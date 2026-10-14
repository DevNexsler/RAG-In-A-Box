"""The compose thread caps must bound a real Lance index-maintenance process (#4083).

Rayon, the lance/lancedb tokio runtimes and OpenBLAS each size themselves to the
host's cores unless the environment says otherwise. On the 32-core production host
one index-run process peaked at 157 threads, and the next pool spawn under the
container's pids_limit failed with EAGAIN inside optimize_indices. This runs the
same Lance calls in a child process carrying only the compose defaults, so a
pool that stops honouring its knob shows up as a thread count that tracks cores.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# This child peaks at ~40 threads under the compose caps and ~112 with only the
# two LANCE_* caps on 32 cores (rayon, tokio and OpenBLAS each add one per core).
THREAD_BUDGET = 64

_CHILD = r"""
import json, os, tempfile, threading
import numpy as np, pyarrow as pa, lance

peak = 0
done = threading.Event()

def sample():
    global peak
    while not done.is_set():
        peak = max(peak, len(os.listdir("/proc/self/task")))
        done.wait(0.01)

threading.Thread(target=sample, daemon=True).start()

def batch(start, n):
    vectors = np.random.default_rng(start).random((n, 64), dtype=np.float32)
    return pa.table({
        "doc_id": [f"d{i}" for i in range(start, start + n)],
        "text": [f"chunk {i} ledger payment notice" for i in range(start, start + n)],
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), 64),
    })

uri = os.path.join(tempfile.mkdtemp(), "chunks.lance")
ds = lance.write_dataset(batch(0, 8192), uri)
ds.create_index("vector", index_type="IVF_PQ", num_partitions=8, num_sub_vectors=8, metric="cosine")
ds.create_scalar_index("doc_id", "BTREE")
ds.create_scalar_index("text", "INVERTED")
ds = lance.write_dataset(batch(8192, 512), uri, mode="append")
ds.optimize.optimize_indices(num_indices_to_merge=2)
done.set()
peak = max(peak, len(os.listdir("/proc/self/task")))
print(json.dumps({"peak_threads": peak}))
"""


def _compose_thread_env() -> dict[str, str]:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = {}
    for item in compose["services"]["doc-organizer"]["environment"]:
        match = re.fullmatch(r"(\w*THREADS\w*)=\$\{\1:-(\d+)\}", item)
        if match:
            env[match.group(1)] = match.group(2)
    return env


def test_lance_index_maintenance_stays_within_thread_budget_under_compose_caps():
    thread_env = _compose_thread_env()
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not re.search(r"THREADS", key)
    }
    child_env.update(thread_env)

    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    peak = json.loads(result.stdout.strip().splitlines()[-1])["peak_threads"]

    assert peak <= THREAD_BUDGET, (thread_env, peak)
