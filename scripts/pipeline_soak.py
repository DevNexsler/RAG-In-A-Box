#!/usr/bin/env python3
"""Isolated queue/storage/callback soak. Local HTTP faults; no configured providers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hook_outbox import HookOutbox
from core.index_request_queue import IndexRequestQueue
from hooks.http import send_http_event
from scripts.owned_process import owned_process
from scripts.rag_quality import make_node


def rss_mib() -> float:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('RSS telemetry unavailable')


def run_soak(index_root: Path, *, rounds=10, documents=16, workers=4,
             max_p95_ms=5000.0, max_rss_growth_mib=128.0, max_versions_per_write=4.0) -> dict:
    from lancedb_store import LanceDBStore
    if min(rounds, documents, workers) < 1 or min(max_p95_ms, max_rss_growth_mib, max_versions_per_write) <= 0:
        raise ValueError('positive workload and budgets required')
    if index_root.exists() and any(index_root.iterdir()):
        raise ValueError('soak index must be empty')
    queue = IndexRequestQueue(index_root)
    outbox = HookOutbox(index_root)
    store = LanceDBStore(index_root, 'soak')
    attempts, accepted = {}, set()
    peer_lock = threading.Lock()

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            event = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            event_id = event['event_id']
            with peer_lock:
                attempts[event_id] = attempts.get(event_id, 0) + 1
                code = 503 if attempts[event_id] == 1 else 200
                if code == 200:
                    accepted.add(event_id)
            body = b'{"status":"updated"}'
            self.send_response(code)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    hook = {'name': 'soak', 'url': f'http://127.0.0.1:{server.server_port}/callback',
            'timeout_seconds': 2, 'accepted_statuses': ['updated', 'duplicate']}
    latencies, samples = [], []
    started = time.monotonic()
    # Production keeps its worker pool across batches/checkpoints. Recreating
    # threads every round adds allocator warmup unrelated to sustained load.
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        for round_id in range(rounds):
            def write_document(number):
                before = time.monotonic()
                doc_id = f'documents::{number}'
                request = queue.enqueue('soak', 'documents', f'{number}.md')
                text = f'Document {number}. Committed revision {round_id}. Receipt paid by Visa ****5678.'
                node = make_node(doc_id, text)
                store.upsert_nodes([node])
                newer = queue.enqueue('soak', 'documents', f'{number}.md', force=True)
                if queue.complete(request):
                    raise AssertionError('stale ACK deleted new work')
                store.upsert_nodes([node])  # retry after commit, before ACK
                if not queue.complete(newer):
                    raise AssertionError('latest request was lost')
                event = {'event_id': f'{round_id}:{number}', 'event': 'document.indexed',
                         'doc_id': doc_id, 'rel_path': f'{number}.md'}
                outbox.enqueue(event, hook)
                return (time.monotonic() - before) * 1000

            # Match production: create the Lance table with one serial document
            # before concurrent writers use its now-established schema.
            if round_id == 0:
                latencies.append(write_document(0))
                latencies.extend(executor.map(write_document, range(1, documents)))
            else:
                latencies.extend(executor.map(write_document, range(documents)))
            for _ in range(2):
                for delivery in outbox.due(documents * 2, now=time.time() + 120):
                    owned = outbox.claim(delivery)
                    if owned is None:
                        raise AssertionError('callback claim lost')
                    result = send_http_event(owned.hook, owned.event)
                    if result.accepted:
                        if outbox.complete(owned) is None:
                            raise AssertionError('callback ACK lost')
                    elif result.retryable:
                        outbox.retry(owned, result.outcome, result.error, now=time.time() - 120)
                    else:
                        raise AssertionError('non-retryable callback failure')
            # Reopen public readers each round: persisted state, not cached writes.
            reopened = LanceDBStore(index_root, 'soak')
            if reopened.count_chunks() != documents:
                raise AssertionError('replay duplicated or lost chunks')
            for number in range(documents):
                chunks = reopened.get_doc_chunks(f'documents::{number}')
                if len(chunks) != 1 or f'Committed revision {round_id}.' not in chunks[0].text:
                    raise AssertionError('latest text not durable')
            samples.append({'round': round_id + 1, 'rss_mib': rss_mib(),
                            'retained_versions': len(list(index_root.rglob('*.manifest'))),
                            'storage_bytes': sum(path.stat().st_size for path in index_root.rglob('*') if path.is_file()),
                            'pending_requests': len(queue.pending('soak', limit=documents * 2)),
                            'pending_callbacks': len(outbox.due(documents * 2, now=float('inf')))})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    p95 = sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)]
    growth = max(sample['rss_mib'] for sample in samples) - samples[0]['rss_mib']
    versions_per_write = samples[-1]['retained_versions'] / len(latencies)
    passed = (p95 <= max_p95_ms and growth <= max_rss_growth_mib and versions_per_write <= max_versions_per_write
              and len(accepted) == rounds * documents
              and all(sample['pending_requests'] == sample['pending_callbacks'] == 0 for sample in samples))
    return {'passed': passed, 'mode': 'local-subsystem-soak', 'rounds': rounds, 'workers': workers,
            'writes': len(latencies), 'chunks': store.count_chunks(), 'elapsed_s': time.monotonic() - started,
            'write_p95_ms': p95, 'rss_growth_mib': growth, 'versions_per_write': versions_per_write,
            'pending_requests': samples[-1]['pending_requests'], 'pending_callbacks': samples[-1]['pending_callbacks'],
            'callback_retries': sum(value - 1 for value in attempts.values()), 'samples': samples,
            'budgets': {'max_p95_ms': max_p95_ms, 'max_rss_growth_mib': max_rss_growth_mib,
                        'max_versions_per_write': max_versions_per_write}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--documents', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deadline', type=float, default=600)
    parser.add_argument('--max-p95-ms', type=float, default=5000)
    parser.add_argument('--max-rss-growth-mib', type=float, default=128)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--index-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupted)
        args.output.unlink(missing_ok=True)
        # Parent owns cleanup even if the worker must be killed at the deadline.
        with tempfile.TemporaryDirectory(prefix='rag-soak-') as directory:
            command = [sys.executable, __file__, *(argv if argv is not None else sys.argv[1:]),
                       '--worker', '--index-root', directory]
            try:
                with owned_process(command) as process:
                    code = process.wait(timeout=args.deadline)
                if code and not args.output.exists():
                    args.output.write_text(json.dumps({'passed': False, 'error': 'worker_failed', 'returncode': code}) + '\n')
                return code
            except subprocess.TimeoutExpired:
                args.output.write_text(json.dumps({'passed': False, 'error': 'deadline_exceeded'}) + '\n')
                return 1
            except BaseException:
                args.output.write_text(json.dumps({'passed': False, 'error': 'interrupted'}) + '\n')
                raise
    if args.index_root is None:
        parser.error('worker requires owned index root')
    report = run_soak(args.index_root, rounds=args.rounds, documents=args.documents, workers=args.workers,
                      max_p95_ms=args.max_p95_ms, max_rss_growth_mib=args.max_rss_growth_mib)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
