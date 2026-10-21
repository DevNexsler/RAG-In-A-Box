"""The nightly index backup wrapper against a Docker that knows only real images.

Only the docker CLI is replaced. The fake daemon holds the one image the
production container runs (releases tag it doc-organizer:main-<sha>; no
:latest exists) and, like the real daemon, refuses to run any other image.
The real backup_index.sh, gzip check, publish step and retention run as-is.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_IMAGE = "sha256:" + "2b" * 32

FAKE_DOCKER = f"""#!{sys.executable}
import io, json, os, sys, tarfile
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
containers = {{"doc-organizer": {DEPLOYED_IMAGE!r}}}
if args[0] == "inspect":
    name = args[-1]
    if name not in containers:
        sys.exit("Error: No such object: " + name)
    print(containers[name])
elif args[0] == "exec":
    sys.exit(1)  # indexer not running
elif args[0] == "run":
    rest, mounts = args[1:], {{}}
    while rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("--network", "--entrypoint"):
            rest.pop(0)
        elif flag == "-v":
            source, target = rest.pop(0).split(":")[:2]
            mounts[target] = source
    image = rest[0]
    if image not in containers.values():
        sys.stderr.write(f"Unable to find image '{{image}}' locally\\n")
        sys.exit(125)
    partial = rest[-3].replace("/backup", mounts["/backup"], 1)
    with tarfile.open(partial, "w:gz") as archive:
        member = tarfile.TarInfo("./record")
        member.size = 8
        archive.addfile(member, io.BytesIO(b"fixture\\n"))
else:
    sys.exit("unexpected docker call: " + " ".join(args))
"""


def run_backup(tmp_path, **env_overrides):
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    backups = tmp_path / "backups"
    log = tmp_path / "docker.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("DOC_BACKUP_")}
    env.update(PATH=f"{commands}{os.pathsep}{os.environ['PATH']}",
               DOC_BACKUP_DIR=str(backups), FAKE_DOCKER_LOG=str(log), **env_overrides)
    completed = subprocess.run(["bash", str(ROOT / "scripts/backup_index.sh")],
                               env=env, capture_output=True, text=True, timeout=60)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return completed, calls, backups


def test_backup_runs_tar_from_the_image_the_production_container_runs(tmp_path):
    completed, calls, backups = run_backup(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = next(call for call in calls if call[0] == "run")
    assert DEPLOYED_IMAGE in run
    archives = list(backups.glob("index-*.tar.gz"))
    assert len(archives) == 1
    assert f" OK {archives[0].name} " in (backups / "backup.log").read_text()
    assert not list(backups.glob("*.partial"))


def test_backup_without_a_production_container_fails_before_writing(tmp_path):
    completed, calls, backups = run_backup(tmp_path, DOC_BACKUP_CONTAINER="missing")
    assert completed.returncode != 0
    assert "No such object: missing" in completed.stderr
    assert not [call for call in calls if call[0] == "run"]
    assert not list(backups.glob("index-*"))
