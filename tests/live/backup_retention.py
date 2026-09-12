#!/usr/bin/env python3
"""Restore a real wrapper-produced backup from a private fixture volume."""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
name = "backup-e2e-" + uuid.uuid4().hex[:12]


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout.strip()


try:
    docker("volume", "create", name)
    docker(
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        name + ":/data",
        "alpine:latest",
        "sh",
        "-c",
        "echo business-fixture > /data/record; mkdir /data/chunks__shadow.lance; echo transient > /data/chunks__shadow.lance/skip",
    )
    with tempfile.TemporaryDirectory(prefix="backup-e2e-") as tmp:
        env = dict(os.environ, DOC_BACKUP_VOLUME=name, DOC_BACKUP_DIR=tmp, DOC_BACKUP_CONTAINER="missing-" + name)
        subprocess.run(["bash", str(ROOT / "scripts/backup_index.sh")], env=env, check=True)
        archives = list(Path(tmp).glob("index-*.tar.gz"))
        assert len(archives) == 1
        monthly = list((Path(tmp) / "monthly").glob("*.tar.gz"))
        assert len(monthly) == 1
        assert archives[0].stat().st_ino == monthly[0].stat().st_ino
        with tarfile.open(archives[0], "r:gz") as archive:
            assert archive.extractfile("./record").read() == b"business-fixture\n"
            assert not any("chunks__shadow" in item for item in archive.getnames())
        assert not list(Path(tmp).glob("*.partial"))
        print("PASS real Docker backup restored, transient data excluded, monthly snapshot hardlinked")
        print("image", docker("image", "inspect", "alpine:latest", "--format", "{{.Id}}"))
finally:
    docker("volume", "rm", name)
    assert name not in docker("volume", "ls", "-q").split()
    print("cleanup: fixture containers, volume and private backup files removed")
