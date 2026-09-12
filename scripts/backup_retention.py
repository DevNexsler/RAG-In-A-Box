"""Promote verified immutable backups and retain distinct calendar generations."""

from __future__ import annotations

import argparse
from datetime import datetime
import errno
import hashlib
import gzip
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

NAME = re.compile(r"index-(\d{8}-\d{6})\.tar\.gz\Z")


def stamp(path: Path) -> datetime:
    match = NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a completed backup: {path.name}")
    return datetime.strptime(match[1], "%Y%m%d-%H%M%S")


def archives(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and not p.is_symlink() and NAME.fullmatch(p.name)), key=stamp
    )


def verify(path: Path) -> None:
    # A tar reader can stop before the gzip trailer. Drain the compressed
    # stream as well so CRC/length corruption cannot qualify a restore point.
    with gzip.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is not None:
                        while stream.read(1024 * 1024):
                            pass
        while compressed.read(1024 * 1024):
            pass


def link_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        fd, temporary = tempfile.mkstemp(prefix=".copy-", dir=target.parent)
        os.close(fd)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)


def digest(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def rotate(directory: Path, latest: Path) -> None:
    if latest.parent != directory or latest.is_symlink():
        raise ValueError("latest must be a regular backup in the daily directory")
    date = stamp(latest)
    verify(latest)
    weekly, monthly = directory / "weekly", directory / "monthly"
    for folder in (weekly, monthly):
        if folder.is_symlink():
            raise ValueError("retention directories must not be symlinks")
        folder.mkdir(exist_ok=True)
    periods = [(monthly, lambda p: stamp(p).strftime("%Y-%m"), 3)]
    if date.weekday() == 6:
        week = lambda p: stamp(p).isocalendar()[:2]
        if not any(week(p) == week(latest) for p in archives(weekly)):
            link_copy(latest, weekly / latest.name)
    month = date.strftime("%Y-%m")
    if not any(stamp(p).strftime("%Y-%m") == month for p in archives(monthly)):
        link_copy(latest, monthly / latest.name)
    periods.append((weekly, lambda p: stamp(p).isocalendar()[:2], 4))
    # Keep first successful snapshot in each period, then newest distinct periods.
    remove = []
    keep = archives(directory)[-3:]
    remove.extend(p for p in archives(directory) if p not in keep)
    for folder, key, count in periods:
        groups = {}
        for path in archives(folder):
            groups.setdefault(key(path), []).append(path)
        selected = sorted(groups)[-count:]
        chosen = [groups[k][0] for k in selected]
        keep.extend(chosen)
        remove.extend(p for paths in groups.values() for p in paths if p not in chosen)
    # Older archives may predate validation. Check retained restore points before
    # discarding any redundant/month-expired generation.
    verified = {(latest.stat().st_dev, latest.stat().st_ino)}
    for path in keep:
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity not in verified:
            verify(path)
            verified.add(identity)
    # Existing copies with the same snapshot name can share the same immutable
    # inode after content equality is verified. Preserve distinct-content files.
    by_name = {}
    for path in keep:
        previous = by_name.setdefault(path.name, path)
        if previous == path or previous.stat().st_ino == path.stat().st_ino:
            continue
        if previous.stat().st_dev == path.stat().st_dev and digest(previous) == digest(path):
            fd, temporary_name = tempfile.mkstemp(prefix=".link-", dir=path.parent)
            os.close(fd)
            temporary = Path(temporary_name)
            temporary.unlink()
            try:
                os.link(previous, temporary)
                os.replace(temporary, path)
            except PermissionError:
                # Older root-owned snapshots can be readable but not linkable.
                # New backups are owned by the scheduling user; retain old copies.
                pass
            finally:
                temporary.unlink(missing_ok=True)
    for path in remove:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("latest", type=Path)
    args = parser.parse_args()
    rotate(args.directory, args.latest)


if __name__ == "__main__":
    main()
