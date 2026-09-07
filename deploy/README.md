# deploy/

Version-controlled deployment triggers for RAG-in-a-Box, so the live schedule
derives from this repo (single source of truth) rather than a loose machine-local
crontab line or `~/.config/systemd/user/` file.

## systemd/ — nightly index backup

`doc-organizer-backup.timer` + `.service` run `scripts/backup_index.sh` daily at
04:30 (replaces the old `30 4 * * *` crontab line). Output is appended to
`~/backups/doc-organizer/backup.log`, matching the original cron redirect.

### Install / update

```bash
./deploy/install.sh
```

Symlinks the units into `~/.config/systemd/user/` (symlink, so repo edits win
after a `daemon-reload`), enables the timer, and removes the legacy crontab line.
Idempotent. Requires systemd `--user` with lingering on
(`loginctl enable-linger danpark`) so the timer fires headless.

### Verify

```bash
systemctl --user list-timers doc-organizer-backup.timer
systemctl --user is-enabled doc-organizer-backup.timer   # -> enabled
readlink ~/.config/systemd/user/doc-organizer-backup.timer  # -> this repo
```

## systemd/ — memory watchdog

`doc-organizer-memory-watchdog.timer` runs `scripts/memory_watchdog.py` every
5 minutes. The server's resident set grows between restarts (~0.4 GB/h under
organic load, mostly after index sweeps — 2026-09-07) inside a 10 GiB cgroup it
shares with the ~3.7 GB sweep subprocess; past ~7 GB the next sweep can reach
the ceiling and the kernel kills the server mid-run. The watchdog restarts the
container instead, at a quiet moment: never during an index run
(`/data/index/indexer.pid`), never when the container is unhealthy. Threshold:
`DOC_ORGANIZER_RSS_RESTART_MB` (default 7000). One JSON line per run in
`~/backups/doc-organizer/memory-watchdog.log`; `--dry-run` reports the decision.
Installed by `./deploy/install.sh` alongside the backup timer.
