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
