#!/usr/bin/env bash
# Install the doc-organizer backup trigger FROM THIS REPO as the single source of
# truth. Symlinks the repo's systemd --user units into ~/.config/systemd/user/
# (symlink so repo edits win after a daemon-reload), enables the timer, and
# removes the legacy bare crontab line so the live schedule derives from the repo.
#
# Idempotent — safe to re-run. Requires: systemd --user with lingering enabled
# (loginctl enable-linger <user>) so the timer fires headless.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS=(doc-organizer-backup.service doc-organizer-backup.timer)

mkdir -p "$UNIT_DIR"
for u in "${UNITS[@]}"; do
  ln -sfn "$REPO_DIR/deploy/systemd/$u" "$UNIT_DIR/$u"
  echo "symlinked $u -> $REPO_DIR/deploy/systemd/$u"
done

systemctl --user daemon-reload
systemctl --user enable --now doc-organizer-backup.timer

# Remove the legacy bare crontab line, if present (single source of truth is now
# the repo timer). Guarded so re-runs are no-ops.
if crontab -l 2>/dev/null | grep -q 'scripts/backup_index.sh'; then
  crontab -l 2>/dev/null | grep -v 'scripts/backup_index.sh' | crontab -
  echo "removed legacy crontab line for scripts/backup_index.sh"
fi

echo
echo "Live trigger now derives from: $REPO_DIR/deploy/systemd/"
systemctl --user --no-pager is-enabled doc-organizer-backup.timer
systemctl --user --no-pager is-active doc-organizer-backup.timer
systemctl --user list-timers doc-organizer-backup.timer --no-pager || true
