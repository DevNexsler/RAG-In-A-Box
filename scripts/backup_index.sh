#!/usr/bin/env bash
# Nightly backup of the doc-organizer LanceDB index volume.
#
# Backs up the rag-in-a-box_doc-organizer-data volume root (chunks.lance,
# doc_registry.db, taxonomy, metadata). The expensive contents are the LLM
# enrichment fields and embeddings (~26K LLM calls to regenerate), so a
# point-in-time snapshot is worth keeping even mid-indexing — Lance's
# versioned manifests let the corruption recovery walk back to a clean
# version on restore.
#
# Excluded: chunks__shadow.lance (transient rebuild table), *.corrupt
# (already-dead data), indexer logs.
#
# Retention (GFS): 3 daily; Sunday copies go to weekly/ (keep 4); the first
# backup of each month also goes to monthly/ (keep 3). Granular day-by-day
# rollback for the last 30 days comes from in-dataset Lance version
# tags (#0113), which are far cheaper than full tarballs; these independent
# tarballs cover the "chunks.lance directory is lost/corrupt" case (#0113).
#
# Restore:
#   docker compose stop doc-organizer
#   docker run --rm -v rag-in-a-box_doc-organizer-data:/vol \
#     -v /home/danpark/backups/doc-organizer:/backup alpine \
#     sh -c "rm -rf /vol/chunks.lance && tar xzf /backup/<file>.tar.gz -C /vol"
#   docker compose start doc-organizer

set -euo pipefail

VOLUME="${DOC_BACKUP_VOLUME:-rag-in-a-box_doc-organizer-data}"
BACKUP_DIR="${DOC_BACKUP_DIR:-/home/danpark/backups/doc-organizer}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="index-${STAMP}.tar.gz"
LOG="${BACKUP_DIR}/backup.log"

mkdir -p "${BACKUP_DIR}/weekly" "${BACKUP_DIR}/monthly"
exec 9>"${BACKUP_DIR}/.backup.lock"
flock -n 9 || { echo "backup already running"; exit 0; }
PARTIAL="${OUT}.partial"
trap 'rm -f -- "${BACKUP_DIR}/${PARTIAL}"' EXIT

log() { echo "$(date -Is) $*" >> "${LOG}"; }

# Flag (but don't skip) backups taken while the indexer is writing.
RUNNING=""
if docker exec "${DOC_BACKUP_CONTAINER:-doc-organizer}" test -f /data/index/indexer.pid 2>/dev/null; then
  RUNNING=" (indexer was running — point-in-time snapshot)"
fi

docker run --rm \
  -v "${VOLUME}:/vol:ro" \
  -v "${BACKUP_DIR}:/backup" \
  alpine sh -c '
    tar czf "$1" -C /vol --exclude="chunks__shadow.lance" \
      --exclude="*.corrupt" --exclude="indexer.log*" . && chown "$2:$3" "$1"
  ' sh "/backup/${PARTIAL}" "$(id -u)" "$(id -g)"

gzip -t "${BACKUP_DIR}/${PARTIAL}"
mv "${BACKUP_DIR}/${PARTIAL}" "${BACKUP_DIR}/${OUT}"

SIZE=$(du -h "${BACKUP_DIR}/${OUT}" | cut -f1)
log "OK ${OUT} ${SIZE}${RUNNING}"

# Calendar-aware retention validates restore points before pruning. Weekly and
# monthly paths share immutable daily files on the same filesystem.
python3 "$(dirname "$0")/backup_retention.py" "$BACKUP_DIR" "$BACKUP_DIR/$OUT"
log "retention OK: 3 daily, 4 distinct weeks, 3 distinct months"
