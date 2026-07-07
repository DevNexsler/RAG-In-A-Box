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
# Retention: 3 daily backups; Sunday backups are also copied to weekly/ and
# the 2 newest weeklies are kept (~2 weeks of coarse physical DR). Granular
# day-by-day rollback for the last 30 days comes from in-dataset Lance version
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

VOLUME="rag-in-a-box_doc-organizer-data"
BACKUP_DIR="/home/danpark/backups/doc-organizer"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="index-${STAMP}.tar.gz"
LOG="${BACKUP_DIR}/backup.log"

mkdir -p "${BACKUP_DIR}/weekly"

log() { echo "$(date -Is) $*" >> "${LOG}"; }

# Flag (but don't skip) backups taken while the indexer is writing.
RUNNING=""
if docker exec doc-organizer test -f /data/index/indexer.pid 2>/dev/null; then
  RUNNING=" (indexer was running — point-in-time snapshot)"
fi

docker run --rm \
  -v "${VOLUME}:/vol:ro" \
  -v "${BACKUP_DIR}:/backup" \
  alpine tar czf "/backup/${OUT}" -C /vol \
    --exclude='chunks__shadow.lance' \
    --exclude='*.corrupt' \
    --exclude='indexer.log*' \
    .

SIZE=$(du -h "${BACKUP_DIR}/${OUT}" | cut -f1)
log "OK ${OUT} ${SIZE}${RUNNING}"

# Sunday → keep a weekly copy
if [ "$(date +%u)" = "7" ]; then
  cp "${BACKUP_DIR}/${OUT}" "${BACKUP_DIR}/weekly/${OUT}"
  log "weekly copy ${OUT}"
fi

# Retention: 3 daily, 2 weekly (reduced 2026-07-07: tarballs are ~100G each, disaster-recovery only) (find avoids ls-glob crash when empty)
find "${BACKUP_DIR}" -maxdepth 1 -name 'index-*.tar.gz' -printf '%T@ %p\n' \
  | sort -rn | tail -n +4 | cut -d' ' -f2- | xargs -r rm -f
find "${BACKUP_DIR}/weekly" -maxdepth 1 -name 'index-*.tar.gz' -printf '%T@ %p\n' \
  | sort -rn | tail -n +3 | cut -d' ' -f2- | xargs -r rm -f
