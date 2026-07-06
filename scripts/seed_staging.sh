#!/usr/bin/env bash
# Seed a RUNNING staging stack with a standing test corpus, then index it.
#
# Opt-in and parallel-only: the hermetic gate e2e never calls this. Deposits
# every file under the corpus dir into the app container's /data/documents and
# triggers a per-file index (the same REST path prod's per-attachment hook uses).
# README* files are skipped. This is testing-only data on a throwaway volume —
# never production. To keep the seeded corpus across restarts, stop the stack
# with `down` (NOT `down -v`).
#
# Env overrides (defaults target the base staging stack on :17788):
#   SEED_COMPOSE   compose -f args    (default: -f docker-compose.staging.yml)
#                  CDS: "-f docker-compose.staging.yml -f docker-compose.staging.cds.yml"
#   SEED_SERVICE   app service name   (default: doc-organizer-staging)
#   SEED_URL       app base url       (default: http://127.0.0.1:17788; CDS: :27788)
#   SEED_API_KEY   bearer token       (default: staging-test-key)
#   SEED_CORPUS    corpus dir         (default: staging/fixtures/corpus)
set -euo pipefail

SEED_COMPOSE="${SEED_COMPOSE:--f docker-compose.staging.yml}"
SEED_SERVICE="${SEED_SERVICE:-doc-organizer-staging}"
SEED_URL="${SEED_URL:-http://127.0.0.1:17788}"
SEED_API_KEY="${SEED_API_KEY:-staging-test-key}"
SEED_CORPUS="${SEED_CORPUS:-staging/fixtures/corpus}"

[ -d "$SEED_CORPUS" ] || { echo "corpus dir not found: $SEED_CORPUS" >&2; exit 1; }

shopt -s nullglob
seeded=0 failed=0
for f in "$SEED_CORPUS"/*; do
  name="$(basename "$f")"
  [ -f "$f" ] || continue
  case "$name" in README*|.*) continue ;; esac

  # shellcheck disable=SC2086 -- SEED_COMPOSE intentionally carries multiple -f args
  docker compose $SEED_COMPOSE cp "$f" "$SEED_SERVICE:/data/documents/$name" >/dev/null
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SEED_URL/api/index/document" \
    -H "Authorization: Bearer $SEED_API_KEY" -H "Content-Type: application/json" \
    -d "{\"rel_path\":\"$name\"}")
  if [ "$code" = "200" ]; then
    echo "  ok    $name (HTTP $code)"; seeded=$((seeded + 1))
  else
    echo "  FAIL  $name (HTTP $code)" >&2; failed=$((failed + 1))
  fi
done

echo "Seeded $seeded file(s), $failed failure(s) from $SEED_CORPUS into $SEED_SERVICE."
echo "Persist across restarts with \`down\` (not \`down -v\`)."
[ "$failed" -eq 0 ]
