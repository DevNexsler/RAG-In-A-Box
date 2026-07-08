# Comm / person / phone / call / voicemail lookups

**For any comm / person / phone / call / voicemail / message question, use
Doc-Organizer first — not raw Comm-Data-Store SQL.** Doc-Organizer already
indexes the comm corpus and returns a compact, info-rich answer. Raw SQL against
Comm-Data-Store is a last resort, only for an *exact field* Doc-Organizer
doesn't surface, and only after it has given you a `source_id` to target.

This exists because the wrong path is easy to reach and expensive: one incident
(Hermes W8 P2, Aaron Curet / +1 484-735-8527 voicemail) fell back to
`docker exec psql` and produced **~73,726 chars** of schema + rows, when the
Doc-Organizer path returned the same operational answer in **~858 chars** —
`sor::task/1698: "Callback: +1 484-735-8527 missed calls + unclear voicemail"`.

## The one safe command

Pass arguments as a **JSON object** via `--args`, never as `key=value` shell
tokens:

```bash
npx mcporter call doc-organizer.comm_lookup \
  --args '{"query":"Aaron Curet phone call voicemail","limit":3}' \
  --output json
```

`comm_lookup` returns a compact envelope (normally 1–2k chars, hard-capped
~3k):

```json
{
  "verdict": "found",
  "top_hit": {"doc_id": "sor::task/1698", "title": "...", "source_type": "sor_task", "score": 0.87},
  "key_facts": ["Callback: +1 484-735-8527", "missed calls + unclear voicemail"],
  "source_ids": ["1698"],
  "snippet": "...",
  "hits": [{"doc_id": "...", "source_type": "...", "score": 0.87, "sender": "...", "direction": "...", "sent_at": "...", "source_id": "..."}],
  "missing_exact_fields": [],
  "sql_needed": false,
  "note": "Compact Doc-Organizer result. For an exact field not shown, run ONE targeted SQL query keyed by a source_id above — do not dump the table."
}
```

- `verdict`: `found` | `not_found` | `ambiguous`.
- `sql_needed`: `true` **only** when nothing was found. When `false`,
  Doc-Organizer answered — do not fall back to SQL.
- `source_ids`: use these to target a single follow-up query when you truly need
  an exact, un-indexed field.

`file_search` with `return="slim"` is the lower-level equivalent if you need
raw hits:

```bash
npx mcporter call doc-organizer.file_search \
  --args '{"query":"Aaron Curet phone call voicemail","limit":3,"return":"slim"}' \
  --output json
```

## Footgun: do NOT pipe MCP JSON into a heredoc

```bash
# BROKEN — do not do this:
npx mcporter call doc-organizer.file_search query='...' --output json | python3 - <<'PY'
import sys, json; data = json.load(sys.stdin)   # stdin is EMPTY here
PY
```

`python3 - <<'PY'` makes the heredoc **the program's stdin**, so the piped
mcporter JSON is discarded and you get an empty-stdin / `DOC_SEARCH_PARSE_ERROR`.
Two things are wrong in that line: the heredoc stdin capture, *and*
`query='...'` (key=value) instead of `--args '{...}'` JSON.

If you must post-process, capture first, then parse:

```bash
out=$(npx mcporter call doc-organizer.comm_lookup --args '{"query":"..."}' --output json)
echo "$out" | jq '.top_hit'
# or: echo "$out" | python3 -c 'import sys, json; print(json.load(sys.stdin)["verdict"])'
```

## When raw Comm-Data-Store SQL is actually warranted

Only when **all** of these hold:

1. You already ran `comm_lookup` / `file_search` and it returned `source_ids`.
2. You need a specific exact field that isn't in the compact result (e.g. a full
   phone number, a raw timestamp, a joined row).
3. You run **one** targeted query keyed by a `source_id` — never
   `SELECT *`-style table or schema dumps.
