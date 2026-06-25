# SOR structured query tool (`sor_query`) — design

**Date:** 2026-06-25
**Status:** Approved (design), pending implementation plan
**Component:** RAG-in-a-Box MCP server (doc-organizer)

## Motivation

Hermes agents do exact/aggregate SOR (System-of-Record, NocoDB-backed Postgres)
work — tenant collections reconciliation, ticket status, building/unit/contact
lookups — by hand-writing raw SQL inside `execute_code` via
`npx mcporter call sor-postgres.query` (the generic `@modelcontextprotocol/server-postgres`).
Spend-log analysis showed this burns turns and tokens through two recurring
patterns:

1. **Schema rediscovery.** The SOR uses quoted CamelCase identifiers
   (`"Building Units"`, `"Nick_Name"`, `"Status"`) the model can't guess, so it
   runs repeated `information_schema.columns` probes and trial-and-errors the
   casing (e.g. `"Building Units"` then `building_units`) before a query works.
   Several wasted round-trips up front, every session.
2. **Find-then-SQL detour + unbounded output.** `doc-organizer.file_search`
   (semantic RAG) can't aggregate, join, or return exhaustive/exact rows, so the
   agent does a search, then drops to raw SQL anyway — and that SQL is typically
   `SELECT *` with no `LIMIT`, pretty-printed as `json.dumps(indent=2)`, producing
   7–53K-char dumps that then replay every subsequent turn.

`doc-organizer` (RAG-in-a-Box) is already configured with the SOR Postgres DSN
(the `sources` entry named `sor`, resolved from `${SOR_DSN}`) and indexes the SOR
via `sources/postgres.py`/`PostgresSource` — though that connection lives in the
**indexer subprocess** (`flow_index_vault.py`), not the MCP server process. It is
also the documented "search here FIRST" front door. Adding a structured query tool
here reuses that same DSN config and puts exact retrieval next to semantic
`file_search` in one surface.

## Goals

- Eliminate schema-rediscovery round-trips (hand the agent the schema; never make
  it probe `information_schema`).
- Make exact/joined/aggregated SOR retrieval a single bounded call, so the agent
  stops the find-then-SQL detour and the `SELECT *` dumps.
- Bound output at the source so results do not bloat the replayed transcript.

## Non-goals (YAGNI)

- **No JSON query DSL.** The agent already writes correct SQL; a DSL would
  re-implement SQL, constrain expressiveness, and break on schema change.
- **No fuzzy/semantic entity resolution** inside this tool. A simple SQL `ILIKE`
  substring filter covers `'%rosado%'`-style lookups; genuinely fuzzy resolution
  stays in `file_search`.
- **No writes.** Read-only. SOR mutations remain the job of `nocodb-plus`.
- **No canned/parameterized query library** in v1 (may add hot queries later if
  usage shows a clear top-N).

## Tool surface

Two FastMCP `@mcp.tool()` functions in `mcp_server.py`, alongside `file_search`.

### `sor_query`

```
sor_query(sql: str, limit: int = 50, format: "tsv" | "json" = "tsv") -> str
```

- Runs a **read-only** SQL query against the SOR Postgres and returns rows.
- `limit` — default row cap (see Auto-LIMIT). `format` — `tsv` (default, compact)
  or `json` (when the agent needs structure).
- The tool **description embeds a compact schema of the core SOR tables**
  (auto-generated — see Schema exposure), so the agent writes correct identifiers
  first try.

### `sor_schema`

```
sor_schema(table: str | None = None) -> str
```

- Returns the current schema (table → columns with types) for one table, or a
  compact index of all tables when `table` is omitted.
- For tables not in the `sor_query` inline core set, the agent fetches schema here
  in **one bounded call** instead of N `information_schema` probes.
- Auto-generated from the live DB, cached (see below).

## Guardrails

1. **Read-only enforcement (defense in depth).**
   - Connect using a dedicated least-privilege Postgres role (`sor_readonly`,
     SELECT-only grants). If provisioning that role is out of scope for v1, fall
     back to `default_transaction_read_only = on` on the connection.
   - Wrap execution in a `READ ONLY` transaction.
   - Statement validation: reject anything that is not a single `SELECT`/`WITH`
     statement (no multiple statements, no DDL/DML keywords). Return a clear error.
2. **Auto-LIMIT.** If the query has no effective top-level `LIMIT`, wrap it
   (`SELECT * FROM (<sql>) _sub LIMIT :n`) and fetch `n+1` rows to detect
   truncation. When truncated, prepend a one-line notice
   (`[49 of >50 rows — add LIMIT or narrow the WHERE]`).
3. **Statement timeout.** Set a per-query `statement_timeout` (e.g. 15s) so a
   runaway/cartesian join cannot hang the MCP server.
4. **Compact serialization + cell capping.** Default TSV (no `indent=2`). Cap
   individual cell width (e.g. 500 chars) so a single fat free-text column
   (`Staff_Notes`/`AI_Notes`) cannot blow the budget; truncated cells get an
   ellipsis marker.

## Schema exposure (two-tier, auto-generated, never hand-maintained)

The schema the agent sees is **generated from the live DB**, not a static string:

- `get_sor_schema()` queries `information_schema.columns` (reusing the existing
  SOR Postgres connection config), builds the schema text, and **caches it with a
  short TTL** (e.g. 10 min) plus a refresh on server start.
- **Tier 1 — inline core:** `sor_query`'s description includes the ~6 hot tables
  (Buildings, Building Units, Contacts, Collection Tickets, Tasks, Legal Entities,
  per the `sor` source). Keeps the common path zero-extra-calls and the per-turn
  description size flat as the overall schema grows. **The exact identifiers shown
  are auto-generated from the live schema, not authored** — observed transcripts
  used `collection_tickets` while `config.yaml` indexes `"Collection Tickets"`;
  the generated hint reflects whatever the DB actually exposes, which is precisely
  why the tool eliminates the agent's casing/quoting guesswork.
- **Tier 2 — `sor_schema` on demand:** full/other-table schema fetched only when
  needed.

**Robustness properties (explicit design intent):**

- **Execution is schema-agnostic.** The wrapper never parses/validates/hardcodes
  columns — it passes SQL to Postgres. Adding a column/table is immediately
  queryable with no code change or redeploy.
- **Hint auto-refreshes.** New columns/comment changes appear within the cache TTL
  (or on restart). Nothing to manually keep in sync.
- **Graceful degradation.** If the inline hint is momentarily stale, queries still
  run; worst case the agent does one `sor_schema`/`information_schema` lookup —
  i.e. it never degrades below today's behavior.

## Architecture & data flow

```
agent (mcporter call doc-organizer.sor_query) 
  -> FastMCP @mcp.tool() sor_query(sql, limit, format)
       -> validate read-only (statement check)
       -> wrap: BEGIN READ ONLY; SET statement_timeout; auto-LIMIT subquery
       -> execute via psycopg (new read-only connection opened in the MCP
          process from the `sor` source DSN — NOT a shared/existing connection)
       -> serialize rows (TSV default, cell-capped) + truncation notice
  -> bounded text result
```

- Reuse RAG-in-a-Box's existing SOR Postgres connection configuration: the MCP
  process reads the DSN from `config["sources"]` (the entry named `sor`, resolved
  from `${SOR_DSN}` — `PostgresSource.__init__` already contains the `${ENV}`
  resolution logic to reuse or reference) and opens its **own new** read-only
  connection. Do not introduce new credentials, and do not reuse the indexer
  subprocess's connection — a separate read-only connection (or `sor_readonly`
  role) isolates query load from indexing.
- New code lives in a focused module (e.g. `sor_query.py`) with the SQL execution,
  guardrails, and schema cache; `mcp_server.py` only wires the two `@mcp.tool()`
  entry points to it. Keeps `mcp_server.py` thin and the logic unit-testable
  without the MCP layer.

## Error handling

- Non-SELECT / multi-statement → `ERROR: sor_query is read-only; only a single
  SELECT/WITH is allowed.`
- SQL/Postgres error → return the Postgres error message verbatim (it's the most
  useful signal for the agent to self-correct), prefixed `SOR query error:`.
- Timeout → `ERROR: query exceeded 15s; add a filter or LIMIT.`
- Unknown table in `sor_schema(table)` → list available table names.

## Testing

- **Unit (no DB):** statement validator (accept SELECT/WITH; reject UPDATE/DELETE/
  INSERT/DDL/multi-statement); auto-LIMIT wrapping + truncation detection; TSV/JSON
  serialization + cell capping; schema-text generation from a mocked
  `information_schema` result.
- **Integration (test Postgres, mirrors NocoDB CamelCase tables):** read-only role
  rejects writes; `statement_timeout` fires; a join + a `GROUP BY` aggregate return
  correct rows; `sor_schema` reflects an added column after cache expiry.
- Mirror existing MCP-contract test patterns (`tests/test_mcp_contract.py`).

## Success criteria

- A reconciliation lookup that previously took ~4 turns (2 schema probes + a casing
  retry + a `SELECT *` dump) completes in **1 `sor_query` call** with correct
  identifiers first try.
- `COUNT`/`SUM`/`GROUP BY` and multi-table joins succeed in one call.
- Default result is bounded (TSV, capped rows + cells); no unbounded `SELECT *`
  dumps reach the transcript.
- Zero `information_schema` probes for the core tables in normal use.

## Rollout / verification

- Ship behind the existing MCP server; verify via mcporter (`doc-organizer.sor_query`,
  `doc-organizer.sor_schema`).
- Re-run the LiteLLM spend-log analysis after a usage window: expect SOR-related
  `execute_code` turn counts and average result-byte size to drop, and
  `information_schema` probe frequency to fall to ~0 for core tables.

## Post-launch watch / v2 candidates

These were scoped out of v1 (see Non-goals). Watch for the signals; if a signal
fires, the named v2 is the response.

### 1. Fuzzy entity resolution (primary watch)
`file_search` and `sor_query` are separate tools. A chained
`file_search` → `sor_query` collapses to **one** agent turn when the hand-off is
programmatic (extract an id from the search hit, plug into SQL inside one
`execute_code` block). It costs **two** turns only when the model must *read*
several semantic hits and *judge* which entity is correct before querying.

- **Signal to watch:** in the LiteLLM spend logs, count chained
  `doc-organizer.file_search` → `doc-organizer.sor_query` sequences where query 2
  is NOT programmatically derivable from query 1 (i.e. an LLM round-trip sits
  between them). If this judgment-requiring chain is common, the 2-turn cost is
  recurring.
- **v2 response:** add fuzzy entity resolution — a combined call that resolves a
  fuzzy reference (`"Rosado"`, `"54 S Broad"`) to the exact contact/unit/ticket
  rows and returns them, collapsing the judgment chain into one call. (Note this
  reintroduces semantic ranking into the structured path, so design it as its own
  tool/mode, not a change to `sor_query`'s deterministic contract.)

### 2. Canned / parameterized query library
If usage shows a clear top-N of near-identical `sor_query` SQL shapes (e.g. a
tenant ledger, tickets-by-status, a building roster), add named convenience
queries (`tenant_ledger(contact)`, `tickets_by_status(status)`).

- **Signal:** high frequency of near-duplicate SQL bodies in the spend logs.
- **v2 response:** a small set of named parameterized queries (tiniest tokens,
  safest). Keep `sor_query` for the long tail.

### 3. Read-only role hardening (ops)
v1 enforces read-only via `default_transaction_read_only=on` + statement
validation (psycopg3's extended protocol already blocks multi-statement). For
defense in depth, provision a dedicated `sor_readonly` Postgres role with
`SELECT`-only grants and point this tool's DSN at it. Not required for v1
correctness — track as an ops follow-up.

### 4. Externalized-ref durability (cross-project note)
Unrelated to this tool but adjacent: hermes-lcm `/lcm doctor` flagged orphaned
externalized payload refs (missing JSON files) — mostly pre-v0.18 legacy.
v0.18 #265 hardens this going forward. Mentioned only so it isn't mistaken for a
`sor_query` issue if it surfaces.
