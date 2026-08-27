<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **RAG-In-A-Box** (6935 symbols, 16364 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/RAG-In-A-Box/context` | Codebase overview, check index freshness |
| `gitnexus://repo/RAG-In-A-Box/clusters` | All functional areas |
| `gitnexus://repo/RAG-In-A-Box/processes` | All execution flows |
| `gitnexus://repo/RAG-In-A-Box/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Testing

- `make gate-fast` during development (static → unit → integration); `make gate` before any release — all five tiers including staging-e2e (hermetic compose stack) and live (real providers, real money, preflight-guarded).
- Full operator's manual: [docs/TESTING.md](docs/TESTING.md) (tiers, staging stack, fault injection, live preflight, gate reports).
- Tier markers are auto-derived from the path, in precedence order: `tests/e2e/` → e2e, `_live.py` suffix → live, `.int.test.py` → integration, everything else → unit; explicit markers win.
- New MCP tools are automatically REQUIRED to have e2e coverage and `mcp.tool.<name>` spans — the two-sided coverage check discovers tools via live `list_tools` and fails the gate for any tool without both.
- Only `make gate-real` / `make test-e2e-real` spend money (opt-in real-API e2e; needs a real `OPENROUTER_API_KEY`); plain `make gate` never does. Comm-Data-Store runs its own isolated copy of the staging stack (`docker-compose.staging.cds.yml`, `:27788`) with an env-driven `document.indexed` callback — don't change the hook payload or the `${VAR}` url resolution in `hooks/http.py` without coordinating with CDS. See [docs/TESTING.md](docs/TESTING.md) → "Cross-repo test target (CDS)".
