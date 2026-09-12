"""An index run's own summary must report what it wrote, not how big its queue was.

Regression guard for #1173: `Index stats: added/updated=N` was `len(queue)` —
every examined document counted, including the ones the run explicitly decided
not to index. A run whose documents were all duplicate-skips or all
no-text-extracted still reported a non-zero `added/updated` next to
`completion=100.0%`, so the run's success record could not be used as evidence
that anything reached LanceDB.

These drive the real flow over a real store, so the emitted line is the one an
operator (or a dashboard, or an alert) actually reads.
"""

import ast
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("prefect")
pytest.importorskip("llama_index")

import flow_index_vault as fiv
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider


class _StubEmbedProvider(EmbedProvider):
    """Deterministic vectors — this suite asserts on counters, not on ranking."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


def _config(root: Path, index_root: Path) -> dict:
    return {
        "index_root": str(index_root),
        "sources": [
            {
                "type": "filesystem",
                "name": "documents",
                "root": str(root),
                "scan": {"include": ["**/*.md"], "exclude": []},
            }
        ],
        "chunking": {
            "max_chars": 1800,
            "overlap": 200,
            "semantic": {"enabled": False},
        },
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "media": {"enabled": False},
        "dedupe": {"enabled": True},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "INFO"},
    }


def _run_flow(
    root: Path, index_root: Path, *, terminal_failures: tuple[str, ...] = ()
) -> None:
    """Drive the real flow. `terminal_failures` holds filename stems whose
    processing raises a deterministic (non-transient) error — the terminal skip
    lane. Matched as a substring because the filesystem source stamps the doc id
    into the name it reports (`broken@00002@.md`)."""
    store = LanceDBStore(str(index_root), "chunks")
    taxonomy = MagicMock()
    taxonomy.count.return_value = 0
    process_doc_task = fiv.process_doc_task

    def _process_doc(doc: dict):
        if any(stem in doc.get("rel_path", "") for stem in terminal_failures):
            # The #0569 shape: a deterministic provider rejection that can never
            # succeed on a re-run, so the doc is quarantined instead of retried.
            raise RuntimeError('embeddings error 400: {"message":"invalid input"}')
        return process_doc_task(doc)

    with patch("flow_index_vault.load_config", return_value=_config(root, index_root)), \
         patch("flow_index_vault.get_run_logger", return_value=logging.getLogger("test")), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=_StubEmbedProvider()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.process_doc_task", _process_doc), \
         patch("core.taxonomy.load_taxonomy_store", return_value=taxonomy):
        fiv.index_vault_flow.fn("dummy.yaml")


def _stats_line(caplog) -> str:
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Index stats:")
    ]
    assert len(lines) == 1, f"expected one Index stats line, got {lines}"
    return lines[0]


def _completion_line(caplog) -> str:
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Index run completion:")
    ]
    assert len(lines) == 1, f"expected one completion line, got {lines}"
    return lines[0]


def _chunk_write_ops(caplog) -> int:
    return len([
        record
        for record in caplog.records
        if re.match(r"^(Inserted|Upserted) \d+ chunks: ", record.getMessage())
    ])


_STATS_SKIPPED = re.compile(r"skipped=(\d+)(?: (\{.*?\}))?, deleted=")
_LEDGER_SKIPPED = re.compile(r"^(\d+) docs added to skip ledger \(.*?\): (\{.*\})$")


def _skip_rollup(pattern: "re.Pattern[str]", line: str) -> tuple[int, dict]:
    """(document count, reason breakdown) as one summary line reports them."""
    match = pattern.search(line)
    assert match, f"unparseable skip roll-up: {line}"
    return int(match.group(1)), ast.literal_eval(match.group(2) or "{}")


def _ledger_line(caplog) -> str:
    lines = [
        record.getMessage()
        for record in caplog.records
        if "docs added to skip ledger" in record.getMessage()
    ]
    assert len(lines) == 1, f"expected one skip ledger line, got {lines}"
    return lines[0]


def test_all_skip_run_reports_zero_indexed_with_skip_reasons(tmp_path, caplog):
    """The ticket's run: every queued doc skipped, nothing written to the index."""
    root = tmp_path / "documents"
    root.mkdir()
    (root / "blank.md").write_text("   \n\n")
    (root / "original.md").write_text("shared body text for the duplicate pair\n")
    (root / "copy.md").write_text("shared body text for the duplicate pair\n")

    index_root = tmp_path / "index"
    caplog.set_level(logging.INFO)

    # First run indexes one of the pair and elects it canonical; its twin skips
    # as a duplicate and blank.md skips as contentless.
    _run_flow(root, index_root)

    # Age the skip ledger so both skips come due again (production: "22 due this
    # run, 22 retries restamped"). The indexed doc is unchanged and stays out of
    # the queue, so the second run queues nothing but skips.
    ledger = fiv._load_skip_ledger(index_root)
    for entry in ledger["docs"].values():
        entry["skipped_at"] = 0.0
    assert fiv._save_skip_ledger(index_root, ledger)
    caplog.clear()

    _run_flow(root, index_root)

    line = _stats_line(caplog)
    assert _chunk_write_ops(caplog) == 0, line
    assert "indexed_docs=0" in line, line
    assert "indexed_chunks=0" in line, line
    assert "skipped=2" in line, line
    assert "no_text_extracted" in line, line
    assert "duplicate_of" in line, line


def test_completion_line_carries_the_indexed_count(tmp_path, caplog):
    """`completion=100.0%` is queue drain; it must never stand alone as evidence."""
    root = tmp_path / "documents"
    root.mkdir()
    (root / "blank.md").write_text("   \n\n")

    index_root = tmp_path / "index"
    caplog.set_level(logging.INFO)

    _run_flow(root, index_root)

    line = _completion_line(caplog)
    assert "completion=100.0%" in line, line
    assert "indexed_docs=0" in line, line


def test_mixed_queue_indexed_count_matches_chunk_write_ops(tmp_path, caplog):
    """Acceptance #4: the reported count is the number of real write calls."""
    root = tmp_path / "documents"
    root.mkdir()
    (root / "blank.md").write_text("   \n\n")
    (root / "alpha.md").write_text("alpha body text\n")
    (root / "beta.md").write_text("beta body text\n")

    index_root = tmp_path / "index"
    caplog.set_level(logging.INFO)

    _run_flow(root, index_root)

    line = _stats_line(caplog)
    write_ops = _chunk_write_ops(caplog)
    assert write_ops == 2, line
    assert f"indexed_docs={write_ops}" in line, line
    assert "indexed_chunks=2" in line, line
    assert "skipped=1" in line, line
    assert "no_text_extracted" in line, line


def test_terminal_failure_counts_the_same_on_both_skip_roll_ups(tmp_path, caplog):
    """#2184: `Index stats:` and the skip ledger line must not disagree.

    A terminal (non-transient) processing failure writes a real
    `terminal_error:<ExcType>` skip ledger entry, so the run's two adjacent
    skip roll-ups have to count that document — and name its reason — alike.
    Before the fix the terminal lane left the progress counters untouched, so a
    run with `k` terminal failures reported `skipped=N` next to
    `N+k docs added to skip ledger`, with `terminal_error` in one dict only.
    """
    root = tmp_path / "documents"
    root.mkdir()
    (root / "blank.md").write_text("   \n\n")
    (root / "broken.md").write_text("body text that never reaches the store\n")

    index_root = tmp_path / "index"
    caplog.set_level(logging.INFO)

    _run_flow(root, index_root, terminal_failures=("broken",))

    stats_count, stats_reasons = _skip_rollup(_STATS_SKIPPED, _stats_line(caplog))
    ledger_count, ledger_reasons = _skip_rollup(_LEDGER_SKIPPED, _ledger_line(caplog))

    assert (stats_count, stats_reasons) == (ledger_count, ledger_reasons)
    assert ledger_reasons == {"no_text_extracted": 1, "terminal_error": 1}
    assert stats_count == 2

    # The terminal lane's own ERROR line is an external contract — log-patterns.conf
    # and three Maint-Manager outcome checks match it verbatim.
    assert any(
        re.search(r"^Skipping .* after retries exhausted: ", record.getMessage())
        for record in caplog.records
    )


def test_completion_line_agrees_with_the_skip_ledger_on_terminal_failures(
    tmp_path, caplog
):
    """The third roll-up reads from the same counters, so it moves with them."""
    root = tmp_path / "documents"
    root.mkdir()
    (root / "broken.md").write_text("body text that never reaches the store\n")

    index_root = tmp_path / "index"
    caplog.set_level(logging.INFO)

    _run_flow(root, index_root, terminal_failures=("broken",))

    ledger_count, _ = _skip_rollup(_LEDGER_SKIPPED, _ledger_line(caplog))
    line = _completion_line(caplog)
    assert f"skipped={ledger_count}" in line, line
