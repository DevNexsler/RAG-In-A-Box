"""One source's scan failure must not cost the whole index run (#2020).

`_scan_and_register_sources` iterated every configured source in one unguarded
loop, so an exception from any single `scan()` propagated out of
`index_vault_flow` and killed the run before a single document was processed.
On 2026-09-03 a 21-second Comm-Data-Store Postgres recovery window cost a
complete index cycle: every healthy filesystem source went unscanned and the
run died with `queued=unknown processed=0`.

These drive the real flow over a real store, because the guarantees at stake
are downstream of the scan — the other sources' documents get indexed, and the
failed source's documents, registry rows and degraded-ledger entries survive a
pass that produced no evidence about them.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("prefect")
pytest.importorskip("llama_index")

import flow_index_vault as fiv
from doc_id_store import DocIDStore, strip_id_from_filename
from extractors import ExtractionResult
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider
from sources import build_source as real_build_source
from sources.base import SourceRecord


class _StubEmbedProvider(EmbedProvider):
    """Deterministic vectors — this suite asserts on presence, not on ranking."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


class _RowSource:
    """A database-shaped source whose scan can be told to fail.

    Modelled on `sources/postgres.py`: no filesystem root (so absence can only
    be concluded from the scan) and `scan()` raises from its very first step,
    the way `PostgresSource._get_conn` does when the server is in recovery.
    """

    def __init__(self, name: str, doc_ids: list[str]):
        self.name = name
        self._doc_ids = doc_ids
        self.fail = False

    def scan(self):
        if self.fail:
            raise ConnectionError(
                'connection failed: FATAL: the database system is in recovery mode'
            )
        for doc_id in self._doc_ids:
            yield SourceRecord(
                doc_id=doc_id,
                source_type="other",
                natural_key=f"{self.name}/{doc_id}",
                mtime=1.0,
                size=64,
                metadata={"_text": f"row body for {doc_id}"},
            )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        return ExtractionResult.from_text(record.metadata["_text"])

    def close(self) -> None:
        pass


def _config(root: Path, index_root: Path, safety: dict | None = None) -> dict:
    return {
        "index_root": str(index_root),
        "safety": safety or {},
        "sources": [
            {
                "type": "filesystem",
                "name": "documents",
                "root": str(root),
                "scan": {"include": ["**/*.md"], "exclude": []},
            },
            {"type": "rows", "name": "comm_messages"},
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
    root: Path, index_root: Path, row_source: _RowSource, safety: dict | None = None
) -> None:
    """Drive the real flow with `comm_messages` served by the stub source."""

    def _build_source(source_config, *, registry=None, pdf_config=None):
        if source_config.get("type") == "rows":
            return row_source
        return real_build_source(
            source_config, registry=registry, pdf_config=pdf_config
        )

    store = LanceDBStore(str(index_root), "chunks")
    taxonomy = MagicMock()
    taxonomy.count.return_value = 0
    with patch("flow_index_vault.load_config", return_value=_config(root, index_root, safety)), \
         patch("flow_index_vault.get_run_logger", return_value=logging.getLogger("test")), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=_StubEmbedProvider()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("sources.build_source", _build_source), \
         patch("core.taxonomy.load_taxonomy_store", return_value=taxonomy):
        fiv.index_vault_flow.fn("dummy.yaml")


def _completion_line(caplog) -> str:
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Index run completion:")
    ]
    assert len(lines) == 1, f"expected one completion line, got {lines}"
    return lines[0]


def _indexed_doc_ids(index_root: Path) -> set[str]:
    return set(LanceDBStore(str(index_root), "chunks").list_doc_mtimes())


def _indexed_paths(index_root: Path) -> set[str]:
    """Indexed documents as registry paths — filesystem doc_ids are registry
    ids (`documents::00001`) and their files carry an injected `@id@`, so a
    stripped path map is what makes an assertion readable.
    """
    mappings = DocIDStore(index_root / "doc_registry.db").all_mappings()
    paths = set()
    for doc_id in _indexed_doc_ids(index_root):
        rel_path = mappings.get(doc_id, doc_id)
        parent, _, name = rel_path.rpartition("/")
        name = strip_id_from_filename(name)
        paths.add(f"{parent}/{name}" if parent else name)
    return paths


def test_failed_source_scan_does_not_abort_the_run(tmp_path, caplog):
    """The ticket's run: one source raises, every other source still indexes."""
    root = tmp_path / "documents"
    root.mkdir()
    (root / "first.md").write_text("the first note, indexed by the healthy run\n")

    index_root = tmp_path / "index"
    rows = _RowSource("comm_messages", ["msg-1", "msg-2"])

    # A healthy pass first: both sources index, so the second pass has
    # something to lose.
    _run_flow(root, index_root, rows)
    assert _indexed_paths(index_root) == {
        "first.md", "comm_messages/msg-1", "comm_messages/msg-2",
    }

    # A degraded entry for the failing source, carrying no unresolved streak.
    ledger_path = fiv._degraded_ledger_path(index_root)
    ledger_path.write_text(json.dumps({
        "version": 2,
        "docs": {"comm_messages::msg-1": {"reasons": ["enrichment_failed"], "attempts": 1}},
    }))

    # Now the source is unavailable, and a new document lands meanwhile.
    (root / "second.md").write_text("the second note, deposited during the outage\n")
    rows.fail = True
    caplog.set_level(logging.INFO)
    _run_flow(root, index_root, rows)

    indexed = _indexed_paths(index_root)
    # The healthy source was scanned and its new document processed...
    assert "second.md" in indexed
    # ...and the unavailable source's documents were not read as deletions.
    assert {"comm_messages/msg-1", "comm_messages/msg-2"} <= indexed

    # Its registry rows survive too — a scan that never ran is not evidence
    # that the rows vanished.
    registry = DocIDStore(index_root / "doc_registry.db")
    assert {"comm_messages::msg-1", "comm_messages::msg-2"} <= set(
        registry.all_mappings()
    )

    # Its degraded entry buckets as source_not_scanned: untouched, not aged,
    # so a provider outage can never escalate it out of the retry ledger.
    ledger = json.loads(ledger_path.read_text())
    assert ledger["docs"]["comm_messages::msg-1"] == {
        "reasons": ["enrichment_failed"], "attempts": 1,
    }
    assert "0 unresolved" in "".join(
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("Degraded ledger:")
    )

    # And the run says so: partial, naming the source that failed.
    completion = _completion_line(caplog)
    assert "partial=true" in completion
    assert "failed_sources=comm_messages" in completion


def test_forced_rebuild_does_not_promote_a_shadow_built_from_a_partial_scan(
    tmp_path, caplog
):
    """A shadow promote replaces the whole corpus.

    Building one from a scan that missed a source would delete that source
    wholesale — the mass deletion the per-source guards exist to prevent — so
    an explicit `safety.force_full_rebuild` degrades to an in-place update
    while any source is unscannable.
    """
    root = tmp_path / "documents"
    root.mkdir()
    (root / "first.md").write_text("the first note, indexed by the healthy run\n")

    index_root = tmp_path / "index"
    rows = _RowSource("comm_messages", ["msg-1", "msg-2"])
    _run_flow(root, index_root, rows)

    rows.fail = True
    caplog.set_level(logging.INFO)
    _run_flow(root, index_root, rows, safety={"force_full_rebuild": True})

    assert "Skipping shadow rebuild" in "".join(
        record.getMessage() for record in caplog.records
    )
    assert {"comm_messages/msg-1", "comm_messages/msg-2"} <= _indexed_paths(index_root)


def test_every_source_failing_still_fails_the_run(tmp_path):
    """Isolation is per source, not a blanket swallow: no source scanned, no run."""
    root = tmp_path / "documents"
    root.mkdir()
    index_root = tmp_path / "index"
    rows = _RowSource("comm_messages", ["msg-1"])
    rows.fail = True

    def _fail_filesystem_scan(self):
        raise OSError("source root unavailable")

    with patch("sources.filesystem.FilesystemSource.scan", _fail_filesystem_scan):
        with pytest.raises(Exception) as excinfo:
            _run_flow(root, index_root, rows)

    assert "comm_messages" in str(excinfo.value)
    assert "documents" in str(excinfo.value)
