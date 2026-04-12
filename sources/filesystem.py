"""FilesystemSource: scans a directory tree using scan.include/exclude globs
and extracts text via the existing extract_text() dispatcher.

This is a refactor of the legacy scan_vault_task + extract dispatch that
previously lived inline in flow_index_vault.py — zero behavior change,
proven by tests/sources/test_filesystem_source.py.
"""

from pathlib import Path
from typing import Iterator

from sources.base import SourceRecord
from doc_id_store import DocIDStore
from extractors import ExtractionResult, extract_text


# Satisfies the Source protocol from sources.base structurally — no explicit
# inheritance needed. Runtime isinstance checks against Source() work because
# the protocol is @runtime_checkable.
class FilesystemSource:
    """Indexes files matching glob patterns under a directory root."""

    def __init__(
        self,
        name: str,
        root: str | Path,
        scan_config: dict,
        registry: DocIDStore,
        pdf_config: dict | None = None,
    ):
        self.name = name
        self._root = Path(root)
        self._include = scan_config.get("include", ["**/*.md"])
        self._exclude = scan_config.get("exclude", [])
        self._registry = registry
        self._pdf_config = pdf_config or {}
        self._ocr_provider = None  # Set by flow after instantiation; see Task 8

    def set_ocr_provider(self, provider):
        """Injected by flow_index_vault after config is loaded. OCR is a
        flow-level concern, not a source-level one — filesystem doesn't
        know how to build an OCR provider and shouldn't have to."""
        self._ocr_provider = provider

    def scan(self) -> Iterator[SourceRecord]:
        """Delegate to the legacy scan_vault_task which owns the walk +
        rename + registry logic. Yield SourceRecord per returned dict."""
        from flow_index_vault import scan_vault_task, _RUNTIME

        # scan_vault_task reads _RUNTIME["doc_id_store"] — preserve that
        # contract for now. Task 8 will change scan_vault_task to accept
        # the registry as an argument.
        prev = _RUNTIME.get("doc_id_store")
        _RUNTIME["doc_id_store"] = self._registry
        try:
            records = scan_vault_task(self._root, self._include, self._exclude)
        finally:
            if prev is None:
                _RUNTIME.pop("doc_id_store", None)
            else:
                _RUNTIME["doc_id_store"] = prev

        # Backfill source_name on rows this scan just registered.
        # register()'s None-sentinel behavior (Task 2) preserves existing
        # source_name on conflict, so scan_vault_task's register() calls
        # don't clobber. But on fresh inserts the default is 'documents',
        # so we run an explicit UPDATE to set the source_name to this
        # instance's name.
        for r in records:
            self._registry.set_source_name(r["doc_id"], self.name)

        for r in records:
            yield SourceRecord(
                doc_id=r["doc_id"],
                source_type=r["ext"],
                natural_key=r["rel_path"],
                mtime=r["mtime"],
                size=r["size"],
                metadata={
                    "ext": r["ext"],
                    "abs_path": r["abs_path"],
                    "rel_path": r["rel_path"],
                },
            )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        return extract_text(
            file_path=record.metadata["abs_path"],
            ext=record.metadata["ext"],
            ocr_provider=self._ocr_provider,
            pdf_strategy=self._pdf_config.get("strategy", "text_then_ocr"),
            min_text_chars=self._pdf_config.get("min_text_chars_before_ocr", 200),
            ocr_page_limit=self._pdf_config.get("ocr_page_limit", 200),
        )

    def close(self) -> None:
        pass  # Filesystem has no handle to close
