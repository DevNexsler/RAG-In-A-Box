"""FilesystemSource: scans a directory tree using scan.include/exclude globs
and extracts text via the existing extract_text() dispatcher.

This is a refactor of the legacy scan_vault_task + extract dispatch that
previously lived inline in flow_index_vault.py — zero behavior change,
proven by tests/sources/test_filesystem_source.py.
"""

from pathlib import Path
from glob import escape as glob_escape
import logging
import re
from typing import Iterator

from communication_context import communication_metadata_from_sidecar
from sources.base import SourceRecord
from doc_id_store import DocIDStore
from extractors import ExtractionResult, extract_text
from core.source_types import canonical_source_type

logger = logging.getLogger(__name__)
_INJECTED_ID_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)@[^@]+@$")


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
        self._no_rename = scan_config.get("no_rename", [])
        self._registry = registry
        self._pdf_config = pdf_config or {}
        self._ocr_provider = None  # Set by flow after instantiation; see Task 8
        self._media_provider = None

    def set_ocr_provider(self, provider):
        """Injected by flow_index_vault after config is loaded. OCR is a
        flow-level concern, not a source-level one — filesystem doesn't
        know how to build an OCR provider and shouldn't have to."""
        self._ocr_provider = provider

    def set_media_provider(self, provider):
        """Injected by flow_index_vault for local audio/video extraction."""
        self._media_provider = provider

    def scan(self) -> Iterator[SourceRecord]:
        """Scan files without depending on the Prefect task runtime."""
        from flow_index_vault import scan_filesystem_records

        records = scan_filesystem_records(
            self._root,
            self._include,
            self._exclude,
            doc_id_store=self._registry,
            no_rename_prefixes=self._no_rename,
        )

        # Backfill source_name on rows this scan just registered.
        # register()'s None-sentinel behavior (Task 2) preserves existing
        # source_name on conflict, so scan_vault_task's register() calls
        # don't clobber. But on fresh inserts the default is 'documents',
        # so we run an explicit UPDATE to set the source_name to this
        # instance's name.
        for r in records:
            self._registry.set_source_name(r["doc_id"], self.name)

        for r in records:
            abs_path = Path(r["abs_path"])
            metadata = {
                "ext": r["ext"],
                "abs_path": r["abs_path"],
                "rel_path": r["rel_path"],
            }
            metadata.update(_communication_sidecar_metadata(abs_path))
            yield SourceRecord(
                doc_id=r["doc_id"],
                source_type=canonical_source_type(r["ext"]),
                natural_key=r["rel_path"],
                mtime=r["mtime"],
                size=r["size"],
                metadata=metadata,
            )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        return extract_text(
            file_path=record.metadata["abs_path"],
            ext=record.metadata["ext"],
            ocr_provider=self._ocr_provider,
            media_provider=self._media_provider,
            pdf_strategy=self._pdf_config.get("strategy", "text_then_ocr"),
            min_text_chars=self._pdf_config.get("min_text_chars_before_ocr", 200),
            ocr_page_limit=self._pdf_config.get("ocr_page_limit", 200),
        )

    def close(self) -> None:
        pass  # Filesystem has no handle to close


def _communication_sidecar_metadata(media_path: Path) -> dict[str, str]:
    if media_path.suffix.lower() == ".json":
        return {}
    sidecar_path = _find_communication_sidecar(media_path)
    if sidecar_path is None:
        return {}
    try:
        return communication_metadata_from_sidecar(media_path, sidecar_path)
    except Exception as exc:
        logger.debug("Could not read communication sidecar %s: %s", sidecar_path, exc)
        return {}


def _find_communication_sidecar(media_path: Path) -> Path | None:
    exact = media_path.with_suffix(".json")
    if exact.exists():
        return exact

    match = _INJECTED_ID_SUFFIX_RE.match(media_path.stem)
    if not match:
        return None

    prefix = match.group("prefix")
    candidates = sorted(media_path.parent.glob(f"{glob_escape(prefix)}@*.json"))
    return candidates[0] if candidates else None
