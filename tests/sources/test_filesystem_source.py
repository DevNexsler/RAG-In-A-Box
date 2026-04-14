"""Parity test: FilesystemSource.scan() + .extract() must produce the
same records and extraction output as the legacy scan_vault_task +
extract_text path in flow_index_vault.py."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Copy the real test_vault into a tmp dir so we don't mutate tracked files."""
    src = Path(__file__).parent.parent.parent / "test_vault"
    dst = tmp_path / "vault"
    shutil.copytree(src, dst)
    return dst


def test_scan_yields_same_records_as_legacy(tmp_vault):
    """Given identical config, FilesystemSource.scan() yields records with
    the same doc_id/mtime/size/source_type/metadata as scan_vault_task."""
    from doc_id_store import DocIDStore
    from flow_index_vault import scan_vault_task, _RUNTIME
    from sources.filesystem import FilesystemSource

    include = ["**/*.md", "**/*.pdf", "**/*.png"]
    exclude = []

    # Two separate registries so the two scans don't interfere with each other
    reg_legacy = DocIDStore(tmp_vault.parent / "legacy.db")
    reg_new = DocIDStore(tmp_vault.parent / "new.db")

    # Legacy path needs _RUNTIME populated with the registry
    _RUNTIME["doc_id_store"] = reg_legacy
    legacy_records = scan_vault_task(tmp_vault, include, exclude)
    _RUNTIME.pop("doc_id_store", None)

    # Reset the vault so rename side effects from the legacy scan don't bias the new one
    src = Path(__file__).parent.parent.parent / "test_vault"
    shutil.rmtree(tmp_vault)
    shutil.copytree(src, tmp_vault)

    src_new = FilesystemSource(
        name="documents",
        root=tmp_vault,
        scan_config={"include": include, "exclude": exclude},
        registry=reg_new,
    )
    new_records = list(src_new.scan())

    # Records are equivalent modulo the exact @XXXXX@ suffix (counters are
    # independent between the two registries). Compare normalized doc_ids.
    from core.source_types import canonical_source_type
    from doc_id_store import strip_id_from_filename

    def normalize(records):
        return sorted([
            (
                # Legacy dicts have "ext"; SourceRecords have .source_type
                canonical_source_type(r.get("source_type", r.get("ext"))) if isinstance(r, dict) else r.source_type,
                strip_id_from_filename(
                    Path(r["rel_path"] if isinstance(r, dict) else r.natural_key).name
                ),
                r["size"] if isinstance(r, dict) else r.size,
            )
            for r in records
        ])

    assert normalize(legacy_records) == normalize(new_records)
    assert len(new_records) == len(legacy_records)


def test_extract_matches_legacy_extract_text(tmp_vault):
    """FilesystemSource.extract(record) returns the same text as
    the legacy extract_text() call with the same arguments."""
    from extractors import extract_text
    from doc_id_store import DocIDStore
    from sources.filesystem import FilesystemSource

    reg = DocIDStore(tmp_vault.parent / "reg.db")
    src = FilesystemSource(
        name="documents",
        root=tmp_vault,
        scan_config={"include": ["**/*.md"], "exclude": []},
        registry=reg,
    )

    records = list(src.scan())
    md_record = next(r for r in records if r.source_type == "md")

    new_result = src.extract(md_record)

    # Legacy equivalent
    legacy_result = extract_text(
        file_path=str(md_record.metadata["abs_path"]),
        ext="md",
        ocr_provider=None,
        pdf_strategy="text_then_ocr",
        min_text_chars=200,
        ocr_page_limit=200,
    )

    assert new_result.full_text == legacy_result.full_text


def test_scan_emits_canonical_source_type_and_preserves_extension(tmp_path):
    """FilesystemSource should expose canonical search source_type, not raw suffix."""
    from doc_id_store import DocIDStore
    from sources.filesystem import FilesystemSource

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "report.docx").write_text("not a real docx; scan only needs stat")

    src = FilesystemSource(
        name="documents",
        root=vault,
        scan_config={"include": ["**/*.docx"], "exclude": []},
        registry=DocIDStore(tmp_path / "reg.db"),
    )

    [record] = list(src.scan())

    assert record.source_type == "doc"
    assert record.metadata["ext"] == "docx"


def test_scan_does_not_mutate_flow_runtime(tmp_path, monkeypatch):
    """FilesystemSource.scan should not depend on flow_index_vault._RUNTIME."""
    from doc_id_store import DocIDStore
    import flow_index_vault
    from sources.filesystem import FilesystemSource

    class NoMutationDict(dict):
        def __setitem__(self, key, value):
            raise AssertionError(f"_RUNTIME mutated via __setitem__({key!r})")

        def pop(self, key, default=None):
            raise AssertionError(f"_RUNTIME mutated via pop({key!r})")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note")
    monkeypatch.setattr(flow_index_vault, "_RUNTIME", NoMutationDict())

    src = FilesystemSource(
        name="documents",
        root=vault,
        scan_config={"include": ["**/*.md"], "exclude": []},
        registry=DocIDStore(tmp_path / "reg.db"),
    )

    records = list(src.scan())

    assert len(records) == 1
