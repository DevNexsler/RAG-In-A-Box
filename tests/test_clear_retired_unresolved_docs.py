"""#2074 — the one-time clear of registry-retired terminal degraded entries.

The terminal ledger (``degraded_unresolved.json``) exists to hold documents the
pipeline can no longer resolve, i.e. possible silent loss. Truncating it would
also discard a genuinely lost document, so the clear must be driven by the
registry's own record of ids *we* retired — never by the ledger alone.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from doc_id_store import DocIDStore
from scripts.clear_retired_unresolved_docs import (
    clear_retired,
    is_registry_retired,
    open_registry,
)

RETIRED_NAMESPACED = "comm_messages::zoho_mail/<7r3ekCixTQ6Uw7mj7RXcJQ@geopod-ismtpd-canary-0>"
RETIRED_BARE = "documents::abc12"
NEVER_RETIRED = "comm_messages::zoho_cliq/1780327866430_15958014910122"

TERMINAL_ENTRY = {"attempts": 1, "reasons": ["enrichment_failed"], "unresolved_runs": 3}


def _index_root(tmp_path: Path, ledger_docs: dict) -> Path:
    """Build an index root holding a terminal ledger and a real registry.

    The registry is populated through ``DocIDStore`` itself so the fixture's
    notion of "retired" is the registry's, not the test's.
    """
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "degraded_unresolved.json").write_text(
        json.dumps({"docs": ledger_docs}, indent=2, sort_keys=True), encoding="utf-8"
    )

    store = DocIDStore(index_root / "doc_registry.db")
    # A namespaced comm id: registered and deleted under its full key, which is
    # how the 16 pre-#2022 entries sit in prod.
    store.register(RETIRED_NAMESPACED, "zoho_mail/mangled.json", source_name="comm_messages")
    store.delete(RETIRED_NAMESPACED)
    # A legacy bare row reached through its all_mappings() key: DocIDStore.delete
    # retires the BARE id, so the ledger key and the retired key differ.
    store.register("abc12", "notes/legacy.md", source_name="documents")
    store.delete(RETIRED_BARE)
    store.close()
    return index_root


def _ledger(index_root: Path) -> dict:
    return json.loads((index_root / "degraded_unresolved.json").read_text(encoding="utf-8"))


def test_registry_retirement_is_recognised_for_both_stored_key_forms(tmp_path):
    """A ledger key is retired whether the registry stored it namespaced or bare."""
    index_root = _index_root(tmp_path, {})
    with open_registry(index_root / "doc_registry.db") as connection:
        assert is_registry_retired(connection, RETIRED_NAMESPACED) is True
        assert is_registry_retired(connection, RETIRED_BARE) is True
        assert is_registry_retired(connection, NEVER_RETIRED) is False


def test_dry_run_reports_the_retired_entries_and_writes_nothing(tmp_path, capsys):
    docs = {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY), RETIRED_BARE: dict(TERMINAL_ENTRY)}
    index_root = _index_root(tmp_path, docs)
    before = (index_root / "degraded_unresolved.json").read_bytes()

    dropped, kept = clear_retired(index_root, apply=False)

    assert sorted(dropped) == sorted(docs)
    assert kept == {}
    assert (index_root / "degraded_unresolved.json").read_bytes() == before
    assert "DRY-RUN" in capsys.readouterr().out


def test_apply_leaves_the_ledger_with_zero_entries(tmp_path):
    docs = {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY), RETIRED_BARE: dict(TERMINAL_ENTRY)}
    index_root = _index_root(tmp_path, docs)

    dropped, kept = clear_retired(index_root, apply=True)

    assert len(dropped) == 2 and kept == {}
    assert _ledger(index_root) == {"docs": {}}


def test_an_id_the_registry_never_retired_is_kept_even_under_apply(tmp_path):
    """The sink's whole purpose: an unexplained id is a possible lost document."""
    docs = {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY), NEVER_RETIRED: dict(TERMINAL_ENTRY)}
    index_root = _index_root(tmp_path, docs)

    dropped, kept = clear_retired(index_root, apply=True)

    assert list(dropped) == [RETIRED_NAMESPACED]
    assert list(kept) == [NEVER_RETIRED]
    assert _ledger(index_root) == {"docs": {NEVER_RETIRED: TERMINAL_ENTRY}}


def test_apply_is_idempotent(tmp_path):
    docs = {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY)}
    index_root = _index_root(tmp_path, docs)

    clear_retired(index_root, apply=True)
    after_first = (index_root / "degraded_unresolved.json").read_bytes()
    dropped, kept = clear_retired(index_root, apply=True)

    assert dropped == {} and kept == {}
    assert (index_root / "degraded_unresolved.json").read_bytes() == after_first


def test_unknown_top_level_ledger_fields_survive_the_clear(tmp_path):
    """The ledger's shape is owned by the flow (#2022 may add fields) — don't eat them."""
    index_root = _index_root(tmp_path, {})
    (index_root / "degraded_unresolved.json").write_text(
        json.dumps(
            {"docs": {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY)}, "version": 2},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    clear_retired(index_root, apply=True)

    assert _ledger(index_root) == {"docs": {}, "version": 2}


def test_a_missing_registry_refuses_to_drop_anything(tmp_path):
    """No ground truth means no proof of retirement — keep every entry."""
    index_root = _index_root(tmp_path, {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY)})
    (index_root / "doc_registry.db").unlink()
    before = (index_root / "degraded_unresolved.json").read_bytes()

    with pytest.raises(FileNotFoundError):
        clear_retired(index_root, apply=True)

    assert (index_root / "degraded_unresolved.json").read_bytes() == before


def test_a_corrupt_ledger_is_not_read_as_empty(tmp_path):
    """Silently treating unreadable JSON as 'nothing to clear' would hide entries."""
    index_root = _index_root(tmp_path, {})
    (index_root / "degraded_unresolved.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        clear_retired(index_root, apply=True)


def test_an_absent_ledger_is_already_clear(tmp_path):
    index_root = _index_root(tmp_path, {})
    (index_root / "degraded_unresolved.json").unlink()

    assert clear_retired(index_root, apply=True) == ({}, {})


def test_cli_defaults_to_dry_run(tmp_path):
    """--apply is opt-in, matching reopen_capped_ocr_docs.py / backfill_unledgered_stub_docs.py."""
    index_root = _index_root(tmp_path, {RETIRED_NAMESPACED: dict(TERMINAL_ENTRY)})
    before = (index_root / "degraded_unresolved.json").read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "clear_retired_unresolved_docs.py"

    result = subprocess.run(
        [sys.executable, str(script), "--index-root", str(index_root)],
        capture_output=True, text=True, check=True,
    )

    assert "DRY-RUN" in result.stdout
    assert (index_root / "degraded_unresolved.json").read_bytes() == before
