"""Filename @XXXXX@ tokens are identity claims, not identity grants.

External producers (email-attachment capture, comms review copies) mint their
own @NNNNN@ tokens inside filenames they deposit into the vault. The scan must
never let such a token steal a live doc ID from its registered file, adopt a
token this registry never issued, or re-mint an ID that is already taken.

Regression tests for ticket #0390 (193 collisions / 66 ids since April;
IDs 00001+00002 re-pointed to unrelated producer files on 2026-07-19).
"""

from glob import escape as glob_escape

from doc_id_store import DocIDStore
from flow_index_vault import scan_filesystem_records


def _vault_with_owner(tmp_path):
    """Create a vault with one indexed document and return its identity."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "readme.md").write_text("# The real document")
    registry = DocIDStore(tmp_path / "registry.db")
    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )
    assert len(records) == 1
    return registry, records[0]["doc_id"], records[0]["rel_path"]


def test_producer_minted_token_cannot_steal_live_id(tmp_path):
    """A deposited file carrying a live ID must not re-point that ID.

    Mirrors the per-attachment index path (resolve_single_record), which scans
    exactly one file: the registered owner is not part of the scan, so only the
    registry can defend the ID.
    """
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)

    deposit = tmp_path / "0-attachments"
    deposit.mkdir()
    impostor = deposit / f"junk.metadata-only@{owner_id}@.md"
    impostor.write_text("producer metadata, unrelated to the owner")

    records = scan_filesystem_records(
        tmp_path,
        [glob_escape(f"0-attachments/{impostor.name}")],
        [],
        doc_id_store=registry,
    )

    # The live ID still points at its registered file...
    assert registry.lookup_path(owner_id) == owner_rel
    # ...and the impostor got a fresh identity instead of the stolen one.
    assert len(records) == 1
    assert records[0]["doc_id"] != owner_id
    assert f"@{owner_id}@" not in records[0]["rel_path"]
    # The rejected claim is visible in the audit log.
    events = registry.audit_log(doc_id=owner_id, event=DocIDStore.COLLISION)
    assert events, "forged identity claim must log a collision event"


def test_full_scan_owner_keeps_id_regardless_of_walk_order(tmp_path):
    """Whichever file the walk visits first, the registered owner wins."""
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)

    # Two impostors in directories that sort before AND after the owner's.
    for dirname in ("0-attachments", "zz-attachments"):
        (tmp_path / dirname).mkdir()
        (tmp_path / dirname / f"copy@{owner_id}@.md").write_text(
            f"unrelated deposit in {dirname}"
        )

    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )

    assert registry.lookup_path(owner_id) == owner_rel
    by_rel = {r["rel_path"]: r["doc_id"] for r in records}
    assert by_rel[owner_rel] == owner_id
    assert len(set(by_rel.values())) == 3  # every file has a distinct ID


def test_unissued_token_is_not_adopted_on_populated_registry(tmp_path):
    """A token the registry never issued is not an identity claim at all.

    Adopting it would plant the ID in the counter's future space, minting a
    guaranteed collision for whenever the counter catches up.
    """
    registry, _owner_id, _owner_rel = _vault_with_owner(tmp_path)

    (tmp_path / "notes" / "report@zzzzz@.md").write_text(
        "carries a token this registry never issued"
    )
    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )

    stray = next(r for r in records if "report" in r["rel_path"])
    assert stray["doc_id"] != "zzzzz"
    assert registry.lookup_path("zzzzz") is None
    assert "@zzzzz@" not in stray["rel_path"]  # forged token stripped on rename


def test_bootstrap_scan_on_empty_registry_still_adopts_tokens(tmp_path):
    """Rebuilding a lost registry from filename tokens must keep working."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "recipe@abc12@.md").write_text("# Recipe")
    registry = DocIDStore(tmp_path / "registry.db")

    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )

    assert records[0]["doc_id"] == "abc12"
    assert registry.lookup_path("abc12") == "notes/recipe@abc12@.md"


def test_moved_file_keeps_id_on_populated_registry(tmp_path):
    """A genuine move (old path gone) still carries the ID to the new path."""
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)

    archive = tmp_path / "archive"
    archive.mkdir()
    owner_name = owner_rel.split("/")[-1]
    (tmp_path / owner_rel).rename(archive / owner_name)

    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )

    assert len(records) == 1
    assert records[0]["doc_id"] == owner_id
    assert registry.lookup_path(owner_id) == f"archive/{owner_name}"


def test_next_id_skips_ids_already_registered_or_retired(tmp_path):
    """The counter must never re-mint an ID that already names a document."""
    registry = DocIDStore(tmp_path / "registry.db")
    # An adopted token sits exactly where the counter will land next...
    registry.register("00001", "adopted/one.md")
    # ...followed by a retired ID.
    registry.register("00002", "gone/two.md")
    registry.delete("00002")

    fresh = registry.next_id()

    assert fresh == "00003"
    assert registry.lookup_path(fresh) is None
    assert not registry.is_retired(fresh)
