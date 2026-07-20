"""Filename @XXXXX@ tokens are identity claims, not identity grants.

External producers (email-attachment capture, comms review copies) mint their
own @NNNNN@ tokens inside filenames they deposit into the vault. The scan must
never let such a token steal a live doc ID from its registered file, adopt a
token this registry never issued, or re-mint an ID that is already taken.

Regression tests for ticket #0390 (193 collisions / 66 ids since April;
IDs 00001+00002 re-pointed to unrelated producer files on 2026-07-19).
"""

import json
from glob import escape as glob_escape

from doc_id_store import DocIDStore, extract_id_from_filename
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


def test_adjudicated_deposit_token_logs_collision_only_once(tmp_path):
    """A no-rename deposit's foreign token is adjudicated once, then silent.

    Deposit-owned files keep their producer-minted token forever (renaming
    them creates a re-deposit loop), so the registry resolves them by path.
    Only the first sweep may record the collision — re-logging the same
    adjudication every sweep buries the audit log and makes "new collision
    events" useless as a health signal.
    """
    registry, owner_id, _owner_rel = _vault_with_owner(tmp_path)

    deposit = tmp_path / "email-attachments"
    deposit.mkdir()
    deposit_rel = f"email-attachments/junk.metadata-only@{owner_id}@.md"
    (tmp_path / deposit_rel).write_text(
        "producer metadata carrying the owner's token"
    )

    def sweep():
        return scan_filesystem_records(
            tmp_path, ["**/*.md"], [],
            doc_id_store=registry,
            no_rename_prefixes=["email-attachments/"],
        )

    first = {r["rel_path"]: r["doc_id"] for r in sweep()}
    assigned = first[deposit_rel]
    assert assigned != owner_id
    events = registry.audit_log(doc_id=owner_id, event=DocIDStore.COLLISION)
    assert len(events) == 1  # the adjudication itself is recorded

    second = {r["rel_path"]: r["doc_id"] for r in sweep()}
    assert second[deposit_rel] == assigned  # identity is stable...
    events = registry.audit_log(doc_id=owner_id, event=DocIDStore.COLLISION)
    assert len(events) == 1  # ...and re-adjudicated silently


def _record_owner_content_identity(registry, vault_root, owner_id, owner_rel):
    """Simulate the dedupe pass recording the owner's exact content identity."""
    from core.dedupe import compute_file_identity

    ident = compute_file_identity(vault_root / owner_rel)
    registry.claim_canonical_by_exact_hash(
        owner_id, ident.size_bytes, ident.content_hash, hash_algo="blake3"
    )


def test_vanished_owner_token_on_different_content_is_not_a_move(tmp_path):
    """Deleting the owner must not let a producer copy inherit its identity.

    The ticket's headline case: the registered holder of a token disappears,
    then a producer file carrying the same token appears elsewhere. The bytes
    don't match the registry's recorded content identity, so it is a
    collision, not a move.
    """
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)
    _record_owner_content_identity(registry, tmp_path, owner_id, owner_rel)

    (tmp_path / owner_rel).unlink()
    (tmp_path / "notes" / f"impostor@{owner_id}@.md").write_text(
        "unrelated producer bytes"
    )

    records = scan_filesystem_records(
        tmp_path, ["**/*.md"], [], doc_id_store=registry
    )

    assert len(records) == 1
    assert records[0]["doc_id"] != owner_id
    # The vanished owner's row is untouched (left for the reap, not re-pointed).
    assert registry.lookup_path(owner_id) == owner_rel
    events = registry.audit_log(doc_id=owner_id, event=DocIDStore.COLLISION)
    assert events, "rejected move-claim must log a collision event"


def test_moved_file_with_matching_content_hash_keeps_id(tmp_path):
    """A genuine move — same bytes, old path gone — still carries the ID."""
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)
    _record_owner_content_identity(registry, tmp_path, owner_id, owner_rel)

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
    moves = registry.audit_log(doc_id=owner_id, event=DocIDStore.MOVED)
    assert moves, "a genuine move is still recorded as moved"


def test_conflicting_communication_sidecar_is_retokenized_but_not_indexed(tmp_path):
    """Excluded sidecars still participate in global token adjudication.

    Production's remaining duplicate-token pairs were communication sidecars:
    the scan skipped them before checking the filename token, so a full-vault
    remediation pass could never reach zero duplicate live tokens.
    """
    registry, owner_id, owner_rel = _vault_with_owner(tmp_path)

    deposit = tmp_path / "email-attachments"
    deposit.mkdir()
    sidecar = deposit / f"message__mm0@{owner_id}@.json"
    sidecar.write_text(json.dumps({
        "schema_version": 2,
        "message": {"source_message_id": "msg-1"},
        "media": {"storage_path": "message__mm0.jpg"},
    }))

    records = scan_filesystem_records(
        tmp_path, ["**/*.md", "**/*.json"], [], doc_id_store=registry
    )

    assert registry.lookup_path(owner_id) == owner_rel
    assert not any(r["rel_path"].endswith(".json") for r in records)
    renamed = list(deposit.glob("message__mm0@*.json"))
    assert len(renamed) == 1
    fresh_id = extract_id_from_filename(renamed[0].name)
    assert fresh_id is not None and fresh_id != owner_id
    fresh_rel = str(renamed[0].relative_to(tmp_path))
    assert registry.lookup_path(fresh_id) == fresh_rel

    collisions = registry.audit_log(doc_id=owner_id, event=DocIDStore.COLLISION)
    assert len(collisions) == 1
    scan_filesystem_records(
        tmp_path, ["**/*.md", "**/*.json"], [], doc_id_store=registry
    )
    assert registry.audit_log(
        doc_id=owner_id, event=DocIDStore.COLLISION
    ) == collisions
