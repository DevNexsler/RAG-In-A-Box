"""Taxonomy tool surface: full CRUD round-trip over the real MCP transport.

ADAPTATION (documented): the plan called for `file_taxonomy_import` to bulk-load
2 terms, but the tool is deprecated in production (mcp_server._file_taxonomy_import_impl
returns {"error": true, "code": "deprecated"} — SQLite seed DBs were removed).
The test asserts the deprecation contract instead, and the "bulk 2 terms" step
uses two file_taxonomy_add calls.
"""
import pytest

pytestmark = pytest.mark.anyio

TAG = "e2e-zanzibar"
TAG_ID = f"tag:{TAG}"
BULK = ("e2e-bulk-alpha", "e2e-bulk-beta")


async def test_taxonomy_crud_roundtrip(indexed_corpus, mcp_session):
    s = mcp_session

    # -- cleanup from any earlier (failed) run: deletes may 'not_found', fine
    for name in (TAG, *BULK):
        await s.call_tool_json("file_taxonomy_delete", {"id": f"tag:{name}"})

    # -- add
    added = await s.call_tool_json("file_taxonomy_add", {
        "kind": "tag", "name": TAG,
        "description": "E2E test tag for the zanzibar lighthouse survey",
        "aliases": "zanzibar-survey",
    })
    assert not added.get("error"), added
    assert added["id"] == TAG_ID
    assert added["kind"] == "tag" and added["name"] == TAG
    assert added["status"] == "active"

    # duplicate add must be rejected
    dup = await s.call_tool_json("file_taxonomy_add", {
        "kind": "tag", "name": TAG, "description": "dup"})
    assert dup.get("error") and dup.get("code") == "duplicate", dup

    # -- get
    got = await s.call_tool_json("file_taxonomy_get", {"id": TAG_ID})
    assert got["id"] == TAG_ID
    assert got["description"].startswith("E2E test tag"), got
    assert got["aliases"] == "zanzibar-survey"

    # -- search (semantic over sim embeddings + name matching)
    found = await s.call_tool_json("file_taxonomy_search", {
        "query": "zanzibar", "kind": "tag", "top_k": 5})
    assert isinstance(found, list), found
    assert any(e.get("id") == TAG_ID for e in found), found

    # -- list
    listed = await s.call_tool_json("file_taxonomy_list", {"kind": "tag"})
    assert isinstance(listed, list), listed
    assert TAG_ID in [e.get("id") for e in listed]

    # -- update
    updated = await s.call_tool_json("file_taxonomy_update", {
        "id": TAG_ID, "description": "Updated by the e2e suite", "status": "archived"})
    assert not updated.get("error"), updated
    assert updated["description"] == "Updated by the e2e suite"
    assert updated["status"] == "archived"
    # archived entries drop out of the default (active) listing
    active = await s.call_tool_json("file_taxonomy_list", {"kind": "tag"})
    assert TAG_ID not in [e.get("id") for e in active]

    # -- import: deprecated contract (see module docstring)
    imported = await s.call_tool_json("file_taxonomy_import", {})
    assert imported.get("error") is True, imported
    assert imported.get("code") == "deprecated", imported

    # -- bulk add 2 terms (replacement for the deprecated import path)
    for name in BULK:
        row = await s.call_tool_json("file_taxonomy_add", {
            "kind": "tag", "name": name, "description": f"bulk term {name}"})
        assert row.get("id") == f"tag:{name}", row
    listed = await s.call_tool_json("file_taxonomy_list", {"kind": "tag"})
    ids = [e.get("id") for e in listed]
    assert all(f"tag:{n}" in ids for n in BULK), ids

    # -- delete everything, list confirms deletion
    for name in (TAG, *BULK):
        deleted = await s.call_tool_json("file_taxonomy_delete", {"id": f"tag:{name}"})
        assert deleted.get("deleted") is True, deleted
    remaining = await s.call_tool_json("file_taxonomy_list", {"kind": "tag", "status": "active"})
    remaining_ids = [e.get("id") for e in remaining]
    assert TAG_ID not in remaining_ids
    assert all(f"tag:{n}" not in remaining_ids for n in BULK)
    gone = await s.call_tool_json("file_taxonomy_get", {"id": TAG_ID})
    assert gone.get("error") and gone.get("code") == "not_found", gone
