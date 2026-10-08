"""Tests for the sources: backward-compat shim in core/config.py."""

from pathlib import Path

import pytest
import yaml

from core.config import load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    """Write a YAML config file AND create any referenced paths so load_config
    validation doesn't reject it."""
    p = tmp_path / "test_config.yaml"
    # Ensure documents_root / vault_root / source roots exist on disk
    roots = []
    if "documents_root" in data:
        roots.append(data["documents_root"])
    if "vault_root" in data:
        roots.append(data["vault_root"])
    for src in data.get("sources", []):
        if src.get("type") == "filesystem" and "root" in src:
            roots.append(src["root"])
    for r in roots:
        Path(r).mkdir(parents=True, exist_ok=True)
    data.setdefault("index_root", str(tmp_path / "index"))
    Path(data["index_root"]).mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))
    return p


def test_old_style_config_synthesizes_single_filesystem_source(tmp_path):
    """documents_root: /path expands to sources: [{type: filesystem, name: documents, root: /path}]."""
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
    })
    cfg = load_config(str(cfg_path))
    assert "sources" in cfg
    assert len(cfg["sources"]) == 1
    src = cfg["sources"][0]
    assert src["type"] == "filesystem"
    assert src["name"] == "documents"
    assert src["root"] == str(tmp_path / "vault")


def test_new_style_config_loads_sources_as_is(tmp_path):
    """New-style sources: list is returned unchanged (preserving order, types, and names)."""
    cfg_path = _write_config(tmp_path, {
        "sources": [
            {"type": "filesystem", "name": "docs", "root": str(tmp_path / "vault")},
            {"type": "postgres", "name": "comm", "dsn": "postgresql://...", "tables": []},
        ],
    })
    cfg = load_config(str(cfg_path))
    assert len(cfg["sources"]) == 2
    assert cfg["sources"][0]["name"] == "docs"
    assert cfg["sources"][1]["type"] == "postgres"


def test_mixing_old_and_new_style_is_an_error(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "sources": [{"type": "filesystem", "name": "docs", "root": str(tmp_path / "vault")}],
    })
    with pytest.raises(ValueError, match="Cannot use both.*documents_root.*sources"):
        load_config(str(cfg_path))


def test_backward_compat_preserves_scan_include_exclude(tmp_path):
    """scan.include/exclude at the top level should flow into the synthesized source."""
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "scan": {
            "include": ["**/*.md", "**/*.pdf"],
            "exclude": ["**/.git/**"],
        },
    })
    cfg = load_config(str(cfg_path))
    src = cfg["sources"][0]
    assert src["scan"]["include"] == ["**/*.md", "**/*.pdf"]
    assert src["scan"]["exclude"] == ["**/.git/**"]


def test_communication_context_defaults(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
    })

    cfg = load_config(str(cfg_path))

    assert cfg["communication_context"] == {
        "enabled": True,
        "window_before": 5,
        "window_after": 5,
        "max_time_window_minutes": 15,
        "same_channel_only": True,
        "include_batch": True,
    }


def test_communication_context_must_be_mapping(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "communication_context": "enabled",
    })

    with pytest.raises(ValueError, match="communication_context must be a mapping"):
        load_config(str(cfg_path))


def test_communication_context_window_must_be_non_negative(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "communication_context": {"window_before": -1},
    })

    with pytest.raises(
        ValueError,
        match="communication_context.window_before must be a non-negative integer",
    ):
        load_config(str(cfg_path))


def test_communication_context_same_channel_only_must_remain_true(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "communication_context": {"same_channel_only": False},
    })

    with pytest.raises(
        ValueError,
        match="communication_context.same_channel_only must remain true",
    ):
        load_config(str(cfg_path))


def test_project_comm_messages_config_example_exports_source_channel_id():
    config_example = Path("config.yaml.example").read_text()

    assert "c.source_channel_id" in config_example
    assert "c.name AS channel_name" in config_example
    assert (
        "metadata_columns: [source, source_message_id, source_channel_id, "
        "channel_name, sender, subject, sent_at, direction, thread_id]"
    ) in config_example


def test_project_comm_messages_config_example_scopes_a_phone_line_by_counterparty():
    """A Quo line is shared by everyone who texts it: its conversation is the line plus
    the counterparty (incoming: from; outgoing: to), so conversation context never
    mixes two people's texts (2026-10-07, PFG collection ticket 103)."""
    config_example = " ".join(Path("config.yaml.example").read_text().replace("# ", "").split())

    assert (
        "CASE WHEN m.source IN ('quo', 'openphone') THEN "
        "CASE r.payload->'data'->'object'->>'direction' "
        "WHEN 'outgoing' THEN CASE jsonb_typeof(r.payload->'data'->'object'->'to') "
        "WHEN 'array' THEN r.payload->'data'->'object'->'to'->>0 "
        "ELSE r.payload->'data'->'object'->>'to' END "
        "ELSE r.payload->'data'->'object'->>'from' END "
        "END AS thread_id"
    ) in config_example


def test_project_comm_messages_config_example_indexes_subject_and_body():
    uncommented = Path("config.yaml.example").read_text().replace("# ", "")
    config_example = " ".join(uncommented.split())

    assert (
        "CASE "
        "WHEN NULLIF(BTRIM(COALESCE(m.subject, '')), '') IS NULL THEN m.body "
        "WHEN NULLIF(BTRIM(COALESCE(m.body, '')), '') IS NULL "
        "THEN 'Subject: ' || BTRIM(m.subject) "
        "ELSE 'Subject: ' || BTRIM(m.subject) "
        "|| E'\\n\\nBody:\\n' || m.body "
        "END AS _text"
    ) in config_example
    assert (
        "WHERE ( "
        "NULLIF(BTRIM(COALESCE(m.subject, '')), '') IS NOT NULL "
        "OR NULLIF(BTRIM(COALESCE(m.body, '')), '') IS NOT NULL "
        ") AND m.canonical_message_id IS NULL"
    ) in config_example


@pytest.mark.parametrize(
    "config_path", ["config.staging.yaml", "config.staging.realmedia.yaml"]
)
def test_staging_comm_messages_index_subject_and_body(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    postgres_source = next(
        source for source in config["sources"] if source["type"] == "postgres"
    )
    message_table = next(
        table
        for table in postgres_source["tables"]
        if table["source_type"] == "pg_message"
    )
    query = " ".join(message_table["query"].split())

    assert "WHEN NULLIF(BTRIM(COALESCE(subject, '')), '') IS NULL THEN body" in query
    assert "THEN 'Subject: ' || BTRIM(subject)" in query
    assert "E'\\n\\nBody:\\n' || body" in query
    assert "NULLIF(BTRIM(COALESCE(subject, '')), '') IS NOT NULL" in query
    assert "NULLIF(BTRIM(COALESCE(body, '')), '') IS NOT NULL" in query
    assert "subject" in message_table["metadata_columns"]


def test_a_quo_message_source_without_thread_id_warns_while_context_is_on(tmp_path, caplog):
    """r6 review: counterparty-scoped context needs the comm_messages query to export
    thread_id; the live config did not, and nothing said so. Loading such a config
    warns (context falls back to unscoped_line), naming the source."""
    import logging

    quo_query = "SELECT m.source, m.source_message_id FROM messages m WHERE m.source IN ('quo', 'zoho_cliq')"
    table = {"source_type": "pg_message", "query": quo_query,
             "metadata_columns": ["source", "source_message_id", "sender", "direction"]}
    source = {"type": "postgres", "name": "comm_messages", "dsn": "postgres://x", "tables": [table]}
    cfg_path = _write_config(tmp_path, {"sources": [source]})
    with caplog.at_level(logging.WARNING, logger="core.config"):
        load_config(str(cfg_path))
    assert any("comm_messages" in r.getMessage() and "thread_id" in r.getMessage() for r in caplog.records)

    # The live query reads every source without naming quo; it exports source_channel_id.
    caplog.clear()
    table.update(query="SELECT m.source, c.source_channel_id FROM messages m",
                 metadata_columns=["source", "source_message_id", "source_channel_id", "sender"])
    cfg_path = _write_config(tmp_path, {"sources": [source]})
    with caplog.at_level(logging.WARNING, logger="core.config"):
        load_config(str(cfg_path))
    assert any("thread_id" in r.getMessage() for r in caplog.records)

    caplog.clear()
    table["metadata_columns"].append("thread_id")
    cfg_path = _write_config(tmp_path, {"sources": [source]})
    with caplog.at_level(logging.WARNING, logger="core.config"):
        load_config(str(cfg_path))
    assert not any("thread_id" in r.getMessage() for r in caplog.records)
