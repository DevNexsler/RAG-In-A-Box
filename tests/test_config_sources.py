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
        "channel_name, sender, sent_at]"
    ) in config_example
