import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def _base_config(tmp_path: Path, dedupe_block: str = "") -> Path:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    return _write_config(
        tmp_path,
        f"""
documents_root: "{docs_root}"
index_root: "{tmp_path / 'index'}"
{dedupe_block}
""".strip()
        + "\n",
    )


def test_load_config_sets_dedupe_defaults(tmp_path):
    config = load_config(_base_config(tmp_path))

    assert config["dedupe"]["enabled"] is False
    assert config["dedupe"]["mode"] == "exact"
    assert config["dedupe"]["hash_algo"] == "blake3"
    assert isinstance(config["dedupe"]["archive_root"], str)
    assert config["dedupe"]["archive_root"] == str(tmp_path / "index" / "duplicates")
    assert config["dedupe"]["archive_duplicates"] is True
    assert config["dedupe"]["update_canonical_metadata"] is True
    assert config["dedupe"]["skip_duplicate_indexing"] is True


def test_load_config_uses_index_root_env_override_for_default_archive_root(tmp_path, monkeypatch):
    override_root = tmp_path / "index-from-env"
    monkeypatch.setenv("INDEX_ROOT", str(override_root))

    config = load_config(_base_config(tmp_path))

    assert config["index_root"] == str(override_root)
    assert config["dedupe"]["archive_root"] == str(override_root / "duplicates")


@pytest.mark.parametrize(
    ("dedupe_block", "match"),
    [
        ("dedupe:\n  enabled: 'false'\n", "dedupe.enabled"),
        ("dedupe:\n  mode: fuzzy\n", "dedupe.mode"),
        ("dedupe:\n  hash_algo: sha256\n", "dedupe.hash_algo"),
        ("dedupe:\n  archive_duplicates: 'false'\n", "dedupe.archive_duplicates"),
        ("dedupe:\n  update_canonical_metadata: false\n", "dedupe.update_canonical_metadata"),
        ("dedupe:\n  skip_duplicate_indexing: false\n", "dedupe.skip_duplicate_indexing"),
    ],
)
def test_load_config_rejects_invalid_dedupe_values(tmp_path, dedupe_block, match):
    with pytest.raises(ValueError, match=match):
        load_config(_base_config(tmp_path, dedupe_block))


def test_load_config_allows_archive_duplicates_false(tmp_path):
    config = load_config(
        _base_config(
            tmp_path,
            "dedupe:\n  archive_duplicates: false\n",
        )
    )

    assert config["dedupe"]["archive_duplicates"] is False


def test_load_config_rejects_nonsensical_archive_root_type(tmp_path):
    with pytest.raises(ValueError, match="dedupe.archive_root"):
        load_config(
            _base_config(
                tmp_path,
                "dedupe:\n  archive_root: 123\n",
            )
        )


def test_load_config_rejects_blank_archive_root(tmp_path):
    with pytest.raises(ValueError, match="dedupe.archive_root"):
        load_config(
            _base_config(
                tmp_path,
                "dedupe:\n  archive_root: ''\n",
            )
        )
