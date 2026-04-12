"""Tests for the sources.build_source dispatcher."""

import pytest

from sources import build_source
from sources.filesystem import FilesystemSource
from sources.postgres import PostgresSource


def test_dispatch_filesystem(tmp_path):
    from doc_id_store import DocIDStore

    reg = DocIDStore(tmp_path / "r.db")
    (tmp_path / "vault").mkdir()
    src = build_source({
        "type": "filesystem",
        "name": "docs",
        "root": str(tmp_path / "vault"),
        "scan": {"include": ["**/*.md"], "exclude": []},
    }, registry=reg)
    assert isinstance(src, FilesystemSource)
    assert src.name == "docs"


def test_dispatch_postgres():
    src = build_source({
        "type": "postgres",
        "name": "comm",
        "dsn": "postgresql://user:pass@host:5432/db",
        "tables": [],
    })
    assert isinstance(src, PostgresSource)
    assert src.name == "comm"


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source({"type": "slack", "name": "x"})


def test_missing_name_raises():
    with pytest.raises(KeyError):
        build_source({"type": "filesystem", "root": "/tmp"})
