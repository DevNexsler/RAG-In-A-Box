from unittest.mock import patch

import sor_query


def test_validate_accepts_select():
    assert sor_query.validate_select("SELECT 1") is None
    assert sor_query.validate_select("  select * from \"Contacts\"  ") is None


def test_validate_accepts_with_cte():
    assert sor_query.validate_select("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_validate_rejects_non_select():
    assert sor_query.validate_select("UPDATE t SET a=1") is not None
    assert sor_query.validate_select("") is not None
    assert sor_query.validate_select("   ") is not None


def test_wrap_adds_outer_limit_and_returns_effective():
    wrapped, eff = sor_query.wrap_with_limit("SELECT * FROM \"Contacts\"", 50)
    assert eff == 50
    assert "LIMIT 51" in wrapped          # fetch n+1 to detect truncation
    assert "_sub" in wrapped


def test_wrap_clamps_to_hard_max():
    _, eff = sor_query.wrap_with_limit("SELECT 1", 99999)
    assert eff == sor_query.HARD_MAX_ROWS


def test_wrap_floor_of_one():
    _, eff = sor_query.wrap_with_limit("SELECT 1", 0)
    assert eff == 1


def test_serialize_tsv_basic():
    rows = [{"id": 1, "name": "Rosado"}, {"id": 2, "name": "Fedak"}]
    out = sor_query.serialize(rows, "tsv", eff=50)
    assert out.splitlines()[0] == "id\tname"
    assert "1\tRosado" in out
    assert "of >" not in out             # not truncated


def test_serialize_truncation_notice():
    rows = [{"id": i} for i in range(51)]   # eff+1 fetched
    out = sor_query.serialize(rows, "tsv", eff=50)
    assert out.startswith("[50 of >50 rows")
    assert len(out.splitlines()) == 1 + 1 + 50   # notice + header + 50 rows


def test_serialize_cell_capping():
    rows = [{"notes": "x" * 1000}]
    out = sor_query.serialize(rows, "tsv", eff=50, cell_cap=100)
    assert "…" in out
    assert "x" * 1000 not in out


def test_serialize_json_format():
    import json
    rows = [{"id": 1, "name": "Rosado"}]
    out = sor_query.serialize(rows, "json", eff=50)
    assert json.loads(out) == [{"id": 1, "name": "Rosado"}]


def test_serialize_empty():
    assert "0 rows" in sor_query.serialize([], "tsv", eff=50)


def test_resolve_dsn_from_config_env(monkeypatch):
    monkeypatch.setenv("SOR_DSN", "postgresql://u@localhost/sor")
    fake = {"sources": [{"type": "postgres", "name": "sor", "dsn": "${SOR_DSN}"}]}
    with patch("sor_query.load_config", return_value=fake):
        assert sor_query.resolve_sor_dsn() == "postgresql://u@localhost/sor"


def test_resolve_dsn_missing_raises(monkeypatch):
    monkeypatch.delenv("SOR_DSN", raising=False)
    with patch("sor_query.load_config", return_value={"sources": []}):
        import pytest
        with pytest.raises(ValueError):
            sor_query.resolve_sor_dsn()
