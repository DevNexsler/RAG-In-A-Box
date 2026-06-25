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


def test_format_schema_text_quotes_camelcase():
    schema = {
        "Building Units": [("id", "integer"), ("Unit_Name", "text")],
        "collection_tickets": [("id", "integer"), ("Status", "text")],
    }
    txt = sor_query.format_schema_text(schema, ["Building Units", "collection_tickets"])
    assert '"Building Units"("id", "Unit_Name")' in txt
    assert 'collection_tickets("id", "Status")' in txt   # snake table unquoted


def test_get_sor_schema_caches(monkeypatch):
    calls = {"n": 0}

    class FakeCur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *_): calls["n"] += 1
        def fetchall(self):
            return [{"table_name": "Contacts", "column_name": "id",
                     "data_type": "integer"}]

    class FakeConn:
        def cursor(self): return FakeCur()

    monkeypatch.setattr(sor_query, "_get_readonly_conn", lambda: FakeConn())
    sor_query._SCHEMA_CACHE["data"] = None
    s1 = sor_query.get_sor_schema(refresh=True)
    s2 = sor_query.get_sor_schema()           # cached, no 2nd execute
    assert s1 == s2
    assert calls["n"] == 1
    assert s1["Contacts"] == [("id", "integer")]


def _fake_conn(rows):
    class FakeCur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *_): pass
        def fetchall(self): return rows
    class FakeConn:
        def cursor(self): return FakeCur()
        def rollback(self): pass
    return FakeConn()


def test_sor_query_impl_happy_path(monkeypatch):
    monkeypatch.setattr(sor_query, "_get_readonly_conn",
                        lambda: _fake_conn([{"id": 1, "Status": "Open"}]))
    out = sor_query.sor_query_impl('SELECT id, "Status" FROM "Collection Tickets"')
    assert out.splitlines()[0] == "id\tStatus"
    assert "1\tOpen" in out


def test_sor_query_impl_rejects_write(monkeypatch):
    out = sor_query.sor_query_impl("DELETE FROM x")
    assert out.startswith("ERROR")


def test_sor_query_impl_db_error_is_returned(monkeypatch):
    import psycopg

    class BoomConn:
        def cursor(self): raise psycopg.Error("relation \"Nope\" does not exist")
        def rollback(self): pass

    monkeypatch.setattr(sor_query, "_get_readonly_conn", lambda: BoomConn())
    out = sor_query.sor_query_impl("SELECT * FROM \"Nope\"")
    assert "SOR query error" in out and "Nope" in out


def test_sor_schema_impl_unknown_table(monkeypatch):
    monkeypatch.setattr(sor_query, "get_sor_schema",
                        lambda **k: {"Contacts": [("id", "integer")]})
    out = sor_query.sor_schema_impl("Missing")
    assert "Unknown table" in out and "Contacts" in out


def test_build_description_falls_back_on_db_down(monkeypatch):
    def boom(**k): raise RuntimeError("db down")
    monkeypatch.setattr(sor_query, "get_sor_schema", boom)
    desc = sor_query.build_sor_query_description()
    assert "SELECT" in desc  # still returns usable static guidance
