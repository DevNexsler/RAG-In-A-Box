import os
import pytest

import sor_query

DSN = os.environ.get("SOR_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SOR_TEST_DSN not set")


@pytest.fixture()
def seeded(monkeypatch):
    import psycopg
    with psycopg.connect(DSN) as conn:
        conn.execute('DROP TABLE IF EXISTS "Buildings"')
        conn.execute('CREATE TABLE "Buildings" (id int primary key, "Nick_Name" text)')
        conn.execute('INSERT INTO "Buildings" VALUES (1, \'Carriage\'), (2, \'Linden\')')
        conn.commit()
    monkeypatch.setattr(sor_query, "resolve_sor_dsn", lambda *a, **k: DSN)
    sor_query._CONN = None
    sor_query._SCHEMA_CACHE["data"] = None
    yield


def test_readonly_blocks_writes(seeded):
    out = sor_query.sor_query_impl('SELECT 1; ', )  # validator catches multi/non-select first
    # A write that passes the SELECT check still fails at the DB (read-only txn):
    out = sor_query.sor_query_impl('WITH x AS (DELETE FROM "Buildings" RETURNING 1) SELECT * FROM x')
    assert "error" in out.lower()


def test_aggregation_and_count(seeded):
    out = sor_query.sor_query_impl('SELECT count(*) AS n FROM "Buildings"')
    assert "n" in out.splitlines()[0]
    assert "2" in out


def test_limit_truncation(seeded):
    out = sor_query.sor_query_impl('SELECT * FROM "Buildings"', limit=1)
    assert out.startswith("[1 of >1 rows")


def test_schema_reflects_new_column(seeded):
    sor_query.get_sor_schema(refresh=True)
    import psycopg
    with psycopg.connect(DSN) as conn:
        conn.execute('ALTER TABLE "Buildings" ADD COLUMN "Risk_Score" int')
        conn.commit()
    refreshed = sor_query.get_sor_schema(refresh=True)
    assert any(c == "Risk_Score" for c, _ in refreshed["Buildings"])
