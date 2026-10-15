"""The SOR live tests seed their database destructively (DROP TABLE, CREATE,
INSERT, ALTER) from SOR_TEST_DSN, outside any config the live preflight
checks. Their fixture must refuse a production database before it connects
(#4193). Production databases come from the production container
(scripts/live_preflight.production_storage), so these tests fake that.
"""
from unittest.mock import MagicMock

import pytest

import scripts.live_preflight as lp
import tests.test_sor_query_integration_live as sor_live
from tests.test_sor_query_integration_live import seeded  # noqa: F401  (fixture under test)

SCRATCH_DSN = "postgresql://sor:sor@localhost:15432/sor_scratch"
PROD_SOR_DSN = "postgresql://admin:secret@localhost:5432/sor"


@pytest.fixture
def connects(monkeypatch):
    """Every DSN the fixture connects to; the connection is a no-op fake."""
    calls = []

    def fake_connect(dsn, *args, **kwargs):
        calls.append(dsn)
        return MagicMock()

    monkeypatch.setattr("psycopg.connect", fake_connect)
    return calls


def _production(monkeypatch, *databases):
    storage = lp.ProductionStorage(paths=frozenset(), volumes=frozenset(),
                                   databases=frozenset(databases))
    monkeypatch.setattr(lp, "production_storage", lambda: storage)


def test_seeding_refuses_a_production_database(monkeypatch, connects, request):
    _production(monkeypatch, "sor", "comm_data_store")
    monkeypatch.setattr(sor_live, "DSN", PROD_SOR_DSN)

    with pytest.raises(pytest.fail.Exception) as refused:
        request.getfixturevalue("seeded")

    assert connects == []
    assert "SOR_TEST_DSN" in str(refused.value) and "'sor'" in str(refused.value)
    assert "secret" not in str(refused.value)  # never print DSN credentials


def test_seeding_fails_closed_when_production_is_unknown(monkeypatch, connects, request):
    def no_container():
        raise RuntimeError("cannot inspect production container doc-organizer")

    monkeypatch.setattr(lp, "production_storage", no_container)
    monkeypatch.setattr(sor_live, "DSN", SCRATCH_DSN)

    with pytest.raises(RuntimeError, match="doc-organizer"):
        request.getfixturevalue("seeded")

    assert connects == []


def test_seeding_runs_against_a_scratch_database(monkeypatch, connects, request):
    _production(monkeypatch, "sor", "comm_data_store")
    monkeypatch.setattr(sor_live, "DSN", SCRATCH_DSN)

    request.getfixturevalue("seeded")

    assert connects == [SCRATCH_DSN]
