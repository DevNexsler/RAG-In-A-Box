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
