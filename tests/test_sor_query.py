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
