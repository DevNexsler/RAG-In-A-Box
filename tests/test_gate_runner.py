# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
from scripts.gate import TIERS, next_tier_allowed


def test_tier_order():
    assert [t.name for t in TIERS] == ["static", "unit", "integration", "staging-e2e", "live"]


def test_fail_fast():
    results = {"static": True, "unit": False}
    assert next_tier_allowed("integration", results) is False


def test_live_requires_all_prior():
    results = {"static": True, "unit": True, "integration": True, "staging-e2e": True}
    assert next_tier_allowed("live", results) is True
