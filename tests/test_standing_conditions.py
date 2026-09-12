"""Tests for the standing-condition announcer (#2101).

A state that no run can change must be reported when it appears, changes or
clears — and on a periodic digest so it never disappears — but not on every
run, or it drowns the level it is reported at (117 of 118 production ERROR
lines in one 24h window were a single unchanged condition).
"""

import json

from core import standing_conditions as sc


def _observe(state, members, **kwargs):
    return sc.observe(state, "degraded_capped", members, **kwargs)


def test_first_observation_announces_and_stamps_first_seen():
    state, announcement = _observe({}, {"documents::001Og": ["blocked"]}, now=100.0)

    assert announcement.announce is True
    assert announcement.transition == "entered"
    assert announcement.first_seen_at == 100.0
    assert state["conditions"]["degraded_capped"]["member_count"] == 1


def test_unchanged_members_do_not_re_announce_and_do_not_rewrite_state():
    state, _ = _observe({}, {"documents::001Og": ["blocked"]}, now=100.0)

    next_state, announcement = _observe(
        state, {"documents::001Og": ["blocked"]}, now=200.0
    )

    assert announcement.announce is False
    assert announcement.transition == "standing"
    assert announcement.first_seen_at == 100.0
    assert next_state is state  # nothing to persist


def test_added_member_re_announces():
    state, _ = _observe({}, {"documents::001Og": ["blocked"]}, now=100.0)

    _, announcement = _observe(
        state,
        {"documents::001Og": ["blocked"], "documents::001Oo": ["blocked"]},
        now=200.0,
    )

    assert announcement.announce is True
    assert announcement.transition == "changed"
    assert announcement.first_seen_at == 100.0  # the condition itself is older


def test_changed_values_re_announce_even_when_the_keys_are_identical():
    # {doc_id: [reasons]} and {reason: [doc_ids]} are both valid membership
    # shapes, so a change on either side has to count.
    state, _ = _observe({}, {"documents::001Og": ["blocked"]}, now=100.0)

    _, announcement = _observe(
        state, {"documents::001Og": ["blocked", "ocr_describe_failed"]}, now=200.0
    )

    assert announcement.announce is True
    assert announcement.transition == "changed"


def test_emptied_set_reports_cleared_once_then_stays_quiet():
    state, _ = _observe({}, {"documents::001Og": ["blocked"]}, now=100.0)

    state, announcement = _observe(state, {}, now=200.0)
    assert announcement.announce is False
    assert announcement.cleared is True
    assert announcement.transition == "cleared"
    assert state["conditions"] == {}

    _, announcement = _observe(state, {}, now=300.0)
    assert (announcement.announce, announcement.cleared) == (False, False)
    assert announcement.transition == "absent"


def test_unchanged_condition_is_re_announced_once_per_digest_interval():
    members = {"documents::001Og": ["blocked"]}
    state, _ = _observe({}, members, now=0.0)

    _, quiet = _observe(state, members, now=sc.DIGEST_INTERVAL_SECONDS - 1)
    assert quiet.announce is False

    state, digest = _observe(state, members, now=sc.DIGEST_INTERVAL_SECONDS)
    assert digest.announce is True
    assert digest.transition == "digest"

    # The digest restarts the window rather than firing every run afterwards.
    _, after = _observe(state, members, now=sc.DIGEST_INTERVAL_SECONDS + 1)
    assert after.announce is False


def test_announce_round_trips_through_the_state_file(tmp_path):
    members = {"documents::001Og": ["blocked"]}

    assert sc.announce(tmp_path, "degraded_capped", members, now=100.0).announce
    assert not sc.announce(tmp_path, "degraded_capped", members, now=200.0).announce

    payload = json.loads(sc.state_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["conditions"]["degraded_capped"]["first_seen_at"] == 100.0


def test_conditions_are_independent_of_each_other(tmp_path):
    assert sc.announce(tmp_path, "degraded_capped", {"a": ["x"]}, now=1.0).announce
    assert sc.announce(tmp_path, "actionable_skips", {"b": ["y"]}, now=1.0).announce
    assert not sc.announce(tmp_path, "degraded_capped", {"a": ["x"]}, now=2.0).announce


def test_unreadable_state_never_silences_the_condition(tmp_path):
    sc.state_path(tmp_path).write_text("{ not json", encoding="utf-8")

    assert sc.announce(tmp_path, "degraded_capped", {"a": ["x"]}, now=1.0).announce


def test_unwritable_state_keeps_announcing_rather_than_going_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "save", lambda *_args, **_kwargs: False)

    first = sc.announce(tmp_path, "degraded_capped", {"a": ["x"]}, now=1.0)
    second = sc.announce(tmp_path, "degraded_capped", {"a": ["x"]}, now=2.0)

    assert first.announce and second.announce


def test_announcement_is_falsy_when_there_is_nothing_to_report(tmp_path):
    assert not sc.announce(tmp_path, "degraded_capped", {}, now=1.0)


def test_a_member_without_reasons_is_still_a_member(tmp_path):
    # The flow builds {doc_id: reasons}; a ledger entry with no recorded
    # reasons is still a parked document and must not be dropped.
    assert sc.announce(tmp_path, "degraded_capped", {"documents::a": []}, now=1.0)
