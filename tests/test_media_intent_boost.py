"""Media-intent boost: surface attachments when the query names the medium.

Without this an attachment loses every top-k slot to the messages that quote its
conversation context (the message matches that text at ~1.0; the attachment
carries it inside a multi-KB describe).
"""
from core.storage import SearchHit
from search_hybrid import _apply_media_intent_boost, _ensure_media_intent_slots


def _hit(doc_id, score, source_type):
    return SearchHit(
        doc_id=doc_id, loc="c:0", snippet="", text="", score=score,
        source_type=source_type,
    )


def test_boost_lifts_video_above_quoting_message():
    msg = _hit("m1", 1.0, "pg_message")
    vid = _hit("v1", 0.8, "video")
    out = _apply_media_intent_boost([msg, vid], "163 Washington video walkthrough", weight=0.35)
    assert out[0].doc_id == "v1", "video should outrank the message that quotes it"
    assert msg.score == 1.0, "non-matching hits must be untouched"


def test_boost_targets_only_the_named_medium():
    img = _hit("i1", 0.8, "img")
    vid = _hit("v1", 0.8, "video")
    _apply_media_intent_boost([img, vid], "show me the photo", weight=0.5)
    assert img.score > 0.8, "img boosted for a photo query"
    assert vid.score == 0.8, "video untouched for a photo query"


def test_no_media_term_is_a_noop():
    msg = _hit("m1", 1.0, "pg_message")
    vid = _hit("v1", 0.8, "video")
    out = _apply_media_intent_boost([msg, vid], "163 Washington rent adjustment", weight=0.35)
    assert [h.doc_id for h in out] == ["m1", "v1"]
    assert vid.score == 0.8


def test_zero_weight_disables():
    vid = _hit("v1", 0.8, "video")
    _apply_media_intent_boost([vid], "video walkthrough", weight=0.0)
    assert vid.score == 0.8


def test_empty_hits_safe():
    assert _apply_media_intent_boost([], "video", weight=0.35) == []


# --- media-intent quota: guarantee inclusion when boosting can't win ---------


def test_quota_injects_video_when_messages_sweep_every_slot():
    """The prod failure: ten message hits, the video buried in the pool."""
    hits = [_hit(f"m{i}", 1.0 - i * 0.05, "pg_message") for i in range(10)]
    pool = hits + [_hit("v1", 0.31, "video"), _hit("v2", 0.20, "video")]
    out = _ensure_media_intent_slots(hits, pool, "163 Washington video walkthrough", min_slots=2)
    ids = [h.doc_id for h in out]
    assert "v1" in ids and "v2" in ids, "the named medium must be represented"
    assert len(out) == 10, "quota must not inflate the result count"
    assert ids[0] == "m0", "the strongest hits are preserved"


def test_quota_is_satisfied_and_noop_when_medium_already_present():
    hits = [_hit("v1", 0.9, "video"), _hit("v2", 0.8, "video"), _hit("m1", 0.7, "pg_message")]
    pool = hits + [_hit("v3", 0.1, "video")]
    out = _ensure_media_intent_slots(hits, pool, "video walkthrough", min_slots=2)
    assert [h.doc_id for h in out] == ["v1", "v2", "m1"], "already satisfied -> untouched"


def test_quota_ignores_queries_naming_no_medium():
    hits = [_hit(f"m{i}", 0.9, "pg_message") for i in range(3)]
    pool = hits + [_hit("v1", 0.5, "video")]
    out = _ensure_media_intent_slots(hits, pool, "163 Washington rent adjustment", min_slots=2)
    assert [h.doc_id for h in out] == ["m0", "m1", "m2"], "no medium named -> no injection"


def test_quota_no_candidate_of_that_type_is_safe():
    hits = [_hit("m1", 0.9, "pg_message")]
    out = _ensure_media_intent_slots(hits, list(hits), "video walkthrough", min_slots=2)
    assert [h.doc_id for h in out] == ["m1"], "nothing to inject -> unchanged"


def test_quota_does_not_duplicate_an_already_returned_hit():
    vid = _hit("v1", 0.5, "video")
    hits = [_hit("m1", 0.9, "pg_message"), vid]
    pool = [vid, _hit("v2", 0.4, "video")]
    out = _ensure_media_intent_slots(hits, pool, "video", min_slots=2)
    assert [h.doc_id for h in out].count("v1") == 1, "no duplicates"
    assert "v2" in [h.doc_id for h in out]


def test_quota_zero_slots_disables():
    hits = [_hit("m1", 0.9, "pg_message")]
    pool = hits + [_hit("v1", 0.5, "video")]
    assert _ensure_media_intent_slots(hits, pool, "video", min_slots=0) == hits
