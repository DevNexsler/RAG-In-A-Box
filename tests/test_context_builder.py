import pytest
import context_builder as cb


def test_normalize_contact():
    c = cb.normalize_contact(email="A@B.com", phone="(484) 761-4094",
                             name=" Jess ", lead_id=2444206,
                             latest_inbound_at="2026-08-27T14:00:00Z")
    assert c == {"email": "a@b.com", "phone_e164": "+14847614094",
                 "name": "Jess", "lead_id": "2444206",
                 "latest_inbound_at": "2026-08-27T14:00:00Z"}


def test_normalize_requires_identifier():
    with pytest.raises(ValueError):
        cb.normalize_contact(phone="123")  # 3 digits is not a phone


def test_exact_hit_filters_fuzzy_false_positive():
    contact = {"email": "jessbrown816@gmail.com", "phone_e164": "+14847614094",
               "name": "Jessica Ann Brown"}
    leslie = {"sender": "Leslie Huberheide", "channel": "+16107095575",
              "snippet": "415leslie@gmail.com"}
    jess = {"sender": "Jessica Ann Brown", "channel": "+14847614094",
            "snippet": "tour request"}
    assert cb.exact_hit(leslie, contact) is False
    assert cb.exact_hit(jess, contact) is True


def test_exact_hit_rejects_cross_field_digit_concatenation():
    # No single field contains the contact's full phone digits, but the OLD
    # concatenated-haystack implementation joined "channel"'s digits directly
    # against "snippet"'s digits and spelled out the target number across the
    # seam ("1484761" + "4094" == "14847614094") -- a false positive no field
    # actually carries.
    contact = {"email": None, "phone_e164": "+14847614094", "name": None}
    hit = {"sender": "Ops Bot", "channel": "+1484761", "snippet": "ext 4094 for billing"}
    assert cb.exact_hit(hit, contact) is False


def test_derived_flag_true_false_unknown():
    contact = {"latest_inbound_at": "2026-08-27T14:00:00Z"}
    ok_after = {"status": "ok", "latest_inbound_at": None, "outbound_evidence":
                [{"at": "2026-08-27T15:00:00+00:00"}]}
    ok_before = {"status": "ok", "latest_inbound_at": None, "outbound_evidence":
                 [{"at": "2026-08-27T10:00:00+00:00"}]}
    err = {"status": "error:db down", "outbound_evidence": []}
    assert cb.derive_flags(contact, ok_after)["our_outbound_after_latest_inbound"] is True
    assert cb.derive_flags(contact, ok_before)["our_outbound_after_latest_inbound"] is False
    assert cb.derive_flags(contact, err)["our_outbound_after_latest_inbound"] == "unknown"


def test_build_context_degrades_per_source():
    contact = cb.normalize_contact(email="a@b.com")
    deps = {"factbook": lambda c: (_ for _ in ()).throw(RuntimeError("down")),
            "cds": lambda c: {"status": "ok", "inbound_count_30d": 1,
                               "latest_inbound_at": "2026-08-27T14:00:00Z",
                               "latest_outbound_at": None, "outbound_evidence": []},
            "comm": lambda c: {"status": "no_exact_hit", "hits": []}}
    out = cb.build_context(contact, deps)
    assert out["factbook"]["status"].startswith("error:")
    assert out["cds"]["status"] == "ok"
    assert out["derived"]["our_outbound_after_latest_inbound"] is False
    assert "elapsed_ms" in out and out["contact"]["email"] == "a@b.com"
