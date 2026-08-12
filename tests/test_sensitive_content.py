import pytest

from core.sensitive_content import (
    REDACTION_MARKER,
    contains_sensitive_content,
    sanitize_sensitive_content,
)


@pytest.mark.parametrize(
    "credential",
    [
        "AES-PRERELEASE:synthetic-token-0123456789abcdef:expires=2099-01-01T00:00:00Z",
        "Authorization: Bearer syntheticBearerToken0123456789abcdef",
        "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB",
        "github_pat_11AA0synthetic0123456789abcdefghijklmnopqrstuvwxyz",
        "sk-proj-synthetic0123456789abcdefghijklmnopqrstuvwxyz",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.synthetic-signature-value",
        "api_key=synthetic-secret-value-0123456789",
        "https://example.test/reset?access_token=synthetic-reset-token-0123456789",
        "-----BEGIN PRIVATE KEY-----\nc3ludGhldGljLWtleS1tYXRlcmlhbA==\n-----END PRIVATE KEY-----",
    ],
)
def test_common_credential_shapes_are_redacted(credential):
    decision = sanitize_sensitive_content(
        f"Migration note: {credential}",
        source_type="pg_message",
        metadata={"sender": "Pat"},
    )

    assert credential not in decision.text
    assert REDACTION_MARKER in decision.text
    assert decision.finding_kinds
    assert decision.quarantine is False


def test_sensitive_nested_metadata_is_redacted_without_changing_normal_values():
    secret = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB"
    decision = sanitize_sensitive_content(
        "Routine message",
        source_type="pg_message",
        metadata={
            "sender": "Pat",
            "headers": {"authorization": f"Bearer {secret}"},
            "labels": ["permit", secret],
            "sequence": 42,
        },
    )

    assert secret not in str(decision.metadata)
    assert decision.metadata["sender"] == "Pat"
    assert decision.metadata["labels"][0] == "permit"
    assert decision.metadata["sequence"] == 42


def test_system_or_credential_only_messages_are_quarantined():
    credential = "AES-PRERELEASE:synthetic-token-0123456789abcdef:expires=2099-01-01"

    system = sanitize_sensitive_content(
        f"Credential: {credential}",
        source_type="pg_message",
        metadata={"sender": "System"},
    )
    credential_only = sanitize_sensitive_content(
        credential,
        source_type="pg_message",
        metadata={"sender": "Pat"},
    )

    assert system.quarantine is True
    assert credential_only.quarantine is True


def test_normal_operational_text_is_not_changed_or_flagged():
    text = "Permit 24-0198 moved to Thursday. Call 555-0102 at 10:30 AM."
    decision = sanitize_sensitive_content(
        text,
        source_type="pg_message",
        metadata={"sender": "Pat", "channel_name": "Operations"},
    )

    assert decision.text == text
    assert decision.metadata == {"sender": "Pat", "channel_name": "Operations"}
    assert decision.finding_kinds == ()
    assert decision.quarantine is False
    assert contains_sensitive_content(text) is False
