"""Deterministic credential detection and redaction for indexable content.

Findings intentionally contain only categories and offsets. Secret values must
never enter logs, reports, or audit output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


REDACTION_MARKER = "[REDACTED CREDENTIAL]"


@dataclass(frozen=True)
class SensitiveFinding:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class SensitiveContentDecision:
    text: str
    metadata: dict[str, Any]
    finding_kinds: tuple[str, ...]
    quarantine: bool


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            r".*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        # Explicit header form. The key names the credential, so the value needs
        # no further shape test.
        "authorization_header",
        re.compile(
            r"(?i)\b(?:proxy-)?authorization\s*:\s*"
            r"(?:bearer|basic)?\s*[A-Za-z0-9._~+/=-]{16,}"
        ),
    ),
    (
        # Bare "Bearer <token>" / "Basic <token>" written inline in prose. Taking
        # any 16+ token characters here redacts ordinary English, because "/" and
        # "-" are token characters: live index rows contained "Can maintenance
        # complete the basic entry-point/conditions" and "basic bedroom/bathroom"
        # — 47 such matches over 25 documents, not one a credential. So the value
        # must look like a token rather than words: either it carries a digit or
        # base64 punctuation, or it is one unbroken mixed-case run.
        #
        # Residual gap, accepted knowingly: an all-lowercase, digit-free,
        # separator-free value (e.g. "bearer abcdefghijklmnopq") is not matched,
        # because that is indistinguishable from a long word. The explicit
        # "Authorization:" form above catches it whenever the header names it,
        # and secret_assignment catches "token=..." style.
        "authorization_header",
        re.compile(
            r"(?i)\b(?:bearer|basic)\s+(?:"
            r"(?=[A-Za-z0-9._~+/=-]*[0-9+=_])[A-Za-z0-9._~+/=-]{16,}"
            r"|(?-i:(?=[A-Za-z]*[a-z])(?=[A-Za-z]*[A-Z])[A-Za-z]{16,})"
            r")"
        ),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{30,255})\b"),
    ),
    (
        "provider_api_key",
        re.compile(r"\b(?:sk[-_](?:proj[-_])?[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "expiring_token",
        re.compile(
            r"\b[A-Z][A-Z0-9_-]{2,32}:[A-Za-z0-9._~+/-]{12,}:"
            r"(?:expires|expiry|exp)=[^\s,;]+",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"refresh[_-]?token|session[_-]?token|client[_-]?secret|"
            r"password|passwd|secret)\s*[=:]\s*[\"']?"
            r"(?!\$\{|<|\[REDACTED)(?!changeme\b|example\b|placeholder\b)"
            r"[A-Za-z0-9._~+/%:@-]{8,}[\"']?",
        ),
    ),
)

_SENSITIVE_METADATA_KEYS = re.compile(
    r"(?i)(?:authorization|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|client[_-]?secret|password|passwd|secret)"
)
_SYSTEM_MARKER_KEYS = {
    "is_system",
    "system_message",
    "is_synthetic",
    "synthetic",
    "is_test",
    "test_message",
}
_SYSTEM_IDENTITY_RE = re.compile(
    r"(?i)^(?:system(?: message)?|automation|automated|bot|test|synthetic)$"
)


def find_sensitive_content(text: str) -> tuple[SensitiveFinding, ...]:
    """Return secret-safe finding metadata for recognized credential shapes."""
    findings = [
        SensitiveFinding(kind, match.start(), match.end())
        for kind, pattern in _PATTERNS
        for match in pattern.finditer(text or "")
    ]
    return tuple(sorted(findings, key=lambda item: (item.start, item.end, item.kind)))


def contains_sensitive_content(text: str) -> bool:
    return bool(find_sensitive_content(text))


def _redact_text(text: str) -> tuple[str, tuple[SensitiveFinding, ...]]:
    findings = find_sensitive_content(text)
    if not findings:
        return text, ()

    spans: list[tuple[int, int]] = []
    for finding in findings:
        if spans and finding.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], finding.end))
        else:
            spans.append((finding.start, finding.end))

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.extend((text[cursor:start], REDACTION_MARKER))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), findings


def redact_sensitive_text(text: str) -> str:
    """Replace recognized credentials while preserving surrounding content."""
    return _redact_text(text)[0]


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return (
        not normalized
        or normalized.startswith("${")
        or normalized in {"changeme", "example", "placeholder", "redacted"}
        or REDACTION_MARKER.lower() in normalized
    )


def _sanitize_value(value: Any, *, key_hint: str = "") -> tuple[Any, set[str]]:
    if isinstance(value, str):
        if (
            _SENSITIVE_METADATA_KEYS.fullmatch(key_hint)
            and len(value.strip()) >= 8
            and not _is_placeholder(value)
        ):
            return REDACTION_MARKER, {"sensitive_metadata_field"}
        redacted, findings = _redact_text(value)
        return redacted, {finding.kind for finding in findings}
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        kinds: set[str] = set()
        for key, nested in value.items():
            clean, nested_kinds = _sanitize_value(nested, key_hint=str(key))
            sanitized[str(key)] = clean
            kinds.update(nested_kinds)
        return sanitized, kinds
    if isinstance(value, list):
        sanitized_list = []
        kinds: set[str] = set()
        for nested in value:
            clean, nested_kinds = _sanitize_value(nested)
            sanitized_list.append(clean)
            kinds.update(nested_kinds)
        return sanitized_list, kinds
    if isinstance(value, tuple):
        clean, kinds = _sanitize_value(list(value))
        return tuple(clean), kinds
    return value, set()


def sanitize_metadata(metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], tuple[str, ...]]:
    clean, kinds = _sanitize_value(metadata or {})
    return clean, tuple(sorted(kinds))


def _is_system_message(metadata: Mapping[str, Any]) -> bool:
    for key, value in metadata.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in _SYSTEM_MARKER_KEYS and (
            value is True or str(value).strip().lower() in {"1", "true", "yes"}
        ):
            return True
        if normalized_key in {"sender", "author", "message_type", "kind"} and (
            _SYSTEM_IDENTITY_RE.fullmatch(str(value).strip()) is not None
        ):
            return True
    return False


def _is_credential_only(redacted_text: str) -> bool:
    remainder = redacted_text.replace(REDACTION_MARKER, " ")
    remainder = re.sub(
        r"(?i)\b(?:auth(?:entication)?|credential|password|secret|token|temporary|expires?)s?\b",
        " ",
        remainder,
    )
    return not re.sub(r"[^A-Za-z0-9]+", "", remainder)


def sanitize_sensitive_content(
    text: str,
    *,
    source_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> SensitiveContentDecision:
    """Apply one policy before enrichment, embedding, persistence, and hooks."""
    raw_metadata = metadata or {}
    clean_text, text_findings = _redact_text(text or "")
    clean_metadata, metadata_kinds = sanitize_metadata(raw_metadata)
    finding_kinds = tuple(
        sorted({finding.kind for finding in text_findings} | set(metadata_kinds))
    )
    quarantine = bool(finding_kinds) and source_type == "pg_message" and (
        _is_system_message(raw_metadata) or _is_credential_only(clean_text)
    )
    return SensitiveContentDecision(
        text=clean_text,
        metadata=clean_metadata,
        finding_kinds=finding_kinds,
        quarantine=quarantine,
    )
