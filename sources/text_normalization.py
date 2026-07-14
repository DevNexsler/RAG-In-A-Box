"""Configurable text normalization for database-backed sources."""

from dataclasses import dataclass
import re
from typing import Mapping


_CLIQ_MENTION_RE = re.compile(r"\{@([0-9]+)\}")
_CLIQ_SOURCE = "zoho_cliq"
_CLIQ_NORMALIZER_VERSION = "zoho_cliq_mentions:v1"


@dataclass(frozen=True)
class NormalizedText:
    """Normalized source text plus change-detection and indexing decisions."""

    text: str
    change_hash_salt: str = ""
    should_index: bool = True


class ZohoCliqMentionNormalizer:
    """Expand Cliq mention tokens using Comm-Data-Store participants."""

    def __init__(self, display_names: Mapping[str, str]):
        self._display_names = {
            str(participant_key): str(display_name).strip()
            for participant_key, display_name in display_names.items()
            if str(display_name).strip()
        }

    @classmethod
    def from_connection(cls, connection) -> "ZohoCliqMentionNormalizer":
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT participant_key, display_name
                FROM participants
                WHERE source = 'zoho_cliq'
                  AND participant_key IS NOT NULL
                  AND NULLIF(BTRIM(display_name), '') IS NOT NULL
                """
            )
            display_names = {
                str(row["participant_key"]): str(row["display_name"])
                for row in cursor
            }
        return cls(display_names)

    def normalize(self, text: str, row: Mapping) -> NormalizedText:
        if str(row.get("source") or "").lower() != _CLIQ_SOURCE:
            return NormalizedText(text)

        matches = list(_CLIQ_MENTION_RE.finditer(text))
        if not matches:
            return NormalizedText(text)

        text_without_mentions = _CLIQ_MENTION_RE.sub("", text)
        if not any(character.isalnum() for character in text_without_mentions):
            return NormalizedText(
                "",
                change_hash_salt=_CLIQ_NORMALIZER_VERSION,
                should_index=False,
            )

        def replace(match: re.Match[str]) -> str:
            display_name = self._display_names.get(match.group(1))
            return f"@{display_name}" if display_name else match.group(0)

        return NormalizedText(
            _CLIQ_MENTION_RE.sub(replace, text),
            change_hash_salt=_CLIQ_NORMALIZER_VERSION,
        )


def build_text_normalizer(name: str | None, connection):
    """Build configured normalizer, rejecting unknown names early."""
    if not name:
        return None
    if name == "zoho_cliq_mentions":
        return ZohoCliqMentionNormalizer.from_connection(connection)
    raise ValueError(f"Unknown postgres text_normalizer: {name!r}")
