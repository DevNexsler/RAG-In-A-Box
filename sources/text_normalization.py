"""Configurable text normalization for database-backed sources."""

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Mapping


_CLIQ_MENTION_RE = re.compile(r"\{@([0-9]+)\}")
_CLIQ_SOURCE = "zoho_cliq"
_CLIQ_NORMALIZER_VERSION = "zoho_cliq_mentions:v1"
_HTML_MESSAGE_NORMALIZER_VERSION = "html_message:v1"
_HTML_HINT_RE = re.compile(
    r"<(?:!doctype|html|head|body|table|div|p|span|a\s|img\s|style|footer|nav)\b",
    re.IGNORECASE,
)
_HIDDEN_STYLE_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
    re.IGNORECASE,
)
_INVISIBLE_TEXT = str.maketrans("", "", "\u00ad\u034f\u200b\u200c\u200d\u2060\ufeff")


class _VisibleEmailHTMLParser(HTMLParser):
    """Extract visible semantic text without serializing HTML attributes."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div",
        "dl", "dt", "figcaption", "figure", "h1", "h2", "h3", "h4",
        "h5", "h6", "hr", "li", "main", "ol", "p", "pre", "section",
        "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
    _SUPPRESSED_TAGS = {
        "footer", "head", "nav", "noscript", "script", "style", "svg",
        "template",
    }
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._suppressed_depth = 0

    def _break(self) -> None:
        if self._pieces and self._pieces[-1] != "\n":
            self._pieces.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value or "" for key, value in attrs}
        hidden = (
            "hidden" in attributes
            or attributes.get("aria-hidden", "").lower() == "true"
            or bool(_HIDDEN_STYLE_RE.search(attributes.get("style", "")))
            or attributes.get("role", "").lower() in {"contentinfo", "navigation"}
        )
        suppress = self._suppressed_depth > 0 or tag in self._SUPPRESSED_TAGS or hidden
        if suppress:
            if tag not in self._VOID_TAGS:
                self._suppressed_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._suppressed_depth:
            self._suppressed_depth -= 1
            return
        if tag.lower() in self._BLOCK_TAGS:
            self._break()

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self._pieces.append(data)

    def text(self) -> str:
        visible = "".join(self._pieces).translate(_INVISIBLE_TEXT)
        lines: list[str] = []
        for line in visible.splitlines():
            collapsed = " ".join(line.split())
            if collapsed and (not lines or collapsed != lines[-1]):
                lines.append(collapsed)
        return "\n".join(lines)


@dataclass(frozen=True)
class NormalizedText:
    """Normalized source text plus change-detection and indexing decisions."""

    text: str
    change_hash_salt: str = ""
    should_index: bool = True


def normalize_source_text(source_type: str, text: str) -> NormalizedText:
    """Apply structural normalization implied by source record type.

    Message bodies may be plain text or HTML in the same database column. HTML
    is reduced to visible text so href/src/style payloads never become embedding
    input. Plain text remains byte-for-byte intact, including meaningful URLs.
    """
    if source_type != "pg_message" or not _HTML_HINT_RE.search(text):
        return NormalizedText(text)

    parser = _VisibleEmailHTMLParser()
    parser.feed(text)
    parser.close()
    return NormalizedText(
        parser.text(),
        change_hash_salt=_HTML_MESSAGE_NORMALIZER_VERSION,
    )


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
