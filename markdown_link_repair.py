"""Keep Markdown links valid when persistent Doc IDs rename vault files.

The document scanner appends ``@XXXXX@`` to filenames.  This module turns the
registry into an alias table from the old, human filename to the current one
and rewrites local Markdown and Obsidian links atomically.  Running it on every
full scan also repairs links left broken by older scanner versions.
"""

from __future__ import annotations

import bisect
import fnmatch
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote, unquote

from doc_id_store import extract_id_from_filename, strip_id_from_filename


_LINK_RE = re.compile(
    r"(?P<wiki_open>!?\[\[)(?P<wiki_body>[^\]\n]+)(?P<wiki_close>\]\])"
    r"|(?P<inline_prefix>!?\[[^\]\n]*\]\(\s*)"
    r"(?:(?P<inline_open><)(?P<inline_angle>[^>\n]+)(?P<inline_close>>)"
    r"|(?P<inline_plain>(?:\\.|[^)\s])+))"
    r"|(?P<reference_prefix>^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*)"
    r"(?:(?P<reference_open><)(?P<reference_angle>[^>\n]+)(?P<reference_close>>)"
    r"|(?P<reference_plain>\S+))",
    re.MULTILINE,
)
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class LinkRepairResult:
    files_changed: int = 0
    links_rewritten: int = 0
    changed_paths: tuple[str, ...] = ()


def _normal_rel(path: str) -> str | None:
    normalized = posixpath.normpath(path.replace("\\", "/")).lstrip("/")
    if normalized in ("", ".", "..") or normalized.startswith("../"):
        return None
    return normalized


def build_doc_id_aliases(
    vault_root: str | Path,
    registered_paths: Iterable[str],
    renamed_paths: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return unambiguous old-path -> current-path aliases.

    Registry aliases provide the self-heal path for historical breakage.
    Explicit renames from the current scan win because they are known to have
    happened successfully during this process.
    """
    root = Path(vault_root)
    candidates: dict[str, set[str]] = {}

    for raw_current in registered_paths:
        current = _normal_rel(raw_current)
        if current is None or extract_id_from_filename(posixpath.basename(current)) is None:
            continue
        try:
            if not (root / current).is_file():
                continue
        except OSError:
            continue
        clean_name = strip_id_from_filename(posixpath.basename(current))
        clean = posixpath.join(posixpath.dirname(current), clean_name)
        # A live tokenless file owns its own path; do not redirect it.
        try:
            if (root / clean).exists():
                continue
        except OSError:
            continue
        candidates.setdefault(clean, set()).add(current)

    aliases = {
        old: next(iter(currents))
        for old, currents in candidates.items()
        if len(currents) == 1
    }
    for raw_old, raw_new in (renamed_paths or {}).items():
        old = _normal_rel(raw_old)
        new = _normal_rel(raw_new)
        if old is not None and new is not None:
            aliases[old] = new
    return aliases


def _split_suffix(destination: str) -> tuple[str, str]:
    indexes = [i for marker in ("?", "#") if (i := destination.find(marker)) >= 0]
    if not indexes:
        return destination, ""
    index = min(indexes)
    return destination[:index], destination[index:]


def _decode_path(path: str) -> str:
    return unquote(path.replace("\\ ", " "))


def _render_path(path: str, original: str) -> str:
    if "%" in original:
        return quote(path, safe="/%:@-._~!$&'()*+,;=")
    if "\\ " in original:
        return path.replace(" ", "\\ ")
    return path


def _relative_target(source_rel: str, target_rel: str, original: str) -> str:
    if original.startswith("/"):
        return "/" + target_rel
    rendered = posixpath.relpath(target_rel, posixpath.dirname(source_rel) or ".")
    if original.startswith("./") and not rendered.startswith("."):
        rendered = "./" + rendered
    return rendered


def _resolve_standard_link(
    source_rel: str, destination: str, aliases: Mapping[str, str]
) -> str | None:
    path, suffix = _split_suffix(destination)
    if not path or path.startswith("//") or _SCHEME_RE.match(path):
        return None
    decoded = _decode_path(path)
    if decoded.startswith("/"):
        old_rel = _normal_rel(decoded)
    else:
        old_rel = _normal_rel(posixpath.join(posixpath.dirname(source_rel), decoded))
    if old_rel is None or old_rel not in aliases:
        return None
    replacement = _relative_target(source_rel, aliases[old_rel], decoded)
    return _render_path(replacement, path) + suffix


def _wiki_lookup(
    source_rel: str,
    raw_path: str,
    aliases: Mapping[str, str],
    basename_aliases: Mapping[str, str],
) -> tuple[str, str] | None:
    decoded = _decode_path(raw_path)
    source_dir = posixpath.dirname(source_rel)
    # An explicit attachment extension (``image.png``) is a real target as-is.
    # Also try ``.md`` because Obsidian hides that suffix from note links, so a
    # note named ``invoice.pdf.md`` is conventionally linked as ``invoice.pdf``.
    lookup_paths = [decoded]
    if not decoded.lower().endswith(".md"):
        lookup_paths.append(decoded + ".md")

    for lookup_path in lookup_paths:
        candidates: list[str | None]
        if lookup_path.startswith("/"):
            candidates = [_normal_rel(lookup_path)]
        elif lookup_path.startswith(("./", "../")):
            candidates = [_normal_rel(posixpath.join(source_dir, lookup_path))]
        elif "/" in lookup_path:
            # Obsidian folder links are normally vault-relative.  A source-relative
            # fallback keeps ordinary Markdown authoring conventions useful too.
            candidates = [
                _normal_rel(lookup_path),
                _normal_rel(posixpath.join(source_dir, lookup_path)),
            ]
        else:
            candidates = [
                _normal_rel(posixpath.join(source_dir, lookup_path)),
                _normal_rel(lookup_path),
            ]

        for candidate in candidates:
            if candidate is not None and candidate in aliases:
                return aliases[candidate], "local"

        basename = posixpath.basename(lookup_path)
        target = basename_aliases.get(basename)
        if target:
            return target, "basename"
    return None


def _render_wiki_target(
    source_rel: str, target_rel: str, original_path: str, resolution: str
) -> str:
    omitted_md = not original_path.lower().endswith(".md")
    if resolution == "basename" or "/" not in original_path:
        rendered = posixpath.basename(target_rel)
    elif original_path.startswith("/"):
        rendered = "/" + target_rel
    elif original_path.startswith(("./", "../")):
        rendered = _relative_target(source_rel, target_rel, original_path)
    else:
        rendered = target_rel
    if omitted_md and rendered.lower().endswith(".md"):
        rendered = rendered[:-3]
    return rendered


def _protected_spans(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Locate fenced and inline code so examples are never rewritten."""
    spans: list[tuple[int, int]] = []
    fence: str | None = None
    fence_start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        marker_match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        marker = marker_match.group(1) if marker_match else None
        if fence is None and marker:
            fence = marker[0]
            fence_start = offset
        elif fence is not None and marker and marker[0] == fence:
            spans.append((fence_start, offset + len(line)))
            fence = None
        offset += len(line)
    if fence is not None:
        spans.append((fence_start, len(text)))

    fenced = tuple(spans)
    for match in re.finditer(r"(?<!`)`+[^`\n]*`+(?!`)", text):
        if not any(start <= match.start() < end for start, end in fenced):
            spans.append(match.span())
    spans.sort()
    return [start for start, _ in spans], spans


def _is_protected(position: int, starts: list[int], spans: list[tuple[int, int]]) -> bool:
    index = bisect.bisect_right(starts, position) - 1
    return index >= 0 and position < spans[index][1]


def rewrite_markdown_links(
    text: str, source_rel: str, aliases: Mapping[str, str]
) -> tuple[str, int]:
    """Rewrite local links in one Markdown document, returning text and count."""
    basename_candidates: dict[str, set[str]] = {}
    for old, new in aliases.items():
        basename_candidates.setdefault(posixpath.basename(old), set()).add(new)
    basename_aliases = {
        name: next(iter(targets))
        for name, targets in basename_candidates.items()
        if len(targets) == 1
    }
    starts, spans = _protected_spans(text)
    rewritten = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewritten
        if _is_protected(match.start(), starts, spans):
            return match.group(0)

        if match.group("wiki_body") is not None:
            body = match.group("wiki_body")
            target_and_anchor, separator, label = body.partition("|")
            raw_path, anchor = _split_suffix(target_and_anchor)
            resolved = _wiki_lookup(source_rel, raw_path, aliases, basename_aliases)
            if resolved is None:
                return match.group(0)
            target_rel, resolution = resolved
            new_path = _render_wiki_target(source_rel, target_rel, raw_path, resolution)
            rewritten += 1
            new_body = new_path + anchor + (separator + label if separator else "")
            return match.group("wiki_open") + new_body + match.group("wiki_close")

        if match.group("inline_prefix") is not None:
            destination = match.group("inline_angle") or match.group("inline_plain")
            replacement = _resolve_standard_link(source_rel, destination, aliases)
            if replacement is None:
                return match.group(0)
            rewritten += 1
            if match.group("inline_angle") is not None:
                replacement = "<" + replacement + ">"
            return match.group("inline_prefix") + replacement

        destination = match.group("reference_angle") or match.group("reference_plain")
        replacement = _resolve_standard_link(source_rel, destination, aliases)
        if replacement is None:
            return match.group(0)
        rewritten += 1
        if match.group("reference_angle") is not None:
            replacement = "<" + replacement + ">"
        return match.group("reference_prefix") + replacement

    return _LINK_RE.sub(replace, text), rewritten


def _matches_exclude(rel_path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            if fnmatch.fnmatch(rel_path, suffix) or fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(posixpath.basename(rel_path), suffix):
                return True
        elif fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def repair_markdown_links(
    vault_root: str | Path,
    aliases: Mapping[str, str],
    *,
    exclude: Iterable[str] = (),
    logger=None,
) -> LinkRepairResult:
    """Repair all Markdown files under a vault using atomic replacements."""
    if not aliases:
        return LinkRepairResult()
    root = Path(vault_root)
    changed_paths: list[str] = []
    links_rewritten = 0
    visited: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() not in {".md", ".markdown"}:
                continue
            path = Path(dirpath) / filename
            rel_path = path.relative_to(root).as_posix()
            if _matches_exclude(rel_path, exclude):
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            repaired, count = rewrite_markdown_links(original, rel_path, aliases)
            if not count or repaired == original:
                continue

            temp = path.with_name(f".{path.name}.doc-link-repair-{os.getpid()}.tmp")
            try:
                mode = path.stat().st_mode
                temp.write_text(repaired, encoding="utf-8")
                os.chmod(temp, mode)
                os.replace(temp, path)
            except OSError as exc:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
                if logger is not None:
                    logger.warning("Cannot repair Markdown links in %s: %s", rel_path, exc)
                continue
            changed_paths.append(rel_path)
            links_rewritten += count

    if logger is not None and links_rewritten:
        logger.info(
            "Doc-ID link repair: rewrote %d link(s) in %d Markdown file(s)",
            links_rewritten,
            len(changed_paths),
        )
    return LinkRepairResult(len(changed_paths), links_rewritten, tuple(changed_paths))
