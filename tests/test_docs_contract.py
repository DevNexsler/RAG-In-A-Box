"""Documentation contract: what the docs say `doc_id` is must match what the code mints.

`doc_id` used to be the documents-root-relative path. It stopped being one when ids
became persistent identifiers: `doc_id_store` mints a zero-padded 5-char base-62 token
and the indexing flow namespaces it per source (`documents::0a3Zq`), while the path
moved to `rel_path`. Prose that still equates the two points a reader at an identifier
no path-addressed surface accepts.

These guards match the *claim* ("doc_id is a path relative to the documents root"),
not example payload *values* — a stale `"doc_id": "ops/deployment.md"` in a sample
response is a different defect class, tracked by its own tickets.
"""

import re
from pathlib import Path

import doc_id_store

ROOT = Path(__file__).resolve().parents[1]

# `doc_id` has to be the subject of the line for a path claim to be about it.
_DOC_ID = re.compile(r"\bdoc_id\b")

# The claim, in the phrasings docs actually use for it.
_PATH_CLAIMS = (
    re.compile(r"\bpath[\s-]*relative\b", re.I),          # "Path-relative to documents root"
    re.compile(r"\brelative\s+(?:\w+\s+)?path\b", re.I),  # "relative path", "relative file path"
    re.compile(r"\bdoc[_ ]?id\s+path\b", re.I),           # "Download file by doc_id path"
    # "doc_id is a path", "doc_id is the vault-relative path to the file"
    re.compile(r"\bdoc_id\b[^.|]{0,40}?\bis\s+(?:an?|the)\s+(?:[\w-]+\s+){0,2}path\b", re.I),
)


def _docs() -> list[Path]:
    """The published prose surface: docs/ plus repo-root markdown.

    Test fixtures and the historical record (`plans/`, `worklogs/`) are out —
    they are snapshots of what was true when written, not statements of contract.
    """
    return sorted(ROOT.glob("docs/**/*.md")) + sorted(ROOT.glob("*.md"))


def _path_claims(text: str) -> list[tuple[int, str]]:
    """(line number, line) for each line calling `doc_id` a path."""
    return [
        (n, line.strip())
        for n, line in enumerate(text.splitlines(), start=1)
        if _DOC_ID.search(line) and any(claim.search(line) for claim in _PATH_CLAIMS)
    ]


def _schema_row(field: str) -> str:
    """Description cell of `field`'s row in architecture.md's LanceDB schema table."""
    doc = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    row = re.search(rf"^\|\s*`{re.escape(field)}`\s*\|(.*)$", doc, re.M)
    return row.group(1) if row else ""


def test_path_claim_matcher_flags_the_claim_and_not_its_correction():
    """The lint has to survive the fix: it must flag the claim and pass the correction."""
    flagged = "| `doc_id` | string | Path-relative to documents root (e.g. `Projects/notes.md`) |"
    assert _path_claims(flagged)
    assert _path_claims("| `GET /api/documents/{path}` | Done | Download file by doc_id path |")
    assert _path_claims("The doc_id is the path of the file relative to documents_root.")

    # Corrections and neighbouring, accurate lines must stay clean.
    assert not _path_claims("| `doc_id` | string | Minted id `documents::0a3Zq` — not a path, see `rel_path` |")
    assert not _path_claims("| `rel_path` | string | Document path relative to documents root |")
    assert not _path_claims("| `/api/documents/{doc_id}` | GET | Download a file by path |")
    assert not _path_claims('      "doc_id": "ops/deployment.md",')


def test_no_doc_describes_doc_id_as_a_path():
    offenders = [
        f"{doc.relative_to(ROOT)}:{n}: {line}"
        for doc in _docs()
        for n, line in _path_claims(doc.read_text(encoding="utf-8"))
    ]
    assert not offenders, "docs describe doc_id as a path:\n  " + "\n  ".join(offenders)


def test_architecture_documents_doc_id_as_a_minted_identifier():
    """The schema table's example must be a value `doc_id_store` could actually mint."""
    row = _schema_row("doc_id")
    assert row, "docs/architecture.md has no `doc_id` row in the LanceDB schema table"

    example = re.search(r"e\.g\.\s*`([^`]+)`", row)
    assert example, f"doc_id row carries no `e.g.` example: {row.strip()}"

    source, sep, minted = example.group(1).partition("::")
    assert sep and source, f"doc_id example {example.group(1)!r} is not `<source_name>::<id>`"
    assert len(minted) == doc_id_store._ID_LEN and set(minted) <= set(doc_id_store._BASE62), (
        f"doc_id example {minted!r} is not a {doc_id_store._ID_LEN}-char base-62 id"
    )


def test_architecture_points_readers_from_doc_id_to_rel_path():
    """Whoever came for the path needs to be told where it moved."""
    assert "rel_path" in _schema_row("doc_id"), (
        f"doc_id row must point at rel_path for the document path: {_schema_row('doc_id').strip()}"
    )
    assert "path" in _schema_row("rel_path").lower(), (
        "LanceDB schema table must document `rel_path` as the document path"
    )
