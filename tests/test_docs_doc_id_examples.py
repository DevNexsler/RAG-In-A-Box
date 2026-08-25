"""Docs lint: example payloads must show `doc_id` as a namespaced persistent id.

`doc_id` is the source-namespaced registry id (`documents::00001`, see
`DocIDStore.all_mappings`), not a path — the path is `rel_path`, and every
search hit carries both (`_hit_to_dict` / `_slim_hit_to_dict` in
`mcp_server.py`). Docs that print a path as the `doc_id` value teach callers to
join it onto a filesystem root or pass it back as a path, which fails for every
non-filesystem source (comm messages, SOR tasks) and for renamed files.

Prose claims are caught by reading; a bare JSON example value is not, so lint
the values themselves: any `"doc_id"` shown in a ```json example block must be
namespaced. Elision placeholders assert nothing and are exempt.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

_JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.DOTALL)
_DOC_ID_VALUE = re.compile(r'"doc_id"\s*:\s*"([^"]*)"')
_PLACEHOLDERS = {"...", "…"}


def _doc_id_examples() -> list[tuple[Path, int, str]]:
    """Every `doc_id` value shown in a fenced json block under docs/, with its line."""
    found: list[tuple[Path, int, str]] = []
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in _JSON_BLOCK.finditer(text):
            body = block.group(1)
            block_line = text.count("\n", 0, block.start()) + 1
            for match in _DOC_ID_VALUE.finditer(body):
                line = block_line + body.count("\n", 0, match.start()) + 1
                found.append((path, line, match.group(1)))
    return found


def test_docs_contain_doc_id_examples():
    """Guard the lint itself: a regex that matches nothing would pass vacuously."""
    assert _doc_id_examples(), "no doc_id example values found under docs/ — lint is inert"


@pytest.mark.parametrize(
    "path,line,value",
    [pytest.param(p, ln, v, id=f"{p.relative_to(ROOT)}:{ln}") for p, ln, v in _doc_id_examples()],
)
def test_doc_id_example_is_namespaced_not_a_path(path: Path, line: int, value: str):
    if value in _PLACEHOLDERS:
        return
    assert "::" in value, (
        f"{path.relative_to(ROOT)}:{line} shows doc_id as {value!r}. "
        "doc_id is a source-namespaced persistent id (e.g. 'documents::00001'); "
        "show the path as rel_path instead."
    )
