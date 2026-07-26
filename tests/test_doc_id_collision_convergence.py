"""A doc-ID sweep must converge, and must say so.

Ticket #0545: after #0390's claim policy shipped, production still logged
`ID collision` warnings that neither converged nor repeated — 62 / 0 / 43 / 117
across four consecutive scans of the *same* vault, the same token re-assigned
to a different victim every run. Two defects sit behind that shape:

1. **Order dependence.** The walk visited directories in filesystem order, so
   which file won a contested token — and which fresh IDs the allocator handed
   out — changed between otherwise identical scans.
2. **Silence.** The warning says what the scan *will* do; nothing recorded the
   fresh ID it minted, and no per-scan total distinguished "62 claims resolved"
   from "62 claims re-decided for the fourth time". A sweep that cannot be shown
   to converge is indistinguishable from one that never does.

These tests pin the observable contract: identical vault -> identical decisions,
every re-tokenization names its replacement ID, and every scan reports
contested / re-tokenized / remaining.
"""

import logging

from doc_id_store import DocIDStore, extract_id_from_filename
from flow_index_vault import scan_filesystem_records

SCAN_INCLUDE = ["**/*"]
SCAN_EXCLUDE = ["registry.db*"]


class _CapturingLogger(logging.Logger):
    """Records formatted warning/info lines for assertion."""

    def __init__(self):
        super().__init__("doc-id-sweep-test")
        self.lines: list[str] = []

    def handle(self, record):  # pragma: no cover - logging plumbing
        self.lines.append(record.getMessage())

    def warning(self, msg, *args, **kwargs):
        self.lines.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.lines.append(msg % args if args else msg)

    def text(self) -> str:
        return "\n".join(self.lines)


def _scan(tmp_path, registry, logger=None, no_rename_prefixes=None):
    return scan_filesystem_records(
        tmp_path,
        SCAN_INCLUDE,
        SCAN_EXCLUDE,
        doc_id_store=registry,
        logger=logger,
        no_rename_prefixes=no_rename_prefixes,
    )


def _decisions(records):
    """The scan's identity decisions, order-independent."""
    return sorted((r["rel_path"], r["doc_id"]) for r in records)


def _vault_with_contested_tokens(tmp_path, token="0000t"):
    """A vault where several producer files all claim one unissued token.

    The registry is seeded with a real document first: an empty registry is the
    documented bootstrap case where filename tokens *are* adopted verbatim, and
    production's registry is never empty.
    """
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "readme.md").write_text("# the real document")
    registry = DocIDStore(tmp_path / "registry.db")
    scan_filesystem_records(
        tmp_path, ["notes/*.md"], SCAN_EXCLUDE, doc_id_store=registry
    )
    assert registry.count() == 1

    for owner in ("alice", "bob", "carol"):
        d = tmp_path / "email-attachments" / owner
        d.mkdir(parents=True)
        (d / f"msg__mm0@{token}@.pdf").write_text(f"{owner} attachment body")
    return registry


def test_repeat_scan_of_unchanged_vault_makes_identical_decisions(tmp_path):
    """Two scans of an unchanged vault must agree, file for file.

    This is the 62 / 0 / 43 / 117 signature: the same vault, minutes apart,
    adjudicated differently every time.
    """
    registry = _vault_with_contested_tokens(tmp_path)
    first = _decisions(_scan(tmp_path, registry))
    second = _decisions(_scan(tmp_path, registry))
    assert first == second
    registry.close()


def test_scan_is_independent_of_filesystem_walk_order(tmp_path, monkeypatch):
    """The same vault must produce the same IDs whatever order the walk yields.

    `os.walk` order is a filesystem detail — it varies with directory hash
    seeding, inode reuse, and concurrent writes. Reversing it here stands in
    for that variance: whichever order the OS happens to hand back, the sweep's
    decisions must be the same, or a rescan re-adjudicates a vault that never
    changed. This is the mechanism behind 62 / 0 / 43 / 117.
    """
    import flow_index_vault

    real_walk = flow_index_vault.os.walk

    def reversed_walk(top, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, **kwargs):
            dirnames.reverse()
            yield dirpath, dirnames, list(reversed(filenames))

    decisions = []
    for reverse in (False, True):
        vault = tmp_path / f"vault-{reverse}"
        vault.mkdir()
        registry = _vault_with_contested_tokens(vault)
        with monkeypatch.context() as m:
            if reverse:
                m.setattr(flow_index_vault.os, "walk", reversed_walk)
            records = _scan(vault, registry)
        # Compare identity per attachment *owner*, not per absolute path: the
        # two vaults differ only by root name.
        decisions.append(sorted(
            (r["rel_path"].split("/")[1], r["doc_id"])
            for r in records if r["rel_path"].startswith("email-attachments/")
        ))
        registry.close()

    assert decisions[0] == decisions[1]


def test_retokenization_logs_the_fresh_id(tmp_path):
    """A rejected claim must name the ID it was actually given.

    Production logged 222 re-assignment warnings and not one replacement ID:
    `grep -niE 'allocated|fresh id|assigned new'` returned zero matches, so no
    operator could tell a resolved claim from a re-decided one.
    """
    registry = _vault_with_contested_tokens(tmp_path)
    logger = _CapturingLogger()
    records = _scan(tmp_path, registry, logger=logger)

    fresh_ids = {
        r["doc_id"] for r in records
        if r["rel_path"].startswith("email-attachments/")
        and extract_id_from_filename(r["rel_path"]) == r["doc_id"]
    }
    assert len(fresh_ids) == 3, "each re-tokenized file should carry its new ID"
    assert "0000t" not in fresh_ids, "the contested claim must not be granted"
    for doc_id in fresh_ids:
        assert doc_id in logger.text(), (
            f"fresh ID {doc_id} was minted but never logged:\n{logger.text()}"
        )
    registry.close()


def test_scan_reports_contested_retokenized_and_remaining(tmp_path):
    """Every scan must publish a per-scan sweep total.

    Without it, "62 collisions" and "117 collisions" look like the same kind of
    event, and a non-converging sweep reads as progress.
    """
    registry = _vault_with_contested_tokens(tmp_path)
    logger = _CapturingLogger()
    _scan(tmp_path, registry, logger=logger)

    summary = [ln for ln in logger.lines if "contested" in ln]
    assert summary, f"no sweep summary emitted:\n{logger.text()}"
    line = summary[-1]
    assert "re-tokenized" in line and "remaining" in line, line
    # Three files claimed one unissued token; all three are re-tokenized and
    # nothing is left physically contested.
    assert "3 contested" in line, line
    assert "3 re-tokenized" in line, line
    assert "0 remaining" in line, line
    registry.close()


def test_converged_vault_reports_a_clean_sweep(tmp_path):
    """Once the sweep has run to completion, a rescan is quiet.

    Acceptance item 1: zero `ID collision` warnings on an unchanged vault.
    """
    registry = _vault_with_contested_tokens(tmp_path)
    _scan(tmp_path, registry)

    logger = _CapturingLogger()
    _scan(tmp_path, registry, logger=logger)

    assert "ID collision" not in logger.text(), logger.text()
    summary = [ln for ln in logger.lines if "contested" in ln][-1]
    assert "0 contested" in summary, summary
    assert "0 remaining" in summary, summary
    registry.close()


def test_unresolvable_claim_is_counted_as_remaining(tmp_path):
    """A claim the sweep cannot physically clear must not report as resolved.

    Deposit-owned paths are never renamed, so a producer-minted token stays on
    disk after the registry re-tokenizes the file. That residue is real — it is
    what keeps the vault's token space contested — and belongs in `remaining`,
    not silently in `re-tokenized`.
    """
    registry = _vault_with_contested_tokens(tmp_path)
    logger = _CapturingLogger()
    _scan(
        tmp_path, registry, logger=logger,
        no_rename_prefixes=["email-attachments/"],
    )

    # Filenames untouched: every file still carries the contested token.
    on_disk = list((tmp_path / "email-attachments").rglob("*.pdf"))
    assert len(on_disk) == 3
    assert all("@0000t@" in p.name for p in on_disk)

    summary = [ln for ln in logger.lines if "contested" in ln][-1]
    assert "3 contested" in summary, summary
    assert "3 remaining" in summary, summary
    registry.close()
