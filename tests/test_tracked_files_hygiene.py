"""Tracked-file hygiene for generated superpowers scratch.

A ``.gitignore`` rule only stops untracked paths from being added. It does
nothing for a path that is already in the index, which is how
``.superpowers/sdd/2026-08-14-hook-delivery-outbox/final-fix-report.md`` stayed
on main after ``808ca96`` swept a generated final-fix report in beside the real
change. The sibling report under ``1688-index-maintenance`` was removed by hand
in ``256e84f`` without an ignore rule, so the next broad ``git add`` could do
it again.

The guard asks git itself, using only committed ``.gitignore`` files — not
``.git/info/exclude`` or the operator's global excludes. ``docs/superpowers/``
is intentional source and must stay addable.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]

# The generated report that was committed, plus the next variant of that artifact.
SUPERPOWERS_SCRATCH = (
    ".superpowers/sdd/2026-08-14-hook-delivery-outbox/final-fix-report.md",
    ".superpowers/sdd/next-session/final-fix-report.md",
)

# Tracked design docs that share the word "superpowers" but are source.
TRACKED_SOURCE_NEIGHBOURS = (
    "docs/superpowers/plans/2026-08-14-hook-delivery-outbox.md",
    "docs/superpowers/specs/2026-08-14-hook-delivery-outbox-design.md",
)


def _git_env(**overrides: str) -> dict[str, str]:
    # An inherited GIT_DIR / GIT_INDEX_FILE (a hook, a wrapper) would point git
    # at some other repository or index than the one under test.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(overrides)
    return env


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_git_env(),
        check=check,
        capture_output=True,
        text=True,
    )


def tracked_ignored_paths(repo: Path) -> list[str]:
    """Index entries that the repository's committed ``.gitignore`` files exclude."""
    result = _git(
        repo,
        "ls-files",
        "--cached",
        "--ignored",
        "--exclude-per-directory=.gitignore",
        "-z",
        check=False,
    )
    assert result.returncode == 0, f"git ls-files failed in {repo}: {result.stderr.strip()}"
    return sorted(path for path in result.stdout.split("\0") if path)


def test_committed_gitignore_ignores_superpowers_scratch_not_specs(tmp_path: Path) -> None:
    """The committed rules must ignore ``.superpowers/`` scratch and leave specs addable.

    If the rule is missing, the next broad ``git add`` tracks another generated
    report and the index guard never sees it, because an untracked ignored path
    is not an index entry.
    """
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=repo,
        env=_git_env(HOME=str(home), XDG_CONFIG_HOME=str(home), GIT_CONFIG_NOSYSTEM="1"),
        check=True,
        capture_output=True,
        text=True,
    )
    shutil.copyfile(REPO_ROOT / ".gitignore", repo / ".gitignore")
    paths = (*SUPERPOWERS_SCRATCH, *TRACKED_SOURCE_NEIGHBOURS)
    for name in paths:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--force", *paths],
        cwd=repo,
        env=_git_env(HOME=str(home), XDG_CONFIG_HOME=str(home), GIT_CONFIG_NOSYSTEM="1"),
        check=True,
        capture_output=True,
        text=True,
    )

    assert tracked_ignored_paths(repo) == sorted(SUPERPOWERS_SCRATCH)


def test_superpowers_scratch_is_untracked() -> None:
    """Acceptance: the index has no ``.superpowers`` path, and the rule matches it.

    ``git check-ignore`` hides a directory that still contains tracked files, so
    this has to run against the real index after the untrack, not a force-add.
    """
    listed = _git(REPO_ROOT, "ls-files", ".superpowers")
    assert listed.stdout == ""
    ignored = _git(REPO_ROOT, "check-ignore", "-q", ".superpowers/", check=False)
    assert ignored.returncode == 0


def test_repository_tracks_nothing_its_gitignore_excludes() -> None:
    offenders = tracked_ignored_paths(REPO_ROOT)
    assert offenders == [], (
        f"tracked paths match .gitignore: {offenders}. An ignore rule never untracks a "
        "file. Run `git rm --cached <path>`, or add a `!` negation to .gitignore if the "
        "path must stay tracked."
    )
