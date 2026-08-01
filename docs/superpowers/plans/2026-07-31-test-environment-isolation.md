# Test Environment Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make documented gate commands run with project Python and fresh run-local Prefect state without touching developer or production data.

**Architecture:** Make selects an overridable interpreter, preferring project virtual environment. Gate creates an atomic unique Prefect home beneath artifact directory, scopes it to gate execution with restoration in `finally`, and lets every child process inherit it.

**Tech Stack:** GNU Make, Python 3.12, pytest, Prefect, `tempfile`.

---

## File Structure

- `Makefile`: interpreter selection and all Python command entrypoints.
- `scripts/gate.py`: unique Prefect-home creation and environment lifetime.
- `tests/test_gate_runner.py`: regression coverage for isolation, uniqueness, restoration, and Makefile portability.

### Task 1: Portable Make interpreter

**Files:**
- Modify: `Makefile:1-28`
- Test: `tests/test_gate_runner.py`

- [ ] **Step 1: Add failing Makefile contract test**

```python
def test_makefile_uses_overridable_project_python():
    text = (REPO_ROOT / "Makefile").read_text()
    assert "PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)" in text
    assert "\tpython " not in text
    assert text.count("\t$(PYTHON) ") == 8
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_gate_runner.py::test_makefile_uses_overridable_project_python -q`

Expected: FAIL because `PYTHON` definition is absent and recipes use literal `python`.

- [ ] **Step 3: Run GitNexus impact analysis**

Run impact analysis for file target `Makefile`, upstream direction. Warn before editing if risk is HIGH or CRITICAL.

- [ ] **Step 4: Implement minimal Makefile change**

Add:

```make
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
```

Replace each recipe's leading `python` with `$(PYTHON)`.

- [ ] **Step 5: Verify GREEN and command resolution**

Run:

```bash
.venv/bin/python -m pytest tests/test_gate_runner.py::test_makefile_uses_overridable_project_python -q
make -n gate-fast
```

Expected: test PASS; dry run begins `.venv/bin/python scripts/gate.py --fast`.

- [ ] **Step 6: Commit Task 1**

Before commit, run `gitnexus_detect_changes(scope="staged")`. Then commit only `Makefile` and `tests/test_gate_runner.py` with `fix: use project Python for gate commands`.

### Task 2: Isolate Prefect state per gate invocation

**Files:**
- Modify: `scripts/gate.py:1-300`
- Test: `tests/test_gate_runner.py`

- [ ] **Step 1: Add failing isolation tests**

Add hermetic tests that call `monkeypatch.chdir(tmp_path)`, monkeypatch `gate.dispatch`, call `gate.main()` twice with one `--run-dir`, and capture `os.environ["PREFECT_HOME"]` during dispatch. Changing directory prevents real report-script execution. Assert both values:

```python
assert first.parent == run_dir
assert second.parent == run_dir
assert first != second
assert first.is_dir() and second.is_dir()
```

Set ambient `PREFECT_HOME` before calls and assert it is restored after each `main()` return. Add failure-path test where dispatch raises; assert restoration still occurs.

- [ ] **Step 2: Verify RED**

Run isolation tests directly. Expected: FAIL because gate reuses ambient Prefect home and creates no run-local directory.

- [ ] **Step 3: Run GitNexus context and impact analysis**

Run `gitnexus_context(name="main", file_path="scripts/gate.py")`, then upstream impact for `main`. Run impact for any existing helper changed. Warn before editing if risk is HIGH or CRITICAL; cover every d=1 dependent.

- [ ] **Step 4: Implement atomic isolation with restoration**

Import `tempfile`. After `run_dir.mkdir(...)`, create directory atomically:

```python
prefect_home = Path(tempfile.mkdtemp(prefix="prefect-home-", dir=run_dir))
```

Move gate execution body into focused helper if needed. Save ambient `PREFECT_HOME`, set run-local value before any child dispatch, and restore exact prior state in `finally`, including exception paths. Do not delete created directory.

- [ ] **Step 5: Verify targeted GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_gate_runner.py -q
.venv/bin/python -m pytest tests/sources/test_filesystem_source.py::test_scan_yields_same_records_as_legacy tests/test_change_hash_diff.py tests/test_index_safety_guards.py tests/test_scan.py -q
```

Expected: all selected tests PASS; stale `~/.prefect/prefect.db` remains untouched.

- [ ] **Step 6: Commit Task 2**

Before commit, run `gitnexus_detect_changes(scope="staged")`. Commit only `scripts/gate.py` and relevant test changes with `fix: isolate Prefect state for gate runs`.

### Task 3: Full verification

**Files:**
- Verify only.

- [ ] **Step 1: Run full fast gate through documented entrypoint**

Run: `make gate-fast`

Expected: static PASS, unit PASS, integration PASS, overall PASS. No staging, live, or real-provider tier runs.

- [ ] **Step 2: Verify production state untouched**

If `~/.prefect/prefect.db` exists, capture `stat` metadata and checksum before and after gate; expected unchanged. If absent, confirm it remains absent. Confirm new Prefect directories exist only beneath latest `.evals/gate-runs/` artifact directory.

- [ ] **Step 3: Verify repository scope**

Run `gitnexus_detect_changes(scope="compare", base_ref="HEAD~2")` and `git status --short`. Expected affected scope: Makefile, gate runner, tests; unrelated user-owned untracked docs unchanged.

- [ ] **Step 4: Refresh GitNexus after commits**

Check `.gitnexus/meta.json` embeddings count. Run `npx gitnexus analyze`; add `--embeddings` only if count was nonzero.
