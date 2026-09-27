#!/usr/bin/env python3
"""Curated safety mutations in disposable tracked-file copies. Never edits checkout."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    before: str
    after: str
    tests: tuple[str, ...]
    occurrences: int = 1


MUTATIONS = (
    Mutation('queue-incarnation', 'core/index_request_queue.py', 'AND incarnation = ?', 'AND (? IS NOT NULL)',
             ('tests/test_queue_properties.int.test.py',), 2),
    Mutation('queue-revision', 'core/index_request_queue.py', 'AND revision = ?', 'AND (? IS NOT NULL)',
             ('tests/test_queue_properties.int.test.py',), 2),
    Mutation('callback-incarnation', 'core/hook_outbox.py', 'AND incarnation = ?', 'AND (? IS NOT NULL)',
             ('tests/test_outbox_properties.int.test.py',), 2),
    Mutation('card-grounding', 'core/enrichment_postprocess.py', 'return repaired, corrections', 'return dict(enrichment), []',
             ('tests/test_enrichment_invariants.py',)),
    Mutation('rpc-correlation', 'factbook_client.py', 'data.get("id") != payload["id"]', 'False',
             ('tests/contracts/test_factbook_identity.int.test.py',)),
    Mutation('failed-source-retirement', 'flow_index_vault.py', 'and entry_source not in failed_sources', 'and True',
             ('tests/contracts/test_source_retirement.int.test.py::test_retirement_requires_successful_scan_of_own_source[failed_scan]',)),
    Mutation('single-probe', 'core/resilience.py', 'if state.get("probe_in_flight"):', 'if False:',
             ('tests/test_provider_recovery_contract.py',)),
)


def run_test_process(repo: Path, selectors: tuple[str, ...], timeout: float, report_path: Path) -> dict:
    report_path.unlink(missing_ok=True)
    env = {key: value for key, value in os.environ.items() if key not in {'PYTHONPATH', 'PYTEST_ADDOPTS'}}
    env['PYTHONPATH'] = str(repo)
    # No project .env/config files are copied. Selectors are hermetic tests only.
    log_path = report_path.with_suffix('.log')
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'pytest', *selectors, '-q',
                                    f'--junitxml={report_path}'], cwd=repo, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {'status': 'timeout'}
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if not report_path.exists():
        return {'status': 'error', 'returncode': code}
    suites = ET.parse(report_path).getroot().iter('testsuite')
    totals = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, '0'))
    if totals['errors'] or not totals['tests'] or code not in (0, 1) or totals['tests'] == totals['skipped']:
        status = 'error'
    else:
        status = 'killed' if code == 1 and totals['failures'] else 'survived'
    return {'status': status, 'returncode': code, **totals}


def run_mutations(root: Path, mutations=MUTATIONS, *, timeout: float = 180, artifacts: Path | None = None) -> dict:
    if timeout <= 0 or not mutations:
        raise ValueError('positive timeout and nonempty mutation set required')
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    rows = []
    with tempfile.TemporaryDirectory(prefix='rag-mutations-') as directory:
        scratch = Path(directory)
        baseline = scratch / 'baseline'
        baseline.mkdir()
        for name in filter(None, files):
            source = root / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f'tracked file unavailable or symlink: {name}')
            destination = baseline / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        evidence = artifacts or scratch / 'reports'
        evidence.mkdir(parents=True, exist_ok=True)
        checked = {}
        for index, mutation in enumerate(mutations):
            selectors = mutation.tests
            if selectors not in checked:
                checked[selectors] = run_test_process(baseline, selectors, timeout, evidence / f'baseline-{index}.xml')
            if checked[selectors]['status'] != 'survived' or checked[selectors].get('skipped'):
                rows.append({'name': mutation.name, 'status': 'baseline_failed', 'baseline': checked[selectors]})
                continue
            candidate = scratch / f'mutant-{index}'
            shutil.copytree(baseline, candidate, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
            path = candidate / mutation.path
            original = path.read_text()
            if original.count(mutation.before) != mutation.occurrences:
                rows.append({'name': mutation.name, 'status': 'invalid', 'reason': 'anchor mismatch'})
                continue
            changed = original.replace(mutation.before, mutation.after)
            try:
                ast.parse(changed)
            except SyntaxError:
                rows.append({'name': mutation.name, 'status': 'invalid', 'reason': 'syntax'})
                continue
            path.write_text(changed)
            result = run_test_process(candidate, selectors, timeout, evidence / f'mutant-{index}.xml')
            rows.append({'name': mutation.name, **result})
    return {'mutations': rows, 'passed': all(row['status'] == 'killed' for row in rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    report = run_mutations(ROOT, timeout=args.timeout, artifacts=args.output.parent / 'mutation-details')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
