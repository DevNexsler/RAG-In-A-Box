"""Mutation runner distinguishes assertion kills from broken test infrastructure."""
import subprocess

from scripts.mutation_gate import Mutation, run_mutations


def test_mutations_report_killed_survived_and_invalid_without_editing_checkout(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('def accept(value):\n    return value > 0\n')
    (repo / 'test_sample.py').write_text('from sample import accept\ndef test_positive():\n    assert accept(1)\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    source = (repo / 'sample.py').read_text()
    mutations = [
        Mutation('killed', 'sample.py', 'value > 0', 'value < 0', ('test_sample.py',)),
        Mutation('survived', 'sample.py', 'value > 0', 'value >= 0', ('test_sample.py',)),
        Mutation('invalid', 'sample.py', 'value > 0', 'value +', ('test_sample.py',)),
        Mutation('test-error', 'sample.py', 'value > 0', 'missing_name', ('test_sample.py',)),
    ]
    report = run_mutations(repo, mutations, timeout=30)
    assert not report['passed']
    assert [row['status'] for row in report['mutations']] == ['killed', 'survived', 'invalid', 'killed']
    assert (repo / 'sample.py').read_text() == source


def test_process_crash_cannot_reuse_stale_junit(tmp_path):
    from scripts.mutation_gate import run_test_process
    (tmp_path / 'conftest.py').write_text('import os\nos._exit(1)\n')
    (tmp_path / 'test_sample.py').write_text('def test_ok():\n    assert True\n')
    report = tmp_path / 'result.xml'
    report.write_text('<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0"/></testsuites>')
    assert run_test_process(tmp_path, ('test_sample.py',), 10, report)['status'] == 'error'
