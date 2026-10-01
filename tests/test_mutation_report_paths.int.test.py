"""Mutation reports resolve before child pytest changes working directory."""
from pathlib import Path

from scripts.mutation_gate import run_test_process


def test_relative_junit_path_is_collected_from_child_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / 'test_example.py').write_text('def test_example():\n    assert True\n')
    reports = tmp_path / 'reports'
    reports.mkdir()
    monkeypatch.chdir(tmp_path)
    result = run_test_process(checkout, ('test_example.py',), 30, Path('reports/baseline.xml'))
    assert result['status'] == 'survived', result
    assert result['tests'] == 1
    assert (reports / 'baseline.xml').exists()
    assert not (checkout / 'reports').exists()
