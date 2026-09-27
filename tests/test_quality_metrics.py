"""Worked ranking judgments protect evaluator correctness."""
import pytest
from scripts.rag_quality import ranking_metrics


def test_duplicate_chunks_do_not_hide_wasted_retrieval_slots():
    metrics = ranking_metrics(['a', 'a', 'b'], ['a', 'b'], 2)
    assert metrics['recall_at_k'] == .5
    assert metrics['reciprocal_rank'] == 1
    assert metrics['ndcg_at_k'] == pytest.approx(.6131471927654584)


def test_relevant_result_at_second_rank():
    metrics = ranking_metrics(['wrong', 'right'], ['right'], 2)
    assert metrics['recall_at_k'] == 1
    assert metrics['reciprocal_rank'] == .5
    assert metrics['ndcg_at_k'] == pytest.approx(.6309297535714575)


def test_unanswerable_query_requires_abstention():
    assert ranking_metrics([], [], 3)['recall_at_k'] == 1
    assert ranking_metrics(['hallucinated'], [], 3)['recall_at_k'] == 0


def test_live_quality_cli_requires_explicit_spend_opt_in(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/rag_quality.py'
    result = subprocess.run([sys.executable, str(script), '--live-config', str(tmp_path / 'missing.yaml'),
                             '--output', str(tmp_path / 'quality.json')], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--allow-paid' in result.stderr
    assert not (tmp_path / 'quality.json').exists()


def test_live_preflight_failure_cannot_leave_old_passing_report(tmp_path, monkeypatch):
    import json
    import subprocess
    from scripts.rag_quality import main
    output = tmp_path / 'quality.json'
    output.write_text('{"passed":true}')
    monkeypatch.setattr(subprocess, 'call', lambda *args, **kwargs: 1)
    assert main(['--live-config', 'config_test.yaml', '--allow-paid', '--output', str(output)]) == 1
    assert not json.loads(output.read_text())['passed']
