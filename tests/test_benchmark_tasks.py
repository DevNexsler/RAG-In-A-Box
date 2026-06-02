from core.benchmarking.tasks import get_task


def test_get_task_returns_enrichment_task():
    task = get_task("enrichment")

    assert task.name == "enrichment"
    assert task.default_score_mode == "standard"


def test_get_task_rejects_unknown_task():
    try:
        get_task("missing")
    except ValueError as exc:
        assert "unknown benchmark task" in str(exc)
    else:
        raise AssertionError("expected ValueError")
