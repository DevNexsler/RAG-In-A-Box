"""Local scorer contracts: wire policy, occurrence alignment and bounded work."""

import asyncio
import json
import threading
import time

import httpx
import pytest
import yaml

from core.rerank_cancellation import rerank_cancel_event
from core.storage import SearchHit
from providers.rerank import qwen_bulk as qwen
from search_hybrid import build_reranker


def hits(count=3):
    return [SearchHit(doc_id=f"doc-{i}", loc="0", snippet=str(i), text=f"passage {i}", score=1.0)
            for i in range(count)]


def response(payload, scores=None, content=None, finish="stop", status=200):
    passages = json.loads(payload["messages"][1]["content"])["passages"]
    if scores is None:
        scores = {p["id"]: int(p["id"][1:]) / 100 for p in reversed(passages)}
    return httpx.Response(status, json={"choices": [{"finish_reason": finish, "message": {
        "content": content if content is not None else json.dumps({"scores": scores}),
    }}]}, request=httpx.Request("POST", "http://gateway/v1/chat/completions"))


def install(monkeypatch, callback):
    async def post(self, url, **kwargs):
        return await callback(url, kwargs["json"])
    monkeypatch.setattr(httpx.AsyncClient, "post", post)


def scorer(**kwargs):
    return qwen.QwenBulkReranker(api_key="test-only", base_url="http://gateway/v1", **kwargs)


def test_default_and_compose_migrate_legacy_provider(monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-only")
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    assert isinstance(build_reranker({"search": {"reranker": {"enabled": True}}}), qwen.QwenBulkReranker)
    monkeypatch.setenv("DOC_ORGANIZER_RERANKER_PROVIDER", "qwen-bulk")
    r = build_reranker({"search": {"reranker": {"enabled": True, "provider": "deepinfra",
                                               "model": "Qwen/Qwen3-Reranker-8B", "api_key": "paid-key"}}})
    assert r.model == "qwen-bulk"
    assert r.api_key == "test-only"
    from pathlib import Path
    env = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]["doc-organizer"]["environment"]
    assert "DOC_ORGANIZER_RERANKER_PROVIDER=${DOC_ORGANIZER_RERANKER_PROVIDER:-qwen-bulk}" in env
    assert "LITELLM_BASE_URL=${LITELLM_BASE_URL:-http://host.docker.internal:4000/v1}" in env


def test_gateway_credentials_precedence_and_no_paid_fallback(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "preferred")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secondary")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://override/v1/")
    r = qwen.QwenBulkReranker()
    assert r.api_key == "preferred" and r.base_url == "http://override/v1"
    monkeypatch.delenv("LITELLM_API_KEY")
    assert qwen.QwenBulkReranker().api_key == "secondary"
    monkeypatch.delenv("LITELLM_MASTER_KEY")
    with pytest.raises(ValueError, match="LITELLM"):
        qwen.QwenBulkReranker()
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker({"search": {"reranker": {"enabled": True, "provider": "qwen-fast"}}})


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        scorer(timeout=timeout)


@pytest.mark.parametrize("count", [1, 32, 64, 70, 90])
def test_maps_reversed_keys_to_original_documents_and_bounds_payload(monkeypatch, count):
    sent = []
    async def answer(url, payload):
        sent.append(payload)
        assert url == "http://gateway/v1/chat/completions"
        assert payload["model"] == "qwen-bulk" and payload["priority"] == 0
        assert payload["temperature"] == 0 and payload["num_retries"] == 0
        assert payload["disable_fallbacks"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert 0 < payload["metadata"]["inference_deadline_seconds"] <= 10
        passages = json.loads(payload["messages"][1]["content"])["passages"]
        assert len(passages) <= 32
        assert all(0 < len(p["text"]) <= 4000 for p in passages)
        bounded = {k: v for k, v in payload.items() if k != "metadata"}
        assert (len(json.dumps(bounded, ensure_ascii=False).encode()) + payload["max_tokens"] + 512
                <= qwen.PROMPT_TOKEN_BUDGET)
        return response(payload)
    install(monkeypatch, answer)
    candidates = hits(count)
    for h in candidates:
        h.text = "invoice 1234567890 金額 \"\\\n" * 700
    ranked = scorer().rerank("invoices due " * 2000, candidates)
    assert [h.doc_id for h in ranked] == [f"doc-{i}" for i in reversed(range(count))]
    assert len(sent) == (count + 31) // 32
    assert {p["id"] for s in sent for p in json.loads(s["messages"][1]["content"])["passages"]} == {
        f"D{i+1}" for i in range(count)}
    assert all(h.text.endswith("invoice 1234567890 金額 \"\\\n") for h in ranked)


def test_duplicate_occurrences_keep_own_original_scores_and_stable_ties(monkeypatch):
    async def answer(url, payload):
        return response(payload, scores={"D3": 1, "D1": 0, "D2": 1})
    install(monkeypatch, answer)
    candidates = hits()
    for h, score in zip(candidates, [9, 9, 1]):
        h.doc_id = "same-doc"
        h.text = "identical text"
        h.score = score
    a, b, c = candidates
    ranked = scorer().rerank("query", candidates)
    assert ranked == [b, c, a]
    assert [a.score, b.score, c.score] == pytest.approx([.4, 1, .6])
    async def tied(url, payload):
        return response(payload, scores={"D3": .5, "D2": .5, "D1": .5})
    install(monkeypatch, tied)
    candidates = hits()
    assert scorer().rerank("query", candidates) == candidates


@pytest.mark.parametrize("content,finish,status", [
    ('{"scores":[0.1,0.2,0.3]}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"X":0}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":0,"D1":1}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":true}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":NaN}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":Infinity}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":-0.1}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":1.1}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":".3"}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":null}}', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":1},"extra":0}', "stop", 200),
    ('{"scores":', "stop", 200),
    ('{"scores":{"D1":0,"D2":0,"D3":1}}', "length", 200),
    ('{}', "stop", 401), ('{}', "stop", 402), ('{}', "stop", 503),
])
def test_rejects_entire_operation_without_mutating_hits(monkeypatch, content, finish, status):
    calls = []
    async def answer(url, payload):
        calls.append(url)
        return response(payload, content=content, finish=finish, status=status)
    install(monkeypatch, answer)
    candidates = hits()
    with pytest.raises(RuntimeError, match="Qwen bulk reranker failed"):
        scorer().rerank("query", candidates)
    assert [h.score for h in candidates] == [1, 1, 1]
    assert len(calls) == 1


def test_failed_batch_cancels_and_awaits_siblings(monkeypatch):
    cancelled = []
    async def answer(url, payload):
        ids = json.loads(payload["messages"][1]["content"])["passages"]
        if ids[0]["id"] == "D1":
            await asyncio.sleep(.01)
            return response(payload, content='{"scores":{}}')
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    install(monkeypatch, answer)
    candidates = hits(70)
    with pytest.raises(RuntimeError):
        scorer().rerank("query", candidates)
    assert cancelled and all(cancelled)
    assert all(h.score == 1 for h in candidates)


def test_total_deadline_includes_multibatch_and_cancels_transport(monkeypatch):
    calls, ended = [], []
    async def answer(url, payload):
        calls.append(payload["metadata"]["inference_deadline_seconds"])
        try:
            await asyncio.sleep(.12)
            return response(payload)
        finally:
            ended.append(True)
    install(monkeypatch, answer)
    began = time.monotonic()
    with pytest.raises(RuntimeError):
        scorer(timeout=.22).rerank("query", hits(90))
    assert time.monotonic() - began < .5
    assert len(calls) == len(ended) and len(calls) >= 2
    if len(calls) == 3:
        assert calls[2] < calls[0] - .08


def test_queue_wait_is_bounded_without_gateway_request(monkeypatch):
    async def forbidden(url, payload):
        pytest.fail("queued request must not reach gateway")
    install(monkeypatch, forbidden)
    qwen._SLOTS.acquire()
    qwen._SLOTS.acquire()
    try:
        with pytest.raises(RuntimeError):
            scorer(timeout=.1).rerank("query", hits())
    finally:
        qwen._SLOTS.release()
        qwen._SLOTS.release()


@pytest.mark.parametrize("via_event", [False, True])
def test_cancellation_propagates_and_releases_slots(monkeypatch, via_event):
    started, ended = threading.Event(), threading.Event()
    async def answer(url, payload):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            ended.set()
    install(monkeypatch, answer)
    async def run():
        event = threading.Event()
        token = rerank_cancel_event.set(event)
        try:
            task = asyncio.create_task(scorer().arerank("query", hits()))
            while not started.is_set():
                await asyncio.sleep(.001)
            if via_event:
                event.set()
            else:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            rerank_cancel_event.reset(token)
    asyncio.run(run())
    assert ended.is_set()
    assert qwen._SLOTS.acquire(blocking=False)
    assert qwen._SLOTS.acquire(blocking=False)
    qwen._SLOTS.release()
    qwen._SLOTS.release()


def test_large_pool_reserves_chat_and_output_overhead_within_global_64k():
    r = scorer()
    documents = ["1234567890" * 400] * 200
    cost = 0
    for start in range(0, len(documents), 32):
        batch = documents[start:start + 32]
        p = r._payload("invoice", batch, [f"D{i+1}" for i in range(start, start + len(batch))],
                       min(qwen.PROMPT_TOKEN_BUDGET, 64000 // 7))
        cost += len(json.dumps(p, ensure_ascii=False).encode()) + p["max_tokens"] + 512
    assert cost <= 64000


def test_impossible_overhead_rejected_and_empty_pool_never_calls_gateway(monkeypatch):
    async def forbidden(url, payload):
        pytest.fail("no request expected")
    install(monkeypatch, forbidden)
    assert scorer().rerank("query", []) == []
    with pytest.raises(ValueError, match="prompt budget"):
        scorer()._payload("query", ["document"], ["D1"], 1)
    with pytest.raises(RuntimeError):
        scorer().rerank("1" * 2000, hits(90))
    event = threading.Event()
    event.set()
    token = rerank_cancel_event.set(event)
    try:
        with pytest.raises(asyncio.CancelledError):
            scorer().rerank("query", hits())
    finally:
        rerank_cancel_event.reset(token)


def test_process_wide_slots_bound_overlapping_fresh_adapters(monkeypatch):
    active = peak = 0
    async def answer(url, payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.02)
            return response(payload)
        finally:
            active -= 1
    install(monkeypatch, answer)
    async def run():
        results = await asyncio.gather(*(scorer().arerank("query", hits(33)) for _ in range(3)))
        assert all(len(result) == 33 for result in results)
    asyncio.run(run())
    assert peak == 2 and active == 0


def test_refusal_remembered_without_retries_and_valid_recovery_required(monkeypatch):
    from core.resilience import CIRCUITS, EndpointCircuits
    now = [0.0]
    circuit = EndpointCircuits(clock=lambda: now[0])
    monkeypatch.setattr(qwen, "CIRCUITS", circuit)
    calls = []
    mode = ["refused"]
    async def answer(url, payload):
        calls.append(url)
        if mode[0] == "refused":
            return response(payload, status=402)
        if mode[0] == "partial":
            return response(payload, content='{"scores":{}}')
        return response(payload)
    install(monkeypatch, answer)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            scorer().rerank("query", hits())
    assert len(calls) == 1 and circuit.tripped()
    now[0] += 100000
    mode[0] = "partial"
    with pytest.raises(RuntimeError):
        scorer().rerank("query", hits())
    assert circuit.tripped(), "invalid HTTP 200 cannot recover circuit"
    now[0] += 100000
    mode[0] = "valid"
    assert len(scorer().rerank("query", hits())) == 3
    assert not circuit.tripped() and not CIRCUITS.tripped()
