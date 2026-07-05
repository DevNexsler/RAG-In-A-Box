# Contract tests for the staging provider simulator (staging/provider_sim/app.py).
#
# In-process via httpx.ASGITransport — no docker, unit tier.
# The sim must speak the exact HTTP dialects production provider code speaks:
# OpenRouter (embeddings + chat), DeepInfra (rerank), DeepSeek OCR2, Ollama.

import importlib.util
import json
import math
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "staging" / "provider_sim" / "app.py"
# .evals is gitignored, so linked worktrees don't carry it — prefer the
# repo-relative path, fall back to the main checkout's copy.
_TRACES_CANDIDATES = (
    ROOT / ".evals" / "llm-traces",
    Path("/home/danpark/projects/RAG-in-a-Box/.evals/llm-traces"),
)
TRACES_DIR = next(
    (p for p in _TRACES_CANDIDATES if p.is_dir()), _TRACES_CANDIDATES[0]
)

# Keys parse_enrichment_response / the enrichment JSON schema require
# (doc_enrichment.py: _ENRICHMENT_KEYS_RAW + _CONTEXT_KEYS_RAW).
ENRICHMENT_KEYS = [
    "summary",
    "doc_type",
    "entities_people",
    "entities_places",
    "entities_orgs",
    "entities_dates",
    "topics",
    "keywords",
    "key_facts",
    "suggested_tags",
    "suggested_folder",
    "importance",
    "atomic_entities_people",
    "atomic_entities_places",
    "atomic_entities_orgs",
    "atomic_entities_dates",
    "atomic_topics",
    "context_entities_people",
    "context_entities_places",
    "context_entities_orgs",
    "context_entities_dates",
    "context_topics",
    "context_key_facts",
    "context_relationship",
    "context_confidence",
    "context_source_message_ids",
    "context_warning",
]


def _load_module():
    spec = importlib.util.spec_from_file_location("provider_sim_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sim_module():
    return _load_module()


@pytest.fixture(scope="module")
def app(sim_module):
    return sim_module.app


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://provider-sim:9999"
    ) as c:
        await c.post("/admin/reset")
        yield c


def _chat_payload(text, response_format=None, model="openai/gpt-4.1-mini"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a document metadata extractor."},
            {"role": "user", "content": text},
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---------------------------------------------------------------------------
# OpenRouter embeddings
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_embeddings_shape_and_order(client):
    inputs = ["first document text", "second entirely different text", "third one"]
    resp = await client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding", "input": inputs},
        headers={"Authorization": "Bearer sk-fake-key-ignored"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == len(inputs)
    for i, item in enumerate(data):
        assert item["index"] == i
        assert len(item["embedding"]) == 768
        assert all(isinstance(v, float) for v in item["embedding"])


@pytest.mark.anyio
async def test_embeddings_deterministic_and_normalized(client):
    async def embed(text):
        resp = await client.post(
            "/api/v1/embeddings", json={"model": "m", "input": [text]}
        )
        return resp.json()["data"][0]["embedding"]

    a1 = await embed("the quick brown fox")
    a2 = await embed("the quick brown fox")
    b = await embed("completely unrelated insurance claim")
    assert a1 == a2, "same text must give identical vectors"
    assert a1 != b, "different text must give different vectors"
    norm = math.sqrt(sum(v * v for v in a1))
    assert abs(norm - 1.0) < 1e-6, "vectors must be unit-norm"


# ---------------------------------------------------------------------------
# OpenRouter chat completions — enrichment JSON, plain text, media routing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_enrichment_json_schema(client):
    resp = await client.post(
        "/api/v1/chat/completions",
        json=_chat_payload(
            "Extract metadata from this document.\n\nInsurance claim #123.",
            response_format={"type": "json_schema", "json_schema": {"name": "enrichment"}},
        ),
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # must be VALID JSON
    for key in ENRICHMENT_KEYS:
        assert key in parsed, f"enrichment JSON missing required key: {key}"
    assert isinstance(parsed["summary"], str) and parsed["summary"]
    assert isinstance(parsed["topics"], list)
    assert isinstance(parsed["key_facts"], list)
    assert isinstance(parsed["doc_type"], list)
    assert isinstance(parsed["importance"], (int, float))
    assert 0.0 <= float(parsed["importance"]) <= 1.0


@pytest.mark.anyio
async def test_chat_enrichment_json_object_and_determinism(client):
    payload = _chat_payload(
        "Some other document text", response_format={"type": "json_object"}
    )
    r1 = await client.post("/api/v1/chat/completions", json=payload)
    r2 = await client.post("/api/v1/chat/completions", json=payload)
    c1 = r1.json()["choices"][0]["message"]["content"]
    c2 = r2.json()["choices"][0]["message"]["content"]
    assert c1 == c2, "same input must yield identical content"
    json.loads(c1)
    # Different input text must produce different deterministic values
    r3 = await client.post(
        "/api/v1/chat/completions",
        json=_chat_payload("A different doc", response_format={"type": "json_object"}),
    )
    assert r3.json()["choices"][0]["message"]["content"] != c1


@pytest.mark.anyio
async def test_chat_plain_text(client):
    resp = await client.post(
        "/api/v1/chat/completions", json=_chat_payload("Summarize this text please.")
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content
    with pytest.raises(json.JSONDecodeError):
        # plain-text branch, not the JSON branch
        json.loads(content)


@pytest.mark.anyio
async def test_chat_media_audio_routing(client):
    payload = {
        "model": "gpt-4o-audio",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRiQAAABXQVZF", "format": "wav"},
                    },
                ],
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.0,
    }
    r1 = await client.post("/api/v1/chat/completions", json=payload)
    r2 = await client.post("/api/v1/chat/completions", json=payload)
    c1 = r1.json()["choices"][0]["message"]["content"]
    assert c1.startswith("[transcript] ")
    assert c1 == r2.json()["choices"][0]["message"]["content"]
    # Different media payload -> different marker
    payload2 = json.loads(json.dumps(payload))
    payload2["messages"][0]["content"][1]["input_audio"]["data"] = "T3RoZXJBdWRpbw=="
    r3 = await client.post("/api/v1/chat/completions", json=payload2)
    c3 = r3.json()["choices"][0]["message"]["content"]
    assert c3.startswith("[transcript] ")
    assert c3 != c1


@pytest.mark.anyio
async def test_chat_media_video_routing(client):
    payload = {
        "model": "qwen-vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                ],
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.0,
    }
    resp = await client.post("/api/v1/chat/completions", json=payload)
    content = resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("[transcript] ")


# ---------------------------------------------------------------------------
# DeepInfra rerank — model name contains a slash (path param)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rerank_scores_lexical_overlap(client):
    query = "roof damage insurance claim"
    docs = [
        "recipe for chocolate cake with vanilla frosting",
        "the roof damage insurance claim was filed in March",
        "roof repair notes",
    ]
    resp = await client.post(
        "/v1/inference/Qwen/Qwen3-Reranker-8B",
        json={"queries": [query], "documents": docs},
    )
    assert resp.status_code == 200
    scores = resp.json()["scores"]
    assert len(scores) == len(docs)
    assert all(isinstance(s, (int, float)) for s in scores)
    # doc 1 contains every query word -> must outrank both others
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    # determinism
    resp2 = await client.post(
        "/v1/inference/Qwen/Qwen3-Reranker-8B",
        json={"queries": [query], "documents": docs},
    )
    assert resp2.json()["scores"] == scores


# ---------------------------------------------------------------------------
# DeepSeek OCR2 — multipart /extract and /describe
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/extract", "/describe"])
async def test_ocr_multipart(client, route):
    files = {"file": ("scan_001.png", b"\x89PNG fake image bytes", "image/png")}
    r1 = await client.post(route, files=files)
    assert r1.status_code == 200
    text = r1.json()["text"]
    assert text.startswith("[ocr] ")
    assert "scan_001.png" in text
    r2 = await client.post(route, files=files)
    assert r2.json()["text"] == text, "same file must give identical OCR text"
    r3 = await client.post(
        route, files={"file": ("scan_001.png", b"different bytes", "image/png")}
    )
    assert r3.json()["text"] != text, "different bytes must change OCR text"


# ---------------------------------------------------------------------------
# Ollama /api/chat — NDJSON streaming and non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ollama_stream_true_ndjson(client):
    resp = await client.post(
        "/api/chat",
        json={
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": "describe the attached image"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert len(lines) >= 2
    for chunk in lines[:-1]:
        assert chunk["done"] is False
        assert isinstance(chunk["message"]["content"], str)
    final = lines[-1]
    assert final["done"] is True
    assert final["message"]["content"] == ""
    assembled = "".join(chunk["message"]["content"] for chunk in lines)
    assert assembled


@pytest.mark.anyio
async def test_ollama_stream_false(client):
    resp = await client.post(
        "/api/chat",
        json={
            "model": "qwen3-vl",
            "messages": [{"role": "user", "content": "describe the attached image"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["message"]["content"], str)
    assert body["message"]["content"]


# ---------------------------------------------------------------------------
# Webhook sink
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sink_roundtrip_and_reset(client):
    p1 = {"event": "indexed", "doc_id": "abc"}
    p2 = {"event": "failed", "doc_id": "def"}
    for p in (p1, p2):
        resp = await client.post("/hooks/sink", json=p)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
    received = (await client.get("/hooks/received")).json()
    assert received["events"] == [p1, p2]
    await client.post("/admin/reset")
    received = (await client.get("/hooks/received")).json()
    assert received["events"] == []


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fault_armed_429_exhausts(client):
    arm = await client.post(
        "/admin/fault",
        json={"route_prefix": "/api/v1/embeddings", "fault": "429", "times": 2},
    )
    assert arm.status_code == 200
    payload = {"model": "m", "input": ["hello"]}
    for _ in range(2):
        resp = await client.post("/api/v1/embeddings", json=payload)
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "0"
    resp = await client.post("/api/v1/embeddings", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1
    # unrelated routes were never affected
    resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_fault_header_garbage(client):
    resp = await client.post(
        "/api/v1/embeddings",
        json={"model": "m", "input": ["hello"]},
        headers={"X-Sim-Fault": "garbage"},
    )
    assert resp.status_code == 200
    assert resp.text == "not json {"
    assert resp.headers["content-type"].startswith("application/json")
    with pytest.raises(json.JSONDecodeError):
        resp.json()
    # single-shot: next request without header is normal
    resp = await client.post("/api/v1/embeddings", json={"model": "m", "input": ["x"]})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_fault_header_timeout_delays(client):
    start = time.monotonic()
    resp = await client.post(
        "/api/v1/embeddings",
        json={"model": "m", "input": ["hello"]},
        headers={"X-Sim-Fault": "timeout", "X-Sim-Fault-Seconds": "0.2"},
    )
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_fault_armed_timeout_delays_then_recovers(client):
    arm = await client.post(
        "/admin/fault",
        json={
            "route_prefix": "/api/v1/embeddings",
            "fault": "timeout",
            "times": 1,
            "seconds": 0.1,
        },
    )
    assert arm.status_code == 200
    payload = {"model": "m", "input": ["hello"]}
    start = time.monotonic()
    resp = await client.post("/api/v1/embeddings", json=payload)
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert elapsed >= 0.1, "armed timeout fault must delay the first call"
    start = time.monotonic()
    resp = await client.post("/api/v1/embeddings", json=payload)
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert elapsed < 0.1, "charge exhausted: second call must be fast again"


@pytest.mark.anyio
async def test_admin_fault_rejects_unknown_fault_name(client):
    resp = await client.post(
        "/admin/fault",
        json={"route_prefix": "/api/v1/embeddings", "fault": "flaky", "times": 1},
    )
    assert resp.status_code == 400, "unknown fault names must not silently no-op"
    # nothing was armed — traffic flows normally
    resp = await client.post("/api/v1/embeddings", json={"model": "m", "input": ["x"]})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_admin_fault_armed_count_excludes_exhausted(client):
    await client.post(
        "/admin/fault",
        json={"route_prefix": "/api/v1/embeddings", "fault": "429", "times": 1},
    )
    resp = await client.post("/api/v1/embeddings", json={"model": "m", "input": ["x"]})
    assert resp.status_code == 429  # consumes the only charge
    arm = await client.post(
        "/admin/fault",
        json={"route_prefix": "/v1/inference", "fault": "429", "times": 1},
    )
    assert arm.json()["armed"] == 1, "exhausted faults must not inflate the count"


@pytest.mark.anyio
async def test_hooks_received_returns_copy_not_live_list(client, sim_module):
    # Over HTTP serialization always copies, so assert directly on the handler:
    # it must return a snapshot, not the live SINK_EVENTS list.
    await client.post("/hooks/sink", json={"event": "one"})
    body = await sim_module.hooks_received()
    assert body["events"] == [{"event": "one"}]
    assert body["events"] is not sim_module.SINK_EVENTS


@pytest.mark.anyio
async def test_fault_armed_garbage_with_reset(client):
    await client.post(
        "/admin/fault",
        json={"route_prefix": "/v1/inference", "fault": "garbage", "times": 1},
    )
    resp = await client.post(
        "/v1/inference/Qwen/Qwen3-Reranker-8B",
        json={"queries": ["q"], "documents": ["d"]},
    )
    assert resp.text == "not json {"
    # reset clears fault state too
    await client.post(
        "/admin/fault",
        json={"route_prefix": "/v1/inference", "fault": "429", "times": 5},
    )
    await client.post("/admin/reset")
    resp = await client.post(
        "/v1/inference/Qwen/Qwen3-Reranker-8B",
        json={"queries": ["q"], "documents": ["d"]},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fidelity vs recorded production traces (skipped when traces absent)
# ---------------------------------------------------------------------------


def _load_recorded_chat_response():
    if not TRACES_DIR.is_dir():
        return None
    for trace_file in sorted(TRACES_DIR.glob("*.jsonl")):
        try:
            with open(trace_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue  # skip malformed production-recorded lines
                    response = entry.get("response")
                    url = entry.get("request", {}).get("url", "")
                    if (
                        entry.get("success")
                        and "chat/completions" in url
                        and isinstance(response, dict)
                        and "choices" in response
                    ):
                        return response
        except OSError:
            continue
    return None


_RECORDED = _load_recorded_chat_response()


@pytest.mark.skipif(
    _RECORDED is None, reason="no recorded llm traces in .evals/llm-traces"
)
@pytest.mark.anyio
async def test_fidelity_chat_response_keys_match_recorded_trace(client):
    resp = await client.post(
        "/api/v1/chat/completions",
        json=_chat_payload(
            "Extract metadata from this document.",
            response_format={"type": "json_object"},
        ),
    )
    sim_body = resp.json()
    missing = [k for k in _RECORDED if k not in sim_body]
    assert not missing, f"sim chat response missing recorded top-level keys: {missing}"
