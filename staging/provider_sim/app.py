"""Provider simulator for the staging gate.

Speaks the exact HTTP dialects the production doc-organizer code speaks:

- OpenRouter:  POST /api/v1/embeddings, POST /api/v1/chat/completions
- DeepInfra:   POST /v1/inference/{model}  (model contains slashes)
- DeepSeek OCR2: POST /extract, POST /describe  (multipart, field "file")
- Ollama:      POST /api/chat  (NDJSON streaming or single JSON)
- Webhook sink + fault-injection admin endpoints for gate tests.

All responses are deterministic: derived from content hashes, never from
randomness or wall-clock time. State (sink, faults) is in-memory only.
"""

import asyncio
import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="provider-sim")

# --------------------------------------------------------------------------
# In-memory state (single event loop; no locks needed)
# --------------------------------------------------------------------------

SINK_EVENTS: list[Any] = []
ARMED_FAULTS: list[dict[str, Any]] = []  # {route_prefix, fault, times, seconds}

# The embedding model's context window, expressed in characters: a hermetic
# image can't fetch BPE ranks, so the sim has no tokenizer. The staging config
# sets embeddings.max_input_tokens far enough below this that any input the
# client bounded is comfortably under the window whatever its token density,
# and any unbounded one is far over it — which is what a real provider does
# with an oversized input: reject the WHOLE batch, deterministically (#0569).
MAX_EMBED_INPUT_CHARS = 200_000

# --------------------------------------------------------------------------
# Deterministic helpers
# --------------------------------------------------------------------------


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def fake_embedding(text: str, dim: int = 768) -> list[float]:
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = [(h[i % 32] / 255.0) - 0.5 for i in range(dim)]
    for tok in text.lower().split()[:64]:
        th = hashlib.md5(tok.encode()).digest()
        for j in range(dim):
            vec[j] += ((th[j % 16] / 255.0) - 0.5) * 0.1
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# Keys required by doc_enrichment.py (_ENRICHMENT_KEYS_RAW + _CONTEXT_KEYS_RAW).
_ENRICHMENT_STRING_KEYS = (
    "suggested_folder",
    "context_relationship",
    "context_confidence",
    "context_warning",
)
_ENRICHMENT_LIST_KEYS = (
    "doc_type",
    "entities_people",
    "entities_places",
    "entities_orgs",
    "entities_dates",
    "topics",
    "keywords",
    "key_facts",
    "suggested_tags",
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
    "context_source_message_ids",
)


# Summary the overshoot_budget fault returns, so a test can tell that answer
# apart from the one a (wasted) retry would have produced.
_OVERSHOOT_BUDGET_SUMMARY = "Overshot-budget enrichment marker."


def _fake_enrichment(text: str) -> str:
    """Minimal valid enrichment JSON with values derived from the text hash."""
    h = _sha12(text.encode())
    obj: dict[str, Any] = {
        "summary": f"Simulated enrichment summary for document {h}.",
        "importance": round(int(h[:2], 16) / 255.0, 3),
    }
    for key in _ENRICHMENT_STRING_KEYS:
        obj[key] = ""
    obj["suggested_folder"] = f"sim/{h[:4]}"
    for key in _ENRICHMENT_LIST_KEYS:
        obj[key] = [f"sim-{key.replace('_', '-')}-{h[:6]}"]
    return json.dumps(obj)


def _media_parts(messages: list[Any]) -> list[dict]:
    parts = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "input_audio",
                    "video_url",
                ):
                    parts.append(part)
    return parts


def _messages_text(messages: list[Any]) -> str:
    chunks = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def _lexical_overlap(query: str, doc: str) -> float:
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    if not q_tokens:
        return 0.0
    return round(len(q_tokens & d_tokens) / len(q_tokens), 6)


# --------------------------------------------------------------------------
# Fault injection — implemented once, as middleware
# --------------------------------------------------------------------------


def _fault_response(fault: str) -> Response | None:
    if fault == "429":
        return JSONResponse(
            {"error": {"code": 429, "message": "simulated rate limit"}},
            status_code=429,
            headers={"Retry-After": "0"},
        )
    if fault == "garbage":
        return Response(
            content="not json {", status_code=200, media_type="application/json"
        )
    if fault == "reasoning_only":
        return JSONResponse(
            {
                "id": "sim-reasoning-only",
                "object": "chat.completion",
                "model": "sim-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "simulated output-budget exhaustion",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 1,
                    "total_tokens": 13,
                },
            }
        )
    if fault == "overshoot_budget":
        # A COMPLETE structured answer that bills more completion tokens than
        # the request allowed. Production's LiteLLM route bills reasoning
        # tokens it then strips from the message, so `completion_tokens >
        # max_tokens` says nothing about truncation — 272 of 274 such
        # responses were valid enrichment JSON (#1097). The distinctive
        # summary is how a test tells this answer from the one a retry would
        # get: keeping it must cost exactly one call.
        enrichment = json.loads(_fake_enrichment(""))
        enrichment["summary"] = _OVERSHOOT_BUDGET_SUMMARY
        return JSONResponse(
            {
                "id": "sim-overshoot-budget",
                "object": "chat.completion",
                "model": "sim-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(enrichment),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    # Far above any configured max_output_tokens.
                    "completion_tokens": 6201,
                    "total_tokens": 6213,
                },
            }
        )
    if fault == "hangup":
        # Every other fault here is an ANSWER the provider gave. This one models
        # the provider not being there at all — a container recreate mid-flight
        # (#0619) — by cutting the connection after the headers, with no body.
        # The client raises httpx.RemoteProtocolError, the connection-level
        # failure shape production sees as [Errno 111] Connection refused.
        async def _cut():
            raise RuntimeError("simulated connection hangup")
            yield b""  # unreachable; makes _cut an async generator

        return StreamingResponse(_cut(), media_type="application/json")
    return None  # "timeout" delays, then falls through to normal handling


@app.middleware("http")
async def fault_middleware(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/admin"):
        # Single-shot header fault for direct testing.
        header_fault = request.headers.get("X-Sim-Fault")
        if header_fault:
            if header_fault == "timeout":
                await asyncio.sleep(
                    float(request.headers.get("X-Sim-Fault-Seconds", "10"))
                )
            else:
                response = _fault_response(header_fault)
                if response is not None:
                    return response
        else:
            for armed in ARMED_FAULTS:
                if armed["times"] > 0 and path.startswith(armed["route_prefix"]):
                    armed["times"] -= 1
                    if armed["times"] <= 0:
                        ARMED_FAULTS.remove(armed)  # keep the armed count meaningful
                    if armed["fault"] == "timeout":
                        await asyncio.sleep(armed["seconds"])
                    else:
                        response = _fault_response(armed["fault"])
                        if response is not None:
                            return response
                    break
    return await call_next(request)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/")
async def health() -> dict:
    return {"ok": True}


@app.post("/api/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    for i, text in enumerate(inputs):
        if not text:
            # The gateway's own request validation, which runs before any
            # upstream sees the batch: an empty input is rejected outright and
            # takes the WHOLE batch with it, naming the offending slot
            # (measured 2026-08-27, zod `too_small`). Because the input never
            # gains characters, the rejection is permanent (#1687).
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": json.dumps(
                            [
                                {
                                    "origin": "string",
                                    "code": "too_small",
                                    "minimum": 1,
                                    "inclusive": True,
                                    "path": ["input", i],
                                    "message": (
                                        "Too small: expected string to have "
                                        ">=1 characters"
                                    ),
                                }
                            ],
                            indent=2,
                        ),
                    }
                },
            )
        if not text.strip():
            # One layer further in: an input that normalizes to nothing is
            # rejected by the upstream provider instead. The real route does
            # this *intermittently* — `[" "]` measured 400/400/200 across three
            # runs, depending on which provider the gateway picked. The sim
            # answers deterministically, and rejects: a simulator stricter than
            # the real route cannot produce a false green, a more permissive
            # one can (#1658).
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": (
                            'HTTP 400: {"detail":"Prompt must not be empty"}'
                        ),
                    }
                },
            )
        if len(text) > MAX_EMBED_INPUT_CHARS:
            # Same shape OpenRouter returns when one input overruns the model's
            # context window: the WHOLE batch is rejected with a 400, and it
            # never gets smaller on retry (#0569).
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": (
                            "This model's maximum context length is "
                            f"{MAX_EMBED_INPUT_CHARS} characters. However, your "
                            f"prompt contains {len(text)}. Please reduce the "
                            "length of the input prompt."
                        ),
                    }
                },
            )
    return {
        "object": "list",
        "model": body.get("model", "sim-embed"),
        "data": [
            {"object": "embedding", "index": i, "embedding": fake_embedding(text)}
            for i, text in enumerate(inputs)
        ],
    }


@app.post("/api/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await request.json()
    messages = body.get("messages", [])
    media = _media_parts(messages)
    text = _messages_text(messages)

    if media:
        marker = _sha12(json.dumps(media, sort_keys=True).encode())
        kinds = "+".join(sorted({part.get("type", "") for part in media}))
        content = f"[transcript] simulated {kinds} transcript {marker}"
    else:
        response_format = body.get("response_format") or {}
        if response_format.get("type") in ("json_schema", "json_object"):
            content = _fake_enrichment(text)
        else:
            content = (
                f"Simulated summary ({_sha12(text.encode())}): "
                f"{text[:200] or 'empty prompt'}"
            )

    prompt_tokens = max(1, len(text.split()))
    completion_tokens = max(1, len(content.split()))
    # Extra top-level keys mirror recorded OpenRouter/litellm responses
    # (production only parses choices[0].message.content).
    return {
        "id": f"sim-{_sha12(json.dumps(body, sort_keys=True).encode())}",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "sim-model"),
        "provider": "provider-sim",
        "system_fingerprint": None,
        "service_tier": "default",
        "metadata": {},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/inference/{model:path}")
async def rerank(model: str, request: Request) -> dict:
    body = await request.json()
    queries = body.get("queries", [])
    query = queries[0] if queries else ""
    documents = body.get("documents", [])
    return {
        "model": model,
        "scores": [_lexical_overlap(query, doc) for doc in documents],
    }


async def _ocr(file: UploadFile) -> dict:
    data = await file.read()
    return {"text": f"[ocr] {file.filename} {_sha12(data)}"}


@app.post("/extract")
async def ocr_extract(file: UploadFile) -> dict:
    return await _ocr(file)


@app.post("/describe")
async def ocr_describe(file: UploadFile) -> dict:
    return await _ocr(file)


@app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    marker = _sha12(json.dumps(messages, sort_keys=True).encode())
    text = _messages_text(messages)
    content = f"Simulated ollama response {marker}: {text[:120] or 'empty prompt'}"

    if body.get("stream", True):
        words = content.split(" ")
        # Three deterministic chunks + terminal done marker.
        step = max(1, math.ceil(len(words) / 3))
        pieces = [" ".join(words[i : i + step]) for i in range(0, len(words), step)]

        async def ndjson():
            for i, piece in enumerate(pieces):
                prefix = " " if i else ""
                yield (
                    json.dumps(
                        {
                            "model": body.get("model", "sim"),
                            "message": {"role": "assistant", "content": prefix + piece},
                            "done": False,
                        }
                    )
                    + "\n"
                )
            yield (
                json.dumps(
                    {
                        "model": body.get("model", "sim"),
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                    }
                )
                + "\n"
            )

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    return {
        "model": body.get("model", "sim"),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


# --------------------------------------------------------------------------
# Webhook sink + admin
# --------------------------------------------------------------------------


@app.post("/hooks/sink")
async def hooks_sink(request: Request) -> dict:
    SINK_EVENTS.append(await request.json())
    return {"ok": True}


@app.get("/hooks/received")
async def hooks_received() -> dict:
    return {"events": list(SINK_EVENTS)}


@app.post("/admin/reset")
async def admin_reset() -> dict:
    SINK_EVENTS.clear()
    ARMED_FAULTS.clear()
    return {"ok": True}


_KNOWN_FAULTS = {
    "429",
    "timeout",
    "garbage",
    "reasoning_only",
    "overshoot_budget",
    "hangup",
}


@app.post("/admin/fault")
async def admin_fault(request: Request):
    body = await request.json()
    fault = body.get("fault")
    if fault not in _KNOWN_FAULTS:
        # A typo'd fault name must not silently no-op — downstream gate tests
        # would misread the resulting normal responses as "retry recovered".
        return JSONResponse(
            {"error": f"unknown fault {fault!r}; expected one of {sorted(_KNOWN_FAULTS)}"},
            status_code=400,
        )
    ARMED_FAULTS.append(
        {
            "route_prefix": body["route_prefix"],
            "fault": fault,
            "times": int(body.get("times", 1)),
            "seconds": float(body.get("seconds", 10)),
        }
    )
    armed = sum(1 for f in ARMED_FAULTS if f["times"] > 0)
    return {"ok": True, "armed": armed}
