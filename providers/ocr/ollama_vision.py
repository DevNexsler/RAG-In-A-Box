"""OCR via Ollama vision-language models (e.g. qwen3-vl:8b)."""

import base64
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract ALL text from this image, preserving the original layout "
    "as closely as possible.\n\n"
    "After the extracted text, if the image contains any non-text visual elements "
    "(charts, graphs, diagrams, tables, maps, photos, signatures, stamps, logos), "
    "add a section starting with '--- Visual Elements ---' and briefly describe each one.\n\n"
    "If there is no text and no visual elements, return an empty string."
)

_DESCRIBE_PROMPT = (
    "Analyze this image thoroughly and provide:\n\n"
    "1. **TEXT**: Extract ALL visible text exactly as it appears, preserving layout. "
    "If no text is visible, write 'No visible text.'\n\n"
    "2. **DESCRIPTION**: Provide a detailed description of the image contents including:\n"
    "   - What type of image this is (photo, screenshot, diagram, map, chart, "
    "document scan, handwritten note, receipt, invoice, floor plan, etc.)\n"
    "   - Key subjects, objects, people, or scenes depicted\n"
    "   - Any data, measurements, or quantities shown\n"
    "   - Spatial layout and relationships between elements\n"
    "   - Colors, labels, annotations, or markings\n"
    "   - Context clues about what this image relates to\n\n"
    "Format your response as:\n"
    "--- Text ---\n"
    "[extracted text here]\n\n"
    "--- Description ---\n"
    "[detailed description here]"
)

# qwen3-vl:8b IGNORES the API's think=False and still emits a long <think>
# reasoning trace; on dense images (pay stubs, statements, busy screenshots) that
# trace alone consumes the entire num_predict budget, so the answer ("content")
# comes back empty (done_reason=length, content_len=0). Raising num_predict does
# not help (the reasoning just grows). A plain-language system instruction DOES
# suppress it — verified on a dense pay stub: content_len 0 -> 2962, done_reason
# length -> stop, within budget. This is the actual fix for the empty-describe
# class, distinct from the hang guards below.
_NO_REASONING_SYSTEM = (
    "You are a precise document OCR tool. Output ONLY the final answer in the "
    "requested format. Do not produce any reasoning, analysis, planning, or "
    "<think> blocks."
)


# Ollama processes one vision request at a time per model. With the indexer
# running 8-wide, naive concurrent calls queue server-side and expire against
# the HTTP timeout (901 describe timeouts in one run). Serialize app-side so
# waiting happens without burning the request timeout.
_VISION_GATE = threading.Semaphore(int(os.environ.get("OLLAMA_VISION_CONCURRENCY", "1")))

_MAX_IMAGE_DIM = 1024

# Output token caps (a ceiling, not a target — images that finish early stop
# early and pay nothing extra, so this does not slow or pad normal images).
# qwen3-vl emits a hidden reasoning trace even with think=False; on busier
# images that trace alone overran the old 800-token cap, leaving content empty
# and the image with only a metadata stub (~8.5% of images). A generous describe
# cap lets the answer land after even a long reasoning trace — the worst
# observed image needed ~4.5k tokens. The only images that take longer are the
# ones that would otherwise have produced no description at all.
_EXTRACT_NUM_PREDICT = 800
# 5120 comfortably covers the ~4.5k-token worst case described above while
# bounding the worst generation to ~4-5 min. The old 10240 let pathological
# generations run ~9-10 min — the main driver of the ~300s describe stalls
# (one slow describe holding the vision gate stalls the whole indexer). Floor
# is intentionally NOT 1500-2000 (that produced metadata-only stubs). Env-tunable.
_DESCRIBE_NUM_PREDICT = int(os.environ.get("OLLAMA_DESCRIBE_NUM_PREDICT", "5120"))

# An empty describe is almost always transient — a call that timed out or got
# starved under concurrent indexing load, not a deterministic refusal (the same
# image describes fine on a fresh call). Retry a few times with a short, growing
# backoff before falling back to a metadata-only stub.
_DESCRIBE_EMPTY_RETRIES = 2
_DESCRIBE_RETRY_BACKOFF = 2.0

# --- Hang guards ---
# Root cause of indexer freezes: every describe funnels through the single-permit
# _VISION_GATE, and the old stream=False call blocked in ONE recv for the entire
# generation (Ollama sends no bytes until done). One slow/stuck describe holding
# the gate froze all 8 worker threads. These bound it so a bad describe degrades
# one doc instead of freezing the run.
#
# Max silence between streamed chunks before httpx aborts the read. Covers
# time-to-first-token (image decode + prompt eval, ~15s warm); afterward tokens
# flow ~55ms apart, so a longer gap means a genuinely stuck generation.
_VISION_READ_GAP = float(os.environ.get("OLLAMA_VISION_READ_GAP", "90"))
# Bounded wait to ACQUIRE the vision gate — a backstop so that if a holder ever
# fails to release within its wall-clock deadline, waiters degrade their image
# rather than block forever. Must exceed the per-call wall deadline (self.timeout).
_VISION_GATE_TIMEOUT = float(os.environ.get("OLLAMA_VISION_GATE_TIMEOUT", "600"))

# NOTE: model keep-alive (how long ollama keeps qwen3-vl resident) is intentionally NOT
# set from this client. It lives on the Mac Mini as OLLAMA_KEEP_ALIVE=-1 (pin forever) —
# the single source of truth for the vision host's memory policy, alongside deepseek's
# idle_timeout=0. A per-request keep_alive here would OVERRIDE that server default, so we
# omit it and defer to the host (where both vision models are pinned; 32GB fits both).


def _downscale(image_bytes: bytes) -> bytes:
    """Resize large images before sending — vision prefill cost scales with
    pixels and attachment photos are often full-resolution camera shots."""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= _MAX_IMAGE_DIM:
            return image_bytes
        img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


class OllamaVisionOCR(OCRProvider):
    """Use an Ollama vision model for text extraction and image description."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:8b",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        logger.info("OllamaVisionOCR: %s model=%s", self.base_url, self.model)

    def _call(self, file_path: Path, prompt: str, num_predict: int) -> str:
        image_bytes = _downscale(file_path.read_bytes())
        b64 = base64.b64encode(image_bytes).decode("ascii")
        # Bounded acquire: a stuck prior describe can no longer freeze this
        # worker (and the whole pool) indefinitely. On timeout we raise, which
        # propagates to extractors' except-block -> note_degradation -> the
        # image is degraded to a metadata stub and indexing continues.
        if not _VISION_GATE.acquire(timeout=_VISION_GATE_TIMEOUT):
            raise TimeoutError(
                f"vision gate not acquired within {_VISION_GATE_TIMEOUT}s "
                f"(a prior describe is stuck); degrading {file_path.name}"
            )
        try:
            return self._request(b64, prompt, num_predict)
        finally:
            _VISION_GATE.release()

    def _request(self, b64: str, prompt: str, num_predict: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _NO_REASONING_SYSTEM},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            # Stream so httpx's read timeout becomes an inter-token gap (resets
            # per chunk) AND so we can enforce a hard wall-clock deadline by
            # checking elapsed time per chunk and closing the socket. With
            # stream=False, Ollama sent zero bytes until the ENTIRE generation
            # finished, so the call blocked in one recv for the whole generation
            # (~300s on verbose images) — and one such call holding _VISION_GATE
            # froze the whole 8-worker indexer.
            "stream": True,
            # think=False is sent but qwen3-vl:8b IGNORES it (it still reasons) —
            # _NO_REASONING_SYSTEM above is what actually forces a direct answer.
            # Kept anyway as a no-cost hint. (keep_alive is intentionally omitted — the
            # Mac Mini's OLLAMA_KEEP_ALIVE=-1 pins the model; see the module note above.)
            "think": False,
            # temperature=0 (greedy) -> deterministic, faithful description every
            # time (sampled decoding occasionally produced near-empty stubs).
            "options": {"num_predict": num_predict, "temperature": 0},
        }
        # connect short so a dead host can't hang; read = max silence BETWEEN
        # chunks (NOT a total bound). The total bound is the wall deadline below.
        timeout = httpx.Timeout(
            connect=10.0, read=_VISION_READ_GAP, write=30.0, pool=10.0
        )
        # SO_KEEPALIVE (tuned low) detects a half-open socket — peer gone without
        # FIN/RST — instead of blocking forever (the rare infinite-hang class).
        transport = httpx.HTTPTransport(
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 30),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 10),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 3),
            ]
        )
        deadline = time.monotonic() + self.timeout
        parts: list[str] = []
        with httpx.Client(transport=transport, timeout=timeout) as client:
            with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if time.monotonic() > deadline:
                        # Raising exits the context manager, which closes the
                        # socket and aborts the server-side generation.
                        raise TimeoutError(
                            f"ollama describe exceeded wall deadline "
                            f"{self.timeout:.0f}s"
                        )
                    if not line:
                        continue
                    obj = json.loads(line)
                    parts.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        break
        return "".join(parts).strip()

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        return self._call(file_path, _EXTRACT_PROMPT, _EXTRACT_NUM_PREDICT)

    def describe(self, file_path: str | Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        text = self._call(file_path, _DESCRIBE_PROMPT, _DESCRIBE_NUM_PREDICT)
        # Empty is almost always transient (timeout/contention under load), not
        # a deterministic refusal — retry before falling back to a metadata stub.
        attempt = 0
        while not text and attempt < _DESCRIBE_EMPTY_RETRIES:
            attempt += 1
            logger.warning(
                "Vision describe returned empty for %s; retrying (%d/%d)",
                file_path, attempt, _DESCRIBE_EMPTY_RETRIES,
            )
            time.sleep(_DESCRIBE_RETRY_BACKOFF * attempt)
            text = self._call(file_path, _DESCRIBE_PROMPT, _DESCRIBE_NUM_PREDICT)
        if not text:
            logger.warning(
                "Vision describe still empty after %d retries for %s",
                _DESCRIBE_EMPTY_RETRIES, file_path,
            )
        return text
