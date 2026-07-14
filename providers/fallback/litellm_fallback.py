"""One OpenAI-compatible LiteLLM client used as the fallback for every modality.

Contract (matches core.fallback.resolve_with_fallback's fallback_call):
  - unreachable -> raises a transient error (httpx.ConnectError / 5xx via is_transient)
  - reachable but empty -> returns ""
Per-modality difference is ONLY (endpoint, model, prompt, encoder).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Callable

import httpx

from core.resilience import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BACKOFF,
    TransientError,
    call_with_retry,
    is_transient,
)

Encoder = Callable[[Path, str], list]  # (path, prompt) -> OpenAI content parts


def image_encoder(path: Path, prompt: str) -> list:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lstrip(".") or "png"
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/{suffix};base64,{b64}"}},
    ]

def audio_encoder(path: Path, prompt: str) -> list:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    fmt = path.suffix.lstrip(".") or "mp3"
    return [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
    ]


def video_encoder(path: Path, prompt: str) -> list:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    fmt = path.suffix.lstrip(".") or "mp4"
    return [
        {"type": "text", "text": prompt},
        {"type": "video_url",
         "video_url": {"url": f"data:video/{fmt};base64,{b64}"}},
    ]

# The client shape below is identical for all three modalities.


class LiteLLMFallback:
    def __init__(self, endpoint: str, model: str, prompt: str, encoder: Encoder,
                 *, api_key: str | None = None, timeout: float = 300.0,
                 attempts: int = DEFAULT_ATTEMPTS,
                 backoff: tuple[float, ...] = DEFAULT_BACKOFF):
        self.base_url = endpoint.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.encoder = encoder
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.api_key = (api_key or os.environ.get("LITELLM_API_KEY", "")
                        or os.environ.get("LITELLM_MASTER_KEY", ""))

    def run(self, file_path: str | Path) -> str:
        path = Path(file_path)
        content = self.encoder(path, self.prompt)
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": content}],
                   "temperature": 0.0}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        def _once() -> str:
            resp = httpx.post(f"{self.base_url}/chat/completions",
                              json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()  # 5xx -> HTTPStatusError -> is_transient True
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

        # call_with_retry re-raises the transient exc UNCHANGED on exhaustion, so an
        # unreachable endpoint propagates as a transient error to resolve_with_fallback.
        try:
            return call_with_retry(_once, attempts=self.attempts, backoff=self.backoff,
                                   label=f"litellm-fallback {self.model}")
        except Exception as exc:
            if is_transient(exc):
                raise
            # A misconfigured/erroring fallback (401 bad key, 404 wrong model, 4xx,
            # malformed body) must be treated as UNREACHABLE -> transient, so the doc
            # retries and self-heals once the config is fixed. It must never cap the
            # doc (#0251) nor be mistaken for a confirmed blank (#0264).
            raise TransientError(
                f"litellm fallback {self.model} failed: {type(exc).__name__}: {exc}"
            ) from exc
