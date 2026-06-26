#!/usr/bin/env python3
"""Verify the PRIMARY fix: make qwen3-vl emit content by suppressing its reasoning.
think:False is ignored by qwen3-vl:8b, so test other suppression methods and see
which yields non-empty content_len at the prod budget (num_predict=5120).
Run inside the doc-organizer container (-w /app)."""
import base64, io, os, time
import httpx
from PIL import Image
import providers.ocr.ollama_vision as ov

URL = "http://192.168.68.70:11434/api/chat"
MODEL = "qwen3-vl:8b"
BASE = ov._DESCRIBE_PROMPT
IMGS = {
    "001bf": "quo-attachments/jordan-grace-wilder/2026-05/2026-05-20T16-53-36Z__msg23234__mm3781@001bf@.jpg",
    "001iP": "quo-attachments/destiny/2026-05/2026-05-29T18-18-14Z__msg27291__mm4166@001iP@.png",
    "001gS": "quo-attachments/14015002730/2026-05/2026-05-27T22-59-01Z__msg26182__mm4076@001gS@.jpg",
}


def b64(rel):
    im = Image.open(os.path.join("/data/documents", rel)).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call(name, messages, label, np=5120, think=False):
    payload = {"model": MODEL, "messages": messages, "stream": False, "think": think,
               "options": {"num_predict": np, "temperature": 0}, "keep_alive": "10m"}
    t0 = time.time()
    try:
        d = httpx.post(URL, json=payload, timeout=400.0).json()
    except Exception as e:
        print(f"  [{label}] {name}: ERROR {type(e).__name__}: {e}", flush=True); return None
    dt = time.time() - t0
    m = d.get("message", {}) or {}
    c = m.get("content") or ""; th = m.get("thinking") or ""
    ok = "  <-- WORKS" if len(c) > 0 else ""
    print(f"  [{label}] {name}: done={d.get('done_reason')} eval={d.get('eval_count')} "
          f"content_len={len(c)} thinking_len={len(th)} t={dt:.0f}s{ok}", flush=True)
    print(f"     content[:240]={c[:240]!r}", flush=True)
    return c


if __name__ == "__main__":
    img = b64(IMGS["001bf"])

    def umsg(text):
        return {"role": "user", "content": text, "images": [img]}

    # M1: qwen3 soft switch /no_think appended to the user prompt
    call("001bf", [umsg(BASE + "\n\n/no_think")], "suffix /no_think")
    # M2: /no_think as a system message
    call("001bf", [{"role": "system", "content": "/no_think"}, umsg(BASE)], "system /no_think")
    # M3: natural-language system instruction banning reasoning
    call("001bf", [{"role": "system", "content":
                    "You are a precise document OCR tool. Output ONLY the final answer "
                    "in the requested format. Do not produce any reasoning, analysis, "
                    "planning, or <think> blocks."}, umsg(BASE)], "system no-reason")
