#!/usr/bin/env python3
"""Confirm the winning suppression (NL system instruction) generalizes to the
other two empty-describe images before coding the fix."""
import base64, io, os, time
import httpx
from PIL import Image
import providers.ocr.ollama_vision as ov

URL = "http://192.168.68.70:11434/api/chat"
MODEL = "qwen3-vl:8b"
BASE = ov._DESCRIBE_PROMPT
SYS = ("You are a precise document OCR tool. Output ONLY the final answer in the "
       "requested format. Do not produce any reasoning, analysis, planning, or "
       "<think> blocks.")
IMGS = {
    "001iP": "quo-attachments/destiny/2026-05/2026-05-29T18-18-14Z__msg27291__mm4166@001iP@.png",
    "001gS": "quo-attachments/14015002730/2026-05/2026-05-27T22-59-01Z__msg26182__mm4076@001gS@.jpg",
}


def b64(rel):
    im = Image.open(os.path.join("/data/documents", rel)).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def run(name, rel):
    payload = {"model": MODEL, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": BASE, "images": [b64(rel)]},
    ], "stream": False, "think": False,
        "options": {"num_predict": 5120, "temperature": 0}, "keep_alive": "10m"}
    t0 = time.time()
    d = httpx.post(URL, json=payload, timeout=400.0).json()
    dt = time.time() - t0
    m = d.get("message", {}) or {}
    c = m.get("content") or ""; th = m.get("thinking") or ""
    ok = "  <-- WORKS" if len(c) > 0 else "  <-- STILL EMPTY"
    print(f"  {name}: done={d.get('done_reason')} eval={d.get('eval_count')} "
          f"content_len={len(c)} thinking_len={len(th)} t={dt:.0f}s{ok}", flush=True)
    print(f"     content[:240]={c[:240]!r}", flush=True)


if __name__ == "__main__":
    for n, r in IMGS.items():
        run(n, r)
