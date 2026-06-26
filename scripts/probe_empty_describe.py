#!/usr/bin/env python3
"""Diagnose WHY qwen3-vl describe returns empty for the 3 skipped images.
Replicates the provider's request (think=False, thumbnail 1024, _DESCRIBE_PROMPT)
but NON-streaming so we can read done_reason / eval_count / content vs thinking.
Run inside the doc-organizer container (-w /app)."""
import base64, io, os, time
import httpx
from PIL import Image
import providers.ocr.ollama_vision as ov

URL = "http://192.168.68.70:11434/api/chat"
MODEL = "qwen3-vl:8b"
PROMPT = ov._DESCRIBE_PROMPT

IMGS = {
    "001bf": "quo-attachments/jordan-grace-wilder/2026-05/2026-05-20T16-53-36Z__msg23234__mm3781@001bf@.jpg",
    "001iP": "quo-attachments/destiny/2026-05/2026-05-29T18-18-14Z__msg27291__mm4166@001iP@.png",
    "001gS": "quo-attachments/14015002730/2026-05/2026-05-27T22-59-01Z__msg26182__mm4076@001gS@.jpg",
}


def b64(rel):
    im = Image.open(os.path.join("/data/documents", rel)).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def probe(name, rel, num_predict, think):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64(rel)]}],
        "stream": False, "think": think,
        "options": {"num_predict": num_predict, "temperature": 0},
        "keep_alive": "10m",
    }
    t0 = time.time()
    try:
        d = httpx.post(URL, json=payload, timeout=600.0).json()
    except Exception as e:
        print(f"  {name} np={num_predict} think={think}: ERROR {type(e).__name__}: {e}", flush=True)
        return
    dt = time.time() - t0
    m = d.get("message", {}) or {}
    c = (m.get("content") or "")
    th = (m.get("thinking") or "")
    print(f"  {name} np={num_predict} think={think}: done_reason={d.get('done_reason')} "
          f"eval_count={d.get('eval_count')} prompt_eval={d.get('prompt_eval_count')} "
          f"content_len={len(c)} thinking_len={len(th)} t={dt:.0f}s", flush=True)
    print(f"     content[:200]={c[:200]!r}", flush=True)
    if th:
        print(f"     thinking[:200]={th[:200]!r}", flush=True)


if __name__ == "__main__":
    print("== 001bf (dense pay stub) — replicate prod, then higher budget, then think-visible ==", flush=True)
    probe("001bf", IMGS["001bf"], 5120, False)   # exact prod setting
    probe("001bf", IMGS["001bf"], 16384, False)  # does more token budget fix it?
    probe("001bf", IMGS["001bf"], 8192, True)    # surface the hidden reasoning
    print("== 001gS (MPO photo) — format/other cause ==", flush=True)
    probe("001gS", IMGS["001gS"], 5120, False)
