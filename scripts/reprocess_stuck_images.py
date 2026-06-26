#!/usr/bin/env python3
"""End-to-end verify: reprocess the 3 previously empty-describe docs through the
real single-doc pipeline (now with reasoning suppressed) and confirm real chunks."""
import time
import lancedb
from flow_index_vault import index_document_flow

DOCS = {
    "001bf": "quo-attachments/jordan-grace-wilder/2026-05/2026-05-20T16-53-36Z__msg23234__mm3781@001bf@.jpg",
    "001gS": "quo-attachments/14015002730/2026-05/2026-05-27T22-59-01Z__msg26182__mm4076@001gS@.jpg",
    "001iP": "quo-attachments/destiny/2026-05/2026-05-29T18-18-14Z__msg27291__mm4166@001iP@.png",
}

for name, rel in DOCS.items():
    t0 = time.time()
    res = index_document_flow("config.yaml", target=rel, source_name="documents", force=True)
    dt = round(time.time() - t0, 1)
    did = res.get("doc_id", f"documents::{name}")
    tbl = lancedb.connect("/data/index").open_table("chunks")
    df = tbl.search().where(f"doc_id = '{did}'").limit(3).to_pandas()
    txt = (str(df.iloc[0]["text"])[:200] if len(df) else "")
    print(f"{name}: status={res.get('status')} chunks={len(df)} t={dt}s", flush=True)
    print(f"   text[:200]={txt!r}", flush=True)
