# DeepSeek-OCR2 Local Service — Project Scope

## Overview

A lightweight local HTTP service that runs DeepSeek-OCR2 (3B vision-language model) on Apple Silicon via `mlx-vlm`. Replaces Gemini Cloud OCR for document indexing, eliminating per-page API costs.

## Problem

The Document Organizer indexes PDFs and images by OCR'ing them through Gemini 2.5 Flash (cloud). This costs money per page and requires an API key. A local OCR service using DeepSeek-OCR2 would be free, fast, and private.

## Architecture

```
Document Organizer (indexer)
    │
    ├── Ollama (embeddings, enrichment LLM)
    ├── llama-server (reranking)
    └── DeepSeek-OCR2 Service ← NEW (this project)
            │
            ├── mlx-vlm (Apple Silicon inference)
            └── FastAPI (HTTP API)
```

The OCR2 service is a **separate process** (separate repo, separate venv). The Document Organizer calls it via HTTP, same as it calls Gemini today. No in-process model loading — avoids the GPU memory contention issues we had with MLX.

## Integration Contract

The Document Organizer's `OCRProvider` interface requires two methods:

### `POST /extract` — PDF page OCR
- **Input**: Image file (PNG rendered from PDF page)
- **Output**: Extracted text preserving layout
- **Used by**: `extractors.py` → `_ocr_page()` renders PDF page to PNG, sends to OCR provider

### `POST /describe` — Standalone image analysis
- **Input**: Image file (PNG, JPG, etc.)
- **Output**: Extracted text + visual description of image contents
- **Used by**: `extractors.py` → `extract_image()` for standalone images (.png, .jpg in vault)

### Request Format
```
POST /extract
Content-Type: multipart/form-data
Body: file=<image_bytes>

Response: { "text": "extracted text..." }
```

```
POST /describe
Content-Type: multipart/form-data
Body: file=<image_bytes>

Response: { "text": "--- Text ---\n...\n--- Description ---\n..." }
```

### Health Check
```
GET /health
Response: { "status": "ok", "model": "deepseek-ocr2", "device": "mps" }
```

## Model Details

| Property | Value |
|----------|-------|
| Model | DeepSeek-OCR-2 (3B parameters) |
| Format | MLX (via mlx-community or manual conversion) |
| Memory | ~2-3 GB on Apple Silicon |
| Framework | mlx-vlm |
| Speed | Fast on M-series chips (Metal acceleration) |

## Project Structure

```
deepseek-ocr2-service/
├── server.py              # FastAPI app with /extract, /describe, /health
├── ocr_engine.py          # mlx-vlm model loading and inference
├── prompts.py             # OCR and describe prompts (mirroring Gemini prompts)
├── requirements.txt       # mlx-vlm, fastapi, uvicorn, python-multipart
├── config.yaml            # model name, host, port, max_tokens
├── README.md
└── tests/
    ├── test_extract.py    # Unit tests with sample images
    └── test_describe.py
```

## Prompts

The service should use equivalent prompts to the existing Gemini OCR provider for consistent output:

### Extract prompt (PDF pages)
```
Extract ALL text from this document page, preserving the original layout
as closely as possible.

After the extracted text, if the page contains any non-text visual elements
(charts, graphs, diagrams, tables, maps, photos, signatures, stamps, logos),
add a section starting with '--- Visual Elements ---' and briefly describe each one.

If there is no text and no visual elements, return an empty string.
```

### Describe prompt (standalone images)
```
Analyze this image thoroughly and provide:

1. TEXT: Extract ALL visible text exactly as it appears, preserving layout.
2. DESCRIPTION: Provide a detailed description of the image contents including
   type, subjects, data/measurements, spatial layout, colors, labels, context.

Format as:
--- Text ---
[extracted text]

--- Description ---
[detailed description]
```

## Document Organizer Integration

After the OCR2 service is running, the Document Organizer needs a small change:

### New OCR provider: `providers/ocr/deepseek_ocr2_local.py`
- Implements `OCRProvider` (extract + describe)
- Sends image bytes via HTTP POST to the OCR2 service
- Configurable base_url and timeout

### Config change in `config.yaml`
```yaml
ocr:
  enabled: true
  provider: "deepseek_ocr2"          # NEW — local DeepSeek-OCR2
  base_url: "http://localhost:8790"   # OCR2 service port
  timeout: 120                        # seconds per request
  # provider: "gemini"               # cloud fallback
```

### Add provider to `providers/ocr/__init__.py`
```python
elif provider == "deepseek_ocr2":
    from providers.ocr.deepseek_ocr2_local import DeepSeekOCR2Local
    return DeepSeekOCR2Local(
        base_url=ocr_cfg.get("base_url", "http://localhost:8790"),
        timeout=ocr_cfg.get("timeout", 120),
    )
```

## Startup & Lifecycle

```bash
# Start the service (stays running in background)
cd deepseek-ocr2-service
python server.py                    # or: uvicorn server:app --host 0.0.0.0 --port 8790

# First run downloads model weights (~2-3 GB) from mlx-community
# Subsequent runs load from cache (~5-10s startup)
```

The service should:
- Load model on startup (not per-request)
- Log model load time and memory usage
- Handle concurrent requests (queued — one inference at a time)
- Graceful shutdown on SIGTERM/SIGINT

## Performance Expectations

| Metric | Estimate |
|--------|----------|
| Model load time | 5-10s (from cache) |
| Memory usage | ~2-3 GB |
| PDF page OCR | ~2-5s per page |
| Image describe | ~3-8s per image |
| Throughput | Sequential (1 at a time) |

## Testing Plan

1. **Unit tests**: Sample PDF pages and images with known text → verify extraction accuracy
2. **Integration test**: Run Document Organizer indexer with OCR2 provider against test vault
3. **Comparison test**: Same documents through Gemini vs OCR2 → compare quality

## Dependencies

```
mlx-vlm>=0.3.0        # MLX vision-language model inference
fastapi>=0.110.0       # HTTP API framework
uvicorn>=0.27.0        # ASGI server
python-multipart       # File upload handling
pillow                 # Image processing
```

## Out of Scope

- GPU/CUDA support (Apple Silicon only for now)
- Batch/async processing (sequential is fine for indexing)
- Model fine-tuning
- OCR for non-Latin scripts (evaluate later)
- Auto-start/stop lifecycle management (manual start for now; can add later like llama-server)

## Future Considerations

- Add idle timeout and auto-shutdown (like llama-server reranker)
- Document Organizer could auto-start the OCR2 service when needed
- If DeepSeek-OCR2 gets GGUF/Ollama support, migrate to Ollama for unified orchestration
- Evaluate running on VPS if needed (would use transformers instead of mlx-vlm)
