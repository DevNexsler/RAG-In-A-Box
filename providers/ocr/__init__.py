"""OCR providers: extract text from images or PDF pages.

Supports separate providers for extract (PDF pages) vs describe (images):

    ocr:
      enabled: true
      provider: "deepseek_ocr2"          # default for both extract and describe
      base_url: "http://localhost:8790"
      describe:                           # override just the describe provider
        provider: "ollama_vision"
        model: "qwen3-vl:8b"
"""

from providers.ocr.base import OCRProvider

__all__ = ["OCRProvider"]


def _build_single_provider(cfg: dict) -> OCRProvider | None:
    """Build one OCRProvider from a flat config dict."""
    provider = cfg.get("provider", "none")

    if provider == "none":
        return None

    elif provider == "gemini":
        from providers.ocr.gemini_ocr import GeminiOCR
        return GeminiOCR(model=cfg.get("model", "gemini-2.0-flash"))

    elif provider == "deepseek_ocr2":
        from providers.ocr.deepseek_ocr2_local import DeepSeekOCR2Local
        return DeepSeekOCR2Local(
            base_url=cfg.get("base_url", "http://localhost:8790"),
            timeout=cfg.get("timeout", 120.0),
        )

    elif provider == "ollama_vision":
        from providers.ocr.ollama_vision import OllamaVisionOCR
        return OllamaVisionOCR(
            base_url=cfg.get("base_url", "http://localhost:11434"),
            model=cfg.get("model", "qwen3-vl:8b"),
            timeout=cfg.get("timeout", 120.0),
        )

    else:
        raise ValueError(f"Unknown OCR provider: {provider}")


_DESCRIBE_PROMPT = "Describe this image in detail for document search."
_EXTRACT_PROMPT = "Transcribe all text in this image verbatim."


def _build_fallback_run(cfg: dict | None, prompt: str):
    if not cfg:
        return None
    if cfg.get("provider") != "litellm":
        raise ValueError(f"Unknown fallback provider: {cfg.get('provider')}")
    if not cfg.get("model"):
        raise ValueError("fallback.model is required (no implicit default)")
    from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
    client = LiteLLMFallback(cfg["endpoint"], cfg["model"], prompt, image_encoder,
                             api_key=cfg.get("api_key"))
    return client.run


def _compose_primary(extract_prov: OCRProvider | None,
                     describe_prov: OCRProvider | None) -> OCRProvider | None:
    """Compose the bare primary OCRProvider from extract/describe providers."""
    if extract_prov is None and describe_prov is None:
        return None
    if extract_prov is describe_prov:
        return extract_prov

    from providers.ocr.composite import CompositeOCRProvider
    return CompositeOCRProvider(
        extract_provider=extract_prov or describe_prov,
        describe_provider=describe_prov or extract_prov,
    )


def build_ocr_provider(config: dict) -> OCRProvider | None:
    """Build an OCR provider from the ocr section of config.yaml.

    Supports split config: separate providers for extract (PDF pages)
    and describe (images). Falls back to a single provider when no
    extract/describe subsections are present. When OCR is enabled, the
    composed primary is always wrapped in a FallbackOCRProvider so that a
    reachable-but-empty describe/extract can recover via an optional
    per-method LiteLLM fallback (built from the `.fallback` subsection).
    """
    ocr_cfg = config.get("ocr", {})
    if not ocr_cfg.get("enabled", False):
        return None

    default = _build_single_provider(ocr_cfg)

    extract_cfg = ocr_cfg.get("extract")
    describe_cfg = ocr_cfg.get("describe")

    if not extract_cfg and not describe_cfg:
        extract_prov = describe_prov = default
    else:
        extract_prov = _build_single_provider(extract_cfg) if extract_cfg else default
        describe_prov = _build_single_provider(describe_cfg) if describe_cfg else default

    describe_fb = _build_fallback_run(
        (describe_cfg or {}).get("fallback") or ocr_cfg.get("describe", {}).get("fallback"),
        _DESCRIBE_PROMPT,
    )
    extract_fb = _build_fallback_run(
        (extract_cfg or {}).get("fallback"),
        _EXTRACT_PROMPT,
    )

    primary = _compose_primary(extract_prov, describe_prov)
    if primary is None:
        return None

    from providers.ocr.fallback import FallbackOCRProvider
    return FallbackOCRProvider(primary, describe_fallback=describe_fb,
                               extract_fallback=extract_fb)
