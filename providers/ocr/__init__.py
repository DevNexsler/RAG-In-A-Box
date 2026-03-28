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


def build_ocr_provider(config: dict) -> OCRProvider | None:
    """Build an OCR provider from the ocr section of config.yaml.

    Supports split config: separate providers for extract (PDF pages)
    and describe (images). Falls back to a single provider when no
    extract/describe subsections are present.
    """
    ocr_cfg = config.get("ocr", {})
    if not ocr_cfg.get("enabled", False):
        return None

    default = _build_single_provider(ocr_cfg)

    extract_cfg = ocr_cfg.get("extract")
    describe_cfg = ocr_cfg.get("describe")

    # No overrides — use the single default provider
    if not extract_cfg and not describe_cfg:
        return default

    extract_prov = _build_single_provider(extract_cfg) if extract_cfg else default
    describe_prov = _build_single_provider(describe_cfg) if describe_cfg else default

    if extract_prov is None and describe_prov is None:
        return None
    if extract_prov is describe_prov:
        return extract_prov

    from providers.ocr.composite import CompositeOCRProvider
    return CompositeOCRProvider(
        extract_provider=extract_prov or describe_prov,
        describe_provider=describe_prov or extract_prov,
    )
