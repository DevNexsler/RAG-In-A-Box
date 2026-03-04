"""OCR providers: extract text from images or PDF pages."""

from providers.ocr.base import OCRProvider

__all__ = ["OCRProvider"]


def build_ocr_provider(config: dict) -> OCRProvider | None:
    """Build an OCR provider from the ocr section of config.yaml.

    Returns None if OCR is disabled or provider is 'none'.
    """
    ocr_cfg = config.get("ocr", {})
    if not ocr_cfg.get("enabled", False):
        return None

    provider = ocr_cfg.get("provider", "none")

    if provider == "none":
        return None

    elif provider == "gemini":
        from providers.ocr.gemini_ocr import GeminiOCR
        model = ocr_cfg.get("model", "gemini-2.0-flash")
        return GeminiOCR(model=model)

    elif provider == "deepseek_ocr2":
        from providers.ocr.deepseek_ocr2_local import DeepSeekOCR2Local
        return DeepSeekOCR2Local(
            base_url=ocr_cfg.get("base_url", "http://localhost:8790"),
            timeout=ocr_cfg.get("timeout", 120.0),
        )

    else:
        raise ValueError(f"Unknown OCR provider: {provider}")
