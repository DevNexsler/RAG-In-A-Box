"""LLM providers for document enrichment.

Supports multiple backends:
  - "ollama": Qwen3-14B via Ollama /api/chat with structured JSON output
  - "baseten": Qwen3-14B-AWQ via Baseten-hosted vLLM (cloud)
  - "openrouter": Any model via OpenRouter (cloud, OpenAI-compatible)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMGenerator(Protocol):
    """Protocol for LLM generators used by doc_enrichment."""

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str: ...


def build_llm_provider(config: dict) -> LLMGenerator | None:
    """Build an LLM generator from config.  Returns None if disabled or unavailable.

    Provider selection (enrichment.provider):
      - "ollama" (default): OllamaGenerator via Ollama API
      - "baseten": BasetenGenerator via Baseten-hosted vLLM
      - "openrouter": OpenRouterGenerator via OpenRouter API
    """
    enrichment_cfg = config.get("enrichment", {})
    if not enrichment_cfg.get("enabled", False):
        return None

    provider = enrichment_cfg.get("provider", "ollama")

    if provider == "openrouter":
        try:
            from providers.llm.openrouter_llm import OpenRouterGenerator

            return OpenRouterGenerator(
                model=enrichment_cfg.get("model", "minimax/minimax-m2.5"),
                api_key=enrichment_cfg.get("api_key"),
                timeout=enrichment_cfg.get("timeout", 300.0),
            )
        except Exception as exc:
            logger.warning(
                "Failed to init OpenRouter LLM provider, enrichment disabled: %s", exc
            )
            return None

    if provider == "baseten":
        try:
            from providers.llm.baseten_llm import BasetenGenerator

            return BasetenGenerator(
                model_id=enrichment_cfg.get("model_id", ""),
                model_name=enrichment_cfg.get("model", "qwen3-14b"),
                api_key=enrichment_cfg.get("api_key"),
                timeout=enrichment_cfg.get("timeout", 300.0),
            )
        except Exception as exc:
            logger.warning(
                "Failed to init Baseten LLM provider, enrichment disabled: %s", exc
            )
            return None

    if provider == "ollama":
        try:
            from providers.llm.ollama_llm import OllamaGenerator

            return OllamaGenerator(
                model_name=enrichment_cfg.get("model", "qwen3:14b-udq6"),
                base_url=enrichment_cfg.get("base_url", "http://localhost:11434"),
            )
        except Exception as exc:
            logger.warning(
                "Failed to init Ollama LLM provider, enrichment disabled: %s", exc
            )
            return None

    logger.warning("Unknown enrichment provider %r", provider)
    return None
