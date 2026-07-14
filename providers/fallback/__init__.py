from providers.fallback.litellm_fallback import LiteLLMFallback  # noqa: F401


def build_litellm_fallback(cfg, prompt, encoder):
    """Build a fallback .run callable from a `.fallback` config block, or None.

    Raises ValueError on an unknown provider or a missing model (no implicit default).
    """
    if not cfg:
        return None
    if cfg.get("provider") != "litellm":
        raise ValueError(f"Unknown fallback provider: {cfg.get('provider')}")
    if not cfg.get("model"):
        raise ValueError("fallback.model is required (no implicit default)")
    from providers.fallback.litellm_fallback import LiteLLMFallback
    client = LiteLLMFallback(cfg["endpoint"], cfg["model"], prompt, encoder,
                             api_key=cfg.get("api_key"))
    return client.run
