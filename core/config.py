"""Load and validate config from config.yaml. Fail fast on missing/invalid keys."""

from pathlib import Path
from typing import Any

import yaml

# Load .env file if present (for GEMINI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set manually


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config. Raises if file missing or required keys absent."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Config file is empty")

    required = ["vault_root", "index_root"]
    for key in required:
        if key not in raw:
            raise ValueError(f"Config missing required key: {key}")

    vault = Path(raw["vault_root"])
    if not vault.exists():
        raise ValueError(f"vault_root does not exist: {vault}")

    # --- Validate chunking parameters ---
    chunk_cfg = raw.get("chunking", {})
    max_chars = chunk_cfg.get("max_chars")
    if max_chars is not None:
        if not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError(f"chunking.max_chars must be a positive integer, got {max_chars!r}")
    overlap = chunk_cfg.get("overlap")
    if overlap is not None:
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError(f"chunking.overlap must be a non-negative integer, got {overlap!r}")
        effective_max = max_chars if max_chars is not None else 1800
        if overlap >= effective_max:
            raise ValueError(
                f"chunking.overlap ({overlap}) must be less than max_chars ({effective_max})"
            )

    # --- Validate search parameters ---
    search_cfg = raw.get("search", {})
    for key in ("vector_top_k", "keyword_top_k", "final_top_k"):
        val = search_cfg.get(key)
        if val is not None:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"search.{key} must be a positive integer, got {val!r}")

    return raw
