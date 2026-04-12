"""Load and validate config from config.yaml. Fail fast on missing/invalid keys."""

import os
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
    """Load YAML config. Raises if file missing or required keys absent.

    Accepts either ``documents_root`` or ``vault_root`` for the source
    directory (legacy single-source mode), or a top-level ``sources:`` list
    (new multi-source mode).

    After loading, the returned dict always contains a ``sources`` key
    (a non-empty list of source dicts).  Legacy keys ``documents_root`` and
    ``vault_root`` are preserved in the dict so downstream code that hasn't
    been refactored yet continues to work.

    Env var notes
    -------------
    ``DOCUMENTS_ROOT``: only applied when the config uses legacy
    ``documents_root``/``vault_root`` style.  For new-style ``sources:``
    configs it is ignored (the variable is ambiguous when multiple filesystem
    sources are present).
    ``INDEX_ROOT``: always applied regardless of source style.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Config file is empty")

    if "index_root" not in raw:
        raise ValueError("Config missing required key: index_root")

    # --- Determine which shape we're dealing with ---
    has_legacy_key = bool(raw.get("documents_root") or raw.get("vault_root"))
    has_sources_key = "sources" in raw

    if has_legacy_key and has_sources_key:
        # Allow the case where the config was produced by a previous load_config call
        # that synthesized a single 'documents' source from documents_root (round-trip
        # via yaml.dump / yaml.safe_load).  Detect this by checking that sources is a
        # single-entry list with name='documents' and type='filesystem'.  If so, strip
        # the synthesized sources key so the legacy path re-synthesises it correctly
        # (applying DOCUMENTS_ROOT env override and fresh path validation).
        sources_list = raw.get("sources") or []
        _is_shim = (
            isinstance(sources_list, list)
            and len(sources_list) == 1
            and isinstance(sources_list[0], dict)
            and sources_list[0].get("name") == "documents"
            and sources_list[0].get("type") == "filesystem"
        )
        if _is_shim:
            del raw["sources"]
            has_sources_key = False
        else:
            raise ValueError(
                "Cannot use both 'documents_root' and 'sources' in the same config. "
                "Pick one: either keep 'documents_root' (legacy single-source) or "
                "switch to 'sources:' (multi-source)."
            )

    if not has_legacy_key and not has_sources_key:
        raise ValueError(
            "Config missing required key: documents_root (or vault_root)"
        )

    # --- Env var overrides (for VPS / container deployments) ---
    # INDEX_ROOT always applies.
    env_index_root = os.environ.get("INDEX_ROOT")
    if env_index_root:
        raw["index_root"] = env_index_root

    if has_legacy_key:
        # Legacy single-source mode: accept either documents_root or vault_root.
        docs_root = raw.get("documents_root") or raw.get("vault_root")

        # DOCUMENTS_ROOT env var only applies to legacy mode; for new-style
        # sources: configs it is ignored because the variable is ambiguous
        # when multiple filesystem sources exist.
        env_docs_root = os.environ.get("DOCUMENTS_ROOT")
        if env_docs_root:
            docs_root = env_docs_root

        # Normalise: both keys resolve to the same value
        raw["documents_root"] = docs_root
        raw["vault_root"] = docs_root

        # Validate that the legacy root exists on disk (fail fast)
        docs_path = Path(docs_root)
        if not docs_path.exists():
            raise ValueError(f"documents_root does not exist: {docs_path}")

        # --- Sources shim: synthesize a single filesystem source ---
        raw["sources"] = [{
            "type": "filesystem",
            "name": "documents",
            "root": docs_root,
            "scan": raw.get("scan", {}),
        }]

    else:
        # New-style sources: mode.
        # Note: a top-level 'scan:' key is not forwarded to sources in new-style mode.
        # Each source in the list should carry its own 'scan' sub-key if needed.
        if raw["sources"] is None:
            raise ValueError(
                "'sources' key is present but has no value — "
                "provide at least one source entry"
            )
        # Validate structural requirements.
        if not isinstance(raw["sources"], list) or not raw["sources"]:
            raise ValueError("'sources' must be a non-empty list")

        # DOCUMENTS_ROOT env var: in new-style mode, apply it to the first
        # filesystem source's root (same as legacy mode applies it to
        # documents_root). This lets host-side tests override the container
        # path without knowing which source index it is.
        env_docs_root = os.environ.get("DOCUMENTS_ROOT")
        fs_overridden = False

        seen_names: set[str] = set()
        for i, src in enumerate(raw["sources"]):
            if not isinstance(src, dict):
                raise ValueError(f"sources[{i}] must be a mapping")
            if "type" not in src:
                raise ValueError(f"sources[{i}] missing required key 'type'")
            if "name" not in src:
                raise ValueError(f"sources[{i}] missing required key 'name'")
            if src["name"] in seen_names:
                raise ValueError(f"duplicate source name: {src['name']!r}")
            seen_names.add(src["name"])

            if src["type"] == "filesystem":
                if "root" not in src:
                    raise ValueError(f"sources[{i}] (filesystem) missing required key 'root'")
                if env_docs_root and not fs_overridden:
                    src["root"] = env_docs_root
                    fs_overridden = True
                root_path = Path(src["root"])
                if not root_path.exists():
                    raise ValueError(
                        f"sources[{i}] (filesystem) root does not exist: {root_path}"
                    )

        # Populate documents_root / vault_root for downstream callers that
        # still reference config["documents_root"] (MCP server, REST API).
        # Uses the first filesystem source's root, or falls back to index_root
        # for pure-postgres configs that have no filesystem at all.
        first_fs = next((s for s in raw["sources"] if s["type"] == "filesystem"), None)
        docs_root = first_fs["root"] if first_fs else raw.get("index_root", "")
        raw.setdefault("documents_root", docs_root)
        raw.setdefault("vault_root", docs_root)

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
