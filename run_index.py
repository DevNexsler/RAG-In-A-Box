#!/usr/bin/env python3
"""CLI entrypoint: run the indexer flow. Usage: python run_index.py [config.yaml]"""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.config import load_config
from prefect_server import PrefectServer


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    # Configure root logger from config (Prefect flow/task logs are independent)
    log_level = config.get("logging", {}).get("level", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

    with PrefectServer():
        from flow_index_vault import index_vault_flow
        index_vault_flow(config_path)


if __name__ == "__main__":
    main()
