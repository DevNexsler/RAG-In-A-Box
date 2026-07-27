#!/usr/bin/env python3
"""CLI entrypoint: run the indexer flow. Usage: python run_index.py [config.yaml]"""

import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.config import load_config
from core.logging_setup import configure_logging_from_config
from prefect_server import PrefectServer


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    # Configure root logger from config (Prefect flow/task logs are independent).
    # One record == one physical line, warnings captured (#0546).
    configure_logging_from_config(config)

    with PrefectServer():
        from flow_index_vault import index_vault_flow
        index_vault_flow(config_path)


if __name__ == "__main__":
    main()
