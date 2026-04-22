from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmarking.cases import prepare_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM enrichment benchmark utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-cases")
    prepare.add_argument("--trace-dir", default=".evals/llm-traces")
    prepare.add_argument("--bench-dir", default=".evals/benchmarks")
    prepare.add_argument("--limit", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-cases":
        result = prepare_cases(
            trace_dir=args.trace_dir,
            out_dir=args.bench_dir,
            limit=args.limit,
            seed=args.seed,
        )
        print(f"Prepared {result.selected_count} cases")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
