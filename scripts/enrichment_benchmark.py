from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmarking.cases import build_labeling_status, load_case, prepare_cases, write_gold_stub
from core.benchmarking.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM enrichment benchmark utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-cases")
    prepare.add_argument("--trace-dir", default=".evals/llm-traces")
    prepare.add_argument("--bench-dir", default=".evals/benchmarks")
    prepare.add_argument("--limit", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)

    show_case = subparsers.add_parser("show-case")
    show_case.add_argument("--bench-dir", default=".evals/benchmarks")
    show_case.add_argument("--case-id", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--bench-dir", default=".evals/benchmarks")

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--bench-dir", default=".evals/benchmarks")
    run_cmd.add_argument("--model", required=True)
    run_cmd.add_argument("--run-id", required=True)
    run_cmd.add_argument("--max-cases", type=int)

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

    if args.command == "show-case":
        case = load_case(bench_dir=args.bench_dir, case_id=args.case_id)
        gold_path = write_gold_stub(case, bench_dir=args.bench_dir)
        print(f"Case ID: {case.case_id}")
        print(f"Title: {case.title}")
        print(f"Source Type: {case.source_type}")
        print(f"Category: {case.category}")
        print(f"Difficulty: {case.difficulty}")
        print(f"Trace: {case.trace_file}:{case.trace_line}")
        print()
        print("Prompt:")
        print(case.prompt)
        print()
        print("Baseline Output (gpt-4.1-mini):")
        print(case.baseline_response)
        print()
        print(f"Gold File: {gold_path}")
        return 0

    if args.command == "status":
        result = build_labeling_status(bench_dir=args.bench_dir)
        print(f"Total Cases: {result['total_cases']}")
        print(f"Labeled: {result['labeled']}")
        print(f"Remaining: {result['remaining']}")
        print(f"Next Unlabeled Case: {result['next_case_id'] or 'none'}")
        return 0

    if args.command == "run":
        result = run_benchmark(
            bench_dir=args.bench_dir,
            model=args.model,
            run_id=args.run_id,
            max_cases=args.max_cases,
        )
        print(f"Run ID: {args.run_id}")
        print(f"Model: {args.model}")
        print(f"Cases: {result.summary['case_count']}")
        print(f"Average Total Score: {result.summary['average_total_score']}")
        print(f"Parse Failures: {result.summary['parse_failures']}")
        print(f"Artifacts: {result.run_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
