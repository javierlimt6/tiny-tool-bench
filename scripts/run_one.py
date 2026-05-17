"""scripts/run_one.py — Phase 1 single-sweep CLI (see PLAN.md §4.1)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml

from bench.harness.runner import run
from bench.scoring.aggregate import summarize
from bench.types import CorrectnessResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tiny-fc-bench",
        description="Run one (config, dataset) sweep.",
    )
    parser.add_argument("config_path", type=Path, help="Path to YAML config")
    parser.add_argument(
        "--max-n", type=int, default=None, help="Limit prompts (smoke test)"
    )
    args = parser.parse_args(argv)

    with args.config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.max_n == 0:
        _print_summary(config, summarize([]))
        return 0

    results_path = run(config, max_n=args.max_n)
    results = list(_load_correctness(results_path))
    summary = summarize(results)
    _print_summary(config, summary)
    return 0


def _load_correctness(path: Path) -> Iterator[CorrectnessResult]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield CorrectnessResult(**json.loads(line)["correctness"])


def _print_summary(config: dict, summary: dict) -> None:
    name = config.get("display_name", "(unnamed)")
    n = summary["n"]
    print(f"{name} (n={n}):")
    print(
        f"  Strict accuracy: {summary['strict_accuracy']:.2f} "
        f"[{summary['ci_low']:.2f}, {summary['ci_high']:.2f}]"
    )
    print(f"  Level 1 (called): {summary['level_1_rate']:.2f}")
    print(f"  Level 2 (name):   {summary['level_2_rate']:.2f}")
    print(f"  Level 3 (args):   {summary['level_3_rate']:.2f}")
    print(f"  Level 4 (vals):   {summary['level_4_rate']:.2f}")
    print(f"  Parse failures:   {summary['parse_failures']}")


if __name__ == "__main__":
    sys.exit(main())
