"""scripts/run_one.py — Phase 1 single-sweep CLI (see PLAN.md §4.1)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml

from bench.harness.runner import run
from bench.scoring.aggregate import summarize, summarize_timing
from bench.types import CorrectnessResult, ResultRow


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
        _print_summary(config, summarize([]), {"n": 0})
        return 0

    results_path = run(config, max_n=args.max_n)
    rows = list(_load_rows(results_path))
    summary = summarize(row.correctness for row in rows)
    timing_summary = summarize_timing(row.timing for row in rows)
    _print_summary(config, summary, timing_summary)
    return 0


def _load_rows(path: Path) -> Iterator[ResultRow]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield ResultRow.model_validate_json(line)


def _load_correctness(path: Path) -> Iterator[CorrectnessResult]:
    """Kept for backward-compat with anything importing the helper externally."""
    for row in _load_rows(path):
        yield row.correctness


def _print_summary(config: dict, summary: dict, timing: dict) -> None:
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
    if n == 0 or timing.get("n", 0) == 0:
        return
    print("  Prefill (median tokens):")
    if timing.get("prefill_tokens_p50") is not None:
        print(f"    Total:   {int(timing['prefill_tokens_p50'])}")
    if timing.get("schema_tokens_p50") is not None:
        print(f"    Schema:  {int(timing['schema_tokens_p50'])}")
    if timing.get("query_tokens_p50") is not None:
        print(f"    Query:   {int(timing['query_tokens_p50'])}")
    if timing.get("decode_tokens_p50") is not None:
        print(f"  Decode  (median tokens):     {int(timing['decode_tokens_p50'])}")
    if timing.get("ttft_ms_p50") is not None:
        print(
            f"  Latency (median): ttft={timing['ttft_ms_p50']:.0f}ms  "
            f"total={timing['total_ms_p50']:.0f}ms  "
            f"ratio={timing['ttft_ms_p50']/timing['total_ms_p50']:.2f}"
        )


if __name__ == "__main__":
    sys.exit(main())
