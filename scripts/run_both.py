"""scripts/run_both.py — Phase 2 side-by-side comparison CLI (see PLAN.md §4.2).

Runs two configs serially through ``bench.harness.runner.run`` (the runner is
sequential; no parallelism here), then loads each result JSONL, summarises it,
and prints a side-by-side comparison table. The first cross-model BFCL v3
``simple`` comparison this benchmark produces lives at the end of this pipe.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import yaml

from bench.harness.runner import run
from bench.scoring.aggregate import summarize, summarize_timing
from bench.types import ResultRow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tiny-fc-bench-compare",
        description="Run two (config, dataset) sweeps and print a side-by-side table.",
    )
    parser.add_argument("config_a", type=Path, help="Path to YAML config A")
    parser.add_argument("config_b", type=Path, help="Path to YAML config B")
    parser.add_argument(
        "--max-n", type=int, default=None, help="Limit prompts (smoke test)"
    )
    args = parser.parse_args(argv)

    config_a = _load_config(args.config_a)
    config_b = _load_config(args.config_b)

    if args.max_n == 0:
        empty_summary = summarize([])
        empty_timing = {"n": 0}
        _print_comparison(
            config_a,
            empty_summary,
            empty_timing,
            config_b,
            empty_summary,
            empty_timing,
        )
        return 0

    # Sequential: runner is single-stream; no parallelism, see PLAN.md §3 rule 6.
    path_a = run(config_a, max_n=args.max_n)
    path_b = run(config_b, max_n=args.max_n)

    summary_a, timing_a = _summarise_file(path_a)
    summary_b, timing_b = _summarise_file(path_b)

    _print_comparison(config_a, summary_a, timing_a, config_b, summary_b, timing_b)
    return 0


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_rows(path: Path) -> Iterator[ResultRow]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield ResultRow.model_validate_json(line)


def _summarise_file(path: Path) -> tuple[dict, dict]:
    rows = list(_load_rows(path))
    return (
        summarize(row.correctness for row in rows),
        summarize_timing(row.timing for row in rows),
    )


def _print_comparison(
    config_a: dict,
    summary_a: dict,
    timing_a: dict,
    config_b: dict,
    summary_b: dict,
    timing_b: dict,
) -> None:
    name_a = config_a.get("display_name", "(unnamed)")
    name_b = config_b.get("display_name", "(unnamed)")
    n_a = summary_a["n"]
    n_b = summary_b["n"]
    # Headline n is A's count if equal across both (the expected case under
    # identical --max-n); else show both for transparency.
    n_header = str(n_a) if n_a == n_b else f"{n_a} vs {n_b}"
    print(f"Comparison on BFCL v3 simple (n={n_header}):")
    print()

    rows: list[tuple[str, str, str]] = [
        ("Strict accuracy", _fmt_accuracy(summary_a), _fmt_accuracy(summary_b)),
        ("Level 1 (called)", _fmt_rate(summary_a, "level_1_rate"), _fmt_rate(summary_b, "level_1_rate")),
        ("Level 2 (name)", _fmt_rate(summary_a, "level_2_rate"), _fmt_rate(summary_b, "level_2_rate")),
        ("Level 3 (args)", _fmt_rate(summary_a, "level_3_rate"), _fmt_rate(summary_b, "level_3_rate")),
        ("Level 4 (vals)", _fmt_rate(summary_a, "level_4_rate"), _fmt_rate(summary_b, "level_4_rate")),
        ("Parse failures", _fmt_int(summary_a, "parse_failures"), _fmt_int(summary_b, "parse_failures")),
        ("Prefill schema/query (p50)", _fmt_schema_query(timing_a), _fmt_schema_query(timing_b)),
        ("Decode tokens p50/p90/p99", _fmt_tail(timing_a, "decode_tokens", _fmt_int_val), _fmt_tail(timing_b, "decode_tokens", _fmt_int_val)),
        ("ttft_ms p50/p90/p99", _fmt_tail(timing_a, "ttft_ms", _fmt_float_val), _fmt_tail(timing_b, "ttft_ms", _fmt_float_val)),
        ("total_ms p50/p90/p99", _fmt_tail(timing_a, "total_ms", _fmt_float_val), _fmt_tail(timing_b, "total_ms", _fmt_float_val)),
        ("ttft/total ratio (p50)", _fmt_ratio(timing_a), _fmt_ratio(timing_b)),
    ]

    label_width = max(len(r[0]) for r in rows) + 2
    col_a_width = max(len(name_a), max(len(r[1]) for r in rows)) + 2
    print(f"  {'':<{label_width}}{name_a:<{col_a_width}}{name_b}")
    for label, val_a, val_b in rows:
        print(f"  {label:<{label_width}}{val_a:<{col_a_width}}{val_b}")


def _fmt_accuracy(summary: dict) -> str:
    if summary["n"] == 0:
        return "-"
    return (
        f"{summary['strict_accuracy']:.2f} "
        f"[{summary['ci_low']:.2f}, {summary['ci_high']:.2f}]"
    )


def _fmt_rate(summary: dict, key: str) -> str:
    if summary["n"] == 0:
        return "-"
    return f"{summary[key]:.2f}"


def _fmt_int(summary: dict, key: str) -> str:
    if summary["n"] == 0:
        return "-"
    return str(summary[key])


def _fmt_schema_query(timing: dict) -> str:
    if timing.get("n", 0) == 0:
        return "-"
    schema = timing.get("schema_tokens_p50")
    query = timing.get("query_tokens_p50")
    schema_str = f"{int(schema)}" if schema is not None else "-"
    query_str = f"{int(query)}" if query is not None else "-"
    if schema_str == "-" and query_str == "-":
        return "-"
    return f"{schema_str} / {query_str}"


def _fmt_int_val(val: float) -> str:
    return f"{int(val)}"


def _fmt_float_val(val: float) -> str:
    return f"{val:.0f}"


def _fmt_tail(timing: dict, base: str, fmt: Callable[[float], str]) -> str:
    """Format ``base_p50/p90/p99`` as 'X / Y / Z'; '-' when no rows or all None."""
    if timing.get("n", 0) == 0:
        return "-"
    parts = [
        fmt(timing[f"{base}_{q}"]) if timing.get(f"{base}_{q}") is not None else "-"
        for q in ("p50", "p90", "p99")
    ]
    if all(p == "-" for p in parts):
        return "-"
    return " / ".join(parts)


def _fmt_ratio(timing: dict) -> str:
    if timing.get("n", 0) == 0:
        return "-"
    ttft = timing.get("ttft_ms_p50")
    total = timing.get("total_ms_p50")
    if ttft is None or total is None or total == 0:
        return "-"
    return f"{ttft / total:.2f}"


if __name__ == "__main__":
    sys.exit(main())
