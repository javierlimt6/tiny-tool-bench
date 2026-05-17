"""Pure aggregation over CorrectnessResult lists (see PLAN.md §3, §4.1).

Reporter-free so Unit 4's CLI can call ``summarize`` directly. Bootstrap CI is
now vectorised with numpy (already a transitive dep of pandas/torch). A second
helper ``summarize_timing`` reports median prefill/query/schema/decode tokens
so the schema-heavy-prefill story is visible per-run.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from bench.types import CorrectnessResult


def summarize(
    results: Iterable[CorrectnessResult],
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict:
    """Return strict accuracy, per-level rates, and a 95% bootstrap CI on L4."""
    items = list(results)
    n = len(items)
    if n == 0:
        return {
            "n": 0,
            "strict_accuracy": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "level_1_rate": 0.0,
            "level_2_rate": 0.0,
            "level_3_rate": 0.0,
            "level_4_rate": 0.0,
            "parse_failures": 0,
        }

    l1 = np.fromiter((int(r.level_1_called) for r in items), dtype=float, count=n)
    l2 = np.fromiter((int(r.level_2_name) for r in items), dtype=float, count=n)
    l3 = np.fromiter((int(r.level_3_args) for r in items), dtype=float, count=n)
    l4 = np.fromiter((int(r.level_4_values) for r in items), dtype=float, count=n)
    parse_failures = sum(1 for r in items if r.parse_failed)

    ci_low, ci_high = _bootstrap_ci(l4, seed=seed, n_bootstrap=n_bootstrap)

    return {
        "n": n,
        "strict_accuracy": float(l4.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "level_1_rate": float(l1.mean()),
        "level_2_rate": float(l2.mean()),
        "level_3_rate": float(l3.mean()),
        "level_4_rate": float(l4.mean()),
        "parse_failures": parse_failures,
    }


def summarize_timing(timings: Iterable[dict]) -> dict:
    """Report tokens (medians) and latency (p50/p90/p99) per run.

    Each input dict is one row's ``timing`` block. Token fields may be absent
    on older adapters; percentile helpers skip None. Latency tails (p90/p99)
    are first-class so the schema-scaling-vs-decode-variance story is
    decidable from the CLI summary without re-loading the JSONL.
    """
    rows = list(timings)
    if not rows:
        return {"n": 0}

    def pct(key: str, q: float) -> float | None:
        xs = [r[key] for r in rows if r.get(key) is not None]
        return float(np.percentile(xs, q)) if xs else None

    return {
        "n": len(rows),
        "prefill_tokens_p50": pct("prefill_tokens", 50),
        "query_tokens_p50": pct("query_tokens", 50),
        "schema_tokens_p50": pct("schema_tokens", 50),
        "decode_tokens_p50": pct("decode_tokens", 50),
        "decode_tokens_p90": pct("decode_tokens", 90),
        "decode_tokens_p99": pct("decode_tokens", 99),
        "ttft_ms_p50": pct("ttft_ms", 50),
        "ttft_ms_p90": pct("ttft_ms", 90),
        "ttft_ms_p99": pct("ttft_ms", 99),
        "total_ms_p50": pct("total_ms", 50),
        "total_ms_p90": pct("total_ms", 90),
        "total_ms_p99": pct("total_ms", 99),
    }


def _bootstrap_ci(
    values: np.ndarray,
    seed: int,
    n_bootstrap: int,
    lower: float = 2.5,
    upper: float = 97.5,
) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [lower, upper])
    return float(lo), float(hi)
