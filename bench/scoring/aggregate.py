"""Pure aggregation over CorrectnessResult lists (see PLAN.md §3, §4.1).

Kept reporter-free so Unit 4's CLI can call ``summarize`` directly. Bootstrap CI
uses stdlib ``random`` to avoid adding numpy as a hard dep.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

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

    l1 = [int(r.level_1_called) for r in items]
    l2 = [int(r.level_2_name) for r in items]
    l3 = [int(r.level_3_args) for r in items]
    l4 = [int(r.level_4_values) for r in items]
    parse_failures = sum(1 for r in items if r.parse_failed)

    strict_accuracy = sum(l4) / n
    ci_low, ci_high = _bootstrap_ci(l4, seed=seed, n_bootstrap=n_bootstrap)

    return {
        "n": n,
        "strict_accuracy": strict_accuracy,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "level_1_rate": sum(l1) / n,
        "level_2_rate": sum(l2) / n,
        "level_3_rate": sum(l3) / n,
        "level_4_rate": sum(l4) / n,
        "parse_failures": parse_failures,
    }


def _bootstrap_ci(
    values: list[int],
    seed: int,
    n_bootstrap: int,
    lower: float = 0.025,
    upper: float = 0.975,
) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    return _percentile(means, lower), _percentile(means, upper)


def _percentile(sorted_values: list[float], q: float) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = int(q * (n - 1))
    return sorted_values[idx]
