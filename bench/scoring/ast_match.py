"""Hierarchical 4-level AST scorer for tool calls (see PLAN.md §3, §4.1).

The four levels are nested: each requires the previous to hold. Irrelevance
prompts (``gold is None``) are scored separately as a binary correct/incorrect.
"""

from __future__ import annotations

from typing import Any

from bench.types import CorrectnessResult, ParsedToolCall


_EPS = 1e-6


def score(
    parsed: ParsedToolCall | None,
    gold: ParsedToolCall | None,
    parse_failed: bool,
    metadata: dict | None = None,
) -> CorrectnessResult:
    """Score a single parsed call against the gold call.

    BFCL convention: if ``metadata["possible_answer"]`` is present and ``gold``
    is not None, level-4 value matching admits any of the allowed values per
    argument. ``gold`` still drives the level-3 key-set check.
    """
    if parse_failed:
        return CorrectnessResult(
            level_1_called=False,
            level_2_name=False,
            level_3_args=False,
            level_4_values=False,
            parse_failed=True,
            notes="parse_failed",
        )

    if gold is None:
        called = parsed is not None
        notes = "irrelevance: false-positive call" if called else "irrelevance: correct"
        return CorrectnessResult(
            level_1_called=called,
            level_2_name=False,
            level_3_args=False,
            level_4_values=False,
            parse_failed=False,
            notes=notes,
        )

    if parsed is None:
        return CorrectnessResult(
            level_1_called=False,
            level_2_name=False,
            level_3_args=False,
            level_4_values=False,
            parse_failed=False,
            notes="no_call",
        )

    level_1 = True

    level_2 = parsed.function_name == gold.function_name
    if not level_2:
        return CorrectnessResult(
            level_1_called=True,
            level_2_name=False,
            level_3_args=False,
            level_4_values=False,
            parse_failed=False,
            notes="wrong_name",
        )

    level_3 = set(parsed.arguments) == set(gold.arguments)
    if not level_3:
        return CorrectnessResult(
            level_1_called=True,
            level_2_name=True,
            level_3_args=False,
            level_4_values=False,
            parse_failed=False,
            notes="wrong_args",
        )

    allowed = _allowed_values(metadata, gold.function_name)
    level_4 = _values_match(parsed.arguments, gold.arguments, allowed)
    notes = "ok" if level_4 else "wrong_values"
    return CorrectnessResult(
        level_1_called=True,
        level_2_name=True,
        level_3_args=True,
        level_4_values=level_4,
        parse_failed=False,
        notes=notes,
    )


def _allowed_values(metadata: dict | None, function_name: str) -> dict[str, list[Any]] | None:
    """Return the BFCL per-arg allowed-value lists, or None when absent."""
    if not metadata:
        return None
    possible_answer = metadata.get("possible_answer")
    if not possible_answer:
        return None
    first = possible_answer[0]
    if not isinstance(first, dict):
        return None
    arg_choices = first.get(function_name)
    if not isinstance(arg_choices, dict):
        return None
    return {k: v for k, v in arg_choices.items() if isinstance(v, list)}


def _values_match(
    parsed_args: dict[str, Any],
    gold_args: dict[str, Any],
    allowed: dict[str, list[Any]] | None,
) -> bool:
    for key, gold_value in gold_args.items():
        parsed_value = parsed_args[key]
        if allowed is not None and key in allowed:
            if not any(_values_equal(parsed_value, candidate) for candidate in allowed[key]):
                return False
        else:
            if not _values_equal(parsed_value, gold_value):
                return False
    return True


def _values_equal(p: Any, g: Any, eps: float = _EPS) -> bool:
    if isinstance(p, float) or isinstance(g, float):
        try:
            return abs(float(p) - float(g)) <= eps
        except (TypeError, ValueError):
            return False
    if isinstance(p, list) and isinstance(g, list):
        if len(p) != len(g):
            return False
        return all(_values_equal(pi, gi, eps) for pi, gi in zip(p, g))
    if isinstance(p, dict) and isinstance(g, dict):
        if set(p) != set(g):
            return False
        return all(_values_equal(p[k], g[k], eps) for k in g)
    return p == g
