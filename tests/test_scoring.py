"""Cover all four scoring levels plus irrelevance and BFCL possible_answer."""

from __future__ import annotations

from bench.scoring.aggregate import summarize
from bench.scoring.ast_match import score
from bench.types import CorrectnessResult, ParsedToolCall


def _gold() -> ParsedToolCall:
    return ParsedToolCall(function_name="foo", arguments={"x": 1, "y": "hello"})


def test_no_call_level_1_false() -> None:
    result = score(parsed=None, gold=_gold(), parse_failed=False)
    assert result.level_1_called is False
    assert result.level_2_name is False
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.parse_failed is False
    assert result.notes == "no_call"


def test_wrong_name_stops_at_level_2() -> None:
    parsed = ParsedToolCall(function_name="bar", arguments={"x": 1, "y": "hello"})
    result = score(parsed=parsed, gold=_gold(), parse_failed=False)
    assert result.level_1_called is True
    assert result.level_2_name is False
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.notes == "wrong_name"


def test_wrong_arg_keys_stops_at_level_3() -> None:
    parsed = ParsedToolCall(function_name="foo", arguments={"x": 1, "z": "hello"})
    result = score(parsed=parsed, gold=_gold(), parse_failed=False)
    assert result.level_1_called is True
    assert result.level_2_name is True
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.notes == "wrong_args"


def test_wrong_value_stops_at_level_4() -> None:
    parsed = ParsedToolCall(function_name="foo", arguments={"x": 2, "y": "hello"})
    result = score(parsed=parsed, gold=_gold(), parse_failed=False)
    assert result.level_1_called is True
    assert result.level_2_name is True
    assert result.level_3_args is True
    assert result.level_4_values is False
    assert result.notes == "wrong_values"


def test_exact_match_all_levels_true() -> None:
    parsed = ParsedToolCall(function_name="foo", arguments={"x": 1, "y": "hello"})
    result = score(parsed=parsed, gold=_gold(), parse_failed=False)
    assert result.level_1_called is True
    assert result.level_2_name is True
    assert result.level_3_args is True
    assert result.level_4_values is True
    assert result.notes == "ok"


def test_float_tolerance_within_eps() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"temp": 1.0})
    parsed = ParsedToolCall(function_name="foo", arguments={"temp": 1.0000001})
    result = score(parsed=parsed, gold=gold, parse_failed=False)
    assert result.level_4_values is True
    assert result.notes == "ok"


def test_float_tolerance_outside_eps_fails() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"temp": 1.0})
    parsed = ParsedToolCall(function_name="foo", arguments={"temp": 1.5})
    result = score(parsed=parsed, gold=gold, parse_failed=False)
    assert result.level_4_values is False


def test_irrelevance_correct_when_no_call() -> None:
    result = score(parsed=None, gold=None, parse_failed=False)
    assert result.level_1_called is False
    assert result.level_2_name is False
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.notes == "irrelevance: correct"


def test_irrelevance_false_positive_when_call_emitted() -> None:
    parsed = ParsedToolCall(function_name="foo", arguments={})
    result = score(parsed=parsed, gold=None, parse_failed=False)
    assert result.level_1_called is True
    assert result.level_2_name is False
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.notes == "irrelevance: false-positive call"


def test_parse_failed_short_circuits() -> None:
    result = score(parsed=None, gold=_gold(), parse_failed=True)
    assert result.level_1_called is False
    assert result.level_2_name is False
    assert result.level_3_args is False
    assert result.level_4_values is False
    assert result.parse_failed is True
    assert result.notes == "parse_failed"


def test_bfcl_possible_answer_allows_any_listed_value() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"x": 1})
    parsed = ParsedToolCall(function_name="foo", arguments={"x": 3})
    metadata = {"possible_answer": [{"foo": {"x": [1, 2, 3]}}]}
    result = score(parsed=parsed, gold=gold, parse_failed=False, metadata=metadata)
    assert result.level_4_values is True
    assert result.notes == "ok"


def test_bfcl_possible_answer_rejects_disallowed_value() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"x": 1})
    parsed = ParsedToolCall(function_name="foo", arguments={"x": 99})
    metadata = {"possible_answer": [{"foo": {"x": [1, 2, 3]}}]}
    result = score(parsed=parsed, gold=gold, parse_failed=False, metadata=metadata)
    assert result.level_4_values is False
    assert result.notes == "wrong_values"


def test_list_values_compared_elementwise() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"xs": [1, 2.0, "a"]})
    parsed = ParsedToolCall(function_name="foo", arguments={"xs": [1, 2.0000001, "a"]})
    result = score(parsed=parsed, gold=gold, parse_failed=False)
    assert result.level_4_values is True


def test_nested_dict_values_recurse() -> None:
    gold = ParsedToolCall(function_name="foo", arguments={"d": {"k": 1.0}})
    parsed = ParsedToolCall(function_name="foo", arguments={"d": {"k": 1.0000001}})
    result = score(parsed=parsed, gold=gold, parse_failed=False)
    assert result.level_4_values is True


def test_summarize_empty() -> None:
    summary = summarize([])
    assert summary["n"] == 0
    assert summary["strict_accuracy"] == 0.0
    assert summary["parse_failures"] == 0


def test_summarize_mix_of_levels() -> None:
    results = [
        CorrectnessResult(True, True, True, True, False, "ok"),
        CorrectnessResult(True, True, True, False, False, "wrong_values"),
        CorrectnessResult(True, True, False, False, False, "wrong_args"),
        CorrectnessResult(True, False, False, False, False, "wrong_name"),
        CorrectnessResult(False, False, False, False, True, "parse_failed"),
    ]
    summary = summarize(results, seed=42, n_bootstrap=200)
    assert summary["n"] == 5
    assert summary["level_1_rate"] == 4 / 5
    assert summary["level_2_rate"] == 3 / 5
    assert summary["level_3_rate"] == 2 / 5
    assert summary["level_4_rate"] == 1 / 5
    assert summary["strict_accuracy"] == 1 / 5
    assert summary["parse_failures"] == 1
    assert 0.0 <= summary["ci_low"] <= summary["strict_accuracy"] <= summary["ci_high"] <= 1.0
