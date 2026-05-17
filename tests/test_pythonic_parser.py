"""Pythonic parser cases (vendored from vLLM; PLAN.md §3 design rule 2)."""

from __future__ import annotations

from bench.parsers.pythonic import parse_pythonic
from bench.types import ParsedToolCall


def test_single_call_in_list() -> None:
    parsed, failed = parse_pythonic('[get_weather(city="Singapore")]')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="get_weather", arguments={"city": "Singapore"})


def test_bare_call_without_list() -> None:
    parsed, failed = parse_pythonic('get_weather(city="Tokyo")')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="get_weather", arguments={"city": "Tokyo"})


def test_numeric_and_bool_literals() -> None:
    parsed, failed = parse_pythonic('[adjust_volume(level=7, mute=False)]')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="adjust_volume", arguments={"level": 7, "mute": False})


def test_nested_list_value() -> None:
    parsed, failed = parse_pythonic('[set_route(waypoints=["A", "B", "C"])]')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="set_route", arguments={"waypoints": ["A", "B", "C"]})


def test_multiple_calls_returns_first() -> None:
    parsed, failed = parse_pythonic('[f(x=1), g(y=2)]')
    assert failed is False
    assert parsed is not None
    assert parsed.function_name == "f"


def test_empty_list_fails() -> None:
    parsed, failed = parse_pythonic('[]')
    assert parsed is None
    assert failed is True


def test_syntax_error_fails() -> None:
    parsed, failed = parse_pythonic('[get_weather(city=Singapore]')  # missing closing paren
    assert parsed is None
    assert failed is True


def test_positional_args_rejected() -> None:
    """Pythonic tool calls use kwargs only; positional means we mis-parsed."""
    parsed, failed = parse_pythonic('[get_weather("Singapore")]')
    assert parsed is None
    assert failed is True


def test_non_call_expression_fails() -> None:
    parsed, failed = parse_pythonic('"just a string"')
    assert parsed is None
    assert failed is True


def test_empty_string_fails() -> None:
    parsed, failed = parse_pythonic('')
    assert parsed is None
    assert failed is True
