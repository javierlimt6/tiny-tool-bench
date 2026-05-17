"""JSON-native parser cases (PLAN.md §4.2 Phase 2; mirrors tests/test_hermes_parser.py)."""

from __future__ import annotations

from bench.parsers.json_native import parse_json_native
from bench.types import ParsedToolCall


def test_well_formed_single_object() -> None:
    parsed, failed = parse_json_native('{"name": "foo", "arguments": {"x": 1}}')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="foo", arguments={"x": 1})


def test_list_wrapped_returns_first() -> None:
    parsed, failed = parse_json_native(
        '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
    )
    assert failed is False
    assert parsed is not None
    assert parsed.function_name == "a"


def test_arguments_as_json_string_is_decoded() -> None:
    """Some models emit arguments as a JSON-encoded string; decode once (vLLM's fallback)."""
    parsed, failed = parse_json_native('{"name": "foo", "arguments": "{\\"x\\": 1}"}')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="foo", arguments={"x": 1})


def test_arguments_as_non_json_string_fails() -> None:
    parsed, failed = parse_json_native('{"name": "foo", "arguments": "not json"}')
    assert parsed is None
    assert failed is True


def test_arguments_as_list_fails() -> None:
    """`set(parsed.arguments)` over a list would silently corrupt scoring; reject."""
    parsed, failed = parse_json_native('{"name": "foo", "arguments": [1, 2, 3]}')
    assert parsed is None
    assert failed is True


def test_malformed_json_fails() -> None:
    parsed, failed = parse_json_native("not json at all")
    assert parsed is None
    assert failed is True


def test_missing_name_key_fails() -> None:
    parsed, failed = parse_json_native('{"arguments": {}}')
    assert parsed is None
    assert failed is True


def test_empty_string_fails() -> None:
    parsed, failed = parse_json_native("")
    assert parsed is None
    assert failed is True
