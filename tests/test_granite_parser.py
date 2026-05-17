"""Granite parser cases (vendored from vLLM; PLAN.md §3 design rule 2)."""

from __future__ import annotations

from bench.parsers.granite import parse_granite
from bench.types import ParsedToolCall


def test_single_call_object() -> None:
    parsed, failed = parse_granite('<|tool_call|>{"name": "get_weather", "arguments": {"city": "SG"}}')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="get_weather", arguments={"city": "SG"})


def test_call_as_list() -> None:
    parsed, failed = parse_granite('<|tool_call|>[{"name": "get_weather", "arguments": {"city": "SG"}}]')
    assert failed is False
    assert parsed == ParsedToolCall(function_name="get_weather", arguments={"city": "SG"})


def test_arguments_as_json_string() -> None:
    parsed, failed = parse_granite(
        '<|tool_call|>{"name": "foo", "arguments": "{\\"x\\": 1}"}'
    )
    assert failed is False
    assert parsed == ParsedToolCall(function_name="foo", arguments={"x": 1})


def test_no_tag_fails() -> None:
    parsed, failed = parse_granite('I cannot help with that')
    assert parsed is None
    assert failed is True


def test_empty_payload_fails() -> None:
    parsed, failed = parse_granite('<|tool_call|>')
    assert parsed is None
    assert failed is True


def test_malformed_json_fails() -> None:
    parsed, failed = parse_granite('<|tool_call|>not json')
    assert parsed is None
    assert failed is True


def test_missing_name_fails() -> None:
    parsed, failed = parse_granite('<|tool_call|>{"arguments": {}}')
    assert parsed is None
    assert failed is True


def test_arguments_as_list_fails() -> None:
    parsed, failed = parse_granite('<|tool_call|>{"name": "foo", "arguments": [1, 2]}')
    assert parsed is None
    assert failed is True


def test_trailing_text_after_json() -> None:
    """raw_decode handles trailing text after the JSON object."""
    parsed, failed = parse_granite(
        '<|tool_call|>{"name": "foo", "arguments": {}}\n[end of call]'
    )
    assert failed is False
    assert parsed is not None
    assert parsed.function_name == "foo"
