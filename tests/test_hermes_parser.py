"""Hermes parser cases (PLAN.md §4.1)."""

from __future__ import annotations

from bench.parsers.hermes import parse_hermes
from bench.types import ParsedToolCall


def test_well_formed() -> None:
    parsed, failed = parse_hermes(
        '<tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>'
    )
    assert failed is False
    assert parsed == ParsedToolCall(function_name="foo", arguments={"x": 1})


def test_surrounding_prose() -> None:
    parsed, failed = parse_hermes(
        'sure thing! <tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call> done'
    )
    assert failed is False
    assert parsed is not None
    assert parsed.function_name == "foo"


def test_no_tags() -> None:
    parsed, failed = parse_hermes("I cannot help with that")
    assert parsed is None
    assert failed is True


def test_malformed_json() -> None:
    parsed, failed = parse_hermes("<tool_call>not json</tool_call>")
    assert parsed is None
    assert failed is True


def test_missing_name_key() -> None:
    parsed, failed = parse_hermes('<tool_call>{"arguments": {}}</tool_call>')
    assert parsed is None
    assert failed is True


def test_multiple_blocks_returns_first() -> None:
    parsed, failed = parse_hermes(
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>'
    )
    assert failed is False
    assert parsed is not None
    assert parsed.function_name == "a"
