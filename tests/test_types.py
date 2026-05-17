"""Round-trip the canonical dataclasses through dataclasses.asdict (PLAN.md §3)."""

from __future__ import annotations

from dataclasses import asdict

from bench.types import (
    CorrectnessResult,
    GenerationResult,
    ParsedToolCall,
    PromptRecord,
    ToolSchema,
)


def test_tool_schema_round_trip() -> None:
    obj = ToolSchema(
        name="get_weather",
        description="Get the weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    restored = ToolSchema(**asdict(obj))
    assert restored == obj


def test_parsed_tool_call_round_trip() -> None:
    obj = ParsedToolCall(function_name="get_weather", arguments={"city": "Singapore"})
    restored = ParsedToolCall(**asdict(obj))
    assert restored == obj


def _rebuild_prompt_record(d: dict) -> PromptRecord:
    """asdict() recurses into nested dataclasses, so reconstruct by hand."""
    return PromptRecord(
        id=d["id"],
        source=d["source"],
        category=d["category"],
        user_message=d["user_message"],
        tools=[ToolSchema(**t) for t in d["tools"]],
        gold_call=ParsedToolCall(**d["gold_call"]) if d["gold_call"] is not None else None,
        metadata=d["metadata"],
    )


def test_prompt_record_round_trip() -> None:
    obj = PromptRecord(
        id="rec-1",
        source="bfcl_v3",
        category="simple",
        user_message="hello",
        tools=[
            ToolSchema(name="a", description="d-a", parameters={"type": "object"}),
            ToolSchema(name="b", description="d-b", parameters={"type": "object"}),
        ],
        gold_call=ParsedToolCall(function_name="a", arguments={"x": 1}),
        metadata={"split": "test"},
    )
    assert _rebuild_prompt_record(asdict(obj)) == obj


def test_prompt_record_round_trip_irrelevance() -> None:
    obj = PromptRecord(
        id="rec-2",
        source="bfcl_v3",
        category="simple",
        user_message="irrelevant prompt",
        tools=[ToolSchema(name="a", description="", parameters={})],
        gold_call=None,
        metadata={},
    )
    assert _rebuild_prompt_record(asdict(obj)) == obj


def test_generation_result_round_trip() -> None:
    obj = GenerationResult(
        raw_text="<tool_call>...</tool_call>",
        prefill_tokens=128,
        decode_tokens=32,
        ttft_ms=45.5,
        total_ms=210.0,
        parse_failed=False,
    )
    restored = GenerationResult(**asdict(obj))
    assert restored == obj


def test_correctness_result_round_trip() -> None:
    obj = CorrectnessResult(
        level_1_called=True,
        level_2_name=True,
        level_3_args=True,
        level_4_values=False,
        parse_failed=False,
        notes="value mismatch on `unit`",
    )
    restored = CorrectnessResult(**asdict(obj))
    assert restored == obj
