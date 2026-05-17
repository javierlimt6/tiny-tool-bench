"""Round-trip the canonical pydantic models via model_dump / model_validate_json (PLAN.md §3)."""

from __future__ import annotations

from bench.types import (
    CorrectnessResult,
    GenerationResult,
    ParsedToolCall,
    PromptRecord,
    ResultRow,
    ToolSchema,
)


def test_tool_schema_round_trip() -> None:
    obj = ToolSchema(
        name="get_weather",
        description="Get the weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    restored = ToolSchema.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_parsed_tool_call_round_trip() -> None:
    obj = ParsedToolCall(function_name="get_weather", arguments={"city": "Singapore"})
    restored = ParsedToolCall.model_validate_json(obj.model_dump_json())
    assert restored == obj


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
    restored = PromptRecord.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_prompt_record_round_trip_personal_ai() -> None:
    # Verify the new "personal_ai" source is accepted
    pa = PromptRecord(
        id="pa-1",
        source="personal_ai",
        category="timer",
        user_message="set a timer",
        tools=[ToolSchema(name="set_timer", description="", parameters={})],
        gold_call=ParsedToolCall(function_name="set_timer", arguments={"duration_seconds": 300}),
        metadata={},
    )
    assert PromptRecord.model_validate_json(pa.model_dump_json()) == pa


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
    restored = PromptRecord.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_generation_result_round_trip() -> None:
    obj = GenerationResult(
        raw_text="<tool_call>...</tool_call>",
        prefill_tokens=128,
        decode_tokens=32,
        ttft_ms=45.5,
        total_ms=210.0,
        parse_failed=False,
        query_tokens=12,
        schema_tokens=98,
    )
    restored = GenerationResult.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_generation_result_token_split_optional() -> None:
    """query_tokens/schema_tokens default to None so older runners still validate."""
    obj = GenerationResult(
        raw_text="x",
        prefill_tokens=10,
        decode_tokens=2,
        ttft_ms=1.0,
        total_ms=2.0,
        parse_failed=False,
    )
    assert obj.query_tokens is None
    assert obj.schema_tokens is None


def test_correctness_result_round_trip() -> None:
    obj = CorrectnessResult(
        level_1_called=True,
        level_2_name=True,
        level_3_args=True,
        level_4_values=False,
        parse_failed=False,
        notes="value mismatch on `unit`",
    )
    restored = CorrectnessResult.model_validate_json(obj.model_dump_json())
    assert restored == obj


def test_result_row_round_trip() -> None:
    """ResultRow is the JSONL contract — runner writes, CLI reads via this schema."""
    parsed = ParsedToolCall(function_name="get_weather", arguments={"city": "SG"})
    correctness = CorrectnessResult(
        level_1_called=True,
        level_2_name=True,
        level_3_args=True,
        level_4_values=True,
        parse_failed=False,
        notes="ok",
    )
    row = ResultRow(
        prompt_id="simple_0",
        raw_text="<tool_call>...</tool_call>",
        parsed=parsed,
        correctness=correctness,
        timing={
            "prefill_tokens": 200,
            "decode_tokens": 25,
            "ttft_ms": 800.0,
            "total_ms": 1200.0,
            "schema_tokens": 180,
            "query_tokens": 12,
        },
        config_hash="abcd1234abcd1234",
        git_commit="0" * 40,
    )
    restored = ResultRow.model_validate_json(row.model_dump_json())
    assert restored == row
    assert restored.parsed == parsed
    assert restored.correctness == correctness
