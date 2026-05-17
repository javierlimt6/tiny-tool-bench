"""Canonical pydantic models shared across adapters, parsers, scoring, runner, CLI.

These are the only abstraction that matters; see PLAN.md §3 for the contract.
We migrated from `@dataclass(slots=True)` to `pydantic.BaseModel` so the JSONL
runner and the CLI loader can serialise/deserialise via one schema each side
(`model_dump_json` ⇄ `model_validate_json`) instead of relying on field-name
coincidence between `dataclasses.asdict` and a `Model(**dict)` splat.

Two fields were added beyond PLAN.md §3's original spec:
- `GenerationResult.query_tokens` / `schema_tokens`: optional prefill breakdown
  so the schema-heavy-prefill story (BFCL tools dominate input) is reportable
  per-row, not inferred.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]  # OpenAI-style JSON schema


class ParsedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_name: str
    arguments: dict[str, Any]


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: Literal["bfcl_v3", "ood", "adversarial", "personal_ai"]
    category: str
    user_message: str
    tools: list[ToolSchema]
    gold_call: ParsedToolCall | None = None  # None for irrelevance
    metadata: dict[str, Any] = {}


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str
    prefill_tokens: int
    decode_tokens: int
    ttft_ms: float
    total_ms: float
    parse_failed: bool
    # Phase 1.1 additions: prefill breakdown for the schema-vs-query story.
    query_tokens: int | None = None
    schema_tokens: int | None = None


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_1_called: bool
    level_2_name: bool
    level_3_args: bool
    level_4_values: bool
    parse_failed: bool
    notes: str


class ResultRow(BaseModel):
    """One JSONL row written by the runner, read back by the CLI/reporter."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    raw_text: str
    parsed: ParsedToolCall | None
    correctness: CorrectnessResult
    timing: dict[str, float | int]
    config_hash: str
    git_commit: str
