"""Canonical dataclasses shared across adapters, parsers, scoring, and runner.

These are the only abstraction that matters; see PLAN.md §3 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict  # OpenAI-style JSON schema


@dataclass(slots=True)
class ParsedToolCall:
    function_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class PromptRecord:
    id: str
    source: Literal["bfcl_v3", "ood", "adversarial"]
    category: str
    user_message: str
    tools: list[ToolSchema]
    gold_call: ParsedToolCall | None  # None for irrelevance
    metadata: dict


@dataclass(slots=True)
class GenerationResult:
    raw_text: str
    prefill_tokens: int
    decode_tokens: int
    ttft_ms: float
    total_ms: float
    parse_failed: bool


@dataclass(slots=True)
class CorrectnessResult:
    level_1_called: bool
    level_2_name: bool
    level_3_args: bool
    level_4_values: bool
    parse_failed: bool
    notes: str
