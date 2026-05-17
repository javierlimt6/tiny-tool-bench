"""BFCL v3 `simple` loader (see PLAN.md §3 repo layout, §4.1 Phase 1)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.types import ParsedToolCall, PromptRecord, ToolSchema


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "bfcl" / "BFCL_v3_simple.json"


def load_bfcl_simple(path: Path | None = None) -> Iterator[PromptRecord]:
    """Yield BFCL v3 `simple` prompts as PromptRecord.

    The vendored file is JSONL (one JSON object per line) despite the `.json`
    extension. Each upstream record has shape:

        {
          "id":       "simple_0",
          "question": [[{"role": "user", "content": "..."}]],
          "function": [{"name": ..., "description": ..., "parameters": ...}, ...],
          "possible_answer": [{"<function_name>": {"<arg>": [allowed, values], ...}}]
        }

    `possible_answer[0]` is a single-key dict whose key is the gold function name
    and whose value maps each argument to a list of acceptable values. We keep
    the first acceptable value per argument as a representative `gold_call`, and
    stash the full `possible_answer` list in `metadata["possible_answer"]` so the
    scorer can check any of the allowed values later.
    """
    src = path if path is not None else _DEFAULT_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"BFCL `simple` data not found at {src}. "
            "Phase 1 vendors this file at data/bfcl/BFCL_v3_simple.json relative "
            "to the repo root. Pass an explicit path to load_bfcl_simple() if "
            "running from a non-editable install."
        )
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _to_record(raw)


def _to_record(raw: dict[str, Any]) -> PromptRecord:
    tools = [
        ToolSchema(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        )
        for fn in raw.get("function", [])
    ]

    # BFCL nests questions: question[0] is the first conversation turn list.
    user_message = ""
    question = raw.get("question") or []
    if question and question[0]:
        first_turn = question[0][0]
        user_message = first_turn.get("content", "")

    possible_answer = raw.get("possible_answer")
    gold_call = _extract_gold_call(possible_answer)

    metadata: dict[str, Any] = {}
    if possible_answer is not None:
        metadata["possible_answer"] = possible_answer

    return PromptRecord(
        id=raw["id"],
        source="bfcl_v3",
        category="simple",
        user_message=user_message,
        tools=tools,
        gold_call=gold_call,
        metadata=metadata,
    )


def _extract_gold_call(possible_answer: Any) -> ParsedToolCall | None:
    if not possible_answer:
        return None
    first = possible_answer[0]
    if not isinstance(first, dict) or not first:
        return None
    function_name = next(iter(first))
    arg_choices = first[function_name] or {}
    # Pick a representative value per arg: the first allowed value, skipping the
    # empty-string "any value is fine" sentinel BFCL uses for optional args.
    arguments: dict[str, Any] = {}
    for arg, choices in arg_choices.items():
        if not isinstance(choices, list) or not choices:
            continue
        chosen = next((c for c in choices if c != ""), choices[0])
        arguments[arg] = chosen
    return ParsedToolCall(function_name=function_name, arguments=arguments)
