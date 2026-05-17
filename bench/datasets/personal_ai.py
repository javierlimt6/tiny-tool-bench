"""Personal-AI slice loader (PLAN.md §5 #8).

Concatenates per-category JSONL files in data/personal_ai/. Skips missing
files so a partially-landed phase branch is still usable. Each row uses the
BFCL JSONL shape with category-scoped IDs (personal_ai_<category>_NNN).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.types import ParsedToolCall, PromptRecord, ToolSchema

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "personal_ai"
_CATEGORIES = ("timer", "message", "calendar", "weather", "music", "navigation")


def load_personal_ai(root: Path | None = None) -> Iterator[PromptRecord]:
    """Yield PromptRecords from every per-category JSONL in root.

    Missing per-category files are skipped — a partially-merged phase branch
    still loads what's available. Categories load in a deterministic order
    so result JSONLs are byte-stable across reruns.
    """
    src_root = root if root is not None else _DEFAULT_ROOT
    for category in _CATEGORIES:
        path = src_root / f"{category}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield _to_record(json.loads(line))


def _to_record(raw: dict[str, Any]) -> PromptRecord:
    tools = [
        ToolSchema(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        )
        for fn in raw.get("function", [])
    ]
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
        source="personal_ai",
        category=raw.get("category", "personal_ai"),
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
    arguments: dict[str, Any] = {}
    for arg, choices in arg_choices.items():
        if not isinstance(choices, list) or not choices:
            continue
        chosen = next((c for c in choices if c != ""), choices[0])
        arguments[arg] = chosen
    return ParsedToolCall(function_name=function_name, arguments=arguments)
