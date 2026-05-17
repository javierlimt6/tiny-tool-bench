"""Granite-style tool call parser (PLAN.md §3 design rule 2; Phase 3: Granite 4.0 350M).

Granite emits ``<|tool_call|>{"name": "func", "arguments": {...}}`` or a JSON list of objects.

Vendored from vLLM's GraniteToolParser (Apache-2.0,
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/granite_tool_parser.py),
trimmed to the non-streaming text→ParsedToolCall function. Phase 1 keeps only the first call on
multi-call output.
"""

from __future__ import annotations

import json

from bench.types import ParsedToolCall

_TAG = "<|tool_call|>"


def parse_granite(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
    idx = raw_text.find(_TAG)
    if idx < 0:
        return None, True

    payload = raw_text[idx + len(_TAG):].strip()
    if not payload:
        return None, True

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None, True

    if isinstance(obj, list):
        if not obj:
            return None, True
        obj = obj[0]

    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return None, True

    arguments = obj["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, True
    if not isinstance(arguments, dict):
        return None, True

    return ParsedToolCall(function_name=obj["name"], arguments=arguments), False
