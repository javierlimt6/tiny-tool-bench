"""Hermes-style `<tool_call>` parser (PLAN.md §3 design rule 2).

Mirrors vLLM's Hermes2ProToolParser
(https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py).
Phase 1 evaluates single-tool-call categories only, so on multi-block
output we deliberately keep just the first block.
"""

from __future__ import annotations

import json
import re

from bench.types import ParsedToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_hermes(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
    match = _TOOL_CALL_RE.search(raw_text)
    if match is None:
        return None, True

    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, True

    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return None, True

    return ParsedToolCall(function_name=obj["name"], arguments=obj["arguments"]), False
