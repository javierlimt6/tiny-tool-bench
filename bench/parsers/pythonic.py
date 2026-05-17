"""Pythonic tool call parser (PLAN.md §3 design rule 2; Phase 3: LFM2.5 350M).

LFM2.5 emits a pythonic call like ``[function_name(arg1=value1, arg2='hello')]``.

Vendored from vLLM's PythonicToolParser (Apache-2.0,
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/pythonic_tool_parser.py),
trimmed to the non-streaming text→ParsedToolCall function. Parses safely via ``ast``: no eval,
no exec. Phase 1 keeps only the first call on multi-call output.
"""

from __future__ import annotations

import ast
from typing import Any

from bench.types import ParsedToolCall


def parse_pythonic(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
    text = raw_text.strip()
    if not text:
        return None, True

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None, True

    body = tree.body
    if isinstance(body, ast.List):
        if not body.elts:
            return None, True
        call_node = body.elts[0]
    elif isinstance(body, ast.Call):
        call_node = body
    else:
        return None, True

    if not isinstance(call_node, ast.Call) or not isinstance(call_node.func, ast.Name):
        return None, True

    if call_node.args:
        # Pythonic tool calls use kwargs only; positional args mean we mis-parsed.
        return None, True

    arguments: dict[str, Any] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            return None, True  # **kwargs unpack not supported
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None, True

    return ParsedToolCall(function_name=call_node.func.id, arguments=arguments), False
