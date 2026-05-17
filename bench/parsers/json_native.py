"""JSON-native tool call parser (PLAN.md §4.2 Phase 2; Needle 26M).

Needle emits raw JSON without a wrapper tag: either ``{"name": ..., "arguments": ...}``
or a list of such objects. Mirrors the JSON-string-arguments fallback from
``bench/parsers/hermes.py`` and the ``raw_decode`` trailing-text tolerance from
``bench/parsers/granite.py``. Phase 1/2 evaluates single-tool-call categories only,
so on a multi-call list we deliberately keep just the first object.
"""

from __future__ import annotations

import json

from bench.types import ParsedToolCall


def parse_json_native(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
    payload = raw_text.strip()
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
    # Some models emit arguments as a JSON-encoded string (vLLM's parser does the
    # same fallback). Decode once; if it's still not a dict, that's a parse failure.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, True
    if not isinstance(arguments, dict):
        return None, True

    return ParsedToolCall(function_name=obj["name"], arguments=arguments), False
