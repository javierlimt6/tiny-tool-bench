"""Needle 26M adapter (PLAN.md §4.2 Phase 2; JAX, JSON-native output).

Cactus's Needle is a 26M Simple Attention Network distilled from Gemini 3.1,
shipped as a JAX/Flax checkpoint that emits raw JSON tool calls of the form
``[{"name": ..., "arguments": {...}}]`` — no wrapper tag, no Hermes-style
escape token. The companion parser is ``bench.parsers.json_native``.

The upstream public API (``needle.generate``) is synchronous-with-stdout-stream:
when ``stream=True`` it writes each decoded token to ``sys.stdout`` between
forward passes, but does NOT expose a Python callback or generator. To recover
a real TTFT measurement that mirrors what Qwen 2.5 0.5B's adapter captures via
``TextIteratorStreamer`` (see ``bench/adapters/qwen25_05b.py``), we redirect
``sys.stdout`` to a thin wrapper that timestamps the first decoded-token write.
Upstream emits a leading bare ``"\\n"`` after the encoder pass and before the
decode loop starts; we skip whitespace-only writes so TTFT reflects the first
real token, not the encoder-done marker. PLAN.md §2 lists single-stream latency
as a first-class metric and Phase 1 (PR #5) established that approximating TTFT
as ``total_ms / decode_tokens`` is wrong by ~20× on this cohort, so the real
measurement matters.

Default checkpoint path is ``checkpoints/needle.pkl`` (the upstream convention,
produced by ``cd needle && source ./setup``; the 52.6 MB pickle lives in the
``Cactus-Compute/needle`` HuggingFace repo). Pass an absolute path via
``model_id=`` to load from elsewhere. The tokenizer is fetched separately from
the same HuggingFace repo on first ``get_tokenizer()`` call.
"""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from typing import Any

from bench.adapters.base import Adapter
from bench.types import GenerationResult, PromptRecord


def _convert_params_for_needle(
    parameters: dict, function_description: str = ""
) -> dict[str, dict]:
    """Convert BFCL/OpenAI parameters → Needle's canonical per-arg descriptor format.

    Per Cactus's README (https://huggingface.co/Cactus-Compute/needle), Needle expects
    each parameter as a full descriptor dict, not a bare type string:

        {"location": {"type": "string", "description": "City name.", "required": true}}

    BFCL ships ``{"type":"object","properties":{<arg>:{"type":<t>,"description":...}},
    "required":[<arg>, ...]}`` — we translate by lifting ``properties`` to the top
    level and folding the outer ``required`` list into a per-arg boolean.

    Earlier iterations of this adapter flattened to ``{arg: type_string}``, which
    dropped per-arg ``description`` and ``required`` fields entirely. The model
    then had no way to know which args were required and what they were for,
    causing the L2→L3 collapse observed on the personal-AI slice (0.68 → 0.28).
    """
    if not isinstance(parameters, dict):
        return {}
    props = parameters.get("properties", {})
    required_set = set(parameters.get("required", []) or [])
    converted: dict[str, dict] = {}
    for arg_name, spec in props.items():
        if not isinstance(spec, dict):
            converted[arg_name] = {
                "type": "string",
                "description": "",
                "required": arg_name in required_set,
            }
            continue
        converted[arg_name] = {
            "type": str(spec.get("type", "string")),
            "description": str(spec.get("description", "")),
            "required": arg_name in required_set,
        }
        # Preserve enum if present — Cactus's README doesn't show it but the
        # constrained decoder reads it when available.
        if "enum" in spec:
            converted[arg_name]["enum"] = spec["enum"]
    return converted


class _TimedStdout(io.TextIOBase):
    """Stdout wrapper that records perf-counter time of the first non-whitespace write.

    Needle's ``generate(stream=True)`` writes ``"\\n"`` once after the encoder pass
    and before decoding starts (see ``needle/model/run.py``). We treat that bare
    newline as the encoder-done marker and time the next write as TTFT.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first_token_t: float | None = None
        self.token_writes: int = 0

    def write(self, s: str) -> int:  # type: ignore[override]
        # Ignore the leading "\n" emitted before the decode loop starts; time
        # the first non-whitespace write so TTFT == encode + 1-step decode.
        if s.strip():
            if self.first_token_t is None:
                self.first_token_t = time.perf_counter()
            self.token_writes += 1
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        return None


class NeedleAdapter(Adapter):
    def __init__(
        self,
        model_id: str = "checkpoints/needle.pkl",
        device: str | None = None,
    ) -> None:
        # ``model_id`` here is a checkpoint path (filesystem), not an HF repo id —
        # Needle's pickled JAX params don't fit transformers' from_pretrained model.
        # The arg name matches the Qwen adapter's signature on purpose; callers
        # treat it as an opaque identifier.
        self.model_id = model_id
        self.device = device
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._params: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        from needle import (
            SimpleAttentionNetwork,
            get_tokenizer,
            load_checkpoint,
        )

        params, config = load_checkpoint(self.model_id)
        self._params = params
        self._model = SimpleAttentionNetwork(config)
        self._tokenizer = get_tokenizer()
        self._loaded = True

    def generate(self, prompt: PromptRecord) -> GenerationResult:
        if not self._loaded:
            self.load()

        # Needle expects the README-canonical format with full per-arg descriptors:
        #     [{"name":"get_weather","description":"...","parameters":{
        #         "location":{"type":"string","description":"City name.","required":true}}}]
        # See _convert_params_for_needle for the BFCL → Needle translation.
        tools_compact = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _convert_params_for_needle(t.parameters, t.description),
            }
            for t in prompt.tools
        ]
        tools_json = json.dumps(tools_compact, separators=(",", ":"))

        # Prefill breakdown — replicates ``needle.model.run._build_encoder_input``
        # so we can report ``query_tokens`` / ``schema_tokens`` like the Qwen
        # adapter does. Counts include the ``<tools>`` separator in the schema
        # bucket. Truncation matches upstream (max_enc_len=1024 by default).
        query_tokens, schema_tokens, prefill_tokens = self._prefill_breakdown(
            prompt.user_message, tools_json
        )

        from needle import generate as needle_generate

        timed_stdout = _TimedStdout()
        t0 = time.perf_counter()
        with redirect_stdout(timed_stdout):
            raw_text = needle_generate(
                self._model,
                self._params,
                self._tokenizer,
                query=prompt.user_message,
                tools=tools_json,
                stream=True,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0

        # decode_tokens: count the per-token stdout writes (excluding the
        # encoder-done newline and the trailing newline). This matches the
        # real number of decoder steps, which is what PLAN.md §2's "decode
        # tokens reported separately" means.
        decode_tokens = timed_stdout.token_writes

        if timed_stdout.first_token_t is not None:
            ttft_ms = (timed_stdout.first_token_t - t0) * 1000.0
        else:
            # Degenerate case: model emitted nothing (no tool call). TTFT is
            # undefined; report total_ms so downstream timing math doesn't
            # divide by zero. Mirrors the Qwen adapter's fallback.
            ttft_ms = total_ms

        return GenerationResult(
            raw_text=raw_text,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            parse_failed=False,
            query_tokens=query_tokens,
            schema_tokens=schema_tokens,
        )

    def _prefill_breakdown(
        self, query: str, tools_json: str
    ) -> tuple[int | None, int | None, int]:
        """Return (query_tokens, schema_tokens, prefill_tokens).

        Mirrors ``needle.model.run._build_encoder_input``: the encoder input is
        ``[query_tokens..., <tools>, tools_tokens...]`` truncated to
        ``max_enc_len`` (1024). ``schema_tokens`` accounts for the separator.
        Returns ``(None, None, 0)`` if the tokenizer is unavailable, so
        tokeniser quirks don't sink a sweep.
        """
        try:
            from needle.dataset.dataset import DEFAULT_MAX_ENC_LEN

            max_enc_len = DEFAULT_MAX_ENC_LEN
            q_toks = self._tokenizer.encode(query)
            t_toks = self._tokenizer.encode(tools_json)
            max_query = max_enc_len - 2
            if len(q_toks) > max_query:
                q_toks = q_toks[:max_query]
            remaining = max_enc_len - len(q_toks) - 1
            t_toks_trunc = t_toks[:remaining]
            query_tokens = len(q_toks)
            # +1 accounts for the ``<tools>`` separator that lives in the schema bucket.
            schema_tokens = len(t_toks_trunc) + 1
            prefill_tokens = query_tokens + schema_tokens
            return query_tokens, schema_tokens, prefill_tokens
        except Exception:
            return None, None, 0
