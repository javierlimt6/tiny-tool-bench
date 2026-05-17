"""Qwen 2.5 0.5B Instruct adapter (PLAN.md §4.1 Phase 1 reference model).

Uses HuggingFace transformers with greedy decoding and the tokenizer's
built-in chat template, which emits Hermes-style `<tool_call>` blocks
when `tools=` is provided.
"""

from __future__ import annotations

import time

from bench.adapters.base import Adapter
from bench.types import GenerationResult, PromptRecord


class Qwen25_05B_Adapter(Adapter):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        dtype: str = "float16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        )
        if self.device is not None:
            self._model = self._model.to(self.device)
        self._loaded = True

    def generate(self, prompt: PromptRecord) -> GenerationResult:
        if not self._loaded:
            self.load()

        messages = [{"role": "user", "content": prompt.user_message}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in prompt.tools
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        prefill_tokens = int(input_ids.shape[1])

        t0 = time.perf_counter()
        output_ids = self._model.generate(
            input_ids, max_new_tokens=512, do_sample=False
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        new_token_ids = output_ids[0, prefill_tokens:]
        decode_tokens = int(new_token_ids.shape[0])
        raw_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # ttft_ms is approximated as per-token average wall time; true TTFT
        # requires a streaming callback which Phase 1 deliberately skips.
        ttft_ms = total_ms / max(1, decode_tokens)

        return GenerationResult(
            raw_text=raw_text,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            parse_failed=False,
        )
