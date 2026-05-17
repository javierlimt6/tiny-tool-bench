"""Qwen 2.5 0.5B Instruct adapter (PLAN.md §4.1 Phase 1 reference model).

Uses HuggingFace transformers with greedy decoding and the tokenizer's
built-in chat template, which emits Hermes-style `<tool_call>` blocks
when `tools=` is provided. `ttft_ms` is measured for real by running
`model.generate` in a background thread with a `TextIteratorStreamer`
attached, then timing the arrival of the first emitted chunk in the
main thread — PLAN.md §2 lists single-stream latency as a first-class
metric so the prefill/decode split needs to be genuine.
"""

from __future__ import annotations

import time
from threading import Thread
from typing import Any

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
        # Qwen 2.5 ships a generation_config with temperature/top_p/top_k that trigger
        # spurious "do_sample=False but X is set" warnings under greedy decoding. None
        # them out so the greedy contract is explicit and the logs aren't noisy.
        for attr in ("temperature", "top_p", "top_k"):
            setattr(self._model.generation_config, attr, None)
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

        inputs = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        prefill_tokens = int(input_ids.shape[1])

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        captured: dict[str, Any] = {}

        def _run() -> None:
            captured["output_ids"] = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=False,
            )

        t0 = time.perf_counter()
        thread = Thread(target=_run)
        thread.start()

        ttft_ms: float | None = None
        chunks: list[str] = []
        for chunk in streamer:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            chunks.append(chunk)
        thread.join()
        total_ms = (time.perf_counter() - t0) * 1000.0

        output_ids = captured["output_ids"]
        decode_tokens = int(output_ids.shape[1] - prefill_tokens)
        raw_text = "".join(chunks)
        # Degenerate case: model emitted nothing (immediate EOS). TTFT is undefined; report total.
        if ttft_ms is None:
            ttft_ms = total_ms

        return GenerationResult(
            raw_text=raw_text,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            parse_failed=False,
        )
