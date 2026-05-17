"""Abstract adapter interface — one adapter per model (PLAN.md §3 design rule 1).

Adapters are responsible for lazy weight loading and producing a
`GenerationResult` from a `PromptRecord`. Parsing is the parser's job.
"""

from __future__ import annotations

import abc

from bench.types import GenerationResult, PromptRecord


class Adapter(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None:
        """Lazily load model weights and tokenizer. Must be idempotent."""

    @abc.abstractmethod
    def generate(self, prompt: PromptRecord) -> GenerationResult:
        """Run a single prompt through the model and return raw output + timings."""
