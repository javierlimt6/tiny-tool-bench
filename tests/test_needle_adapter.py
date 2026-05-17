"""Needle adapter contract tests — no weights downloaded (PLAN.md §4.2).

Mirrors ``tests/test_qwen_adapter.py``: mocks every external import so the
suite stays offline. The Needle package is mocked at the symbols the adapter
imports (``needle.SimpleAttentionNetwork``, ``needle.load_checkpoint``,
``needle.get_tokenizer``, ``needle.generate``).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from bench.adapters.needle import _flatten_params
from bench.types import PromptRecord


def test_flatten_params_openai_to_needle_format() -> None:
    """BFCL ships OpenAI schemas; Needle wants a flat {arg: type_string} dict."""
    schema = {
        "type": "object",
        "properties": {
            "base": {"type": "number", "description": "base length"},
            "height": {"type": "number"},
        },
        "required": ["base", "height"],
    }
    assert _flatten_params(schema) == {"base": "number", "height": "number"}


def test_flatten_params_empty_schema() -> None:
    assert _flatten_params({}) == {}
    assert _flatten_params({"type": "object"}) == {}


def test_flatten_params_missing_type_defaults_to_string() -> None:
    schema = {"properties": {"x": {"description": "no type"}}}
    assert _flatten_params(schema) == {"x": "string"}


@pytest.fixture(autouse=True)
def _stub_needle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a stub ``needle`` module so ``import needle`` succeeds offline.

    Each test replaces the relevant attributes with its own mocks via ``patch``,
    but the package itself must be importable for ``from needle import ...`` to
    even reach ``patch``'s lookup.
    """
    needle_pkg = types.ModuleType("needle")
    needle_pkg.SimpleAttentionNetwork = MagicMock()  # type: ignore[attr-defined]
    needle_pkg.TransformerConfig = MagicMock()  # type: ignore[attr-defined]
    needle_pkg.load_checkpoint = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[attr-defined]
    needle_pkg.get_tokenizer = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    needle_pkg.generate = MagicMock(return_value="")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", needle_pkg)

    needle_dataset = types.ModuleType("needle.dataset")
    needle_dataset_dataset = types.ModuleType("needle.dataset.dataset")
    needle_dataset_dataset.DEFAULT_MAX_ENC_LEN = 1024  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle.dataset", needle_dataset)
    monkeypatch.setitem(sys.modules, "needle.dataset.dataset", needle_dataset_dataset)


def test_constructor_is_lazy() -> None:
    from bench.adapters.needle import NeedleAdapter

    adapter = NeedleAdapter()
    assert hasattr(adapter, "generate")
    assert hasattr(adapter, "load")
    assert adapter._loaded is False


def test_constructor_stores_config() -> None:
    from bench.adapters.needle import NeedleAdapter

    adapter = NeedleAdapter(model_id="/tmp/custom.pkl", device="cpu")
    assert adapter.model_id == "/tmp/custom.pkl"
    assert adapter.device == "cpu"
    assert adapter._loaded is False


def test_load_invokes_needle_api() -> None:
    """``load()`` calls ``needle.load_checkpoint`` once with the configured path,
    plus constructs the model and tokenizer."""
    from bench.adapters.needle import NeedleAdapter

    with patch("needle.load_checkpoint") as mock_load, patch(
        "needle.SimpleAttentionNetwork"
    ) as mock_arch, patch("needle.get_tokenizer") as mock_tok:
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_arch.return_value = MagicMock()
        mock_tok.return_value = MagicMock()

        adapter = NeedleAdapter(model_id="checkpoints/needle.pkl")
        adapter.load()

        mock_load.assert_called_once_with("checkpoints/needle.pkl")
        assert mock_arch.call_count == 1
        assert mock_tok.call_count == 1
        assert adapter._loaded is True


def test_load_is_idempotent() -> None:
    from bench.adapters.needle import NeedleAdapter

    with patch("needle.load_checkpoint") as mock_load, patch(
        "needle.SimpleAttentionNetwork"
    ) as mock_arch, patch("needle.get_tokenizer") as mock_tok:
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_arch.return_value = MagicMock()
        mock_tok.return_value = MagicMock()

        adapter = NeedleAdapter()
        adapter.load()
        adapter.load()

        assert mock_load.call_count == 1
        assert mock_arch.call_count == 1
        assert mock_tok.call_count == 1


def test_generate_shape_with_mocked_model() -> None:
    """End-to-end mock: ``generate`` produces a populated ``GenerationResult``
    with real TTFT captured via the stdout-redirect hook (whitespace skipped)."""
    from bench.adapters.needle import NeedleAdapter

    fake_tokenizer = MagicMock()
    # Distinct token counts for query vs tools so the prefill split is checkable.
    fake_tokenizer.encode.side_effect = lambda text: (
        [1, 2, 3] if "hi" in text else [10, 11, 12, 13, 14]
    )

    def fake_needle_generate(model, params, tokenizer, query, tools, stream):
        # Emulate upstream: encoder-done newline first, then decoded tokens.
        # See ``needle/model/run.py``'s ``generate`` for the exact stdout pattern.
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.stdout.write('[{"name":"f","')
        sys.stdout.flush()
        sys.stdout.write('"arguments":{}}]')
        sys.stdout.flush()
        sys.stdout.write("\n")
        return '[{"name":"f","arguments":{}}]'

    with patch("needle.generate", side_effect=fake_needle_generate):
        adapter = NeedleAdapter()
        adapter._loaded = True
        adapter._tokenizer = fake_tokenizer
        adapter._model = MagicMock()
        adapter._params = MagicMock()

        record = PromptRecord(
            id="t",
            source="bfcl_v3",
            category="simple",
            user_message="hi",
            tools=[],
            gold_call=None,
            metadata={},
        )
        result = adapter.generate(record)

        assert result.raw_text == '[{"name":"f","arguments":{}}]'
        assert result.parse_failed is False
        # Only the two non-whitespace token writes count toward decode_tokens.
        assert result.decode_tokens == 2
        assert result.ttft_ms >= 0.0
        assert result.ttft_ms <= result.total_ms
        # prefill_tokens uses Needle's encoder layout: query + tools + sep.
        assert result.prefill_tokens > 0
        assert result.query_tokens == 3
