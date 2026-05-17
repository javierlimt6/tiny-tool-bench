"""Adapter contract tests — no weights downloaded (PLAN.md §4.1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from bench.adapters.qwen25_05b import Qwen25_05B_Adapter
from bench.types import PromptRecord


def test_constructor_is_lazy() -> None:
    adapter = Qwen25_05B_Adapter()
    assert hasattr(adapter, "generate")
    assert hasattr(adapter, "load")
    assert adapter._loaded is False


def test_constructor_stores_config() -> None:
    adapter = Qwen25_05B_Adapter(model_id="custom/model", device="cpu", dtype="float32")
    assert adapter.model_id == "custom/model"
    assert adapter.device == "cpu"
    assert adapter.dtype == "float32"
    assert adapter._loaded is False


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_invokes_transformers_with_model_id(
    mock_tok: MagicMock, mock_model: MagicMock
) -> None:
    mock_tok.return_value = MagicMock()
    mock_model.return_value = MagicMock()

    adapter = Qwen25_05B_Adapter(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    adapter.load()

    mock_tok.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct")
    assert mock_model.call_args.args == ("Qwen/Qwen2.5-0.5B-Instruct",)
    assert adapter._loaded is True


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_is_idempotent(mock_tok: MagicMock, mock_model: MagicMock) -> None:
    mock_tok.return_value = MagicMock()
    mock_model.return_value = MagicMock()

    adapter = Qwen25_05B_Adapter()
    adapter.load()
    adapter.load()

    assert mock_tok.call_count == 1
    assert mock_model.call_count == 1


@patch("transformers.TextIteratorStreamer")
def test_generate_uses_streamer_for_real_ttft(mock_streamer_cls: MagicMock) -> None:
    """`generate` must drive a streamer (real TTFT), not divide total by tokens."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }

    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 13), dtype=torch.long)

    mock_streamer = MagicMock()
    mock_streamer.__iter__.return_value = iter(["foo ", "bar", "!"])
    mock_streamer_cls.return_value = mock_streamer

    adapter = Qwen25_05B_Adapter()
    adapter._loaded = True
    adapter._tokenizer = tokenizer
    adapter._model = model

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

    # The streamer was constructed with skip_prompt=True so we time the first DECODE chunk.
    assert mock_streamer_cls.called
    call_kwargs = mock_streamer_cls.call_args.kwargs
    assert call_kwargs.get("skip_prompt") is True

    # model.generate was invoked with the streamer AND attention_mask kwargs.
    assert model.generate.called
    gen_kwargs = model.generate.call_args.kwargs
    assert "streamer" in gen_kwargs
    assert "attention_mask" in gen_kwargs

    # apply_chat_template was called with return_dict=True so we got the mask.
    chat_kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert chat_kwargs.get("return_dict") is True

    assert result.raw_text == "foo bar!"
    assert result.prefill_tokens == 10
    assert result.decode_tokens == 3
    assert result.parse_failed is False
    assert 0.0 <= result.ttft_ms <= result.total_ms


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_load_clears_sampling_keys_in_generation_config(
    mock_tok: MagicMock, mock_model: MagicMock
) -> None:
    """Qwen's bundled generation_config defaults trigger spurious warnings under greedy."""
    mock_tok.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.generation_config = MagicMock()
    fake_model.generation_config.temperature = 0.7
    fake_model.generation_config.top_p = 0.8
    fake_model.generation_config.top_k = 20
    mock_model.return_value = fake_model

    adapter = Qwen25_05B_Adapter()
    adapter.load()

    assert fake_model.generation_config.temperature is None
    assert fake_model.generation_config.top_p is None
    assert fake_model.generation_config.top_k is None
