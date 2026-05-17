"""Adapter contract tests — no weights downloaded (PLAN.md §4.1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bench.adapters.qwen25_05b import Qwen25_05B_Adapter


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
