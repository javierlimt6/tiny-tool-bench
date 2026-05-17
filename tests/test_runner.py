"""Sequential runner with fakes wired via monkeypatch (PLAN.md §3, §4.1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bench.harness import runner
from bench.types import GenerationResult, ParsedToolCall, PromptRecord, ToolSchema


_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: PromptRecord) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            raw_text=f"<tool_call>{prompt.id}</tool_call>",
            prefill_tokens=10 + self.calls,
            decode_tokens=2,
            ttft_ms=12.5,
            total_ms=42.0,
            parse_failed=False,
        )


def _fake_parser(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
    return ParsedToolCall(function_name="foo", arguments={"x": 1}), False


def _make_prompts() -> list[PromptRecord]:
    tool = ToolSchema(name="foo", description="d", parameters={"type": "object"})
    return [
        PromptRecord(
            id=f"p-{i}",
            source="bfcl_v3",
            category="simple",
            user_message=f"hello {i}",
            tools=[tool],
            gold_call=ParsedToolCall(function_name="foo", arguments={"x": 1}),
            metadata={},
        )
        for i in range(3)
    ]


def _fake_dataset():
    return iter(_make_prompts())


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {
        "fake_adapter": _FakeAdapter,
        "fake_parser": _fake_parser,
        "fake_dataset": _fake_dataset,
    }

    def fake_resolve(registry: dict, key: str):
        return table[key]

    monkeypatch.setattr(runner, "_resolve", fake_resolve)


def test_run_writes_jsonl_with_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    config = {
        "adapter": "fake_adapter",
        "parser": "fake_parser",
        "dataset": "fake_dataset",
        "output_dir": str(tmp_path),
    }

    results_path = runner.run(config)

    assert results_path.exists()
    assert results_path.name == "results.jsonl"
    assert results_path.parent.parent == tmp_path

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    for line in lines:
        row = json.loads(line)
        assert _HEX_16.match(row["config_hash"]), row["config_hash"]
        assert row["git_commit"] == "unknown" or _HEX_40.match(row["git_commit"]), row["git_commit"]
        assert "level_1_called" in row["correctness"]
        assert row["correctness"]["level_1_called"] is True
        assert "total_ms" in row["timing"]
        assert row["timing"]["total_ms"] == 42.0
        assert row["parsed"] == {"function_name": "foo", "arguments": {"x": 1}}


def test_run_respects_max_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    config = {
        "adapter": "fake_adapter",
        "parser": "fake_parser",
        "dataset": "fake_dataset",
        "output_dir": str(tmp_path),
    }

    results_path = runner.run(config, max_n=2)
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_run_writes_parse_failed_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_parser(raw_text: str) -> tuple[ParsedToolCall | None, bool]:
        return None, True

    table = {
        "fake_adapter": _FakeAdapter,
        "fake_parser": failing_parser,
        "fake_dataset": _fake_dataset,
    }

    def fake_resolve(registry: dict, key: str):
        return table[key]

    monkeypatch.setattr(runner, "_resolve", fake_resolve)

    config = {
        "adapter": "fake_adapter",
        "parser": "fake_parser",
        "dataset": "fake_dataset",
        "output_dir": str(tmp_path),
    }
    results_path = runner.run(config)
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["correctness"]["parse_failed"] is True
        assert row["correctness"]["notes"] == "parse_failed"
        assert row["parsed"] is None
