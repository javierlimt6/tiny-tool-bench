"""Offline tests for the Phase 1 CLI (see PLAN.md §4.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_one


_VALID_YAML = """\
display_name: "Test Sweep"
adapter: qwen25_05b
parser: hermes
dataset: bfcl_v3_simple
output_dir: results/runs
"""


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_VALID_YAML, encoding="utf-8")
    return cfg_path


def _write_fake_results(tmp_path: Path) -> Path:
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "correctness": {
                "level_1_called": True,
                "level_2_name": True,
                "level_3_args": True,
                "level_4_values": True,
                "parse_failed": False,
                "notes": "ok",
            }
        },
        {
            "correctness": {
                "level_1_called": True,
                "level_2_name": True,
                "level_3_args": False,
                "level_4_values": False,
                "parse_failed": False,
                "notes": "wrong_args",
            }
        },
        {
            "correctness": {
                "level_1_called": False,
                "level_2_name": False,
                "level_3_args": False,
                "level_4_values": False,
                "parse_failed": True,
                "notes": "parse_failed",
            }
        },
    ]
    with results_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return results_path


def test_main_missing_config_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        run_one.main(["nonexistent.yaml"])


def test_main_max_n_zero_skips_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_config(tmp_path)

    def _boom(*args: object, **kwargs: object) -> Path:
        raise AssertionError("runner.run must not be invoked when --max-n 0")

    monkeypatch.setattr(run_one, "run", _boom)

    rc = run_one.main([str(cfg_path), "--max-n", "0"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "Test Sweep (n=0):" in captured.out
    assert "Strict accuracy: 0.00" in captured.out
    assert "Parse failures:   0" in captured.out


def test_main_runs_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_config(tmp_path)
    fake_results = _write_fake_results(tmp_path)

    def _fake_run(config: dict, max_n: int | None = None) -> Path:
        return fake_results

    monkeypatch.setattr(run_one, "run", _fake_run)

    rc = run_one.main([str(cfg_path)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "Test Sweep (n=3):" in captured.out
    assert "Strict accuracy:" in captured.out
    assert "Parse failures:   1" in captured.out
