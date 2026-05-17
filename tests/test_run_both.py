"""Offline tests for the Phase 2 comparison CLI (see PLAN.md §4.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_both


_CONFIG_A_YAML = """\
display_name: "Model A on BFCL v3 simple"
adapter: qwen25_05b
parser: hermes
dataset: bfcl_v3_simple
output_dir: results/runs
"""

_CONFIG_B_YAML = """\
display_name: "Model B on BFCL v3 simple"
adapter: needle
parser: json_native
dataset: bfcl_v3_simple
output_dir: results/runs
"""

_CONFIG_NO_NAME_YAML = """\
adapter: needle
parser: json_native
dataset: bfcl_v3_simple
output_dir: results/runs
"""


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _make_row(
    *,
    prompt_id: str,
    level_4: bool,
    parse_failed: bool = False,
    ttft_ms: float = 800.0,
    total_ms: float = 1300.0,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "raw_text": "<tool_call>{}</tool_call>",
        "parsed": None
        if parse_failed
        else {"function_name": "foo", "arguments": {}},
        "correctness": {
            "level_1_called": not parse_failed,
            "level_2_name": not parse_failed,
            "level_3_args": not parse_failed,
            "level_4_values": level_4 and not parse_failed,
            "parse_failed": parse_failed,
            "notes": "parse_failed" if parse_failed else "ok",
        },
        "timing": {
            "prefill_tokens": 200,
            "decode_tokens": 25,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "schema_tokens": 180,
            "query_tokens": 15,
        },
        "config_hash": "0" * 16,
        "git_commit": "unknown",
    }


def _write_results(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_main_requires_two_config_paths() -> None:
    # argparse exits with SystemExit when required positionals are missing.
    with pytest.raises(SystemExit):
        run_both.main([])
    with pytest.raises(SystemExit):
        run_both.main(["only_one.yaml"])


def test_main_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    existing = _write_config(tmp_path / "a.yaml", _CONFIG_A_YAML)
    with pytest.raises(FileNotFoundError):
        run_both.main([str(existing), str(tmp_path / "missing.yaml")])
    with pytest.raises(FileNotFoundError):
        run_both.main([str(tmp_path / "missing.yaml"), str(existing)])


def test_main_max_n_zero_skips_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_a = _write_config(tmp_path / "a.yaml", _CONFIG_A_YAML)
    cfg_b = _write_config(tmp_path / "b.yaml", _CONFIG_B_YAML)

    def _boom(*args: object, **kwargs: object) -> Path:
        raise AssertionError("runner.run must not be invoked when --max-n 0")

    monkeypatch.setattr(run_both, "run", _boom)

    rc = run_both.main([str(cfg_a), str(cfg_b), "--max-n", "0"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "Comparison on BFCL v3 simple (n=0):" in captured.out
    # Both column headers must appear in the table.
    assert "Model A on BFCL v3 simple" in captured.out
    assert "Model B on BFCL v3 simple" in captured.out
    # Every metric row exists; with zero rows we print "-" placeholders.
    assert "Strict accuracy" in captured.out
    assert "Level 1 (called)" in captured.out
    assert "Level 4 (vals)" in captured.out
    assert "Parse failures" in captured.out
    assert "Prefill schema/query (p50)" in captured.out
    assert "Decode tokens p50/p90/p99" in captured.out
    assert "ttft_ms p50/p90/p99" in captured.out
    assert "total_ms p50/p90/p99" in captured.out
    assert "ttft/total ratio (p50)" in captured.out


def test_main_happy_path_prints_both_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_a = _write_config(tmp_path / "a.yaml", _CONFIG_A_YAML)
    cfg_b = _write_config(tmp_path / "b.yaml", _CONFIG_B_YAML)

    results_a = _write_results(
        tmp_path / "a.jsonl",
        [
            _make_row(prompt_id="a-0", level_4=True, ttft_ms=1000.0, total_ms=1600.0),
            _make_row(prompt_id="a-1", level_4=True, ttft_ms=1100.0, total_ms=1700.0),
            _make_row(prompt_id="a-2", level_4=False, ttft_ms=1200.0, total_ms=1800.0),
        ],
    )
    results_b = _write_results(
        tmp_path / "b.jsonl",
        [
            _make_row(prompt_id="b-0", level_4=True, ttft_ms=400.0, total_ms=600.0),
            _make_row(prompt_id="b-1", level_4=False, ttft_ms=420.0, total_ms=620.0),
            _make_row(
                prompt_id="b-2",
                level_4=False,
                parse_failed=True,
                ttft_ms=440.0,
                total_ms=640.0,
            ),
        ],
    )

    calls: list[tuple[str, int | None]] = []

    def _fake_run(config: dict, max_n: int | None = None) -> Path:
        calls.append((config["display_name"], max_n))
        return results_a if config["display_name"].startswith("Model A") else results_b

    monkeypatch.setattr(run_both, "run", _fake_run)

    rc = run_both.main([str(cfg_a), str(cfg_b)])
    assert rc == 0

    # Both runner invocations happened serially, in the order A then B.
    assert [name for name, _ in calls] == [
        "Model A on BFCL v3 simple",
        "Model B on BFCL v3 simple",
    ]

    captured = capsys.readouterr()
    out = captured.out
    # Headline and both column headers visible.
    assert "Comparison on BFCL v3 simple (n=3):" in out
    assert "Model A on BFCL v3 simple" in out
    assert "Model B on BFCL v3 simple" in out
    # Both columns have non-placeholder values across every row. We assert
    # this by checking that every metric label has at least two numeric
    # tokens on its line (one per model).
    for label in [
        "Strict accuracy",
        "Level 1 (called)",
        "Level 2 (name)",
        "Level 3 (args)",
        "Level 4 (vals)",
        "Parse failures",
        "Prefill schema/query (p50)",
        "Decode tokens p50/p90/p99",
        "ttft_ms p50/p90/p99",
        "total_ms p50/p90/p99",
        "ttft/total ratio (p50)",
    ]:
        line = next(ln for ln in out.splitlines() if label in ln)
        # Strip the label, count any digit groups in the remainder.
        remainder = line.split(label, 1)[1]
        digit_groups = [tok for tok in remainder.replace("/", " ").replace(",", " ").replace("[", " ").replace("]", " ").split() if any(c.isdigit() for c in tok)]
        assert len(digit_groups) >= 2, f"row {label!r} lacks two value columns: {line!r}"

    # Parse failures specifically: A has 0, B has 1.
    pf_line = next(ln for ln in out.splitlines() if "Parse failures" in ln)
    pf_tokens = [t for t in pf_line.split() if t.isdigit()]
    assert pf_tokens == ["0", "1"], pf_line


def test_main_missing_display_name_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_a = _write_config(tmp_path / "a.yaml", _CONFIG_NO_NAME_YAML)
    cfg_b = _write_config(tmp_path / "b.yaml", _CONFIG_B_YAML)

    def _boom(*args: object, **kwargs: object) -> Path:
        raise AssertionError("runner must not be invoked when --max-n 0")

    monkeypatch.setattr(run_both, "run", _boom)

    rc = run_both.main([str(cfg_a), str(cfg_b), "--max-n", "0"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "(unnamed)" in captured.out
    assert "Model B on BFCL v3 simple" in captured.out
