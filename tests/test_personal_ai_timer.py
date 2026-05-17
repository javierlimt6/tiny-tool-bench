"""Validate the timer.jsonl file shape (personal-AI slice, PLAN.md §5 #8)."""
from __future__ import annotations
import json
from pathlib import Path


def test_timer_jsonl_well_formed() -> None:
    path = Path("data/personal_ai/timer.jsonl")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert 25 <= len(rows) <= 40, f"expected 25-40 rows, got {len(rows)}"
    seen_ids = set()
    for r in rows:
        assert r["category"] == "timer"
        assert r["id"].startswith("personal_ai_timer_"), r["id"]
        assert r["id"] not in seen_ids, f"duplicate id: {r['id']}"
        seen_ids.add(r["id"])
        assert isinstance(r["question"], list) and r["question"], r["id"]
        assert r["question"][0][0]["role"] == "user"
        assert isinstance(r["question"][0][0]["content"], str)
        assert len(r["function"]) == 1, r["id"]
        fn = r["function"][0]
        assert "name" in fn and "parameters" in fn
        assert isinstance(r["possible_answer"], list) and len(r["possible_answer"]) == 1
        func_name = fn["name"]
        assert func_name in r["possible_answer"][0], f"{r['id']}: possible_answer doesn't reference {func_name}"
        # Allowed values must be lists
        for arg, vals in r["possible_answer"][0][func_name].items():
            assert isinstance(vals, list) and len(vals) >= 1, f"{r['id']}.{arg} allowed-values must be non-empty list"


def test_timer_anchor_tools_present() -> None:
    path = Path("data/personal_ai/timer.jsonl")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    tool_names = {r["function"][0]["name"] for r in rows}
    for anchor in ("set_timer", "cancel_timer", "list_timers", "add_minutes_to_timer"):
        assert anchor in tool_names, f"missing anchor tool: {anchor}"
