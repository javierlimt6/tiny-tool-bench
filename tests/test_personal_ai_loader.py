"""Loader tests for the personal-AI slice (PLAN.md §5 #8)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from bench.datasets.personal_ai import _CATEGORIES, load_personal_ai
from bench.types import PromptRecord


def test_load_personal_ai_yields_prompt_records_when_present() -> None:
    """If sibling category files exist, the loader yields valid PromptRecords."""
    root = Path("data/personal_ai")
    present = [c for c in _CATEGORIES if (root / f"{c}.jsonl").exists()]
    if not present:
        pytest.skip("no per-category files present yet (run after Units 1-6 merge)")
    records = list(load_personal_ai())
    assert len(records) >= 25 * len(present)  # each category targets ~25-30
    for r in records:
        assert isinstance(r, PromptRecord)
        assert r.source == "personal_ai"
        assert r.category in _CATEGORIES
        assert r.user_message
        assert len(r.tools) >= 1


def test_load_personal_ai_skips_missing_categories(tmp_path: Path) -> None:
    """An empty root yields zero records (no exception)."""
    records = list(load_personal_ai(root=tmp_path))
    assert records == []


def test_load_personal_ai_preserves_possible_answer_in_metadata(tmp_path: Path) -> None:
    """The scorer reads metadata['possible_answer']; loader must populate it."""
    row = {
        "id": "personal_ai_timer_001",
        "category": "timer",
        "question": [[{"role": "user", "content": "Set a timer for 5 minutes"}]],
        "function": [{"name": "set_timer", "description": "...", "parameters": {"type": "object", "properties": {"duration_seconds": {"type": "integer"}}, "required": ["duration_seconds"]}}],
        "possible_answer": [{"set_timer": {"duration_seconds": [300]}}],
    }
    (tmp_path / "timer.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = list(load_personal_ai(root=tmp_path))
    assert len(records) == 1
    r = records[0]
    assert r.gold_call is not None
    assert r.gold_call.function_name == "set_timer"
    assert r.gold_call.arguments == {"duration_seconds": 300}
    assert r.metadata["possible_answer"] == row["possible_answer"]
