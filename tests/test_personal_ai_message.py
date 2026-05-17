"""Validate the message.jsonl file shape (personal-AI slice, PLAN.md §5 #8)."""
from __future__ import annotations
import json
from pathlib import Path


def test_message_jsonl_well_formed() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/message.jsonl").open() if l.strip()]
    assert 25 <= len(rows) <= 40
    seen = set()
    for r in rows:
        assert r["category"] == "message"
        assert r["id"].startswith("personal_ai_message_") and r["id"] not in seen
        seen.add(r["id"])
        assert r["question"][0][0]["role"] == "user"
        assert len(r["function"]) == 1
        fn = r["function"][0]
        assert "name" in fn and "parameters" in fn
        assert len(r["possible_answer"]) == 1 and fn["name"] in r["possible_answer"][0]
        for arg, vals in r["possible_answer"][0][fn["name"]].items():
            assert isinstance(vals, list) and vals


def test_message_anchor_tools_present() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/message.jsonl").open() if l.strip()]
    names = {r["function"][0]["name"] for r in rows}
    for anchor in ("send_message", "read_messages", "reply_to_last_message", "mark_as_read"):
        assert anchor in names, f"missing: {anchor}"
