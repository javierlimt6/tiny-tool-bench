"""Validate the weather.jsonl file shape (personal-AI slice, PLAN.md §5 #8)."""
from __future__ import annotations
import json
from pathlib import Path


def test_weather_jsonl_well_formed() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/weather.jsonl").open() if l.strip()]
    assert 25 <= len(rows) <= 40
    seen = set()
    for r in rows:
        assert r["category"] == "weather"
        assert r["id"].startswith("personal_ai_weather_") and r["id"] not in seen
        seen.add(r["id"])
        assert r["question"][0][0]["role"] == "user"
        assert len(r["function"]) == 1
        fn = r["function"][0]
        assert len(r["possible_answer"]) == 1 and fn["name"] in r["possible_answer"][0]
        for arg, vals in r["possible_answer"][0][fn["name"]].items():
            assert isinstance(vals, list) and vals


def test_weather_anchor_tools_present() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/weather.jsonl").open() if l.strip()]
    names = {r["function"][0]["name"] for r in rows}
    for anchor in ("get_current_weather", "get_forecast", "get_alerts"):
        assert anchor in names, f"missing: {anchor}"
