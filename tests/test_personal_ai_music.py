"""Validate the music.jsonl file shape (personal-AI slice, PLAN.md §5 #8)."""
from __future__ import annotations
import json
from pathlib import Path


def test_music_jsonl_well_formed() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/music.jsonl").open() if l.strip()]
    assert 25 <= len(rows) <= 40
    seen = set()
    for r in rows:
        assert r["category"] == "music"
        assert r["id"].startswith("personal_ai_music_") and r["id"] not in seen
        seen.add(r["id"])
        assert r["question"][0][0]["role"] == "user"
        assert len(r["function"]) == 1
        fn = r["function"][0]
        assert len(r["possible_answer"]) == 1 and fn["name"] in r["possible_answer"][0]
        # Allow empty-args dicts for pause/skip
        args_map = r["possible_answer"][0][fn["name"]]
        for arg, vals in args_map.items():
            assert isinstance(vals, list) and vals


def test_music_anchor_tools_present() -> None:
    rows = [json.loads(l) for l in Path("data/personal_ai/music.jsonl").open() if l.strip()]
    names = {r["function"][0]["name"] for r in rows}
    for anchor in ("play_song", "play_playlist", "pause", "skip", "add_to_queue", "set_volume"):
        assert anchor in names, f"missing: {anchor}"
