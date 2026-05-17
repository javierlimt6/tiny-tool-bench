"""Loader sanity for the vendored BFCL v3 `simple` data (PLAN.md §4.1)."""

from __future__ import annotations

from bench.datasets.bfcl_v3 import load_bfcl_simple


def test_load_bfcl_simple_records() -> None:
    records = list(load_bfcl_simple())
    assert len(records) >= 390, f"expected at least 390 records, got {len(records)}"

    for rec in records:
        assert rec.source == "bfcl_v3"
        assert rec.category == "simple"
        assert rec.id, "id must be non-empty"
        assert len(rec.tools) >= 1, f"{rec.id} has no tools"
        if rec.gold_call is not None:
            assert rec.gold_call.function_name, f"{rec.id} gold_call has empty function_name"
