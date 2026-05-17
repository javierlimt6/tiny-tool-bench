"""Sequential JSONL runner (see PLAN.md §3 design rules, §4.1 Phase 1).

The runner WRITES JSONL only; printing the summary is Unit 4's CLI job. JSONL
is opened ``buffering=1`` so a crash leaves every completed prompt durable on
disk. Each row carries ``config_hash`` + ``git_commit`` for provenance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bench.scoring.ast_match import score


_ADAPTERS: dict[str, str] = {
    "qwen25_05b": "bench.adapters.qwen25_05b:Qwen25_05B_Adapter",
}

_PARSERS: dict[str, str] = {
    "hermes": "bench.parsers.hermes:parse_hermes",
}

_DATASETS: dict[str, str] = {
    "bfcl_v3_simple": "bench.datasets.bfcl_v3:load_bfcl_simple",
}


def run(config: dict, max_n: int | None = None) -> Path:
    """Execute ``config`` and write ``results.jsonl`` under a timestamped dir."""
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    git_commit = _git_commit()

    output_dir = Path(config["output_dir"]) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_cls = _resolve(_ADAPTERS, config["adapter"])
    parser_fn = _resolve(_PARSERS, config["parser"])
    dataset_fn = _resolve(_DATASETS, config["dataset"])

    adapter = adapter_cls()
    prompts = _load_dataset(dataset_fn, config)
    if max_n is not None:
        prompts = itertools.islice(prompts, max_n)

    results_path = output_dir / "results.jsonl"
    with open(results_path, "w", buffering=1, encoding="utf-8") as fh:
        for prompt in tqdm(prompts, desc="prompts"):
            gen = adapter.generate(prompt)
            parsed, parse_failed = parser_fn(gen.raw_text)
            correctness = score(parsed, prompt.gold_call, parse_failed, prompt.metadata)
            row = {
                "prompt_id": prompt.id,
                "raw_text": gen.raw_text,
                "parsed": _asdict_or_none(parsed),
                "correctness": dataclasses.asdict(correctness),
                "timing": {
                    "prefill_tokens": gen.prefill_tokens,
                    "decode_tokens": gen.decode_tokens,
                    "ttft_ms": gen.ttft_ms,
                    "total_ms": gen.total_ms,
                },
                "config_hash": config_hash,
                "git_commit": git_commit,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return results_path


def _resolve(registry: dict[str, str], key: str) -> Any:
    if key not in registry:
        raise KeyError(f"unknown {key!r}; registered: {sorted(registry)}")
    module_path, _, attr = registry[key].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _load_dataset(dataset_fn: Any, config: dict) -> Any:
    if "dataset_path" in config:
        return dataset_fn(Path(config["dataset_path"]))
    return dataset_fn()


def _git_commit() -> str:
    # Run from inside the package so we read the bench repo's HEAD, not whatever
    # repo the user happens to be cd'd into when invoking the CLI.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _asdict_or_none(obj: Any) -> dict | None:
    return None if obj is None else dataclasses.asdict(obj)
