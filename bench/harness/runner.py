"""Sequential JSONL runner (see PLAN.md §3 design rules, §4.1 Phase 1).

The runner WRITES JSONL only; printing the summary is Unit 4's CLI job. JSONL
is opened ``buffering=1`` so a crash leaves every completed prompt durable on
disk. Each row carries ``config_hash`` + ``git_commit`` for provenance.
"""

from __future__ import annotations

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
from bench.types import ResultRow


_ADAPTERS: dict[str, str] = {
    "qwen25_05b": "bench.adapters.qwen25_05b:Qwen25_05B_Adapter",
    "needle": "bench.adapters.needle:NeedleAdapter",
}

_PARSERS: dict[str, str] = {
    "hermes": "bench.parsers.hermes:parse_hermes",
    "pythonic": "bench.parsers.pythonic:parse_pythonic",
    "granite": "bench.parsers.granite:parse_granite",
    "json_native": "bench.parsers.json_native:parse_json_native",
}

_DATASETS: dict[str, str] = {
    "bfcl_v3_simple": "bench.datasets.bfcl_v3:load_bfcl_simple",
    "personal_ai": "bench.datasets.personal_ai:load_personal_ai",
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
            timing: dict[str, float | int] = {
                "prefill_tokens": gen.prefill_tokens,
                "decode_tokens": gen.decode_tokens,
                "ttft_ms": gen.ttft_ms,
                "total_ms": gen.total_ms,
            }
            if gen.query_tokens is not None:
                timing["query_tokens"] = gen.query_tokens
            if gen.schema_tokens is not None:
                timing["schema_tokens"] = gen.schema_tokens
            row = ResultRow(
                prompt_id=prompt.id,
                raw_text=gen.raw_text,
                parsed=parsed,
                correctness=correctness,
                timing=timing,
                config_hash=config_hash,
                git_commit=git_commit,
            )
            fh.write(row.model_dump_json() + "\n")

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
