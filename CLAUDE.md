# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`tiny-tool-bench` (a.k.a. `tiny-fc-bench`) is a standardised function-calling benchmark for sub-2B language models. It runs the cohort (Needle 26M, FunctionGemma 270M, Qwen 2.5 0.5B, Granite 4.0 350M, LFM2.5 350M, Gemma 4 E2B) against BFCL v3, a synthetic OOD set, and an adversarial slice — same parser per output format, same hardware tier, so the numbers are directly comparable.

The full design — phases, deferrals, discipline rules — lives in `PLAN.md` at the repo root. **`PLAN.md` is intentionally untracked** (it's the working design doc the maintainer keeps locally). Always read it before substantive changes; ask the user for the file if it's missing.

Phase status: Phase 1 (Qwen 2.5 0.5B on BFCL `simple`) is shipped on `main`. See `ROADMAP.md` for what's next.

## Common commands

```bash
# Dev install with tests
pip install -e ".[dev]"

# Full suite
pytest tests/

# Single file
pytest tests/test_hermes_parser.py -v

# Single test
pytest tests/test_scoring.py::test_bfcl_possible_answer_allows_any_listed_value

# Run a sweep (downloads Qwen 2.5 0.5B ~1GB on first call)
python scripts/run_one.py configs/qwen25_05b_bfcl_simple.yaml --max-n 10   # smoke
python scripts/run_one.py configs/qwen25_05b_bfcl_simple.yaml              # full ~400, ~23min CPU

# GPU/SSH env check before Phase 2+ work
python scripts/check_env.py
```

Phase-2+ extras:
- `pip install -e ".[needle]"` — JAX for the Needle adapter (Phase 2)
- `pip install -e ".[gpu]"` — bitsandbytes for INT4 Gemma 4 E2B
- `pip install -e ".[report]"` — matplotlib for the Phase 4 Pareto plots
- `pip install flash-attn --no-build-isolation` — CUDA-only, manual, 10–20 min compile

Install order on a fresh GPU box: `torch` → `jax[cuda12]` → `pip install -e .` → flash-attn last. Torch and JAX both want to own the CUDA runtime; resolving simultaneously fights.

## Architecture (the big picture)

### One contract: `bench/types.py`

Six pydantic `BaseModel`s with `ConfigDict(extra="forbid")` are the only abstraction that matters:

- `ToolSchema`, `ParsedToolCall`, `PromptRecord` — input side
- `GenerationResult` — adapter output (includes optional `query_tokens` / `schema_tokens` for the schema-vs-query prefill breakdown)
- `CorrectnessResult` — scorer output
- `ResultRow` — one JSONL line: ties everything together with `config_hash` + `git_commit` provenance

Every other module imports from `bench.types`. Adding a field is a contract change that ripples — update all writers AND readers; `extra="forbid"` will surface drift loudly. JSONL round-trip is `row.model_dump_json()` ⇄ `ResultRow.model_validate_json(line)` — both sides share the same schema.

### Six layers, strict separation (PLAN.md §3 design rules)

```
datasets (BFCL JSONL loader)         → yields PromptRecord
adapters/<model>.py                  → PromptRecord → GenerationResult (one file per model)
parsers/<format>.py                  → raw_text → (ParsedToolCall | None, parse_failed)  (one per output format, no per-prompt fallbacks)
scoring/ast_match.py                 → 4-level hierarchical: called → name → arg-keys → values
scoring/aggregate.py                 → summary dict with bootstrap CI on L4
harness/runner.py                    → sequential loop, writes JSONL line-buffered
scripts/run_one.py                   → CLI, reads JSONL back, prints summary
```

### Registry-based dispatch

`bench/harness/runner.py` has three string-keyed registries:

```python
_ADAPTERS = {"qwen25_05b": "bench.adapters.qwen25_05b:Qwen25_05B_Adapter", ...}
_PARSERS  = {"hermes": "...:parse_hermes", "pythonic": "...", "granite": "..."}
_DATASETS = {"bfcl_v3_simple": "...:load_bfcl_simple"}
```

YAML config keys must match these registry keys exactly. **The key is `adapter:` (not `model:`)** — common mistake.

Adding a new model = (1) write `bench/adapters/<model>.py` extending `Adapter`, (2) register it, (3) write a config YAML. The runner, scorer, and CLI need zero changes.

### Adapter shape (Qwen 2.5 0.5B is the reference)

- Lazy `load()`: HF weights only at first `generate()`. Tests must mock `transformers.AutoModelForCausalLM` etc. and assert no actual download.
- `generate()` runs `model.generate(streamer=TextIteratorStreamer(...))` in a daemon thread; main thread stamps `ttft_ms` on the first emitted chunk. **Do not approximate TTFT as `total_ms / decode_tokens`** — that's the per-token average and was off by ~20× on this cohort (see PR #5).
- After `load()`, clear `temperature/top_p/top_k` on `model.generation_config` (set to `None`) — Qwen ships sampling defaults that trigger `UserWarning` spam under `do_sample=False`.
- Always pass `attention_mask` to `model.generate()` — Qwen's `pad_token == eos_token`, so HF can't safely infer it.
- transformers 5.x uses `dtype=` (not `torch_dtype=`).

### JSONL is the persistence layer

No DB, no async, no caching. The runner opens the results file with `buffering=1` so a crash at prompt N leaves N-1 rows durable. `results/runs/<ISO8601>/results.jsonl` is the canonical artifact; the reporter (Phase 4) reads it, the CLI's summary reads it. `results/runs/` is `.gitignore`d.

### Tests must be offline

Adapter tests mock `transformers.AutoModelForCausalLM.from_pretrained` + `AutoTokenizer.from_pretrained` + (when exercising `generate`) `transformers.TextIteratorStreamer`. The HF download path is reserved for the integration smoke run by humans (`scripts/run_one.py --max-n 5`).

## Project conventions

- **No `Co-Authored-By: Claude` trailer** on commits in this repo (project rule; applies to subagent workers too).
- **Phase gates are absolute** (PLAN.md §7): don't start Phase N+1 until Phase N prints a number. New ideas → `ROADMAP.md`, not mid-build implementation.
- **One canonical parser per output format**: no per-prompt fallbacks. Unparseable output → `parse_failed=True`, not a parser fix.
- **vLLM is the parser reference**, not a runtime dep. New parsers should vendor (Apache-2.0 with provenance comment) from `vllm.entrypoints.openai.tool_parsers/*`, trimmed to the non-streaming `text → ParsedToolCall` function.
- Pin direct deps in `pyproject.toml` `dependencies = [...]`. `requirements.txt` mirrors for fallback installs.
- Python 3.11+, full type hints, `from __future__ import annotations` at module top, `pathlib.Path` (never `os.path`).
- `@dataclass(slots=True)` was the original types choice; PR #7 migrated to pydantic `BaseModel`. New types should follow the pydantic pattern.

## Known cohort version constraints (for Phase 2+)

- `transformers >= 5.0.0` is required: LFM2.5 needs `>= 4.55`, Gemma 4 E2B needs `>= 5.0`.
- `accelerate` is a **runtime** requirement when transformers uses `device_map="auto"` — missing it surfaces as a confusing mid-load `ImportError`, not at install time.
- `sentencepiece` for FunctionGemma + Granite tokenisers.
- `einops` for LFM2.5's hybrid LIV-conv + GQA architecture.
- `bitsandbytes` for INT4 Gemma 4 E2B (RTX 4060 8GB VRAM can't fit FP16).
- Needle needs a custom MLX-Swift port for the iPhone tier (v3); on the GPU tier it uses JAX via the `[needle]` extra.

## Phase-1 findings worth knowing (motivates Phase 2+ decisions)

On Qwen 2.5 0.5B on BFCL `simple` (n=400, CPU):

- **Prefill dominates wall-clock**: TTFT/total = 0.61 at p50. ~80% of prefill is tool-schema JSON, not the user query (`schema_tokens=202 vs query_tokens=21` typical). Prefill cost scales with **tool count and complexity**, not query length.
- **p99 tail is decode-length-driven, not noise**: `r(decode, total) = +0.71` vs `r(prefill, total) = +0.54`. Model occasionally produces 100–185 decode tokens (vs median 36). Greedy is reproducible per-prompt; the tail is model output variance.
- These two findings together justify ROADMAP.md's v2 tool-count-scaling experiment (sweep {1, 5, 15, 30} tools, plot TTFT vs tool count).

## Where things live

- `PLAN.md` (untracked) — full design, phase plans, discipline rules, definition of done.
- `ROADMAP.md` — Phase 2/3/4 + v2/v3 deferrals + long-tail.
- `README.md` — user-facing: setup, run, methodology notes.
- `HISTORY.md` (gitignored) — human-readable narrative of what's shipped, when, why.
- `data/bfcl/SOURCE.md` — provenance of the vendored BFCL JSONL (URL, retrieval date, upstream commit).
