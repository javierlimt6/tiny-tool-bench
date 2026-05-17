# tiny-fc-bench

A standardised function-calling benchmark for sub-2B language models, designed for the on-device deployment tier (phones, watches, wearables, embedded hardware). It runs the full cohort (Needle, FunctionGemma, Qwen 2.5 0.5B, Granite 4.0 350M, LFM2.5 350M, Gemma 4 E2B) on BFCL v3, a synthetic OOD set, and an adversarial slice, with one canonical parser per output format on consistent hardware, so the numbers are directly comparable.

## Setup

```bash
git clone https://github.com/<user>/tiny-fc-bench.git
cd tiny-fc-bench
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -v
```

## Run a single sweep

```bash
python scripts/run_one.py configs/qwen25_05b_bfcl_simple.yaml
```

This prints a strict-accuracy table for Qwen 2.5 0.5B on BFCL v3 `simple`. The command becomes available once all four Phase 1 units land; see `PLAN.md` for status.

## Note on model downloads

Model weights are not vendored. On first run, Hugging Face Transformers downloads Qwen 2.5 0.5B (~1 GB) to `~/.cache/huggingface/` automatically. Subsequent runs reuse the cache.

## Status

Phase 1 (skeleton + Qwen 2.5 on BFCL `simple`) is in progress. See `PLAN.md` §4.1 for the phase plan and `PLAN.md` §11 for the v1 definition of done.

## More

Full design, scope, and roadmap: [`PLAN.md`](PLAN.md).
