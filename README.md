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

Smoke test the plumbing on 10 prompts (downloads Qwen 2.5 0.5B ~1GB on first run):

```bash
python scripts/run_one.py configs/qwen25_05b_bfcl_simple.yaml --max-n 10
```

You should see something like:

```
Qwen 2.5 0.5B on BFCL v3 simple (n=10):
  Strict accuracy: 0.50 [0.20, 0.80]
  Level 1 (called): 0.90
  Level 2 (name):   0.70
  Level 3 (args):   0.60
  Level 4 (vals):   0.50
  Parse failures:   1
```

Drop `--max-n` for the full sweep (~400 prompts, 10–30 min depending on hardware). Results land in `results/runs/<timestamp>/results.jsonl` for downstream analysis.

## Note on model downloads

Model weights are not vendored. On first run, Hugging Face Transformers downloads Qwen 2.5 0.5B (~1 GB) to `~/.cache/huggingface/` automatically. Subsequent runs reuse the cache.

## Status

Phase 1 (skeleton + Qwen 2.5 on BFCL `simple`) is in progress. See `PLAN.md` §4.1 for the phase plan and `PLAN.md` §11 for the v1 definition of done.

## More

Full design, scope, and roadmap: [`PLAN.md`](PLAN.md).
