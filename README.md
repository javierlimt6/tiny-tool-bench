# tiny-tool-bench

A standardised function-calling benchmark for sub-2B language models, designed for the on-device deployment tier (phones, watches, wearables, embedded hardware). It runs the full cohort (Needle, FunctionGemma, Qwen 2.5 0.5B, Granite 4.0 350M, LFM2.5 350M, Gemma 4 E2B) on BFCL v3, a synthetic OOD set, and an adversarial slice, with one canonical parser per output format on consistent hardware, so the numbers are directly comparable.

## Setup

```bash
git clone https://github.com/javierlimt6/tiny-tool-bench.git
cd tiny-tool-bench
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

The summary also reports a prefill breakdown (schema vs. query tokens) and median TTFT/total latency. On BFCL `simple` the median input is ~266 tokens of which ~250 are tool-schema JSON; the user message is typically 10–20 tokens. Prefill cost therefore scales with the number and complexity of declared tools, not with query length.

## Methodology notes

- **Greedy, single-stream, batch size 1.** No sampling, no batching. Numbers are reproducible per-prompt: same input → bit-identical output.
- **TTFT is measured for real**, not approximated. `model.generate` runs in a background thread with a `TextIteratorStreamer`; the main thread stamps the first emitted chunk. Reporting `total_ms / decode_tokens` (a common shortcut) would have under-reported TTFT by ~20× on this cohort because prefill dominates.
- **The p99 latency tail is decode-length-driven, not measurement noise.** On the Phase 1 Qwen sweep, `r(decode_tokens, total_ms) = +0.71` while `r(prefill_tokens, total_ms) = +0.54`. The slowest prompts are ones where the model occasionally emits 100–185 decode tokens (vs. median 36) — that's a model output-length variance, not a scheduling or thermal artefact.
- **Single canonical parser per format.** No per-prompt fallbacks. If a model emits unparseable output, that is reported as `parse_failed` rather than rescued by a custom handler. This makes the numbers more defensible than leaderboards that permit per-row custom handlers.
- **Strict accuracy uses BFCL's `possible_answer` semantics**: the gold call is one representative value per argument; the scorer admits any value listed in the upstream `possible_answer` allowed-values list per argument.

## Note on model downloads

Model weights are not vendored. On first run, Hugging Face Transformers downloads Qwen 2.5 0.5B (~1 GB) to `~/.cache/huggingface/` automatically. Subsequent runs reuse the cache.

## Status

Phase 1 (skeleton + Qwen 2.5 on BFCL `simple`) is in progress. See `PLAN.md` §4.1 for the phase plan and `PLAN.md` §11 for the v1 definition of done.

## More

Full design, scope, and roadmap: [`PLAN.md`](PLAN.md).
