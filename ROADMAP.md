# Roadmap

Items explicitly deferred past Phase 1 v0.1.0. See `PLAN.md` for the original phase plan.

## Phase 2 (v0.2.0)

- **Needle adapter** (JAX, custom SAN format) — single biggest integration risk; hard 8h cap per PLAN.md §4.2.
- Schema converter: OpenAI tool format → Needle's compact `[{"name":..., "parameters":...}]`.
- JSON-native parser for Needle output.
- `scripts/run_both.py` comparing Qwen and Needle on BFCL `simple`.

## Phase 3 (v0.3.0)

- FunctionGemma adapter (developer-role system message, escape-token parser).
- Granite 4.0 350M adapter.
- LFM2.5 350M adapter (pythonic parser).
- Gemma 4 E2B adapter (INT4 on 8GB VRAM via `bitsandbytes`).
- Extend BFCL loader: `multiple`, `parallel`, `live_simple`, `live_multiple`, `irrelevance`.
- Full sweep on RTX 4060.

## Phase 4 (v0.4.0)

- `scripts/generate_ood.py`: 100 prompts in held-out ontologies via Claude API.
- `data/adversarial/`: 50 hand-curated prompts.
- Run all six models on both new datasets.
- `scripts/report.py`: aggregates JSONL → markdown tables + Pareto plot (matplotlib via `[report]` extra).
- README replacement with positioning, methodology, results, caveats, prior art.
- Tag `v0.1.0`.

## v2 additions (post-v1, 15–20h, PLAN.md §5)

1. MacBook Air M4 + MLX adapters for the cohort (Needle deferred to v3).
2. **INT4 quantisation as first-class result** — generate GGUF/MLX checkpoints, document conversion per model, INT4 in headline tables.
3. OOD expansion to 200 prompts with 3 paraphrases each → paraphrase-invariance probe.
4. Adversarial expansion to 100 prompts incl. prompt-injection-via-tool-output cases.
5. Sampling-variance bootstrap on bottom-quartile borderline prompts (10-run resampling).
6. McNemar paired tests vs Needle (`scipy.stats.mcnemar`).
7. **Tool-count scaling slice** — sweep {1, 5, 15, 30} tools per prompt, plot TTFT vs tool count. Motivated by the Phase 1 finding that schema tokens dominate prefill (`memory/project_phase1_findings.md`). The most directly actionable result for on-device-assistant builders and does not exist publicly for this size class.
8. Personal-AI slice — 100–200 prompts mimicking on-device assistant usage (timer, message, calendar, weather, music, navigation). Operationalises Cactus's narrow claim.

## v3 additions (post-v2, 15–20h, PLAN.md §6)

1. iPhone 16 Pro Max via MLX Swift — single largest risk; minimal Swift host app loading MLX models from disk.
2. `llama-cpp-python` cross-runtime comparison (GGUF Q4_K_M for all GGUF-supporting models).
3. Multi-turn BFCL categories (Needle scored N/A explicitly).
4. LLM-judge stage-2 scoring for OOD failures (frontier model rescores strict-AST rejections).
5. Energy and cold-start measurement where platform permits.
6. Updated cohort with any new sub-2B function-callers shipped by then.

## Long-tail (post-v3, no commitment)

- Constrained decoding comparison (XGrammar, outlines, lm-format-enforcer).
- Fine-tuning recipe comparison (cohort fine-tuned on the OOD set).
- Multi-language BFCL (Java, JavaScript, SQL).
- Real production tool catalogues (HubSpot, Asana, GitHub API surfaces).

## Discipline

- Phase gates are absolute (PLAN.md §7). No Phase N+1 until Phase N prints a number.
- New ideas → this file, never mid-build implementation.
- Commit and push at every phase boundary.
- Sunday-evening hard ship deadline applies to v1 even if 2 of 6 models fail (PLAN.md §11 fallback).
