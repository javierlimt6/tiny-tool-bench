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

## Implementation principle for v2/v3

**Reuse existing libraries; don't rebuild.** Apple, llama.cpp, and the MLX community have already shipped the runtime layer for the Mac and iOS tiers. Each v2/v3 item below names the specific library it leans on. The benchmark is the *measurement layer* on top — we don't reimplement model architectures or quantization pipelines that already exist.

The single exception is Needle (custom SAN architecture). On Mac, try `jax[metal]` first; on iPhone, document an "N/A pending Cactus runtime" until upstream publishes its iOS port.

## v2 additions (post-v1, 15–20h, PLAN.md §5)

1. **MacBook Air M4 + MLX adapters** for the cohort (Needle deferred to v3). **Reuse `mlx-lm`** (Apple official): each adapter is a ~60 LOC wrapper around `mlx_lm.load` + `mlx_lm.generate`. Auto-converts HF weights to MLX on first load; native streaming for real TTFT. No bespoke MLX checkpoint conversion needed.
2. **INT4 quantisation as first-class result.** **Reuse `mlx-lm --quantize`** (one flag) and llama.cpp GGUF Q4_K_M for the five Transformers-based cohort members. Bespoke quant only needed for any model that lacks a published quantized checkpoint on HF Hub.
3. OOD expansion to 200 prompts with 3 paraphrases each → paraphrase-invariance probe.
4. Adversarial expansion to 100 prompts incl. prompt-injection-via-tool-output cases.
5. Sampling-variance bootstrap on bottom-quartile borderline prompts (10-run resampling).
6. McNemar paired tests vs Needle (`scipy.stats.mcnemar`).
7. **Tool-count scaling slice** — sweep {1, 5, 15, 30} tools per prompt, plot TTFT vs tool count. Motivated by the Phase 1 finding that schema tokens dominate prefill. The most directly actionable result for on-device-assistant builders and does not exist publicly for this size class.
8. Personal-AI slice — 100–200 prompts mimicking on-device assistant usage. **Already shipped early on `v2/personal-ai` branch** during v1 when the BFCL Needle 0.01 result motivated immediate investigation. Merge into main is gated on the v0.1.0 ship.

## v3 additions (post-v2, 15–20h, PLAN.md §6)

1. **iPhone 16 Pro Max via MLX Swift** — the largest single risk. **Reuse `mlx-swift-examples`** (Apple): fork the Mistral/Llama sample iOS app, swap the prompt-source loop to read YAML config + JSONL dataset and write results back via Files app or HTTP. ~1-2 days of Swift work for the five Transformers-based cohort members, NOT a from-scratch port. Needle on iPhone: "N/A pending Cactus runtime" unless their iOS code ships in time.
2. **`llama-cpp-python` cross-runtime comparison** (GGUF Q4_K_M for all GGUF-supporting models). One adapter file (~80 LOC), pip-installable, Metal on Mac, cross-compiles for iOS. Per-token timing for free. Needle has no GGUF conversion → N/A on this runtime.
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
