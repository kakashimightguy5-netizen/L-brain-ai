# 129 Limitations-of-Transformers Map vs L (2026-10-11)

Source: pasted report "Transformer architecture: limitations, research, and fixing ideas" (90 issues T01-T90, 27 references). Goal stated by user: L should be built against the real limitations of transformers for general intelligence, not as a narrow model. No claim of AGI/ASI is made or supportable (report T64: no accepted sufficiency result).

## Where L stands (design, not yet all measured)
- Addressed by design: T01 quadratic cost, T03 KV growth, T05 prefill (no attention; recurrence + constant state).
- Partly there: T21 compressed-state recall (fast-weight delta memory + hash-recall hippocampus = the report's idea 1, exact memory + compressed state); T08 fixed compute per token (integer spikes allow zero-skip, NOT yet measured on a trained model); T23-T25 durable/versioned memory (hippocampus only; no versioning/consolidation).
- Existing evidence: parity state-tracking tests (`parity_L0.py`, results `q3-q5_parity_L`).
- Not addressed (need objectives/tools/verifiers): T35, T46, T51-T56, T62, T64.

## New test suite: `lim_suite.py` (code/), jobs `1619kh*` (queued, 3 seeds x 3 tasks x {L5, transformer})
- kv: in-context recall, train 4-8 pairs, test up to 128 (T14/T15).
- hop2: two-hop lookup with distractors, train <=4 chains, test up to 32 (T19).
- reverse: fixed facts trained forward only, test reverse direction (T54).
- Models: L5Model d128/4 layers (936,357 params) vs TModel d128/4 layers (990,208 params, learned positions to 1024). Same steps (3000), same data. Harness smoke-tested for 6 steps on CPU only (no results).
- Rules: 3 seeds, MEASURED accuracy, report where L loses.
- Next suite items: counting/parity depth, continual learning with forgetting, calibration/abstention, joules per verified answer.

## Update 2026-10-11 — what the big labs actually disclose (research, 5 searches + 4 page fetches)
**Fact boundary:** Anthropic does not publish Claude's architecture (searches found only third-party speculation; not used). OpenAI's gpt-oss model card (arXiv 2508.10925) discloses only "mixture-of-experts transformer" in its abstract (the fetch showed no attention-pattern, context or parameter details; no limitations section). Gemini 2.5 report (arXiv 2507.06261): sparse-MoE transformer, native text/vision/audio, up to 1M-token context, smaller models trained by distillation. So "beating Claude/GPT/Gemini's architecture" cannot be a literal head-to-head: we can only compare against public designs and our own matched transformer.

**Limitations the labs themselves admit (Gemini 2.5 report):** agent tendency to repeat actions from its history beyond ~100K tokens (anecdotal); effective use of long context for agents is "a new research frontier"; struggled with raw pixels (Game Boy screen); hallucination in agentic runs; prompt-injection resilience weaker for Pro than Flash; safety violation rates slightly higher with visible thinking traces.

**Public open designs worth benchmarking against:** Qwen3-Next: 75% Gated DeltaNet linear-attention layers + 25% gated attention, 80B total / ~3B active MoE (512 experts, 10 routed + 1 shared), reported >10x prefill/decode throughput beyond 32K vs Qwen3-32B (vendor numbers; efficiency "depends heavily on the implementation"). DeepSeek-V3.2-Exp: sparse attention with a lightning indexer + top-k selection, core attention O(L^2)->O(Lk) (vendor claim; post gives no figures). Both are efficiency moves; neither fixes memory/consistency limits.

**Gap L can target:** "stale/conflicting history" and exact long-range recall at constant state. New suite task `kvupd` (rewritten key, ask latest value; train 4-8 events, test to 128) added for L5 vs transformer, 3 seeds (`1619kh*kvupd_*`). Honest definition of a win: L5 >= matched transformer on recall/update tasks at lengths beyond training, at lower memory and energy, and no loss on the compression-gate quality; anything else is reported as a loss.
