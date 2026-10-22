---
tags: [experiments, round-5, internal, lightweight]
created: 2026-10-06
status: internal research record
---
# Round 5: The 3 Partial Modules + Lightweight Proof

Back to [[20 - Final Benchmark Summary]] · [[21 - Honest Verdict on the Architecture]]

## 1. Genome-coded skills → replaced by low-rank "epigenetic" adapters
**Finding:** at equal skill size, a plain low-rank column **beats the frozen-genome column every time**, including few-shot learning.
| Skill size | Genome column | **Low-rank adapter (same size)** | Full column |
|---|---|---|---|
| ~4.1k params (48× smaller) | 74.5% | **83.4%** | 87.5% (199k) |
| ~8.6k (23×) | 78.2% | **86.9%** | |
| ~11k (18×) | 86.3% (with low-rank correction) | **87.1%** | |
| Few-shot, 10 / 50 / 200 per class | 62.5 / 71.9 / 77.4% | **69.8 / 77.4 / 81.7%** | 70.6 / 78.4 / 83.4% |
- Evolving the genome across 3 environments and using bigger codes helped a little (74.5 → 79.8%), but not enough.
- **Root cause:** a frozen genome evolved in another world *constrains* a new skill; an adapter of the same size has full freedom.
- **Decision:** skills = **low-rank adapters** (✅ 87.1% at 18× smaller ≈ full column). The genome stays a research item only for compressing whole wide networks (where it beat low-rank in Round 2).

## 2. Early exit → savings grow with depth ✅ (root cause confirmed)
| ~0.9M params | Compute saved | Bits per char |
|---|---|---|
| 4 layers, gate 0.95 | 25% | 2.374 |
| **8 layers, gate 0.95** | **32%** | **2.371** |
| 4 layers, gate 0.9 | 41% | 2.423 |
| **8 layers, gate 0.9** | **47%** | **2.425** |
Full depth: 8-layer 2.345 vs 4-layer 2.355. Layers 6–7 already ≈ final (2.344), so bigger models are expected to save more.

## 3. "I don't know this" for never-imagined inputs → open research problem 🟡
| At 5% false referral | Inverted digits (imagined) | Rotated fashion (imagined) | **Mirrored digits (asym.)** | **Fashion 180°** | Noise |
|---|---|---|---|---|---|
| Self-knowing "mine" | 98.2% | 89.1% | 10.9% | 29.9% | 99.9% |
| + content ("what") familiarity | 96.1% | 77.9% | 26.6% | 19.9% | 95.1% |
| + arrangement ("where") familiarity | 94.8% | 77.5% | 9.2% | 18.3% | 96.4% |
| Imagination transfer (flips imagined on fashion) | 87.4% | 64.3% | **28.1%** | 29.8% | 78.9% |
- **Root cause:** mirrored digits and upside-down clothes fall *inside the normal variation* of known data in every generic feature space (invariant V1 features, pixel layout). Detecting them needs transformation-specific knowledge. This is the general **near-out-of-distribution** problem, still open in ML.
- **Best practical design:** a broad imagination library + human referral.

## 4. Lightweight claim: measured on a plain 2-core CPU, no GPU
| What | Measured |
|---|---|
| Train shared V1 once (12k images) | **7.4 s** |
| Learn a **new skill** (12k images) | **8.3 s** |
| Skill size | **0.4 MB** (V1 0.12 MB) |
| Time per answer (full system) | **2.8 ms** (0.7 ms batched) |
| Our LM vs standard | **0.93 MB vs 3.3 MB**; **3.6 vs 4.2 ms**/char |
| Train an LM from scratch (2,500 steps) | ~10–15 min |
**Line for reviewers:** "Our model trains and runs on a laptop CPU. A GPU only speeds up our research experiments (100+ runs of baselines, seeds and ablations), not the model's needs."

## Updated module status: 13 ✅ / 1 🟡
- Genome skills → adapters ✅
- Early exit ✅ (grows with depth)
- Never-imagined abstention 🟡 (open research)

## Code
`lab-code/`: big6_genomic_capacity.py · big6b_fewshot.py · big7_familiarity.py · big8_lightweight.py · lm_deep_exit.py · results_big6*/big7*/big8*/lm_deep_exit.json
