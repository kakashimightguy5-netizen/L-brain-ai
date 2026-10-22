---
tags: [L, scale-up, gpu, results, honest]
created: 2026-10-08
status: internal research record
---
# 107 - First scale-up: L-2 at 8.6M vs transformer at 9.2M (laptop RTX 3050)

Previous: [[105 - L-2 Candidate and Remaining Flaws Fixed]] · [[106 - L-3 Learning New Languages Without Forgetting]]

## Setup
- **Hardware:** the user's RTX 3050 6 GB, through the job runner at `D:\AI\SpikingBrain-study\L-architecture\runner`.
- **Data:** CPython code, file-disjoint (38 MB train / 2.1 MB test).
- **Training:** 6,000 steps × 32 × 256 bytes (49M bytes), lr 1e-3, 1 seed each.

| | Transformer (outside competitor) | **L-2** (d=384, 6 layers, homeostatic spikes, gated trajectory, energy budget 0.03) |
|---|---|---|
| Params | 9.22M | 8.59M |
| **Held-out code, bits/byte** | **1.080** | 1.092 (+1.1%) |
| Exact copy 120 / 512 / 2,048 bytes | 90% / cannot / cannot | **94% / 96% / 95%** |
| Copy with 5% typos (120 / 2,048) | 82% / cannot | **93% / 90%** |
| Copy with renamed variables (120 / 2,048) | 65% / cannot | **77% / 86%** |
| Active state / FFN neurons | dense (100%) | **24–57% / 4–10%** |
| Training time on RTX 3050 | **49 min (0.49 s/step)** | 170 min (1.70 s/step) |

"Cannot" means the text is longer than the transformer's 256-byte context.

## Honest reading
- **At about 1M params L was 8% better on quality. At 9M the transformer is 1.1% better.** L's quality lead did **not** hold at this scale (single seed; seed-to-seed variation was ±0.05 at 1M for the transformer).
- **L still clearly wins on:**
  - exact and fuzzy copying, and context beyond 256 bytes;
  - very sparse activity (FFN 4–10% active);
  - constant cost per token at long context (notes 102–103).
- **L trains 3.5× slower on the GPU.** The cause is tooling: there is no fused scan kernel on Windows yet, so it uses a log-depth scan written in plain PyTorch. This is not an inherent cost of the architecture.
- **Candidate causes of the quality gap (to test, not assume):**
  1. The energy-budget penalty (0.03) costs more at larger scale. Test: **L 9M with no penalty (queued)**.
  2. Variance between seeds. Test: **transformer 9M seed 2 (queued)**.
  3. Spike-count quantisation limits a bigger model's precision.
  4. Learning rate / steps tuned for the transformer.

## Extra seeds finished on the GPU (1M-scale tests; bits/byte)
| Variant | Seeds | Mean |
|---|---|---|
| L-1 (reference) | 3 | 1.353 |
| Gated trajectory, no homeostasis | 3 | 1.349 (1.3486 / 1.3491 / 1.3506) |
| Homeostasis only | 3 | 1.351 |
| Strong energy budget (0.1) | 3 | 1.383 (state 31%, FFN 5–6% active) |
| Layer pairs trained separately | 3 | 1.396 |
| Transformer (reference) | 3 | 1.466 |

Homeostasis is neutral for language quality (its value is long-range state tracking). Gating fixes the trajectory cost.

## L-3 follow-up (cloud CPU)
| | Python | JavaScript | C | Average | Python forgotten |
|---|---|---|---|---|---|
| Columns rank 32 (2 seeds) | 1.34 | 0.94 | 1.66 | 1.314 | +0.03 |
| Columns rank 64 (2 seeds) | 1.337 | 0.870 | 1.60 | **1.270** | +0.03 |
| Rank 64 with **rolling routing** (re-choose every 32 bytes) | 1.334 | 0.876 | **1.564** | — | +0.026 |

Rolling routing helps a little (C 1.588 → 1.564). The Python loss that remains comes from misrouted passages, not from forgetting.

## Next
- Find the root cause of the 9M quality gap (2 runs queued overnight).
- A faster GPU scan.
- 3 seeds at 9M.
