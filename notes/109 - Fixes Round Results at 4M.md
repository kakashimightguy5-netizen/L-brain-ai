---
tags: [L, L-4, results, 4M, energy-budget, effort-gate, root-cause, honest]
created: 2026-10-08
status: internal research record
---
# 109 - Fixes round: results at 4M, and the root cause of the 9M loss

Previous: [[108 - L-4 Fixes Fast-Weight Memory Effort Gate Routing]] · [[107 - First Scale-Up to 9M on Laptop GPU]] · Next: [[110 - L-5 Error-Correcting Memory Breaks the Ceiling]]

## Setup
- **Data:** CPython code, file-disjoint (38 MB train / 2.1 MB test).
- **Training:** 3,000 steps × 16 × 256 bytes, lr 2e-3, cloud CPU (2 cores).
- **Scoring:** bits/byte on held-out code (lower is better).

## Main comparison at about 4M params
| Model | Params | Seeds | Bits/byte | vs transformer |
|---|---|---|---|---|
| Transformer (outside competitor) | 4.35M | 3 | 1.2786 / 1.3147 / 1.2937 → **1.296 ± 0.018** | — |
| L-2 (gated trajectory + homeostasis, λ 0.03) | 3.96M | 2 | 1.2311 / 1.2325 → 1.232 | −4.9% |
| L-4 (+ fast-weight memory, λ 0.03) | 3.94M | 3 | 1.2271 / 1.2314 / 1.2266 → 1.228 ± 0.003 | −5.2% |
| **L-4, light energy budget (λ 0.01)** | 3.94M | 3 | 1.2181 / 1.2201 / 1.2201 → **1.219 ± 0.001** | **−5.9%** |
| L-4, no energy budget (λ 0) | 3.94M | 1 | 1.2171 | −6.1% |
| (L-2 at 3.28M, a smaller-width run by mistake) | 3.28M | 1 | 1.2468 | −3.8% |

- L stays clearly ahead at 4M, with 10% fewer params.
- L is also far more stable across seeds (± 0.001–0.003 vs ± 0.018).

## Energy budget: quality vs sparsity at 4M (L-4, seed 0)
| λ | Bits/byte | State active (by layer) | FFN active |
|---|---|---|---|
| 0 | 1.2171 | 63–78% | 7–13% |
| **0.01** | **1.2181** | **43–64%** | **5–10%** |
| 0.03 | 1.2271 | 27–49% | 4–7% |

- **λ 0.01 is the sweet spot.** It is as good as no budget (+0.001) with about 25% fewer active state neurons.
- The cost of λ 0.03 is 0.8% at 4M vs 1.1% at 1M. **It does not grow with scale.**
- New default: λ 0.01.

## Copying (L-4 at 4M, 3 seeds)
| Test | 120 bytes | 512 | 2,048 |
|---|---|---|---|
| Exact | 94% | 96% | 95% |
| 5% typos | 94% | 92% | 89–90% |
| Renamed variables | 78–80% | 83–84% | 86–87% |
| Transformer (exact / typos / renamed) | 84–86% / 77–78% / 62–66% | cannot | cannot |

## Effort gate v2 at 8 layers (1.13M params, 1 seed)
| Model | Full depth | Gate 0.99 | Gate 0.95 | Gate 0.9 |
|---|---|---|---|---|
| Plain L-4, 8 layers (no gate) | **1.3437** | — | — | — |
| L-4 + effort gate v2, 8 layers | 1.389 | 1.401 at 83% compute | 1.450 at 72% | 1.501 at 66% |
| (4 layers, note 108) | 1.385 | 1.392 at 86% | 1.421 at 77% | 1.456 at 72% |

- **Still does not pay off.** The training overhead (+0.045) is larger than the 17% compute it saves.
- The hypothesis carried over from transformer note 22 (savings grow with depth) was **not confirmed** for L.
- Verdict: **effort gate parked.** L's per-token cost is already low and constant.

## Where the 9M gap came from
| Hypothesis | Evidence |
|---|---|
| Energy budget hurts at scale | ❌ Not at 4M (cost 0.8% and shrinking) |
| Transformer seed variance | 🟡 Seed spread ±0.018; it cannot explain a consistent flip |
| Learning rate (laptop used 1e-3, cloud 2e-3) | ❌ At 4M: L 1.2181 → 1.2597 (+3.4%), transformer 1.2786 → 1.3148 (+2.8%). Both are hurt about equally, and L still leads by 4.2% at 1e-3 |
| Spike-count precision | ⏹ Not needed (root cause found) |
| **More training data/compute** (9M used 4× more tokens) | ✅ **ROOT CAUSE** (table below) |

## Root cause: L learned faster but had a lower ceiling
| 4M models | 3,000 × 16 (short) | 6,000 × 32 (4× data) | Gain from more data |
|---|---|---|---|
| Transformer | 1.296 (3 seeds) | **1.086** (1.0824 / 1.0892) | −0.210 |
| L-4 (λ 0.01) | **1.219** (3 seeds) | 1.103 (1 seed) | −0.116 |

- **L-4 was more data-efficient** (6% better on little data) **but saturated earlier.** With 4× data the transformer passed it by 1.6%, the same pattern as at 9M.
- **Explanation:** a transformer reads every byte of its 256-byte window exactly. L compresses the past into a fixed-size state, and the L-4 memory write was purely additive, so new information piles onto old and blurs. This is the known capacity limit of linear-recurrent / fast-weight models.
- L kept its copy and long-context advantages at 4× data: exact 96% at 2,048 bytes, typos 90%, renamed 87%.
- **Fix:** L-5 → [[110 - L-5 Error-Correcting Memory Breaks the Ceiling]].

## Sustainability line
- At equal or better quality, L uses fewer params and 5–10% active FFN neurons.
- Measured energy per token (note 103): 2.2× lower at 256 context, 11× lower at 16K.
- Fewer active neurons mean less memory traffic: the main driver of power, cooling water and hardware turnover.

## Code
`lab-code-L/`: results_L1_*4Mc*.json · results_L1_*4M_long*.json · results_L1_L4_8L_s0.json · results_L4E2_L4E2_8L_s0.json · train_L4E2.py (now reads DM/NL) · train_L1.py (RESUME checkpoints, ACC gradient accumulation)
