---
tags: [L, L-4, fixes, fast-weights, effort-gate, routing, gpu-kernel]
created: 2026-10-08
status: internal research record
---
# 108 - L-4: implementing the fixes for L's weaknesses

Previous: [[107 - First Scale-Up to 9M on Laptop GPU]] (L lost on quality at 9M by about 0.9%) · Next: [[109 - Fixes Round Results at 4M]]

## Diagnosis that started this round
| Weak point | Fix tried | Status |
|---|---|---|
| 1. Working memory too small (one number per neuron) | **Fast-weight memory:** a small matrix per head, rewritten while reading (short-term synaptic plasticity). Fixed cost per token | ✅ Built, exact (chunked = token-by-token, error 2e-7) |
| 2. Hippocampus exact-only | Learned associative recall now comes from the fast-weight memory; the exact index is kept | ✅ (via 1) |
| 3. Spike counts too coarse | −8..8 test at 9M | ⏹ Superseded: the root cause was found elsewhere (note 109) |
| 4. Energy budget may hurt at scale | Tested at 4M (note 109) | ✅ λ 0.01 is the sweet spot |
| 5. Same effort for every token | Effort gate v1 and v2 | 🟡 v2 works but does not pay off (parked, note 109) |
| 6. Slow GPU training | **CuPy GPU kernel** (runtime-compiled, self-verified) | ✅ **Scan 152 ms → 5.5 ms (28× faster)** on the RTX 3050 |
| 8. Column misrouting | Margin toward the original column, calibrated on training passages | ❌ Calibrated margin = 0 (no help). Rolling routing helps C a little |

**Honest naming:** the fast-weight memory belongs mathematically to the "fast weights / linear attention" family. It is not softmax attention. It never looks back over past tokens, and its cost per token is fixed.

## Results at about 1M params (cloud CPU, 3,000 steps, bits/byte, lower is better)
| Model | Params | Seeds | Held-out code |
|---|---|---|---|
| Transformer | 1.23M | 3 | 1.466 ± 0.052 |
| L-2 (gated trajectory + homeostasis) | 1.18M | 3 | 1.350 ± 0.002 |
| **L-4 (+ fast-weight memory, slimmer feed-forward)** | **1.15M** | **3** | **1.3467 ± 0.0002** |

L-4 also keeps or improves copying:
- exact 95–96% up to 2,048 bytes;
- typos 89–92%;
- renamed variables 83–86%.

It is sparse: 37–38% of state neurons active in the first layer.

## Effort gate (think less on easy tokens)
| Version | Full depth | Gate 0.99 | Gate 0.95 | Gate 0.9 |
|---|---|---|---|---|
| v1 (skipped token is invisible to deeper layers) | 1.370 | 1.495 at 75% compute | 1.620 at 59% | 1.679 at 50% |
| **v2** (skipped token still writes into deeper memories + exit practice in training) | 1.385 | **1.392 at 86%** | 1.421 at 77% | 1.456 at 72% |

- **Root cause of v1's failure:** a recurrent memory that misses tokens loses information, and training never practised skipping. v2 fixes both. The cost of saving compute drops about 10× (14% saved for +0.007).
- **The gate does not pay off yet:** the deep supervision and exit practice raise the full-depth error from 1.347 to 1.385.

## L-3 routing (3rd seed, rank-64 columns)
- Average over Python, JavaScript and C: **1.255**; Python forgotten +0.025.
- C with rolling routing: 1.554.
- **3-seed mean, rank 64:** 1.271 / 1.269 / 1.255.
- The calibrated confidence margin was 0, so remaining misrouting needs a different fix (a learned router from surprise patterns).

## Results at about 4M params (cloud CPU, 3,000 × 16 × 256)
| Model | Params | Bits/byte | Exact copy 2,048 | Renamed 512 / 2,048 | FFN active |
|---|---|---|---|---|---|
| Transformer (2 seeds) | 4.35M | 1.2786 / 1.3147 | cannot | 62–66% at 120 only | dense |
| L-2 (1 seed) | 3.96M | 1.2311 | 95.2% | 83.3 / 86.6% | 3.5–6.7% |
| **L-4 (1 seed)** | **3.94M** | **1.2271** | 95.0% | **84.0 / 86.8%** | 3.9–7.2% |

More seeds and the full analysis: [[109 - Fixes Round Results at 4M]].

## GPU results from the laptop
- Transformer 9M, seed 2: **1.083** (seed 1: 1.080). The gap to L-2 9M (1.092) holds at about 0.9%.

## Code
`lab-code-L/`: l4_model.py (FastWeights, L4Model, L4EModel, L4E2Model) · cuscan.py · setup_cupy.py · train_L4E.py · train_L4E2.py · train_k8.py · results_L1_L4_s*.json · results_L4E*_*.json · results_L3_columns_r64_s2.json
