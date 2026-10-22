---
tags: [L, L-5, delta-rule, results, scaling, breakthrough-candidate]
created: 2026-10-09
status: internal research record (confirmation seeds + 9M GPU test running)
---
# 110 - L-5: error-correcting memory breaks the ceiling

Previous: [[109 - Fixes Round Results at 4M]] (found that L-4 saturates with more data)

## What changed from L-4
1. **Error-correcting fast-weight memory (delta rule).** Brain analogue: error-driven synaptic plasticity.
   `S_t = a_t S_{t-1} + b_t (v_t − a_t S_{t-1} k_t) k_tᵀ` and `o_t = S_t q_t`.
   - Before writing, the memory reads what it already stores for this key and writes only the **difference**.
   - Old associations get corrected instead of piling up and blurring.
   - `b_t` is a learned write strength; `a_t` is learned forgetting.
   - Training uses an exact chunk-parallel form (unit-lower-triangular solve per 64-token chunk). It was verified against the token-by-token loop: **max error 4e-15**.
2. **Short delay line.** Brain analogue: dendritic/axonal delays. A depthwise causal convolution (kernel 4) on the memory's query/key/value gives exact access to the last 4 bytes. It starts as an identity.
3. **Kept from L-4:**
   - spiking diagonal neurons with sign-flipping (state tracking);
   - homeostasis;
   - hippocampus (exact recall);
   - gated trajectory replay.
4. **Energy budget:** λ 0.01.
5. **Cost per token stays constant.** No attention, no growing cache. Params +0.6%.
6. **Honest naming:** this is the "delta rule / DeltaNet" family of fast-weight memories (Schmidhuber 1992; Yang et al. 2024).

## Screening at 1M (3,000 × 16, λ 0.03; bits/byte)
| Model | Seeds | Bits/byte | s/step |
|---|---|---|---|
| Transformer | 3 | 1.466 | — |
| L-4 | 3 | 1.3467 | 0.92 |
| L-5 without delay line (delta rule only) | 1 | 1.3274 | 0.77 |
| **L-5 (delta + delay line, 4 memory blocks)** | **3** | **1.3215 / 1.3198 / 1.3197 → 1.320 ± 0.001** | 0.94 |
| L-5, 2 memory blocks | 1 | 1.3339 | 0.82 |
| L-5, 8 memory blocks | 1 | 1.3139 | 1.02 |
| L-5, 16 memory blocks | 1 | 1.3112 | 1.33 |

- Both parts help: the delta rule gives −1.4% and the delay line −0.6% more.
- Many small memory blocks beat a few big ones. 8 blocks is the speed/quality sweet spot.
- **L-5 is 10% better than the transformer at 1M.**

## 4M, normal budget (3,000 × 16, λ 0.01)
| Model | Bits/byte |
|---|---|
| Transformer (3 seeds) | 1.296 |
| L-4 (3 seeds) | 1.219 |
| **L-5 (1 seed)** | **1.1982 (−7.5% vs transformer)** |

## 4M, 4× data (6,000 × 32): the ceiling test
| Model | Params | Bits/byte | vs transformer |
|---|---|---|---|
| Transformer (2 seeds) | 4.35M | 1.0824 / 1.0892 → 1.086 | — |
| L-4 | 3.94M | 1.1031 | +1.6% (lost) |
| **L-5** | **3.96M** | **1.0675** | **−1.7% (wins)** |

- **The data-efficiency advantage no longer turns into a loss with more training.** L-5 gains 0.131 from 4× data, vs L-4's 0.116.
- **Copying still holds:** exact 95% at 2,048 bytes; typos 90%; renamed 87%.
- **Sparsity:** FFN 5–11% active; state 36–65% active.
- **Training speed on cloud CPU:** 4.5 s/step vs 2.5 for the transformer. This is a tooling gap: the C++ scan is not used for the delta memory.

## Running
- **Cloud, 4× data:** L-5 second seed, and L-5 with 8 memory blocks (about 8–9 h on 2 CPU cores).
- **Laptop RTX 3050:** **L-5 at 8.8M (8 memory blocks) vs transformer 9.2M**, 6,000 × 32 at lr 2e-3. This is the size where L-2 lost (note 107).

## Code
- `lab-code-L/l5_model.py`: DeltaMemory, delta_chunk (exact chunk form), delta_loop (reference), L5Model.
- `train_L1.py`: model kind `L5`, env CONV / HEADS / ACC.
- `results_L1_L5*.json`.
