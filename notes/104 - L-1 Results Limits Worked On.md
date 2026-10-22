---
tags: [L, L-1, results, limits, sustainability, decoupled-training]
created: 2026-10-07
status: internal research record
---
# 104 - L-1: working on the limits of L-0

Previous: [[102 - L-0 Results First Non-Transformer Model]] · [[103 - L Real Energy Measurement on Laptop]] · [[101 - Training Without GPU Clusters Plan]]

## Setup
- **Data:** a 20× bigger corpus (CPython `Lib`, 38 MB train / 2.1 MB test). Train and test are split **by file**, so no test code is seen in training.
- **Training:** 3,000 steps of 16×256 bytes, 1 CPU thread per run.
- **Models:** L-1 (1.22M params) vs a transformer (1.23M params) trained identically, **3 seeds each**. Ablations use 1 seed unless noted.
- **Copy tests:** a held-out code passage is shown, then shown again. The second copy is either:
  - exact;
  - with 5% typos;
  - with variables **renamed**.

  Scoring counts only the bytes that did not change.

## Limit-by-limit outcome
| L-0 limit | What was done | Outcome |
|---|---|---|
| Slow training (Python scan) | Compiled C++/OpenMP scan with an exact hand-written backward pass (zero error vs the loop) | Scan 12× faster; **L-1 0.64 s/step vs transformer 0.61** (L-0 was 2.1) ✅ |
| Small data / possibly undertrained transformer | 20× data, file-disjoint split, 3 seeds | L still wins clearly (below) ✅ |
| Energy only estimated | Real laptop measurement kit | **Measured: 2.2× less energy/token at 256 context, 11× at 16K** (note 103) ✅ |
| State neurons not sparse (70–80% active) | Spike-rate energy budget | 51% active at medium setting, **34% at strong setting**, small quality cost ✅ |
| Exact-only memory (no fuzzy recall) | (a) LSH brain-state hashing; (b) **trajectory replay** | (a) gave no gain → dropped. (b) **+6.6 points on typos, +5.2 on renamed code** ✅ |
| Layer dependency (decoupled cost 0.10) | Blocks of 2 layers instead of 1 | Gap shrinks to **0.04 bits/byte** ✅ (blocks of 1: 0.11) |
| Parity fades at 16× length | Diagnosis: L without spikes | Without spikes: 100% / 99–100% / 71–100% → spike rounding is part of the cause 🟡 (fix next) |

## Main results (3 seeds; bits/byte, lower is better)
| Model | Held-out code | Exact copy 120 / 512 / 2,048 bytes | Typos 5% (512 / 2,048) | Renamed (512 / 2,048) |
|---|---|---|---|---|
| Transformer | 1.466 ± 0.052 | 77% / cannot / cannot | 70% at 120 only | 63% at 120 only |
| **L-1** | **1.353 ± 0.001** | **95% / 96% / 95%** | 89% / 86% | 81% / 85% |
| **L-1 + trajectory replay** (2 seeds) | 1.377 ± 0.004 | **94% / 96% / 95%** | **92% / 90%** | **83% / 86%** |

"Cannot" means the text is longer than the transformer's 256-byte context. L-1 is **8% better on quality** and **50× more stable across seeds** (± 0.001 vs ± 0.052).

### Ablations (1 seed each)
| Variant | Bits/byte | State / FFN neurons active |
|---|---|---|
| No energy budget | 1.338 | 78% / 15% |
| Medium energy budget (main) | 1.353 | 51% / 10% |
| Strong energy budget | 1.381 | **34% / 6%** |
| Without LSH fuzzy memory | 1.347 | 51% / 9% |
| Trajectory replay ON vs OFF at test time (same model): typos 512 | 92% vs 85% | — |
| Trajectory replay ON vs OFF: renamed 512 | 83% vs 77% | — |

### Layer separation (training without the full layer dependency)
| Training | Bits/byte | Gap to end-to-end |
|---|---|---|
| End-to-end (3 seeds) | 1.353 | — |
| **Blocks of 2 layers** (2 separate machines possible) | **1.396** | **+0.043 (3%)** |
| Blocks of 1 layer (fully local) | 1.462 | +0.109 |
| Transformer, end-to-end | 1.466 | — |

Even fully local L (every layer on its own) equals an end-to-end transformer. Training in pairs of layers costs only 3%.

## Honest limits
- **Still small:** about 1.2M params, 38 MB of data, 3,000 steps. The scale-up is needed next (laptop RTX 3050 / several CPUs).
- **Trajectory replay costs a little:** it improves fuzzy recall, but held-out code quality is 1.377 vs 1.353. Next: let it run only when trusted (gate the input embedding too).
- **Spike rounding hurts long state tracking:** a fix is needed (e.g. finer spike counts on the state, or a stochastic spike surrogate).
- **Ablations are single seeds.** Energy was measured for L-0, not yet for L-1.

## Next
1. Parity fix for spikes; trajectory gating; repeat ablations with 3 seeds.
2. Real parallel training of layer pairs in separate processes (the demonstration for training without GPU clusters).
3. Scale L to 10–50M params; measure L-1 energy on the laptop.

## Code
- Vault: `lab-code-L/` (l1_model.py · lscan.py · train_L1.py · results_L1_*.json).
- Laptop: `D:\AI\SpikingBrain-study\L-architecture\`.
