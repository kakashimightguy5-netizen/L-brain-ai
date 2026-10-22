---
tags: [L, L-2, results, flaws-fixed, parallel-training, calibration, memory]
created: 2026-10-07
status: internal research record
---
# 105 - L-2 candidate, and the remaining flaws fixed

Previous: [[104 - L-1 Results Limits Worked On]] · [[103 - L Real Energy Measurement on Laptop]]

All runs:
- data: CPython corpus (38 MB train / 2.1 MB test, split by file);
- 3,000 steps, CPU;
- comparison: a same-size transformer (1.466 ± 0.052 bits/byte, 3 seeds).

## Flaw by flaw
| Flaw (from notes 102–104) | Fix | Result |
|---|---|---|
| Spike rounding breaks long state tracking | **Homeostatic gain control:** the spike threshold adapts to the layer's current activity (brain-like adaptive threshold) | Parity at 16× training length: **64% → 87%** (3 seeds: 94 / 100 / 68%). Language quality unchanged (1.351) ✅ |
| Trajectory replay cost quality (1.377) | **Gated replay:** a "still on track?" gate driven by the last-recall-was-right signal scales the replay | **1.3496 ± 0.0018 (3 seeds)**, the best L so far, and the fuzzy-recall gains are kept ✅ |
| Layer separation only simulated | **Real two-process training:** lower cortex (layers 1–2) and upper cortex (layers 3–4) as separate OS processes. Activity flows up, nothing flows back | **0.38 s/step vs 0.67 single-process (1.8× faster on 2 cores).** Quality 1.409, still better than the transformer. The upper process waited only 8% of the time ✅ |
| Hippocampus index grows with text | **Capped index with LRU forgetting** (fixed memory) | Copy from 32K bytes back: **88% at 192 KB fixed** and 85% at 48 KB (vs 93% uncapped, 1.5 MB) ✅ |
| Self-knowledge untested | **Calibration test** (Problem Atlas category 3) | L knows better when it may be wrong (table below) ✅ |
| Single-seed ablations | 3 seeds for the L-2 candidate | ± 0.0018 ✅ (other ablations: laptop GPU queue) |

## L-2 candidate = L-1 + homeostatic spikes + gated trajectory replay (3 seeds)
| | Transformer | L-1 | **L-2 candidate** |
|---|---|---|---|
| Held-out code, bits/byte | 1.466 ± 0.052 | 1.353 ± 0.001 | **1.350 ± 0.002** |
| Exact copy 120 / 512 / 2,048 bytes | 77% / — / — | 95 / 96 / 95% | 94 / 96 / 95% |
| Copy with 5% typos, 512 / 2,048 | — | 89 / 86% | **92 / 90%** |
| Copy with renamed variables, 512 / 2,048 | — | 81 / 85% | **83 / 86%** |
| Same, trajectory switched off at test time (typos / renamed, 512) | — | — | 85% / 77% |
| Active state / feed-forward neurons | dense | 51 / 10% | 50 / 10% |

## Self-knowledge: does the model know when it is unsure? (3 seeds each)
| | Accuracy (next byte) | Calibration error (ECE, lower is better) | Accuracy on its 50% most confident answers | AUROC confidence: right vs wrong |
|---|---|---|---|---|
| Transformer | 70.6–72.7% | 0.0076–0.0097 | 96.6–97.6% | 0.901–0.907 |
| **L-1** | **73.7–73.8%** | **0.0028–0.0044** | **98.3%** | **0.915–0.916** |

## Fixed-size hippocampus (L-1, exact copy accuracy)
| Copy distance | No cap | Cap 32K entries | Cap 8K (192 KB) | Cap 2K (48 KB) |
|---|---|---|---|---|
| 2,048 bytes | 95.4% | 95.4% | 95.4% | 95.4% |
| 8,192 bytes | 93.4% | 93.4% | 93.4% | 79.6% |
| 32,768 bytes | 92.6% (1.5 MB) | 92.6% (768 KB) | 87.8% | 85.0% |

## Laptop GPU runner (prepared)
`D:\AI\SpikingBrain-study\L-architecture\runner\START_L_RUNNER.ps1` and the queued jobs were prepared, but the laptop link dropped before they could be copied. They will be copied when it reconnects.

## Still open
- Scale-up to 10–50M params (needs the laptop GPU).
- Energy of L-1/L-2 measured on the laptop.
- Real growth of skill columns for new programming languages (L-3).
- A training protocol for abstention and asking (Problem Atlas category 3).

## Code
`lab-code-L/`: parallel_demo.py · eval_extra.py · l1_model.py (gated trajectory, homeostasis) · results_parallel_demo.json · results_eval_extra.json · results_L1_L2cand_s*.json
