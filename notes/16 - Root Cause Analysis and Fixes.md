---
tags: [root-cause, experiments, integration, genome, round-2]
created: 2026-10-06
status: lab round 2 (toy scale, CPU, 3-5 seeds)
---
# Root Cause Analysis and Fixes (Round 2)

Back to [[00 - Concept and Roadmap]] · Round 1: [[07 - Experiment Results]] · ChatGPT's parallel work: [[13 - Extended Prototype Results]], [[15 - Temporal Stress Replication]] · Ideas: [[17 - Innovation Proposals]]

> Kakashi's instruction: don't patch and retry. Find **why** the two strongest features (all modules combined, DNA genome) failed, fix the root, and make them run. If something can't be fixed, explain why.

## Result in one line
**Both now run.** Integration was failing for **4 root causes** (one was a flaw in our benchmark); after fixing them, the full system **equals or beats a parameter-matched replay baseline with ~47% fewer parameters**. The genome had **3 root causes**; with learned "gene-expression" codes it **beats equal-size low-rank compression by up to 12 points**, and **genomic columns** add each new domain for ~1.1k parameters (7.6× smaller) with zero forgetting.

*Scale warning: toy CPU experiments (8×8 digits, synthetic domains). Mechanism evidence, not LLM-scale proof.*

---

## Part A: Integration ("all modules combined")

### Diagnosis (RC1)
Split the error into routing vs column quality using an **oracle router** (perfect routing):

| Test shift | System | Router accuracy | **Columns with perfect routing** |
|---|---|---|---|
| Clean | 95.1% | 95.4% | 99.6% |
| Noise | 54.5% | 57.9% | **94.5%** |
| Contrast | 55.1% | 55.8% | **98.8%** |
| Translate | 19.8% | 25.6% | 60.3% |

- Columns were **95–97% confident even on other domains' inputs**: they didn't know what they didn't know.

### The 4 root causes, and what fixed each
| # | Root cause | Evidence | Fix | Result |
|---|---|---|---|---|
| 1 | **Columns can't say "not mine"** (no self-knowledge). Theory agrees: routing accuracy is bounded by each module's out-of-distribution detection (Kim et al., NeurIPS 2022) | 95–97% confidence on foreign inputs | **Self-knowing columns**: extra NOT-MINE output, trained on neighbours' memories; old columns re-learn it in sleep when a new neighbour arrives | Beat replay baseline on clean (94.5 vs 91.4) and 4/5 shifts, incl. never-seen blur (72.4 vs 58.2), with 38% fewer params |
| 2 | **Dreams too strong destroy content** | Strength 1.0: contrast fell 66% → 30% | **Mild dreams** (strength 0.3): "weird but recognisable" | Contrast 66 → **77%**, salt-pepper 64 → 68%; beats baseline on every shift |
| 3 | **Sleep caused self-forgetting**: columns re-trained on a tiny memory at a high learning rate overwrote their own knowledge | Healing gave +0.2 pts (baseline +14) | **Protected sleep**: recalibration updates only the mine/not-mine head; healing uses a gentle rate + distillation anchor to pre-sleep self (synaptic consolidation) | Healing now works: noise +5 pts, blur +2.8 |
| 4 | **Benchmark flaw**: in split-digits, "domain" = digit pair, so routing **is** the whole task | Joint domain classifier on noisy: 64.7% ≈ joint digit classifier 64.5% | **Realistic benchmark**: domains differ in look AND task (normal/identify, rotated/mod-3, inverted/>4, mirrored/identify) | Router accuracy on noisy inputs 58% → **86%** |
| ✗ | *Hypothesis rejected:* scores of separately trained columns are on different scales | Platt calibration: no gain (64.9%) | — | Recorded as negative |
| — | *Not fixable by routing:* translation | Every model fails (14–28%) | Needs spatial structure (convolution / position-aware sensory layer) | Separate root cause: representation, not routing |

### Final integration result (realistic 4-domain benchmark, 5 seeds)
| System | Params | Clean | Noisy (healed) | Blur (never dreamt) |
|---|---|---|---|---|
| Naive sequential | 25k | 21.3% | 13.0% | 13.3% |
| Replay + same dreams + same healing | 25.3k | 83.9% | 57.2% | 61.9% |
| **Full system** (self-knowing columns + mild dreams + protected healing sleep) | **13.5k** | **84.9%** | **58.2%** | **63.2%** |

➡️ Equal-or-better with **~half the parameters**, and only one column active per input. Margins are within seed noise (SD ≈ 2–3 pts), so this is **"works and is efficient"**, not "wins big".

---

## Part B: DNA genome compression

### Diagnosis (RC2)
| Check | Finding |
|---|---|
| D1 Optimisation | Loss curves **bounce** (0.51 → 0.93 → 0.10). Stable runs of genome-16 reach **94–96%**; the old "58.7%" was an unstable run |
| D2 Do neuron IDs carry information? | Fitting a genome to trained weights: true IDs R² 0.355; **shuffled hidden IDs 0.352** (no difference); shuffled pixel IDs 0.279 |
| D3 Better than a trivial compressor? | Same budget: **low-rank explains 80%** of the weights vs genome 35% |

### The 3 root causes and fixes
| # | Root cause | Fix |
|---|---|---|
| 1 | Unstable training (high learning rate; every update rescales the whole layer) | lr 3e-3 + gradient clipping |
| 2 | **Hidden neuron IDs are meaningless labels.** In biology, identity = gene expression, which tracks function | **Learned per-neuron "gene-expression" codes** (cost ∝ neurons, not neurons²) + **one shared genome for all layers** |
| 3 | Genome less expressive than low-rank | Nonlinear genome over codes (with a bilinear term) |

### Results: learned-code genome vs equal-size low-rank (3 seeds)
| Hidden width | Genome d=2 | Low-rank | Genome d=4 | Low-rank | Compression |
|---|---|---|---|---|---|
| 128 | 73.5% | 73.9% | 89.7% | 92.4% | 11× / 6× |
| 512 | **82.7%** | 75.3% | **94.0%** | 89.4% | 14× / 7.5× |
| 2048 | **80.9%** | 69.1% | **92.7%** | 90.9% | 15× / 8× |
| (full network) | 97.9% | | | | 1× |

➡️ The genome's advantage **grows with layer width**, which is the right direction for LLM-size layers. Still below the full network.

### Genomic columns (fusion of genome + growing columns)
Genome learned with domain 1, then **frozen (DNA)**; each new domain column learns only its neurons' codes.
| Columns | Mean accuracy | Params per new column | Forgetting |
|---|---|---|---|
| Independent full columns | 94.1% | 8,645 | 0 |
| Shared frozen low-rank basis | 88.1% | 901 | 0 |
| **Genomic columns (d=8)** | **91.2%** | **1,133** | **0 by construction** |

### Learn big, sleep small (three-tier memory): 🟡 partial
Learn a domain with a big column, then distil it into genome codes during sleep using only memory + dreams:
| Memory per class | Slept genomic | (direct on full data) |
|---|---|---|
| 10 | 77.8% | 91.0% |
| 30 | 83.4% | 91.0% |
| 30 + imaginative mixup dreams | 81.5% ✗ | |
**Why it falls short:** a small memory doesn't cover the input space, so the teacher can't transfer its knowledge. **Workaround now:** compress before the data is discarded.

---

## What's still open
- Translation and spatial shifts need a position-aware sensory layer
- Error attribution: decide whether a mistake was the router's or the column's, and heal only the responsible part
- Scale up: real language model (Qwen3-0.6B + LoRA columns) on the RTX 3050
- Genome on one large LLM layer (4096 wide), where compression should pay off most

## Code and raw results
`lab-code/`: rc1_integration_diagnosis.py · rc1_fix_selfknowing_dreams.py · rc1_integration_v2.py · rc1_calibrated.py · rc1_real_domains.py (`python rc1_real_domains.py protected`) · rc2_genome_diagnosis.py · rc2_genome_fix.py · rc2_genomic_columns.py · rc2_learn_big_sleep_small.py · results_rc*.json
