---
tags: [experiments, results, validation]
created: 2026-10-06
status: lab round 1 (toy scale)
---
# Experiment Results: Lab Round 1

Back to [[00 - Concept and Roadmap]] · Plan: [[06 - Validation Gaps]] · Papers: [[08 - Paper Deep Reads]] · Code: `lab-code/`

> **Scale warning:** these are **toy experiments** on a CPU (8×8 handwritten digits, 1,797 images; synthetic graph tasks), 3–5 random seeds each. They test whether each *idea* works mechanically. They do **not** prove anything at LLM scale.

## Scoreboard
| # | Idea tested | Verdict | Key number |
|---|---|---|---|
| E1 | Auto-growing domain columns (no forgetting) | ✅ **Works** | 94.9% vs 19.6% naive, 90.8% replay |
| E2 | Self-healing (memory → sleep) | ✅ **Works, with rules** | Direct weight edits destroy the model (97% → 52%); sleep consolidation fixes 86% of mistakes, +11 pts on new inputs, −3 pts old |
| E2 | Intrinsic self-correction (no feedback) | ❌ **Fails** | Breaks more answers than it fixes (6.8% vs 5.8%) |
| E3 | Effort gate (think when needed) | ✅ **Works** on reasoning; ❌ useless on perception | **Same 97.3% accuracy with 62.5% less compute**; steps track difficulty (r ≈ 0.79) |
| E4 | Spiking neurons | ✅ **Works** | 95.3% vs 97.2%, ~11% neurons firing, ~3.7× less energy (*estimate, neuromorphic only*) |
| E5 | Genome (DNA) compression | 🟡 **Weak / unstable** | 92.8% at 3.8× compression (best); other sizes fail |
| E6 | **Integration of all modules** | ❌ **Fails in v1** | Replay baseline 69.7% vs ours 62.5% on shifted data; the router is the weak link |

---

## E1: Auto-growing domain columns
**Setup:** 5 domains arrive one by one (digit pairs 0/1, 2/3, …). At test time the model is **not told** the domain (the hardest setting, called class-incremental). Each column = small classifier + autoencoder "novelty sensor". New column only when data looks novel to every existing column. 5 seeds.

| Method | Avg accuracy | First domain at end | Params |
|---|---|---|---|
| Joint training (upper bound, sees all data) | 97.2% | 98.9% | 9.0k |
| Naive fine-tuning | 19.6% | **0%** (total forgetting) | 9.0k |
| EWC (classic anti-forgetting) | 19.8% | 0% | 9.0k |
| Replay (stores 20 images per class) | 90.8% | 87.0% | 9.0k |
| **Growing columns (ours)** | **94.9%** | **93.4%** | 24.1k total, 12.9k active |

- Growth decisions: grew on **25/25** new domains; **0/5** false growth when an old domain was shown again.
- Router picked the right column 95.3% of the time.
- ⚠️ Costs: 2.7× the parameters, and routing cost grows with every column (all novelty sensors run on each input).
- ⚠️ In later runs (E6) the sensor **missed 1 in 5 new domains**, so growth needs a backup signal.

## E2: Self-healing (Kakashi's idea)
**Setup:** model trained on clean digits, deployed on noisy digits (stream accuracy 56%). Mistakes get correct labels from **external feedback** (like a user correction or a test result). 5 seeds.

| Healing method | Reliability (fixed stays fixed) | Generalisation (new noisy inputs) | Locality (old clean skill) |
|---|---|---|---|
| No healing | 0% | 54.4% | 97.3% |
| **Direct weight edits** (fix each mistake in the weights) | 38.6% | 37.2% ↓ | **52.3%** 💥 |
| **Hippocampus memory** (store fixes + router, threshold 1%) | **100%** | 53.6% | 96.2% |
| **Sleep consolidation** (memory + small replay → weights) | 86.0% | **65.2%** ↑ | 94.4% |
| Intrinsic self-correction (no feedback, switch low-confidence answers) | — | net **−0.98 pts** | — |

**Lessons (they match the papers exactly):**
1. Fixing mistakes **directly in the weights destroys the model** (same as ROME/MEMIT collapsing after 100s of edits).
2. A **side memory** gives perfect, instant fixes with almost no damage (like WISE/GRACE), but doesn't generalise.
3. **Sleep consolidation** turns stored fixes into general skill (+11 pts on new inputs).
4. A model **can't reliably judge its own mistakes** without an outside signal: it flagged errors with only 66% precision / 30% recall and made things worse (same as Huang et al., ICLR 2024).

➡️ **Self-healing design rule:** *detect with an external signal → write to hippocampus instantly → consolidate during sleep → never edit weights directly.*

## E3: Effort gate (Think-When-Needed)
- **v1 (noisy digits): failed.** Looping didn't help perception (lost pixels can't be "thought back"), and the halting head became overconfident from overfitting. *Lesson: effort gates help reasoning, not perception, and halting must be trained on fresh data.*
- **v2/v3 bugs caught:** a too-easy task (root identifiable in one step), then a generic network that couldn't learn "follow the pointer".
- **Final (looped graph message-passing block, 12.7k params):** follow a pointer k times in a random 16-node graph (k = difficulty, 0–15). 3 seeds:

| Model | Accuracy | Steps |
|---|---|---|
| Fixed 1 step | 33.6% | 1 |
| Fixed 8 steps | 72.7% | 8 |
| Fixed 18 steps | 97.3% | 18 |
| **Effort gate (thr 0.95)** | **97.3%** | **6.7 avg (−62.5%)** |

Steps used rise almost linearly with difficulty (1 step for k=0–1, ~13 for k=14), correlation ≈ 0.79.

![[assets/e3_effort_gate.png]]

➡️ **This is the pilot result for the symposium paper** ([[Think-When-Needed Paper]]).

## E4: Spiking neurons (snnTorch LIF, 8 timesteps)
| Sparsity penalty | Accuracy | Neurons firing per step | Est. energy |
|---|---|---|---|
| Normal ANN | 97.2% | 58% active | 40.6 nJ |
| SNN, none | 95.9% | 28% | 16.5 nJ |
| SNN, 0.02 | 95.7% | 19% | 13.3 nJ |
| **SNN, 0.1** | **95.3%** | **11%** | **10.9 nJ (3.7× less)** |
| SNN, 0.3 | 10.2% 💥 | 0% (network died) | — |

⚠️ Energy uses standard 45nm estimates (multiply 4.6 pJ vs add 0.9 pJ). The saving is real **only on neuromorphic hardware**. On a GPU, spiking is *slower and costlier* (8 timesteps). Too much sparsity kills the network.

## E5: Genome (DNA) compression
| Version | Compression | Accuracy (full net 97.5%) |
|---|---|---|
| v1 (tanh genome, 3 layers) | 2–25× | ~10–48% ❌ (vanishing weights; fixed with "homeostatic" scaling, still weak) |
| **v2 (Fourier IDs, 2 layers), genome 8** | **3.8×** | **92.8%** |
| v2, genome 16 | 1.8× | 58.7% (unstable) |
| v2, genome 32 | 0.8× | 84.4% |

Paper (PNAS 2024) reached 322× on MNIST because compression grows with network size. **Unproven for language models; high risk; needs proper work.**

## E6: Integration (the most important test)
5 domains learned, then the world shifts (noisy inputs) and mistakes are healed. Compared with a parameter-matched replay model getting the **same healing data**. 5 seeds.

| System | Clean after | Shifted after |
|---|---|---|
| Replay baseline, no healing | 91.6% | 58.0% |
| **Replay baseline + healing** | 92.9% | **69.7%** 🏆 |
| Columns + healing (autoencoder router) | 85.1% | 54.0% |
| Columns + healing + **learned, self-healing router** | **93.4%** | 62.5% |
| Same + soft routing | 93.1% | 62.9% |

**Diagnosis:** the "thalamus" (router) sends only 54–66% of shifted inputs to the right column, so routing errors compound. **A monolithic model with replay generalises better under shift at this scale.**

➡️ **Biggest open problem found: robust routing under distribution shift.** Ideas to test next: a shared slow "cortex" trunk under the columns, HiCL-style sparse codes + prototypes, routing that heals during sleep with more data, and routing by meaning (embeddings) instead of raw input.

---

## What this means for the blueprint
| Module | Status after Round 1 |
|---|---|
| Domain columns | Keep: strong on clean continual learning; **fix routing** |
| Hippocampus + sleep (self-healing) | Keep: **validated design rule** |
| Effort gate | Keep: **validated on reasoning** (symposium-ready) |
| Spiking | Keep, but only pays off on neuromorphic hardware |
| Genome | Park as long-term research (high risk) |
| Self-correction without feedback | **Drop**: always pair self-healing with an external signal (tests, users, verifiers, tools) |
| Integration | ❌ **Not yet working.** This is the next research priority |

## Next experiments (Round 2)
- [ ] Router robustness: shared trunk + column heads; sparse-code prototype routing (HiCL)
- [ ] Scale E1/E2 to a real small language model (e.g. Qwen3-0.6B + LoRA columns) on the RTX 3050
- [ ] Effort gate inside a small language model (layer-loop + halting), measured in real joules (nvidia-smi)
- [ ] Spiking columns on a real language task (snnTorch on GPU)
- [ ] Genome: try it on one large layer (where compression should pay off)
