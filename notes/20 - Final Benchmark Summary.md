---
tags: [benchmarks, summary, status]
created: 2026-10-06
status: consolidated scoreboard after Rounds 1–8
---
# Final Benchmark Summary (as of 06 Oct 2026)

Back to [[00 - Concept and Roadmap]] · Details: [[07 - Experiment Results]] · [[16 - Root Cause Analysis and Fixes]] · [[18 - Round 3 Big Benchmarks]] · [[19 - Round 4 Failures Fixed]] · [[22 - Round 5 Partial Modules and Lightweight Proof]] · [[23 - Round 6 Never-Imagined Abstention]] · [[24 - Round 7 Abstention Scores Raised and Noise Removed]] · [[25 - Round 8 Language Continual Learning (Phase 2)]]

> **Status:** no symposium events. The project continues as long-term research.
> **Scale:** CPU only (2 cores), small models (0.2–2M params), datasets: MNIST, Fashion-MNIST (70k each), WikiText-2, Python source code, Wisconsin breast cancer (569 patients), Tiny Shakespeare (1.1M chars), synthetic reasoning tasks. Every comparison uses a baseline trained the same way; 2–10 seeds.

## Scoreboard: best result per module — **15 / 15 modules working**
| # | Module (brain analogue) | Benchmark | **Ours** | Baseline | Verdict |
|---|---|---|---|---|---|
| 1 | **Effort gate** (thalamus) | Multi-step reasoning (16-node pointer chasing) | **97.3% with 62.5% less compute** | Fixed 18 steps 97.3% | ✅ Strong |
| 2 | **V1 + saccade** (retina/visual cortex) | MNIST, objects moved 8 px | **91.1%**, 30k params | MLP 8.4%, 235k params | ✅ Strong |
| | | Fashion, moved 8 px | **69.5%** | MLP 14.9% | ✅ (rest = object cut off frame) |
| 3 | **Full integrated system** | 4 domains learned in sequence, shifted inputs, after healing | **86.7%** shifted / **92.9%** clean, **100k active params** | Replay (same size) 82.0% / 86.0%, 407k active | ✅ Win, 4× fewer active params |
| 4 | **Growing columns** (cortical columns) | 5 domains learned in sequence (no task hints) | **94.9%** | Replay 90.8% · naive 19.6% | ✅ |
| 5 | **Self-healing** (error monitor + protected sleep) | Real clinical: new hospital with equipment drift | **Hosp B 94.4 → 96.1%**, cancer detection **88.5 → 94.9%**, **Hosp A kept 96.4%** | Direct weight edits collapse to 65–74% | ✅ Strong |
| 6 | **Safe referral** (self-knowledge) | Breast cancer, defer ~9% least-confident cases | **98.8% (A) / 96.9% (B)** accuracy on kept cases | 96.2 / 94.4% without referral | ✅ |
| 7 | **"I don't know this"** (counterfactual imagination in sleep) | Refuse unseen domain types at 5% false referral | **98.8% / 92.4%** correctly refused | 0.1% / 63% before | ✅ imagined 89% / 97% · ✅ never-imagined (mirrored / upside-down / fashion 180°) **93% / 90% / 85%**, noise 100%, 4 seeds, spread ≤ ±0.7 ([[24 - Round 7 Abstention Scores Raised and Noise Removed]]) |
| 8 | **Genome** (DNA with learned gene codes) | Equal-size comparison, width 2048 | **92.7%** (d4) / **80.9%** (d2) | Low-rank 90.9% / 69.1% | ✅ Beats low-rank, gap grows with width |
| 9 | **Skills as low-rank "epigenetic" adapters** (replaced genome-coded skills) | Fashion on frozen V1 | **87.1% at 18× smaller** | Full column 87.5% · genome column 86.3% | ✅ |
| 10 | **Sleep compression** (best: natural-statistics lucid dreams) | Distil a skill from 200 memories | **69.8%** | Old lucid 57.5% · memory 67.6% · ceiling 74.8% | ✅ Best sleep method |
| 11 | **Spiking neurons** | Digits | **95.3%**, ~11% neurons firing, ~**3.7× less energy (estimate)** | ANN 97.2% | ✅ (energy saving needs neuromorphic hardware) |
| 12 | **Looped LM from scratch** | Tiny Shakespeare, bits/char ↓ | **2.433 at 233k params** | Same params + same compute: 2.509 · 4× params baseline: 2.311 | ✅ Parameter-efficient |
| 13 | **Early-exit LM** (think when needed) | Tiny Shakespeare, 8 layers | **32% compute saved** at 2.371 · **47%** at 2.425 | Full depth 2.345 | ✅ Savings grow with depth |
| 14 | **Weak-feedback healing** (right/wrong only) | 4-domain system | **85.9%** | Naive 85.5% · full-label ceiling ~86.7% | ✅ At ceiling |
| 15 | **Continual-learning language model** (childhood trunk + growing columns + surprise routing) | Shakespeare → Wikipedia → Python code, learned in sequence | **2.06 bits/byte avg, forgetting +0.015**, routing 99%, new-domain detection AUROC 0.92 / 0.97 | Standard 3.06 (forgets +1.59) · replay 2.55 · joint (all data at once) 2.06 | ✅ Matches joint training without old data ([[25 - Round 8 Language Continual Learning (Phase 2)]]) |

## Design principles we established (each backed by a test)
1. **Self-knowledge:** every module learns "mine / not mine"; routing depends on it.
2. **Protected sleep:** consolidate without disturbing existing knowledge; never edit weights directly.
3. **Dreams must look like life:** mild, natural-statistics dreams help; adversarial or extreme dreams hurt.
4. **Developmental order:** sensory layers mature first, then higher areas and the genome.
5. **Counterfactual imagination:** imagined style/content combinations teach what *isn't* mine.
6. **Negative evidence is safe, ownership needs confirmation** (blame assignment).
7. **Balanced healing:** heal from all confirmed cases, class-balanced, not only from mistakes.
8. **Loop for reasoning and parameter efficiency; exit early for compute efficiency.**
9. **Modularity pays when domains are more separable than their contents** (routing-headroom test).

## Still open (next research)
- Never-imagined abstention: longer pose-sense training on GPU (curve not flat yet); add more senses (scale, colour, texture) for unknowns outside the pose vocabulary
- Deeper LMs on WikiText-2 on the RTX 3050 (gpu-kit ready)
- Genome-LoRA for language models; brain-style energy budget across modules
- Medical image data (MedMNIST / chest X-ray), downloaded on the laptop
- Real energy measurements on GPU / neuromorphic / FPGA
- **Compute:** move to the RTX 3050 or a free Kaggle GPU (CPU runs take hours)
