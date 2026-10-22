---
tags: [experiments, benchmarks, round-3, internal]
created: 2026-10-06
status: internal research record (not for submission)
---
# Round 3: Big Benchmarks, Open Problems Closed, From-Scratch LM

Back to [[16 - Root Cause Analysis and Fixes]] · [[17 - Innovation Proposals]]

> **Internal research note.** It records everything, including what didn't work, so we know where to go next. The submission packets in `20 - Symposiums 2026` contain only scope, achievements and future scope.

**Data used (real, full-size):** MNIST (70k), Fashion-MNIST (70k), Wisconsin Diagnostic Breast Cancer (569 patients), Tiny Shakespeare (1.1M characters). Data hosts like Hugging Face and Zenodo are blocked in the workspace, so the data came from GitHub mirrors and scikit-learn. CPU only (2 cores).

---

## 1. Open problem: shifted images ✅ CLOSED
**Root cause:** a fully connected sensory layer gives every pixel its own weight, so a moved object lands on weights that never learned it.
**Fix (brain):** a **V1 retina layer** (shared local receptive fields + complex-cell pooling) plus a **saccade** (fixate on the object, i.e. re-centre on its centre of mass).

| MNIST: shift (px) → | 0 | 2 | 4 | 6 | 8 | Params |
|---|---|---|---|---|---|---|
| MLP | 96.7 | 78.5 | 30.3 | 10.6 | 8.4 | 235k |
| MLP + shifted rehearsal | 95.9 | 95.0 | 90.4 | 63.3 | 25.8 | 235k |
| MLP + saccade | 96.6 | 96.6 | 95.7 | 92.6 | 84.0 | 235k |
| V1 | 97.8 | 97.3 | 96.4 | 93.3 | 84.6 | 30k |
| **V1 + saccade** | **97.7** | **97.7** | **97.6** | **96.2** | **91.1** | **30k** |

Fashion-MNIST, V1 + saccade: 86.3 / 84.5 / 81.2 / 76.9 / 69.5 (MLP at 8 px: 14.9).
**The remaining Fashion drop is explained:** an 8-px shift pushes **20% of a clothing item out of the frame** (digits 10%). The information is gone from the image.

## 2. Open problem: "learn big, sleep small" ✅ ROOT CAUSES FOUND, mostly closed
| Step | Finding |
|---|---|
| Small scale (Round 2) | Looked like a memory-coverage problem |
| Real scale, raw pixels | Coverage costs only 2–4 pts. **Real cause: a genome frozen on raw pixels transfers poorly** (Fashion 70.7% vs big column 87.3%) |
| V1 + genome trained jointly | **Failed (52%).** Cause: the genome bottleneck starved V1 of learning signal |
| **Developmental order** (V1 matures first → genome on normalised V1 features) | **MNIST genomic column 98.7%** |

Fashion (genome learned on MNIST, then frozen):
| Column | Params | Accuracy |
|---|---|---|
| Big column (awake) | 199k | 87.5% |
| Genomic, direct | **4.4k (45× smaller)** | 75.7% |
| Genomic, slept from full data | 4.4k | 75.2% (distillation is lossless) |
| Genomic, slept from 200 images + dreams | 4.4k | 69.6% |
| Genomic, generative memory (same storage) | 4.4k | 69.3% |
- Remaining gaps: **6 pts from memory coverage**, and a **capacity/compression trade-off** (45×) that's set by code size.
- Lucid (teacher-refined) dreams **hurt** (56%).

## 3. Big integrated system ✅ WIN
4 domains learned in sequence: MNIST digits, Fashion items, rotated MNIST parity, inverted Fashion groups. 12k train images each, 3 seeds. Shared V1 + saccade front end. Self-knowing columns with protected sleep and mild dreams. Healing on a noisy, shifting stream.

| After healing | Clean | Shifted (noise + 4 px) | Active params / input | Total params |
|---|---|---|---|---|
| Replay baseline (same front end, memory, dreams, healing) | 86.0% | 82.0% (±0.8) | 407k | 407k |
| **Full system** | **92.9%** | **86.7% (±0.3)** | **100k** | 397k |
- Routing on shifted inputs: 87% before healing → **98%** after.
- **Blame-aware healing = blind healing** (86.3 vs 86.7). No gain once sleep has fixed routing.

## 4. Safe referral / abstention
| At 5% referral of known cases | Refused correctly (before → after counterfactual sleep) |
|---|---|
| Inverted digits (imagined type) | 0.1% → **98.8%** |
| Rotated fashion (imagined type) | 63% → **92.4%** |
| Mirrored digits (never imagined) | 11.6% → 14.6% |
| Fashion 180° (never imagined) | 46% → 43% |
| Pure noise | 100% → 100% |
- **Root cause of the failure:** shortcut self-knowledge. Columns learned "my style" (e.g. inverted) instead of "my content".
- **Fix: counterfactual imagination** (known styles × other known contents become "not mine" during sleep). It works for imagined combinations.
- Never-imagined transforms remain open; part of this is genuinely ambiguous (0, 1 and 8 look the same mirrored).

## 5. Healthcare: real clinical data (breast cancer, 10 seeds)
Hospital A trains the model; Hospital B has equipment-calibration drift; clinicians confirm cases.
| | Hosp A acc | Hosp B acc | B sensitivity | B specificity |
|---|---|---|---|---|
| Trained at A | 96.2% | 94.4% | 88.5% | 97.9% |
| Direct weight edits on B mistakes | 73.8% 💥 | 64.8% 💥 | — | — |
| Protected sleep (corrections only) | 95.9% | 89.5% | 96.6% | 85.3% |
| **Protected sleep, class-balanced, all confirmed cases** | **96.4%** | **96.1%** | **94.9%** | **96.8%** |
- **Safe referral:** deferring ~9% of least-confident cases gives **98.8% (A) / 96.9% (B)** accuracy on kept cases.
- **Root cause of the middle row:** corrections were almost all missed cancers, so healing became biased toward "malignant". Fixed with class balance + all confirmed cases.

## 6. Language model trained FROM SCRATCH (Tiny Shakespeare, char-level, 2,500 steps)
| Model | Params | Compute / token | Val bits per char |
|---|---|---|---|
| Standard transformer (4 layers) | 0.83M | 1× | **2.311** |
| Standard, 4× wider | 3.23M | 4× | 2.256 |
| Looped single block + gate (fixed 3 loops) | 0.86M | 3× | 2.317 |
| Looped + gate (adaptive 0.9) | 0.86M | 3.6× | 2.319 |
| **Early-exit (4 distinct layers) + consistency gate** | 0.83M | 1× → **0.8×** | 2.337 full / **2.366 at 20% saved** / 2.406 at 32% saved |
- **Looping doesn't pay for character-level language** (pattern recall), the same root cause as in perception. It does pay for multi-step reasoning (62.5% saved in E3).
- **Early exit** gives a smooth quality/compute trade-off. Small at 4 layers; expected to grow with depth (CALM).
- The from-scratch models generate Shakespeare-style text (samples in results_lm_from_scratch.json).

---

## Next experiments (Round 4)
1. Train a **deeper** early-exit LM (8–12 layers) on WikiText-2: does skippable depth grow with model depth?
2. **Genome-LoRA**: generate an LM's skill adapters from the genome
3. Abstention for never-imagined transforms: content-based familiarity (prototype distance in V1 feature space)
4. **Metabolic energy budget** across gate + columns + spikes
5. Medical images (MedMNIST / chest X-ray), downloaded on Kakashi's laptop
6. Larger compute: Kaggle GPU or the RTX 3050 (the device shell is currently unavailable)

## Code and raw results
`lab-code/`: big_common.py · big1_translation.py · big2_sleep_small.py · big2b_v1_genomic.py (failed joint version, kept for the record) · big2c_v1_genomic_staged.py · big3_integration.py · big3b_abstention.py · big3c_counterfactual.py · health1_breast_cancer.py · health1b_balanced.py · lm_from_scratch.py · lm_early_exit.py · results_big*.json · results_health*.json · results_lm*.json
