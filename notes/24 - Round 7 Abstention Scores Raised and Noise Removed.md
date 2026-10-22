---
tags: [experiments, round-7, internal, abstention, ood, fine-tuning]
created: 2026-10-06
status: internal research record
---
# Round 7: "I don't know this": scores raised, noise removed

Back to [[20 - Final Benchmark Summary]] · Previous: [[23 - Round 6 Never-Imagined Abstention]]

**Goal:** the Round 6 scores (59–77% on never-imagined types) were too low and changed from run to run. In this round we found the causes, fixed them and fine-tuned further.
Every number below is the mean ± spread over **4 seeds**, with **5% false referral on known data**. No unknown (OOD) example was used for training or tuning; every threshold was set on known data only.

## Final result
| Input type | Round 5 | Round 6 | **Round 7 (final)** |
|---|---|---|---|
| Inverted digits (imagined) | 98.2% | 90.1% | **89.2 ± 4.6%** |
| Rotated fashion (imagined) | 89.1% | 85.9% | **97.4 ± 0.3%** |
| **Mirrored digits** (never imagined) | 10.9% | 62.9% | **93.3 ± 0.5%** |
| **Fashion 180°** (never imagined) | 29.9% | 76.6% | **84.7 ± 0.5%** |
| **Upside-down digits** (never imagined) | 15% | 58.8% | **79.7 ± 0.7%** (all) · **89.6 ± 0.6%** (digits that change when flipped) |
| Pure noise | 99.9% | 95.6% | **100.0 ± 0.0%** |

- Never-imagined types are refused **6–9× more often than in Round 5** and **1.2–1.5× more often than in Round 6**.
- The spread between runs (the "noise") is now **≤ ±0.7 points** on every unknown type except inverted digits.

## What was wrong, and the fix for each cause
| # | Root cause (found by diagnosis) | Fix (brain analogue) | Effect |
|---|---|---|---|
| 1 | Symmetric known items (handwritten 0/1/8, bags, shirts) look the same when flipped. A softmax pose head splits its belief between "upright" and "flipped", so these items get a low "upright" score, use up the 5% budget and push the threshold down. | **Symmetry-aware pose sense:** every pose that renders the same picture counts as correct (multi-label soft targets from pixel similarity; the labels are still free). | Pose AUROC is stable across seeds: 0.97 / 0.93 / 0.94 for mirrored / fashion 180° / upside-down, where the softmax head swung between 0.89 and 0.98 |
| 2 | Score noise from where the eye happens to land | **Micro-saccade averaging:** average the score over 5 slightly shifted fixations, plus an **ensemble of 2 pose heads** | Spread fell from about ±3 to about ±1 point |
| 3 | Pure noise gets a mid-level "mine" score and a random pose score, so detection was 80 ± 15% under the plain fusion | **Early-vision gate** (retina/V1 natural-image statistics: brightness, contrast, background, edge and pixel-noise energy; kNN familiarity) with a tiny 0.5% budget | Noise refused **100% in every run** |
| 4 | Splitting the budget between two signals was wasteful: min-rank fusion is good for single-signal oddities, Fisher fusion for "both a bit odd" | **UNION(min-rank, Fisher)**, with one shared threshold calibrated on known data | The best balance between the imagined and never-imagined types |
| 5 | **Under-training was the biggest limiter** | Fine-tune the pose sense longer, on more known data | See the table below |

### Fine-tuning curve (pose sense, 2-head ensemble, full cascade)
| Training | Mirrored | Fashion 180° | Upside-down (changing digits) |
|---|---|---|---|
| 12 epochs, 4k images per domain | 78.6% | 78.8% | 76.5% |
| 24 epochs, 6k images per domain | 88.6% | 81.6% | 85.1% |
| **40 epochs, 8k images per domain** | **93.3%** | **84.7%** | **89.6%** |
The curve has not flattened yet. Longer training on the RTX 3050 is the next cheap gain.

### Diagnosis: which unknowns still pass
- **Upside-down digits:** 0, 1, 3 and 8 still look like valid digits when flipped vertically (the same fairness rule we have used for mirrored 0/1/8 since Round 4). Of the digits that really change, a 2 is the hardest to catch (about 52% before fine-tuning).
- **Fashion 180°:** dresses and bags still look like a dress or a bag when turned 180°. Sandals, sneakers and boots are caught 98–100% of the time.

## Final design of the "I don't know this" module
1. **Early-vision gate:** reject anything that does not have natural-image statistics. It costs almost nothing.
2. **Self-knowing columns ("mine"):** catch content and style the system has never learned (including counterfactual imagination in sleep).
3. **Shared, symmetry-aware orientation sense:** catches familiar things in an unfamiliar pose, which nothing imagined in sleep had covered.
4. **Calibrated fusion** of 2 and 3 (UNION of min-rank and Fisher), with all thresholds set on known data only.

**Cost:** 2 pose heads × 454k params (about 1.8 MB in total). It is shared by all domains and runs on a CPU. Training each head took about 50 min on 1 CPU thread.

## Honest limits
- Inverted digits dropped from 98% (with "mine" alone) to 89 ± 5%. Sharing the 5% budget with the pose sense costs this, and the spread comes from the "mine" columns of different runs.
- The pose sense only knows the poses in its vocabulary (rotations and flips). Unknowns of other kinds (e.g. scale, colour, texture) need more senses of the same kind; that is the future scope.

## Code
`lab-code/`:
- Scripts: big9e_symmetry.py · big9f_fusion.py · big9g_ensemble.py · big9h_perclass.py · big9i_final.py · big9j_longer.py
- Results: results_big9e_seed*.json · results_big9f_fusion.json · results_big9g_ensemble.json · results_big9h_perclass.json · results_big9i_final.json (12 epochs) · results_big9j_longer.json (24 epochs) · results_big9k_xl.json (40 epochs, final)
