---
tags: [experiments, round-6, internal, abstention, ood]
created: 2026-10-06
status: internal research record
---
# Round 6: "I don't know this" for never-imagined inputs

Back to [[20 - Final Benchmark Summary]] · [[22 - Round 5 Partial Modules and Lightweight Proof]]

**Goal:** refuse input types that the system never saw and never imagined during sleep (mirrored digits, upside-down digits, clothes rotated 180°). It must keep refusing the imagined types and must not refer more than 5% of normal inputs.

## Idea: an orientation sense ("which way is up")
The brain knows how familiar things normally look; an upside-down face feels wrong immediately. We gave the model the same sense.
- A small **canonical-view head** sees known images in different poses (upright, rotated 90/180/270°, mirrored, flipped) and learns to say which pose it is seeing.
- The pose labels come for free (self-supervised). It never sees any of the test unknowns.
- At test time we use **P(upright)**: a low value means "this is not how the things I know look", so the input is referred to a human.

## Step 1: one head per column (BIG-9). Modest gain.
Mirrored digits went from 18% to 27% refused, and clothes at 180° from 31% to 44%.

## Step 2: root-cause diagnosis (BIG-9b)
| AUROC | Mirrored digits | Fashion 180° | Upside-down digits |
|---|---|---|---|
| "mine" score | 0.63 | 0.76 | 0.46 |
| Head of the column that **claimed** the input | 0.81 | 0.87 | 0.66 |
| Head of the column that **truly owns** that content | **0.98** | **0.96** | **0.97** |

**Root cause:** a strange input is routed to the wrong column, so the wrong expert judges it. The orientation knowledge itself is good. A secondary cause was that the heads were under-trained.

## Step 3: fixes compared (BIG-9c, 2 seeds, 12 epochs)
| Way of judging (+ mine, OR rule, 5% false referral) | Mirrored | Fashion 180° | Upside-down |
|---|---|---|---|
| A. Claiming column's head | 41.7% | 51.8% | 23.3% |
| B. Any column says "upright" | 39.1% | 27.8% | 30.4% |
| C. Most confident head judges | 15.5% | 24.7% | 15.1% |
| **D. One shared orientation sense for all domains** | **46.2%** | **63.1%** | **49.9%** |

The routing problem disappears when no routing is needed. The brain has one shared system for "which way is up", and it does not keep a separate one per skill.

## Step 4: refined shared sense (BIG-9d, 2 seeds): final result
The shared head was made wider (conv 32→64, fc 128) and trained for 8 epochs on all known data. Its score is fused with "mine" by rank.

| At 5% false referral on known data | Inverted digits (imagined) | Rotated fashion (imagined) | **Mirrored digits** | **Fashion 180°** | **Upside-down digits** | Noise |
|---|---|---|---|---|---|---|
| Round 5 best ("mine" only) | 98.2% | 89.1% | 10.9–18.2% | 29.9–31.2% | 15.2% | 99.9% |
| **Ours: mine + shared orientation sense** | **90.1%** | **85.9%** | **62.9%** | **76.6%** | **58.8%** | **95.6%** |
| AUROC of the orientation sense alone | — | 0.94 | **0.94** | **0.95** | **0.92** | — |

- Never-imagined refusals improved **3.5× / 2.5× / 3.9×** over Round 5.
- Imagined types still have about 86–90% refused, and noise 96%.
- The extra cost is one small shared head of about 0.4M params, for all domains together.

## Honest limits
- The sense only catches changes that are in its **pose vocabulary**. In the fair control where flips are left out of the vocabulary, mirrored digits drop back to about 28% and upside-down digits to about 18%. Clothes at 180° stay at about 68%, because rotations are still in the vocabulary.
- A gap remains between AUROC 0.93 (shared sense) and 0.97 (the true owner judging).
- **Future scope:**
  - Add more self-supervised "senses" (colour constancy, scale, texture) so that more kinds of unknowns are covered.
  - Route by content before judging orientation.
  - Train for longer on the GPU.

## Status: ✅ working (was 🟡). The scoreboard is now 14 / 14.

➡ Improved further in [[24 - Round 7 Abstention Scores Raised and Noise Removed]] (93% / 90% / 85%, noise 100%).

## Code
`lab-code/`: big9_canonical.py · big9b_diag.py · big9c_fix.py · big9d_shared.py · results_big9*.json
