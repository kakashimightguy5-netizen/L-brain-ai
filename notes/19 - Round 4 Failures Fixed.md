---
tags: [experiments, root-cause, round-4, internal]
created: 2026-10-06
status: internal research record (not for submission)
---
# Round 4: The Three Failures, Root Causes and Fixes

Back to [[18 - Round 3 Big Benchmarks]]

| Failure (Round 3) | Root cause (measured) | Fix | Before → After |
|---|---|---|---|
| Lucid dreams hurt | Dreams were **adversarial**: teacher confidence 1.00 (nothing to teach) and **2× further from real data** than real images are from each other | **Natural-statistics lucid dreams**: confident, but V1 feature statistics must match real memories | 57.5% → **69.8%**, now the **best sleep method** (memory 67.6, generative 68.1; ceiling 74.8) |
| Looped LM didn't pay | **Unfair design**: one wide block looped cost as much per loop as the whole baseline; noisy halting target ("is my guess right?"); state drift | Narrow block looped ×4 (same compute as baseline), consistency halting, later-loop weighting, self-distillation, state normalisation | Same params + same compute: **2.433 vs 2.509** bits per char (4 thin distinct layers). **3.6× fewer params** than the 4-layer baseline |
| Blame-aware healing: no gain | (a) With full labels, the label already revealed the owner, so the test couldn't show a difference. (b) With weak feedback, the blamed **runner-up was the true owner only 60%** of the time, so wrong columns were taught to claim foreign data | **Never assign ownership from a guess:** use the "not mine" veto + content correction; give ownership only after a **second-opinion confirmation** | Weak-feedback blame 71.6% → **85.9%** (naive 85.5%; full-label ceiling ~86.7) |

---

## 1. Lucid dreams (Fashion, genomic student, 6,000-dream pool per method)
| Sleep source | Teacher confidence | Unnaturalness (1.0 = like real data) | Student accuracy |
|---|---|---|---|
| Real data (reference) | 0.89 | 0.95 | (ceiling 74.8%) |
| Memory + mild dreams | 0.85 | 0.96 | 67.6% |
| Generative memory | 0.82 | 1.03 | 68.1% |
| Old lucid (sign-gradient, confident) | **1.00** | **1.97** | 57.5% |
| Edge-of-knowledge (maximise teacher entropy) | 0.48 | 1.12 | 57.3% |
| Mix: generative + edge | 0.64 | 1.07 | 67.4% |
| **Natural-statistics lucid** | 0.99 | 1.12 | **69.8%** |
- **Learned:** naturalness matters more than difficulty.
- Boundary-only dreams fail even when natural, because they don't cover typical cases.
- Principle: **dreams must look like life.**

## 2. From-scratch language models (Tiny Shakespeare, 2,500 steps, bits per char ↓)
| Model | Params | Compute / token | Bits per char |
|---|---|---|---|
| Baseline: 4 distinct layers d128 | 827k | 1× | 2.311 |
| Control: 1 layer d128 (same params as looped) | 232k | 0.25× | 2.572 |
| Control: 4 distinct layers d64 (same params, same compute) | 217k | ~1× | 2.509 |
| **Looped narrow ×4** | **232k** | 1× | **2.450** |
| **Looped narrow ×4 + state normalisation** | **233k** | 1× | **2.433** (loop 3) / 2.440 (loop 4) |
| Looped + norm, adaptive gate 0.95 | 233k | **0.76×** | 2.458 |
| Early exit 4 layers + self-distillation, gate 0.95 | 827k | **0.75×** | 2.374 |
| Early exit + self-distillation, gate 0.9 | 827k | **0.59×** | 2.423 |
- **Looping = parameter efficiency:** at equal size and compute it beats distinct thin layers (2.433 vs 2.509), and it beats the 1-layer model with identical parameters by 0.14.
- **Early exit = compute efficiency:** 25% saved for +0.06 bits per char.
- Still open: loop 4 ≈ loop 3 (mild "overthinking"); the adaptive gate stops early. Expected to improve with deeper/larger models.

## 3. Blame-aware healing under weak feedback (right/wrong only; 3 seeds)
| Healing | Shifted | Clean |
|---|---|---|
| None | 69.8% | 88.1% |
| Naive (blame the content of every mistake) | 85.5% | 93.1% |
| Blame-aware v1 (runner-up told "mine" on a guess) | 71.6–74.0% 💥 | — |
| **Veto + content** | **85.9%** | **93.2%** |
| **Second opinion (confirmed runner-up) + veto + content** | **86.0%** | 92.9% |
- Diagnosis: runner-up = true owner in only **60%** of routing-blamed cases. Acting on that guess dropped routing from 83% to 81.5%.
- Principle: **negative evidence is safe; positive ownership needs confirmation.**
- A second opinion is needed for ~60% of mistakes (one extra yes/no from the user).

## Code and raw results
`lab-code/`: lm_fixes.py · lm_fixes2.py · big4_lucid_fix.py · big5_blame_weak_feedback.py · big5b_blame_diag.py · big5c_blame_fix.py · results_lm_fixes*.json · results_big4_lucid_fix.json · results_big5*.json

## Next
- Deeper (8–12 layer) early-exit and looped LMs on WikiText-2: does skippable depth grow?
- GPU access (Kaggle / RTX 3050) to cut experiment time from hours to minutes
