# 139 - Overnight summary (14-15 Oct 2026, IST). All MEASURED unless marked.
## 1. Forgetting (notes 137, 138)
- Two tasks: statistics-matched dreams (only the 256-number byte histogram of task A kept) keep task A fully: 71.8% after B vs 71.6% before, 3 seeds; B 93.9% (plain 96.5%). Plain training keeps ~2%. Works for the transformer too (96% of A kept), so it is a training-procedure result; L5 learns the tasks ~2x better at matched size.
- Three tasks A->B->C: unigram dreams protect A but can lose B (seed 0: 3.5%). Bigram dreams (256x256 counts per task, no episodes) fix most of it: A 82.6 / B 85.7 / C 98.9% (seed 0). More seeds running.
## 2. Reversal curse (notes 134, 135)
Sleep reverse replay with a familiarity gate: L5 91% reverse / 99% forward; transformer reaches the same with a stabilising anchor. Procedure win, not L5-only.
## 3. Lightweight L5 (note 136)
- Fair control: the same 6000-step distillation with full-precision weights improves L5 to 0.8923 bits/byte (3 seeds). This is the new reference (old 0.9359).
- Ternary at the original width (d=128) fails: +3.7% to +4.6% vs the control.
- WIDER ternary students trained fresh with distillation: d=192 at step 8000 = 0.8647 bits/byte (3.1% BETTER than the fp d=128 control); d=256 at step 6000 = 0.869. Final 12000-step values and seeds 1-2 + an fp d=192 control were queued, but the laptop link dropped around 03:35 IST (laptop asleep or offline?) so those are unread.
- Weight memory (MODELLED, embeddings/norms/head at fp16): ternary d=192 ~7.6 Mbit vs fp16 d=128 ~15.0 Mbit vs fp16 matched transformer ~15.8 Mbit; ternary matrices need adds only, no multiplies, but d=192 does ~2.2x more add operations than d=128.
## 4. Competition / sparsity (note 135)
k-winner-take-all with learning losers: 94.5% at 20% firing vs 97.4% at ~29% firing (3 seeds, delay-copy). Homeostatic thresholds alone were worse.
## Open
Read the laptop finals when it reconnects; if wide ternary holds over 3 seeds and against an fp d=192 control, run the limitation suite on it; bigram dreams 3 seeds; try bigram dreams on the transformer.
