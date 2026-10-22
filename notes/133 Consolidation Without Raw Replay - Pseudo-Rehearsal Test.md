# 133 - Consolidation without stored examples: pseudo-rehearsal from a frozen old self (2026-10-12/13 IST)
Question (from an outside review, via the user): can knowledge of task A survive learning task B if the slow weights are distilled toward an older self on noise/latent inputs, with no stored A examples?
Correction to the proposal: L5's fast-weight memory is per-sequence state, already wiped after every sequence, so there is no persistent fast store to teach from. What was tested instead: the A-trained model is frozen as teacher; while training on B the student also minimises KL to the teacher on pseudo-inputs. Code: cont_suite2.py (same tasks/model as cont_suite.py, L5 d=128, 4 layers, 936,357 params, 2500 steps A + 2500 steps B).
## Seed 0 (MEASURED, CPU; chance 10%; whole-answer accuracy; A_after_A = 73.8%)
| condition | A after B | B after B | stored A examples? |
|---|---|---|---|
| none (sequential) | 0.8% | 96.9% | no |
| raw replay, 25% of batches (earlier run, cont_suite.py) | 48.0% | 94.9% | 64 episodes |
| KD on random bytes (kdnoise) | 44.1% | 99.2% | none |
| KD on teacher's own samples (kdself) | 15.6% | 97.3% | none (256 model-generated sequences) |
| KD on random content in A's format (kdfmt) | 66.4% | 96.9% | none, but LEAKS the task format |
## Reading
- One seed only. No claim until seeds 1-2 finish (running).
- Distillation from the old self on pure noise kept ~60% of A's accuracy (44.1 of 73.8) while B stayed at 99%, comparable to raw replay. Surprising enough to need replication; alternative explanation: a KL anchor on any inputs just limits weight drift (a generic regulariser), not 'consolidation'. A control with KD on B-style inputs or an L2-to-old-weights anchor is needed to separate these.
- The format-prior variant retains most of A (66.4 of 73.8) but uses knowledge of A's grammar, so it is not an unsupervised result.
- Costs (MODELLED): the teacher is a frozen second copy of the weights (2x weight memory during consolidation) and the extra forward passes about double training compute. Not a free lunch for an FPGA.
- Transformer under the same procedure: not run.
## Next
Seeds 1-2 (none/kdnoise/kdfmt); L2-anchor control; transformer comparison; larger task sets (A then B then C).

## Controls (seed 0, MEASURED; eval noise about ±5 points: the same A checkpoint scored 73.8% and 63.3% on two random eval draws)
- L2 anchor to old weights (lambda 1.0): A after B 60.2%, B after B only 12.5% -> pinned the model, did not learn B. One strength only; not a fair trade-off curve.
- KD on task-B-style inputs (kdb): A after B 14.1%, B 96.5% -> keeping the old outputs only on B-like inputs does not protect A. So the noise result is not just 'any KL anchor'; the input distribution matters (random bytes cover regions of input space that B-like data does not).
- Seeds 1-2, condition none: A after B 5.5% (A_after_A 66.4%) and 0.0% (78.5%); B 94.9% / 96.9%. kdnoise and kdfmt for seeds 1-2 still running.

## Replication of kdnoise (3 seeds, MEASURED)
| seed | A after A | none: A after B | kdnoise: A after B | kdnoise: B after B |
|---|---|---|---|---|
| 0 | 73.8% | 0.8% | 44.1% | 99.2% |
| 1 | 66.4% | 5.5% | 62.9% | 92.2% |
| 2 | 78.5% | 0.0% | 34.0% | 96.9% |
| mean | 72.9% | 2.1% | 47.0% | 96.1% |
Reading: KD on random-byte noise from the frozen old self retains about two thirds of A (47.0 of 72.9) with no stored A examples; B learned to 92-99%. Seed spread is large (34-63%), so the size of the effect is uncertain; the direction held in all 3 seeds. Compared with raw replay: only seed 0 measured (48%). Not yet shown: transformer under the same procedure, longer task sequences (A,B,C), whether it holds at language scale. Cost (MODELLED): frozen teacher copy + roughly 2-3x training compute during consolidation. kdfmt seeds 1-2 still running.

## Forgetting-fix comparison (seed 0 only, MEASURED; the A checkpoint scored 63.3% on this eval draw, 73.8% on the first; eval noise about ±5 points)
| method | stored A examples | A after B | B after B |
|---|---|---|---|
| none | 0 | 0.8% | 96.9% |
| KD on random bytes, strength 1 (main run) | 0 | 44.1% | 99.2% |
| KD on random bytes, strength 5 | 0 | 43.0% | 95.3% |
| KD on random bytes, strength 25 | 0 | 52.7% | 96.9% |
| EWC (graded Fisher penalty, 1000) | 0 (needs A samples to compute importance) | 21.1% | 99.2% |
| Binarised EWC (freeze top 30% by importance) | 0 (same) | 26.2% | 72.3% |
| Micro-buffer, 8 raw A episodes, 25% of batches | 8 | 28.9% | 94.5% |
| Raw replay, 64 episodes (earlier run) | 64 | 48.0% | 94.9% |
Reading: noise-KD is the best fix so far and beats 8-episode replay, EWC and binarised EWC on this seed; stronger pull (25) looks a bit better but is within eval noise. Nothing here restores A to its starting level (best 52.7% against 63-74% before), so forgetting is reduced, NOT fixed. Binarised EWC hurt B (72%). One seed each; only the main noise-KD setting has 3 seeds (47.0% mean).
Not tested: context gating, k-WTA, hyperdimensional binding, metaplastic cascade synapses (need model changes and, for gating, a task ID at test time).

## Strong noise-distillation, 3 seeds, FIXED 1024-episode eval (MEASURED, 2026-10-14 IST)
Distillation strength 100 on random bytes, no stored A examples, same 2500+2500 steps.
| seed | A before B | A after B | kept | B after B |
|---|---|---|---|---|
| 0 | 70.1% | 63.5% | 90.5% | 93.0% |
| 1 | 71.3% | 70.4% | 98.8% | 90.9% |
| 2 | 73.4% | 56.2% | 76.5% | 95.5% |
| mean | 71.6% | 63.4% | 88.5% | 93.1% |
(Plain sequential training: A after B ~2%.) Reading: the strongest result so far; forgetting reduced from ~97% lost to ~11% lost on average, B still learned (91-96%, a few points below the ~96% of plain training). NOT fully fixed: seed 2 still loses 17 points. Next: adapters on frozen slow weights + noise KD; repeated short consolidation rounds; A->B->C sequence; transformer under the same procedure.
