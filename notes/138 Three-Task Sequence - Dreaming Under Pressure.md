# 138 - Three-task sequence A -> B -> C (2026-10-15 IST, MEASURED, cloud CPU, cont3.py)
Tasks: A letters a-m -> digits; B letters n-z -> letters a-j; C digit pairs -> letters k-t. 2500 steps each, L5 d=128, fixed 1024-episode eval, chance 10%.
## Seed 0
| condition | after A | after B (A / B) | after C (A / B / C) |
|---|---|---|---|
| plain sequential | A 83.2% | 9.5% / 97.2% | 0.0% / 0.1% / 99.5% |
| unigram dreams (strength 25) | A 83.2% | 83.9% / 96.4% | 82.0% / 3.5% / 98.1% |
| bigram dreams (strength 25) | A 83.2% | 84.4% / 98.0% | 82.6% / 85.7% / 98.9% |
## Reading
- Unigram dreams keep A through two later tasks, but B collapses when C is learned. Likely cause: B and C both answer with letters after '='; a 256-number histogram never produces '? key key =' patterns, so the dream never probes the B mapping that C overwrites. A survives because its digit answers do not compete with C's letter answers.
- Fix being tested: BIGRAM dreams (per-task 256x256 transition counts, still no stored episodes) so dreams contain the local structure (': v ;', '? k k =') where interference happens. Seed 1 of plain/unigram also running.

## Bigram dreams result (seed 0, MEASURED, 2026-10-15 ~05:30 IST)
Bigram dreams (local structure, still no episodes) rescue B: 3.5% -> 85.7% after C, while A stays (82.6%) and C is learned (98.9%). B still loses ~12 points (98.0 -> 85.7).
Seed 1: plain A/B/C after C = 0.0 / 3.5 / 98.6%; unigram after B = 83.0 / 94.0% (C running). Running: bigram seed 1; bigram strength 100 seed 0.

Seed 1 unigram dreams after C: A 82.9 / B 78.6 / C 89.9% (seed 0 had B 3.5%) -> large seed variance; bigram is the more reliable setting so far (seed 1 bigram and strength-100 bigram running).

## Update (2026-10-15 ~07:00 IST, MEASURED)
- Bigram dreams seed 1 (strength 25): after C A 84.1 / B 94.1 / C 90.8%. Two seeds so far: A 82.6-84.1, B 85.7-94.1, C 90.8-98.9 (plain: A 0, B ~0-3.5, C ~99).
- Bigram dreams strength 100 (seed 0): A 83.0 / B 70.2 / C 96.7% -> worse than 25 (too stiff).
- Running: bigram seed 2; transformer plain vs bigram (seed 0).

## Update (2026-10-15 ~09:00 IST, MEASURED)
- L5 bigram dreams seed 2: task A only learned to 38% on this seed (training failure on A, other seeds 83%); after C: A 37.9 (kept) / B 69.5 / C 98.4%.
- Transformer seed 0: plain after C = A 0.0 / B 0.0 / C 32.6%; bigram dreams after C = A 22.2 / B 13.1 / C 35.7%. Dreams help the transformer too, but it learns each task poorly (~31-37% vs L5 ~83-98%) at matched size.
- Summary over L5 bigram seeds 0-2 after C: A 82.6 / 84.1 / 37.9 (its own starting level), B 85.7 / 94.1 / 69.5, C 98.9 / 90.8 / 98.4. Plain training: A and B ~0.
- Seed 3 (MEASURED): plain after C = A 0.0 / B 1.0 / C 99.9%; bigram dreams after C = A 88.5 (started 88.7) / B 81.9 / C 98.6%.
- Bigram-dream summary, 4 seeds, after C: A kept at its starting level in all 4 (82.6, 84.1, 37.9*, 88.5; *seed 2 never learned A); B 85.7 / 94.1 / 69.5 / 81.9 (mean 82.8; ~98 before C); C 90.8-98.9. Plain training: A 0, B 0-3.5. Remaining weakness: the most recent old task (B) still loses ~15 points.

## Recency-weighted bigram dreams, 4 seeds (MEASURED, 2026-10-16 ~05:15 IST, cloud CPU, cont3.py RECENT=2.5, LAM 25)
Dream mix weights newer tasks higher (factor 2.5); still no stored episodes. After task C:
| seed | A (start -> after C) | B | C |
|---|---|---|---|
| 0 | 83.2 -> 83.3 | 93.6 | 98.7 |
| 1 | 74.9 -> 75.7 | 89.5 | 86.6 |
| 2 | 95.5 -> 94.9 | 92.4 | 97.2 |
| 3 | 88.7 -> 88.3 | 96.7 | 99.4 |
| mean | 85.6 -> 85.5 | 93.0 | 95.5 |
- Versus plain bigram dreams (mean B 82.8, one seed with A never learned): recency lifts B by about 10 points (93.0 vs 82.8) with A fully retained (no loss on any seed) and C still learned (mean 95.5, seed 1 low at 86.6).
- Plain sequential training for reference: A 0, B 0-3.5, C ~99.
- Caveats: synthetic key->answer tasks, small L5 (d=128), 4 seeds, container resets meant runs resumed from checkpoints (resume is exact, weights and optimizer state). Seed 1 is a weak seed (A only 75% before any forgetting) and its C is also lower. B still loses 3-8 points relative to right after B (about 90-98), so retention is good, not perfect.
- Verdict: good on this toy; not evidence for real-language continual learning. Next: test sleep on the real L5 language benchmark.

## Seeds 4 and 5 added, all 6 seeds (MEASURED, 2026-10-16 ~07:45 IST, recency 2.5, LAM 25)
| seed | A start -> after C | B (after C) | C |
|---|---|---|---|
| 4 | 37.6 -> 38.2 | 94.1 | 98.1 |
| 5 | 41.3 -> 38.5 | 84.7 | 98.4 |
- Seeds 4 and 5 are WEAK on task A (about 40% before any forgetting); forgetting did not make it worse (A kept within 3 points). B 94.1 / 84.7, C 98.1 / 98.4.
- Mean over seeds 0-3 (healthy A): A 85.6 -> 85.5, B 93.0, C 95.5. Mean over all 6: A 69.8 (dragged down by weak seeds 1, 4, 5 that never learned A well), B 91.8, C 96.4.
- Reading: forgetting is controlled on all 6 seeds (A never loses more than about 3 points of what it learned), but task A is only learned well on 3 of 6 seeds. That is a training problem, not a forgetting problem.
- Fix being tested (flag A_MIN, default off, validation episodes separate from the fixed eval): after task A, keep training in extra rounds (half length, lr 1e-3, up to 3) until validation accuracy >= 0.8. Running on seeds 1, 4, 5 (FTAG _recent_amin). Smoke-tested; results to follow.

## A_MIN fix results, weak seeds 1, 4, 5 (MEASURED, 2026-10-16 ~18:30 IST, cloud CPU, A_MIN=0.8, A_ROUNDS=3, recency 2.5, LAM 25, 1 run per seed)
Accuracy after task C; old = recency run without A_MIN, new = with A_MIN.
| seed | A old -> new | B old -> new | C old -> new | extra A rounds |
|---|---|---|---|---|
| 1 | 75.7 -> 89.8 | 89.5 -> 89.3 | 86.6 -> 86.0 | 1 |
| 4 | 38.2 -> 93.6 | 94.1 -> 89.3 | 98.1 -> 97.0 | 2 (plateau, then breakthrough) |
| 5 | 38.5 -> 92.6 | 84.7 -> 88.9 | 98.4 -> 93.7 | 2 |
| mean | 50.8 -> 92.0 | 89.4 -> 89.1 | 94.4 -> 92.2 | |
- A is fixed on all three weak seeds (+41 points mean). B is flat on average (seed 4 -4.8, seed 5 +4.2, i.e. seed variance). C costs about 2.2 points on average (seed 5: -4.7, a real cost).
- Healthy seeds 0, 2, 3 are unaffected by design (validation already >= 0.8, A_MIN is a no-op).
- Caveats: toy key->value recall, one run per seed, so a 3-5 point move on B or C is inside seed variance. Not evidence for real language.
- Next: RECENT=4 re-run on seeds 4 and 1 to see if B/C cost shrinks (running); CONV=8 delay-line test on hop2/kvupd (hop2 seed 0: no clear difference vs CONV=4).

## PROJECT PAUSED for one week (2026-10-16, ~19:15 IST, user decision)
- All cloud jobs stopped and the scheduled check-in removed. Nothing is running.
- Finished: A_MIN seeds 1, 4, 5 (above); CONV=8 hop2 seed 0 (no clear difference vs CONV=4).
- Unfinished, resumable from checkpoints in the cloud workspace only if it survives (checkpoints may be lost; rerun is cheap): CONV=8 hop2 seeds 1-2 and kvupd seeds 0-2 (8000 steps each, `c8queue.sh`); RECENT=4 + A_MIN on seeds 4 and 1 (`bqueue.sh`, FTAG _recent4_amin).
- On resume: relaunch `c8queue.sh` and `bqueue.sh`, then compare against `/lim8k` (CONV=4) and the recent_amin table above. Laptop comparison runs (1500f0/f1 transformer, 1500g0-g2 mixA d=192 ternary) are still the user's to run.
- Tool note: rea-agents is installed globally in the cloud workspace; `rea-agents setup` was blocked by a permission check and not run.
