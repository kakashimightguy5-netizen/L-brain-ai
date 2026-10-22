# 136 - Fair control for the ternary L5 result (2026-10-14 IST, MEASURED on laptop GPU, 6x1024 eval windows)
Question: cft2_tern_hid_b (all matrices ternary except head/trust, distilled from LEdge_s0) scored 0.9065 bits/byte vs the 0.9359 baseline. Was that ternary being good, or just 6000 extra training steps?
Control: identical recipe (compress_ft.py 1.0 6000 all6, HID 1, LRN 2e-3, KD from LEdge_s0) with full-precision weights (QAT none), 3 seeds.
| run | bits/byte |
|---|---|
| original baseline LEdge_s0 | 0.9359 |
| full-precision control s0 / s1 / s2 | 0.8936 / 0.8936 / 0.8897 (mean 0.8923, -4.7% vs 0.9359) |
| ternary (cft2_tern_hid_b, s0) | 0.9065 (+1.6% vs control mean) |
Reading:
- The extra distillation fine-tune itself improves L5 by ~4.7%. The fair reference for the 0.5% gate is now 0.8923, not 0.9359.
- Against the fair control, ternary is +1.6% -> FAILS the 0.5% gate (one ternary seed; seeds 1-2 running).
- Earlier statement that ternary 'beat the baseline' was an unfair comparison; corrected here.
## Next (queued on laptop GPU)
Ternary student initialised from and distilled by the stronger full-precision control (TEACHER flag added to compress_ft.py, default unchanged), LR 1e-3 cosine, 3 seeds (jobs 1500c0-2). Skipped superseded jobs kbf1-5, kbg0-1.

## Ternary seeds 1-2 (MEASURED, 2026-10-15 00:05 IST)
cft2_tern_hid_b s1 0.9484, s2 0.9444 (s0 0.9065). Mean 0.9331 = +4.6% vs the fair full-precision control (0.8923). Seed 0 was a lucky run; ternary with this recipe clearly FAILS the gate. Both seeds spiked when quantisation reached full strength (ALPHA_RAMP 0.5 -> step 3000: 1.01 / 1.045) and only partly recovered.
Running now: cft4 ternary initialised from and taught by the stronger control models, LR 1e-3 (jobs 1500c0-2, ~80 min).

## cft4: ternary initialised from and taught by the stronger control (MEASURED, 2026-10-15 01:20 IST)
s0 0.9255, s1 0.9175, s2 0.9327; mean 0.9252 = +3.7% vs control 0.8923 -> FAILS the gate, but better than cft2 seeds (mean 0.9331).
Pattern in every ternary run: quality jumps +8-11% when quantisation reaches full strength, then recovers a lot only in the final low-LR phase (e.g. 0.964 at step 5000 -> 0.926 at 6000). Diagnosis: the d=128 model is too narrow for 1.58-bit weights and needs a longer low-LR recovery.
Running now (laptop GPU): (1) ternary 12000 steps, ramp 0.25, LR 5e-4 (1500d0); (2) WIDER ternary students trained fresh with KD from the control: d=192 (1.95M params) and d=256 (3.31M params), 12000 steps (STUDENT_D flag added to compress_ft.py, default off). Size check (MODELLED): d=256 ternary ~5.2 Mbit of weights vs d=128 fp32 ~30 Mbit, still ~6x smaller and multiply-free; more add operations though.

## Wide ternary L5 - results (MEASURED, laptop GPU, 2026-10-15 09:30 IST, 6x1024 eval windows, bits/byte)
Fresh students trained 12000 steps with KD from the full-precision control (STUDENT_D flag), all matrices ternary except head/trust.
| model | weights | bits/byte |
|---|---|---|
| fp d=128 control (from LEdge + 6000 KD steps), 3 seeds | full precision, ~15 Mbit fp16 | 0.8923 mean |
| ternary d=192 s1 / s2 | ternary, ~7.6 Mbit (MODELLED) | 0.8261 / 0.8369 |
| ternary d=192 s0 | | 0.8647 at step 8000; run interrupted by laptop sleep, resuming |
| fp d=192 s0 (same recipe) | full precision, ~31 Mbit fp16 | 0.8050 |
Reading:
- Ternary d=192 is ~6-7% BETTER than the fp d=128 model at about half its weight memory and with multiply-free matrices. Lightweight win vs the d=128 model, 2 seeds final.
- Against fp at the SAME width (d=192) ternary is ~3-4% worse: ternarisation still costs quality; width buys it back.
- Caveat (training budget differs): the wide students had 12000 fresh steps; the d=128 control had LEdge pretraining + 6000 steps. Same-recipe d=128 controls (fp and ternary, fresh 12000 steps) are queued (1500c45/c46) to settle this.
- Cost (MODELLED): d=192 needs ~2.2x more add operations per byte than d=128 needs multiply-adds; adds are much cheaper than multiplies in hardware, but energy has not been measured.

## Update 2026-10-15 ~10:35 IST (MEASURED)
- Ternary d=192 seed 0 (resumed after the laptop sleep): 0.8230. Three-seed ternary d=192: 0.8230 / 0.8261 / 0.8369, mean 0.8287 = -7.1% vs the fp d=128 control (0.8923).
- Ternary d=256 seed 0 (resumed): 0.8009 at step 10000 of 12000 (~11 Mbit MODELLED).
- Running: ternary d=192 with int8 embeddings (0.9371 at step 4000), same-recipe fp d=128 fresh control; ternary d=128 fresh control queued.

## Same-recipe control settles the fairness question (MEASURED, 2026-10-15 12:51 IST, seed 0 for controls)
All trained fresh, 12000 steps, KD from the same teacher, same LR/schedule; only width and weight type differ.
| model | weight memory (MODELLED) | bits/byte |
|---|---|---|
| fp d=128 (same recipe) | ~15 Mbit fp16 | 0.8471 |
| ternary d=192 (3 seeds) | ~7.6 Mbit | 0.8230 / 0.8261 / 0.8369, mean 0.8287 |
| ternary d=192 + int8 embeddings | ~5 Mbit (embeddings 8-bit; head/trust still 16-bit) | 0.8340 |
| ternary d=256 | ~11.3 Mbit | 0.7863 |
| fp d=192 (same recipe) | ~31 Mbit fp16 | 0.8050 |
Reading: against a same-recipe fp d=128 model, ternary d=192 is still 2.2% better at half the weight memory, and the int8-embedding version is 1.5% better at about one third of the memory, with no multiplies in the ternary matrices. The earlier '7% better' figure overstated the win because the old control had a different training budget; corrected here. Ternary d=256 (0.786) beats fp d=192 (0.805) at about one third of its memory. Ternary d=128 same-recipe control still running (1500c46).
- Ternary d=128 same recipe (seed 0): 0.8941 vs fp d=128 same recipe 0.8471 -> ternary at the SAME width is 5.5% worse. Width is what makes ternary work: ternary d=192 (0.8287) beats fp d=128 at half the memory.
