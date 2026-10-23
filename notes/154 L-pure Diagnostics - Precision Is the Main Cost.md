# 154 - L-pure diagnostics: weight precision is the main cost (2026-10-22 IST, late) - MEASURED, 1 seed, 200-window scoring (evalbig)

Yardstick: mean bits/byte over 200 x 1024-byte windows of py_big_val (second half), +/- standard error ~0.03. Bar: <= 1.78 (within 10% of dense 1.6195). 6-window scores used earlier were unreliable (base8: 2.07 on 6 windows vs 2.43 on 200).

| model | bits/byte | vs dense | adds/token | verdict |
|---|---|---|---|---|
| dense d256 (teacher8) | 1.6195 | 0 | n/a (float matmul) | baseline |
| sparse K32, FLOAT weights (diagnostic, needs multiplies) | 1.7855 | +10.2% | 180,224 | FAIL by 0.004 (at the bar, diagnostic only) |
| sparse K32 ternary, no teacher (base8) | 2.4316 | +50.1% | 180,224 | FAIL |
| sparse K32 ternary + distillation from teacher8 (kd8) | 1.9836 | +22.5% | 180,224 | FAIL, but distillation helped: 2.43 -> 1.98 |
| sparse K32 two-term ternary (levels -1.5..1.5, two8) | 1.8876 | +16.6% | 360,448 (MODELLED upper bound) | FAIL, but best matrix-free so far without a teacher; -22% vs base8 |

dense-ternary diagnostic was stopped at step 4000 (2.34, no verdict).

## Reading
- Removing connections (K=32 of 256) costs about 10% even with float weights: the wiring cost is real but near the bar.
- Ternary precision is the bigger cost: 1.7855 (float) -> 2.43 (ternary). A second ternary term at half scale (still adds and shifts, no multiplies) recovers most of it: 1.89.
- Distillation helps (earlier conclusion from the 6-window check was wrong). 1 seed each; differences between models exceed the error margins.
- Previous 'distillation did not help' read (2.05 vs 2.11 on 6 windows) is retracted.

## Next (queue_L3.ps1)
two-term ternary + distillation (twokd8) and three-term + distillation (threekd8), lpure_lm10.py, same 200-window scoring. Expect ~1.8 if effects add; then 3 seeds, memory block, ops scorecard.
