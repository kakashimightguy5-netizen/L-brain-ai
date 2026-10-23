# 152 - L-pure 12000-step results and the broken-teacher bug (2026-10-22 IST) - MEASURED, 1 seed, laptop RTX 3050

Setup: py_big, BS 16, T 256, 12000 steps, eval = 6x1024 windows. No memory block, no distillation unless stated.

| run | params type | final bits/byte | vs dense d256 |
|---|---|---|---|
| dense d256 (control, float matmul) | dense | 1.7266 | 0 |
| dense d128 (control) | dense | 1.8871 | +9.3% |
| MaskTern K=32 d=256 (matrix-free: adds only in matrices) | 180,224 adds/token | 2.0497 | +18.7% |
| teacher_dense_d256 (lpure_lm6) | - | 18.6971 (never trained) | INVALID |
| kd_mask_K32_d256 (lpure_lm6) | - | 3.2127 | INVALID |

Gap of sparse K32 d256 vs width-matched dense d256: +18.7%. Vs the smaller dense d128: +8.6%.

## The teacher bug (found by reading the code)
In lpure_lm6.py the lines `opt.zero_grad(); loss.backward(); clip; opt.step(); sch.step()` were indented INSIDE `if tch is not None:`. Without a teacher no optimiser step ever ran, so teacher_dense_d256 stayed at random init (18.70 bits). The KD student then distilled from a random teacher, hence 3.21. Not an architecture result. Fixed in lpure_lm7.py (steps moved outside the if). lpure_lm5.py is unaffected, which is why its runs trained.

## Next
1. Retrain dense d256 teacher with lpure_lm7.py (saves .pt). 2. Redo KD student K32 d256. 3. If gap still >10%: add AssocMem block, hybrid with a few ternary dense layers, 3 seeds.
