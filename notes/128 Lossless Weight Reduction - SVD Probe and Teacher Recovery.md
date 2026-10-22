# 128 Lossless weight reduction: SVD probe and teacher recovery
Date: 2026-10-10 (IST). User rule: efficiency must NOT cost quality; the deployed model must stay small (no big matrices restored).

## Finding 1 (MEASURED): trained L-Edge matrices are nearly full-rank
SVD-factorise the TRAINED baseline (LEdge_s0, 936,357 params) with no retraining; eval on the 6x1024 windows (baseline 0.9359 bits/byte). Script: `svd_probe.py`.
| rank kept | dense4 (qkv,o,inp,f1) | all6 (+f2,out) | params (all6) |
|---|---|---|---|
| 0.75 | +1.5% | +1.3% | 1,051,045 (MORE than baseline) |
| 0.5 | +11.5% | +11.9% | 772,517 |
| 0.375 | +25.0% | +28.7% | 633,253 |
| 0.25 | +56.9% | +60.9% | 493,989 |
Conclusion: plain low-rank cannot be lossless at d=128. Rank only saves params below ~0.6.

## Finding 2 (MEASURED, single seed, CPU): teacher recovery works
Student = SVD init (all6, rank 0.5, 772,517 params), fine-tuned with loss 0.5*KL(teacher)+0.5*CE; teacher = baseline, used only in training and discarded. `compress_ft.py`.
step 0: +11.9% | 50: +6.4% | 100: +5.7% | 150: +4.2% | 200: +4.3% | 250: +3.8% | 300: +3.6%. Still improving at 300 steps.

## Queued on the laptop (3000 steps each, gate: within ~0.5% of 0.9359 to keep)
- 1619i0 rank 0.5; 1619i1 rank 0.375; 1619i2 ternary only (QAT, full rank); 1619i3 ternary + rank 0.5; 1619j0 = i3 on CPU.
- From scratch, for comparison: 1619f125/f25/f375 (low-rank all), 1619g0 share layers, 1619g1 share + tie head.
- Skipped on purpose (expected quality loss): 1619f50, 1619g2, 1619g3.

## Flags added (all default off)
LOWRANK_ALL=1 (also factorise f2,out), SHARE_LAYERS=1 (layers reuse layer-0 inp/out/f1/f2), TIE_HEAD=1 (head = emb). Toy equivalence checked; flag-off path unchanged. Full model with all three: 323,749 params (-65%), MODELLED not trained.

## Caveats
Single seed; CPU 300-step run only; share/tie/ternary quality not yet known. Sharing and tying cut stored weights, not training FLOPs. Next: read the laptop results, 3 seeds for anything within gate.


## Update 2026-10-11 05:05 IST — corrected results (env-leak bug)

**Bug (MEASURED from logs):** the runner (`START_L_RUNNER.ps1`) cleared only a fixed list of env vars between jobs, so `QAT_MAP`, `QAT_SKIP`, `ALPHA_RAMP` etc. leaked from one job into later ones. Jobs started after the first mixed-precision run (`kb0`) silently became mixed-precision runs. Detected because logs printed a `QAT_MAP` line the job file never set, and the svd-init bpb matched the mix runs. Fix: new jobs set every var explicitly (blank = off); runner patched (effective on next restart). Gate: within 0.5% of 0.9359 bits/byte; baseline 0.9359; single-run numbers unless "3 seeds".

Naming: **Mix A** = ternary on f1/f2, int4 elsewhere. **Mix B** = ternary on f1/f2, int8 elsewhere.

| Variant | Params | Result vs baseline | Verdict |
|---|---|---|---|
| fp low-rank 0.5 (true), seeds 0/1/2 | 772,517 | -3.28 / -3.71 / -1.71 % | PASS, 3 seeds |
| fp low-rank 0.375 (true, `kbe1`) | 633,253 | -0.12 % at 3000/6000 | running |
| Mix B rank 0.5, seeds 0/1/2 (6000 steps) | 772,517 | -1.31 / -1.14 / -1.30 % | PASS, 3 seeds |
| Mix B rank 0.5, 9000 steps HID3 KDT2 (`kd0`, mislabelled ternary) | 772,517 | -0.13 % | pass, no better than 6000 steps |
| Mix B rank 0.375 (`kbb0`, 6000) / (`kbd0`, 9000) | 633,253 | +0.70 % / +1.14 % | fail |
| Mix B rank 0.25 (`kba2`) | 493,989 | +2.62 % | fail |
| Mix B full rank (`kc0`) | 936,357 | -3.13 % | pass, 1 seed |
| Mix A full rank (`kc2`, 3000) | 936,357 | -0.61 % | pass, 1 seed (seeds 1,2 queued) |
| Mix A rank 0.5 (`kc1`, 3000 / `kbc2`, 9000) | 772,517 | +0.58 % / +1.39 % | fail |
| all-ternary rank 0.5 (`ka2`, valid) | 772,517 | +3.73 % | fail |
| true int4 rank 0.5 (`kbe2`) | 772,517 | +7.39 % at 1000 | running |
| true int4 full rank (`kbe3`) | 936,357 | pending | running |

Notes: eval is 6 short windows (noisy; step-to-step swings of 1-2 % are normal); rank-0.5 fp is the safest keeper so far, Mix B is the first with ternary weights that passes 3 seeds. Storage/energy for Mix B is MODELLED, not measured (bits per weight and bytes still to be computed). Speed/energy numbers from `kb_train_graph` were taken with Ollama on the GPU and are invalid; to be rerun alone.


## Update 2026-10-11 ~09:30 IST — modelled storage of the keepers (MODELLED, not measured on hardware)

Assumptions: matrices = qkv, o, inp, f1, f2, out (720,896 weights; dims per layer: qkv 384x128, o 128x128, inp 256x128, f1 256x128, f2 128x256, out 128x128, x4 layers); everything else (embeddings, head, norms, biases, gates = 215,461 params) kept fp16; ternary counted at log2(3)=1.585 bits; per-row scale overhead ignored in the first column and counted as fp16 per 32 weights (+0.5 bit) in the second (upper bound; actual per-row pow2 scales are cheaper).

| Variant | Params | KB (no scale) | KB (+scale bound) | vs fp32 | vs fp16 |
|---|---|---|---|---|---|
| baseline fp32 (as trained) | 936,357 | 3657.6 | - | 1.00x | 0.5x |
| baseline fp16 | 936,357 | 1828.8 | - | 2.00x | 1.00x |
| fp low-rank 0.5, fp16 | 772,517 | 1508.8 | - | 2.42x | 1.21x |
| fp low-rank 0.375, fp16 | 633,253 | 1236.8 | - | 2.96x | 1.48x |
| Mix B rank 0.5 (tern f1/f2, int8 rest) | 772,517 | 810.9 | 844.9 | 4.51x | 2.26x |
| Mix A full rank (tern f1/f2, int4 rest) | 936,357 | 695.5 | 739.5 | 5.26x | 2.63x |
| int4 all matrices, full rank | 936,357 | 772.8 | 816.8 | 4.73x | 2.37x |
| int4 all matrices, rank 0.5 | 772,517 | 692.8 | 726.8 | 5.28x | 2.64x |

**Finding:** the non-matrix parameters (embeddings: emb, head, 3x remb, temb, hemb; 23% of all params) are 431 KB in fp16 and are now the largest part of every quantised variant (more than 60% of Mix A's 695 KB). Next lever = quantise them: new flag `QAT_EMB` (default off) + `other:int8` in `QAT_MAP`. Jobs queued: `1619kbg0` (Mix A full rank + int8 embeddings/head), `1619kbg1` (Mix B rank 0.5 + int8 embeddings/head). Modelled saving if quality holds: about 215 KB (fp16 -> int8 for those tables).

Also: results for kbf0-kbf5 (fp rank 0.375 seeds 1,2; int4 full rank seeds 1,2; int4 rank 0.375; fp rank 0.25 9000 steps) pending; the runner's 3 GPU slots were held by old sweep jobs 1622/1623/1624_E_*_s2.
