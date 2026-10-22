# 142 - Barcode memory + sleep loops on hop2 (seed 0 only; MEASURED, cloud CPU, 8000 steps, 2026-10-21 IST)
Flags (default off, code l5_model.py, backup l5_model_before_barcode.py): BARCODE=2 BARCODE_KF=4 (fixed random 2x sparse expansion of keys/queries, keeps top 25%); SLEEP=2 (second pass over the context starting from the consolidated state). Same recipe as the lim8k baseline (CONV=4). 64 eval items per cell, so +-6 points is noise.
| chains | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| baseline s0 | 1.00 | .594 | .422 | .344 | .234 | .156 |
| baseline mean of 3 | 1.00 | .552 | .385 | .312 | .188 | .130 |
| BARCODE s0 | 1.00 | .594 | .281 | .188 | .250 | .156 |
| SLEEP=2 s0 | 1.00 | .500 | .281 | .234 | .188 | .172 |
Reading: no benefit at seed 0; chains 4-8 are lower (0.19-0.23 vs 0.34-0.42), long chains unchanged, all inside or near noise. One seed only. Not a win; a negative or null result so far.
Note: runs survived 60+ minutes while I kept making calls (previously died about every 10 min when idle) - supports the user's idle-reset theory, one observation.
Queued (rqueue.sh): BCSL s0, then seeds 1-2 of all three variants.
Likely reason (HYPOTHESIS): the failure at long chains is length generalisation (train <=4 chains, test up to 32), not key collisions, so decorrelating keys does not address it.
