---
tags: [L, fixes, results]
created: 2026-10-10
status: MEASURED bits/byte (L-Edge d128 4L, 3000 steps, ONE seed, EVAL 6x1024 windows); MACs MODELLED; chip nJ ESTIMATED (45 nm Horowitz). Never fit on EVAL.
---
# 124 - Fix variants vs L-Edge baseline (0.9359 bits/byte)
| variant | bits/byte | delta | multiplies/byte | additions-only/byte | weights KB | chip nJ/byte (EST) |
|---|---|---|---|---|---|---|
| baseline LEdge_s0 | 0.9359 | - | 605,422 | 0 | 740.6 | 896 |
| ternary + low-rank 0.25 (tlr25) | 1.1218 | +19.9% | 0 | 293,388 | 105.2 | 100.5 |
| dense-input spike coding, theta 0.5 (dspk05, TRAINED with the spike path) | 0.9554 | +2.1% | 499,286 (-17.5%) | 0 | 740.6 | 739 (-17.5%) |
Reading: training WITH the dense-input spike path (+2.1%) beats the post-hoc version (+3.3% at theta 0.5, ~41% of MACs skippable) in quality, but the realised multiply reduction is only 17.5% because the zero share of dense inputs after training is 14-47% per matrix (most matrices ~14-18%; a few ~42-47%). Quality cost is small; the saving is modest. Not yet evaluated: tern, lr25, dspk05_tern, brk, mix, mixcf (cloud chain running; container restarts every ~10-30 min, runs resume).
Caveat: single seed; differences of ~1% are within seed noise for L-Edge, so +2.1% should be confirmed on seeds 1/2 (laptop jobs 1650-1655 in queue).
