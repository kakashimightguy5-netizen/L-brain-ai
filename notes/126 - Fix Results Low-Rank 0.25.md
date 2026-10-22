# 126 - Fix Results: Low-Rank 0.25 (L-Edge, 1 seed)
Time: 2026-10-10 11:30 IST. Cloud CPU run; eval on 6x1024 EVAL windows vs LEdge_s0 baseline. bits/byte MEASURED; chip nJ ESTIMATED.

| Variant | bits/byte | vs 0.9359 | multiplies/byte | weights KB | chip nJ/byte (EST) |
|---|---|---|---|---|---|
| baseline | 0.9359 | - | 605,422 | 740.6 | 896 |
| dense-spike 0.5 | 0.9554 | +2.1% | 499,286 | 740.6 | 739 |
| low-rank 0.25 (lr25) | 1.0343 | +10.5% | 291,977 (-51.8%) | 420.6 | 432 |
| ternary | 1.0229 | +9.3% | 0 | 185.2 | 208 |
| ternary + low-rank 0.25 | 1.1218 | +19.9% | 0 | 105.2 | 101 |

Reading: low-rank halves multiplies and weights but costs +10.5%, worse than ternary alone (+9.3%) and ternary cuts more. Dense-spike is the cheapest in quality (+2.1%) for -17.5% multiplies. One seed each; 3-seed confirmation pending.
Moved 2026-10-10 11:35 IST: dspk05_tern, brk, mix, mixcf now queued on the laptop GPU as jobs 1619a-1619d (front of queue); cloud chain stopped.
