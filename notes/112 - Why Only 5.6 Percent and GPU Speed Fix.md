---
tags: [L, L-5, diagnosis, gpu, speed, honest]
created: 2026-10-09
status: internal research record (GPU profile + fix experiments queued on laptop)
---
# 112 - Why L-5 is "only" 5.6% better, and the GPU speed fix

Previous: [[111 - L-5 Wins at 9M on Laptop GPU]]

## Per-byte diagnosis (4M models trained on 4× data, 102,400 held-out bytes)
| Byte group | Share of bytes | Transformer | L-4 | L-5 | Who wins |
|---|---|---|---|---|---|
| ALL | 100% | 1.099 | 1.123 | **1.087** | L-5 −1.1% |
| Position 0–16 in window | 6% | 1.831 | 1.844 | 1.828 | tie |
| Position 16–128 | 44% | 1.315 / 1.081 | — | **1.288 / 1.057** | L-5 (−2%) |
| Position 128–256 | 50% | 0.936 | 0.979 | 0.933 | **tie** (the transformer can look back over everything here) |
| Start of an identifier | 10% | 3.458 | 3.480 | **3.390** | L-5 |
| Spaces (indentation) | 27% | **0.170** | 0.173 | 0.172 | transformer (slightly) |
| Digits | 1.6% | **2.884** | 2.943 | 2.913 | transformer |
| Hippocampus recall correct | 34% | 0.141 | 0.125 | **0.122** | L-5 |
| **Hippocampus recall WRONG** | 8.5% | **2.579** | 2.670 | 2.610 | **transformer: L over-trusts wrong recalls** |
| Bytes where one model is >1 bit better | — | 5.8% of bytes | — | 6.2% of bytes | — |

## Why the average margin is only a few percent
1. **A big part of the loss cannot be removed by any model.** Choosing a new identifier (start of a word) costs both models about 3.4 bits; spaces cost both 0.17. Only the *reducible* part can differ, so a 5.6% total gap is a much larger share of the reducible error. At fixed params and data, a few % is the typical size of an architecture gain in published comparisons.
2. **The two models win on different bytes, and the wins cancel.** About 6% of bytes favour each model by more than 1 bit. L's wins (copying, mid-range context, new identifiers) are partly cancelled by three specific weaknesses:
   - **(a) over-trusting wrong hippocampus recalls** (8.5% of bytes);
   - **(b) the second half of the window**, where the transformer reads every byte exactly and L-5 only ties (L-4 lost here clearly; L-5 caught up but did not pass);
   - **(c) counting** (indentation spaces, digits), which is where coarse spike counts are weakest.
3. **Fixes being tested (laptop GPU, 4M on 4× data, same seed):**
   - **8a:** baseline (8 memory blocks);
   - **8b:** + "was this memory right last time?" signal per hippocampus order (fixes a);
   - **8c:** spike counts −8..8 instead of −4..4 (fixes c);
   - **8d:** 16 memory blocks (fixes b; best at 1M).

## GPU speed fix (L-5 took 133 min vs 55 min for the transformer at 9M)
1. **Delta memory rewritten in "WY / UT form":** all chunks are processed at once with big batched matrix products and one batched triangular solve. Only a tiny dk×dk update runs chunk by chunk.
   - Exactness: the output matches the token loop to 4e-15, and the gradients match to 5e-14.
   - CPU speed: 0.21 → 0.125 s per call (1.7×). The GPU gain is being measured.
2. **Hippocampus and trajectory replay moved to the GPU (`fastmem.py`).** "Most recent earlier occurrence" is computed by a stable sort, and the replay pointer by a cumulative max.
   - **0 mismatches** vs the python versions (1,280 + 32,768 positions, with forced long repeats).
   - Removes per-step CPU work and GPU↔CPU copies.
3. **Profile job (7a):** measures transformer vs L-5 (old path, new path) and per-component times on the RTX 3050 once the GPU is free.

## Code
`lab-code-L/`: l5_model.py (delta_chunk2, rhit option) · fastmem.py · profile_L5.py · where_loses.py · results_where_loses.json
