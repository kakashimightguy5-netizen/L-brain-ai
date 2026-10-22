---
tags: [L, M3, energy, results]
created: 2026-10-10
status: MEASURED on laptop RTX 3050 6GB (M3: 3 seeds; energy bench: n=5 repeats, 1 checkpoint)
---
# 118 - M3 verdict and first energy-bench results
## M3 domain experts (3 seeds, mean bits/byte over Python/JS/C; lower = better)
| mode | active FFN params/byte | s0 | s1 | s2 | mean ± sd |
|---|---|---|---|---|---|
| dense | 334,656 | 1.4287 | 1.4261 | 1.4329 | 1.4292 ± 0.0028 |
| dense_small (equal active) | 84,096 | 1.4764 | 1.4738 | 1.4820 | 1.4774 ± 0.0034 |
| token-routed experts | 84,096 | 1.4415 | 1.4405 | 1.4443 | 1.4421 ± 0.0016 |
| segment-routed (ours) | 84,096 | 1.4728 | 1.4720 | 1.4799 | 1.4749 ± 0.0036 |
- Token routing beats equal-size dense by 0.035 bits/byte and sits 0.013 from the full dense FFN. Segment routing is only 0.003 better than equal-size dense (about 1 sd): NO quality gain.
- Segment routing DOES specialise by language (MI/H(domain), layers 2-3: 0.48-0.55 and 0.25-0.38 across seeds; token routing ~0.01) and switches only 4.5-7.4 times per 1000 bytes (token ~600). Value = power-gating whole expert blocks for long stretches, not quality.
## Energy bench (L-6 9M vs transformer 9M, ctx 256, batch 1, CUDA graphs, n=5, NVML; idle 11.08 W)
- Quality-matched decode: L 0.8721 vs T 0.9134 bits/byte (task from eff_bench Part A). Energy ABOVE IDLE: L 16.65 vs T 16.62 mJ/token = TIE. Total: L 30.8 vs T 29.1 mJ/token; L slower (1.28 vs 1.12 ms/token).
- Batch/context scaling (per-token metric of eff_bench Part B): ctx16384 batch1 L 15.9 vs T 303; batch4 4.9 vs 177.6; T cannot run batch 16/64 at 16K (KV cache > 3.5 GB). ctx256 batch64 L 1.74 vs T 11.4; ctx2048 batch64 1.79 vs 27.6. L's advantage grows with context and batch; at ctx256 batch1 it is a tie.
- TRAINING is a loss for L: 3.55 vs 1.24 mJ/token (2.9x more energy), 613 vs 197 ms/step, peak VRAM 6.66 vs 1.6 GB, both power-capped (SwPowerCap).
## Power-limit sweep: NOT VALID. nvidia-smi -pl returned "not supported in current scope" for every setting (55..12 W), so all rows ran at the default limit (identical numbers). Needs an Administrator shell; GPU state was left unchanged (restored, default 35 W). No sweet-spot claim can be made yet.
