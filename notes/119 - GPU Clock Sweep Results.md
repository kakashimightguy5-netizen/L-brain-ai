---
tags: [L, energy, clock-sweep, results]
created: 2026-10-10
status: MEASURED (NVML energy counter, RTX 3050 6GB laptop, idle 11.66 W, n=3 repeats per cell, 1 checkpoint, 1 run of the sweep)
---
# 119 - GPU clock-lock sweep (replaces the invalid power-limit sweep)
Method: `nvidia-smi -pl` is refused on this laptop even as Administrator ("not supported in current scope"), so graphics clocks were locked with `-lgc MHz,MHz` (accepted), measured, then `-rgc` restored the default (verified: rc 0). Clocks 2100/1800 both ran at an ACTUAL 1702 MHz (power-capped); 1500/1200/900/600 ran at the requested clock.
| setting (actual MHz) | L train mJ/tok | s | Transformer train mJ/tok | s | L batch-1 gen mJ/tok (above idle) | s |
|---|---|---|---|---|---|---|
| 2100 (1702) | 3.77 | 3.72 | 1.29 | 1.17 | 34.2 (19.1) | 0.66 |
| 1800 (1702) | 3.44 | 3.72 | 1.15 | 1.18 | 27.5 (13.1) | 0.63 |
| 1500 | 3.19 | 3.80 | 1.20 | 1.17 | 28.3 (12.7) | 0.68 |
| 1200 | 2.79 | 4.08 | 1.06 | 1.31 | 25.3 (7.5) | 0.78 |
| 900 | 2.48 | 4.67 | 0.96 | 1.65 | 25.8 (3.3) | 0.99 |
| 600 | 2.85 | 6.20 | 1.13 | 2.37 | 33.5 (1.4) | 1.41 |
Findings: (1) Sweet spot for TOTAL energy per token is around 900-1200 MHz: L training 2.48 vs 3.77 mJ/tok (-34%) at +25% time; transformer training -26%; L generation about -25% at 1200 MHz (+18% time). Below ~900 the idle floor (11.7 W x longer time) dominates and total energy rises again. (2) Both models gain similarly, so the L-vs-transformer training gap does NOT close: L stays ~2.6x the transformer's energy per token (2.48 vs 0.96). (3) Caution: 2100 and 1800 ran at the same actual clock yet differ by 20% in generation energy, so single-cell differences of that size are within noise; only the trend across 1500 -> 900 is reliable. (4) Idle = GPU 11.66 W; per-token numbers include the idle floor, "above idle" excludes it.
