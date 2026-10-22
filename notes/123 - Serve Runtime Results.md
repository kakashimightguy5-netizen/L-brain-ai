---
tags: [L, serving, energy, results]
created: 2026-10-10
status: MEASURED GPU energy (NVML, n=5) + MODELLED work/SPARK numbers. ONE checkpoint (L5E_post3k_s0), seed 0 only; seeds 1/2 failed (no L5E checkpoints exist for them).
---
# 123 - Energy-proportional serving runtime, first real results
Setup: 8 streams x 1024 bytes EVAL; policies fitted on a separate CAL split (never on EVAL). Easy bytes = lowest-50% full-depth loss, hard = top 20%. Idle for this run 5.62 W.

| policy | bits/byte (vs full 0.9292) | modelled work | work on easy / hard bytes | GPU mJ/token total (above idle), batch 1 |
|---|---|---|---|---|
| full | 0.9292 | 1.000 | 1.00 / 1.00 | 137.6 +- 4.8 (57.0) |
| high | 0.9361 (+0.7%) | 0.906 | 0.82 / 1.00 | 137.3 (56.2) |
| balanced | 0.9405 (+1.2%) | 0.872 | 0.77 / 0.99 | 146.0 +- 13.4 (61.0) |
| low (halt 0.98 + recall bypass) | 0.9898 (+6.5%) | 0.806 | 0.72 / 0.94 | 126.2 +- 3.2 (52.5) |
| balanced + energy cap 0.85 | 1.1539 (+24%) | 0.743 | 0.71 / 0.78 | 128.4 (52.7) |
| full + cap 0.75 | 1.3251 (+43%) | 0.672 | 0.67 / 0.67 | 116.4 +- 2.6 (47.1) |

## Reading it honestly
1. The policy IS energy-proportional in the modelled sense: easy bytes get 0.72-0.82 of the work, hard bytes ~1.0 (full pays equally for both).
2. But the saving is small and costs quality: best defensible point is `balanced` (-12.8% modelled work for +1.2% bits). `low` saves 19% modelled work for +6.5% bits.
3. MEASURED GPU energy barely moves (high/balanced within noise of full; low -8%; cap 0.75 -15% at +43% bits). The runtime is Python/launch-bound: the GPU averages only ~9.5 W, i.e. mostly idle floor, so skipping layers does not translate into watts at batch 1. The 137 mJ/token is ~4.5x the plain GraphStepper decode (30.8) - this runtime is NOT the efficient path; it is a policy testbed.
4. Chip-level SPARK nJ/byte (ESTIMATED, event-driven model): 1137 full -> 1038 balanced (-8.7%) -> 986 low (-13%). Estimates, not measurements.
5. Consistent with note 120: depth-gating is a ~10-15% lever here, not a game changer.
Failed: serve_s1 / serve_s2 (FileNotFoundError: L5E_post3k_s1/s2.pt never trained; the L5E seed 1/2 jobs were killed during the slow-run reset).
