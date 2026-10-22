# 148 - NOMAT: sparse ternary wiring distilled from trained L5 (2026-10-22 IST) - MEASURED, 1 seed

Setup: teacher LEdge_s0 (0.9359 bits/byte on 6x1024 eval windows). Every Linear except the output head replaced by sparse ternary wiring (each output keeps its K strongest inputs by |w|, weights in {-1,0,+1}, per-output gain), then 1500 steps of distillation + CE (AdamW 5e-4). Adds/token counts only the replaced layers.

| K | adds/token vs dense MACs | params | bpb after 1500 steps | vs teacher |
|---|---|---|---|---|
| 32 | 0.227x (165,024 vs 725,632) | 380,906 | 1.0935 | +16.85% |
| 16 | 0.114x | 298,394 | 1.2946 | +38.34% |
| 8 | 0.057x | 257,138 | 1.4760 | +57.72% |
| 4 | 0.028x | 236,510 | 1.6487 | +76.18% |

Reading (honest):
- Quality gate (0.5%) is FAILED by a wide margin at every K. Naive pruning of a trained dense model into sparse ternary does not work at this budget. Fewer connections = monotonically worse.
- K=32 was still improving slowly (+17.5% at 1000, +18.9% at 1250, +16.9% at 1500): not converged, but unlikely to reach the gate in 1500 steps.
- This tests "prune a trained model", not "design the model sparse from the start". It does not show sparse wiring cannot work; it shows the dense mixing in the trained model carries a lot that K<=32 cannot recover quickly.
- Still dense/multiplying: output head, delta-memory rank-1 state update, norms, per-output gains.
- Time per K varied (2483 s for K=32 vs 875 s for K=16): other GPU load during the first run; ms/step not comparable.

Next: (1) train sparse-wired models FROM SCRATCH at larger width (d=192-256, K=16-32) instead of pruning; (2) structured mixing instead of random sparse wiring (butterfly/shift permutations between sparse layers so information reaches all channels in log steps); (3) keep ternary dense d=192 (adds only, 0.8287 bpb) as the reference point; (4) only then replace delta memory and head.
