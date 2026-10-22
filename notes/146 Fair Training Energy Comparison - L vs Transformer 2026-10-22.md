# 146 - Fair training-energy comparison, both optimised (2026-10-22 IST) - MEASURED, B=16, T=256, laptop RTX 3050 6GB

Transformer TModel d=152 (tf_ab.py) vs L5 d=128 (compile_ab.py). Same energy.py method. L eager/compiled run in the same session as the transformer. Note: compile_ab prints "shares the GPU with other running jobs" as a static label; L step times were jittery (ms_all 100-144), so treat L numbers as +/-15%.

| model / variant | ms/step | kernels/step | mJ/token |
|---|---|---|---|
| Transformer eager fp32 | 95.2 | 342 | 0.910 |
| Transformer compiled fp32 | 77.7 | 270 | 0.827 |
| Transformer eager bf16 | 80.8 | 456 | 0.736 |
| Transformer compiled bf16 (best) | 48.7 | 314 | 0.412 |
| L5 eager fp32 | 244.7 | 2365 | 2.183 |
| L5 compiled + STE + bf16 (best) | 113.6 | 1298 | 0.839 |

Reading (honest):
- Best vs best: L uses ~2.0x the training energy per token and ~2.3x the step time of the transformer. We have NOT closed the gap.
- Eager vs eager: 2.4x. Compile+bf16 cut L by 2.6x but the transformer by 2.2x, so the optimisations helped both almost equally.
- L best (0.839) is still better than the transformer's unoptimised fp32 (0.910), but that is not a fair comparison.
- Earlier 1.83 mJ/token for L was inflated by GPU sharing; clean value is 0.84.
- Cause: 1298 vs 314 kernels. L's scan + delta memory + spike elementwise ops are sequential/launch-bound.
- Per-token energy is not energy-to-quality; L's quality edge (note 111) at 9M is not accounted for here. A bpb-at-equal-joules curve is still owed.

Next levers: profile the 1298 kernels (scan, delta chunks, spikes, hippocampus 33 ms); fuse spikes/elementwise into the scan kernel; CUDA graph around compiled layers; energy-to-target-bpb comparison.
