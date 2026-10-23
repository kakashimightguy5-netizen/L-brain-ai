# 155 - L-pure: more weight levels + distillation (2026-10-23 IST) - MEASURED, 1 seed, 200-window scoring

Bar <= 1.7815 bits/byte (within 10% of dense 1.6195). Std error about 0.03 per score. Adds/token MODELLED (upper bound: levels x connections).

| model | bits/byte | vs dense | adds/token | verdict |
|---|---|---|---|---|
| dense d256 | 1.6195 | 0 | n/a | baseline |
| sparse K32 float (diagnostic, needs multiplies) | 1.7855 | +10.2% | 180,224 | FAIL by 0.004 |
| K32 ternary, no teacher | 2.4316 | +50% | 180,224 | FAIL |
| K32 ternary + KD | 1.9836 | +22.5% | 180,224 | FAIL |
| K32 two-term, no teacher | 1.8876 | +16.6% | 360,448 | FAIL |
| K32 two-term + KD (twokd8) | 1.8656 | +15.2% | 360,448 | FAIL |
| K32 three-term + KD (threekd8) | 1.8249 | +12.7% | 540,672 | FAIL, closest matrix-free so far |

## Reading
- Precision levers work but with diminishing returns: 1 term 1.98 -> 2 terms 1.87 -> 3 terms 1.82 (with KD). The float-weight limit for K32 wiring is 1.7855, so precision is nearly used up.
- The remaining ~10% is the cost of keeping only 32 connections per output (float-sparse +10.2%). Next lever: more connections (K64) or learned wiring, which costs adds.
- Dense d256 needs about 1.6M multiply-accumulates per token for its matrices; threekd8 uses about 0.54M adds (MODELLED). Add is cheaper than multiply-accumulate, but this is not measured energy.
- 1 seed each; differences of ~0.04 are near the error margin. A second seed of threekd8 is queued.

## Next (queue_L4.ps1)
K64 two-term + KD, K64 three-term + KD, and a second seed of threekd8.
