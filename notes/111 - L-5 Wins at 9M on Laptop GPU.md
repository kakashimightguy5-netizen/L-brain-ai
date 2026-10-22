---
tags: [L, L-5, 9M, gpu, results, scaling]
created: 2026-10-09
status: internal research record (seed 1 queued on laptop; 4M confirmation seeds running on cloud)
---
# 111 - L-5 wins at 9M on the laptop GPU

Previous: [[110 - L-5 Error-Correcting Memory Breaks the Ceiling]] · Where L lost before: [[107 - First Scale-Up to 9M on Laptop GPU]]

## Setup
Same as note 107:
- CPython code, file-disjoint;
- 6,000 steps × 32 × 256 bytes (49M bytes);
- RTX 3050 6 GB.

L-5 uses 8 memory blocks, energy budget λ 0.01, lr 2e-3, and gradient accumulation 2 (same effective batch).

| Model | Params | Held-out code, bits/byte | Training time |
|---|---|---|---|
| Transformer (3 runs: lr 1e-3 ×2, lr 2e-3) | 9.22M | 1.080 / 1.083 / 1.083 → **1.082** | 49–55 min |
| L-2 (note 107) | 8.59M | 1.092 (lost by 0.9%) | 170 min |
| **L-5** | **8.81M** | **1.0218 (−5.6% vs transformer)** | 133 min (GPU shared for the first ~50 min) |

## L-5 at 9M, other abilities
| Test | Result |
|---|---|
| Exact copy at 120 / 512 / 2,048 bytes | 92.5% / 96.1% / 94.8% (transformer: 89.6% / cannot / cannot) |
| 5% typos at 2,048 bytes | 89.5% |
| Renamed variables at 2,048 bytes | 86.8% |
| FFN neurons active | 2–3% in the first layers (very sparse) |
| GPU memory | 2.2 GB |

## Reading
- **The scaling problem from note 107 is solved for this size.** L lost at 9M because its additive memory saturated with more data (note 109). The error-correcting memory plus the 4-byte delay line (L-5) turns that loss into a clear win.
- **The margin at 9M (5.6%) is bigger than at 4M with the same data (1.7%).** Single seed so far; seed 1 is queued.
- **Training is still slower than the transformer** (about 2.4×), but this is tooling. The delta memory has no fused kernel yet; the diagonal scan uses the CuPy kernel.

## Trend (L vs transformer, same data per size)
| Size | Training budget | L version | L vs transformer |
|---|---|---|---|
| 1M | 3,000 × 16 | L-5 | −10% |
| 4M | 3,000 × 16 | L-5 | −7.5% |
| 4M | 6,000 × 32 | L-5 | −1.7% |
| 9M | 6,000 × 32 | L-5 | **−5.6%** |

## Next
- 2nd seed at 9M (queued), plus 4M confirmation seeds (cloud).
- A fused GPU kernel for the delta memory, to close the training-speed gap.
- Scale to about 25M, and train longer.
- Energy per token for L-5 measured on the laptop (sustainability line).
