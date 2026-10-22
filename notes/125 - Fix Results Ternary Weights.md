# 125 - Fix Results: Ternary Weights (L-Edge, 1 seed)
Time: 2026-10-10 09:22 IST. Run on cloud CPU (laptop unplugged). Eval: 6x1024 EVAL windows, baseline LEdge_s0.

| Variant | bits/byte | vs 0.9359 | multiplies/byte | adds/byte | weights KB | chip nJ/byte (ESTIMATED) |
|---|---|---|---|---|---|---|
| baseline | 0.9359 | - | 605,422 | 0 | 740.6 | 896 |
| dense-spike 0.5 (dspk05) | 0.9554 | +2.1% | 499,286 | 0 | 740.6 | 739 |
| ternary (tern) | 1.0229 | +9.3% | 0 | 605,813 | 185.2 | 208 |
| ternary + low-rank 0.25 (tlr25) | 1.1218 | +19.9% | 0 | 293,388 | 105.2 | 101 |

Reading: ternary alone removes every multiply and cuts weights 4x for +9.3% bits/byte (MEASURED, one seed; nJ is ESTIMATED). Still 185 KB, above the ~103.5 KB SPARK block SRAM, so tlr25 is the only variant near fitting, at a 20% quality cost. Next: lr25, dspk05_tern, brk, mix, mixcf; 3-seed confirmation pending.
