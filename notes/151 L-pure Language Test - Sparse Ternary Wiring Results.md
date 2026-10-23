# 151 - L-pure byte-LM test: sparse ternary wiring from scratch (2026-10-22 IST) - MEASURED, 1 seed, 3000 steps, 6x1024 eval windows

Setup: lpure_lm*.py, 4 layers, no memory block / hippocampus (mixing question isolated), batch 16x256, AdamW 2e-3 one-cycle. Numbers are comparable only to each other (NOT to the L5 0.83 baseline).

| run | bits/byte | vs dense fp d=128 | adds per token | note |
|---|---|---|---|---|
| dense fp d=128 (control) | 2.0389 | 0 | ~463k multiply-adds | |
| fixed random sparse K=32 | 3.2353 at step 1000 (stopped) | far worse | 98,304 | slow gather; random wiring never learned |
| gradual pruning K=128 (ternary, no sparsity), forced ternary | 2.377 | +16.6% | 360,448 | |
| gradual pruning K=64 | 2.431 | +19.2% | 196,608 | |
| gradual pruning K=32 | 2.5128 | +23.2% | 98,304 | |
| gradual pruning K=16 | 2.5141 | +23.3% | 49,152 | |
| gradual pruning K=32, d=256 | 2.2601 | +10.8% | 180,224 | |
| ramped ternary K=128 | 2.5013 | +22.7% | 360,448 | ramp hurt (+5% vs forced) |
| ramped ternary K=32 d=128 | 2.5224 | +23.7% | 98,304 | ~ same as no ramp |
| ramped ternary K=32 d=256 | 2.2435 | +10.0% | 180,224 | ~ same as no ramp (within noise ~0.03) |

Key-value recall (cloud CPU, 4 pairs, 2500 steps, 1 seed): L-pure add/shift memory 97.3%, same without memory 42.6%, dense-matrix control 99.2%. 8 pairs running.

Findings:
1. Removing connections is cheap: K=128 -> K=16 (8x fewer) costs ~7 points; K=16 ~ K=32.
2. Most of the loss is the ternary training at this short budget (K=128 ternary already +16.6%): jump when quantisation reaches full strength, slow recovery (same pattern as notes 136).
3. Quantisation ramp is not a fix (neutral to harmful).
4. Width helps: d=256 K=32 +10% at 180k adds (~2.6x fewer ops than dense d=128).
5. Caveats: dense control is d=128 only and underfit at 3000 steps; a width-matched fp d=256 control is owed. Noise ~0.03 bpb; 1 seed.
Mechanism note: gain must follow the CURRENT fan-in (first masked run diverged to 36 bits/byte when it did not).
Next: 12000-step runs (recovery phase) for mask K32 d256 vs dense d128 and d256; then add the add/shift memory block to the LM; 3 seeds.
