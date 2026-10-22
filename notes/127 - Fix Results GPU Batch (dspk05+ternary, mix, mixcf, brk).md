# 127 - Fix Results: GPU batch (L-Edge, 1 seed each)
Time: 2026-10-10 12:15 IST. Trained on the laptop GPU (RTX 3050) as jobs 1619a-d/e. Metric: training-time VAL bits/byte (MEASURED, same split for every row, 3000 steps). Baseline LEdge_s0 = 1.3381 (CPU). The earlier fix_eval numbers (0.9359 baseline) use a different 6x1024 window set; percentages agree (e.g. ternary +9.2% here, +9.3% there).

| Variant | val bits/byte | vs baseline | copy exact n=512 | copy renamed n=512 | device |
|---|---|---|---|---|---|
| baseline LEdge_s0 | 1.3381 | - | 0.9595 | 0.8364 | CPU |
| dense-spike 0.5 | 1.3638 | +1.9% | 0.9595 | 0.8358 | CPU |
| lowrank 0.25 | 1.4363 | +7.3% | 0.9619 | 0.8271 | CPU |
| ternary | 1.4613 | +9.2% | 0.9565 | 0.8308 | CPU |
| dense-spike 0.5 + ternary | 1.4754 | +10.3% | 0.9604 | 0.8357 | GPU |
| ternary + lowrank 0.25 | 1.6038 | +19.9% | 0.9507 | 0.8261 | CPU |
| mixcopy + colpos (mix) | 1.3417 | +0.3% | 0.9619 | 0.8200 | GPU |
| mix + copy-gate feature (mixcf) | 1.3407 | +0.2% | 0.9619 | 0.8287 | GPU |
| BRACKET counter channel (brk) | 1.3408 | +0.2% | 0.9629 | 0.8325 | GPU (resumed from CPU ckpt at step 1200) |

Reading (single seed, differences under ~0.5% are within noise until 3 seeds):
- mix, mixcf and brk do not improve bits/byte or copy accuracy at this size; they cost nothing measurable either. No evidence yet that they help L-Edge.
- dense-spike on top of ternary adds about 1 point over ternary alone (+10.3% vs +9.2%); the 17.5% multiply saving of dense-spike is moot once ternary already removes all multiplies.
- Best size/quality trade so far: ternary (+9.2%, 0 multiplies, 185 KB). Dense-spike alone is the cheapest in quality (+1.9%).
Bug fixed: train_L1.py resume crashed with 'RNG state must be a torch.ByteTensor' when a GPU job resumed from an old checkpoint; now set_rng_state(ck['rng'].cpu()).
Open: 3-seed confirmation (jobs 1650-1662 in the 16xx set), SPARK fit (ternary 185 KB vs ~103.5 KB on-chip).
