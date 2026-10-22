# Efficiency stack report - L vs a same-size transformer (DRAFT, 2026-10-10)

Hardware: laptop RTX 3050 6GB. "MEASURED" = NVML total board energy; "total" includes the 11.08 W idle floor, "above idle" subtracts it. Data-centre numbers (PUE, water, CO2, 1e9 queries/day) are ESTIMATED / PROJECTION from `configs/sustainability.yaml`. Nothing here was measured on an FPGA or in a data centre.

## 1. Headline
1. **Quality-matched, batch 1, ctx 256: a tie.** L-6 9M 0.8721 bits/byte vs transformer 0.9134 (L better on quality). Energy: above idle 16.65 vs 16.62 mJ/token, total 30.8 vs 29.1 (MEASURED, n=5). L is slower (1.28 vs 1.12 ms/token).
2. **Long context and batching: L uses far less energy.** The transformer's cost grows with the KV cache; L carries a fixed-size state. Table 2. At ctx 16384 the transformer cannot run batch 16/64 on this card.
3. **Training: L loses.** 3.55 vs 1.24 mJ/token total (2.9x), 613 vs 197 ms/step, peak VRAM 6.66 vs 1.6 GB (MEASURED). Power draw is similar (47 vs 52 W), so the energy gap is the time gap. The old profile shows the L step is elementwise/launch-bound; job 1741 (CUDA graph / bf16) is PENDING.
4. **Clock locking helps both models equally.** 900-1200 MHz cuts total energy ~25-34% for an 18-25% time penalty (MEASURED, one run; vault note 119). It does not close the training gap (L stays ~2.6x).
5. **"Think less" depth gating has a small ceiling.** A perfect oracle gate saves ~19% of modelled work at ~0 quality loss; the best realisable confidence gate ~8-11% (MODELLED, one checkpoint; vault note 120).

## 2. Batch x context scaling (MEASURED, random weights - cost does not depend on weight values; mJ/token, n=5)
| point | L total | T total | T/L total | L above idle | T above idle | T/L above idle |
|---|---|---|---|---|---|---|
| ctx 256 | batch 1 | 25.31 | 55.21 | 2.2x | 11.06 | 26.04 | 2.4x |
| ctx 256 | batch 4 | 6.31 | 20.88 | 3.3x | 2.69 | 12.89 | 4.8x |
| ctx 256 | batch 16 | 3.42 | 17.46 | 5.1x | 2.14 | 11.73 | 5.5x |
| ctx 256 | batch 64 | 2.43 | 15.71 | 6.5x | 1.74 | 11.38 | 6.5x |
| ctx 2048 | batch 1 | 31.48 | 109.30 | 3.5x | 17.29 | 50.32 | 2.9x |
| ctx 2048 | batch 4 | 8.78 | 43.24 | 4.9x | 5.21 | 27.97 | 5.4x |
| ctx 2048 | batch 16 | 3.83 | 41.99 | 11.0x | 2.54 | 28.91 | 11.4x |
| ctx 2048 | batch 64 | 2.47 | 37.84 | 15.3x | 1.79 | 27.58 | 15.4x |
| ctx 16384 | batch 1 | 30.26 | 596.63 | 19.7x | 15.85 | 303.31 | 19.1x |
| ctx 16384 | batch 4 | 8.47 | 251.50 | 29.7x | 4.91 | 177.65 | 36.2x |
| ctx 16384 | batch 16 | 4.16 | does not fit (KV cache > 3.5 GB) | n/a | 2.87 | does not fit (KV cache > 3.5 GB) | n/a |
| ctx 16384 | batch 64 | 2.39 | does not fit (KV cache > 3.5 GB) | n/a | 1.71 | does not fit (KV cache > 3.5 GB) | n/a |

Batch-1 rows are noisy (L std up to 15 mJ/token in the 'ctx 256 batch 1' cell, vs 30.8 +- 0.7 in the quality-matched run); trust the trend, not single cells.

## 3. Sustainability accounting (ESTIMATED; scenario = PROJECTION)
# Sustainability accounting (ESTIMATED; scenario = PROJECTION)

Factors (configs/sustainability.yaml): PUE 1.56, on-site WUE 0.55 L/kWh (IT), off-site 3.14 L/kWh (not in table), grid 0.727 kg CO2/kWh (India CEA FY2023-24, CO2 only). Scenario: 1e+09 queries/day x 2000 bytes.

| system | J/byte | IT kWh per 1M bytes | facility kWh | water on-site L | CO2 kg | projection: facility GWh/yr at 1e9 queries/day | CO2 kt/yr | on-site water ML/yr |
|---|---|---|---|---|---|---|---|---|
| L-6 9M (GraphStepper) - laptop GPU, batch 1 (MEASURED, total board energy) | 0.0308 | 0.00857 | 0.0134 | 0.00471 | 0.00972 | 9.76 | 7.09 | 3.44 |
| Transformer 9M (KV cache, CUDA graph) - laptop GPU, batch 1 (MEASURED, total board energy) | 0.0291 | 0.00807 | 0.0126 | 0.00444 | 0.00915 | 9.19 | 6.68 | 3.24 |
| L-6 ctx 256 | batch 16 (MEASURED, random weights) | 0.00342 | 0.000951 | 0.00148 | 0.000523 | 0.00108 | 1.08 | 0.787 | 0.382 |
| Transformer ctx 256 | batch 16 (MEASURED, random weights) | 0.0175 | 0.00485 | 0.00757 | 0.00267 | 0.0055 | 5.52 | 4.02 | 1.95 |
| L-6 ctx 256 | batch 64 (MEASURED, random weights) | 0.00243 | 0.000674 | 0.00105 | 0.000371 | 0.000764 | 0.767 | 0.558 | 0.271 |
| Transformer ctx 256 | batch 64 (MEASURED, random weights) | 0.0157 | 0.00437 | 0.00681 | 0.0024 | 0.00495 | 4.97 | 3.61 | 1.75 |
| L-6 ctx 2048 | batch 16 (MEASURED, random weights) | 0.00383 | 0.00106 | 0.00166 | 0.000585 | 0.00121 | 1.21 | 0.88 | 0.427 |
| Transformer ctx 2048 | batch 16 (MEASURED, random weights) | 0.042 | 0.0117 | 0.0182 | 0.00641 | 0.0132 | 13.3 | 9.66 | 4.68 |
| L-6 ctx 2048 | batch 64 (MEASURED, random weights) | 0.00247 | 0.000686 | 0.00107 | 0.000377 | 0.000778 | 0.781 | 0.568 | 0.275 |
| Transformer ctx 2048 | batch 64 (MEASURED, random weights) | 0.0378 | 0.0105 | 0.0164 | 0.00578 | 0.0119 | 12 | 8.7 | 4.22 |
| L-6 ctx 16384 | batch 16 (MEASURED, random weights) | 0.00416 | 0.00116 | 0.0018 | 0.000635 | 0.00131 | 1.32 | 0.956 | 0.464 |
| L-6 ctx 16384 | batch 64 (MEASURED, random weights) | 0.00239 | 0.000664 | 0.00104 | 0.000365 | 0.000753 | 0.756 | 0.549 | 0.266 |

Sensitivity (all outputs are linear in J/byte, PUE and the factors): PUE 1.1-2.0 changes facility energy, heat and CO2 by 0.71x-1.28x; on-site water spans 0.3-1.8 L/kWh (6x); CO2 spans 0 (renewable-matched) to 0.9 kg/kWh. A model-side saving of s % lowers every column by exactly s %, so which assumption dominates depends on the column: water is dominated by WUE (6x range), CO2 by the grid factor, energy by PUE (1.8x).

## 4. Clock sweep
Vault note 119: sweet spot 900-1200 MHz; below ~900 MHz the idle floor dominates. The 2100 and 1800 cells ran at the same actual clock (1702 MHz) yet differ ~20% in generation energy, so differences of that size in single cells are noise.

## 5. Rebound / Jevons caution
Cheaper inference tends to raise total usage. Luccioni, Strubell and Crawford (2025, arXiv 2501.16548) argue that efficiency gains alone do not reduce total AI resource use. Per-query savings here are an upper bound on real-world savings unless demand is held fixed; the 1e9 queries/day scenario holds demand fixed by construction.

## 6. Open / not yet measured
- Serving runtime with energy-proportional policies: `15q_serve_s0` running; seeds 1/2 lack L5E checkpoints.
- Training-energy levers (CUDA graph, bf16): job 1741 PENDING.
- Quality-matched comparison uses ONE trained checkpoint per model; the scaling table uses random weights.
- One laptop GPU only; data-centre GPUs have different idle floors and batch-1 behaviour.

## 7. Five numbers for a pitch (each with its caveat)
1. Quality-matched batch 1: energy tie (not a win), L better on quality.
2. Batched serving ctx 256, batch 64: see table 2 (MEASURED, random weights).
3. Long context ctx 16384, batch 1/4: see table 2 - the largest gap; the transformer cannot run batch 16/64.
4. Clock lock 900-1200 MHz: -25..-34% energy, both models.
5. Training: L costs ~2.9x the transformer - a loss.
