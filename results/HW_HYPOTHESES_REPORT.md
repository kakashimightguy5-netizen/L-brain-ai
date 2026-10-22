# SPARK hardware hypotheses H1–H10: report (draft 1, 2026-10-10)

**Labels.** MEASURED = model activity or quality measured on held-out Python code. MODELLED = computed from that activity with the stated model. ESTIMATED = needs a per-operation energy or a hardware datasheet figure.
**Fitting rule.** Tables, hot sets, predictors and thresholds are fitted on CAL (first half of the held-out file). Every number below is measured on EVAL (second half).
**Models.** L-6 9M seed 0 (50,000 bytes per split) and L-Edge (d128, 4 layers, 3,000 steps). Seeds 1 and 2 of L-6 are queued on the laptop (`hwh_traces.py`).
**Not finished yet.** H4 and H6 need 3-seed training on the laptop (jobs 1600–1626, queued). Their rows say PENDING; nothing is claimed for them.
Everything is in `results/hw_hypotheses/` (CSV + PNG).

## Verdict table

| H | Verdict | Key MEASURED / MODELLED fact |
|---|---|---|
| H1 energy proportionality | **FAIL as built, fix queued** | Work per byte does not follow difficulty: Spearman(events, loss) 0.006 on L-6 and 0.19 on L-Edge; hardest vs easiest decile work 1.00×. About 92% of per-byte work is dense matrices (qkv, o, inp, f1). |
| H2 prefetch | **DROP (L-6), borderline (L-Edge)** | Best honest predictor on L-6: hit 51%, waste 1.07× (needs ≥70% and ≤1×). L-Edge linear predictor: hit 70.1%, waste 1.03× — misses the waste rule by 0.03. Jaccard between consecutive active sets only 0.14 (L-6) and 0.16 (L-Edge). |
| H3 hot neurons | **DROP** | Top 10% of FFN neurons cover 21% of events (L-6) and 17% (L-Edge); 50% cover 68% / 63%. No small fixed hot cache. |
| H4 spike groups | **PENDING training; trace side says NO-GO for fetch** | Naive grouping of the current model raises fetched columns 2.6× at G=4 and 5.2× at G=8 for 11% / 24% fewer index entries. Training with groups (3 seeds) pending. |
| H5 address-sorted queue | **KEEP (small)** | Sorted vs unsorted SDRAM cycles: 0.99 (L-6 f2, 8-bit), 0.96 (4-bit), 0.90 (state→out, 4-bit); 0.81 / 0.76 on L-Edge. Largest saving is event-driven vs dense (f2 at 5% of dense cycles). |
| H6 block-scaled 4-bit QAT | **PENDING training** | Post-training 4-bit on L-Edge (no QAT): mx4 pow2 B32 +2.2% bits/byte vs int8, per-channel int4 +5.6%. QAT results pending. |
| H7 precision on demand | **DROP** | Confidence-gated 4/8-bit is no better than random at the same 4-bit share (mx4: 0.9935 at 40% vs random 0.9908 at 25% / 0.9947 at 50%). The previous byte's confidence does not tell which bytes tolerate 4 bits. |
| H8 multi-stream | **BUILD (if on-chip state allows)** | Weight traffic per byte falls to 0.27× at 4 streams, 0.14× at 8, 0.04× at 32. On-chip state grows linearly: L-Edge 36.5 KB per stream. |
| H9 clock gating | **KEEP, fine-grained only** | Idle share of the FFN-f2 unit: 90% per neuron, 47% for 8-neuron blocks, 13% for 32-neuron blocks (L-Edge). Whole-layer idleness is 0.5%. Gate-able share of all work ≤ 15.7% (per neuron), ≤ 8.2% (8-blocks). |
| H10 analog noise | **KEEP as optional** | 5% weight variation costs +0.2% bits/byte, 10% costs +1.3%. Read noise is worse: 5% costs +3.3%, 10% +13.7%. Noise-aware fine-tuning gave no clear gain over a matched no-noise fine-tune (1 seed). |

## Details

### H1 — energy proportionality (MEASURED events, MODELLED work, ESTIMATED energy)
- Spike events per byte average 1,381 on L-6 (p5–p95: 1,173–1,573) and 344 on L-Edge. The event count is nearly constant and slightly lower on harder bytes.
- Modelled work is 79% of an all-dense implementation on both models, and flat across difficulty deciles.
- ESTIMATED chip energy (45 nm per-operation energies, Horowitz 2014): about 9.7 µJ per byte for L-6 (12.2 µJ if all dense) and 0.89 µJ for L-Edge. These are order-of-magnitude values, not Tang Nano numbers.
- Reading: L is sparse where it spikes (FFN 5%, state 55% non-zero on L-6) but the dense projections dominate, so cost does not follow difficulty. The effort-gated runtime (`serve/`) is the fix for this; its results are in the efficiency-stack report.

### H2 — prefetch
- On L-6 the predictors tried were: last step, union of the last 2/4/8 steps, a byte-conditioned table, and a linear model on the previous state spikes plus the input byte. None reaches 70% hit at ≤1× waste.
- On L-Edge the linear predictor reaches 70.1% / 1.03×. This is a borderline result on one seed, so it is reported as NOT a go.

### H3 — hot neurons
- Never-firing FFN neurons: 0% on both CAL and EVAL, so there is nothing to prune either.
- Coverage curves are close to the diagonal (uniform). The 4-bit cache for 10% of L-Edge neurons would be 6.5 KB and would serve only 17% of events.

### H4 — grouped spikes (trace side only, current ungrouped model)
- G=4: 84.4 active groups per byte vs 101.8 spikes (L-Edge); fetching whole groups would read 337.6 columns, a 2.3× waste.
- Real answer needs the trained `SPIKE_GROUPS` flag; jobs are queued.

### H5 — address-sorted queue (MODELLED, assumptions in script docstring)
- Assumptions: SDRAM row 2 KB, burst 32 B, 6 activation and 8 burst cycles.
- Spikes scanned in index order from a bitmap are already sorted, so a sorter is only needed when spikes arrive out of order.

### H7 — precision on demand (MEASURED, exact sequential simulation on L-Edge, PTQ weights)
- float32 0.9848, int8 0.9857, mx4 pow2 B32 1.0077, int4 per-channel 1.0405 bits/byte.
- Gated vs random at equal share: no gain, for both 4-bit formats and all three shares.
- The useful by-product: block-scaled 4-bit costs only +2.2% without any QAT.

### H8 — multi-stream (MEASURED activity, MODELLED traffic)
- Dense weights are read once per step for all B streams (traffic per byte ∝ 1/B). Event weights are read for the union of the streams' active columns, so they grow with B but sub-linearly: union vs sum of active columns is 0.79 at B=2 and 0.13 at B=32 on L-Edge. Net weight traffic per byte is 0.27× at B=4 and 0.14× at B=8 (L-6 and L-Edge agree within 1%).
- Cost: per-stream state (membrane, conv buffer, fast-weight memory) must stay on chip: 36.5 KB per stream for L-Edge, 200 KB per stream for L-6. Hippocampus tables per stream are not counted.

### H9 — clock gating (MEASURED idle share)
- State→out unit: 53% idle per neuron but only 8.7% for 4-neuron blocks, so gating there is not useful.

### H10 — analog noise (MEASURED quality under SIMULATED noise; noise model is an assumption)
- Pretrained L-Edge, bits/byte (clean 1.2764): weight variation 1% 1.2767, 2% 1.2768, 5% 1.2795, 10% 1.2934, 20% 1.3551. Read noise 1% 1.2772, 2% 1.2814, 5% 1.3190, 10% 1.4512. Five chips per weight-variation point, three runs per read-noise point.
- Fine-tuning 500 steps hurt the clean model by itself (1.2885 no-noise control, 1.2853 noise-aware). At 5% weight variation: control 1.2918, noise-aware 1.2880. The difference is within about 2σ of chip-to-chip spread. Single seed, so no conclusion.

## BUILD / KEEP / DROP

| Item | Decision |
|---|---|
| Event-driven out and f2 projections | BUILD (already the design) |
| Multi-stream interleaving (H8) | BUILD if the on-chip budget allows 4–8 streams |
| Address-sorted queue (H5) | KEEP, low priority |
| Fine-grained clock gating (H9) | KEEP, per-neuron or 4-neuron blocks only |
| Analog-friendly weights (H10) | KEEP as an option, weight noise only |
| Hot-neuron cache (H3) | DROP |
| Prefetch predictor (H2) | DROP |
| Precision on demand (H7) | DROP |
| Spike groups (H4), block 4-bit QAT (H6) | PENDING 3-seed training |

## Caveat
These are single-checkpoint trace results (L-6 s0, L-Edge s0). The s1/s2 traces are queued; conclusions that sit close to a threshold (H2 L-Edge) should not be read as established.
