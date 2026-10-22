---
tags: [experiment, temporal-stress, replication]
created: 2026-10-05
---
# Temporal stress replication

Back to [[14 - Temporal Stress Results]] · [[13 - Extended Prototype Results]] · [[12 - Extended Prototype Protocol]]

## Why we replicated
Seed-42 exploration suggested long-delay recall retention for LIF but poor pulse-order retention. We replicated the entire fixed grid across seeds 42-46 rather than selecting the favourable example. Exact original model settings, original independent train/validation datasets and best-validation selection recreated; all 40/40 original short-task test scores matched. No architecture tuning from stress scores. Five-seed stress is still exploratory because its design followed primary results.

![[SpikingBrain Assets/temporal-stress-five-seed.png]]

## Full results
512 fresh balanced inputs per condition/seed. Trained horizon 12; tested lengths 12/24/48/96; additional signal-channel noise SD 0 or .2. There are 320 seed/model/task/condition evaluations; all logits finite: True.

| Task | Length | Signal noise SD | Model | Accuracy mean +/- sample SD |
|---|---:|---:|---|---:|
| delayed_recall | 12 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | analog | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | RNN | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.0 | LIF | 100.00 +/- 0.00% |
| delayed_recall | 12 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 12 | 0.2 | analog | 84.38 +/- 3.36% |
| delayed_recall | 12 | 0.2 | RNN | 98.48 +/- 3.41% |
| delayed_recall | 12 | 0.2 | LIF | 94.61 +/- 4.52% |
| delayed_recall | 24 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 24 | 0.0 | analog | 68.55 +/- 19.96% |
| delayed_recall | 24 | 0.0 | RNN | 59.10 +/- 53.98% |
| delayed_recall | 24 | 0.0 | LIF | 78.71 +/- 25.41% |
| delayed_recall | 24 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 24 | 0.2 | analog | 66.80 +/- 2.06% |
| delayed_recall | 24 | 0.2 | RNN | 52.54 +/- 47.38% |
| delayed_recall | 24 | 0.2 | LIF | 74.80 +/- 24.37% |
| delayed_recall | 48 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 48 | 0.0 | analog | 50.16 +/- 0.35% |
| delayed_recall | 48 | 0.0 | RNN | 68.71 +/- 45.53% |
| delayed_recall | 48 | 0.0 | LIF | 69.80 +/- 27.57% |
| delayed_recall | 48 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 48 | 0.2 | analog | 55.20 +/- 1.86% |
| delayed_recall | 48 | 0.2 | RNN | 70.62 +/- 41.63% |
| delayed_recall | 48 | 0.2 | LIF | 71.80 +/- 25.01% |
| delayed_recall | 96 | 0.0 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.0 | analog | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.0 | RNN | 20.27 +/- 44.57% |
| delayed_recall | 96 | 0.0 | LIF | 69.77 +/- 27.60% |
| delayed_recall | 96 | 0.2 | memoryless | 50.00 +/- 0.00% |
| delayed_recall | 96 | 0.2 | analog | 50.74 +/- 4.08% |
| delayed_recall | 96 | 0.2 | RNN | 31.56 +/- 39.58% |
| delayed_recall | 96 | 0.2 | LIF | 70.31 +/- 26.69% |
| pulse_order | 12 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 12 | 0.0 | analog | 100.00 +/- 0.00% |
| pulse_order | 12 | 0.0 | RNN | 99.96 +/- 0.09% |
| pulse_order | 12 | 0.0 | LIF | 100.00 +/- 0.00% |
| pulse_order | 12 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 12 | 0.2 | analog | 66.72 +/- 1.53% |
| pulse_order | 12 | 0.2 | RNN | 98.16 +/- 0.86% |
| pulse_order | 12 | 0.2 | LIF | 96.60 +/- 0.74% |
| pulse_order | 24 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 24 | 0.0 | analog | 92.19 +/- 5.29% |
| pulse_order | 24 | 0.0 | RNN | 83.87 +/- 16.46% |
| pulse_order | 24 | 0.0 | LIF | 33.95 +/- 26.03% |
| pulse_order | 24 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 24 | 0.2 | analog | 57.85 +/- 2.63% |
| pulse_order | 24 | 0.2 | RNN | 78.63 +/- 16.20% |
| pulse_order | 24 | 0.2 | LIF | 36.33 +/- 18.68% |
| pulse_order | 48 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 48 | 0.0 | analog | 50.39 +/- 0.60% |
| pulse_order | 48 | 0.0 | RNN | 34.96 +/- 40.57% |
| pulse_order | 48 | 0.0 | LIF | 43.01 +/- 24.50% |
| pulse_order | 48 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 48 | 0.2 | analog | 51.95 +/- 2.88% |
| pulse_order | 48 | 0.2 | RNN | 39.02 +/- 35.19% |
| pulse_order | 48 | 0.2 | LIF | 45.51 +/- 22.05% |
| pulse_order | 96 | 0.0 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.0 | analog | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.0 | RNN | 53.36 +/- 37.23% |
| pulse_order | 96 | 0.0 | LIF | 47.70 +/- 17.07% |
| pulse_order | 96 | 0.2 | memoryless | 50.00 +/- 0.00% |
| pulse_order | 96 | 0.2 | analog | 50.12 +/- 3.38% |
| pulse_order | 96 | 0.2 | RNN | 54.49 +/- 28.29% |
| pulse_order | 96 | 0.2 | LIF | 50.08 +/- 13.50% |

## Limits
Delay/noise shift is out-of-distribution. Results are sensitive to initialization, validation-selected epoch and fixed neuron/readout choices. This does not prove general SNN superiority, language capability or hardware energy savings. The conventional RNN is a basic tanh network, not a tuned GRU/LSTM. Models have different parameter counts. Report variability, both tasks and both noise settings together.

## Records
- [[SpikingBrain Assets/temporal-stress-five-seed.json]]
- D:\AI\SpikingBrain-study\experiments\run_temporal_replication.py
- D:\AI\SpikingBrain-study\experiments\temporal-five-seed-states.pt
- Rerun `py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\run_temporal_replication.py'`
- Runtime 37.7 s. All new artifacts/cache/temp on D:.

## Plot review
The figure uses means with full observed seed ranges; the table reports sample SD. Replaced clipped SD bars with full ranges so the plot exposes every seed's variation. Numerical results unchanged.

