---
tags: [experiment, variable-delay, spiking, gru, lstm]
created: 2026-10-06
---
# Variable-delay memory results

Back to [[17 - Variable Delay and Real Query Protocol]] · [[08 - SpikingBrain Process Log]] · [[15 - Temporal Stress Replication]]

## Design and controls
Five seeds, two temporal tasks, six models, 120 training updates/model. Train lengths 8/16/24/32, 64 balanced examples each; independent validation of same sizes. Noise SD .1 in signal/nuisance channels; final query identical across labels. Adam lr .01, clip norm 1; select lowest validation CE checkpoint every 10 updates. No test-based choice. 512 fresh balanced test examples for every seed/task/length/noise condition.

Parameters: {'memoryless': 722, 'analog': 722, 'RNN': 746, 'LIF': 722, 'GRU': 730, 'LSTM': 728}. Matched within about 3.3%, not matched expressivity or operation count. Conventional recurrent layers use native packed-sequence PyTorch; analog and LIF mask padding. LIF decay .95, threshold 1, recurrent spikes, detached reset, sigmoid surrogate derivative, readout final-three-step spike mean. All 811/811 implementation controls passed if no failures in JSON.

![[SpikingBrain Assets/variable-delay-comparison.png]]

## Complete test results
Accuracy mean +/- sample SD; CPU time is mean of per-seed medians, 512-input batch, 3 warmups + 10 measured runs. All figures show full observed seed ranges.

| Task | Model | Length | Signal noise SD | Accuracy | Batch ms |
|---|---|---:|---:|---|---:|
| delayed_recall | memoryless | 12 | 0.1 | 50.00 +/- 0.00% | 0.401 |
| delayed_recall | memoryless | 12 | 0.2 | 50.00 +/- 0.00% | 0.495 |
| delayed_recall | memoryless | 32 | 0.1 | 50.00 +/- 0.00% | 0.456 |
| delayed_recall | memoryless | 32 | 0.2 | 50.00 +/- 0.00% | 0.463 |
| delayed_recall | memoryless | 48 | 0.1 | 50.00 +/- 0.00% | 0.381 |
| delayed_recall | memoryless | 48 | 0.2 | 50.00 +/- 0.00% | 0.404 |
| delayed_recall | memoryless | 96 | 0.1 | 50.00 +/- 0.00% | 0.544 |
| delayed_recall | memoryless | 96 | 0.2 | 50.00 +/- 0.00% | 0.498 |
| delayed_recall | analog | 12 | 0.1 | 99.02 +/- 0.37% | 4.705 |
| delayed_recall | analog | 12 | 0.2 | 87.07 +/- 1.25% | 5.091 |
| delayed_recall | analog | 32 | 0.1 | 74.84 +/- 2.74% | 13.351 |
| delayed_recall | analog | 32 | 0.2 | 62.46 +/- 1.00% | 13.027 |
| delayed_recall | analog | 48 | 0.1 | 62.03 +/- 2.36% | 20.199 |
| delayed_recall | analog | 48 | 0.2 | 55.51 +/- 2.79% | 22.391 |
| delayed_recall | analog | 96 | 0.1 | 50.39 +/- 2.53% | 49.002 |
| delayed_recall | analog | 96 | 0.2 | 49.02 +/- 1.86% | 44.611 |
| delayed_recall | RNN | 12 | 0.1 | 80.00 +/- 44.72% | 3.168 |
| delayed_recall | RNN | 12 | 0.2 | 80.00 +/- 44.72% | 2.822 |
| delayed_recall | RNN | 32 | 0.1 | 100.00 +/- 0.00% | 6.422 |
| delayed_recall | RNN | 32 | 0.2 | 100.00 +/- 0.00% | 6.879 |
| delayed_recall | RNN | 48 | 0.1 | 100.00 +/- 0.00% | 10.489 |
| delayed_recall | RNN | 48 | 0.2 | 100.00 +/- 0.00% | 11.041 |
| delayed_recall | RNN | 96 | 0.1 | 100.00 +/- 0.00% | 21.060 |
| delayed_recall | RNN | 96 | 0.2 | 99.92 +/- 0.17% | 21.027 |
| delayed_recall | LIF | 12 | 0.1 | 100.00 +/- 0.00% | 8.980 |
| delayed_recall | LIF | 12 | 0.2 | 99.84 +/- 0.35% | 10.831 |
| delayed_recall | LIF | 32 | 0.1 | 89.41 +/- 23.67% | 23.694 |
| delayed_recall | LIF | 32 | 0.2 | 89.49 +/- 23.39% | 26.003 |
| delayed_recall | LIF | 48 | 0.1 | 89.14 +/- 24.28% | 39.920 |
| delayed_recall | LIF | 48 | 0.2 | 89.22 +/- 24.11% | 36.301 |
| delayed_recall | LIF | 96 | 0.1 | 90.43 +/- 21.40% | 76.964 |
| delayed_recall | LIF | 96 | 0.2 | 90.00 +/- 22.36% | 78.554 |
| delayed_recall | GRU | 12 | 0.1 | 100.00 +/- 0.00% | 7.628 |
| delayed_recall | GRU | 12 | 0.2 | 100.00 +/- 0.00% | 7.950 |
| delayed_recall | GRU | 32 | 0.1 | 100.00 +/- 0.00% | 18.959 |
| delayed_recall | GRU | 32 | 0.2 | 100.00 +/- 0.00% | 18.899 |
| delayed_recall | GRU | 48 | 0.1 | 100.00 +/- 0.00% | 28.107 |
| delayed_recall | GRU | 48 | 0.2 | 100.00 +/- 0.00% | 29.227 |
| delayed_recall | GRU | 96 | 0.1 | 100.00 +/- 0.00% | 56.730 |
| delayed_recall | GRU | 96 | 0.2 | 100.00 +/- 0.00% | 58.507 |
| delayed_recall | LSTM | 12 | 0.1 | 100.00 +/- 0.00% | 6.808 |
| delayed_recall | LSTM | 12 | 0.2 | 100.00 +/- 0.00% | 7.285 |
| delayed_recall | LSTM | 32 | 0.1 | 100.00 +/- 0.00% | 18.146 |
| delayed_recall | LSTM | 32 | 0.2 | 100.00 +/- 0.00% | 18.317 |
| delayed_recall | LSTM | 48 | 0.1 | 100.00 +/- 0.00% | 28.223 |
| delayed_recall | LSTM | 48 | 0.2 | 99.96 +/- 0.09% | 26.849 |
| delayed_recall | LSTM | 96 | 0.1 | 100.00 +/- 0.00% | 50.475 |
| delayed_recall | LSTM | 96 | 0.2 | 100.00 +/- 0.00% | 49.883 |
| pulse_order | memoryless | 12 | 0.1 | 50.00 +/- 0.00% | 0.418 |
| pulse_order | memoryless | 12 | 0.2 | 50.00 +/- 0.00% | 0.429 |
| pulse_order | memoryless | 32 | 0.1 | 50.00 +/- 0.00% | 0.479 |
| pulse_order | memoryless | 32 | 0.2 | 50.00 +/- 0.00% | 0.437 |
| pulse_order | memoryless | 48 | 0.1 | 50.00 +/- 0.00% | 0.461 |
| pulse_order | memoryless | 48 | 0.2 | 50.00 +/- 0.00% | 0.473 |
| pulse_order | memoryless | 96 | 0.1 | 50.00 +/- 0.00% | 0.449 |
| pulse_order | memoryless | 96 | 0.2 | 50.00 +/- 0.00% | 0.471 |
| pulse_order | analog | 12 | 0.1 | 83.44 +/- 1.11% | 4.604 |
| pulse_order | analog | 12 | 0.2 | 68.79 +/- 1.55% | 4.947 |
| pulse_order | analog | 32 | 0.1 | 88.71 +/- 1.87% | 13.493 |
| pulse_order | analog | 32 | 0.2 | 75.31 +/- 1.00% | 12.758 |
| pulse_order | analog | 48 | 0.1 | 87.93 +/- 1.92% | 20.498 |
| pulse_order | analog | 48 | 0.2 | 74.34 +/- 1.57% | 18.484 |
| pulse_order | analog | 96 | 0.1 | 79.06 +/- 2.93% | 47.286 |
| pulse_order | analog | 96 | 0.2 | 70.43 +/- 3.13% | 47.265 |
| pulse_order | RNN | 12 | 0.1 | 99.92 +/- 0.11% | 3.224 |
| pulse_order | RNN | 12 | 0.2 | 95.27 +/- 1.38% | 2.836 |
| pulse_order | RNN | 32 | 0.1 | 99.69 +/- 0.17% | 6.599 |
| pulse_order | RNN | 32 | 0.2 | 87.46 +/- 3.08% | 6.518 |
| pulse_order | RNN | 48 | 0.1 | 98.09 +/- 1.60% | 10.336 |
| pulse_order | RNN | 48 | 0.2 | 78.71 +/- 1.59% | 9.894 |
| pulse_order | RNN | 96 | 0.1 | 93.40 +/- 3.69% | 20.427 |
| pulse_order | RNN | 96 | 0.2 | 66.72 +/- 5.82% | 17.813 |
| pulse_order | LIF | 12 | 0.1 | 79.34 +/- 10.71% | 9.739 |
| pulse_order | LIF | 12 | 0.2 | 79.77 +/- 8.91% | 9.073 |
| pulse_order | LIF | 32 | 0.1 | 58.67 +/- 6.80% | 25.547 |
| pulse_order | LIF | 32 | 0.2 | 57.42 +/- 4.05% | 24.766 |
| pulse_order | LIF | 48 | 0.1 | 53.24 +/- 3.39% | 38.868 |
| pulse_order | LIF | 48 | 0.2 | 53.44 +/- 3.86% | 34.786 |
| pulse_order | LIF | 96 | 0.1 | 53.40 +/- 1.99% | 77.597 |
| pulse_order | LIF | 96 | 0.2 | 51.64 +/- 1.79% | 72.063 |
| pulse_order | GRU | 12 | 0.1 | 99.45 +/- 0.70% | 7.836 |
| pulse_order | GRU | 12 | 0.2 | 97.23 +/- 1.53% | 7.157 |
| pulse_order | GRU | 32 | 0.1 | 99.45 +/- 0.35% | 19.281 |
| pulse_order | GRU | 32 | 0.2 | 89.53 +/- 1.53% | 18.483 |
| pulse_order | GRU | 48 | 0.1 | 94.96 +/- 3.03% | 27.757 |
| pulse_order | GRU | 48 | 0.2 | 79.65 +/- 1.76% | 28.323 |
| pulse_order | GRU | 96 | 0.1 | 79.57 +/- 7.49% | 54.925 |
| pulse_order | GRU | 96 | 0.2 | 65.66 +/- 3.64% | 59.856 |
| pulse_order | LSTM | 12 | 0.1 | 99.84 +/- 0.21% | 7.338 |
| pulse_order | LSTM | 12 | 0.2 | 98.55 +/- 0.56% | 6.966 |
| pulse_order | LSTM | 32 | 0.1 | 98.48 +/- 0.52% | 19.185 |
| pulse_order | LSTM | 32 | 0.2 | 90.39 +/- 1.38% | 18.069 |
| pulse_order | LSTM | 48 | 0.1 | 85.00 +/- 4.08% | 26.581 |
| pulse_order | LSTM | 48 | 0.2 | 77.70 +/- 5.74% | 26.692 |
| pulse_order | LSTM | 96 | 0.1 | 57.50 +/- 11.16% | 50.953 |
| pulse_order | LSTM | 96 | 0.2 | 56.56 +/- 5.95% | 52.241 |

## Interpretation limits
Lengths 48/96 are outside training; noise .2 is a noise shift. Memoryless chance performance is expected from identical final inputs, not a competitive sequence model. Analog projection has many parameters but is still a linear leaky integrator. Best-loss checkpoints differ from earlier best-accuracy short-horizon experiment, so improvements cannot be uniquely attributed to variable lengths. Shared train/test split sizes and optimizer budgets improve fairness but do not prove one model family generally superior. CPU dense spiking simulation is not neuromorphic power measurement.

## Reproduce and saved states
- D:\AI\SpikingBrain-study\experiments\run_variable_delay.py
- [[SpikingBrain Assets/variable-delay-results.json]]
- Checkpoints variable-delay-seed42.pt through variable-delay-seed46.pt under D:\AI\SpikingBrain-study\experiments.
- `py -3.12 -B 'D:\AI\SpikingBrain-study\experiments\run_variable_delay.py'`
- PyTorch 2.14.0+cpu, CPU threads 2. New temp/cache on D:, bytecode disabled. Runtime 469.7 s. Source SHA256 in JSON.
