# Learned weak-signal memory results

[[34 - Weak Benchmark Improvement Protocol]] · [[08 - SpikingBrain Process Log]]

Five seeds, 15 fits; independent first and second signs; additive noise also contaminates real pulses. This removes the old shortcut where the second event always revealed the first sign. Fresh 512-example tests at each of18 conditions per model/seed. Shared192training and192validation examples,120updates, checkpoints chosen by validation loss. RNN746/LIF722/gated148parameters: equal update budgets, not matched architecture or compute. Training noise .1/.2; .3 and lengths96/192 stress generalization. This harder task differs from all previous memory runs, so use the matched rows here rather than claiming historical accuracy changes.

The candidate learns an input-strength write gate, binary threshold-neuron features, and an analog persistent synaptic trace protected by occupancy. It is a hybrid purpose-built architecture, not a fully spiking LIF model or the full SpikingBrain checkpoint. It does not receive true event times. Long noisy intervals can cross the gate before a true event: protecting a mistaken write preserves the mistake. Missing-event confidence is not reliable. Do not replace the contract-based old detector where incomplete-input rejection matters.

Raw accuracy counts every binary prediction. Coverage and correct fraction use a confidence threshold selected on validation for95% accepted quality. That validation target does not hold automatically at longer delays/heavier noise. The old protected model uses its existing exactly-two-events validity contract; raw accuracy for it includes invalid default-feature predictions and is not a usable answered score. Correct fraction penalizes abstentions and is the relevant comparison for that control.

| Model | Length | Amplitude | Noise SD | Raw accuracy ± sample SD | Coverage | Correct fraction of all inputs |
|---|---:|---:|---:|---:|---:|---:|
| RNN | 32 | 1.0 | 0.1 | 61.17 ± 19.31% | 1.45% | 1.33% |
| RNN | 32 | 1.0 | 0.2 | 57.15 ± 14.47% | 2.38% | 1.99% |
| RNN | 32 | 1.0 | 0.3 | 55.90 ± 9.16% | 3.98% | 2.89% |
| RNN | 32 | 0.6 | 0.1 | 59.34 ± 17.49% | 0.66% | 0.62% |
| RNN | 32 | 0.6 | 0.2 | 57.38 ± 11.35% | 1.68% | 1.37% |
| RNN | 32 | 0.6 | 0.3 | 53.83 ± 8.94% | 3.05% | 2.03% |
| RNN | 96 | 1.0 | 0.1 | 55.43 ± 11.40% | 0.66% | 0.55% |
| RNN | 96 | 1.0 | 0.2 | 53.55 ± 8.57% | 1.45% | 0.98% |
| RNN | 96 | 1.0 | 0.3 | 51.45 ± 2.62% | 2.85% | 1.60% |
| RNN | 96 | 0.6 | 0.1 | 52.97 ± 9.19% | 0.31% | 0.27% |
| RNN | 96 | 0.6 | 0.2 | 52.03 ± 5.74% | 1.09% | 0.70% |
| RNN | 96 | 0.6 | 0.3 | 51.02 ± 5.44% | 2.66% | 1.76% |
| RNN | 192 | 1.0 | 0.1 | 53.79 ± 7.21% | 1.60% | 0.70% |
| RNN | 192 | 1.0 | 0.2 | 53.12 ± 7.41% | 4.18% | 1.88% |
| RNN | 192 | 1.0 | 0.3 | 50.66 ± 3.11% | 6.25% | 2.81% |
| RNN | 192 | 0.6 | 0.1 | 52.11 ± 6.37% | 1.09% | 0.70% |
| RNN | 192 | 0.6 | 0.2 | 51.25 ± 5.15% | 3.71% | 1.68% |
| RNN | 192 | 0.6 | 0.3 | 50.90 ± 3.23% | 5.86% | 2.77% |
| LIF_membrane | 32 | 1.0 | 0.1 | 67.42 ± 10.72% | 7.19% | 6.80% |
| LIF_membrane | 32 | 1.0 | 0.2 | 65.23 ± 8.39% | 10.62% | 9.45% |
| LIF_membrane | 32 | 1.0 | 0.3 | 63.09 ± 6.21% | 14.26% | 11.33% |
| LIF_membrane | 32 | 0.6 | 0.1 | 63.52 ± 6.64% | 3.09% | 2.73% |
| LIF_membrane | 32 | 0.6 | 0.2 | 60.20 ± 4.66% | 5.35% | 4.69% |
| LIF_membrane | 32 | 0.6 | 0.3 | 59.26 ± 3.61% | 10.94% | 8.01% |
| LIF_membrane | 96 | 1.0 | 0.1 | 53.01 ± 8.91% | 9.22% | 7.23% |
| LIF_membrane | 96 | 1.0 | 0.2 | 54.02 ± 5.85% | 10.59% | 7.07% |
| LIF_membrane | 96 | 1.0 | 0.3 | 53.44 ± 2.34% | 13.24% | 7.85% |
| LIF_membrane | 96 | 0.6 | 0.1 | 53.91 ± 6.41% | 7.54% | 5.66% |
| LIF_membrane | 96 | 0.6 | 0.2 | 51.80 ± 3.10% | 9.65% | 5.86% |
| LIF_membrane | 96 | 0.6 | 0.3 | 51.48 ± 3.44% | 12.89% | 7.19% |
| LIF_membrane | 192 | 1.0 | 0.1 | 50.23 ± 2.60% | 14.26% | 8.01% |
| LIF_membrane | 192 | 1.0 | 0.2 | 51.60 ± 1.18% | 14.77% | 7.70% |
| LIF_membrane | 192 | 1.0 | 0.3 | 49.96 ± 1.02% | 16.99% | 8.98% |
| LIF_membrane | 192 | 0.6 | 0.1 | 51.64 ± 2.43% | 14.41% | 7.93% |
| LIF_membrane | 192 | 0.6 | 0.2 | 50.86 ± 0.86% | 15.51% | 8.05% |
| LIF_membrane | 192 | 0.6 | 0.3 | 50.62 ± 0.54% | 16.80% | 8.67% |
| gated | 32 | 1.0 | 0.1 | 100.00 ± 0.00% | 100.00% | 100.00% |
| gated | 32 | 1.0 | 0.2 | 95.98 ± 1.30% | 97.19% | 94.30% |
| gated | 32 | 1.0 | 0.3 | 78.95 ± 3.37% | 93.16% | 75.31% |
| gated | 32 | 0.6 | 0.1 | 96.02 ± 1.39% | 94.61% | 92.54% |
| gated | 32 | 0.6 | 0.2 | 87.93 ± 1.94% | 87.38% | 79.88% |
| gated | 32 | 0.6 | 0.3 | 71.45 ± 2.18% | 86.68% | 64.53% |
| gated | 96 | 1.0 | 0.1 | 99.96 ± 0.09% | 99.96% | 99.96% |
| gated | 96 | 1.0 | 0.2 | 84.30 ± 4.26% | 84.30% | 74.73% |
| gated | 96 | 1.0 | 0.3 | 63.36 ± 0.63% | 85.04% | 55.82% |
| gated | 96 | 0.6 | 0.1 | 97.85 ± 1.11% | 93.95% | 93.12% |
| gated | 96 | 0.6 | 0.2 | 76.21 ± 3.00% | 76.45% | 62.58% |
| gated | 96 | 0.6 | 0.3 | 58.01 ± 1.37% | 83.75% | 49.41% |
| gated | 192 | 1.0 | 0.1 | 99.92 ± 0.17% | 98.09% | 98.09% |
| gated | 192 | 1.0 | 0.2 | 69.14 ± 4.07% | 73.63% | 55.47% |
| gated | 192 | 1.0 | 0.3 | 55.47 ± 0.79% | 82.89% | 46.80% |
| gated | 192 | 0.6 | 0.1 | 96.80 ± 2.78% | 86.84% | 86.21% |
| gated | 192 | 0.6 | 0.2 | 64.61 ± 3.62% | 68.48% | 48.36% |
| gated | 192 | 0.6 | 0.3 | 53.83 ± 2.31% | 82.11% | 45.43% |
| old_protected | 32 | 1.0 | 0.1 | 99.22 ± 0.57% | 96.52% | 96.52% |
| old_protected | 32 | 1.0 | 0.2 | 93.63 ± 2.23% | 75.00% | 74.96% |
| old_protected | 32 | 1.0 | 0.3 | 86.99 ± 0.67% | 51.68% | 49.26% |
| old_protected | 32 | 0.6 | 0.1 | 53.52 ± 5.80% | 1.99% | 1.99% |
| old_protected | 32 | 0.6 | 0.2 | 59.45 ± 6.09% | 5.66% | 5.59% |
| old_protected | 32 | 0.6 | 0.3 | 63.63 ± 2.28% | 17.54% | 13.98% |
| old_protected | 96 | 1.0 | 0.1 | 99.22 ± 0.46% | 96.52% | 96.52% |
| old_protected | 96 | 1.0 | 0.2 | 92.77 ± 2.33% | 72.77% | 72.54% |
| old_protected | 96 | 1.0 | 0.3 | 81.80 ± 1.48% | 35.04% | 31.09% |
| old_protected | 96 | 0.6 | 0.1 | 54.80 ± 7.80% | 2.70% | 2.70% |
| old_protected | 96 | 0.6 | 0.2 | 60.00 ± 4.65% | 5.86% | 5.66% |
| old_protected | 96 | 0.6 | 0.3 | 63.40 ± 1.39% | 21.68% | 15.27% |
| old_protected | 192 | 1.0 | 0.1 | 98.98 ± 0.77% | 96.45% | 96.45% |
| old_protected | 192 | 1.0 | 0.2 | 93.05 ± 2.29% | 72.66% | 72.58% |
| old_protected | 192 | 1.0 | 0.3 | 77.27 ± 5.92% | 22.70% | 19.49% |
| old_protected | 192 | 0.6 | 0.1 | 54.10 ± 6.23% | 2.07% | 2.07% |
| old_protected | 192 | 0.6 | 0.2 | 59.38 ± 3.89% | 6.29% | 5.74% |
| old_protected | 192 | 0.6 | 0.3 | 59.57 ± 1.04% | 23.95% | 14.57% |

## Incomplete-input diagnostics

These inputs have no valid first-event label. Rejection is the intended behavior; binary accuracy is meaningless.

| Model | Condition | Rejection |
|---|---|---:|
| RNN | missing_first | 98.95% |
| RNN | empty | 99.22% |
| LIF_membrane | missing_first | 90.94% |
| LIF_membrane | empty | 91.33% |
| gated | missing_first | 39.38% |
| gated | empty | 38.83% |

15/15 finite gradient checks;270 saved learned-model test scores reproduced exactly. Additional45 reset/padding/finite checks passed across saved15models. All360matched test-condition rows plus30diagnostics, training histories and confidence choices are in [[SpikingBrain Assets/weak-memory-results.json]]. Runtime 187.6s. CPU only; no energy or biological fidelity claim.
