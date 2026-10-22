# State tracing and neuron ablation results

[[25 - State Tracing and Language Challenge Protocol]] · [[23 - Longer Order Memory Results]] · [[08 - SpikingBrain Process Log]]

## Frozen-state trace
Previously saved five-seed checkpoints, independent balanced fixed-position sequences, train256/test512. StandardScaler fit on probe training only; logistic C1 fixed. Original network weights untouched. These probes use privileged diagnostic labels, not deployable readouts. Pulses at4/40, query95. Step39 is before second signal and directly tests first-sign retention. After second pulse, decoder can infer first sign from the opposite second sign; high post-second accuracy does not prove original memory survived. Fixed positions differ from original random-position task. Paired state-distance statistics and original readout accuracy in [[SpikingBrain Assets/state-trace-results.json]].

![[SpikingBrain Assets/state-trace-comparison.png]]

| Frozen model | Representation | Step | Probe mean +/- SD |
|---|---|---:|---|
| RNN | hidden | 5 | 100.00 +/- 0.00% |
| RNN | hidden | 39 | 98.59 +/- 1.40% |
| RNN | hidden | 41 | 100.00 +/- 0.00% |
| RNN | hidden | 95 | 98.98 +/- 1.53% |
| LIF_spikes | membrane | 5 | 100.00 +/- 0.00% |
| LIF_spikes | membrane | 39 | 81.48 +/- 12.92% |
| LIF_spikes | membrane | 41 | 99.92 +/- 0.11% |
| LIF_spikes | membrane | 95 | 69.96 +/- 18.42% |
| LIF_spikes | instantaneous_spikes | 5 | 93.01 +/- 11.35% |
| LIF_spikes | instantaneous_spikes | 39 | 61.29 +/- 14.72% |
| LIF_spikes | instantaneous_spikes | 41 | 81.13 +/- 12.59% |
| LIF_spikes | instantaneous_spikes | 95 | 56.80 +/- 13.37% |
| LIF_membrane | membrane | 5 | 100.00 +/- 0.00% |
| LIF_membrane | membrane | 39 | 91.29 +/- 7.80% |
| LIF_membrane | membrane | 41 | 99.84 +/- 0.16% |
| LIF_membrane | membrane | 95 | 64.38 +/- 11.71% |
| LIF_membrane | instantaneous_spikes | 5 | 82.38 +/- 13.30% |
| LIF_membrane | instantaneous_spikes | 39 | 54.88 +/- 4.77% |
| LIF_membrane | instantaneous_spikes | 41 | 77.89 +/- 16.96% |
| LIF_membrane | instantaneous_spikes | 95 | 50.47 +/- 1.75% |

## Fresh matched neuron comparisons
25 fits, five seeds; fresh independent train/val/test inputs; same initialization across all LIF variants. Train16/32/64,120updates, Adam .01, gradient clip1, validation minimum CE. Beta .95 vs .99; soft subtractive vs hard zero reset; final-three spikes vs final membrane readout. All LIF722params, RNN746. Membrane readout uses analog state. Test32 is in train range;96/192 exceed64. SD reflects five runs, not broad model-family proof. Changes can affect both optimization and retained information; traces alone do not establish a unique causal mechanism.

![[SpikingBrain Assets/neuron-ablation-comparison.png]]

| Variant | Test length | Noise | Mean +/- sample SD | Seed range |
|---|---:|---:|---|---|
| RNN | 32 | 0.1 | 99.34 +/- 0.78% | 98.05-100.00% |
| RNN | 32 | 0.2 | 91.52 +/- 3.66% | 85.94-94.92% |
| RNN | 96 | 0.1 | 98.28 +/- 1.42% | 96.09-100.00% |
| RNN | 96 | 0.2 | 75.39 +/- 5.22% | 69.73-83.20% |
| RNN | 192 | 0.1 | 96.33 +/- 2.92% | 92.38-99.61% |
| RNN | 192 | 0.2 | 64.73 +/- 5.38% | 60.16-73.24% |
| spikes_base | 32 | 0.1 | 56.95 +/- 3.58% | 53.32-62.30% |
| spikes_base | 32 | 0.2 | 57.19 +/- 2.24% | 53.91-60.16% |
| spikes_base | 96 | 0.1 | 51.84 +/- 2.29% | 48.83-54.69% |
| spikes_base | 96 | 0.2 | 51.88 +/- 0.86% | 50.59-52.73% |
| spikes_base | 192 | 0.1 | 50.43 +/- 1.81% | 48.44-52.34% |
| spikes_base | 192 | 0.2 | 50.39 +/- 1.65% | 48.63-52.93% |
| membrane_base | 32 | 0.1 | 89.53 +/- 2.54% | 85.16-91.21% |
| membrane_base | 32 | 0.2 | 81.48 +/- 3.46% | 77.34-85.35% |
| membrane_base | 96 | 0.1 | 78.20 +/- 4.90% | 71.29-83.20% |
| membrane_base | 96 | 0.2 | 70.00 +/- 1.53% | 68.75-72.66% |
| membrane_base | 192 | 0.1 | 66.41 +/- 2.68% | 61.91-69.14% |
| membrane_base | 192 | 0.2 | 61.09 +/- 1.20% | 59.38-62.30% |
| membrane_decay99 | 32 | 0.1 | 75.20 +/- 17.54% | 52.54-94.53% |
| membrane_decay99 | 32 | 0.2 | 72.73 +/- 11.70% | 58.59-86.91% |
| membrane_decay99 | 96 | 0.1 | 66.25 +/- 14.34% | 50.78-81.25% |
| membrane_decay99 | 96 | 0.2 | 61.56 +/- 8.89% | 51.37-70.90% |
| membrane_decay99 | 192 | 0.1 | 59.34 +/- 11.35% | 50.00-74.61% |
| membrane_decay99 | 192 | 0.2 | 56.41 +/- 5.48% | 50.00-62.30% |
| membrane_hard_reset | 32 | 0.1 | 91.09 +/- 2.83% | 86.13-92.97% |
| membrane_hard_reset | 32 | 0.2 | 81.91 +/- 2.88% | 78.12-85.55% |
| membrane_hard_reset | 96 | 0.1 | 75.74 +/- 6.50% | 67.19-83.98% |
| membrane_hard_reset | 96 | 0.2 | 66.64 +/- 2.10% | 64.65-69.73% |
| membrane_hard_reset | 192 | 0.1 | 62.54 +/- 3.38% | 58.20-67.38% |
| membrane_hard_reset | 192 | 0.2 | 58.52 +/- 2.57% | 54.69-61.91% |

Controls 245/245; all150 recorded scores reproduced exactly from saved disk states. Full histories/individual results [[SpikingBrain Assets/neuron-ablation-results.json]]. No test-based hyperparameter selection. No energy claim or full SpikingBrain inference. All newly created artifacts on D:; existing CPU libraries reused. Runtime 140.9s.
