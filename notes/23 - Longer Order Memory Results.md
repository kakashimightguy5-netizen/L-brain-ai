# Longer event-order memory results

[[21 - Embedding and Longer Order Protocol]] · [[18 - Variable Delay Memory Results]] · [[08 - SpikingBrain Process Log]]

Five seeds, 20 model fits, equal120 updates, noise .1 and independent balanced train/validation sets at16/32/64. Validation minimum CE selects checkpoint; test never selects. RNN746, GRU730, LIF variants722 parameters. Both LIF variants use identical initial weights; spike readout averages final3 steps, membrane readout uses final post-reset state (hybrid analog output). Test512examples/condition.

![[SpikingBrain Assets/long-order-comparison.png]]

| Variant | Test length | Noise | Mean +/- sample SD | Seed range |
|---|---:|---:|---|---|
| RNN | 32 | 0.1 | 99.49 +/- 0.64% | 98.44–100.00% |
| RNN | 32 | 0.2 | 90.98 +/- 1.74% | 88.09–92.77% |
| RNN | 96 | 0.1 | 99.14 +/- 1.39% | 96.68–100.00% |
| RNN | 96 | 0.2 | 78.48 +/- 5.01% | 74.41–86.91% |
| RNN | 192 | 0.1 | 98.59 +/- 2.04% | 95.12–100.00% |
| RNN | 192 | 0.2 | 67.54 +/- 5.14% | 62.70–74.61% |
| GRU | 32 | 0.1 | 99.22 +/- 0.82% | 97.85–100.00% |
| GRU | 32 | 0.2 | 91.84 +/- 2.44% | 87.70–94.14% |
| GRU | 96 | 0.1 | 95.43 +/- 1.52% | 93.16–97.07% |
| GRU | 96 | 0.2 | 72.23 +/- 4.86% | 66.41–77.34% |
| GRU | 192 | 0.1 | 81.91 +/- 10.80% | 65.82–91.80% |
| GRU | 192 | 0.2 | 58.75 +/- 1.74% | 55.66–59.77% |
| LIF_spikes | 32 | 0.1 | 61.37 +/- 6.24% | 53.91–69.92% |
| LIF_spikes | 32 | 0.2 | 60.43 +/- 5.01% | 53.91–67.38% |
| LIF_spikes | 96 | 0.1 | 54.57 +/- 4.74% | 51.56–62.50% |
| LIF_spikes | 96 | 0.2 | 52.93 +/- 3.69% | 48.05–58.01% |
| LIF_spikes | 192 | 0.1 | 53.24 +/- 2.98% | 49.80–56.64% |
| LIF_spikes | 192 | 0.2 | 51.72 +/- 1.31% | 50.39–53.12% |
| LIF_membrane | 32 | 0.1 | 89.10 +/- 4.31% | 81.84–93.36% |
| LIF_membrane | 32 | 0.2 | 76.95 +/- 3.36% | 72.27–80.86% |
| LIF_membrane | 96 | 0.1 | 74.77 +/- 4.64% | 67.38–80.08% |
| LIF_membrane | 96 | 0.2 | 69.34 +/- 3.63% | 65.23–73.83% |
| LIF_membrane | 192 | 0.1 | 63.75 +/- 4.71% | 57.62–69.73% |
| LIF_membrane | 192 | 0.2 | 59.22 +/- 2.12% | 56.25–61.33% |

## Controls and limitations
205/205 implementation checks. All120 saved test scores reproduced exactly after disk reload. CPU dense simulation, no power claim or full SpikingBrain execution. Length32 is in training range;96/192 exceed maximum64. Longer-order training differs from previous8–32; compare both LIF readouts within this experiment, not as an isolated readout intervention against old scores. Improvement here is restricted to synthetic binary pulse ordering, not language/episodic understanding. Full validation histories and every seed score [[SpikingBrain Assets/long-order-results.json]]. Runtime266.5s. Five states saved D:\AI\SpikingBrain-study\experiments\long-order-seed42.pt through46.
