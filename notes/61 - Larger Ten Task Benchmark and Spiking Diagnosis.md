# Larger benchmark and ten-task results

The official MNIST and Fashion-MNIST test sets provide20000 examples. All global predictions choose among20classes without a task or domain label. Domain-subset accuracies still require the correct global20-class label; these are not isolated10-class benchmark scores. Each model trained on30000selected training examples total, not all120000 available training images.6000 held-out training-file examples were used for validation. Official test examples were not used for training or configuration choice. Exact pixel-pattern overlap auditing found zero train-validation, train-test or validation-test matches in these selected splits.

Three initialization runs share one official test set; they are not three independent populations. Two additional spiking configurations are one-run pilot diagnostics. Baseline validation choices and packed-control choices were registered before any official test result. The packing control changes validation subset size; its packed360control exposes that calibration difference. Larger packing comparisons use the same1200-example calibration subset.

MNIST source: https://github.com/cvdfoundation/mnist . Fashion-MNIST source: https://github.com/zalandoresearch/fashion-mnist . All eight source archives and SHA256 provenance are saved on D:. IDX dimensions, magic and counts checked.

All models learned ten alternating digit/clothing tasks. Each had2880 optimizer updates. ANN and LIF share103060 parameters and identical initial matrices; CNN has54020 parameters and a spatial architecture. All stored model-plus-reference files are below1.5 MiB. Neural weights remain float32. Memory-only results are deterministic, not three seed replicates. Stored bytes exclude optimizer state, test arrays and dataset files; runtime decoded memory uses more RAM.


## original_float32

| Method | Condition | Runs | Global20-class accuracy | Digit subset | Clothing subset | Stored model + references |
|---|---|---:|---:|---:|---:|---:|
| memory360_likelihood | clean | 1 | 57.97% (one deterministic/pilot result) | 61.51% | 54.43% | 1.081MiB |
| memory360_likelihood | single20 | 1 | 57.73% (one deterministic/pilot result) | 61.62% | 53.83% | 1.081MiB |
| memory360_likelihood | single35 | 1 | 56.77% (one deterministic/pilot result) | 59.85% | 53.68% | 1.081MiB |
| memory360_distance | clean | 1 | 58.71% (one deterministic/pilot result) | 60.99% | 56.43% | 1.081MiB |
| memory360_distance | single20 | 1 | 54.80% (one deterministic/pilot result) | 52.60% | 57.00% | 1.081MiB |
| memory360_distance | single35 | 1 | 43.62% (one deterministic/pilot result) | 35.21% | 52.04% | 1.081MiB |
| memory480_likelihood | clean | 1 | 58.34% (one deterministic/pilot result) | 62.64% | 54.04% | 1.441MiB |
| memory480_likelihood | single20 | 1 | 57.94% (one deterministic/pilot result) | 62.85% | 53.03% | 1.441MiB |
| memory480_likelihood | single35 | 1 | 56.76% (one deterministic/pilot result) | 60.58% | 52.94% | 1.441MiB |
| memory480_distance | clean | 1 | 58.27% (one deterministic/pilot result) | 61.89% | 54.64% | 1.441MiB |
| memory480_distance | single20 | 1 | 54.86% (one deterministic/pilot result) | 53.86% | 55.86% | 1.441MiB |
| memory480_distance | single35 | 1 | 43.58% (one deterministic/pilot result) | 35.97% | 51.18% | 1.441MiB |
| ANN | clean | 3 | 67.71% ± 2.11 | 74.89% | 60.53% | 1.476MiB |
| ANN_combined | clean | 3 | 69.07% ± 2.67 | 75.73% | 62.42% | 1.476MiB |
| ANN | single20 | 3 | 63.54% ± 1.31 | 70.58% | 56.50% | 1.476MiB |
| ANN_combined | single20 | 3 | 65.84% ± 2.80 | 72.24% | 59.43% | 1.476MiB |
| ANN | single35 | 3 | 54.62% ± 1.14 | 59.95% | 49.30% | 1.476MiB |
| ANN_combined | single35 | 3 | 60.48% ± 1.25 | 65.70% | 55.27% | 1.476MiB |
| CNN | clean | 3 | 73.60% ± 0.92 | 84.33% | 62.87% | 1.290MiB |
| CNN_combined | clean | 3 | 74.92% ± 0.56 | 84.79% | 65.04% | 1.290MiB |
| CNN | single20 | 3 | 69.31% ± 0.87 | 80.90% | 57.71% | 1.290MiB |
| CNN_combined | single20 | 3 | 71.93% ± 0.47 | 82.02% | 61.85% | 1.290MiB |
| CNN | single35 | 3 | 57.23% ± 3.48 | 66.03% | 48.42% | 1.290MiB |
| CNN_combined | single35 | 3 | 64.65% ± 1.72 | 71.52% | 57.77% | 1.290MiB |
| LIF_T8_h1.0 | clean | 3 | 60.51% ± 2.52 | 74.07% | 46.95% | 1.476MiB |
| LIF_T8_h1.0_combined | clean | 3 | 66.26% ± 1.99 | 74.72% | 57.79% | 1.476MiB |
| LIF_T8_h1.0 | single20 | 3 | 56.48% ± 2.30 | 69.25% | 43.71% | 1.476MiB |
| LIF_T8_h1.0_combined | single20 | 3 | 63.65% ± 1.61 | 71.83% | 55.47% | 1.476MiB |
| LIF_T8_h1.0 | single35 | 3 | 49.04% ± 2.54 | 59.00% | 39.08% | 1.476MiB |
| LIF_T8_h1.0_combined | single35 | 3 | 58.91% ± 0.78 | 65.38% | 52.45% | 1.476MiB |
| LIF_T8_h0.5 | clean | 1 | 52.28% (one deterministic/pilot result) | 64.19% | 40.37% | 1.476MiB |
| LIF_T8_h0.5_combined | clean | 1 | 64.03% (one deterministic/pilot result) | 70.39% | 57.68% | 1.476MiB |
| LIF_T8_h0.5 | single20 | 1 | 49.29% (one deterministic/pilot result) | 60.63% | 37.95% | 1.476MiB |
| LIF_T8_h0.5_combined | single20 | 1 | 62.08% (one deterministic/pilot result) | 67.93% | 56.23% | 1.476MiB |
| LIF_T8_h0.5 | single35 | 1 | 42.25% (one deterministic/pilot result) | 51.60% | 32.91% | 1.476MiB |
| LIF_T8_h0.5_combined | single35 | 1 | 58.44% (one deterministic/pilot result) | 62.78% | 54.10% | 1.476MiB |
| LIF_T4_h0.5 | clean | 1 | 56.63% (one deterministic/pilot result) | 70.75% | 42.51% | 1.476MiB |
| LIF_T4_h0.5_combined | clean | 1 | 64.05% (one deterministic/pilot result) | 71.32% | 56.79% | 1.476MiB |
| LIF_T4_h0.5 | single20 | 1 | 52.08% (one deterministic/pilot result) | 64.18% | 39.97% | 1.476MiB |
| LIF_T4_h0.5_combined | single20 | 1 | 61.72% (one deterministic/pilot result) | 68.34% | 55.10% | 1.476MiB |
| LIF_T4_h0.5 | single35 | 1 | 44.32% (one deterministic/pilot result) | 51.67% | 36.98% | 1.476MiB |
| LIF_T4_h0.5_combined | single35 | 1 | 57.40% (one deterministic/pilot result) | 62.18% | 52.62% | 1.476MiB |

## lossless_packed

| Method | Condition | Runs | Global20-class accuracy | Digit subset | Clothing subset | Stored model + references |
|---|---|---:|---:|---:|---:|---:|
| packed_memory360 | clean | 1 | 57.97% (one deterministic/pilot result) | 61.51% | 54.43% | 0.274MiB |
| packed_memory360 | single20 | 1 | 57.73% (one deterministic/pilot result) | 61.62% | 53.83% | 0.274MiB |
| packed_memory360 | single35 | 1 | 56.77% (one deterministic/pilot result) | 59.85% | 53.68% | 0.274MiB |
| packed_memory1200 | clean | 1 | 63.00% (one deterministic/pilot result) | 70.50% | 55.50% | 0.908MiB |
| packed_memory1200 | single20 | 1 | 62.41% (one deterministic/pilot result) | 70.28% | 54.54% | 0.908MiB |
| packed_memory1200 | single35 | 1 | 60.78% (one deterministic/pilot result) | 67.68% | 53.88% | 0.908MiB |
| packed_memory1920 | clean | 1 | 66.72% (one deterministic/pilot result) | 76.53% | 56.90% | 1.452MiB |
| packed_memory1920 | single20 | 1 | 66.20% (one deterministic/pilot result) | 76.23% | 56.17% | 1.452MiB |
| packed_memory1920 | single35 | 1 | 64.73% (one deterministic/pilot result) | 74.09% | 55.38% | 1.452MiB |
| ANN_packed360 | clean | 3 | 68.97% ± 2.84 | 75.77% | 62.18% | 0.668MiB |
| ANN_packed360 | single20 | 3 | 65.75% ± 2.95 | 72.25% | 59.25% | 0.668MiB |
| ANN_packed360 | single35 | 3 | 60.43% ± 1.34 | 65.71% | 55.15% | 0.668MiB |
| ANN_packed1200 | clean | 3 | 70.20% ± 1.89 | 78.22% | 62.17% | 1.303MiB |
| ANN_packed1200 | single20 | 3 | 67.46% ± 1.72 | 75.55% | 59.36% | 1.303MiB |
| ANN_packed1200 | single35 | 3 | 62.12% ± 0.63 | 69.35% | 54.89% | 1.303MiB |
| CNN_packed360 | clean | 3 | 74.78% ± 0.79 | 84.60% | 64.96% | 0.483MiB |
| CNN_packed360 | single20 | 3 | 71.79% ± 0.71 | 81.84% | 61.74% | 0.483MiB |
| CNN_packed360 | single35 | 3 | 64.48% ± 1.98 | 71.31% | 57.64% | 0.483MiB |
| CNN_packed1200 | clean | 3 | 75.49% ± 0.57 | 85.98% | 65.00% | 1.117MiB |
| CNN_packed1200 | single20 | 3 | 72.73% ± 0.45 | 83.64% | 61.83% | 1.117MiB |
| CNN_packed1200 | single35 | 3 | 65.75% ± 1.60 | 74.11% | 57.38% | 1.117MiB |
| LIF_T8_h1.0_packed360 | clean | 3 | 66.19% ± 1.95 | 74.72% | 57.66% | 0.669MiB |
| LIF_T8_h1.0_packed360 | single20 | 3 | 63.60% ± 1.57 | 71.81% | 55.39% | 0.669MiB |
| LIF_T8_h1.0_packed360 | single35 | 3 | 58.76% ± 0.86 | 65.22% | 52.30% | 0.669MiB |
| LIF_T8_h1.0_packed1200 | clean | 3 | 67.97% ± 1.74 | 77.31% | 58.64% | 1.303MiB |
| LIF_T8_h1.0_packed1200 | single20 | 3 | 65.47% ± 1.43 | 74.73% | 56.21% | 1.303MiB |
| LIF_T8_h1.0_packed1200 | single35 | 3 | 60.69% ± 0.76 | 68.46% | 52.92% | 1.303MiB |

## Forgetting and domain confusion

| Original model | First task retained, clean test | Neural validation forgetting | Wrong domain, clean test |
|---|---:|---:|---:|
| ANN | 87.17% | 24.07points (neural model) | 0.62% |
| ANN_combined | 88.90% | 24.07points (neural model) | 0.40% |
| CNN | 91.49% | 21.72points (neural model) | 0.95% |
| CNN_combined | 93.13% | 21.72points (neural model) | 0.45% |
| LIF_T8_h1.0 | 85.47% | 25.25points (neural model) | 1.58% |
| LIF_T8_h1.0_combined | 88.20% | 25.25points (neural model) | 0.34% |

Forgetting is mean best-minus-final validation accuracy on the nine earlier task pairs. It separates old-skill loss from merely adding harder new classes. Combined-system forgetting was not measured at every stage; its first-task final test retention is reported, and the neural metric is explicitly labelled.

Latest-task acquisition check (same saved predictions): ANN achieved 88.75% clean test accuracy on the last clothing pair (validation 87.94%); CNN achieved 92.30% clean test accuracy on the last clothing pair (validation 92.17%); LIF_T8_h1.0 achieved 9.78% clean test accuracy on the last clothing pair (validation 9.94%). The spiking failure includes not acquiring the newest task, as well as losing earlier skills. Global clean validation can prefer preserving earlier performance over a poorly learned new task; this does not prove gradients vanished.

## Spiking diagnosis

The network uses784 pixel inputs,128 LIF hidden neurons and20 linear output scores. Voltage evolves as0.9*previous voltage + learned constant current; crossing the threshold emits a spike and subtracts the threshold. Mean spike counts multiplied by threshold feed the output layer. Time steps simulate internal state over the same input; they do not supply fresh observations. Surrogate gradients enable native training in existing PyTorch. This dense CPU simulation is not event-driven neuromorphic hardware.

| Pilot configuration | Final clean validation | Last-task mean firing | Silent hidden units | Saturated samples | First-layer gradient norm |
|---|---:|---:|---:|---:|
| LIF_T8_h1.0 | 62.63% | 0.1246 | 15.62% | 8.88% | 5.24185 |
| LIF_T8_h0.5 | 52.17% | 0.0830 | 16.41% | 6.50% | 2.47151 |
| LIF_T4_h0.5 | 57.05% | 0.1078 | 19.53% | 9.40% | 1.84426 |

Selected solely by pilot clean validation: LIF_T8_h1.0. Lower threshold/shorter simulation did not automatically improve learning. Activity and gradient records diagnose representation limits, but this comparison does not prove a single causal explanation for all weakness. Class scarcity in replay and training budget also affect both ANN and LIF.

## Lossless storage control

Original8-bit pixels were unnecessarily retained as32-bit floats. Packing round(normalizedpixel*255) to uint8 and decoding float32/255 reproduced every selected reference exactly. The first18 of each60-reference class matched the original replay references. Original neural training and weights remained unchanged; extra retained examples support inference only. This is not neural weight quantization, retraining on discarded old data or extra test observations.

Packing the same360 references preserves original memory computation (exact validation probability checks) while lowering stored bytes. It allows1200 references with neural weights or1920 references without neural weights below the same1.5 MiB ceiling. More references are additional retained information; the cost advantage comes from lossless encoding. The new sigma/temperature/blend calibration differs from the original full-validation calibration, so packed360versus packed1200is the fair within-control comparison.

## Repair and harm

| Packed combined method | Condition | Incorrect-to-correct | Correct-to-incorrect |
|---|---|---:|---:|
| ANN_packed360 | clean | 5.54% | 4.28% |
| ANN_packed360 | single20 | 7.67% | 5.46% |
| ANN_packed360 | single35 | 11.55% | 5.74% |
| ANN_packed1200 | clean | 6.33% | 3.84% |
| ANN_packed1200 | single20 | 8.73% | 4.81% |
| ANN_packed1200 | single35 | 12.60% | 5.10% |
| CNN_packed360 | clean | 4.03% | 2.85% |
| CNN_packed360 | single20 | 6.13% | 3.65% |
| CNN_packed360 | single35 | 11.44% | 4.19% |
| CNN_packed1200 | clean | 4.66% | 2.77% |
| CNN_packed1200 | single20 | 6.87% | 3.44% |
| CNN_packed1200 | single35 | 12.40% | 3.88% |
| LIF_T8_h1.0_packed360 | clean | 9.86% | 4.17% |
| LIF_T8_h1.0_packed360 | single20 | 11.83% | 4.71% |
| LIF_T8_h1.0_packed360 | single35 | 14.32% | 4.59% |
| LIF_T8_h1.0_packed1200 | clean | 11.24% | 3.77% |
| LIF_T8_h1.0_packed1200 | single20 | 13.24% | 4.26% |
| LIF_T8_h1.0_packed1200 | single35 | 15.74% | 4.10% |

No rejected predictions were excluded. Neural-memory changes sometimes damage correct predictions; both fractions are counted. The corruption model is synthetic Gaussian noise plus pixel dropout with clipping, and memory knows that family. It is not autonomous truth detection.

## Comparable CPU timings

| Cached predictor | Batch | Median time |
|---|---:|---:|
| ANN | 1 | 0.424ms |
| ANN | 64 | 14.007ms |
| CNN | 1 | 0.825ms |
| CNN | 64 | 23.897ms |
| LIF_T8_h1.0 | 1 | 1.580ms |
| LIF_T8_h1.0 | 64 | 17.998ms |
| CNN_packed1200 | 1 | 0.938ms |
| CNN_packed1200 | 64 | 31.852ms |
| memory_packed1920 | 1 | 24.108ms |
| memory_packed1920 | 64 | 68.014ms |

These timings use two BLAS and Torch threads and cache decoded references. They use the same early clean digit inputs, so uncertainty-gate use is input-dependent; they are not dataset-wide latency, GPU benchmarks or energy savings. The original workers used16 NumPy BLAS threads. After all choices were frozen, evaluation was resumed with2BLAS threads and shared memory-score caches to avoid redundant computation. Full neural reload and balanced100-image deployed-predictor checks verified the resumed calculations. No model was retrained or recalibrated.

## Practical conclusion

The earlier98.41% 8x8digit result did not establish performance on this harder28x28,20-class,ten-task problem. Use the above within-round comparisons. Neither a neuron-threshold change nor a small raw-pixel memory is enough to claim a general learning system. Keep failures visible and base architecture choices on actual gains relative to simpler memory-only inference. Full SpikingBrain language-model inference and fine-tuning remain untested.

Runnable: try_packed_larger_model.py. Default is packed CNN with1200 references; pass a saved packed LIF checkpoint for the spiking version, or packed-memory-1920.pt for memory alone. Input JSON accepts784 normalizedpixels per example. Class0-9 means digit0-9; class10-19 means clothing class0-9. Default demonstration uses an official test example and reports its label after predicting. It may be wrong; examples were not selected to show success.

![[SpikingBrain Assets/larger-ten-task/larger-results.png]]