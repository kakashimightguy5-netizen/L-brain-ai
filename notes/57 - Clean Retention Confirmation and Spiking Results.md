# Clean retention and neural-memory confirmation

The fixed test set contains 765 previously unused digit images from the UCI original training file. Training 2293 and validation 765 images are separate. No exact image overlap was found with the previously used 1797 sklearn digits images. This random split is not writer-disjoint. Five runs vary initialization; they share the same test set. Accuracy variability below is initialization standard deviation, not an independent-dataset confidence interval.

All 20 checkpoints and their validation decisions were frozen before the first final test evaluation. Every model learned the five digit-pair tasks sequentially, with no future-class training inputs. Each used 1200 updates. Old validation examples remained available. CNN 20 and CNN 50 have 37930 weights; ANN 50 and LIF 50 have 9610 weights with identical initial matrices. Spiking adds eight simulated time steps, so equal updates do not mean equal computation.

## Frozen test results

| Method | Input | Accuracy mean ± SD | First-pair retention | Deployment storage |
|---|---|---:|---:|---:|
| CNN 20 | clean | 96.60% ± 0.56 | 95.82% | 152.9 KiB |
| CNN 20_combined | clean | 97.73% ± 0.41 | 98.56% | 205.2 KiB |
| CNN 20_memory | clean | 93.46% ± 0.00 | 97.39% | 51.6 KiB |
| CNN 20 | single 35 | 71.87% ± 1.28 | 73.73% | 152.9 KiB |
| CNN 20_combined | single 35 | 77.54% ± 0.71 | 80.26% | 205.2 KiB |
| CNN 20_memory | single 35 | 75.69% ± 0.00 | 83.01% | 51.6 KiB |
| CNN 20 | single 50 | 48.99% ± 0.75 | 51.76% | 152.9 KiB |
| CNN 20_combined | single 50 | 54.12% ± 1.10 | 58.30% | 205.2 KiB |
| CNN 20_memory | single 50 | 52.29% ± 0.00 | 60.78% | 51.6 KiB |
| CNN 50 | clean | 97.62% ± 0.39 | 96.99% | 152.9 KiB |
| CNN 50_combined | clean | 98.41% ± 0.21 | 97.65% | 282.5 KiB |
| CNN 50_memory | clean | 98.30% ± 0.00 | 97.39% | 128.9 KiB |
| CNN 50 | single 35 | 75.27% ± 1.04 | 77.65% | 152.9 KiB |
| CNN 50_combined | single 35 | 82.35% ± 0.51 | 88.37% | 282.5 KiB |
| CNN 50_memory | single 35 | 81.83% ± 0.00 | 89.54% | 128.9 KiB |
| CNN 50 | single 50 | 52.29% ± 1.19 | 55.95% | 152.9 KiB |
| CNN 50_combined | single 50 | 60.84% ± 0.47 | 66.41% | 282.5 KiB |
| CNN 50_memory | single 50 | 60.52% ± 0.00 | 67.32% | 128.9 KiB |
| ANN 50 | clean | 94.64% ± 0.39 | 93.33% | 40.4 KiB |
| ANN 50 | single 35 | 71.97% ± 0.63 | 76.86% | 40.4 KiB |
| ANN 50 | single 50 | 53.12% ± 1.02 | 59.48% | 40.4 KiB |
| LIF 50 | clean | 94.56% ± 0.31 | 93.46% | 40.4 KiB |
| LIF 50 | single 35 | 72.13% ± 0.39 | 76.47% | 40.4 KiB |
| LIF 50 | single 50 | 52.71% ± 0.65 | 58.17% | 40.4 KiB |

Storage excludes embedded test arrays, histories and evaluation metadata. Neural-only models report saved weights. Memory-only rows report raw reference arrays and labels without serialization metadata. Combined rows report runnable neural+reference+configuration files. Tiny ANN/LIF still require a separate 500-example buffer to resume continual training. The larger CNN memory budget is 2.5 times the references, not a free improvement.

## Repairs and damage

| Combined method | Input | Wrong-to-correct fraction | Correct-to-wrong fraction |
|---|---|---:|---:|
| CNN 20_combined | clean | 1.70% | 0.58% |
| CNN 20_combined | single 35 | 8.18% | 2.51% |
| CNN 20_combined | single 50 | 8.73% | 3.61% |
| CNN 50_combined | clean | 1.49% | 0.71% |
| CNN 50_combined | single 35 | 10.88% | 3.79% |
| CNN 50_combined | single 50 | 14.75% | 6.20% |

Memory also breaks some correct decisions; those errors are counted. No predictions were rejected. Clean validation guarded the combination but did not guarantee zero test harm.

## Retention and speed

| Model | Neural backward forgetting on validation | Neural single-input median | Complete noisy-input median |
|---|---:|---:|
| CNN 20 | 1.83 points | 0.128 ms | 6.454 ms |
| CNN 50 | 1.43 points | 0.129 ms | 10.670 ms |
| ANN 50 | 1.76 points | 0.023 ms | 0.187 ms |
| LIF 50 | 1.24 points | 0.237 ms | 1.303 ms |

Timing is local CPU inference, not GPU performance, energy measurement or a cross-device benchmark. Complete timing includes neural inference and conditional memory use for one saved noisy input per run; it is not dataset-wide latency. Backward forgetting is mean prior-task best-minus-final validation accuracy for the neural model, not the combined system. First-pair test retention for each decision method is reported above.

## Conclusion and limitations

The clean 97% retention goal was achieved by the combined models on this frozen evaluation. CNN 50 neural-only also reached 97.62%. The 200-exemplar combination reached 97.73% clean and 77.54% noisy;500 exemplars reached 98.41% clean and 82.35% noisy. This remains below the 90% single-noisy-input target. Results are not directly comparable to previous dataset means because the training population and sample count changed. The within-round paired comparisons are the supported improvements.

The LIF network ran natively with existing CPU PyTorch and matched a conventional same-parameter network within small accuracy differences. It did not beat the CNN or reach 97% clean. No speed or energy advantage is established, and it is not a SpikingBrain language model. Spiking comparison eligibility was determined entirely on validation; all five seeds qualified.

The combined model is an external training-example-memory decision layer. It does not fix pure EWC or demonstrate autonomous truth discovery. Full language-model fine-tuning remains untested.

Runnable: clean_memory_inference.py --model CNN50, CNN20, ANN50 or LIF50. Default seed 1262; default saved input is clean, --input-json accepts one or more 64 pixel normalized[0,1]vectors, --neural-only disables memory. Always set temporary files and bytecode configuration to D: when running existing installed Python.

![[SpikingBrain Assets/clean-memory-confirmation/clean-memory-results.png]]