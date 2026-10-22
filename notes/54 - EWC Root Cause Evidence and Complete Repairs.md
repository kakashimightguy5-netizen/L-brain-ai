# EWC root cause and repair results

These are handwritten-digit prototype experiments on CPU, not full SpikingBrain language-model tests. Five fresh runs confirm the single-observation schedule and the EWC confidence-control change. Other comparisons use three diagnosis runs. Splits overlap across runs because the same digits dataset is reused. Three-run calibration and replay follow-ups used already inspected test splits and are exploratory.

## What fails in pure EWC

1. Ten-class score competition: EWC trains on two current digits at a time without replaying old training inputs. Historical selected EWC checkpoints scored23.00% globally but about72.78% when the correct digit pair was supplied. Task identity is extra information, so this does not rescue deployed accuracy.

2. Weak importance estimates after confident fitting: original empirical Fisher gradients became very small. In three fresh runs its penalty gradient at the second task was0.024%-3.18% of the current-task gradient just after one update. Expected Fisher increased protection, but did not consistently prevent forgetting. These measurements are local snapshots, not averages over the entire training trajectory.

3. Preservation versus learning: very strong penalties did not solve the class competition and sometimes prevented learning later pairs. Per-task records include old- and new-class accuracy; several later pairs fell to0%. Giving the correct task still did not recover perfect accuracy, showing actual discrimination loss as well as score imbalance.

4. Excessive confidence: the first new-task loss was16.93-40.60 before confidence control in the diagnosis. Label smoothing0.1 reduced that shock to roughly3-4 in the early diagnostic runs. The frozen five-run comparison below tests whether this helps final retention. It is not evidence that pure EWC now meets the replay baseline.

EWC protects important parameters rather than storing or reconstructing old examples. Published comparisons distinguish task-incremental and class-incremental settings and report regularization failures in the latter: https://doi.org/10.1038/s42256-022-00568-3 . Original EWC paper: https://doi.org/10.1073/pnas.1611835114 .

## All results

Mean accuracy and standard deviation across the listed number of runs. No rejected examples or additional test observations are used in these rows.

### ewc-root-cause

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| empirical1000 | clean | 3 | 31.11% ± 4.20 |
| empirical1000 | single35 | 3 | 22.04% ± 5.37 |
| expected1000 | clean | 3 | 32.59% ± 5.01 |
| expected1000 | single35 | 3 | 23.80% ± 1.37 |
| expected100000 | clean | 3 | 28.15% ± 4.32 |
| expected100000 | single35 | 3 | 21.85% ± 2.84 |
| expected10000000 | clean | 3 | 24.81% ± 4.86 |
| expected10000000 | single35 | 3 | 21.48% ± 2.70 |
| replay | clean | 3 | 95.09% ± 1.16 |
| replay | single35 | 3 | 66.20% ± 3.69 |

### ewc-bias-calibration

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| empirical1000_bias_calibrated | clean | 3 | 39.26% ± 2.52 |
| empirical1000_bias_calibrated | single35 | 3 | 23.80% ± 0.98 |
| expected1000_bias_calibrated | clean | 3 | 35.83% ± 9.43 |
| expected1000_bias_calibrated | single35 | 3 | 22.31% ± 6.90 |
| expected100000_bias_calibrated | clean | 3 | 33.33% ± 5.01 |
| expected100000_bias_calibrated | single35 | 3 | 20.37% ± 1.31 |
| expected10000000_bias_calibrated | clean | 3 | 27.13% ± 2.26 |
| expected10000000_bias_calibrated | single35 | 3 | 20.83% ± 3.33 |
| replay_bias_calibrated | clean | 3 | 95.46% ± 1.40 |
| replay_bias_calibrated | single35 | 3 | 66.39% ± 3.37 |

### single-focus

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| clean | clean | 3 | 96.67% ± 1.82 |
| clean | single35 | 3 | 62.50% ± 1.00 |
| clean | single50 | 3 | 43.89% ± 3.61 |
| mixed60 | clean | 3 | 97.69% ± 1.16 |
| mixed60 | single35 | 3 | 74.44% ± 3.27 |
| mixed60 | single50 | 3 | 55.65% ± 1.31 |
| single60 | clean | 3 | 97.96% ± 0.80 |
| single60 | single35 | 3 | 76.11% ± 2.82 |
| single60 | single50 | 3 | 56.76% ± 1.53 |
| single120 | clean | 3 | 98.24% ± 1.37 |
| single120 | single35 | 3 | 77.87% ± 2.31 |
| single120 | single50 | 3 | 57.41% ± 2.23 |

### single-kernel

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| augmented_RBF | clean | 3 | 98.06% ± 1.21 |
| augmented_RBF | single35 | 3 | 74.63% ± 0.85 |
| augmented_RBF | single50 | 3 | 56.11% ± 1.92 |

### single-confirm

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| clean | clean | 5 | 95.44% ± 2.12 |
| clean | single35 | 5 | 62.28% ± 2.56 |
| clean | single50 | 5 | 42.89% ± 3.78 |
| mixed60 | clean | 5 | 97.89% ± 1.22 |
| mixed60 | single35 | 5 | 75.06% ± 2.34 |
| mixed60 | single50 | 5 | 53.61% ± 1.21 |
| single60 | clean | 5 | 97.50% ± 1.36 |
| single60 | single35 | 5 | 75.83% ± 2.60 |
| single60 | single50 | 5 | 55.00% ± 1.40 |
| single120 | clean | 5 | 98.00% ± 1.01 |
| single120 | single35 | 5 | 77.22% ± 1.67 |
| single120 | single50 | 5 | 57.50% ± 1.58 |

### replay-single

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| replay_single | clean | 3 | 94.17% ± 1.27 |
| replay_single | single35 | 3 | 64.63% ± 1.85 |
| replay_single | single50 | 3 | 48.61% ± 0.73 |
| balanced_replay_single | clean | 3 | 94.72% ± 0.96 |
| balanced_replay_single | single35 | 3 | 68.98% ± 3.45 |
| balanced_replay_single | single50 | 3 | 50.83% ± 1.21 |

### continual-likelihood

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| continual_memory20 | clean | 3 | 94.44% ± 0.96 |
| continual_memory20 | single35 | 3 | 74.91% ± 0.85 |
| continual_memory20 | single50 | 3 | 57.13% ± 2.97 |
| continual_memory50 | clean | 3 | 97.31% ± 1.12 |
| continual_memory50 | single35 | 3 | 81.57% ± 0.16 |
| continual_memory50 | single50 | 3 | 63.80% ± 1.28 |
| continual_memory100 | clean | 3 | 97.96% ± 1.43 |
| continual_memory100 | single35 | 3 | 82.96% ± 0.42 |
| continual_memory100 | single50 | 3 | 65.65% ± 1.67 |

### ewc-confidence

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| smooth_empirical1000 | clean | 3 | 38.15% ± 6.42 |
| smooth_empirical1000 | single35 | 3 | 27.22% ± 1.21 |
| smooth_expected1000 | clean | 3 | 33.89% ± 2.17 |
| smooth_expected1000 | single35 | 3 | 22.04% ± 3.20 |
| smooth_expected100 | clean | 3 | 29.44% ± 4.01 |
| smooth_expected100 | single35 | 3 | 23.43% ± 2.97 |

### ewc-confirm

| Method | Condition | Runs | Accuracy mean ± SD |
|---|---|---:|---:|
| empirical1000 | clean | 5 | 28.61% ± 5.77 |
| empirical1000 | single35 | 5 | 21.94% ± 5.78 |
| smooth_empirical1000 | clean | 5 | 34.61% ± 4.12 |
| smooth_empirical1000 | single35 | 5 | 25.22% ± 4.84 |

## What the comparison does and does not establish

EWC confidence control stays memory-free during training, but old validation examples select checkpoints, as in the baseline. It is not a no-old-data experiment. Expected Fisher has higher computation cost than empirical Fisher. The very large penalty sweep is a diagnosis, not equal-capacity model expansion.

Supervised bias calibration used180 old/new labelled validation examples and a separate179-example selection subset. It is not pure EWC. Its incomplete recovery supports the conclusion that offsets alone are insufficient.

Single60 versus mixed60 uses the same parameter count and robust epoch budget. Single120 adds60 robust epochs, and restarts optimization from the validation-selected Single60 checkpoint. It is a resource tradeoff, not an equal-compute improvement. All receive30 clean pretraining epochs.

The augmented RBF classifier did not establish a stronger noisy solution. Its support vectors and serialized sizes are in the raw records. Larger models were not promoted simply because they were tried.

The replay follow-up also changed checkpoint selection to noisy/clean validation; gains are not attributed solely to noise range. Balanced sampling is a separately evaluated change.

The likelihood decision repair uses exactly the same200 k-center-selected exemplars as replay for the20/class setting. Its500- and1000-reference settings spend more memory. Reference arrays plus labels occupy52,800;132,000;264,000bytes respectively, before checkpoint metadata. The1000-reference version retains nearly the entire training split and should not be called a small-memory result. This is an external exemplar decision method, not native neural self-healing.

A single-noisy-observation90% goal is still not achieved by these tested neural repairs. Full-model native fine-tuning, autonomous correction without reliable feedback, and full language benchmarks remain unverified.

248 prediction arrays exactly reproduced after checkpoint/model reload. All recorded finite/checkpoint checks passed.

## Reproduce and inspect

Source, raw results, selected models, validation grids and histories are archived with SHA256 hashes under SpikingBrain Assets/EWC-root-repairs. Runnable demo: D:\AI\SpikingBrain-study\experiments\try_root_cause_repairs.py. Modes memory, single, ewc. For single use --seed1042; for memory/EWC use --seed942. Memory argument20/50/100 sets references per class. The default demonstration may misclassify a saved input; it was not cherry-picked to show success. Set bytecode/caches/temp to D: before running existing Python.

![[SpikingBrain Assets/EWC-root-repairs/root-cause-results.png]]