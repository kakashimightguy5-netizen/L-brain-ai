# 69 - Append Only Category Prototype Routing Follow Up

Registered after the first growth result revealed 64% automatic accuracy. Prior 6000-image scores are inspected; do not tune on them.

Keep all 60 experts and the existing nearest-reference routing policy unchanged. Add separately versioned routing nodes learned only from each arriving class's own 1500 training images. Use k-means 4,16,32 centers/class, with either raw pixels or per-image unit normalization (no cross-class fitted normalization). Seeds 1762+class, n_init 3, max_iter 50. Each task's center node is immutable after creation. Evaluate incremental task routing to show that old center bytes stay fixed while automatic decisions may change when candidates are appended.

Choose the common center configuration by global task routing accuracy on existing validation data only, after all tasks are available. No holdout labels tune centers, configuration or expert weights. Use a new 4000-image holdout, 200/global class, seed 1752, excluding all earlier training, validation, official tests, the 10000-image repair holdout and the 6000-image growth holdout, including exact pixel duplicates.

Report the new automatic router, unchanged nearest router, supplied task context and previous shared model on exactly the same new holdout. Test clean and one noise 0.2 observation. Three expert initializations share one routing-node initialization and the same final holdout; do not claim three independent router replications. Retain the old routing version for reproducibility. Added centers consume extra resources; storage may exceed the previous 1.5 MiB cap.

Deploy center nodes into the integrity-checked append-only libraries, preserving every old manifest entry and file hash. Test exact prototype reload, node damage recovery and invalid-backup refusal using isolated copies. This improves category selection if successful; it cannot guarantee automatic old-answer preservation for all unseen inputs.
