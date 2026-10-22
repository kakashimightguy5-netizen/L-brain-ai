# 71 - Latest Growing Knowledge and Healing Results

New learning adds separate modules and category cluster nodes; existing expert weights and stored knowledge are frozen. 270 old-module checks passed. Existing routing policy remains unchanged and selectable. New cluster routing improves automatic category selection.

| Model | Previous shared model | Growing model, automatic clusters | Growing model, supplied category |
|---|---:|---:|---:|
|CNN|81.10%|88.49%|99.39%|
|LIF|72.11%|88.39%|99.25%|

All scores: same fresh 4000-image holdout, three expert initializations. Supplied category is an oracle task ID in this image experiment. Growing models use more parameters/storage.

Self-healing: verified recovery of damaged knowledge, expert weights and category-cluster files; exact restored predictions. Recovery fails safely if no verified snapshot survives. It does not fix wrong facts or routing mistakes.

Read notes 66–70 for registered protocols, failures, tests, noisy scores, resource costs and remaining limitations. All processing, models and reports are on D:.

One noisy observation: CNN86.04%, LIF85.73% with automatic cluster routing. Additional isolated category-mapping damage probe:98.75% healthy,50.00% unchecked damaged,98.75% restored; all original predictions exactly recovered. Successful Windows demos: automatic cluster routing for both CNN/LIF, and stable explicit-category LIF.
