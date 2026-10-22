# EWC root-cause and single-observation protocol

Registered before new test evaluation, 2026-10-06. All artifacts, caches and notes stay on D:. No full language models are involved.

Fresh splits 942-944, three runs for diagnosis. Same full-resolution CNN, same initial weights, five digit-pair tasks and 30 epochs per task. Compare corrected empirical Fisher lambda1000; model-expected Fisher lambda1000,100000,10000000; and 20-example-per-class replay. Expected Fisher sums probability-weighted squared gradients over all ten predicted labels for 32 current-task examples. It does not retain old input examples. Empirical Fisher uses true-label gradients for 64 examples. These estimators have different compute budgets, explicitly reported.

Each task records Fisher scale, validation performance, old-task versus newest-task test accuracy, predicted class histogram, and within-pair accuracy given an oracle task. Oracle task identity is a diagnosis with extra information, not a deployable ten-class result. Penalty-to-new-task gradient ratios are recorded after the first parameter update of task2. Validation includes old classes, as in the prior protocol; no old training examples are replayed by EWC, but old validation examples remain available for checkpoint selection. The untrained epoch0 model is eligible for checkpointing in every variant.

No claim that EWC can be fixed by changing lambda alone. A large penalty may preserve old classes while preventing learning new ones. Replay is a separate mechanism, not renamed pure EWC.

Single-observation investigation on the same fresh splits: same initialized CNN, 60 clean-then-robust epochs using averaged independent observations versus 60 using strictly one noisy observation; extend the single-observation branch to 120 epochs to measure additional training, clearly marked as double budget. Inputs and noise corruption remain identical at evaluation. No test-based model selection. Clean validation and one noisy validation stream select checkpoints with score0.3clean+0.7noisy. Record all seeds, failures and reloaded predictions.

The previous 60-epoch robust schedule spends some updates on averaged observations. This may limit one-observation performance; it is a hypothesis until the matched comparison is complete. Changes from the earlier clean-pretrained schedule must be explicitly documented.
