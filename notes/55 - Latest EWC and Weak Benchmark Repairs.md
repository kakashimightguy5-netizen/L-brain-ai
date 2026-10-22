# Latest EWC explanation and repairs

Pure EWC failed because sequential two-class training distorts ten-class decisions, weak importance estimates can underprotect old weights, and large penalties can block new learning. Historical 23% accuracy increased to about 72.78% when an oracle supplied the correct task, but that is extra information and still below perfect accuracy. Both score imbalance and discrimination loss are present.

Five fresh runs: confidence-controlled EWC achieved 34.61% clean accuracy versus 28.61% for unchanged EWC. This remains far below replay. Confidence control uses label smoothing 0.1; it retains no old training examples, but both versions use old validation data for checkpoint selection.

Five fresh runs: targeted single-observation training with 120 robust epochs achieved 77.22% noisy accuracy versus 75.06% for the old 60-epoch schedule. Extra training costs are explicit. The matched 60-epoch comparison is in note 54.

Three exploratory runs: changing the replay decision method to noise-aware likelihood inference raised noisy accuracy from 66.20% to 74.91% using the same 200 exemplars. Using 500 exemplars reached 81.57%, with 2.5 times reference-storage cost. This is memory inference, not pure EWC or a healed neural model.

All attempted methods, failures, clean/noisy results, variability and limitations are in [[54 - EWC Root Cause Evidence and Complete Repairs]]. Protocols are notes 48–53. The 90% single-input goal remains unmet. Keep replay as the supported retention method; keep pure EWC as an experimental baseline.

All outputs are on D:. No new models or environments were downloaded. Existing installed Python was executed with bytecode disabled and D: temp/cache paths.
