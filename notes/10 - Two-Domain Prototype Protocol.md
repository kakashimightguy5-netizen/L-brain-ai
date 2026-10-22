---
tags: [experiment, protocol, continual-learning]
created: 2026-10-05
---
# Two-domain prototype protocol

Back to [[08 - SpikingBrain Process Log]]

## Registered before running
- CPU only, 2 threads, seeds 42-46; all new files and caches on D:.
- Synthetic controlled task, not language: domain A predicts sign(x0+0.5*x1); domain B predicts its opposite. Inputs have 8 independent Gaussian features. Domain marker is a separate noisy scalar (-1 for A, +1 for B). Strong conflict is deliberate, and limits generality.
- Independently generated train (384/domain), validation (256/domain), and test (512/domain) sets. Test set does not select model or early-exit threshold.
- Comparable shared MLP (10->32->2, 418 parameters) and two expert MLPs (8->20->2, 444 total parameters); verify actual counts. Shared sequential A then B vs joint A+B; 80 updates/domain sequential and 160 joint updates. Experts train 80 updates/domain. Batch sizes differ for joint vs sequential; report this limitation.
- Evaluate A before and after B, both-domain test accuracy, learned vs oracle routing, corrupted domain markers and parameter growth. Growth is explicitly triggered by new domain ID; not autonomous semantic discovery.
- Compare actual LIF dynamics to expert ANN: membrane u=0.85*u+input_current; hard threshold 1; subtract spike after firing; sigmoid surrogate derivative for training; reset state per input; 8 simulation steps. Same trainable topology and parameter counts. Custom transparent PyTorch implementation, not an official SpikingBrain block and not snnTorch.
- Adaptive effort: select confidence threshold using validation only, minimum 2 steps, maximum 8. Choose minimum mean steps among thresholds with <=1 percentage-point validation accuracy loss. Evaluate accuracy, mean steps and measured CPU batch time on untouched test inputs. Falling back to fixed 8 is allowed. Reduced steps are not power measurements.
- Memory: exact-match explicit key/value facts with random labels, measured separately from learned classification. Test known, absent, wrong-domain, unrelated, updated and deleted facts. Retrieval of known labels is not learning/generalization.
- Save raw per-seed results, tests, charts and rerun script in vault. Do not assert forgetting solved or spiking superior from this toy task.

## Follow-up controls registered after initial run
- Initial run showed frozen expert retention and adaptive CPU overhead. Added rehearsal using 64 A training examples during B training, starting from the same A-trained shared model. This increases B-stage batch size and is reported as a confound.
- Added independently trained decay multiplier beta=0 (no membrane history) experts with identical starting seed/topology/update count to LIF.
- Corrected all-modules timing to include the same accuracy calculation as routed timing; original timing comparison is superseded.
- Allocate expert B only when new domain B arrives; save seed-42 expert/router state dictionaries for reuse.
- Preserve initial raw results; final run is a follow-up exploratory comparison, not a fresh confirmatory study.


