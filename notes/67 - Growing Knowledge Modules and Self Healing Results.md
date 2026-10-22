# 67 - Growing Knowledge Modules and Self Healing Results

The append-only mechanism succeeded at preserving existing modules: all 270 earlier-module regression checks passed across six growing libraries. Parameters and old-category validation predictions (clean and noisy) remained exactly identical after later additions. This is preservation of the old module, not a guarantee that an automatic router keeps choosing it.

The new experiment trains one independent two-class expert for each arriving task. Earlier experts remain frozen; the new expert needs no old training examples. Each library has ten experts. Three initializations per architecture share the same partitions and holdouts. Each expert receives 288 updates of batch 64, giving 2880 updates per full library, equal in count to the earlier shared model. Task-specific data allocation, classifier scope and parameter capacity differ substantially.

| Architecture | Method | Fresh clean % | Fresh noise 0.2 % | Interpretation |
|---|---|---:|---:|---|
|CNN|Prior shared model|80.37|76.28|Global 20-class baseline, no task context|
|CNN|Frozen growing experts + supplied task|99.40|99.03|Oracle two-class task context supplied in image test|
|CNN|Frozen experts + automatic router|64.54|59.83|Global 20-class task-agnostic predictions|
|LIF|Prior shared model|72.19|66.49|Global 20-class baseline, no task context|
|LIF|Frozen growing experts + supplied task|99.01|98.74|Oracle two-class task context supplied in image test|
|LIF|Frozen experts + automatic router|64.44|59.74|Global 20-class task-agnostic predictions|

The fresh final holdout contains 6000 images, 300/global class. All models and the router were frozen before evaluation. Exact pixel overlap with original training, validation, official tests and the earlier 10,000-image holdout is zero; there are no duplicate fresh images. The earlier holdout is reported separately as previously inspected confirmation data. No test labels select weights or router settings. Supplied task context is derived from labels to evaluate an oracle routing scenario; its high score does not demonstrate automatic category understanding.

Automatic router: nearest among 1200 retained training images in raw pixel space. It does not change old weights, but new candidates can change which expert answers an old input. Aggregated incremental validation routing changes: {"LIF": {"total_old_validation_decisions": 81000, "changed": 2892, "correct_to_wrong": 1814, "wrong_to_correct": 0}, "CNN": {"total_old_validation_decisions": 81000, "changed": 2892, "correct_to_wrong": 1833, "wrong_to_correct": 0}}. Counts include repeated queries across stages/seeds, not unique independent samples.

| Architecture | Expert parameters | Expert checkpoint bytes | Router bytes | Active state total bytes | Verified backup bytes |
|---|---:|---:|---:|---:|---:|
|CNN|528500|2153848|961944|3115792|3115792|
|LIF|1007380|4056804|961944|5018748|5018748|

Growth uses more parameters and storage than the previous 1.5 MiB cap. Recovery snapshots duplicate active bytes on disk; manifests and filesystem overhead are additional. Inference runs on the CPU. LIF hidden units remain genuine eight-step simulated spiking units with analog input currents/readout. Neither the CNN nor the LIF implementation is a biological neuron simulation of the human brain.

Self-healing tests passed: append-only frontend/backend knowledge, duplicate-node rejection, checksum damage detection, verified recovery, exact parameter/prediction restoration, healthy access without unnecessary repair, and rejection when live and backup copies are both damaged. The integrated persistent predictor restored a corrupted expert before answering. These faults were injected into isolated copies; healthy source checkpoints were not damaged.

Trusted-but-wrong fact test: a deliberately incorrect statement with a valid checksum remained incorrect. This correctly exposes the limit: integrity recovery does not validate truth, heal routing errors or correct an inaccurate model. A trusted manifest and at least one valid snapshot are required; this is snapshot-based operational healing.

The persistent knowledge example stores HTML, CSS and React under frontend, and database knowledge under backend, using stable explicit category/key lookup. Restarting the store and appending new nodes preserved old records and answers. This tests the requested data structure, not natural-language reasoning or acquisition of an executable frontend coding skill.

Runnable deployment: growth/libraries/{CNN,LIF}-{1662,1663,1664}; use try_growing_model.py with --kind, --seed, --index, and optional --category 1 through 10. Omit --category for automatic routing. Models and the router are integrity checked during reads; corrupted files are recovered only from verified snapshots. Fresh holdout includes reference labels for demonstration; supplied category context is explicit, never silently inferred from those labels.

Main remaining work: improve automatic category/relevance routing and grow modules only when existing modules cannot handle a genuinely new task. Current growth is explicitly triggered by the task boundary and has no learned novelty detector. Add feedback-verified correction nodes for semantic mistakes, preserving original versions. More realistic frontend-skill tasks are needed before claiming useful coding-skill learning.