# Stronger architecture and evidence-acquisition protocol

[[41 - Latest Repair Findings and Runnable Tests]] · [[08 - SpikingBrain Process Log]]

Registered before new fits. This round targets the practical causes of the weak scores, while retaining the old records. All work, models, temporary files and results stay on D:. Existing native CPU PyTorch is reused. No full SpikingBrain checkpoint tuning is claimed.

## Stronger sensory representation and correction

Five new seeds442–446, same stratified60/20/20digit partitions (1,078/359/360). Compare spatial CNN (~18.5kparameters) with approximately matched flat MLP (~18.7k), using the same augmentation and update budget.30clean epochs then60robust epochs, Adam.001, gradientclip1; validation checkpoint every5epochs. Robust updates combine original clean training images with corrupted training images. Corruption strengths.1–.5; training noisy inputs occasionally average3or5independent corruptions of the SAME training image, preparing an evidence-integration model. No test images/labels fit anything. Keep clean pretrained checkpoints as no-healing baselines.

Evaluate clean, one observation atnoise.35, mixednoise0–.7, and shiftednoise.5. Rawaccuracy and classconfidence rejectioncoverage both reported. A noise-aware exemplar classifier uses only stored clean TRAINING images/labels, with likelihood accounting for Gaussian corruption, clipping and dropout. Kernelnoise/temperature and CNN–memory blending chosen using validation only. This is supervised prior memory plus inference correction, NOT autonomous intrinsic truth detection. Compare repair vs harm fractions on the same single-observation tests.

## Extra observations as an explicit separate regime

Acquire1/3/5/9INDEPENDENT noisy measurements of each underlying digit in the synthetic sensor simulator. The model receives only observations, never the underlying clean testimage or itslabel. Average observedpixels beforeclassification. Firstobservationidenticaltoordinary single-observation test. These regimes supply additional information and must NEVER be presented as improved same-input inference. An effort gate obtainsmoreobservationswhenconfidenceis low; confidence threshold selected on separatevalidation to match9-observationaccuracywithin.5pointwhile minimizing observations. Report accuracy, average observations, latency and failurecases. Generating extra perturbations of an already corrupted input does not count as new independent evidence.

## Continual learning and integration

Freshly initialized CNNs, no joint pretrained future-class features. Five tasks0/1…8/9;30epochs/task, currentclassesplus20/classoldmemory. Compare no-replayfine-tuning, corrected EWC without replay, ordinaryreplay, class-balancedreplay and CNN shared-column integration. All replay variants selectrepresentative memories from currenttasktrainingonly; oldmemoriespersist. Balance losses so numerousnewexamplesdo not overwhelm sparseoldexamples. All classifiers retain10-way class-incremental inference, with no true domain supplied. Noiseaugmentation on training only. Report finalclean/noisy/firsttaskretention and routing diagnostics, with validation-only EWC strength selection. Structural changes/extra replay information named explicitly.

## Stable weight-generating genome

Five new seeds542–546. A small dense training-only teacher supplies SVD initialization to per-neuron gene codes. A shared decoder generates eachweight from the dotproduct of pre/post codes plus a bounded small learned nonlinear residual, with no global layer renormalization. Compare rank8/rank12 gene models with plain SVD factors; CE+teacherKLdistillation300updates, validation checkpointselection. This bilinear foundation is mathematically low-rank; any extra nonlinear benefit must be demonstrated, not assumed. Quantize stored code/decoder tables to per-rowint8, retain scales/biases, and decode tofloat32CPU for inference. Report quality before/afterquantization and actual filebytes. This is storagequantization, not an int8compute-speed claim.

Aspirational targets:90%single-observationnoisyaccuracy,95%continualretention/multi-observationaccuracy,97%cleanaccuracy,compressionwithin1pointofteachingmodel. Report misses honestly. All saved decisions independently reproduce, data partitions checked, and all failures stay visible.
