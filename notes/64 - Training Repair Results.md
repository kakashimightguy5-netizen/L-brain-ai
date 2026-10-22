# 64 - Training Repair Results

The old training used only 18 remembered images per class despite retaining 60 for inference. Near the last task, uniform class sampling allocated only 10% of updates to the current pair. The old checkpoint rule could also preserve weights that had not acquired the current pair. These are plausible causes; the interventions below measure their effect.

All models use the same 2,880 training updates, 64 examples/update, original architecture, ten sequential tasks, and at most1.5MiB serialized state. New runs retain and train on 60 images/class. No untrained entry checkpoint is eligible in any new variant; this shared change limits attribution of the replay-only versus historical comparison.

Pilot variants: replay60 keeps uniform class sampling; current_half allocates half the batch to current classes; acquisition_checkpoint adds equal weighting of current and previous validation accuracy when selecting checkpoints. The first-task batch is entirely current.

Validation-only common selection: **acquisition_checkpoint**. Validation averages: {'replay60': 0.7132222056388855, 'current_half': 0.7431111137072245, 'acquisition_checkpoint': 0.7646111051241556}. The selected variant has pilot seed 1562 and confirmation seeds 1563/1564. Historical controls use 1362–1364; the fresh seed change means historical differences are not paired causal estimates.

The new final evaluation uses10,000 previously unused source indices from original training files, 500/class. Their labels were withheld from training and method selection. Official test images were not used for this round’s selection. Exact-pixel overlap audit: {'train': 0, 'validation': 0, 'previous_official_test': 0, 'within_holdout_duplicate_images': 0}. After excluding every image matching prior train, validation or official-test images, the strict secondary holdout contains 10000 images.

| Model | Method | Runs | Clean % | Noise0.2 % | Newest task clean % | Strict clean % | Confirmation-only clean % |
|---|---|---:|---:|---:|---:|---:|---:|
|ANN|historical18|3|67.97|63.78|88.47|67.97|pilot only|
|ANN|replay60|1|67.96|65.22|0.00|67.96|pilot only|
|ANN|current_half|1|75.51|68.92|95.50|75.51|pilot only|
|ANN|acquisition_checkpoint|3|76.04|70.24|96.37|76.04|76.14499926567078|
|CNN|historical18|3|73.81|69.32|92.77|73.81|pilot only|
|CNN|replay60|1|81.32|77.41|91.40|81.32|pilot only|
|CNN|current_half|1|81.53|78.21|97.40|81.53|pilot only|
|CNN|acquisition_checkpoint|3|80.96|76.63|97.20|80.96|80.17500042915344|
|LIF|historical18|3|60.40|56.39|9.83|60.40|pilot only|
|LIF|replay60|1|66.33|61.10|0.00|66.33|pilot only|
|LIF|current_half|1|67.51|61.92|45.30|67.51|pilot only|
|LIF|acquisition_checkpoint|3|72.25|66.45|67.47|72.25|71.99499905109406|

The noisy condition is one corrupted observation, not multiple observations. Means across initialization seeds share the same holdout; they are not independent populations. Pilot rows are exploratory validation-selected comparisons. The two confirmation runs are reported separately to make selection bias visible.

Checks: every trained parameter finite, every serialized checkpoint below the byte cap, exact native predictions after checkpoint reload for both conditions. No memory correction is applied to these reported results, so improvements reflect changed learned weights. Spiking models still use genuine eight-step LIF hidden activity and an analog readout. These tests do not establish general language ability, biological self-correction, neuromorphic energy savings, or pure-EWC success.

Runnable models are repair-{ANN,CNN,LIF}-acquisition_checkpoint-{1562,1563,1564}.pt in the experiment directory. Use the native model loader, or the supplied demonstration script. Detailed epoch-by-epoch new/past validation scores and all predictions are in repair-results.json.

Direct checkpoint evidence: on pilot LIF task 8, all-seen selection retained epoch6 with new-task accuracy 0.33%; epoch12 had 94.67% new-task accuracy but was discarded because aggregate accuracy favored old tasks. This within-run evidence identifies the selection failure more directly than comparing final models from different seeds. Acquisition-aware selection changes the retention/acquisition tradeoff; it does not eliminate forgetting.
