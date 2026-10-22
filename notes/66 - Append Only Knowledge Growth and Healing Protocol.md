# 66 - Append Only Knowledge Growth and Healing Protocol

2026-10-06. Registered before training and evaluation. This is an engineered expandable architecture inspired by the user's idea, not a claim that biological learning creates a neuron for each fact.

Create a separate CNN or eight-step LIF expert for each of the ten existing two-class digit/clothing tasks. Train only the new expert on the current task; freeze and checksum every earlier expert. Each expert uses the previous hidden architecture with a two-class output head. Train 12 epochs, 24 batches/epoch, batch 64, Adam 0.001, the same clean/noisy objective as the previous experiment. Select a checkpoint on that task's clean validation accuracy and CE only. Seeds 1662,1663,1664. No training replay is needed for frozen experts.

Test all earlier validation predictions and exact parameter bytes before and after every module addition. Explicit category routing is a hard task identifier supplied to inference; for the image benchmark this is oracle task context derived from the label, a substantial advantage over the previous global 20 classifier. Do not compare that score as if it solves task-agnostic classification.

Also test a separate automatic router using nearest retained image in raw pixel space, 60 training images per class, fixed before evaluation. It accesses retained images for routing only, not for neural training. Measure routing accuracy and old answers changed by new router candidates, despite frozen experts. This exposes the limits of append-only learning when the category is unknown.

Reuse the previous 10,000-image holdout as a previously inspected confirmation set, not a newly untouched test. Create a second fresh 300/class holdout from unused original training indices, excluding original training, validation, previous holdout, and exact pixels matching previous official tests. Seed 1652. Evaluate it only after all six model-library fits are frozen. Report both holdouts and per-task/noisy results. No test labels tune weights or router rules.

Growth deliberately permits more parameters and bytes than the earlier 1.5 MiB cap. Report expert counts, parameters, active inference bytes, retained router images, backup bytes and growth per task. This is not a same-capacity superiority claim.

Self-healing: hash each immutable checkpoint against a trusted manifest; keep verified snapshots. Corrupt copies in a dedicated fault-test directory, never the healthy source checkpoints. Test detection, restoration, exact recovered parameters/predictions, repeated healthy access, and refusal when both live copy and backup are damaged. Recovery from a trusted backup is operational integrity recovery, not intrinsic biological healing or autonomous correction of wrong facts.

Create an illustrative frontend knowledge library with append-only category-linked fact nodes. Adding React must preserve HTML/CSS records and answers. Test duplicate-ID rejection, new category creation, damaged knowledge recovery and invalid-backup refusal. Retrieval here is explicit category/key lookup, not natural language understanding.

Record code, scores, regression checks, fault injections, failures, resource costs and limits in the Obsidian vault. All writes and processing files stay on D:.
