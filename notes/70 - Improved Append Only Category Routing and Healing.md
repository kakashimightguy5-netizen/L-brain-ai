# 70 - Improved Append Only Category Routing and Healing

The first append-only expert experiment preserved old modules but automatic nearest-image routing achieved only about 64%. The follow-up replaced outlier-prone single-reference matching with independently learned category cluster nodes, while retaining the old policy and every old expert unchanged. This improves global predictions rather than relying on oracle task context.

Configuration was selected on validation only: per-image unit-normalized features and 32 k-means centers per class, 64 centers per two-class category. The six frozen configurations tested raw/unit features with 4/16/32 centers; selected validation task routing accuracy was 88.55%. Each class center fit sees only its own 1500 training images. No global feature normalization is learned across categories. Center files are frozen after addition; adding new candidates can still change automatic decisions.

All comparisons below use exactly the same fresh 4000-image holdout (200/global class), with no exact-image overlap with training, validation, official tests or either previous holdout. Expert weights and routing configuration were frozen before this holdout was evaluated. Three expert initializations share one selected routing fit and one population. These are initialization means, not three independent datasets or routing replications.

| Architecture | Method | Clean % | One noise 0.2 observation % |
|---|---|---:|---:|
|CNN|Previous shared model|81.10|76.70|
|CNN|Growing experts + unchanged nearest-image router|64.32|59.27|
|CNN|Growing experts + learned category clusters|88.49|86.04|
|CNN|Growing experts + supplied task (oracle)|99.39|99.13|
|LIF|Previous shared model|72.11|66.47|
|LIF|Growing experts + unchanged nearest-image router|64.32|59.22|
|LIF|Growing experts + learned category clusters|88.39|85.73|
|LIF|Growing experts + supplied task (oracle)|99.25|98.95|

Supplied task scores use an oracle two-class task ID derived from the image label. They remain an easier scenario and do not demonstrate automatic understanding. The cluster-router row is the global 20-class comparison. Learned category cluster routing is not natural-language semantic mapping.

Preservation checks:270 earlier-expert parameter/prediction checks passed in the first growth experiment. Every old expert/router entry and hash stayed identical after appending the cluster nodes to six deployed libraries. Legacy nearest routing still reproduced exactly its old balanced 100-image predictions; the new policy reproduced its own frozen predictions. Old learned center files remained byte-identical across category additions. Automatic decision changes during selected-policy growth: {"old_task_queries": 27000, "changed_old_task_routes": 418, "correct_to_wrong": 340, "old_center_bytes_identical": true}. Therefore unchanged stored knowledge does not imply unchanged automatic answers when a new category competes for a query.

Healing checks: previously tested knowledge/weight recovery from verified snapshots; now damaged category cluster files also restore exact integrated predictions. When the cluster and its backup are both damaged, the predictor refuses recovery. Trusted-but-wrong facts, wrong category decisions and model reasoning errors are not repaired by checksums. A trusted manifest and an intact verified snapshot remain required.

Storage:
- CNN: 4,186,770 bytes required for new expert+cluster policy; 5,148,714 bytes with old router retained; 5,148,714 verified backup bytes.
- LIF: 6,089,726 bytes required for new expert+cluster policy; 7,051,670 bytes with old router retained; 7,051,670 verified backup bytes.

These growing architectures exceed the earlier 1.5 MiB limit and use 10 independent experts. CNN experts total 528,500 parameters; LIF experts total 1,007,380 parameters. Improved capacity and task isolation are part of the change. This is not evidence of equal-capacity superiority or a literal human-brain neuron implementation. Current growth is triggered by known task boundaries; novelty detection and choosing when to grow remain future work.

Runnable command: try_growing_model.py --kind LIF --seed 1662 --routing prototype --index 0. CNN is also supported. Add --category 1 through 10 for explicit context. Omit --routing or set nearest to reproduce the original routing version; its default is intentionally preserved. Inputs may be supplied as normalized 784-pixel JSON. Files live in the growth/libraries directories on D:, with per-node manifests and verified snapshots.

Remaining weaknesses: routing errors still limit automatic accuracy below the expert ceiling. New categories can still divert old queries, so a universal no-change guarantee is only available with a stable explicit category/version. Semantic self-correction needs independently verified feedback and append-only corrections. The frontend example is an immutable fact library; realistic coding-skill acquisition has not been demonstrated.

Additional meaningful routing fault test: deliberately redirected one category mapping in an isolated two-category library. On 80 fresh probe images, unchecked damaged routing accuracy fell from 98.75% to 50.0%. Integrated integrity healing restored 98.75%, with exactly identical old predictions and routing node bytes. These images were chosen by class for fault diagnosis only, not for tuning the router.
