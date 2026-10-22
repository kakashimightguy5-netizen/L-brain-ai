# Latest stronger prototype results

The strongest supported configuration combines robust training with noise-aware training-example memory. It achieved 83.56% average accuracy on one noisy observation. The neural model alone achieved 74.39%. This remains below the 90% goal.

Nine independent observations achieved 95.61%. The adaptive observation gate achieved 95.44%. This needs additional observations from a real sensor or data source; repeated computation on one input does not supply them.

Validation-selected teacher-initialized INT8 gene codes achieved 96.67% clean accuracy, compared with 96.72% for the dense teacher, at about 4.6 times smaller stored size. Plain low-rank models remain equally competitive. Directly trained gene codes reached only 94.72%, versus 96.72% for the dense baseline.

Balanced replay retained 95.72% clean accuracy after learning five tasks sequentially. Pure EWC retained only 23.00%. Shared columns retained 94.56% and did not establish an advantage over replay.

See [[46 - Stronger Benchmarks and Remaining Failures]] for every condition, variability, consolidation results and failures. The new consolidation follow-up is exploratory because its test splits were already inspected.

## Runnable demonstrations

All code and saved models are under D:\AI\SpikingBrain-study\experiments. Set PYTHONDONTWRITEBYTECODE=1 and TEMP/TMP to the existing D: temp directory before launching Python. Run try_stronger_prototype.py with --mode single, evidence, continual or genes. Optional --index selects a saved test example. These demos classify digits using saved checkpoints; they do not run the full SpikingBrain language models.

## Process record

Registered protocols: notes 42 through 45. Completed five-seed rounds: matched neural/perception experiments, stable gene compression, fresh continual learning, direct gene training, fresh learned density correction, and post-continual consolidation. Initial pooled spatial-model validation failed; that diagnostic was preserved and the architecture revision used fresh seeds. A NumPy checkpoint scalar was converted to a plain float for safe checkpoint loading. The first explanation blaming a clean-accuracy guard was not supported by the diagnostic and was withdrawn. The learned density correction attempt also failed to improve consistently.

No new downloads, environments, experiment outputs or caches were placed on C:. Existing Python was read and executed from its installed location. A restricted Python launch could not see the interpreter; rerunning with authorized device access succeeded.

The benchmark notes distinguish completed work from proposals. No claim is made that all weak mechanisms are fixed.

Latest continuation: [[55 - Latest EWC and Weak Benchmark Repairs]], with complete root-cause evidence in [[54 - EWC Root Cause Evidence and Complete Repairs]].

