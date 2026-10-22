# 63 - Training Root Cause Repair Protocol

Registered before new training. All files, caches and logs stay on D:.

Test three changes in order: train with the existing 60-per-class lossless memory budget; allocate half each batch to the current task; select checkpoints by equal current/past validation accuracy. All variants use 12 epochs, 24 updates per epoch, batch64, identical architectures and initial weights for pilot seed 1562. Checkpoints are chosen after training epochs, excluding the untrained task entry checkpoint. That shared change must be considered when comparing historical models.

Choose one common method using the mean ANN/CNN/LIF clean validation score, then repeat it with fresh seeds 1563 and1564. Only after all training and selection are frozen, evaluate 500 unused original-training images per global class. Exclude every earlier train and validation image. Previously inspected official tests are not used for selection or this new final evaluation.

Compare native predictions, clean and one noisy observation; report newest-task learning and past-task retention separately. The 1.5MiB cap covers serialized inference state and lossless retained images, not working RAM. This is a small prototype experiment, not SpikingBrain LLM fine tuning.
