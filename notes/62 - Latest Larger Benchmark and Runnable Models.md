# Latest larger-benchmark result

Completed ten alternating digit/clothing tasks and evaluated the full 20000 official test images with no task identity at inference. Three initialization runs share those test examples. Training used 30000 selected images; validation used 6000.

Original clean 20-class means: ANN 67.71%, CNN 73.60%, selected LIF 60.51%. These are a harder benchmark than earlier 8 x 8 digits.

Packed 1200-reference combined means: ANN 70.20%, CNN 75.49%, LIF 67.97%. Packed memory alone with 1920 references: 66.72%. All are below the same 1.5 MiB stored-model ceiling. Packing is lossless pixel encoding, not neural-weight quantization. Extra references support inference; neural training remained unchanged.

The spiking pilot selected 8 steps and threshold 1 using validation. Other thresholds/time steps failed to improve it. Spiking has no established energy or speed advantage in this CPU simulation. Its neural model scored only 9.78% on the newest clothing pair, so its weakness includes failing to acquire that task.

See [[61 - Larger Ten Task Benchmark and Spiking Diagnosis]] for domain-specific accuracy, noise, forgetting, harm, bytes, timings and limits. Protocols [[59 - Larger Benchmark and Ten Task Frozen Protocol]] and [[60 - Lossless Reference Packing Storage Control]] were registered before official-test evaluation. Raw predictions, datasets/provenance, deployment models and code are archived under SpikingBrain Assets/larger-ten-task. Everything stays on D:.
