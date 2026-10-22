---
tags: [spikingbrain, experiment, neurons, validation]
created: 2026-10-05
---
# SpikingBrain neuron experiments

Back to [[00 - Concept and Roadmap]] · [[08 - SpikingBrain Process Log]] · [[09 - SpikingBrain Architecture and Prototype Decisions]]

## Result
**15/15 checks passed**, on CPU, using the official unmodified 7B encoder and QuantLinear implementations. This demonstrates encoding and a toy linear layer, not trained language-model inference. No model weights downloaded.

## Storage and reproducibility
- Source: D:\AI\SpikingBrain-study\SpikingBrain-7B-main and SpikingBrain2.0-main
- Runnable script: D:\AI\SpikingBrain-study\experiments\run_demo.py
- Raw results: D:\AI\SpikingBrain-study\experiments\results.json
- Python: 3.12.10, PyTorch: 2.14.0+cpu, seed 42, CPU threads 2.
- New temporary files and matplotlib cache are on D:. Python bytecode writes disabled. Existing Python runtime is read from C:; nothing installed there.
- Rerun in PowerShell: `Set-Location 'D:\AI\SpikingBrain-study\experiments'; py -3.12 -B .\run_demo.py`
- Source SHA256 hashes and all test results are in [[SpikingBrain Assets/results.json]]. Upstream source files were not modified.

## What the neurons actually do
Binary count: +5 becomes five +1 spikes. Ternary count: -5 becomes five -1 spikes. Signed bitwise: +5 becomes [1,0,1], -5 becomes [-1,0,-1], decoded with weights [4,2,1]. Bit positions represent numerical weights, not equal-duration biological firing rates.

Although classes are named LIF, these count encoders do not implement a leaky membrane differential equation. Their internal state is a remaining count or bit position. They should not be described as evidence of biological membrane simulation or learning.

![[SpikingBrain Assets/spike-encoding.png]]

## Small encoding comparison
Nine inputs: [-15,-7,-5,-1,0,1,5,7,15]. Binary uses absolute values; ternary and signed bitwise retain the sign. Timing: 5 warmups and 30 measured CPU calls per mode; medians below. These small-input timings are illustrative, not full-model benchmarks.

| Encoding | Steps | Nonzero events | Median CPU ms |
|---|---:|---:|---:|
| Binary count (absolute values) | 15 | 56 | 0.337 |
| Ternary count (signed) | 15 | 56 | 0.230 |
| Bitwise (signed) | 4 | 20 | 0.186 |

Fewer nonzero events or encoding steps do not establish lower energy use. PyTorch uses dense tensors and bitwise spike positions carry different weights.

## QuantLinear connection
The official path sets threshold to mean(abs(activation))/3, rounds activations into counts, reconstructs them and applies group-rounded weights through a conventional dense linear operation. This is fake quantization in floating-point tensors, not a packed INT8 weight implementation.

Toy dimensions: 8 inputs, 6 outputs, batch 4; random untrained weights; group size 4; weight scale 0.125. Independent quantized-reference maximum error: 0.00000000. Float-original maximum error: 0.565780; mean absolute error: 0.238223. This quantization error says nothing about trained model quality.

![[SpikingBrain Assets/linear-layer-comparison.png]]

## Tests
| Check | Result |
|---|---|
| Binary round trip 0..255 | PASS |
| Ternary round trip -255..255 | PASS |
| Unsigned bitwise round trip 0..255 | PASS |
| Signed bitwise round trip -255..255 | PASS |
| Twos complement with explicit sign correction | PASS |
| binary zero round trip | PASS |
| ternary zero round trip | PASS |
| bitwise zero round trip | PASS |
| Binary rejects negative | PASS |
| Bitwise unsigned rejects negative | PASS |
| binary rejects fractional input | PASS |
| ternary rejects fractional input | PASS |
| bitwise rejects fractional input | PASS |
| Official fake quantization mixed-sign input | PASS |
| Official QuantLinear matches independent reference | PASS |

## Observed edge cases and API limitations
- **binary insufficient T=2 for value 5**: Short windows silently lose count/bit information. Observed: `{"decoded": [2.0], "expected": 5}`
- **ternary insufficient T=2 for value 5**: Short windows silently lose count/bit information. Observed: `{"decoded": [2.0], "expected": 5}`
- **bitwise insufficient T=2 for value 5**: Short windows silently lose count/bit information. Observed: `{"decoded": [1.0], "expected": 5}`
- **Reused binary encoder after value 1 then value 5**: Automatic T persists for binary/ternary objects; use fresh instances or explicitly set T. Observed: `{"steps": 1, "decoded": [1.0]}`
- **Twos-complement direct official decoder**: Direct decoder sums unsigned bit weights; subtract 2**T for sign-bit entries to recover signed values. Observed: `{"decoded": [11.0, 5.0]}`
- **Signed fake quantizer with all-positive input**: Wrapper requires zero-point handling for this case; direct encoder works. Observed: `{"error": "x_zero is required for bidirectional mode when x is non-negative."}`

Use fresh encoders or explicit sufficient T. For two's complement direct decoding, apply signed correction. The experiment correction is outside upstream code. The wrapper zero-point conventions deserve separate tests before production use.

## Limits
No language quality evaluation, training, GPU throughput test, full 5B/7B forward pass, power measurement or proof of reduced forgetting. Full-model offloading remains unverified.

## Sources
- https://github.com/BICLab/SpikingBrain-7B/blob/main/W8ASpike/Int2Spike/neuron.py
- https://github.com/BICLab/SpikingBrain-7B/blob/main/W8ASpike/quant_linear.py
- https://github.com/BICLab/SpikingBrain2.0
