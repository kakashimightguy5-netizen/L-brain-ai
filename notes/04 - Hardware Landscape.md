---
tags: [research, hardware, neuromorphic]
created: 2026-10-05
---
# Hardware Landscape for Brain-Inspired AI

Back to [[01 - Deep Research Report]]

## The real bottleneck
- GPU compute grew **3.0× every 2 years**, but memory bandwidth only **1.6×** (the memory wall).
- Each token moves the model's weights from memory to compute, and **moving data costs more than computing** (true in chips *and* in the brain).
- **Storage (HDD → SSD → NVMe) helps capacity and loading**, but NVMe is still far slower than on-chip memory. The fix is **compute inside or next to memory**.

## Options compared
| Approach | Brain principle | Best result found | Maturity | Our use |
|---|---|---|---|---|
| **Near-memory digital (IBM NorthPole)** | Synapse = memory + compute | 3B LLM, <1 ms/token, **72.7× energy efficiency** vs GPUs | Research product | Design principle for our chip |
| **Neuromorphic spiking (Intel Loihi 2 / Hala Point)** | Spikes, event-driven | 1.15B neurons at 2.6 kW; 370M LLM: 3× throughput, 2× less energy | Research | Target for spiking columns |
| **Analog in-memory (gain cells, phase-change memory)** | Analog synapses | Attention **up to 70,000× less energy** than GPUs (1.5B model) | Prototype | Long-term |
| **Processing-in-memory (HBM-PIM, AiM)** | Local compute | Early products for LLM inference | Early commercial | Watch |
| **Memristors** | Plastic synapses, on-chip learning | Fully integrated edge-learning chips (*Science* 2023) | Lab | Long-term synapse device |
| **Flash/NVMe inference (LLM in a Flash)** | Long-term memory store | Runs models 2× larger than RAM; 4–5× CPU, 20–25× GPU speedup vs naive | Software technique, works today | **Hippocampus store, usable now** |
| **Living neurons (CL1)** | Real biology | 200k neurons influencing a 350M LLM | Toy | ❌ Not our path |

## Our hardware ladder
1. **Software simulation** (snnTorch, Brian2, Intel Lava)
2. **FPGA board**: event-driven columns + on-chip memory + NVMe hippocampus, on our own PCB
3. **Tiny Tapeout**: fabricate a small neuron/synapse tile (open-source shuttle; check cost and schedule on tinytapeout.com)
4. **Silicon partners** for analog / PIM / memristor at scale

## Sources
- NorthPole LLM: https://research.ibm.com/blog/northpole-llm-inference-results
- Hala Point: https://www.datacenterdynamics.com/en/news/intel-unveils-115bn-neuron-neuromorphic-system-hala-point/
- Loihi 2 LLM: https://arxiv.org/abs/2503.18002
- Analog attention: https://www.nature.com/articles/s43588-025-00854-1
- Memristor edge learning: https://www.science.org/doi/10.1126/science.ade3483
- PIM: https://www.digitimes.com/news/a20251218PD206/samsung-sk-hynix-hbm-ai-inference-data.html
- LLM in a Flash: https://arxiv.org/abs/2312.11514
- Memory wall: https://arxiv.org/abs/2403.14123
- Tiny Tapeout: https://tinytapeout.com/
