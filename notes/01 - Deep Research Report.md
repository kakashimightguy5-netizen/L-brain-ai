---
tags: [research, deep-research, ai-architecture, neuroscience, hardware]
created: 2026-10-05
status: v1
---
# Deep Research Report: Biology + Tech for Low-Compute AI

Back to [[00 - Concept and Roadmap]] · Details: [[02 - LLM Evolution]] · [[03 - Brain Fundamentals for AI]] · [[04 - Hardware Landscape]] · [[05 - Local Model Setup]]

> **Research question (Kakashi, 05 Oct 2026):** Can we reduce the compute time, power and *data-centre requirement* of AI by embedding biology into the architecture (software first, then hardware)? What already exists, what doesn't, and what will we do?

---

## 0. Short answer
- **Yes, per answer.** The evidence that AI can get radically cheaper per answer is strong: capability per parameter doubles about every **3.5 months** (Densing Law, *Nature Machine Intelligence* 2025), Google cut energy per prompt **33× in one year**, and brain-like chips show **~70× better energy efficiency** (IBM NorthPole) and up to **70,000× for attention** in analog memory (*Nature Computational Science* 2025).
- **But not automatically in total.** Per-answer gains have so far been *eaten by more usage* (Jevons paradox) and by reasoning models that write **300–800× more tokens**. Data-centre power still went **415 → 485 TWh** (2024 → 2025).
- **Where we can win:** make capable models run **on small, local, brain-like hardware**, which moves work *out of* data centres. That reduces the **requirement** for data centres, which is exactly Kakashi's goal.

---

## 1. WHAT ALREADY EXISTS

### 1A. In AI software
| Brain idea | Already built | Scale reached | Note |
|---|---|---|---|
| Sparse firing (few neurons active) | Mixture-of-Experts (DeepSeek-V3: 671B total / **37B active**; Qwen3-Next 80B / **3B active**) | Frontier | Mainstream |
| Spiking neurons | SpikingBrain-7B / 76B (2025), **SpikingBrain2.0-5B** (Apr 2026), SpikeLLM, SpikingLM (ICML 2026) | 5–76B | ⚠️ Mostly *converted from Qwen*; spikes are **simulated on GPUs**, not event-driven |
| Efficient "memory-state" processing instead of full attention | Mamba hybrids: IBM Granite 4.0 (**>70% less RAM**), NVIDIA Nemotron-H, Qwen3-Next | Up to ~80B | Mainstream in 2026 |
| Hippocampus-like fast memory | Google **Titans** (learned memory at test time), Meta **Memory Layers** (beat dense models with 2× compute) | 1–8B | Research → early adoption |
| Multi-speed learning (fast/slow synapses) | Google **Nested Learning / HOPE** (2025) | 340M–1.3B | Research |
| Hippocampal circuit (DG/CA3/CA1) | **HiCL** (2025): pattern separation + completion, single expert per input, **4–6× fewer FLOPs** | Small (CIFAR) | Research |
| Sleep replay against forgetting | Sleep-like replay (*Nature Communications* 2022), sleep-inspired consolidation for LLMs (2026) | Small | Research |
| Dendrites (a neuron = a mini-network) | Dendritic ANNs: **10–1000× fewer parameters** for the same accuracy (*Nature Communications* 2025) | Small (images) | Research |
| DNA-like compression | **Genomic bottleneck** (PNAS 2024): a tiny "genome" network generates the weights; works untrained on images/games | Small | Research. *This is the real "data as DNA" idea.* |
| Adaptive effort (think only when needed) | ACT, PonderNet, CALM, TRM (7M params), GPT-5's fast/thinking router | Up to frontier | Mainstream as a *router*, research as an *architecture* |
| Growing new modules for new domains | Progressive nets, expandable MoE adapters (CVPR 2025), expandable spiking nets (2026) | Small–mid | Research; mostly vision |
| Running models from storage (SSD) | Apple **"LLM in a Flash"**: runs models 2× larger than RAM, **4–5× (CPU) / 20–25× (GPU)** faster than naive loading | 7B-class | Proven technique |

### 1B. In hardware
| Approach | Example | Result |
|---|---|---|
| Memory + compute on the same chip (like synapses) | **IBM NorthPole**: 3B LLM, <1 ms/token, **72.7× more energy-efficient** than GPUs | 16 cards in one 2U server |
| Neuromorphic (spiking) | **Intel Hala Point**: 1.15B neurons, 128B synapses, **2.6 kW max** (owl-brain scale); Loihi 2 LLM (370M): **3× throughput, 2× less energy** | Research systems |
| Analog in-memory compute | Gain-cell attention (*Nature Computational Science* 2025): **up to 70,000× less energy** for attention; IBM phase-change memory chips | Prototypes |
| Processing-in-memory (PIM) | Samsung HBM-PIM, SK hynix AiM | Early products |
| Memristor synapses | Memristor chips with on-chip learning (*Science* 2023; many 2025–26 papers) | Lab chips |
| Living neurons | Cortical Labs CL1 (2025); 200k neurons on an LLM (Mar 2026) | Toy-scale. **Not our path.** |
| DIY chip fabrication | Tiny Tapeout (open-source mini-ASIC shuttles) | Cheap real chips for students |

---

## 2. WHAT DOESN'T EXIST YET (the gaps)
*Based on this search; "doesn't exist" means we found no public evidence. Re-check before claiming.*

1. **No complete "brain blueprint" LLM.** Each organ exists separately (sparse cortex, hippocampal memory, sleep, dendrites, spikes, effort control, genome compression), mostly at **tiny scale** and never **integrated into one model**.
2. **No LLM that grows its own domain modules automatically** (neurogenesis-style: detect a new domain, grow a node, route to it) at language-model scale. Existing work is mainly vision benchmarks.
3. **No true event-driven spiking LLM at multi-billion scale with quality parity.** SpikingBrain's spikes are a GPU *approximation*; real neuromorphic LLMs are ≤ ~370M.
4. **No LLM with brain-like "sleep" consolidation at scale** (fast memory → slow weights offline), only early papers.
5. **No genome-compressed LLM**: the genomic bottleneck has never been applied to language models.
6. **No architecture with *structured cognition* built in** (why → who → what → how → impact before acting); today it's only prompting or agents. See [[Structured Think-Before-Build Flow]].
7. **No hardware co-designed for this combination**: neuromorphic chips run spikes, NorthPole runs transformers, PIM runs matrix maths. None is built for *sparse + growing + memory-centric* models.
8. **No standard full-cascade efficiency metric** (accuracy per joule + memory + storage + carbon per answer).

---

## 3. WHAT WE ARE GOING TO DO

### 3A. The architecture (working blueprint, each part mapped to a brain organ)
| Our module | Brain analogue | Built from (existing work) | Our novelty |
|---|---|---|---|
| **Cortex**: base language ability | Neocortex (slow, stable knowledge) | Upcycle an open model (e.g. Qwen3-4B) to a hybrid linear-attention + sparse model, as SpikingBrain2.0 did | Starting point, not novel |
| **Domain columns**: one module per domain (frontend, gym, PCB…) | Cortical areas / columns | MoE experts + expandable adapters | **Auto-growth**: a new column is created when a novel domain is detected |
| **Hippocampus**: fast memory / "second brain" (Kakashi's "DNA data") | Hippocampus | Memory Layers, Titans, HiCL | Stored on **NVMe SSD** with flash-paging; learns instantly without retraining |
| **Sleep cycle**: consolidation | Sleep replay | Sleep-replay papers | Offline transfer from hippocampus to domain columns, so nothing is forgotten |
| **Thalamus gate**: effort control | Thalamus / attention + "fast vs slow thinking" | ACT, CALM, TRM, our [[Think-When-Needed Paper]] | Halts early on easy inputs, loops on hard ones |
| **Prefrontal planner**: structured thinking | Prefrontal cortex (planning) | Agent planners | Why/who/what/how/impact stages *inside* the model's loop |
| **Spiking execution**: event-driven, mostly silent | Neurons and spikes | SpikingBrain, Loihi LLMs | Only active columns spike; ready for neuromorphic hardware |
| **Genome**: compressed weight generator | DNA (blueprint) | Genomic bottleneck | Store a small genome on device and unfold columns when needed, so storage shrinks |
| **Error Monitor**: self-healing *(added 06 Oct, Kakashi's idea)* | Anterior cingulate (error detection) + cerebellum (correction) | WISE, SCoRe, Amazon self-healing | Detect with an **external signal** → write fix to hippocampus instantly → **sleep consolidates** → never edit weights directly. Validated in [[07 - Experiment Results]] (E2) |

> **Lab Round 1 status (06 Oct):** effort gate ✅, self-healing ✅, growing columns ✅ (clean), spiking ✅, genome 🟡, **integration ❌ (routing under shift)**. See [[07 - Experiment Results]].

### 3B. Hardware track (after the architecture proves itself)
1. **Simulate** spiking and event-driven columns in software (snnTorch / Lava).
2. **FPGA prototype**: event-driven columns with memory next to compute (the NorthPole principle), on our own PCB with NVMe for the hippocampus.
3. **Tiny Tapeout**: fabricate a small neuron/synapse tile as a real chip.
4. **Partnerships** for analog/PIM silicon (research labs, India Semiconductor Mission programmes: *verify*).

> 🔧 **Correction on storage:** the bottleneck isn't how *big* the storage is. It's **moving data between memory and compute** on every token. The brain spends ~35× more energy on *communication* than computation (PNAS 2021). So the redesign goal is **compute where the memory is** (NorthPole, analog, PIM). Faster NVMe helps as the hippocampus store, but alone it doesn't fix it.

### 3C. How we prove it (metrics)
- Accuracy per joule, per GB of memory, per GB of storage
- Forgetting rate when learning new domains
- % of the model active per answer
- Can it run on a laptop/edge board, i.e. **zero data centre needed**?

### 3D. Research → startup path
Paper (symposium) → open-source tiny prototype → publication → FPGA demo board → startup ("brain-inspired edge AI that doesn't need a data centre").

---

## 4. Honest risks
- Each component is proven only at small scale; **integration may fail** or cost quality.
- Spiking training is hard; neuromorphic tooling is immature.
- Big labs move fast (GPT-5 already routes fast vs slow thinking; Google works on continual learning). **Our edge must be the integration plus edge hardware**, not any single trick.
- Jevons paradox: cheaper AI means more AI. We reduce the *per-answer* requirement; total demand also depends on policy and usage.

## Sources
See [[Paper Library]] plus: Densing Law (https://www.nature.com/articles/s42256-025-01137-0) · Epoch price trends (https://epoch.ai/data-insights/llm-inference-price-trends) · SpikingBrain (https://arxiv.org/abs/2509.05276) · SpikingBrain2.0 (https://arxiv.org/html/2604.22575v1) · Granite 4.0 (https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) · Qwen3-Next (https://vllm.ai/blog/2025-09-11-qwen3-next) · Titans (https://arxiv.org/pdf/2501.00663) · Nested Learning (https://www.marktechpost.com/2025/11/08/nested-learning-a-new-machine-learning-approach-for-continual-learning-that-views-models-as-nested-optimization-problems-to-enhance-long-context-processing/) · Memory Layers (https://arxiv.org/abs/2412.09764) · HiCL (https://arxiv.org/html/2508.16651v2) · Dendritic ANNs (https://www.nature.com/articles/s41467-025-56297-9) · Genomic bottleneck (https://www.pnas.org/doi/10.1073/pnas.2409160121) · LLM in a Flash (https://arxiv.org/abs/2312.11514) · NorthPole (https://research.ibm.com/blog/northpole-llm-inference-results) · Hala Point (https://www.datacenterdynamics.com/en/news/intel-unveils-115bn-neuron-neuromorphic-system-hala-point/) · Analog attention (https://www.nature.com/articles/s43588-025-00854-1) · Loihi 2 LLM (https://arxiv.org/abs/2503.18002) · Sleep replay (https://www.nature.com/articles/s41467-022-34938-7) · Brain communication energy (https://www.pnas.org/doi/10.1073/pnas.2008173118) · DeepSeek-V3 (https://arxiv.org/pdf/2412.19437)
