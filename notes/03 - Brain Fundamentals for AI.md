---
tags: [research, neuroscience, biology]
created: 2026-10-05
---
# Brain Fundamentals for AI Design

Back to [[01 - Deep Research Report]]

*Textbook-level facts plus key papers. Numbers are widely cited estimates.*

## Scale and energy
- ~**86 billion neurons**, ~10¹⁴–10¹⁵ synapses
- ~**20 W** (about 20% of the body's energy)
- **Most neurons are silent most of the time** (sparse coding): only a small fraction fire at any moment
- **Communication costs ~35× more energy than computation** in the cortex (Levy & Calvert, PNAS 2021). **Lesson: move less data.**
- **Memory and compute are the same place**: a synapse both stores and computes. There's no von Neumann bottleneck. **Lesson for hardware.**

## Brain organs → AI design lessons
| Brain part | What it does | AI lesson | Existing AI work |
|---|---|---|---|
| **Neuron + dendrites** | One neuron computes like a **5–8-layer deep network** (Beniaguev et al., *Neuron* 2021) | Smarter units mean far fewer parameters | Dendritic ANNs: 10–1000× fewer params |
| **Synapse** | Stores strength, learns locally (STDP: timing-based) | Local learning, no global backprop | Spiking nets with STDP; memristor synapses |
| **Spikes** | All-or-none events; information in timing | Event-driven, compute only on change | Spiking LLMs, Loihi |
| **Neocortex** | Slow, stable knowledge; repeated **columns** | Modular, reusable units | MoE; Thousand Brains (Numenta) |
| **Hippocampus** | Fast learning of new memories; **pattern separation** (DG) and **completion** (CA3) | Fast memory module beside slow weights | HiCL, Titans, Memory Layers |
| **Complementary Learning Systems** | Hippocampus learns fast, cortex learns slowly, and sleep transfers between them | Solves catastrophic forgetting | CLS-inspired continual learning |
| **Sleep** | Replays and consolidates memories, prunes synapses | Offline consolidation phase | Sleep-replay nets (*Nature Communications* 2022) |
| **Thalamus** | Gates and routes information to cortex | Router and effort gate | MoE routers, halting |
| **Prefrontal cortex** | Planning, goals, working memory | Structured planning before acting | Agent planners; [[Structured Think-Before-Build Flow]] |
| **Basal ganglia + dopamine** | Action selection; dopamine = **reward prediction error** | Reinforcement learning (TD learning came from here) | RL in all reasoning models |
| **Neuromodulators** (dopamine, serotonin, acetylcholine, noradrenaline) | Change *how* the brain learns (learning rate, exploration, focus) | Context-dependent learning rates | Meta-learning, neuromodulated nets |
| **Astrocytes / glia** | Support, regulate synapses and energy | Slow regulatory layer | Early research |
| **Myelin** | Speeds up important long connections | Optimise high-traffic paths | Hardware interconnect design |
| **Neurogenesis** | New neurons (limited in adults, mainly hippocampus) | Grow new units for new knowledge | Expandable networks |
| **DNA / genome** | ~750 MB blueprint that **builds** a brain holding vastly more information | Compress weights into a small generator | **Genomic bottleneck** (PNAS 2024) |

## Five principles to copy
1. **Sparse**: activate only what's needed
2. **Local**: compute where the memory is
3. **Event-driven**: work only when something changes
4. **Two-speed learning**: fast memory + slow knowledge + sleep consolidation
5. **Compressed blueprint**: grow the network from a small genome

## Correction to our earlier analogy
Kakashi's "data = DNA" is closest to the **hippocampus / memory store** (stored experiences). The biological **DNA** analogue is the **genome-compressed weight generator**. We use both in the blueprint.

## Sources
- Brain energy, communication vs computation: https://www.pnas.org/doi/10.1073/pnas.2008173118
- Attwell & Laughlin energy budget: https://journals.sagepub.com/doi/10.1097/00004647-200110000-00001
- Single neuron as deep net: https://www.biorxiv.org/content/10.1101/613141v2
- Dendritic ANNs: https://www.nature.com/articles/s41467-025-56297-9
- Complementary Learning Systems update: https://www.sciencedirect.com/science/article/pii/S1364661316300432
- Sleep-like replay: https://www.nature.com/articles/s41467-022-34938-7
- Genomic bottleneck: https://www.pnas.org/doi/10.1073/pnas.2409160121
- Thousand Brains Project: https://www.numenta.com/wp-content/uploads/2024/06/Short_TBP_Overview.pdf
