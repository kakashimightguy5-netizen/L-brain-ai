---
tags: [validation, research-plan, risks]
created: 2026-10-05
---
# Validation Gaps: What Still Needs Research

Back to [[00 - Concept and Roadmap]] · [[01 - Deep Research Report]]

> **Question (Kakashi):** The core architecture is set. Does it need more research? Does the problem statement need more research or solution validation?
> **Answer:** Yes to both. The architecture is a **blueprint built from literature**, not a validated design. Nothing has been tested yet, and most papers were read at abstract level (🔎).

## 1. Problem statement: too broad, needs ONE headline
Right now it bundles **7 problems**: energy, data centres, GPUs, memory/storage, forgetting, structured thinking, growth. Judges, investors and reviewers need **one**.
Candidates:
- **A. "Capable AI without a data centre":** run a useful model on edge hardware at X joules per answer
- **B. "Learning without forgetting":** add new domains instantly, with no retraining and no forgetting
- **C. "Thinking only when needed":** compute proportional to difficulty (the symposium paper)
- [ ] Pick one headline; the others become supporting benefits

Missing validation:
- [ ] **Baseline numbers measured ourselves** (e.g. Qwen3-4B on the RTX 3050: energy, latency and memory per answer)
- [ ] **Who exactly suffers** and how much (edge/offline users, Indian SMEs, hospitals, defence/air-gapped sites)
- [ ] **Market/user validation** for the startup path: 10–15 interviews
- [ ] A Jevons-paradox position (we cut the per-answer requirement; we don't promise lower global demand)

## 2. Architecture: 8 open risks
| # | Module | Unknown | Risk |
|---|---|---|---|
| 1 | Integration | Do the modules work *together*, or does combining them hurt quality? | 🔴 High |
| 2 | Spiking execution | Spiking nets still lose quality on language; training is hard | 🔴 High |
| 3 | Auto-growing columns | How to *detect* a new domain reliably, and how to prevent endless growth? | 🔴 High |
| 4 | Genome compression | Never tried on language models | 🔴 High |
| 5 | Sleep consolidation | Design undefined (what to replay, when) | 🟡 Medium |
| 6 | Hippocampus on NVMe | Latency of SSD lookups per token | 🟡 Medium |
| 7 | Effort gate | Halting training is unstable (known ACT issue) | 🟡 Medium |
| 8 | Planner stage inside the model | Is it architecture or just prompting? | 🟡 Medium |

## 3. Research not done yet
- [ ] **Deep-read** the key papers (we only have abstracts): SpikingBrain 1/2, HiCL, Titans/HOPE, Memory Layers, genomic bottleneck, TRM, dendritic ANNs
- [ ] **Competitor scan** (needed for a startup): Liquid AI, Cartesia (state-space models), BrainChip, Innatera, SynSense (neuromorphic), Numenta, Rain AI, IBM NorthPole: what they've built and where the gap is
- [ ] **Licences** of base models (Qwen, Llama, Granite) for commercial use
- [ ] **Evaluation benchmarks**: language quality, continual learning, energy
- [ ] **Tooling**: snnTorch vs SpikingJelly vs Lava; FPGA spiking frameworks
- [ ] **Patent check** on the combined idea

## 4. Validation plan: cheapest tests first, with kill criteria
| Step | Test | Where | Kill / pivot if… |
|---|---|---|---|
| V1 | Desk: deep prior-art + competitor scan | Research | Someone already built the full combination, so pivot to a narrower novelty |
| V2 | Baselines: measure energy/latency/memory of small open models | RTX 3050 | n/a (needed for every claim) |
| V3 | **Riskiest first:** auto-growing domain columns without forgetting (tiny model, 2–3 domains) | RTX 3050 | Forgetting isn't lower than normal fine-tuning plus replay |
| V4 | Hippocampus memory + sleep consolidation | RTX 3050 | No gain over plain RAG |
| V5 | Spiking version of the winning modules | RTX 3050 | Quality drop > agreed threshold |
| V6 | User interviews (problem headline) | Field | Nobody has the problem badly enough |

## 5. Don't forget
⚠️ **Symposium first:** ZORYNX (09 Oct), ARVENA (10 Oct), ARVEXIS (14 Oct). The **Think-When-Needed pilot experiment hasn't been run yet**, and the slides aren't built. This long-term project shouldn't eat those 4 days.
