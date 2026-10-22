---
tags: [innovation, proposals, novelty, roadmap]
created: 2026-10-06
---
# Innovation Proposals

Back to [[16 - Root Cause Analysis and Fixes]] · [[01 - Deep Research Report]]

> Kakashi: propose ideas that don't exist yet, or that exist but don't work well, and do better.
> **Novelty status** is from web searches on 06 Oct 2026. "No match found" is not proof; a formal prior-art and patent search is still needed.

## A. Tested in our lab
| # | Proposal | Closest prior art | What's new in ours | Lab status |
|---|---|---|---|---|
| 1 | **Self-knowing columns**: each module has a NOT-MINE output; old modules re-learn it during sleep when a new neighbour arrives | CIL = task-ID × within-task, OOD-based routing (Kim 2022, HAT+CSI) | Lightweight self-rejection inside each module + **sleep re-calibration from neighbours' memories**; no separate OOD model | ✅ Works |
| 2 | **Dream-intensity principle**: dreams must be "weird but recognisable" | Overfitted-brain hypothesis (Hoel 2021); augmentation | Measured a **sweet spot** for routing robustness; too-strong dreams cut accuracy by more than half | ✅ Finding |
| 3 | **Protected sleep**: recalibrate only the self-knowledge head; heal with gentle updates anchored to the pre-sleep self | LwF distillation, synaptic consolidation | Diagnosed **sleep-induced self-forgetting** in modular systems and separated *who-am-I* updates from *what-I-know* updates | ✅ Turned healing from +0 to +5 pts |
| 4 | **Genomic columns**: one frozen genome (DNA) + per-neuron gene-expression codes; each new skill = only new codes | Hypernetwork CL (von Oswald 2020; partial hypernetworks 2023) use **task-ID embeddings** + regularisation | **Neuron-level** codes; genome frozen after the first domain, so **zero forgetting by construction**; cost ∝ neurons; shared across layers | ✅ 7.6× smaller per skill, 91% vs 94% |
| 5 | **Learned gene-expression genome** (fix of the genomic bottleneck) | Genomic bottleneck (PNAS 2024) uses fixed Gray-code IDs | Showed fixed hidden IDs carry **no information**; learned codes beat equal-size low-rank as width grows | ✅ +2 to +12 pts vs low-rank |
| 6 | **Modularity headroom test**: before choosing modules, measure domain-ID accuracy under shift vs task accuracy; modules pay only when domains are more separable than their contents | MoE/CL theory | A practical **go/no-go diagnostic** for modular designs | ✅ Explained our failure |
| 7 | **Three-tier memory**: hippocampus (exact) → column (fast learning) → genome codes (compressed), consolidated in sleep | "Language Models Need Sleep" (Google, Jun 2026: consolidation + dreaming); distillation | Compression *into genome codes* as the final consolidation tier | 🟡 Loses 8–13 pts without original data |

## B. New proposals not tested yet (ranked by promise)
1. **Blame-aware self-healing.** With self-knowing columns, every mistake can be attributed: *wrong column claimed it* → heal the router heads only; *right column, wrong answer* → heal that column only. This fixes the weak attribution behind Amazon's self-healing failures.
2. **Genome-LoRA for LLMs.** Each new skill of a language model = LoRA adapters *generated* by one frozen genome from per-neuron codes, so a skill costs **kilobytes, not megabytes**. Phones could hold thousands of skills. *(Check against HyperLoRA / Text-to-LoRA.)*
3. **Metabolic budget controller.** Like the brain's fixed ~20 W, a single joule budget is shared across the effort gate (loops), column activation and spike rate. The model learns where to spend energy per query. Directly targets the data-centre problem.
4. **Effort gate per column.** Hard domains get more thinking loops, easy domains fewer (combines [[Think-When-Needed Paper]] with columns).
5. **Self-knowledge as abstention.** If no column says "mine", the system answers *"I don't know this domain yet"* and triggers growth, instead of hallucinating.
6. **Position-aware sensory genome.** Fix translation failures by giving input neurons geometric codes (as in real retina/cortex maps) while hidden codes are learned.
7. **Sleep scheduling by surprise.** Sleep is triggered when the error monitor's surprise rate crosses a threshold, not on a timer; healing frequency adapts to how fast the world changes.

## C. Honest competitive picture
- Google (Titans, Nested Learning, "LMs Need Sleep") is moving on **memory + sleep** for LLMs.
- BICLab (SpikingBrain) leads **spiking LLMs**; IBM/Intel lead **brain-like chips**.
- **Our angle: modular, self-knowing, self-healing, genome-compressed skills that run on small hardware.** No single published system combines these as far as our searches found.

## Sources
- Kim et al., A Theoretical Study on Solving Continual Learning: https://arxiv.org/pdf/2211.02633
- Hoel, The overfitted brain hypothesis: https://www.cell.com/patterns/fulltext/S2666-3899(21)00094-5
- von Oswald et al., Continual learning with hypernetworks: https://arxiv.org/pdf/1906.00695
- Partial Hypernetworks for Continual Learning: https://arxiv.org/html/2306.10724v1
- Behrouz et al., Language Models Need Sleep (Jun 2026): https://arxiv.org/html/2606.03979v1
- Genomic bottleneck: https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2.full.pdf
