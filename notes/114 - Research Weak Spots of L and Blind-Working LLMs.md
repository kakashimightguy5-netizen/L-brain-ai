---
tags: [L, research, literature, weak-spots, metacognition, hardware, LLM-limits]
created: 2026-10-09
status: research map (3 parallel research passes; ~190 sources checked; items marked as in the source reports)
---
# 114 - Research: L's weak spots, "blind working" LLMs, and brain-inspired fixes

Previous: [[113 - Fixing Everything Phase 0-3 Progress]] · Plans: `docs/L_metacognition_plan.md`

## A. L's own weak spots: what the field already knows
| L weak spot (measured) | What the literature says | Key sources | What to adopt |
|---|---|---|---|
| Lower ceiling with more data (fixed-size state) | It is the **associative-recall gap**. Recurrent models need state that grows with the number of key-value pairs; transformers copy exponentially long strings. **A single attention layer or a retrieval module closes the gap** (formal result). | Zoology/MQAR 2312.04927 · Repeat After Me 2402.01032 · RNNs are not Transformers (Yet) 2402.18510 · Based 2402.18668 | Run MQAR on L with/without hippocampus and delta memory. A tiny exact local window is the cheapest known patch. |
| Additive memory saturates → delta rule fixed it | Exactly the published story: linear attention = additive fast weights, and the delta rule fixes the capacity overload. Gated DeltaNet + sliding window is today's best recipe. Kimi Linear (48B) validates gated delta + sparse full attention at production scale. | Schlag 2102.11174 · DeltaNet 2406.06484 · Gated DeltaNet 2412.06464 · Kimi Linear 2510.26692 · hybrid ratios 1:3–1:7 (2507.06457) | Our memory family is the right bet. Per-channel learning rates (RWKV-7), Householder multi-step (DeltaProduct). |
| GPU training 3.2× slower; triangular solves | Solved in-kernel by FLA's chunkwise WY/UT Triton kernels (`chunk_gated_delta_rule`). Spikes inside a chunked kernel = **nobody has done it**. | FLA 2312.06635 · Tiled FLA 2503.14376 · Mamba fused scan 2312.00752 | Use FLA (needs Triton → Linux or triton-windows). Fuse spike+STE ourselves — first-of-kind. |
| Sparsity not usable on GPU (union across batch) | **Independently confirmed**: MLP sparsity vanishes as batch grows (Polar Sparsity 2505.14884). SpikingBrain (7B/76B) trains integer spike counts **densely on GPU** and exploits sparsity only on neuromorphic hardware. No spiking LLM reports a measured GPU training speed-up. | Polar Sparsity · SpikingBrain 2509.05276 · Deja Vu, PowerInfer, TEAL (batch-1 wins only) | Exploit sparsity at batch-1 decode and on our own hardware; stop expecting it on GPU training. |
| "Think less" skipping breaks recurrent state | Confirmed: token pruning of SSMs causes large drops (2410.14725). The known fixes: **never skip the state update** — cheap partial update (Skim-RNN), vary FFN/depth only (MoD, Mixture-of-Recursions), propagate state (CALM). | Skim-RNN 1711.02085 · Skip RNN 1708.06834 · CALM 2207.07061 · MoD 2404.02258 · MoR 2507.10524 | Our skim design + post-hoc gate is on the right track; next step: skim = state + memory only, FFN skipped. |
| Copy over-trust | Sentinel "none of the above" inside the pointer softmax; adaptive λ from entropy, match length, count, disagreement. **No calibrated (ECE) copy gate published** in byte/recurrent LMs. | Pointer Sentinel 1609.07843 · adaptive kNN-MT 2105.13022 · efficient kNN-LM 2109.04212 · kNN-LM fails in generation 2305.14625 | Our calibrated mixture = sentinel-like; add disagreement + match length features; evaluate on generation. |
| Counting / indentation | Bounded, saturating units can't count; need one **unbounded additive integer counter channel** (a=1, b∈{−1,0,+1}). Negative eigenvalues (we have) enable parity; modern linear RNNs count worse than classic RNNs. | Weiss 1805.04908 · Grazzi 2411.12537 · Chang & Bisk 2405.20131 · DeltaProduct 2502.10297 | Add an integer counter channel; our layout place cells are a hand-made version. |
| Integer / FPGA deployment | Quantise projections, keep state & decay at 16-bit (Quamba/MambaQuant). FPGA templates: LightMamba, HFRWKV; MatMul-free LM on Loihi 2 (3× throughput, 2× less energy, preliminary). **No FPGA delta-rule memory and no hardware hash-recall LM published** → L-HW would be first. | Quamba 2410.13229 · LightMamba 2502.15260 · HFRWKV 2601.02135 · Loihi 2 LLM 2503.18002 | Matches our L-HW choices (int8 weights, 24-bit state). |

**Novelty that survived the search:** hash-based exact n-gram store + delta-rule memory in one end-to-end LM; spikes fused into a chunkwise linear-recurrence kernel; FPGA delta-rule memory; calibrated copy gate with ECE.

## B. "Blind working" in today's LLMs (verified)
- **Both arXiv IDs from the brief exist:**
  - 2605.25284 (Su & Cardie 2026): LLMs **recognise ambiguity but rarely ask**, and retrieved context makes asking even rarer.
  - 2601.07767 (RiskEval 2026): confidence does not drive decisions; models rarely abstain even when they should.
- **Recurring theme 2025–2026:** models *have* the signals (uncertainty, ambiguity) but **don't act on them**.
  - A monitor-to-control harness gave +8.6 points without training (2605.14186).
  - This supports building the uncertainty → action link into the architecture.
- **Asking:** asking helps only when gated by uncertainty (MediQ 2406.00922: naive asking hurts, confidence-gated +22%). STaR-GATE, CollabLLM, CIGAsk 2609.24290 (information-gain reward). Multi-turn: −39% when the spec arrives in pieces (2505.06120).
- **Abstention:**
  - Training and benchmarks reward guessing (Why LMs Hallucinate 2509.04664).
  - Reasoning training lowers abstention by 24% (AbstentionBench 2506.09038).
  - Semantic entropy works but is costly (Nature 2024).
- **Effort:** overthinking (2412.21187), underthinking (2501.18585), AdaptThink/Thinkless (learn when to think), CALM/MoD (per-token).
- **Procedural memory:** all working skill memories live **outside the weights** (Voyager, AWM, Memp, ExpeL). Memory Layers at Scale (2412.09764) is the closest in-weight store.
- **MoE:** experts follow **syntax/token**, not domain (Mixtral; OpenMoE; part-of-speech routing). Domain experts need **sequence-level routing** (DEMix, BTX).
- **Sycophancy:** caused partly by preference training (Sharma 2023; SycEval 58%).
- **Generic answers:** the "Artificial Hivemind" (NeurIPS 2025) measures homogeneity directly.

## C. Brain-inspired precedents for our metacognitive controller
- **Uncertainty gate (3 paths): partly novel.**
  - Precedents:
    - neuroscience: Kiani & Shadlen's opt-out; arbitration by uncertainty (Daw 2005), value of information (Keramati 2011) and reliability (Lee 2014); expected value of control (Shenhav 2013);
    - AI: SOFAI / SOFAI-LM, a metacognitive arbiter between fast and slow solvers.
  - Novel: one per-step 3-way gate *inside* a spiking, attention-free LM, mapped to clock-gating.
  - Predicted risks:
    - byte-level entropy confuses spelling ambiguity with ignorance → pool uncertainty per word or segment;
    - confident errors are where calibration is worst;
    - the ask path collapses without an abstention reward;
    - ACT-style halting is fragile → use a fixed budget (MoD).
- **Skill memory with promotion and demotion: mechanism has precedent, combination novel.**
  - Precedents:
    - habits strengthen with repetition (Miller 2019);
    - reliability tracked from prediction errors (Lee 2014);
    - instance theory of automatization (Logan 1988);
    - episodic control / kNN-LM;
    - Voyager (promotion, but no demotion).
  - Risks:
    - trivial skills get promoted → weight promotion by surprise;
    - near-miss / hash-collision errors (amortised inference);
    - open-loop multi-byte skills run unchecked;
    - multi-step skills don't distil;
    - the cache's value fades as the weights learn → consolidate by replay.
- **Segment-routed experts: least novel.**
  - Precedents: DEMix, context-dependent gating, active dendrites, thalamic routing.
  - **Closest whole-system precedent to L: TRC² (arXiv 2602.22479)** — cortical columns + thalamic routing + hippocampal pathway → cite it and differentiate.
  - Risk: routers fail to specialise unless the domains are separable.
- **Energy biology:** only ~1% of cortex is active at once (Lennie 2003); food scarcity cuts synaptic ATP 29% by lowering precision (Padamsey 2022) → under an energy budget, expect fine distinctions to suffer first.
- **Hardware:**
  - verified LM deployments: Loihi 2 (MatMul-free LM), SpiNNaker2 (EGRU LM), NorthPole (3B LLM, 72.7× tokens/s/W, dense integer, near-memory);
  - **no published metacognitive gate with power gating on neuromorphic silicon** → credible novelty for SPARK.

## What changes in our plan because of this research
1. Skim path must always update state; skip only the FFN/deep compute (Skim-RNN, MoD) — already our direction.
2. Add an integer counter channel for indentation and brackets.
3. Add disagreement + match-length features to the copy gate; evaluate on generation.
4. Uncertainty gate: pool per word/segment; report AUROC2/meta-d′; reward abstention.
5. Skill promotion weighted by surprise; audit and demote; consolidate by replay later.
6. Experts: route on segment posteriors; expect weak specialisation unless domains are separable (Python/JS/C test running).
7. Hardware story: train dense on GPU (field standard), exploit sparsity on SPARK; cite SpikingBrain and Polar Sparsity for the GPU limitation.
