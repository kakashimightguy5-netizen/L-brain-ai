# 141 - Brain-principle fixes for L's 9 weak spots (2026-10-21 IST). Research + PROPOSALS. Nothing below is tested on L yet.
Source quality: paper claims are the authors' own, read from fetched pages. Items marked HYPOTHESIS are my adaptation, not a published result.

## Research read
- Hippocampal "barcode" model (eLife reviewed preprint 103512): a random chaotic recurrent net turns correlated inputs into near-orthogonal sparse patterns (barcodes); Hebbian plasticity binds barcode+content; recall from the cue alone. Separates index from content; tunable search radius trades separation vs completion. The text reports NO capacity figure and only 3-cache tests.
- "Language Models Need Sleep" (arXiv 2605.26099): when context fills, run N extra recurrent passes over the context to refine fast weights, then clear the KV cache. Rule-110 task: 10% (no loop) -> ~20% (2 loops) -> >30% (3-4 loops). Synthetic tasks only; compute cost not shown in the text I read.
- SCM sleep-consolidated memory (arXiv 2604.20943): importance-scored concept graph, NREM-like strengthening, pruning/forgetting. Prototype, self-designed benchmarks, no code; REM-style dreaming showed no benefit. Treat as idea source only.
- Generalising E-prop to deep nets (arXiv 2512.24506): theory only; the paper states no experiments show it works across depth. E-prop vs BPTT neural similarity (ICML 2025): comparable only when task accuracy is matched; architecture/initialisation matter more than the rule.
- SLTT (arXiv 2302.14311): drop temporal gradient paths in spiking nets; memory constant in time steps, -64% (CIFAR-10) to -70% (ImageNet) memory, ~same accuracy (vision, not language).
- FPGA SNN accelerators: search found several; the key multiplier-free FPGA paper page was blocked, so no hardware numbers were read. Nothing here supports a SPARK number.

## Fix proposals per weakness
1. Training cost -> truncated credit (SLTT / e-prop idea). Backprop through the spatial/depth path every step, drop the long temporal path through neuron state, keep the gradient through the fast-weight path; optionally K random time steps. Risk: L needs long dependencies, so truncation may hurt recall. Test: memory/time vs full BPTT, 3 seeds, 0.5% gate. Plus fused spike kernel + CUDA graphs (engineering, job 1741).
2. Ternary same-width penalty -> brain uses many cheap synapses, not few precise ones. HYPOTHESIS: widen (d=192-256, already works), add per-channel learned gain (homeostatic scaling), and a very sparse high-precision residual ("few strong synapses"). Test at d=128 and 192 vs same-recipe fp control.
3. Recall -> hippocampal separation. HYPOTHESIS: a fixed random sparse expansion + k-WTA (dentate-gyrus-like barcode) in front of the delta-memory key, so keys are decorrelated before binding; keep content binding Hebbian/delta. Add the "sleep loops": extra recurrent passes over the context before answering (multi-hop). No parameters needed for the barcode. Test on hop2 / chains8 / chains32 / kv, 3 seeds, vs CONV=4 baseline in /lim8k.
4. Local learning -> fast/slow split like the brain: local rules for the fast-weight memory (already delta-rule), backprop for slow weights; local losses only between blocks. Do not expect pure local learning to close the gap: no published result closes it at depth.
5. Competition / k-WTA -> use it where it needs no learning (the barcode expansion above), not across the whole LM; inhibition-style normalisation instead of hard k.
6. Sleep scope -> make it a scheduled phase (replay from stored statistics + extra recurrent passes), test on real text with >=5 tasks and on the transformer. Do not copy REM-style dreaming (no benefit in SCM).
7. Memory/skills -> importance (surprise) scores, pruning, versioned writes; prototype as a small store beside the hippocampus, measured on retention and false-recall.
8. Hardware -> no brain principle fixes this; needs Gowin toolchain, one ternary matvec RTL + resource report, then hash memory/state sizing. Barcode memory is also cheaper to build than hash tables (fixed random expansion), worth checking on the board.
9. Evidence -> downstream tasks, long generation, scale to 50-100M, joules per verified answer; this is measurement work, not a brain idea.

## Order I would run
A) barcode expansion + sleep loops on hop2/chains (fast, cheap, directly attacks weak spot 3). B) truncated-credit training (weak spot 1). C) ternary gain+residual (2). D) sleep-as-phase on real text (6). Others after.
