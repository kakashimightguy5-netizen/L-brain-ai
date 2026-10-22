# 140 - Status report: where L is weak, and what is left for a complete architecture (2026-10-21 IST)
Labels: MEASURED = run and read; MODELLED = computed from a model; ESTIMATED = paper number. Nothing here is an AGI/ASI claim; all evidence is on small models and mostly toy tasks.

## 1. What is solid (MEASURED)
- L5 (d=128, 4 layers, ~936k params): attention-free spiking recurrence + delta-rule fast-weight memory + hash recall. At 9M it beats a matched transformer on quality (0.8721 vs 0.9134 bits/byte, note 111/121).
- Inference energy: tie at batch 1/ctx 256; L uses 2-30x less energy at long context and batching (transformer KV cache does not fit at ctx 16384, batch 16+). (note 121)
- Wide ternary: ternary d=192 (3 seeds, mean 0.8287) beats same-recipe fp d=128 (0.8471) by 2.2% at half the weight memory (~7.6 Mbit MODELLED), multiply-free matrices. Ternary d=256 0.7863 beats fp d=192 0.8050. (note 136)
- Sleep consolidation (dreams from stored statistics, no episodes): two tasks A kept fully; three tasks A->B->C works on seeds 0-5 after the A_MIN fix (weak seeds' task A 50.8 -> 92.0 mean, C -2.2 pts avg). (notes 137-139)
- Reversal curse fixed by sleep reverse replay (91% reverse), but it is a procedure win that also works for a transformer.

## 2. Where L is weak
A. Quality/training
1. Training is slower and heavier than a transformer: 2.9x energy/token, ~3x step time, 6.66 vs 1.6 GB VRAM (MEASURED). The step is launch/elementwise bound; CUDA-graph/bf16 job 1741 never read.
2. Ternary at the SAME width costs 5.5% (0.8941 vs 0.8471). Width buys it back but adds ~2.2x add operations; real energy of that trade is not measured.
3. Fixed-size state: recall ceiling. Multi-hop at long chains is low (chains8 ~0.34, chains32 ~0.16-0.22). CONV=8 delay line: no clear gain on hop2 (1 seed, unfinished).
B. Brain-inspired parts not yet working
4. Local / backprop-free learning: every rule pays +14-20% bits/byte (DFA, local losses); the gap widens with steps; predictive coding matches backprop only at ~40x compute. No win; only block-local hybrids remain untested in real L5 layers.
5. k-WTA competition: 94.5% at 20% firing vs 97.4% at ~29% on one delay-copy task; unstable, homeostasis alone worse. Not integrated into the LM.
6. "Think less" gating: ceiling ~8-11% modelled savings; skipping must never skip the state update.
C. Memory / continual learning
7. Sleep is a training-procedure add-on, not a built-in phase; tested only on synthetic key->value tasks, 3 tasks, ~10% chance level. Not tested on real language, many tasks, or the transformer for bigram dreams.
8. Cost of dreaming: C drops ~2 pts; seeds 1,4,5 needed extra rounds on task A. RECENT=4 re-run unfinished.
9. No versioned/durable memory, no forgetting policy, no skill promotion/demotion (all design only).
D. Evidence quality
10. Mostly toy tasks and byte-level LM at <10M params. No scale test, no downstream benchmarks, no long-generation evaluation, no tool use or reasoning verification. Several early "wins" were corrected (ternary vs baseline; 7% -> 2.2%).
11. Laptop comparison runs you owe: 1500f0/f1 transformer same recipe, 1500g0-g2 mixA d=192 ternary.
E. Hardware (SPARK)
12. FPGA fit is ESTIMATED only: old tlr25 model 105 KB vs ~103.5 KB block SRAM, +19.9% quality loss; hash-recall tables and fast-weight state unquantified. No synthesis, no place-and-route, no board power. The 100.5 nJ/byte figure is a 45 nm model, not FPGA.
13. Sparsity cannot be used on GPU training (confirmed in the literature); the efficiency story depends on custom hardware that does not exist yet.

## 3. What is left to build a complete architecture
1. Make wide-ternary mixA the default L core; finish the same-recipe fp/ternary controls and the full limitation suite on it (3 seeds).
2. Fold sleep into the model as a scheduled phase (bigram/statistics dreams + reverse replay); validate on real text with >=5 tasks; try on the transformer for fairness.
3. Stronger recall: more delay taps + content-addressed fast-weight read; MQAR with/without hippocampus; tiny exact local window.
4. Metacognition: uncertainty gate (pooled per word), calibrated copy gate, abstention reward; skill memory with surprise-weighted promotion and demotion.
5. Training cost: fused spike+STE kernel (FLA chunked delta rule), CUDA graphs, bf16; target training energy within ~1.5x of the transformer.
6. Block-local learning inside real layers (hybrid, not pure local), optional k-WTA after stability fixes.
7. Hardware: Gowin toolchain, one ternary matvec + resource report, then hash memory and state in SDRAM; measure real power.
8. Evaluation: downstream tasks, generation quality, long context, joules per verified answer, against a same-recipe transformer.
9. Scale: 9M -> 50-100M to check that the wins and the ternary penalty hold.

## 4. Unfinished runs (from the pause)
CONV=8 hop2 seeds 1-2 and kvupd 0-2 (`c8queue.sh`); RECENT=4 on seeds 4 and 1 (`bqueue.sh`). Nothing running now. rea-agents installed in the cloud workspace only; setup not run.
