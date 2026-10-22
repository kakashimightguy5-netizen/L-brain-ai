# 130 - Brain principles for L: research and experiment plan
Date: 2026-10-11 (IST). Prompted by: "we are creating a human brain... dont inherit any transformer properties... dont rely on only maths and matrices".

## Honest framing
- Nobody can build a human brain today. Neuroscience does not know the full learning algorithm, and any computer, including a neuromorphic one, still runs mathematics. What we can do is copy **mechanisms** the brain uses (event-driven spikes, local learning rules, two memory systems, sleep replay, structured innate wiring) and test each against a matched transformer.
- Every mechanism below becomes a flag, default off, tested with 3 seeds. A mechanism is kept only if it wins on a measured task. Claims about "brain-level" intelligence are not supported by anything here.

## What the brain does that transformers do not (sources checked 2026-10-11)
1. **Two learning systems.** Kumaran, Hassabis, McClelland (2016): neocortex learns slowly and generalises; hippocampus learns single episodes quickly (pattern separation in dentate gyrus, pattern completion in CA3). One system alone suffers catastrophic interference. Replay during rest and sleep trains the cortex in an interleaved way, and replay can be biased toward surprising or rewarding events. Update: new facts consistent with existing knowledge (a schema) can enter cortex fast. [Source: https://web.stanford.edu/class/psych205/papers/KumaranHassabisMcC16CLSUpdate.pdf]
2. **Sleep replay protects old memories.** Thalamocortical spiking models show interleaved replay of new and old traces prevents forgetting (eLife 51005; bioRxiv 2025.06.25.661579). I could not read the eLife page text, so I only cite its title and claim.
3. **Innate structure, not blank slates.** Zador (2019): the genome holds about 1 GB (3x10^9 nucleotides) but the brain has about 10^14 synapses; specifying them all would need about 3.7x10^15 bits, at least six orders of magnitude more than the genome holds. So evolution encodes compact wiring rules, and animals learn fast partly because most structure is innate. Lesson: spend effort on a good built-in scaffold and a meta-learned learning rule, not only on training data. [https://pmc.ncbi.nlm.nih.gov/articles/PMC6704116]
4. **One neuron is a small network.** Beniaguev, London, Segev: a deep network needed 5 to 8 layers and about 1,000 artificial units to mimic one rat cortical pyramidal neuron at 99% accuracy (dendrites are the cause). Caveats: model-to-model comparison; unclear if it helps AI. [Quanta write-up: https://www.quantamagazine.org/how-computationally-complex-is-a-single-neuron-20210902/]
5. **Local learning without backprop.** Predictive coding: each layer predicts the layer below, errors are local, weights change by local rules; it can approximate backprop under conditions and fits associative memory best. Limits: rarely beats backprop, does not yet scale to ImageNet-size images, inference needs several settling iterations. [https://ar5iv.labs.arxiv.org/html/2202.09467]
6. **Sensorimotor models with reference frames.** Thousand Brains Project (Numenta): many small cortical-column-like modules each model whole objects as features at locations, and vote. Claims continual learning and few-sample learning, but the paper shows no quantitative results and is early (77-object benchmark in simulation). [https://arxiv.org/html/2412.18354v1]
7. **Three-factor (neuro-Hebbian) rules and eligibility traces.** (From my background knowledge, not from a page I could read: the sources I searched are PMC6079224 and arXiv 2312.09257.) Synapse change needs pre-activity, post-activity and a slower neuromodulator signal such as dopamine; an eligibility trace lasting seconds bridges the delay to reward. Candidate for credit assignment with no backward pass.
8. **Energy.** One review quotes about 10 W for the brain (Merkle) and 1,287 MWh to train GPT-3. The same page's "3,000 times fewer words for a child" figure is not derived and I do not rely on it. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11099251]

## What L already has vs what is missing
| Brain mechanism | L now | Gap |
|---|---|---|
| Fast one-shot memory (hippocampus) | delta-rule fast weights + hash recall | no consolidation into slow weights |
| Slow cortex | trained diagonal recurrence | trained by backprop only |
| Event-driven spikes | integer spikes | zero-skip benefit not measured on a trained model |
| Sleep replay | none | add offline replay phase |
| Pattern separation | hash recall | not tested for interference |
| Dendritic nonlinearity | plain FFN | untested |
| Local learning | none | predictive coding / three-factor untested |
| Innate scaffold, meta-learned rule | none | untested |

## Experiment plan (all flags default off, 3 seeds, matched transformer baseline, accuracy plus forgetting)
- **B1 Consolidation / sleep replay (first, cheapest):** train on task A, learn task B episodically in fast weights, then run an offline replay phase that distils fast-weight content into slow weights; measure A-accuracy retained and B-accuracy, versus no-replay and versus a transformer fine-tuned sequentially. Metric: forgetting = A_before - A_after. Gate: L forgets less than the transformer at equal params.
- **B2 Surprise-biased replay:** replay chosen by prediction error versus uniform; measures if the CLS "goal-weighted replay" claim helps here.
- **B3 Dendritic units:** replace each FFN unit by a 2-branch nonlinear unit at matched parameter count; test on the limitation tasks and on bits/byte with the 0.5% gate.
- **B4 Local learning:** predictive-coding or three-factor update for the fast-weight path only (not the whole net), compare to the delta rule.
- **B5 Meta-learned initialisation (innate scaffold):** meta-train the starting weights/learning rate across many small tasks so new tasks need few examples; measure examples-to-threshold.
- Existing limitation suite stays the yardstick (kv, kvupd, hop2, reverse) plus one new continual-learning task for B1/B2.

## Current evidence on the limitation suite (8000 steps, MEASURED, 2 of 3 seeds so far)
- Key-to-value recall: L5 reaches 98 to 100% at 4 pairs and 95 to 97% at 8 pairs; the matched transformer stays near 30 to 42% at 4 pairs. Both degrade beyond 16 pairs (L5 about 25% at 128). The transformer baseline may be under-tuned; a learning-rate check is still owed.

## B1 first result (MEASURED, seed 0 only, CPU, 2500 steps A then 2500 steps B; whole-answer accuracy, chance ~10%)
| Model | A after A | none: A after B | none: B | replay(25% buffer): A after B | replay: B |
|---|---|---|---|---|---|
| Transformer (990,208) | 41% | 0% | 36% | 30% | 31% |
| L5 (936,357) | 74% | 1% | 97% | 48% | 95% |
- Both models forget task A completely when trained only on B. L5 is NOT more resistant to forgetting; its fast-weight memory is not consolidated into the slow weights, so the brain-style two-system benefit is not yet present.
- A small replay buffer (hippocampal-replay analogue) restores part of A in both (L5 retains 65% of its A score, transformer 74%), and L5 learns both tasks far better than the transformer. This is one seed and the transformer learns A only weakly, so no winner is declared. Seeds 1 and 2 are not yet run.
- Takeaway: replay helps, but the real brain mechanism (fast store teaching the slow store during an offline phase, without raw stored data) is what L still lacks. Next: generative/self-replay from the fast-weight store (B1b).
