---
tags: [experiments, round-8, internal, language-model, continual-learning, phase-2]
created: 2026-10-07
status: internal research record
---
# Round 8: Brain-style language model that learns new domains without forgetting (Phase 2 prototype)

Back to [[20 - Final Benchmark Summary]] · [[21 - Honest Verdict on the Architecture]] · Roadmap: [[00 - Concept and Roadmap]] (Phase 2: "learns domain B without forgetting A; only relevant modules fire")

**Question:** do the principles proven on images (self-knowing columns, developmental order, novelty detection, growth) carry over to **language**?

## Setup (all trained from scratch on CPU, byte level, context 128)
- **Domains, learned one after another:** A = Shakespeare → B = Wikipedia (WikiText-2) → C = Python source code (CPython standard library, 1.8 MB).
- **Same trunk for every system:** 4 layers, d = 128, 0.88M params. Each domain gets 1,500 steps of batch 32.
- **Metric:** bits per byte on held-out text, after all three domains (lower is better). All systems are scored on positions 32–127, so the routing of our system gets no unfair peek.

| System | What it does |
|---|---|
| Standard (seq) | Fine-tunes A → B → C (what normally happens) |
| Replay | seq + 25% of every batch replayed from a small buffer (3% of each old domain) |
| Joint | Trained on all three domains mixed at once. **Not continual**: it needs all the data together. It serves as the upper reference. |
| **Ours** | **Childhood trunk** learned on A, then frozen (developmental order). **Each new domain grows a low-rank column**. **Routing by predictive surprise:** every column predicts the first 32 bytes and the least surprised one continues (predictive coding). **Novelty:** if even the best column is surprised, the text belongs to a new domain and a column is grown. No labels are needed. |

## Results (mean ± spread over 2 seeds, joint 1 seed)
| System | Shakespeare (A) | Wikipedia (B) | Code (C) | **Average** | Forgetting of A | Active params |
|---|---|---|---|---|---|---|
| Standard (seq) | 3.91 | 3.37 | 1.89 | 3.06 ± 0.02 | **+1.59** | 0.88M |
| Replay | 3.24 | 2.41 | 1.99 | 2.55 ± 0.04 | +0.92 | 0.88M |
| Replay, wider (size-matched to ours, small) | 3.34 | 2.39 | 1.97 | 2.57 | +1.04 | 1.10M |
| Replay, wider (size-matched to ours, large) | 3.79 | 2.42 | 1.93 | 2.71 | +1.54 | 1.47M |
| Joint (needs all data at once) | 2.25 | 1.98 | 1.96 | 2.06 | n/a | 0.88M |
| **Ours, small columns (rank 16, +137k params each)** | 2.33 | 2.07 | 2.16 | **2.18 ± 0.00** | **+0.004** | 1.01M |
| **Ours, large columns (rank 64, +549k each)** | 2.34 | **1.91** | 1.94 | **2.06 ± 0.01** | **+0.015** | 1.42M |

### Self-knowledge on language
- **Routing by surprise picks the right column 99.0%** of the time, from 32 bytes of text, with no domain labels.
- **Novelty before learning** (is incoming text a new domain?):

  | Incoming domain | AUROC | Flagged as new (5% false alarm) |
  |---|---|---|
  | Wikipedia (vs Shakespeare) | 0.92 | 70% |
  | Python code (vs Shakespeare + Wikipedia) | 0.97 | 93% |

  This is the trigger of the growth rule: the system knows when it needs a new column.

## What this shows
1. **Forgetting essentially solved:** Shakespeare stays within 0.004–0.015 bits of its original quality, where standard training loses 1.59 bits.
2. **Ours beats replay by a large margin** (2.06–2.18 vs 2.55) at every model size. Making the replay model bigger does not help: the large one forgets more.
3. **Large columns match joint training (2.06 vs 2.06)** without ever revisiting old data. On Wikipedia it is even better (1.91 vs 1.98).
4. **Only the relevant modules fire:** one trunk plus one column per token. Columns are added only when novelty says so.

## Honest limits
- **Capacity trade-off:** with small columns, the new domains trail full fine-tuning (code 2.16 vs 1.89). The frozen trunk learned only Shakespeare, so the columns must do more work. Large columns close the gap but cost more parameters.
- **Routing cost:** our routing runs every column over the first 32 bytes. That is about 25% extra compute per extra column for each new text, paid once per passage, not per token.
- **Scale:** this is still under 2M params on CPU, and Wikipedia novelty (0.92) is weaker than code novelty (0.97) because both are English prose.
- **Next:**
  - Add early exit and looping inside each column (both already proven).
  - Grow a column automatically from a mixed stream.
  - A broader childhood (more varied first data).
  - Scale to 10–50M params on the RTX 3050 / Kaggle.

## Code
`lab-code/`: big10_lm_continual.py (`python big10_lm_continual.py {seq|replay|joint|ours} SEED 1500`; env RANK = column rank, DM = width) · big10_summary.py · results_big10_*.json
