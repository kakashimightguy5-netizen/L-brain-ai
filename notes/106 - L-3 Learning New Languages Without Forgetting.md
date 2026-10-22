---
tags: [L, L-3, continual-learning, skill-columns, results]
created: 2026-10-07
status: internal research record (1 seed per system)
---
# 106 - L-3: learning new programming languages without forgetting

Previous: [[105 - L-2 Candidate and Remaining Flaws Fixed]] · Brain principle origin: [[25 - Round 8 Language Continual Learning (Phase 2)]]

## Setup
- **Starting point:** the trained L-2 candidate (Python, 1.18M params).
- **Languages learned in order:**
  - JavaScript: lodash, 3.7 MB train;
  - C: Redis, 12.6 MB train.
  - Test files are separate from training files.
- **Training:** 1,500 steps per language, CPU.
- **Scoring:** bits/byte on held-out code (lower is better), counted from byte 33 onward. The first 32 bytes are used for routing.

| System | What it does |
|---|---|
| Normal fine-tuning | Updates all weights on each new language |
| Replay | Fine-tuning + 25% of every batch replayed from a small buffer (3% of each old language) |
| **L skill columns** | Base **frozen**. Each new language grows a skill column: a rank-32 add-on to every layer, +280k params (24%). **Routing by surprise:** each column predicts the first 32 bytes and the least surprised one continues. No labels are needed. |

## Results after learning Python → JavaScript → C
| System | Python | JavaScript | C | **Average** | Python forgotten |
|---|---|---|---|---|---|
| Normal fine-tuning | 2.032 | 1.718 | **1.502** | 1.751 | **+0.72** |
| Replay | 1.613 | 1.136 | 1.545 | 1.431 | +0.31 |
| **L skill columns** | **1.337** | **0.927** | 1.674 | **1.313** | **+0.03** |

- **Routing by surprise** picks the right column 95% (Python), 99% (JavaScript) and 92% (C) of the time. The small Python loss (+0.03) comes only from the 5% of misrouted passages; Python's own weights never change.
- **New-language detection before learning** (the trigger for growing a column):

  | Incoming language | AUROC | Flagged as new (5% false alarm) |
  |---|---|---|
  | JavaScript | 0.88 | 36% |
  | C | 0.91 | 67% |

## Honest reading
- **Skill columns win clearly overall** and essentially solve forgetting.
- **On the newest language alone, full fine-tuning is better.** For C: 1.50 vs 1.67. A rank-32 column on a Python-trained base has less freedom than retraining everything. Round 8 showed that larger columns close this gap.
- **Detection of JavaScript as new is moderate:** 36% flagged at a 5% false-alarm rate. Code languages share many byte patterns with Python.
- One seed per system.

## Also finished
- Layer pairs trained separately, seed 2: 1.392 → mean of 2 seeds **1.394** (end-to-end 1.353, transformer 1.466).

## Next
- Larger and adaptive columns; replay inside columns (sleep).
- Repeat with 3 seeds on the laptop GPU.
- Grow a column automatically from a mixed stream.

## Code
`lab-code-L/l3_continual.py` · `results_L3_*.json`
