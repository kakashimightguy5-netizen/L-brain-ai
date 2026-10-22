---
tags: [research-scope, L, problem-atlas, llm-limitations]
created: 2026-10-07
status: scope recorded; research waiting for approval
---
# 99 - Widened Research Scope: the LLM Problem Atlas

Previous: [[98 - Architecture L Charter]] · Related idea: [[Structured Think-Before-Build Flow]]

**User decision (2026-10-07):** widen the research. Find all the problems of current frontier and open-source LLMs, study why they happen, and propose L as the model that fixes them.

## Research method: four questions for every capability
| # | Question | Example |
|---|---|---|
| 1 | **Can't do it:** what do current LLMs fail at, and *why*? | Exact long copying, reliable counting, learning after deployment |
| 2 | **Does it, but wrongly:** right-looking output with the wrong method or hidden errors | Code that passes demos but breaks edge cases; confident wrong answers |
| 3 | **Does it, but inefficiently:** correct result at too high a cost | Attention cost on long text, all parameters firing for easy tokens, huge GPUs |
| 4 | **Does it without understanding:** generic, repetitive protocol for every task; never grasps the user's need or the product it builds | Every landing page looks the same; no discovery of who the users are or why the product exists |

## Who is studied
- **Frontier models:** closed, best available.
- **Open-source models:** Qwen, Llama, DeepSeek, Mistral, Gemma and others.

## The Problem Atlas (output)
For each problem:
1. The evidence (papers, benchmarks, our own tests).
2. The root cause in the architecture or training.
3. Whether scale fixes it.
4. **How L fixes it**, and the test that proves it.

## Planned problem areas (first draft)
- Learning and memory:
  - forgetting;
  - no learning after deployment;
  - exact recall and copying.
- Self-knowledge:
  - hallucination;
  - over-confidence;
  - not knowing when to ask.
- Efficiency:
  - quadratic attention;
  - KV-cache memory;
  - dense compute;
  - same effort for every token;
  - energy and data-centre dependence.
- Reasoning:
  - counting and state tracking;
  - long multi-step tasks;
  - planning.
- Agents and tools:
  - schema handling;
  - loops;
  - repeated calls (seen in our own CPU coder tests, notes 92–96).
- Understanding the need:
  - generic templates;
  - no discovery;
  - no product understanding;
  - no domain-expert thinking.
- Data and training:
  - needs huge data;
  - cannot update a single fact cheaply.

## Plan (not started)
1. **Literature and benchmark sweep:** what is known and proven.
2. **Our own probes:** small, reproducible tests on open models run locally.
3. **Write the atlas.**
4. **Map each problem to an L component and a test** (feeds L-0 to L-3 in [[98 - Architecture L Charter]]).
