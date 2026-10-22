---
tags: [cpu-coder, review, handoff, claude]
created: 2026-10-07
status: review only (no training, no services started)
---
# 96 - Claude Takeover Review of the CPU Coder (from Codex)

Previous: [[95 - Latest CPU Skill Training Status]] · [[94 - V5 Targeted Follow-up Results]] · [[92 - Targeted V4 Results and OpenCode Root Causes]]

**Scope of this step:**
- Read the handoff (CLAUDE_HANDOFF.md, CLAUDE_REVIEW.md, manifests), notes 58–95, and the raw V5 evidence: test-results.json, legacy-heldout-results.json, real-opencode-results.json, retention-results.json, and the copy train/test data.
- Nothing was trained, started, changed or deleted. The baseline and all checkpoints are untouched.
- The project is taken as the base as-is (Qwen2.5-Coder-0.5B + LoRA skill "columns" + earlier growth, spiking and frontend modules). Work continues on our path: modular skills, self-knowing routing, protected retention, root cause before training.

## Current state
| Part | Status |
|---|---|
| Base | Qwen2.5-Coder-0.5B-Instruct, 494,032,768 params, frozen (hash verified) |
| Skills | Rank-4 LoRA on q/v (270,336 coefficients each): V4 selector (selected), V5 copier (selected), V5 selector (research only) |
| Runtime | 3 separate merged Q8 GGUFs (531 MB each) + a hand-written Python gateway (port 8797) that routes on keywords ("exact contents" / "verbatim") and tool-name sets |
| Default | Baseline only; the hybrid is opt-in. All services are paused. |

## Successful results (confirmed in the raw files)
- Copying, fresh mixed: 7/16 → 11/16.
- Copying, varied: 9/12 → 11/12. Patterned copying: 16/16 retained.
- Independent JS: 3/4 → 4/4. Python: 4/4. Retention: 6/6.
- Scoped read/glob/grep: 12/12. Code vs tool: 8/8.
- Hybrid V4 selector + V5 copier: 3/3 real OpenCode reads (tiny sample).
- About 66–68 tokens/s on CPU for short outputs.

## New findings from the raw evidence
1. **4 of the 5 remaining fresh-copy failures share one cause: leading whitespace on the FIRST line is lost.**
   - Every test whose first line starts with a tab or spaces failed (4/4); every other test but one passed.
   - The model got all content right. It only dropped the indentation of line 1, e.g. `\tTab_480…` came back as `Tab_480…`.
   - Training had 15/56 such rows, so the model has seen this.
   - Strong suspicion: the response is trimmed after generation (llama.cpp chat/tool-call parser or the harness), not a model error. This must be verified by raw completion before any training.
   - If it is confirmed, the true score is likely 15/16 without retraining.
   - The 5th failure (#13) is a real model error: an HTML-style `<…>` wrapper was hallucinated around the last line.
   - The prior-diverse failure also lost its first-line indentation.
2. **The "fresh" copy test is not unseen content.** All 46/46 test lines use the same 8 line templates as training (only the numbers differ). It measures template copying, not general copying.
3. **The broad tool regression (4/11 vs baseline 8/11) has specific causes:**
   - **Name-slot confusion:** an argument value is used as the tool name (`"john_doe"`, `"Inception"`). In the V5 training data, the tool name was always written in the user text ("Use lookup_record to…"), so the adapter learned to copy names from the prompt (a shortcut).
   - **Asks instead of calling:** 3 cases ask a clarifying question when a call was expected.
   - **Nested object schema:** the schema itself was copied into the arguments.
   - **One garbage output** ("metre").
4. **The "result" metric is too weak.**
   - 6/10 fail by re-emitting the tool call after the result came back.
   - The 4 "passes" contain junk ("_PS", "idelity", "</tool_response>"). They pass only because the check is "non-empty and no new call".
5. **Runtime cost:** 3 merged GGUFs hold 3 copies of the base in RAM. llama.cpp can serve one base with hot-swappable LoRA adapters (about 1 MB each), which fits the lightweight goal.
6. **The router is hand-written keyword rules, not learned.** This is where our proven "self-knowing / surprise routing" (Round 8: 99% routing, novelty AUROC 0.92–0.97) can replace rules.

## Proposed next steps (waiting for approval)
- **Step 0: diagnostics, no training**
  - Re-run the 5 failed copy cases via raw completion and in fp32 transformers to confirm the trimming hypothesis, and fix it in the pipeline if confirmed.
  - Replace the weak "result" check with a grounding and clean-stop check.
  - Freeze a genuinely unseen evaluation set: new copy line families; Glaive held-out conversations with new function names, competing tools, number/boolean/enum/array/nested arguments, and missing-argument cases.
  - Re-score all existing checkpoints on it.
- **Step 1: V6 broad-schema selector (training)**
  - Root-cause curriculum:
    - Tool names never appear in the user text.
    - 2–5 competing tools per request.
    - Typed and nested arguments.
    - "Ask for missing required arguments" cases.
    - Post-tool answers that stop cleanly.
    - Replay of V4 scoped data.
  - Glaive train split only (Apache-2.0), disjoint from the test set.
  - Accept only if it reaches at least the baseline on broad tools, keeps scoped 12/12, and passes retention.
- **Step 2: one base + LoRA hot-swap and a learned router**
  - Serve one base with LoRA hot-swap (one base in RAM).
  - Replace the keyword gateway with learned self-knowing routing (adapter surprise/confidence with baseline fallback, from Round 8).
- **Step 3:**
  - Expand the retention suite.
  - Re-run the real OpenCode tests.
  - Record everything here.
- **Option:** use the RTX 3050 for training if allowed. So far training has been CPU-only by choice.
