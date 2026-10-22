# 143 - Master Research Directive (pasted 2026-10-22, cut off in Part 17): mapping to what Project L already has
Rules kept: no restart, nothing overwritten, baselines preserved, MEASURED / MODELLED / ESTIMATED labels, no AGI claim, null results reported. Prior-art statements below come from my training knowledge and have NOT been re-verified this session (marked UNVERIFIED); Part 13 literature check is still to be done per mechanism.

## 1. Status per part (honest)
| Part | Topic | In L today | Gap |
|---|---|---|---|
| 1-2 | Foundations / brain as principles | Notes 114, 129, 130, 141; matrix-free survey (131); brain principles judged by measurement | No written answer per principle to the 5-point test (established / uncertain / computational form / advantage / test). |
| 4 | Predictive state | L5 fixed-size state (diag recurrence + delta fast weights); long-context energy 2-30x lower (MEASURED, note 121) | No predictive-coding state, no information-loss detector, no selective-update test. |
| 5 | Memory types | Working (state), semantic-ish (hippocampus hash recall), parametric (weights) | No episodic/procedural memory, no forgetting policy, no provenance, no memory-vs-no-memory ablation on unseen reasoning. |
| 6 | Adaptive compute | Skim/gate prototypes; oracle ceiling 19%, realisable 8-11% (MODELLED, note 120) | Controller cost not measured; matched-accuracy comparison missing. |
| 7 | Alternatives to backprop | DFA, local losses, predictive coding tested: +14-20% bits/byte, PC ~40x compute (MEASURED, note 132) | Forward-forward, equilibrium prop, hybrid block-local untested. |
| 8 | Continual learning | Sleep consolidation, 3-task A->B->C, 6 seeds (notes 137-139) | Forward/backward transfer, sample efficiency, memory growth not reported; synthetic tasks only. Statistics-store growth must be reported (256x256 counts per task = grows with tasks). |
| 9 | World models | none | untouched |
| 10 | Unified mechanism | see section 2 | candidate maths written below, nothing validated |
| 11 | Information theory | none beyond bits/byte | untouched |
| 12 | Five candidates | partial (A, B, D components exist) | no formal comparison |
| 14 | Resource-aware objective | efficiency measured post hoc | no cost term in loss |
| 15 | Benchmarks | lim_suite (kv, kvupd, hop2, reverse), cont3, bits/byte windows | no compositional, math, OOD, few-shot suites; mostly toy |
| 16 | Failure modes to avoid | we corrected several overclaims (ternary vs baseline 7%->2.2%) | evidence still toy-scale |

## 2. Three explicit candidate mechanisms for Part 10
Notation: state S_t in R^{dv x dk} per head, key k_t, value v_t, query q_t (unit norm), decay a_t in (0,1), write gate b_t in (0,1).

### M1 - Delta-rule fast-weight state as prediction, memory, and learning (what L5 already implements)
- Prediction: v^_t = a_t S_{t-1} k_t (what the memory expects for key k_t). Error: u_t = v_t - v^_t.
- Update: S_t = a_t S_{t-1} + b_t u_t k_t^T. This is one online gradient step on 1/2 ||v_t - S k_t||^2 (with decay), so prediction, storage and learning are the same operation.
- Retrieval: o_t = S_t q_t. Reasoning across steps = repeated read/write over S.
- Surprise signal: ||u_t|| / b_t (already recorded in code, RECORD flag).
- Complexity: O(dv dk) per token per head, constant in context. Memory capacity ~ dk orthogonal keys per head (MEASURED ceiling: kv N128 acc 0.26).
- Learning objective: next-byte cross-entropy through the chunked solve; fast weights learn within sequence, slow weights across.
- Prior art (UNVERIFIED): fast weight programmers / DeltaNet (Schlag et al. 2021), Gated DeltaNet, RWKV-7 style; test-time-training layers treat the state as a model trained by gradient. Overlap is large; the difference in L is only the integration with spikes, hippocampal hash recall and sleep.
- Falsifiable: if the delta state is only a linear-attention variant, a matched Gated DeltaNet baseline should equal L5 on kv/hop2/kvupd within noise. NOT yet run; should be.

### M2 - Surprise-gated consolidation from fast state to slow weights (sleep)
- Slow weights W; frozen copy W_old; fast state from M1.
- Wake: M1 only. Surprise s_t = ||u_t||. Tokens with s_t above threshold tau are marked for consolidation (statistics-only version: per-task bigram count matrix N_c in N^{256x256}).
- Sleep: sample dream sequences x~ from N_c (no episodes), minimise L_sleep = KL(p_{W_old}(.|x~) || p_W(.|x~)) + L_new, with recency weight rho and strength lambda (MEASURED settings: RECENT=2.5, LAM=25).
- Memory growth: O(256^2) per task = 65k integers; unbounded over tasks unless counts are merged (not yet tested).
- Prior art (UNVERIFIED): complementary learning systems, generative/pseudo-rehearsal (Shin et al. 2017), learning without forgetting (distillation). Difference: dreams come from n-gram statistics of old tasks, not a generator or stored data.
- Falsifiable: on a 6+ task sequence, retention must stay within gate while memory growth is reported; plain distillation without dreams must fail. Transformer control exists for 2 tasks only.

### M3 - Uncertainty-allocated compute using the same error signal
- Controller reads surprise s_t and an output-entropy H_t. Budget loop: if s_t < tau_1 and H_t low, emit; else run n extra recurrent passes over S (re-read with S_f as init) up to n_max; if still high, retrieve (hippocampus) or abstain.
- Objective: L = L_task + lambda_C * E[passes] + lambda_R * ECE-penalty; or constrained form min E[passes] s.t. acc >= acc_target.
- Complexity: expected passes x O(dv dk). Controller cost must be measured.
- Prior art (UNVERIFIED): adaptive computation time, mixture-of-depths, early exit, "sleep loops" (arXiv 2605.26099). 
- Falsifiable: at matched accuracy, mean operations must fall vs fixed depth; and surprise must have AUROC > 0.5 for errors. Existing measurement: oracle gate ceiling only 19% (note 120), so this candidate is bounded.
- Status: sleep-loop part tested (SLEEP=2) - no gain on hop2 (note 142).

## 3. Candidates A-E ranking (EVIDENCE ONLY, provisional)
| Candidate | Evidence we hold | Provisional rank |
|---|---|---|
| A Predictive recurrent | L5 beats matched transformer at 9M; kv/hop2 above transformer (MEASURED) | 1 (already our core) |
| B Memory-centric | hippocampus + delta; no memory-ablation on reasoning | 2 |
| D Continual cognitive | sleep works on toy tasks; growth unmeasured | 3 |
| C Adaptive sparse | bounded ceiling 8-19% | 4 |
| E Unified | M1 does prediction+memory+learning in one rule; M2/M3 not integrated | untested; do not rank above A |

## 4. Proposed order of work (small, controlled, falsifiable)
1. Matched Gated-DeltaNet baseline on lim_suite (tests whether L5's memory is genuinely different). 
2. Memory ablations on unseen reasoning (Part 5): L5 with/without hippocampus and delta memory.
3. Continual suite extended: 6 tasks, transfer metrics, statistics-store growth (Part 8).
4. Resource-aware objective: add measured op-count term, constrained version, check for reward hacking (Part 14).
5. Literature pass per mechanism with real citations (Part 13).
6. World-model and information-theoretic parts last; they have no existing code.
Running now: hop2 wide-training and think-out-loud tests (note 142); results pending.
