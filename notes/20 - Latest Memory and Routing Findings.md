---
tags: [research, results, spiking, routing]
created: 2026-10-06
---
# Latest memory and routing findings

Protocol [[17 - Variable Delay and Real Query Protocol]], complete memory results [[18 - Variable Delay Memory Results]], language results [[19 - Real Query Routing Results]]. Previous snapshot [[16 - Latest Prototype Findings and Demo]].

## Long-delay memory
Five seeds, approximately 722–746 parameters/model, equal 120-update budgets. Train lengths 8–32; evaluate length 96. This table uses signal noise SD .1; full noise-shift results, seed ranges, validation histories and timing are in note 18.

| Model | First-sign recall | Two-pulse order |
|---|---:|---:|
| memoryless | 50.00% | 50.00% |
| analog | 50.39% | 79.06% |
| RNN | 100.00% | 93.40% |
| LIF | 90.43% | 53.40% |
| GRU | 100.00% | 79.57% |
| LSTM | 100.00% | 57.50% |

LIF recall succeeded in four seeds and fell near chance in the fifth. At long delays the LIF order task stays near chance. Ordinary recurrent baselines are useful controls; these results do not establish a spiking advantage. Variable delays, noisier training and checkpoint selection differ from earlier experiments, so this is not an isolated causal test of variable length alone. CPU simulation does not measure neuromorphic energy savings.

## Natural-language routing
CLINC150 is a crowdsourced public English dataset (Larson et al., EMNLP 2019), CC BY 3.0; source https://github.com/clinc/oos-eval and paper https://aclanthology.org/D19-1131/. It is not production customer data. TF-IDF three-domain router across five training subsets:
- Familiar intents: 97.96% raw domain accuracy; answers 84.37% of supported queries, with 99.98% accuracy among accepted queries.
- Unseen intents inside familiar domains: 59.41% raw accuracy; only 16.15% coverage, and 89.01% accepted accuracy.
- Unsupported-query rejection: 90.95% on standard evaluation.
- Manual negation/topic-switch probes: only 2/8 correct per seed. Seed 42 rejects three desired supported queries and incorrectly accepts three unsupported requests because of negated topic mentions.

High accepted accuracy on familiar intents does not imply semantic understanding. This router should remain an experiment, not an automatic decision-maker. A topic prediction is not a generated answer.

## Verified saved artifacts
Memory implementation checks 811/811; routing checks 37/37. Independent fresh-process checkpoint loads reproduced all 480 memory test scores exactly; empty-query rejection also checked. These are implementation/reproduction checks, separate from behavioral failures above.

All new artifacts, temp and cache are on D:. Existing CPU Python libraries were executed without writing bytecode. No pretrained 5B/7B checkpoint was downloaded in this experiment. Scripts, five memory checkpoints, router, dataset attribution/license and hashed manifest are archived under SpikingBrain Assets.

Try a query without training: `py -3.12 -B "D:\AI\SpikingBrain-study\experiments\try_real_router.py" --query "What is the balance of my checking account?"`. This demonstration deliberately preserves the measured limitations.

## Next proposal
Prioritize a language router that handles negation and unfamiliar intents, compared on the same fixed splits and coverage metrics. Test an available local embedding model against this lexical baseline before adding any new large download. For temporal memory, extend the order-task training horizon and compare alternative state/readout choices against RNN/GRU/LSTM; keep budgets and validation-based selection explicit. Continue examining SpikingBrain source separately from claims about these small LIF networks. These are proposals, not completed tests.

Next completed follow-up: [[24 - Embedding and Order Findings]].
