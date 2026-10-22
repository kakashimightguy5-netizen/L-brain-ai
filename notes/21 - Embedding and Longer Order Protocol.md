---
tags: [protocol, embedding, spiking]
created: 2026-10-06
---
# Embedding routing and longer-order memory protocol
Registered before fitting. Follow-up to [[20 - Latest Memory and Routing Findings]]. All newly created experiment files, temp and cache on D:. Existing Ollama service/model and existing CPU Python libraries reused; no model downloads or package installs. Existing service/OS may maintain their own normal logs.

## Frozen local embeddings
Use installed nomic-embed-text:latest via localhost Ollama, record digest/model metadata. Use classification: prefix, full 768-dimensional normalized embedding, truncate false. Embed each text independently; cache frozen features on D:. Encoder is never fine-tuned. Official CLINC duplicate-filtered splits, domains banking/travel/work, training subsets seeds 42–46, held-out intents, logistic C grid .1/1/10, validation-only rejection threshold selection and all eight probes exactly as note 17. Compare with recorded lexical baseline. Fixed encoder preprocessing includes test texts but learns no statistics or vocabulary from them. External pretraining contamination cannot be ruled out. Report raw accuracy, coverage, accepted accuracy and unknown rejection; negation probes are diagnostics, never tuning data. Save CPU classifier and test serialized inference. Encoder runtime/GPU allocation observed, no energy claims.

## Longer-horizon event-order learning
Five seeds 42–46; pulse-order task only. Compare RNN hidden24 (746 params), GRU hidden13 (730), recurrent LIF hidden24 (722) with final-three-step spike readout, and same LIF with final membrane readout (722). Membrane readout is a hybrid analog output, not a fully spike-only architecture. Beta .95, threshold1, detached reset, sigmoid surrogate as prior. Identical balanced training/validation 64 examples per length 16/32/64, independent split seeds; noise SD .1. Same initialization seed for both LIF variants; RNN/GRU fixed per-kind offsets. Adam lr .01,120 full-batch updates, gradient clip1; choose minimum validation CE every10 updates. No test selection.

Test512 independent balanced examples perseed at lengths32/96/192, signal noise .1/.2. Length32 trained,96/192 exceed longest training length64. Record all scores, mean/sample SD, finite gradients/logits, padding invariance, reset, exact reload. Longer training horizon is shared by all four models; new scores are not directly comparable to prior train8–32 as a pure readout effect. LIF variants within this experiment isolate readout choice more closely. Save all five checkpoints. No pretrained SpikingBrain inference in these tiny prototypes.
