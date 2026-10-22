---
tags: [protocol, variable-delay, gru, lstm, clinc150]
created: 2026-10-06
---
# Variable-delay memory and real-query routing protocol

Back to [[08 - SpikingBrain Process Log]] · [[16 - Latest Prototype Findings and Demo]]

Registered before model fitting. CPU only, two threads, seeds 42-46; all new data, scripts, checkpoints, figures and caches on D:. Existing Python libraries reused. These experiments do not run pretrained SpikingBrain language models.

## Memory stability
Compare six models with approximately matched trainable parameter counts: memoryless MLP (hidden 120, 722 params), analog leaky integrator (hidden 120, 722), tanh RNN (hidden 24, 746), recurrent LIF (hidden 24, 722), GRU (hidden 13, 730), LSTM (hidden 11, 728). Recurrent layers for conventional baselines use native PyTorch packed sequences. LIF decay .95, threshold 1, detached subtractive reset, sigmoid surrogate gradient. LIF output is mean spikes over final 3 valid timesteps. The matching is parameter count, not mathematical expressivity or computational cost.

Tasks: delayed first-sign recall and positive/negative pulse ordering. Independent balanced train/validation sets, 64 examples at each length 8,16,24,32 (256 total), noise SD .1 in the signal channel and nuisance channel. Variable lengths are masked or packed; no padding updates hidden state. Final query identical across labels. Adam lr .01, 120 full-batch updates, gradient norm clipping 1 for all models. Choose lowest validation cross-entropy checkpoint every 10 updates; no test selection.

Tests: 512 fresh balanced examples per seed/task at 12,32,48,96 steps, signal-noise SD .1 and .2. 48/96 unseen during training. Report all conditions, means, sample SD, per-model parameters and measured CPU inference median over 10 repetitions after 3 warmups. Save all trained state dictionaries. Check padding invariance, state reset, finite gradients/logits and save/load.

## Public natural-language routing
Use CLINC150 (Larson et al., EMNLP 2019), downloaded from official clinc/oos-eval at a pinned commit. This is crowdsourced natural-language intent data, not live customer traffic. CC BY 3.0 license, attribution and checksums retained. Official domain mapping identifies banking, travel and work (3 supported domains). Remaining domains and original out-of-scope records are rejection tests, not additional classes.

Retain official train/validation/test splits. Normalize whitespace and case for duplicate audit; prevent duplicate texts crossing splits, retaining earliest split record and report exclusions. For each seed sample 80 of the 100 official train examples per supported intent. Fit TF-IDF only on that selected training subset, word uni/bigrams, min_df=2, max_features=6000. Logistic regression C=.1,1,10 selected by validation routing accuracy (ties prefer lower C). Calibration uses supported validation and unsupported validation only, maximizing coverage subject to >=95% accepted-query accuracy and <=10% unsupported acceptance. Threshold grid .34-.99 in .01 steps plus 1.1 fallback. Empty feature vectors abstain. Report clean accuracy, accepted accuracy, coverage, unknown rejection and unknown-category breakdown.

Secondary transfer control: hold out lexicographically last 3 intents per supported domain; train on remaining 12/domain, validate only those known intents, evaluate held-out supported-domain intents separately. No held-out examples fit vocabulary or choose C/threshold. This is transfer within predefined domains, not discovering new domains.

Eight manually authored negation/topic-switch probes are diagnostics only; label expected target explicitly. Never use them to tune the model. Save their successes and failures. They are not CLINC examples or a statistically meaningful benchmark.

## Integrity and deliverables
Record split counts, hashes, selected checkpoints/hyperparameters, full per-seed test results, plots, scripts and CPU timings in vault. Reproduce saved inference. Known-domain classification and unknown rejection must be reported together; abstentions are not counted as successful classified answers. No energy, real semantic understanding, production-readiness or language-generation claims.

## Sources
- https://github.com/clinc/oos-eval
- https://aclanthology.org/D19-1131/
- D:\AI\SpikingBrain-study\experiments\datasets\clinc150\manifest.json
- D:\AI\SpikingBrain-study\experiments\datasets\clinc150\LICENSE
