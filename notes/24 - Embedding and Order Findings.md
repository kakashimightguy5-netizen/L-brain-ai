# Embedding and longer-order findings

[[21 - Embedding and Longer Order Protocol]] · [[22 - Local Embedding Routing Results]] · [[23 - Longer Order Memory Results]] · [[08 - SpikingBrain Process Log]]

## What succeeded and what remains uncertain
Small local models run and their recorded results are reproducible. This is an engineering success. Better performance or human-like reasoning must be assessed separately. These are custom tiny networks plus a frozen existing embedding encoder, not executions of pretrained SpikingBrain5B/7B.

## Routing comparison
CLINC150 public crowdsourced queries; same official filtered splits and five training subsets. Frozen Nomic137M classification embeddings versus word/bigram TF-IDF features, then supervised three-domain logistic classifier. Vocabulary/feature fitting and threshold selection follow their own documented protocols. No test-based hyperparameter selection. External pretrained-corpus overlap cannot be ruled out.

| Evaluation | Metric | Lexical baseline | Nomic embeddings |
|---|---|---:|---:|
| standard | accuracy_without_rejection | 97.96% | 98.76% |
| standard | coverage | 84.37% | 39.27% |
| standard | accepted_accuracy | 99.98% | 100.00% |
| standard | unsupported_rejection | 90.95% | 93.58% |
| heldout_intents | accuracy_without_rejection | 59.41% | 62.59% |
| heldout_intents | coverage | 16.15% | 9.19% |
| heldout_intents | accepted_accuracy | 89.01% | 100.00% |
| heldout_intents | unsupported_rejection | 90.54% | 91.67% |

Standard seed43 selected the pre-registered reject-all fallback: zero supported-query coverage. Its accepted accuracy is undefined and excluded from the displayed100% aggregate, which therefore averages only four nonempty runs. This threshold-grid failure must not be presented as successful routing. The five-seed coverage mean includes the zero-coverage run. Other runs accept about48–50% of familiar queries; transfer coverage is only9.19%. High accepted accuracy largely reflects heavy abstention.

Manual negation diagnostic correct counts per seed: [3, 3, 4, 2, 2]/8; lexical baseline2/8 eachseed. Probes are only eight authored examples, not a robust benchmark. Raw accuracy, acceptance accuracy and coverage must be read together. Rejected supported queries are unanswered; accepted accuracy alone can hide poor coverage. Unsupported rejection is calibrated on validation, not guaranteed on test.

## Event-order experiment
Five seeds,120updates/model, train16/32/64, signal noise .1; table uses test noise .1. Complete .2noise results and SD in note23.

| Variant | Length | Mean | Seed range |
|---|---:|---:|---|
| RNN | 32 | 99.49% | 98.44–100.00% |
| RNN | 96 | 99.14% | 96.68–100.00% |
| RNN | 192 | 98.59% | 95.12–100.00% |
| GRU | 32 | 99.22% | 97.85–100.00% |
| GRU | 96 | 95.43% | 93.16–97.07% |
| GRU | 192 | 81.91% | 65.82–91.80% |
| LIF_spikes | 32 | 61.37% | 53.91–69.92% |
| LIF_spikes | 96 | 54.57% | 51.56–62.50% |
| LIF_spikes | 192 | 53.24% | 49.80–56.64% |
| LIF_membrane | 32 | 89.10% | 81.84–93.36% |
| LIF_membrane | 96 | 74.77% | 67.38–80.08% |
| LIF_membrane | 192 | 63.75% | 57.62–69.73% |

LIF_spikes reads last3spikes; LIF_membrane reads remaining final membrane state. The latter is a hybrid analog readout. Both start from identical weights; RNN/GRU are approximately parameter-matched controls. Longer train lengths change comparison with prior experiments. Performance is on synthetic binary order tasks, not language comprehension or episodic memory. No energy advantage measured.

## Reproducibility and files
Embedding controls 27/27; memory controls 205/205. All120 memory scores reproduced from disk; all8 saved routing probe probabilities/decisions independently reproduced. These checks do not make behavioral errors successes. Archive copies SHA256-verified, including full frozen-feature cache, scripts and five memory checkpoints; manifest [[SpikingBrain Assets/embedding-order-manifest.json]]. All new experiment data/temp/cache/scripts on D:. Existing services may keep normal system logs. No model downloads/installations.

Runnable local router: `py -3.12 -B "D:\AI\SpikingBrain-study\experiments\try_embedding_router.py" --query "What is the balance of my checking account?"`. Requires existing local Ollama service and model. It predicts a topic or abstains; it does not generate answers.

## Next proposal
Use the measured comparison to choose the baseline rather than assuming spiking or embedding superiority. For routing, collect a separate larger negation/unfamiliar-request test set before changing rules; keep coverage and unknown rejection visible. For memory, separate state retention from final readout and compare broader seed ranges before adding domain-column complexity. Full pretrained SpikingBrain inference remains a separate compatibility/memory problem.

Sources: https://github.com/clinc/oos-eval ; https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 ; https://docs.ollama.com/api/embed .

Latest completed follow-up: [[28 - Memory Trace and Language Challenge Findings]].
