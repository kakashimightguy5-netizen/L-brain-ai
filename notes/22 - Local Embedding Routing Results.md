# Local embedding routing results

[[21 - Embedding and Longer Order Protocol]] · [[19 - Real Query Routing Results]] · [[08 - SpikingBrain Process Log]]

Existing local nomic-embed-text:latest, digest 0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f, {'parent_model': '', 'format': 'gguf', 'family': 'nomic-bert', 'families': ['nomic-bert'], 'parameter_size': '137M', 'quantization_level': 'F16', 'context_length': 2048, 'embedding_length': 768}. Frozen classification-prefixed 768-dimensional normalized features, 13103 independent text encodings; no encoder fine-tuning, no fitted corpus preprocessing. Identical filtered official CLINC splits and five sampled training subsets; same validation-only C and rejection calibration as lexical baseline. External pretrained-data overlap cannot be excluded. CLINC is crowdsourced, not live customer traffic.

![[SpikingBrain Assets/embedding-routing-comparison.png]]

| Evaluation | Metric | Mean +/- sample SD |
|---|---|---|
| standard | accuracy_without_rejection | 98.76 +/- 0.16% |
| standard | coverage | 39.27 +/- 21.97% |
| standard | accepted_accuracy | 100.00 +/- 0.00% |
| standard | correct_answer_fraction | 39.27 +/- 21.97% |
| standard | unsupported_rejection | 93.58 +/- 3.59% |
| heldout_intents | accuracy_without_rejection | 62.59 +/- 0.91% |
| heldout_intents | coverage | 9.19 +/- 0.71% |
| heldout_intents | accepted_accuracy | 100.00 +/- 0.00% |
| heldout_intents | correct_answer_fraction | 9.19 +/- 0.71% |
| heldout_intents | unsupported_rejection | 91.67 +/- 0.58% |

## Manual probes, seed42
Diagnostic only; never fitted/tuned. Decisions include abstentions.

| Text | Expected | Decision | Correct |
|---|---|---|---|
| I am not asking about my bank balance. I want to know my paid time off balance. | work | unknown | False |
| Don't book a flight; check when my salary will be paid. | work | unknown | False |
| This is not about my salary; I need the baggage allowance for my flight. | travel | unknown | False |
| Not a flight question: what is the balance of my checking account? | banking | unknown | False |
| Please ignore my bank balance. How do I cook an omelette? | unknown | unknown | True |
| I don't want to schedule a meeting; tell me the weather. | unknown | unknown | True |
| I don't need flight information. Tell me a joke. | unknown | unknown | True |
| I don't need a recipe, I need to check my account balance. | banking | unknown | False |

All seeds' manual scores [3, 3, 4, 2, 2]. Report coverage with accepted accuracy; embedding features do not guarantee comprehension. Encoder is substantially larger than lexical classifier; parameter/compute cost is not matched. 768-float feature cache adds storage.

Checks 27/27. Full per-seed results/calibration/errors/runtime in [[SpikingBrain Assets/embedding-routing-results.json]]. Features cache D:\AI\SpikingBrain-study\experiments\cache\clinc-nomic-classification; saved classifier D:\AI\SpikingBrain-study\experiments\nomic-domain-router.joblib. Existing Ollama service may keep normal service logs; all newly created experiment files/temp/cache on D:. Sources https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 and https://docs.ollama.com/api/embed. No model downloads or installs.

## Calibration failure and aggregation clarification
Standard seed43 selected the registered1.1 reject-all fallback; supported coverage0%, accepted accuracy undefined. The100% accepted-accuracy aggregate excludes that run and includes only four nonempty runs. Coverage mean39.27% includes allfive. This is a failure of the registered calibration policy under the available grid/constraint, not a successful classifier decision. No post-test threshold retuning was done. Negation seed43 score3/8 comes entirely from rejecting the three expected-unknown probes; it does not establish understanding.

