# Independent language challenge results

[[25 - State Tracing and Language Challenge Protocol]] · [[22 - Local Embedding Routing Results]] · [[08 - SpikingBrain Process Log]]

100 authored families, 300 texts: 50 supported and 50 unsupported tasks, each expressed neutrally, reframed, and with a negated distracting topic. Labels and text frozen before evaluating either router. Saved seed42 classifiers and thresholds unchanged. No fitting, calibration or rule additions. Dataset SHA256 9a965ac683e6cbd43e73a9b59d953ead82333a5933cfea3cdb885f827703c0e1. No exact overlap with CLINC or the previous eight probes. Templates within family are dependent; this is a controlled diagnostic, not a representative human-language benchmark. Reframing uses fixed sentence wrappers, not diverse semantic paraphrases. Unsupported credit-card topics intentionally test boundaries near banking; labels are our chosen support policy.

| Router | Style | Correct decision incl. rejection | Supported coverage | Accepted accuracy | Unsupported rejection |
|---|---|---:|---:|---:|---:|
| lexical | all | 54.3% | 34.7% | 92.3% | 76.7% |
| lexical | neutral | 69.0% | 50.0% | 100.0% | 88.0% |
| lexical | reframed | 67.0% | 42.0% | 100.0% | 92.0% |
| lexical | negation | 27.0% | 12.0% | 33.3% | 50.0% |
| embedding | all | 56.3% | 30.7% | 100.0% | 82.0% |
| embedding | neutral | 64.0% | 46.0% | 100.0% | 82.0% |
| embedding | reframed | 64.0% | 42.0% | 100.0% | 86.0% |
| embedding | negation | 41.0% | 4.0% | 100.0% | 78.0% |

Entire family correct across all three styles: [('lexical', 0.23), ('embedding', 0.36)]. Unsupported abstention is correct; supported abstention is a failure to answer. Accepted accuracy must be read with coverage. Larger sample count does not remove template bias. Full texts/labels/predictions including failures [[SpikingBrain Assets/language-challenge-results.json]] and [[SpikingBrain Assets/language-challenge.json]]. Controls 5/5. All new features/results/scripts on D:, existing local Ollama reused. This tests topic routing, not answer generation.
