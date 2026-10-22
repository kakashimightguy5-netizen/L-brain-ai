# Latest clean-retention result

The combined neural-and-memory model passed the 97% clean-accuracy target after sequential learning on a frozen, previously unused test set.

| Configuration | Clean accuracy | One noisy observation |
|---|---:|---:|
| CNN, 200 exemplars | 96.60% | 71.87% |
| CNN + memory, same 200 exemplars | 97.73% | 77.54% |
| CNN, 500 exemplars | 97.62% | 75.27% |
| CNN + memory, same 500 exemplars | 98.41% | 82.35% |
| Small conventional ANN, 500 exemplars | 94.64% | 71.97% |
| Small LIF spiking network, 500 exemplars | 94.56% | 72.13% |

Five initialization runs share the same 765-example test set. This is a new data partition from the previously unused UCI training file, not five independent populations or a writer-disjoint evaluation. All models and validation decisions were frozen before final testing. No old pretrained weights or future-task training images were used.

Memory is valuable when combined with the neural model, but it also changes some correct predictions into errors; those harms are included. Enlarging memory has a storage cost. The 90% noisy-input target remains unmet. The spiking model runs locally but shows no advantage over the matched ANN. Pure EWC remains an experimental baseline.

See [[57 - Clean Retention Confirmation and Spiking Results]] for first-task retention, forgetting, timing, storage, failures and limits. Protocol: [[56 - Frozen Clean Retention and Neural Memory Protocol]]. Code, runnable deployment models, data provenance and raw evidence are archived in SpikingBrain Assets/clean-memory-confirmation. Everything was saved on D:.
