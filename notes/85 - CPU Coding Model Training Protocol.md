# CPU coding model training protocol

User requested Python, Java, C, C++, React, HTML, CSS, JavaScript, Node, Express, agent creation and tool calling; OpenCode integration; lightweight fast CPU-only inference; superiority over 4B–48B models as aspiration.

Hardware verified: Intel i7-13620H, 10 cores/16 threads, 15.71 GiB RAM, about 84.7 GiB initial free D: capacity. Existing installed PyTorch 2.14.0+cpu and Transformers 5.17.0. GPU unavailable to this training runtime and explicitly excluded. Windows OpenCode C: npm launcher was a 479-byte non-executable placeholder. Installed the matching official Windows binary 1.18.30 in a separate D: runtime and redirected XDG config/data/cache/state and temp to D:. Local model provider registered successfully. Existing C: install not changed.

## Model identity and scope
Pretrained Qwen/Qwen2.5-Coder-0.5B-Instruct; pinned revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25. Official model card: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct . Apache-2.0 license retained. This is a Qwen-derived Transformer adaptation, not a newly pretrained language model or spiking architecture. Existing 19,935-parameter frontend selectors remain separate and unchanged.

Custom real LoRA weight training: freeze base parameters, rank 4/alpha 8 on 48 q_proj/v_proj modules, train 270,336 coefficients. V1 used 37 assistant-authored examples, one epoch, CPU. Initial setup failed on tokenizer return type before training; fixed explicit input_ids access, preserved failed log. V1 trained in about 52 seconds. Training covers listed categories but does not demonstrate mastery. No independent dataset or broad cross-model benchmark yet.

## Honest initial evaluation
Four tiny Python/JavaScript function tasks and two structured read/glob calls. V1 better followed code-only output, but copied schemas into tool arguments. Original strict results preserved; fair executable extraction of unchanged function bodies yields base 6/6 semantic tasks, V1 4/6. Therefore V1 has a genuine tool-call regression; cannot claim an improved coding model from this run. Direct CPU dynamic-INT8 generation observed roughly 13–16 tokens/s on short adapter tasks; measurement includes prompt overhead and is not comparable to prior GPU-assisted Qwen 9B without a matched benchmark.

V2: separate checkpoint, 63 examples with natural-language tool requests, two epochs. Same failed prompts are development cases, not held-out wins. More training remains an experiment; lower loss is not evidence of dependable agent behavior. V1 retained. Inference server temporarily stopped to reduce memory pressure during V2; will restart after evaluation.

## Deployment work
Downloaded official CPU-only llama.cpp b11429 (commit d81235049); converter source pinned to the same tag. Plan: merge adapter into base, export FP16, package GGUF, test Q8/4-bit CPU generation and memory. OpenCode provider uses local-only API; deterministic read/glob/grep agent, three-step limit for first integration test. Test marker in new D: scratch workspace, unseen by training. A wrapper parser accepts model-produced XML calls or bare JSON, checks supplied tool name/required arguments, never invents calls/values.

## Constraints and unfinished objective
The user authorized local training/configuration; no paid cloud compute or messaging. All new models, checkpoints, code, runtime installs, caches, temp files and records on D:. Base weights remain immutable, old frontend records/checkpoints preserved. No claim of beating 4B, 8B, 9B, 14B, 27B, 30B, 35B or 48B models. A 48B dense model's 4-bit weight payload alone is about 24 GB decimal before overhead, exceeding this laptop's 16 GB RAM. Quantization does not confer larger-model capability. Broad stronger coding/agents require a much larger validated corpus, independent tests and adequate training budget.

Results will be recorded after actual model and OpenCode tests. This note is protocol/progress, not a completion claim.
