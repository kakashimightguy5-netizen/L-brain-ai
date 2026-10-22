---
tags: [research, llm-history]
created: 2026-10-05
---
# LLM Evolution: What Each Generation Improved

Back to [[01 - Deep Research Report]]

## Kakashi's question
As GPT-2 → GPT-3 → GPT-4 → GPT-5 → newer versions arrive, what are they improving? Better outputs? Fewer tokens? What are they trying to do next?

## Timeline
| Model | Year | Main leap |
|---|---|---|
| GPT-2 | 2019 | 1.5B params, ~1k-token context, fluent text |
| GPT-3 | 2020 | 175B params: **scale** brought few-shot learning |
| ChatGPT (GPT-3.5) | 2022 | **RLHF**: follows instructions, chat |
| GPT-4 | 2023 | Much better reasoning, images in, longer context (details undisclosed) |
| GPT-4o | 2024 | Native voice/vision, **much cheaper and faster** |
| o1 / o3 | 2024–25 | **Reasoning via RL**: thinks longer before answering (test-time compute) |
| DeepSeek-V3 / R1 | 2024–25 | Open, **MoE 671B / 37B active**, ~2.7M H800 GPU-hours for the final run: efficiency shock |
| GPT-5 | Aug 2025 | **Router**: a fast model and a deep "thinking" model, auto-chosen per question |
| 2026 wave (GPT-5.x, Claude 4.6–5, Gemini 3.x, DeepSeek V4, Qwen3.5) | 2026 | Agents, long autonomous coding, **adjustable reasoning effort**, cheaper tiers *(industry sources; verify specifics)* |

## Answering the question: better outputs or fewer tokens?
- **Outputs: much better.** Each generation improved reasoning, coding and instruction-following.
- **Cost per token: down massively.** The price to reach a fixed benchmark score fell **9× to 900× per year** (Epoch AI). Capability per parameter doubles every ~3.5 months (Densing Law).
- **Tokens per answer: UP for reasoning models.** Reasoning mode uses **300–800× more tokens** and ~30× more energy on average. *Cheaper per token, but many more tokens per answer.*
- **Context window:** ~1k (GPT-2) → hundreds of thousands to 1M+ tokens today.

## The 9 trends across generations
1. Scale (more params and data), then **efficiency** (MoE, hybrids, distillation)
2. Alignment (RLHF, instruction following)
3. Longer context
4. Multimodality (text + images + audio + video)
5. **Reasoning** through test-time compute
6. Tool use and **agents**
7. Price per capability collapsing
8. **Adaptive effort**: routers and effort settings (fast vs deep)
9. Architecture shifts: MoE, Mamba/linear hybrids, memory layers

## What labs are trying to do next (2026)
- **Continual learning:** update knowledge without retraining (Titans, Nested Learning)
- **World models:** learning how environments work (Genie, V-JEPA)
- **Orchestration:** small models delegating to specialists
- **Refinement:** propose → critique → revise loops
- **Data wall:** human text (~300T tokens) could be exhausted between **2026 and 2032** (Epoch), pushing toward synthetic data and efficiency
- **Energy wall:** power is now the bottleneck (see [[Research - AI Compute Crisis]])

➡️ **Link to our project:** the labs' next goals (continual learning, memory, adaptive effort, efficiency) are **exactly the brain features** in our blueprint. We're aligned with where the field is heading, and that's both validation and competition.

## Sources
- Epoch price trends: https://epoch.ai/data-insights/llm-inference-price-trends
- Densing Law: https://www.nature.com/articles/s42256-025-01137-0
- DeepSeek-V3 cost analysis: https://www.interconnects.ai/p/deepseek-v3-and-the-actual-cost-of
- 2026 research trends: https://venturebeat.com/technology/four-ai-research-trends-enterprise-teams-should-watch-in-2026
- Data exhaustion: https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data
- 2026 release timeline (industry, unverified): https://jobsecuritymeter.com/guides/frontier-ai-models-2026
