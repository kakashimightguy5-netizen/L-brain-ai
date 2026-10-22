---
tags: [project, ai-architecture, neuromorphic, hardware, roadmap]
created: 2026-10-05
status: concept
---
# Brain-Inspired Model: Concept & Roadmap

Related: [[Brain-Inspired AI]] · [[Where Transformer Compute Really Goes]] · [[Think-When-Needed Paper]] · [[Research - AI Compute Crisis]]

## Kakashi's vision (05 Oct 2026)
- Work on **both software (architecture) and hardware**. **Architecture first, then hardware.**
- Leave generic transformers behind and design a **new brain-inspired architecture**, researching spiking neurons, synapses and real biology.
- **No living neurons.** Build an **electrical chip** that behaves like neurons (cheaper than lab-grown cells). Design PCBs and structures.
- **Target:** a **4B–7B parameter model** that runs comfortably on our own hardware (no million-GPU clusters).
- **Core concept:**
  - **Data = DNA**: stored information, kept as the model's **second brain** (external memory it learns from).
  - **Neurons = parameters.**
  - **Every domain is a neuron cluster** (e.g. frontend developer). When it learns something new, it **creates a new node**. When a frontend question comes in, the model goes into that domain's nodes and accesses all of them.
  - Uses a DNN to manage **relationships and inheritance** between data and nodes.

## Claude's analysis

### What's strong about the concept
It combines three properties today's LLMs lack:
1. **Knowledge stored outside the weights** (DNA / second brain), so it's updatable without retraining
2. **Domain-organised modules**, so only the relevant part activates (brain-like sparsity, like Think-When-Needed)
3. **Growth when learning**: new nodes for new knowledge, so old knowledge isn't overwritten (solves **catastrophic forgetting**, a major unsolved LLM problem)

### Biology check (to make the analogy precise)
- In the brain, **DNA is the blueprint**, not where memories live. Memories live in **synapses** (connection strengths).
- Adults grow few new neurons (mainly in the hippocampus). Learning mostly creates **new synapses and strengthens existing ones**.
- Better mapping: data/second brain ↔ **hippocampus (fast memory)**, domain modules ↔ **cortical areas**, new nodes ↔ **new synapses/connections**. Keep "DNA" as the project metaphor if you like, but present the accurate mapping to judges.

### Prior art to check (novelty ladder: don't skip)
| Part of our idea | Existing work |
|---|---|
| Knowledge outside weights | RETRO (retrieval), Meta "Memory Layers at Scale" (2024), knowledge graphs + RAG |
| Domain modules + routing | Mixture-of-Experts (DeepSeek, Mixtral), domain-expert MoE ("Branch-Train-Mix") |
| Growing new nodes when learning | Progressive Neural Networks (2016), Dynamically Expandable Networks, neurogenesis-inspired continual learning |
| Spiking/sparse | SpikingBrain, SpikeLLM, Loihi 2 LLM |
| Adaptive effort | ACT, PonderNet, CALM, TRM, our Think-When-Needed |
➡️ Each piece exists separately. **A possible novel contribution is the combination**: spiking, domain-modular, growth-on-learning, with external memory, sized to run on small custom hardware. Needs a proper prior-art pass to confirm nobody has combined them.

### Reality check: compute and cost
- Training a **7B model from scratch** properly (~140B+ tokens) needs roughly **6×10²¹ operations**, about **15,000+ H100 GPU-hours**, tens of thousands of dollars. Not feasible for us at the start.
- ✅ Feasible path: **prove the architecture at tiny scale (10M–100M params)** on a laptop or Colab, then scale only when results justify it. Look for compute grants (e.g. the IndiaAI Mission GPU portal: *verify eligibility*).
- Alternative shortcut: **attach our modules and memory to an existing open model** (1–4B) instead of training from scratch.

### Reality check: hardware
- A **PCB alone can't be a neuron chip**: the neurons must live in a chip (FPGA or custom silicon). The PCB carries the chip, memory and power.
- Custom chip design (ASIC) at full scale costs crores. Realistic hardware ladder:
  1. **Software simulation** of spiking neurons (snnTorch, Brian2, Lava)
  2. **FPGA**: program our neuron/synapse design onto a reconfigurable chip (low-cost boards exist)
  3. **Tiny Tapeout**: a community program that manufactures tiny real chips cheaply (*verify current cost/schedule*)
  4. Our own **PCB** around the FPGA/chip: power, memory, I/O
  5. Long term: analog synapses (memristors), custom neuromorphic ASIC

## Roadmap
| Phase | Goal | Scale | Output |
|---|---|---|---|
| **0 (now → 14 Oct)** | Symposiums | — | Think-When-Needed paper (already brain-inspired) |
| **1. Spec** | Write the architecture: domain modules, growth rule, memory ("DNA"), router, spiking neurons | Paper design | Architecture doc + diagrams |
| **2. Tiny prototype** | Prove: (a) learns domain B without forgetting domain A; (b) only relevant modules fire; (c) energy per answer | 10M–100M params, laptop/Colab | Experiments vs. a same-size transformer |
| **3. Scale up** | 0.5–1B params, or bolt onto an open model | Rented/granted GPUs | Benchmark results, paper |
| **4. Hardware v1** | Spiking modules on an FPGA + our PCB | Small model | Working demo board |
| **5. Target** | 4–7B model on our own hardware | Grants / partners | Product / research paper |

## Phase 1 questions (answering our own "why / who / what / how / impact" rule, see [[Structured Think-Before-Build Flow]])
- [ ] **Why:** which LLM problem do we solve first? (forgetting? energy? updatability?) Pick ONE headline.
- [ ] **Who:** who uses a 4–7B brain-like model on small hardware? (edge devices, Indian-language users, offline use)
- [ ] **What:** exact definition of a "node", a "domain" and the growth rule
- [ ] **How:** how are nodes created, connected and pruned? How does routing pick domains?
- [ ] **Impact:** which metric proves it's better? (accuracy per joule, forgetting rate, params activated per answer)
- [ ] Full prior-art pass on the combined idea
