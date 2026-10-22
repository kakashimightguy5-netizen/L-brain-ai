# 75 - College Landing Page Qwen Comparison Protocol

Compare on one fixed fictional college brief, HTML/CSS/vanilla JS, offline dependencies, 1440px and390px renders. Qwen is the already installed qwen3.5:9b, local Q4_K_M checkpoint (API metadata parameter count9,653,104,368). One generation, temperature0.2, seed1902, thinking disabled,8192 context,5000 output-token limit, automatic GPU/CPU allocation. Preserve request, unedited output, finish reason, timing and runtime allocation. Formatting extraction only; no hidden fixes before testing.

Existing frontend prototype is a small immutable fact/template retriever, not an LLM. Send it exactly the same brief and record whether it can produce a complete page. Do not represent agent-written HTML as the original prototype's output.

As an explicitly assisted extension, append a human-authored college page-composition template with a structured plan to a separate copy of its knowledge library, retaining all old records. Show the rendered template output separately; this is demonstration of taught template reuse, not a fair proof of independent generative superiority. Do not use Qwen output as that template.

Freeze shared behavioral/content criteria before viewing Qwen output: required sections and three programs, offline rendering, one h1, title/viewport, anchors, mobile menu/aria state, valid local form feedback, invalid submission refusal, labels, accessible names, keyboard focus, no horizontal overflow, and no runtime errors. Render both complete pages in a real headless browser with D: profiles/cache. Screenshots and functional checks are local. Preserve failing original output; do not tune a one-shot benchmark through silent repairs.

Save every new artifact, browser profile, cache, log and report on D:. Keep earlier learned nodes and models unchanged. No public hosting or live admissions backend; this is a local model comparison within the existing experiment project. Report small single-brief scope, arbitrary human visual judgments separately from objective checks, and all assistance differences.
