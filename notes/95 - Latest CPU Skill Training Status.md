# Latest CPU Skill Training Status

2026-10-07. V5 follow-up completed, CPU-only on D:.

- Fresh mixed copying: 7/16 previous adapter → 11/16 new adapter.
- Earlier varied copying: 9/12 → 11/12; patterned copying retained 16/16.
- Independent JavaScript: 3/4 → 4/4, including firstOrNull zero case; fresh first-item tests 8/8 include zero, false, empty string and null.
- Original retention: 6/6. Scoped tool selection 12/12, code discrimination 8/8.
- Unseen four-schema test: previous adapter 3/4 → new 4/4, but baseline was also 4/4.
- Broad tool benchmark regressed 5/11 → 4/11; baseline 8/11. Do not promote the new selector for general agent use.
- All-V5 OpenCode pipeline failed 0/2 before reading. Hybrid V4 selector + V5 copier passed 3/3 actual file reads, including the earlier failed multiline file. This is a tiny, specialized workflow test.

Default remains baseline. Optional D: launchers start-targeted-v5-hybrid.ps1 and opencode-v5-hybrid.ps1 use the tested combination. Both checkpoints, raw failures and data are retained. No broader-model superiority claim. Base remains 494,032,768 parameters; each skill adapter 270,336 coefficients. These are transformer skills, not a newly trained spiking LLM.

Next: improve broad-schema transfer with varied schema types and competing tools, validate truly unseen file/task families, and investigate remaining five fresh-copy failures without feeding benchmark rows into training. See [[94 - V5 Targeted Follow-up Results]] for evidence.
