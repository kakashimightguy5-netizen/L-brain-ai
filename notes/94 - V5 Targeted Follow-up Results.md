# V5 Targeted Follow-up Results

Completed 2026-10-07. CPU-only native LoRA continuation on frozen Qwen2.5-Coder-0.5B-Instruct; all new files on D:. Base: 494,032,768 parameters; each separate skill adapter: 270,336 coefficients. Two skills each trained one pass over 56 examples, including eight earlier training examples replayed. Copy continued V4 copy-diverse; selection continued V4 selection. No benchmark test rows were added to training. This is not a spiking LLM trained from scratch.

## Controlled diagnosis

The real multiline control repeated read with both auto and disabled tool calling. Removing tools produced malformed file contents. Thus disabling tools alone does not fix the learned task. The V5 gateway preserves raw generated text when blocking post-read tool calls, avoiding the prior gateway hiding repeated-read JSON as an empty answer. It never invents a file answer. Copy curriculum combines randomized content lines, actual argument strings and Windows headers; selection covers new simple schemas and JS empty/zero/false cases.

## Exact benchmark results

| Model | Test group | Category | Passed |
| --- | --- | --- | --- |
| prior | fresh-copy | mixed-multiline-copy | 7/16 |
| copy-next | fresh-copy | mixed-multiline-copy | 11/16 |
| copy-next | prior-diverse | diverse-copy | 11/12 |
| copy-next | prior-pattern | exact-copy | 16/16 |
| prior | fresh-selection | js-first | 8/8 |
| prior | fresh-selection | schema-tool | 8/8 |
| selection-next | fresh-selection | js-first | 8/8 |
| selection-next | fresh-selection | schema-tool | 8/8 |
| selection-next | prior-scoped | tool-selection | 12/12 |
| selection-next | prior-scoped | code-discrimination | 8/8 |
| base | fresh-unseen-schemas | unseen-schema | 4/4 |
| prior | fresh-unseen-schemas | unseen-schema | 3/4 |
| selection-next | fresh-unseen-schemas | unseen-schema | 4/4 |

Original retention: 6/6.

Independent earlier suite (freshly run, not trained on):

```json
{
  "v5-selection": {
    "python": {
      "passed": 4,
      "total": 4
    },
    "js": {
      "passed": 4,
      "total": 4
    },
    "tool": {
      "passed": 4,
      "total": 11
    },
    "result": {
      "passed": 4,
      "total": 10
    }
  }
}
```

Actual OpenCode tests: 0/2. Both require an actual completed tool result and exact final text, ignoring only trailing CR/LF.

```json
[
  {
    "name": "v5-one",
    "reads": 0,
    "passed": false,
    "answer": ">v5-one.txt\none two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty thirty forty fifty sixty seventy eighty ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred ninety ninety hundred one hundred twenty one hundred twenty three hundred thirty four hundred fifty six hundred seventy eight hundred",
    "expected": "Novel_CASE: 72814 / []?!"
  },
  {
    "name": "v5-many",
    "reads": 0,
    "passed": false,
    "answer": "\u0432\u0435\u0441\u0442 v5-many.txt",
    "expected": "head_Zero=0\n    Preserve_FALSE=false!\n<p>KeepCASE & spaces</p>\nlast_39172?"
  }
]
```

Validation losses (not task accuracy):
- copy-next: warm start 0.213295, final 0.021918; 801.1 seconds.
- selection-next: warm start 0.483582, final 0.348544; 432.4 seconds.

## Interpretation and reproducibility

Fresh randomized test identifiers are separate, but several line and task families are shared with training; synthetic scores do not establish general coding superiority. Numeric/boolean tool schemas use names absent from training and are labeled separately. Earlier test suites are regression checks, not newly untouched benchmarks. No larger 4B–48B competitor comparison was performed. The baseline default is unchanged; specialist integration remains opt-in until broad correctness and retention justify promotion. Independent merged Q8 variants duplicate the shared base at runtime.

Raw outputs, checkpoints, datasets, controls, timing data, gateway adjustments and process logs: D:\AI\SpikingBrain-study\experiments\cpu-coder\targeted-v5. Large model files are referenced by SHA-256 rather than duplicated into Obsidian. Training success, native completion scores and pipeline correctness are separate claims.

See [[92 - Targeted V4 Results and OpenCode Root Causes]] for prior failures.


## Final hybrid integration correction

The all-V5 selector pipeline failed both real OpenCode cases before executing a read (0/2). Synthetic scoped 12/12 did not establish actual integration transfer. Switching only the selector backend back to V4, with the new V5 copying adapter retained, passed 3/3 real cases: fresh single-line, fresh mixed multiline, and the earlier failed multiline file. All require actual completed read tool output and exact generated contents. This small opt-in pipeline result is separate from native model scores and is not general agent reliability. The new selector fixes the independent JavaScript suite (4/4 vs prior 3/4), but broad tools regress to 4/11 vs prior 5/11 and base 8/11. Therefore it is not selected for broad agent use. Original retention: 6/6.

Selected experiment: baseline for general tasks, V4 selector for scoped read/glob/grep, V5 copy-next for exact read-result copying. Default user configuration remains baseline. start-targeted-v5-hybrid.ps1 and opencode-v5-hybrid.ps1 resume the opt-in pipeline. Old checkpoints and the V5 selector remain available for research.
