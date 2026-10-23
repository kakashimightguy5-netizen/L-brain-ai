# 153 - Second invalid distillation attempt: leftover TEACHER env var (2026-10-22 IST) - MEASURED

Result of the lpure_lm7.py reruns (laptop, 12000 steps):

| run | final val bits/byte | verdict |
|---|---|---|
| teacher7_d256 (dense d256, meant to be a plain teacher) | 3.0223 (train loss 1.91) | FAIL - INVALID |
| kd7_mask_K32_d256 (student of teacher7) | 2.6996 | FAIL - INVALID |
| diag run via lpure_lm8.py | crashed: loaded a wrong checkpoint | not run |

## Cause (found in the result json, not a model bug)
teacher7_d256.json has "teacher": "teacher_dense_d256.pt". The PowerShell tab still had $env:TEACHER set from the earlier broken distillation run, and environment variables persist between runs in the same tab. So the 'plain' dense teacher was secretly being distilled from the untrained random teacher (KDW 0.5), which pushed it to 3.02 instead of the ~1.73 of long_dense_d256 (lpure_lm5.py, same architecture, 1,713,152 params). The student then learned from that poor teacher. The lm8 crash had the same cause (TEACHER pointed at an old L5 checkpoint in that tab).
lpure_lm5.py vs lpure_lm7.py code was diffed: the only differences are the teacher option and checkpoint saving; the training step is identical.

## Fix
Clear every env var before each run, or open a new tab. Use new tags: teacher8_d256, kd8_mask_K32_d256, diag_sparse_float_K32_d256, diag_dense_tern_d256.
Lesson: always print the json 'teacher' field and the first loss line before trusting a run.
