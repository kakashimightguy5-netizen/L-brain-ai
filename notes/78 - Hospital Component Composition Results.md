# Hospital component composition — 2026-10-06

Local output: http://127.0.0.1:8766/hospital.html . Fictional Harborview Hospital, local-only demonstration, HTML/CSS/JavaScript.

## Results
Hospital page: 22/22 shared browser checks. Required specialties, labeled form, invalid-input blocking, no-transmission confirmation, keyboard focus, mobile menu state, semantic sections, desktop/mobile overflow and no external requests/runtime errors passed. Tested real Chrome at 1440x1000 and 390x844. Screenshots reviewed; initial extraction lost the intro wrapper and retained some college copy. Assistant corrected text, wrapper and supplied a heart illustration, then repeated checks. These corrections are manual engineering, not intrinsic self-healing. The checks did not detect the stale prose, so their scope is limited.

12 existing records (11 original frontend records plus the taught college page) remained byte-identical. Eight generic component topics added in a separate branch; UTF-8 corrections appended as v2 without overwriting v1. 46/46 original alias routing decisions preserved; this is a replay regression set, not unfamiliar wording generalization. Component JSON verified equal to stored immutable component records.

## What was actually built
Assistant extracted eight HTML structures from the existing college lesson and supplied domain text separately. The new composer applies mappings, keeps CSS and interactions, and changes the illustration. No complete hospital HTML was taught as a new lesson. However the assistant wrote the composer, hospital copy and corrections after reading the requested task. Thus this is a heavily assisted domain transfer experiment, not a blind held-out test or autonomous neural generation. The page keeps the same college layout. No new Qwen run or direct hospital Qwen comparison was performed; prior college scores remain separate.

## Main unresolved weakness
Raw router selects html.college_page for the hospital brief (score about 0.286). It does not independently choose/arrange generic blocks. Treating that result as a completed hospital page would be wrong. Working output was assembled by the assistant's deterministic composer. A learned planner, safe task selection, variable card counts/layouts and stronger content-specific tests remain unimplemented. Old stored bytes are preserved; this does not guarantee every possible old answer unchanged.

## Process and evidence
All new files/profiles/caches/screenshots on D:. Existing C: executables/packages read only. First launch used a directory before it existed and executed nothing; created directory and reran. A correction command had a quoting parse failure and changed nothing; used a saved Python correction script. An integrity audit caught Windows-default decoding in initial component registration. Appended verified UTF-8 v2 records, kept v1 records, and reran the audit. Saved successful checks, router result, source, generic components, hospital data and integrity evidence. No medical recommendations, claims, real appointments or public deployment.

## Next proposal
Build an explicit generic page-spec interface and validate domain/task selection before rendering. Then freeze components and composer, choose new briefs afterward, and test novel section order/card counts plus content-specific checks. Keep raw model routing separately scored. Actual automatic composition is the remaining objective; the present result establishes reusable engineering components only.

![Hospital preview](SpikingBrain Assets/hospital-composition/hospital-preview.png)
