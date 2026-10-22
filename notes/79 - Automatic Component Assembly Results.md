# Automatic component selection and assembly — 2026-10-06

Local gallery: http://127.0.0.1:8767/ . Five pages automatically assembled from existing structural blocks. Supported domains: college/university, hospital/clinic, library, bakery.

## Measured results
10/10 planner cases: five compositions and five clarification cases. 92/92 browser requirement assertions across five outputs: library 19/19, bakery 19/19, clinic 19/19, college 19/19, library without form 16/16. These are a different suite from the earlier 22-check college comparison, not directly interchangeable scores. Real Chrome tested 1440px and 390px; screenshots saved. Varied section order, 1/2/3/4/5 cards, omitted sections and no-form behavior. Empty and invalid input blocked; valid forms show explicit no-transmission confirmation. No runtime errors or external requests. Visual review of bakery output: retains college book/building illustrations and a three-column grid with empty space for two cards. Those design limitations are not detected by the functional suite.

## Honest capability statement
The assistant wrote a deterministic planner and generic renderer. It selects a domain from an explicit keyword table, parses brand/cards/sections from a bounded grammar, and composes stored structures with parameterized text. It is automatic execution of supplied rules, not a learned neural planner, unconstrained text-to-code model or evidence of superiority over Qwen. No Qwen test repeated this round. It does not change FrontBrain.route; that raw model still has the previously recorded college-routing weakness. The separate planner avoids that pathway for supported briefs. It follows requested section order rather than discovering the best order.

Planner source hash frozen before concrete test fixtures: D62A337684637F1ACB20CB52C1AD421984D3A6554860668DC92F15C25CE2D427. It was not modified after the suite. The assistant designed both implementation and fixtures; this is a regression/development test, not independently authored or blinded held-out evaluation. Test domain vocabulary is intentionally supported. General free-form wording, adversarial/security tests, broader brands/cards and visual diversity remain untested.

## Preservation
12 original records preserved. All 26 component branch live records and backups match stored hashes, including earlier corrected component versions. No old records or routing config edited. Old alias routing was 46/46 in prior hospital audit; not rerun here because FrontBrain, nodes and routing did not change. No new complete-page lesson or neural fine-tuning.

## Brief format
Example: Create a library page for "Lakeside Library". Cards: Fiction | History | Science | Children. Sections: environment, cards, form, steps.
Brand must be quoted. Cards are pipe-separated, 1–6. Sections are comma-separated from cards, steps, environment, form. Unknown domains, conflicting domains, duplicate sections, missing required fields and excessive card counts return clarification. This explicit format is the current input contract.

## Artifacts and process
All new sources, pages, plans, profiles, screenshots and logs on D:. Existing C: runtimes read only. Frozen planner then generated fixtures then real browser checks then integrity audit. Gallery includes all five outputs and links to plan evidence. No public hosting or real submissions. Evidence archived in SpikingBrain Assets/automatic-composition with SHA-256 manifest.

## Next proposal
Independent briefs in ordinary language, stronger domain/brand consistency and visual tests, adaptive grid/art variants, and ultimately a learned planner evaluated against these deterministic rules. Keep raw retrieval, rule planning and rendered output separately scored.
