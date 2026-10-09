# Joint roadmap: getting more intelligence out of the Rineya small cores (BWM-001 + RCWM-T1), 2026-10-08

Merged by BWM-001 from both sessions' ideas. Owners: **[B]** BWM-001, **[R]** RCWM-T1, **[J]** joint. The goal is
intelligence measured rigorously: correct planning, accurate consequence prediction, rules that hold out of range, and
retained general skill. Every test is pre-registered, paired, and has an out-of-distribution arm.

## What both programmes have taught us

1. **Small LoRAs learn data-forced shortcuts that look perfect in distribution.**
   - BWM F-01: a legality rule.
   - RCWM PREREG-2: a prerequisite copy shortcut that worsens with seed.
   - Fix shortcuts with data or explicit checks, and always audit out of range.
2. **Deferring to the frozen base repairs only errors the base itself gets right.**
   - BWM F-03/F-06: confident errors survive.
   - RCWM offline result today: deferring on adapter/base disagreement does *not* fix the copy shortcut, because Titan
     does not know the precondition rule either.
3. **Learned world models generalize better than learned plan generators, and search turns that into capability.**
   - BW_F is exact up to 15 blocks.
   - World model + beam search reaches 0.63 against 0.13 on hard-style problems.
   - Frozen no-world-model arm on PlanBench-hard: 4.5%.
4. **The skill lives in blocks 8-15 (RCWM R12/R14).** Restricting the update there cuts off-task drift by about 1/3.
5. **Titan's edge is exact copying; plain modules can beat it on rule arithmetic.** That points to hybrids:
   - copy with the core;
   - compute with a rule step or tool.

## Ranked ideas: expected value per GPU turn

| # | Idea | Hypothesis / test (pre-registered) | GPU turns | Owner |
|---|---|---|---:|---|
| 1 | **Finish the frozen Blocksworld runs** (hard 8 shards, mirror, PlanBench-600) | S-beam beats S-direct on hard, paired McNemar; mirror gap < 10 pp | ~3 (queued) | B |
| 2 | **Out-of-range audit of the ledger fix** (k = 8/16, balances beyond range, long event lists) | Exact match and regret degrade gracefully with k on a fresh bank; this is the honesty check before any claim | < 1 | R |
| 3 | **Beam search over RCWM ledger**, with the T2 LoRA as F inside the BWM planner | Goal-reaching success and regret beat PREREG-2's fixed-candidate selection at matched model calls; tests whether near-oracle prediction becomes planning | ~1 | J (R runs, B code) |
| 4 | **Search distillation into the proposer (expert iteration)**: train Pa on F-verified beam solutions of generated 6-12-block problems | The *model itself* plans further: frozen S-direct on the hard mirror rises well above 13%, paired vs Pa_v2; then once on hard with a new addendum | 2-3 (traces 1-2, train 1) | B |
| 5 | **De-shortcut rule data (T2b)**: copy share 66% -> ~30%, precondition minimal pairs, app-del quota, P upweighted, plus a self-distillation control; **run with LoRA on blocks 8-15 only as a second arm** (merges R1 and R2) | Frozen gates G3 (app-del CI > 0) and G4 (false copy not up), then a sealed bank; 8-15 arm non-inferior with drift ratio < 0.8 | 3-4 | R |
| 6 | **Explicit legality checker trained on minimal pairs, shared across domains** (RW2 legality, Blocksworld preconditions, RCWM prerequisite) | A small verifier LoRA recovers rule errors that deferral cannot, on out-of-range keys | 1-2 | J |
| 7 | **Two more seeds of BW_F and Pa_v2** | The world-model effect replicates across seeds; required before any public release | 2 | B |
| 8 | **Exact arithmetic inside the step** (scratchpad, or a tool call reported separately) | Register mix/wrap and ledger scale become exact at chain k = 4 on a sealed bank | ~1 | R |
| 9 | **Transfer of the recipe** to a second domain (PlanBench Logistics, or Mystery Blocksworld Regime B with rules in the prompt) | Learned F + search beats S-direct out of size range in a new domain; the generality claim | 3-4 | B |

## Zero-GPU work (CPU or offline, in parallel)

- [B] Input-novelty signal for confident errors (F-06; hidden-state distance probe on stored keys).
- [B] The devfreeze package and repo export after each milestone.
- [R] MLP-vs-Titan analysis to define the "copy with the core, compute with a module" hybrid.
- [J] Shared shortcut-audit checklist: out-of-range limits and balances, minimal pairs, seed trend of shortcut metrics.

## Dropped (one line each)

- Disagreement or confidence deferral as the prerequisite fix: the base is wrong too (RCWM offline, today).
- Proposer amortization in RegisterWorld: there is no compute advantage over search in a 5-action world (BWM P-05).
- Dense merges of rule LoRAs: they bake shortcuts in irreversibly; keep adapters unmerged.
- Blanket verifier calls: worse than F alone in distribution (BWM F-04: 0.760 vs 0.886).

## Proposed GPU-turn order (strict alternation, about 2-3 turns per programme per day)

| Day | BWM turns | RCWM turns |
|---|---|---|
| 1 (now-tomorrow) | #1 hard shards 0-3 -> 4-7 | PREREG-2 job -> L1 screen (queued) |
| 2 | #1 mirror + PlanBench-600 -> #4 trace generation | #2 out-of-range ledger audit -> #5 T2b seed 0 (full + 8-15 arms) |
| 3 | #4 distillation training + frozen mirror eval | #3 beam over ledger (BWM planner) -> #5 T2b seed 1 |
| 4 | #7 seeds (F, Pa) | #5 T2b seed 2 -> #8 exact arithmetic |
| 5 | #6 legality checker (joint) -> devfreeze package -> #9 scoping | #6 legality checker (joint) |

Release gates are unchanged. Nothing goes to Hugging Face or vault promotion without the user's explicit go-ahead.

## Amendments from RCWM's judged brainstorm (30 ideas, 3 judges), accepted

**New evidence**
- **(i) Computed-predicate rules are not learned from bare-answer rows.** Register branch choice sits at chance
  (0.51) before and after T2, in all 3 seeds, while the in-branch arithmetic is right. This is the same class as BWM's
  RW2 `|2x| <= L` legality shortcut: one cross-programme finding, not two.
- **(ii)** Only 24% of T2's update energy sits in blocks 8-15, and 22% sits in blocks 0-7, where Titan's own delta
  was zero.
- **(iii)** Seed consensus: where 3 seeds agree (85% of steps), they are right 0.995 of the time. This is usable as a
  certificate, but likely blind to shortcuts the seeds share.
- **(iv)** KL drift missed Titan's abstention loss, so retention must be measured by correctness.

**Edits to the ranking**
- **#2 becomes a "rule or trigger?" out-of-range map** [R], PREREG-3, ~1.5 GPU-h, no training. It probes:
  - unseen statuses with posted twins, and inverted rule text;
  - 4-5-digit balances, and amounts 13-500;
  - renamed or extra accounts, and reordered JSON;
  - k = 4 chains.
- **#5 splits.** **5a** [R]: precondition representation, with balanced data and minimal pairs, rendered three ways
  (bare int / bit scratchpad / bitstring + applicable flag). **5b** [J]: trace-in-answer targets
  `{"cmp":..,"calc":..,"after":..}` with a filler-token control. 5b shares one trainer and one target format with
  BWM's RW2 legality scratchpad (factorized "compare, then answer"). That makes **#6 (legality checker) concrete**:
  teach the predicate explicitly in the answer, across RW2 legality, Blocksworld preconditions and RCWM
  prerequisite/branch.
- **New #5c [J]: correctness-based retention suite.** 8 code-graded categories × 75 items, a fresh sealed draw per
  confirmation; ~0.9 GPU-h once, then ~0.2 per arm. For BWM, the adapters are task-scoped and never merged, so the
  base's general skill is unchanged by construction. The suite still measures adapter-on behaviour before any
  release.
- **The 8-15 arm** becomes a "dissect T2" item [R]: LoRA confined to 8-15 vs 16-27, an α sweep, and SD/NAT controls.
- **#3 (beam with T2 as F) stays:** it is the direct planning test.
- **New CPU-only items** [R]: no-LLM learners at Titan's matched 629 ledger rows on the out-of-range strata; long-horizon
  (k = 8/16) and seed-consensus certificates.

**Added to "do not do"**
- Tuning on spent or reserved instruments (pilot TEST, t2_eval/supp, R14's 248, L1's sets).
- Latent single-pass composition (single-shot k4 exact = 0.00).
- Reporting tool tiers as model ability.

The turn order is unchanged, with #2 = out-of-range rule/trigger map and #5 = 5a/5b/5c.
