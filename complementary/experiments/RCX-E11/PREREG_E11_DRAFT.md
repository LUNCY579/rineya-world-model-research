# PREREG RCX-E11 (DRAFT): do macros mined from verified plans raise expansion-budgeted search success on larger problems, and does a learned subgoal policy compose to unseen goal structures?

| | |
|---|---|
| Status | **DRAFT, not frozen.** Written 2026-10-11 and updated the same day with the coordinator's decisions on DESIGN Q1–Q7. Q8 was resolved by the coordinator in CX D-28: the random control excludes the mined macros' full keys, not bare step patterns. Updated again on 2026-10-11 with the consolidated E11 review edits (review workflow wf_99b7f078-3bf) and the coordinator's settings: `exclude_noop_macros=True`, `b_label_target="achiever"`, descriptive arm G added, compute-matched thresholds {500, 1,000, 2,500, 5,000}. The coordinator freezes it after the smoke: copy it to `PREREG_E11.md` (status FROZEN), set its sha256 as `PREREG_SHA256` in `run_e11.py`, and log it in DECISIONS (CX D-nn) before any non-smoke run. |
| Outcomes known at drafting | **None for E11.** No E11 code has run on any E11 instance; there has been no smoke run and no search. Known: R-04 / R-07 (the A3 profile); the mining-source statistics in DESIGN.md §3–4 (custody checks and the segment-frequency table of the 211 teacher plans, which is training data); the composition of the M, R and S libraries from a dry run of the mining code on those plans (DESIGN §4.3). Under the current rule, S shares no macro or step signature with M, and its three L = 4 slots needed the fill tier; with the no-op rule (§2), S changes 4 macros and R changes 1, and S's three L = 4 slots and one L = 3 slot need the fill tier. Also known: the unit-test results (37 tests). Review-time design information, none on E11 instances or seeds: (a) a baselines-and-oracles pilot on scratch instances (`e11-review-scratch*`), with no learned selector. At n = 10–15: OPT_A/BU ≈ 0.78 on 72 T/H/TH instances (0.88 on S), RF/BU ≈ 0.85, RND/BU ≈ 1.47. At n = 7: a selector always picking from A(s) is 1–3% shorter than BU, RF 5%, OPT_A 10%. A(s) and A_A(s) are disjoint on 18% of 600 scratch rows. (b) Row overlap measured with `20261256:tests`: 100% / 99.6% / about 70% of val and test_id rows at n = 4 / 5 / 6 are renaming-equivalent to a training row. (c) Branching on 1,350 random-walk states at n = 12/14/15 (scratch seed): P has 4.6–6.2 actions per state; M adds 1.9–2.8 macro bindings; R's tier-1 macro has 2.5–6.2. (d) Accounting probes of macro search on scratch instances (blind GBFS at n = 12/15 with 150 expansions; budget-derivation reruns at n = 6–8). These measured accounting only. (e) A training-side library dry run with the no-op rule and annotation alignment. M: 11 of 14 literals established, 3 pending. R: 2 established, 2 pending, 2 destroyed, 8 unrelated. S: 6 established, 1 destroyed. |
| Track and tier | RSM-001 E11, CPU mechanism tier, exact dynamics, external specialist (tier B). No Rineya-core claim. |
| Design | `DESIGN.md` (same directory). Code: `rcx/bw/macros.py`, `rcx/bw/subgoal.py`, `run_e11.py`, `tests/test_e11.py`. |

## 1. Hypotheses

**E11a, macro reuse:**

- **H-a1.** Adding a library of K = 8 parameterised macros mined from BWM's 211 verified B3 plans to the action space
  raises GBFS success at a fixed expansion budget on fresh, larger instances (n = 12, 14, 15; the mining source has
  n = 6–10).
- **H-a2.** The gain comes from plan structure. It exceeds the gain from matched random sound macros (R) and from
  macros mined by the same pipeline from order-shuffled plans (S).

**E11b, compositional subgoal policy:**

- **H-b1.** A subgoal selector trained on exact achiever-level labels (DESIGN §5.2: the unmet facts whose ACHIEVE call
  begins a shortest sequence of ACHIEVE calls to the goal; `b_label_target="achiever"`) from n ≤ 7 goals with at most
  2 towers of height at most 4 chooses subgoal orders that need fewer actions than bottom-up ordering, when executed
  without search by the same fixed achiever on n = 10–15 goals with more and/or taller towers. Its success is within
  2 instances of BU's (a fixed count margin, not a statistical test).
- **H-b2 (descriptive).** Its label accuracy on small out-of-distribution structures (more or taller towers, n ≤ 7)
  stays close to its in-distribution accuracy on test_id rows unseen in train or val, compared per n.

## 2. Arms

### E11a

Every arm is crossed with two heuristics and two algorithms:

- **Heuristics:**
  - A3 = `b4_hook.LearnedUnmet(runs/RCX-E07/main/models/gnn_s20261276.pt, hybrid_lambda=1.0)`: the B4 hook itself,
    with single-state 1-thread forwards and a fresh hook (empty cache, zero counters) per search run, so h_calls
    equals A3 evaluations. There is no batched variant;
  - `misplaced_blocks`.
- **Algorithms:** GBFS, and beam with width 4. Both use `macros.gbfs_macro` / `beam_macro`, which equal
  `search.gbfs` / `beam_search` when the library is empty.

| Arm | Action space |
|---|---|
| **P** | primitives only |
| **M** | primitives + the mined library: the top 8 by occurrences × (L − 1), with support ≥ 5 plans, accepted by validation (DESIGN §4.2–4.3) |
| **R** | primitives + 8 random sound macros matched to M by L and G, and by V where possible (tiers recorded); not M's keys; no internal no-op pairs |
| **S** | primitives + 8 macros mined and ranked like M from the same plans with each plan's actions shuffled (seed `20261256:shuffle`). It **excludes every candidate whose abstract step signature is in M** (whatever its goal annotation), so M ∩ S = ∅. Candidates with an internal no-op (an adjacent inverse pair such as `pick-up ?x; put-down ?x`) are excluded from M, R and S (`exclude_noop_macros=True`): such a macro only generates end states of its reduced form, so a padded copy could carry an M pattern into S. It matches M's length mix: per M macro, the next most frequent accepted shuffled-plan macro, at the first tier that has one (same L with support ≥ 5; same L with support ≥ 1; any L with support ≥ 5; any L with support ≥ 1). The tier is recorded per macro. |
| **G** (descriptive) | primitives + `macros.goal_move_library()`: `pick-up ?0; stack ?0 ?1 \| on(?0,?1)` and `unstack ?0 ?1; stack ?0 ?2 \| on(?0,?2)`; not in the decision rule |

### E11b

Every arm runs without search except GC-GBFS. Every arm except GC-GBFS uses the fixed achiever ACHIEVE(on(x, y)).
Deterministic arms stop on a repeated decision state. All arms stop at 6·|goal| skill calls or 16n actions.

**Composition test.** E11's composition test is the goal-structure holdout: training goals have at most 2 towers of
height at most 4, and evaluation goals have more and/or taller towers. **The registry's literal AB/BA/conditional
template-level holdout is not run**, for three reasons:

- the skills here are parameterised over goal facts, not named templates;
- the order of achieving goal facts and towers, which is what AB vs BA means here, is exactly what the selector must
  choose on unseen structures;
- Blocksworld offers no named parent templates to hold out.

| Arm | Selector |
|---|---|
| **L-r0** (primary), L-r1, L-r2 | the learned SubgoalNet, seeds 20261259 / 816062654 / 250692302 |
| **BU** (primary comparator) | bottom-up by tower |
| **RND** | a fixed random priority per instance |
| **RF** | ready-first; descriptive except for one E11b ADVANCE condition (L-r0 vs RF point estimate ≤ 0, §5) |
| **GC-GBFS** | primitive GBFS with h = unmet goal facts, budget 2,000 |

## 3. Data, instances and seeds

- **Teacher.**
  - `BW_DISTILL_r1/solutions.jsonl`, sha256 `fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c`,
    read-only and pinned (mining fails closed on a mismatch).
  - Only rows whose `verified` is exactly JSON true are used (211; `exact_valid_for_analysis_only` is never used);
    mining stops unless there are 240 rows, 211 verified and one plan per problem id. A plan that fails exact
    re-verification is excluded and counted.
  - Off-limits and never opened: `distill_rows.jsonl`, `mix_*.jsonl`, `fcache.jsonl`, adapters.
- **E11a test.** 120 fresh instances, seed 20261256: n ∈ {12, 14, 15} × 40.
  - Goals: `B.generate(full_goal_frac=1.0)`.
  - Exclusions: BWM recipe plus selection sets.
  - Disjoint by instance key from R-04 E08, R-04 E07 OOD test, and E07b TEST-B / VAL-B / A1. Goal classes are
    disjoint from the A3 GNN's R-04 train/val classes (checked); a post-draw exclusion re-check must find 0.
  - Optimal lengths: A*, budget 300k.
- **E11b small sets, seed 20261257.**
  - train / val / test_id: the 24 S-structure partitions at n = 4–7, with 12 / 3 / 4 instances per class.
  - ood_small: the 10 T/H partitions at n = 5–7, 6 instances per class.
  - Hand-empty states are sampled (150 / 150 / 100 / 100 per instance) with two exact label sets: the achiever-level
    set A_A(s) (training target; primary label accuracy) and the optimal-plan set A(s) (descriptive).
  - Instances are unique across sets (renaming signature); BWM row-level exclusion as in E07. Rows are not separated
    by instance: instances of one goal class share their state space up to renaming. Every row carries `seen` (its
    `subgoal.canonical_row_key` occurs in an earlier split; order train, val, test_id, ood_small). Epoch selection
    uses val rows unseen in train. In-distribution accuracy uses test_id rows unseen in train or val; all-row accuracy
    is secondary.
- **E11b execution set, seed 20261258.** 128 fresh instances: strata S / T / H / TH × n ∈ {10, 12, 14, 15} × 8, with
  the partition uniform within the stratum. The **OOD pool** is T ∪ H ∪ TH (96 instances); S is the size-only control.
  Optimal lengths by A*, budget 300k, and the exact achiever-level optimum OPT_A (A* over ACHIEVE calls,
  `subgoal.achiever_search`, budget 200k; descriptive ceiling).
- **Seeds.** Only the block 20261256–20261259 and values derived from it (accepted by the coordinator, Q4; recorded in
  DECISIONS at the freeze):

  | Seed | Uses |
  |---|---|
  | 20261256 | E11a instances. Derived: `20261256:random_macros` (R), `20261256:shuffle` (S), `20261256:mono_check`, `20261256:bootstrap` (reserved; unused); unit tests only: `20261256:tests` |
  | 20261257 | E11b small instances: `20261257:train`, `:val`, `:test_id`, `:ood_small`; states `20261257:states:<split>` |
  | 20261258 | E11b execution instances `20261258:exec`; random order `20261258:random_order:<instance id>` (execution runs and label-accuracy rows); `20261258:bootstrap` |
  | 20261259 | selector r0 |
  | 816062654 | selector r1 = sha256("20261259:r1")[:4] mod (2³¹ − 1) |
  | 250692302 | selector r2 = sha256("20261259:r2")[:4] mod (2³¹ − 1) |

## 4. Endpoints

### E11a PRIMARY (one test)

GBFS with A3 at 200 expansions on all 120 instances: success of **M vs P**, exact two-sided McNemar on paired instances
at α = 0.05.

- Success = a plan found within budget and valid under `B.execute`.
- Smaller budgets are derived from the 2,000-expansion run (budget-monotone) and spot-checked.

### E11a attribution family (grades ADVANCE vs REVISE only)

At the primary cell, **M vs R** and **M vs S**, by exact McNemar. Holm-adjusted across the two.

### E11a compute-matched secondary (pre-declared; grades wording only)

- At A3/GBFS: M vs P success within 1,000 heuristic calls, exact two-sided McNemar, α = 0.05, derived from the
  2,000-expansion runs. The same test is reported at 1,000 simulator steps.
- It must be exact (no censored run in either arm); otherwise it is reported as not evaluable.
- It does not change ADVANCE, REVISE or STOP.
- Wording rule: "compress search" or "saves compute" is allowed only if this test favours M at p < 0.05. Otherwise a
  positive primary reads: "mined macros raise success per node expansion; no saving per heuristic evaluation is
  shown".

### E11a secondary (descriptive)

- success at {50, 200, 500, 2000} for every arm × heuristic × algorithm, overall and per n;
- every M-vs-control McNemar;
- expansions (median; solved median);
- h_calls and prim_steps (solved means), and children per expansion;
- plan length and suboptimality against A* where covered;
- macro use in solved plans;
- the compute-matched view: success within h-call and simulator-step thresholds {500, 1,000, 2,500, 5,000}, derived
  from the 2,000-expansion runs (success with h_calls or prim_steps ≤ t). A cell is exact for an arm only if none of
  its expansion-budget-stopped runs used ≤ t; otherwise the count is a lower bound, marked censored, and not tested;
- library descriptors, validation counts, per-library annotation alignment and internal no-op counts.

### E11b PRIMARY (one estimate with a CI)

On the OOD pool (96 instances), the mean over instances of log(len_{L-r0} / len_{BU}), where a failed run counts as
16n actions. The interval is a 95% percentile bootstrap over instances: 5,000 resamples, seed `20261258:bootstrap`,
`rcx.stats.cluster_bootstrap_diff`. exp(mean) is reported as the geometric-mean length ratio.

### E11b secondary (descriptive)

- success per arm and stratum;
- L-r0 vs RND and vs RF (same statistic);
- r1 and r2 point estimates;
- per-stratum geometric ratio vs BU (the composition gap: S vs T / H / TH);
- suboptimality against A* where covered;
- geometric length ratio vs OPT_A per arm and stratum; for L-r0 on the OOD pool, the fraction of BU's excess closed,
  (mean log len_BU − mean log len_L) / (mean log len_BU − mean log OPT_A);
- GC-GBFS success at {50, 200, 500, 2000};
- top-1 label accuracy against A_A(s) and against A(s), on all rows and on unseen rows (the in-distribution basis), on
  test_id and ood_small, overall, per stratum, per n and per class, for L-r0..r2 and for BU, RF and RND on the same
  states.

## 5. Decision rules (mechanical; implemented in `run_e11.py` `a_decision` / `b_decision`)

### E11a decision rule

**Validity gates.** Any failure makes the outcome **INVALID**; fix it, log a deviation and rerun into a new directory.

| Gate | Condition |
|---|---|
| G1 | \|M\| = \|R\| = \|S\| = 8; M and S share no key and no step signature; every library macro exhaustively accepted and free of internal no-op pairs |
| G2 | no found-but-invalid plan in any run |
| G3 | zero symbolic-vs-replay mismatches during search |
| G4 | every monotonicity spot-check rerun equals its derived record |

Rows are evaluated top to bottom; the first match is the outcome.

| Outcome | Condition (at the primary cell unless stated) |
|---|---|
| **ADVANCE** | primary p < 0.05 with M-only > P-only discordants, **and** for both R and S: Holm-adjusted p < 0.05 and M-only > control-only discordants |
| **REVISE** | primary p < 0.05 in M's favour, attribution not met (macros help, but not specifically mined ones) |
| **STOP** | primary p < 0.05 in P's favour, **or** M ≤ P in success count at both 200 and 2,000 expansions for GBFS with both heuristics |
| **INCONCLUSIVE** | otherwise |

**Interpretation clause (E11a), fixed now.** Arm G is descriptive. If ADVANCE holds but M − G < 5 successes at the
primary cell, the report must say: "the mined library's gain is matched by goal-gated single-move macros; it is not
specific to mined multi-step structure". The wording of any compute-saving claim is governed by the E11a
compute-matched secondary (§4).

### E11b decision rule

Data checks are validity gates, enforced by `b_data`, which writes no data on a failure: disjoint classes and
instances; strata recomputed from every goal equal the assigned strata; every ood_small goal is non-S; no ood_small row
shares a renaming class with an ID-split row; execution instances are disjoint by key from earlier RCX test/selection
sets and E11a; both label sets are non-empty and exact; OPT_A plans are valid and ≥ the A* optimum. Any failure makes
the outcome **INVALID**.

Rows are evaluated top to bottom; the first match is the outcome.

| Outcome | Condition (OOD pool) |
|---|---|
| **ADVANCE** | primary CI upper bound < 0 **and** point estimate ≤ log 0.95 (at least 5% fewer actions than BU), **and** L-r0 success ≥ BU success − 2, **and** the r1 and r2 point estimates are both < 0, **and** the L-r0 vs RND point estimate < 0, **and** the L-r0 vs RF point estimate ≤ 0 |
| **STOP** | L-r0 vs RND point estimate ≥ 0 (no better than random order), **or** L-r0 success < 0.8 × BU success |
| **REVISE** | not ADVANCE or STOP, and L-r0 vs RND CI upper bound < 0 (the learned ordering transfers but does not meet ADVANCE: it does not beat BU by the margin, or it is longer than RF) |
| **INCONCLUSIVE** | otherwise |

**Interpretation clause (E11b), fixed now.** RF enters the decision only through the ADVANCE condition on the L-r0 vs RF
point estimate; every other RF comparison is descriptive. If every other ADVANCE condition holds but L-r0 is longer
than RF (L-r0 vs RF point estimate > 0), the report must say: "the learned order transfers to unseen goal structures
and beats bottom-up ordering, but does not beat a ready-first handcrafted rule". The report also states the fraction
of BU's excess over OPT_A that L-r0 closes. ADVANCE supports "the learned subgoal order transfers to unseen goal
structures under a fixed achiever", not composition in RSM-001's sense.

## 6. Multiplicity

- There is one primary per sub-experiment, each at α = 0.05 (two-sided McNemar; 95% CI).
- E11a and E11b test different hypotheses and are decided separately. There is no correction across them, so the
  family-wise error over the two primaries is ≤ 0.10.
- E11a's attribution pair is Holm-adjusted and only grades ADVANCE vs REVISE. The attribution pair is tested only if
  the primary favours M (serial gatekeeping), so the family-wise error over the primary and the pair is ≤ 0.05.
- Every other comparison is descriptive; its p-values or CIs are reported but not used for claims.

## 7. What a positive result would NOT establish

### E11a

- **Compute savings.** Expansions are the budget unit, and macro children add heuristic calls and simulator steps. The
  compute-matched secondary (1,000 h-calls, A3/GBFS) decides whether a saving claim is allowed; the threshold table is
  descriptive and censored where the 2,000-expansion cap binds.
- **That the gain needs mined multi-step structure rather than goal-aligned gating.** Mined annotations are
  progress-aligned (11 of 14 literals established, 3 pending), and primitives are never gated. R's annotations are
  random, so M vs R mixes pattern and alignment; M vs S keeps the annotation rule fixed. Per-library alignment is
  reported.
- **That Rineya learned skills.** The library is an external specialist (tier B).
- **Transfer to BWM's LM forward model or beam planner.** Dynamics here are exact.
- **Domain-general skills.** The mined macros encode B3's own "dismantle, then build goal towers" strategy. They may not
  transfer to other planners' plans or domains.
- **Which aspect of order matters.** A gain over S shows that action order matters, not which aspect of it.

### E11b

- **Superiority over a well-designed handcrafted ordering rule.** ADVANCE requires only that L-r0's point estimate is
  not longer than RF's; no test against RF is made (see the RF clause).
- **Learned low-level skills.** The achiever is fixed and hand-written; only the ordering is learned.
- **No-search competence of an LM.**
- **Generality.** The test covers neither non-on-fact goals, other init distributions, nor other domains.
- **Near-optimal primitive plans.** Achiever-level labels optimise composed length under the fixed achiever, whose
  optimum is itself above h* (pilot: about 1.05 × h* at n = 7); suboptimality against A* is reported separately.
- **Composition in RSM-001's sense** (no sequence-memorization control).

### Limitations of the design (stated before any outcome)

- **Power is low.** The sample sizes, 120 paired E11a instances and a 96-instance E11b OOD pool, were set without a
  pilot (accepted by the coordinator, Q3). The review's power sketch (exact binomial, anchored on R-07 for E11a):
  - E11a: the effect detectable with 80% power is about +15 to +20 of 120. Joint ADVANCE power is 0.1–0.2 if R and S
    each recover half of a +20 effect, and about 0.7 if they recover none.
  - E11b: ADVANCE power is 0.23–0.48 at a true −5% (depending on the SD of the per-instance log ratio, 0.15–0.40),
    and 0.40–0.89 at −7%.
  - Each extra L-r0 failure where BU succeeds adds about +0.02 to the mean log ratio.
  - INCONCLUSIVE or STOP does not rule out an effect on RSM-001's +5-point scale.
- **One low-level skill.** E11b uses a single clear-to-table achiever (Q5). A GBFS single-subgoal achiever is future
  work, and results may depend on the achiever.
- **No literal AB/BA template holdout** (Q6, §2).
- **Weak S support at L = 4 and one L = 3 slot (no-op rule).** These S slots are filled from low-support shuffled
  segments (recorded tiers), because sound segments of that length are rare after shuffling once internal no-op
  pairs are excluded.
- **S is not a decision-count control, and the selector has limited depth.**
  - S goals have at most 6 on-facts at n = 10–15, while the out-of-distribution goals have up to 14, so the strata
    also differ in the number of subgoal decisions.
  - The 4-round GNN selector cannot see the full height of the tallest out-of-distribution towers.
  - A failure on the tallest towers may therefore reflect receptive field rather than composition.

## 8. Compute estimate (CPU, single lane; the A3 hook runs single-threaded)

| Phase | Estimate |
|---|---|
| E11a `a_data` (120 instances + A*) | about 4–6 min |
| E11a `a_mine` (mining + exhaustive validation, at most about 0.4 s per pattern) | about 1–2 min |
| E11a `a_search` (1,920 runs at 2,000 expansions for P, M, R, S) | about 1.3–2 h expected, ≤ 3 h upper; dominated by A3 macro arms, where macro children multiply hook calls at about 1 ms each; per-run A3 hooks (no cross-arm cache reuse) [+ arm G: +480 runs]; re-estimated from the smoke |
| E11b `b_data` (draws, signatures, both exact label sets, exec A* and OPT_A) | about 8–12 min |
| E11b `b_train` (3 seeds, about 43k rows, ≤ 20 min cap each) | about 10–20 min |
| E11b `b_eval` | about 5 min |
| **Total** | **about 2–3 h expected, ≤ 4 h upper** |
| Smoke | about 3–6 min |

RAM is under 1.5 GB (A3 cache ≤ 50k states per run). Guard on, ≤ 4 threads, nothing else in the CPU lane.

## 9. Execution and freeze procedure

1. Q1–Q8 are resolved (DESIGN §9; Q8 in CX D-28: R excludes M's full keys, not bare step patterns). Set
   `exclude_noop_macros` and `b_label_target` as frozen, then run the smoke (steps 2–3). After the smoke, freeze by
   copying the draft to `PREREG_E11.md` (status FROZEN, with the date). Set `PREREG_SHA256` in `run_e11.py`, and log
   the CX D-nn decision with the seed allocation.
2. Run `python -B -m unittest tests.test_e11`.
3. Run the smoke in the CPU lane: `run_e11.py --smoke --threads 2`, which writes to `runs/RCX-E11/smoke`. Smoke
   numbers are plumbing only, never results.
4. Run the full design: `run_e11.py --threads 4`, which writes to `runs/RCX-E11/main` and is resumable. The report phase
   computes the decisions mechanically.
5. Code may be fixed after the freeze only without changing endpoints, arms, instances or decision rules. Every
   deviation is logged in DECISIONS.
