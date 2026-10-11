# PREREG RCX-E11 (DRAFT): do skills distilled from verified plans compress search on larger problems, and does a learned subgoal policy compose to unseen goal structures?

| | |
|---|---|
| Status | **DRAFT, not frozen.** Written 2026-10-11 and updated the same day with the coordinator's decisions on DESIGN Q1–Q7. Q8 was resolved by the coordinator in CX D-28: the random control excludes the mined macros' full keys, not bare step patterns. The coordinator freezes it after the smoke: copy it to `PREREG_E11.md` (status FROZEN), set its sha256 as `PREREG_SHA256` in `run_e11.py`, and log it in DECISIONS (CX D-nn) before any non-smoke run. |
| Outcomes known at drafting | **None for E11.** No E11 code has run on any E11 instance; there has been no smoke run and no search. Known: R-04 / R-07 (the A3 profile); the mining-source statistics in DESIGN.md §3–4 (custody checks and the segment-frequency table of the 211 teacher plans, which is training data); the composition of the M, R and S libraries from a dry run of the mining code on those plans (DESIGN §4.3). Under the current rule, S shares no macro or step signature with M, and its three L = 4 slots needed the fill tier. Also known: the unit-test results. |
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

- **H-b1.** A subgoal selector trained on exact optimal-plan labels from n ≤ 7 goals with at most 2 towers of height at
  most 4 chooses subgoal orders that need fewer actions than bottom-up ordering, when executed without search by a fixed
  achiever on n = 10–15 goals with more and/or taller towers. Its success is non-inferior.
- **H-b2 (descriptive).** Its label accuracy on small out-of-distribution structures (more or taller towers, n ≤ 7)
  stays close to in-distribution accuracy.

## 2. Arms

### E11a

Every arm is crossed with two heuristics and two algorithms:

- **Heuristics:**
  - A3 = `b4_hook.LearnedUnmet(runs/RCX-E07/main/models/gnn_s20261276.pt, hybrid_lambda=1.0)`: the B4 hook itself,
    with single-state 1-thread forwards and a fresh cache per instance. There is no batched variant;
  - `misplaced_blocks`.
- **Algorithms:** GBFS, and beam with width 4. Both use `macros.gbfs_macro` / `beam_macro`, which equal
  `search.gbfs` / `beam_search` when the library is empty.

| Arm | Action space |
|---|---|
| **P** | primitives only |
| **M** | primitives + the mined library: the top 8 by occurrences × (L − 1), with support ≥ 5 plans, accepted by validation (DESIGN §4.2–4.3) |
| **R** | primitives + 8 random sound macros matched to M by L and G, and by V where possible (tiers recorded); not M's keys |
| **S** | primitives + 8 macros mined and ranked like M from the same plans with each plan's actions shuffled (seed `20261256:shuffle`). It **excludes every candidate whose abstract step signature is in M** (whatever its goal annotation), so M ∩ S = ∅. It matches M's length mix: per M macro, the next most frequent accepted shuffled-plan macro, at the first tier that has one (same L with support ≥ 5; same L with support ≥ 1; any L with support ≥ 5; any L with support ≥ 1). The tier is recorded per macro. |

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
| **RF** (descriptive) | ready-first |
| **GC-GBFS** | primitive GBFS with h = unmet goal facts, budget 2,000 |

## 3. Data, instances and seeds

- **Teacher.**
  - `BW_DISTILL_r1/solutions.jsonl`, sha256 `fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c`,
    read-only and pinned (mining fails closed on a mismatch).
  - Only rows with `verified == true` are used (211). A plan that fails exact re-verification is excluded and counted.
  - Off-limits and never opened: `distill_rows.jsonl`, `mix_*.jsonl`, `fcache.jsonl`, adapters.
- **E11a test.** 120 fresh instances, seed 20261256: n ∈ {12, 14, 15} × 40.
  - Goals: `B.generate(full_goal_frac=1.0)`.
  - Exclusions: BWM recipe plus selection sets.
  - Disjoint by instance key from R-04 E08, R-04 E07 OOD test, and E07b TEST-B / VAL-B / A1.
  - Optimal lengths: A*, budget 300k.
- **E11b small sets, seed 20261257.**
  - train / val / test_id: the 24 S-structure partitions at n = 4–7, with 12 / 3 / 4 instances per class.
  - ood_small: the 10 T/H partitions at n = 5–7, 6 instances per class.
  - Hand-empty states are sampled (150 / 150 / 100 / 100 per instance) with exact label sets.
  - Instances are unique across sets; row-level exclusion as in E07.
- **E11b execution set, seed 20261258.** 128 fresh instances: strata S / T / H / TH × n ∈ {10, 12, 14, 15} × 8, with
  the partition uniform within the stratum. The **OOD pool** is T ∪ H ∪ TH (96 instances); S is the size-only control.
  Optimal lengths by A*, budget 300k.
- **Seeds.** Only the block 20261256–20261259 and values derived from it (accepted by the coordinator, Q4; recorded in
  DECISIONS at the freeze):

  | Seed | Uses |
  |---|---|
  | 20261256 | E11a instances. Derived: `20261256:random_macros` (R), `20261256:shuffle` (S), `20261256:mono_check`, `20261256:bootstrap`; unit tests only: `20261256:tests` |
  | 20261257 | E11b small instances: `20261257:train`, `:val`, `:test_id`, `:ood_small`; states `20261257:states:<split>` |
  | 20261258 | E11b execution instances `20261258:exec`; random order `20261258:random_order:<instance id>`; `20261258:bootstrap` |
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

### E11a secondary (descriptive)

- success at {50, 200, 500, 2000} for every arm × heuristic × algorithm, overall and per n;
- every M-vs-control McNemar;
- expansions (median; solved median);
- h_calls and prim_steps (solved means), and children per expansion;
- plan length and suboptimality against A* where covered;
- macro use in solved plans;
- success within h-call thresholds {1k, 5k, 20k, 100k} (the compute-matched view);
- library descriptors and validation counts.

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
- GC-GBFS success at {50, 200, 500, 2000};
- top-1 label accuracy on test_id and ood_small, overall, per stratum, per n and per class, for L-r0..r2 and for BU, RF
  and RND on the same states.

## 5. Decision rules (mechanical; implemented in `run_e11.py` `a_decision` / `b_decision`)

### E11a decision rule

**Validity gates.** Any failure makes the outcome **INVALID**; fix it, log a deviation and rerun into a new directory.

| Gate | Condition |
|---|---|
| G1 | \|M\| = \|R\| = \|S\| = 8; M and S share no key and no step signature; every library macro exhaustively accepted |
| G2 | no found-but-invalid plan in any run |
| G3 | zero symbolic-vs-replay mismatches during search |
| G4 | every monotonicity spot-check rerun equals its derived record |

| Outcome | Condition (at the primary cell unless stated) |
|---|---|
| **ADVANCE** | primary p < 0.05 with M-only > P-only discordants, **and** for both R and S: Holm-adjusted p < 0.05 and M − control ≥ +5 successes (of 120) |
| **REVISE** | primary p < 0.05 in M's favour, attribution not met (macros help, but not specifically mined ones) |
| **STOP** | primary p < 0.05 in P's favour, **or** M ≤ P in success count at both 200 and 2,000 expansions for GBFS with both heuristics |
| **INCONCLUSIVE** | otherwise |

### E11b decision rule

Data checks are validity gates: disjoint classes and instances, strata as assigned, non-empty label sets. Any failure
makes the outcome **INVALID**.

| Outcome | Condition (OOD pool) |
|---|---|
| **ADVANCE** | primary CI upper bound < 0 **and** point estimate ≤ log 0.95 (at least 5% fewer actions than BU), **and** L-r0 success ≥ BU success − 2, **and** the r1 and r2 point estimates are both < 0 |
| **STOP** | L-r0 vs RND point estimate ≥ 0 (no better than random order), **or** L-r0 success < 0.8 × BU success |
| **REVISE** | not ADVANCE or STOP, and L-r0 vs RND CI upper bound < 0 (the learned ordering transfers but does not beat BU) |
| **INCONCLUSIVE** | otherwise |

**Interpretation clause, fixed now.** RF is descriptive. If ADVANCE holds but L-r0 is more than 5% longer than RF (geo
ratio vs RF > 1.05), the report must say: "the learned order transfers compositionally but does not beat a ready-first
handcrafted rule".

## 6. Multiplicity

- There is one primary per sub-experiment, each at α = 0.05 (two-sided McNemar; 95% CI).
- E11a and E11b test different hypotheses and are decided separately. There is no correction across them, so the
  family-wise error over the two primaries is ≤ 0.10.
- E11a's attribution pair is Holm-adjusted and only grades ADVANCE vs REVISE.
- Every other comparison is descriptive; its p-values or CIs are reported but not used for claims.

## 7. What a positive result would NOT establish

### E11a

- **Compute savings.** Expansions are the budget unit, and macro children add heuristic calls and simulator steps. The
  h-call-matched table decides whether any saving survives.
- **That Rineya learned skills.** The library is an external specialist (tier B).
- **Transfer to BWM's LM forward model or beam planner.** Dynamics here are exact.
- **Domain-general skills.** The mined macros encode B3's own "dismantle, then build goal towers" strategy. They may not
  transfer to other planners' plans or domains.
- **Which aspect of order matters.** A gain over S shows that action order matters, not which aspect of it.

### E11b

- **Superiority over a well-designed handcrafted ordering rule** (see the RF clause).
- **Learned low-level skills.** The achiever is fixed and hand-written; only the ordering is learned.
- **No-search competence of an LM.**
- **Generality.** The test covers neither non-on-fact goals, other init distributions, nor other domains.
- **Near-optimal composed plans.** Optimal-plan labels do not imply this; suboptimality is reported separately.

### Limitations of the design (stated before any outcome)

- **Power was not simulated.** The sample sizes, 120 paired E11a instances and a 96-instance E11b OOD pool, were set
  without a pilot or power simulation (accepted by the coordinator, Q3). A null result at these sizes does not rule out
  a small effect.
- **One low-level skill.** E11b uses a single clear-to-table achiever (Q5). A GBFS single-subgoal achiever is future
  work, and results may depend on the achiever.
- **No literal AB/BA template holdout** (Q6, §2).
- **Weak S support at L = 4.** S's length-4 slots are filled from low-support shuffled segments (recorded tiers),
  because sound 4-step segments are rare after shuffling.

## 8. Compute estimate (CPU, single lane; the A3 hook runs single-threaded)

| Phase | Estimate |
|---|---|
| E11a `a_data` (120 instances + A*) | about 4–6 min |
| E11a `a_mine` (mining + exhaustive validation, at most about 0.4 s per pattern) | about 1–2 min |
| E11a `a_search` (1,920 runs at 2,000 expansions) | about 1.3–2 h expected, ≤ 3 h upper; dominated by A3 macro arms, where macro children multiply hook calls at about 1 ms each |
| E11b `b_data` (draws, signatures, exact labels, exec A*) | about 6–10 min |
| E11b `b_train` (3 seeds, about 43k rows, ≤ 20 min cap each) | about 10–20 min |
| E11b `b_eval` | about 5 min |
| **Total** | **about 2–3 h expected, ≤ 4 h upper** |
| Smoke | about 3–6 min |

RAM is under 1.5 GB (A3 cache ≤ 50k states per instance). Guard on, ≤ 4 threads, nothing else in the CPU lane.

## 9. Execution and freeze procedure

1. Q1–Q7 are resolved (DESIGN §9); resolve Q8 (whether R also excludes M's step signatures). Freeze by copying the
   draft to `PREREG_E11.md` (status FROZEN, with the date). Set `PREREG_SHA256` in `run_e11.py`, and log the CX D-nn
   decision with the seed allocation.
2. Run `python -B -m unittest tests.test_e11`.
3. Run the smoke in the CPU lane: `run_e11.py --smoke --threads 2`, which writes to `runs/RCX-E11/smoke`. Smoke
   numbers are plumbing only, never results.
4. Run the full design: `run_e11.py --threads 4`, which writes to `runs/RCX-E11/main` and is resumable. The report phase
   computes the decisions mechanically.
5. Code may be fixed after the freeze only without changing endpoints, arms, instances or decision rules. Every
   deviation is logged in DECISIONS.
