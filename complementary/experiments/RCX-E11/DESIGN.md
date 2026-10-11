# RCX-E11: reusable skills for Blocksworld planning (RSM-001; CPU mechanism tier; tier B specialist)

**Status:** DRAFT design, written 2026-10-11 and updated the same day with the coordinator's decisions on Q1–Q7 (§9).
Nothing has run: no smoke run, no full run, no outcome.

- Prereg draft: `PREREG_E11_DRAFT.md`. It is not frozen.
- Runner: `run_e11.py`. A full run refuses to start until a frozen `PREREG_E11.md` exists and its sha256 is pinned in
  the runner.
- Code:
  - `rcx/bw/macros.py`
  - `rcx/bw/subgoal.py`
  - fast tests in `tests/test_e11.py`: 13 tests, about 0.4 s, no torch.

## 1. Question

RSM-001 asks whether search experience can be turned into reusable skills that do more than memorise complete plans.
E11 tests two parts of that question at the CPU mechanism tier, with exact dynamics:

- **E11a: reuse of mined skills on larger problems.** We mine parameterised macro-operators (open-loop options of 2–4
  primitive actions) from BWM's 211 verified B3 plans. Does adding them to the action space compress heuristic search
  on fresh, larger instances (n = 12, 14, 15; the mining source has n = 6–10)? We test this against two controls:
  - sound macros of matched size that were not mined;
  - macros mined from the same plans with their action order shuffled.

  The controls test whether the gain comes from *plan structure* rather than from any multi-step jump.
- **E11b: composition to unseen goal structures.** A subgoal selector chooses which goal fact a fixed achiever skill
  should establish next. It is trained on exact optimal-plan labels from small problems with at most 2 goal towers of
  height at most 4. Does it compose those skills, with no search, on goals with **more** towers and/or **taller**
  towers than any training goal (n = 10–15)?

**Mapping to the registry.** E11 is "novel option composition". E11b's held-out goal structures are the composition
test:

- more towers are parallel compositions of tower-building skills;
- taller towers are longer serial compositions;
- the selector's choice depends on the state, which is the conditional part.

**The registry's literal AB/BA/conditional template-level holdout is not run** (coordinator decision on Q6). The
goal-structure holdout is E11's composition test, for three reasons:

- In this design the skills are parameterised over goal facts, not named templates A and B. "AB vs BA" corresponds to
  the order in which goal facts or towers are achieved, and that order is exactly what the selector must choose on
  unseen structures.
- A literal template holdout needs a skill vocabulary with named parent templates, which RCX's Blocksworld setting
  does not have.
- The structure holdout is checkable with exact labels, both at n ≤ 7 (label accuracy) and at n = 10–15 (execution).

**Mapping to RSM-001's required controls:**

| RSM-001 control | E11 counterpart |
|---|---|
| "same traces without option boundaries" | arm P (primitives only) |
| "random boundaries" | arms R and S |
| "exact shortest-path teacher where affordable" | A* optima; E11b's exact labels |
| "no-search student evaluation" | E11b execution |
| "deduplicate at environment and parent level" | instance-level dedup and splits |

**Tier.** A macro library or a selector is an external specialist (tier B). Nothing here is a Rineya-core claim.

## 2. Why this is complementary, and the gate

BWM owns E10 (its B3 search distillation). RCX reads B3 *outputs* only after BWM opened the gate (COLLABORATION.md,
2026-10-10 about 20:30; `runs/BW_B3_EVAL/B3_REPORT.json` exists).

E11 adds what B3 does not have:

- explicit, symbolically validated skills (initiation condition, binding, policy, termination);
- a test of whether those skills transfer as search operators;
- a compositional-generalisation test of a subgoal policy.

## 3. Data custody

**Teacher plans.** The only file read is `Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_DISTILL_r1/solutions.jsonl`.

- **sha256:** `fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c`, read 2026-10-11. It is pinned in
  `run_e11.py`, and mining fails closed on a mismatch.
- **What is used:** rows with `verified == true`, 211 of 240.
  - Each plan is re-verified with `B.execute`, the exact simulator, and its `f_states` are compared with the exact
    trajectory.
  - At drafting this was checked once, read-only: 211/211 plans are exactly valid, 211/211 F-trajectories equal the
    exact ones, all goals are on-facts only, and plan lengths run from 2 to 58 (mean 20.2, 4,254 actions in total).
  - `DISTILL_META.json` is hashed, not otherwise used.
- **Never opened:** `distill_rows.jsonl`, `mix_v2c.jsonl`, `mix_v3.jsonl`, `fcache.jsonl` and the adapters.
- **Never evaluated on:** b3fresh, mirror, hard-110, PB-600, seeds_fresh, val_large, val_v2, and the 240 distill
  problems themselves. Their instances are used only through `EXCLUSIONS.json` / `excl.with_selection_sets` to build
  exclusions, as in E07/E07b.
- RCX writes nothing in the BWM bundle. BWM code is imported pinned and bytecode-free (`bwm_import`), and a bytecode
  check runs at the end of every invocation.

**A3 heuristic.** `runs/RCX-E07/main/models/gnn_s20261276.pt` (R-04 GNN seed 20261276), read-only, wrapped as
`b4_hook.LearnedUnmet(hybrid_lambda=1.0)`. This is the R-07 A3 key, the RCX module for B4 arm (a). The hook itself is
used, with single-state forwards at 1 thread; there is no batched variant, because exact parity with B4 matters more than
about 1 h of CPU (coordinator decision on Q2). The sha256 of the model and of its sidecar are recorded in
`inputs.json`, and `hook.describe()` is written to `a/a3_hook.json`.

**Evaluation instances.** All are fresh and drawn with E11 seeds. They pass BWM's exclusion recipe plus the selection
sets.

- E11a instances are also disjoint, by instance key, from every earlier RCX test or selection set at n ≥ 8:
  - R-04 E08 and the E07 OOD test;
  - E07b TEST-B, VAL-B and A1.
- E11b instances are unique across all E11b splits. Small instances use the renaming signature; large ones use a raw
  key.

## 4. E11a: macro discovery and reuse

### 4.1 Representation (`rcx/bw/macros.py`)

**What a macro is.** A macro is a pair (steps, goal_on).

- `steps` is a sequence of 2–4 primitive patterns over variables numbered by first appearance, for example
  `pick-up ?0; stack ?0 ?1`. Distinct variables bind distinct blocks.
- `goal_on` is a **static initiation condition**: pairs (i, j) such that `on(?i, ?j)` must be a goal fact.

**Preconditions and effects** come from STRIPS regression over the 4-op schemas, which equal `bwmn/blocks.py`; a test
checks them against `B.step` on every ground action of every state for n ≤ 4.

- A step precondition destroyed by an earlier step makes the macro *inconsistent*.
- Effects compose to (s − dele) ∪ add.

**Initiation set.** I(s, goal) is the set of injective bindings with pre ⊆ s and goal_on ⊆ goal. It is computed as a
small join over the state's facts. Policy = the ground sequence; termination = after its last step.

**Why goal annotations.** Mining records every goal on-relation among a segment's blocks. Without this,
`pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0` would have about T³ groundings per state (T = number of clear
blocks, up to 15). With the goal relation it has at most about n, and it becomes a goal-directed skill ("build two
levels of a goal tower").

### 4.2 Validation (the soundness gate)

Every macro used anywhere passes `validate`, a bounded exhaustive check against `B.step`. It runs on **every** valid
state of n_check = min(V + 2, 6) blocks (at least 3) and checks:

1. **Consistent:** the regression succeeded.
2. **Sound:** every binding satisfying `pre` replays step by step with `B.applicable`/`B.step` and ends exactly in
   (s − dele) ∪ add.
3. **Complete:** every binding under which the steps are executable satisfies `pre`. Such bindings are found
   independently, by walking `B.step`.
4. **Non-vacuous:** at least one execution exists. This rejects mutex preconditions, such as two held blocks or two
   blocks on one block, which regression cannot see.
5. **Non-identity:** some execution changes the state. This rejects `pick-up ?0; put-down ?0` and
   `stack ?0 ?1; unstack ?0 ?1`; the B3 plans contain the latter 25 times.

**Why soundness holds at every n.** Soundness follows from STRIPS composition once the schemas equal `B.step`. The
exhaustive check is an independent guard against regression or join bugs.

**Speed-up.** A cheap mutex filter on `pre` rejects vacuous patterns before enumeration. Its rejections are marked
`shortcut`. Every *accepted* macro is always exhaustively checked.

**Cost.** Measured: 0.4 s for a 4-variable macro at n_check = 6.

### 4.3 Libraries (K = 8 each)

**M (mined).**

- Scan every contiguous segment of length 2, 3 or 4 of the 211 plans, at all offsets.
- Abstract each segment (variables by first appearance, plus goal_on). Drop segments with more than 4 variables.
- Count occurrences and support (the number of distinct plans).
- Score = occurrences × (L − 1): primitive steps saved, summed over uses.
- Take the top K by score (ties: support, then key) with support ≥ 5 that pass validation.

The mining source alone, computed at drafting, predicts this library (it is not a result):

1. `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)`: 598 occurrences, 195 plans
2. `stack ?0 ?1; pick-up ?2; stack ?2 ?0 | …`
3. `pick-up ?0; stack ?0 ?1; pick-up ?2 | …`
4. `pick-up ?0; stack ?0 ?1 | on(?0,?1)`: 1,139 occurrences
5. `stack ?0 ?1; pick-up ?2; stack ?2 ?0; pick-up ?3 | 3 goal literals`
6. `stack ?0 ?1; pick-up ?2 | …`
7. `unstack ?0 ?1; put-down ?0 | -`
8. `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?3 | on(?0,?1),on(?2,?3)`

These skills encode B3's "unstack to the table, then build goal towers bottom-up" strategy.

**Dry run of the mining code at drafting (training-side only).** The mining code was run once on the 211 plans with
n_check_max = 4, to check the pipeline; it took 1.7 s. It did not touch evaluation instances or run any search.

- **M:** exactly the 8 macros above; no candidate was rejected.
- **R:** 7 macros drawn at tier 0 and 1 at tier 1 (`pick-up ?0; stack ?0 ?1 | -`, filling the (L = 2, V = 2, G = 0)
  slot).
- **S, first rule (superseded):** the plain pipeline shared 2 macros with M (`pick-up ?0; stack ?0 ?1 | on(?0,?1)` and
  `stack ?0 ?1; pick-up ?2 | on(?0,?1),on(?2,?0)`), because short plans keep adjacent pairs under shuffling. This led to
  the Q7 decision.
- **S, current rule (re-run, 0.5 s):**
  - It shares no key and no step signature with M; 2 candidates carrying M's signatures were skipped.
  - Its length mix equals M's: 3 macros of L = 4, 2 of L = 3, 3 of L = 2.
  - The L = 2 and L = 3 slots come from support ≥ 5 (tier 0).
  - The three L = 4 slots need the fill tier (support 1–2): shuffled plans rarely contain sound 4-step segments. Two of
    the three contain no-op fragments, such as `put-down ?0; pick-up ?0` inside a longer macro.
- 46 step patterns were checked: 17 accepted, 11 inconsistent, 15 vacuous (all caught by the mutex shortcut) and 3
  identity.

The full run validates with n_check up to 6, and its libraries are recorded in `a/libraries.json`.

**R (random sound; matched).** For each M macro in order, draw an accepted macro with the same length L, the same
number of goal literals G and, where possible, the same number of variables V. The draw is uniform over the
cheap-filtered pattern universe, in seeded order, taking the first pattern that validation accepts. goal_on is a random
consistent goal fragment of size G. Tiers, used in order:

| Tier | Constraint |
|---|---|
| 0 | same (L, V, G); step pattern not in M, not yet drawn |
| 1 | same (L, V, G); M's step pattern allowed with a different goal_on (see Q8) |
| 2 | same (L, G), any V ≤ 4 |

The tiers exist because small classes are tiny. For L = 2 and V = 2 only three sound, non-identity patterns exist, and
M uses two of them. The tier used is recorded per macro. R's full keys never equal M's.

**S (shuffled-plan mining; coordinator decision on Q7).** Shuffle every plan's actions (seeded), then mine and rank
exactly as for M (`macros.select_matched_library`), with two changes:

- **Disjointness.** Every candidate whose *abstract step signature* (the variable-abstracted action sequence) is in M
  is excluded, whatever its goal annotation. S therefore shares no macro and no action pattern with M; this is stricter
  than excluding identical keys.
- **Length matching.** S is matched to M's length mix. For each M macro in order, take the next most frequent accepted
  shuffled-plan macro, using the first tier that has one:

  | Tier | Constraint |
  |---|---|
  | 0 | same L, support ≥ 5 |
  | 1 | same L, support ≥ 1 |
  | 2 | any L, support ≥ 5 |
  | 3 | any L, support ≥ 1 |

  The tier is recorded per macro. Within one length, score order equals frequency order.

S keeps the plans' action vocabulary and frequencies but destroys their sequential structure. Gate G1 requires
\|S\| = 8 and M ∩ S = ∅ (by key and by step signature); a unit test asserts the disjointness.

### 4.4 Search

`gbfs_macro` and `beam_macro` (width 4) are exactly `search.gbfs` and `search.beam_search` with one addition.

- **Unchanged:** successor order, global duplicate detection, the goal test at generation time, ties, the budget check
  before each expansion, and plan validation by `B.execute`. Tests show the empty-library case reproduces `search.gbfs`
  and `search.beam_search` exactly.
- **Added:** after the primitive children of an expanded node come the end states of every macro binding, in library
  order with sorted bindings. Each is replayed with `B.step`, and the symbolic effect is cross-checked; mismatches are
  counted as gate G3.
- **Cost accounting:**
  - One expansion = one popped node, which generates all its primitive and macro children. A macro edge costs the same
    single expansion as a primitive edge.
  - `prim_steps` counts `B.step` calls during search, and `h_calls` counts heuristic evaluations. Both are reported as
    compute-side cost.
  - If an intermediate macro state satisfies the goal, the plan stops there. Plan length is always counted in
    primitives.

### 4.5 Arms, instances and budgets

**Arms:** P (primitives), M, R, S.

**Heuristics:**

- **A3:** `LearnedUnmet(gnn_s20261276, hybrid_lambda=1.0)`, the hook itself, with single-state 1-thread forwards and a
  fresh cache per instance. There is no batched variant (Q2).
- `misplaced_blocks`.

**Algorithms:** GBFS, and beam with width 4.

**Instances:** 120, seed 20261256, n ∈ {12, 14, 15} × 40.

- Full on-fact goals: `B.generate(full_goal_frac=1.0)`, as in E07b.
- Optimal lengths by A* with the admissible misplaced bound and a budget of 300k expansions. E07b's coverage at
  n = 14/15 was 20/25 and 16/25.

**Budgets:** {50, 200, 500, 2000} expansions. One run per (instance, arm, heuristic, algorithm) at 2,000; smaller
budgets are derived (budget-monotone, as in E07b) and spot-checked by direct reruns.

**Metrics:** success (a plan valid under `B.execute`), expansions, plan length in primitives, suboptimality against A*
where covered, h_calls, prim_steps, children per expansion, macro uses in solved plans, and success within heuristic-call
thresholds. The last is the compute-matched view.

## 5. E11b: compositional subgoal policy

### 5.1 Skill, selector and execution (`rcx/bw/subgoal.py`)

**The skill.** One fixed, parameterised skill per goal fact: ACHIEVE(on(x, y)). From a hand-empty state:

1. move every block above x to the table, top first;
2. do the same for y;
3. pick up or unstack x, and stack it on y.

It always succeeds, ends hand-empty, and may break goal facts achieved earlier. This single achiever is kept
(coordinator decision on Q5). **Future work, not run:** a GBFS single-subgoal achiever as a second low-level skill.

**The selector** picks the next unmet goal fact. Execution alternates selector and skill until the goal holds. It uses
no search, with these stops:

- a cycle stop for deterministic selectors: a repeated decision state means an infinite loop;
- at most 6 × |goal| skill calls;
- at most 16 n primitive actions.

### 5.2 Labels (exact)

For a hand-empty, non-goal state s with unmet goal facts U(s), the label set A(s) is the set of facts in U(s) that some
optimal plan from s achieves (by its `stack x y` step) before any other fact of U(s).

- A(s) is computed by a DFS over the optimal-move DAG of `space.GoalLabels`.
- It equals a brute-force enumeration of all optimal plans on a 4-block instance (test).
- Training uses the set-marginal likelihood −log Σ_{f ∈ A(s)} softmax(logits)_f.
- Accuracy means top-1 ∈ A(s).

### 5.3 Model

`SubgoalNet` uses RelGNN-style typed message passing over `encode.graph(state, goal)`: hidden 48, 4 rounds, sum
aggregation, LayerNorm residual. The logit of candidate on(x, y) is MLP([h_x, h_y, h_x ⊙ h_y, graph mean]).

- Training: AdamW, learning rate 2e-3, batch 64, 10 epochs, with a 20-minute time cap per seed.
- Selection: the best epoch on validation top-1.
- Seeds: r0, plus replicates r1 and r2 (section 6).

### 5.4 Goal structures and data

A goal's structure is its multiset of tower heights (singletons included). Let k be the number of towers of height ≥ 2
and hmax the tallest tower.

| Stratum | Rule | Role |
|---|---|---|
| S | k ≤ 2 and hmax ≤ 4 | the training structures |
| T | k ≥ 3 and hmax ≤ 4 | more towers |
| H | k ≤ 2 and hmax ≥ 5 | taller towers |
| TH | k ≥ 3 and hmax ≥ 5 | both |

**Small sets (seed 20261257; derived string seeds per split).** All are fresh. Goals are built per partition with a
random block assignment; the init comes from BWM's `random_towers` (p_stack 0.6). The usual exclusions apply, and
instances are unique across splits.

| Set | Contents | States per instance |
|---|---|---|
| train / val / test_id | all 24 S partitions of n = 4–7; 12 / 3 / 4 instances per class | 150 / 150 / 100 hand-empty states, sampled uniformly, init included, row-level exclusion |
| ood_small | the 10 T/H partitions possible at n ≤ 7: (2,2,2)@6, (3,2,2)@7, (2,2,2,1)@7, (5)@5, (6)@6, (5,1)@6, (7)@7, (6,1)@7, (5,2)@7, (5,1,1)@7; 6 instances each | 100 |

**Execution set (seed 20261258).** 4 strata × n ∈ {10, 12, 14, 15} × 8 = 128 instances.

- The partition is drawn uniformly among the stratum's partitions at n.
- A* gives optimal lengths with a budget of 300k.
- The OOD pool T ∪ H ∪ TH has 96 instances. Stratum S at n = 10–15 is the size-only control.

### 5.5 Arms

| Arm | Description |
|---|---|
| learned r0 / r1 / r2 | the trained selector |
| BU | bottom-up by tower: towers by (height desc, bottom name); the lowest unmet fact of the first unfinished tower |
| RND | a fixed random priority over goal facts per instance |
| RF | ready-first, descriptive: a fact whose support is in final position, preferring clear x and y; else BU |
| GC-GBFS | primitive GBFS with h = unmet goal facts (BWM's ranking), budget 2,000; smaller budgets derived |

**Metrics:**

- success;
- plan length;
- suboptimality against A* where covered;
- length ratio against BU per stratum;
- label accuracy on test_id and ood_small, for the learned selectors and for BU, RF and RND on the same states.

## 6. Seeds (RCX E11 block 20261256–20261259; no other seed is used)

| Seed | Use |
|---|---|
| 20261256 | E11a test instances (`data.fresh_instances`, guard retargeted to the E11 block as E07b did). Derived string seeds: `:random_macros` (R), `:shuffle` (S), `:mono_check`, `:bootstrap`. Unit tests use `20261256:tests` only. |
| 20261257 | E11b small instances: `:train`, `:val`, `:test_id`, `:ood_small`, and `:states:<split>` |
| 20261258 | E11b execution instances: `:exec`; random order `:random_order:<instance id>`; `:bootstrap` |
| 20261259 | E11b selector r0 = 20261259. Replicates r1 = 816062654 and r2 = 250692302 are derived as sha256("20261259:r<k>")[:4] mod (2³¹ − 1), because torch needs integer seeds (accepted, Q4). |

The coordinator records these in DECISIONS.md at the freeze; no entry has been made yet.

## 7. Implementation map

**`rcx/bw/macros.py`:**

- `schema`, `regress`, `abstract`, `Macro`: bindings, ground, apply_symbolic, replay, JSON round trip;
- `validate`, `ValidationCache`, `pre_mutex`;
- `mine`, `rank`, `select_library` (M), `select_matched_library` (S: length-matched, M-disjoint),
  `shuffled_plans`, `pattern_universe`, `random_library` (R);
- `gbfs_macro`, `beam_macro`.

**`rcx/bw/subgoal.py`:**

- `goal_towers`, `structure`, `stratum_of`, `partitions`, `partitions_in`, `make_goal`, `make_instance`;
- `label_set`, `handempty_ids`;
- `achieve`, `execute`;
- `done_blocks`, `BottomUp`, `ReadyFirst`, `RandomOrder`;
- the lazy-torch `SubgoalNet` with `encode_row`, `collate`, `set_nll`, `fit`, `predict_choice`, `LearnedSelector`.

**`experiments/RCX-E11/run_e11.py`:**

- **Phases:** `a_data`, `a_mine`, `a_search`, `b_data`, `b_train`, `b_eval`, `report`. `--smoke` uses a tiny config.
- **Resumability:** each phase is skipped when its outputs exist, and one JSONL record is written per run, with torn
  last lines repaired.
- **Guards:**
  - ThermalGuard courtesy;
  - ≤ 4 threads;
  - refusal to write into protected trees (BWM bundle, `../bwm`, `../rcwm`, `runs/RCX-E07`, `runs/RCX-E01*`,
    `experiments/`).
- **Provenance:**
  - `config.json`, frozen;
  - `inputs.json`: sha256 of solutions.jsonl, DISTILL_META, the A3 model and sidecar, EXCLUSIONS.json, BWM pins and the
    planner pin;
  - `code_shas.json`, appended at every start; a full run aborts if code changed mid-run;
  - the prereg draft and frozen sha;
  - A3 hook provenance (`hook.describe()`) in `a/a3_hook.json`.
- **The report phase** computes every endpoint and the decision mechanically, as drafted.

**`tests/test_e11.py`:**

- schema = `B.step`;
- regression hand cases;
- sound macros accepted;
- unsound and useless macros rejected: inconsistent, self-pair, wrong claimed effect, too-weak or too-strong claimed
  precondition, vacuous, identity;
- the bindings join equals brute force;
- macro application equals primitive replay;
- mining counts;
- library selection and the random control;
- the S library is disjoint from M (key and step signature), length-matched, with tiers recorded;
- an empty library equals `search.gbfs` / `search.beam_search`;
- macro-search plans are valid;
- 4-block label hand cases;
- labels equal brute force over optimal plans;
- achiever plus baseline execution from every 5-block state;
- strata and partitions.

## 8. Resolved ambiguities (design choices; listed for review)

1. **"Each macro application = one expansion"** is read as: a node expansion generates all primitive *and* macro
   children, so reaching a state through a macro costs one expansion of its parent. The alternatives (per-child or
   per-primitive accounting) appear as the reported `h_calls` and `prim_steps`.
2. **Goal-annotated macros** (§4.1) are the parameterisation. Without them, macros branch at about T³ and the A3 cost
   explodes.
3. **R matches (L, G) exactly and V where the pattern universe allows** (the tiers). Exact (L, V) matching is infeasible
   for L = 2.
4. **S uses M's mining and ranking, excludes M's step signatures and matches M's length mix** (Q7, §4.3). Its
   (V, G) mix is not forced.
5. **E11b's skill is a deterministic clear-to-table achiever,** not GBFS on a single subgoal. It is total, cheap and
   exactly specified (Q5; the GBFS achiever is future work).
6. **E11b's primary is a plan-length ratio, not success.** With a total achiever, BU nearly always succeeds, so success
   alone cannot separate selectors. Length needs no A* optimum, which is incomplete at n = 14–15. Failures are scored as
   the 16 n cap.

## 9. Coordinator decisions (2026-10-11) and remaining open item

| Question | Decision |
|---|---|
| Q1. E11a primary | **Accepted:** GBFS with the A3 hook at 200 expansions; M vs P by exact McNemar; Holm-graded ADVANCE / REVISE against R and S. |
| Q2. A3 implementation | **The faithful per-state hook only.** The batched variant was removed from the runner. |
| Q3. Sample size | **Accepted:** 120 (E11a) and 96 (E11b OOD pool). The prereg states that power was not simulated. |
| Q4. Derived seeds | **Accepted:** r1 = 816062654, r2 = 250692302, listed in the prereg. Recorded in DECISIONS at the freeze. |
| Q5. Achiever | **The single clear-to-table achiever is kept.** A GBFS single-subgoal achiever is future work. |
| Q6. AB/BA holdout | **The goal-structure holdout is E11's composition test.** The literal AB/BA holdout is not run (§1). |
| Q7. S/M overlap | **S excludes every M step signature and matches M's length mix,** with tiers recorded (§4.3). A unit test asserts the disjointness. |

**Q8 (open, raised by this update).** R's tier 1 may reuse one of M's step signatures with a *different* goal
annotation. R's full keys never equal M's.

- In the dry run this happened once: `pick-up ?0; stack ?0 ?1 | -`, filling the (L = 2, V = 2, G = 0) slot. Its
  initiation set is a superset of M's `pick-up ?0; stack ?0 ?1 | on(?0,?1)`.
- For the same reason as Q7, the coordinator may want R to exclude M's step signatures too. That means dropping tier 1,
  so the slot would be filled at tier 2 (same L and G, any V); it is a one-line change in `random_library`.
- The draft leaves R as it is until decided.
