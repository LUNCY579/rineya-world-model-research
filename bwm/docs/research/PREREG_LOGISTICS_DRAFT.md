<!-- DRAFT produced 2026-10-10 by a read-only scoping workflow (3 scouts, 2 designers, 1 synthesis); NOT frozen. Items marked [VERIFY] must be checked before any freeze. BWM decisions on the open questions: DECISIONS D-67. -->

# Addendum LG (item 9): out-of-size transfer of the learned-F + imagined-search recipe to PlanBench Logistics

**Status: DRAFT. Not frozen, not approved by the user.** Proposed file name: `PREREG_LOGISTICS.md`.

Written on 2026-10-10 (evening, CDT) from read-only CPU scoping. When it was written:
- no Logistics generator, data row, adapter, planner run or model output existed;
- no BWM model had seen any Logistics instance.

Numbers marked **[VERIFY]** rest on facts this draft could not check. Each must be checked or replaced before the freeze.

The freeze records in DECISIONS.md:
- this file's SHA-256;
- the vendored-asset manifest;
- the `runs/LG_data/META.json` SHA;
- the test-list SHA;
- the adapter SHA-256s;
- the selected settings.

`PREREG_PLANBENCH.md` (adfc4354…), `PREREG_B3_DISTILL.md` and `PREREG_ITEM7_SEEDS.md` (956a0dc4…) stay byte-identical.

All B2 rules carry over unless this file changes them:
- Success means VAL accepts the plan on the instance PDDL.
- One plan per instance, with no execution feedback. The exact simulator and VAL are never consulted while planning.
- Planner settings are chosen on generated sets only. Each test set is run once per frozen system.
- Intervals are Clopper-Pearson 95% CIs. Paired contrasts use exact two-sided McNemar (`bwmn/stats.mcnemar_exact`, stats.py line 24).

## Question

Does the B2 result carry over to a second PlanBench domain?

- In Blocksworld-hard, models trained only on problems with 6 or fewer blocks solved 73/110 with the learned world model plus search (S-beam).
- The same proposer alone (S-direct) solved 5/110.
- Sources: P-10; `runs/BW_TEST_B2/hard_{beam,direct}/summary.json`.

Item 9 tests: "learned F + search beats S-direct out of size range in a new domain". Sources: `research/JOINT_ROADMAP_2026-10-08.md` line 38; the user chose Logistics in D-65.5 (DECISIONS.md line 765).

**Which dimensions are out of range:**
- **Packages and goals (fully out of range).** Training uses 1-3 packages; testing uses PlanBench Logistics Task-1 instances with 4-7 packages. Every PlanBench goal is exactly one `(at p l)` per package (286/286 files), so the goal count is out of range as well.
- **Maps (in range).** Training and test both have 2-3 cities, 1-3 locations per city, one truck and one airport per city, and 1-3 airplanes, never more airplanes than cities.
- **Optimal plan length (only partly out of range).**
  - PlanBench instances with 1-3 packages: 3-24 steps.
  - Generated instances with 1-3 packages: about 3-27 steps [VERIFY].
  - The test list: 12-39 steps.

## Prerequisites (all before the freeze; CPU only)

1. **Vendoring.** This is a write and needs user approval.
   - The only local copy of Logistics is an upstream clone in a Temp scratchpad: `…/bc6adc92-…/scratchpad/LLMs-Planning`. Its HEAD is fc638a1a…, the commit that `bench/planbench/SOURCE.md` cites.
   - Copy into `bench/planbench/logistics/`:
     - `generated_domain.pddl` (md5 641642606bb9c846d70478af4b3f3599);
     - `generated_basic/instance-1..286.pddl`;
     - `prompts/task_1_plan_generation.json` (the ground-truth plans);
     - the 8 published Logistics result files (7 Task-1 files plus GPT-4 backprompting).
   - Record their SHA-256s in `SOURCE.md` and `LEDGER.json`.
   - `scripts/lg_published.py` re-scores every published plan with the local VAL. Expected: 0 disagreements on 1,450 plans [VERIFY].
2. **New files only:** `bwmn/logistics.py`, `bwmn/domain_planner.py`, `scripts/lg_*.py`, `tests/test_logistics.py`.
   - Not edited: `bwmn/blocks_planner.py` (sha 18a5f22b…, pinned by D-63), `blocks.py` (b1e28ed5…), `lm.py` (88a1e3a2…), `mystery.py` and `bwmn/__init__.py`.
   - `blocks_eval.py`, `bw_select.py`, `bw_freeze.py` and `bw_run_tests.py` are imported, never edited.
3. The tests listed in the implementation plan pass.
4. The harness check (see Systems) has run.

## What stays the same, and declared deviations

| Component | B2 (Blocksworld) | LG (this addendum) |
|---|---|---|
| Core | Titan-Sparse FP32, frozen | same |
| Adapters | two unmerged LoRAs (F, Pa) | same |
| Trainer | `train_generic.py`, batch 16, lr 1e-4; F 600 steps seed 0; Pa 1000 steps seed 1 | same |
| Max-len | 256 | **768** (about 13 tokens per action vs 5.3; generated 1-3-package Pa rows reach 690 tokens, review LG-SPEC-1) |
| F output | delta `{add, del, legal}`, `B.forward_target_delta` / `B.parse_forward_delta` | same functions, unchanged |
| Pa rows | outcome-conditioned, action-only; every suffix of an optimal plan, plus one noisy plan | same recipe |
| Planner | `plan_beam` / `plan_direct` (sha 18a5f22b), default `goal_met` / `unmet` | same file, unmodified, default hooks |
| Selection rule | `bw_freeze.py` lines 30-36 | same rule, new grid |
| K | 8 (selected) | **4 (fixed, for cost)** |
| New code | (none) | the domain module: simulator, renderer, parser, generator, signature; planner subclasses; `lg_*` scripts |

**Declared deviations:**
1. Max-len is 768 instead of 256 (peak memory to be re-measured in the bundleL1 smoke test).
2. K is fixed at 4. B2 selected K = 8; in B2's selection, K4 vs K8 gave val_large 0.600 vs 0.633 at 134.3 vs 256.7 s per instance, and val_v2 0.98 vs 0.98 (`runs/BW_SELECT_v1/select.jsonl`).
3. Token caps are rescaled to about 14 tokens per action. max_depth goes from 80 to 100, the S-direct M from 256 to 704, and the parse cap from 60 to 96 actions.
4. The F random walk starts on a teacher-plan state. B2's code starts at init (`blocks_data.py` lines 105-108), though its comment says "states along the optimal plan".
5. A quarter of the illegal F actions violate argument types. Blocksworld has a single type, so it has no analogue.
6. Half of the training instances keep PlanBench names. B2 used names a..o for every instance.
7. The planner grid is new. C1 is B2's allocation.
8. A futility gate is added.
9. The primary test set is a stratified subset, whereas B2 ran full sets.

## Domain module `bwmn/logistics.py` (pure python)

- **State.** The full PDDL fact frozenset, static facts included (type facts, `in-city`, `airport`), parsed by `B.parse_problem`. Each state is therefore a self-contained key for plan_beam's seen set and for the F memo and cache.
- **Semantics follow VAL.**
  - The 6 operators of `generated_domain.pddl`, with preconditions including the type facts; effects are applied delete-then-add.
  - A self-loop such as `drive-truck t l l c` or `fly-airplane a l l` is a legal no-op. VAL prints "Plan valid" for both on instance-2.
  - An illegal action changes nothing and is labelled `legal: false`, the B2 convention.
  - `mystery.ground` is not reused, because it rejects repeated objects (mystery.py lines 202-206).
- **Grounding is typed:** 14-720 ground actions per PlanBench instance, against about 265k-315k untyped.
- **Teacher.**
  - A* with an admissible package-movement bound. Self-loops are excluded.
  - A 30 s timeout per instance; failures are dropped and counted.
  - Optimality check: on every PlanBench instance with 3 or fewer packages that it solves, its plan length must equal the Fast Downward `astar(lmcut())` ground truth (`plan-bench/prompt_generation.py` line 49).
- **Rendering** (B2 style, through `B.canonical` and `B.render_chat`):
  - **State:** `{"map":{"airports":[…],"cities":{c:[locs]}},"packages":{p:place},"planes":{a:loc},"trucks":{t:loc}}`. Types come from the group, not from the name.
  - **F prompt:** `render_chat(SYS_FORWARD_DELTA_LG, canonical({"action","state"}))`.
    - Target: `B.forward_target_delta`. For a legal action that is not a self-loop it is one add and one del fact.
    - Size: at most 25 tokens plus EOS on every PlanBench ground-truth step, against F's 40-token cap (blocks_planner.py line 73).
  - **Pa prompt:** `render_chat(SYS_PROPOSER_LG, canonical({"goal":{p:l},"outcome","state"}))`.
    - Target: `B.target_actions` with PDDL operator names, e.g. `"load-truck p0 t0 l0-1"`.
    - Cost: 13.0 tokens per action on average on PlanBench ground-truth plans (maximum 13.88).
  - **System prompts** state the operator rules, as B2's do (blocks.py lines 280-288 and 324-329). The final text is fixed and hashed with the module. The drafts measure 164 tokens (F) and 84 tokens (Pa) [VERIFY final text].
- **Parser.** `parse_action` checks the operator name, the arity (3/3/3/3/4/3) and that every argument is a problem object.
  - It does **not** check types. Type preconditions are F's job, as in the PDDL, where types are preconditions.
  - `parse_plan` and `parse_plan_prefix` mirror blocks.py, with a 96-action cap.
- **Names.** In all 286 files, PlanBench names encode the map: `l<c>-0` is the airport of `c<c>`, and `t<c>` sits in `c<c>`.
  - 50% of training instances keep PlanBench names.
  - The other 50% get random indices: the airport index is random, and location, truck and plane indices are decoupled from city indices.
  - Package names come from p0..p11, so the test names p3..p6 occur in training.
- **Signature (used for exclusion).** An exact renaming-invariant canonical form. It permutes cities and the non-airport locations within each city, and treats vehicles and packages as multisets of their positions and goals. Expected: 285 classes over the 286 PlanBench files, because instances 47 and 48 are renamings of each other [VERIFY in tests].
- **Problem files.** `write_problem` emits `(:domain logistics-strips)`. `val_check = partial(B.val_check, domain_path=<vendored domain>)`.

## Training data (`scripts/lg_data.py` → `runs/LG_data/`, seed 20261401)

- **Instances: 6,000 train plus 400 validation**, the B2 v2 counts (`runs/BW_data_v2/META.json`).
  - Cities {2, 3}; one city size per instance, drawn from {1, 2, 3}.
  - One truck per city at a random location of its city; one airport per city.
  - 1 to #cities airplanes, each at a random airport.
  - Package count drawn from [1, 2, 2, 3, 3, 3], the analogue of B2's block mix [3, 4, 4, 5, 5, 5, 6] (blocks_data.py line 45).
  - Package origins and goals are uniform over all locations, and every package has a goal. Instances whose goal already holds are skipped.
- **Exclusion:**
  - Drop any instance whose signature equals one of the 286 PlanBench signatures. This covers instance 1 and the 73 in-range instances.
  - Drop suffix rows whose intermediate (state, goal) signature hits PlanBench (the B2 guard, blocks_data.py lines 80-84).
  - Split train and validation by signature, **symmetrically** (P-18): B2's guard (blocks_data.py lines 59-66) checks
    only validation signatures before a train insert, so an isomorph drawn into train first could later enter
    validation; 303/400 of B2's val_v2 were renamed training problems. Here an instance goes to validation only if its
    signature, and every suffix (state, goal) signature of its trajectory, is absent from train, and vice versa.
    Duplicates within train are kept, as in B2.
  - Signatures of the harness set, LG-VAL-OUT and MIRROR-80 are excluded everywhere.
  - All counts go to META.json.
- **Pa rows:**
  - every suffix of the optimal plan, with outcome `success`;
  - one noisy plan per instance, with one action replaced by a random typed ground action, labelled with the outcome it actually achieves.
- **F rows:** 12,000 unique (state, action) keys, plus 1,500 validation rows.
  - Start state: chosen uniformly among the teacher plan's states, then 0-7 random legal steps.
  - Action: 50% legal, drawn uniformly from the applicable typed actions EXCLUDING self-moves, except that self-moves (legal no-ops) make up exactly 5% of the legal rows so the no-op semantics is still taught (review LG-SPEC-2: uniform drawing would make about 36% of legal rows self-moves).
  - The other 50% are illegal: 75% typed actions whose preconditions fail, 25% actions with the right arity but a wrong-type argument.
  - Labels come from the VAL-semantics simulator.
- **Length.** Rows over 768 tokens (prompt + target + EOS) are dropped and counted before training, because `lm.train_sft` raises on overlength rows (lm.py lines 413-414; D-58 precedent).
  - Expected count: 0 at 768 (the largest generated 1-3-package row seen in the module tests is 690 tokens).
  - Measured on PlanBench 1-3-package ground-truth rows with the draft prompts: at most 588 tokens.
  - Designers' generated samples reached Pa 567-617 and F 316-329 tokens [VERIFY on the generated data].
- No PlanBench instance, plan or prompt enters any row.

## Adapters (`scripts/train_generic.py`, unchanged)

| Adapter | Rows | Steps | Batch | lr | Seed | Max-len |
|---|---|---:|---:|---|---:|---:|
| `LG_F_s0` | `runs/LG_data/forward_delta_train.jsonl` | 600 | 16 | 1e-4 | 0 | 768 |
| `LG_Pa_s1` | `runs/LG_data/proposer_actions_train.jsonl` | 1000 | 16 | 1e-4 | 1 | 768 |

- These are the `BW_Fdelta_s0` and `BW_Pa_v2_s1` recipes (their `metrics.json` args), except max-len.
- Checkpoints resume across turns (lm.py lines 437-452).
- There is no checkpoint selection: the final adapters are frozen.

## Generated validation sets (selection only; never test sets)

- **LG-VAL-IN:** 30 instances, 10 per package count 1-3, taken every k-th from the signature-disjoint validation split.
- **LG-VAL-OUT:** 20 instances, 5 per package count 4-7, seed 20261402.
  - Within each package count, (cities, city size, airplanes) is drawn from the empirical shapes of PlanBench instances with that package count (the B2 mirror precedent).
  - Signatures of PlanBench, train and validation are excluded.
  - Scored with the exact simulator, so no teacher is needed.
- These are the only sets on which planner settings are compared.

## Systems

- **S-beam (primary):** Titan-Sparse + `LG_F_s0` + `LG_Pa_s1`, running `BP.plan_beam` through `bwmn/domain_planner.py`.
  - Default hooks: `goal_met` = goal ⊆ state, `unmet` = |goal − state| (blocks_planner.py lines 279-280). There is no Logistics progress hook, so the domain rules stay inside F.
  - K = 4: candidate 0 is greedy; the others are sampled at T 1.0, top-p 0.95, with per-candidate CRN seeds.
  - P = 1 (prefix parse), eval_seed 0, max_depth 100, F max_new 40.
- **S-direct (no world model):** the same Pa, greedy, K = 1, `plan_direct`, M = 704 tokens. That leaves room for the longest PlanBench optimal plan: 48 actions = 619 target tokens. This is the D-48(a) analogue.
- **Grid.** K = 4 and P = 1 throughout. Every configuration has the same worst-case envelope: E×K×R = 1,152 rollout steps (so B2's 2,000 F-query cap never binds) and about 17k proposer tokens per instance.

| Config | W | E | R | M | Role |
|---|---:|---:|---:|---:|---|
| C1 | 4 | 24 | 12 | 184 | B2's frozen allocation (FREEZE_B2_beam), with M rescaled |
| C2 | 4 | 12 | 24 | 352 | longer window for delivery plateaus |
| C3 | 2 | 6 | 48 | 688 | near-whole-plan rollouts, fewer expansions |

M = 14R + 16.

Why the window R matters:
- `unmet` drops only when a package arrives.
- If no candidate makes goal progress within its first R actions, the child advances only one step (line 320 picks the earliest minimum).
- Optimal Logistics plans batch their deliveries, which leaves long stretches with no goal progress. Generated evidence: gaps of more than 12 steps between deliveries in 9/40 A*-optimal three-package plans (`designer_lean/gaps_gen.py`) [VERIFY].
- A designer also measured these gaps on PlanBench ground-truth plans. That is disclosed here; no model output was involved, and selection uses generated sets only.

**Harness check** (before the freeze; CPU only; no model; the D-44 analogue):
- ExactD runs through the unmodified `plan_beam` at C1, C2 and C3.
- Two scripted proposers:
  - (a) sequential single-package delivery;
  - (b) phase-batched: all first legs, then all flights, then all last legs. This reproduces the plateaus of optimal plans.
- In both, 20% of candidates have one action perturbed.
- The harness set has 40 instances, 10 per package count 4-7, seed 20261405.
- A configuration that reaches the goal on less than 90% of instances under proposer (a) is replaced before the freeze, through a recorded amendment.
- The results are reported.

## Frozen selection rule (`scripts/lg_freeze.py`; the `bw_freeze.py` rule, lines 30-36)

- Among C1-C3 evaluated on both LG-VAL-IN and LG-VAL-OUT, pick the highest mean success, (LG-VAL-IN + LG-VAL-OUT) / 2.
- Constraint: at most 2,000 unique F queries per instance on each set.
- Ties go to the configuration with fewer proposer tokens per instance.
- S-direct is fixed and is not a selection candidate.
- The freeze writes `runs/FREEZE_LG_beam.json` and `runs/FREEZE_LG_direct.json`. They contain:
  - the args;
  - the adapter SHA-256s;
  - this file's SHA-256;
  - the data META SHA and the test-list SHA;
  - the SHAs of blocks_planner.py, blocks.py, lm.py, logistics.py and domain_planner.py.
- `lg_eval.py` refuses a test run if its args differ from the freeze record or if its `summary.json` already exists (the `blocks_eval.py` lines 86-97 pattern).

## Futility gate (generated data only; protects the once-only test)

- **GO** iff, on LG-VAL-OUT, the selected S-beam solves at least 4/20 **and** solves at least 3 more of the same 20 than S-direct.
- On **NO-GO**:
  - no PlanBench test is run, and every PlanBench Logistics instance stays reserved;
  - the item-9 reading is "not supported on generated out-of-range problems";
  - a revised recipe needs a new addendum and re-selection on generated sets.
- The decision is written to `runs/LG_GATE.json`.

## Test sets (each run once per frozen system)

- **PB-OUT-80 (primary).** PlanBench Logistics Task-1 instances (IDs 2-286) with 4-7 packages. The population is 23 / 42 / 47 / 100 instances with 4 / 5 / 6 / 7 packages.
  - Sampling: 20 per package count. Sort each count's IDs and take index floor(i·n_k/20) for i = 0..19.
  - Membership depends only on package count and ID.
  - IDs, in order:
    - 4 packages: 16, 17, 18, 19, 30, 31, 32, 75, 76, 77, 78, 79, 80, 136, 138, 139, 140, 141, 142, 143
    - 5 packages: 34, 36, 81, 83, 85, 87, 89, 91, 93, 95, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165
    - 6 packages: 38, 40, 42, 98, 100, 102, 105, 107, 109, 112, 114, 116, 119, 169, 171, 174, 176, 178, 181, 183
    - 7 packages: 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281
  - sha256(json.dumps(ids)) = 24dcb050195149e2d82af339e8d2e7846194ccdfd950fc64c62b1b7949bda249.
  - FD-optimal length 12-39, mean 26.0. By package count (min/mean/max): 4: 12/22.1/30; 5: 13/23.9/38; 6: 14/26.4/39; 7: 26/31.6/38.
  - (cities, planes) counts: (2,1) 6; (2,2) 6; (3,1) 12; (3,2) 41; (3,3) 15.
  - 63 of the 80 IDs fall in 2-200: all of the 4-6-package strata and 3 of the 7-package stratum.
- **MIRROR-80 (contamination and generalization control; S-direct only in the core).**
  - One generated twin per PB-OUT-80 instance, with the same shape (cities, city size, airplanes, packages), fresh random positions and goals, and PlanBench naming. Seed 20261403.
  - Signature-disjoint from PlanBench, train, validation, LG-VAL-OUT and the harness set. "Train" means the
    signature of EVERY training row's (state, goal), suffix rows included, not only problem-initial ones (P-18:
    seeds_fresh/0 was a renamed suffix row of a Pa training problem). The same rule applies to LG-VAL-OUT and the
    harness set.
  - Generated and hashed before any training; a contamination check against the final training rows (P-18 method)
    is rerun after training data is written and before any test run, and its result is recorded.
- **Reserved.** No model runs these under this addendum except through Stage C:
  - the other 132 PB-OUT instances (3 / 22 / 27 / 80 with 4 / 5 / 6 / 7 packages);
  - the 73 in-range instances.
  - Instance 1, PlanBench's one-shot example, is used nowhere.
  - RCWM and RCX are told, through the user, that all PlanBench Logistics instances, MIRROR-80 and the LG sets are reserved for BWM (D-64 pattern).
- **Shards.** S-beam runs PB-OUT-80 in 8 shards, assigned by index % 8 (D-50). A shard cut by the turn cap restarts from scratch with identical seeds and a warm F cache.

## Registered test and pre-set readings

**P1 (primary):** S-beam vs S-direct on PB-OUT-80, paired by instance, exact two-sided McNemar.

| Reading | Condition |
|---|---|
| **Transfer: YES** | S-beam − S-direct ≥ +0.20 (at least 16 of 80) **and** p < 0.01 |
| **PARTIAL** | b > c and p < 0.05, but YES is not met |
| **NO** | otherwise |
| **NOT RUN** | the futility gate said NO-GO |

At c = 0, p < 0.01 needs b ≥ 8 (mcnemar_exact(8, 0) = 0.0078), so in practice the +0.20 bar is what binds.

**Registered flag (descriptive; B2 precedent):** S-direct success on PB-OUT-80 minus S-direct success on MIRROR-80. A gap above 10 pp is reported as possible contamination of the proposer. There is a single primary test, so no multiplicity correction applies to it.

## Report-only (no gate)

- **Success:**
  - success per system on every set, with CP 95% CIs;
  - the paired difference, with a Newcombe (method 10) 95% CI;
  - success by package count (n = 20 each) and by FD-optimal length bin (12-19 / 20-29 / 30-39);
  - an estimate reweighted to PB-OUT's package mix (23 / 42 / 47 / 100).
- **Plan quality:**
  - the number of VAL vs exact-executor disagreements (expected 0);
  - verified rate and verification precision, P(VAL success | F-verified);
  - failure modes: budget, dead_end, unparseable;
  - plan length relative to optimal on solved instances.
- **Compute per instance:**
  - proposer tokens; unique F keys needed, counting cache hits (FR-02 lesson); F tokens; wall time in seconds;
  - FLOPs, estimated afterwards with `bwmn/flops.py`.
- **F out of range:**
  - Every F key queried in the PB-OUT-80 S-beam run is replayed against the exact simulator on CPU, with no model, giving step exactness and false-legal / false-illegal rates by package count.
  - A generated F audit (`lg_forward_grade.py`, seed 20261404): 60 keys per package count 1-7, half of them illegal.
- **Diagnostic after the freeze:** the exact-F ceiling on LG-VAL-OUT (the frozen S-beam with ExactD in place of F).
- **Selection sets:**
  - results for every configuration;
  - the gap between LG-VAL-OUT and PB-OUT-80. It is descriptive only, because the selection set is biased upward.
- **Published comparators:** paired exact McNemar against each published per-instance file on the IDs it covers, Holm-adjusted across the 7 files, labelled "reference; different regime".
- **Caveats:** all numbers are single-seed per adapter. A non-significant result is not read as equivalence.

## Conditional Stage C (registered now; report-only)

Stage C runs only if P1 reads YES or PARTIAL. Each step needs an explicit user GPU approval, and the steps run in this order:

- **C-1:** S-beam on MIRROR-40 (the first 10 twins per package count, by index), compared with the 40 PB-OUT-80 twins it pairs with. A gap above 10 pp is flagged. The twins have identical shapes, so no standardization is needed (the P-10 raw-vs-standardized lesson).
- **C-2:** both arms on PB-IN-73, the in-range PlanBench instances. Published comparators cover 72 of them.
- **C-3:** both arms on PB-OUT-REST-132. This gives the full PB-OUT-212 estimate (descriptive) and the published pairing on all 127 PB-OUT IDs in 2-200.

Seed replication is not part of this addendum; it would need its own prereg.

## Published comparators (reference only; paired by instance on identical IDs; same local VAL)

Files are under `bench/planbench/logistics/results/` once vendored (now: `CLONE/llm_planning_analysis/results/logistics/`).

| System, setting | Correct on PB-OUT-80 (IDs covered) | Correct on the whole file | Paper source |
|---|---:|---:|---|
| o1-preview, zero-shot PDDL | 59/63 | 188/200 (IDs 1-200) | Valmeekam et al., "Planning in Strawberry Fields", arXiv 2410.02162, Table 2 (94%) |
| GPT-4, one-shot NL | 4/64 | 28/200 (IDs 2-201) | Valmeekam et al., NeurIPS 2023, arXiv 2305.15771, Table 1 (14%) |
| GPT-4, one-shot PDDL | 6/64 | 28/200 | 2305.15771, Table 2 (14%) |
| GPT-4, zero-shot NL | 1/63 | 15/200 | 2305.15771, Table 1 (7.5%) |
| GPT-4, zero-shot PDDL | 0/63 | 11/200 | 2305.15771, Table 2 (5.5%) |
| LLaMA-3.1-405B, zero-shot PDDL | 0/63 | 19/200 | local file only; no paper table located [VERIFY] |
| GPT-3.5-turbo-instruct, one-shot NL | 2/64 | 6/200 | 2305.15771, Table 1, "Instruct-GPT3.5" (3%) [VERIFY label] |

- The LLaMA and o1 files say `prompt_type: oneshot`, but none of their queries contains an example [VERIFY].
- The paper tables give only aggregates; which 200 instances they used is not stated. The per-instance files are what gets paired.

**Not comparable** (cited, never paired):
- GPT-4 with VAL backprompting: 35/50 on instances GPT-4 had already failed (2305.15771, Table 4). The verifier is in the loop.
- LLM-Modulo and AoT+ figures (arXiv 2501.13545) [VERIFY].
- AutoPlanBench closed-loop Act/ReAct (arXiv 2311.09830) [VERIFY].
- o1-preview on Randomized Logistics: 52% (2410.02162, Table 2). A scout found 63/200 of its embedded problem names illegal for VAL, and 147/200 valid on re-execution [VERIFY].
- Fast Downward: 100% (2410.02162, Table 2).

**Literature context** (each uses its own generated Logistics set, not PlanBench IDs). Published SFT size extrapolation collapses:
- Bohnet et al., arXiv 2406.13094: 99.8% in range, 10.8% after training on 1-2 packages and testing on 3-5 [VERIFY].
- Huang, Lipovetzky and Cohn, arXiv 2412.10675: 100% in distribution, 14.0% on 17-32-action plans [VERIFY].
- Fritzsche, Gestrin and Seipp, arXiv 2508.07743: 0.00 extrapolation coverage on Logistics [VERIFY].

## Comparability ledger (stated up front)

- **Regime.** BWM is a domain-trained specialist with a learned simulator and search, and it plans without feedback. The published systems are zero- or one-shot generalists. Fast Downward solves 100%. o1-preview solves 59/63 of the covered PB-OUT-80 IDs, so S-beam will very likely score lower; that is not a failure criterion.
- **Input.** BWM reads PDDL that code parses into compact JSON. The published prompts were natural-language or PDDL text.
- **Compute.** API compute is unknown, so no FLOP-matched claim is made against API models.

## Allowed claims

**If YES:**
- "Trained only on generated Logistics problems with at most 3 packages, the frozen 0.6B Rineya core with two LoRA adapters, planning without execution feedback, solves X/80 of a pre-registered stratified sample of PlanBench Logistics instances with 4-7 packages (CI). The same proposer without the world model solves Y/80 (paired McNemar p)."
- "The planner code, the trainer, the LoRA recipe and the selection rule were B2's. Only the domain module, max-len and the planner settings changed, and the settings were re-selected on generated sets. With Blocksworld-hard, the world-model-plus-search effect holds out of size range in two PlanBench domains (one seed per domain)."

**If PARTIAL:** the numbers above, worded as "a positive effect below the registered bar".

**Published numbers** may be quoted only paired on identical IDs and with the regime caveat.

**Not allowed:**
- "domain-general planner" or "general world model": there are two domains, each with its own adapters, generator and parser;
- "beats o1-preview";
- "beats GPT-4 at planning" without the regime caveat;
- any comparison with verifier-in-the-loop numbers;
- "zero-shot";
- "Logistics solved";
- any statement about obfuscated or randomized Logistics, or about reserved instances that were not run;
- calling the 80 instances "PlanBench Logistics" without the "4-7-package sample of 80" qualifier (a full-212 number exists only after C-3);
- seed-robustness wording;
- any claim from a configuration, step count or adapter chosen after seeing PlanBench results;
- FLOP-matched claims against API models.

## Budget (for scheduling only; it never changes a setting or a test-set size)

All per-turn figures are planning estimates of ±50%.
- The selection runs measure the cost per instance before the test. That measurement changes only the shard plan.
- If the test run overruns, it continues in later turns; the test set is never subset after measurement.
- No GPU is used until this file is frozen (D-65.5). The work is queued after bundleE, then bundleH (if B3 is met), then bundleI (D-66), unless the user reorders.
