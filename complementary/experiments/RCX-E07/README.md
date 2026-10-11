# RCX-E07/E08: relational value learning for Blocksworld planning (runner)

**Tier:** CPU mechanism tier. A learned heuristic is an external specialist (tier B); no Rineya-core claim.

**Plan:** `DESIGN.md`. **Infrastructure:** `rcx/bw/*`, tested in `tests/test_bw.py`. **Exclusions:**
`EXCLUSIONS.json` (BWM recipe, CX D-11).

## How to run (from `complementary/`)

```
python -B experiments/RCX-E07/run_e07.py --out runs/RCX-E07/smoke --phase all --smoke --threads 2   # < 2 min CPU
python -B experiments/RCX-E07/run_e07.py --out runs/RCX-E07/main  --phase all                       # full design
```

- **Phases.** `data`, `train`, `e07`, `e08`, `report`, run in that order by `all`. A phase is skipped when its outputs
  exist, so re-running the same command resumes. `e08` also resumes per search run (`e08_runs.jsonl`).
- **Config freeze.** `<out>/config.json` freezes the configuration at first use. A later invocation with a different
  configuration aborts, so use a new `--out` instead.
- **Overrides.** `--seeds`, `--threads` (default 4), `--train-per-size`, `--val-per-size`, `--test-per-size`,
  `--k-states`, `--epochs`, `--time-cap-s`, `--astar-budget`, `--opt-budget`, `--e08-per-size` and
  `--budgets 500,2000`.
- **Courtesy.** `rcx.guard.ThermalGuard().check()` is called every ≥ 30 s inside loops, after every epoch and at phase
  start. It only reads the RCWM watchdog log and pause flag, and never spawns a subprocess. Guard sleep is logged and
  excluded from the training time cap.
- **Determinism.** All seeds are fixed. Float results are bit-reproducible for the same `--threads`.
- **Outputs.**

  | File | Contents |
  |---|---|
  | `data/{train,val,test}.jsonl.gz`, `data/instances.json`, `data/meta.json` | Data, instances, allocation and stats |
  | `models/{gnn,mlp}_s<seed>.{pt,json}` | Weights (the `.pt` files are git-ignored) and training history |
  | `e07_metrics.json`, `e07_scores.jsonl.gz` | E07 metrics, plus per state: h\*, predictions, policy correctness, ties |
  | `e08_instances.json`, `e08_runs.jsonl`, `e08_metrics.json` | E08 instances (with A\* optimal lengths), runs, metrics |
  | `REPORT.md`, `metrics.json` | Tables plus a provenance block: code shas, EXCLUSIONS sha, BWM pins, seeds, config, wall times, guard wait, BWM bytecode check |
  | `timings.json`, `log.txt` | Wall times and log |

## Design as implemented (coordinator spec, 2026-10-10), with the resolutions of ambiguities

### Data

The goal-class constraint drives the data. A goal made only of `on` pairs is fixed up to renaming by its multiset of
chain lengths, so there are only p(n) − 1 goal classes per size: 4, 6, 10 and 14 for n = 4, 5, 6 and 7. This holds
for full and partial goals alike.

**Decision CX D-17.** Classes are disjoint across splits, with one instance per goal class (`unique='goal'`) within
train and val.
- **Why not more instances per class.** At n ≤ 6 every state of an instance's goal is labelled, so further instances of
  the same goal class only add renamed duplicate rows.
- **Compensating at n = 7, where states are sampled.** `train_states_large` was raised from 3,000 to 10,000.
- **Effect.** Train therefore has fewer than "40 per size" instances, and this is reported as a deviation. Class
  diversity, not instance count, is the binding constraint, and it is shared by every model compared.

The runner resolves the allocation as follows.

1. **Class allocation per size** (`allocate_classes`).
   - *Discovery.* 3,000 draws per size of `B.generate(full_goal_frac=0.5)` with the split seed 20261275. Only the
     invariant filter is applied, which is cheap.
   - *Eligible classes.* Classes seen at least 5 times. In practice this is all of them.
   - *Split.* A seeded shuffle, then:
     - test: ⌈0.25·C⌉ classes, at the ID sizes (5 and 6) only;
     - val: max(1, round(0.2·C)) classes;
     - train: the rest.
   - *Full design.* n=4: train 3 / val 1. n=5: train 3 / val 1 / test 2. n=6: train 5 / val 2 / test 3. n=7: train
     11 / val 3.
2. **Train.** `fresh_instances(20261270, ..., full_goal_frac=0.5, unique='goal', accept=train classes)`.
   - One instance per train goal class, with `--train-per-size` as a cap. The cap of 40 never binds.
   - States: all states for n ≤ 6, or a uniform sample of 3,000 at n = 7.
   - With all states labelled, more instances of the same class would only duplicate rows exactly (up to renaming).
   - This does not hold at n = 7, where states are sampled: there the cap costs distinct rows (11 classes × 3,000
     instead of 40 × 3,000). It is a deviation from "40 instances/size". Allowing repeats of a train class within
     train (`unique='instance'`; classes stay disjoint across splits through `accept`) would restore it, at about 6×
     the train rows, mostly renaming-duplicates at n ≤ 6. That choice is open for the coordinator.
3. **Val.** Seed 20261271, the same per-class scheme, at most 10 instances per size, 300 states per instance. Val is
   used only for checkpoint selection.
4. **Test.** Seed 20261272, 15 instances per size.
   - ID sizes (5, 6) draw only from test-allocated classes. Instances can repeat a class but are deduped by instance
     (renaming signature).
   - OOD sizes (8, 10, 12) are unconstrained; their classes are disjoint by size.
   - Disjointness of goal classes across splits is asserted.
5. **Row exclusion.** `exclude_rows=True` everywhere, including test rows: a row whose (state, goal) would itself be a
   reserved instance is dropped. Counts are reported.
   - **val_v2 supplement** (`excl.with_selection_sets`, data phase only). BWM's selection set val_v2
     (`runs/BW_data_v2/instances_val.json`: 400 instances, 4–6 blocks) was drawn with `B.generate(full_goal_frac=0.5)`,
     which is this runner's own generator, and BWM's recipe does not cover it. Without the supplement, the smoke run
     had one train and one ID-test instance that were val_v2 instances, and the full design's ID test had one more
     (`test/20261272/6`).
   - The supplement reads the file only to compute renaming signatures and invariants. Instances and rows that match
     a val_v2 instance are excluded as `selection_signature`. The file's sha256 and counts are recorded in
     `meta.json` and in the provenance.
   - Cost: about 1 s, plus up to roughly 1 min of extra n = 6 row signatures.
6. **Test states.** The init plus K = 60 random-walk endpoints, with walk lengths uniform in 0..2n and uniform random
   applicable actions, deduped in order.
   - Labels: exact h\* and the set of optimal actions.
     - n ≤ 8: full state space.
     - n ≥ 9: `space.astar_optimal`, one A\* run per state that continues through f ≤ h\* and then marks
       optimal-path nodes backwards. It agrees with exact labels on 320 states in tests.
   - A budget failure is stored as `miss = "astar_budget"`. Such states are excluded from every model identically
     and counted in the report.

### Models

- **Matched training.** RelGNN (default, `node_sum` readout, 123,969 params) and FlatMLP (default, 158,721 params) are
  trained on identical rows, in the identical shuffle order for a given seed, with the identical budget:
  - `--epochs` (default 12), batch 64, AdamW at lr 2e-3, Huber loss with δ = 1;
  - the same time cap (default 3,600 s, excluding guard sleep). A cap hit is reported as a budget mismatch.
- **Checkpoint selection.** The checkpoint is the epoch with the best val MAE (early stopping by selection, not by
  truncation, so the budgets stay matched).
- **Seeds.** 20261276–20261278.

### E07 metrics

- **Coverage.** Every metric is computed per size and for the pooled ID and OOD groups.
- **Error metrics (learned models only).**
  - MAE and RMSE.
  - "Calibrated exact": `round(pred) == h*`.
- **Spearman correlation (all models).** Computed within each size. For the pooled ID and OOD groups it is the
  unweighted mean of the per-size values (per-size values are in `spearman_per_size`), because a correlation pooled
  across sizes would reward merely predicting n (h\* grows with n). Reported as `–` when a predictor is constant in
  any size of the group, such as `h0`.
- **Policy accuracy.**
  - Computed over states with h\* > 0 and a known optimal set.
  - A state counts as correct when every successor that minimises the predicted h is optimal.
  - Ties use a tolerance of 1e-5·max(1, |min|). The tie rate is reported.
- **Paired CIs.**
  - Method: `rcx.stats.seed_parent_bootstrap`. Instances are the resampled clusters; each instance's states are
    averaged per seed and then over seeds. 5,000 resamples. The RNG seed is `"20261275:bootstrap"`, derived from
    the allocated split seed instead of `rcx.stats`' default 17. Seeds are averaged, not resampled, so a CI is
    conditional on the trained seeds; seed-to-seed spread is reported separately (mean ± sd and per-seed values).
  - Comparisons: GNN − MLP and GNN − best handcrafted.
  - Scopes: each OOD size and pooled OOD, plus pooled ID as context only.
  - "Best handcrafted" is the handcrafted heuristic with the highest instance-macro policy accuracy *in that scope*
    (the statistic the bootstrap compares; all five are listed in `handcrafted_instance_macro`). It is chosen on the
    test states themselves, so the baseline is optimistically selected: conservative for a "GNN better" claim,
    anti-conservative for a "GNN worse" claim.

### E08

- **Instances.** Seed 20261273, sizes {8, 10, 12, 14, 15}, 20 per size, full goals, exclusion-checked and distinct
  from the E07 test instances. Their goal classes are disjoint from the train/val classes (n ≥ 8 > 7), and a guard
  checks this.
- **Optimal plan length.** `astar_hstar(init)` within `--opt-budget` (300,000 expansions). Coverage is reported;
  suboptimality is len/opt over successes with a known optimum.
- **Search.**
  - GBFS (exact `B.step` dynamics; plans checked with `B.execute`) with budgets {500, 2000} for: GNN seed-1, MLP
    seed-1, `unmet_goal_facts`, `misplaced_blocks` and `h_add`.
  - `h0` runs at 500 only (the "small budget").
  - FlatMLP is n/a at 14 and 15 blocks, because its encoding caps n at 12 by construction.
  - GBFS is deterministic, so a larger-budget run that succeeded within a smaller budget is reused as that budget's
    run (`derived_from`).
- **Paired tests.** Exact McNemar for GNN vs `unmet_goal_facts` and for GNN vs `misplaced_blocks`, per budget,
  pooled and per size.

## Estimated full-run cost (from the 2026-10-10 smoke run, 2 threads; indicative)

The smoke-based extrapolation (`REPORT.md` → "Estimated full-run CPU time") gives about **0.8 h expected and
1.8 h upper bound**. The split:

| Component | Estimate |
|---|---|
| Test-state A\* labelling at n = 10/12 | ~20 min (heavy-tailed) |
| Training of 6 models | ~12 min |
| E08 GBFS | ≤ 22 min upper bound |
| E08 optimal-length A\* | ≤ 15 min upper bound |
| Everything else | a few minutes |

Peak RAM is dominated by the n = 8 state space (~180 MB) and the encoded training rows (< 0.5 GB).

## RCX-E07b (runner `run_e07b.py`, prereg `PREREG_E07B.md`)

**Question.** Can the learned relational value keep its low-budget GBFS advantage and close R-04's high-budget gap?
At 2,000 expansions R-04's GNN solved 65/100, against 87 for misplaced_blocks and 89 for h_add. The design is
frozen in `PREREG_E07B.md` (sha256 `8a4cbde0…`, CX D-20). The runner verifies that sha at every start and aborts if
the prereg changed.

### How to run (from `complementary/`)

```
CUDA_VISIBLE_DEVICES= python -B experiments/RCX-E07/run_e07b.py --smoke --threads 2 --out <scratch dir>  # ~2-3 min
CUDA_VISIBLE_DEVICES= python -B experiments/RCX-E07/run_e07b.py --threads 4                             # full run
```

- **Phases.** `data_b`, `train_b`, `select`, `e08b`, `policy`, `report`, run in that order by the default `--phase
  all`. Finished work is skipped (per phase, per model seed, and per GBFS run), so re-running the same command resumes.
- **Freezes.** `<out>/config.json` freezes the configuration, and `<out>/inputs.json` freezes the prereg sha and the
  sha256 of every R-04 input. A mismatch on either aborts the run.
- **Output directory.** The full run writes only to `runs/RCX-E07/e07b`, as the prereg fixes. A smoke run may use any
  `--out` except the R-04 directory, the BWM bundle, `../bwm` and `../rcwm`.
- **Read-only R-04 inputs.** The R-04 run directory `runs/RCX-E07/main` is never written. The runner reads from it:
  - the A0 models `models/gnn_s2026127{6,7,8}.pt`, loaded as saved and never retrained;
  - the train and val rows;
  - the E07 test states and the stored E07 scores;
  - the E08 instances.
- **Code reuse.** `run_e07.py` is imported as a module (`train_one`, `encode_rows`, `metric_block`, `Ctx`, I/O
  helpers) and is not modified, so R-04 stays reproducible.
- **Compute.** CPU only, ≤ 4 threads (enforced), ThermalGuard checks as in R-04.
- **Smoke configuration.**
  - Data: TEST-B and VAL-B have 2 instances per size. R-04 train rows are subsampled with stride 37. A1 uses 2 classes
    × 300 states.
  - Training: 2 seeds per arm, 1 epoch.
  - Search: budgets {20, 50, 100, 300}, with 300 standing in for 2,000 and 50 for 200. A\* runs with 20k expansions.
  - Policy: 2 R-04 instances per size.

| Output | Contents |
|---|---|
| `data/{testb,valb,a1}_instances.json`, `data/a1_n8.jsonl.gz`, `data/meta.json` | Instances (TEST-B with A\* optimal lengths), the A1 n = 8 rows, generation stats, disjointness checks, A1 class discovery |
| `models/{A1,A2}_s<seed>.{pt,json}` | Weights (git-ignored) and training history; A2's json also holds the ranking-data statistics |
| `valb_runs.jsonl`, `select.json` | VAL-B GBFS runs, the λ and BEST selection with its full table |
| `testb_runs.jsonl`, `e08b_metrics.json` | TEST-B GBFS runs, all cells, seed aggregates, McNemar tests, the PRIMARY decision, the monotonicity check |
| `policy_scores.jsonl.gz`, `policy_metrics.json` | E07-style policy accuracy, A0's reproduction of R-04's stored scores |
| `REPORT_E07B.md`, `metrics_e07b.json` | The report (the PRIMARY decision first) and provenance: code shas, prereg sha, EXCLUSIONS sha, R-04 input shas, seeds, config, wall times, guard wait, BWM bytecode check |

### Design as implemented

**Instances.**
- **Draws.** TEST-B (seed 20261290; n 10, 12, 14 and 15, 25 each) is drawn first, then VAL-B (seed 20261291; n 12 and
  14, 15 each). Both use `data.fresh_instances(full_goal_frac=1.0)` with R-04's exclusion machinery,
  `excl.with_selection_sets(load_exclusions())`. The EXCLUSIONS sha must equal R-04's `2f211de3…`.
- **Disjointness.** One shared `seen` set is seeded with the instance keys of R-04's E08 instances and of its E07 OOD
  test instances. The runner then asserts that TEST-B and VAL-B are disjoint from each other, from R-04's E08 set, from
  R-04's E07 test set, and from R-04's train/val goal classes.
- **Seed guard.** `data.py` only allows seeds 20261270–89. For the E07b draws the guard is temporarily retargeted to the
  reserved block 20261290–99; `data.py` itself is unchanged.
- **Optimal lengths.** Each TEST-B instance gets its optimal plan length from `astar_hstar` with a budget of 300k
  expansions, as in R-04.

**A1 (n = 8, full goals).**
- **"Fresh" class.** A goal class that does not occur among R-04's E07 test instances at n = 8, which are the only
  n = 8 states E07b evaluates (in the policy endpoint). 11 of the 21 n = 8 classes qualify.
- **"Largest-available".** The highest count in 50,000 discovery draws of `B.generate(n=8, full_goal_frac=1.0)`. The
  seed is `"20261298:discovery"`, and generate() = None and invariant-excluded draws are dropped, as in R-04's
  `allocate_classes`. A class needs at least 5 draws; ties go by class key. The top 10 are taken.
- **Instances.** One instance per chosen class (`unique='goal'`), drawn with the spare seed 20261298.
- **States.** 10,000 per instance, sampled with seed `"20261298:states"` (init included), with exact h\* from the n = 8
  space and R-04's row exclusion.
- **Training.** R-04's rows plus the n = 8 rows, trained with R-04's `train_one` unchanged: 12 epochs, batch 64, AdamW
  at lr 2e-3, Huber loss, a 3,600 s cap, and the checkpoint chosen by val MAE on R-04's val rows. Seeds 20261292–94.

**A2 (ranking hinge).**
- **Pairs.** For every R-04 training row with h\* > 0, the runner reads its successors and their exact h\* from the
  instance's full state space (n ≤ 7). A successor is optimal if h\*(succ) = h\*(s) − 1. Every pair (optimal o,
  non-optimal q) contributes `relu(0.5 + f(o) − f(q))`.
- **Loss.** The hinge is averaged over a row's pairs, then over the batch rows that have pairs, and added to R-04's
  Huber loss with weight 1.
- **Exclusions.** Successors whose (state, goal) is a reserved instance under the row-level exclusion are dropped.
- **Batch construction.** It uses R-04's minibatch of 64 rows in R-04's shuffle order for the seed. The rows' graphs
  are followed by their successors' graphs, all collated into one batch and scored in one forward pass.
- **Graph reuse.** A successor that is itself a training row reuses that row's graph. Every other successor is encoded
  once, roughly 250k extra graphs (~0.3 GB).
- **Training.** Otherwise as R-04: same epochs, cap, optimizer and checkpoint rule. Seeds 20261295–97.

**A3 (hybrid key).** The key is `h = GNN_A0-seed1 + λ·misplaced_blocks`.
- **λ choice.** λ ∈ {0.25, 0.5, 1.0} is chosen on VAL-B by success@2000, then success@200, then the smaller λ.
- **Baseline.** λ = 0 is A0 seed 1 itself.

**BEST.** Among A1 seed 1, A2 seed 1 and A3(λ\*), the arm with the highest VAL-B success@2000, then success@200, then
prereg order A1, A2, A3. VAL-B also runs A0 seed 1 and misplaced_blocks, as context only.

**Search (e08b).**
- **Runs.** One GBFS run per (TEST-B instance, heuristic) at 2,000 expansions, for:
  - A0 × 3 seeds;
  - A1 × 3 seeds;
  - A2 × 3 seeds;
  - A3(λ\*);
  - the baselines misplaced_blocks, h_add and unmet_goal_facts.
- **Smaller budgets** (50, 200, 500) are derived, as R-04 did:
  - success@b means a valid plan found within ≤ b expansions;
  - this is exact because `search.gbfs` only checks the budget before each pop and applies the goal test at
    generation, and neither its pop order nor its heuristic calls depend on the budget, so a b-budget run is a prefix
    of the 2,000 run;
  - `e08b` re-runs a few (instance, heuristic) pairs directly at every smaller budget and aborts on any difference.

**PRIMARY decision.** Computed exactly by the prereg rule on TEST-B n ∈ {12, 14, 15} (75 instances), with exact
McNemar from `rcx.stats`:
- **CLOSED:** BEST@2000 ≥ misplaced@2000 − 3, and BEST@200 > misplaced@200 with McNemar p < 0.05.
- **IMPROVED:** BEST@2000 − A0@2000 ≥ 8 with McNemar p < 0.05, without CLOSED.
- **NOT IMPROVED:** otherwise.

**Seeds in paired tests.** Every paired test uses seed 1 of each arm (A0 = 20261276, the R-04 E08 model; A1 =
20261292; A2 = 20261295). Seed mean ± sd is descriptive.

**Secondary outputs** (all descriptive):
- success at every budget and size, plus seed mean ± sd;
- suboptimality and expansions;
- McNemar for A0 vs A1, A2 and A3 at 2,000 and at 200, on the 75 primary and on all 100 instances (seed 1 and
  seed-matched);
- policy accuracy at n = 8, 10 and 12 on R-04's E07 test states (R-04's exact definition) for A0, A1 and A2. This
  includes a bootstrap of A1 − A0 and A2 − A0 with seed 20261299, and a check that A0 reproduces R-04's stored
  predictions. A3 is reported as an extra.

### Ambiguities resolved (also listed in the report)

1. **"Fresh" A1 classes.** The prereg does not say fresh relative to what. If the n = 8 classes of R-04's E08 set
   (unused by E07b) were also excluded, only 5 classes would remain (`1,1,1,1,4`, `1,1,1,5`, `1,1,2,2,2`, `2,2,2,2`,
   `2,2,4`), and "10" would be impossible. The implemented reading leaves exactly 11 classes, which fits "the 10
   largest-available". `data/meta.json` records both lists.
2. **"Largest-available".** Read as generator frequency (discovery count), mirroring R-04's discovery step.
3. **The hinge form.** Read as all-pairs, averaged per row; the sentence admits either all-pairs or an "∃ optimal
   below all" reading. The all-pairs form implies the other one.
4. **Seeds in the paired tests.** Seed 1, as in the selection rule and in R-04's E08 tests.
5. **Ties left after the prereg's own tie-break.** λ: the smaller λ. BEST: prereg listing order.

### Estimated cost of the full run (from R-04's timings at 4 threads; indicative)

| Phase | Estimate | Basis |
|---|---|---|
| `data_b` | 2–5 min | TEST-B A\* at R-04's mean of 0.03 / 0.17 / 2.2 / 1.2 s per instance for n = 10 / 12 / 14 / 15, heavy-tailed; 50k discovery draws; 10 × 10k n = 8 labels |
| `train_b`, A1 | ~16 min | ~195k rows → ~37k steps × ~8.5 ms per seed, × 3 seeds |
| `train_b`, A2 | ~26–41 min | 20,724 steps at ~25–40 ms per step (each step scores ~5× more graphs: rows plus successors), × 3 seeds; plus ~1 min to build the rank data |
| `select` | 3–5 min | 30 VAL-B instances × 6 GNN-based keys; a budget-exhausting n = 12–15 GNN run costs ~2–3 s |
| `e08b` | ~20–25 min | 100 × 10 GNN-based keys at ~1–1.3 s per run (R-04 mean 0.74 s over n = 8–15), plus h_add ~2 min |
| `policy`, `report` | ~1 min | |
| **Total** | **~1.1–1.6 h** (upper ~2 h) | |

The smoke report extrapolates the same total from its measured ms/step and s/expansion.
