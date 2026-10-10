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

"Dedupe by `goal_key` so no goal class repeats across train/val/test" therefore cannot coexist with 40 train instances
per size. The runner resolves this as follows.

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
3. **Val.** Seed 20261271, the same per-class scheme, at most 10 instances per size, 300 states per instance. Val is
   used only for checkpoint selection.
4. **Test.** Seed 20261272, 15 instances per size.
   - ID sizes (5, 6) draw only from test-allocated classes. Instances can repeat a class but are deduped by instance
     (renaming signature).
   - OOD sizes (8, 10, 12) are unconstrained; their classes are disjoint by size.
   - Disjointness of goal classes across splits is asserted.
5. **Row exclusion.** `exclude_rows=True` everywhere, including test rows: a row whose (state, goal) would itself be a
   reserved instance is dropped. Counts are reported.
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
- **Spearman correlation (all models).** Reported as `–` when a predictor is constant, such as `h0`.
- **Policy accuracy.**
  - Computed over states with h\* > 0 and a known optimal set.
  - A state counts as correct when every successor that minimises the predicted h is optimal.
  - Ties use a tolerance of 1e-5·max(1, |min|). The tie rate is reported.
- **Paired CIs.**
  - Method: `rcx.stats.seed_parent_bootstrap`. Instances are the resampled clusters; each instance's states are
    averaged per seed and then over seeds. 5,000 resamples.
  - Comparisons: GNN − MLP and GNN − best handcrafted.
  - Scopes: each OOD size and pooled OOD, plus pooled ID as context only.
  - "Best handcrafted" is the handcrafted heuristic with the highest state-pooled policy accuracy *in that scope*,
    which is the strongest baseline and therefore conservative for any GNN claim.

### E08

- **Instances.** Seed 20261273, sizes {8, 10, 12, 14, 15}, 20 per size, full goals, exclusion-checked and distinct
  from the E07 test instances.
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
