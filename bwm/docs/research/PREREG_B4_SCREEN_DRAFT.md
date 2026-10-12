# PREREG B4-SCREEN (DRAFT): a one-arm futility screen for B4 arm (a), paired against item 7's R0 beam run

| | |
|---|---|
| Status | **DRAFT**, written 2026-10-11 ~20:45 CDT. Not frozen. Joint BWM-001 × RCX. BWM owns execution, the instances and the final text; RCX owns arm (a)'s module and co-signs (D-76a). |
| Written before | `runs/FREEZE_I7_*` exists and before any item-7 evaluation output exists (`runs/BW_SEEDS/` absent at writing; `BW_Pa_v2_s2`/`s3` not yet trained). |
| Registered by | `research/PREREG_ITEM7_ADDENDUM_A_DRAFT.md` §B (the item-7 addendum that names R0's beam run as this screen's control). Both are finalised (DRAFT dropped from the names), hashed into DECISIONS and co-signed before `scripts/bw_seeds.py freeze` runs. |
| Decision source | D-75.1 (user: one-turn B4 futility screen after item 7, then B6, then item 9); D-76a (RCX agreement: pins, call, hygiene, attribution rule, NO-GO floor); panel `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` (`final.gpu_sequence`, B4-screen step; `top_now_cpu` item 6). |
| Runner | `scripts/b4_screen.py` (draft; sha256 at drafting `2bb8c548…`; after the independent verifier's two fixes (§6 certificate `objs`, §7.1 R0-output guard) `cb2a7f63…`; recorded again by the freeze). Stub tests `tests/test_b4_screen.py` (24 pass; `8b00f546…` at drafting, `a25c1831…` after the verifier's fixes). |
| Tier | B: the treatment is an external CPU module (RCX's GNN value + misplaced_blocks) inside BWM's frozen LM planner. |

## 1. Question and role

**Question.** Inside the pinned B2 `plan_beam` with BWM's frozen F and proposer, does replacing the default child-ranking
key (unmet goal facts) with RCX's hybrid hook show enough signal on seeds_fresh-40 to justify the full B4 (about 7 GPU
turns, D-70 / P-14 follow-up)?

**Role.** This is a futility screen, not a test of a claim.

- **GO** unlocks the co-signed B4 primary (§8). It does **not** establish that the hook improves planning.
- **NO-GO** closes B4 at the screen, worded "not supported at screen". It does **not** establish that the hook has no
  effect: the screen has low power for effects below about +9 pp (§5.3).

**Why a screen.** The forensics (F-09) show most frozen-beam hard failures stall at an imagined state with exactly one
unmet goal fact, so the ranking key looks binding. But RCX's evidence (X-02, X-05) comes from GBFS with exact dynamics,
not from a proposer-driven beam with a learned F. One arm against an existing control costs about one turn; the full B4
costs about seven.

## 2. Fixed components and pins

### 2.1 Planner, models and arguments: exactly item 7's R0 S-beam

| Component | Value |
|---|---|
| Planner | `bwmn/blocks_planner.py` sha256 `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` (D-57/D-59/D-63), imported read-only, never patched. `bwmn/blocks.py` `b1e28ed5…`. |
| Arguments | `runs/FREEZE_I7_R0_beam.json` `args` (written by `bw_seeds.py freeze`) = `runs/FREEZE_B2_beam.json` (`412a6ee5…`) `args`: proposer lm, mode actions, K 8, W 4, E 24, R 12, M 80, P 1, max_depth 80, F lm delta, eval_seed 0. The runner refuses unless the R0 record's args equal FREEZE_B2_beam's and its adapter shas match (`check_r0_is_b2`). |
| F | `runs/BW_Fdelta_s0/adapter` (`2ed0bb00…`) |
| Proposer | `runs/BW_Pa_v2_s1/adapter` (`ee1c5aec…`). Pa_v2 only; no Pa_v3 cross in the screen. |
| Batch budgets | `blocks_eval.py` defaults (F 5,000, proposer 6,000 tokens); outputs do not depend on them (D-48c). |
| Shards | 4, instance index % 4: the same partition as R0 (`bw_seeds.BEAM_SHARDS`). |

### 2.2 The hook (arm a): the D-76a call, nothing else

```python
LearnedUnmet.from_file("runs/RCX-E07/main/models/gnn_s20261276.pt", invalid_policy="fallback", hybrid_lambda=1.0)
```

passed as `unmet=` to `plan_beam`. Value = GNN(state, goal) + 1.0 × misplaced_blocks for a consistent imagined state
(clipped to ±500); 1,000 + misplaced on a sanitised copy for an inconsistent one (the invalid-state penalty); 1e6 if
even that fails. Lower is closer. The hook is used only where `plan_beam` calls `unmet` (child ranking, the earliest
best state along a rollout, the best submitted prefix); solution detection is unchanged (`goal <= state`).

**Pins** (RCX worktree `rineya-complementary-worktree/complementary`, read-only):

| File | sha256 | Source |
|---|---|---|
| `rcx/bw/b4_hook.py` | `a0b4f88c32797feba95115d232f733b8dd47610c9e3018af6b90b297bfdeec81` | D-76a |
| `rcx/bw/models.py` | `4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52` | D-76a |
| `rcx/bw/encode.py` | `32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a` | D-76a |
| `rcx/bw/space.py` | `29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4` | D-76a |
| `runs/RCX-E07/main/models/gnn_s20261276.pt` | `9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04` | D-76a |
| `runs/RCX-E07/main/models/gnn_s20261276.json` (sidecar) | `2ccdb2798fc784327c153fe7a6dd5479b2646f0cb29314aad3610eed1aa4a892` | D-76a |
| `rcx/bw/bwm_import.py` | `84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e` | **BWM addition**: executed by the hook import (module only; `load()` is never called) |
| `rcx/bw/__init__.py` | `4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa` | **BWM addition** (docstring only) |
| `rcx/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) | **BWM addition** |
| `runs/RCX-E07/main/data/instances.json` | `1e39f9bbc25a7e857ccbfee97c996638fed63d606c334efef88fc3b7a32f4e31` | **BWM addition**: the GNN's train/val/test instances, used only by the pre-check (§4.3) |

All ten match at drafting (verified by the dry run). RCX is asked to confirm the four BWM additions at co-sign.

### 2.3 How the hook is loaded and checked (runner, GPU process)

- **Sha-verifying importer.** `rcx`, `rcx.bw` and `rcx.bw.*` are loaded by a meta-path importer that executes each
  file's bytes only if their sha256 equals the pin (no bytecode is read or written). An unpinned `rcx` module raises.
  The import refuses if any `rcx` package is already loaded. The hook module itself imports numpy only; torch and
  `models.py` load lazily inside `from_file`.
- **Order.** The planner and core are loaded first (as `blocks_eval.py` does), then the hook is built. The runner asserts
  that the host's CUDA view (`torch.cuda.is_available`, `device_count`, `CUDA_VISIBLE_DEVICES`) is unchanged afterwards
  (`models.py` patches them; `b4_hook` restores them). The hook runs on CPU at 1 thread and restores the thread count.
- **Identity check, at the start and the end of every shard** (`verify_hook`, via `describe()`): model and sidecar
  shas, `config_source == "sidecar json 'config'"`, the four D-76a code shas, `invalid_policy`, `hybrid_lambda`,
  `round_to is None`, the hook's own planner pin `18a5f22b…`, truthiness (`plan_beam` binds `unmet or default`).
- **Meter.** The hook is passed through `HookMeter`, a truthy pass-through that returns the hook's value unchanged and
  counts calls, penalty-band calls (value ≥ 1,000), failsafe-band calls and seconds, per instance. The dry run checks
  pass-through equality.

## 3. Arm and control

| | Arm **a** (run by this screen) | Control **R0** (item 7's run, not re-run) |
|---|---|---|
| Run | `runs/BW_B4_SCREEN/a_beam/` (4 shards, merged by `bw_run_tests.merge`) | `runs/BW_SEEDS/R0_beam/` |
| `unmet` | the hook (§2.2) | default `len(goal − state)` = B4 arm **c** |
| Everything else | identical (§2.1) | identical |

**Pairing.** The same 40 ids (`seeds_fresh/<index>`), the same CRN seed function `_seed(eval_seed, id, "<round>.<slot>",
j)`, the same shard partition. Round 0 is identical in both arms by construction (same initial state, same proposals,
same F answers, and solutions are detected without the hook), so every instance R0 solves in its first round is a tie;
discordance can arise only on instances that survive round 1.

**Control = c, certified as c+pen.** B4's primary contrasts are against c+pen (the default key plus RCX's invalid-state
penalty). On a run where no F-accepted imagined state is inconsistent, c+pen makes exactly the same choices as c. The
analysis checks this on R0's caches with the hook's own `is_consistent` (§6, certificate). If it fails, the control is
labelled c (not c+pen) in the report; the full B4 runs c+pen in any case.

**F cache.** The arm starts cold (it does not reuse R0's caches). This keeps the arm's F audit independent and makes the
cross-run cache check (§6) meaningful. It costs about 10-20 GPU-minutes (F is roughly a sixth of beam time; the keys
shared with R0 are a fraction of that).

## 4. Instances

### 4.1 The set

`runs/BW_val_large/instances_seeds_fresh.json`, sha256 `6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc`
(D-66): n = 40, seed 20261014, mirror generator, the hard set's block distribution stratified (6:1, 8:3, 9:5, 10:5,
11:4, 12:5, 13:6, 14:8, 15:3). Excluded at generation: hard-set renaming signatures and the invariants of hard,
val_large, mirror, b3fresh and the 240 distillation problems (637). Loaded exactly as `blocks_eval.py`'s `_json_set`.

### 4.2 Hygiene (D-76a)

RCX excluded seeds_fresh from the GNN's training and selection data (`rcx/bw/excl.py` lists
`instances_seeds_fresh.json` as an extra reserved file).

### 4.3 Registered pre-check: goal classes the GNN was trained on

**Rule.** An instance is dropped from the paired analysis (both arms) iff its goal renaming class (RCX's `goal_key`:
`chains:<n>:<sorted goal-tower heights>`, exact for on-only disjoint-chain goals) is a goal class of the GNN's
training or validation (selection) instances (`runs/RCX-E07/main/data/instances.json`, splits `train` and `val`).
The thresholds of §5 are unchanged by a drop. The runner re-implements `goal_key` (RCX's version calls `BI.load()`,
which refuses inside a BWM process) and refuses unless it agrees with the stored `goal_key` on every E07 row.

**Computed at drafting** (CPU, read-only, goal structure only; `b4_screen.py precheck`, scratch copy
`%TEMP%/claude/bwm_d75_scratch/C/precheck/PRECHECK.json`):

- agreement with RCX's stored keys 103/103; 28 train/val goal classes, all at 4-7 blocks (4: 3, 5: 4, 6: 7, 7: 14);
- seeds_fresh instances at ≤ 8 blocks: `seeds_fresh/0` (6 blocks, `chains:6:1,1,4`: an E07 **test** class, not
  train/val, kept), `/1` (`chains:8:1,2,5`), `/2` (`chains:8:1,7`), `/3` (`chains:8:1,3,4`); no 8-block class was
  trained;
- **drops: none; n = 40.**

The freeze stage re-runs the pre-check and writes `runs/BW_B4_SCREEN/PRECHECK.json`; the analysis uses that file.

### 4.4 Exposure

seeds_fresh-40 is item 7's reserved set. After this screen it has been used by four S-beam systems (R0, R1, R2, a) and
three S-direct systems. Nothing in item 7 is selected using the screen's result, and the screen never touches hard-110.

## 5. Decision rule (fixed in advance)

### 5.1 Paired table

Success = the VAL verdict (`val` field; `bw_seeds.py analyze` reads it the same way). On the n paired instances
(40 minus any §4.3 drops): g = a-only successes, l = R0-only successes.

### 5.2 Rule

> **GO iff g − l ≥ 2** (the hook arm solves at least two more of the paired 40 than R0 does). Otherwise NO-GO.

- **Why ≥ 2 (BWM decision, 2026-10-11, before FREEZE_I7_*):**
  - It sits between RCX's D-76a floor (NO-GO iff net ≤ 0) and the panel's net ≥ 3 with sign p ≤ 0.20.
  - The verifier's simulation showed the stricter rule kills a +9 pp effect 36-49% of the time.
  - A screen exists to avoid spending about 7 turns on nothing, and a false GO costs GPU time but not validity,
    because the full B4 has its own registered test.
- The one-sided sign p and the two-sided McNemar p are reported but do not gate.
- Net ≤ 1 is always NO-GO.
- Binary: no "inconclusive" branch, no extension of n, no second look.

**Attribution (D-76a).** If more than 5% of hook calls return a penalty-band value (≥ 1,000), a GO is reported as "not
credited to the learned critic alone", and c+pen is mandatory in the full B4. The call-level share decides; the
evaluation-level share (`fallbacks / evaluations`) is reported beside it. Hook error counters (`failsafe`,
`model_errors`, `encode_errors`, `nonfinite`, `clipped`) are reported with the decision; they are part of the
registered module's behaviour and do not change it.

### 5.3 Why this rule (operating characteristics)

**Revision D-77 (2026-10-11, before FREEZE_I7_*): the registered rule is now net ≥ 2.** P(GO) on the paired 40, from
`b4_screen.oc_table` (exact enumeration). Columns 2 and 3 give P(GO) under the registered rule and under the earlier
draft's rule, net ≥ 3 with one-sided sign p ≤ 0.20:

| Scenario | net ≥ 2 (registered) | earlier draft | B4 primary power (n = 110, α 0.025) |
|---|---:|---:|---:|
| null, 5% discordant | 0.132 | 0.037 | 0.002 |
| null, 10% discordant | 0.220 | 0.090 | 0.006 |
| null, 20% discordant | 0.296 | 0.120 | 0.007 |
| null, 30% discordant | 0.332 | 0.134 | 0.008 |
| harm −5 pp | 0.070 | 0.020 | 0.000 |
| +5 pp, low loss | 0.590 | 0.365 | 0.196 |
| +9 pp, low loss | 0.822 | 0.635 | 0.564 |
| +9 pp | 0.783 | 0.558 | 0.396 |
| +13.6 pp (H2 discordance) | 0.920 | 0.774 | 0.765 |
| +20 pp | 0.980 | 0.916 | 0.958 |

A screen should rarely kill a real effect. A false GO costs about 7 GPU turns of full B4, whose own registered test
then guards validity. The earlier text of this section (kept below for the record) analysed the stricter rule.

#### Earlier analysis (stricter rule, superseded by D-77)

P(GO) at n = 40 for candidate rules, by the true per-instance rates of a-only and R0-only outcomes (exact multinomial
enumeration; `python -B scripts/b4_screen.py oc`, scratch copy `%TEMP%/claude/bwm_d75_scratch/C/oc/b4s_oc.json`):

| Scenario (P a-only, P R0-only) | **Registered**: net ≥ 3 & p₁ ≤ 0.20 | Panel: net ≥ 3 | Task draft: (net ≥ 4 & L ≤ 1) or p₂ < 0.10 | net ≥ 4 | D-76a floor: net ≥ 1 | Context: B4 primary power, n = 110, α 0.025 |
|---|---|---|---|---|---|---|
| null, 5% discordant (.025, .025) | **0.037** | 0.037 | 0.008 | 0.008 | 0.348 | 0.002 |
| null, 10% discordant (.05, .05) | **0.090** | 0.101 | 0.035 | 0.039 | 0.398 | 0.006 |
| null, 20% discordant (.10, .10) | **0.120** | 0.186 | 0.043 | 0.106 | 0.429 | 0.007 |
| null, 30% discordant (.15, .15) | **0.134** | 0.234 | 0.030 | 0.155 | 0.442 | 0.008 |
| harm −5 pp (.05, .10) | **0.020** | 0.029 | 0.007 | 0.010 | 0.148 | 0.000 |
| +5 pp, low loss (.075, .025) | **0.365** | 0.385 | 0.201 | 0.215 | 0.778 | 0.196 |
| +5 pp (.10, .05) | **0.342** | 0.412 | 0.191 | 0.262 | 0.732 | 0.128 |
| +9 pp, low loss (.115, .025) | **0.635** | 0.678 | 0.448 | 0.505 | 0.919 | 0.564 |
| +9 pp (.14, .05) | **0.558** | 0.655 | 0.334 | 0.509 | 0.878 | 0.396 |
| +9 pp, high loss (.165, .075) | **0.514** | 0.640 | 0.244 | 0.510 | 0.847 | 0.315 |
| +13.6 pp, P-14's H2 discordance (.182, .045) | **0.774** | 0.851 | 0.525 | 0.753 | 0.961 | 0.765 |
| +20 pp (.25, .05) | **0.916** | 0.958 | 0.711 | 0.921 | 0.991 | 0.958 |

Reasons for the registered rule:

1. **A bounded false GO at any discordance.** Under any null (equal a-only and R0-only rates), the exact one-sided test
   keeps P(GO) ≤ 0.20 conditional on the number of discordant pairs, so at every discordance level; with the net ≥ 3
   floor the realised rate is 0.04-0.13 for 5-30% discordance. "Net ≥ 3" alone (the panel's rule) has no such bound
   and climbs to 0.19-0.23 at 20-30% discordance. High discordance is plausible here: a new ranking key changes which
   beam children survive on every instance not solved in round 1, so instances can flip both ways.
2. **Power where the full B4 has power.** The full B4 primary (n = 110, Holm m = 2, first step α 0.025) has power
   0.32-0.56 at +9 pp, 0.77 at an H2-sized effect and only 0.13-0.20 at +5 pp. A screen should pass the effects the
   primary can detect: the registered rule passes 0.51-0.64 at +9 pp and 0.77 at +13.6 pp. For an H2-sized effect, the
   chance of passing both the screen and the primary is about 0.77 × 0.77 ≈ 0.59 (independent instance sets).
3. **Why not the task draft's rule** [(net ≥ 4 and at most 1 loss) or two-sided p < 0.10]. With 2 R0-only outcomes it
   needs 9 a-only outcomes (net 7). It would kill an H2-sized effect 47% of the time (P(GO) 0.525) and a +9 pp effect
   55-76% of the time. Its false-GO rate (≤ 0.043) is lower than needed: a false GO costs GPU turns but no false claim,
   because the full B4 primary is a separate, Holm-controlled test on fresh instances.
4. **Why not the D-76a floor alone** (net ≥ 1): it would GO 35-44% of the time under the null.

Costs of the two errors, for the record: a false GO spends about 7 GPU turns on a full B4 that then most likely reports
NOT SUPPORTED; a false NO-GO forgoes a tier-B improvement of the planner's ranking key (B6, the panel's top capability
lever, proceeds either way).

## 6. Report-only outputs (no reading attached)

`runs/BW_B4_SCREEN/B4S_REPORT.{json,md}`, written by `b4_screen.py analyze`:

- per arm: success with Clopper-Pearson 95% CI; the paired table, net, one-sided sign p, two-sided exact McNemar p;
- strata 6-10 and 11-15 blocks (paired counts and McNemar p, descriptive) and success by block count;
- failure taxonomy per arm: solved / dead_end / budget / verified-but-invalid; verified-plan precision; VAL
  disagreements with the executor;
- cost per instance: proposer tokens, F model queries (cold caches in both arms), rounds, expansions (arm; R0 has the
  mean only), wall time; hook seconds;
- hook counters and inconsistency reasons, call-level and evaluation-level penalty shares;
- anytime curves: success within r rounds, r = 1..12 (exact, D-48c/D-50: the first r rounds do not depend on any later
  cap) and success within T proposer tokens, T = 2k-16k (approximate);
- **R0 failures stalled at 1 unmet goal fact** (D-76a): the number of unmet goal facts in the executor's state after
  each R0 failure's submitted prefix (the best node), its distribution, and how many of the stalled-at-1 instances arm a
  solves;
- **certificate c ≡ c+pen on R0**: every F-accepted prediction in R0's beam caches checked with the hook's
  `is_consistent`, with the parent state's blocks as `objs` (counts and reasons). The cache does not record the goal,
  and the hook, called with the goal, penalises a goal block that vanished from the imagined state; passing the
  parent's blocks catches every vanished block (and flags a new name as `unknown_block`), so the check is at least as
  strict as the penalty;
- **F cache cross-check**: every (state, action) key answered in both runs must have the same F text (F is frozen and
  memoised; a check of the batch-invariance that pairing and the round curves rely on). Differences are reported, not
  reconciled;
- the arm's F audit with the FREPLAY definitions (set `b4s_a_beam`; item-7 addendum A §A.1).

Before anything is parsed, the analysis writes `runs/BW_B4_SCREEN/ANALYSIS_INPUTS.json` (sha256 of every plans,
summary, hook and cache file it reads).

## 7. Execution and cost

### 7.1 Commands (all `python -B`, from `native_v1/`)

| Step | When | Command | Writes |
|---|---|---|---|
| freeze (CPU, seconds) | right after `bw_seeds.py freeze`, before `bw_seeds.py run` | `scripts/b4_screen.py freeze --prereg research/PREREG_B4_SCREEN.md --addendum research/PREREG_ITEM7_ADDENDUM_A.md` | `runs/FREEZE_B4S_a_beam.json`, `runs/BW_B4_SCREEN/PRECHECK.json` |
| run (GPU) | the BWM GPU turn after item 7's analysis (D-75.1) | `scripts/b4_screen.py run` | `runs/BW_B4_SCREEN/a_beam/shard_{0..3}of4/`, merged `a_beam/` |
| analyze (CPU) | after both R0 and a exist | `scripts/b4_screen.py analyze` | `runs/BW_B4_SCREEN/B4S_REPORT.{json,md}`, `ANALYSIS_INPUTS.json` |

- **Freeze record** (refuses to overwrite): the R0 args verbatim, the hook spec, the control's freeze sha, every pin of
  §2, the rule constants, the pre-check result, the shas of this prereg (final), the addendum, PREREG_ITEM7_SEEDS
  (`956a0dc4…`) and the runner. It refuses if any R0 beam output exists (any file under `runs/BW_SEEDS/R0_beam/`,
  including a running shard's `progress.txt` or `fcache.jsonl`), unless `--allow-after-r0` is passed, in which
  case the late freeze is recorded with the list of R0 files present. It refuses a file named DRAFT unless
  `--allow-draft`.
- **Run guards:** every shard re-verifies the freeze (R0 freeze sha and args, runner sha, every pin, adapter shas) and
  refuses if its `summary.json` exists (run once). `plans.jsonl` is written before `summary.json` (the completion
  marker). An interrupted shard restarts from scratch with identical seeds and a warm F cache (as item 7). No retries
  under the same id otherwise.
- **Lead's queue change (not made by this draft):** insert a `b4s-freeze` job between bundleI's `i7-freeze` and
  `i7-run` (marker `runs/FREEZE_B4S_a_beam.json`, est 30 s). If it fails, item 7 proceeds unchanged (the release gate
  is not blocked by the screen), and the screen is either frozen late with `--allow-after-r0` or dropped; the choice is
  recorded in DECISIONS before the screen's GPU run.

### 7.2 GPU cost estimate

- **Expected: about 3.5 GPU-h (3.3-3.7), one BWM turn.** The frozen beam's planning time was 291-316 s per instance
  on hard-110 (316), mirror-110 (294) and H2-hard (291), which have the same block mix as seeds_fresh (`wall_s`, which
  excludes model loading and VAL: add a few minutes per shard). 40 instances in 4 shards of about 50-55 min. A rough
  fit over those 24 shards (wall ≈ 978 s per shard + 22 s per 1k proposer tokens + 264 s per 1k F queries; residuals
  up to ~1.1 ks per shard) gives 3.6 h at B2's per-instance mix.
- **Pessimistic: about 6 GPU-h.** If the hook drove every instance to the E24 budget (budget failures used about 15.0k
  proposer tokens vs an 8.0k mean on hard), one shard would spill into the next turn. Shards resume.
- **Hook CPU inside the GPU process: under a minute per run (estimate; not measured, because torch may not be imported
  in this CPU track).** About 150-200 unique imagined states per instance (B2-hard: 187 unique F keys per instance,
  ~80% legal), one single-thread forward of a 124k-parameter GNN each (estimated 1-5 ms), plus an LRU cache. Measured
  and reported (`HookMeter` seconds).
- **Memory:** CPU tensors only; no VRAM change; host RAM +~50 MB.
- **What the screen saves:** NO-GO saves about 7 GPU turns of full B4 (5 arms, about 2 turns each, minus the screen).

## 8. Consequences

- **GO:** the co-signed B4 primary runs as drafted (D-70; P-14 follow-up): P1 a vs c+pen and P2 b vs c+pen, Holm m = 2,
  fresh seed-20261410 set, n = 110, frozen W4/E24, Pa_v2; Pa_v3 cross secondary; arm b's V-LoRA trained first. If the
  attribution rule fired, the GO wording says so and c+pen is mandatory (it is already the comparator).
- **NO-GO:** "B4 arm (a) not supported at screen (seeds_fresh-40, paired vs R0, pre-set futility rule)". Arms a0 and b
  are not run under B4; B6 proceeds (D-75.1). A different module needs a new prereg.
- **Either way:** the result and the per-item records (arm, n, success, F queries, expansions, plan, hook counters) go
  to RCX (D-70 §7).

## 9. Not claimed; limits

- Tier B: the arm uses an external module, and the GNN was trained on exact h* (simulator labels), so arm a is not
  simulator-free and is not a model-capability result for the LM planner.
- One run, one seed, n = 40: low power below about +9 pp (§5.3).
- seeds_fresh is a generated hard-style set, not PlanBench; Pa_v2 only; Blocksworld only.
- The screen cannot separate "learned value" from "misplaced_blocks"; that is arm a0 of the full B4.
- The control is item 7's R0 run, executed before the arm in a different turn. Pairing rests on the D-48c/D-50
  invariances (per-instance CRN, batch-independent sampling, frozen memoised F), which §6's cache cross-check tests.

## 10. Dry-run evidence (CPU, no model, no torch)

`python -B scripts/b4_screen.py --dry-run` (and `--inconsistent 0`), 2026-10-11 ~20:40 CDT. The pinned planner bytes
run with a stub `bwmn.lm`: a scripted F (exact on legal actions; F-09-style "legal" answers on illegal actions, which the
support check rejects; an optional share of inconsistent answers) and a noisy teardown-and-build proposer with CRN
seeds. The arm is the pinned `LearnedUnmet` with only its GNN forward replaced by 0.0. 8 generated instances (5-8
blocks, dry-run seed 7411; seeds_fresh is hashed, never planned on), with the frozen B2 beam args.

- All pins verified; the RCX files executed are exactly `rcx/__init__.py`, `rcx/bw/__init__.py`, `b4_hook.py`,
  `bwm_import.py`, `encode.py`, `space.py` (not `models.py`); torch, transformers and peft were not imported; no
  bytecode appeared in the RCX tree.
- The arm in 4 shards merged by `bw_run_tests.merge` equals the unsharded arm instance for instance (plans, verified,
  rounds, tokens; with the exact stub, hook calls 2,477 both ways).
- The meter is a pure pass-through. The freeze-record preview's args equal what `bw_seeds.py freeze` writes for R0.
- With an exact stub F: certificate holds, 0 penalty-band calls, F cache cross-check 0 differences (446-469 common
  keys), verified-plan precision 1.0, VAL agrees with the executor. With 5% inconsistent stub answers (the default):
  certificate fails and 15.4% of hook calls are penalty-band (the attribution path works).
- The dry-run outcomes (exact stub: control 6/8, arm 5/8) mean nothing about B4.
- Tests: `python -B -m pytest tests/test_b4_screen.py -q -p no:cacheprovider` → 24 passed; with
  `tests/test_domain_planner.py` in either order, 52 passed.

## 11. For RCX's co-signature

1. The four BWM pin additions (§2.2) and the loader (§2.3).
2. The pre-check rule and its drafting result (§4.3: no drops).
3. The decision rule and attribution rule (§5); the D-76a floor is implied.
4. Control = item 7's R0 run, with the c ≡ c+pen certificate (§3).
5. Outputs sent to RCX (§8).
