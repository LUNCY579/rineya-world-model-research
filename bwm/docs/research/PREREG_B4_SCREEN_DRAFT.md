# PREREG B4-SCREEN (DRAFT): a one-arm futility screen for B4 arm (a), paired against item 7's R0 beam run

| | |
|---|---|
| Status | **DRAFT**, written 2026-10-11 ~20:45 CDT; revised 2026-10-12 ~02:30 CDT for RCX's co-sign review (`rineya-complementary-worktree/complementary/experiments/RCX-B4/COSIGN_REVIEW_B4_SCREEN.md`, commit 2657079, sha256 `9dd352ee…`: "co-sign with edits", 13 required edits). BWM accepted all 13 as D-79, as recorded in RCX's `complementary/COLLABORATION.md` (2026-10-12 ~01:50, commit f3031f6). BWM's DECISIONS.md D-79 entry does not carry it yet; the lead adds it there. Not frozen. Joint BWM-001 × RCX. BWM owns execution, the instances and the final text. RCX owns arm (a)'s module and co-signs (D-76a). The co-signature binds to the final shas in this header and to nothing else (§11). |
| Written before | `runs/FREEZE_I7_*` exists and before any item-7 evaluation output exists. Re-checked at the revision (2026-10-12 01:41 CDT): no `runs/FREEZE_I7_*`, no `runs/BW_SEEDS/`, `BW_Pa_v2_s2` at step 300/1000, `BW_Pa_v2_s3` not trained. |
| Registered by | `research/PREREG_ITEM7_ADDENDUM_A.md` §B (drafted as `research/PREREG_ITEM7_ADDENDUM_A_DRAFT.md`), the item-7 addendum that names R0's beam run as this screen's control. Both are finalised (DRAFT dropped from the names), hashed into DECISIONS and co-signed before `scripts/bw_seeds.py freeze` runs. |
| Decision source | D-75.1 (user: a one-turn B4 futility screen after item 7, then B6, then item 9). D-76a (RCX agreement: pins, call, hygiene, attribution rule, NO-GO floor). D-77 (BWM: GO iff net ≥ 2). D-79 as sent to RCX (the 13 co-sign edits accepted; edit-5 route: B4 declares the screen's outcome as known and forbids any change to arm (a)). Panel `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` (`final.gpu_sequence`, B4-screen step; `top_now_cpu` item 6). |
| Runner | `scripts/b4_screen.py` with stub tests `tests/test_b4_screen.py` (60 pass). Their sha256s are in the next two rows, and they replace every earlier runner and test sha, including the bytes RCX reviewed (`14ef3d99…`, `d09301e7…`). `b4_screen.py freeze` refuses unless the runner and tests on disk are exactly these bytes and the addendum names the same runner sha, in the late-freeze path as well (edit 2). The runner's canary sha is still a placeholder (§2.4). When RCX sends it, BWM writes it into the runner, and these two rows change once more, before RCX co-signs. |
| Runner sha256 | `3ca01448fbffb91d42c6167a0a90a3146366e401c7b4eca72cae54498f404489` |
| Tests sha256 | `0349b70b8a810432131de920bc4103ef699e38ba548002361b478d8f470fe440` |
| Tier | B: the treatment is an external CPU module (RCX's GNN value + misplaced_blocks) inside BWM's frozen LM planner. |

## 0. Outcomes known at registration (co-sign edit 12)

**Known before this text was written** (none of it is an item-7 or seeds_fresh outcome):

- **B2, frozen S-beam:**
  - hard: 73/110 (P-08), S-direct 5/110; mirror: 61/110 (P-10).
  - Round-1 solves: 0/110 on hard and 1/110 on the mirror (recomputed from `runs/BW_TEST_B2/{hard,mirror}_beam/plans.jsonl`).
- **H2 (P-14):** Pa_v3 in the same beam solved 88/110 on hard. Against B2 S-beam it was 20 H2-only and 5 B2-only, a
  discordance of 25/110 = 0.227.
- **F-09:** 28 of B2-hard's 37 failures end with exactly one unmet goal fact in the executor's state after the
  submitted prefix (recomputed; mirror 39 of 49).
- **FREPLAY (P-15):**
  - 0 false-legal-accepted keys and 0 wrong states in every set; false-illegal 6 (B2-hard), 1 (mirror) and 5 (H2-hard).
  - The support-check share R2 is 13.0-27.8%, which triggers the wording consequence.
  - R1 bounds: [72, 73] on B2-hard and [87, 88] on H2-hard.
- **RCX:**
  - R-04 (X-02): the 124k-parameter GNN transfers out of distribution.
  - E07b VAL-B/TEST-B (X-05): VAL-B selected A3, the hybrid, with λ = 1; TEST-B was CLOSED, with hybrid 55/75 vs
    misplaced_blocks 54/75 at 2,000 expansions and 39 vs 19 at 200.
  - E11 (X-07).
  - All of these use GBFS with exact dynamics, not this beam.
- **This screen's own work:** the stub dry runs (§10) and the D-77 operating-characteristics simulation (§5.3).

**Not known:**
- Any item-7 output (no freeze record, no evaluation).
- Any outcome of any system on seeds_fresh. Only seeds_fresh's goal structure was read, for the pre-check and the
  hygiene check (§4.2-4.3).

**State at the 2026-10-11 21:54 reboot (I-18 / D-79):**
- `BW_Pa_v2_s2` was at step 300/1000, checkpointed every 50 steps; bundleI resumes it from the checkpoint after RCX's
  GPU-idle window.
- `BW_Pa_v2_s3` is not trained.
- R1 and R2 cannot exist yet, so no item-7 system can have been evaluated.

## 1. Question and role

**Question.** Inside the pinned B2 `plan_beam`, with BWM's frozen F and proposer, does replacing the default
child-ranking key (unmet goal facts) with RCX's hybrid hook show enough signal on seeds_fresh-40 to justify the full B4
(about 7 GPU turns, D-70 / P-14 follow-up)?

**Role.** This is a futility screen, not a test of a claim.

- **GO** unlocks the co-signed B4 primary (§8). It does **not** establish that the hook improves planning, and the
  screen's net and p are not effect estimates (§8.1).
- **NO-GO** closes B4 at the screen, worded "not supported at screen". It does **not** establish that the hook has no
  effect: the screen has low power for effects below about +9 pp (§5.3).
- **VOID** (§2.4) and **INSTRUMENT FAULT** (§3) are not decisions. They are registered validity outcomes.

**Why a screen.**
- The forensics show that most frozen-beam hard failures stall one goal fact short (F-09: 28/37), so the ranking key
  looks binding.
- But RCX's evidence (X-02, X-05) comes from GBFS with exact dynamics, not from a proposer-driven beam with a learned F.
- One arm against an existing control costs about one turn; the full B4 costs about seven.

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

**The rest of the paired code path is pinned (co-sign edit 6).** R0 runs through these files; the arm runs through them
too. They are enforced (not only recorded) by the freeze and by every run-shard and analyze stage:

| File | sha256 |
|---|---|
| `bwmn/lm.py` (generation; the planner's `lm`, also checked on the imported module) | `88a1e3a26b518cebd17e0b8197d1dd10872d549a4a27a69744ed59266895c0ac` |
| `bwmn/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| `scripts/blocks_eval.py` (R0's driver) | `b3666424151c78368500535d2571f0eb5b9e546b11764696d99f427a7794acb6` |
| `scripts/bw_run_tests.py` (`merge`, used for R0 and for the arm) | `de7022d1eacebab271f7624e962d40d58aeee435494da079a622a03f80a34542` |
| `scripts/bw_seeds.py` (item 7's runner) | `b783a1916bce655cf93e3d55bedcb276dae3dab2bbdc214e00ff633c828f3b08` |

**Library and driver versions (edit 6).** R0 and arm a run in different turns, so the freeze records the versions of
torch, transformers, peft, accelerate, tokenizers, safetensors and numpy, plus Python and the NVIDIA driver:
- The package versions come from package metadata, without importing anything.
- The driver version comes through NVML via ctypes; no `nvidia-smi` process is spawned.
- The screen freeze immediately precedes R0's run in the same bundle.
- Every arm shard refuses on any change, including when the loaded `torch.__version__` differs.

### 2.2 The hook (arm a): the D-76a call, nothing else

```python
LearnedUnmet.from_file("runs/RCX-E07/main/models/gnn_s20261276.pt", invalid_policy="fallback", hybrid_lambda=1.0)
```

It is passed as `unmet=` to `plan_beam`. Values (lower is closer):

| State | Value |
|---|---|
| Consistent imagined state | GNN(state, goal) + 1.0 × misplaced_blocks, clipped to ±500 |
| Inconsistent state (the invalid-state penalty) | 1,000 + misplaced_blocks on a sanitised copy |
| Encode, model or nonfinite error on a consistent state (a critic error) | 1,000 + misplaced_blocks on a sanitised copy |
| Even that fails | 1e6 (failsafe) |

**Where the hook acts (RCX acknowledgement, co-sign review).** `plan_beam` uses `unmet` at exactly three places, and
the hook acts at all three. This is how the pinned planner uses `unmet`; the planner is not edited.
1. **Child ranking:** each child's key (u, prefix length, candidate rank), and the beam selection.
2. **The earliest-best truncation point along a rollout:** `t_best = argmin (um[t], t)`. This includes the initial
   best, `unmet(init, goal)`, called once per instance.
3. **The prefix that unsolved instances submit:** the running best (u, prefix length).

Solution detection is unchanged (`goal <= state`). `B4_HOOK_NOTES.md` lists the same three sites, and RCX confirms them
for this text.

**The instance registry is off.** The D-76a call passes no `instances=`, and `verify_hook` checks that the registry
has 0 goals. Consequences:
- `unknown_block` is never flagged.
- RCX's known limitation: a *non-goal* block that vanished from an imagined state is not flagged either
  (`b4_hook.py:206-216`), so such a state is scored by the model rather than penalised.
- §6 reports how many keys of this kind the arm met. The count is report-only and does not change the attribution
  rule.

**Pins** (RCX worktree `rineya-complementary-worktree/complementary`; copied at the freeze into the screen's own
snapshot, §2.3):

| File | sha256 | Source |
|---|---|---|
| `rcx/bw/b4_hook.py` | `a0b4f88c32797feba95115d232f733b8dd47610c9e3018af6b90b297bfdeec81` | D-76a |
| `rcx/bw/models.py` | `4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52` | D-76a |
| `rcx/bw/encode.py` | `32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a` | D-76a |
| `rcx/bw/space.py` | `29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4` | D-76a |
| `runs/RCX-E07/main/models/gnn_s20261276.pt` | `9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04` | D-76a |
| `runs/RCX-E07/main/models/gnn_s20261276.json` (sidecar) | `2ccdb2798fc784327c153fe7a6dd5479b2646f0cb29314aad3610eed1aa4a892` | D-76a |
| `rcx/bw/bwm_import.py` | `84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e` | BWM addition: executed by the hook import (module only; `load()` is never called) |
| `rcx/bw/__init__.py` | `4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa` | BWM addition (docstring only) |
| `rcx/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) | BWM addition |
| `runs/RCX-E07/main/data/instances.json` | `1e39f9bbc25a7e857ccbfee97c996638fed63d606c334efef88fc3b7a32f4e31` | BWM addition: the GNN's train/val/test instances, used only by the pre-check (§4.3) |
| `experiments/RCX-B4/canary/B4_HOOK_CANARY.json` | **placeholder**: RCX sends the sha after its step A (generator `experiments/RCX-B4/make_canary.py`, commit c45ca0a) | edit 8: golden values (§2.4) |

RCX recomputed all ten D-76a and BWM-addition pins at review and found that they match. That includes the four BWM
additions.

### 2.3 How the hook is loaded and checked (runner, GPU process)

**Snapshot (edit 10).**
- At the freeze, every RCX file of §2.2 is sha-verified in RCX's worktree and copied into
  `runs/BW_B4_SCREEN/rcx_snapshot/`, where the copy is verified again. The freeze record lists the snapshot's shas.
- `run`, `run-shard` and `analyze` load and check RCX code and data only from the snapshot. RCX's worktree is
  therefore not bound after the freeze, and RCX's commitment to keep its pinned files byte-stable ends at the freeze.

**Sha-verifying importer.**
- `rcx`, `rcx.bw` and `rcx.bw.*` are loaded by a meta-path importer. It executes each file's bytes only if their
  sha256 equals the pin; no bytecode is read or written.
- An unpinned `rcx` module raises, and the import refuses if any `rcx` package is already loaded.
- The hook module itself imports numpy only. torch and `models.py` load lazily inside `from_file`.

**Order.**
- The planner and core are loaded first, as `blocks_eval.py` does; then the canary hook and the run hook are built.
- The runner asserts that the host's CUDA view (`torch.cuda.is_available`, `device_count`, `CUDA_VISIBLE_DEVICES`) is
  unchanged afterwards. `models.py` patches them, and `b4_hook` restores them.
- The hook runs on CPU at 1 thread and restores the thread count.

**Identity check, at the start and at the end of every shard** (`verify_hook`, via `describe()`). It checks:
- the model and sidecar shas, and `config_source == "sidecar json 'config'"`;
- the model file, which must be the snapshot's;
- the four D-76a code shas;
- `invalid_policy`, `hybrid_lambda` and `round_to is None`;
- the registry, which must have 0 goals;
- the hook's own planner pin, `18a5f22b…`;
- truthiness, because `plan_beam` binds `unmet or default`.

At the end, the hook's call counter must equal the meter's.

**Meter.** The hook is passed through `HookMeter`, a truthy pass-through that returns the hook's value unchanged.
Per instance it counts:
- calls;
- **penalty-band** calls (value ≥ 1,000: invalid state or critic error), split by the hook's own
  `is_consistent(state, goal)` into invalid-state and critic-error calls;
- failsafe-band calls;
- seconds.

The tests check that the meter passes values through unchanged, and that the arm path with
`HookMeter(lambda s, g: len(g - s))` reproduces the control's rows field for field (edit 13).

**`hook.json` per shard.** Each hook shard writes, before its `plans.jsonl` and `summary.json`:
- `describe()` at the start and at the end;
- the meter;
- the canary result;
- the RCX modules actually executed (the importer's `executed` map);
- `torch.__version__` and the recorded versions.

### 2.4 Canary and VOID rule (co-sign edit 8)

**Canary.** RCX supplies a sha-pinned golden file, `experiments/RCX-B4/canary/B4_HOOK_CANARY.json`. It holds about 20
consistent and 5 flagged-inconsistent (state, goal) probes:
- The probes are taken from RCX-E07 **test** instances, never from seeds_fresh.
- The expected values were computed with exactly the D-76a call, on a fresh hook, in file order.
- The file also records the expected hook counters after all probes.

The runner's `CANARY_SHA256` is a placeholder until RCX sends the sha. While it is a placeholder, `freeze` and
`run-shard` refuse.

**The check.** At the start of every hook shard, before any planning:
1. A **fresh** hook (the registered call, from the snapshot) scores every probe.
2. The shard asserts |value − expected| ≤ 1e-5 per probe.
3. It asserts that the counters after all probes equal the recorded ones.
4. It also checks that the file's hook call is the D-76a call and that its tolerance is no looser than 1e-5.

A canary failure stops the shard before planning, and the screen is **VOID**.
- The failure is logged in `ATTEMPTS.jsonl` and is final: no shard of that arm runs afterwards.
- `analyze` then writes the VOID report even though the arm is incomplete.
- The canary's own hook is separate from the run's hook, so the run hook's counters contain planning calls only. `b4_screen.py canary` runs the same check as a
CPU pre-flight; it is not part of the registration.

**VOID rule.** The screen is **VOID**, which is not NO-GO, if any of the following holds:
- The arm's hook counters, merged from the 4 shard records, show `model_errors`, `encode_errors`, `nonfinite` or
  `failsafe` > 0 (`failsafe` includes analysis errors). `analyze` recomputes the merge from the shard `hook.json` files
  and refuses if `hook_merged.json` differs from it.
- `model_evals` = 0.
- Any hook shard lacks a passed canary record.
- The R0 outputs the screen reads do not match addendum A.0's sha table (§7.1, edit 9).

When the screen is VOID:
- The report states VOID and the reasons, with no GO/NO-GO reading.
- A void screen needs a new prereg.
- VOID takes precedence over INSTRUMENT FAULT, which takes precedence over GO/NO-GO.

## 3. Arm and control

| | Arm **a** (run by this screen) | Control **R0** (item 7's run, not re-run) |
|---|---|---|
| Run | `runs/BW_B4_SCREEN/a_beam/` (4 shards, merged by `bw_run_tests.merge`) | `runs/BW_SEEDS/R0_beam/` |
| `unmet` | the hook (§2.2) | default `len(goal − state)` = B4 arm **c** |
| Everything else | identical (§2.1) | identical |

**Pairing.** Both arms use:
- the same 40 ids (`seeds_fresh/<index>`);
- the same CRN seed function `_seed(eval_seed, id, "<round>.<slot>", j)`;
- the same shard partition.

Round 0 is identical in both arms by construction: the same initial state, the same proposals and the same F answers,
and solutions are detected without the hook. So every instance solved in round 1 is a tie, and discordance can arise
only on instances that survive round 1. On hard, B2 solved 0/110 in round 1 (mirror 1/110), so almost every instance
is exposed to the ranking key.

**Pairing integrity (co-sign edit 7).**
- **R0's arguments.** Each R0 shard's `summary.json` records `blocks_eval.py`'s full argument namespace. It must equal
  the frozen S-beam configuration on `seeds_fresh`, with shard `i/4`, `--val`, the R0 freeze record and R0's shard
  directory, and have no other key. This is asserted before the GPU run (`run` and every `run-shard` refuse on a
  failure, so arm a is not run) and again in `analyze`, which then writes the INSTRUMENT FAULT report below.
- **Round-1 identity check.** An instance solved (verified) in round 1 by **either** arm must be solved in round 1 by
  both arms, with the same plan and the same proposer tokens.
- **Pre-registered consequence of a failure of either check: INSTRUMENT FAULT.**
  - No decision is read.
  - Arms c (the default key) and a are re-run in **one** GPU turn by this runner (`run --rerun`, writing
    `runs/BW_B4_SCREEN/rerun/{c,a}_beam/`), and analysed with `analyze --rerun` under the same rule.
  - If the re-run also fails the identity check, the screen closes without a reading, and a new prereg is needed.

**Control = c, certified as c+pen.**
- B4's primary contrasts are against c+pen: the default key plus RCX's invalid-state penalty.
- On a run where no F-accepted imagined state is inconsistent, c+pen makes exactly the same choices as c.
- The analysis checks this on R0's caches with the hook's own `is_consistent` (§6, certificate).
- If the check fails, the control is labelled c, not c+pen, in the report. The full B4 runs c+pen in any case.

**F cache.**
- The arm starts cold; it does not reuse R0's caches. This keeps the arm's F audit independent and makes the
  cross-run cache check (§6) meaningful.
- It costs about 10-20 GPU-minutes: F is roughly a sixth of beam time, and the keys shared with R0 are a fraction of
  that.

## 4. Instances

### 4.1 The set

`runs/BW_val_large/instances_seeds_fresh.json`, sha256 `6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc`
(D-66):
- n = 40, seed 20261014, mirror generator.
- The hard set's block distribution, stratified: 6:1, 8:3, 9:5, 10:5, 11:4, 12:5, 13:6, 14:8, 15:3.
- Excluded at generation: the hard set's renaming signatures, and the invariants of hard, val_large, mirror, b3fresh
  and the 240 distillation problems (637 invariants).
- Loaded exactly as `blocks_eval.py`'s `_json_set` loads it.

### 4.2 Hygiene (D-76a; method and verified results)

- **RCX's exclusions.** RCX excluded seeds_fresh from the GNN's training and selection data. `rcx/bw/excl.py` lists
  `instances_seeds_fresh.json` as an extra reserved file, and RCX's EXCLUSIONS.json (`2f211de3…`, which includes
  seeds_fresh `6fccb945…`) predates the E07 data and the model (RCX review).
- **No shared instances.**
  - RCX verified that seeds_fresh-40 shares no invariant and no instance with any RCX set: E07, E08, E07b, E11.
  - BWM's test confirms that no seeds_fresh (objs, init, goal) equals an E07 or VAL-B instance
    (`test_hygiene_goal_classes_shared_with_valb_but_no_instance`).
- **Shared goal *classes*.** Classes are shared, though instances are not (method: RCX's `goal_key`, re-implemented
  as `goal_chain_key` and checked against RCX's stored keys on all 103 E07 rows):
  - seeds_fresh ids 0, 2, 3, 9 and 21 share a goal class with E07 **test** instances;
  - seeds_fresh ids 18, 19, 20, 22 and 29 share a goal class with E07b's **VAL-B** selection set.
- **These ids are not dropped.** A class is not an instance, the GNN never trained on E07 test or VAL-B, and VAL-B
  selected only λ and the arm (A3), not the GNN's weights. The trained classes are handled by §4.3.
- **Two further class overlaps, for completeness** (same method, computed by BWM):
  - E07b's TEST-B shares classes with seeds_fresh ids 9, 11, 19, 20, 22, 29, 33, 35, 37, 38 and 39. TEST-B is a test
    set.
  - E07b's A1-n8 training data shares a class with id 1. That data trained arm A1's models, not the pinned GNN
    `gnn_s20261276` (E07 main).

### 4.3 Registered pre-check: goal classes the GNN was trained on

**Rule.**
- An instance is dropped from the paired analysis (in both arms) iff its goal renaming class is a goal class of the
  GNN's training or validation (selection) instances (`runs/RCX-E07/main/data/instances.json`, splits `train` and
  `val`).
- The class is RCX's `goal_key`, `chains:<n>:<sorted goal-tower heights>`, which is exact for on-only disjoint-chain
  goals.
- The thresholds of §5 are unchanged by a drop.
- The runner re-implements `goal_key`, because RCX's version calls `BI.load()`, which refuses inside a BWM process. The
  runner refuses unless its version agrees with the stored `goal_key` on every E07 row.

**Scope.** The pre-check covers only the GNN's own E07 train/val classes. B4's exclusion is broader: BWM's full recipe,
plus RCX's E07/E07b/E11 train and val goal classes at n ≤ 8, plus RCX's test instances. The screen uses only the
pre-check.

**Computed at drafting** (CPU, read-only, goal structure only; `b4_screen.py precheck`):
- agreement with RCX's stored keys: 103/103;
- 28 train/val goal classes, all at 4-7 blocks (4: 3, 5: 4, 6: 7, 7: 14);
- seeds_fresh instances at ≤ 8 blocks:
  - `seeds_fresh/0` (6 blocks, `chains:6:1,1,4`): an E07 **test** class, not train/val, so it is kept;
  - `/1` (`chains:8:1,2,5`), `/2` (`chains:8:1,7`) and `/3` (`chains:8:1,3,4`);
  - no 8-block class was trained;
- **drops: none; n = 40.**

The freeze stage re-runs the pre-check and writes `runs/BW_B4_SCREEN/PRECHECK.json`. The analysis takes the drop_ids
from the **frozen record** and refuses if `PRECHECK.json` differs from it (edit 1).

### 4.4 Exposure

- seeds_fresh-40 is item 7's reserved set. After this screen it has been used by four S-beam systems (R0, R1, R2, a)
  and three S-direct systems, plus c and a again if the re-run of §3 happens.
- Nothing in item 7 is selected using the screen's result, and the screen never touches hard-110.
- The screen's 40 are never pooled into B4 (§8.1).

## 5. Decision rule (fixed in advance)

### 5.1 Paired table

- Success = the VAL verdict (the `val` field, read the same way as `bw_seeds.py analyze` reads it).
- Every row of both arms must carry `val`; otherwise the analysis refuses (edit 1).
- On the n paired instances (40 minus any §4.3 drops): g = a-only successes, l = R0-only successes.

### 5.2 Rule

> **GO iff g − l ≥ 2** (the hook arm solves at least two more of the paired 40 than R0 does). Otherwise NO-GO.

- **Why ≥ 2** (BWM decision D-77, 2026-10-11, before `FREEZE_I7_*`):
  - It sits above RCX's D-76a floor (NO-GO iff net ≤ 0).
  - The stricter rule of the earlier draft (net ≥ 3 with one-sided sign p ≤ 0.20, **superseded**, §5.3) would have
    killed a +9 pp effect 36-49% of the time.
  - A screen exists to avoid spending about 7 turns on nothing. A false GO costs GPU time but not validity, because
    the full B4 has its own registered test.
- **No p-value is part of the rule (co-sign edit 3).** The one-sided sign p and the two-sided exact McNemar p are
  reported only, and the frozen rule has no p-value term.
- Net ≤ 1 is always NO-GO.
- Binary: no "inconclusive" branch, no extension of n, no second look.
- The rule constants (net ≥ 2, the 5% attribution threshold) and the pre-check drop_ids are read from the **frozen
  record** by `analyze` (edit 1). It refuses on any mismatch with the runner's constants or with `PRECHECK.json`, and if
  the frozen rule carries a p-value term.
- **Precedence:** VOID (§2.4) > INSTRUMENT FAULT (§3) > GO/NO-GO.

**Attribution (D-76a).**
- If more than 5% of hook calls return a value in the **penalty band (invalid state or critic error;
  value ≥ 1,000)**, a GO is reported as "not credited to the learned critic alone", and c+pen is mandatory in the full
  B4.
- Exactly 5% (for example 50 of 1,000 calls) does not trigger the rule. A test covers this boundary.
- The call-level share decides. Reported beside it (no reading):
  - the call-level split: invalid state / critic error / failsafe;
  - the evaluation-level split: invalid states / encode + model + nonfinite errors / failsafe, which includes analysis
    errors;
  - the evaluation-level share, (fallbacks + analysis_errors) / evaluations.
- On a non-void screen the critic-error count is 0 by the VOID rule, so the penalty band then holds invalid states
  only.

### 5.3 Why this rule (operating characteristics)

**Registered rule (D-77): net ≥ 2.** P(GO) at n = 40 is computed by exact enumeration (`b4_screen.oc_dgrid` and
`oc_table`; `python -B scripts/b4_screen.py oc`).

**Grid by discordance (co-sign edit 4).** Define d = P(a-only) + P(R0-only) and the effect e = P(a-only) − P(R0-only).
Where they overlap (d 0.10-0.30; null, +9 pp, +13.6 pp), the grid equals RCX's table from its two independent
enumerations exactly. A test checks this.

| d | null | +5 pp | +9 pp | +13.6 pp |
|---|---:|---:|---:|---:|
| 0.05 | 0.132 | 0.601 | n/a | n/a |
| 0.10 | 0.220 | 0.590 | 0.870 | n/a |
| 0.15 | 0.266 | 0.578 | 0.813 | 0.968 |
| 0.20 | 0.296 | 0.569 | 0.777 | 0.933 |
| **0.227 (H2 anchor)** | **0.308** | 0.565 | 0.762 | 0.917 |
| 0.30 | 0.332 | 0.558 | 0.731 | 0.881 |
| 0.40 | 0.354 | 0.550 | 0.703 | 0.844 |

**The null false-GO rate, stated plainly.**
- **The false-GO rate under the null is about 0.30-0.36 at plausible discordance** (d = 0.20-0.40; 0.308 at H2's
  discordance d = 0.227).
- It rises with d and is **at most 0.44 at any discordance** (supremum 0.437, at d = 1).
- High discordance is plausible here. B2 solves 0/110 hard instances in round 1, so almost every instance is exposed to
  the ranking key, and a new key changes which beam children survive on every one of them.
- This is the price of a screen that rarely kills a real effect. A false GO costs about 7 GPU turns of full B4, whose
  own registered, Holm-controlled test on fresh instances then guards validity.

**Scenario table.** The registered rule beside the superseded one, with the full B4 primary for context:

| Scenario (P a-only, P R0-only) | **net ≥ 2 (registered)** | superseded draft rule: net ≥ 3 & p₁ ≤ 0.20 | B4 primary power (n = 110, α 0.025) |
|---|---:|---:|---:|
| null, 5% discordant (.025, .025) | 0.132 | 0.037 | 0.002 |
| null, 10% discordant (.05, .05) | 0.220 | 0.090 | 0.006 |
| null, 20% discordant (.10, .10) | 0.296 | 0.120 | 0.007 |
| null, 30% discordant (.15, .15) | 0.332 | 0.134 | 0.008 |
| harm −5 pp (.05, .10) | 0.070 | 0.020 | 0.000 |
| +5 pp, low loss (.075, .025) | 0.590 | 0.365 | 0.196 |
| +5 pp (.10, .05) | 0.578 | 0.342 | 0.128 |
| +9 pp, low loss (.115, .025) | 0.822 | 0.635 | 0.564 |
| +9 pp (.14, .05) | 0.783 | 0.558 | 0.396 |
| +9 pp, high loss (.165, .075) | 0.755 | 0.514 | 0.315 |
| +13.6 pp, P-14's H2 discordance (.182, .045) | 0.920 | 0.774 | 0.765 |
| +20 pp (.25, .05) | 0.980 | 0.916 | 0.958 |

The superseded column is shown for comparison only.

#### Superseded analysis (the earlier draft's rule, net ≥ 3 with p₁ ≤ 0.20; superseded by D-77, not part of the rule)

Kept for the record. Every "registered" below refers to the earlier draft, **not** to this prereg. P(GO) at n = 40 for
candidate rules, by the true per-instance rates of a-only and R0-only outcomes (exact multinomial enumeration):

| Scenario (P a-only, P R0-only) | Superseded: net ≥ 3 & p₁ ≤ 0.20 | Panel: net ≥ 3 | Task draft: (net ≥ 4 & L ≤ 1) or p₂ < 0.10 | net ≥ 4 | D-76a floor: net ≥ 1 | Context: B4 primary power, n = 110, α 0.025 |
|---|---|---|---|---|---|---|
| null, 5% discordant (.025, .025) | 0.037 | 0.037 | 0.008 | 0.008 | 0.348 | 0.002 |
| null, 10% discordant (.05, .05) | 0.090 | 0.101 | 0.035 | 0.039 | 0.398 | 0.006 |
| null, 20% discordant (.10, .10) | 0.120 | 0.186 | 0.043 | 0.106 | 0.429 | 0.007 |
| null, 30% discordant (.15, .15) | 0.134 | 0.234 | 0.030 | 0.155 | 0.442 | 0.008 |
| harm −5 pp (.05, .10) | 0.020 | 0.029 | 0.007 | 0.010 | 0.148 | 0.000 |
| +5 pp, low loss (.075, .025) | 0.365 | 0.385 | 0.201 | 0.215 | 0.778 | 0.196 |
| +5 pp (.10, .05) | 0.342 | 0.412 | 0.191 | 0.262 | 0.732 | 0.128 |
| +9 pp, low loss (.115, .025) | 0.635 | 0.678 | 0.448 | 0.505 | 0.919 | 0.564 |
| +9 pp (.14, .05) | 0.558 | 0.655 | 0.334 | 0.509 | 0.878 | 0.396 |
| +9 pp, high loss (.165, .075) | 0.514 | 0.640 | 0.244 | 0.510 | 0.847 | 0.315 |
| +13.6 pp, P-14's H2 discordance (.182, .045) | 0.774 | 0.851 | 0.525 | 0.753 | 0.961 | 0.765 |
| +20 pp (.25, .05) | 0.916 | 0.958 | 0.711 | 0.921 | 0.991 | 0.958 |

The earlier draft gave these reasons for the superseded rule. They are recorded, not adopted:

1. **A bounded false GO at any discordance.** The exact one-sided test keeps P(GO) ≤ 0.20 conditional on the number of
   discordant pairs. With the net ≥ 3 floor, the realised rate was 0.04-0.13 for 5-30% discordance. "Net ≥ 3" alone
   climbs to 0.19-0.23. D-77 accepted a higher false-GO rate (above) in exchange for power.
2. **Power where the full B4 has power.** The full B4 primary (n = 110, Holm m = 2, first step α 0.025) has power
   0.32-0.56 at +9 pp, 0.77 at an H2-sized effect and only 0.13-0.20 at +5 pp. The superseded rule passed 0.51-0.64 at
   +9 pp; the registered rule passes 0.75-0.82.
3. **Why not the task draft's rule** [(net ≥ 4 and at most 1 loss) or two-sided p < 0.10]: it would kill an H2-sized
   effect 47% of the time.
4. **Why not the D-76a floor alone** (net ≥ 1): it would GO 35-44% of the time under the null.

**Costs of the two errors:**
- A false GO spends about 7 GPU turns on a full B4 that then most likely reports NOT SUPPORTED.
- A false NO-GO forgoes a tier-B improvement of the planner's ranking key. B6, the panel's top capability lever,
  proceeds either way.

## 6. Report-only outputs (no reading attached)

`runs/BW_B4_SCREEN/B4S_REPORT.{json,md}`, written once by `b4_screen.py analyze`. A second analysis refuses.

**Status and pairing:**
- the status (GO / NO-GO / VOID / INSTRUMENT_FAULT) with its wording;
- the round-1 identity check;
- per arm: success with a Clopper-Pearson 95% CI;
- the paired table, net, the one-sided sign p and the two-sided exact McNemar p (reported only);
- strata 6-10 and 11-15 blocks (paired counts and McNemar p, descriptive) and success by block count.

**Outcomes per arm:**
- the failure taxonomy: solved / dead_end / budget / verified-but-invalid;
- verified-plan precision;
- **unverified VAL successes**, so that a GO driven by the choice of submitted prefix is visible;
- VAL disagreements with the executor.

**Cost per instance:**
- proposer tokens, F model queries (cold caches in both arms), rounds, expansions (the arm has them per instance; R0
  has the mean only), wall time and hook seconds;
- `ATTEMPTS.jsonl`, an append-only record of every shard attempt. Shards that restarted with a warm F cache are
  flagged, because their F-query cost is a lower bound.

**Hook:**
- counters and inconsistency reasons;
- the **penalty band (invalid state or critic error)**, split at call and evaluation level (§5.2).

**Anytime curves:**
- success within r rounds, r = 1..12. This is exact (D-48c/D-50: the first r rounds do not depend on any later cap);
- success within T proposer tokens, T = 2k-16k. This is approximate.

**unmet = 1 report, both directions** (descriptive, on the executor's **true** state after the submitted prefix; an
invalid prefix is counted separately):
- R0's failures: the number of unmet goal facts, its distribution, how many stall at exactly 1 (D-76a), how many of
  those arm a solves, and a's solve rate on R0's other failures;
- the mirror quantity: a's failures, and R0's solves among them.

**Certificate c ≡ c+pen on R0.**
- Every F-accepted prediction in R0's beam caches is checked with the hook's `is_consistent`, with the parent state's
  blocks as `objs` (counts and reasons).
- The cache does not record the goal, and the hook, called with the goal, penalises a goal block that vanished from
  the imagined state.
- Passing the parent's blocks catches every vanished block (and flags a new name as `unknown_block`), so the check is
  at least as strict as the penalty.

**Unpenalised-inconsistency certificate on the arm** (report-only; RCX's hook limitation, §2.2).
- Every F-accepted prediction in the arm's caches is checked with the instance's blocks ("strict"). Each strict
  inconsistency is then classed as:
  - penalised by the hook as it was called (with the goal; registry off);
  - unpenalised: a vanished non-goal block;
  - undetermined: the parent holds a name no instance of the shard has, so the goal is unknown.
- This shows how many inconsistent keys went unpenalised and so can undercount the attribution share.

**F cache cross-check.**
- Every (state, action) key answered in both runs must have the same F text. F is frozen and memoised, so this checks
  the batch-invariance that pairing and the round curves rely on.
- Differences are reported, not reconciled.

**The arm's F audit:** FREPLAY definitions, set `b4s_a_beam` (item-7 addendum A §B.6:
`scripts/item7_addendum_a.py faudit-screen`, on files whose shas this analysis recorded first).

**Inputs hashed before any outcome is parsed.** Before any plans, summary, hook or cache file of either arm is parsed,
the analysis writes `runs/BW_B4_SCREEN/ANALYSIS_INPUTS.json`: the sha256 of every such file it reads, and of the freeze,
pre-check, attempts, canary and A.0 files. Only the registration files (the freeze records, `PRECHECK.json`, the A.0
sha table) and `ATTEMPTS.jsonl` are read before that, to re-verify the freeze and the VOID conditions; R0's files are
only hashed against A.0's table. R0's shard `summary.json` arguments are read early only when arm a never ran (§3). A
rerun of a failed analysis attempt must find identical shas.

**Not an effect estimate.** The screen's net and p are not effect estimates, and they are not quoted as such anywhere,
including in B4 (§8.1).

## 7. Execution and cost

### 7.1 Commands (all `python -B`, from `native_v1/`)

| Step | When | Command | Writes |
|---|---|---|---|
| freeze (CPU, seconds) | right after `bw_seeds.py freeze`, before `bw_seeds.py run` | `scripts/b4_screen.py freeze --prereg research/PREREG_B4_SCREEN.md --addendum research/PREREG_ITEM7_ADDENDUM_A.md` | `runs/FREEZE_B4S_a_beam.json`, `runs/BW_B4_SCREEN/PRECHECK.json`, `runs/BW_B4_SCREEN/rcx_snapshot/` |
| canary pre-flight (CPU, optional, not registered) | any time after RCX sends the canary | `scripts/b4_screen.py canary` | nothing |
| run (GPU) | the BWM GPU turn after item 7's analysis and addendum A.0 (D-75.1) | `scripts/b4_screen.py run` | `runs/BW_B4_SCREEN/a_beam/shard_{0..3}of4/`, merged `a_beam/`, `ATTEMPTS.jsonl` |
| analyze (CPU) | after both R0 and a exist | `scripts/b4_screen.py analyze` | `runs/BW_B4_SCREEN/B4S_REPORT.{json,md}`, `ANALYSIS_INPUTS.json` |
| re-run (GPU), only after an INSTRUMENT_FAULT report (§3) | one GPU turn | `scripts/b4_screen.py run --rerun`, then `analyze --rerun` | `runs/BW_B4_SCREEN/rerun/{c,a}_beam/`, `B4S_REPORT_RERUN.{json,md}` |

**The freeze record** is written once. It holds:
- the R0 args verbatim, the hook spec (registry off) and the control's freeze sha;
- every pin of §2, including the code path;
- the snapshot's file shas and the canary sha;
- the library and driver versions;
- the rule constants: net ≥ 2 and the 5% threshold, with no p-value term;
- the VOID rule and the pre-check result;
- the shas of this prereg (final), the addendum and PREREG_ITEM7_SEEDS (`956a0dc4…`);
- the registered runner and test shas;
- whether the freeze is late, with the list of item-7 files present.

**The freeze refuses:**
- if the runner or tests on disk differ from the registered shas in this header, or if the addendum does not name the
  same runner sha. This applies in the late path too (edit 2);
- while the canary sha is the placeholder (edit 8);
- if a file is named DRAFT, unless `--allow-draft` is passed;
- if any pin, the snapshot copy or `check_r0_is_b2` fails.

**No forking after R0 (co-sign edit 9).**
- **The guard covers all item-7 outputs.** That means every file under `runs/BW_SEEDS/`, because R0_direct reveals
  per-instance outcomes and a running shard's `progress.txt` shows per-round solve counts. It also covers bundleI's log
  `runs/BW_SEEDS.stdout` once it shows that an item-7 run has started.
- **If such a file exists, the freeze refuses unless `--allow-after-r0` is passed.** If `b4s-freeze` fails, the late
  freeze with `--allow-after-r0` is **mandatory**, and it records the files present. The screen is never "dropped" by
  choice.
- **Once `runs/BW_SEEDS/SEEDS_REPORT.json` or `runs/BW_SEEDS/ADDENDUM_A_INPUTS.json` exists, the freeze refuses
  outright.** The screen can then never run, and it is void.
- **The screen runs whatever item 7's outcome is.** It uses only R0 outputs whose sha256 equals addendum A.0's table
  (`ADDENDUM_A_INPUTS.json`). That covers R0's merged and per-shard `plans.jsonl` and `summary.json` and the shard
  `fcache.jsonl`. Otherwise the screen is VOID: `run` and `run-shard` refuse, and `analyze` writes a VOID report
  (also when the arm was never run).
- Addendum A.0 itself refuses unless the screen is frozen, or unless it is told explicitly that the screen was never
  frozen (in which case the screen is void).

**Run guards.** Every shard:
- re-verifies the freeze: R0's freeze sha and args, the runner sha, the canary sha, every pin (RCX files in the
  snapshot), the code path and the adapter shas;
- checks the versions, R0's shard args and R0's A.0 shas;
- runs the canary;
- refuses if its own `summary.json` exists, so it runs once.

Ordering and recovery:
- `hook.json`, then `plans.jsonl`, then `summary.json` (the completion marker).
- The merged `hook_merged.json` is written before the merged `summary.json` marker.
- An interrupted shard restarts from scratch with identical seeds and a warm F cache, as in item 7, and the restart is
  logged in `ATTEMPTS.jsonl`. There are no other retries under the same id.

**Analyze guards (co-sign edit 1).** `analyze`:
- calls `load_and_verify_freeze` (the same checks as run);
- takes the rule constants and drop_ids from the frozen record (§5.2);
- asserts R0's shard args. If they fail and arm a was therefore never run, it writes the INSTRUMENT FAULT report of
  §3 (VOID first, if a VOID condition also holds), which unlocks `run --rerun`;
- requires all 4 `hook.json`, each with calls > 0, calls ≥ the shard's instances, meter calls = hook calls, the
  registered hook record and `models.py` actually executed;
- refuses unless `hook_merged.json` equals the merge of the 4 shard records;
- requires the arm shards' own args;
- requires `val` on every row;
- refuses to overwrite `B4S_REPORT.json`.

**Lead's queue change (not made by this draft):**
- Insert a `b4s-freeze` job between bundleI's `i7-freeze` and `i7-run` (marker `runs/FREEZE_B4S_a_beam.json`,
  est 30 s).
- If `b4s-freeze` fails, item 7 is not blocked: the release gate never waits for the screen. The late freeze is then
  mandatory and must come before `i7-analyze` and before addendum A.0, because both of their outputs close the window.
  So the lead holds `i7-analyze` (and A.0) until `runs/FREEZE_B4S_a_beam.json` exists.

### 7.2 GPU cost estimate

**Expected: about 3.5 GPU-h (3.3-3.7), one BWM turn.**
- The frozen beam's planning time was 291-316 s per instance on hard-110 (316), mirror-110 (294) and H2-hard (291).
  These sets have the same block mix as seeds_fresh. (`wall_s` excludes model loading and VAL; add a few minutes per
  shard.)
- 40 instances in 4 shards of about 50-55 min each.
- A rough fit over those 24 shards gives 3.6 h at B2's per-instance mix: wall ≈ 978 s per shard + 22 s per 1k proposer
  tokens + 264 s per 1k F queries, with residuals up to about 1.1 ks per shard.
- The canary adds seconds per shard.

**Pessimistic: about 6 GPU-h.** If the hook drove every instance to the E24 budget, one shard would spill into the next
turn; shards resume. (Budget failures used about 15.0k proposer tokens, against an 8.0k mean on hard.)

**Hook CPU inside the GPU process: under a minute per run** (an estimate, not measured, because torch may not be
imported in this CPU track).
- About 150-200 unique imagined states per instance (B2-hard: 187 unique F keys per instance, about 80% legal).
- One single-thread forward of a 124k-parameter GNN each (estimated 1-5 ms), plus an LRU cache.
- It is measured and reported (`HookMeter` seconds).

**Memory:** CPU tensors only; no VRAM change; host RAM +about 50 MB.

**Re-run after an INSTRUMENT FAULT:** two arms, about 7 GPU-h, which needs a full turn or more.

**What the screen saves:** a NO-GO saves about 7 GPU turns of full B4 (5 arms at about 2 turns each, minus the screen).

## 8. Consequences

- **GO:**
  - The co-signed B4 primary runs as drafted (D-70; P-14 follow-up): P1 a vs c+pen and P2 b vs c+pen, Holm m = 2,
    fresh seed-20261410 set, n = 110, frozen W4/E24, Pa_v2; Pa_v3 cross secondary; arm b's V-LoRA trained first.
  - If the attribution rule fired, the GO wording says so and c+pen is mandatory (it is already the comparator).
- **NO-GO:**
  - The wording is "B4 arm (a) not supported at screen (seeds_fresh-40, paired vs R0, pre-set futility rule)".
  - Arms a0 and b are not run under B4, and B6 proceeds (D-75.1). A different module needs a new prereg.
- **VOID:** no reading. B4 is neither unlocked nor closed by this screen, and a new prereg is needed for any screen.
- **INSTRUMENT FAULT:** the pre-registered re-run of §3. If that fails too, VOID applies.
- **In every case:** the result and the per-instance records go to RCX (D-70 §7). The records cover arm, n, success,
  F queries, expansions, plan and hook counters.

### 8.1 Independence of B4 from the screen (co-sign edit 5)

- **Separate sets.** B4's 110 instances (seed 20261410) exclude seeds_fresh's renaming signatures (B4 draft §4: BWM's
  full exclusion recipe includes seeds_fresh). This is a binding condition on B4's set: its generator must check it at
  generation and refuse on any match. BWM generates the set together with the B4 freeze, after item 7. The screen's 40
  are **never pooled** into B4.
- **B4's design is fixed independently.** B4's n (110), α (Holm m = 2, first step 0.025), Holm family (a vs c+pen,
  b vs c+pen), arms (c, c+pen, a0, a, b) and endpoints are **not changed on screen data**.
- **No effect estimates.** The screen's net and p are not quoted as effect estimates, in B4 or anywhere else.
- **Co-signing the screen does not co-sign B4.** Route chosen by BWM (D-79):
  - B4's prereg declares the screen's outcome as **known** at its registration;
  - it **forbids any change to arm (a)**: the module, the D-76a call and the pins of §2.2 stay exactly as they are.

## 9. Not claimed; limits

- **Tier B:** the arm uses an external module, and the GNN was trained on exact h\* (simulator labels). So arm a is not
  simulator-free, and the result is not a model-capability result for the LM planner.
- **Power:** one run, one seed, n = 40, with low power below about +9 pp. The null false-GO rate is 0.30-0.36 at
  plausible discordance (§5.3).
- **Scope:** seeds_fresh is a generated hard-style set, not PlanBench. Pa_v2 only. Blocksworld only.
- **Arm composition:** the screen cannot separate "learned value" from "misplaced_blocks"; that is arm a0 of the full
  B4.
- **Registry off:** the hook does not penalise a vanished non-goal block (§2.2). §6 reports how often that happened.
- **The control runs in a different turn.** The control is item 7's R0 run, executed before the arm.
  - Pairing rests on the D-48c/D-50 invariances: per-instance CRN, batch-independent sampling and a frozen memoised F.
  - The round-1 identity check (§3), the pinned code path, the recorded versions (§2.1) and §6's cache cross-check test
    these invariances.

## 10. Dry-run evidence (CPU, no model, no torch)

`python -B scripts/b4_screen.py --dry-run --inconsistent 0` and the default `--inconsistent 0.05`, re-run on the revised
runner on 2026-10-12. Outputs are in scratch, under `%TEMP%/claude/bwm_b4cosign_scratch/dryrun_{exact,inc05}/`.

**Setup.**
- The pinned planner bytes run with a stub `bwmn.lm`:
  - a scripted F: exact on legal actions, F-09-style "legal" answers on illegal actions (which the support check
    rejects), and an optional share of inconsistent answers;
  - a noisy teardown-and-build proposer with CRN seeds.
- The arm is the pinned `LearnedUnmet` with only its GNN forward replaced by 0.0.
- 8 generated instances (5-8 blocks, dry-run seed 7411), with the frozen B2 beam args. seeds_fresh is hashed, never
  planned on.

**Integrity results.**
- All pins verified, including the code path.
- The RCX files executed are exactly `rcx/__init__.py`, `rcx/bw/__init__.py`, `b4_hook.py`, `bwm_import.py`,
  `encode.py` and `space.py` (not `models.py`).
- torch, transformers and peft were not imported, and no bytecode appeared in the RCX tree.
- The arm in 4 shards, merged by `bw_run_tests.merge`, equals the unsharded arm instance for instance (plans, verified,
  rounds, tokens). With the exact stub, hook calls were 2,477 both ways.
- The meter is a pure pass-through. The freeze-record preview's args equal what `bw_seeds.py freeze` writes for R0.

**Exact stub F:**
- the certificate holds;
- 0 penalty-band calls;
- 0 unpenalised or undetermined inconsistent keys;
- the F cache cross-check finds 0 differences (469 common keys);
- verified-plan precision 1.0, and VAL agrees with the executor;
- the round-1 identity check holds (3 instances solved in round 1 in each arm).

**5% inconsistent stub answers:**
- the certificate fails (70 of 232 accepted predictions are inconsistent);
- 15.4% of hook calls (361/2,340) are penalty-band, all of them invalid-state;
- the evaluation-level share is 0.216;
- the arm certificate finds 38 strict inconsistencies: 27 penalised, 0 unpenalised, and 11 undetermined (hallucinated
  block names);
- the attribution path works.

**Outcomes.** The dry-run outcomes (exact stub: control 6/8, arm 5/8; 5% stub: control 5/8, arm 6/8) mean nothing
about B4. A dry run has no canary, so the would-be VOID check lists only "no passed canary".

**Tests.**
- `python -B -m pytest tests/test_b4_screen.py -q -p no:cacheprovider` → 60 passed.
- They cover the 13 edits, including the GPU path end to end with model loading stubbed (edit 13): run-shard ×4 →
  merge → analyze, the refusal cases, the VOID cases (including a run that never completed), the instrument fault
  (round-1 identity and R0 arguments), the re-run, the freeze and the snapshot.
- Together with `tests/test_item7_addendum_a.py` (9) and `tests/test_domain_planner.py` (28), 97 pass in either
  order. The whole `tests/` directory passes too (234).

## 11. For RCX's co-signature

1. The pins, including the four BWM additions (recomputed by RCX), the code-path pins and the versions (§2.1-2.2); the
   loader, snapshot and identity checks (§2.3).
2. The canary and the VOID rule (§2.4). RCX supplies the canary and its sha.
3. The pre-check rule and its drafting result (§4.3: no drops), and the hygiene statement (§4.2).
4. The decision rule and the attribution rule (§5); the D-76a floor is implied. The operating characteristics (§5.3).
5. Control = item 7's R0 run, with the c ≡ c+pen certificate, the round-1 identity check and its consequence (§3).
6. Independence of B4 (§8.1); outputs sent to RCX (§8).

**What the co-signature binds.** It binds to the final runner and test shas in this header, to this text's final
sha256 and to the addendum's.
- **Finalisation** drops "DRAFT" from the title and changes the Status row only. `b4_screen.py final-diff --draft D
  --final F` refuses any other change, and RCX re-verifies that diff.
- **RCX's commitments:** the canary file; receiving the per-instance records; keeping its pinned files byte-stable
  until BWM's freeze has copied them into the snapshot (§2.3). RCX commits no GPU and no CPU-lane time.
