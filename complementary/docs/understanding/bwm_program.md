# BWM-001 research report: forward world model and planning program (read-only survey, 2026-10-09 ~22:25 CDT)

## 0. Provenance and which copy is current

- **The live copy is `C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/`.** Every result and decision below comes from here unless noted.
- **The repo copy `rineya-world-model-research/bwm/` (branch main) is stale.** Its last BWM commit is `ed94c45` ("final Blocksworld-hard result (73/110 vs 5/110), trace schema, roadmap"), and its `docs/DECISIONS.md` stops at P-08/D-52.
  - The local copy adds D-53 to D-58, F-08 and P-09.
  - The repo's `bwm/code/bwmn/blocks_planner.py` has sha `b151c26a…`. The local copy has `18a5f22b…`, which adds the D-53 hooks. `blocks.py` is identical in both (`b1e28ed5…`).
- **The bundle's top level (`README.md`, `RELEASE_NOTES.md`, `program/`, `configs/`, `schemas/`, `src/rineya_bwm/`) is the original v1.0 spec dated 5 Oct.**
  - It contains an "outcome-conditioned world modeling" spec, a CPU reference (65 tests, 551 RegisterWorld episodes, a 120,228-param MLP forward sidecar), and an unqualified native scaffold.
  - Its backlog statuses are stale: ENG-02 to ENG-15 still read "PENDING", and the acceptance gates G1 to G6 read "NOT EXECUTED".
  - The native work in `native_v1/` superseded it. The design is in `native_v1/DESIGN.md` and every change is logged in `native_v1/DECISIONS.md` (D-01 to D-58).

## 1. Environments

**(a) RegisterWorld-v2 (RW2)**, `native_v1/bwmn/rw2.py`
- **State:** an integer x. **Actions:** `MOVES=("add1","sub1","double","halve")` plus `"finish"`.
- **Law:** `exact_step(x, action, limit) -> (next_x, legal, terminated)`. A move to |x|>limit is illegal and leaves x unchanged; halving an odd x is illegal. It is the same law as the reference `src/rineya_bwm/core.py exact_step`.
- **Oracles and helpers:** `dist_to(goal, limit)` (reverse BFS), `opt_moves`, `optimal_next_actions`, `sample_optimal_plan`, `execute(task, actions)`, `greedy_episode(task, lookahead, step_fn)`.
- **Tasks and data:** `class Task`, `build_pools(seed=20261005)`, `eval_tasks(pools, pool, split, n, seed, infeasible=...)`, `build_training_rows`, `proposer_sft`, `forward_sft`.
- **Difficulty knobs:**
  - limits: `ID_LIMITS=(32,48,64)`, `OOD_RANGE_LIMITS=(96,128)`;
  - opt_moves 1-8 (ID) and 9-12 (OOD-long);
  - tight budget H = opt+1+slack, with slack 0-2 at eval and 0-6 in training;
  - infeasible twins.
- **Split:** goal-held-out, by hash of the goal value, 65/10/5/20 (D-06).
- **Prompts:** `SYS_FORWARD`/`forward_prompt(x, action, limit)` -> `{"legal","terminated","x"}`.

**(b) Blocksworld (PlanBench 4-op STRIPS)**, `native_v1/bwmn/blocks.py`
- **State:** a frozenset of facts `('clear',x) ('ontable',x) ('handempty',) ('holding',x) ('on',x,y)`.
- **Actions:** `('pick-up',x) ('put-down',x) ('stack',x,y) ('unstack',x,y)`. An inapplicable action is an illegal no-op.
- **Simulator and oracles:** `applicable`, `step(s,a)->(s',legal)`, `ground_actions(objs)` (2n²), `goal_met`, `bfs_plan(init, goal, objs, max_states=200000)`, `execute(init, goal, plan)` (VAL semantics).
- **PDDL I/O and VAL:** `parse_problem`, `write_problem`, `plan_to_pddl`, `val_check(problem_path, plan, domain_path)`. The VAL binary is `bench/planbench/VAL/bin/validate/validate.exe`.
- **Generator:** `generate(rng, n_blocks, name_pool=NAMES(a..o), full_goal_frac)` built on `random_towers(objs, rng, p_stack=0.6)` (p_stack 0.85 for hard-style targets).
- **Leakage guards:** `signature` (renaming-canonical, n≤7), `test_signatures()`, `hard_signatures()`, plus `scripts/blocks_val_large.py invariant()` for larger n.
- **LM rendering:**
  - `state_json` has keys `clear/hand/on/table`;
  - `proposer_prompt`, `forward_prompt`, `forward_prompt_delta`, `forward_target_delta`;
  - delta output is `{"add":[...],"del":[...],"legal":bool}`;
  - parsers: `parse_forward_delta(text, s)` (returns None if a deleted fact is absent), `parse_plan`, `parse_plan_prefix`.
- **Bench assets** (`bench/planbench/`, LLMs-Planning MIT commit fc638a1a):
  - 600 Task-1 instances (500 `generated_basic` + 100 `generated_basic_3`);
  - `blocksworld_hard` (110; 6-15 blocks; `info.json` holds reference plan length);
  - `instances_mystery` (603);
  - `unsolvable_blocksworld`;
  - `LEDGER.json`: published plans re-scored with local VAL.
- **Validated harness (B-01):** BFS lengths match on 120/120, the oracle planner scores 602/602, and VAL agrees with the executor on 602/602.
- **Leakage finding:** the 602 instances are only 347 distinct problems up to renaming.

**(c) Generated Blocksworld sets** (`runs/`)

| Set | Contents | Seed / use |
|---|---|---|
| `BW_data` | 12k forward rows (~50% illegal), plus delta versions, proposer rows | seed 20261006 |
| `BW_data_v2` | 53,682 proposer rows, half with whole-configuration goals | seed 20261007 |
| `BW_val_large/instances_val_large.json` | 120 instances, 6-15 blocks | seed 20261008; selection |
| `instances_mirror_hard.json` | 110 instances, hard block-count mix | seed 20261009; a B2 **test** set |
| `instances_b3_fresh.json` | 110 instances, sha `575c45b1…` | seed 20261011; **sealed for B3; nothing may run on it before FREEZE_B3** |

The distillation problems use seed 20261010.

**(d) Mystery Blocksworld and generic STRIPS**, `bwmn/mystery.py` (1,271 lines)
- Contents: a generic `Domain/Schema` PDDL parser, `step(dom, s, a)`, `bfs_plan(dom, …)`, `find_isomorphism`, Regime A (renaming) and Regime B (`random_domain`, `regime_b`, `regime_b_sample`, with rules given in the prompt).
- Status: only a smoke test ran (`runs/_mystery_smoke/SUMMARY.json`: executor agrees with blocks on 122,064 transitions; VAL agrees).
- Must-fix items in `TODO_BEFORE_RUNS.md` are still open. No model has been trained on it.

**(e) FlagWorld**, `bwmn/flags.py`, `flags_planner.py`, `scripts/flags_run.py`
- This is the Forge v2 `prerequisite_plan` family: bitstring state, 2n+2 actions with pre_on/pre_off/add/delete, finish, tight budgets, about 1/4 impossible worlds.
- It was written 10-06 (DESIGN §8 "E3"). There are **no `runs/FL_*` directories and no DECISIONS entry**, so my inference is that it was built but never run.

**(f) Reference RW v1** (`src/rineya_bwm/core.py`): `exact_step`, `shortest_plan`, `EmpiricalForward`, `EmpiricalReverse`, `select_plan`, `returns_to_go`.

## 2. Models, planners and value usage

**Backbone**
- Frozen **Rineya Titan-Sparse** (Qwen3-0.6B geometry, FP32, sha a28629c9…), loaded with `lm.load_core("titan-sparse")` from `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/…`.
- `lm.MODELS` also lists titan-transition, titan-exp, supa and qwen3-0.6b.

**Adapters**
- Recipe (D-03): LoRA r16/α32 on q,k,v,o,gate,up,down (10.09M params each), never merged; AdamW (0.9, 0.95), lr 1e-4, WSD 10/70/20; loss on target + EOS only.
- Run environment: `hermes-agent` venv, Python 3.11, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0.

**Forward models (F), all goal-blind**

| Model | Training | Measured accuracy |
|---|---|---|
| RW2 `F_titan_main` | 1,455 ID transitions, 300 steps | ID 0.997; \|x\|>64 0.995; legality-flip doubles 0.365 (F-01) |
| `BW_F_s0` (state format) | — | 3-10 blocks 1.000; 12 and 15 blocks 0.967 (F-05) |
| `BW_Fdelta_s0` (delta format, **the frozen F**) | 600 steps, 12k transitions, 4-6 blocks | 12 blocks 0.983, 15 blocks 1.000, ID 0.990 (F-07) |

**F-08 surface audit (`runs/BW_Fdelta_s0/surface_audit.json`)**

| Rendering | Exact match |
|---|---|
| Training format | 1.000 |
| Unseen block names | 0.992 |
| Shuffled lists | 1.000 |
| **Reversed JSON key order** | **0.458** |

BWM's own conclusion is that the model is "format-locked".

**Proposers**
- RW2: `P_actions_s1` and `P_states_s1` (action-only vs state-and-action targets).
- Blocksworld: `BW_Pa_v2_s1` (1,000 steps, action-only, outcome-conditioned).

**Planners**
- **RW2** (`bwmn/planner.py`):
  - `run_episodes(tasks, proposer, fwd, rule="rerank"|"gate")` executes one action and then replans.
  - Rerank uses `predicted_utility` = success − 0.01·len − 0.1·illegal, computed from F rollouts.
  - Baselines: `greedy_under_f`, `bfs_under_f(call_cap)`.
- **Blocksworld** (`bwmn/blocks_planner.py`), all planning **in imagination, with no execution feedback**:
  - `plan_direct`: proposer candidate 0 as is.
  - `plan_instances`: the replan-in-imagination planner.
  - `plan_beam(instances, proposer, fwd, *, eval_seed, width, max_expansions, max_depth=80, max_rollout, log, goal_met=None, unmet=None)`:
    - a node is (imagined state, prefix);
    - K proposals per node are rolled out under F with `rollout_many`;
    - solved if a rollout reaches the goal with all steps legal;
    - otherwise each child is the point of best goal progress;
    - no revisits; the beam is ranked by **(unmet goal facts, prefix length, candidate rank)**.
  - F wrappers: `ExactF`, `LMForwardBW(model, tok, adapter, fmt, cache=JSONL, max_batch_tokens)`.
  - Proposers: `LMProposerBW` (CRN seeds: candidate 0 greedy, the others T=1.0/top-p 0.95), `OracleProposer`.
  - Scoring: `score()`.

**Value or critic: none is learned anywhere.**
- The beam heuristic is the hand-coded count `len(goal - state)` on F's predicted state.
- RW2 uses a hand-coded utility from F rollouts.
- `scripts/abstain_score.py` (outcome likelihood ratio and an F-reject signal for infeasibility AUROC) exists, but I found no run output (no ABST* directory).

## 3. Headline results and tiers of assistance

### 3.1 Measured, frozen and run once (P-08)

- **Test set:** PlanBench Blocksworld-hard, 110 instances, 6-15 blocks, 20-40-step optimal plans; one plan per instance, scored by VAL.
- **S-beam:** **73/110 = 0.664** (Clopper-Pearson 95% CI 0.567-0.751).
- **S-direct:** **5/110 = 0.045** (0.015-0.103).
- **Paired comparison:** beam-only 68 / direct-only 0, exact McNemar p = 6.8e-21.
- **Verification:** VAL agrees on all 220 plans, and verification precision is 73/73. All 37 failures were unverified: 23 `dead_end` and 14 `budget` (my tally from `hard_beam/plans.jsonl`).
- **Success by block count** (S-beam successes / S-direct successes / n):

| Blocks | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| Beam / direct / n | 1/0/1 | 1/0/1 | 6/2/7 | 14/3/14 | 10/0/13 | 9/0/12 | 8/0/13 | 11/0/18 | 11/0/21 | 2/0/10 |

- **Cost:** 186.9 unique F queries per instance and 7,991.9 proposer tokens per instance; 34,770 s of wall time over 8 shards.
- **Published zero-shot references** (unpaired Fisher tests, a different regime):
  - o1-preview 23.6%, p = 2.4e-10;
  - o1-mini 10%, p = 1.2e-18;
  - DeepSeek-R1 53.6%, two-sided p = 0.073, so "above R1" is not claimed.
- **Training saw at most 6 blocks.**
- **My own computation (not reported by BWM):** successful plans average 1.16× the reference length in `info.json`; only 32.9% are at or below the reference length; the maximum is 1.73×.

### 3.2 Frozen configurations

**`runs/FREEZE_B2_beam.json`**
- Settings: `planner=beam, K=8, W=4, E=24, R=12, M=80, P=1, fwd=lm delta (runs/BW_Fdelta_s0/adapter), pa=runs/BW_Pa_v2_s1/adapter, eval_seed=0, max_depth=80`.
- Hashes: prereg sha `adfc4354…`, pa sha `ee1c5aec…`, fwd sha `2ed0bb00…`.
- Selection rule: maximum mean success over val_large and val_v2, subject to ≤ 2,000 F queries per instance.
- Selection results: K8-M80 mean 0.807 (val_large 0.633, val_v2 0.98); K4-M80 0.790; K4 0.740.

**`FREEZE_B2_direct.json`:** `planner=direct, K=1, M=256`.

### 3.3 Tiers of assistance (distinguishing what each system gets)

| Tier | System | Result |
|---|---|---|
| T0 | **S-direct**: the proposer LoRA alone, one greedy plan, no F, no search | hard 5/110; mirror 6/110; val_large 0.117 |
| T1 | **S-beam**: proposer + learned F LoRA (internal model) + external hand-coded beam search; code reads the goal facts and counts unmet goals on F's predicted states; the domain rules live only in F; no simulator or VAL while planning | hard 73/110 |
| T2 | **Exact-F oracle harness**: ExactF + scripted myopic proposer (D-44) | 110/110; mean 49 expansions; 79,745 exact-F calls. A harness test, not a result |
| External regime (excluded) | o1-preview + VAL loop | 98.2%; verifier in the loop; explicitly excluded from comparison |

The joint roadmap lists "Reporting tool tiers as model ability" under "do not do".

### 3.4 Other measured results (validation-grade, single seed)

- **P-06:** on val_large (30), beam 0.50 at K4 vs direct 0.117. The README cites 0.63 vs 0.13 for K8.
- **P-09 (interim mirror, shards 0-4 = 70 instances):** S-beam 38/70 = 0.543 vs S-direct 4/70 on the same instances, +34/-0, p = 1.2e-10. The hard set on the same shard indices scored 44/70. The interim hard-minus-mirror gap is +8.6 pp against a pre-registered flag at > 10 pp.
- **RW2:**
  - F rerank A01 0.675 vs A00 0.620, p = 0.001 (P-03).
  - K = 12 0.82 vs K = 4 0.67 (P-04).
  - Compute frontier (P-05): proposer + F sits on the same frontier as BFS-under-F; BFS-256 reaches 1.00 at ~9.3k tokens. "No amortization claim for RW2."
  - Gated adapter-off verifier (F-03/F-04): flips 0.32 -> 0.949; range BFS-256 0.720 -> 0.863.

## 4. Trace schemas

**(a) Joint trace-target schema (D-52; `native_v1/bwmn/traces.py`; RCWM D-34)**
- One JSON object with keys in **fixed literal order** `{"cmp":…, "calc":…, "after":…}`. Keys are never sorted, and `parse_trace` rejects any other order.
- **RW2:**
  - add1/sub1/double: `cmp={"cand","slack":L-|c|,"ok"}`;
  - halve: `{"x","rem","ok"}`;
  - finish: `null`;
  - `calc` = c if ok, else x;
  - `after` = `{"legal","terminated","x"}`.
- **Blocksworld:**
  - `cmp` = `{<precondition fact str>: bool, …, "ok": legal}` (from `bw_preconditions`);
  - `calc` = `{"add","del"}`;
  - `after` = the delta answer.
- **Arms:** `plain` (after only), `trace`, `filler` (`{"fill":" ." * n,"after":…}`).
  - Per D-55 / amendment 4b, n is a **constant per (domain, action kind)**, set by `filler_length` because a per-row filler length leaks the answer.
- **Helpers:** `rw2_target`, `bw_target`, `trace_head_tokens`, `after_text`.
- **Status:** planned arms only. No trace-trained BWM model exists yet.

**(b) Bundle `schemas/` (v1 spec, RegisterWorld reference)**
- `episode.schema.json`: steps `{x, action, next_x, legal, terminated, reward, return_to_go}`, plus success, total_return and split.
- `evaluation_record.schema.json`: arm ∈ A00/A10/A01/A11/control, success, abstained, violations, input/output tokens, forward_calls, wall_seconds.
- `sft.schema.json`: role ∈ policy/reverse/forward.
- `paired_parent.schema.json` and `run_manifest.schema.json`.

**(c) Blocksworld planner output rows (`plans.jsonl`):** `{id, n_blocks, success, valid, len, verified, rounds, gen_tokens, fail, plan[], val}`.

## 5. Roadmap, active experiments, GPU and thermal rules

### 5.1 Running now (verified in the process list and the lock file)

- **Job:** `blocks_eval.py --set mirror --planner beam … --freeze runs/FREEZE_B2_beam.json --shard 5/8`, run by `bw_run_tests.py`, inside `bundle.py runs/bundles/bundleD.json`, inside `gpu_run.py`, inside `orchestrate.py` (pid 32100).
- **Lease:** BWM holds `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock` (`bwm:bundle-bundleD`) since 22:05 with a 16,200 s budget, so until about 02:35 on 10-10.
- **Peer:** RCWM `rcwm-t1:t2b-prereq` has been waiting since 22:03.
- **Shard 5 history:** it was killed twice, at 07:19 (after 8,535 s) and at 18:38 (after 2,835 s), each with rc 1073807364 = 0x40010004. The cause is not recorded in DECISIONS. It restarts with a warm F cache (`fcache.jsonl`).

### 5.2 Remaining queue (`runs/jobs.json`)

1. **bundleD:** mirror beam shards 5-7, then the PlanBench-600 direct run, then PlanBench-600 beam (4 shards).
2. **bundleE (B3 search distillation, pre-registered in `PREREG_B3_DISTILL.md`, sha `e6aa884b…`):**
   - `bw_distill.py search --n 240` (K=4, F-verified plans only) -> `rows` -> `bw_b3_prep.py mix`;
   - train Pa_v3 and Pa_v2c (400 steps, `--init-adapter BW_Pa_v2_s1`, max-len 640);
   - `FREEZE_B3.json` -> `bw_b3_eval.py run/analyze`;
   - primary test: S-direct Pa_v3 vs Pa_v2c on b3fresh, needing ≥ +10 pp and p < 0.05;
   - hard runs H1/H2 happen only if that bar is met.
3. **Then:** a2-train-Fvlab, P_actions_s1_1600, E2 F sample-efficiency trainings (titan/supa/qwen3 r256/r1024), e2-xlimB.
4. **bundleF:** the RW2 4-arm GPU pilot (pilotG).

**Inference:** the `est_s 1800` for the distillation search looks optimistic. Validation beam K8 took about 7,700 s per 30 instances, although 6-10 blocks at K=4 are cheaper.

### 5.3 Joint roadmap (`research/JOINT_ROADMAP_2026-10-08.md`)

| # | Item | Owner |
|---|---|---|
| 1 | Finish frozen runs | B |
| 2 | RCWM out-of-range map (PREREG-3, done: TRIGGER) | R |
| 3 | Ledger beam with T2 as F | J |
| 4 | Search distillation (= B3) | B |
| 5a | Precondition representation | R |
| 5b | Trace / filler | J |
| 5c | Correctness retention suite | J |
| 6 | Shared legality checker LoRA trained on minimal pairs | J |
| 7 | Two more seeds of BW_F and Pa_v2 | B |
| 8 | Exact arithmetic in the step | R |
| 9 | Second domain (Logistics or Mystery Regime B) | B |

- **Zero-GPU items:** an input-novelty / hidden-state probe for confident errors (F-06); devfreeze packaging.
- **Planned:** D-54 key-order input augmentation, tested on held-out permutations.
- **Day plan:** about 2-3 GPU turns per programme per day, under strict alternation.

### 5.4 GPU and thermal rules

- **Hardware:** a shared RTX 4070 Laptop 8 GB that also drives the display.
- **Rules:**
  - D-20: 70% VRAM cap.
  - D-22: proportional duty cap, idle = 3.0 × GPU busy time (~25% duty; `BWM_DUTY_RATIO`).
  - D-21: `lm.thermal_guard` pauses at ≥ 82 °C until ≤ 75 °C, samples every 10 s, treats an unreadable temperature as hot, and honours the RCWM flag `…/RCWM-T1/runs/.thermal_pause`. RCWM's `thermal_watchdog.py` is the only flag writer.
  - CPU guard: tz01 ≥ 98 °C until ≤ 90 °C.
  - D-39: host-RAM floor of 3 GB for GPU loads and 12 GB for CPU-lane loads.
  - D-25/D-35: CPU lanes never touch CUDA (`BWM_DEVICE=cpu`).
  - One GPU process at a time, under the lease.
  - D-26: one-for-one alternation.
  - D-41/D-45: bundles of at most 4.5 h.
  - D-31: keep-awake.
  - **Stop rule:** any further TDR under a BWM job halts BWM GPU work.
- **Health now** (`runs/health_now.json` and `health.log`): GPU 63-70 °C, tz01 about 94-96 °C, about 11-16 GB RAM free.

## 6. Critic, value, successor-feature, distillation and skill work (to avoid duplicating RVM-001 / RSM-001)

| Area | Status |
|---|---|
| Learned critic / value model | **None, done or planned.** The beam uses the hand-coded unmet-goal count; RW2 uses a hand-coded utility. |
| Successor features | **None found** (grep across docs and code). |
| Skills / macros / options | **None.** Proposals are multi-action plans truncated to R=12; no reusable macro library. |
| Search distillation, Blocksworld | **B3 is pre-registered, coded and queued; not yet run** (no `runs/BW_DISTILL_r1`). Teacher = F-verified beam plans; rows = suffixes with F-predicted states. |
| Search distillation, RW2 | **Written, never run** (no `SD_*` runs): `scripts/self_distill.py` ("HC2"; F-verified or exact-verified, mixed 1:1). DESIGN §8 "E4" and the program's "verified_plan_distillation" are the same idea. |
| Verifier / legality | Done: the uncertainty-gated adapter-off verifier (F-03/F-04/F-06). Planned: a shared legality-checker LoRA (#6) and trace targets (#5b). |
| Seed-consensus certificates | Observed on the RCWM side (seeds agree on 85% of steps and are then right 0.995 of the time); not a BWM item. |

**Inference:** the clearly open gaps are learned value or heuristic functions for node ranking and pruning, and dead-end or budget prediction. All 37 hard failures were unverified, so a critic could support abstention or better expansion. Plan-length optimality is also unaddressed (1.16× the reference length).

## 7. Interfaces a complementary researcher can import without modifying BWM

- **Import pattern** (as RCWM does):
  `sys.dont_write_bytecode=True; sys.path.insert(0, ".../native_v1"); from bwmn import blocks as B, blocks_planner as BP, rw2, stats, traces, mystery`
  - `blocks_planner`, `planner` and `traces` all pull in `lm`, which imports torch, so torch is needed even for CPU-only use.
  - `BENCH` resolves relative to the module, so do not copy `bwmn` elsewhere without `bench/`.
- **Planner freeze:** `blocks_planner.py` is **frozen at sha 18a5f22b…** until the ledger-beam run finishes (D-57). Use it read-only.
- **Simulators and oracles:** `B.step`, `B.applicable`, `B.bfs_plan`, `B.execute`, `B.val_check`; `rw2.exact_step`, `rw2.dist_to`; `mystery.step`, `mystery.bfs_plan`.
- **Planning:** `BP.plan_beam(..., goal_met=, unmet=)` is domain-generic: states and actions must be hashable, and instances need `id/init/goal`. The proposer API is `propose(items, eval_seed) -> [[{"plan","claimed","n_tokens"}]*K]`. The F API is `predict(keys) -> [(s', legal) | None]`.
- **Learned F:** `BP.LMForwardBW` needs a model with an adapter named "fwd". **Pass your own `cache` path**; the default in `blocks_eval` writes into the run directory.
- **Evaluation:**
  - `scripts/blocks_eval.py` supports `--set val|val_v2|val_large|mirror|hard|planbench`, `--planner replan|beam|direct`, `--proposer oracle|lm`, `--fwd exact|lm` and `--shard`. Test sets require `--freeze` and refuse a second run.
  - `BP.score`; `stats.wilson`, `stats.mcnemar_exact`, `stats.paired_table`, `stats.holm`, `stats.crossed_cluster_bootstrap`; `flops.generation_flops`.
- **Generators:** `B.generate`, `scripts/blocks_val_large.make` / `invariant`, `scripts/blocks_data.py`. Use new seeds (20261005-20261011 and 20261141-43 are taken) and exclude the hard, val_large, mirror and b3fresh invariants.
- **Instruments not to touch:**
  - **Spent or reserved:** hard-110, mirror, PlanBench-600 and **b3fresh**. The roadmap says: "Tuning on spent or reserved instruments" is forbidden.
  - **Usable:** `BW_data*` jsonl, val sets (selection-only).
- **Adapters** (local only, not in git):
  - `runs/BW_Fdelta_s0/adapter`, `runs/BW_Pa_v2_s1/adapter`, also packaged in `release/Rineya-Titan-Sparse-BWM-Blocksworld-0.1.0-devfreeze/adapters/{world_model_delta,proposer_v2}`;
  - RW2: `runs/F_titan_main/adapter`, `runs/P_actions_s1`, `runs/P_states_s1`.
- **GPU:** use `bwmn/gpu_lease.lease(label, job)` and the alternation protocol. Never use the GPU outside the lease (incident I-15).

## 8. Joint work with RCWM

- **Shared infrastructure:**
  - the GPU lease, alternation and bundle agreements (D-13, D-18, D-26, D-41, D-45);
  - the shared thermal flag;
  - the BWM trainer (`lm.py`, `train_generic.py`, `gpu_lease.py`), vendored into RCWM's `vendor_bwm/`;
  - the shared private repo `LUNCY579/rineya-world-model-research` (D-51; `bwm/` and `rcwm/` folders).
- **Joint findings:**
  - F-01 corresponds to RCWM's "computed-predicate rules not learned".
  - F-08 corresponds to RCWM PREREG-3 (T2 reordered keys 0.37-0.47); the shared rule D-35 is that every claim gets a surface-form audit plus an out-of-range audit.
  - Trace schema D-52 = RCWM D-34; filler amendment D-55.
- **Item 3, ledger beam** (`research/PREREG_LEDGER_BEAM_DRAFT.md`, draft v1):
  - RCWM's kit (`rcwm/experiments/RCWM-T1/scripts/lbeam_f.py`, `lbeam_problems.py`, `lbeam_proposer.py`; `docs/ledger_beam/INTERFACE.md` rev 3) calls `BP.plan_beam` with `EnumerateProposer`, `LF.guarded(LedgerF(arm))` and W2/E12.
  - Arms: T2 / OFF / ORACLE.
  - Co-primaries: T2 vs OFF on all instances and on the needs-lookahead stratum.
  - Bar: ≥ +0.20 with Holm p < 0.01; near-oracle requires the T2/ORACLE ratio ≥ 0.80.
  - N = 180 fresh instances; about 2.5 GPU-h.
  - ORACLE sizing: 0.85 at W2E12.
  - RCWM runs it and holds the answer key.
  - **Not yet frozen or run.**
- **Planned joint items:** #5c retention suite (RCWM froze it with a dev draw, RCWM D-36), #6 legality checker.

## 9. Files modified in the last 24 h (local bundle, excluding binaries)

- **Docs:**
  - `native_v1/DECISIONS.md` (22:11);
  - `PREREG_B3_DISTILL.md` (22:09; the header says "Written 22:30", which does not match the file time);
  - `research/PREREG_LEDGER_BEAM_DRAFT.md` (12:35).
- **Code:**
  - `bwmn/traces.py` (08:02);
  - `scripts/bw_distill.py` (12:04), `bw_b3_prep.py` (22:09), `bw_b3_eval.py` (22:10);
  - `train_generic.py` (22:08, `--init-adapter`), `watch_events.py` (22:06), `bw_surface_audit.py` (07:21).
- **Data:** `runs/BW_val_large/instances_b3_fresh.json` (22:09).
- **Queues:** `runs/jobs.json`, `bundles/bundle{D,E,F}.json` (22:11), `jobs_cpu.json`.
- **Runs:**
  - `runs/BW_TEST_B2/mirror_beam/shard_{0..5}of8/*` (08:37-22:09);
  - `BW_Fdelta_s0/surface_audit.*` (07:29);
  - orchestrator and health logs.
- **Repo export clone:** `gh_export/rcwm/...` (12:03; the RCWM export: T2b kits, lbeam kit, retention).

**Not found:**
- any learned value, critic or successor-feature code;
- any FlagWorld, Mystery-model, RW2 self-distillation or abstention-scoring run outputs;
- the merged mirror S-beam summary (pending);
- any PlanBench-600 result;
- a second seed of any adapter.