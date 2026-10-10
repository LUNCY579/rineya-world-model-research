# RCI complementary program: decision-ready synthesis (as of 2026-10-09, about 22:25 CDT)

**Path keys used throughout**
- **BUNDLE** = `C:/Users/clclc/BusinessProject/WorldModel/rineya_complementary_research_2026-10-09` (the spec, read-only)
- **RCWM** = `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1`. This is the live copy and is newer than the repo mirror.
- **BWM** = `C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1`. This is the live copy; the repo copy `bwm/` is stale at `ed94c45`.
- **REPO** = `C:/Users/clclc/BusinessProject/WorldModel/rineya-world-model-research` (main at `a223884`; RCWM's live export target)
- **WT** = `C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree` (branch `research/rineya-complementary-intelligence-v1`, at `a223884`, clean, no remote yet)
- **FR** = `datasets/rineya_high_quality_factory_v1/` on `origin/research/rineya-high-quality-dataset-v1` (HEAD `7a7d055`). It is currently checked out read-only as a detached worktree in this session's scratchpad (`.../scratchpad/dataset_ro/...`).

**Evidence status.**
- The bundle contains no native measurements: `09_quality/validation_summary.json` says `native_experiments_run: 0`.
- Every number below is either recorded by a peer program [M], a plan [P], or an inference [I].
- The bundle's evidence is pinned at `a223884`. Both peers' live bundles have moved on since then: RCWM added D-38, R17 and I-31; BWM added D-53 to D-58, F-08 and P-09.

---

## 1. Peer programs

### 1.1 Who has the GPU right now [M]

| Item | State |
|---|---|
| Card | One RTX 4070 Laptop, 8,188 MiB. It also drives the display. 6,303 MiB is in use. |
| Lease holder | `bwm:bundle-bundleD` (pid 32060, worker 31372). Acquired 22:05:30, budget 16,200 s, so it ends about **02:35 on 10-10**. Job: mirror beam shard 5/8. |
| Waiting | `rcwm-t1:t2b-prereq` (pid 42092, since 22:03). This is the 5a screen: about 3.83 h (one turn) or 6.10 h (two turns). |
| `turn.json` | `last: rcwm` |
| Thermal | No `.thermal_pause`. GPU 63–70 °C, tz01 about 94–96 °C, about 11–16 GB RAM free. |

**[I] Near-term outlook.** After 5a comes BWM bundleE (B3 distillation, about 11,940 s, an estimate the reader called optimistic). After that come RCWM PREREG-LB (2.5 GPU-h in the RCWM report, about 1.7 in the coordination report), 5b, and the retention confirmation runs. Under strict alternation of 2–3 turns per programme per day, **there is no idle GPU slot for at least a day**.

### 1.2 BWM-001 (forward world model and planning)

**Owns (joint roadmap `BWM/research/JOINT_ROADMAP_2026-10-08.md`):**
- #1 frozen Blocksworld runs: hard result final at 73/110 vs 5/110; mirror interim 38/70; PB-600 pending
- #4 search distillation (B3, preregistered in `PREREG_B3_DISTILL.md`, sha `e6aa884b…`, queued as bundleE)
- #7 extra seeds
- #9 second domain (Logistics or Mystery Regime B)
- planned D-54 key-order input augmentation
- code for joint item #3

**Queue (`BWM/runs/jobs.json`):**
1. bundleD: mirror shards 5–7, then PB-600 direct, then PB-600 beam
2. bundleE: B3
3. a2-train-Fvlab, P_actions_s1_1600, E2 sample-efficiency trainings, e2-xlimB
4. bundleF: RW2 pilotG

**Not done anywhere in BWM:**
- learned value, critic or successor features
- options or macros
- GNNs
- runs on FlagWorld or Mystery (only a smoke test exists)
- RW2 self-distillation runs (`scripts/self_distill.py` is written but has never run)

### 1.3 RCWM-T1 (consequence world model)

**Owns:**
- PREREG-3 (done; verdict TRIGGER)
- 5a prerequisite representation (now waiting for the GPU)
- 5b register trace / plain / filler. Its identification fix wf_84e8f08d-166 is being edited right now: `scripts/t2br_common.py` was modified at 22:20.
- PREREG-LB, which RCWM runs and for which it holds the answer key (hook 70 reserved)
- the 5c retention suite (frozen 08:03; R17 selected 5 of 8 categories)
- the thermal watchdog (the only writer of the pause flag)

**Planned but not started:**
- #5 dissect T2: LoRA on blocks 8–15 vs 16–27, an α sweep, and **self-distillation (SD) and natural-marginal (NAT) controls**
- #6 long horizons (k=8/16) and seed-consensus certificates
- RCWM-roadmap #8: **no-LLM learners at matched labels** (S-roll and a digit transformer on "Titan's 629 ledger rows", scored on the out-of-range strata, CPU only)
- PREREG-4: a T3 adapter combining ledger with the winning P and R formats, around the end of week 2
- joint #8: exact arithmetic in the step

**RCWM's "do not" list:**
- dense merges
- tuning on spent instruments
- more ON/OFF deferral or voting
- latent single-pass composition
- self-label distillation now
- reporting tool tiers as model ability

### 1.4 Dataset factory (a third session, CPU-only, macOS, authored by Luncy-May)

- 3,340 active approved records across 648 families and all ten tracks.
- **No model has been trained or evaluated** on them.
- **No sealed split exists.**
- Nothing in it is addressed to us. Its implicit requests to later sessions are G05 (A/B/C/D plus matched SFT), G06 (an acquisition/retention matrix) and G10 (authorized training plus independent sealed evaluation).

### 1.5 Do-not-touch list

- **GPU and thermal control files**
  - Any GPU work outside the lease.
  - `turn.json` and the `waiting.{rcwm,bwm}` markers. The protocol is hard-coded for two parties.
  - The `.thermal_pause` flag. Only RCWM's watchdog writes it.
  - Reclaiming any lock whose `worker_pids` is non-empty.
- **Frozen code**
  - `BWM/bwmn/blocks_planner.py`, frozen at sha `18a5f22b…` (D-57) until the ledger-beam run finishes.
  - `RCWM/rcwmt/runtime.py`.
  - Freeze-bound docs.
- **Weights**
  - Registered T2 adapters `RCWM/runs/t2/t2_s{0,1,2}/adapter` (s0 sha `92bc7c74…`). Load them only unmerged in FP32 and never use them to label out of range.
  - BWM adapters under `BWM/runs/*/adapter`.
  - Titan-Sparse FP32 (sha `a28629c9…`). It must not be converted to BF16.
- **Spent or sealed instruments**
  - BWM: hard-110, mirror, PlanBench-600, and **b3fresh** (`BWM/runs/BW_val_large/instances_b3_fresh.json`, sha `575c45b1…`, sealed for B3).
  - RCWM: `HSKILL_CONFIRM_V2*` (never open), `bank_confirm.json`, `bank_confirm2.json`, pilot TEST, `t2_eval`/supp, R14's 248 prompts, L1's 456 prompts and its OOD bank, SELECT-VIEWS-001, ledger-beam dev seeds 20261141–43, `bank_p3_oor.jsonl` (the PREREG-3 bank, already used), and the retention suite's sealed draws.
  - RCI-001/RCI-002 calibration banks, spent for confirmation.
- **Taken seeds:** BWM 20261005–20261011; RCWM retention dev draw `dev_20261121`; ledger-beam 20261141–43; factory 20261009.
- **Repo and working copies**
  - Do not write into peer folders: `rcwm/`, `bwm/`, `tools/`, `datasets/`, the root `README.md` and `.gitignore`.
  - Never check out, reset or clean REPO or `Rineya_BWM_001_Research_Bundle/gh_export`.
  - Do not write data inside the RCWM or BWM bundles. Their freshness harvesters scan those trees (I-28).
- **Factory:** do not run `python -m factory.catalog`, `-m factory.status`, any `generate()`, the `factory.pipeline` release step or `write_*`. All of them write files.

---

## 2. Overlap map

| Track | Peers already did | Peers doing or queued | Factory data | Genuinely complementary (open) | Coordinate on |
|---|---|---|---|---|---|
| **RCM-001** | PREREG-3 found the TRIGGER failure (EV03). T2 recipe. D-35 surface audit. R16: models are key-order-locked and name-invariant. Joint trace format D-34/D-52 with filler rule 4b. | 5a (P-family surface diversity with a held-out surface stage E3). 5b (R-family trace vs plain vs filler). Planned: #5 SD/NAT controls, CPU no-LLM learners on 629 rows, joint #8 exact arithmetic, PREREG-4 T3. | v0001 (240) and v0003 (1,200) causal rule-AST data on `{x,y}`, with not(rule) twins, answer_only vs explicit_trace, and forward/reverse key order. Only train and an exposed dev_challenge split. | **Ledger rule repair**: training on balanced apply/no-op data with nonce statuses and inverted-rule pairs so the T2 trigger becomes the rule. A 2x2 of surface × trace on ledger, with a canonicalization control. A rule parser plus exact interpreter ceiling. E02 and E03. No one has a ledger counterfactual-data arm. [I, based on the reports] | Register predicates overlap 5b, so start with ledger only. SD control overlaps RCWM #5. CPU learners overlap RCWM #8. Ledger domain kits are RCWM's: "extend T2b kits; do not regenerate or retune frozen banks." |
| **RGM-001** | Blocksworld simulator and VAL harness. F-08 surface audit (reversed keys 0.458). plan_beam with a hand-coded unmet-goal heuristic. 73/110. | D-54 key-order augmentation. B3. #9 second domain. Mystery/STRIPS parser (smoke test only). FlagWorld built but never run. | v0001 (72) and v0002 (360) graph and blocks tasks with held-out names and larger counts. Trajectory export. | E07: text vs canonical vs flat MLP vs GNN with gold-graph and parsed-graph tiers. E08: a learned heuristic with exact dynamics. All 37 hard failures were unverified (23 dead_end, 14 budget). E09 program induction. | D-54 is effectively a canonicalization/augmentation arm. #9 second domain. Seeds and exclusion invariants. Use the planner read-only. |
| **RVM-001** | Uncertainty-gated adapter-off verifier (F-03/F-04). RCWM goal-factorized decisions (`decide.py`, 256 goals). | Joint #6: a shared legality-checker LoRA on minimal pairs, not started. **This directly overlaps E04.** `abstain_score.py` exists but has never run. | v0002, v0003 and v0004 critic data (384 active), with process_reward and preference exports. Residual plan-length shortcut: 0.812 vs a 0.743 majority. | Successor features (none exist anywhere). A value/dead-end critic for beam ranking. A held-out error-producer critic. A CPU logistic legality critic as the baseline. | Ownership of #6. The "blanket verifier" regression is a mandatory baseline. |
| **RSM-001** | None. | **B3 = E10**, already preregistered and queued. The bundle says: "Extend the existing BWM search-distillation item." | v0001: 24 rows, which is too small. | E11 option composition (AB/BA/conditional) and E12 DAgger. Both come after B3 outputs exist. | Do not run E10. Request B3 artifacts once B3 finishes. |

---

## 3. RCM-001 launch spec: E01 "Rule semantics factorial"

**Status.** E01 is `Planned` with `approved: false`. It depends on "G0: audit / teacher qualification / preregistration"; G0 is not defined anywhere in the bundle. Anything below not quoted from BUNDLE is a **proposal** to be frozen in the preregistration.

### 3.1 Hypothesis (H01)

> "Can supplied rule semantics control behavior when vocabulary, arithmetic range and predicate structure change?"

- **Mechanism.** Split each transition into three parts: applicability `C(rule,state,action)`, calculation `A(state,action)`, and assembly. "Treat the rule as observed; do not claim to discover hidden causal structure."
- **Prior expectation.** High confidence in fewer surface shortcuts; moderate for new predicate combinations; low for language transfer. "More balanced data alone may explain most of the gain."

### 3.2 Environments

- The bundle says: "Start with ledger statuses and register predicates." These are the RCWM `ledger_replay` and `register_machine` families in `RCWM/rcwmt/env.py`.
- **Proposal:** run the E01 pilot on **ledger first**. RCWM's 5b is already screening register traces, and register strict-`>` predicates belong to E02's computed-predicate scope.
- **[I] Important labeling gap.** `env.step` hard-codes "applied only if `status == "posted"`". Inverted or nonce-required-status rules therefore need a **new rule-parameterized interpreter**. Qualify it with `TEACHER_QUALIFICATION.md`, and cross-check it against `env.step`, `p3_common.rule_gold` and the FDS oracles on the posted-rule subset.
- **T2 may not label** any nonce or out-of-range data ("T2 is not an unrestricted ledger oracle").

### 3.3 Splits (custody follows COMMON_PROTOCOL)

- **Parents.** About 200 independent ledger parents for the pilot. Assign each whole parent to TRAIN, DEV or SEALED **before rendering**. Store canonical semantic hashes plus a `family_group_id` for twins.
- **TRAIN.** Balanced apply/no-op pairs, opposite rule meanings, nonce status words, and input permutations. Keep the literal target order `cmp, calc, after`. Use a fixed filler length per (family, action kind), chosen from TRAIN only.
- **Factor pools (proposal).** Use disjoint TRAIN, DEV and SEALED pools for each of:
  - status vocabulary
  - rule-paraphrase templates
  - renderers (key orders, account-naming schemes)

  The SEALED status words must not appear in PREREG-3's list (pending, draft, cancelled, reversed, held, approved, active, ok, dax, wug, Void/VOID/voided, unposted, not_posted) or in the factory's xelth/quom, because all of those are exposed.
- **OOD composite (primary).** Held-out semantics plus held-out surface forms. Report each axis separately: surface, semantic, and combined. The numerical axis (M_bal/M_amt) is report-only or protected in E01; it is primary only in E02.
- **Confirmation.** "Must use unseen parent groups and renderers." Its size is set by simulation from pilot discordance. Never confirm on old banks.
- **Audit.** Use `BUNDLE/08_reference_tools/audit_splits.py`. Map SEALED to `test`, because the tool accepts only `train|dev|test`.

### 3.4 Arms and baselines

**Factorial cells (all "student" LoRA adapters).** These follow the T2 recipe: r16/α32 on q,k,v,o,gate,up,down; frozen FP32 Titan-Sparse; unmerged.

| | Ordinary data | Counterfactual / surface-diversified data |
|---|---|---|
| Answer-only | F11 | F21 |
| Predicate trace `{cmp,calc,after}` | F12 | F22 |

**Required controls (BUNDLE E01 row):**
- existing unmerged T2 (s0 is the registered artifact)
- matched self-distillation
- balanced answer-only training
- canonicalized input
- trace with fixed filler length per family/action kind

Match oracle label access, and report both processed and supervised tokens. Include both same-answer/different-predicate pairs and opposite-answer pairs.

**Additional anchors (proposed, drawn from peer interfaces):**
- **OFF:** Titan-Sparse with `disable_adapter()` (tier A).
- **Ceiling:** a rule parser plus the exact interpreter (a B-type tool tier, labelled as privileged).
- **Rule-based shortcut floors**, each scored on the same items:
  - COPY (no-op always)
  - IGNORE_STATUS (always apply; this already exists as `LedgerF(arm="IGNORE_STATUS")` in `RCWM/scripts/lbeam_f.py`)
  - TRIGGER (no-op iff the status case-folds to a trained no-op token)
  - CONTAINS-POSTED substring cue
  - majority class
  - status-only lookup
- **RCWM-style approach.** T2 itself, plus F11 trained as a fresh T2-style recipe on balanced data. Together these separate "more balanced data" from counterfactual structure.

### 3.5 Operationalizing PREREG-3's rule-vs-trigger failure as anti-shortcut tests

**Source [M]:** EV03 and `RCWM/runs/bank_p3_oor/analysis/PREREG_3_RESULTS.md`.
- Unseen no-op statuses scored 0.01–0.18 against 0.99–1.00 on the trained "void".
- N_voidcase scored 0.94–0.99.
- 1,605 of 2,030 errors were applied_as_posted.
- Key order scored 0.37–0.47.
- Status-sensitive pairs scored 0.00–0.03 for both base models.

Reuse PREREG-3's **stratum taxonomy and verdict logic**, but **not its rows**, and use fresh vocabularies.

| Test | Construction | Shortcut detected | Proposed pass rule |
|---|---|---|---|
| T1 Unseen-status no-op | Fresh nonce non-posted statuses, about 8 strata | Token trigger | Each stratum ≥ 0.80 with Wilson lower bound ≥ 0.70, in every seed (the PREREG-3 RULE bar) |
| T2 Trigger near-miss | Case and morphology variants of TRAIN's no-op words | Spelling-level trigger (diagnostic) | Report only. A high score here with a low T1 score is the TRIGGER signature. |
| T3 Posted-like negations | Analogues of unposted / not_posted / "not posted" | Substring cue | Same bar as T1 |
| T4 Inverted-rule twins | Same observation and action; rule flipped (for example, the required status is a nonce word, so "posted" becomes the no-op) | Ignoring the rule text | Pair accuracy (both twins exact). The OFF and T2 baselines are expected near 0. |
| T5 Rule ablation or swap | Rule text removed, or replaced with an unrelated rule | Rule not read | Accuracy on rule-dependent pairs must drop to the floor |
| T6 Error direction | Classify each error as applied_as_posted, false-copy or format | Trigger and copy shortcuts | TRIGGER signature = more than half of errors are applied_as_posted. Also track false-copy, since T2's prerequisite false-copy rose by +0.53 to +0.80. |
| T7 Surface | Top-level key swap, all keys permuted, word or letter account names, 5–8 accounts | Format lock (R16) | Held-out orders only; this is the D-54 logic |
| T8 Magnitude | Analogues of M_bal and M_amt | Range-bound arithmetic | Protected or report-only in E01 |
| T9 Twin guard | Posted, in-distribution twins | Over-correction toward no-op | T_posted ≥ 0.80 and copied ≤ 0.10 |
| T10 Floor validation (CPU, before training) | Score every rule-based shortcut floor on the DEV composite | An instrument that cannot tell RULE from TRIGGER | TRIGGER, COPY and IGNORE must score near 0 on T4 pairs and far below the ceiling on T1 and T3. If not, redesign the composite before any GPU spend. |

**Secondary per-arm verdict (proposal).** Classify each arm with PREREG-3's own definitions:
- **RULE:** all unseen-status strata at the bar in all seeds, plus the twin guard.
- **TRIGGER:** unseen non-trigger strata ≤ 0.30, trigger strata ≥ 0.80, and more than half of errors applied_as_posted.
- **MIXED:** anything else.

### 3.6 Metrics

- **Primary:** "parent-macro exact transition accuracy on a predeclared held-out semantic/surface composite."
- **Secondary:**
  - predicate (`cmp.ok`) accuracy
  - false-copy rate
  - false-apply rate
  - exact K4/K8/K16 rollout
  - worst stratum
  - format errors counted separately; malformed output is a failure

### 3.7 Gates

- **Development target.** +10 pp over the **strongest matched non-oracle baseline** on the OOD composite, a positive corrected paired interval, and no protected-stratum regression beyond the preregistered margin. The margins are not defined yet.
- **Failure readings:**
  - If gains vanish with canonicalized input, credit the interface.
  - If only known predicates improve, narrow the claim.
  - If traces lose to balanced answer-only, keep the simpler recipe.
- **Retention.** Every trained candidate also goes through **E33** under the D-36 gate.

### 3.8 Statistical protocol

- The unit is the semantic parent. Renderers and twins stay inside the parent's cluster.
- Use `paired_metrics.analyze`: average variants within each parent, then a percentile bootstrap over parents. Use exact McNemar only when there is one binary row per parent.
- Apply a Holm correction over a small primary family. There is one primary endpoint and one primary comparison.
- Freeze the arms, partitions, token accounting and stopping rules before training.
- The pilot uses one seed. A replicated claim needs at least 3 independent seeds.
- Record every attempted confirmation.
- "A positive point estimate with a wide interval is not a pass."
- Outcomes are ADVANCE / REVISE / STOP / INCONCLUSIVE.

### 3.9 Compute [I]

- The RCM shared pilot cap is 4 GPU-h. It is not an allocation.
- T2 s0 took 1,858.8 s to train, with a 2.61 GiB peak.
- Four factorial cells plus about three trained controls is roughly 3.5 h of training alone at T2 scale, before evaluation.
- So: profile about 20 steps first, then shrink the pilot (fewer steps, or the 2x2 cells before the controls), or request a larger cap.
- Native execution requires explicit operator approval (LAUNCH_CHECKLIST item 6).

### 3.10 What a meaningful CPU-only E01 looks like (label it non-native)

1. A generator plus a rule-parameterized interpreter, qualified against `env.step`, `p3_common.rule_gold` and the FDS oracles.
2. Split audit, semantic hashes, and filler lengths chosen from TRAIN.
3. **T10 floor validation** plus the interpreter ceiling on DEV.
4. Power simulation from assumed discordance. The `SimulatedF` presets SIM_T2_P3 and SIM_OFF in `lbeam_f.py` are a possible source.
5. Small non-LLM learners (flat MLP or tiny transformer) trained on ordinary vs counterfactual data at matched labels. This tests the data factor on CPU.
   - Coordinate with RCWM-roadmap #8, which covers S-roll and a digit transformer on 629 rows.
   - The CPU reference package `REPO/rcwm/experiments/reference/rcwm/` (60 tests) may serve. The exact path is uncertain.
6. Optionally, OFF-arm inference on a small DEV slice using the CPU torch recipe, only with at least 12 GB RAM free. Throughput has not been measured.

`load_registered_t2()` is GPU-only, so T2 cannot be evaluated on CPU through it. None of this establishes Rineya efficacy.

### 3.11 Reusing factory RCM data

- **Location:** FR `data/RCM-001/releases/v0001` (240 rows, 30 families) and `v0003` (1,200 rows, 150 families). v0002 is superseded.
- **Loading:** only through `factory.catalog.catalog()`. Never concatenate release directories.
- **Schema (required fields):** `sample_id, semantic_parent_id, semantic_family_id, track_id, skill_tags, difficulty, generator_version, oracle_version, generation_seed, source_provenance, semantic_task, rendered_input, verified_target, target_format, split, review_status, review_receipt_ids, record_hash`.
  - `semantic_task`: `{kind:causal, state:{x,y}, rule AST, precondition AST, events[], annotation}`.
  - Targets: `{after}` for answer_only, or `{after, steps:[{cmp,calc,after}]}` for explicit_trace.
- **Sealed status.** Nothing in the factory is sealed, so nothing there restricts use for sealing reasons. But the dev_challenge split is **exposed**, so the data **cannot serve as confirmation or as a sealed OOD instrument**.
- **Known weaknesses:**
  - The nested dev gate is redundant in 16 of 30 families.
  - 214 train variants share AST motifs with dev.
  - Only the first status is held out.
- **Best use:** an auxiliary DEV or second-domain development substrate. Generate fresh held-out sets in memory by calling the pure builders and `oracles.solve` with new indices and vocabularies, writing nothing into FR.
- **Training restriction:** "Only independently approved active release bytes may be considered for a separately authorized future training experiment."
- **[I] Trace order may conflict.** The factory's top-level trace order `{after, steps}` may not satisfy the "literal `cmp, calc, after`" rule, so a converter is needed. Verify the actual key order first.

---

## 4. Reusable interfaces (import read-only)

**Import hygiene for every import below:**
- Set `sys.dont_write_bytecode=True` and insert the path with `sys.path.insert(0, …)`.
- For the factory, run `python -B`.
- For CPU torch, set `CUDA_VISIBLE_DEVICES=""` before importing torch and patch `torch.cuda.is_available`/`device_count`. Never use `-1`.
- Venv: `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv` (Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0).
- Pin the sha256 of every imported file.

| Purpose | Module and functions | Path | Notes |
|---|---|---|---|
| Paired statistics | `mcnemar_exact`, `quantile`, `analyze(rows, resamples, seed)` | `BUNDLE/08_reference_tools/paired_metrics.py` | Stdlib only. No Holm, seed hierarchy, noninferiority or power. |
| Split audit | `audit_records(records)` | `BUNDLE/08_reference_tools/audit_splits.py` | Splits must be `train`/`dev`/`test`. Trusts the supplied hashes. |
| Bundle integrity | `validate(root)` | `BUNDLE/08_reference_tools/validate_bundle.py` | |
| Holm, Wilson, crossed bootstrap | `stats.wilson`, `mcnemar_exact`, `paired_table`, `holm`, `crossed_cluster_bootstrap`; `flops.generation_flops` | `BWM/bwmn/` | Whether `stats` pulls in torch is not stated; check. |
| Ledger and register dynamics, prompts, parser | `step`, `rollout`, `applicable`, `features`, `gen_start`, `gen_event`, `one_step_messages`, `multi_step_messages`, `parse_state`, `dumps`, `SYSTEM_ONE/MULTI`, `cross_check` | `RCWM/rcwmt/env.py` | Stdlib. The ledger rule is hard-coded to `"posted"`. |
| PREREG-3 helpers | `rule_gold`, `messages(before, ev, ko)`, `parse_general`, `score_onestep`, `wilson` | `RCWM/scripts/p3_common.py` | Torch-free. |
| Ledger F arms and wrappers | `LedgerF(T2/OFF/ORACLE/IGNORE_STATUS)`, `RangeGuardF`, `SurfaceAuditF`, `BudgetF`, `SimulatedF` | `RCWM/scripts/lbeam_f.py`, `lbeam_problems.py`, `lbeam_proposer.py` | `load_registered_t2()` is GPU-only. |
| Surface renderers, trace parsers | t2b kits | `RCWM/scripts/t2bp_common.py`, `t2br_common.py` | `t2br_common.py` is **being edited live**; pin it after the fix lands. |
| External oracle | `ORACLES` | `C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/oracles.py` | Validated 384/384. |
| Trace schema | `rw2_target`, `bw_target`, `filler_length`, `parse_trace`, `after_text` | `BWM/bwmn/traces.py` | Imports `lm`, so it needs torch. Literal order `{"cmp","calc","after"}`. Filler format `{"fill":" . .","after":…}`. |
| Blocksworld and STRIPS (RGM) | `B.step`, `applicable`, `bfs_plan`, `execute`, `val_check`, `generate`; `rw2.exact_step`; `mystery.step` | `BWM/bwmn/blocks.py`, `rw2.py`, `mystery.py` | Do not copy `bwmn` without `bench/`. |
| Planner | `BP.plan_beam(..., goal_met=, unmet=)` | `BWM/bwmn/blocks_planner.py` | Frozen at `18a5f22b…`. Pass your own `cache` path to `LMForwardBW`. |
| Retention graders | `ret_grade.py`, `ret_gen.py` | `RCWM/scripts/` | Custody belongs to RCWM. Do not open or recreate the protected instrument. |
| Factory | `catalog()`, `oracles.solve`, `factory.common.{canonical, digest, record_hash, load_jsonl}`, `factory.lineage.semantic_fingerprint`, `generators.causal.generate.CONTRACT` | FR | Read-only only (see §1.5). |

**Data contracts** (`BUNDLE/04_experiments/DATA_CONTRACTS.md`, `task_example.json`, `target_example.json`):
- **Task record:** `parent_id, split, generator_version, observation, rule, action, goal, evaluator_only{…}`.
- **Target record:** `cmp{observed_status, required_status, ok}, calc{balance}, after{balance}`.
- **Learned-output capture record:** experiment, run, seed, parent, renderer, tier, hashes, output, parse validity, calls/tokens, wall time.

---

## 5. Protocol and reporting requirements

**Preregistration** (`BUNDLE/06_templates/PREREGISTRATION.md`; header "Explicit execution approval: NOT GRANTED"):
1. **Claim and scope:** one hypothesis, task family, observations, model fingerprint, tier, practical effect, and what a positive result would not establish.
2. **Custody and overlap:** commits, file SHA-256, model, tokenizer and adapter hashes, generator, parser, the list of spent/reserved/sealed banks, parent-before-render grouping, and hash checks.
3. **Arms and fairness:** information available, teacher access, labels, processed and supervised tokens, hyperparameter search effort, inference budget, a **non-LM deterministic baseline**, the A/B/C/D tiers, and matched SFT.
4. **Endpoints and sampling:** primary endpoint, direction, unit, protected strata, scoring of errors/timeouts/invalid parses, clusters, renders per parent, seeds, and a blinded power simulation.
5. **Decision rule:** GO / NO-GO / INCONCLUSIVE, multiplicity, threshold, retention margins, and resource limits.
6. **Execution and safety:** cap and stop policy, restart rules, partial-output rules.
7. **Freeze and deviations:** freeze hash, timestamp, reviewers, create-only outputs, and a deviation log recording whether outcomes were known. No rerun under the same ID.

**Teacher qualification** (`TEACHER_QUALIFICATION.md`):
- Status is UNQUALIFIED by default.
- Record the supported range and the full error taxonomy, including rejected labels.
- Define a fail-closed eligibility predicate before generation.
- Requalify on any change.
- My rule interpreter needs this.

**Decision log** (`DECISION_LOG.csv`) header: `decision_id, timestamp_timezone, experiment_id, prior_status, decision, evidence_path, outcomes_known, reason, author, approver`.

**Run manifest** (`RUN_MANIFEST.template.json`):
- `schema_version "RCI-001/planning-v1"`, `status "DRAFT_UNAUTHORIZED"`
- `experiment_id, repo_commit, model/tokenizer/adapter_sha256, registered_precision, teacher_qualification, split_manifest_sha256, preregistration_sha256, independent_seed, output_directory, gpu_lease, thermal_watchdog`
- `authorized:false, paid_services_authorized:false, publication_authorized:false`
- The runner must fail closed.

**Results report and model card:**
- The claim status includes INVALID.
- Report what actually ran, numerators and denominators, B−A, C−A, matched SFT, per-category results under both graders, each OOD axis, and attribution by ablation.
- The model card says "NOT A RELEASE" and gives the artifact type: adapter / external / modular / consolidated.

**A/B/C/D tiers:**
- **A** = frozen core with no assistance (OFF).
- **B** = core plus the specialist or tool (for example, the exact interpreter).
- **C** = fresh student without assistance.
- **D** = C plus the same assistance as B.
- A matched-SFT control is mandatory.
- Pin prompts, decoding, adapter state and allowed observations.
- E01's trained cells are C-type component artifacts. The full four-arm test is E31, which "requires a separately approved confirmation budget."
- Never call an adapter on a dense backbone a dense checkpoint.

**E33 retention:**
- No-go on any Holm-McNemar-detected category loss, a category net below −11, or a pooled net below −4% of items, under **both** graders. The old −5/75 rule is diagnostic only.
- RCWM's suite has 5 of 8 categories selected (R17). Each confirmation mints its own sealed draw. No confirmation has run yet.
- KL and greedy agreement are diagnostics only.
- A dense-student claim needs a prospectively powered noninferiority margin.

**Custody:**
- Record base and adapter hashes before and after every run.
- Loader parity before training.
- Graders stay immutable during evaluation.

**Launch checklist** (6 items):
1. Not a duplicate; custody of commit, checkpoints, tokenizer and data known.
2. Teacher qualified, or an exact interpreter supplies labels; T2 stays unmerged FP32.
3. Splits, overlap checks, trace order and filler policy pass; no gold state reaches the model.
4. Controls and budgets fixed; power design matches the claim.
5. CPU tests pass; on-device loader parity, memory, lease, watchdog and restart rules verified.
6. Native execution explicitly approved.

"A failed checklist item is a blocker." `AGENT_HANDOFF.md` says: one experiment only, and "Never fabricate a successful native run."

---

## 6. Coordination

### 6.1 Lease and thermal rules

**GPU:**
- **CPU-only by default.** GPU use needs a request to both sessions plus operator go-ahead.
- Use the lock `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock`, via a copy of `RCWM/rcwmt/lease_vendored.py` (token-checked release, 256 MiB foreign-holder rule).
- Use a distinct label prefix. `rcx:` is proposed; `rcx`/`rci:` had zero hits, but re-grep first.
- Never write `turn.json` or the waiting markers.
- Take the card only when the lock is absent and no live waiter exists, or through a pre-agreed one-shot `.yield` file: `RCWM/runs/.yield*.json` or `BWM/runs/.yield.json`.
- Turns are at most 4.5 h. Kill the process tree (`taskkill /F /T`) at the cap.
- In-process limits: `set_per_process_memory_fraction(0.70)`; VRAM floor 3.6 GiB; host-RAM floor 3 GB; duty cap of at least 1.5×.
- Check `RCWM/runs/.thermal_pause` between batches.

**Watchdog thresholds:** pause at ≥ 76 °C, resume at ≤ 66 °C; pause on battery; pause below 3 GB RAM; pause at tz01 ≥ 98 °C.

**CPU work:**
- At most 6 threads.
- Load models only with at least 12 GB RAM free.
- Pause at tz01 ≥ 98 °C until ≤ 90 °C.
- No per-sample subprocesses (I-27). Read `thermal_log.txt` or `health_now.json` instead.

### 6.2 Folder and COLLABORATION.md conventions

- **Where to write:** only under `WT/complementary/`, on `research/rineya-complementary-intelligence-v1`.
- **Push and merge:** push to an origin branch of the same name. No merge to main without the operator. If main is ever touched, run `git pull --rebase` and use the commit prefix `complementary:` with the Co-Authored-By trailer.
- **Files:** at most 25 MB each, otherwise listed in `complementary/EXCLUDED_LARGE_FILES.json`. No weights without approval. LF line endings. Run the `tools/export_bwm.py` secret/email regex before every push.
- **Docs:**
  - `complementary/docs/COLLABORATION.md` holds scope, IDs, GPU policy and data hygiene.
  - `complementary/docs/REQUESTS.md` holds requests: ID, addressee, files read with sha, resources, status.
  - Also deliver every request by message, because peers log agreements in their own DECISIONS.md.
- **IDs:**
  - Prefix our IDs (for example "CX D-01").
  - Do not use RCI-001 as a run or bank label.
  - Cite peer results as commit plus path plus sha.
- **Cleanup:** proposing a `complementary/` row in the root README needs both peers plus the operator. Remove the scratchpad `dataset_ro` worktree (`git worktree remove`) when done, because it is registered in the shared repo.

### 6.3 Proposed initial messages

**To RCWM-T1:**
> Complementary session (CPU-only for now, label `rcx:` if ever GPU) planning bundle E01 "rule semantics factorial" on the **ledger** family: balanced apply/no-op, nonce statuses, inverted-rule twins, answer-only vs `cmp/calc/after` trace, canonicalized-input control — aimed at turning PREREG-3's TRIGGER into RULE. Requests: (1) Confirm no ledger rule-repair arm is planned in #5/PREREG-4, and that register stays with 5b. (2) OK to import read-only, sha-pinned: `rcwmt/env.py`, `p3_common.py`, `lbeam_f.py`, and `t2br_common.py` after wf_84e8f08d-166 lands? (3) OK to reuse PREREG-3's stratum taxonomy with fresh nonce words and fresh seeds? Which seed ranges/exclusions should I avoid? (4) Your #8 no-LLM learners: I'll only run CPU learners on ordinary-vs-counterfactual data. Want to share a baseline instead? (5) Would you run D-36 retention confirms for any future candidate, or should I request a draw? No GPU ask now; I won't write turn or waiting files.

**To BWM-001:**
> Complementary session, CPU-only for now. Requests: (1) Confirm B3 is bundle E10. I won't run search distillation; may I request B3 artifacts later for option-composition/DAgger work? (2) RGM E07 (text/canonical/flat-MLP/GNN on fresh Blocksworld via `B.generate`): which seed range is free, and should I exclude the hard/val_large/mirror/b3fresh invariants exactly as you do? Is D-54 key-order augmentation scheduled? I'll treat it as yours and only add a CPU canonicalization control. (3) Joint #6 legality checker: who owns it? I can offer a CPU logistic critic baseline only. (4) I'll import `bwmn` read-only at a pinned sha, use my own F cache, and never touch `blocks_planner.py` 18a5f22b. No GPU ask now.

---

## 7. Risks, ambiguities and open questions

1. **G0 is undefined.** Proposal: G0 = the launch checklist plus teacher qualification plus a frozen prereg.
2. **Name collisions.**
   - "RCI-001" is already an existing spent calibration bank (`rineya-dataset-retention-records/rci001-20260918/RCI_SEAL.json`).
   - RCM/RGM/RVM/RSM-001 are also the factory's track IDs.
   - Bundle templates hard-code `RCI-001/…` schema strings.
   - Open question: rename runs (e.g. `RCX-E01`) while keeping citations to the bundle?
3. **"Strongest matched non-oracle baseline" is adaptive if picked after outcomes.** Fix the candidate set and select on DEV before SEALED.
   - Open question: is F11 (balanced answer-only) a control or a factorial cell?
4. **Undefined quantities:** protected-stratum margins, confirmation n, the RGM "substantially close" threshold, and per-experiment caps. Decision vocabulary also differs: ADVANCE/REVISE/STOP vs GO/NO-GO/INCONCLUSIVE/INVALID.
5. **Labels need a new rule-parameterized interpreter.** `env.step` hard-codes "posted". Whether `rule_gold` is also hard-coded is not stated.
6. **Overlap risks.**
   - With RCWM: 5b (register traces), #5 SD control, #8 CPU learners, PREREG-4.
   - With BWM: D-54 (augmentation) and B3 (= E10).
   - Joint #6 vs E04.
7. **Split vocabularies don't match.** Bundle TRAIN/DEV/SEALED vs `audit_splits` train/dev/test vs factory train/dev_challenge.
8. **Factory caveats.** Exposed dev; same-model reviewers ("correlated-error risk"); possible trace-order conflict; ledger vs `{x,y}` AST domain mismatch; RSM/RBM/RJP/RDP/ROM pilots too small; RBM v2 PENDING and failed.
9. **GPU contention and cap.** The 4 GPU-h RCM cap probably cannot cover a 2x2 plus controls at T2 scale [I]. GPU access is a day or more away. A shard-5 crash (rc 0x40010004) is still unexplained, and under BWM's stop rule a TDR halts BWM GPU work.
10. **Stale vs live sources.** Peer bundles are newer than `a223884`. `t2br_common.py` is changing as of this writing. PREREG-LB GPU estimates disagree (2.5 vs 1.7 h).
11. **Path uncertainties:**
    - the RCWM CPU reference package path
    - whether `bwmn/stats` imports torch
    - "629 ledger rows" vs T2's 501 ledger rows (in its 1,501-row training set)
12. **Retention custody belongs to RCWM.** E33 for our candidates needs their cooperation or a new draw.
13. **Approval.** No native run, branch push, root-README edit or GPU lease without explicit operator approval. Peer or agent messages do not count as approval.