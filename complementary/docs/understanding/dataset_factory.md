# Rineya High-Quality Dataset Factory v1: read-only research report

**Target:** branch `research/rineya-high-quality-dataset-v1`, HEAD `7a7d055`, which matches `origin/research/rineya-high-quality-dataset-v1`.
**Factory root (FR):** `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-rineya-complementary-research-2026-10-09/3669d0b6-f9a8-4068-a5c7-094164c55025/scratchpad/dataset_ro/datasets/rineya_high_quality_factory_v1`

How to read the labels below:
- **[M]** a result recorded in the repo as actually executed.
- **[M-me]** something I measured myself during this read-only pass.
- **[P]** a plan or proposal.
- **[I]** my own inference.

Seven files under `reviews/adjudication/` did not check out because their paths are too long on Windows: `belief-gap-v1-execution-history/*`, `belief-gap-v1-source-snapshots/*` and `claim-gap-v2-final-machine-evidence/{leakage.json,oracle.jsonl}`. None of them are needed to use the active data.

---

## 1. Who produced it, why, and current status

**Git authorship.** All 12 dataset commits are by `Luncy-May` (operator account), between 2026-10-09 18:49 and 22:09 CDT:
- 517a51a: "preserve factory foundation and pending causal pilot"
- 8c2571e: "release independently approved causal and critic pilots"
- 265f768: "approve all ten exact-verified pilot tracks"
- ee7a91c: "publish P0 production milestone…"
- eac4777, 61530fc: backup and handoff records
- 75934fb: typed-JSON fix
- 98cd4b3: "Release reviewed REM history controls…"
- 2ceaa55: REM backup receipt
- e65f3c2: "frozen belief design"
- 7a7d055: "Preserve belief-gap reviews, whole-batch quarantine, and failed lexical revision evidence"

The branch is based on main `a223884`, authored by LUNCY579 (RCWM/BWM work). The history below the dataset commits is RCWM/BWM: PREREG-1/2/3, L1, and the BWM Blocksworld-hard result of 73/110 vs 5/110.

**Execution environment.** The work was done on macOS:
- Factory path: `/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset`
- Python: `/opt/anaconda3/bin/python3`
- Git worktree: `.rineya-dataset-git-worktree`
- Repo: private `LUNCY579/rineya-world-model-research`

**Session structure.** This was one Claude-agent orchestration. `research/ROLE_REGISTRY.json` lists:
- root `/root`, acting as causal producer, improvement engineer and gate-patch author
- `/root/relational_critic_producer`, producer for every other track
- `/root/research_architect`, which wrote the curriculum and later acts as adjudicator
- `/root/formal_oracle`
- `/root/expert_reviewer`
- `/root/adversarial_auditor`
- `/root/backup_manager`

The registry says these are "all same underlying model" and that there is "correlated-error risk". It also says there is "No fictional 12-agent independence claim".

**Purpose.** The README describes the content as "Original synthetic CPU-only data for ten complementary research tracks". The factory turns the complementary research bundle's track cards into exact, oracle-verified synthetic datasets.

[M-me] The bundle files it cites, with their SHA-256 in `CURRICULUM.md` and `DATASET_BLUEPRINT.json`, are byte-identical to your current directory `C:\Users\clclc\BusinessProject\WorldModel\rineya_complementary_research_2026-10-09`. I checked five of them: `README.md`, `04_experiments/tracks.json`, `experiment_registry.json`, `01_report/Rineya_Complementary_Research_Report.md` and `02_research_tracks/RCM-001.md`.

**Status [M]** (from PROJECT_STATUS and `manifests/ACTIVE_RELEASE_CATALOG.json`):
- **3,340 active approved canonical records from 648 source families, covering all 10 tracks.**
- P0 milestone reached: at least 150 reviewed families each for RCM, RGM and RVM; 12-family pilots for the other seven tracks.
- In progress: the RBM belief evidence-gap wave. v1 (1,296 rows) is entirely quarantined. v2 (1,728 rows) is PENDING and has already failed a residual-shortcut stage (details in section 2).
- Last full remote-byte-verified checkpoint for the session state: `session-final` = `eac4777`, 778 files, 191,806,213 bytes.
- Later receipts: `memory-gap-v2-final` = `98cd4b3` is BACKED_UP. The belief commits e65f3c2 and 7a7d055 are "await the next verified archival checkpoint".

---

## 2. Per-track datasets

### Common layout (all tracks)

Each track has `FR/data/<TRACK>/` with these subdirectories:
- `semantic_parents/<batch>.jsonl`
- `candidate_renderings/<batch>.jsonl`
- `reviewed/`
- `accepted/`
- `quarantine/`
- `manifests/` (memory and belief tracks)
- `releases/vNNNN/`

Each release directory contains `MANIFEST.json`, `CHECKSUMS.sha256`, `QUALITY_REPORT.md`, `canonical.jsonl`, `oracle_receipts.jsonl`, `review_receipts.jsonl` (3 roles × records), `train-structured.jsonl`, `train-sft.jsonl`, `dev_challenge-structured.jsonl`, `dev_challenge-sft.jsonl`, plus track-specific exports.

**Split names.** Only `train` and `dev_challenge` exist; the schema enum allows nothing else. The split is assigned per source family before rendering, via `factory/common.py::split_for(index)`, which returns `"dev_challenge" if index % 10 >= 8 else "train"`. That puts 20% of families in dev.

**There is no sealed test split anywhere.** Every audit carries `"structural_template_holdout": false` and the scope string "fresh generated development data; not a sealed generalization instrument".

**Canonical fields** (`schemas/canonical/v1.schema.json`, required): `sample_id, semantic_parent_id, semantic_family_id, track_id, skill_tags, difficulty, generator_version, oracle_version, generation_seed (20261009), source_provenance, semantic_task, rendered_input, verified_target, target_format, split, review_status, review_receipt_ids, record_hash`. Rows also usually carry `structural_difficulty, challenge_axes, source_family_key, renderer (forward|reverse key order), reviewed_candidate_hash`.

**Export formats:**
- SFT: `{"messages":[user, assistant]}`
- structured: `{"input","target"}`
- preference: `{"prompt","chosen","rejected"}`
- process_reward: `{"input","candidate","labels"}`

**Gate.** A record is released only when four things join by exact record hash: the independent oracle receipt, the expert receipt, the adversarial receipt and the adjudicator receipt.

**Review totals [M]** (DATASET_QUALITY_REPORT):

| Role | Records | ACCEPT | REVISE |
|---|---:|---:|---:|
| Expert | 5,176 | 4,294 | 882 |
| Adversarial | 4,408 | 4,220 | 188 |
| Adjudication | 4,168 | 4,028 | 140 |

### Active release inventory [M; re-verified M-me]

[M-me] `factory.catalog.catalog()` loaded on this machine read-only (with `-B`, so no bytecode was written) and returned 3,340 rows / 648 families. I re-ran `oracles.solve` against all 2,280 active RCM, RGM, RVM and RSM records: **0 mismatches**.

| Track | Release (batch) | Records | Families | train / dev | Status |
|---|---|---:|---:|---|---|
| RCM-001 | v0001 (pilot-v2, gen 1.1.0) | 240 | 30 | 192/48 | ACTIVE |
| RCM-001 | v0002 (expansion-v1) | 1200 | 150 | 960/240 | SUPERSEDED by v0003 (identical canonical sha `4744be7c…`; source-blob provenance issue) |
| RCM-001 | v0003 (expansion-v1, gen 1.2.0) | 1200 | 150 | 960/240 | ACTIVE |
| RGM-001 | v0001 (pilot-v3, 1.2.0) | 72 | 30 | 48/24 | ACTIVE |
| RGM-001 | v0002 (expansion-v3, 1.3.2) | 360 | 150 | 240/120 | ACTIVE |
| RVM-001 | v0001 (pilot-v2) | 60 | 30 | 48/12 | SUPERSEDED; erratum `quality_reports/errata/RVM-001-v0001.json` (6 false equal-route tags) |
| RVM-001 | v0002 (expansion-v2, 1.1.2) | 300 | 150 | 240/60 | ACTIVE |
| RVM-001 | v0003 (pilot-v4) | 60 | 30 | 48/12 | ACTIVE (supersedes v0001) |
| RVM-001 | v0004 (claim-gap-v2, 1.2.1) | 24 | 12 | 20/4 | ACTIVE |
| RSM-001 | v0001 (extended-pilot-v2) | 24 | 12 | 20/4 | ACTIVE |
| REM-001 | v0001 (extended-pilot-v3) | 120 | 12 | 100/20 | ACTIVE |
| REM-001 | v0002 (memory-gap-v2) | 768 | 12 | 640/128 | ACTIVE |
| RMM-001 | v0001 (extended-pilot-v3) | 76 | 12 | 60/16 | ACTIVE |
| RBM-001 | v0001 (extended-pilot-v2) | 24 | 12 | 20/4 | ACTIVE |
| RJP-001 / RDP-001 / ROM-001 | v0001 (extended-pilot-v2) | 24 each | 12 each | 20/4 | ACTIVE |

**Quarantined (non-empty) [M]:**
- `RBM-001/quarantine/belief-gap-v1.jsonl` (1,296)
- `REM-001/quarantine/memory-gap-v1.jsonl` (768; targets correct, but false property metadata)
- `RVM-001/quarantine/claim-gap-v1.jsonl` (24; renderer-hash alias bug)

**Pending:** `RBM-001/candidate_renderings/belief-gap-v2.jsonl` (1,728 rows, not approved).

### RCM-001 (causal / counterfactual rules)

**Generator.** `generators/causal/generate.py::generate(count, batch, offset)`.

**Task.** The model executes a supplied rule AST and precondition over 1–4 events on state `{x, y}`:
- Ops: `add`, `transfer`, `scale`, `swap`.
- Six predicate mechanisms (`predicate_0..5`): `eq`/`ne` status, `ge` amount threshold, `all(eq, ge)`, `any(ne, lt)`, `not(all(ne, gt))`.
- A `CONTRACT` text prefix defines the semantics. Prompts include the distractor `"annotation":"weather record: cloudy"`.

**Renderings per family: 8.**
- Counterfactual twin: `twin0` uses the rule, `twin1` uses `{"op":"not", ...}` of it.
- Formats: `answer_only` targets `{after}`; `explicit_trace` targets `{after, steps:[{cmp:{rule,precondition}, calc:{operation}, after}]}`.
- Key order: `forward` / `reverse`.

**Dev challenge axes:**
- `heldout_first_status`: nonce statuses `xelth` / `quom`.
- `larger_magnitudes`: ×100.
- `heldout_predicate_composition`: only for families with index ≥ 30, where an extra nested `all(rule, any(gt amount±1, lt state.x))` wrapper is added.

v0003 dev = 240 records, all carrying all three axes. Difficulty counts in v0003: L4 720, L3 320, L1 160.

**Example (abbreviated), `RCM-001-expansion-v1-p0030-twin0-answer_only-forward`, train:**
```
semantic_task: {kind:causal, state:{x:13,y:28}, rule:{op:eq,args:[{var:event.status},{const:"cancelled"}]},
  precondition:{op:ge,args:[{var:state.x},{var:event.amount}]},
  events:[{status:"cancelled",amount:6,op:"scale",source:"x",destination:"y"}], annotation:"weather record: cloudy"}
verified_target: {"after":{"x":78,"y":28}}   skill_tags:[rule_conditioning,counterfactual,conditional_execution,predicate_0]
review_receipt_ids: [expert-…, adv-tags-…, adj-expansion-v1-…]
```

**Oracle.** `oracles/independent.py::_causal`. Each receipt reads "independently authored state and factual metadata oracle recomputation".

**Expert scores** (one sample receipt; this may be a per-mechanism template rather than per-record): semantic_correctness 5, target_correctness 5, shortcut_resistance 3, diversity 3, generalization_value 3.

**Shortcut audit [M]** (`audits/shortcut_detection/expansion-v1.json`; stripped-feature lookups, no model):
- RCM length_only dev accuracy 0.50, below the majority baseline of 0.617.
- status_only 0.617, plan_length_only 0.617; these equal the majority baseline.
- `copy_state_accuracy` 0.473: a copy-the-input baseline gets the final state right 47.3% of the time.
- 'status_words_imply_execution' counterexamples exist: the same status leads to both outcomes across twins.

**Known limits [M]** (adjudication and `reviews/adjudication/EXPANSION_V1_SCOPE_REVIEW.md`):
- "150 families instantiate six base predicate families plus optional added guards… not new arithmetic or causal mechanisms".
- The nested dev gate is meaningful in only 14 of 30 dev families; in 16 it is "always true", i.e. redundant. Removing the wrapper changes 25/60 twin traces and 21/60 final states.
- "214 current training logical variants sharing AST motifs with previously exposed development".
- Only the first status is held out; later statuses overlap with train.

### RGM-001 (relational graphs and blocks)

**Generator.** `generators/relational/generate.py`, which calls `expansion.py::diversify` for families with index ≥ 30.

**Two task kinds:**
- `graph`: directed weighted edges with fuel; plan replay; first-illegal-step stop; `shortest_plan` uses BFS with lexicographic tie-break.
- `blocks`: an `on` map, `goal_on` (possibly partial), clear-block legality.

**Target:** `{steps:[{legal, after}], after, valid, reaches_goal, shortest_plan|null}`.

**Mechanisms in v0002:** branched_resource_graph 184, multiple_initial_table_stacks 118, single_successor_resource_graph 26, multiple_explicit_goal_table_roots 18, block_goal_composition 14.

**Renderings:** base × {forward, reverse}. Each dev family also gets a **`matched` variant with the opposite reaches_goal outcome** (RGM-FIX-004). That fix responded to the earlier v2 issue where all 6 dev parents had reaches_goal=false, which a constant-false label exploited at 100%.

**Dev axes:** `heldout_names` (Nacre/Obsidian/… vs Aster/Beryl/…) and `larger_graph_or_block_count` (7–8 nodes vs 3–6; 5 blocks vs 3–4).

**Example (abbreviated), `RGM-001-expansion-v3-p0030-base-forward`:**
```
task: {kind:graph, nodes:[Cedar,Dune,Beryl,Ember,Aster], edges:[[Cedar,Dune,3],[Dune,Beryl,2],…,[Aster,Cedar,6]],
  start:Cedar, goal:Aster, fuel:17, plan:[Dune,Aster]}
target: {steps:[{legal:true,after:{node:Dune,fuel:14}},{legal:true,after:{node:Aster,fuel:11}}],
  after:{node:Aster,fuel:11}, valid:true, reaches_goal:true, shortest_plan:[Beryl,Aster]}
```

**Extra export.** `*-trajectories.jsonl` contains valid, goal-reaching cases only: v0001 16+12, v0002 94+60.

**Shortcut audit [M]** (`audits/shortcut_detection/expansion-v3.json`):
- kind_only dev accuracy 0.783, equal to the majority baseline.
- length / plan_length 0.433.
- goal_label_by_plan_length 0.40, vs a 0.50 baseline.
- always-false goal label 0.50.
- Dev reaches_goal is balanced 30/30.

**Verdict history.** Expansion-v1 had 40 RGM REVISE decisions, all factual-tag errors (e.g. 26 non-branching graphs tagged "branched"). They were fixed in expansion-v3 (0 metadata failures).

### RVM-001 (validity critic / value)

**Generator.** `generators/critic/generate.py`, with `claim_controls.py::augment` (v0004) and `route_facts.py`.

**Task.** A world graph plus 9–17 candidates, each `{candidate_id (opaque cNNNNN), plan, claimed_after}`.

**Target.** `{evaluations:[{candidate_id, valid, claim_correct, reaches_goal, remaining_moves, progress, usefulness(-1/0/1/2), first_error(int|"claim"|null), repair_plan}], ranking:[[ids…],…]}`. Ranking uses the tie-groups tuple `(valid∧claim_correct, reaches_goal, usefulness, −remaining_moves, −len(plan))`.

**Example labels (v0004 row, abbreviated):**
```
{"candidate_id":"c36271","claim_correct":false,"first_error":"claim","progress":1,"reaches_goal":false,"remaining_moves":1,"repair_plan":["Cedar","Beryl"],"usefulness":1,"valid":true}
```

**Derived exports** (`factory/exports.py::specialized`):
- process_reward: one row per candidate. v0002 2,494/638; v0003 496/124; v0004 332/66.
- preference: strict pairs only, at most one each from the equal / longer / shorter plan-length buckets, selected by semantic hash rather than ID. v0002 720/180; v0003 144/36; v0004 60/12.
- The README says "These are not per-step scalar rewards".

**Shortcut audit [M]** (expansion-v2): plan_length_only dev accuracy **0.812 vs majority 0.743**, and candidate_length_only 0.790 vs 0.721. This is the "residual plan-length correlation" the handoff mentions. Position and ID buckets sit at the majority baseline. Every one of the 150 families has equal-length good/bad pairs and longer-valid vs shorter-invalid pairs.

**Claim-gap v0004 [M].** All 24 same-plan groups contain true, overclaim, underclaim and wrong-location claims. Two gaps remain:
- The global pure-fuel error mix is still 48 overclaim vs 24 underclaim.
- Only 2 families are in dev.

Expansion-v1 originally had "all 286 fuel-only incorrect claims overstate available fuel" (RVM-GAP-001).

### RSM-001 (skills)

**Generator.** `generators/skills/generate.py::make_task`, run through `generators/other_tracks/generate.py`.

**Task.** Execute supplied acyclic procedures: `add`/`scale`/`swap`/`call`/`if`. Target: `{after, executed_primitives, call_order, conditions}`.

**Mechanisms (4 renderings each):** repeated_procedure, conditional_subgoal, state_dependent_reuse, conditional_composition, arithmetic_composition, repeated_conditional.

**Dev axes:** `heldout_names` plus `larger_procedure_library` (adds the 4th procedure `cascade`).

**Example (dev, abbreviated):**
```
procedures:{sprout:[add x 1], threshold:[if x ge 2 then call sprout else add y -1], weave:[scale y 1, swap], cascade:[call threshold, call weave]}
main:[call threshold, call weave, call threshold, call cascade], state:{x:2,y:1}
target:{after:{x:1,y:1}, call_order:[threshold,sprout,weave,threshold,cascade,threshold,weave], conditions:[true,false,false], executed_primitives:7}
```

**Scope string:** "Finite supplied procedure execution; no learned hierarchy or skill acquisition claim."

**Shortcut audit:** "Twelve finite families, two exposed development families… No trained baseline."

### Other tracks (brief)

**REM-001 v0002 (memory-gap-v2), 768 records / 12 families.** Built from 48 base histories with query and key rotations, giving 3,072 rotation-level and 6,144 rendered query observations. Nine per-row properties are covered, e.g. `version_over_recency`, `authoritative_tombstone`, `present_null_distinction`, `absence_control`, all witnessed by `oracles/memory_controls_independent.py`.

Residual issues [M]:
- Status majority is 65.625%.
- A same-key event-count lookup gets 84.375% status accuracy but only 12.5% exact answer accuracy.
- Future-only absence is not covered.
- Dev has 2 families.

**RBM-001.** v0001 has 24 approved rows. A prospective audit found that "support-size/observed-fact shortcut rules match all 72 query flags".
- belief-gap-v1: adjudicated **all 1,296 REVISE** (432 lexical defects; 864 family dependency holds). Observation tokens `stone`/`unknown` predict emptiness perfectly.
- belief-gap-v2 (`data/RBM-001/manifests/belief-gap-v2.shortcut_audit.json`): the frozen "prospective_marker_and_auxiliary_token_empty" rule still gets **status_accuracy 1.0** on all 1,728 rows. bounded_decision: "Preserve actual first v2 PENDING evidence and failed residual shortcut stage; no approval."

**RMM, RJP, RDP, ROM.** Each is a 12-family finite proxy built in `generators/other_tracks/generate.py` and `generators/memory/generate.py`, scored by `oracles/extended_independent.py`.

---

## 3. Has any model training or evaluation been done?

**No.** No model has been loaded, trained or evaluated on these datasets. Supporting statements:
- Every MANIFEST: `"research_effectiveness": "untested; no model training performed"` and `"data_custody": "exposed train/dev data; no sealed evaluation"`.
- README: "No model training or efficacy evaluation is performed."
- PROJECT_STATUS: "No model loading, training or weights/architecture changes… model benefit remains untested."
- Latest addenda: "No training, efficacy result or sealed evaluation was performed."

**What was executed instead [M]:**
- Independent oracle recomputation.
- Schema, duplicate and leakage audits.
- Stripped-feature lookup baselines (majority, length, ID, position, token tables).
- Metamorphic tests: key order, annotation, status renaming.
- LLM-agent expert, red-team and adjudicator reviews. For belief-gap the agents also made "input-only predictions" that were frozen before seeing gold. [I] That is review methodology, not a model-efficacy evaluation.
- 231 unit tests passed (`quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json`, "Ran 231 tests… OK"), plus 7 backup-helper tests.

The only model results quoted come from the source research, not from this factory: BWM 73/110 vs 5/110; RCWM PREREG-2/3; L1 "DO NOT PROMOTE". `CURRICULUM.md` lists them as F01–F08.

---

## 4. Training-readiness statements and constraints

From `TRAINING_READINESS.md`:
- "Only independently approved active release bytes may be considered for a **separately authorized future training experiment**."
- "Pending candidates and quarantine files are excluded from student exports."
- "The factory does not establish transfer, retention, larger-world success or intelligence improvement."
- "Train/dev records are exposed; separate custody is needed for a future sealed evaluation."

How to load:
- "Load through `factory.catalog.catalog()`… do not concatenate historical release directories". Concatenating would include superseded RCM v0002 and RVM v0001 and the 6-row erratum.

**Immutability.** "Release directories are immutable. Corrected data requires a new release version and explicit supersession record". The handoff adds "never switch or clean the research checkout".

**Sealed sets.** None exist in the factory. "Create no sealed evaluation under current exposed custody." "Exposed dev challenges are not sealed tests." The blueprint `known_benchmark_policy` says "Frozen Rineya evaluations and existing PlanBench-hard instruments stay historical, never training/development input."

**Reuse constraints** in CURRICULUM:
- RCM: "do not regenerate or retune their frozen banks" (RCWM T2b and joint cmp/calc/after kits).
- RGM: "preserve the completed 110-item hard result as historical evidence, not a tunable development set".
- F06: choose filler policy "from TRAIN only; do not copy a test-derived maximum".
- F07: the factory "does not open or recreate the protected retention instrument".

**Interpretation limits** stated throughout:
- Same-model reviewer correlation.
- Structural template overlap.
- Source families are world instances, not mechanisms ("150 worlds do not equal 150 independently new mechanisms").
- Renderings and twins are correlated.

---

## 5. Python interfaces you could import read-only

Run from FR with `python -B` or `PYTHONDONTWRITEBYTECODE=1` so no `__pycache__` is written. `ROOT` resolves from `factory/common.py`.

**Pure or read-only functions:**
- `factory.catalog.catalog()` returns `(rows, report)`. It is read-only. **Do not** run `python -m factory.catalog`: its `__main__` rewrites `manifests/ACTIVE_RELEASE_CATALOG.json`. The same applies to `-m factory.status`.
- `oracles.solve(task)` and `oracles.OracleError` (`oracles/independent.py`; it delegates to `extended_independent.solve`). Supported kinds: causal, graph, blocks, critic, skills, modules, episodic, belief, predictive_abstraction, constraint_repair, agent_history.
- `oracles.metadata_independent.check(row)`.
- `oracles.memory_controls_independent.{analyze, check, check_batch, audit_probes}`.
- `oracles.belief_controls_independent.{analyze, check, check_batch}`.
- `factory.reference.{solve, causal, graph_run, shortest_graph, blocks_move, shortest_blocks, critic}`: the generator-side reference, kept separate from the oracle.
- `generators.other_tracks.reference.{solve, contracts}`.
- `factory.exports.specialized(rows, split)` and the `PREFERENCE_RULES` text.
- `factory.lineage.{source_key, semantic_fingerprint, cross_release}`.
- `factory.common.{canonical, digest, record_hash, load_jsonl, split_for}`.

**Task builders (pure; these return a task without writing):**
- `generators.skills.generate.make_task(index, split, rng)`
- `generators.memory.generate.{make_modules, make_episodic}`
- `generators.other_tracks.generate.{make_belief, make_predictive, make_repair, make_agents}`
- `generators.relational.generate.{graph_parent, block_parent}` and `generators.relational.expansion.diversify`
- `generators.critic.generate.{world_parent, candidates_for}`

**Contract texts:** `generators.causal.generate.CONTRACT`, `generators.relational.generate.{GRAPH_CONTRACT, BLOCK_CONTRACT}`, `generators.critic.generate.CONTRACT`.

**Avoid (these write files):** every `generate()` function writes to `data/.../candidate_renderings`. Also avoid `factory.pipeline` audit/release, `write_*` and `atomic`.

**Dependencies:** `requirements.txt` is `jsonschema>=4.0,<5`, needed only by `pipeline.schema_validate`. The factory targets Python 3.12; [M-me] catalog loading and oracle re-checks worked under Python 3.11.15 here.

---

## 6. Requests or handoffs addressed to the complementary or model-research sessions

**No message in the factory is explicitly addressed to "the complementary research session" or to a model-research session.** [M; grep for complementary, model session, A/B/C/D, authorized training]

The connections are indirect:
- The bundle in your directory is the factory's **upstream specification**. Its track cards, `tracks.json`, `experiment_registry.json`, `predictions.json`, `COMMON_PROTOCOL.md`, `DATA_CONTRACTS.md`, `REUSE_AND_BRANCH_POLICY.md` and `INTEGRATION_AND_TRANSFER.md` are hash-pinned.
- Each track is linked to registry experiments: RCM E01–E03, RVM E04–E06, RGM E07–E09, RSM E10–E12, RMM E13–E15, RBM E16–E18, REM E19–E21, RJP E22–E24, RDP E25–E27, ROM E28–E30.
- RESEARCH_GAPS: "The 35 registry experiments and 13 bundle predictions are planned, unmeasured source records… this CPU dataset role executes none of those studies."

**Implicit requests to a future model or training session [P]:**
- G05: "future A/B/C/D plus matched SFT controls required for internalization claim".
- G06: "later model acquisition/retention matrix with prospective powered margins".
- G10: "separate authorized training and independent sealed evaluation in the future".
- Blueprint `assistance_claim_policy`: "Later A/B/C/D arms and matched SFT distinguish assistance from internalization; no training is authorized or executed by this role."

**Next-session (factory-internal) priorities [P]** (NEXT_SESSION_HANDOFF and `research/research_gap_analysis/PRODUCTION_MILESTONE.md`):
1. Broaden causal primitives (carry/wrap/modulo) and make nested branches decisive.
2. Add new relational topology and dependency mechanisms.
3. Extend critic faults beyond the fuel/graph scaffold, with sign balance.
4. Memory: future-only absence, relevance and unresolved conflict.
5. RSM/RMM: dependent composition plus protected postconditions (AB vs BA).
6. Belief: information-gathering decisions. Belief-gap v2 next steps are to freeze the candidate and checker, create stripped input-only reviews, and "If a different perfect stripped-input shortcut survives… publish a new public revision".

**Reuse points named for model work [P]** (RESEARCH_GAPS "Existing-work overlap"): RCWM T2b, the joint cmp/calc/after kits, BWM planner/search-distillation interfaces and surface-audit controls. RHF, triadic memory, CSS, RHD and recurrence need "an exact branch/interface audit before reuse".

**[I] Practical takeaways:**
- **What's usable for training or eval now:** the active RCM (1,440), RGM (432) and RVM (384) data is exact-verified and reviewed, and suitable for SFT, preference or process-reward experiments.
- **What isn't held out:** the dev_challenge split is exposed, template-overlapping development data. Any OOD or efficacy claim needs a freshly generated held-out set under separate custody. Holding out mechanism templates (not just seeds) can be done by calling the pure builders and oracles with new indices or vocabularies, without writing into FR.
- **Track sizes:** RSM, RBM, RJP, RDP and ROM have only 12 families and 4 dev rows each, which is too small for meaningful training or eval.