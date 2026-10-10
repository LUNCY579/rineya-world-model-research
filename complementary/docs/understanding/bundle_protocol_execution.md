# Rineya complementary research bundle (RCI-001 v1.0): protocol, execution, templates, evidence and tools

Root: `C:/Users/clclc/BusinessProject/WorldModel/rineya_complementary_research_2026-10-09`

**How claims are labelled.** **[M]** means measured or recorded as a result. **[P]** means a plan, proposal or rule. **[I]** means my own inference.

`09_quality/validation_summary.json` records `"native_experiments_run": 0`, 16 reference unit tests passing, 35 experiments, 10 tracks, 36 sources (5 private), 13 predictions and 16 risks. The only numerical Rineya results in this bundle are summaries the bundle's authors read from the private repository. Nobody re-ran any of them here.

---

## 1. Common protocol (`04_experiments/COMMON_PROTOCOL.md`)

Header: "Status: proposed; existing preregistrations remain authoritative." [P]

### 1.1 Unit of independence and splits
- **Build parents first.** Generate semantic **parent groups** before anything else. A parent is a world/rule instance plus its counterfactual, paraphrase, permutation and trajectory variants. The whole parent goes to **TRAIN, DEV or SEALED before serialization**.
- **Seeds are not enough.** "Changes in random seeds alone do not prevent structural leakage."
- **What to hold out.** Hold out whole templates, rule compositions and renderer variants.
- **Hashes.** Store **canonical semantic hashes** as well as raw-file hashes.
- **Old banks are spent.** The old frozen confirmations are now historical evidence. Do not tune on them, mine their failures into training, or call a re-run on them a fresh confirmation. Build new dev mirrors with independent seeds and templates, and reserve a separate confirmation instrument.
- **Public benchmarks** are external checks with possible pretraining exposure. Private fresh synthetic instruments give a cleaner rule-acquisition claim.
- **Pilot size:** "about **200 independent parents per selected domain**" plus an explicit small set of strata. This is "a planning starting point, not a power guarantee." Multiple renderings of the same 200 parents are not thousands of samples.
- **Confirmation size** is chosen by simulation, using the pilot's discordance and a prespecified effect margin.
- **Seeds:** 1 training seed for engineering. A replicated efficacy claim needs **at least 3 genuinely independent training seeds**.

### 1.2 OOD matrix (each axis reported separately)
| Split | What changes |
|---|---|
| Same-distribution, fresh payload | Nothing (tests interpolation) |
| Surface | Names, field labels, input order, paraphrase |
| Numerical | Magnitude, thresholds |
| Semantic | Predicate meaning, composition |
| Structural | Graph motifs, object counts |
| Temporal | Horizon, observation delay |
| Goal | Reward weights |
| Domain | Environment, or the skill's language-facing application |

- Never aggregate the axes into a single OOD claim without reporting every component.
- Hold out **combined shifts** as well as single-factor shifts, e.g. large objects in unseen renderings.
- Include **copy/no-op and touched-entity floors** so a gain cannot be explained by label imbalance.

### 1.3 Budget matching and leakage controls
- **Training log:** parents, transitions, oracle labels, processed tokens, supervised target tokens, optimizer steps, LR schedules, parameter counts, tuning trials, wall time.
- **Inference log:** generated tokens, all specialist/validator calls, graph extraction, retrieval, search expansions, cache hits, denoising steps, end-to-end latency.
- **Matching:** run a **matched-label** comparison and, where meaningful, a separate **matched-compute** comparison. They are not automatically the same experiment. "Equal token counts do not imply equal FLOPs or equal wall time."
- **Pricing:** canonicalization, graph extraction, exact execution and memory access must be priced and labelled by information tier. Replay or cached-price estimates do not establish latency gains.
- **Filler control:**
  - Use a **constant length per public domain/action kind, chosen from TRAIN only**. Never derive it from an individual gold answer or hidden trace.
  - Match aggregate supervised-token budgets separately, or report the remaining mismatch as a covariate.
  - Keep target ordering stable. Malformed output counts as a failure.

### 1.4 Statistics and decision rules
- **Paired, parent-level comparisons** throughout.
- **Binary outcomes:** **exact McNemar** on discordant counts.
- **Regret or repeated variants:** **clustered paired bootstrap over semantic parents**. Goal weights and renderings are not independent evidence.
- **Reporting seeds:** show individual seed results plus pooled parent summaries. With 3 seeds, broad seed-stability claims remain limited.
- **Pre-confirmation freeze:**
  - one primary endpoint and one primary comparison per experiment;
  - minimum practical effect, retention margins, multiplicity family, budget and stopping rules.
- **Multiplicity:**
  - **Holm adjustment** for a small prespecified family of primary tests;
  - secondary metrics are descriptive unless separately registered;
  - keep an experiment-level record of *all* attempted confirmations ("A fresh bank for every iterative tweak does not eliminate adaptive multiple testing").
- **Outcomes:** **ADVANCE / REVISE / STOP / INCONCLUSIVE**.
  - "A positive point estimate with a wide interval is not a pass."
  - "A nonsignificant loss is not proof of equivalence."
  - No claim may be promoted from synthetic utilities, mocked adapters or unit tests.

### 1.5 Retention and process safety
- **Existing gate:** preserve the registered **D-36 gate** on the frozen retention workflow.
- **New dense-student gate:**
  - define a prospective overall **noninferiority margin**;
  - simulate its power;
  - protect critical categories individually: correctness, instruction compliance, arithmetic, retrieval fidelity, abstention, role consistency.
  - KL and greedy agreement are diagnostics only.
- **Custody and runtime:**
  - record base and adapter hashes before and after every run;
  - require loader parity before training;
  - verify cache identity;
  - keep graders immutable during evaluation.
  - Any change to runtime, tokenizer, precision or export needs an amendment or a separate arm.

---

## 2. Data contracts (`DATA_CONTRACTS.md`, `task_example.json`, `target_example.json`) [P]

These are "proposed exchange records, not replacements for a registered RCWM/BWM schema." Adapters convert into the existing interfaces without mutating frozen datasets. Gold labels live in a separate evaluator/teacher lane. The policy sees only observations, stated rules, the goal and its own permitted history.

### Required lineage
- Immutable semantic `parent_id`.
- Canonical identity/hash.
- Generator version.
- Split assigned **before** surface rendering. Renderer and goal variants inherit the parent's split.
- A **family/group key** for counterfactual twins. Hashes alone miss semantic overlap, so generator-level lineage and held-out mechanism checks are also required.

### Task record (`task_example.json`, flagged `synthetic_example: true`, `not_benchmark_data: true`)
```
parent_id, split ("train"), generator_version,
observation {balance, status}, rule (string), action {kind, amount}, goal (string),
evaluator_only {gold_after, explanation}
```

### Target record (`target_example.json`, flagged `not_model_output: true`)
The order `cmp`, `calc`, `after` is fixed:
```
target: { cmp: {observed_status, required_status, ok},
          calc: {balance},
          after: {balance} }
```
- Exact labels come from a validated interpreter.
- No per-row filler length, filename, order or provenance feature may encode the answer.

### Learned-output capture record (fields)
- experiment, run, independent seed, parent, renderer, inference tier;
- model hashes, generated output, parse validity;
- permitted model calls/tokens, wall time.

The evaluator appends exact scores separately. A next-state gold feature must never be attached to a router, and evaluation correctness must never be fed into a policy.

**[I] Split-label mismatch.** The protocol uses TRAIN/DEV/SEALED, but `audit_splits.py` only accepts lowercase `train|dev|test`. SEALED has to be mapped to `test` before using the tool.

---

## 3. Experiment registry guide and caps (`EXPERIMENT_REGISTRY_GUIDE.md`, `tracks.json`, `experiment_registry.json`) [P]

### Status
All 35 experiments are **Planned**, with `approved: false`, `actual_gpu_hours: null` and `result: null`.
- E01–E30 cover the 10 tracks.
- E31–E35 cover CORE: four-arm transfer, second-domain replication, retention, factorial synergy and final confirmation.

### Registry fields
`id, track, name, priority, status, purpose, primary_endpoint_draft, dependencies, required_controls, development_target, evaluation_unit, sample_plan, seed_plan, compute_policy, source_ids, approved, actual_gpu_hours, result`

### Pilot GPU-hour caps (`pilot_gpu_hour_cap`)
These are shared ceilings for one initial pilot per track. They are not allocations or authorization.

| Track | Name | Priority | Cap (GPU-h) |
|---|---|---|---|
| RCM-001 | Causal and counterfactual rule models | P0 | 4 |
| RVM-001 | Validity critics, value models and successor features | P0 | 3 |
| RGM-001 | Relational graphs and neuro-symbolic models | P0 | 4 |
| RSM-001 | Hierarchical policies and reusable skills | P0 | 4 |
| RMM-001 | Modular experts, continual learning and meta-learning | P1 | 3 |
| RBM-001 | Belief-state models and metacognitive control | P1 | 2 |
| REM-001 | Episodic, associative and procedural memory | P1 | 2 |
| RJP-001 | Latent predictive and energy-based models | P2 | 2 |
| RDP-001 | Diffusion and revisable proposal models | P2 | 2 |
| ROM-001 | Social, opponent and character-belief models | P2 | 2 |

The P0 caps sum to 15 h and all ten sum to 28 h. The guide says 28 h "is not a recommendation to launch them all."

### Recommended order
1. Finish shared controls and existing active registered work.
2. **E01 + E07.**
3. Independently qualify **E04**.
4. **E10 + E31**.

**E33** accompanies every training candidate. E32 and E35 qualify a claim. E34 is only needed to claim a specific interaction. P1/P2 pilots wait for a documented bottleneck and a separate go-ahead.

Development targets such as "+10 points" or "25% fewer expansions" describe a useful pilot. They are not expected gains.

### Key drafts (endpoint → target)
- **E01 Rule semantics factorial**
  - Endpoint: parent-macro exact transition accuracy on held-out semantics and surface forms.
  - Target: +10 pp over the strongest matched non-oracle baseline on the OOD composite.
  - Controls: unmerged T2, matched self-distillation, balanced answer-only training, canonicalized input, fixed-filler trace training.
- **E04 Independent validity critic**
  - Endpoint: false-accept rate and valid-step recall on unseen error producers.
  - Target: +5 pp goal success at matched compute, *or* ≥25% fewer expansions at noninferior success.
  - Controls: random, constant and shuffled-label critics; deterministic validator ceiling.
- **E05 Goal-factorized value transfer**
  - Endpoint: goal-selection regret on held-out reward weights with fixed dynamics.
- **E06 Value-guided bounded search**
  - Endpoint: exact success at matched compute.
- **E07 Relational transition comparison** (GNN vs flat/text/canonicalized)
  - Endpoint: exact state-delta accuracy across object counts and renderings.
  - Controls: canonicalization-only control, randomized edge labels; graph extraction must be charged.
- **E08 Learned search heuristic.**
- **E09 Bounded program induction.**
- **E10 Verified search distillation**
  - Endpoint: unaided student planning success at a fixed decoding budget.
  - Target: +5 points no-search success.
- **E11 Novel option composition.**
- **E12 Student-state aggregation.**
- **E31 A/B/C/D four-arm transfer + matched SFT**
  - Dependencies: E01/E07 teacher eligibility, then qualified E04/E10 outputs.
- **E34 Selected factorial synergy**
  - Endpoint: S_AB − S_A − S_B + S_0 with parent-clustered uncertainty.

---

## 4. Hardware and compute (`05_execution/HARDWARE_AND_COMPUTE.md`) [P]

### Envelope
- **RTX 4070 Laptop, 8 GB.** Refresh RAM, free VRAM, disk and thermal conditions before execution. Historical free-memory numbers are not a capacity check.
- 0.6B FP32 weights alone are about **2.4 GB decimal (~2.24 GiB)**. That is a lower bound; activations, gradients, optimizer state, caches and overhead come on top.

### Precision and residency
- Preserve the **registered FP32 teacher lane**. BF16, quantization, merging and library upgrades are separate, parity-qualified experiments.
- Avoid holding several full backbones in memory at once. Prefer one frozen backbone with sequential adapter calls.

### Calibration before full runs
- Profile **about 20 steps** plus a representative search batch.
- Measure throughput, activation peak, search branching, cache growth and CPU/RAM load.
- Then extrapolate, and decide whether to simplify, checkpoint or request a larger cap.

### Device and lease rules
- **Default to CPU** for generators, interpreters, schema checks, split audits, classical baselines and small state models.
- **Single authorized GPU lease** plus the **existing thermal watchdog**.
- **Fail closed** if thresholds, lease ownership, memory or checkpoint custody cannot be verified.
- Do not disable the watchdog, invent a safe temperature, or launch an unattended competing process. "No Codex tasks or new schedules."

### Portfolio shares after calibration
These are planning shares, not approved time.

| Area | Share |
|---|---|
| Rule/relational pilots | 35% |
| Critics/skill distillation | 20% |
| Replication and retention | 25% |
| Routing/memory/belief baselines | 15% |
| Optional latent/diffusion/social exploration | 5% |

An optional track stops early if its prerequisite or simple-baseline comparison fails.

---

## 5. Integration and dense transfer (`INTEGRATION_AND_TRANSFER.md`) [P]

### Four mandatory ability tiers
| Arm | Definition |
|---|---|
| **A** | Unchanged frozen core, no assistance |
| **B** | A + specialist/tool interface |
| **C** | Freshly trained student, no assistance |
| **D** | C + the identical assistance used by B |

- **Contrasts:**
  - **B−A** = system assistance;
  - **C−A** = training transfer;
  - **D−B** = whether student training helps an assisted system;
  - **D−C** = remaining benefit of assistance.
- **Matched-SFT control** gets comparable accepted examples, label access and token budget. Without it, a C−A gain may just be more task training.
- **Pinned across arms:** tokenizers, prompts, decoding, adapter state, allowed observations.
- **No-search claims:** A and C must run without teacher search or hidden retrieved solutions.

### Artifact types
Report these three separately:
- frozen base + adapter;
- modular multi-adapter inference;
- genuinely consolidated dense export.

**Registered T2 stays unmerged in FP32.** A new student may be consolidated only in an independent, approved lane with functional parity and retention tests. "No universal ban on all future merges is implied."

### Teacher eligibility
- A teacher must pass the exact domain/rule/surface range it will label. "T2 is not an unrestricted ledger oracle."
- For new statuses, ranges or formats, use the exact generator/interpreter or another qualified teacher.
- Teacher uncertainty is never a hard positive label.
- Log rejected examples and the reasons.
- Separate proposal from verification. Plans accepted by a learned forward model must still pass the **exact independent environment** before entering the gold stream.
- Keep a small blinded manual audit of the parser and checker.

### Training sequence
1. Competent supervised student on simple verified targets.
2. Predicate/trace targets vs answer-only vs fixed-filler controls.
3. DAgger-style exact corrections on TRAIN trajectories only.
4. Preference optimization only where correctness is grounded, keeping an SFT-only comparator.
5. Shorter traces, bounded recurrence or direct answers, each as a distinct inference tier.

### Dense-transfer bridge tasks
Compositional instructions, prerequisite constraints, evidence-grounded updates, simple algorithm execution, fictional role consistency. Style, correctness, factuality and instruction compliance are scored separately; no weighted total may hide a catastrophic category loss.

---

## 6. Branch and reuse policy (`REUSE_AND_BRANCH_POLICY.md`) [P]

**Verified starting points:**
- RCWM T2b and its retention designs;
- BWM search/planner work;
- the **IDR-001** branch README, which is CPU-tested and *native-unqualified*. Reuse its role separation, competence masks and audit interfaces, but "do not count it as a successful native router."

**Overlap candidates** (not verified as executable together): RHF-001, factorized/triadic memory, CSS, RHD, layered post-training, inference-compute/recurrence branches. The bundle is "not a complete remote-branch census."

**Rules:**
- Before any new branch:
  - inspect refs and files;
  - record exact commits;
  - either extend the existing track or document the missing experiment.
- RCI track IDs are planning labels, **not created Git branches**.
- The private world-model records are evidence, not a workspace for rewriting history.
- New experiment folders go only in a separately authorized branch.
- Checkpoints are read-only.
- Avoid unrelated app/game directories.
- Preserve lease coordination with other manually authorized sessions.

---

## 7. Launch checklist (`LAUNCH_CHECKLIST.md`) [P]

Every item must be **evidenced**, not assumed:
1. The experiment is not a duplicate of an active or archived branch. Custody of source commit, checkpoints, tokenizer and data is known.
2. The teacher is qualified for every label-producing stratum, or an independently tested exact interpreter supplies the labels. T2 stays unmerged in FP32.
3. These all pass:
   - parent-group splits;
   - canonical/semantic overlap checks;
   - test custody;
   - trace order;
   - fixed filler policy.

   No gold or evaluator-only state reaches the deployed model.
4. Simple controls and training/inference budgets are fixed. Parent-level power and noninferiority design match the claim. Existing preregistrations are not silently rewritten.
5. CPU tests pass. On the actual device, verify:
   - qualified loader parity;
   - actual memory measurement;
   - single GPU lease;
   - thermal watchdog;
   - safe restart rules.
6. Native execution is **explicitly approved**. No paid service, publication, promotion or scheduler is inferred from approval of the plan.

"A failed checklist item is a blocker." CPU planning may continue, but it must not be presented as native efficacy.

---

## 8. 90-day roadmap (relative, gated; `90_DAY_ROADMAP.md`) [P]

| Days | Phase | Work |
|---|---|---|
| 1–7 | Evidence and baselines | Read pinned records. Qualify loader and independent simulator. Finalize manifest and split rules. Small parity sample. Exact/simple baselines. Deliver signed-off pilot protocol and measured compute estimates. Do not rerun the full historical sealed eval. |
| 8–21 | First pilots | RCM surface×semantics factorial. RGM text/flat/graph/canonicalization comparison. Small CPU critic. A failed rule-generalization gate blocks unrestricted teacher reuse. Pick one mechanism per domain. |
| 22–42 | Replication | Extra seeds. RVM ranking/validity at matched cost. First RCM×RGM interaction. Teacher inventory, error map, retention report. "A simple symbolic/graph hybrid may be the best practical outcome." |
| 43–63 | Direct-skill transfer | Exact-verified TRAIN trajectories. RSM next-action distillation plus matched SFT. A/B/C/D comparison. Optional DAgger arm. New language-facing bridge domain ("do not mistake held-out numeric values for a new domain"). |
| 64–77 | Cumulative capabilities | IDR routing over successful specialists. Sequential retention matrix. At most one memory/belief experiment and at most one optional latent/diffusion/social pilot. |
| 78–90 | Confirmation | Freeze primaries. Fresh sealed parents. Powered retention. Private report of all arms. Separate promotion review. No public upload. |

---

## 9. Templates (`06_templates/`) [P]

### `PREREGISTRATION.md`
Status DRAFT; "Explicit execution approval: NOT GRANTED."

| Section | Content |
|---|---|
| Claim and scope | One primary hypothesis; task family; permitted observations; model fingerprint; inference tier; practical effect; what a positive result would *not* establish |
| Custody and overlap | Repo commits, file SHA-256, model/tokenizer/adapter hashes, generator version, parser; list of historical/spent/reserved/sealed banks; parent grouping before rendering; canonical and byte-hash checks |
| Arms and fairness | Each control/intervention; information available; teacher access; train labels; processed/supervised tokens; HP search effort; inference budget; a **non-LM/deterministic baseline**; A/B/C/D tiers; matched SFT |
| Endpoints and sampling | Primary endpoint; direction; independent unit; secondaries and protected strata; error/timeout/invalid-parse scoring; parent clusters; renders/goals per parent; train seeds; blinded power/operating-characteristic simulation |
| Decision rule | GO / NO-GO / INCONCLUSIVE criteria; paired analysis; multiplicity; practical threshold; retention margins; resource limits |
| Execution and safety | Cap/stop policy; restart and partial-output rules |
| Freeze and deviations | Freeze hash, timestamp, reviewers; create-only confirmation outputs; deviation log stating whether outcomes were known; no rerun under the same ID |

### `TEACHER_QUALIFICATION.md`
- **Header and scope:**
  - teacher ID and hashes;
  - role: proposer / dynamics / critic / value / interpreter;
  - default status UNQUALIFIED;
  - supported domain, rule family, names, input schema/order, numeric range, horizon, observation boundary, adapter state.
- **Evidence:**
  - sealed qualification cases;
  - independent exact validator;
  - parser self-tests;
  - per-stratum results including rejected labels.
- **Error taxonomy:** false application, false copying, impossible state, arithmetic error, format error, missing observation.
- **Acceptance:** a fail-closed eligibility predicate defined before generation.
- **Requalification** is required for new data, rendering, precision, loader or merge state.

### `RESULTS_REPORT.md`
- **Claim status:** GO / NO-GO / INCONCLUSIVE / INVALID.
- **What actually ran:** hardware, loader parity, seeds, completed parents, missing outputs, actual GPU wall time, cost, calls, tokens, tier.
- **Primary result:**
  - numerators/denominators;
  - paired effect and uncertainty;
  - corrected test;
  - **B−A** and **C−A** plus matched SFT.
- **Protection and generalization:** per-category correctness under both graders; retention gate; each OOD axis; actual latency/memory.
- **Attribution and limits:** which ablation removes the gain.
- **Decision:** next step, protected artifacts, hashes, deviations.

### `MODEL_CARD.md`
"NOT A RELEASE … unqualified candidate."
- **Artifact:** architecture, parameter counts, base/adapter fingerprints, precision, tokenizer, loader, trained parts. Artifact type is one of adapter / external specialist / modular / consolidated dense.
- **Training data:** origin, license, generator, filters, counts, provenance, contamination audits.
- **Supported and unsupported use:** domains, formats, ranges, horizons.
- **Deployment:** tools, budget, residency, measured resources, fallback.
- **Licenses:** listed separately for weights, code, data and assets.

### `DECISION_LOG.csv` (header only)
`decision_id,timestamp_timezone,experiment_id,prior_status,decision,evidence_path,outcomes_known,reason,author,approver`

### `RUN_MANIFEST.template.json`
`schema_version:"RCI-001/planning-v1"`, `status:"DRAFT_UNAUTHORIZED"`, then:
- `experiment_id, repo_commit`
- `model_sha256, tokenizer_sha256, adapter_sha256, registered_precision`
- `teacher_qualification, split_manifest_sha256, preregistration_sha256`
- `independent_seed, output_directory, gpu_lease, thermal_watchdog`
- `authorized:false, paid_services_authorized:false, publication_authorized:false`
- notes: the runner "must fail closed when required fields or permissions are absent."

### `AGENT_HANDOFF.md` (six steps, one experiment only)
1. Read the evidence read-only and confirm the experiment is not already implemented or running.
2. Restate the question and the strongest simple control. Record commits, custody, hardware and blockers. Treat historical banks as spent.
3. Write the preregistration. Use parent-first holdouts, a fixed filler per family/action kind, and `cmp, calc, after` order.
4. Implement only within approved scope. CPU schema/parser/scorer/split tests first.
5. Get explicit native authorization. Honor the lease and watchdog. Stop on parity, integrity, retention or safety failure. Never auto-merge, publish, delete or widen a teacher's domain.
6. Run sealed confirmation once. Report A/B/C/D. Issue GO / NO-GO / INCONCLUSIVE.

Deliverables: code and tests, source manifest, prereg, per-parent outputs, paired analysis, compute record, report, handoff. "Never fabricate a successful native run."

---

## 10. Evidence about native Rineya results (`07_evidence/`)

### Pinned source
World-model repo commit **`a223884d55ff47baf3890c7d4a82054d07223215`** (9 Oct 2026), read through the GitHub connector. Nothing was re-run.

The core is a native frozen **0.6B Titan-Sparse** model with specialized adapters. The open question is "broad transfer to general-purpose, retained, unaided dense behavior."

### Ledger entries
All are [M] as recorded in the repo; none were reproduced here.

| ID | Experiment | Status | Finding | Scope | Limitation |
|---|---|---|---|---|---|
| EV01 | RCWM PREREG-1 | Confirmed | Normalized regret **0.448** on pooled ledger/register 4-step plan selection | 300 parents; 256 goals/eval; horizon 4 | Not a latent-readout superiority claim; some comparators at/below an answer-free floor |
| EV02 | RCWM PREREG-2 | Confirmed | Ledger regret **0.544 → 0.056**; exact 4-step state **0.029 → 0.698** | 150 fresh ledger parents; fixed T2 seed-0 adapter; within-generator bank | Specialized gain; not OOD rule mastery or no-forgetting |
| EV03 | RCWM PREREG-3 | Confirmed failure boundary | Unseen no-op statuses **0.01–0.18** vs trained "void" **0.99–1.00** | 3 T2 seeds; 1,500 one-step audit rows; 80 chains | Gain is "trigger-bound" |
| EV04 | BWM hard planning | Frozen | **73/110** with world model + beam vs **5/110** without | PlanBench Blocksworld-hard; 1 seed; trained ≤6 blocks | Hybrid-system gain only |
| EV05 | BWM surface audit | Exploratory | Training rendering **1.000** vs reversed JSON keys **0.458** | 120 keys at 5 and 10 blocks (RCWM joint decision R16) | Size transfer ≠ serialization invariance; canonical frozen result intact |
| EV06 | L8-15 localization | Promotion failed | Prerequisite acquisition **96/128 → 88/128**; DO NOT PROMOTE | PREREG-L1, gate C2 | Less drift ≠ preservation |
| EV07 | Fixed filler control | Design correction | Per-row trace-matched filler length **leaked answers** | D-36 amendment 4b: fixed per family/domain and action kind | Augment input order, keep `cmp/calc/after` |
| EV08 | Retention workflow | Frozen design, not a pass | No-go on Holm-McNemar category loss, **category net < −11**, or **pooled net < −4%** of items, under **both graders** | D-36; old **−5/75** rule diagnostic only | Non-detection ≠ ≤2-pt loss |
| EV09 | IDR routing | Implementation only | CPU-tested routing and tiny optimizer paths | ChatBubble branch | Native efficacy unqualified |

### Repository sources (`repository_sources.json`)
- **R01** RCWM-T1 `REPORT.md` (blob `4074833f…`).
- **R02** BWM-001 `README.md` (`36337e54…`).
- **R03** RCWM `DECISIONS.md` D-35/R16/D-36/D-37 (`44c4738d…`).
- **R04** `bwm/docs/research/JOINT_ROADMAP_2026-10-08.md` (`0d00e7d1…`).
- **R05** ChatBubble branch `research/rineya-id-balancing-skill-routing-v1`, file `experiments/rineya_id_balancing_skill_routing_v1/README.md` (blob `c6e81f91…`, commit null).

### Evidence taxonomy
Record-confirmed / Exploratory / Implementation-only / Literature-supported / Proposed. The notes say "these labels are not interchangeable."

---

## 11. Reusable reference tools (`08_reference_tools/`, stdlib only, Python ≥3.10)

There is no network, GPU or scheduling in any of them. Test output in `09_quality/unit_tests.txt` [M]: 16 tests, all OK.

### `audit_splits.py`
- **API:** `audit_records(records: list[dict[str, Any]]) -> dict`
- **Required fields:** `parent_id`, `split ∈ {'train','dev','test'}`, `canonical_input_sha256` (64-hex).
- **Optional field:** `family_group_id`.
- **Checks:** flags any `parent_id`, `family_group_id` or hash (lowercased) that appears in more than one split.
- **Returns:** `{schema:'RCI-001/split-audit-v1', passed, records, split_counts, collisions[{field,value,first_split,other_split}], limitation}`.
- **Errors:** raises `ValueError` on empty or invalid input.
- **CLI:** `python audit_splits.py <file.jsonl>`. Exit codes: 0 pass, 1 collisions, 2 error.
- **Limitation:** it trusts the supplied hashes and does not detect semantic duplicates.

### `paired_metrics.py`
- `mcnemar_exact(baseline_only: int, candidate_only: int) -> float`
  - two-sided exact binomial, no continuity correction;
  - returns 1.0 when n=0 or the counts are balanced.
- `quantile(values: list[float], q: float) -> float` — linear interpolation.
- `analyze(rows, resamples=5000, seed=17) -> dict`
  - **Inputs:** rows need `parent_id`, `baseline`, `candidate`, optional `variant_id`.
  - **Validation:**
    - `resamples` must be in [100, 100000];
    - at least 2 parents;
    - parent/variant pairs must be unique;
    - scores must be finite.
  - **Method:** averages variants within each parent, weights parents equally, then runs a percentile bootstrap over parents.
  - **McNemar:** computed only if every parent has exactly one binary row.
  - **Returns:** `schema:'RCI-001/paired-reference-v1', parents, rows, baseline_parent_macro, candidate_parent_macro, candidate_minus_baseline, percentile_interval_95, resamples, bootstrap_seed, baseline_only, candidate_only, mcnemar_two_sided_exact, interpretation, limitations`.
- **CLI:** `python paired_metrics.py <pairs.csv> [--resamples N] [--seed S]`.
- **Not handled:** seed hierarchy, Holm correction, power, noninferiority.
- **Sign:** follows the score, so a negative change is good for regret.
- **Synthetic example** [M, synthetic only], 6 parents: diff +0.1667, 95% CI [−0.333, 0.667], discordant 1 vs 2, p = 1.0.

### `validate_bundle.py`
- **API:** `validate(root: Path) -> dict`
- **Checks:** every `MANIFEST.json` `files[{path,bytes,sha256}]` entry. It rejects absolute paths, `..` and repeated paths, flags unlisted files, and honors `excluded_self_referential_files`.
- **CLI:** `python validate_bundle.py [root]`. Default root is the bundle.

### `test_reference_tools.py`
`SplitTests` (6), `StatsTests` (8), `ManifestTests` (2).

**[I] Gaps you would need to fill:**
- Holm adjustment across a family of tests.
- Seed-level hierarchical bootstrap, e.g. resampling seeds then parents.
- Noninferiority interval checks against a margin.
- Power simulation from pilot discordance.

---

## 12. Literature relevant to the four focus areas (`03_resources/ANNOTATED_RESOURCES.md`)

Sources were "verified on 9 October 2026". No results were reproduced, and licenses are not cleared.

### Causal / counterfactual (RCM-001, partly RJP)
- **P01** LLMs Lean on Priors, Not PL Semantics (arXiv 2510.03415; code `EngineeringSoftware/PLSemanticsBench`). Counterfactual semantics expose reliance on familiar operators. Use rule inversions and held-out operator definitions, not just renamed tokens. Full text checked (v3).
- **P02** Towards Causal Representation Learning (2102.11107). Separate rule-conditioned prediction from causal discovery.
- **P17** Causal-JEPA (2602.11389). Object-level latent masking. Ablate object vs random masking vs supervised transitions. Masking is not an identifiable intervention.
- **P33** NAR with Causal Regularisation (PMLR v202, Bevilacqua 2023). Bridges counterfactual training and graph algorithm traces.
- **P21** IFEval (2311.07911). Used for the instruction-rule bridge (E03).

### Graph / neuro-symbolic (RGM-001)
- **P05** Learning to Search and Searching to Learn (PMLR v306, 2026; `maichmueller/generalized-search-for-planning`). Relational GNN Q-values with weighted A* search. This is a heuristic, not a learned world model. Compare heuristics, learned transitions and exact search separately. Methods checked.
- **P06** Neural Algorithmic Reasoning (2105.02761). Algorithm traces, size extrapolation, exact transition checks.
- **P07** Lifting Traces to Logic (2605.01293; `sh-jj/Lifting-Traces-to-Logic-NSI`). Induces programs with preconditions, arguments, control flow and termination. Program execution counts as tool assistance. Full text checked.
- **P23** PlanBench (2206.10498). Make fresh dev mirrors; the Blocksworld-hard set is already spent.
- **P24** CLRS (2205.15659; `google-deepmind/clrs`). Small subset with size separation. Hints count against the label budget.
- **P30** DreamCoder (2006.08381). Compact operator libraries as skill teachers. Bound grammar, depth and evaluations.

### Value / PRM / successor features (RVM-001)
- **P03** GenPRM (2504.00891, AAAI 2026). Generated plus code-based verification. Start with cheap legality heads.
- **P04** Successor Features (1606.05312). Test unseen reward weights with fixed dynamics. The guarantees do not cover new dynamics or nonlinear rewards.
- **P22** ProcessBench (2412.06559). First-error-step detection, as a hard external critic test. Expect floor effects at 0.6B.
- **P26** On Calibration (Guo 2017). Fit temperature on a separate dev set; report Brier score and selective risk.
- **P28** DreamerV3 (2301.04104). Borrow modular interfaces only.

### Hierarchical skills (RSM-001)
- **P34** Sutton–Precup–Singh options (1999). Option = initiation + policy + termination. It does not discover options.
- **P08** Expert Iteration (1705.08439). Distill verified search into the proposer and test without search. Include teacher search cost.
- **P09** DAgger (Ross 2011). Relabel student-visited **TRAIN** states only.
- Also listed for this track: P05, P07, P30, P23.

### Continual/modular, supporting cumulative skills
- **P11** PASs-MoE (2601.13020). Compare against IDR.
- **P25** EWC (1612.00796). Replay and regularization baselines.
- **P31** LoRA (2106.09685).
- **P10** MAML (1703.03400).
- **P36** DPO (2305.18290). Optional, and only after verified acquisition.

### Reading order (`READING_ORDER.md`)
1. R01 §3b/3c, R03 D-35/D-36, P01, P33.
2. P05, P06, P23, P24 (gold-state vs parsed-state vs language-only arms; dynamics vs search-ranking).
3. P03, P22, P04, P08, P09.
4. R05, P11, P25, P10, P31.
5. Only then: memory (P12–P15, P26, P27), latent (P16, P17, P29, P32), proposals (P18, P19), role (P20, P35, P21), and P36.

For each paper, capture:
- the input/output contract;
- train/test information;
- compute regime;
- strongest control;
- the biggest non-transferable assumption;
- the smallest decision-changing experiment.

### Related risks (`risk_register.csv`)
- K01 shortcut masquerading as semantics.
- K02 teacher and critic share wrong rules.
- K03 leakage.
- K04 gain is just extra compute.
- K05 forgetting hidden by averages.
- K06 laptop instability.
- K07 GNN helped by a privileged parser.
- K08 critic rejects valid candidates.
- K15 scope explosion ("Only P0 core first").

Remaining risks K09–K14 and K16 cover routing, meta-learning, memory, latent collapse, diffusion, social inference and licensing.