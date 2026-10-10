# RCI-001 bundle: master report, RCM-001 and RGM-001

**Scope.** Read-only. Bundle root: `C:/Users/clclc/BusinessProject/WorldModel/rineya_complementary_research_2026-10-09`. I read in full:

- `01_report/Rineya_Complementary_Research_Report.md`
- `02_research_tracks/RCM-001.md`
- `02_research_tracks/RGM-001.md`
- `04_experiments/experiment_registry.json`: E01–E03, E07–E09 and E31–E35, plus the dependency column for all 35 rows
- `04_experiments/predictions.json`: all 13 entries
- `04_experiments/tracks.json`: the RCM-001 and RGM-001 entries, and caps for all ten tracks

I also checked these for context:

- `07_evidence/evidence_ledger.json` (the only file that uses the "PREREG-3" label)
- `07_evidence/repository_sources.json`
- `04_experiments/COMMON_PROTOCOL.md`, `DATA_CONTRACTS.md`, `EXPERIMENT_REGISTRY_GUIDE.md`, `risk_register.json`, `task_example.json`, `target_example.json`
- `05_execution/REUSE_AND_BRANCH_POLICY.md`, `LAUNCH_CHECKLIST.md`, `INTEGRATION_AND_TRANSFER.md`, `90_DAY_ROADMAP.md`
- `06_templates/TEACHER_QUALIFICATION.md`, `PREREGISTRATION.md`, `AGENT_HANDOFF.md`
- `README.md`

**Evidence status.**
- **(a) Measured:** nothing in this bundle. The registry guide says all 35 experiments are "Planned" with `approved: false`, `actual_gpu_hours: null` and `result: null`. Every prediction has `"observation": "Not measured in this bundle"`, except H35: "No quantitative productivity effect measured".
- **Measured elsewhere:** the only measured numbers are repository-recorded results from the pinned world-model commit `a223884d55ff47baf3890c7d4a82054d07223215` (R01–R05). The bundle says it did not rerun them.
- **(b) Plans:** everything else is a plan or proposal.
- **(c) Inference:** my own inferences are marked **[Inference]**.

---

## 1. Overall thesis and the A/B/C/D assistance tiers

**Program.** "Program label: RCI-001. Scope: Rineya 0.6B, complementary specialists, and verified transfer." It is "a research plan and resource kit, not a model release or an implementation of ten trained specialists."

**Central bet (verbatim).** "a compact language model can learn more effectively when different computational jobs are taught and evaluated separately: interpret a rule, predict a transition, assess validity, estimate value, select a skill, maintain a belief, retrieve evidence, and express an answer. The goal is not a permanent collection of ten expensive models. It is a controlled route from useful external assistance to reusable skills and, where possible, better unaided dense-model behavior."

**Executive decision (verbatim).** "**Start with causal-rule repair and relational models. Then add independently qualified critics and distill verified planning.** Develop memory, routing and belief-state work against proven bottlenecks. Keep latent, diffusion and social-model experiments bounded until their prerequisites pass. This is a ten-track portfolio, not a recommendation to launch ten GPU jobs at once."

**Expected outcomes.** "The strongest realistic near-term outcome is a more reliable hybrid system with reduced surface shortcuts and some cheaper direct decisions. The ambitious outcome is retained, cross-domain improvement inside a fresh Rineya 0.6B student. Neither outcome is assumed."

**Framing of the unestablished claim.** Existing programs already use "a native frozen 0.6B Titan-Sparse core with specialized adapters and procedures… The unestablished claim is broad transfer to general-purpose, retained, unaided dense behavior."

**Component contracts (§2).**
- **Forward model:** "predict what follows from state, action and rule, independently of which outcome is desirable."
- **Value:** "estimate usefulness for a specified goal."
- **Critic:** "assess whether a candidate transition or step is valid."
- "A goal must not contaminate a goal-blind dynamics model… A router must never mix fundamentally different operations in one expert pool."

**Priorities.**
- P0: RCM, RGM, RVM, RSM. P1: RMM, RBM, REM. P2: RJP, RDP, ROM.
- First dependency path: "shared evidence controls -> RCM rule repair and RGM representation comparison -> qualified RVM evaluation -> RSM search distillation -> four-arm dense-transfer tests."
- "In the first cycle, prefer an exact rule interpreter, a flat MLP, a small GNN, logistic critics and fixed routing before adding complexity. A simple module that wins is an asset, not a disappointment."

**A/B/C/D tiers (§5 "Four mandatory ability tiers", verbatim; repeated in `05_execution/INTEGRATION_AND_TRANSFER.md`).**

> Arm A is the unchanged frozen core without auxiliary assistance. Arm B is the same core with the specified specialist/tool interface. Arm C is a freshly trained student without that assistance. Arm D is that student with the identical assistance used by B. Pin tokenizers, prompts, decoding, adapter state and allowed observations. For a no-search claim, A and C must truly run without teacher search or hidden retrieved solutions.
>
> B-A measures system assistance. C-A measures training transfer into the evaluated student. D-B tests whether student training helps an assisted system. D-C measures the remaining benefit of assistance. A separate matched-SFT control must receive comparable accepted examples, label access and token budget; otherwise any C-A gain might simply be more task training.

Related rules from the same section:
- "Do not call a student adapter a new dense checkpoint merely because it is loaded on a dense backbone. Report frozen-base + adapter, modular multi-adapter inference, and genuinely consolidated dense export separately."
- "Registered T2 remains unmerged in FP32."
- Teacher eligibility: "T2 is not an unrestricted ledger oracle. For new statuses, new numerical ranges or unsupported formats, use the exact generator/interpreter or another independently qualified teacher."

The tiers are put into practice as **E31 "Four-arm assistance and student transfer"** (CORE, P0):
- Primary endpoint draft: "Exact held-out task success for A/B/C/D and matched-SFT control."
- Dependency: "E01/E07 teacher eligibility, followed by qualified E04/E10 outputs."

**Synergy definition (§4).** "I = S_AB - S_A - S_B + S_0 on the same parent instances and equal resource budget." This is put into practice as E34. "The first three proposed interactions are RCM x RGM, RVM x RSM, and REM x RBM."

---

## 2. RCM-001: causal and counterfactual rule models

From the track card, the report and `tracks.json`. Status: "Planned; not executed." Priority P0. `pilot_gpu_hour_cap: 4`, "not a runtime estimate".

**Purpose.** "Learn which mechanism governs a transition, not which familiar token tends to precede an answer."

**Question.** "Can supplied rule semantics control behavior when vocabulary, arithmetic range and predicate structure change?"

**Mechanism.**
- Factor the transition into applicability `C(rule,state,action)`, a calculation `A(state,action)` and a state assembler.
- Use counterfactual pairs in which "only the rule or one precondition changes."
- "Initially treat the rule as observed; do not claim to discover hidden causal structure."

**Design and environments.**
- "Start with ledger statuses and register predicates." These are the RCWM ledger/register domains.
- "Generate TRAIN/DEV/SEALED groups before rendering."
- TRAIN = "balanced apply/no-op pairs, opposite rule meanings, nonce status words, and input permutations."
- "Preserve literal target order cmp, calc, after."
- Pilot is a 2x2 factorial: **(answer-only vs predicate trace) × (ordinary vs counterfactual/surface-diversified data).**
- "Then hold out entire predicate compositions, thresholds and renderers."
- Ceiling: "A rule parser plus exact interpreter supplies a transparent ceiling."

**Data contract** (`DATA_CONTRACTS.md` and the synthetic examples).
- Input = observation, rule and action. Supervised target = ordered object `cmp`, `calc`, `after`, with labels from "a validated interpreter."
- The synthetic example is illustrative only:
  - observation `{balance: 7, status: "pending"}`
  - rule "Apply the amount only when status equals posted."
  - action `{kind: add, amount: 3}`
  - target `cmp.ok=false`, `after.balance=7`

**Baselines and controls (identical text in all three experiment rows E01–E03).** "Compare the existing unmerged T2 behavior, matched self-distillation, balanced answer-only training, canonicalized input, and trace training with fixed filler length per family/action kind. Match oracle label access and report both processed and supervised tokens. Test cases with the same answer but different predicates as well as opposite answers."

**Metrics.**
- Primary: "parent-macro exact transition accuracy on a predeclared held-out semantic/surface composite."
- Secondary: "predicate accuracy, false-copy rate, false-apply rate, exact K4/K8/K16 rollout and worst stratum. Classify format errors separately."

**Numeric gate (identical in all three rows, and in H01).** "+10 percentage points over the strongest matched non-oracle baseline on the OOD composite, positive corrected paired interval, and no protected-stratum regression beyond its preregistered margin. Confirmation must use unseen parent groups and renderers."

**Failure interpretation.**
- "If gains vanish with input canonicalization, credit the interface repair rather than causal learning."
- "If only known predicates improve, narrow the claim."
- "If traces lose to balanced answer-only data, retain the simpler recipe and investigate predicate labels rather than adding model complexity."

**Transfer path.**
- "Distill verified applicability decisions and transition traces into a fresh 0.6B student adapter; subsequently reduce trace length and test without the specialist."
- "Bridge to natural-language constraints with conflicting or inverted rules. Do not merge the registered FP32 T2 adapter."

**Reading anchors.** P01, P02, P33, P21.

**Risk.** K01 "Shortcut masquerading as semantics" (High, owner "RCM/E01"):
- Mitigation: "Fresh surface, inverted-rule and computed-predicate holdouts."
- Response: "Do not broaden the rule-learning claim."

### RCM-001 experiments

All three rows share these fields:
- `evaluation_unit`: "Independent parent/task cluster; renderer/goal variants remain within cluster"
- `sample_plan`: "Use train/dev pilots to simulate paired power; freeze confirmation n before opening outcomes"
- `seed_plan`: "Pilot seed only; >=3 independent training seeds for replication claim"
- `compute_policy`: "Track cap is a proposed shared pilot ceiling, not a per-experiment allocation or execution permission"
- `approved: false`

| ID | Name | Purpose | Primary endpoint (draft) | Dependency |
|---|---|---|---|---|
| **E01** | Rule semantics factorial | "Surface x trace pilot on held-out renderers" | "Parent-macro exact transition accuracy on held-out semantics and surface forms" | "G0: audit / teacher qualification / preregistration" |
| **E02** | Computed-predicate extrapolation | "New thresholds, composed predicates and arithmetic ranges" | "Exact transitions on unseen predicate compositions and numerical ranges" | "E01 passes or documented independent rationale" |
| **E03** | Instruction-rule bridge | "Transfer predicate handling to unseen language constraints" | "Constraint compliance on held-out instruction templates; task correctness separately" | "E02 passes or documented independent rationale" |

**OOD splits for RCM.**
- E01: held-out renderers/surface plus semantics, nonce status words and input permutations.
- E02: numerical split (thresholds, arithmetic ranges) and semantic split (composed predicates).
- E03: domain/language-facing split (held-out instruction templates). P21 (IFEval) is the anchor: "Pair structured-rule gains with a fresh, rule-composition instruction benchmark."

**Ablations.** The 2x2 factorial, the canonicalized-input control and the fixed-filler trace control.

**[Inference]** "G0" is never defined in any file I read. It appears to mean the shared gate: audit, teacher qualification and preregistration, consistent with `LAUNCH_CHECKLIST.md` and `TEACHER_QUALIFICATION.md`. The registry also leaves the 2x2 arms, n, and the protected-stratum margins to a future preregistration ("Freeze the exact arms, parent-level partitions, label/token accounting and stopping rules before training").

---

## 3. RGM-001: relational graphs and neuro-symbolic models

Status "Planned; not executed." P0. `pilot_gpu_hour_cap: 4`.

**Purpose.** "Represent objects and relationships directly, and compare learned computation with explicit rule execution."

**Question.** "Is the current generalization limit caused by serialization, relational representation, arithmetic execution, or search guidance?"

**Mechanism.**
- "permutation-equivariant object embeddings and invariant aggregation; represent relations with typed arguments."
- "Keep two experiments separate: a learned graph transition predictor and a graph heuristic used with exact dynamics. A third bounded program-induction experiment learns small conditional procedures."

**Design and environments.**
- "Begin on generated Blocksworld and prerequisite graphs."
- Arms: "text, canonical text, an equal-label flat MLP, and a small message-passing GNN."
- "Train on small object counts; test larger counts and unseen object-name/rendering combinations. Hold out graph motifs as well as size."
- Heuristic: "use weighted A-star or beam search over the exact simulator and compare to zero/simple handcrafted heuristics. Transfer only after matching the same action enumeration interface."
- Reuse: BWM domains, planner interfaces and surface audit.

**Baselines and controls (identical text in E07–E09).** "Exact parser/simulator ceiling; canonicalization-only control; nonrelational capacity-matched predictor; randomized edge labels; permutation tests; fixed heuristic with identical search budget. Charge graph extraction and candidate enumeration. Distinguish invalid states from legal actions with no state change."

**Metrics.** "Exact transition accuracy, relation precision/recall, legal-action recall, goal success at equal expansions, end-to-end latency, and generalization by object count/motif. Compare raw text and gold-graph inputs as different information tiers."

**Numeric gate (identical in E07–E09 and H07).** "substantially close the held-out serialization gap and improve OOD planning by +10 points or reduce expansions by 25% at noninferior success. The learned approach must justify itself against canonicalization and exact algorithms."
- **[Inference]** "Substantially close" is not quantified anywhere.

**Failure interpretation.**
- "If the exact symbolic model wins cheaply, use it as a teacher or tool and retain the neural result as negative evidence."
- "If a GNN works only with gold graphs, the language-to-graph interface remains the bottleneck."
- "Do not attribute privileged symbolic input to the dense core."

**Transfer path.**
- "Distill typed relation extraction, applicability and next-action choices. Test gold-graph assistance, learned extraction and no-graph inference separately."
- "Do not equate the 2026 GNN heuristic paper's known-dynamics setup with Rineya's learned transition model." This refers to P05 (Aichmüller et al., PMLR v306, "relational GNN action values with weighted A-star search and Q-learning").

**Reading anchors.** P05, P06, P07, P24, P30, P33.

**Risk.** K07 "GNN is helped by privileged parser" (Medium, owner "RGM/E07"):
- Mitigation: "Equal canonical facts and parser oracle ablation."
- Response: "Attribute gain to interface or available information."

### RGM-001 experiments

The shared fields (unit, sample plan, seed plan, compute policy) are the same as for RCM.

| ID | Name | Purpose | Primary endpoint (draft) | Dependency |
|---|---|---|---|---|
| **E07** | Relational transition comparison | "GNN versus flat/text/canonicalized models" | "Exact state-delta accuracy across object counts and input renderings" | "G0: audit / teacher qualification / preregistration" |
| **E08** | Learned search heuristic | "Exact dynamics with matched heuristic/search budgets" | "Exact goal success and node expansions with independently checked plans" | "E07 passes or documented independent rationale" |
| **E09** | Bounded program induction | "Conditional symbolic skills with execution checks" | "Interpreter-validated program success on unseen task compositions" | "E08 passes or documented independent rationale" |

**OOD splits for RGM.**
- Structural: larger object counts and held-out graph motifs.
- Surface: unseen object-name/rendering combinations, and reversed JSON keys per the BWM audit.
- Combined shifts are required. COMMON_PROTOCOL example: "train on small objects in familiar renderings and test large objects in unseen renderings."

**Ablations.** Text vs canonical text vs flat MLP vs GNN; randomized edge labels; permutation tests; raw-text vs gold-graph information tiers; parser-oracle ablation (K07).

**[Inference]** The controls text is copied at track level into every row, so E07 lists "fixed heuristic with identical search budget" even though that control belongs to E08.

### Shared protocol and compute (apply to both tracks)

Sources: `COMMON_PROTOCOL.md` and report §6 and §8.

**Data custody.**
- "Allocate the entire parent to TRAIN, DEV or SEALED before serialization."
- Pilot size: "about 200 independent parents per selected domain… not a power guarantee."
- Seeds: one training seed for engineering; "at least three genuinely independent training seeds for a replicated efficacy claim."

**OOD matrix (8 split types).** Same-distribution fresh payload, surface, numerical, semantic, structural, temporal, goal and domain. "Do not aggregate these into a single OOD claim without reporting every component." Include "copy/no-op and touched-entity floors."

**Statistics and decisions.**
- Paired parent-level comparisons: exact McNemar for binary outcomes, clustered paired bootstrap for regret or repeated variants, Holm for a small primary family.
- "one primary endpoint and one primary comparison per experiment."
- Decisions are ADVANCE / REVISE / STOP / INCONCLUSIVE.
- **[Inference]** E35 and the templates use GO / NO-GO / INCONCLUSIVE instead; this is a minor vocabulary inconsistency.

**Filler.** "constant length per public domain/action kind, chosen from TRAIN only."

**Retention (E33 accompanies every training candidate).** Preserve the D-36 gate: "no-go on a Holm-McNemar-detected category loss, category net below -11, or pooled net below -4% of items, under both graders." The old -5/75 rule is diagnostic only.

**Compute.**
- Hardware: RTX 4070 Laptop 8 GB envelope. 0.6B FP32 weights are about 2.4 GB decimal (about 2.24 GiB), a lower bound only.
- Profile about 20 steps before approving a full run. Single GPU lease and the existing thermal watchdog.
- Caps: RCM 4 h, RVM 3 h, RGM 4 h, RSM 4 h. P0 total = 15 h; all ten tracks = 28 h. These are "proposed shared ceilings for an INITIAL pilot per track… neither per-experiment allocations, runtime predictions nor execution authorization."
- Proposed post-calibration portfolio shares: "35% rule/relational pilots, 20% critics/skill distillation, 25% replication and retention, 15% routing/memory/belief baselines, and 5% optional latent/diffusion/social exploration."

### Cross-track (RCM × RGM)

- **Report §4:** "RCM changes what must be learned; RGM changes how it is represented… Train both flat and graph models on ordinary versus counterfactual rule data. If only the data factor matters, keep the simpler representation. If the graph factor matters only with gold input, the parser is the next problem."
- **E34** "Selected factorial synergy" (CORE, P1):
  - Endpoint: "Interaction S_AB - S_A - S_B + S_0 with parent-clustered uncertainty."
  - Dependency: "Two independently qualified components; compatible interfaces and matched budgets."
- **Roadmap:** Days 22–42 "Run the first RCM x RGM interaction."

---

## 4. "Shortcut generalization failures in RCWM PREREG-3"

The master report does **not** use the label "PREREG-3." It describes the finding without the ID in §1:

> "RCWM's ledger result is substantial within its tested distribution: decision regret changed from 0.544 to 0.056 and four-step exact final-state accuracy from 0.029 to 0.698. But the next audit found that the trained no-op cue generalized poorly to unfamiliar status strings, with accuracies of 0.01-0.18 on those strata. A model can acquire valuable task behavior while still failing the intended semantic rule. [R01]"

The label appears only in `07_evidence/evidence_ledger.json`:

- **EV03:**
  - `experiment: "RCWM PREREG-3"`, `status: "Confirmed failure boundary"`
  - `finding: "Unseen no-op statuses 0.01-0.18 versus trained void 0.99-1.00"`
  - `scope: "Three T2 seeds; 1,500 one-step audit rows; 80 chains"`
  - `limitation: "The positive in-distribution effect remains real but is trigger-bound"`
  - source R01 (RCWM-T1 REPORT.md, blob `4074833f…`, sections "3, 3b, 3c")
- **EV02** (the preceding positive result): `"RCWM PREREG-2"`
  - "Ledger regret 0.544 -> 0.056; exact four-step state 0.029 -> 0.698"
  - scope "150 fresh ledger parents; fixed T2 seed-0 adapter; within-generator bank"
  - limitation "not OOD rule mastery or broad no-forgetting"
- **EV01** `"RCWM PREREG-1"`: normalized regret 0.448; 300 parents; 256 goals; horizon 4.

The bundle cites two related surface-shortcut findings that are **not** PREREG-3:
- **EV05, BWM surface audit (exploratory):** "Training rendering 1.000; reversed JSON keys 0.458" on 120 keys at 5 and 10 blocks (RCWM decision R16). "Object-count generalization and serialization invariance are separate axes."
- **EV07, filler leak:** "Per-row trace-matched filler length leaked answer information." This led to the D-36 amendment 4b: fixed filler per family/domain and action kind.

**[Inference]** The bundle does not give more detail on PREREG-3: no per-status breakdown, no list of the "unfamiliar status strings", and no per-seed numbers. Only the 0.01–0.18 range versus 0.99–1.00 on the trained "void" status is stated.

**How the findings drive RCM/RGM.**
- They motivate RCM's nonce status words, opposite-rule pairs and canonicalized-input control.
- They lead to the teacher restriction: "T2 is not an unrestricted ledger oracle."
- Roadmap: "A failed rule-generalization gate blocks unrestricted teacher reuse."
- The BWM reversed-keys result motivates RGM's serialization-gap target.

---

## 5. Predictions and their IDs

All have `revision_rule: "Revise only after a logged pilot; never back-fit a confirmatory prediction"` unless noted.

- **H01 (RCM-001).**
  - Hypothesis: the RCM question.
  - Expectation: "High directional confidence for reducing the already-observed surface shortcuts; moderate confidence for new predicate combinations; low confidence for immediate broad language transfer. More balanced data alone may explain most of the gain."
  - Falsifier: the canonicalization / known-predicates / answer-only fallbacks.
  - Target: the +10 pp gate.
- **H07 (RGM-001).**
  - Expectation: "High confidence that an explicit relational representation removes avoidable name/order dependence when extraction is correct; moderate confidence in size transfer; uncertain benefit once parsing costs and strong symbolic baselines are included."
  - Target: the serialization gap / +10 / −25% expansions gate.
- **H31 (CORE).**
  - Hypothesis: "Verified specialists can teach an unaided 0.6B student some reusable decisions."
  - Expectation: "Moderate confidence for narrow task transfer; low confidence for broad retained intelligence gains in the first cycle."
  - Falsifier: "B improves while C does not beat matched SFT; conclude assistance, not internalization."
  - Target: "C beats A and matched SFT on a sealed task suite with protected retention."
  - Revision rule: "Preserve negative unaided results and report assistance separately."
- **H34 (CORE, the RCM×RGM tie-in).**
  - Hypothesis: "Rule repair and relational structure can reinforce one another."
  - Expectation: "Moderate confidence for a useful combination; low confidence for statistically super-additive effects."
  - Falsifier: "Interaction interval crosses zero or cost matching removes the advantage."
  - Target: "positive paired interaction on a predeclared scale, plus a useful cost frontier."
  - Revision rule: "An additive combination can still be worthwhile; do not relabel it synergy."
  - Sources: P02, P05, P06, P33.
- **H35 (CORE).** "Better evidence controls improve research efficiency even when candidate models fail." Observation: "No quantitative productivity effect measured."
- **Other track predictions** (not RCM/RGM): H04 RVM, H10 RSM, H13 RMM, H16 RBM, H19 REM, H22 RJP, H25 RDP, H28 ROM. Thirteen records in total.
- There are **no** predictions for E02, E03, E08 or E09 individually. IDs follow the pattern H01, H04, H07…, one per track.

Report §7 framing:
- "The expectation table… is a set of directional forecasts with qualitative confidence, not calibrated probabilities."
- "The most plausible first improvement is reduced surface sensitivity from diversified counterfactual data plus reliable parsing or relational representation. This need not imply learned causal structure."

---

## 6. Recommended first experiments and ordering

- **Report §12 (closing):** "The recommended next action is to finalize the shared protocol and execute E01 and E07 before expensive ensemble training. These two experiments directly distinguish semantic learning, data coverage, serialization repair and relational inductive bias. Their outcome determines whether Rineya should invest next in critics, skill distillation, better interfaces, or more fundamental rule acquisition."
- **`EXPERIMENT_REGISTRY_GUIDE.md`:** "Recommended starting order: finish shared controls and existing active registered work; E01 and E07; independently qualify E04; then E10 and E31. E33 accompanies every training candidate. E32 and E35 qualify a claim, while E34 is needed only to claim a specific interaction. P1/P2 pilots wait for a documented bottleneck and a separate go-ahead."
- **README:** "Begin with E01 (causal-rule semantics) and E07 (relational comparison), after shared controls and existing registered work are respected."
- **Within-track chains:** E01 → E02 → E03 and E07 → E08 → E09, each "passes or documented independent rationale."
- **Cross-track dependency:** E31 depends on "E01/E07 teacher eligibility, followed by qualified E04/E10 outputs."
- **90-day roadmap (relative, "not a schedule"):**
  - Days 1–7: evidence and baselines; qualify the loader and independent simulator. "Do not rerun the entire historical sealed evaluation as a development loop."
  - Days 8–21: "Run the RCM surface/semantics factorial and RGM text/flat/graph/canonicalization comparison"; build a CPU critic. "Select one promising mechanism per domain without opening the new confirmation bank."
  - Days 22–42: seeds, RVM, first RCM×RGM interaction, eligible teacher inventory.
  - Days 43–63: RSM distillation and A/B/C/D.
  - Days 64–77: IDR routing, retention matrix, one memory/belief experiment.
  - Days 78–90: frozen confirmation.

---

## 7. Explicit "do not duplicate / do not touch" statements about BWM and RCWM

**RCM reuse.** "Extend RCWM T2b and the joint cmp/calc/after kits; do not regenerate or retune their frozen banks. Existing generalization diagnostics are possible reuse candidates; remote availability needs confirmation."

**RGM reuse.** "Reuse BWM domains, planner interfaces and surface audit; preserve the completed 110-item hard result as historical evidence, not a tunable development set."
- EV04: "73/110… versus 5/110"; one seed; trained on ≤6 blocks.
- P23 boundary: "The already-read Blocksworld-hard set cannot become a fresh confirmatory instrument."

**T2 artifact.**
- "Do not merge the registered FP32 T2 adapter."
- "the registered artifact itself stays intact."
- TEACHER_QUALIFICATION: "T2's registered ledger range must not be silently widened."

**Frozen confirmations.** "The old frozen confirmations are now historical evidence. Do not tune on them, mine their failures into training, or call a re-run on them a fresh confirmation. Build new development mirrors with independent seeds/templates."

**Preregistrations and gates.**
- "existing preregistrations remain authoritative" (COMMON_PROTOCOL).
- "no universal gate overrides an existing preregistration" (E31–E35).
- Preserve D-36.
- Numeric targets are "not… already-approved changes to existing preregistrations."

**Branch policy.**
- "The verified current starting points are RCWM T2b and retention designs, BWM search/planner work and the IDR-001 branch README."
- "Before any new branch, inspect refs and files, record exact commits, and either extend the existing track or document the missing experiment. The proposed RCI track IDs are planning labels, not created Git branches."
- "Treat the private world-model records as source evidence, not a workspace in which to rewrite history."

**Data contracts.** "proposed exchange records, not replacements for a registered RCWM/BWM schema. Adapters must convert into existing interfaces without mutating frozen datasets."

**Launch checklist.** "The selected experiment is not a duplicate of an active or archived branch."

**Related reuse notes from other tracks.**
- RSM: "Extend the existing BWM search-distillation item."
- RVM: reuse RCWM goal-factorized comparisons; the joint roadmap's "blanket-verifier regression" is "a mandatory baseline."
- RJP: "Existing latent-head negatives in RCWM must be included."

---

## 8. Gaps and not found

- "G0" is never defined.
- E01 has no exact arm list beyond the 2x2 plus controls.
- No confirmation n or protected-stratum margins for any RCM/RGM experiment.
- The "serialization gap" target is not quantified.
- No dataset sizes beyond "about 200 independent parents per selected domain."
- No per-experiment GPU caps.
- No PREREG-3 per-status detail.
- The report's evidence on PREREG-3 cites only R01 sections 3/3b/3c, read as text with no rerun.
- `Rineya_Research_Tracker.xlsx` and the PDF were not inspected. The README says they mirror the same content.