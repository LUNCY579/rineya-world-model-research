# RCI-001 complementary research bundle: RVM-001, RSM-001, the other tracks, all 35 experiments, 13 predictions and 16 risks

Bundle root: `C:/Users/clclc/BusinessProject/WorldModel/rineya_complementary_research_2026-10-09`

## 0. Status: what is measured and what is planned

- **(a) Measured in this bundle: nothing native.** `09_quality/validation_summary.json` says `"native_experiments_run": 0`. All 35 registry entries have `"status": "Planned"`, `"approved": false`, `"actual_gpu_hours": null` and `"result": null`. All 13 predictions have `"observation": "Not measured in this bundle"`, except H35, which says "No quantitative productivity effect measured". Every track card begins: "Status: proposed research. No native experiment in this bundle."
- **(a′) Prior results the bundle cites but did not rerun.** These come from `07_evidence/evidence_ledger.json`, EV01–EV09, pinned at commit `a223884d55ff47baf3890c7d4a82054d07223215`:
  - RCWM ledger regret went from 0.544 to 0.056.
  - Four-step exact state went from 0.029 to 0.698.
  - Unseen no-op statuses scored 0.01–0.18.
  - BWM scored 73/110 with world model + beam versus 5/110 without.
  - The surface audit scored 1.000 on the training rendering versus 0.458 with reversed JSON keys.
  - The L8-15 localization prerequisite fell from 96/128 to 88/128 ("DO NOT PROMOTE").
  - IDR has "Implementation evidence only".
- **(b) Plans.** All of the following are plans:
  - track mechanisms
  - experiments
  - development targets, described as "proposed development gates rather than promised gains" in the README
  - GPU caps
  - the 90-day roadmap
  - the allocation shares (35% / 20% / 25% / 15% / 5%)
- **(c) My inferences** are marked as such below.

### Structural facts
- **Fields the registry does not have.** It has no "gate", "compute cap" or "first-pilot" field. Its fields are: id, track, name, priority, status, purpose, primary_endpoint_draft, dependencies, required_controls, development_target, evaluation_unit, sample_plan, seed_plan, compute_policy, source_ids, approved, actual_gpu_hours, result.
- **Gates.** The `dependencies` field acts as the gate.
  - The first experiment in each track depends on `"G0: audit / teacher qualification / preregistration"`. G0 is not defined anywhere else in the bundle.
  - The later experiments in a track depend on `"E(n-1) passes or documented independent rationale"`.
- **Caps are per track, not per experiment.** `pilot_gpu_hour_cap` in `04_experiments/tracks.json`:

  | Track | Cap (GPU-h) |
  |---|---|
  | RCM | 4 |
  | RVM | 3 |
  | RGM | 4 |
  | RSM | 4 |
  | RMM | 3 |
  | RBM | 2 |
  | REM | 2 |
  | RJP | 2 |
  | RDP | 2 |
  | ROM | 2 |

  - Totals: 28 GPU-h across all tracks; 15 GPU-h for P0.
  - Policy text for E01–E30: "Track cap is a proposed shared pilot ceiling, not a per-experiment allocation or execution permission."
  - Policy text for E31–E35: "Requires a separately approved confirmation budget."
  - The registry guide calls the caps "neither per-experiment allocations, runtime predictions nor execution authorization."
- **Recommended starting order** (`EXPERIMENT_REGISTRY_GUIDE.md`): "finish shared controls and existing active registered work; E01 and E07; independently qualify E04; then E10 and E31. E33 accompanies every training candidate. E32 and E35 qualify a claim, while E34 is needed only to claim a specific interaction. P1/P2 pilots wait for a documented bottleneck and a separate go-ahead."
- **Shared pilot defaults** (`COMMON_PROTOCOL.md`):
  - about 200 independent parents per domain
  - one seed for engineering; at least 3 independent seeds for a replicated claim
  - paired parent-level McNemar test or clustered bootstrap, with Holm correction
  - outcomes are ADVANCE / REVISE / STOP / INCONCLUSIVE
- **Consistency check.** The JSON and CSV versions of the registry, predictions and risks agree row for row (35 / 13 / 16 rows, no differing fields). The report's track sections match the track cards, minus the cards' "Registered experiment candidates" and "Source access" blocks.

---

## 1. RVM-001: validity critics, value models and successor features (P0, 3 GPU-h)

- **Purpose:** "Separate whether a step is valid, whether a state is useful, and what future features a policy produces."
- **Research question / H04:** "Can a small evaluator improve decision quality or reduce search without sharing the forward model's shortcuts?"
- **Mechanism.** Three distinct interfaces:
  - C(s,a,s_next,rule) predicts validity.
  - V(s,g) estimates goal progress or remaining cost.
  - ψ^π(s,a) predicts discounted future features under a specified policy.
  - "Use Q^π_w=ψ^π·w only when rewards are linear in the chosen features and dynamics remain fixed."
- **Minimal viable experiment:**
  1. Train a logistic/MLP legality critic on exact labels. Include model-generated near misses, incorrect copies, impossible transitions and format-corrupted answers.
  2. "Evaluate on errors from a held-out generator to avoid learning producer identity."
  3. Train a scalar value head and a successor-feature head on identical trajectories.
  4. Hold out goal-weight directions. Include outside-span weights and a separately reported changed-dynamics stress test.
- **Baselines and controls:**
  - random and constant scorers
  - deterministic validator ceiling
  - frozen confidence
  - scalar head with matched capacity
  - consequence features at equal labels
  - identical candidate banks
  - shuffled-label critic
  - "Do not give only the learned critic the rule or goal. Apply hard legality before optimizing soft goal scores."
- **Metrics:**
  - Critic: false-accept rate, recall at fixed false-accept rate, first-error identification.
  - Planning: exact goal success, normalized regret under the same candidate set, model calls, expansions, p50/p95 latency.
  - Goal transfer: in-span versus out-of-span performance.
- **Development target:** "+5 percentage points goal success at matched end-to-end compute, or at least 25% fewer expansions at noninferior success. A critic must improve the deployed planner, not only its classification AUROC."
- **Expected direction:** moderate confidence for selected error categories, lower for every domain. "Successor features are most promising for changing goals under fixed dynamics, not arbitrary new worlds."
- **Fallbacks:**
  - If blanket checking harms success, examine false rejection; use a bounded trigger only after calibration.
  - "If all candidate plans are wrong, repair proposer coverage rather than refining rankings."
  - If changed dynamics defeat successor features, record the assumption boundary.
- **Dense path:** "Use independently validated step labels and action preferences as supervision. Audit critic-selected data with the exact validator before distillation. Outcome preference training is optional only after competent SFT."
- **Reuse:** RCWM goal-factorized comparisons and the queued verification-guided rollout. "The joint roadmap reports a blanket-verifier regression; that is a mandatory baseline, not a reason to assume a new verifier helps" (from R04; the regression itself is not quantified in the bundle).
- **Experiments:**

  | ID | Name | Description | Primary endpoint | Depends on |
  |---|---|---|---|---|
  | E04 | Independent validity critic | Held-out error producers and near-miss labels | False-accept rate and valid-step recall on unseen error producers | G0 |
  | E05 | Goal-factorized value transfer | Successor features versus scalar value at equal labels | Goal-selection regret on held-out reward weights, fixed dynamics | E04 |
  | E06 | Value-guided bounded search | Budget-normalized search with an audited critic | Exact goal success at matched compute; expansions at noninferior success | E05 |

- **Sources:** P03 GenPRM, P04 Successor Features, P22 ProcessBench, P26 Calibration.
- **Risks owned:** K02 and K08 (see section 6).

## 2. RSM-001: hierarchical policies and reusable skills (P0, 4 GPU-h)

- **Purpose:** "Convert expensive verified planning into reusable next-action policies and conditional subroutines."
- **Research question / H10:** "Can search experience improve unaided planning and novel skill compositions rather than memorizing complete plans?"
- **Mechanism:**
  - Expert iteration, then optional dataset aggregation (DAgger, P09) on student-visited TRAIN states.
  - Options are defined by an initiation condition, argument bindings, a policy and a termination condition.
  - "Distinguish an externally executed option from a procedure internalized in model weights."
- **Minimal viable experiment:**
  1. Generate exact-validated solutions on fresh development worlds with variable object counts and horizons.
  2. Train a next-action student before a full-plan student.
  3. Induce a small subgoal library and train a controller.
  4. Keep primitive skills A and B in training, but hold out AB, BA and conditional A/B compositions at the parent-template level.
  5. Compare an SFT-only student with a student that also receives corrected self-rollout states.
- **Baselines and controls:**
  - equal-token generic task SFT
  - the same traces without option boundaries
  - random boundaries
  - fixed-depth planning
  - exact shortest-path teacher where affordable
  - no-search student evaluation
  - deduplication at environment and parent level
- **Metrics:**
  - no-search goal success, action legality, skill-boundary accuracy, task length reached, search savings, composition interaction
  - both orders of sequential acquisition, plus interruption/recovery cases
- **Development target:** "+5 points direct no-search success on fresh larger/harder tasks, with a positive paired interval and protected skill retention. Claim composition only when held-out compositions beat matched sequence-memorization controls."
- **Expected direction:** moderate confidence that verified short-horizon traces improve the proposer. Lower confidence that long-horizon or novel conditional compositions survive removal of the planner. "Partial amortization is already useful."
- **Fallbacks:**
  - If assisted performance rises but direct performance does not, keep the hybrid and diagnose exposure bias, teacher mismatch or compression limits.
  - Abandon hierarchy in tiny action spaces if options add overhead.
- **Dense path:** "Use Rineya as the next-action apprentice, optionally with progressively shorter verified traces… Full dense consolidation is a separate approval-gated experiment, never an automatic adapter merge."
- **Reuse:** "Extend the existing BWM search-distillation item". Reuse RHD/CSS and layered post-training only after their branch and native status are checked.
- **Experiments:**

  | ID | Name | Description | Primary endpoint | Depends on |
  |---|---|---|---|---|
  | E10 | Verified search distillation | Next-action versus full-plan supervised targets | Unaided student planning success at fixed decoding budget | G0 |
  | E11 | Novel option composition | AB/BA/conditional compositions not seen in training | Exact success on held-out skill orderings and prerequisite combinations | E10 |
  | E12 | Student-state aggregation | Exact relabeling of student-visited TRAIN states | Exact success on student-visited held-out states; original suite retention | E11 |

- **Sources:** P08 Expert Iteration, P09 DAgger, P07 programmatic skill induction, P34 Options.

## 3. RMM-001: modular experts, continual learning and meta-learning (P1, 3 GPU-h)

- **Purpose:** preserve acquired skills and route only among specialists that are competent for the operation.
- **Question:** can sequentially acquired skills coexist without router drift, competence dilution or shared failures?
- **Mechanism:**
  - One frozen backbone with audited, unmerged expert adapters and a role-specific router.
  - Compare fixed routes, learned routes, IDR quality-constrained balancing, replay and EWC-style regularization.
  - Meta-learning applies only to small adapters or the router.
- **Minimal viable experiment:**
  - Three validated skills in ABC, ACB and CBA orders; measure all skills after each stage.
  - Keep forward predictors, proposers and critics in separate pools.
  - Few-shot test on held-out rule families.
- **Controls:**
  - single adapter
  - fixed correct routing (reported as privileged)
  - no routing; random routing
  - equal trainable parameters
  - replay-only
  - nonuniform target utilization
  - "A router cannot recover a predicate no expert understands."
- **Metrics:** acquisition/retention matrix, worst-task loss, routing regret, abstention, label efficiency, memory, active compute, parameter bytes, adapter-on versus adapter-off behavior.
- **Development target:** retain prior skills within powered noninferiority margins, match new-skill acquisition, and outperform fixed routing under shifted mixtures.
- **Expected direction:** low confidence that a sophisticated router helps before specialists are qualified.
- **Reuse:** branch `research/rineya-id-balancing-skill-routing-v1` (IDR-001; "CPU-tested, native Rineya/CUDA efficacy unqualified").
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E13 | Qualified expert routing | IDR versus fixed routes with competence masks | G0 |
  | E14 | Sequential skill retention | ABC/ACB/CBA acquisition and retention matrix | E13 |
  | E15 | Few-shot adapter adaptation | Held-out rule families and label-budget curves | E14 |

## 4. RBM-001: belief-state models and metacognitive control (P1, 2 GPU-h)

- **Purpose:** track what is known and uncertain, and whether more information or computation is worthwhile.
- **Mechanism:** a distribution over small hidden states, or structured unknown/known/conflicting fields. A separate budget controller chooses answer / inspect / retrieve / verify / continue, using observable features and a fixed cost model.
- **Minimal viable experiment:**
  - Partially observed variants of existing worlds.
  - Compare no belief, exact Bayesian belief and a visible-state summary (the interface ceiling).
  - Then a learned recurrent filter, with delayed observations, parameter changes and contradictory evidence.
  - Test an entropy rule before a learned allocator.
- **Controls:**
  - oracle belief as a ceiling only
  - equal observation access
  - fixed extra compute
  - random allocation
  - input-blind difficulty prior
  - temperature-scaled confidence
  - no gold hidden state except in the labeled oracle arm
- **Metrics:** belief log loss/Brier score, success, wasted observations, clarification utility, selective risk at fixed coverage, success per wall-clock time. Recalibrate only on DEV.
- **Development target:** +5 points at fixed observation/compute cost, or 25% fewer unnecessary inspections at comparable success.
- **Reuse:** "The failed prerequisite deferral is a required negative case."
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E16 | Belief interface ceiling | Oracle beliefs versus equal-history summaries | G0 |
  | E17 | Compact belief filtering | Learned tracker under missing/noisy observations | E16 |
  | E18 | Compute and inspect controller | Fixed/random/input-blind versus learned allocation | E17 |

## 5. REM-001: episodic, associative and procedural memory (P1, 2 GPU-h)

- **Mechanism:** a separate store, writer, retriever and reader. Records carry source, event time, validity interval, confidence and supersession links. Facts are kept separate from verified procedures.
- **Minimal viable experiment:**
  - Multi-session histories with corrections, distractors, conflicts, delayed relevance and new names.
  - Compare full context, recency, simple retrieval and learned memory at identical retrieved tokens.
  - Add a style/paraphrase shift.
  - Run LongMemEval only after the local design is frozen.
- **Controls:** oracle retrieval ceiling, random records, shuffled source/time metadata, the same reader for all arms, equal storage/write budgets, and no future observations or test solutions in memory.
- **Metrics:** fact accuracy, update correctness, temporal attribution, evidence support, abstention, procedure execution, write amplification, retrieval latency, read/write/consolidation tokens.
- **Development target:** +5 points at matched context and storage budget, or a substantial token reduction with noninferior accuracy and no rise in stale-fact errors.
- **Reuse:** RHF-001 on `research/rineya-hebbian-fact-memory-v1` and the factorized/triadic memory branches are "not fully re-audited here".
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E19 | Provenance-aware memory baseline | Recency/retrieval/full-context matched readers | G0 |
  | E20 | Learned write policy | Value-aware storage with corrections and distractors | E19 |
  | E21 | Memory transfer and updates | Held-out paraphrases, facts and workflow changes | E20 |

## 6. RJP-001: latent predictive and energy-based models (P2, 2 GPU-h)

- **Mechanism:** three separate choices — masked representation prediction, object-level masking, and an energy score over discrete transitions. Uses anti-collapse regularization (VICReg) and an explicit decoder or validator. "A latent compatibility score is not a calibrated probability or proof of causality."
- **Minimal viable experiment:** structured states and one-step transitions (not video). Include exact equality and arithmetic cases. Try latent rollouts with periodic exact decoding only after one-step validation.
- **Controls:**
  - equal-size supervised encoder
  - random frozen projection
  - reconstruction baseline
  - shuffled transitions
  - decoder-only baseline
- **Metrics:** exact decoded state, validity ranking, OOD rule compliance, downstream planning at matched time, latent bytes, rollout error by depth, plus a full-text baseline.
- **Development target:** "25% lower measured inference cost at powered noninferior exact task success."
- **Reuse:** "Existing latent-head negatives in RCWM must be included."
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E22 | Structured-state JEPA pilot | Predict representations versus supervised state encoding | G0 |
  | E23 | Object-mask causal-bias audit | Intervention pairs and lossless state checks | E22 |
  | E24 | Energy-ranked candidates | Contrastive compatibility at matched search cost | E23 |

## 7. RDP-001: diffusion and revisable proposal models (P2, 2 GPU-h)

- **Purpose:** an alternative proposer, used only where proposal coverage is the limit.
- **Mechanism:** a small discrete masked-plan denoiser or an offline frozen proposal bank. "An iterative masked-plan denoiser is not a reproduced general-purpose diffusion language model."
- **Minimal viable experiment:** first, the oracle-best candidate ceiling at several budgets. Then compare autoregressive sampling, local edit/repair, randomized search and masked-plan denoising, all with the same validator, and measure wall time.
- **Controls:** the same checker, vocabulary, horizon, total evaluations and tuning budget. Repair the same initial proposals, to separate denoising from simply getting more chances.
- **Development target:** "+5 points independently verified plan coverage at equal end-to-end compute, followed by actual selected-success improvement."
- **Expected direction:** "Low confidence of near-term net value on 8GB… Do not train an 8B diffusion model from scratch."
- **Dense path:** distill verified diverse solutions and repairs, with a plain SFT control at equal accepted examples and tokens.
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E25 | Candidate coverage ceiling | Current sampling versus oracle-best reranking (pass@K) | G0 |
  | E26 | Bounded plan denoising | Small masked-action proposals versus local repair | E25 |
  | E27 | Verified revision distillation | Transfer diverse corrections to an AR student | E26 |

## 8. ROM-001: social, opponent and character-belief models (P2, 2 GPU-h)

- **Mechanism:** separate true world state, per-character information, traits and goals. Style is a separate control variable.
- **Minimal viable experiment:** synthetic fictional scenarios (missed events, changed promises, conflicting goals, corrected facts). Hold out identities. Compare role prompts, full history, explicit belief cards and a learned tracker.
- **Controls:** same history, shuffled identity labels, fixed style prompt, oracle belief cards as a privileged ceiling, impossible-knowledge probes. "This program does not modify the user's game repository."
- **Metrics:** belief-consistent action rate, unsupported-knowledge leakage, promise/relationship consistency, correction recovery, instruction validity, blinded style judgments reported separately.
- **Development target:** "+10 points structured belief consistency on novel scenarios with no deterioration in external instruction adherence, plus blinded style noninferiority."
- **Experiments:**

  | ID | Name | Description | Depends on |
  |---|---|---|---|
  | E28 | False-belief state tracking | Distinct observations and hidden beliefs by character | G0 |
  | E29 | Character consistency | Traits, promises and corrections under novel events | E28 |
  | E30 | Cooperative social planning | Held-out goals with separately evaluated dialogue | E29 |

## 9. CORE experiments E31–E35

These five share the following:
- Controls: "Frozen teacher and baseline custody; independently graded outcomes; equal-label/token/cost controls."
- Target: "Freeze track-appropriate practical effect and powered retention margins before confirmation; no universal gate overrides an existing preregistration."
- Seeds: at least 3.
- Budget: "Requires a separately approved confirmation budget."

### E33, quoted exactly from the registry
- id: `E33`
- track: `CORE`
- priority: `P0`
- name: "Protected capability retention"
- purpose: "Detect damage that activation drift and KL can miss"
- primary_endpoint_draft: "Correctness under both preregistered graders; category and pooled changes"
- dependencies: "Existing retention custody and applicable D-36 gate preserved"
- sources: R01–R05

### The D-36 gate (Report §1, EV08)
- Wording: "no-go on a Holm-McNemar-detected category loss, category net below -11, or pooled net below -4% of items, under both graders. The old -5/75 category flag remains diagnostic. A future claim of no more than two percentage points of loss needs a separately powered noninferiority design; a nonsignificant 75-item category result cannot establish it."
- EV08 status: "Frozen design; not a model pass."
- **Stricter dense-student gate** (COMMON_PROTOCOL): "define an overall noninferiority margin prospectively, simulate its power, and protect critical categories individually. Include correctness, instruction compliance, arithmetic, retrieval fidelity, abstention and role consistency where relevant; KL and greedy agreement are diagnostics, not sufficient guarantees."
- **Why it exists:** the L8-15 screen failed (96→88/128) despite less drift.
- **Linked risk:** K05.
- **Role in the plan:** "E33 accompanies every training candidate" (registry guide, and README: "E33 retention accompanies every trained candidate").

### The other CORE experiments
- **E31, four-arm assistance and student transfer (P0).**
  - Endpoint: exact held-out success for arms A/B/C/D and a matched-SFT control.
  - Depends on: "E01/E07 teacher eligibility, followed by qualified E04/E10 outputs."
- **E32, second-domain replication (P0).**
  - Endpoint: domain-specific exact success and worst-stratum performance.
  - Depends on: winning recipe fixed; second generator and split sealed.
- **E34, selected factorial synergy (P1).**
  - Endpoint: I = S_AB − S_A − S_B + S_0, with parent-clustered uncertainty.
  - Depends on: two independently qualified components.
  - The first proposed 2×2 designs are RCM×RGM, RVM×RSM and REM×RBM.
- **E35, frozen confirmation and decision (P0).**
  - Outcome: GO / NO-GO / INCONCLUSIVE.
  - Depends on: E31, E32 and E33; E34 only for a synergy claim; "no tuning on final data."

---

## 10. All 35 experiments

- **Depends-on** shows "G0" or the upstream experiment id. For E01–E30, an upstream id means "passes or documented independent rationale".
- **Cap** is the shared track pilot cap in GPU-h; "Sep." means it needs a separately approved confirmation budget.
- **First-pilot** is not a field in the bundle; this column is my inference.
  - **REC** = named in the recommended starting order.
  - **R** = runs alongside training candidates (E33 only).
  - **G0** = entry point of its track.

| ID | Track | Pri | Short title | Depends-on | Cap | First-pilot |
|---|---|---|---|---|---|---|
| E01 | RCM | P0 | Rule semantics factorial | G0 | 4 | REC (first) |
| E02 | RCM | P0 | Computed-predicate extrapolation | E01 | 4 | – |
| E03 | RCM | P0 | Instruction-rule bridge | E02 | 4 | – |
| E04 | RVM | P0 | Independent validity critic | G0 | 3 | REC ("independently qualify") |
| E05 | RVM | P0 | Goal-factorized value transfer (successor features vs. scalar) | E04 | 3 | – |
| E06 | RVM | P0 | Value-guided bounded search | E05 | 3 | – |
| E07 | RGM | P0 | Relational transition comparison | G0 | 4 | REC (first) |
| E08 | RGM | P0 | Learned search heuristic | E07 | 4 | – |
| E09 | RGM | P0 | Bounded program induction | E08 | 4 | – |
| E10 | RSM | P0 | Verified search distillation | G0 | 4 | REC |
| E11 | RSM | P0 | Novel option composition | E10 | 4 | – |
| E12 | RSM | P0 | Student-state aggregation | E11 | 4 | – |
| E13 | RMM | P1 | Qualified expert routing | G0 | 3 | G0 (waits for bottleneck) |
| E14 | RMM | P1 | Sequential skill retention | E13 | 3 | – |
| E15 | RMM | P1 | Few-shot adapter adaptation | E14 | 3 | – |
| E16 | RBM | P1 | Belief interface ceiling | G0 | 2 | G0 (waits) |
| E17 | RBM | P1 | Compact belief filtering | E16 | 2 | – |
| E18 | RBM | P1 | Compute and inspect controller | E17 | 2 | – |
| E19 | REM | P1 | Provenance-aware memory baseline | G0 | 2 | G0 (waits) |
| E20 | REM | P1 | Learned write policy | E19 | 2 | – |
| E21 | REM | P1 | Memory transfer and updates | E20 | 2 | – |
| E22 | RJP | P2 | Structured-state JEPA pilot | G0 | 2 | G0 (waits) |
| E23 | RJP | P2 | Object-mask causal-bias audit | E22 | 2 | – |
| E24 | RJP | P2 | Energy-ranked candidates | E23 | 2 | – |
| E25 | RDP | P2 | Candidate coverage ceiling | G0 | 2 | G0 (waits) |
| E26 | RDP | P2 | Bounded plan denoising | E25 | 2 | – |
| E27 | RDP | P2 | Verified revision distillation | E26 | 2 | – |
| E28 | ROM | P2 | False-belief state tracking | G0 | 2 | G0 (waits) |
| E29 | ROM | P2 | Character consistency | E28 | 2 | – |
| E30 | ROM | P2 | Cooperative social planning | E29 | 2 | – |
| E31 | CORE | P0 | Four-arm assistance and student transfer | E01/E07 eligibility → qualified E04/E10 | Sep. | REC |
| E32 | CORE | P0 | Second-domain replication | Winning recipe fixed; second domain sealed | Sep. | – (qualifies a claim) |
| E33 | CORE | P0 | Protected capability retention | Retention custody + D-36 preserved | Sep. | R (with every trained candidate) |
| E34 | CORE | P1 | Selected factorial synergy | Two qualified components | Sep. | – (only for a synergy claim) |
| E35 | CORE | P0 | Frozen confirmation and decision | E31/E32/E33 (+E34 for synergy) | Sep. | – |

Priority counts: 16 × P0, 10 × P1, 9 × P2.

## 11. Predictions (13)

All have "Not measured", except H35 ("No quantitative productivity effect measured"). For H01–H28 the `revision_rule` is "Revise only after a logged pilot; never back-fit a confirmatory prediction."

| ID | Track | Expectation (condensed) | Falsifier / alternative (condensed) |
|---|---|---|---|
| H01 | RCM | High: fewer surface shortcuts. Moderate: new predicate combinations. Low: broad language transfer. "More balanced data alone may explain most of the gain." | Gains vanish with canonicalization → credit the interface. Traces lose to balanced answer-only → keep the simpler recipe. |
| H04 | RVM | Moderate: critic helps selected error categories. Successor features best for changing goals under fixed dynamics. | Blanket-checking harm → bounded trigger. All candidates wrong → fix coverage. |
| H07 | RGM | High: relational representation removes name/order dependence when extraction is correct. Uncertain once parsing costs are charged. | Exact symbolic model wins cheaply → use it as teacher. GNN works only with gold graphs → interface bottleneck. |
| H10 | RSM | Moderate: short-horizon traces help. Lower: long or novel compositions without the planner. | Assisted up, direct flat → keep the hybrid. Option overhead → drop hierarchy. |
| H13 | RMM | Moderate: isolation reduces interference. Low: a fancy router helps before qualification. | Balancing activates weak experts → competence masks. Static routes suffice → prefer them. |
| H16 | RBM | High: correct beliefs help in ambiguous tasks. Moderate-to-low: a cheap learned 0.6B tracker gets there. | Oracle helps, learned does not → repair inference. |
| H19 | REM | High: scoped assistance when context-limited. Uncertain: learned writer beats retrieval. | Gain due only to more tokens → memory-policy claim fails. |
| H22 | RJP | Low-to-moderate efficiency gain. "A compact latent can discard the bit that matters." | Smooth but wrong → lossless structure. No gain over a supervised encoder → defer the track. |
| H25 | RDP | Low near-term value on 8GB. | Local repair matches denoising → keep the cheaper method. |
| H28 | ROM | Moderate for scoped role-play. Low for unaided dense transfer. | Style up while facts fail → reject the joint score. |
| H31 | CORE | "Verified specialists can teach an unaided 0.6B student some reusable decisions." Moderate for narrow transfer, low for broad first-cycle gains. | "B improves while C does not beat matched SFT; conclude assistance, not internalization." Target: C beats A and matched SFT with protected retention. |
| H34 | CORE | Rule repair × relational structure: moderate for a useful combination, low for super-additivity. | Interaction interval crosses zero, or cost matching removes the advantage. "Do not relabel [additive] synergy." |
| H35 | CORE | Better evidence controls improve research efficiency (high qualitative confidence). | Audits don't change decisions or cost more than they save. Track rejected teachers, invalid runs, duplicates. |

## 12. Risks (16)

Format: id, risk [severity], owner — mitigation; stop/response.

- **K01** Shortcut masquerading as semantics [High], RCM/E01 — fresh surface, inverted-rule and computed-predicate holdouts; do not broaden the claim.
- **K02** Teacher and critic share wrong rules [High], RVM/E04 — exact independent validator and held-out error producer; "Reject pseudo-labels outside teacher support."
- **K03** Label or renderer leakage [High], ALL — parent-first splits, semantic hash audit, fixed filler length; invalidate the comparison and issue a new bank.
- **K04** Apparent gain is extra compute [High], ALL — total tokens/calls/latency and a matching cost frontier; report the tradeoff.
- **K05** Catastrophic forgetting hidden by averages [High], E33 — both correctness graders; per-category noninferiority; "Keep candidate unpromoted."
- **K06** Native laptop instability [High], ALL — lease, thermal watchdog, serial jobs, preflight; stop safely and preserve logs.
- **K07** GNN helped by privileged parser [Medium], RGM/E07 — equal canonical facts and parser-oracle ablation; attribute the gain to the interface.
- **K08** Critic rejects valid candidates [Medium], RVM/E04–E06 — false-reject rate, oracle ceiling, planner-level test; "Use simpler/no critic when better."
- **K09** Router balances incompetent experts [High], RMM/E13 — independent competence qualification; "No quota for unqualified experts."
- **K10** Meta-learning test tasks resemble train tasks [Medium], RMM/E15 — hold out generators and budgets; narrow the claim.
- **K11** Memory becomes stale or leaks answers [High], REM/E19–E21 — provenance, versioning, split-scoped stores, deletion checks; reject and rerun clean.
- **K12** Latent collapse or arithmetic loss [Medium], RJP/E22 — variance checks plus exact downstream tests; prefer an explicit structured model.
- **K13** Diffusion adds cost without valid diversity [Medium], RDP/E25–E27 — pass@K at equal budget; keep the autoregressive baseline.
- **K14** Social inference uses inaccessible knowledge [High], ROM/E28 — observation-scoped belief states; reject gold-state leakage.
- **K15** Research scope explodes [High], CORE — only the P0 core first, gated promotion; park noncritical branches.
- **K16** Publication or license violation [High], ALL — private delivery and rights review; no public redistribution or model release.

Severity counts: 11 High, 5 Medium.

---

## 13. Cross-cutting themes

### Critic and teacher qualification
- **Template:** `06_templates/TEACHER_QUALIFICATION.md`.
  - "Status: UNQUALIFIED until evidence below is completed."
  - Roles covered: proposer / dynamics / critic / value / interpreter.
  - Must specify domain, rule family, names, input schema/order, numerical range, horizon, observation boundary and adapter state.
  - "A successful canonical-format result is not permission to label novel forms. T2's registered ledger range must not be silently widened."
  - Record false application, false copying, impossible state, arithmetic error, format error and missing observation separately, including rejected labels.
  - Use a "fail-closed eligibility predicate before generation"; "uncertainty is not a gold answer."
  - "Do not grade a learned predictor solely with the same predictor or a critic trained to mimic it."
  - Requalification is required for new data, rendering, precision, loader or merge state.
- **Report §5:**
  - "T2 is not an unrestricted ledger oracle."
  - Learned-model-accepted plans must still pass the exact environment before entering the gold stream.
  - Keep a blinded manual audit of the parser and checker.
  - "Never allow a model under evaluation to write its own grading rules."
- **Report §4:** "A consensus of three models that share the same shortcut is not a certificate."
- **Roadmap:** days 8–21 develop the critic on CPU. "A failed rule-generalization gate blocks unrestricted teacher reuse." Days 22–42 deliver an "eligible teacher inventory".

### Successor features
- **Glossary:** "expected discounted future feature totals under a specified policy… It is not a universal solution to arbitrary new rules."
- **P04 boundary** (from resources.csv): "Do not extend its transfer guarantees to new dynamics or arbitrary nonlinear rewards."
- **Testing:** E05 tests successor features against a matched-capacity scalar head. It holds out goal-weight directions, splits in-span from out-of-span weights, and runs a separate changed-dynamics stress test. The goal split is one axis of the OOD matrix ("changes reward weights").

### Search distillation
- **Experiments:** E10–E12 in RSM.
- **Report §4:** "Test no-search performance explicitly rather than inferring internalization from successful assisted runs… In a tiny action space, plain search may remain superior."
- **Roadmap days 43–63:** exact-verified TRAIN trajectories, RSM next-action distillation with matched-SFT controls, A/B/C/D tiers, and an aggregation arm "if exposure bias is evident".
- **Planned allocation:** 20% of GPU share to critics and skill distillation.

### Dense transfer
- **Source:** `05_execution/INTEGRATION_AND_TRANSFER.md` and Report §5.
- **Four arms:**
  - A = frozen core, no assistance
  - B = core plus specialist/tool
  - C = fresh student, unassisted
  - D = student plus the same assistance as B
- **Contrasts:**
  - B−A = system assistance
  - C−A = training transfer
  - D−B = whether student training helps the assisted system
  - D−C = remaining value of assistance
  - A matched-SFT control is mandatory.
- **Artifact labeling:** "Do not call a student adapter a new dense checkpoint merely because it is loaded on a dense backbone." Report frozen-base + adapter, modular multi-adapter, and genuinely consolidated dense export separately. "Registered T2 remains unmerged in FP32." Consolidation happens only in an independent, approved lane with parity and retention tests.
- **Training sequence:**
  1. competent SFT
  2. predicate/trace targets versus answer-only and fixed-filler controls
  3. exact corrections on TRAIN student states
  4. preference optimization only once correctness is grounded, with an SFT-only comparator
  5. shorter traces or direct answers as separate tiers
- **Bridge tasks:** compositional instructions, prerequisite constraints, evidence-grounded updates, algorithm execution, fictional role consistency. These are scored separately: "A single weighted total must not hide a catastrophic category loss."

### Data contract
- `DATA_CONTRACTS.md`: the target is an ordered `cmp`, `calc`, `after` object. Gold sits in a separate evaluator lane. There is no per-row filler (EV07: per-row trace-matched filler "leaked answer information").

## 14. Not found / caveats
- No explicit "gate", per-experiment compute cap or first-pilot flag field exists in `experiment_registry.json`. The table above derives them as described.
- "G0" appears only as a dependency string; nothing defines it further.
- The "blanket-verifier regression" (R04) is mentioned but not quantified anywhere in the bundle.
- RCM-001 and RGM-001 were outside the requested reading. Their E01–E03 and E07–E09 entries above come only from the registry and predictions.