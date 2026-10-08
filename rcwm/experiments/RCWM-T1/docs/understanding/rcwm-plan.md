# RCWM bundle reader report: sections 04–10, specs, roadmap, configs and validation

Bundle root: `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05`. I read every file assigned to me plus the reference code (`reference/rcwm/*.py`). I modified nothing. I ran two read-only CPU scripts from the scratchpad. One counts coverage in the demo dataset with numpy. The other runs the bundled 40,325-param toy checkpoint (`validation/demo/synthetic_demo_weights.pt`) on CPU with `weights_only=True`. There was no GPU use and no training. Free RAM was measured at about 4.2 GB, not 2 GB.

---

## (1) The first decisive experiment (G1 → G2)

**Framing** (main plan §"The first decisive experiment"). Use a deterministic text/symbolic environment with:
- exact state snapshots
- executable, legal-validated actions
- an independent outcome checker
- multiple action branches from the same starting state
- a frozen Rineya language backbone

Compare a matched scalar Transformer, a measured-feature predictor and a learned-consequence predictor. Hold out complete problem families and goal combinations. The success claim is deliberately narrow: "on held-out parents, the consequence representation supports better goal-conditioned decisions at a measured cost and without changing the frozen language model." A null result counts as informative.

**Environment** (05 "Data program"):
- KeyDoor (`reference/rcwm/synthetic.py`) is only a mechanics example.
- G1 needs a broader symbolic suite: prerequisite chains, bounded program execution, document-evidence retrieval, and resource-constrained parallel subtasks.
- It must be small enough that an oracle can exhaustively evaluate candidate plans.
- Each environment needs immutable snapshots, legal-action validation, an independent outcome check and a controllable observation function.
- Four axes vary separately: context length, decision horizon, branching factor and partial observability.
- TextWorld/ALFWorld and then sandboxed web tasks come only after the exact-simulator gates pass.

**Collection** (05):
- Mix legal random actions, heuristic actions, the frozen Rineya proposer and deliberately chosen alternatives. Log which policy chose each action and what information it had.
- Keep failures, invalid attempts, no-ops and timeouts with explicit status labels.
- Plans must be chosen before branch execution: `plan_origin=open_loop_precommitted`. `validate_open_loop()` in `data.py` rejects anything else.
- Use branching interventions from restored snapshots, with all branches in the same split.
- Balance the training sampler by transition type, but keep a separate natural-prevalence calibration set.

**Arms** (07 model matrix; `specs/EVALUATION_PROTOCOL.md`):

| ID | Arm | Gate |
|---|---|---|
| B0 | legal random / heuristic | G1 |
| B2 | scalar trajectory Transformer (strong, attention-based, long context, multi-output allowed) | G1, G2 |
| B3 | measured consequence features + goal readout | G1, G2 |
| O1 | exact simulator evaluator on the same candidates (ceiling, not deployable) | G1 |
| B4 | learned continuous consequences + readout — "the central scientific arm" | G2 |
| B1 | frozen Rineya direct action/answer | enters once native capture is qualified (G0-native) |

- **Primary contrast:** `B4 minus matched B2` (`configs/evaluation_gates.yaml`).
- **Matching:** B2 must match B4 on context, parameters, labels, tuning budget, representation-learning/pretraining budget, candidate set and inference budget. A plain MLP over local state does not count as a baseline.
- **Candidate sets:** every evaluator scores the same hashed CandidateSet, recorded before scoring.

**Sidecar config** (`configs/pilot_sidecar.yaml`; `ARCHITECTURE_CONTRACT.md`):
- Shape: width 256, 3 layers, 4 heads, latent_dim 64, max_action_horizon 4, `mask_mode: prospective_causal`.
- Encoder: frozen, `max_text_tokens: 512`, `cache_action_boundaries_only: true`.
- Optimisation: AdamW with lr ∈ {1e-4, 3e-4, 1e-3} (dev-only sweep), batch_micro 1, effective_batch 32, grad clip 1.0, `max_steps: null` (null blocks the run).
- Loss weights: grounded_features 1.0; latent_prediction, contrastive, variance_covariance and vq all 0.0 ("Enable one mechanism at a time; learned-target arm requires target-encoder design approval").
- Plan-level screen ranges: width 128–256, 2–4 layers, 32–64 latent dimensions.
- Batch tensors: context `[B,Lc,dc]`, actions `[B,H]`, targets `[B,H,F]`, target-validity `[B,H,F]`, step-validity `[B,H]`, split/parent index. The model must expose both `predict_step` and `predict_plan`.

**How B4 is built** (05 Stage B; `TRAINING_RECIPE.md` T1–T3):
- `target_z = stop_grad(E_target(record))` and `pred_z = F(available_prefix, candidate_action_prefix)`.
- Loss: L_pred = mean‖ẑ − sg(z)‖² over masked steps.
- Add exactly one anti-collapse mechanism: a grounded observable decoder, CPC, or VICReg-style variance/covariance. They are compared later, one at a time (T2).
- Start with a fixed target encoder; compare against an EMA encoder later.
- Latent dimensions to try: {32, 64, 128} (`feature_registry.yaml`).
- The readout u_ω(z,g) is linear first and fit only on the permitted calibration labels. A small nonlinear readout is a separate arm. A hand-written goal vector must never be dotted into the learned latent.
- Monitor collapse: variance, rank, decoder quality, held-out linear and nonlinear probes, and real versus shuffled action labels.

**Splits** (`DATA_AND_SPLITS.md`; `data.py::split_for_parent`):
- Assign the parent-family key before generating any variants; every descendant inherits the parent's split.
- `split_for_parent` hashes sha256 into train/dev/test at 70/15/15.
- Proposed sizes: **2,000 train / 300 dev / 500 sealed test parents**, with "several matched branches per parent". These are a starting budget, not a power calculation.
- A new-goal calibration set must be separate from the held-out evaluation parents.
- Normalizers, codebooks, pretraining and readouts are fit on TRAIN only.
- Freeze the final sample size from dev-pilot variance before the sealed set is opened.

**Metrics** (07; EVALUATION_PROTOCOL):
- **Primary:** actual independent-outcome utility / task success under a fixed candidate set, plus candidate regret (`metrics.py::candidate_regret`).
- **Secondary:** consequence fidelity, Brier/log score, calibration by operator, OOD rejection coverage, label-efficiency curves, runtime/cost, and candidate-oracle gap (O1 − selected).
- **Controls:** shuffled actions, future-leak perturbation, and a search-width sweep at 1/4/16/32 candidates comparing predicted utility with real utility. A widening gap means the controller is exploiting model error.

**Statistics** (07):
- The unit is the parent family. Average paired outcomes within each parent, then run a parent-level paired bootstrap (`paired_parent_bootstrap`; the config sets 10,000 draws at 0.95 confidence).
- Seeds 17, 29, 43. Report seed uncertainty separately.
- Power formula: n ≈ ((1.96+0.84)σ/δ)². With σ=0.25 and δ=0.03 this gives **≈545 independent parents**. Note that the proposed 500 test parents falls short of this.

**Gates:**
- **G1:** the oracle solves the intended class; a tiny diagnostic set can be overfit; the measured-feature model learns nontrivial action dependence. Report failures stratified by observation ambiguity, label validity and optimisation. A broken instrument blocks everything downstream.
- **G2:** B2/B3/B4 on paired held-out parents, reporting an effect estimate, confidence interval and resource accounting.
- **Promotion** requires all of:
  - a positive paired CI
  - a preregistered minimum effect
  - no cost or retention violation
  - every null field in `evaluation_gates.yaml` filled in. Currently null: `primary_endpoint`, `minimum_practical_effect`, `retention_noninferiority_margin`, `max_total_latency_ratio`, `final_parent_count`. `null_threshold_policy: BLOCK_PROMOTION`.

---

## (2) Gated experiments and hypotheses

**Hypotheses** (07):
- **H1, useful representation:** learned consequence features beat a matched scalar Transformer at candidate selection. Falsified if the gain disappears once context, parameters, labels or tuning are matched.
- **H2, transferable utility:** a frozen representation supports new in-span goals with fewer grounding labels than refitting a scalar evaluator. Falsified if the gain depends on leakage, a task-specific encoder update, or properties that are not actually represented.
- **H3, temporal composition:** gains appear when outcomes depend on several separated actions. Falsified if gains persist after dependencies are removed, or are explained by a larger context window alone.
- **H4, discovered structure:** clustering or VQ improves label efficiency, compositional transfer or interpretability without harming rare outcomes. Falsified if continuous or random-projection controls do equally well, codes collapse, or clusters track task IDs.
- **H5, deployable integration:** the native typed-slot route or a sidecar improves real task performance within cost and preservation limits. Falsified if planning overhead, the real executor, or retention below the noninferiority margin erases the gain.

**Gate chain:** G0 provenance/mechanics → G1 instrument → G2 learned utility → G3 reuse/discovery → G4 composition/horizon → G5 native adaptation/preservation → G6 optional distillation or imagined RL.

**Registry** (`roadmap/experiment_registry.csv`, 21 rows):

| ID | Experiment | Status / acceptance focus |
|---|---|---|
| G0-01 | provenance & capture | native blocked until approved |
| G0-02 | mask/split adversaries | CPU subset tested |
| G1-B0, B2, B3, O1 | B3 and O1 | "toy mechanics included" |
| G2-B1 | frozen Rineya direct policy | needs G0-native |
| G2-B4 | learned continuous consequences | proposed |
| G3-01 | sparse grounding budget curve, B2/B3/B4 | nested identical labeled parents |
| G3-02 | train-only k-means (B5) | utility only implemented |
| G3-03 | learned VQ | deferred |
| G3-04 | new-goal in-span revaluation (B4/B6) | proposed |
| G3-05 | out-of-span goal property | deferred |
| G4-01 | direct successor vs rollouts (B6) | deferred |
| G4-02 | long-context dependency sweep (B2/B4) | deferred |
| G4-03 | synchronous parallel DAG | deferred |
| G5-B7 | native typed-slot adapter | retention noninferiority + measured memory |
| G6-01 / G6-02 | planner distillation / short imagined RL | real anchors + exploitation sweep |
| G7-01 / G7-02 | compute-depth controller / visual extension | deferred |

**G3 protocol details** (04):
- Nested label budgets: 1, 5, 10, 25 and 100% of the same TRAIN parents, identical across arms.
- k-means with K ∈ {16, 64, 128} fit on TRAIN only, centers frozen before dev/test assignment.
- Compare hard, soft and unclustered representations at equal downstream capacity, plus random-projection and random-label controls.
- For VQ, log code usage, entropy, dead codes, turnover and distortion.
- Inspect clusters only after the quantitative screen, using blinded descriptions tested on fresh examples.
- **Primary label-efficiency endpoint:** the number of grounding labels needed to reach a preregistered regret or success target, with uncertainty and full data-production cost.

**G4 grid** (06):
- Context bands 512 / 2,048 / 4,096 / 8,192 tokens; horizons 1/2/4/8; dependency distance varied independently.
- Hold relevant information fixed while adding distractors; separately, increase the number of truly relevant prerequisites.
- Compare full history, bounded retrieval and hierarchical summaries.
- Parallel work uses synchronous-round DAGs scored on makespan, violations, wasted work and joint-success labels. Joint outcomes cannot be taken as the product of marginals.
- Ablation: terminal-only grounding vs measured prerequisite features vs learned future embeddings, with matched label costs.

---

## (3) Most promising publishable directions and the authors' risks

**Claim the authors call strongest** (09): "under a frozen data, compute and preservation budget, a learned consequence representation improved held-out goal-conditioned action selection or reduced the required grounding labels compared with a matched scalar evaluator." Their candidate findings are:
- fewer reward labels needed for new goals
- better choice among plans that need held-out skill compositions
- higher accuracy at equal total reasoning cost
- distinctions preserved across domains where scalar evaluators fail

**My ranking:**

1. **H2 label efficiency (G3-01 / G3-04), B4 vs B2.** This is the most tractable to publish. The scalar baseline must refit or relabel for every new goal, while B3/B4 only refit a linear readout. The measurable outcome is a regret-vs-labels curve.
   - **Caveat:** for in-span goals, B3 (measured features) satisfies in-span revaluation by construction. B4 therefore needs to beat B3 on something to show learned geometry adds value: out-of-span or new-outcome transfer (G3-05, currently deferred), or rare-distinction retention.
2. **Decision-aware consequence fidelity: rare prerequisites vs average error.** The bundle's own demo (section 5 below) shows near-zero average MSE alongside maximal decision regret. Risks 04 and 09 name this directly ("rare critical prerequisite dropped", "rare failure merged away"). A result where regret-weighted or prerequisite-stratified objectives repair it, and B2 or a plain-MSE B4 fails, would be a crisp and novel finding. It is cheap to run on symbolic environments on the 8 GB laptop.
3. **H3 composition and dependency distance (G4-02).** This lines up with Rineya's long-context work. The required control (gains are not explained by context window alone) is demanding.
4. **H4 clustering/VQ is the least likely to win.** The authors expect continuous or random-projection controls may match it, and present "clustering adds interpretability but no control value" as an acceptable negative result.

**Risk register** (09 table, plus 04–07):

| Risk | How it shows up |
|---|---|
| Future-information leakage | Includes the subtle future-action leak: a logged policy action reveals a hidden clue. |
| Representation collapse | Stop-grad alone does not prevent it. |
| Hidden-state aliasing | Same observation, different history branches. |
| Proxy exploitation | Predicted gain rises while real gain falls as search width grows. |
| Goal entanglement | A preference-only goal change alters physical prediction. |
| Rare failure merged away | Codebooks fold catastrophic failures into harmless events. |
| Invalid action semantics | — |
| Policy-mixed successor targets | — |
| Frozen weights, worse system | Behavioural retention degrades even though weights are unchanged. |
| Hidden-cost efficiency | — |
| Goal transfer outside feature span | — |
| Native numerical mismatch | A historical prefix-vs-bulk likelihood discrepancy was recorded for `Rineya-Titan-Experimental-1.0.0-devfreeze`, the 596,049,920-param model. |

The authors single out two shortcuts:
- **Scientific:** "call any useful latent reward predictor a novel world model".
- **Engineering:** "accept a perfect validation loss without proving that the evaluator saw only deployment-available information".

Other identified issues:
- Parallel per-position heads give marginals, not a joint distribution.
- Contrastive objectives can wrongly collapse different successful routes together, and "different action" does not automatically make a negative pair.

**Subject identity (relevant to the user's Titan models).** The bundle requires the "exact Rineya 0.6B-class" checkpoint and rules out substituting stock Qwen3, the older 344M student or a "historical Supa artifact" (`IMPLEMENTATION_HANDOFF_PROMPT.md`). G0 must re-establish identity and hashes and re-qualify prefix/bulk parity. A Titan model from `Rineya_Published_Cores` is therefore usable only if it is pinned and hashed as the subject, and it must pass the prefix-vs-bulk logits and hidden-state parity check, cache on/off, padding/positions and precision before any capture.

---

## (4) Numeric thresholds and budgets

- **Memory, derived:**
  - The 596,049,920-param subject needs ~1.11 GiB for BF16 weights and ~8.88 GiB for full training at 16 B/param, so full fine-tuning is excluded.
  - Caching one 1,024-wide BF16 vector for each of 100k events is ~195 MiB; all 28 layers is ~5.34 GiB.
- **GPU admission guard:** do not admit a pilot expected to exceed **6.5 GiB peak allocated** without a new explicit budget. Reserved memory must be measured. `host_ram_limit_gib: null` and a null max-step limit both block native training.
- **Native start:** 512 tokens, horizon 1, microbatch 1, no backward pass, then increase one dimension at a time. Adapter rank candidates 4/8/16; BF16 first, with quantization as a separate ablation.
- **Sweeps:** horizons {1, 2, 4, 8}; candidate counts {1, 4, 16, 32}; context {512, 2,048, 4,096, 8,192}; latent {32, 64, 128}; K {16, 64, 128}; label budgets {1, 5, 10, 25, 100}%.
- **Statistics:** confidence 0.95; 10,000 bootstrap draws; seeds 17/29/43; ~545 parents at σ=0.25, δ=0.03.
- **Data:** 2,000 / 300 / 500 parents; 70/15/15 hash split.
- **Effort, planning estimates only:**
  - Block 1 (G0): 3–5 sessions
  - Blocks 2–5 (G1–G4): 1–2 weeks each
  - Block 6 (G5/G6): 1–3 weeks, conditional
- **Demo:** 40,325 params, 250 steps, lr 2e-3, wd 1e-4, batch 64, 1,024 train / 256 check samples, horizon 4, ~1.27 s on CPU. The demo model is width 48, 2 layers, 4 heads, latent 24.
- **Validation:** 60 CPU tests passed under Python 3.13.5 / torch 2.10.0+cpu / NumPy 2.3.5. No native, GPU, VQ, self-supervised target encoder, DAG or retention work was run.

---

## (5) The prerequisite-chain prediction failure in the demo

**What the bundle reports** (`validation/demo/demo_result.json`, `KNOWN_LIMITATIONS.md`, report 10):
- Check MSE fell from 0.540924 to 0.002186.
- On the safe plan `take_key → open_door → collect` (`[1,2,3]` from the zero state), predicted cumulative features were [0.985, 1.033, **0.0638**, 0.055, 2.996]. The exact values are [1, 1, **1**, 0, 3].
- Goal scores, with weights avoid_damage = [0,0,10,−20,−1] and prefer_speed = [0,0,10,−1,−2]:

| Goal | Safe [1,2,3] predicted | Safe actual | Shortcut [4,0,0] predicted | Shortcut actual |
|---|---|---|---|---|
| avoid_damage | **−3.46** | **7.00** | −11.50 | −11.20 |
| prefer_speed | −5.41 | 4.0 | 6.64 | 6.6 |

- The run is marked `per_plan_fidelity_qualified: false`.

**My additional analysis** (CPU only, read-only scripts in the scratchpad):

- **Root cause is coverage.** `make_dataset(1024, 4, seed=17)` contains only **2 of 1,024** samples with the exact zero-state `[1,2,3]` prefix; the check set also has 2 of 256.
  - Of 454 `object_gained` events, **418 come from `shortcut` and only 36 from `collect`**.
  - `collect` yields the object in only **4.4%** of its occurrences.
  - Only **4** collect-successes run the full chain from the zero state inside the window.
  - Uniform random 4-step plans make the 3-step conjunctive chain a ~1/125 event per start. MSE is dominated by shortcut events, so the model learns roughly "object ⇔ shortcut".
- **The failure causes maximal decision regret.** I enumerated all 125 three-step plans from the zero state with the bundled checkpoint:
  - Per-feature MAE is tiny at [0.017, 0.026, 0.039, 0.016, 0.015]; object MAE is 0.039.
  - **avoid_damage:** the model picks `[0,0,0]` (wait ×3; predicted −0.51, actual −0.30) over the oracle `[1,2,3]` (actual 7.0). **Regret = 7.3**, essentially the worst possible. `[1,2,3]` is the only damage-free object plan of the 125.
  - **prefer_speed:** it picks `[0,0,4]` and regret is 0.
  - The bundle's own 2-candidate comparison hid this: there both goals still ranked correctly.

**Implication for the research direction.** Near-perfect aggregate fidelity coexists with catastrophic, goal-specific decision failure on the one rare conjunctive prerequisite chain. This is a concrete testbed for the most promising contribution: consequence learning that preserves rare, decision-critical prerequisite distinctions, measured by regret and the candidate-oracle gap rather than MSE. Candidate levers are coverage balancing by transition type, regret- or utility-weighted objectives, and learned latents with grounded decoders. The authors require that any coverage or training fix be logged as a separate experiment, not a silent rewrite of the preserved run.

---

**Files referenced:**
- Gates and config: `configs/evaluation_gates.yaml`, `configs/pilot_sidecar.yaml`, `configs/native_adapter.yaml`, `configs/feature_registry.yaml`
- Specs: `specs/EVALUATION_PROTOCOL.md`, `specs/TRAINING_RECIPE.md`, `specs/NATIVE_8GB_RUNBOOK.md`
- Roadmap: `roadmap/experiment_registry.csv`
- Validation: `validation/demo/demo_result.json`
- Reference code: `reference/rcwm/{synthetic,model,demo,metrics,data,codebook,masks}.py`
- My analysis scripts: `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/{cov.py,regret.py}`