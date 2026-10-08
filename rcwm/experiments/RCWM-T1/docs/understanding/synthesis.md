# RCWM × Titan research map (synthesis of 6 reader reports, 2026-10-05)

Spot-checks I ran myself (read-only):
- All key asset paths below exist.
- Host free RAM is **4.28 GiB**. GPU has **6.5 GB free**, and no `rineya-gpu-0.lock` is held.
- Forge `metadata/train.jsonl.gz` holds **2,000 worlds for each of 8 families**.
- `T_BANK_V5` is unread but **protected** (details in §6).

---

## 1. Core idea of RCWM

1. RCWM (Rineya Consequence World Model) splits "what will happen" from "how much do I want it".
2. A goal-agnostic **consequence model φ** maps a deployment-available history h and a precommitted candidate action prefix a₁..a_H to a vector of consequence features. There are two versions:
   - **B3:** measured, named events such as key_gained, damage and cost.
   - **B4:** a learned latent z, trained against a stop-gradient target encoder with exactly one anti-collapse mechanism.
3. A **goal is a linear readout w_g** over those features, U_g = w_g·ψ, where ψ = Σγᵗφ_t is a successor-feature-style cumulative consequence vector. A new goal therefore costs one weight vector, fit on a small calibration set, rather than a new evaluator.
4. The claim only counts against a **matched scalar trajectory Transformer (B2)**. B2 must have the same context, parameters, labels, tuning budget, hashed candidate set and inference budget. The primary contrast is **B4 − B2**, measured on actual independent-outcome utility and candidate regret.
5. Generalisation is tested on **held-out parent families, held-out goals (in-span and out-of-span) and held-out skill compositions**. Statistics use a parent-level paired bootstrap. The Rineya 0.6B backbone stays frozen and plans must be `open_loop_precommitted`.
6. The bundle's own demo shows why the metric must be decision regret rather than MSE:
   - Aggregate MSE was 0.0022, yet object_gained was predicted at 0.064 against a true 1.0 on the rare take_key→open_door→collect chain.
   - Ranking all 125 three-step plans gives regret 7.3 (the maximum) under avoid_damage.
   - Goal weights d∈(1.8, 11.66) are all mis-ranked. The two goals the demo reported both sit outside that band.
7. Promotion is blocked until the null fields in `evaluation_gates.yaml` are preregistered. A null result counts as informative.

---

## 2. Assets reusable right now

**Models.** All four live under `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/`. Every one is Qwen3ForCausalLM: 28 layers, hidden 1024, 596,049,920 params, tied embeddings, no `auto_map`.

| Dir | Stored as | Role |
|---|---|---|
| `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` | FP32, sha `a28629c9…` | Best transition skill and preservation profile. **Primary encoder.** |
| `Rineya-Titan-Transition-1.0.0-devfreeze` | FP32, sha `a4d73353…` | Sparse + 0.25·P16(δ_H). Transition skill plus boundary abstention, fresh-bank confirmed. |
| `Rineya-Titan-Experimental-1.0.0-devfreeze` | BF16, sha `53720167…` | BF16-rounded δ_T. This is the subject the RCWM bundle names. |
| `Rineya-Supa-1.7.0-devfreeze` | BF16, sha `962abc5b…` | Base for every delta. The no-skill control. |

Deltas that can be recovered locally:
- dSparse = Sparse − FP32(Supa).
- BF16-rounded δ_T = Exp − Supa.
- 0.25·P16(δ_H) ≈ Transition − Sparse.

The full FP32 δ_T for seeds 17/23/29 and δ_H (≈1.26 GB each) are only in the HF vault: `Luncy1978/rineya-published-cores`, revision `33b5f9a4…`, `research-records/delta-weight-screens-20260918/_trained_tensors/`. The files contain trained tensors, so δ = file − Supa. Delta builders:
- `rineya-dataset-retention-records/sparse001-20260917/build_sparse.py`
- `indep00{1..5}-20260921/build_indep.py` (defines P16)
- `int001-20260917/build_arm.py`

**Loading recipe.** Use this exactly.
- Venv: `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe` (torch 2.6.0+cu124, transformers 5.14.1). This is the receipt runtime and has no sklearn.
- For probes needing sklearn, use `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe` (torch 2.12.1, sklearn 1.9.0, peft 0.19.1).

```python
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision("highest"); torch.cuda.set_per_process_memory_fraction(0.78,0)
model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, local_files_only=True,
          low_cpu_mem_usage=True, device_map={"":"cuda"}, attn_implementation="sdpa").eval()
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
gen: do_sample=False, eos_token_id=[151645,151643], pad_token_id=151643
hidden: model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)
```

Loading rules:
- Always pass `dtype=torch.float32`. The v5 default of "auto" would leave Experimental in BF16.
- Never cast Sparse or Transition to BF16.
- Load one model per process.
- Require psutil available RAM ≥ 3.0 GiB before starting.
- Take the GPU lease: `ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/gpu_lease.py`. Another Rineya session (the synth-centralized observer) is alive.

**Task generators, oracles and verifiers.**

*Forge RL v2* (stdlib only), under `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/rineya_reasoning_forge_rl_v2/`:
- `forge_rl/tasks.py`: `make_world`, `ledger_step`, `register_step`, `plan` (Dijkstra), `applicable`, `apply_action`, `solve`.
- `forge_rl/oracle.py`: `replay_transition`, `verify_solution`, `verify_trace`.
- `forge_rl/environment.py`: `PlanningEnvironment`, reset/step, prerequisite_plan only, never run with a model.
- `metadata/{train,validation,test_iid,test_length,test_transfer}.jsonl.gz`: full worlds plus `trace_steps`.
  - Train trace length (min/median/max): ledger 13/28/128, register 12/24/96, prerequisite_plan 0/1/9.
  - Other families: temporal_route, deadline_schedule, budget_selection, evidence_chain, continuous_replan.
- `process/train_transitions.jsonl.gz`: 18,376 valid/invalid `proposed_after` pairs, ready-made verifier data.

*FDS lineage*, under `C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/` (duplicated in `ChatBubble-titan-model-soup-mini-v1`):
- `oracles.py` (`ORACLES[fam](u)`, validated 384/384).
- `generate.py` (`SYSTEM` prompts verbatim, `harvest_exclusions`, `operative_key`, `content_keys`).
- `generate_v2.py` (`GENS`, empirical marginals).
- `score_arms.py` (strict scorer, byte-tested render).

*Exclusion logic*, so fresh data avoids Titan's training and scored parents: `ChatBubble-ced-native-loader-qualification/experiments/reasoning_posttraining_v1/process_arm.py` (`verified_exclusions`), plus `harvest_exclusions()`.

*Other pieces:*
- Multi-step format precedent: `ChatBubble-titan-tangent-task-arithmetic-v1/experiments/delta_multi_reasoning_v1/scout/build_scout.py`.
- Shortcut floors and probes: `rineya-dataset-retention-records/select-views-001/{FLOORS,FAMILY_DECOMPOSITION,STRUCTURAL_FLOOR_PROBE}.json`.
- Abstention scorer: `rineya-dataset-retention-records/hskill-views-002/boundary_rule.py` (`abstains`, `ABSTAIN_PATTERNS`).

**RCWM reference.** Under `…/Rineya_Consequence_World_Model_Bundle_2026-10-05/reference/rcwm/`. The 60 tests pass in hermes.
- Readout: `returns.discounted_features`, `linear_utility`.
- Metrics: `candidate_regret`, `paired_parent_bootstrap`, `brier_score`.
- Data: `split_for_parent` (sha256, 70/15/15), `validate_open_loop`.
- Masks: `causal_mask`, `prefix_plan_mask`, `available_observation_mask`. These use True = **blocked**, the opposite of the HF convention.
- Diagnostics: `TrainOnlyKMeans`; `FeatureTrajectoryTransformer` as a sidecar template (set `state_dim`=1024).
- Data contract: `tools/validate_resources.py` and `schemas/*.json`.
- Not built: a `FrozenHistoryEncoder` implementation, a loader, a real masked/BCE loss, B2, an evaluation loop, and B4 learned targets.

**BWM-001 reuse.** Under `C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/src/rineya_bwm/`:
- `analysis.py`: Wilson interval, paired cluster bootstrap, factorial interaction, sample size.
- `core.py`: `EmpiricalForward` (rejects unsupported transitions), `select_plan` (MPC), `luck_counterexample`.
- `tokenization.py`: `encode_target_only`.
- `scripts/native_adapter.py`: q/v LoRA scaffold. Its default 7.4 GiB RAM guard blocks here.

**Instrument status.**
- Spent: SELECT-VIEWS-001 (selection-only), T_BANK_V4, HSKILL_SOUP_V3, and the LC /256 suite.
- **Never read** (protected by other programmes): T_BANK_V5, C_FRESH, H_FRESH, HSKILL_CONFIRM_V2.
- Every new result needs fresh, seeded, collision-checked banks with shortcut floors published before any model call.

---

## 3. Rineya discoveries a new experiment should exploit

1. **Frozen Rineya cannot do one-step consequences; 384 verified rows install it.**
   - Supa scores 14/384 (0/128 on both prerequisite and register), below the copy floor.
   - Titan-Exp scores 234/384 and Sparse 232/384. RRP-007, with 1,536 rows, reaches 289/384 on all 3 seeds.
   - On fresh T_BANK_V4, Sparse scores 430/864 (L/P/R = 136/126/168 of 288) against BASE 5.
   - The skill is not echoing: 251/346 correct on state-changing rows. It also transfers (time_duration 10→23).
   - **Exploit:** Supa vs Titan is a clean, minimal-delta contrast with the skill absent vs present, made by a 0.056%-norm update. That is ideal for attributing consequence representations to a known weight change.
2. **Multi-step end-to-end is at floor.**
   - DMRF 2-event scout: Supa 1/32, Sparse 1/32. 3 events: 0/6. "3 of 4 balances often right."
   - Puzzles: 102/450, a "capacity ceiling" that survived 11.5× data.
   - **Exploit:** this is exactly where an externally composed consequence model (RCWM rollouts, successor features or decoded latents) could add capability. The partial credit hints that latent knowledge may exceed generative output.
3. **The skill lives in a heavy-tailed head of δ.**
   - The top 25% of elements carry 83.6% of the norm.
   - Pruning to k=0.5 keeps the skill (235); scaling to α=0.5 dilutes it (219).
   - Pruning repairs retention in 3/3 draws (LC +0/+7/+15). LC peaks at 163–164 in the interior.
   - Magnitude, not rank, is the operative variable.
   - **Exploit:** representational dose-response curves along k vs α are a mechanistic test no one has run.
4. **Task arithmetic composes skills by allocation, not interference.**
   - 1.0·T + 0.25·H keeps T (434 vs 430) and adds +67 boundary rows (p = 1.4e-13). The coefficients are separable (interaction ≤ 3 rows).
   - cos(δ_T, δ_H) = 0.0027. Same-skill seeds share a subspace (0.60 against a null of 0.089).
   - Usable δ_H band is [0.10, 0.50].
   - **Exploit:** build β-arms locally and test whether the conversational abstention skill transfers to world-model abstention.
5. **The abstention delta is real but brittle.**
   - High-dose LC "damage" is entirely abstention collapse (UNKNOWN 13→0, answerable rows flat).
   - Full-strength δ_H imports formatting: 342/864 answers gained a trailing period.
   - UNKNOWN-token mass has AUC ≈ 0.8. Generic entropy and margin signals fail ("confidently wrong").
   - **Exploit:** OOD rejection coverage, which RCWM lists as a secondary metric.
6. **Quantization is a hidden variable.**
   - BF16 export zeroes 60.6% of δ_T elements and 24/220 tensors. It behaves like α-shrinkage but is not a magnitude mask (Jaccard 0.615).
   - Titan-Exp's LC 162 is a BF16 artifact; its FP32 parent scored 157.
   - **Exploit:** keep every arm in FP32. Titan-Exp serves as a "rounded-δ" arm.
7. **Shortcut warning.**
   - prerequisite_plan's preconditions are dead in SELECT-VIEWS: `ignore_preconditions` scores 128/128.
   - Ledger `correct_kind_ignore_status` scores 117/128.
   - About 43% of the acquisition gain comes from the dead-constraint family.
   - **Exploit:** this is the LLM analogue of RCWM's rare-prerequisite failure, and a ready-made testbed for it.
8. **Retention numbers are draw-unstable** (LC span 13 across seeds). Acquisition is stable (span 7). Avoid single-draw retention claims and prefer paired designs over fixed models.

---

## 4. Gaps: planned by RCWM/BWM, never executed

Ranked by (a) feasibility on an 8 GB laptop within hours and (b) potential for a substantial, publishable result.

| Rank | Gap (registry id) | (a) Feasibility | (b) Publishability | Note |
|---|---|---|---|---|
| 1 | **G2/G3-04/G3-01 on a frozen Rineya encoder**: B3/B4 vs matched B2 with held-out goals and label-efficiency curves | High: cached FP32 hidden states plus tiny heads | High: the bundle's primary claim, never run on any LLM | Needs fresh environment banks |
| 2 | **Representation-level G0 + consequence probing** (capture at action boundaries; does δ_T write decodable k-step consequences?) | High: forward passes only | High: new mechanistic link between task-vector findings and world-model representation | Prefix/bulk parity must be qualified first |
| 3 | **Decision-aware fidelity / rare prerequisites** (risks 04/09) | High | High: replicates the demo failure in an LLM world model, plus a repair | Titan's precondition shortcut is the natural instrument |
| 4 | **G4-01 direct successor vs rollouts (B6)** and horizon sweep 1/2/4/8 | High: inference only, about 1–2 h per model | Medium–High: tests whether the "capacity ceiling" is an interface problem | Errors compound at per-step p ≈ 0.45–0.6 |
| 5 | **OOD rejection coverage** via the boundary delta | High: about 30 min | Medium: novel cross-skill transfer | Amended prompts add a confound |
| 6 | G4-02 long-context dependency sweep | Medium: 4–8K FP32 (KV 0.9 GiB at 4K; use `logits_to_keep`) | Medium | Must control for "bigger context alone" |
| 7 | G3-02/03 k-means/VQ (B5) | High | Low: the authors expect a null | Run as a diagnostic on existing latents |
| 8 | G5 native typed-slot LoRA with retention non-inferiority; BWM 2×2 (A11 − A01) | Low: RAM-gated (MiMo E0 blocked), CPU AdamW 4 GiB RSS | High | Defer until 1–3 give a signal |
| 9 | G4-03 DAG / parallel; G6 distillation / imagined RL; G7 | Low | Medium | Deferred |

---

## 5. Candidate experiments

Shared protocol for all five:
- FP32, greedy decoding, `enable_thinking=False`, verbatim Forge/FDS system prompts unless an amendment is the variable under test.
- Fresh seeded banks with `harvest_exclusions` and operative-key collision checks against all historical payloads and the Forge parents used in training.
- `split_for_parent` 70/15/15. Seeds 17/29/43 for any trained head.
- `paired_parent_bootstrap` with 10,000 draws. Holm correction across the preregistered primary contrasts.
- Floors computed and published before any model call: copy_before, constant_majority, ignore_preconditions, correct_kind_ignore_status.
- **G0 gate before any capture**, about 30 min:
  - Compare FP32 hidden states at action boundaries between a bulk full-sequence forward and a prefix-only forward on 64 prompts. Tolerance: max |Δ| < 1e-3.
  - Use one consistent path, with no KV-cache mixing and no padding, for all captures.

### E1: Latent consequence probes across the Titan delta family (gap 2, foundation)

**Hypothesis.** The 384-row δ_T writes a linearly decodable consequence representation into the residual stream. It extends beyond the trained horizon (k ≥ 2) even though k-event generation is at floor, and it is carried by the large-magnitude head of δ.

**Design.**
- Fresh chains for prerequisite (8-bit mask; features are the 8 bits plus applicability), ledger (features: sign, argmax account, balance > t, and a regression on each balance), and register (features: r > 48, parity, regression).
- Prompts follow the DMRF `{"before","events":[…]}` format with k ∈ {1, 2, 4, 8}.
- Cache last-token hidden states at layers {4, 8, 12, 16, 20, 24, 28} (FP16 on disk).
- Linear and logistic probes are fit on TRAIN, with the layer chosen on DEV.
- Transfer tests: train the probe on k ≤ 2 and test on k = 4; also test on held-out operation-kind combinations.

**Arms.** All are paired on the same chains.
- Supa (no skill).
- Titan-Exp (BF16-rounded δ_T).
- Titan-Sparse.
- Sparse-delta ×{0.5, 1.5} (an α-line on the sparse head).
- A random-direction delta, per-tensor norm-matched to dSparse (null).
- Titan-Transition.
- Optional: dense FP32 δ_T from the vault, for a k vs α comparison.

**Matched scalar baseline.** A same-capacity probe predicting only the scalar goal utility for the training goals, on the same hidden states.

**Controls.**
- Layer-0 / embedding probe (surface decodability).
- Hewitt–Liang control task (shuffled label mapping), to report selectivity.
- copy_before feature baseline.
- Titan's own generative exact and per-feature accuracy at the same k.

**Metric.** Probe balanced accuracy and per-feature R² at k; selectivity; the latent-minus-generative gap.

**Compute.**
- 3 families × 1,500 chains × 4 k = 18k forwards per model, at about 50 ms each, so about 15–20 min per model. For 7 arms, about 2.5 GPU-h.
- Cache about 260 MB per model. Probes take seconds.
- Generative baselines: about 4k generations for Sparse and Supa at about 0.45 s each, so about 1 h.

**Significant result.**
- Titan-Sparse − Supa probe accuracy at k ≥ 2 is ≥ +10 pp, with the parent-bootstrap lower bound > 0 and selectivity above control, **and** latent accuracy exceeds generative accuracy by ≥ 20 pp at k = 2–4.
- A dose-response that follows the magnitude head (Sparse×0.5 ≥ Exp ≈ α-shrunk) would mirror SPARSE-001 vs INT-001 at the representation level.
- A clean null is publishable too: "transition SFT changes the readout, not the world representation."

### E2: The RCWM primary contrast on a frozen Titan encoder (gap 1)

**Hypothesis (H1/H2).** Goal-agnostic consequence heads on frozen Titan states, plus a linear w_g readout, beat a matched goal-conditioned scalar evaluator at goal-conditioned plan selection. This holds for held-out goals and held-out compositions, and with fewer grounding labels.

**Design.**
- Prerequisite world (primary) and ledger (secondary). Each parent is a fresh start state with **K = 8 precommitted H = 4 plans**, hashed CandidateSet recorded before scoring.
- φ:
  - prerequisite: final bits, plus cost, plus the number of no-op steps;
  - ledger: final balances (normalised), sign bits, negative count, number of posted events.
- Goals are weights over φ:
  - TRAIN goals span subspace S;
  - test uses in-span new mixtures **and** out-of-span directions;
  - held-out compositions are operation-kind pairs never seen in training plans.
- Encoders: Titan-Sparse (primary), with Supa and Titan-Exp as attributions. Cache states for state+plan text at the best E1 layer.

**Arms.**

| Arm | What it is |
|---|---|
| O1 | Exact oracle (ceiling) |
| B0 | Random / heuristic |
| B1 | Titan direct: prompt with state, plans and goal text; choose an index |
| B2 (matched) | MLP([h; w_g]) → U, trained on (parent, candidate, train-goal) utilities. Same parameter count (±5%), labeled parents, learning-rate sweep {1e-4, 3e-4, 1e-3} × 3 seeds, and candidate set as B3/B4 |
| B3 | Head h → φ̂ (BCE for events, Huber for continuous), U = w_g·φ̂ |
| B4 | Predict z = stop_grad(Titan's own last-token state on the **true future state text**) from h. A frozen target encoder means no collapse risk. Then a linear z → φ readout fit on a separate calibration split |
| B4-roll | Chained Titan one-step generation, with exact feature extraction from predicted states |

**Label curve.** Nested 1/5/10/25/100% of TRAIN parents, identical across arms. For out-of-span goals, B2 also gets n ∈ {0, 8, 32, 128} labels per new goal.

**Data.** 2,000 / 300 / 600 parents. 600 exceeds the ≈545 needed for δ = 0.03 at σ = 0.25.

**Metrics.**
- Primary: mean candidate regret (normalised) on TEST, B3/B4 − B2.
- Secondary: success, oracle gap, the label count needed to reach a preregistered regret target, Brier score per feature, and a search-width sweep 1/4/8 (predicted vs real utility).

**Compute.**
- About 23k encoder forwards plus 23k target encodings, so about 40 min per encoder.
- Heads take minutes.
- B1: about 900 generations, about 8 min.
- B4-roll: about 19k one-step generations. Batched by 16, that is about 30–45 min.
- Total about 3–4 GPU-h for the primary encoder.

**Significant result.**
- B3 or B4 − B2 regret difference has a CI excluding 0 on **in-span held-out goals and held-out compositions**. Out-of-span alone is a construction win, as the bundle itself notes.
- B4/B3 reaches the regret target with ≤ 25% of B2's labels.
- Titan-encoder gains exceed Supa-encoder gains, which attributes the effect to the transition skill.
- B4 (Titan-latent target) matching or beating B3 would be the first evidence that a frozen LLM's own state encoding is a usable consequence target.

### E3: Precondition-critical decisions — Titan's shortcut as an LLM instance of the demo failure (gap 3)

**Hypothesis.** Titan's prerequisite skill implements `ignore_preconditions`. On the rare parents where applicability decides the outcome, a Titan-based consequence model incurs large regret despite good average one-step accuracy, exactly like the RCWM demo. A decomposed or rare-weighted model repairs it.

**Design.**
- Fresh prerequisite parents. 70% are natural (all actions applicable) and 30% are **precondition-critical**: the goal is reachable only through a precondition-respecting order, and a "shortcut" plan looks optimal if preconditions are ignored. This is the KeyDoor structure lifted to 8 bits.
- The prompt is amended: "if not applicable, return the state unchanged." Also run the verbatim prompt on applicable rows to measure prompt sensitivity.

**Arms.**
- B4-roll with Titan-Sparse.
- R1, decomposed: Titan answers "applicable? yes/no", then applies (two calls).
- R2: E2's B3 head with applicability as an explicit feature and rare-event reweighting, vs plain-loss B3.
- R3: exact applicability feature (ceiling).
- B2 matched scalar.
- O1.
- Supa.

**Metrics.**
- One-step accuracy split by applicable / inapplicable.
- The fraction of inapplicable rows where the output equals the ignore-preconditions answer.
- Regret on critical vs natural parents.
- The goal-weight band of mis-ranking (the demo's d-sweep).

**Compute.** About 600 parents × 8 candidates × ≤ 4 steps, about 1–1.5 GPU-h batched. Heads take minutes.

**Significant result.**
- Critical-parent regret is ≥ 2× natural-parent regret, with a CI, while natural one-step accuracy stays ≥ 0.7.
- The ignore-preconditions match rate on inapplicable rows is ≥ 50%.
- R1 or R2 cuts critical regret with CI > 0 and no natural-parent loss.

That would be a crisp, general claim: "verified-transition SFT installs a shortcut world model whose failures are invisible to accuracy and visible only to decision regret."

### E4: Factorised rollouts vs direct successor vs end-to-end over horizon (gap 4)

**Hypothesis.** The multi-step "capacity ceiling" is an interface problem. Titan's one-step skill, composed externally, or a direct successor probe, recovers k-step consequences and good decisions well above end-to-end generation.

**Design.**
- Fresh ledger and register chains (and prerequisite), H ∈ {1, 2, 4, 8}, about 400 chains per family.
- Report whether per-step errors compound independently (pᵏ) or are correlated.

**Arms.**
- (a) End-to-end k-event prompt (DMRF format).
- (b) Chained one-step calls with the verbatim prompt.
- (c) Chained with deterministic canonicalisation (JSON re-dump, mod 97).
- (d) Direct successor probe from E1 at layer L.
- (e) Supa chained.
- (f) Oracle.

**Metrics.** Exact final state, per-feature accuracy, and plan regret using E2's candidate sets.

**Compute.** About 400 × 15 steps × 3 families, about 18k generations. Batched, about 1–1.5 GPU-h per model.

**Significant result.**
- Chained exact accuracy at k = 4 is ≥ 5× end-to-end (CI excluding 0).
- Chained or successor regret is within a preregistered margin of the oracle even where state fidelity is low. Decisions survive imperfect fidelity, the RCWM thesis applied to an LLM.

### E5: The boundary delta as world-model abstention / OOD rejection (gap 5)

**Hypothesis.** δ_H, a conversational clarification and abstention skill, transfers to refusing to predict consequences that cannot be simulated, which gives RCWM OOD rejection coverage without new training.

**Design.**
- 600 fresh rows, 50% ill-posed: unknown account or register, missing amount, undefined kind, status outside the rules. The other 50% are well-posed.
- Two prompt conditions: verbatim, and verbatim plus "answer UNKNOWN if undeterminable".

**Arms.** Sparse + β·P16(δ_H) for β ∈ {0, 0.10, 0.25 (= Titan-Transition), 0.50}, built as Sparse + (β/0.25)·(Transition − Sparse) in FP32, plus Supa.

**Metrics.**
- Abstention on ill-posed rows (via `boundary_rule.abstains`, plus literal UNKNOWN).
- False abstention on well-posed rows.
- Transition exact accuracy (T retention).
- Selective plan regret when abstained candidates are discarded.

**Compute.** About 3,000 generations, about 25 min, plus about 5 min of arm builds.

**Significant result.**
- Titan-Transition − Sparse gives ≥ +15 pp abstention on ill-posed rows (McNemar p < 0.01) with ≤ 2 pp false abstention and no T loss.
- A monotone β dose-response.

**Suggested order:**
1. G0 parity (30 min).
2. E1 pilot on prerequisite and ledger (2–3 h).
3. E2 (core).
4. E3 and E4, which share E2's banks and caches.
5. E5 whenever spare GPU time exists.

Fix the primary endpoint, minimum effect (e.g. 0.03 normalised regret), parent count and cost ratio in a PREREG before opening TEST, so the `evaluation_gates.yaml` nulls are filled.

---

## 6. Contradictions and uncertainties across reports

1. **Free host RAM.**
   - Four readers say about 2 GB, rcwm-plan says 4.2 GB, and runtime-env saw 3.82–8.26 GiB swinging within minutes. I measured 4.28 GiB.
   - RAM is volatile because of Unity ×2, browsers and another agent session. Re-check before each launch.
   - The 3.0 GiB probe floor is met; the 3.6 GiB train floor is marginal.
2. **Which venv.**
   - titan-evidence calls Chat-Bubble/.venv the "closest local env". runtime-env recommends hermes (torch 2.6.0, which matches 55+ receipt JSONs) and Chat-Bubble/.venv only when sklearn is needed. The rcwm-code tests ran in hermes.
   - Resolution: hermes for anything compared with CONFIRM-002 numbers; otherwise pick one and record it.
   - No CUDA venv has transformers 4.51.3 except the py3.10 spikingbrain venv, so Lineage-A numbers (SELECT-VIEWS 232/234, LC 161–164) are **not reproducible bit-exactly** here.
3. **Subject identity.**
   - The RCWM bundle demands the "exact Rineya 0.6B-class" subject, names the 596,049,920-param Titan-Experimental, and records a prefix-vs-bulk discrepancy for it.
   - The strongest models are Sparse and Transition (FP32).
   - The MiMo E0 capture failure (logprob error 0.0414 > 0.03125; hidden-state max diff 1.25) happened on a BF16 path. **Whether parity holds in FP32 is unknown.** G0 must decide, and the choice of subject must be pinned and hashed.
4. **Per-family one-step skill depends on the instrument.**
   - Titan prerequisite accuracy is 0.75 on SELECT-VIEWS-001 but 0.44 (126/288) on T_BANK_V4. Ledger is 0.41 vs 0.47; register 0.65 vs 0.58.
   - The two differ in generator (Forge raises on inapplicable actions; FDS v2 is 100% applicable and resamples marginals), harness (DenseReference with an EOS list and 96 tokens vs HF generate with a single EOS and 48 tokens), and transformers version.
   - The prerequisite drop is consistent with the dead-precondition shortcut, but this is unverified. E3's applicable/inapplicable split should resolve it.
5. **Prompt sensitivity.** Titan was trained on exact system prompts, and C5 showed formatting can be imported wholesale. E3 and E5 need amended prompts, and the k-event format of E1 and E4 is untrained. A drop there may be format failure, not missing knowledge. Always include verbatim-prompt one-step controls.
6. **T_BANK_V5 status.** transition-harness says it "appears unread". I confirmed it is unread (`RESULTS_STAGE_A.md`), but `triskill_geo001_parentq_v1/PREREG_PARENTQ.md` lists it under "Never read: any protected bank", together with C_FRESH, H_FRESH and HSKILL_CONFIRM_V2. **Do not use it.** Generate fresh banks instead.
7. **Forge trace statistics.**
   - transition-harness quotes validation lengths with prerequisite at most 5 and describes three families. Train metadata has 8 families × 2,000 worlds and prerequisite traces up to 9 steps.
   - Some Forge TRAIN parents fed Titan's training (384 rows; RRP-007 used 1,536). Building chains from them risks contamination, so prefer freshly generated worlds.
8. **The arithmetic "/47 retained" metric.** BWM-lineage summaries treat it as retention, but `PANEL_IS_TRANSFER_NOT_RETENTION` says the panel measures transfer. Do not use it as a preservation endpoint for new arms.
9. **Titan-Transition.** It is called "promoted" and "the only fresh-bank-confirmed two-skill model". Its LC, arithmetic and retained-47 were never measured, its answerable correctness is descriptively lower (35 vs Sparse 37 vs Base 41), and C3/C5 failed. "Strongest" is endpoint-specific.
10. **The RCWM demo's "check" set.** rcwm-plan treats check MSE as validation. rcwm-code found 41% of check pairs duplicate train pairs and only 7 reachable states, so the demo's fidelity number is not held-out. That makes the bundle's own success figure weaker than it reads, while its failure analysis remains valid.
11. **Delta recovery precision.** Transition − Sparse in FP32 recovers 0.25·P16(δ_H) only up to FP32 rounding, because it was summed in float64. Exact dense FP32 δ_T is only in the HF vault (≈1.26 GB per file, about 2.5 GB of disk headroom needed per pair). The vault revision should be pinned before any k-vs-α arm uses it.
12. **The "capacity ceiling" claim.** It rests on end-to-end exact match. The DMRF partial-credit observation ("3 of 4 balances often right") suggests near-misses, so per-feature metrics in E1 and E4 may contradict the ceiling framing. That is a hypothesis to test, not an established fact.