# RCWM-T1 pilot results (exploratory; living document)

All numbers come from the **exploratory** pilot bank (`data/bank_pilot.json`). The test split has
28 P, 26 L and 26 R parents, with 8 candidate plans each. CIs are 95% parent-level paired
bootstraps and do **not** include layer/λ selection noise. Nothing here is confirmatory, and the
sealed confirmation bank is untouched.

**Revision 2 (2026-10-06 ~13:45):** this revision follows an adversarial verification
(`docs/verification/RULING.md`: code audit, independent sklearn recomputation, alternatives).
- No alignment, layer, split or target bug affects the numbers.
- Two interpretations in revision 1 were wrong and are corrected below (R3, R5).
- The revision-1 R5 table could not be reproduced and is replaced.

## R1. G0 capture qualification passes in FP32

- Prefix-only and bulk forward states at action boundaries: max |Δh| = 1.2e-4 absolute (7.7e-7 relative to the row max).
- Batched vs single forward: 3.9e-4 relative, i.e. fp16 storage rounding.
- A CPU FP32 recomputation of stored rows matches within 4.5e-4 relative at every layer. Each stored row best-matches its own `hidden_states[L]` among all 29 layers.

## R2. A linear change-mask readout decays quickly with horizon

The readout is a ridge probe at the answer slot; the metric is exact change-mask match on TEST.

| Family | k=1 | k=2 | k=4 | Touched-entity rule (k1/k2/k4) |
|---|---|---|---|---|
| L | 0.875 | 0.500 | 0.221 | 0.755 / 0.591 / 0.433 |
| R | 0.433 | 0.091 | 0.010 | — |
| P | copy floor | copy floor | copy floor | — |

Event-boundary tracking reproduces the same curve.

Three things limit this result:
- **The decay is not purely about composition.** A linear readout of *perfect symbolic inputs* decays in parallel: L 0.70 / 0.35 / 0.26. Decoding event 1's own mask also fades as more events are added: 0.82 at k=2, 0.64 at k=4.
- **The decay mixes three causes:** probe limits, interference, and missing composition.
- **P is uninformative here.** The LM head does read P at k=1 (R3), so the probe failure is not evidence that P is absent.

"Absence of latent composition" stays unsupported until multi-step generation and likelihood ranking are scored for Sparse.

## R3. The change-mask probe does not separate Sparse from Supa, but better instruments do (H2b NOT supported)

This is the trained one-step frame, ledger.

| Probe | Sparse | Supa | Diff (95% CI) |
|---|---|---|---|
| Ridge | 0.846 | 0.812 | +0.034 [−0.005, +0.072] |
| MLP | 0.913 | 0.918 | −0.005 [−0.024, +0.014] |
| Logistic | 0.885 | 0.894 | — |

Equivalence is not established: the CIs allow differences of 0.07–0.09. For register the result
depends on which probe is used (ridge +0.014, n.s.; logistic +0.082 [+0.019, +0.144]).

This probe misses the trained skill. On the same Sparse and Supa states (verifier "alternatives"):
- **P, LM head:** on state-changing items, the first digit of the answer is correct 0.84 of the time for Sparse vs 0.60 for Supa. DEV gives 0.96 vs 0.47.
- **R, swapped-in value:** Sparse encodes it better (R² 0.85 vs ≤ 0.67).

So the delta does add consequence content that the change-mask probe cannot see. The revision-1
claim that "Supa encodes the same one-step consequence" is withdrawn.

**Generation.** Supa's generated exact state is near zero wherever the state changes. Under lenient parsing it scores L 0.03–0.06, P 0.00, R 0.04–0.05. Its strict-parse failures on P and R were an instrument artifact:
- Supa wraps its answers as `{"before":…,"after":…}`.
- P answers are capped at 12 new tokens.

That cap and the parser are fixed for future generation (I-09).

## R4. Ledger change-mask decodability is not evidence of model computation

The control rules are base-rate-matched counterfactuals: ledger account references rotated A→B→C→D→A.

| Arm | Ridge selectivity on 180 discriminating test items (26 parents) |
|---|---|
| Sparse | −0.022 [−0.055, +0.014] |
| Supa | 0.000 [−0.032, +0.028] |

- Logistic and k=1 variants range from −0.028 to +0.039; none is significant.
- Input-only learners find the counterfactual slightly easier (+0.03 to +0.05).
- The test rules out an explicit linear encoding of the full true mask. It cannot rule out nonlinear or partial computation.

So ledger decodability does not show the model computed the consequence, and missing selectivity does not show it did not.

## R5. Where the delta moves the residual stream (rewritten)

Layers 0–8 are bit-identical across arms; the delta touches only blocks 8–27. The measurement
uses TEST one-frame items (n = 640) with Supa's top-5 massive dims excluded. Relative L2 shift:

| L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|
| 0.248 | 0.586 | 0.826 | 1.087 | 0.711 |

Most of this is a **constant per-family offset**. That offset accounts for:
- 93–99.6% of the shift energy at L12–20
- 70–84% at L24–28

The offset switches P's first output token from `{"` to a digit. It carries no P content; the
item-specific remainder does.

- **Pooled CKA (0.91–0.99) only reflects that between-family structure is preserved.** Within a family it is lower: ledger 0.79 at L20 and 0.60 at L24. With family means removed it is 0.44–0.86.
- **Sparse is still close to a linear function of Supa:** held-out R² is 0.88–0.99.
- **Cross-arm probe transfer fails, but that is not diagnostic.** Probes also fail across prompt frames within a single arm.

(Bug B1, where the shift metrics pooled families, is fixed in `scripts/mechanism_report.py`.)

## R6. Generation: Titan's one-step skill replicates; chained self-rollout beats end-to-end on L and R

Exact final state on TEST, lenient parse.

| Family | Titan-Sparse e2e k1 / k2 / k4 | Titan chained (chain_one) k2 / k4 | Supa e2e k1 |
|---|---|---|---|
| L | 0.389 / 0.072 / 0.000 | 0.202 / 0.024 | 0.106 |
| R | 0.611 / 0.091 / 0.005 | 0.361 / 0.087 | 0.048 |
| P | 0.196 (0.460 on changed items) / 0.094 / 0.143 | 0.049 / 0.000 | 0.054 (0.000 on changed items) |

- **Chaining helps L and R.** Feeding Titan's own one-step answers back step by step is 2.8–17× better than end-to-end on those families.
- **e2e collapses faster than independent composition would predict** (L k2 0.07 vs 0.39² ≈ 0.15).
- **P is the exception.** Titan applies inapplicable actions even when the multi-step prompt states the no-op rule (H4 shortcut), so chaining compounds the error.

## R7. Decisions (H3; RCWM primary): where a consequence model helps

Setup:
- Normalised regret over 256 held-out goals among the 8 candidate plans of each TEST parent (0 = oracle, ≈1 = random). n = 26–28 parents per family.
- Heads use frozen Titan-Sparse answer-slot states, with the layer chosen on DEV.
- Contrasts are parent-paired bootstraps.

| | L k1 | L k4 | P k1 | P k4 | R k1 | R k4 |
|---|---|---|---|---|---|---|
| B3 consequence head (MLP), full budget | 0.244 | 0.813 | 0.479 | 0.742 | 0.292 | 0.965 |
| B2 scalar, bilinear, in-span goals | 0.286 | 0.826 | 0.546 | 0.734 | 0.319 | 0.966 |
| B2 scalar, bilinear, out-of-span goals | 0.820 | 0.945 | 0.716 | 0.828 | 0.537 | 0.977 |
| S no-LLM learner (same labels) | 0.285 | 0.746 | 0.149 | 0.424 | 0.301 | 0.841 |
| G Titan chained rollouts → φ → w·φ | 0.355 | **0.507** | 0.505 | 0.882 | **0.133** | **0.388** |
| G Titan e2e | 0.321 | 0.842 | 0.538 | 0.886 | 0.129 | 0.818 |
| Rule-ablated simulator (diagnostic) | 0.060 | 0.099 | 0.085 | 0.144 | — | — |

**1. The RCWM factorisation wins on new goals, not on in-span goals.** B3 − B2 on out-of-span goals at full budget:

| | Difference [95% CI] |
|---|---|
| L k1 | −0.575 [−0.631, −0.513] |
| R k1 | −0.245 [−0.308, −0.187] |
| P k1 | −0.237 [−0.368, −0.102] |
| L k4 | −0.132 [−0.192, −0.073] |
| P k4 | −0.085 [−0.152, −0.015] |
| R k4 | n.s. |

In-span, the advantage is small or absent: L k1 −0.042 [−0.064, −0.022], R k1 −0.028 [−0.048, −0.006], the rest n.s. With 8 training parents the scalar evaluator is better on P k1 (+0.125) and R k1 (+0.080).

**2. Frozen-LM features are not a better substrate than a small symbolic learner given the same labels.** B3 − S:

| | Difference [95% CI] |
|---|---|
| P k1 | +0.330 [+0.21, +0.45] |
| P k4 | +0.318 [+0.22, +0.42] |
| R k4 | +0.124 [+0.06, +0.19] |
| L k4 | +0.067 [+0.01, +0.13] |
| L k1 | −0.041 [−0.081, −0.001] |

**3. Titan as an explicit simulator is the best LM-based consequence model for multi-step decisions.** At k=4, chained self-rollout beats the latent head, every scalar evaluator and the no-LLM learner. B3 − G_chain:

| | Difference [95% CI] |
|---|---|
| L k4 | +0.305 [+0.166, +0.443] |
| R k4 | +0.577 [+0.507, +0.641] |

The exception is P (0.88), where the precondition shortcut compounds. No new training is involved: the one-step skill came from Titan's 384-example update, and goals enter only through a linear readout of simulated consequence features (RCWM's "B4-roll" arm).

## R8. Where Titan's one-step skill fails, by operation (pilot TEST)

Chain steps are counted only when the input state was correct, so errors do not compound.

| Family | Operation | n | Accuracy | Dominant error |
|---|---|---|---|---|
| R | swap | 133 | **1.000** | — |
| R | add | 95 | 0.842 | missing mod 97 (7 of 15 errors) |
| R | conditional (other > threshold / ≤) | 48 / 47 | 0.438 / 0.511 | wrong value |
| R | **mix** (r += 2·other + v) | 126 | **0.008** | wrong value; missing mod (41) |
| L | scale, posted | 99 | 0.646 | one balance wrong |
| L | add, posted | 68 | 0.529 | one balance wrong |
| L | transfer, posted | 144 | 0.319 | one or both balances wrong |
| L | **any event with status = void** | 41 | **0.00–0.12** | event applied anyway |
| P | applicable action | 161 | 0.329 | wrong integer |
| P | **inapplicable action** | 117 | **0.009** | action applied anyway |

**What Titan learned.** The 384-row delta taught the surface operations: swap, add, the scale direction. It did not teach:
- conditional rules: void status and preconditions;
- composite arithmetic: mix and modulo.

**Provenance.**
- The training rows were Forge positives, and Forge raises on inapplicable actions, so Titan never saw an inapplicable example.
- SELECT-VIEWS-001 was 100% applicable, so these failures were invisible to the original acquisition metric.

**Implication.** Chained accuracy compounds per-step accuracy.
- Current state: 4-step chains score 0.02 (L) and 0.09 (R).
- If per-step accuracy were 0.9 for every operation, 4-step chains would reach about 0.65.
- So oracle-generated targeted rows for void, inapplicable, mix and modulo cases are likely the highest-leverage next training intervention. This is a hypothesis to test, not a result.

## R9. The constant offset carries output format, not the transition skill (CPU steering preview, D-18)

**Setup.**
- Script: `scripts/run_steer_cpu.py`, using `run_steer.py`'s offsets, hook and scoring unchanged.
- Items: 48 random TEST one-frame items per family (seed 17), run on CPU in FP32.
- Offset: v[fam, L], the TRAIN mean of h_sparse − h_supa at the answer slot. It is added (or subtracted) at the output of block L, at the last prompt position and at every decode step, with α = 1.
- Random control: a Gaussian direction of the same norm, one draw.
- Outputs: `runs/bank_pilot/steer_cpu/`, with per-item correctness. The full-TEST GPU run in `master_chain.sh` stays authoritative for the numbers.

**Exact-state accuracy on items whose state changes (lenient parse):**

| Condition | L (n 40) | P (n 24) | R (n 46) |
|---|---|---|---|
| Supa, no steering | 0.025 | 0.000 | 0.022 |
| Supa + v at L16 / L20 / L24 | 0.000 / 0.050 / 0.050 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| Supa + random at L16 / L20 / L24 | 0.050 / 0.050 / 0.050 | 0.000 / 0.042 / 0.000 | 0.065 / 0.196 / 0.065 |
| Titan-Sparse, no steering | 0.425 | 0.458 | 0.587 |
| Titan-Sparse − v at L16 / L20 / L24 | 0.400 / 0.350 / 0.400 | 0.417 / **0.000** / **0.042** | 0.522 / 0.609 / 0.587 |

**Sufficiency fails.** Adding the offset to Supa does not install the skill at any layer, and it does no better than a random direction of the same norm.
- What it does change is the answer behaviour: Supa gives bare integers on P, or echoes the input JSON on L and R.
- The one random-vector bump (R at L20) is also a format effect. It pushes Supa into a parseable `{"before","after"}` reply, where swaps come out right.

**Necessity fails for the computation.**
- For L and R, subtracting the offset from Titan at any layer changes only a few changed items. Paired, it loses 1–5 and gains 1–4.
- The P collapse at L20/L24 is a format collapse. 35/48 and 46/48 outputs become unparseable, degenerating to strings like `154{"{"{"…`, rather than wrong integers.

**Interpretation.** R5's decomposition stands: 97–99% of the late-layer answer-slot shift for L and R is a constant per-family offset. But that constant carries task mode and output format (when to stop, what shape to emit), not the transition computation. The computation lives in the small item-specific part of the shift, or in how the update acts at other positions and layers.

**Correction to R5.** A large constant share of a fine-tuning shift is not evidence that the skill is a steerable vector.

**Limits.**
- 48 items per family, one random draw, α = 1 only.
- One layer at a time.
- Steering applied only at the answer slot and decode positions, not at prompt positions.

## R10. S-roll: Titan's chained rollout against a generic one-step learner chained the same way

S-roll is PREREG-1 §6's registered but unimplemented "explicit simulator without an LLM".
- **Code and full report:** `scripts/s_roll.py`; `docs/s_roll/S_ROLL_PILOT.md`; numbers in `runs/bank_pilot/analysis/s_roll.json`.
- **Verification:** a build-then-verify workflow with three adversarial verifiers (leakage, evaluation equivalence, learner fairness). Fairness found a must-fix: the L comparator was weak. It was fixed by adding PREREG-1's DEV-regret selection rule and an arm trained on Titan's own 128 rows per family.
- **Learner:** an MLP on `decide.symbolic(state, event)`, trained on gold one-step transitions of pilot TRAIN parents and chained 4× on its own discrete predictions. Regret comes from `decisions.g_only` itself. Titan's regrets reproduce PREREG-1's dry run to 1e-7.

**k=4 normalised regret (Titan chain_one − S-roll; negative = Titan better; 95% parent bootstrap):**

| Family | Titan | vs S-roll at 128 rows (Titan's budget) | vs S-roll on all TRAIN transitions |
|---|---|---|---|
| L | 0.507 | **−0.185 [−0.317, −0.051]** (Titan's own rows: −0.158 [−0.305, −0.008]) | +0.199 [+0.051, +0.345]; with the DEV-regret head +0.456. The lead is gone by 512 transitions (+0.053 [−0.076, +0.189]) |
| P | 0.879 | +0.333 [+0.220, +0.450] (Titan's own rows: +0.127 [−0.025, +0.278]) | +0.626 [+0.491, +0.766] |
| R | 0.386 | **−0.391 [−0.438, −0.342]** (DEV-regret selection) | **−0.239 [−0.296, −0.182]** |

**Where each wins (first steps from the gold start state, Titan vs S-roll on all transitions):**
- Titan's R advantage is register **swap**, i.e. exact copying: 1.000 vs 0.000. An MLP fed numbers as scaled scalars does not learn it. On add (0.85 vs 0.86) and conditional (0.45 vs 0.62) the learner matches or beats Titan, and both fail mix.
- On L the failures are complementary. S-roll does add, transfer and void (0.96–1.00) but not scale (0.09); Titan does scale (0.55) but applies void events (0.10).

**Titan's prerequisite rule is integer addition, not bit-setting (verified directly from the stored chains).** Of 125 applicable first steps, 46 are right, 2 unparseable and 77 wrong.
- 50 of the 77 wrong outputs equal `state + add` exactly.
- 32 of those are unambiguous: the add bit was already set, so the correct bitwise result needs no change, yet Titan outputs the carried integer sum. In 29 of the 32 nothing was deleted, so the right answer was the unchanged state.
- The other 18 are consistent with ignoring the delete bits.
- Together with R8 (inapplicable actions applied anyway), Titan's P skill is "add the action's number", a surface operation. It is not the precondition, delete and set semantics.

**Interpretation.** Relative to a generic chained learner, the LM contributes:
- exact copying (R swap);
- data efficiency on ledger arithmetic, at its own 128-row budget only;
- nothing on P.

So "chained self-rollout is the best LM-based decision method" (R7, and PREREG-1's C0–C4) does not imply that an LM simulator beats a plain learned simulator given abundant transitions.

**Limits.**
- Exploratory: 26–28 TEST parents per family, and TEST was seen during development (disclosed in the report).
- One learner family, with no copy mechanism and no digit inputs.
- The learner's TRAIN transitions come from TEST's own generator, while Titan's 384 rows do not.

## R11. R8, the integer-addition rule and R10 replicate on the confirmation bank

**Status:** post-hoc exploratory replication; not preregistered; no tuning on these data.
- **Data:** the PREREG-1 confirmation bank, run after its confirmatory analysis had been written. 150 parents per family; twins excluded.
- **Settings:** every definition, hyper-parameter, head, selection rule and seed comes from the pilot. The retrained S-roll models are bit-identical to the pilot ones.
- **Sanity:** Titan's regrets through this path equal the confirmatory per-parent values exactly; an oracle gives regret 0.
- **Files:** `docs/replication/CONFIRM_REPLICATION.md`, `scripts/confirm_replicate*.py`. A verifier checked the work; its fixes are applied.

**(A) The per-operation failure map replicates** (chain steps with a correct input state; pilot → confirmation):
- L posted add 0.53 → 0.54, scale 0.64 → 0.58, transfer 0.32 → 0.33; L void 0.07 → 0.10.
- P applicable 0.32 → 0.45; inapplicable 0.009 → 0.008.
- R swap 1.00 → 1.00, add 0.84 → 0.89, conditional 0.47 → 0.47, mix 0.008 → 0.011.
- The P-applicable rise is largely consistent with case mix: reweighted to the pilot cell mix it is 0.39 (post hoc).
- The R add error label replicates only partially. Upward wraps output the unreduced sum r + v (6/7 → 9/9); downward wraps give other wrong values.

**(B) The prerequisite rule is integer addition, decisively** (applicable first steps from the gold start):
- Where `state + add` equals the correct bitwise result, Titan is right 46/53 → **383/415**.
- Where it does not, Titan is right 0/72 → **2/333**.
- 77% of parsed wrong outputs are exactly `state + add` (CI 71–83%), including 163 unambiguous carries.
- Inapplicable actions are applied anyway: right on 4/452.

**(C) S-roll vs Titan replicates** (Titan − S-roll at k=4, pilot → confirmation; negative = Titan better):

| Family | S-roll at 128 rows | S-roll on all TRAIN transitions |
|---|---|---|
| L | −0.185 → **−0.167 [−0.221, −0.111]** | +0.199 → **+0.216 [+0.156, +0.279]** (DEV-regret head +0.457) |
| P | +0.333 → **+0.306 [+0.267, +0.345]** | +0.626 → **+0.579 [+0.532, +0.626]** |
| R | −0.607 → **−0.624**; DEV-regret −0.391 → **−0.393** | −0.239 → **−0.234 [−0.258, −0.209]** |

- Of all 84 contrasts (3 families × k ∈ {1, 2, 4} × arms × selection rules):
  - 68 are significant on both banks with the same sign;
  - 10 go from n.s. to significant in the pilot's direction;
  - 5 are n.s. on both;
  - 0 lose significance;
  - 1 changes sign, and it was n.s. on the pilot.
- Dropping the 55 P parents that overlap TRAIN transitions changes no P sign or significance status.

## R12. The transition skill is computed in blocks 8–15; the delta in blocks 16–27 is dispensable (patching, D-21)

**Setup.**
- Code: `scripts/run_patch.py` (sha d72c0912…), GPU under the lease, 2026-10-07 22:15–23:19. All 640 pilot TEST one-frame items. Outputs: `runs/bank_pilot/patch/patch_results.json`.
- Sanity checks, all passed on the real GPU:
  - identity patches exact (max|d| 0) at every layer, in both models;
  - hooks fire in prefill only;
  - every spliced block hash-equals its source;
  - a full splice reproduces the other model's texts;
  - the hybrid's first token equals the matching prefill-patch condition's on 100% of items.
- Metric: FR = (acc − A_Supa)/(A_Titan − A_Supa) on changed items, with a parent-cluster bootstrap CI.
- Labels: HI ≥ 0.7, LO ≤ 0.2, fixed in D-21 before the run.
- Reference accuracies: A_Titan = 0.433 / 0.529 / 0.595 and A_Supa = 0.034 / 0.000 / 0.080 (L / P / R).
- The delta is zero in blocks 0–7 (D-21). Blocks 8–15 hold 41% of its energy and blocks 16–27 hold 59%.

**Weight-spliced hybrids** (Titan's delta kept only in the named blocks, at every position including decoding; FR [95% CI]):

| Delta kept in | L | P | R |
|---|---|---|---|
| 8–11 | 0.13 LO | 0.83 HI | 0.50 MID |
| **8–15** | **0.85 [0.68, 1.02] HI** | **1.00 [0.76, 1.31] HI** | **0.97 [0.91, 1.03] HI** |
| 8–19 | 0.96 HI | 1.00 HI | 1.01 HI |
| 8–23 | 0.92 HI | 1.00 HI | 1.00 HI |
| 12–27 | 0.56 MID | 0.91 HI | 0.73 HI |
| **16–27** | **0.09 [−0.05, 0.18] LO** | **0.00 LO** | **0.00 [−0.07, 0.07] LO** |
| 20–27 | 0.07 LO | 0.00 LO | −0.07 LO |
| 24–27 | 0.00 LO | 0.00 LO | −0.02 LO |

**Prefill-only patches** (Titan's block-L prompt states written into Supa, or the reverse; Supa then decodes):
- **T2S_all@16:** L 0.24 MID, P 0.80 HI, R 0.45 MID. Titan's prompt encoding through block 15 is enough for P's short answers but only partial for L/R, whose state values are produced at decode positions where Supa's blocks 8–15 lack the delta.
- **T2S_all@24:** LO in every family.
- **T2S_last@L** (answer slot only): LO everywhere.
- **Mismatched-item and difference-swap controls:** LO everywhere, so there is no content leak.
- **S2T_all@16/24** (Supa's prompt encoding written into Titan): L/R stay HI (0.78–1.0), because Titan's own blocks 8–15 recompute at decode positions. P collapses into unparseable output at @24 (parse 0.06), as with the R9 subtraction.

**Interpretation.**
- The transition computation that the 384-row sparse delta installs is localised to its first eight blocks, 8–15.
- That part is sufficient on its own (HI in all three families), and the remaining 59% of the delta, in blocks 16–27, contributes nothing measurable to the skill (LO in all three).
- Within 8–15:
  - P is redundant across 8–11 and 12–15; either suffices.
  - L needs 12–15.
  - R draws on both.
- For multi-token answers (L, R) the delta must be active while the answer is generated, not only while the prompt is read.
- This explains R9: the large late-layer constant offset is a downstream format/mode signal, not the computation.

**Implication to test next.** If the block-16–27 part of the delta carries off-task drift without carrying the skill, localising the delta to blocks 8–15 would keep the skill with less interference. That bears on the programme's sparse-delta retention results. It is exploratory and not yet measured.

**Limits.**
- Pilot TEST one-frame items only; one-step accuracy, not chained decisions.
- A single cut grid (12/16/20/24).
- Exploratory.

## R13. RCWM-T2 seed 0: the rule-coverage LoRA fixes ledger, shortcuts prerequisite by copying, and leaves register unchanged (exploratory)

**Setup.** One D-22 paired turn (train 1,865 s + eval, 5,743 s in total). Analysis by the frozen `scripts/t2_analysis.py` (sha d699dd5a…, verified before running). Report: `docs/t2_analysis/T2_RESULTS.md`.
- Every contrast is adapter ON vs OFF in the same process.
- OFF reproduces the stored Titan-Sparse chains exactly: 100% identical chains in all six mode/family cells.
- The D-19 screen passed (mean targeted gain +0.42), so seeds 1 and 2 run automatically.
- **D-19 verdict for seed 0: "rule learned" FAIL.**
  - G1 (every targeted cell ≥ 0.80): FAIL.
  - G2 (no untouched cell drops): PASS.
  - G3 (P app-del gains): FAIL.
  - G4 (false copying does not rise): FAIL.
  - G5 (gains carry over to chain steps): PASS.
  - G6 (chain k=4 thresholds): FAIL.
  - G7 (regret thresholds): FAIL.
  - G8 (forgetting): PENDING until Supa's off-task capture exists.

**By family (pilot TEST; primary frames; exact state or normalised regret; OFF → ON):**

| | Ledger | Prerequisite (chain_m1) | Register |
|---|---|---|---|
| Rule cells (per-op) | void 0.00–0.08 → **1.00** | inapplicable 0.03 → 0.78, but app-del 0.02 → 0.00, pairs 0.30 | mix ≈ 0 → ≈ 0; add/wrap 0 → 0.20; conditional → 0.55 |
| Untouched cells | posted +0.28 to +0.50 (**improved**) | — | swap and add/nowrap unchanged |
| False copy (target ≠ input) | 0.01 → 0.00 | **0.02 → 0.55** | 0.06 → 0.01 |
| Chain k=1 / 2 / 4 | 0.40 / 0.20 / 0.02 → **0.88 / 0.82 / 0.68** | 0.20 / 0.04 / 0.02 → 0.64 / 0.46 / 0.33 (copy floor 0.61 / 0.45 / 0.31) | 0.61 / 0.36 / 0.09 → 0.61 / 0.37 / 0.09 |
| Decision regret, k=4 | 0.507 → **0.038**; drop +0.469 [+0.354, +0.593] | 0.878 → 0.685; drop +0.193 [+0.088, +0.302] | 0.386 → 0.323; drop +0.063 [+0.016, +0.108] |

- **Gains survive outside the training distribution.** The rule gains carry over to oracle-anchored pilot chain steps 2–4, and for ledger to |balance| > 99 (G5). Examples: void 0.06–0.12 → 1.00; P inapplicable 0.007 → 0.885.
- **P's gain is a copy shortcut.** On P chain steps, applicable state-changing steps fall from 0.46 to 0.31, while unchanged and inapplicable steps rise to 0.83 and 0.92. 66% of P training targets were copies (pre-flight review, finding 1).
- **Forgetting proxy:**
  - first-token KL(OFF‖ON) is 0.064 nats (median 0.033), top-1 agreement 0.82;
  - 64-token greedy replies are identical OFF vs ON on 54% of the 248 prompts;
  - JSON bleed 0.010 → 0.015;
  - the reference KL(Supa‖Sparse) is pending.

**Reading (seed 0 only; seeds 1–2 pending; pilot TEST was used to target the failures, so these are hypothesis-generating).**
- 1,501 targeted rows fix ledger almost completely: 4-step exact 0.02 → 0.68, and regret near oracle. This is the clearest positive result for small-model multi-step consequences in the study.
- Prerequisite needs data without the copy bias: fewer copy targets, more app-changed and app-del rows, and P upweighted (it is 8.4% of supervised tokens).
- Register mix and wrap (composite mod-97 arithmetic) are not learned from about 330 rows.

### R13 addendum: seeds 1 and 2 (read first by the PREREG-2 freeze tool, 15:34 on 2026-10-08)

**Source:** `docs/prereg2/freeze_gate/T2_GATE.md`, from the frozen `t2_analysis.py` on t2_s0, t2_s1 and t2_s2. Same recipe, 190 steps, no checkpoint selection. "Rule learned" FAILs in 0/3 seeds passing. G2 and G5 PASS in all three; G8 is PENDING in all three.

| Metric (ON − OFF, pilot TEST) | s0 | s1 | s2 | Agree |
|---|---|---|---|---|
| **L chain_one k=4 regret drop** (PREREG-2 gate) | +0.469 | **+0.410 [+0.291, +0.537]** | **+0.483 [+0.366, +0.606]** | yes |
| L chain_one k=4 exact | +0.659 | +0.678 | +0.798 | yes |
| L void-disc / posted per-op | +0.96 / +0.41 | +0.96 / +0.44 | +0.96 / +0.46 | yes |
| L chain_m1 k=4 exact (untrained frame) | +0.755 | +0.697 | +0.740 | yes |
| P false copy | +0.53 | +0.75 | +0.80 | yes, **worse** |
| P app add-only per-op | −0.31 | −0.44 | −0.56 | yes, **worse** |
| P chain_m1 regret drop | +0.193 | +0.107 | +0.042 | yes, shrinking |
| R chain_one regret drop | +0.063 | +0.053 | +0.050 | yes, small |
| R mix (wrap 1 / 2) | 0 / 0 | 0 / 0 | 0 / 0 | no change |
| first-token KL(OFF‖ON), nats | 0.064 | 0.095 | 0.068 | all < KL(Supa‖Sparse) 0.129 (R14) |

**Reading.**
- The ledger fix replicates across three seeds with nearly identical magnitude. That is what licensed the PREREG-2 freeze.
- The prerequisite copy shortcut also replicates, and it gets worse with seed: the data's 66% copy targets dominate.
- Register mix and wrap are never learned.
- The forgetting proxy stays below Titan-Sparse's own drift from Supa in every seed.

## R14. A block-localised delta keeps the skill with about a third less off-task drift (D-23; exploratory)

**Setup.**
- Code: `scripts/run_localized.py` (sha 2dfff5f3…) and `localized_analysis.py` (sha 036043e7…, verified before running). Report: `docs/localized/LOCALIZED_RESULTS.md`.
- Four FP32 arms spliced in place with hash gates: Supa; Titan-Sparse (full delta); **loc8_15** (Supa + delta in blocks 8–15 only, 41% of the delta energy); up16_27 (Supa + delta in blocks 16–27 only, 59%).
- Drift: measured against Supa on the 248 neutral off-task prompts. Skill: chain_one on the pilot TEST plans.
- The noise floor (stored vs re-run Titan-Sparse) is exactly 0.
- Run 2026-10-08 05:54–08:53, interrupted by I-27 and resumed from saved stages.

**Pre-stated D-23 verdict: "localisation reduces interference"** in both the primary direction KL(arm‖Supa) and the secondary KL(Supa‖arm).

| Drift vs Supa | Titan-Sparse | loc8_15 / sparse [95% CI] | up16_27 / sparse [95% CI] |
|---|---|---|---|
| mean first-token KL(arm‖Supa), nats | 0.107 | **0.68 [0.62, 0.74]** | 0.17 [0.12, 0.22] |
| mean first-token KL(Supa‖arm), nats | 0.129 | 0.65 [0.59, 0.72] (caution: concentrated) | 0.14 [0.10, 0.19] |
| top-1 first-token disagreement | 0.210 | 0.79 [0.65, 0.93] | 0.33 [0.20, 0.48] |
| 64-token greedy reply disagreement | 0.496 | 0.85 [0.77, 0.94] | 0.35 [0.27, 0.44] |

| Skill | Supa | loc8_15 | Titan-Sparse | up16_27 |
|---|---|---|---|---|
| one-step accuracy on changed first steps, L / P / R | 0.03 / 0.00 / 0.08 | 0.37 / 0.53 / 0.58 (FR 0.85 / 1.00 / 0.97) | 0.43 / 0.53 / 0.60 | 0.07 / 0.00 / 0.08 |
| chain_one k=4 regret, L / P / R | 0.82 / 0.97 / 0.90 | 0.525 / 0.925 / 0.380 | 0.507 / 0.879 / 0.386 | 0.82 / 0.93 / 0.89 |
| loc8_15 − sparse regret [CI] | — | L +0.017 [−0.127, +0.158]; P +0.047 [−0.031, +0.124] (low headroom); R −0.005 [−0.041, +0.029] | — | — |

**Reading.**
- Removing the 59% of the delta that does nothing for the skill (blocks 16–27) costs no measurable decision quality.
- It cuts off-task first-token KL drift by about a third, top-1 flips by about a fifth, and changed greedy replies by about 15%.
- Most drift, about two thirds, comes from the same blocks 8–15 that carry the skill, so localisation alone is not a large retention fix.
- It stacks with magnitude pruning. Titan-Sparse already keeps only the top 25% of entries; localisation drops a further 59% of the remaining energy with no measurable skill loss.

**Reference for T2's forgetting gate G8** (descriptive): KL(Supa‖Sparse) measured here is 0.129 nats. T2 seed 0's KL(OFF‖ON) of 0.064 is about half of it. G8 formally reads `runs/bank_pilot/supa/offtask_logprobs.npy`, which the queue produces later.

**Limits.**
- Exploratory: pilot TEST and 248 off-task prompts; first-token KL and 64-token greedy agreement are proxies for forgetting.
- One cut (8–15 vs 16–27).
- P regret has low headroom; its evidence is the one-step FR.

## Pending

- Sparse multi-step generation, under the fixed parser and caps.
- Teacher-forced likelihood ranking of the gold consequence against heuristic foils, for both arms at k=1, 2 and 4. This directly tests composition and avoids generation-format issues.
- PVR stage 1, and the retrospective verifier.
- Decisions (H3).
- The randdelta, dose, Exp, Transition and Qwen3 arms.
- Off-task mechanism measurements.
