# RCWM-T1 PREREG-2 (DRAFT): does the fixed T2 seed-0 LoRA lower Titan-Sparse's chained ledger decision regret (adapter ON vs OFF, same process, fresh sealed parents)?

**Status:** DRAFT, NOT FROZEN (DECISIONS D-24). Revised after the 3-lens adversarial review (Appendix B). It becomes `PREREG_2.md` only through `scripts/confirm2_gen.py --freeze`, which applies the §9 freeze condition mechanically, and only before any model touches `data/bank_confirm2.json`.

Template: `PREREG_1.md`. Where this document is silent, PREREG-1's conventions apply (endpoint, parsing, regret, tie-breaking, bootstrap form, Holm, deviation policy).

## 1. Claim under test

The claim:
> On fresh ledger parents, adding a fixed LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases (`runs/t2/t2_s0/adapter`, T2 seed 0) to frozen Titan-Sparse **lowers the goal-conditioned regret of its chained four-step self-rollout decisions** (`chain_one`, k = 4) **and raises its four-step exact-state accuracy**, relative to the same model with the adapter switched off in the same process.

Context. The chained rollout is RCWM's "B4-roll". PREREG-1 confirmed it pooled over families, driven by register; it did not show it for ledger alone (ledger −0.072 against F_touched, above the −0.10 per-family threshold). PREREG-2 asks only whether the adapter changes the ledger rollout relative to itself. It does not ask whether the rollout beats other learners (§8 limits).

The evidence motivating it is exploratory (RESULTS_PILOT R13; `docs/t2_analysis/T2_RESULTS.md`): seed 0 only, on pilot TEST, which had been used to diagnose the failures T2 targets (R8). The registered analysis code reproduces R13 on those outputs (dry run, `docs/prereg2/PREREG_2_DRYRUN.md`; a code check, not evidence):

| Pilot TEST, T2 seed 0, adapter OFF -> ON (n parents) | Ledger chain_one (26) | Register chain_one (26) | Prerequisite chain_m1 (28) |
|---|---|---|---|
| k = 4 normalised regret | 0.507 -> 0.038 (ON − OFF −0.469 [−0.593, −0.354]) | 0.386 -> 0.323 (−0.063 [−0.108, −0.016]) | 0.878 -> 0.685 (R13, two-mode call); 0.884 -> 0.688 (−0.196 [−0.307, −0.093]) through the registered single-mode call |
| k = 4 exact-state accuracy | 0.024 -> 0.683 (+0.659 [+0.587, +0.726]) | 0.087 -> 0.091 (+0.005) | 0.018 -> 0.330 (copy-chain floor 0.308) |
| False copy (own-input chain steps) | 0.030 -> 0.000 | 0.049 -> 0.022 | **0.015 -> 0.677** |
| Per-parent SD of ON − OFF regret | 0.314 | 0.122 | 0.279 |

- **Reproduction.** Ledger and register reproduce R13 exactly; P1 equals minus the T2 drop, with the CI mirrored. Prerequisite chain_m1 regret differs by about 0.005 in the registered single-mode call. The reason is the order of the tie-break random stream: `t2_analysis` calls `g_only` with both modes, so chain_m1 gets the second draw. Through the two-mode call the code reproduces R13's 0.878 / 0.685 (`repro_check`). Prerequisite regret is tie-sensitive under copy outputs; ledger P1 is invariant over 20 tie seeds (review).
- **Ledger mechanism caveat.** Most of the pilot ledger gain is on plans without a void event: all-posted plans went 0.039 -> 0.636 exact, any-void plans 0.000 -> 0.759. So the gain is largely posted arithmetic and frame, not only the void rule the rows were stratified toward.
- **Other families.** The prerequisite "gain" is a copy shortcut (R13), and register is essentially unchanged. The confirmatory claim is therefore restricted to the ledger. The other two families are secondary, with registered directional predictions (§6).

## 2. Data

- **Confirmation bank:** `data/bank_confirm2.json`, sha256 `fafb13e573a20eb5045ef204a7cdbf8da4393d21604245e9df26995f5be58aea`.
  - **Builder:** the frozen PREREG-1 builder `rcwmt.bank.build()` (sha `b08b0229…`), called by `scripts/build_confirm2_bank.py` (sha `41f2f484…`, recorded inside the bank). **No model, tokenizer or model output was used** (torch was never imported).
  - **One build.** Exactly one bank build was attempted (`docs/prereg2/bank_build_log.json`). The build is deterministic given the seed and the exclusion union: the engineering review rebuilt it in memory byte-identically (the file differs only by Windows CRLF line endings).
  - **Settings:** seed **20261020**, which no earlier RCWM-T1 builder, analysis or bootstrap has used. 150 parents per family, K = 8 precommitted plans, H = 4, split = test. Prerequisite parents need ≥ 4 distinct final states (131 rejected). No order twins.
  - **Payload freshness (enforced by the builder; re-verified from the written file):** 0 of 14,400 candidate transitions equal any payload of the 293,851-payload exclusion union, and 0 parents equal a parent of an earlier bank. 3 parents were rejected for collisions during the build. A machine-wide scan by the review (204 payload-bearing files) found 0 exact-payload collisions anywhere. The union holds:

    | Source | Payloads |
    |---|---|
    | historical records (`harvest_exclusions`) | 6,464 |
    | `bank_pilot` | 15,629 |
    | `bank_confirm` (incl. order twins) | 13,552 |
    | `bank_s4000` | 255,745 |
    | `t2_train` | 1,501 |
    | `t2_eval` | 600 |
    | `t2_eval_xframe` | 600 |
    | `t2_eval_supp` | 360 |

  - **Operative-content overlap (descriptive).** This is the payload minus fields that cannot change the next state: tick, unused event fields, prerequisite name and cost.
    - **Ledger: 0 parents**, also under the review's broader FDS operative key.
    - **Register: 11 parents.** 10 overlap `bank_s4000`, which no model was trained or scored on (build log). 1 (`T1R-4e04d7f9…`) overlaps the composition_confirmation_v1 bank `T_BANK_V3X.json`, outside the exclusion union (found by the review's scan).
    - **Prerequisite: 121 parents; the family is fresh at payload level only.** The 8-bit state space is small. 117 are in the build log, 27 of them (78 transitions) sharing operative content with `t2_train`, the adapter's training rows. 4 more overlap the composition_confirmation_v2 bank `T_BANK_V4.json` (review scan).
    - The "drop any overlap" sensitivity (§6) uses the build-log parents plus these 5 (`EXTRA_OVERLAP_PARENTS` in the analysis).
  - **Oracle-only structure** (computed from the bank alone, before any model output):
    - Ledger steps: 585 of 4,800 are void.
    - Ledger plans: 497 of 1,200 contain at least one void event (148 parents have one or more such plans); 703 are all-posted (all 150 parents).
    - Copy-chain exact floor at k = 4: L 0.0017, R 0.0025, P 0.117.
    - Ledger k = 4 answer-free floors: F_touched 0.577; the rule-ablated true simulator F_ignore_status 0.120.
  - **Build record:** `docs/prereg2/bank_build_log.json` (sha `0b3a6eaa…`).
- **Fixed artifact under test:** `runs/t2/t2_s0/adapter/adapter_model.safetensors`, sha256 `92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a`. Its `adapter_config.json` has sha256 `50d9a776…`.
  - LoRA r 16, α 32 on q, k, v, o, gate, up and down. PEFT 0.20.0.
  - Trained for a fixed 190 steps (no checkpoint selection), seed 0, lr 1e-4 on `data/t2_train.jsonl` (sha `bf5a2276…`).
- **Off-task prompts:** `data/offtask_prompts.jsonl` (248 prompts in 62 reply classes, sha `ae92870b…`), the set T2 and D-23 use (§5b).
- **Anchor references** (instrument check, §7):
  - `runs/t2/t2_s0/gen_chain_one_base.jsonl` (sha `c082a532…`);
  - `runs/t2/t2_s0/gen_chain_one.jsonl` (sha `5f3a69f3…`).

## 3. Subjects and instrument

- **Model:** Titan-Sparse (`Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, fingerprint prefix `818be9c6b6a3`) with the adapter **OFF** (`PeftModel.disable_adapter()`, which is Titan-Sparse) and **ON**.
  - Both are in **one process**, loaded exactly as `scripts/run_t2.py` loads them.
  - The adapter files are hashed again at load time, and each side's PID and timestamps are recorded.
- **Reference model:** Supa 1.7 (fingerprint prefix `a9952c55f5a1`). It is loaded in the same GPU job, only after Titan-Sparse is unloaded and VRAM is confirmed empty, and only for the off-task reference: first-token log-probs and 64-token greedy replies.
- **Runtime:** as PREREG-1 §3.
  - FP32 with TF32 off; torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0 (hermes venv). The analysis checks every turn's recorded versions.
  - Greedy decoding, `enable_thinking=False`, EOS [151645, 151643].
  - 160-token cap; `generate_texts` batching as amended in PREREG-1 Amendment 1.
- **Parsing:** `env.parse_state(..., lenient=True)`, identical for OFF and ON.
- **Chains (`rcwmt.generation.chain`):**

  | Family | Mode | Role |
  |---|---|---|
  | Ledger | `chain_one` (Titan's one-step frame, which T2 trained) | primary |
  | Register | `chain_one` (trained frame) | secondary |
  | Prerequisite | `chain_m1` (single-event multi-step rule, which T2 trained) | secondary |
  | Ledger | `chain_m1` | transfer secondary |

  Register `chain_m1` and prerequisite `chain_one` were descriptive-only transfer cells in D-19 and are not run.
- **Anchor (instrument identity):** before any sealed-bank generation, the job regenerates the 208 pilot TEST ledger `chain_one` chains, OFF and ON.
  - They must reproduce the T2 seed-0 files with identical states on ≥ 99% of chains on each side; I-22 and R13 found 100%. Otherwise the job stops (§7).
  - Every resumed turn that generates sealed-bank chains re-anchors first. A failed re-anchor is persisted and final.

## 4. Methods

Every method is scored on the same 150 confirmation parents per family and the same goals.

| Code | Definition |
|---|---|
| **G_OFF** | `chain_one` (ledger, register) or `chain_m1` (prerequisite) self-rollout with the adapter off. This is PREREG-1's G_chain(Titan) on new parents; ledger was 0.539 in PREREG-1. Final state → φ (`data.phi_from_state`) → U = w·φ. Unparseable candidates get their parent's mean predicted utility (primary). The last parsed chain state is a sensitivity analysis. |
| **G_ON** | The same, with the adapter on, in the same process. |
| **F_touched** | PREREG-1's answer-free touched-entity floor. |
| **F_copy_s0, copy-chain** | φ of the start state; copy-chain exact floor = P(gold state at k = start state). |
| **F_ignore_status (L), F_ignore_pre (P)** | Rule-ablated true simulators. Diagnostic only, never a comparator for a claim. |

## 5. Endpoint, primary contrasts and decision rule

- **Endpoints (per parent).** Both are paired OFF/ON on the same parent.
  - **(a) Normalised regret:** Σ_g (U* − U_sel) / Σ_g (U* − Ū) over the **256 PREREG-1/T2 test goals (seed 1001)**.
    - Ties are broken at random with seed 17, a fresh generator per call, so OFF and ON get identical tie noise.
    - It is computed through `rcwmt.decisions.g_only`'s own code path, and every call is checked equal to it.
    - A parent with a zero denominator is excluded (none on this bank at k = 1, 2, 4).
  - **(b) Exact-state accuracy at k:** the share of the parent's 8 candidates whose predicted k-step state equals the gold state.
- **Primary analysis set:** ledger, `chain_one`, k = 4, 150 parents.

**Primary contrasts.** Both are ON − OFF, ledger, `chain_one`, k = 4.

| | Contrast | Claim | Threshold on the point estimate |
|---|---|---|---|
| **P1** | regret(ON) − regret(OFF) | < 0 | ≤ **−0.15** |
| **P2** | exact(ON) − exact(OFF) | > 0 | ≥ **+0.25** |

**Test.**
- Paired per-parent differences are bootstrapped with 10,000 parent resamples, seed 17.
- One-sided p = (#{boot mean on the null side, inclusive} + 1) / 10,001.
- Holm step-down (monotone) is applied across P1 and P2 (m = 2).

**Decision rules,** applied in this order:
1. **Reversal.** A primary whose point estimate is in the wrong direction with unadjusted one-sided p < 0.05 is reported as REVERSED. The claim is then not confirmed, whatever the other primary shows.
2. **PASS.** Pj passes iff its Holm-adjusted p < 0.05 **and** its point estimate is beyond its threshold.
   - The thresholds apply to the point estimate; they are not tested margins (as in PREREG-1). With SE about 0.026 (P1) and 0.015 (P2), Holm significance never binds once a point is beyond its threshold, so PASS effectively means "point beyond threshold". A true P1 effect of −0.12 still passes about 12% of the time.
   - **Strong pass:** the **unadjusted** one-sided 95% bootstrap bound is beyond the threshold. This is a level-0.05 test of the margin. Because "claim confirmed" is an intersection-union statement, requiring both strong passes is also a level-0.05 margin test of the conjunction. Strong-pass status is stated in the headline, not only in the table.
3. **Claim confirmed** iff P1 **and** P2 both PASS and neither is reversed. This is an intersection-union test: each contrast at 0.05 already gives a level-0.05 test of the conjunction, so Holm is conservative here and matters only for the partial outcomes. Otherwise each primary is reported with the §8 wording.

**Qualifiers** (wording only; they never change PASS):
- **F_touched.** If mean ON regret is not below F_touched's, the claim carries "not better than an answer-free touched-entity rule". This is **expected never to bind on ledger**: a P1 pass forces ON ≤ OFF − 0.15, about 0.39 at the PREREG-1 OFF baseline, while F_touched is 0.577 on this bank. It can bind only if OFF > 0.727.
- **Copy floor.** If exact(ON) < copy-chain floor + 0.10, the claim carries "not better than copying". This is **expected never to bind on ledger** (floor 0.0017).
- **Copying (can bind).** If ledger `chain_one` false-copy ON − OFF has a 95% CI above 0, the claim carries "with increased copying".
- **Diagnostic.** F_ignore_status (the true simulator minus the void rule; 0.120 on this bank, 0.115 in PREREG-1, 0.099 on pilot TEST) is reported next to ON. P1 can pass on posted-arithmetic gains alone, so the all-posted and any-void exact contrasts are named secondaries (§6).
- **Split halves** are descriptive only (§6). They are no longer a qualifier: random, exchangeable halves carry no information about heterogeneity, and a half-level miss would fire by chance (about 16-31% of the time at a true P1 effect of 0.20).

**Why these thresholds.**
- **P1, −0.15.** This is D-19's regret-drop gate for ledger, set before any T2 data. It is 1.5× PREREG-1's practical threshold, about 28% of the PREREG-1 ledger baseline (0.539), and about one third of the exploratory effect.
- **P2, +0.25.** This is the D-19 k = 4 chain gate, also set before any T2 data. The OFF baseline is 0.024 (pilot and PREREG-1 bank), so it amounts to "ON ≥ about 0.27".
  - P2 is deliberately the more demanding bar: 38% of the pilot effect, against 32% for P1. The claim is about exact simulation as well as decisions.
  - Under proportional shrinkage P2 binds first. The P2 threshold proportional to P1's would be 0.211. The draft keeps +0.25 rather than tune a threshold to the pilot effect ratio.
  - In the band where P1 passes and P2 does not, the §8 wording states only the numbers.
- **Room for shrinkage.** Both thresholds leave room for substantial shrinkage on fresh data (regression to the mean, since pilot TEST was used to target the failures) while ruling out a trivial effect.

**Power** (`docs/prereg2/PREREG_2_DRYRUN.md` §Power). Pilot SDs: P1 0.314, P2 0.182. On the PREREG-1 bank (n = 150), ledger G_chain OFF regret had SD 0.340.

(a) Normal approximation, homogeneous effects (n = 150; SE 0.026 for P1 and 0.015 for P2). Values are at SD × 1 / × 1.25 / × 2:

| True effect (claim direction) | P1 PASS | P1 strong | P2 PASS | P2 strong |
|---|---|---|---|---|
| Half of pilot (0.234 / 0.329) | 0.999 / 0.996 / 0.950 | 0.95 / 0.84 / 0.50 | 1.00 / 1.00 / 0.996 | 1.00 / 0.996 / 0.85 |
| Threshold + 0.05 (0.20 / 0.30) | 0.97 / 0.94 / 0.84 | 0.62 / 0.47 / 0.25 | 1.00 / 0.996 / 0.95 | 0.96 / 0.85 / 0.51 |

(b) Joint power for **claim confirmed**. This is a simulation with the registered bootstrap and Holm code (n_boot 2,000, 200 runs), resampling the 26 pilot parents at coherent (P1, P2) effect pairs. There are three effect models:
- **scaled:** every parent improves by λ × its pilot gain;
- **mixture:** each parent is fully fixed with probability λ, otherwise unchanged; this gives heterogeneous effects, with SD(d2) about 1.9× the pilot's;
- **sd_upper:** SDs at the upper 95% bounds of the pilot SDs (0.433 / 0.251).

| (P1, P2) effect | scaled | mixture | sd_upper |
|---|---|---|---|
| Pilot (0.469, 0.659) | 1.00 | 1.00 | 1.00 |
| Half of pilot (0.234, 0.329) | 1.00 | 0.99 | 0.985 |
| P1 = 0.20 (0.200, 0.281) | 1.00 | 0.88 | 0.895 |
| P1 = 0.169 (0.169, 0.237) | 0.005 | 0.34 | 0.15 |

- **What the power figures mean.** At half the pilot effect, joint power is ≥ 0.985 in every model. At P1 = 0.20 it is about 0.88–0.90 under heterogeneous or high-variance effects, not about 0.97. When the P1 effect is near 0.17, P2 binds and the expected outcome is "PARTIAL: P1 passes".
- **Parent count.** 150 parents per family is kept. It matches PREREG-1 and keeps the secondaries informative. The register regret secondary has 0.88 unadjusted power against half its pilot effect (0.031); prerequisite has 0.99. Both figures are unadjusted, not BH-adjusted.

## 5b. Forgetting guard (registered; not part of the confirmed claim; a re-measurement)

- **Not a prospective test.** The guard re-measures quantities that are partly known before the freeze:
  - **Numerator.** S2 re-measures KL(OFF‖ON) with the same fixed adapter, the same 248 prompts and the same deterministic FP32 instrument that gave 0.064 nats and 0.54 greedy agreement on T2 seed 0.
  - **Denominator.** The queued D-23 localized job measures KL(supa‖sparse) and Supa/Sparse greedy agreement on the same prompts, probably before the freeze.
  - **What is new here.** The guard is measured in the same job as the confirmation chains, by the registered code, under the rule below.
  - **When the rule was fixed.** This section was fixed before any D-23 off-task output existed: `runs/bank_pilot/localized` held only `RUN.json` at the time of `docs/prereg2/PRE_SEED_SNAPSHOT_1.json` and `_2.json`, which hash this file.
  - **Freeze record.** `confirm2_gen.py --freeze --d23-offtask-seen yes|no` records whether D-23's off-task numbers had been seen, plus a listing of D-23's outputs.
- **Statistic.** On the 248 off-task prompts, with first-token log-probs stored in FP16 and renormalised in float64 (`t2_analysis.lp_compare`):
  - per prompt, KL(OFF‖ON);
  - per prompt, the reference KL(Supa‖Sparse), measured **in the same GPU job** (Sparse = OFF).

  Both are KL(before‖after). r = ratio of means. Its interval comes from a reply-class cluster bootstrap (62 classes, 10,000 resamples, seed 17). The prompts are fixed, so the interval treats the reply classes as a sample of possible off-task classes.
- **Guard rule (three outcomes):**
  - **PASS** iff the one-sided 95% upper bound of r < 1;
  - **FAIL** iff the one-sided 95% lower bound of r > 1;
  - otherwise **INCONCLUSIVE**.

  r and its CI are reported in every case. Ratio 1 means the adapter shifts off-task first-token distributions as much as the original 384-example Titan delta did (D-19's G8 yardstick).
- **Descriptives (registered, not guard outcomes):**
  - cumulative drift KL(Supa‖ON) / KL(Supa‖Sparse), with a reading bound of 1.25;
  - 64-token greedy agreement OFF/ON vs Supa/Sparse vs Supa/ON (S6 adds Supa's greedy replies);
  - top-1 agreement;
  - JSON bleed on non-numeric classes.
- **Effect on the claim.** The guard is **not** part of "claim confirmed". Its sentence always accompanies the claim (§8). **Only FAIL changes the headline**; INCONCLUSIVE is stated in the guard sentence.

## 6. Secondary analyses (two-sided parent bootstrap p for ON − OFF ≠ 0; BH q = 0.10 across all 28)

**The 28 contrasts.**
- **Regret and exact state (22).** For each cell (L `chain_one`, R `chain_one`, P `chain_m1`, L `chain_m1`) and each k ∈ {1, 2, 4}: regret ON − OFF and exact-state ON − OFF. That is 24 contrasts, minus the two primaries.
- **False copy (4).** False-copy rate ON − OFF per cell. On every chain step whose own input parses and whose correct next state, given that input, differs from it, a copy is an output equal to the input. The rate is per parent.
- **Void split (2).** Ledger `chain_one` k = 4 exact ON − OFF, per parent:
  - on **all-posted plans** (150 parents);
  - on **any-void plans** (148 parents).

  These separate posted-arithmetic and frame gains from the void rule.

**Notes on the family.**
- About half the contrasts are ledger contrasts expected to be near-certain, which loosens the BH thresholds for register and prerequisite.
- False-copy and step-level rates condition on arm-specific step sets. Their ON − OFF differences are paired by parent, not by step.
- The pilot prerequisite SD (0.279) comes from pilot parents without the ≥ 4-distinct-final-states filter. On the PREREG-1 bank, P `chain_m1` OFF regret had SD 0.191.

**Pre-declared directional predictions** (evaluated by these rules and reported as holds / does not hold; not claims):
- **D1 (prerequisite copy shortcut persists):** P `chain_m1` false-copy ON − OFF > 0, with the 95% CI above 0.
- **D2:** P `chain_m1` k = 4 exact(ON) ≤ copy-chain floor + 0.10. The floor on this bank is 0.117.
- **D3:** P `chain_m1` accuracy on applicable, state-changing, correct-input steps is not higher ON than OFF (point estimate; arm-specific step sets).
- **D4 (register: no large change):** at k = 4, |exact ON − OFF| ≤ 0.05 and |regret ON − OFF| ≤ 0.10 (PREREG-1's per-family practical threshold).
  - D4 can hold alongside a small BH-significant register change; the pilot's −0.063 was significant. Both are reported side by side.

**Prerequisite wording.** The prerequisite family is not content-fresh (121 of 150 parents overlap, §2). OFF parse failures contribute to its regret contrast: pilot k = 4 parse rate OFF 0.616 vs ON 1.000 (see the last-parsed sensitivity).

**Sensitivity analyses** (descriptive; k = 4, every cell):
- last-parsed fill;
- a fresh goal set (seed 1004);
- dropping parents with operative-content overlap with `t2_train`;
- dropping parents with any operative-content overlap (§2: build log plus the review scan).

**Descriptives.**
- Exact-state and parse rates per cell and k.
- Accuracy on correct-input chain steps by operation class (ledger kind × posted/void; register kind × wrap; prerequisite applicability × change).
- Pooled ledger k = 4 exact for any-void versus all-posted plans.
- **Split halves.** The 150 ledger parents are ordered by sha256(`RCWM-PREREG2-half-v1|parent_id`) and split 75 / 75. Both half estimates of P1 and P2 are reported. They are random and exchangeable, and carry no qualifier.
- All floors, at k = 1, 2 and 4.

**Not implemented, therefore exploratory and untested here:**
- S@abundant and S-roll comparators against the adapted rollout;
- the D-19 self-distillation control and an untargeted LoRA, which would separate rule learning from frame adaptation;
- seeds 1 and 2 as artifacts;
- register `chain_m1` and prerequisite `chain_one`.

## 7. Procedure and integrity

1. **Pre-seed snapshots.** Create-only `docs/prereg2/PRE_SEED_SNAPSHOT_<n>.json`, taken by stat and hash only. Each holds the sha256 of this draft, both confirm2 scripts, the builder, `DECISIONS.md` and the prereg2 artifacts. It also lists `runs/t2` and D-23's `runs/bank_pilot/localized`, and records the existence of every seed-1/2 output file.
   - `_1` is the draft as reviewed; `_2` is after the review edits. Both were taken before any seed-1/2 output existed.
   - The freeze uses the latest snapshot. Anything it locks may change afterwards only by a logged typo or crash edit (§9).
2. **Freeze:** `python scripts/confirm2_gen.py --freeze --d23-offtask-seen yes|no` (CPU only). It refuses if:
   - PREREG-2 is already frozen;
   - `runs/bank_confirm2` exists;
   - the gate already failed;
   - the latest snapshot does not predate every seed output;
   - a locked file changed without a logged edit.

   If none applies, it runs the frozen `scripts/t2_analysis.py` (sha `d699dd5a…`) in-process on `runs/t2` with tags `t2_s0,t2_s1,t2_s2`. The output goes to `docs/prereg2/freeze_gate/T2_GATE.{md,json}`, not `docs/t2_analysis/`. The tool then applies §9.
   - **On failure** it writes `docs/prereg2/FREEZE_GATE_FAILED.json` (create-only) and freezes nothing. The failure is final.
   - **On success** it writes `PREREG_2.md` (this draft with the status line and §10 filled in; the draft is kept as the pre-freeze record) and `PREREG_2_FREEZE.json` (create-only). The freeze record holds:
     - every input hash (§10);
     - the gate numbers and the gate output;
     - the snapshot used and any post-seed edit logs;
     - the seed output times;
     - whether D-23's off-task numbers had been seen;
     - the registered runtime.
   - A freeze script separate from `confirm2_gen.py` was not created, because the drafting workflow may create only the D-24 files.
3. **Verification:** `confirm2_gen.py --verify`. Both the GPU job and the analysis call the same verifier, and neither runs unless it passes. The verifier checks:
   - `PREREG_2.md` against its recorded hash;
   - every required input is hashed (all of `rcwmt/*.py` included), with the `PREREG_2_AMENDMENT_*.json` chain applied as `verify_freeze.py` does for PREREG-1;
   - the gate rule, recomputed from the hashed gate output and compared with the recorded numbers;
   - the snapshot-before-seed ordering, and seed `eval.json` unchanged since the freeze.
4. **GPU job:** `scripts/confirm2_gen.py`, one shared lease `rcwm-t1:prereg2`, under every lease, alternation, thermal and RAM rule.
   - **Stage order** (OFF and ON of a stage in one process, written together, create-only):
     - A0 anchor;
     - S1 ledger `chain_one`;
     - S2 off-task first-token OFF/ON plus 64-token greedy;
     - S3 register `chain_one`;
     - S4 prerequisite `chain_m1`;
     - S5 ledger `chain_m1`;
     - unload, VRAM check, pause-flag and resource wait, then S6 Supa off-task log-probs and greedy.
   - **Turn cap (D-22's rules):**
     - 4.5 h measured from lock acquisition (the lock record);
     - no stage starts within 0.5 h of the cap;
     - a stage starts only if 1.25× its estimate, plus a Supa reserve of 1.25× S6 while S6 is outstanding, fits.
   - **Projection** (`--plan`, `docs/prereg2/GEN_PLAN.json`): at most 40,808 generations, 4.32 h in a single pass at the registered planning rate of 2.7 generations/s.
     - Under the stage rule this is two turns: turn 1 runs A0, S1, S2, S3, S4 and S6 (3.33 h); turn 2 runs the re-anchor and S5 (1.19 h). The total is 4.52 GPU-h including the re-anchor and the reload.
     - At T2 seed 0's observed chain rate (6.57 generations/s) it is one turn of 1.84 h.
   - **Completion:** a stage is done only if its files exist and `provenance.json` records them with matching sha256. `DONE.json` lists every output hash.
   - **Outputs:** `runs/bank_confirm2/`.
5. **No peeking.** Between the first sealed-bank output and the registered analysis, no one reads `runs/bank_confirm2/chains/*` or `offtask/*`, or computes any metric from them.
   - Only `provenance.json`, `anchor/*` and the job log may be read, as instrument checks.
   - No analysis-code change is allowed after the first sealed output, except a logged non-statistical crash fix, recorded as an amendment.
6. **Analysis:** run `scripts/confirm2_analysis.py` once, after `--verify` passes. It dies on any failed check:
   - bank and build-log hashes, and 150 parents per family;
   - `DONE.json` and provenance agree on every file hash, and the files are unchanged;
   - every stage is recorded, with all sides of a stage in one process;
   - the anchor passed, and every resumed turn that generated sealed-bank chains passed its re-check;
   - every turn's code hashes equal the frozen or amended versions;
   - every turn's runtime versions, the adapter hash at load, and both fingerprints;
   - unfiltered, well-formed chains for every cell (T2's `chain_errors`; a foreign or unknown row is an error);
   - off-task arrays and greedy replies (OFF/ON and Supa) covering the 248 prompts in order.

   It writes `runs/bank_confirm2/analysis/PREREG_2_RESULTS.json`, `…_PER_PARENT.npz` and the mechanical render `PREREG_2_RESULTS.md` as one set: written as `.partial` files first, then renamed, all create-only.
7. **Report** exactly as computed: REPORT §3b and a RESULTS file rendered from the JSON.

**Deviation policy.**
- **Analysis crash:** a crash before the results set is written allows an identical rerun. A code fix is allowed only for a non-statistical crash. It is recorded as `PREREG_2_AMENDMENT_*.json` (old and new hash, diff, reason) and in DECISIONS.md.
- **GPU crash or reset:** any stage that is not done is moved to `runs/bank_confirm2/_orphans/` (never deleted) and regenerated in full, OFF and ON together. "Not done" covers half-written files, stray `.partial` files, and files without a provenance record. No stage is stitched across processes. A recorded output whose hash changes stops the job.
- **Anchor failure (first or re-check):** the job stops and the failure is persisted. The cause is investigated and logged.
  - If it is a software or environment change, the registered environment (§3) is restored and the job is rerun.
  - Any other remedy is an amendment, recorded before any affected sealed-bank output exists.
- **Supa stage:** if it cannot complete after two attempts (load failure, fingerprint mismatch), an amendment may let the analysis run with the guard "not evaluable". P1 and P2 are unaffected.
- **Operational amendments** (batching, pauses, thermal or lease handling) follow PREREG-1 practice. They are recorded as `PREREG_2_AMENDMENT_*.json` with old and new hashes before any affected output exists. The verifier applies the chain.
- **Threads:** analysis is CPU, numpy-only, at 2 threads, and deterministic.

## 8. Wording

`scripts/confirm2_analysis.py` renders this section mechanically (`wording()`). Every branch is exercised on synthetic inputs in the dry run (`wording_selftest`, 12 branches). The branches are applied in this order:

| Outcome | Headline / report |
|---|---|
| Either primary reversed (wrong direction, unadjusted one-sided p < .05) | "REVERSED: {P1/P2} significantly in the wrong direction". "On fresh parents the adapter makes ledger decisions worse (higher regret) / lowers ledger four-step exact-state accuracy: R13's improvement does not replicate. The claim is not confirmed." |
| P1 and P2 both pass | "CONFIRMED (strong pass: both one-sided 95% bounds beyond the thresholds)" or "CONFIRMED (point estimates beyond the thresholds; not a strong pass: the one-sided 95% bound of {P} is not beyond its threshold)", followed by the claim sentence below |
| P1 passes, P2 does not, P2 significant | "PARTIAL: P1 passes, P2 does not". "The adapter lowers ledger decision regret beyond −0.15 (strong pass: yes/no). Four-step exact-state accuracy rose by e [CI], below the pre-set +0.25." |
| P1 passes, P2 not significant | Same headline. "… Four-step exact-state accuracy did not rise significantly (e [CI]): the decision gain is not accompanied by a significant exact-state gain (consistent with partially correct states)." |
| P2 passes, P1 does not, P1 significant | "PARTIAL: P2 passes, P1 does not". "The adapter raises ledger four-step exact-state accuracy beyond +0.25 (strong pass: yes/no). Decision regret fell by d [CI], short of the pre-set −0.15." |
| P2 passes, P1 not significant | Same headline. "… Decision regret did not fall significantly (d [CI]): the exact-state gain did not translate into a significant decision gain." |
| Neither passes; one or both significant in the predicted direction | "DIRECTIONALLY CONFIRMED ({P} only), below the pre-set practical threshold". "{P}: significant in the predicted direction, point estimate short of the pre-set threshold; {other}: not significant. The claim is not confirmed." |
| Neither significant | "NOT CONFIRMED". "The T2 seed-0 ledger effect (R13) did not replicate on fresh parents: neither primary contrast is significant." |
| Always | One line per primary: point (OFF → ON), 95% CI, one-sided 95% bound, Holm p, threshold, verdict |
| ON regret ≥ F_touched (P1 passed) | Qualifier: "not below the answer-free touched-entity rule (F_touched)" |
| exact(ON) < copy floor + 0.10 (P2 passed) | Qualifier: "not better than copying" |
| Ledger false-copy ON − OFF CI > 0 (a primary passed) | Qualifier: "With increased copying: ledger chain_one false-copy ON − OFF x [CI]" |
| Guard PASS / INCONCLUSIVE | Guard sentence: "Forgetting guard {status}: off-task first-token KL(OFF‖ON) is r× KL(Supa‖Sparse), the Titan delta's own drift (95% CI; one-sided bounds); cumulative KL(Supa‖ON) is c× KL(Supa‖Sparse) (reading bound 1.25); 64-token greedy agreement OFF/ON a vs Supa/Sparse b vs Supa/ON c. A re-measurement of quantities partly known before the freeze (§5b)." |
| Guard FAIL | The same guard sentence, and the headline gains "; at an off-task first-token drift r× that of the Titan delta itself (forgetting guard FAIL)" |
| Guard not evaluable | "No off-task reference was measured; no statement about drift." |

**Claim sentence (both pass):**
> On 150 fresh ledger parents, adding a LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases (T2 seed 0, the fixed artifact) to frozen Titan-Sparse lowered the normalised regret of its chained four-step self-rollout decisions from X_OFF to X_ON (ON − OFF d, 95% CI, one-sided 95% upper bound U, Holm one-sided p; 0 = oracle, 1 = random) and raised four-step exact-state accuracy from a to b (+e, CI, one-sided 95% lower bound L), against the same model with the adapter off in the same process. Both point estimates are beyond the pre-set thresholds (−0.15, +0.25); strong pass: yes/no.

**Limits that always accompany it** (rendered verbatim from `LIMITS`):
- Ledger only, horizon ≤ 4, one adapter (T2 seed 0), Titan-Sparse only, the chain_one frame.
- The plans and the adapter's training rows come from the same event generator and marginals: the bank is fresh at payload level, not out-of-distribution.
- Rule learning is not separated from frame adaptation (no self-distillation or untargeted-LoRA control); posted-arithmetic gains count toward the effect, and the targeting of the training rows was not shown to cause the gain.
- No comparison with non-LM learners at the same label budget: S@abundant (4,000 parents) reached 0.273 on ledger in PREREG-1 and S-roll matched Titan on ledger at about 512 transitions (R10); the adapted model used 384 + 1,501 oracle-labelled rows.
- Register (mix, wrap) and prerequisite are not claimed fixed; the prerequisite prediction is a copy shortcut, and the prerequisite family is fresh at payload level only.
- Retention evidence is limited to the first-token guard and 64-token agreement on 248 prompts, a re-measurement of quantities partly known before the freeze (§5b).

The split-half estimates follow as a descriptive line.

## 9. Freeze condition (stated before any T2 seed-1 or seed-2 output existed)

**Freeze rule.** PREREG-2 is frozen and run **only if T2 seeds 1 AND 2 each show a ledger `chain_one` k = 4 regret drop (OFF − ON; pilot TEST; D-19 instrument; frozen `scripts/t2_analysis.py`, sha `d699dd5a…`) ≥ 0.15, with a 95% paired CI excluding 0.**
- The two numbers per seed are the `E` block's `L|chain_one` `drop` and `drop_ci95` from `t2_analysis.run(["t2_s0", "t2_s1", "t2_s2"])`.
- "Excluding 0" means `drop_ci95[0] > 0`.
- The rule is applied mechanically by `confirm2_gen.py --freeze` (§7.2). No one reads a seed-1/2 number before the freeze tool computes it.

**If either seed fails,** PREREG-2 is not run as drafted, and the seed instability is reported. The failure is recorded in `docs/prereg2/FREEZE_GATE_FAILED.json` and is final.

**The registered artifact is the seed-0 adapter,** the only one fixed before this decision. Seeds 1 and 2 serve only as a replication gate. They are never used to choose an adapter, thresholds, sizes, contrasts or any other part of this document.

**When the condition was stated.**
- It is recorded in DECISIONS D-24. The hand-written "01:50" there runs ahead of the file-system clock; `DECISIONS.md` was last modified at 01:27:11 −0500, 2026-10-08.
- File evidence that it predates every seed output: `docs/prereg2/PRE_SEED_SNAPSHOT_1.json`, written 02:29:23 −0500, hashes `DECISIONS.md` (sha `ac9de968…`, containing D-24) and records that `runs/t2/t2_s1` and `runs/t2/t2_s2` did not exist. `PRE_SEED_SNAPSHOT_2.json` does the same for this revised draft and both scripts, and also records the absence of every seed-1/2 chain file.

**Edits after the latest pre-seed snapshot,** to any section of this draft or to `confirm2_gen.py`, `confirm2_analysis.py` or `build_confirm2_bank.py` (including registered constants):
- Only typo or crash fixes are allowed.
- Each needs a `docs/prereg2/POSTSEED_EDIT_<n>.json` with the file, old and new sha256, type (typo or crash), the diff, and a statement that it does not depend on seed-1/2 results.
- Each must also be logged in DECISIONS.md.
- The freeze tool refuses anything else.
- A new pre-seed snapshot is valid only while no seed-1/2 output exists.

## 10. Freeze record (to be filled at freeze)

`confirm2_gen.py --freeze` replaces this section with the sha256 of every frozen input. The inputs are:
- every file in `confirm2_gen.REQUIRED_INPUTS`;
- all of `rcwmt/*.py`;
- the pre-seed snapshot used;
- the gate output;
- any post-seed edit logs.

Values as of this draft:

| File | sha256 (draft time) |
|---|---|
| `data/bank_confirm2.json` | `fafb13e573a20eb5045ef204a7cdbf8da4393d21604245e9df26995f5be58aea` |
| `docs/prereg2/bank_build_log.json` | `0b3a6eaa57fd03e11d95bbb83cae58d1732ed8e2102c66700560f02799d8c040` |
| `scripts/build_confirm2_bank.py` | `41f2f48498313be996f8f508d87d6200618fd76b264b2a51b5388d793277be84` |
| `scripts/confirm2_gen.py`, `scripts/confirm2_analysis.py` | at freeze (draft-time values in `PRE_SEED_SNAPSHOT_2.json`) |
| `docs/prereg2/GEN_PLAN.json`, `PREREG_2_DRYRUN.{json,md}`, `PREREG_2_DRYRUN_PER_PARENT.npz` | at freeze (the cited projection and dry run; neither is rewritten once frozen) |
| `scripts/t2_analysis.py` | `d699dd5afa432dfef8b1c39cd14af26b92750237c7108328f7a61403decbcaf0` |
| `scripts/run_t2.py` | `ccbb2c6c19f76130d9b2e4885a04ca25d7215975e0be207d307c57bf1c4877f4` |
| `rcwmt/*.py` | as in PREREG-1 §9 + Amendment 1 (`generation.py` `a4052088…`) |
| `runs/t2/t2_s0/adapter/adapter_model.safetensors` | `92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a` |
| `runs/t2/t2_s0/adapter/adapter_config.json` | `50d9a776e093a952ac440a77c9cd7cd8e51eb22a85eb93b383aee1eacc185d9d` |
| `runs/t2/t2_s0/gen_chain_one_base.jsonl` | `c082a5324ae63f76b6550381a5368faf9ae8669f17c7da510ee5234bb44255ae` |
| `runs/t2/t2_s0/gen_chain_one.jsonl` | `5f3a69f3a67b56df4657a31f7a675a07ea15d6b33d9ab4b2dfcb3c9db066f47b` |
| `data/offtask_prompts.jsonl` | `ae92870bef9ab18e99229af71d764d12b0544d40590f2faf2c7d1e9356f3d6bc` |
| `data/t2_train.jsonl` (+ `t2_eval`, `t2_eval_xframe`, `t2_eval_supp`, `t2_eval_substrata`, full sha256 at freeze) | `bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744` |
| `data/bank_pilot.json` | `7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a` |

## Appendix A. Choices made in this draft, with reasons

- **Adapter = seed 0.** It is the only artifact fixed before the PREREG-2 decision (D-24). Choosing among seeds 1 and 2 after seeing them would be selection on the outcome.
- **Ledger only as primary.** R13's register effect is small (−0.063) and its exact accuracy is unchanged. The prerequisite gain is a copy shortcut. Claiming either would confirm an artifact.
- **Two co-primaries, both required.** P1 is the RCWM decision claim. P2 checks that the decision gain comes with better exact simulation, rather than only a change in which states are wrong. Its threshold is the more demanding one (§5).
- **Same goals (seed 1001) and the g_only path.** The goals are a fixed random draw never selected on outcomes. Holding them fixed keeps the instrument identical to R13 and PREREG-1. Freshness is in the parents. A fresh goal set (seed 1004) is a sensitivity analysis.
- **Bootstrap.** PREREG-1's per-family form: parent resampling, 10,000 resamples, seed 17, inclusive one-sided p. Two-sided p for the secondaries, because some have predicted adverse directions.
- **No "both halves" rule.** With a single primary family, random halves carry no substantive information, so they are descriptive. The substantively motivated split (all-posted vs any-void plans) is a named secondary instead.
- **Floors.** F_touched as in PREREG-1, and the copy-chain floor. Both are stated as non-binding on ledger. The copying qualifier and the void-split secondaries are the checks that can bind.
- **Anchor stage.** It costs about 10 GPU-minutes per turn. It proves that the ON arm on the GPU is the R13 artifact: same PEFT wiring, versions and outputs.
- **No twins, no e2e.** Neither enters a registered contrast here. Dropping them saves about 3 GPU-hours at the planning rate: e2e OFF/ON would be about 21,600 generations, and twin chains about 7,200.
- **Freeze tool inside `confirm2_gen.py`.** The drafting workflow may create only the D-24 files, and the runner and the analysis must share one verifier.

## Appendix B. Adversarial review (3 lenses) and dispositions

**Applied:**
- **Title and claim.** Neutral title. "Near-oracle" and "makes … a simulator" removed. The adapter is described as trained on oracle-labelled rows stratified toward rule cases, with no causal attribution to the targeting.
- **Guard.**
  - Relabelled as a re-measurement (§5b) and locked by pre-seed snapshots before any D-23 off-task output existed.
  - The D-23 seen/unseen statement goes into the freeze record.
  - Three-outcome rule with one-sided bounds.
  - Cumulative KL(Supa‖ON) and Supa greedy replies added (+248 generations, +1.5 min).
- **Wording.**
  - The renderer checks reversal first and names which contrast is significant.
  - The partial outcomes are neutral, with no "(significantly)".
  - A guard FAIL changes the headline. Strong-pass status is in the headline.
  - The limits render §8 verbatim.
  - A 12-branch self-test is in the dry run.
- **Freeze.**
  - Mechanical gate via the frozen `t2_analysis`.
  - The verifier recomputes the rule from the hashed gate output.
  - It requires `PREREG_2.md`, a full required-input list and the amendment chain.
  - Both the runner and the analysis call it.
  - Pre-seed snapshots plus logged post-seed edits, with the edit lock widened to every section and both scripts.
- **§1/§8 context.** PREREG-1's ledger result is stated correctly. Comparator and label-budget limits added.
- **Bindings.** The floors are stated as non-binding; a copying qualifier and two void-split secondaries were added.
- **Power.** A joint-power simulation under heterogeneous and high-variance effects was added. The figures quoted earlier ("≥ 0.996 in every scenario", "0.97") are corrected.
- **Reproduction.** §1 is corrected for prerequisite regret (tie-stream order), and the check is added to `repro_check`.
- **Halves.** Made descriptive.
- **Runner.**
  - Crash windows closed: DONE is finalised when every stage is recorded; Supa provenance is saved immediately; stray `.partial` files and unrecorded files make a stage partial; output hashes are recorded in provenance and DONE.
  - A failed re-anchor is persisted.
  - VRAM check, pause-flag and resource wait before the Supa load.
  - D-22 stage margin; deadline from lock acquisition; the full Supa reserve.
  - Per-side PIDs and the adapter hash at load.
  - Create-only `anchor.json`.
  - `--plan` and the default dry run no longer rewrite frozen files.
- **Analysis.**
  - Every turn's re-check, stage coverage, code hashes and runtime versions checked; DONE and provenance hashes checked.
  - Unfiltered chain validation.
  - A clean die on missing provenance or build log, and on a changed build log.
  - Greedy files required.
  - Outputs written as a set.
  - Integrity checks survive `python -O`.
- **Overlap and notes.** The 5 extra-overlap parents were added, D4 narrowed and renamed, notes on the secondary family and on prerequisite wording added, one-build and no-peeking rules stated, and "01:50" corrected to file evidence.

**Rejected or modified, with reasons:**
- **Align the P2 threshold with P1 (0.211).** Not done. +0.25 is the pre-existing D-19 gate, fixed before T2 data. Re-deriving it from the pilot effect ratio would tune a threshold on the exploratory data. The costs are stated instead (joint power, P2 binding first, neutral partial wording).
- **A fresh off-task prompt set to make the guard prospective.** Not done. `rcwmt/offtask.capture_offtask` is frozen and reads `data/offtask_prompts.jsonl`, and no fresh prompt pool was sampled. The guard is not part of the claim, so the honest relabel was chosen (§5b). This is left as an open question.
- **Register the cumulative ratio with a bound of 1.25 as a guard outcome.** Reported as a descriptive reading only, to keep one guard statistic.
- **Full sha256 checks inside `build_confirm2_bank.py`.** Not done. The builder's sha is recorded inside the sealed bank, so editing it would break the bank's provenance. The reviewers verified the T2 inputs with full hashes, and the freeze record hashes them in full.
- **A separate `scripts/freeze_prereg2.py` / `verify_freeze2.py`.** Housed in `confirm2_gen.py` (`--freeze`, `--verify`), because of the file-creation scope.
