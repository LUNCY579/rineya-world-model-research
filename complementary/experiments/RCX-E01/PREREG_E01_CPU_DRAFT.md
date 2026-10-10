# PREREG RCX-E01-CPU (DRAFT, not frozen): does counterfactual rule-varied training make small learners read supplied rules?

The template follows bundle `06_templates/PREREGISTRATION.md`.

**Status:** DRAFT. It is frozen (sha256 recorded in `DECISIONS.md`) only after the DEV engineering pilot and *before* the
SEALED battery is minted. The pilot may change endpoints only in two ways:
- based on floor or ceiling evidence (instrument validity);
- based on IID learnability (whether an arm learned its own training distribution).

Arm-versus-arm DEV differences are **not** used to choose endpoints. Explicit execution approval: CPU tier only. No
native or GPU work is covered.

## 1. Claim and scope

- **Hypothesis H01-CPU.** At a matched label budget, character-level transformers trained on **counterfactually varied,
  in-context rules** with surface diversity (arm `CFSD-ans`) execute supplied rules on fresh vocabulary and inverted
  rules better than the same learners trained on **ordinary varied rules** (arm `VR-ans`).
- **Task family.** The rule-conditioned ledger (`SPEC_RCL.md` v1), with kinds add and transfer.
- **Model.** CharLM d128 / L3 / H4 / ff512 (0.60M params), trained from scratch. **Tier: mechanism (CPU).**
- **Practical effect.** +0.10 on the primary composite (the bundle's development target).
- **What a positive result does not establish:**
  - any Rineya, Titan or Supa behavior;
  - LoRA transfer;
  - natural-language rule following;
  - generalization beyond this world.

## 2. Custody

- **Code.** Commit `<sha at freeze>`. File hashes are recorded in each run's `summary.json` (`code_sha256`).
- **Interpreter.** Qualified (`qualification/QUALIFICATION.json`): Q1 against RCWM `env.step`/`rule_gold`, and Q2/Q3
  against `oracle_indep`.
- **SEALED battery.** `make_battery("sealed", N, seed)`, where `seed = int(sha256(salt)[:8], 16)`. The salt is committed
  as `sha256(salt)` at freeze and revealed only after all arms are trained.
- **Status words.** SEALED status words come from `vocab.pools("sealed")`. They are disjoint from TRAIN and DEV, with
  edit distance ≥ 2 across splits. They exclude PREREG-3 and factory words.
- **Spent instruments.** The DEV battery (`battery_dev_60_20261220`) becomes a development instrument and is never
  used for confirmation.

## 3. Arms and fairness

- **Arms.** The arms are those in `DESIGN.md` §2.
- **Matched budget.** Every arm uses the same `N_TRAIN`, steps, batch size, learning-rate schedule, architecture and
  seed set.
- **Shared parents across formats.** Within a regime, the answer and trace arms train on identical parents.
- **Label access.** Every arm gets one exact label per example, so oracle labels are matched. Trace arms receive more
  supervised target tokens, and this is reported as `target_tokens`.
- **Fixed rule, no tuning.** No per-arm hyperparameter tuning: one configuration is fixed from the pilot's IID
  learnability on `VR-ans` only.
- **Deterministic baselines.** The shortcut floors and the ORACLE ceiling are scored on the same items.
- **Tool tier.** The canonicalizer (`+canon`) is an information tier B control and is reported separately.

## 4. Endpoints

- **Primary endpoint (choose one at freeze):** parent-macro exact transition accuracy on the composite {F2, F3, F5,
  V1, V2, V3, V5}, under surface `in` (P-in) or under the combined held-out surface `ho` (P-ho). Rule for choosing,
  fixed now: choose **P-ho** unless its pilot ORACLE ceiling or IID learnability shows the instrument cannot measure it
  (for example, every arm below 0.55 on P-ho, i.e. near the floors); in that case choose P-in.
- **Primary comparison.** `CFSD-ans` − `VR-ans`. The unit is the parent, with a seed × parent paired cluster
  bootstrap, 3 seeds and 10,000 resamples.
- **Protected strata** (non-inferiority, margin −0.05):
  - F0 posted twins (twin guard);
  - the IID same-distribution split of each arm.
- **Secondary (Holm family of 3):**
  - `CFSD-trace` − `CFSD-ans` (trace effect);
  - `CFSD-trace` − `CFSD-fill` (trace beyond length);
  - `CFSD-ans` − `FXSD-ans` (rule varying vs the T2b-analog fixed rule).

  Everything else is descriptive.
- **Scoring rule.** Malformed output is wrong.

## 5. Decision rule

| Outcome | Condition |
|---|---|
| **ADVANCE** (to the native kit) | primary diff ≥ +0.10, the 95% CI lower bound > 0, and no protected-stratum loss beyond −0.05 |
| **REVISE** | diff > 0 with the CI excluding 0, but < +0.10 |
| **STOP** | CI upper bound < +0.05 |
| **INCONCLUSIVE** | anything else |

**Resources:** CPU only, single lane, about 12 min per arm-seed.

## 6. Execution and safety

- `rcx/guard.py` thermal and RAM courtesy guard.
- Single CPU lane, at most 4 threads.
- Create-only outputs.
- A crash resumes from the checkpoint with identical fingerprints.

## 7. Freeze and deviations

The freeze hash, timestamp and deviation log go here once written. No rerun is allowed under the same ID.
