# RCX-E01R DESIGN (draft): reading the complement of a supplied rule from unseen wording

| | |
|---|---|
| Status | **DRAFT** (2026-10-11). Nothing is frozen and nothing has been trained. The pool-authoring step is running. |
| Origin | R-08: the PREREG_E01N SEALED decision was REVISE (CX D-33) |
| Tier | native-C component artifacts: LoRA on frozen Titan-Sparse, the exact D-03 recipe, CPU |

## 1. What R-08 leaves open

On held-out rule wordings (`ho_tpl`), N-CFSD decides "no-op" correctly for:
- 96–97% of the negative-rule items that need one;
- only 14% of the **positive**-rule items that need one.

Apply items are ≥ 0.89 correct in both polarities. The two spent templates S+ h0 and S− h0 are mirror images ("If the
status is X, carry out the event; for any other status, do nothing" vs "If the status is X, do nothing; for any other
status, carry out the event"), yet their no-op accuracies differ: 0.18 vs 0.96.

**Descriptive reading (post hoc, from R-08's breakdown).**
- On an unseen wording, the adapter decides a *listed* status by the cue attached to it.
- It decides an *unlisted* status by defaulting to "apply".
- That default matches every negative rule, whose complement takes effect, and contradicts every positive rule, whose
  complement has no effect.
- In the F strata (the posted rule, with unseen probe statuses), this leaves `ho_tpl` near 0.5.

**H-E01R.** The residual deficit is *complement inference*: concluding from an unseen wording that all other statuses
have no effect. Reading the listed statuses is not the problem.

**H-alt.** The deficit is specific to the two spent surfaces (h0, h1). On fresh positive wordings the adapters would
then be largely correct.

## 2. Instrument: a blind-authored factorial wording pool (running; see `split_pool.py`)

**Cells.** 2 polarities × 3 forms:

| Form | What the wording states |
|---|---|
| EXPLICIT, listed first | both cases, with the listed case first |
| EXPLICIT, complement first | both cases, with the default (unlisted) case first |
| IMPLICIT | one case only; the other follows from an exclusivity or exception meaning |

**Authoring.**
- Three authors, each in a different register (everyday, formal-regulatory, maximal syntactic variety). Each writes 8
  per cell.
- The authors work only from a prompt. They never see any RCX template, nor the R-06/R-08 wordings.

**Verification (label-blind).**
- Three verifiers each judge every rendered template at one list size (1, 2 or 3 statuses), with one listed and one
  unlisted probe status, in a different order.
- A template passes only if all three verifiers' answers equal the intended semantics on both probes. "Unclear"
  fails.

**Audit** (`rcx/rcl/wording_audit.py`, code). Templates are dropped for:
- a status-like word (any RCX or peer vocabulary);
- near-duplicating any RCX train or held-out template, composition frame or clause (word Jaccard ≥ 0.5, or a shared
  5-word sequence);
- near-duplicating another pool template (Jaccard ≥ 0.6, or a shared 6-word sequence). The cluster keeps one member.

**Split.**
- Per cell, a secret salt orders the survivors and deals them alternately:
  - **P2 (DEV):** visible and committed;
  - **P3 (SEALED):** git-ignored; only its sha256 and sha256(salt) are committed. The lead never reads it.
- P3 is used only by a future salt-committed SEALED battery. The pool is fixed before any model sees it.

## 3. Step A: DEV2 diagnostic (descriptive; no training)

**Battery.**
- `make_battery("dev", …)`-style parents with a fresh DEV2 seed (202612xx), on the single-atom strata: F1–F5, V1, V2,
  V3, V7.
- Surface `p2`: the rule is rendered with a P2 template drawn per parent, stratified over the 6 cells. Everything
  else is canonical, as in `in`.
- Size: about 40 parents per stratum.

**Arms.** N-CFSD s0–s2, N-VR s0, and OFF (reference). About 15–25 min each on CPU, in the single lane.

**Quantities** (descriptive):
- decision accuracy on required-no-op items and on apply items, per cell (polarity × form), with Wilson intervals;
- POS − NEG no-op accuracy within each form (parent-clustered bootstrap);
- the same on `in` (training wording), as the within-battery anchor.

**Readings fixed before the run:**
- **H-alt supported:** N-CFSD POS no-op accuracy on P2 is ≥ 0.80 in every form. The R-08 failure was wording-specific.
  E01R then becomes a small coverage check, and RCM moves on to E02 (computed predicates).
- **H-E01R supported:** POS no-op accuracy is < 0.50 while NEG is ≥ 0.80.
  - EXPLICIT > IMPLICIT for POS: the adapter can use an explicit complement statement but cannot infer one.
  - EXPLICIT ≈ IMPLICIT ≈ low: it does not read complement statements on new wordings at all.
- Anything in between is reported as is, and the recipe choice below names the cell it rests on.

## 4. Step B: candidate recipes (one or two are chosen after step A and preregistered before training)

| Recipe | What changes in training |
|---|---|
| **R1** complement coverage | Restrictive positive wordings with diverse complement statements, from a *separately* blind-authored training pool audited against P2 and P3, at matched rows |
| **R2** complement-focused twins | Each positive-rule parent gets extra twins whose event status is *unlisted*: fresh nonce or real words, rendered under 2–3 training wordings, all with the no-op target. They are balanced by listed-status twins, so the complement case is trained as often as the listed case. |
| **R3** predicate-first target | `trace`-style output that states the membership test and decision (`cmp`) before the state (`after`). This tests the original E01 trace hypothesis. |

Rows, steps and budget are matched to N-CFSD; the control is N-CFSD.

## 5. Step C: confirmation (to be preregistered after step B is chosen)

- A new salt-committed SEALED battery, with surfaces `in` and `p3` (the P3 wordings, unseen by any analysis).
- 3 seeds per new arm. The existing N-CFSD s0–s2 adapters are re-evaluated as the control, which saves about 8 h of
  training.
- Co-primaries and decision rule in the PREREG_E01N style:
  - POS no-op accuracy on `p3`;
  - composite on `p3`;
  - protected `in` composite and posted twins.

## 6. Order and compute

1. Pool authoring and verification (subagents; no CPU-lane load).
2. `split_pool.py` (seconds).
3. Step A (about 1.5–2 h CPU) after E11.
4. Prereg for step B and C.
5. Training: about 2.6 h per adapter.
