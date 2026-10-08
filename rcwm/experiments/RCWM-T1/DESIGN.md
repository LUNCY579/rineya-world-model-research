# RCWM-T1 — Latent consequence readout from frozen Titan

**Status:** design v0 (2026-10-05). The pilot bank is exploratory. Confirmatory claims are reserved
for a fresh bank whose analysis plan is frozen in `PREREG.md` after the pilot.

## Question

The Titan programme installed a one-step state-transition skill into Supa 1.7. It used 384 verified
rows and a 0.056%-norm weight update, and the skill lives in the update's large-magnitude head.
End-to-end *multi-step* generation stays at floor (DMRF: 1/32 for 2 events).

RCWM-001 proposes learning goal-agnostic consequence representations φ from frozen Rineya states,
with goals as readouts w_g·φ, tested against a matched scalar evaluator. This experiment joins the
two lines with three questions:

1. **H1, latent ≥ generative.** Does a light readout on frozen Titan hidden states, captured at
   the answer slot and at each action boundary, predict the k-step consequences of a precommitted
   plan better than Titan can generate them? Two generation modes count here: end-to-end, and a
   chained one-step self-rollout.
2. **H2, attribution.** Is the decodable consequence information caused by the transition delta?
   The test uses five readouts at matched head capacity and labels: Supa vs Titan-Sparse, a dose
   line Supa + α·dSparse with α ∈ {0, 0.5, 1, 2}, a random delta with dSparse's exact support and
   per-tensor norm, BF16-rounded Titan-Experimental, and the pre-Rineya Qwen3-0.6B.
3. **H3, decision value (the RCWM primary contrast).** Consider goal-conditioned selection among
   K=8 precommitted candidate plans on held-out parents, under held-out goals. Does a goal-agnostic
   consequence head plus a linear goal readout (B3) beat a matched goal-conditioned scalar head
   (B2)? Does it hold at small label budgets? Does it beat Titan's generative rollouts and a
   no-LLM symbolic learner trained on the same labels?

A fourth, descriptive question (**H4, shortcut**) asks whether Titan's generation ignores
prerequisite preconditions, and whether the latent readout encodes applicability anyway.

## Environment (rcwmt/env.py, rcwmt/bank.py)

- **Families.** ledger_replay (L), prerequisite_plan (P) and register_machine (R) use the
  FDS-oracle semantics, validated 384/384 historically. Every generated transition is re-checked
  against those oracles.
- **Parents.** Each parent is a start state plus K=8 candidate plans of H=4 events.
  - P: events are drawn from a 6-action library per parent. Preconditions span all bits, so
    applicability changes along a plan. In the pilot, about 52% of steps are applicable.
  - L and R: events are drawn independently from the recorded input marginals.
- **Freshness.** Every one-step sub-transition was checked against all 6,464 historical transition
  payloads. The pilot found 0 collisions.
- **Splits.** Each parent is assigned a split by salted SHA-256, 70/15/15, before any model sees
  data. All candidates of a parent stay in its split.
- **Prompt formats.**
  - multi: the system rule says "apply each event in order" (DMRF precedent); the user turn is
    `{"before", "events":[...]}`, with k ∈ {1, 2, 4}.
  - one: the verbatim one-step Titan frame for the first event, as a trained-format control.
  - Rendering always uses `enable_thinking=False`.

## Capture (rcwmt/capture.py)

- Layers {0, 4, 8, 12, 16, 20, 24, 28} of the residual stream, stored as fp16. Layer 28 is the
  final norm.
- **Answer slot:** the last prompt token, `\n\n` after `</think>`.
- **Action boundaries:** the token holding event j's closing brace. Qwen's BPE merges this with
  the next event's opening (`},{"`), so the position has seen events 1..j plus a content-free `{"`.
  It is a prospective prefix encoding.
- **G0 checks:** batched-padded vs single, and truncated-prefix vs bulk forward, both in FP32 with
  TF32 off.

## Readouts

- **Probe:** ridge regression or logistic probe per feature, with the layer and λ chosen on DEV.
- **Consequence head (B3):** an MLP, 1024 → 256 → |φ|, with BCE for binary features and Huber
  loss for continuous ones. Utility is U_g = w_g · φ̂.
- **Matched scalar (B2):** an MLP on [h; w_g] → U, with the same hidden width and the same parents.
  It is trained on train-goal utilities. The tuning budget is matched: the same LR grid and seeds.
- **No-LLM learner (S):** an MLP on an exact symbolic encoding of the start state and plan. It is
  trained on the same φ labels, so it measures how much the LLM representation adds over learning
  the dynamics from scratch.
- **Generative (G):** parse the generated final state, then compute exact φ. Unparseable outputs get
  a random tie-break.
- **Oracle (O):** the true φ.

## Consequence features and goals

| Family | φ |
|---|---|
| P | 8 final bits, plus cost of applied actions / 10 |
| L | sign_neg ×4, changed ×4, clip(final / 100, ±3) ×4 |
| R | hi (> 48) ×3, parity ×3, changed ×3 |

Goals are Gaussian weight vectors over φ; for P the cost weight is ≤ 0. There are 16 train goals
and 16 disjoint test goals.

**Decision metrics:** normalised regret = (U* − U_sel) / (U* − mean U); top-1 optimal rate; and the
oracle gap. The statistical unit is the parent, analysed with a parent-level paired bootstrap.

## Fidelity metrics

- P: per-bit accuracy and exact 8-bit match. Generation can be scored on the same metrics.
- L and R: binary-feature accuracy, and value MAE where it applies.
- Latent vs generative comparisons always use the same metric on the same test candidates.

## What would count

| Hypothesis | Criterion |
|---|---|
| H1 | Titan-Sparse latent exact-P at k = 4 (or per-feature accuracy) exceeds e2e generation by ≥ 20 pp, with a CI excluding 0 |
| H2 | Sparse − Supa ≥ +10 pp on latent k ≥ 2 metrics; randdelta ≈ Supa; the dose line is monotone or peaked |
| H3 | B3 − B2 regret < 0 with a parent-bootstrap CI excluding 0 on test goals, at ≤ 25% of train labels |

A clean null on any one of these is reportable.
