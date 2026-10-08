# Training recipe / staged, not all-at-once

## T0: mechanics baseline

Frozen observed-state input -> action-token Transformer -> learned bottleneck -> measured features.
The CPU demo implements this. Its objective is supervised MSE on simple synthetic quantities.
It is NOT self-supervised discovery and does not use native Rineya representations.

## T1: frozen target consequence encoder

Define a transition record with actual outcome evidence. Build `target_z = stop_grad(E_target(record))`.
Run `pred_z = F(available_prefix, candidate_action_prefix)` without giving the predictor the record's
future outcome. Add observable decoder targets so a trivial latent cannot count as success.
First use a fixed encoder; later compare EMA targets with an explicit drift/caching protocol.

## T2: representation objective ablations

Compare predictive regression plus grounded anchors, predictive contrastive learning, and a
variance/covariance anti-collapse objective separately [S14,S16]. Same-state different-effect
branches make useful contrasts. Different actions with the same consequence are not automatic
negative pairs. Report target variance, rank and rare-failure separability.

## T3: sparse grounding

Freeze the encoder and fit a linear goal readout on nested identical labeled-parent subsets.
Compare a small nonlinear readout only as a separate arm. Goal text -> objective mapping is another
model whose correctness must be tested. Evaluate in-span and genuinely new-property transfer
separately. Ground truth comes from environment/verifier/preferences, not the predictor itself.

## T4: clustering and VQ

Fit codebooks only on TRAIN. Compare continuous features, k-means, soft assignment, random
projection and learned VQ under matched downstream capacity. Log code usage and rare-error merges.
The provided k-means utility does not implement VQ training [S15].

## T5: future targets

Compare direct cumulative-feature prediction to recursive rollouts. Log continuation policy or exact
plan identity. Match horizon and discount conventions. Include a terminal transition once; no future
contribution after genuine termination. Truncation stops observed data but does not establish zero
continuation value. Auxiliary Bellman-like consistency is added only after direct targets work.

## Optimizer admission

Freeze a parameter allowlist and check gradients against it. Use finite-loss/gradient checks, explicit
accumulation counts and actual token/transition exposure. Development-only tuning has a fixed budget.
Do not interpret an empty/null max-step budget as unlimited permission. Save seed and optimizer state
for reproducibility, and qualify exact native precision before a laptop run.
