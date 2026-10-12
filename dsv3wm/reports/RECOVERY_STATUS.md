# Registered-interface recovery: interim evidence

Recorded 2026-10-12 UTC. Full research goal remains open.

The native 596,049,920-parameter Titan-Sparse parent loaded in FP32, TF32 off,
with the unmerged registered `F_titan_main` adapter. Adapter SHA-256:
`c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466`.
Rank 16, alpha 32, dropout .05, q/k/v/o/gate/up/down. Every original one-step
prompt is byte-identical to `rw2`. No core or owner artifact was modified.

Fresh recovery manifest SHA-256:
`20c593a683fed07e21c226fa77189b1b4be7cea92c186ff27881c4c21d65b9fe`.
TRAIN has 128 independent parents. DEV has 12 ID parents using seen one-step keys
and 12 unseen-limit parents at limits 80/96, including six starts outside inherited
numerical support. The ID cohort tests new compositions of seen transitions.
This is not proof of unseen parent-backbone arithmetic.

| B0 metric | ID | Unseen limits | All |
|---|---:|---:|---:|
| Free-running exact state h1 | 12/12 | 12/12 | 24/24 |
| Free-running exact state h2 | 12/12 | 12/12 | 24/24 |
| Free-running exact state h4 | 12/12 | 12/12 | 24/24 |
| Direct h1 exact state | 12/12 | 12/12 | 24/24 |
| Rollout legality h1/h2/h4 | 12/12 each | 12/12 each | 24/24 each |
| Malformed replies | 0 | 0 | 0 |

Qualification took 88.196 seconds including loading/hashes, with 120 learned
queries, 14,246 prompt tokens and 1,789 output tokens. The conservative processed
position upper bound is 15,960. These are native CPU measurements. Direct h2/h4
was not measured in B0's qualification schedule. Planning and behavioral
retention are pending.

Source: `results/recovery-v1/B0_qualification.json`, prospectively frozen by
`prereg/RECOVERY_FREEZE_V1.json`. The posthoc supplement reconstructs all 120
raw replies without transition execution and verifies original P1 unchanged.
State, legality, termination and cumulative attempted-failure are correct for
every parent at every step. No correct-final/wrong-intermediate cases occurred.
Calibration is unknown because probabilities were not captured.

This fixes the initial screen's interface/undertraining qualification failure.
It does not establish H1/H2/H3. The fixed 24-parent P1 cohort is at ceiling for
B0. A1/A2 beating a degraded A0 would show adaptation preservation, not accuracy
improvement beyond inherited F. No parents will be redrawn to manufacture a gain.

Warm A0/A1/A2 32-step and A5 96-step recipes are frozen in
`RECOVERY_TRAIN_FREEZE_V2.json`. No adapted recovery training has executed at
this checkpoint; the CPU slot is temporarily reserved for coordinated RDW E05.
The next step is training, prospective trained-artifact readout freezes, and
fixed dynamics/planning/behavioral retention evaluation.

V1's editorial `term` spelling was clarified to actual `terminated` before B0
results were read and before recovery training. Explicit AdamW defaults/no
scheduler are bound for all arms: this is a new controlled recipe, not exact
historical retraining. Original V1 trainer bytes are preserved by hash under
`prereg/source_snapshots/`.

Original cold rank-4 pilot records remain unchanged: four arms returned state
0/2 and planning 0/1, so that screen could not discriminate efficacy. Recovery
changes interface, capacity and initialization together and does not isolate
the cause of the prior failure. H4/H5 remain conditional; no integration or
confirmation claim follows from recovery.
