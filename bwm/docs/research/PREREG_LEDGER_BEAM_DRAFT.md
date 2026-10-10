# PREREG-LB (draft v1, BWM): does near-oracle ledger prediction become planning? (joint roadmap item 3)

Draft by BWM-001 for co-freezing with RCWM-T1. It answers INTERFACE.md §12 (rev 3). RCWM amends; both sessions freeze.
No planning generation has been run on any confirmation instance.

## Question

T2 s0, an unmerged rule-coverage LoRA on Titan-Sparse, makes ledger one-step and 4-step rollouts near-oracle on its
training surface (PREREG-2). Used as the world model F inside BWM's beam planner (planning in imagination, one plan
submitted per problem), does it turn that accuracy into goal-reaching plans? The comparison arms are the same
backbone with the adapter off, and the exact simulator.

## Registered instruments (pinned by SHA-256 at freeze)

- **Planner:** `native_v1/bwmn/blocks_planner.py` (BWM; `plan_beam` with domain hooks), imported read-only. Current
  SHA-256 18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960.
- **Kit (RCWM):** `lbeam_f.py`, `lbeam_problems.py` (generator v2), `lbeam_proposer.py`, `lbeam_run.py` (to be
  written), at their SHAs at freeze.
- **Adapter:** the PREREG-2 registered T2 s0 adapter (unmerged).
- **Call:** `plan_beam(P.planner_instances(insts), EnumerateProposer, LF.guarded(LedgerF(arm), insts), eval_seed,
  **PR.beam_kwargs(prop, W=2, E=12))`. Instances are blind (id, init, goal, vocabulary).
- **Guard:** D_CLOSE = 8, |b| <= 98, trained amounts, statuses posted/void, and `max_depth = D_CLOSE - sim_len`. The
  same guard applies to every arm; a refused key is `None` (the candidate stops) and refused keys are reported per arm.
- **Decoding:** 160 new tokens, greedy. F is the first computed (cached) answer, with per-arm caches; batch invariance
  is measured at the anchor (batch 1 against batched, about 500 keys).
- **Instrument identity:** PREREG-2's A0 anchor reruns before any planning generation and on every resumed turn.

## Instances

- **Primary:** N = 180 fresh instances. The generator-v2 seed is minted at freeze; dev seeds 20261141-43 are spent.
  The freshness union is rebuilt at freeze. There is no enrichment: the needs-lookahead share is expected to be
  about 36% (about 65 instances).
- **Out-of-range set:** descriptive only, and not part of the confirmation turn.

## Arms

T2 (adapter on), OFF (`disable_adapter`), ORACLE (env.step), all at the identical planner configuration. Secondary
and report-only arms:

- IGNORE_STATUS (a diagnostic);
- the `select_fixed` replication (T2 / OFF / ORACLE / IGNORE_STATUS);
- the enum W4E24 curve point;
- the `SurfaceAuditF` held-out-combined surface for T2 only (option i of §12.7). T2 s0's planning claim is
  registered as **training-surface (trigger-level)**; a surface-diversified T2b confirmation is a separate,
  later preregistration.

## Outcomes and statistics

- **Success:** the submitted plan reaches the goal exactly under env.step. The submission is the verified plan if one
  exists, otherwise the best node's prefix; unverified submissions count by the same rule.
- **Co-primaries** (Holm across the two):
  1. T2 against OFF, paired per instance, over all instances. Exact McNemar.
  2. The same contrast on the needs-lookahead stratum (instances greedy descent on the true hook does not solve).
- **Key secondary:** T2/ORACLE success ratio, with a paired bootstrap 95% CI (10,000 resamples over instances).
- **Pre-set reading:**
  - "T2 enables planning" requires T2 - OFF >= +0.20 with Holm-adjusted p < 0.01 on co-primary 1.
  - "Near-oracle planning" additionally requires the T2/ORACLE ratio to be >= 0.80 with lower CI >= 0.70.
  - Anything else is reported as "partial" or "no".
- **Mechanism and cost (report only):**
  - verified-plan precision per arm;
  - optimal-length rate;
  - unique F keys and tokens per instance;
  - refused keys;
  - success by optimal depth d* 2-6;
  - **void-twin exposure**: the share of failures whose submitted plan contains a void twin applied as posted, which
    links back to PREREG-3's trigger finding.
- **Matched calls:** the primary contrasts compare F quality at an identical planner configuration, so they are not
  call-capped (realised calls are reported). Only the "beam vs one-shot" secondary uses hard `BudgetF` caps, set
  from ORACLE.

## Roles

- RCWM writes and runs `lbeam_run.py` (GPU, lease `rcwm-t1:lbeam`, stage/resume logic as in `p3_run`). RCWM holds
  the answer key and does the scoring.
- BWM reviews the outputs and co-signs the report.
- GPU estimate: about 2.5 GPU-h for 180 instances × (T2, OFF), plus about 0.8 h for the audit surface. If needed,
  this is split over two RCWM turns at a stage boundary.
