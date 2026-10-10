# DECISIONS: RCX complementary session

Each entry records the timestamp (CDT), the decision, the reason, whether outcomes were already known when it was made,
and the evidence path.

## 2026-10-09

- **CX D-01 Branch, worktree and folder (22:20).**
  - Created branch `research/rineya-complementary-intelligence-v1` from `origin/main` at `a223884`.
  - It lives in a separate worktree, `WorldModel/rineya-complementary-worktree`. The shared checkout
    `rineya-world-model-research` stays on `main` and is not touched.
  - All RCX files live under `complementary/`.
  - A read-only, detached worktree of `origin/research/rineya-high-quality-dataset-v1` was added in this session's
    scratchpad for inspection only. It will be removed with `git worktree remove` when no longer needed.
  - Outcomes known: n/a.
- **CX D-02 Understanding pass (22:25).**
  - Seven parallel read-only readers covered: the bundle core, tracks, protocol, RCWM, BWM, the dataset factory, and
    coordination.
  - A synthesis was then written: `docs/understanding/SYNTHESIS_2026-10-09.md` and the per-area reports.
  - No peer file was modified.
- **CX D-03 Scope split with peers (22:35).**
  - RCX takes **rule-conditioned execution** for RCM-001: rules supplied in context and varied, with opposite meanings,
    nonce words and counterfactual twins.
  - It does **not** take fixed-rule surface diversification, which is RCWM T2b 5a/5b.
  - **Reason:** RCWM PREREG-3 (REPORT §3c) shows two things:
    - T2 learned a `void` trigger;
    - "neither base model reads the rule text" (status sensitivity 0.00–0.03).
  - T2b repairs the fixed rule's surface. Nobody tests whether a model can *read and execute a supplied rule*, which is
    the bundle's RCM research question.
  - E10 search distillation is BWM's B3, so RCX does not run it.
  - Joint item #6 (legality-checker LoRA) remains a peer item unless handed over.
  - Requests were sent (`docs/REQUESTS.md`, CX Q-01, Q-02).
- **CX D-04 Compute posture (22:35).**
  - CPU only. GPU is leased by BWM (bundleD, until about 02:35 on 10-10) with RCWM waiting (5a, 3.8–6.1 GPU-h), then
    BWM B3.
  - Host RAM available is about 5–7 GB, below the 12 GB CPU-lane load floor (D-39). So no 0.6B model is loaded on CPU
    now.
  - tz01 is 94–96 °C against the shared 98 °C GPU-pause threshold. So RCX limits itself to 4 threads and pauses at tz01 ≥
    97 °C.
- **CX D-05 Labels (22:35).**
  - Programme label RCX; decision IDs `CX D-nn`; requests `CX Q-nn`; runs `RCX-Exx`.
  - Reason: `RCI-001` already names a spent calibration bank (`rineya-dataset-retention-records/rci001-20260918`).
  - The seed range is `202612xx`, disjoint from BWM (20261005–11), RCWM (20261121, 20261141–43) and the factory
    (20261009).
- **CX D-06 RCWM confirms the scope split (about 22:55).**
  - RCWM's reply to CX Q-01 (`docs/REQUESTS.md`): no rule-varying ledger arm on RCWM's side, and imports are allowed
    with sha pins.
  - **Binding for RCX:**
    - never call `rcwmt.bank.load_exclusions()` or touch `HSKILL_CONFIRM_V2*`;
    - PREREG-3's 16 strings never appear as RCX statuses (already enforced by `vocab.EXCLUDED`);
    - any future Titan/Supa training data must exclude RCWM's 5a/5b held-out paraphrase #4, field-name variant #4 and
      held-out key orders.
  - Retention confirms for RCX native candidates go to RCWM.
  - Qualification check Q1 already passed against the pinned files: 20,000 posted-rule cases, with
    `world.transition == env.step == p3_common.rule_gold`.
