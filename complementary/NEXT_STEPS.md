# NEXT_STEPS (RCX)

Status: 2026-10-12, about 08:20 CDT. Items are listed in single-CPU-lane order. The CPU lane is shared with BWM and
dsv3wm through the lock file (CX D-36).

1. **RCM-001: RCX-E01R confirmation, branch B4 (R-10, CX D-39).**
   - **a. Done.** The full freeze is CX D-41 (prereg `16e959c5…`, code digest `61ed4df6…`, rows, and audit with
     tokens). The SEALED commitment was minted before any training (CX D-42, `d4635e42…`).
   - **b. Training chain running** (`runs/RCX-E01R/cells/train_chain.sh`). It waits for ≥ 12.5 GB of free RAM and
     the CPU lane. GPU-idle night windows (about 23:00–07:00) were requested from BWM and RCWM, with operator
     approval.
   - **c. Train and smoke.** Train 6 adapters (E-R2 and E-R3, seeds 0–2; about 2.6–3.2 h each on CPU), then smoke each
     on 50 TRAIN prompts.
     - Each block needs ≥ 12.5 GB free RAM, so the blocks are scheduled with BWM and RCWM (GPU-idle windows or nights)
       and run under the lane lock.
   - **d. Evaluate and analyse.** `open-sealed`, then the sealed evals: N-CFSD s0–s2 plus 6 treatments, at `max_new`
     ≥ 160. Then the single `analyze_e01r.py` run.
     - On ADVANCE: request E33 from RCWM via `docs/REQUESTS.md`, and Q-03 cross-scoring.
2. **B4 screen co-sign with BWM.**
   - Canary delivered: `experiments/RCX-B4/canary/B4_HOOK_CANARY.json`, sha `9e9ad339…`.
   - BWM sets `CANARY_SHA256`, applies the final low-severity round, and runs its preflight. Its torch build differs,
     so watch the tolerance.
   - RCX then re-checks the final shas (finalisation may only drop "(DRAFT)") and co-signs.
3. **SUPA probe arm** (R-03 completeness), at a quiet RAM moment (needs ≥ 12 GB free).
4. **RVM-001.**
   - E04 legality critic on `traces.bw_parts` minimal pairs.
   - E05 successor features for goal-weight re-ranking.
5. **RSM-001 follow-ups (R-09).**
   - B5, the macro arm in BWM's beam, after B4. The library is exported.
   - E11c: execution-aware subgoal labels.
6. **SymLM mechanism tier.** Low priority.
