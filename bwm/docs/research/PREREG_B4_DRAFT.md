# PREREG B4 (JOINT BWM-001 × RCX), DRAFT for BWM to edit and co-sign: does a better child-ranking hook improve BWM's frozen imagination beam planner?

| | |
|---|---|
| Status | **DRAFT** by RCX (2026-10-11). Not frozen. Co-signature by both sessions and the operator's go-ahead are required before anything runs. BWM owns execution (GPU turn), the instance set and the final text. RCX owns arm (a)'s module. |
| Tier | Planner with LM proposer and LM forward model (BWM). Arms (a) and (a0) add an *external* CPU heuristic module (tier B). Arm (b) is a model-native value adapter on Titan-Sparse. |
| Origin | RCX R-04 / R-07 (BWM X-02 / X-05). With exact dynamics and GBFS, BWM's default ranking key (unmet goal facts) is weak: one-step policy accuracy 0.06, and 0/75 at 200 expansions at n = 12–15. A GNN-value + misplaced_blocks hybrid solves 39/75 at the same budget. |

## 1. Question and claims

**Question.** Inside the frozen B2 `plan_beam` (sha `18a5f22b…`), with BWM's frozen F and proposer, does replacing the
default `unmet` key with a better state-ranking hook raise imagination-planning success on fresh hard-style Blocksworld
problems at the frozen budget?

**A positive result would establish** that the beam planner's ranking key is a binding constraint at E24, and that the
module (or BWM's native value adapter) relieves it. It would **not** establish:
- that the LM planner is improved without the external module (for arms a and a0);
- transfer beyond Blocksworld;
- any retained dense-model change.

## 2. Fixed components (BWM pins at freeze)

- **Planner.** `bwmn/blocks_planner.py` (sha 18a5f22b…) with the frozen B2 configuration from `FREEZE_B2_beam.json`:
  proposer K8, W4, E24, R12, max depth 80, M80, P1. BWM confirms the exact values at freeze.
- **Models.**
  - F: `BW_Fdelta_s0`, delta format.
  - Proposer: `BW_Pa_v2_s1`, the B2 primary.
- **Pa_v3 factor (secondary).** Crossing with `Pa_v3` happens only if BWM's H2 shows Pa_v3 helps the beam. That
  decision is made before this prereg freezes and stated here.
- **RCX module.**
  - `rcx/bw/b4_hook.py` `LearnedUnmet.from_file(runs/RCX-E07/main/models/gnn_s20261276.pt, invalid_policy="fallback",
    hybrid_lambda=1.0)`.
  - Shas of the model file, sidecar and hook code are recorded at freeze. The hook runs on CPU at 1 thread inside BWM's
    process; RCX's CUDA patch is undone, which is tested.
- **Caches.** BWM's F cache (BWM's path), and the RCX hook's own in-memory cache.

## 3. Arms (all are `unmet=` hooks; everything else identical)

| Arm | Hook | Purpose |
|---|---|---|
| **c** | default `len(goal − state)` (B2) | control (frozen B2 behaviour) |
| **c+pen** | default + RCX invalid-state penalty (1,000 + misplaced on a sanitised copy when F's imagined state is inconsistent) | matched control for (a): isolates the consistency filter |
| **a0** | `misplaced_blocks` + the same penalty | "any better classical key" control |
| **a** | RCX hybrid: GNN value + 1.0 · misplaced_blocks, with the penalty | treatment (RCX module) |
| **b** | BWM value LoRA V on frozen Titan-Sparse (h\* labels from the RCX E07 recipe; optionally a simulator-free variant) | treatment (model-native critic) |

## 4. Instances (BWM generates; fresh)

- **Size.** 110 hard-style full-configuration instances matching hard-110's size distribution (6–15 blocks), from a
  BWM seed (202614xx).
- **Exclusions:**
  - BWM's full recipe (hard-110, val_large, mirror, b3fresh, seeds_fresh, distill-240, PB-600 signatures, val_v2);
  - RCX's E07/E07b/E11 train and val goal classes at n ≤ 8, plus RCX's E07/E08/E07b test instances. This prevents arm
    (a) being scored on goal classes its GNN was trained on.
- **Reference values.** Optimal lengths from A\* where it is feasible (with a coverage report).

## 5. Endpoints

- **Primary family (Holm, 2 tests).** Paired success (VAL-verified, as `bw_select` reports it) at the frozen W4/E24,
  with the B2 proposer. Exact McNemar on the 110 instances:
  - **P1:** a vs c+pen;
  - **P2:** b vs c+pen.
- **Secondary (descriptive).**
  - Contrasts: a vs a0; a vs b; a0 vs c+pen; c+pen vs c (the effect of the penalty alone).
  - Smaller budget: E12 at W4.
  - Plan length vs optimal.
  - Cost: F-queries, proposer calls and expansions, as `bw_select` reports them.
  - Invalid imagined-state rate per arm (from the hook counters).
  - Success by block count.
  - If adopted, the Pa_v3 cross.

## 6. Decision rule (per primary, after Holm)

| Outcome | Condition |
|---|---|
| **SUPPORTED** | Holm-adjusted p < 0.05, and the treatment has more successes than c+pen |
| **NOT SUPPORTED** | otherwise |

The report gives each arm's success rate with a Wilson 95% interval. "Near-oracle" language is not used unless an
oracle arm is run (BWM's call: an exact-F oracle ceiling would be informative but is not required).

## 7. Execution

- **Runner.** BWM's GPU turn, under the lease and thermal rules. RCX supplies the module and the commands; BWM runs
  everything.
- **Outputs.** Per-instance outputs in BWM's run directory. BWM sends RCX the summary and per-item records (arm, n,
  success, F-queries, expansions, plan, hook counters).
- **Retries.** No retries under the same ID. A crashed arm is rerun from scratch under the same seed, and the rerun is
  logged.

## 8. Open items for BWM

1. Confirm the frozen B2 configuration values and file names.
2. Decide on arm (b)'s variant(s), and whether (b) joins the primary family.
3. Decide the Pa_v3 cross.
4. Choose the instance seed and count (110 proposed).
5. Decide whether to add an exact-F oracle ceiling.
6. Agree how to source the h\* labels for (b): RCX can export its labelling recipe, or produce labels for BWM's training
   instances under the agreed exclusions.
