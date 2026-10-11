# PREREG RCX-E07b: can the learned relational value keep its low-budget advantage *and* close the high-budget coverage gap?

| | |
|---|---|
| Status | **FROZEN at commit time** (sha256 recorded in DECISIONS CX D-20). Written 2026-10-10 about 20:15 CDT, after R-04 and before any E07b code or run. |
| Tier | Mechanism, external specialist (tier B), exact dynamics, CPU |
| Motivation (R-04) | RelGNN reaches 0.998 policy accuracy OOD and is the best GBFS critic at ≤ 200 expansions. But it plateaus at 65/100 at 2,000 expansions, with misplaced_blocks at 87 and h_add at 89. All 35 failures are budget exhaustion on long n = 12–15 problems. |

## Hypotheses

- **H-b1 (data).** Adding n = 8 training instances with **full goals** (E08 uses full goals; R-04 trained on half partial
  goals and n ≤ 7) raises high-budget GBFS coverage.
- **H-b2 (objective).** A **successor-ranking loss** (hinge: an optimal successor scores below every non-optimal
  successor of the same state, margin 0.5, added to the Huber regression) raises coverage, because GBFS needs order,
  not calibration.
- **H-b3 (hybrid).** The key `h = GNN + λ·misplaced_blocks`, with λ ∈ {0.25, 0.5, 1.0} chosen on a validation instance
  set, keeps the GNN's low-budget advantage and gains misplaced_blocks' coverage.

## Arms

| Arm | Training |
|---|---|
| **A0 GNN-base** | the 3 R-04 models (seeds 20261276–78), reused unchanged |
| **A1 GNN-n8f** | as R-04, plus n = 8 instances: full goals only, the 10 largest-available fresh goal classes, 10,000 sampled states each, exact h\*. Same epochs and time cap. Seeds 20261292–94. |
| **A2 GNN-rank** | the R-04 data, plus the ranking hinge on every training state with h\* > 0 (its successors and their exact h\*). Seeds 20261295–97. |
| **A3 GNN-hybrid** | A0 models, with λ selected on VAL-B (below) by success@2000, ties broken by success@200. Untrained. |
| **Baselines** | misplaced_blocks, h_add, unmet_goal_facts; plus A3's λ = 0 (≡ A0) |

## Instances (all fresh, exclusion-checked with EXCLUSIONS.json at sha 2f211de3… plus the selection sets; disjoint from R-04's E08 set by instance key)

- **VAL-B** (λ and arm selection only): seed 20261291, n ∈ {12, 14}, 15 each, full goals.
- **TEST-B**: seed 20261290, n ∈ {10, 12, 14, 15}, 25 each = 100 instances, full goals.
- **Seeds:** 20261290–99 are reserved for E07b. Model seeds are listed above, 20261298 is spare, and 20261299 is the
  bootstrap seed.

## Endpoints

- **Primary.** GBFS success at max_expansions = 2,000 on TEST-B n ∈ {12, 14, 15} (75 instances).
- **Primary comparison.** `BEST` vs misplaced_blocks, by exact McNemar on paired instances.
  - `BEST` is the arm among A1, A2 and A3 with the highest VAL-B success@2000 (ties broken by success@200), using seed 1
    of each trained arm.
- **Secondary (descriptive):**
  - success@{50, 200, 500} for all arms;
  - for trained arms, the seed mean ± sd;
  - plan suboptimality;
  - expansions;
  - E07-style policy accuracy at n = 8, 10, 12 on R-04's E07 test states for the trained arms;
  - A0 vs A1/A2/A3 by McNemar at 2,000 and at 200.

## Decision rule (primary)

| Outcome | Condition |
|---|---|
| **CLOSED** (gap closed) | `BEST` success@2000 ≥ misplaced − 3 (of 75), **and** `BEST` success@200 > misplaced success@200 with exact McNemar p < 0.05 |
| **IMPROVED** | `BEST` success@2000 − A0 success@2000 ≥ +8 with McNemar p < 0.05, without meeting CLOSED |
| **NOT IMPROVED** | otherwise |

Multiplicity: one primary test. Everything else is descriptive.

## Execution

- **Compute.** CPU, single lane, ≤ 4 threads, guard on. Estimated at about 1–1.5 h total.
- **Outputs.** Written to `runs/RCX-E07/e07b`, never into an old directory.
- **Code freeze.** Code may be written after this freeze but must not change the endpoints. Any deviation is logged in
  DECISIONS.
