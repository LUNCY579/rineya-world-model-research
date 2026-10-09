# BWM-001: learned world models and planning with a frozen 0.6B Rineya core

Private research records. Not a release. Maintained by the BWM-001 session; exported from
`Rineya_BWM_001_Research_Bundle/native_v1` by `tools/export_bwm.py`. Adapter weights, training data and caches are not
in the repo; each is listed with its size and SHA-256 in `EXCLUDED_LARGE_FILES.json`.

## Question

Can a frozen Rineya Titan-Sparse core (0.6B, FP32) plan better when it gets two LoRA adapters: a goal-blind world model
F, (state, action) -> next state, and a plan proposer? Planning happens *in imagination*: candidates are simulated by F,
and nothing is executed while planning. Two domains: RegisterWorld-v2, a synthetic arithmetic world, and PlanBench
Blocksworld.

## Where to start

| Read | For |
|---|---|
| `docs/DECISIONS.md` | Complete, dated log: decisions (D-), findings (F-, P-), incidents (I-) |
| `docs/PREREG.md`, `docs/PREREG_PLANBENCH.md` | Pre-registrations. Addendum B2 covers Blocksworld-hard. |
| `docs/RESULTS_INTERIM.md` | Interim summary (2026-10-06) |
| `docs/research/SOLUTIONS_AND_BENCHMARK_PLAN.md` | Literature sweep with fact-checked numbers; benchmark plan |
| `results/BW_TEST_B2/` | Frozen Blocksworld test runs (summaries plus per-instance plans, VAL-checked) |
| `code/bwmn/blocks_planner.py`, `code/scripts/blocks_eval.py` | Planner (beam search in imagination) and evaluation driver |

## Key results so far

Single training seed; validation-grade unless marked frozen.

- **The Blocksworld world model generalizes in size (F-05, F-07).** Trained on 4-6-block transitions, it predicts
  generated 8-, 10- and 15-block transitions exactly (delta format: 1.000 at 15 blocks).
- **The world model plus search rescues larger problems (P-06, selection).** On generated hard-style 6-15-block
  problems, beam search under the learned F solves 0.63, against 0.13 for the same proposer without F. Every plan the
  F verified was valid.
- **Frozen, PlanBench Blocksworld-hard (110 instances): the no-F arm solves 5/110 (4.5%).** The world-model arm is
  running in 8 shards (`results/BW_TEST_B2/hard_beam/`). Published zero-shot references: DeepSeek-R1 53.6%,
  o1-preview 23.6%. Those are general models in a different regime.
- **RegisterWorld (F-01/F-03/F-04/F-06; P-03/P-04/P-05).**
  - The forward LoRA learns a data-forced shortcut for a parameterized legality rule.
  - Deferring to the frozen base model (adapter off), gated on the forward model's own confidence, repairs the soft
    errors without training: flips 0.32 -> 0.95.
  - Proposer coverage, not the world model, limited reranking (K = 12: 0.82 vs 0.67).
  - The proposer did not beat plain search under F at matched compute.

## Status

The world-model arm's frozen runs on Blocksworld-hard, the mirror set and PlanBench-600 are in progress. A local
devfreeze package will follow (DECISIONS D-49). Publishing anywhere public requires the user's explicit approval.
