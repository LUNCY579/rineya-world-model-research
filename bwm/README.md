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
- **FINAL, frozen, run once: PlanBench Blocksworld-hard (110; 6-15 blocks).** World model + beam search
  solves **73/110 = 66.4%** (95% CI 56.7-75.1%); the same proposer without the world model solves 5/110 = 4.5%.
  Paired comparison: +68 / -0, p = 6.8e-21. VAL agrees on every plan. Every plan the system verified was valid.
  Training saw <= 6 blocks. Published zero-shot references (a different regime): o1-preview 23.6% (p = 2.4e-10),
  DeepSeek-R1 53.6% (two-sided p = 0.073, so not established). Details: `docs/DECISIONS.md` P-08.
- **RegisterWorld (F-01/F-03/F-04/F-06; P-03/P-04/P-05).**
  - The forward LoRA learns a data-forced shortcut for a parameterized legality rule.
  - Deferring to the frozen base model (adapter off), gated on the forward model's own confidence, repairs the soft
    errors without training: flips 0.32 -> 0.95.
  - Proposer coverage, not the world model, limited reranking (K = 12: 0.82 vs 0.67).
  - The proposer did not beat plain search under F at matched compute.

## Status (2026-10-10)

- **B2 complete (P-10), all frozen and run once:**
  - hard: 73/110 vs 5/110;
  - mirror: 61/110 vs 6/110;
  - PlanBench "600" set (602 instances): 602/602 vs 584/602.

  The pre-registered hard-minus-mirror contamination flag was **raised** (+10.9 pp, Fisher p = 0.13). A post-hoc,
  descriptive analysis attributes the gap to the mirror's taller goal towers; reweighted, the gap is -3.5 pp
  (`results/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json`).
- **Joint item 3 (PREREG-LB) closed:** RCWM's T2 ledger adapter, used as F inside this planner, plans: 127/180 vs
  14/180 with the adapter off, T2/ORACLE 0.794. That is on its training surface only. BWM re-scored it independently
  (D-61, `code/scripts/lb_rescore.py`).
- **B3 search distillation, MET (P-11).** The proposer kept its original training and was further trained, on
  half new rows, using generated 6-10-block plans its own world model had verified (no simulator labels). Without
  search it then solves 22/110 fresh hard-style problems. A same-training control solves 8/110 (paired +18/-4,
  p = 0.004); label precision was 1.0.
- **Registered hard-set runs (P-13, P-14).**
  - **H1:** the proposer alone, without search, solves 13/110 vs 5/110 (+10/-2, p = 0.039). The gain falls within
    the 6-10-block range it was trained on.
  - **H2:** the frozen B2 search with only the proposer swapped solves **88/110 = 80.0%** (CI 71.3-87.0%), vs 73/110
    (paired +20/-5, p = 0.004). Most of the gain is at 11-15 blocks.
  - **Comparison:** the published zero-shot reference (a different regime) is DeepSeek-R1 53.6% and o1-preview
    23.6%.
  - **No leak:** a two-audit check found no route from the hard set into the training data
    (`docs/research/H2_LEAK_CLAIMS_AUDIT.json`).
  - **Erratum:** "o1-mini 10%" is its verifier-in-loop score; direct prompting gives 0.9%.
- **D-54 key-order augmentation, PARTIAL (P-12).** Held-out key orders went from 0.45-0.50 to 0.92-0.95. The cost was
  the legality check on illegal actions.
- **5c retention:** both release adapters RETAINED (RCWM R25, independently re-scored by BWM, D-71), with caveats.
- **In progress:**
  - D-54 key-order augmentation (`docs/PREREG_D54_KEYORDER.md`);
  - B3 H1/H2 on Blocksworld-hard;
  - item 7 seed replication (`docs/PREREG_ITEM7_SEEDS.md`);
  - item 9 PlanBench Logistics, a draft prereg only (`docs/research/PREREG_LOGISTICS_DRAFT.md`).
- **Local package only:** `release/` in the bundle (D-49), not in the vault (D-60). Publishing anywhere public needs
  the user's explicit approval.
