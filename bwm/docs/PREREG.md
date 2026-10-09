# PREREG — BWM-001 native pilot on Titan, RegisterWorld-v2 test evaluation

Status: **DRAFT** (to be frozen after the validation pilot and before any test-split episode is run; the freeze
records this file's SHA-256 in DECISIONS.md). Thresholds come from Source_of_Truth §24–25 unless stated.

## Subject

Frozen `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` (FP32), with three unmerged LoRA adapters:
- F_LM `runs/F_titan_main/adapter`, a goal-blind forward model;
- one action-only proposer per seed;
- one state-and-action proposer per seed.

Runtime: python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0, TF32 off.

## Units, splits, seeds

- Parent = (start, goal) family. One task per parent per split.
- Goal-held-out test (20% of goal values; no test goal appears in any training row). Splits: ID 1,000, OOD-long 530,
  OOD-range 600, novel-limit-only 300, infeasible twins 300, budget+4 control 300.
- Seeds: 5 proposer seeds (data seed = training seed); F fixed.
- Inference: K = 4 (candidate 0 greedy; 1–3 sampled at T = 1.0, top-p 0.95, per-candidate seeds);
  execute one action and replan. Primary F rule: `rerank`.

## Endpoints

**Primary (P1, BWM spec):** ID executed success, A11 − A01.

**Co-primary (P2, program claim):** ID executed success, A11 − G2F. G2F is greedy-2 under the same F_LM, the strongest
model-free heuristic using the same learned world model.

Decision rule per primary (Holm across P1, P2):
- point estimate ≥ +0.05;
- crossed seed × goal-cluster bootstrap 95% lower bound > 0;
- ≥ 4 of 5 seeds positive.

**Confirmatory secondary family (Holm, α = 0.05):**
- S1: A01 − A00 on ID;
- S2: A11 − A10 on ID;
- S3: A10 − A00 on ID;
- S4: A11 − A01 on OOD-range.

**Exploratory, unadjusted:**
- all contrasts on OOD-long, OOD-range and novel-limit-only;
- gate operating points (A01g/A11g) with coverage and success given answered;
- infeasible-twin abstention signals (AUROC);
- outcome-flip and budget+4 controls;
- shuffled-state control (A11 vs A11shuf);
- filtered BC;
- token- and F-call-matched curves (A01k8/k12, BFS16/64/256 under F);
- open-loop success;
- first-action optimality;
- claimed-state mismatch.

## Reporting rules

- Every arm is reported against the floors measured on the same task sets.
- Success abstentions count as failures. F-call counts use per-episode accounting.
- Nonsignificance is not equivalence. Single-seed numbers are labelled as such.
- Failed runs stay preserved under `runs/_failed` and are listed in the report.
