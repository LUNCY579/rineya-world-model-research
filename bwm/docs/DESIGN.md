# BWM-001 native pilot on Titan — design draft v0 (pre-critique)

Status: DRAFT. Nothing below has been run. Amendments are recorded in `DECISIONS.md`.

## 0. Goal

Run the BWM-001 core factorial natively on a Titan backbone, on a task distribution with real
planning headroom, and extend it toward the result the Rineya program lacks: a natively measured,
paired, multi-seed result that a frozen backbone plus an adapter turns a one-step transition skill
into multi-step competence that transfers, with zero backbone retention cost by construction.

## 1. Subject and runtime

- Backbone: `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` (FP32, sha a28629c9…). It stays FP32 and frozen.
  Adapters are never merged.
- Runtime: python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0, RTX 4070 Laptop 8 GB. TF32 off.
- Prompt frame: Titan-native chat template (`system` rule text + `user` compact JSON; thinking disabled; the
  generation prefix ends `<think>\n\n</think>\n\n`). The target is compact JSON + `<|im_end|>`. This deviates
  from the spec's raw `bwm-json-raw-v1` protocol so that Titan's trained frame is engaged (decision D-01).
- Adapter recipe (Rineya-validated): LoRA r16, alpha 32, dropout 0.05, on q,k,v,o,gate,up,down
  (10,092,544 params). AdamW betas (0.9, 0.95), lr 1e-4 peak, WSD 10/70/20 to 0.1×, clip 1.0. Loss on the
  target + EOS only. Gradient checkpointing.

## 2. Domain: RegisterWorld-v2 (RW2)

The semantics are the same as v1 (`exact_step`): add1, sub1, double, halve (illegal if odd), and finish (always
terminates). A move to |x| > limit is illegal and leaves x unchanged. Success = finish while x == goal within the budget H
(finish counts as a step). Reward: success 1, step −0.01, violation −0.1.

What changes is the difficulty. Budgets are tight: H = opt_moves + 1 + slack, with slack ∈ {0, 1, 2}. v1's 95.9%
greedy floor came from loose budgets.

- A parent family is the pair (start, goal, limit). All budget variants and trajectories of a family stay in the
  same split. The split is a hash of the family (70/10/10/10 train/validation/calibration/test).
- ID: limit ∈ {32, 48, 64}, |start|, |goal| ≤ limit, opt_moves ∈ [1, 8].
- OOD-long: same limits, opt_moves ∈ [9, 12]. These families are excluded from training entirely.
- OOD-range: limit ∈ {96, 128}, max(|start|, |goal|) > 64, opt_moves ∈ [1, 8]. No number above 64 appears in training.
- Infeasible: ID-like families with H = opt_moves + 1 − d, d ∈ {1, 2}. The correct behaviour is abstention.

Floors (exact one-step dynamics, model-free; measured on 1,200 sampled tasks at limit 64, slack 0–2):

| split | greedy-1 (exact) | greedy-2 (exact) |
|---|---:|---:|
| ID opt 1–8 | 0.639 | 0.778 |
| OOD-long opt 9–12 | 0.466 | 0.597 |
| OOD-range (L128) | 0.520 | 0.655 |

These floors use the EXACT simulator and are oracle heuristics. They are reported, not beaten by fiat.

## 3. Components

- Proposer Q (LoRA on frozen Titan). User JSON: {"budget":H_remaining,"goal":G,"limit":L,"outcome":"success|failure","x":X}.
  - action-only target: {"actions":["double","add1","finish"]}
  - state-and-action target: {"steps":[{"action":"double","x":12},{"action":"add1","x":13},{"action":"finish","x":13}]}
  - Both families train on identical rows, order and step count. Only the target rendering differs.
- Forward F (goal-blind). Primary is F_LM: a separate LoRA on frozen Titan. User {"action":A,"limit":L,"x":X} →
  {"legal":bool,"terminated":bool,"x":X'}. It never sees goal, outcome or budget. Alternatives are F_MLP (a one-hot
  sidecar for ID only; it abstains on unsupported states) and F_exact (oracle diagnostic).
- Evaluator: the exact executor.

## 4. Data (train families only)

Per family: one shortest plan (random tie-breaking among optimal next actions), one noisy plan (one action
replaced), and one random action sequence of length H. Every suffix becomes a proposer row (from x_t, remaining budget
H−t, outcome = achieved outcome of that suffix). Forward rows are every executed transition, including
illegal and finish transitions.

## 5. Inference loop (every arm)

At each real step, from the real state and remaining budget, generate K = 4 complete-plan candidates conditioned on
outcome=success. Candidate 1 is greedy; candidates 2–4 are sampled (T = 1.0, top-p 0.95). They are generated in
one batch and always charged.

- No-F arms (A00, A10): execute the first action of the first schema-valid candidate in generation order.
- F arms (A01, A11): roll out each valid candidate's ACTIONS under F from the real state. Predicted success =
  the plan ends in finish with predicted x == goal within the remaining budget. Utility = success − 0.01·len − 0.1·illegal.
  Take the argmax among predicted successes (ties go to generation order). If no candidate is predicted to succeed,
  ABSTAIN. Claimed states in A11 are logged, never used (state-consistency rejection is a separate ablation).
- Execute one real action and replan. Finish ends the episode. Abstention or no valid candidate = failure.
- F queries are memoised on (x, action, limit) because F is greedy-deterministic. Total and unique calls are both reported.

## 6. Arms and baselines

Factorial: A00 / A10 / A01 / A11, three proposer seeds per family (6 trainings), one frozen F_LM.

Baselines:
- frozen Titan zero-shot with the same prompt (K = 4, ±F);
- greedy-1/greedy-2 with exact dynamics (floors);
- greedy-1 with F_LM (heuristic proposer + same planner);
- random proposer + F;
- action-only best-of-K' + F at matched generated tokens (K' ≈ 3K);
- A01/A11 with F_exact (headroom);
- success-only SFT action proposer (filtered BC) + F.

## 7. Endpoints and statistics

- Primary: ID-test feasible success, A11 − A01, mean over 3 seeds. Paired by parent; crossed seed×parent bootstrap
  95% CI; per-seed exact McNemar. Promotion criterion (spec): ≥ +5 pp with a CI lower bound > 0.
- Secondary: A01 − A00, A11 − A10, the interaction, A10 − A00; all of these on OOD-long and OOD-range; infeasible-task
  abstention rate (and false abstention on feasible tasks); optimality (steps used vs opt); F one-step accuracy
  (exhaustive (x, a, L) tables, ID and OOD) and rollout accuracy; tokens and F calls; wall time.
- Sizes: ID test 400 parents, OOD-long 300, OOD-range 300, infeasible 150. One budget variant per parent.

## 8. Rineya-specific extensions (after the core)

- E1 Backbone: repeat with Supa 1.7 as the backbone (FP32-upcast). Does Titan's transition prior help?
- E2 F sample efficiency on Titan vs Supa (128/512/2048 transition rows).
- E3 Titan-native domain: STRIPS prerequisite_plan from Forge v2 (Dijkstra oracle, IMPOSSIBLE outcomes, legality).
- E4 Verified-plan distillation: one round of F-verified, executor-checked self-generated plans on train parents.
- E5 Magnitude pruning (top-25% per tensor) of the proposer delta: planning kept? Titan's transition skill
  (SELECT-VIEWS, report-only) damaged less while the adapter is active?
