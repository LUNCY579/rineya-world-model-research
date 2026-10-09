# PREREG addendum — PlanBench Blocksworld (Track B)

Status: **DRAFT, written before any BWM system has been run on a PlanBench instance.** The freeze records this file's
SHA-256 in DECISIONS.md together with the frozen adapter hashes and planner settings.

## System under test (one frozen configuration per run)

- **Backbone and adapters:** frozen Titan-Sparse (FP32) with two unmerged LoRA adapters trained only on our
  generator's instances (`runs/BW_data`, PlanBench excluded up to block renaming):
  - `BW_F`: a goal-blind forward model (state, action) -> {legal, state};
  - `BW_Pa`: an outcome-conditioned action-only proposer.
- **Planner (`bwmn/blocks_planner.plan_instances`):**
  - Plans **without execution feedback**; the exact simulator and VAL are never consulted while planning.
  - Candidates: K proposals per imagined state (candidate 0 greedy, 1..K-1 sampled at T = 1.0 / top-p 0.95 with
    per-candidate seeds).
  - Each candidate is rolled out under BW_F. The shortest candidate predicted legal and goal-reaching is submitted.
  - Otherwise the planner advances one imagined step along the best-scoring candidate, up to a step cap of 24.
  - **Exactly one plan is submitted per instance.**
- **Selection:** K, the step cap and the training checkpoint are chosen on our generated validation set
  (`instances_val.json`) only.

## Test set and scoring

- **Instances:** the 600 PlanBench Task-1 Blocksworld instance IDs present in the published result files
  (500 `generated_basic` + 100 `generated_basic_3`), run once per frozen configuration.
- **Scoring:** plans are scored by the local VAL binary, the same binary used to re-score every published plan
  (`bench/planbench/LEDGER.json`).
- **Statistics:**
  - success rate with a Clopper–Pearson 95% CI;
  - paired exact McNemar against each published system on identical instance IDs;
  - strata by block count and optimal plan length.
- **Contamination control:** a fresh set of 600 instances from our generator, signature-disjoint from training and
  from PlanBench, scored identically. A PlanBench–fresh gap above 5 pp is reported as possible contamination of the
  backbone's pretraining.

## Comparators

1. **Same-harness local arms (fully comparable: same instances, VAL, machine, FLOPs accounting):**
   - frozen Titan without adapters;
   - stock Qwen3-0.6B;
   - Qwen3-4B non-thinking, and Qwen3-4B thinking at fixed token budgets on a stratified subset, run only when
     >= 12 GB RAM is available and no other CPU lane is running.
2. **Published systems (reference; paired by instance, judged by the same VAL), zero-shot Task 1:**

| System | Score (/600) |
|---|---:|
| DeepSeek-R1 | 595 |
| o1-preview | 587 |
| LLaMA-3.1-405B | 386 |
| Claude 3 Opus | 356 |
| o1-mini | 340 |
| GPT-4o | 213 |
| GPT-4 | 210 |
| LLaMA-3-70B | 205 |

## Comparability ledger (stated up front)

- **Regime:** BWM is a domain-trained specialist with a learned simulator and search. The published models are
  zero-shot generalists. Beating GPT-4 or LLaMA-405B is expected and is not claimed as news. Fast Downward solves
  600/600.
- **Input:** our systems read PDDL parsed to compact JSON by code. Published prompts were natural language, and the
  2024 plans were translated to PDDL by GPT-4o.
- **Compute:** API-model compute is unknown (o1's reasoning tokens are hidden), so FLOP-matched claims are made only
  against local Qwen3 models.
- **Allowed headline forms:**
  - (a) "a frozen 0.6B Rineya + 10M-parameter adapters, planning without feedback, reaches X/600 (CI) on PlanBench
    Blocksworld, paired against published systems";
  - (b) iso-FLOP non-inferiority (delta = 5 pp) against Qwen3-4B-thinking on the same harness.
- **Not allowed:**
  - "beats GPT-4 at planning" without the regime caveat;
  - any claim from VAL-in-the-loop runs (those compare only to LLM-Modulo results);
  - Mystery Regime A described as reasoning from prompted rules.

---

# Addendum B2 — PlanBench Blocksworld-hard (written 2026-10-07, before any BWM model saw a hard instance)

Status: **DRAFT.** Only the exact-simulator harness test (D-44) has touched the hard instances: no model was run on
them. The freeze records this file's SHA-256 with the adapter hashes and the selected settings in DECISIONS.md.

## Why this test

- The 600-instance set (3-5 blocks) is close to saturated for domain-trained systems (plain SFT reaches ~90-99% in
  the literature on its own generators). A score there is informative only paired against our own no-F arm.
- **Blocksworld-hard** (110 instances, 6-15 blocks, goals = whole target configurations with 5-12 on-pairs,
  optimal plans 20-40 steps, `bench/planbench/blocksworld_hard`) is where SFT planners trained on small problems
  collapse and where the strongest published zero-shot systems are far from ceiling.
- Our models are trained **only on 3-6-block problems** (BW_F: v1 transitions; Pa: v2 proposer rows, D-43).

## Systems (one frozen configuration each; all plan without execution feedback, one plan per instance)

- **S-beam (primary):** frozen Titan-Sparse + `BW_F_s0` + `BW_Pa_v2_s1`; `blocks_planner.plan_beam`.
- **S-direct (internal no-F arm):** the same Pa, candidate 0 (greedy) submitted as is (`plan_direct`).
- **S-replan (secondary):** `plan_instances` (D-23 planner) with step cap 80.
- **Selection:** K (proposals per node), beam width W, expansion cap E and max depth are chosen on generated sets
  only: `runs/BW_data_v2/instances_val.json` (400, 3-6 blocks) and `runs/BW_val_large/instances_val_large.json`
  (120, 6-15 blocks, hard-style goals, hard instances excluded by signature/invariant). Objective: mean success
  over the two sets subject to <= 2,000 unique F queries per instance on average. The same frozen configuration is
  then run once on the 600 set and once on the 110 hard instances.

## Metrics and statistics

- Success = plan accepted by the local VAL binary on the instance PDDL. Clopper-Pearson 95% CIs. Strata by block
  count and optimal length (from `info.json`).
- Paired exact McNemar: S-beam vs S-direct on the same 110 (and 600) instances. This is the F + search effect.
- Compute per instance: proposer generated tokens, unique F queries and tokens, total forward FLOPs
  (2 x non-embedding params x processed tokens, attention term included as in `bwmn/flops.py`), wall time.
- Contamination/generalization control: `runs/BW_val_large/instances_mirror_hard.json` (110 generated instances
  with the hard set's block-count distribution) scored identically; a hard-mirror gap > 10 pp is reported.

## Published references (aggregate only; different regime: zero-shot generalists, no domain training)

| System (zero-shot, Blocksworld-hard, 110) | Success |
|---|---:|
| DeepSeek-R1 (PlanBench leaderboard / TMLR 2025) | 53.6% (~59/110) |
| o1-preview (TMLR 2025, Table 5, direct prompting) | 23.6% (26/110) |
| o1-mini | 10% |
| o1-preview + VAL feedback loop (LRM-Modulo; verifier-in-loop, NOT comparable) | 98.2% |

## Allowed claims

- "Trained only on <= 6-block problems, a frozen 0.6B Rineya with two LoRA adapters (~20M parameters), planning
  without feedback, solves X/110 Blocksworld-hard instances (CI); published zero-shot reference: R1 53.6%,
  o1-preview 23.6% (different regime)."
- "Adding the learned world model and search changes success from S-direct to S-beam by Y (paired McNemar)."
- Not allowed: "beats o1/R1 at planning" without the regime caveat; any comparison to verifier-in-loop numbers;
  any claim from a configuration selected on hard instances.
