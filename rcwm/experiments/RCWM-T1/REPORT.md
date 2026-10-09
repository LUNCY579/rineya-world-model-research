# RCWM-T1 report: consequence models from a frozen 0.6B Rineya LM

**Date:** 2026-10-06
**Status:** PREREG-1 **confirmed** (2026-10-07; §3). PREREG-2 **confirmed** (2026-10-08; §3b). The L8-15 promotion screen (PREREG-L1) ran on 2026-10-09: **DO NOT PROMOTE**, failing C2 (prerequisite acquisition −8 vs Titan-Sparse; DECISIONS R-L1).

This is the first native execution of the RCWM-001 plan (`../../report/`) on Rineya Titan models. Confirmatory statements are limited to PREREG-1 (§3). Everything else is exploratory, from the pilot bank. Supporting documents:
- full numbers: `RESULTS_PILOT.md`
- every decision and incident: `DECISIONS.md`
- independent checks: `docs/verification/` and `docs/prereg_review/`

## 1. Question

RCWM-001 proposes learning goal-agnostic **consequence representations** φ of executable actions and reading goals out of them (U = w·φ). The claim is tested against a matched scalar evaluator. The Titan programme had installed a one-step state-transition skill into Supa 1.7 with a 384-example, 0.05%-norm sparse update, but multi-step generation was at floor.

RCWM-T1 asks:
1. Where does a frozen Rineya model "know" the consequences of a plan: its residual stream, or its generations?
2. Is that knowledge caused by the Titan delta?
3. Which consequence model makes the best goal-conditioned decisions?

## 2. Setup

- **Tasks:** 3 Titan families (ledger, prerequisite, register). Each parent is a start state plus 8 precommitted 4-event plans.
- **Fresh data:** all banks are checked against 6,464 historical payloads. The pilot has 200 parents per family. The sealed confirmation bank has 150 per family and was built without any model call.
- **Arms:** Titan-Sparse and Supa 1.7, plus delta and dose arms queued for the mechanism study. Everything runs in FP32 with TF32 off. Hidden states are captured at the answer slot and at each action boundary (G0 parity passed).
- **Decision metric:** normalised regret over 256 held-out goals (0 = oracle, ≈ 1 = random).

## 3. Confirmed results (PREREG-1)

**Claim confirmed: all five primary contrasts pass** (`RESULTS_CONFIRM.md`, rendered mechanically from `runs/bank_confirm/analysis/PREREG_1_RESULTS.json`).
- **Run:** 2026-10-07 18:08–18:32, after `verify_freeze.py` passed with Amendment 1 applied and no other differences.
- **Integrity:** sealed bank sha 8bc6e2f8…, 150 ledger + 150 register parents, fingerprints asserted.
- **Reproducibility:** the sparse generations reproduced an earlier, interrupted run byte for byte (I-22).

Registered wording (§8), with X filled in:

> On 300 fresh ledger and register-machine parents, Titan-Sparse chose among 8 four-step plans by chaining its own one-step predictions and scoring the predicted final state with a linear goal readout. Its normalised regret over 256 unseen goals was **0.448** (ledger 0.539, register 0.356; 0 = oracle, 1 = random). That is at least 0.10 lower (Holm one-sided p < .05) than each of the following:
> - an answer-free touched-entity rule;
> - the best of the ridge and MLP consequence heads on its frozen answer-slot states;
> - a symbolic-encoding learner given the same ~1.1k labels per family;
> - base Supa 1.7's best generative procedure;
> - its own single-shot four-step answer.

| | Contrast (k = 4, L+R pooled) | Point | 95% CI | Holm p | Per family (L / R) | Required qualifier |
|---|---|---|---|---|---|---|
| C0 | − answer-free touched-entity rule | −0.195 | [−0.224, −0.165] | 0.0005 | −0.072 / −0.317 | **Pooled effect, driven by register; not shown for ledger.** Ledger is significant (p = 0.005, CI [−0.126, −0.017]) but above the −0.10 per-family threshold |
| C1 | − consequence head B3 | −0.352 | [−0.384, −0.319] | 0.0005 | −0.222 / −0.483 | **Beats a comparator at or below the answer-free floor.** B3 mean 0.800 vs F_touched 0.642, so this does not support superiority over latent readouts in general |
| C2 | − symbolic learner S (equal labels) | −0.340 | [−0.372, −0.308] | 0.0005 | −0.198 / −0.483 | **Beats a comparator at or below the answer-free floor** (S 0.788 vs 0.642) |
| C3 | − Supa 1.7's best generative mode | −0.440 | [−0.473, −0.408] | 0.0005 | −0.306 / −0.575 | none |
| C4 | − its own single-shot 4-step answer | −0.397 | [−0.428, −0.364] | 0.0005 | −0.326 / −0.468 | none |

All five are also "strong passes": the one-sided 95% upper bound is ≤ −0.10 (−0.170, −0.326, −0.313, −0.413, −0.370).

**Limits that always accompany the claim (§8):**
- Horizon 4 only.
- L and R only. For P at k=4 the pre-declared directional prediction holds: G_chain − B3 = +0.277 [+0.236, +0.319], and chain exact-state accuracy is 0.006, below e2e's 0.058.
- The advantage over learned readouts holds at equal label budget. S@abundant (4,000 parents) reached **0.273 on ledger, better than the rollout** (G_chain − S_abundant = +0.266 [+0.206, +0.328]), and 0.594 on register, worse than the rollout (−0.238 [−0.261, −0.214]).
- No claim is made about off-task forgetting.

**Descriptive context** (secondary, BH q = 0.10; see RESULTS_CONFIRM.md):
- The chaining advantage over single-shot generation grows with horizon:
  - k = 1: L −0.010, R −0.005, both n.s.
  - k = 2: L −0.124, R −0.291.
  - k = 4: L −0.326, R −0.468.
- Exact final state at k = 4 stays low even when chaining: ledger 0.024, register 0.092. The decision advantage comes from partially correct states, not from exact simulation.
- B3's goal-factorisation advantage over the matched scalar evaluator replicates on out-of-span goals in every family and horizon (−0.12 to −0.59).

## 3b. Confirmed results (PREREG-2): the rule-coverage adapter fixes ledger rollout decisions

**CONFIRMED, strong pass** (`runs/bank_confirm2/analysis/PREREG_2_RESULTS.md`, rendered mechanically).
- **Freeze:** 2026-10-08 15:34 (`PREREG_2.md` sha f90f2258…), after the §9 seed gate passed (s1 +0.410, s2 +0.483).
- **Run:** one GPU turn, 15:4x–18:38.
- **Analysis:** run once at 18:38.
- **Bank:** `bank_confirm2`, 150 fresh parents per family, sealed, built without any model call, 0 payload overlap.

Registered wording:

> On 150 fresh ledger parents, adding a LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases (T2 seed 0, the fixed artifact) to frozen Titan-Sparse lowered the normalised regret of its chained four-step self-rollout decisions from 0.544 to 0.056 (ON − OFF −0.488, 95% CI [−0.535, −0.443], one-sided 95% bound −0.450, Holm p 0.0002), and raised four-step exact-state accuracy from 0.029 to 0.698 (+0.668 [+0.642, +0.695]).

**Forgetting guard (registered; not part of the claim): PASS.**
- First-token KL(OFF‖ON) is 0.50× [0.34, 0.70] the drift of Titan's own delta, KL(Supa‖Sparse).
- But cumulative drift from Supa is 1.94× KL(Supa‖Sparse), and 64-token greedy agreement with Supa falls from 0.50 (Sparse) to 0.37 (ON). The adapter adds drift on top of Titan's.

**Pre-declared predictions:** all hold.
- D1: the prerequisite copy shortcut persists (false copy +0.56).
- D2: prerequisite exact ≤ copy floor + 0.10.
- D3: prerequisite applicable state-changing steps get worse (0.59 → 0.35).
- D4: register shows no large change (regret −0.019).

**Step level, fresh bank:**
- Ledger void events 0.04–0.13 → 1.00.
- Posted add/scale/transfer 0.30–0.57 → 0.85–0.97.
- Transfer to the untrained multi-event frame is just as strong: k=4 exact +0.744.

**Limits that always accompany the claim (registered):**
- Ledger only; horizon ≤ 4; one adapter (T2 seed 0); Titan-Sparse only; the chain_one frame.
- The plans and the adapter's training rows come from the same event generator and marginals: the bank is fresh at payload level, not out-of-distribution.
- Rule learning is not separated from frame adaptation (no self-distillation or untargeted-LoRA control).
- No comparison with non-LM learners at the same label budget: S@abundant reached 0.273 on ledger, and S-roll matched Titan at about 512 transitions (R10). The adapted model used 384 + 1,501 oracle-labelled rows.
- Register and prerequisite are not claimed fixed.
- Retention evidence is limited to the first-token guard and 64-token agreement on 248 prompts.

## 4. Exploratory findings (pilot)

1. **Titan's one-step skill is real and replicates on fresh data.**

   | Family | Titan exact one-step state | Supa |
   |---|---|---|
   | Ledger | 0.39 | 0.07–0.11 |
   | Register | 0.61 | 0.05 |
   | Prerequisite (items whose state changes) | 0.46 | 0.00 |

2. **Multi-step failure is partly an interface problem.** End-to-end k-step generation collapses faster than independent composition would predict. Feeding Titan's own one-step answers back step by step does much better:
   - register: k=2 rises from 0.09 to 0.36 exact; k=4 from 0.005 to 0.087 (17×);
   - ledger: k=2 rises from 0.07 to 0.20.

   No training is involved.

3. **Serial self-rollout makes the best LM-based decisions at horizon 4 (RCWM "B4-roll").** Normalised regret on the pilot:

   | Method | Ledger | Register |
   |---|---|---|
   | Chained rollout → φ → w·φ | 0.51 | 0.39 |
   | Latent consequence heads on frozen states | 0.75–0.81 | 0.84–0.97 |
   | Scalar evaluators | 0.83–0.95 | 0.97–1.0 |
   | Symbolic learner at equal label budget | 0.75 | 0.84 |
   | Answer-free touched-entity rule | 0.60 | 0.68 |

   Supa's rollouts are near random, so the gain is attributable to the delta: Titan − Supa at register k=4 is −0.61 [−0.64, −0.57].

   Caveat: with 4,000 simulator-labelled parents, a symbolic learner beats the rollout on ledger (≈0.20–0.24) but not on register (0.58).

   **No-LLM chained baseline (S-roll, R10).** A generic MLP one-step learner, chained exactly like Titan:
   - **Ledger:** loses to Titan at Titan's own 128-row budget (−0.185 [−0.317, −0.051]); catches up by about 512 transitions.
   - **Prerequisite:** beats Titan at every budget.
   - **Register:** loses to Titan at every budget, because of register swap (exact copying), which the MLP cannot learn.

   The LM's edge is copying and data efficiency, not "having a simulator".

4. **The RCWM factorisation helps on new goals, and only there.** On goals outside the training span, B3 (consequence head + linear goal) beats a matched scalar evaluator by 0.13–0.58 regret. Within the span the difference is ≈0 to −0.04. This mostly holds by construction.

5. **No latent multi-step world model was found at the answer slot or at action boundaries.**
   - Linear change-mask decodability decays to heuristic floors by k = 2–4.
   - Control-label selectivity is ≈0: a probe can compute the decoded "consequence" from surface features.
   - The LM head does read the trained skill (P first digit: 0.84 Titan vs 0.60 Supa), so the change-mask probe is simply a poor instrument for this skill.
   - An early claim that the delta "only installs a readout" was withdrawn after independent verification.

6. **Mechanism: the large constant offset is format, not skill (R5, R9).**
   - On ledger and register, 97–99% of the delta's effect on late-layer answer-slot states is a constant per-task offset of about 6–8 within-family SDs per dimension.
   - **Sufficiency fails:** adding that offset to Supa at layer 16, 20 or 24 does not install the skill. Accuracy on changed items stays at 0–0.05, no better than a random direction of the same norm.
   - **Necessity fails for the computation:** subtracting it from Titan barely changes ledger and register accuracy. Prerequisite collapses at layers 20 and 24, but into unparseable output, not wrong answers.
   - **Reading:** the constant part sets task mode and output format. The transition computation sits in the small item-specific part, or in how the update acts at other positions.
   - **Status:** CPU preview, 48 items per family. The full GPU run and an off-task shift test with dose arms (forgetting) are queued.
   - **Where the computation is (R12, weight-spliced hybrids on all 640 pilot TEST items):**
     - Keeping Titan's delta only in blocks 8–15, which hold 41% of its energy, recovers the skill in all three families (FR 0.85 / 1.00 / 0.97).
     - Keeping it only in blocks 16–27, which hold 59%, recovers nothing (0.09 / 0.00 / 0.00). Blocks 0–7 are untouched by the delta.
     - For multi-token answers the delta must be active during decoding, not only while the prompt is read.

7. **Per-operation failure map (R8).** Titan's one-step skill is uneven:
   - register swap is 1.00, but mix is 0.008;
   - ledger void-status events are 0.00–0.12 (the event is applied anyway);
   - inapplicable prerequisite actions are 0.009.

   Titan learned the surface operations but not the conditional rules or the composite arithmetic. The training rows never contained inapplicable actions.

   Its prerequisite "rule" is integer addition. 50 of 77 wrong applicable steps output `state + add`, and 32 of those are unambiguous carries where the correct bitwise result was the unchanged state (R10).

   **This replicates on the 450-parent confirmation bank** (R11; post hoc, no tuning). Titan is right on 383/415 applicable steps where `state + add` happens to equal the bitwise answer, and on 2/333 where it does not. The per-operation map and the S-roll comparisons (item 3) replicate too, with 0 of 84 contrasts losing significance.

8. **Shortcut.** Titan applies inapplicable prerequisite actions even when the prompt states the no-op rule. That compounds through chains: prerequisite chained decisions are 0.88 regret.

## 4b. Novelty, read against web-verified prior work (`docs/RELATED_WORK.md`)

Most findings replicate known phenomena in a new, small-model setting (0.6B parameters, 384 examples, a 0.05%-norm sparse delta, exact-match state transitions).

- **Collapse without intermediate steps, recovery with them.** Already established: scratchpads (Nye et al. 2021), the serial-computation limits of transformers without CoT (Li et al. 2024; Merrill et al. 2024), the two-hop curse (Balesni et al. 2024), and Faith and Fate (Dziri et al. 2023).
- **Planning with LM rollouts.** Already established: RAP (Hao et al. 2023), LLM-MCTS (Zhao et al. 2023) and LM web-agent world models (Chae et al. 2025; Gu et al. 2025).
- **Factorised consequence head transfers to out-of-span goals.** This is successor features (Barreto et al. 2017; Borsa et al. 2019). In RCWM-T1 it serves as a confirmation and design check.
- **Probe caveats.** A known methodological lesson (Hewitt & Liang 2019).
- **Fine-tuning acting as a constant activation offset.** Has precedent (Wang, Engels et al. 2025; Minder et al. 2025).

Plausibly new, and narrow:

1. A **matched comparison**: serial self-rollout versus consequence and scalar heads trained on the *same frozen model's* hidden states with equal labels, attributed to a specific 384-example delta. PREREG-1 tests this.
   - Its scope is LM-based methods. Against a no-LLM learner chained the same way, the rollout wins only through exact copying (register) and at Titan's own small data budget (ledger) (R10, exploratory).
2. A **cautionary dissociation**: for an input-conditional skill, 97–99% of the fine-tuning shift at the answer slot is a constant offset, yet steering with it is neither sufficient nor (for the computation) necessary. It controls format and mode only. This qualifies offset-style readings of fine-tuning (Wang, Engels et al. 2025; Minder et al. 2025) for computational skills. It is exploratory; see R9's limits.

## 5. Operational lessons, relevant to all GPU work on this machine

- **The GPU destabilised the laptop.** The RTX 4070 Laptop drives the display. Sustained FP32 jobs reached 89–95 °C and 110–150 W bursts, and coincided with nvlddmkm TDR resets, frozen displays and unexpected shutdowns.
- **Mitigations now shared by both Claude sessions:**
  - a duty-cycle cap that adapts to temperature;
  - an independent thermal watchdog flag, sampled every 1.5 s (pause at 76 °C, resume at 66 °C);
  - on Modern Standby laptops, a Windows power request held only while a GPU job holds the lease on AC power, and a pause on battery (a sleep transition with CUDA active crashed the machine, I-19/I-20);
  - small batches and decode-step pauses;
  - one GPU process at a time;
  - strict alternation between sessions;
  - a lease holder check based on dedicated GPU memory. The stock check blocked forever behind CPU-only torch processes.
  - CPU-only model jobs: set `CUDA_VISIBLE_DEVICES=""` before importing torch and patch `torch.cuda.is_available` to report no GPU. With `-1`, torch 2.6 on this driver corrupted the heap (I-21).

## 6. What this means for the Rineya programme

These are provisional and will be revised once PREREG-1 resolves.

1. **Use explicit serial self-rollout, not latent heads, to get multi-step consequences out of a small model.**
   - The one-step skill a tiny delta installs becomes useful for planning only when composed externally, step by step. That costs inference calls, not training.
   - Benchmark it against a plain learned simulator, though. With a few thousand transitions, a chained MLP beats Titan on ledger and prerequisite. The LM's advantage is exact copying (register swap) and data efficiency at very small budgets (R10).
2. **Acquisition metrics hide rule failures.** SELECT-VIEWS-001 was 100% applicable, so it could not see that Titan ignores void status and preconditions and cannot do register mix (R8). Per-operation and per-rule breakdowns on fresh banks should be standard. Report P's inapplicable cases separately.
3. **Training data must contain the "no-op" and "illegal" cases.** Titan (this study) and BWM-001's forward model (an independent domain) both learn arithmetic and surface operations but not parameterised legality rules when those cases are rare or absent. RCWM-T2 tests the direct fix.
4. **Delta mechanism.** For L and R, most of the late-layer effect of the 0.05%-norm delta is a constant per-task offset, but steering shows that offset only sets output format and mode (R9).
   - A "skill vector" cannot carry the skill on its own.
   - Do not judge a sparse delta's mechanism by the variance share of its activation shift.
   - Test with interventions instead. Weight-spliced hybrids localise the skill to blocks 8–15 (R12).
   - The upper 59% of the delta is dispensable for the skill. Dropping it keeps decision quality and cuts off-task first-token KL drift by about a third (ratio 0.68 [0.62, 0.74]; D-23 pre-stated verdict "reduces interference", R14).
   - Two thirds of the drift sits in the same blocks as the skill, so layer localisation is a modest retention gain that stacks with magnitude pruning, not a cure.
   - **The promotion screen did not pass (PREREG-L1, 2026-10-09, DO NOT PROMOTE).**
     - Passed:
       - C0, C1 and C3–C6 hold;
       - drift is 0.65× Titan-Sparse's [0.60, 0.69];
       - retention is 43 vs 41 /47.
     - Failed:
       - C2 fails on prerequisite acquisition: 96 → 88 /128, net −8 (p 0.021, tolerance −5);
       - arithmetic falls 72 → 63 /150, just above its bar of 62.
     - The lost rows are near-miss sums, off by 1–4. So the upper delta is not needed for the rule, but it does sharpen exact numeric output.
     - "Dispensable" therefore holds for R12's transitions and not for exact arithmetic. L8-15 stays a mechanism result, not a core.
5. **Probing for "world models" in these models needs control tasks and LM-head readouts.** Linear change-mask probes were both insensitive to the trained skill and able to compute answers themselves.
6. **Infrastructure.** Every GPU job on this laptop should use the duty-cycle cap, the thermal watchdog flag, the dedicated-memory lease check and alternation. Without them, the RTX 4070 Laptop reached 95 °C and reset its driver.

## 7. Next steps

0. **PREREG-3: is the T2 ledger fix a rule or a trigger?** Frozen 2026-10-09 02:38 (D-33). A sealed out-of-range bank of 1,500 one-step rows and 80 chains covers:
   - unseen and neutral statuses, void look-alikes and "posted" look-alikes;
   - 4–5-digit balances and large amounts;
   - renamed or extra accounts, and reordered keys.

   The arms are adapter OFF, T2 s0–s2 ON and Supa, with no training, about 1–1.35 GPU-h. It is queued as the next RCWM GPU job. The verdict is RULE, TRIGGER or PARTIAL, with separate magnitude, naming and format verdicts.

1. **RCWM-T2** (3 seeds done, R13 and addendum; PREREG-2 frozen 2026-10-08 15:34 and running on a fresh sealed bank): a rule-coverage LoRA on frozen Titan-Sparse, using 1,501 fresh rows stratified over void, inapplicable, mix and wrap cases.
   - The ledger fix replicates in all three seeds: regret drop +0.47 / +0.41 / +0.48.
   - The prerequisite copy shortcut also replicates, and worsens by seed.
   - **Ledger:** fixed. 4-step exact accuracy 0.02 → 0.68; decision regret 0.507 → 0.038.
   - **Prerequisite:** improved through a copy shortcut (false copy 0.02 → 0.55), not the rule.
   - **Register:** unchanged; mix and wrap are not learned.
   - The strict D-19 "rule learned" gate fails, as designed: it requires every cell in every family.
   - **Next:** confirm the ledger effect across seeds and, if it holds, with a PREREG-2 on a new sealed bank. Rebalance the P data against copying.
2. **Mechanism** (queued):
   - steering sufficiency and necessity of the mode vector. A CPU preview found neither (R9); the full GPU run is queued. The next test is per-item activation patching, at the answer slot and at all prompt positions, to locate the computation;
   - off-task shift and first-token KL across the dose arms (α = 0.5, 1, 2), the random-direction delta, BF16-rounded Titan-Exp and Titan-Transition;
   - this tests whether magnitude pruning reduces off-task drift, which would mechanistically explain the retention benefit seen in SPARSE-001.
3. **Verification-guided rollout** (queued): PVR and retrospective correctness probes that pick among sampled next states. This is a cheap test of whether verification can raise per-step accuracy without training.
4. **Likelihood ranking** (queued): whether Titan's own density prefers the true k-step consequence over heuristic foils. This is a format-free test of composition.
