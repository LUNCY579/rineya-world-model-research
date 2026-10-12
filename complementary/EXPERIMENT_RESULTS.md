# EXPERIMENT_RESULTS (RCX)

This file records only results that were actually measured. Each entry gives:
- the evidence path;
- the **tier**:
  - *instrument*: CPU checks of generators, oracles and floors;
  - *mechanism*: small learners trained from scratch on CPU; not Rineya;
  - *native-A*: a frozen Rineya core, inference only;
  - *native-C*: a trained student on a Rineya core;
- the limitations.

---

## R-01 RCX-E01 instrument qualification (tier: instrument), 2026-10-09/10

**Evidence:** `experiments/RCX-E01/qualification/QUALIFICATION.json`, commits `18f1fa6` and `b128150`.

### Q1: agreement with RCWM on the posted rule

The RCX interpreter on the posted-rule subset was compared with RCWM `rcwmt.env.step` (sha 92abb678…) and
`scripts/p3_common.rule_gold` (sha 737d7984…).
- 20,000 random cases: kinds add/scale/transfer, balances −99..999, amounts −20..99, half posted and half other
  statuses (near-miss and posted-like included).
- Result: **20,000 / 20,000 identical**.

### Q2/Q3: agreement with an independent oracle

`oracle_indep.py` was written blind from `SPEC_RCL.md`, without reading the generator or interpreter. It agrees with the
generator's AST, state and event (in key order), `ok`, `listed`, `after`, and the `ans`/`trace` strings on **76,320 /
76,320 records**. These cover 6 training regimes and the dev and sealed batteries, in both the arith and copy worlds.

### Floor validation on the DEV battery (60 parents per stratum, 13 strata × 6 surfaces)

| Predictor | Primary composite score |
|---|---|
| COPY | 0.50 |
| ALWAYS | 0.50 |
| TRIGGER (no-op iff status ∈ TRAIN no-op words) | 0.50 |
| MAJORITY | 0.50 |
| STATUS_LOOKUP | 0.50 |
| CONTAINS_POSTED | 0.61 |
| RULE_WORD_MATCH (apply iff the status appears in the rule line) | 0.71 |
| ORACLE (exact parser plus interpreter) | **1.00** |

So the battery separates rule reading from every listed shortcut. RULE_WORD_MATCH, the strongest shortcut, fails by
design on negative and inverted rules (V2 = 0.00, V3 = 0.50).

### Split hygiene (tested)

- Zero semantic-hash and zero (state, event-without-status) skeleton overlap between TRAIN, DEV and SEALED.
- DEV and SEALED status words are unseen in TRAIN, with cross-split edit distance ≥ 2.
- The PREREG-3 and factory vocabularies are excluded.

**Limits.** Qualification covers only the closed template pools, two-atom compositions and the listed kinds. The floors
are deterministic heuristics; they are not an exhaustive set of possible shortcuts.

---

## R-02 RCX-E01 CPU learnability pilot (tier: mechanism, DEV only, seed 0), 2026-10-09/10

**Evidence:** `runs/RCX-E01/pilot/*/summary.json`, `runs/RCX-E01/pilot_copy/*`, `DECISIONS.md` CX D-08 and D-09.

**Setup.** Character-level transformer (d128/L3/H4, 0.60M parameters), trained from scratch on CPU with 4 threads.
32,000 examples per arm, batch 32. IID means a fresh-parent split of each arm's own training distribution, with zero
hash overlap with training.

| World | Arm | Steps | IID exact | IID decision | Dominant errors |
|---|---|---|---|---|---|
| arith | VR-ans | 3,000 | 0.522 | 0.523 | false_copy 711 / 717 (collapsed to COPY) |
| arith | FX-ans (T2 analog) | 3,000 | 0.169 | 0.627 | arithmetic 687 / 1,247 |
| arith | VR-trace | 3,000 | 0.409 | 0.584 | false_copy 436, arithmetic 262, applied_when_noop 188 |
| copy | VR-ans | 4,000 | 0.531 | 0.601 | false_copy 392, applied_when_noop 204; copying learned, decision near chance |
| copy | CFSD-ans | 4,000 | 0.505 | 0.545 | false_copy 629; DEV composite `in` 0.444, decision 0.477 |

**Finding (negative, instrument-relevant).** A character-level transformer trained from scratch, at this CPU budget,
does not learn to decide applicability. This holds even in its own training distribution, even with an explicit
`listed`/`ok` trace, and even once arithmetic is removed (copy world).

**Reading.** Deciding membership requires matching multi-character status strings against the words listed in the
rule. That is an induction/binding operation that pretrained LMs already have and that this small model does not acquire
in about 100k examples. This makes the arm-vs-arm data-factor comparison uninformative at this tier.

**Consequences for the design:**
1. A symbol-token learner: identifiers become single tokens with frozen random embeddings, so binding is learnable but
   unseen symbols carry no lexical cue.
2. Moving the decisive E01 test to the native tier, Titan-Sparse LoRA cells on CPU.

**Limits.** One seed, one architecture, a fixed step budget. Longer training (grokking-like transitions) was not tested.

---

## R-03 Native tier-A/B rule-reading probe on frozen Rineya cores (tier: native-A / reference student; DEV, inference only), 2026-10-10

**Evidence:**
- `runs/RCX-E01N/probe1/summary_{OFF,T2}.json`, `preds_*.jsonl.gz`, `rule_sensitivity.json`;
- code: `experiments/RCX-E01N/native_probe.py`, reviewed in wf_53c3f5a8-51d.

### Setup

- **Arms.**
  - **OFF**: Titan-Sparse-Experimental 1.0.0, FP32, frozen.
  - **T2**: the same core with RCWM's registered T2 s0 LoRA, unmerged, FP32, with the adapter and config sha-checked.
  - **SUPA**: not yet run. The load was refused because 11.4 GB of RAM was free, below the 12 GB floor (D-39). It will
    be run when RAM allows.
- **Frame.** Titan's verbatim ledger one-step frame with only the status clause replaced by the supplied rule sentence.
  The user turn is the compact sorted JSON exactly as Titan was trained. Decoding is greedy.
- **Battery.** RCL arith-world DEV battery, seed 20261240: 13 strata × 40 parents, surfaces `in` and `ho_tpl` (held-out
  rule paraphrases), 2,240 prompts per arm.
- **Words.** DEV words are fresh (disjoint from RCX TRAIN and from PREREG-3).
- **Verbatim-frame control.** 800 fixed-rule prompts per arm.
- **CPU cost.** OFF 5,389 s; T2 4,155 s.

### Results (`in` surface; `ho_tpl` within 0.03 everywhere)

| | OFF | T2 s0 |
|---|---|---|
| Primary composite (exact) | 0.325 | 0.400 |
| Decision accuracy, every varied-rule stratum (V1, V2, V3, V5, V6, V7) | 0.50 | 0.50 |
| No-op produced when the supplied rule requires a no-op (V strata) | **1 / 280** (Wilson 95% [0.001, 0.020]) | **1 / 280** ([0.001, 0.020]) |
| V3 rule sensitivity (output changes when the same item's rule polarity flips) | **0.00** | **0.00** |
| V5: rule `S+{x}` or `S-{posted}` with status `posted` → no-op | 0.00 / 0.00 | 0.00 / 0.00 |
| F1 `void` probe → no-op | 0.03 | **1.00** |
| F4 near-miss (`Void`, `VOID`, `voided`, `void_`) → no-op | 0.03 | **0.97** |
| F2 / F3 / F5 unseen real / nonce / posted-like → no-op | 0.03 / 0.00 / 0.00 | **0.00 / 0.00 / 0.00** |
| Posted twins (apply), exact | 0.55–0.72 | 0.72–0.88 |
| Verbatim Titan frame, F strata (exact) | 0.29–0.35 (same as the rule frame) | 0.39–0.93 (same as the rule frame) |
| Format errors | 6 / 2,240 | 4 / 2,240 |

### Reading

1. **Neither model reads a supplied ledger rule.**
   - Frozen Titan-Sparse applies every event regardless of the rule.
   - T2 applies every event except `void` and its case and morphology near-misses.
   - Rule sensitivity is 0.00 for both, even when the in-context rule names the event's own status as ignored.
2. **T2 s0 shows PREREG-3's TRIGGER signature, replicated on an independent RCX instrument with fresh vocabulary.**
   - The no-op fires for `void` and near-misses, which is the "trigger widened" pattern under the shared D-45/D-54
     wording.
   - It never fires for unseen non-posted words (0/40 per stratum), and nearly all such errors are applied-when-no-op.
3. **The rephrased frame is not a confound.** Titan's own verbatim system prompt gives the same F-stratum results.

### Limits

- DEV battery, 40 parents per stratum, one decoding setting, two arms.
- Exact accuracy on apply variants is limited by arithmetic errors, which the decision metric separates out.
- This is *descriptive*: tiers A and B, no training, no confirmation claim.

---

## R-04 RCX-E07/E08 relational value learning for Blocksworld planning (tier: mechanism; external specialist = tier B), 2026-10-10

**Evidence:**
- `runs/RCX-E07/main/REPORT.md`, `e07_metrics.json`, `e08_metrics.json`, `e08_runs.jsonl`, `e08_success_vs_budget.json`;
- code: `experiments/RCX-E07/run_e07.py` (reviewed in wf_28dada83-a37) and `rcx/bw/*`.

### Setup

- **Simulator.** BWM's `blocks.py`, pinned and imported read-only, run with exact dynamics.
- **Instances.** All fresh (seeds 20261270–75). BWM's reserved and selection sets are excluded: hard-110, val_large,
  mirror, b3fresh, seeds_fresh, distill-240, PB-600/unsolvable signatures, val_v2 and v1 val. They are read only to
  build exclusions.
- **Train.** n = 4–7, 22 goal classes (classes are disjoint from val and test), 110k exact-h\* rows.
- **Models.** RelGNN (124k parameters) and FlatMLP (159k), with identical rows, epochs and selection on val MAE.
  3 seeds.
- **E07 test.** 15 instances per size, n ∈ {5, 6} (ID) and {8, 10, 12} (OOD). Random-walk states, exact h\* labels.
- **E08.** GBFS on 100 fresh full-goal instances, n ∈ {8, 10, 12, 14, 15}.

### E07: one-step policy accuracy (argmin successor is optimal; ties strict; mean ± sd over 3 seeds)

| Model | ID (5–6) | OOD 8 | OOD 10 | OOD 12 | OOD pooled |
|---|---|---|---|---|---|
| **RelGNN** | 1.000 ± 0.000 | 1.000 | 0.999 | 0.995 | **0.998 ± 0.001** |
| FlatMLP | 0.989 ± 0.003 | 0.709 | 0.513 | 0.499 | 0.563 ± 0.008 |
| h_add (best handcrafted) | 0.803 | 0.819 | 0.903 | 0.821 | 0.848 |
| misplaced_blocks | 0.560 | 0.600 | 0.633 | 0.377 | 0.526 |
| unmet_goal_facts (BWM's beam ranking) | 0.139 | 0.082 | 0.054 | 0.054 | 0.062 |

Paired instance-clustered bootstrap over OOD instances (n = 45):
- GNN − MLP = **+0.406 [+0.355, +0.456]**;
- GNN − h_add = **+0.148 [+0.113, +0.184]**.

GNN MAE grows with size (0.06 at n = 8, 0.63 at n = 12), but its rank correlation stays high (Spearman ≥ 0.988).

### E08: GBFS success (exact dynamics) vs expansion budget, out of 100 fresh instances, n = 8–15

| Heuristic | 25 | 50 | 100 | 200 | 500 | 1,000 | 2,000 | Mean suboptimality (solved) |
|---|---|---|---|---|---|---|---|---|
| **RelGNN** | **26** | **49** | **54** | **57** | 61 | 61 | 65 | **1.056** |
| misplaced_blocks | 6 | 22 | 36 | 48 | **68** | 78 | 87 | 1.012 |
| h_add | 0 | 3 | 14 | 26 | 63 | 83 | **89** | 2.024 |
| unmet_goal_facts (BWM ranking) | 0 | 2 | 3 | 8 | 14 | 21 | 31 | 1.595 |

**Paired exact McNemar tests (GNN seed 1):**

| Comparison | Budget | Baseline-only | GNN-only | p |
|---|---|---|---|---|
| GNN vs unmet_goal_facts | 500 | 0 | 47 | < 1e-4 |
| GNN vs unmet_goal_facts | 2,000 | 6 | 40 | < 1e-4 |
| GNN vs misplaced_blocks | 500 | 18 | 11 | 0.26 |
| GNN vs misplaced_blocks | 2,000 | 24 | 2 | < 1e-4 |

### Reading

1. **Relational structure generalizes; flat encodings do not.** A GNN trained on ≤ 7 blocks keeps a near-perfect greedy
   policy at 8–12 blocks, while an MLP at matched labels collapses to 0.50–0.71. This is the representation result
   RGM-001 E07 asked for.
2. **A learned value is a strong *low-budget* search critic.**
   - Up to 200 expansions it solves the most instances of every heuristic tested, with near-optimal plans.
   - BWM's hand-coded unmet-goal ranking is the weakest non-trivial heuristic here: policy accuracy 0.06 OOD and
     14/100 solved at 500 expansions.
   - This bears directly on BWM's beam failures (dead_end/budget), where each expansion costs LM calls.
3. **It is brittle at higher budgets.**
   - GNN coverage plateaus: all 35 failures at 2,000 expansions are budget exhaustion on longer problems (mean
     optimal length 29.7 vs 22.7 for solved ones), concentrated at n = 12–15.
   - When it succeeds it needs a median of only 28 expansions.
   - Misplaced-blocks and h_add overtake it beyond about 500 expansions. One misranking deep in a large problem
     traps GBFS in a big region.

### Limits

- Mechanism tier with a specialist heuristic: tier B, not a Rineya-core ability.
- Exact dynamics, not BWM's LM forward model F.
- GBFS, not BWM's beam.
- One E08 seed instance set. The best-handcrafted baseline is selected on test states (conservative for "GNN better").
- E08 full goals vs E07 mixed goals.
- **Next refinements (E07b):** larger-n and full-goal training, a ranking loss, and hybrid tie-breaking with
  misplaced_blocks, tested on a fresh E08 set. Then BWM's beam with the frozen LM F, read-only and coordinated.

---

## R-05 Native E01 cell N-VR seed 0: rule-varied LoRA on Titan-Sparse reads supplied rules, but only in trained wordings (tier: native-C component artifact; DEV pilot, one seed), 2026-10-10

**Evidence:**
- `runs/RCX-E01N/cells/eval_N-VR_s0.json`, `preds_N-VR_s0.jsonl.gz`, `N-VR_s0/TRAIN_META.json` (adapter sha256
  `2f376882…`; weights local, unmerged, never published);
- `runs/RCX-E01N/ANALYSIS_DEV.md`.

### Setup

- **Student.** LoRA r16/α32 on frozen FP32 Titan-Sparse, exactly T2's D-03 recipe: 1,501 rows, 190 × 16 steps,
  lr 1e-4 WSD, AdamW (0.9, 0.95). Recipe parity with the vendored trainer is verified.
- **Training.** On CPU, 9,769 s.
- **Data.** Rule-conditioned ledger, *varied* in-context rules. Canonical wording only (template t0) and real TRAIN
  words only. 30% two-atom compositions. Balanced apply/no-op.
- **Evaluation.** The same DEV battery as R-03 (seed 20261240; fresh DEV words; 13 strata × 40 parents; surfaces `in`
  and `ho_tpl`), in Titan's frame. Greedy decoding.

### Results vs R-03 arms (surface `in` = the canonical wording used in training)

| | OFF | T2 s0 | **N-VR s0** |
|---|---|---|---|
| Primary composite (exact) | 0.325 | 0.400 | **0.964** |
| Decision accuracy | 0.500 | 0.502 | **0.966** |
| No-op when the supplied (varied) rule requires one | 1/280 | 1/280 | **254/280** [0.867, 0.936] |
| V3 rule-polarity sensitivity / both twins exact | 0.00 / 0.00 | 0.00 / 0.00 | **1.00 / 0.975** |
| F2 / F3 unseen real / nonce statuses under the posted rule (exact) | 0.35 / 0.36 | 0.39 / 0.40 | **1.00 / 1.00** |
| F5 posted-like strings | 0.31 | 0.42 | 0.79 |
| V5 rule overrides the `posted` prior | 0.24 | 0.40 | **0.99** |
| V6 held-out compositions / V7 3-word sets (unseen size) | 0.29 / 0.33 | 0.34 / 0.42 | 0.68 / 0.93 |
| M1 / M2 magnitudes (exact; decision) | 0.31 / 0.12 | 0.33 / 0.25 | 0.85 / 0.72 (0.97 / 0.91) |

Paired parent-clustered bootstrap: N-VR − OFF = **+0.639 [+0.608, +0.671]** (`in`, 280 parents).

### Held-out rule paraphrases (`ho_tpl`): composite 0.594, decision 0.597

N-VR − OFF on `ho_tpl` = +0.286 [+0.251, +0.321]. Errors are 445 applied_when_noop, 15 false_copy and 31 arithmetic.

| Held-out wording | Behaviour |
|---|---|
| Positive rule, "If the status is X, carry out the event; for any other status, do nothing." | no-op 0.03 |
| Positive rule, "Balances change only for X events." | no-op 0.01 |
| Negative rule, h0 | no-op 0.93 |
| Negative rule, h1 "Every event takes effect except X events." | no-op 0.19 |
| Compositions | no-op 0.29–0.32 |

### Reading

1. Rule-*varied* training creates genuine in-context rule execution where frozen Titan-Sparse and T2 have none:
   - rule-polarity sensitivity rises from 0.00 to 1.00;
   - the no-op is applied to unseen statuses and to rules that override the `posted` prior.

   In the shared wording, it **generalises to unseen statuses** (F2/F3 1.00). It is partial on posted-like strings
   (F5 0.79).
2. **The capability is bound to the trained wording.** For rule phrasings it never saw, the model mostly reverts to
   Titan's "apply everything" prior. This is a surface-form limit of the same family as PREREG-3/R16. It is the exact
   motivation for the N-CFSD arm, which trains on paraphrases, nonce words and counterfactual twins.

### Limits

- **Scope.** DEV battery, a single seed, a single cell.
- **Controls still missing:**
  - the matched fixed-rule control (N-FX, same generator and budget);
  - N-CFSD;
  - E33 retention (no claim about retained general ability; RCWM will run D-36 on request).
- **Not yet confirmed.** Confirmation requires the frozen PREREG_E01N on a SEALED battery, with 3 seeds.
- **Artifact.** This is a component adapter on a frozen core, not a dense checkpoint.

---

## R-06 Native E01 cell N-CFSD seed 0 vs N-VR seed 0: counterfactual twins, nonce words and paraphrases improve rule reading (tier: native-C component artifact; DEV pilot, one seed each), 2026-10-11

**Evidence:**
- `runs/RCX-E01N/cells/eval_N-CFSD_s0.json`, `preds_N-CFSD_s0.jsonl.gz`;
- `N-CFSD_s0/TRAIN_META.json` (adapter sha256 `97cc48e8…`, local, unmerged);
- `runs/RCX-E01N/ANALYSIS_DEV.md`.

**Setup.** Same recipe, core and DEV battery as R-05. The data regime differs: counterfactual parents (base, rule_flip,
status_swap, set_swap twins), 50% nonce statuses, train paraphrases t0–t3, and 3–4 accounts. Rows, steps and budget
are matched. Supervised tokens are −11.4% vs N-VR (−11.5% vs N-FX) (a reported covariate). Training took 9,302 s on CPU.

| DEV, seed 0 | N-VR | **N-CFSD** | N-CFSD − N-VR (parent-clustered bootstrap, 280 parents) |
|---|---|---|---|
| Composite, `in` (training wordings) | 0.964 | **0.984** | +0.020 [+0.002, +0.039] |
| Composite, `ho_tpl` (held-out paraphrases) | 0.594 | **0.693** | **+0.099 [+0.075, +0.125]** |
| No-op when the rule requires one (`in`) | 254/280 | **276/280** [0.964, 0.994] | |
| F5 posted-like strings | 0.79 | **1.00** | |
| V6 held-out compositions | 0.68 | **0.93** | |
| V3 inverted twins, both exact | 0.975 | **1.000** | |
| PREREG-3-style verdict (F strata, `in`) | partial | **generalises to unseen statuses** | |

Against the R-03 arms, N-CFSD − T2 is +0.584 [+0.562, +0.608] on `in` and +0.300 [+0.266, +0.335] on `ho_tpl`.

### Held-out paraphrases by wording (decision accuracy on no-op items, N-VR → N-CFSD)

| Wording | N-VR | N-CFSD |
|---|---|---|
| Negative h1 "Every event takes effect except X events." | 0.19 | **0.88** |
| Negative h0 | 0.93 | 0.92 |
| Compositions | 0.29–0.32 | **0.79–0.90** |
| Positive h0 "If the status is X, carry out the event; for any other status, do nothing." | 0.03 | 0.29 |
| Positive h1 "Balances change only for X events." | 0.01 | 0.17 |

### Reading

Counterfactual and surface-diverse rule data:
- makes the in-context rule reading more complete (RULE-like on fresh statuses, posted-like strings and compositions);
- partly frees it from the trained wording (+0.10 on held-out paraphrases).

**Residual failure.** Unseen phrasings of *restrictive positive* rules still fall back to Titan's "apply" prior. That is
the next target: more diverse positive phrasings, or a predicate trace (`cmp → calc → after`).

### Limits

- **Scope.** DEV, one seed per arm.
- **Confounds.** The CFSD regime changes several factors at once: twins, nonce words, paraphrases and account count.
- **Not yet done:** retention, and the matched fixed-rule control N-FX.
- **Confirmation.** It needs the frozen PREREG_E01N on a SEALED battery with 3 seeds.

---

## R-07 RCX-E07b, preregistered: a GNN-value + misplaced-blocks hybrid closes the high-budget gap while keeping the low-budget advantage (tier: mechanism, external specialist B; exact dynamics), 2026-10-11

**Evidence:**
- `runs/RCX-E07/e07b/REPORT_E07B.md` and `metrics_e07b.json`;
- prereg `experiments/RCX-E07/PREREG_E07B.md` (sha `8a4cbde0…`, frozen before any E07b code or run; CX D-20);
- clarification CX D-23.

**Setup.**
- TEST-B: 100 fresh full-goal instances, seed 20261290, n ∈ {10, 12, 14, 15}. They are disjoint from R-04's E08 set,
  from R-04's E07 test instances, and from every BWM reserved and selection set.
- VAL-B (30 instances) was used only to choose λ and BEST.
- Arms:
  - A0: the R-04 GNN, 3 seeds.
  - A1: training extended to n = 8 with full goals, 3 seeds.
  - A2: a successor-ranking hinge loss, 3 seeds.
  - A3: GNN + λ·misplaced_blocks, with λ = 1.0 chosen on VAL-B.
- GBFS with exact dynamics.

**PRIMARY (prereg rule, n = 12–15, 75 instances): outcome CLOSED.**
- BEST = A3 (λ = 1.0).
- At 2,000 expansions: 55/75 vs misplaced_blocks 54/75 (8:9, p = 1). That meets the "≥ misplaced − 3" condition.
- At 200 expansions: 39/75 vs 19/75 (McNemar 4:24, **p = 0.00018**).
- Vs A0 at 2,000: 55 vs 35 (0:20, p = 1.9e-6).

| n = 12–15, success / 75 | @50 | @200 | @500 | @2000 | Mean suboptimality (solved, all n) |
|---|---|---|---|---|---|
| **A3 hybrid (λ = 1)** | 26 | **39** | **47** | 55 | 1.045 |
| A0 GNN (3 seeds, mean) | 22.0 | 26.0 | 28.7 | 32.0 | 1.037 |
| A1 n = 8 full-goal (mean) | 26.3 | 28.3 | 32.0 | 36.0 | 1.022 |
| A2 ranking loss (mean ± sd) | 27 ± 8 | 33 ± 11 | 38 ± 10 | 40 ± 9 | 1.052 |
| misplaced_blocks | 8 | 19 | 35 | 54 | 1.007 |
| h_add | 0 | 13 | 34 | **66** | 1.965 |
| unmet_goal_facts (BWM ranking) | 0 | 0 | 1 | 3 | 1.658 |

### Reading

1. **The hybrid has the best overall profile among the heuristics compared.**
   - It roughly doubles misplaced_blocks' success at small budgets, which is where each expansion is costly in an
     LM-driven planner.
   - It matches misplaced_blocks at large budgets.
   - Its plans are near-optimal.
   - h_add has the most coverage at 2,000, but its plans are about 2× optimal length and it is weak at low budgets.
2. **The two learned-only refinements did not help.** Data scaling (A1) did not raise coverage, and the ranking loss
   (A2) was unstable across seeds. So the GNN's high-budget failures are best repaired by combining it with a classical
   heuristic, not by more of the same training.
3. **This A3 hybrid is the RCX module for joint B4 arm (a)** (`rcx/bw/b4_hook.py`, `hybrid_lambda=1.0`), to be tested
   inside BWM's frozen beam planner with LM F under a co-signed prereg.

### Limits

- Tier B specialist with exact dynamics and GBFS. Not BWM's beam with LM F, and not a Rineya-core claim.
- A single selected arm, and seed 1 for the paired tests (A3 inherits A0's seed 1).
- Selection on VAL-B used only 30 instances.

---

## R-08 RCX-E01N SEALED confirmation (PREREG_E01N): counterfactual and paraphrase rule data beats rule-varied data on both co-primaries in every seed, but the held-out-wording gain (+0.090) misses the preregistered +0.10, so the decision is **REVISE** (tier: native-C component artifact; confirmatory, 3 seeds per arm), 2026-10-11

**Evidence:**
- `runs/RCX-E01N/ANALYSIS_SEALED.md` (sha `7e1c6023…`) and `ANALYSIS_SEALED.json` (sha `7b6a6fc4…`): one run at 14:07,
  create-only, decision applied by code;
- `runs/RCX-E01N/cells/eval_sealed_*.json` and `preds_sealed_*.jsonl.gz` (8 cells; commits `2dd1ad0`, `5b848c6`);
- prereg `experiments/RCX-E01N/PREREG_E01N.md` (sha `975cca9a…`, frozen before any SEALED record existed; CX D-26);
- custody: `SEALED_COMMIT.json` (salt commitment minted at 01:54) and `SEALED_OPEN.json` (opened at 08:40, after all
  six adapters existed; committed in `fb27d1d` before the first SEALED eval; CX D-30).

**Setup.**
- **Battery.** SEALED: seed 88527345, sha `c87765ea…`. 2,240 records over 280 parents, surfaces `in` (training
  wordings) and `ho_tpl` (held-out rule paraphrases). Training-overlap parents: 0.
- **Arms.** N-VR (rule-varied) vs N-CFSD (counterfactual twins + nonce statuses + paraphrases), seeds 0–2 each. These
  are LoRA adapters (BWM D-03 recipe, unmerged) on frozen FP32 Titan-Sparse, with matched rows, steps and budget.
- **References.** OFF (frozen core) and T2 (RCWM's T2 adapter).
- **Method.** Seed × parent paired cluster bootstrap, 10,000 resamples. Holm over C1 and C2.

### Confirmatory: N-CFSD − N-VR

| Endpoint | N-VR | N-CFSD | Diff [95% CI] | Holm p | Per seed (s0, s1, s2) |
|---|---|---|---|---|---|
| C1 `in` composite | 0.963 | 0.981 | **+0.018 [+0.006, +0.030]** | 0.002 | +0.015, +0.002, +0.036 |
| C2 `ho_tpl` composite | 0.591 | 0.681 | **+0.090 [+0.069, +0.113]** | 0.0002 | +0.095, +0.121, +0.055 |

**Decision trace.** ADVANCE needs a point estimate ≥ +0.10 on at least one endpoint. Neither reached it (C2 +0.090).
Every other ADVANCE condition held for C2:
- Holm p < 0.05 and CI lower bound > 0;
- all three per-seed diffs > 0;
- C1's CI lower bound > −0.05;
- both protected strata within margin.

STOP did not hold, so the outcome is **REVISE**. Seeds 1–2 alone (excluding the DEV-gated seed 0) give the same
picture: C2 +0.088 [+0.066, +0.111].

**Protected strata (within margin).**
- F0 posted twins (exact, `in`): −0.007 [−0.025, +0.013].
- Format-error rate: 0.001 → 0.000.

**DEV → SEALED replication.** The DEV pilot (R-06, seed 0) gave +0.020 / +0.099. SEALED, on a fresh battery and three
seeds, gives +0.018 / +0.090.

### Against the frozen references (descriptive)

| Arm | Composite `in` | Composite `ho_tpl` | No-op when required, `in` | V3 rule-polarity sensitivity `in` / `ho_tpl` | PREREG-3-style verdict, `in` |
|---|---|---|---|---|---|
| OFF | 0.336 | 0.319 | 4/280 | 0.00 / 0.00 | ignores the rule |
| T2 | 0.367 | 0.364 | 0/280 | 0.00 / 0.00 | TRIGGER (widened to near-misses) |
| N-VR (3 seeds) | 0.963 | 0.591 | 782/840 [0.91, 0.95] | 1.00 / 0.50–0.55 | partial (3/3 seeds) |
| N-CFSD (3 seeds) | **0.981** | **0.681** | **817/840 [0.96, 0.98]** | 0.95–1.00 / **0.95–0.98** | **generalises to unseen statuses (RULE-like), 3/3 seeds** |

N-CFSD − T2 is +0.614 [+0.588, +0.641] on `in` and +0.317 [+0.280, +0.354] on `ho_tpl`.

### Held-out wordings (`ho_tpl`, pooled over seeds): decision accuracy

| Wording family | No-op items, N-VR → N-CFSD | Apply items, N-VR → N-CFSD |
|---|---|---|
| Negative h0 (S−) | 228/252 → 243/252 | 177/192 → 171/192 |
| Negative h1 (S−) "Every event takes effect except X events." | 44/255 → **247/255** | 192/195 → 188/195 |
| AND compositions | 25/33 → **33/33** | 33/33 → 32/33 |
| OR compositions | 27/87 → **73/87** | 84/87 → **66/87** |
| **Positive (S+), h0 and h1** | 24/1,053 (0.02) → **146/1,053 (0.14)** | 1,149/1,173 → 1,120/1,173 |

### Reading

1. **Confirmed, within this task family.** Rule-varied LoRA training makes Titan-Sparse read a supplied rule, where
   the frozen core and T2 ignore it (+0.63 composite on `in`). Counterfactual and paraphrase data improves on that on
   both surfaces, in each of three seeds, on a sealed battery. On training wordings, N-CFSD is RULE-like on unseen
   statuses in all three seeds. This is the PREREG-3 shortcut/trigger failure, removed inside the adapter (no external
   module at inference).
2. **Not confirmed at the preregistered size.** The held-out-wording gain is +0.090, short of +0.10. The threshold
   is not reinterpreted.
3. **The residual failure is specific.** Unseen phrasings of *restrictive positive* rules ("only X events take
   effect") are still read as "apply": N-CFSD decides no-op on only 14% of the no-op items that need one. That error
   direction (`applied_when_noop`) is 286–350 of each N-CFSD cell's `ho_tpl` errors, and leaves the F strata near 0.52–0.58
   on `ho_tpl`. Negative and compositional held-out wordings are mostly solved.
4. **Trade-off.** N-CFSD over-restricts some held-out OR wordings (apply items 84/87 → 66/87), and has slightly more
   `false_copy` errors on `in` (15–30 vs 9–13 per cell).

### Limits

- **Synthetic task family.** This is not a claim of improved general model intelligence. Retention (RCWM D-36 / E33) was
  not run, because the prereg makes it conditional on ADVANCE.
- **Confounds.** The CFSD regime changes several factors at once: twins, nonce words, paraphrases and account count.
  Supervised tokens are about 11% fewer than N-VR.
- **Scope of inference.** Inference is conditional on these six adapters: three seeds per arm, with seed 0 DEV-gated.
  The descriptive N-FX control was not run.
- **The `ho_tpl` wordings are now spent.** Their per-wording results have been seen twice (DEV and SEALED). Any next
  recipe must be judged on a new, blind-authored held-out wording pool and a new sealed battery.

---

## R-09 RCX-E11 (frozen PREREG_E11): macros mined from Rineya-verified plans speed up exact-dynamics search, **E11a ADVANCE**; a learned subgoal order transfers but does not beat simple rules, **E11b REVISE** (tier: mechanism, external specialist B; exact dynamics), 2026-10-11

**Evidence:**
- `runs/RCX-E11/main/REPORT_E11.md`, `metrics_e11.json`, `a/runs.jsonl`, `b/exec_runs.jsonl` (commit `bddfcda`);
- prereg `experiments/RCX-E11/PREREG_E11.md` (sha `c67f036d…`, frozen after a plumbing-only smoke; CX D-31);
- teacher custody: BWM B3 `solutions.jsonl`, sha-pinned. 240 rows; 211 F-verified plans re-executed exactly; 29 not
  verified, skipped.
- Gates: BWM bytecode clean; G1–G3 OK; A3 hook accounting exact (1,200 runs, 0 invalid states, 0 fallbacks); budget
  monotonicity OK.

### E11a: macro discovery and reuse (120 fresh full-goal instances, n = 12/14/15; seed 20261256)

**Arms.** Each arm is primitives plus 8 macros, except P and G:

| Arm | Macros |
|---|---|
| P | none (primitives only) |
| M | mined from the 211 verified plans |
| R | random sound macros matched to M by length and goal size |
| S | mined the same way from action-shuffled plans, disjoint from M |
| G (descriptive) | 2 goal-gated single moves |

**Library M** is mostly tower-building chains, such as `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0`, and one
clearing pair, `unstack; put-down`.

| GBFS + A3 hybrid (primary heuristic) | @50 | **@200** | @500 | @2000 | plan length / optimal |
|---|---|---|---|---|---|
| P | 34 | **52** | 63 | 76 | 1.06 |
| **M** | 75 | **77** | 82 | 85 | 1.08 |
| R | 53 | 69 | 74 | 83 | 1.13 |
| S | 42 | 63 | 69 | 79 | 1.17 |
| G | 49 | 63 | 71 | 84 | 1.04 |

**Primary and attribution.**
- **Primary (M vs P @200):** 77 vs 52 of 120; discordant pairs 25 / 0; exact McNemar p = 6.0e-8.
- **Attribution (Holm):** M vs R 77 vs 69, p = 0.0078; M vs S 77 vs 63, p = 0.00024. Both are significant, so the
  outcome is ADVANCE.
- M − G = 14 (≥ 5), so the gain is not matched by goal-gated single moves.

**Compute-matched view** (descriptive; censored cells marked in the report):
- Per heuristic call (h_calls ≤ 500 / 1,000 / 2,500), M solves 76 / 76 / 82 vs P's 49 / 58 / 69 (p ≤ 0.00025).
- Per primitive step at ≤ 500, M is not ahead (41 vs 43). From 1,000 primitive steps on, M is ahead (66 vs 52,
  p = 0.0013).

### E11b: learned subgoal ordering (128 fresh execution instances, n = 10–15, strata S/T/H/TH; seed 20261258)

**Primary.** Learned order (r0) vs bottom-up on the OOD pool (n = 96):
- mean log length ratio −0.074 [−0.139, +0.0006] (geometric ratio 0.93);
- all three replicates negative: −0.074, −0.084, −0.041;
- vs random order −0.52 [−0.60, −0.45];
- vs ready-first +0.018 (longer).

**Why REVISE.**
- The CI upper bound is not below 0.
- The order is longer than ready-first.
- Success is 96/96 in every ordered method.

**By stratum.**
- The learned order helps on T (0.81 of BU length) and TH (0.87).
- It hurts on tall-tower H goals (1.09–1.21).

**Label accuracy.** On rows of unseen renaming classes, the learned selector matches its achiever-label teacher at
0.997, vs 0.63 for BU and 0.75 for ready-first. OOD it scores 0.97–0.98.

### Reading

1. **E11a (confirmed at tier B, exact dynamics).** Under the RCX A3 heuristic, multi-step macros mined from Rineya's own
   F-verified plans (BWM B3) make greedy search solve more hard instances at low budgets.
   - The gain is +25/−0 at 200 expansions, and +13 to +27 at matched heuristic-call budgets.
   - The specificity is real: the gain beats random sound macros and shuffled-plan macros of matched shape.
   - This is a **search-level skill-reuse mechanism** (RSM-001). The skills come from the dense model's plans, but the
     gain is realized in an external planner. Nothing is internalized in Rineya.
2. **The specificity depends on the heuristic** (descriptive; not in the decision).
   - With misplaced_blocks, *any* macro library helps a lot: S 90 and M 85 vs P 34 at @200; at @2000, S and M are
     both 115.
   - With beam4, R collapses: 23/120 at every budget.
   - So "mined beats shuffled" holds with the learned A3 value, not universally.
3. **E11b.** The selector learns its achiever-level teacher almost perfectly (0.997 on unseen classes). The execution
   result is therefore bounded by the *label definition*, not by learning. The achiever teacher orders tall towers
   poorly (stratum H), and the hand rule ready-first closes more of BU's excess over the achiever optimum: 0.46 vs
   0.20–0.42.

### Limits

- **Tier B.** Exact dynamics, GBFS/beam4, and 120 / 128 instances.
- **Not tested here:** BWM's LM-F beam planner and Rineya's proposer. The A3 heuristic is RCX's GNN hybrid.
- **Smoke disclosure.** The plumbing smoke's 4 + 4 instances may coincide with the first full-run draws; this was
  disclosed at the freeze.
- **E11b** used one achiever definition, and partial tower goals.

---

## R-10 RCX-E01R step A (DEV2 diagnostic, descriptive): on fresh blind-authored wordings N-CFSD reads half of the positive-rule complements, explicit or implicit alike; frozen branch rule gives **B4 → train R2 + R3** (tier: native-C component artifacts, inference only; DEV), 2026-10-12

**Evidence:**
- `runs/RCX-E01R/dev2/DIAG_DEV2.md` and `.json`, `BRANCH_STEP_A.json`, `eval_*.json` and `preds_*.jsonl.gz` (commit
  `f2d12f2`);
- the battery: seed 20261261, sha `38362338…`;
- P2_v2: 45 blind-authored wordings, sha `7e625a59…` (CX D-34);
- the branch rule, frozen before any step A output: PREREG_E01R §2 and `decide_branch.py` `6835b040…` (CX D-37).

**Setup.** No training. The arms are the E01N adapters N-CFSD s0–s2 (pooled) and N-VR s0. Each was evaluated on the
single-atom strata, on two surfaces:
- `in`: canonical training wording;
- `p2`: one fresh P2 wording per parent, stratified by form.

The form is one of EXPLICIT listed-first (LF), EXPLICIT complement-first (CF), or IMPLICIT (IM).

| Decision accuracy | N-CFSD s0–s2 | N-VR s0 |
|---|---|---|
| Anchor `in`: no-op items (POS / NEG) | 0.99 / 1.00 | 0.94 / 1.00 |
| **POS no-op on `p2`** (LF / CF / IM) | **0.51 / 0.35 / 0.64** (pooled 0.50) | 0.27 / 0.06 / 0.21 |
| POS apply on `p2` (LF / CF / IM) | 0.97 / 0.82 / 0.92 | 1.00 / 1.00 / 1.00 |
| NEG no-op on `p2` (LF / CF / IM) | 0.92 / 0.75 / 0.67 (pooled 0.78) | 0.58 / 0.36 / 0.27 |
| NEG apply on `p2` (LF / CF / IM) | 0.74 / 0.62 / 0.67 | 0.92 / 0.70 / 0.78 |

**Branch (`decide_branch.py`, run once).**
- E − I = −0.21, with template-cluster bootstrap 95% CI [−0.50, +0.08], so B3 does not hold.
- p̄ = 0.50 < 0.65, so B2 does not hold.
- a_in = 0.996, so the instrument is sound.
- The result is **B4: treatment arms R2 (complement-focused twins) and R3 (predicate-first trace)**.

### Reading (descriptive; DEV)

1. **R-08's 14% was partly wording-specific.**
   - On 45 fresh, blind-authored wordings, N-CFSD decides about half of the required positive-rule no-ops correctly,
     against 0.14 on the two spent `ho_tpl` wordings.
   - N-VR s0 manages 0.18. The counterfactual and paraphrase recipe's advantage therefore extends to wordings nobody
     tuned on.
2. **The deficit is not "implicit vs explicit".** Implicit positive wordings are read at least as well as explicit ones
   (E − I < 0).
   - The weakest form is **complement-first**, the default case stated before the listed case. It is worst for
     positive no-ops (0.35) and also lowers positive apply decisions (0.82).
   - So the adapter partly reads a rule by its *first clause*, not by its logical structure.
3. **Fresh negative wordings are imperfect too:** no-op 0.67–0.92 and apply 0.62–0.74. R-08's 0.96–0.97 on its two
   negative `ho_tpl` wordings overstated how well negatives are read.
4. **Consequence (mechanical, CX D-37).** The uniform deficit selects R2 and R3. The confirmation tests both against
   N-CFSD on the sealed, never-read P3 wordings (PREREG_E01R).

### Limits

- DEV. Only 45 templates, 6–9 per form, so per-form rates partly reflect template difficulty.
- The P2 wordings were read by the lead (CX D-34).
- Inference only; no training.
- These numbers do not test any recipe. They choose the arms of the next, preregistered experiment.
