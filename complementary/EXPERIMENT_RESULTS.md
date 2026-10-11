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
