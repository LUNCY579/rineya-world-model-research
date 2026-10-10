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
