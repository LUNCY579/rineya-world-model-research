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
