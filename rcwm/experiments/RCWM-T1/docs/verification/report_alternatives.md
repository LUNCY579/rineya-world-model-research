RCWM-T1 verification of claims C1–C4: alternative explanations and overreach

All tests ran on CPU with CUDA_VISIBLE_DEVICES=''. Nothing outside docs/verification/ was modified. "LM head" below means projecting the captured layer-28 answer-slot state (post final norm) through the tied embedding matrix. That matrix is identical in Sparse and Supa (max |diff| = 0).

Bottom line:
- **C1** must be reworded. Its literal statement holds, but it measures the wrong thing: Sparse carries consequence content that Supa lacks, and the change-mask probe cannot see it.
- **C2** is safe as written, provided it is not stretched into "the model did not compute".
- **C3** must be reworded. The R5 "format, not world knowledge" reading is unsupported.
- **C4** must be reworded: "trivial floors" is wrong for L, P adds nothing to the decay story, and "absence of composition" overreaches.

## C1 — one-step change-mask decodability is equal across arms; Supa's generation is ~0

**Strongest alternative:** the metric cannot see the skill the delta installed, so equality is uninformative.

Tests:
1. **Pre-delta baseline.** Layer 8 is bit-identical across arms by construction. Ledger one-frame exact-Δ there is 0.635; the best layer (16) gives 0.846 (Sparse) / 0.812 (Supa). When dev selects a layer ≤ 8, the arms are identical by construction. This happens for P k1 (Supa), the R one-frame arithmetic subset, and several k=4 cells.
2. **Reproduction and equivalence.**
   - One-frame L ridge: +0.034 [−0.005, +0.072]. At fixed layer 20 it is +0.043 [0.000, +0.087], which is borderline.
   - Multi-step-format k1, R: Sparse 0.433 vs Supa 0.341, +0.091 [+0.043, +0.144]. So "indistinguishable" depends on the prompt frame.
   - The CIs leave room for differences of 0.07–0.09, so equivalence is not shown.
3. **Targets that should differ — they do:**
   - **P first answer digit via the LM head.** This is the strongest result.
     - State-changing test items (n=87): Sparse 0.84 vs Supa 0.60.
     - Items where copying s0 gives the wrong first digit (n=39): Sparse 0.67 vs Supa 0.38, diff CI [+0.08, +0.61].
     - DEV replicates it: 0.96 vs 0.47 on changed items, 0.86 vs 0.10 on the copy-fails items.
     - The multi-step k1 frame shows the same: 0.67 vs 0.00.
     - Meanwhile the ridge change-mask probe has both arms at the copy floor. Sparse computes the P consequence at the answer slot in a code the probe does not read.
   - **R new value of the instructed register.**
     - Overall R²: Sparse 0.26 vs Supa 0.10 (+0.16 [+0.05, +0.27]); at k1, 0.24 vs 0.06 (+0.19 [+0.08, +0.30]).
     - Retrieval of the register's old value is equal (0.75 vs 0.76).
     - Split by instruction kind, the gain is almost entirely swap (R² 0.85 vs ≤ 0.67).
     - For add and mix, the new value is not linearly decodable in either arm (R² ≈ 0), even though Sparse scores 0.65 on R generatively. That points to computation during generation, after the answer slot.
   - **Weaker signals** (single split; about 50 contrasts were scanned, so treat as exploratory):
     - L sign bits (one-frame): +0.048 [+0.01, +0.09], but the counterfactual sign bits also improve (+0.043), so it is not selective.
     - L post-event value of the named account at k1: +0.10 [+0.02, +0.18].
     - L start balances: +0.07 / +0.14.
   - **One apparent effect was an artefact.** P applicable-only Δ at k1 showed +0.22, but it vanishes at fixed layers (0.528 vs 0.496). The reported CIs do not include layer/λ selection noise.
4. **Supa's generation, leniently parsed.** Supa wraps P and R answers in a `{"before":…,"after":…}` object, and P was also truncated at 12 tokens. Lenient parsing recovers 95% of P one-frame outputs, but they are mostly s0 echoes (77%). On items whose true state changes, Supa's exact-state accuracy is L 0.03–0.06, P 0.00, R 0.04–0.05.
5. **Two context caveats.**
   - Sparse generation on this bank does not exist yet. The contrast relies on SELECT-VIEWS (Sparse: L 0.41, P 0.75, R 0.65).
   - Latent rank order (L > R > P) is the reverse of Sparse's generative skill order (P > R > L).

**Suggested rewording:** "On an exact change-mask ridge/MLP readout at the answer slot, Titan-Sparse and Supa are not distinguishable in the one-step frame (L +0.034 [−0.005, +0.072]; MLP −0.005 [−0.024, +0.014]; R +0.014 [−0.058, +0.087]; P both at copy floor). This readout is insensitive to the trained skill. On the same states, Sparse's own LM head reads the P consequence (first digit correct on 0.84 of state-changing items vs 0.60 for Supa; DEV 0.96 vs 0.47), and Sparse encodes the swapped-in R value better (R² 0.85 vs ≤ 0.67). Supa's generated states are near 0 on items whose state changes (lenient parse: L 0.03–0.06, P 0.00, R 0.04–0.05)."

Drop the R3 heading "Supa encodes the same one-step consequence as Titan-Sparse"; it is unsupported.

## C2 — the counterfactual is fit as well as the truth, so decodability is not evidence of computation

**Strongest alternatives:** the counterfactual is not matched in difficulty, or the test has no power to detect computation.

1. **Selectivity reproduced on the 180 discriminating ledger items (26 parents).**
   - Sparse −0.020 [−0.055, +0.014]; Supa −0.001 [−0.032, +0.028].
   - k1: −0.020 [−0.063, +0.022] and −0.026 [−0.061, +0.006].
2. **Difficulty matching.** Base rates match (sign bits 0.03–0.06; changed bits 0.27–0.33). But the counterfactual is slightly easier for every input-only learner: on discriminating items, symbolic ridge +0.05, symbolic MLP +0.03, layer-4 ridge +0.05, layer-8 ridge +0.04 / 0.00. So the null expectation is about −0.03 to −0.05, not 0. Relative to that, LM selectivity is roughly +0.01 to +0.03, still about zero.
3. **Power.**
   - The ceiling caps achievable selectivity at 1 − cf = 0.10 (Sparse) or 0.17 (Supa).
   - Injecting the true Δ linearly into the Sparse layer-16 states, even at 0.02 × the mean per-dimension SD, gives selectivity +0.08 to +0.09. The observed upper CI bounds rule this out.
   - So the test excludes an explicit linear encoding of the full true change mask. It cannot detect nonlinear or partial computation, and the P LM-head result shows such probe-invisible content exists.
   - Exact-Δ is dominated by the "changed" bits, which are input-linear under both rules (the touched-entity rule alone scores 0.95 on them).
4. **R counterfactual flaw.** Inverting the conditional is the negation of one comparison bit, which a linear readout handles equally well either way. Only the mix (2× → 3×) half is informative. Separately, the arms do differ in R selectivity at k1: +0.28 [+0.08, +0.52], n=45 items, not seen in the one-frame prompt.

**Verdict:** safe as written for ledger. Add the qualifier: "This shows only that the ledger mask is not linearly encoded beyond what the probe computes from inputs. It is not evidence that the model failed to compute the consequence."

## C3 — the delta massively shifts late states, CKA stays high, probes do not transfer

**Reproduced** on test one-frame items (n=640, excluding Supa's top-5 massive dims):
- Relative L2 from layer 12 to layer 28 is 0.25 / 0.59 / 0.83 / 1.09 / 0.71.
- CKA is 0.99 / 0.93 / 0.98 / 0.94 / 0.91.
- Layers 0–8 are identical, which is trivially expected because the delta starts at block 8.

Alternatives tested:
1. **How much is input-independent.** Shares of shift energy for layers 12 / 16 / 20 / 24 / 28:

   | Constant offset model | L12 | L16 | L20 | L24 | L28 |
   |---|---|---|---|---|---|
   | One global mean | 90% | 82% | 47% | 30% | 41% |
   | Per-family means | 99.6% | 98% | 93% | 70% | 84% |
   | Per family × frame (all test items) | 99.7% | 98.4% | 94.1% | 73.8% | 86.9% |

   The mean shift alone produces median per-dimension |z| of 1.30 / 2.11 / 1.51 / 1.18 / 0.99, out of a total of 1.40 / 2.26 / 1.92 / 1.73 / 1.46. The "~1.6 SD per dimension" headline is mostly a constant offset.
2. **Not a few dims and not the norm.**
   - Beyond the top-5 massive dims, the top 20 dims carry 12–20% of the shift and the top 100 carry 31–44%.
   - The massive dims themselves carry 2–11%.
   - After per-row RMS normalisation, layer 24 still shifts 0.92 (vs 1.09). Norm ratio is 1.0–1.17. The final RMSNorm does not explain the shift.
3. **Sparse is nearly a linear function of Supa.** A ridge map fit on TRAIN explains 90–99.5% of the held-out shift (R² 0.88–0.99). The mean shift alone explains 28–88%.
4. **The mean shift is not behaviourally irrelevant.**
   - On L and R, both arms put '{"' first (KL 0.08–0.13).
   - On P, Supa puts '{"' first on 100% of items and Sparse puts a digit first on 100%.
   - Adding the train-mean shift to Supa switches 100% of P items to a digit. That component is the format switch.
   - It carries no content: first-digit accuracy where copying fails stays 0.38 (one-frame) / 0.21 (k1), vs Sparse 0.67.
   - Linearly mapped Supa also just copies (0.10 correct, 0.85 copy).
   - The content lives in a small item-specific residual that Supa's state does not contain.
5. **Non-transfer is not diagnostic.**
   - Cross-arm transfer stays poor after full linear alignment (0.15–0.41, vs 0.78–0.85 within an arm).
   - Within a single arm, a probe trained on one-frame items and tested on k1 items fails just as badly (raw 0.00–0.02, re-standardised 0.06–0.56). The probes break under any shift.

**Suggested rewording:** "Blocks 8–27 shift late answer-slot states by relative L2 0.25–1.09 (excluding top-5 massive dims). The shift is mostly a constant per-family offset (≥ 93% of shift energy at layers 12–20, 70–84% at layers 24–28) spread over hundreds of dims, and Sparse remains nearly a linear function of Supa (held-out R² 0.88–0.99; CKA 0.91–0.99). The constant offset switches P's first token from a JSON wrapper to a bare integer. The small item-specific remainder carries consequence content Supa lacks (P first digit). Cross-arm probe non-transfer is not informative, because the same probes also fail across prompt wordings within one arm."

Also drop R5's "mostly input-independent … rather than new world knowledge" and "probes do not transfer even after re-standardization" as evidence of anything.

## C4 — multi-step decodability decays to trivial floors

**Reproduced:** Sparse event-boundary tracking recomputed exactly (L 0.865 / 0.524 / 0.322 / 0.192; R 0.452 / 0.115 / 0.024 / 0.010; P at copy floor).

Alternatives tested:
1. **Prompt-format artefact.** Not at k=1: the multi-step k1 frame scores at least as well as the trained one-step frame (L 0.875 vs 0.846; R 0.433 vs 0.385; P first digit identical). For k ≥ 2 this cannot be separated without Sparse generation, which is pending.
2. **Interference, not only failed composition.** Decoding the mask that a single event would cause on its own:
   - L event 1: 0.82 at k2 and 0.64 at k4, vs 0.875 at k1.
   - Later events: 0.45–0.66.
   - Per-event information is itself lost as events are added.
3. **Floors.**
   - L stays above copy and majority (k4: 0.221 vs 0.0 / 0.091) but falls below the touched-entity rule (0.591 at k2, 0.433 at k4).
   - Even the changed bits alone fall to 0.457 at k4, vs 0.889 for the touched rule. Sign bits drop to chance (balanced accuracy 0.50–0.53).
   - A linear readout of perfect symbolic inputs decays in parallel (L 0.70 / 0.35 / 0.26; R 0.24 / 0.10 / 0.01), so a linear probe would decay even on an encoder with no composition.
   - At k=4, dev often selects layer ≤ 8, so arm equality there is by construction.
4. **P is uninformative for this probe**, since it sits at the copy floor even at k=1 where the LM head shows content. However, via the LM head at k=4, on the 21 items where every heuristic predicts the wrong first digit, Sparse is right 0.24 of the time (majority-digit rate 0.29). It matches copy / first-only / last-only 0.48 of the time. That is consistent with no composition, but n is small.

**Suggested rewording:** "Linear change-mask decodability falls to or below simple heuristics by k=2 (L 0.875 / 0.500 / 0.221 vs the touched-entity rule 0.755 / 0.591 / 0.433; R 0.433 / 0.091 / 0.010, at the first-event floor from k=2), both at the answer slot and at event boundaries. P sits at the copy floor at every k, including k=1, which is a probe failure (Sparse's LM head reads the one-step P consequence). So P says nothing about decay. Single-event consequences also fade as events are added, so the decay mixes interference with absent composition. No composed state is linearly readable at these positions, and on 21 contested k=4 items Sparse's own first P digit follows the heuristics, not the truth."

"Absence of composition" stays unsupported until Sparse's multi-step generation is scored.

## Files

All in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/verification/`:

- **Scripts:**
  - `vh.py`
  - `c1c2_tests.py`
  - `c1_gen_lenient.py`
  - `c1_extra.py`
  - `c1_p_first_digit.py`
  - `c13_logit_lens.py`
  - `c3_shift.py`
  - `c_more.py`
  - `c4_p_first_digit.py`
- **Results:**
  - `c1c2_results_all.json`
  - `c1_gen_lenient.json`
  - `c1_extra.json`
  - `c1_p_first_digit.json`
  - `c13_logit_lens.json`
  - `c3_results.json`
  - `c_more_results_all.json`
  - `c4_p_first_digit.json`
  - `c4_tracking_sparse.json`
- **Logs:**
  - `c1c2_log.txt`
  - `c3_log.txt`
  - `c13_log.txt`
  - `c_more_log.txt`