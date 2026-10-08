# RCWM-T1 pilot claims C1–C4: merged review

Three verifiers checked the claims: code-audit, recompute and alternatives. I spot-checked the cited code lines and they match.

## 1. Per-claim verdicts

| Claim | Verdict | Key evidence (verifier) |
|---|---|---|
| **C1** (H2b) | **SUPPORTED-WITH-REWORDING.** The clause "P and R unparseable" is **BUG-AFFECTED** (B3). | **Reproduced exactly:** ridge 0.846/0.812, diff +0.034 [−0.005, +0.072]; MLP 0.913/0.918, −0.005 [−0.024, +0.014]. Logistic: L 0.885/0.894. Supa generation: L 0.067 one-frame, 0.106 k1 (recompute).<br>**Equivalence not shown:** the CIs allow differences of 0.07–0.09. At fixed L20 the diff is +0.043 [0.000, +0.087] (alternatives).<br>**R depends on the probe:** ridge +0.014 [−0.058, +0.087], but logistic +0.082 [+0.019, +0.144] and exact-selected ridge k1 +0.053 [+0.005, +0.106] (recompute).<br>**The probe misses the skill:** on state-changing items, the LM-head P first digit is correct for 0.84 Sparse vs 0.60 Supa. On copy-fails items it is 0.67 vs 0.38, CI [+0.08, +0.61]. DEV gives 0.96 vs 0.47. R swap R² is 0.85 vs ≤0.67 (alternatives).<br>**"Unparseable" is an artifact:** it comes from the P 12-token cap plus the strict parser. With lenient parsing, Supa P one-frame scores 0.406 (audit) / 0.429 (recompute), and all of it is copying. On changed items Supa scores L 0.03–0.06, P 0.00, R 0.04–0.05. |
| **C2** | **SUPPORTED** at its core. Minor rewording: "difficulty-matched" is only base-rate-matched, and a scope qualifier is needed. | **Ridge selectivity on 180 items / 26 parents:** Sparse −0.022 [−0.055/−0.056, +0.014]; Supa 0.000 [−0.032, +0.028].<br>**Other variants:** between −0.028 and +0.039, none significant. Recompute flags Supa logistic at +0.039 [−0.024, +0.104].<br>**The counterfactual is slightly easier** for input-only learners (+0.03 to +0.05), so the null is about −0.03 to −0.05. Measured against that null, LM selectivity is about +0.01 to +0.03.<br>**Power:** a 0.02-SD linear injection of the true Δ gives +0.08 to +0.09, which the CIs exclude. Nonlinear or partial encoding cannot be detected. |
| **C3** | **BUG-AFFECTED** (B1, B2). The qualitative pattern holds; the interpretation does not. | **R5 table can't be reproduced:** recomputed rel-L2 without top-5 dims is 0.248/0.586/0.826/1.087/0.711. Median \|z\| is 1.40/2.26/1.92/1.72/1.46, not 1.03/1.58/1.61/1.61/1.36. CKA is 0.991/0.925/0.979/0.937/0.910.<br>**Pooled CKA hides within-family change:** ledger-only CKA is 0.793 (L20) and 0.604 (L24). Family-mean-removed CKA is 0.437–0.864. Within-family median \|z\| is 5–13 (ledger 6.0–7.4).<br>**The shift is mostly a constant offset:** ≥93% of shift energy at L12–20 and 70–84% at L24–28 is a per-family offset. A linear map from Supa states explains 0.88–0.99 (held-out R²) of Sparse.<br>**The offset is the P format switch** (`{"` vs a digit). It carries no P content.<br>**Non-transfer is not diagnostic:** probes also fail across frames within one arm (raw 0.00–0.02, re-standardized 0.06–0.56). Ledger one-frame Supa→Sparse gives 0.034 vs the claimed 0.139, depending on λ. |
| **C4** | **SUPPORTED-WITH-REWORDING.** | **Reproduced exactly:** answer slot L 0.875/0.500/0.221 and R 0.433/0.091/0.010. Event-boundary tracking: L 0.865/0.524/0.322/0.192, R 0.452/0.115/0.024/0.010.<br>**"Trivial floors" is wrong for L:** at k4 it is above copy/majority (0.0/0.091) but below the touched-entity rule (0.755/0.591/0.433).<br>**A linear readout of perfect symbolic inputs also decays:** L 0.70/0.35/0.26, R 0.24/0.10/0.01.<br>**Interference:** decoding of event 1's single-event mask falls to 0.82 (k2) and 0.64 (k4).<br>**P is a probe failure:** it sits at the copy floor even at k1 (recompute 0.585–0.612 vs the claimed 0.625; floor 0.612), although the LM head reads P.<br>**Arm equality is sometimes by construction:** at k4, dev often picks a layer ≤8. |

**Unresolved between verifiers:**
- **R k1, Supa value:** alternatives reports 0.433 vs 0.341, +0.091 [+0.043, +0.144]. Recompute reports 0.433 vs 0.370 (n.s.) under per-feature selection, and 0.394 vs 0.341 under exact selection. Settle the selection rule before citing either.
- **Lenient P one-frame:** 0.406 (audit) vs 0.429 (recompute).

### Recommended wording

**C1:** "In the one-step frame, an exact change-mask readout at the answer slot does not distinguish Titan-Sparse from Supa on ledger: ridge 0.846 vs 0.812, +0.034 [−0.005, +0.072]; MLP 0.913 vs 0.918, −0.005 [−0.024, +0.014]; logistic 0.885 vs 0.894. Equivalence is not established, because the CIs admit differences of about 0.07–0.09. For register the result depends on the probe (ridge +0.014 [−0.058, +0.087]; logistic +0.082 [+0.019, +0.144]). For P, both arms are at the copy floor. This readout does not capture the trained skill. On the same states, Sparse's LM head reads the P first digit on 0.84 of state-changing items vs 0.60 for Supa (DEV 0.96 vs 0.47), and Sparse encodes the swapped-in R value better (R² 0.85 vs ≤0.67). Supa's generated exact state is near zero wherever the state changes (lenient parse: L 0.03–0.06, P 0.00, R 0.04–0.05). Its strict-parse failures on P and R come from a `{"before","after"}` wrapper and a 12-token P cap, not from missing answers." Also drop the R3 heading "Supa encodes the same one-step consequence as Titan-Sparse".

**C2:** "With base-rate-matched counterfactual rules (ledger accounts rotated A→B→C→D→A), ridge selectivity on 180 discriminating ledger test items (26 parents) is about 0 for both arms: Sparse −0.022 [−0.055, +0.014], Supa 0.000 [−0.032, +0.028]. Logistic and k=1 variants fall between −0.028 and +0.039, none significant. Input-only learners find the counterfactual slightly easier (+0.03 to +0.05). The test rules out an explicit linear encoding of the full true ledger mask, but not nonlinear or partial computation. So ledger change-mask decodability is not evidence that the model computed the consequence, and the lack of selectivity is not evidence that it didn't."

**C3:** "Layers 0–8 are bit-identical across arms; the delta touches only blocks 8–27. On TEST one-frame items (n=640, excluding Supa's top-5 massive dims), the delta shifts late answer-slot states by relative L2 0.248/0.586/0.826/1.087/0.711 at L12/16/20/24/28. Most of this is a constant per-family offset: 93–99.6% of shift energy at L12–20 and 70–84% at L24–28. Pooled CKA (0.91–0.99) reflects between-family structure being preserved. Within a family CKA is lower (ledger 0.79 at L20, 0.60 at L24; family-mean-removed 0.44–0.86). Sparse is still nearly a linear function of Supa (held-out R² 0.88–0.99). The offset switches P's first token from `{"` to a digit but carries no P content; the item-specific remainder does. Ridge probes do not transfer across arms, even after full linear alignment (0.15–0.41 vs 0.78–0.85 within an arm). They also fail across prompt frames within one arm, so non-transfer is not diagnostic." Delete R5's "input-independent … rather than new world knowledge".

**C4:** "Linear change-mask decodability falls to or below simple heuristics by k=2, both at the answer slot and at event boundaries. Ledger scores 0.875/0.500/0.221 vs the touched-entity rule's 0.755/0.591/0.433; register scores 0.433/0.091/0.010. A linear readout of perfect symbolic inputs decays in parallel (L 0.70/0.35/0.26), and single-event consequences also fade as events are added (L event 1: 0.82 at k2, 0.64 at k4). So the decay mixes probe limits, interference and absent composition. P is at the copy floor at every k, including k=1 where the LM head reads P, so P is uninformative here. 'Absence of composition' remains unsupported until Sparse's multi-step generation is scored."

## 2. Confirmed code bugs (by severity)

1. **B1 (High, C3): family pooling.**
   - **Where:** `scripts/mechanism_report.py:50` pools all families' TEST `-one` items. `cka()` (lines 28–30) and `shift()` (33–39) then run on the pooled matrix. The SD at line 37 includes between-family variance, which is 92–97% of the variance at L12–28.
   - **Fix:** compute per family and per family×format. Add CKA with family means removed. Use within-family SD for \|z\|. Report the per-family-mean share of the shift; `rel_l2_mean_part` already exists at line 39. Write the output to `mechanism.json`.
2. **B3 (High, C1): output truncation, strict parser and scoring.**
   - **Truncation:** `rcwmt/generation.py:147-148` returns 12 tokens for P. 84% of Supa's P k1 outputs and 24% of R one-frame outputs hit their caps.
   - **Parser:** `rcwmt/env.py:316-332` requires a bare integer, or a dict with exactly the right keys (line 328).
   - **Scoring:** `rcwmt/fidelity.py:206-208` counts an unparseable answer as all features wrong.
   - **Fix:** raise the P cap to cover the wrapped answer (re-generation is needed; lenient parsing of truncated numbers is unreliable). Add a lenient mode that unwraps `"after"`. Report parse rate, strict exact, lenient exact, the copy-imputed metric (`mi`, lines 209–210) and changed-item-only exact separately.
3. **B2 (Medium, C3): irreproducible table.**
   - The R5 table in `RESULTS_PILOT.md` (12:58) predates `mechanism_report.py` (13:03). There is no `mechanism.json`, and no split×format combination reproduces it.
   - **Fix:** rerun with saved JSON that records the item set, and replace R5.
4. **B4 (Medium-low, P only): P one-frame targets don't match the prompt.**
   - The P `SYSTEM_ONE` prompt (`rcwmt/env.py:44-48`) says the action is applicable. But the `-one` gold (`rcwmt/bank.py:141-146`) uses `env.step`, which makes an inapplicable action a no-op.
   - 44.2% of P TEST first actions are inapplicable. On 24.6% of items, gold differs from the literal instruction.
   - **Fix:** in the next bank, draw only applicable first actions for `-one` items, or state the no-op rule in the prompt. For now, report P one-frame on the consistent subset.
5. **Low, latent: stale index.** `scripts/run_arm.py:53-59` rewrites `index.json` before the skip check, so a changed bank would silently desync it from `last.npy`. **Fix:** check `bank_sha256` against the existing index and abort on mismatch, or write only after deciding to capture.
6. **Low, latent: hand-written capture record.** Sparse's `CAPTURED.json` was written by hand. The fingerprint 818be9c6b6a3 matches, so the chain will skip recapture correctly. **Fix:** regenerate it via code or document the provenance.
7. **Low, documentation.** The delta's relative norm is 0.048% of all parameters (0.052% of blocks 8–27), not 0.056%.

**Methodological caveats (not code bugs):**
- **CIs omit selection noise.** The bootstrap CIs don't include layer/λ selection noise. A P applicable-only +0.22 at k1 vanished at fixed layers (0.528 vs 0.496).
- **The ridge selection rule changes R's significance.** Ridge is selected by per-feature accuracy but scored on exact match. Audit found ≤0.04 change for C4, but recompute shows R significance flips with the rule.
- **Some arm equality is by construction.** Wherever dev picks a layer ≤8, the two arms are equal by construction.

## 3. CPU-only analyses (priority order)

**Global top 3:**
- (a) Score P, and L/R where tractable, by teacher-forced log-probability of the gold answer vs heuristic candidates (copy / first-only / last-only / touched-entity), FP32 on CPU for both arms (audit_02 shows FP32 forwards are feasible). This avoids generation format issues and tests C1 and C4 directly.
- (b) Fix B1 and B2 and regenerate C3.
- (c) Use a nested bootstrap that re-selects layer and λ inside each resample, plus fixed-layer sweeps.

**C1**
1. Nested-selection bootstrap and a fixed-layer sweep (L12–28) for L/R/P with ridge (both selection rules), logistic and MLP. Correct for multiple comparisons. Run a pre-registered TOST equivalence test.
2. Teacher-forced gold-vs-copy log-probability on state-changing items, both arms. This strengthens or falsifies "Sparse computes P/R, Supa does not."
3. Extend the LM-head readout beyond the P first digit (full integer via the next-token chain) and replicate on DEV+TEST pooled.
4. Final lenient re-score table stratified by changed vs unchanged items. Regenerate P with a larger cap on CPU to remove truncation.

**C2**
1. Report LM selectivity minus the input-only null (symbolic ridge/MLP, L4/L8 ridge) as a paired difference-of-differences with parent-level CIs.
2. Compute selectivity restricted to bits that are not input-linear (sign/magnitude bits where the touched-entity rule fails). Add a counterfactual that changes arithmetic, not just labels.
3. Injection-scale sweep to estimate the minimum detectable selectivity. Only 0.02 SD (→ +0.08 to +0.09) has been run.
4. **R counterfactual:** keep only the mix (2×→3×) subset. Replicate the k1 arm difference (+0.28 [+0.08, +0.52], n=45) on DEV.

**C3**
1. Rerun the B1-fixed metrics per family and per family×frame. Report the offset share, family-mean-removed CKA and within-family \|z\|.
2. Ablation: Supa + mean offset + item-specific residual projected onto its top-k PCs. Find the smallest k that restores the LM-head P first digit (0.38 → 0.67), which localizes the content.
3. Use within-arm cross-frame transfer as the reference for cross-arm transfer. Report transfer as a ratio to it.
4. Small-n CPU activation patching (Sparse residual into the Supa forward at L12–24) to test whether the shift is causal for P content.

**C4**
1. MLP probes on stored states and on symbolic inputs at k=1/2/4, to separate linear-readout limits from absent composition.
2. Teacher-forced log-probability of the gold composed state vs heuristic composites at k=2/4, both arms. This is the closest CPU substitute for the pending Sparse multi-step generation.
3. LM-head P first digit at k=2/4 on contested items, pooling DEV and TEST to get n well beyond 21.
4. Per-event single-mask decoding curves at each event boundary vs k, to quantify interference. Run at fixed layers >8 to avoid arm equality by construction.

All verifier scripts and results are in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/verification/`.