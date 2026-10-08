# PREREG-1 review: scope, wording and value

**Evidence.** All numbers are pilot TEST only, CPU only. The script that produced them is `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/prereg_review/scope_pilot_checks.py`. The sealed bank was only inspected for structure: 450 test parents, 8 candidates each, and the twin is stored under `p["twin"]` rather than among the candidates.

**Dry run.** Still running. Its first line (L k1) reproduces `decisions_sparse.log` exactly.

## MUST-FIX before freeze

**1. C3 has no valid pilot basis for R, and the dry run cannot test it.**
- **Cause:** Supa's pilot chains were generated at 12:56–13:01. That was before the 64-token cap (13:45) and before lenient parsing in `generation.chain` (14:30).
- **Pilot R:** 208/208 Supa chains broke, all on a wrapped `{"before","after"}` answer. 116 of those breaks would parse leniently. Every R candidate was therefore imputed, all scores tied, and selection was random (regret 1.007). The §1 value "R, k=4 = 0.994" is an artefact of the instrument.
- **Pilot L:** 40% of Supa's L chains were imputed.
- **Failure scenario:** on the confirmation bank, Supa's chains survive under the frozen instrument. C3 is then tested against an unknown quantity, and §1 misstates the evidence.
- **Fix:**
  - Regenerate Supa `chain_one` and `chain_m1` on pilot TEST with the frozen generation code. This is pilot data, so it is allowed. Update §1.
  - Pre-declare the C3 comparator as the per-family minimum regret over Supa's {e2e, chain_one, chain_m1}. This is conservative for the claim and answers "Supa was handicapped by Titan's frame".

**2. The decision-rule wording does not match the code, and there is no claim mapping.**
- The prereg says "Holm-adjusted one-sided 95% upper bound". The code (`strat_boot`/`holm`) computes the bootstrap share of means ≥ 0, Holm-adjusts that p, and reports an unadjusted two-sided `ci95`.
- Nothing says which combination of C1–C3 confirms the §1 claim.
- **Fix wording:**
  > Cj PASSES iff its Holm-adjusted one-sided bootstrap p (share of 10,000 equal-family-weight resampled means ≥ 0) is < 0.05 AND its point estimate is ≤ −0.10. The §1 claim is confirmed iff all primaries PASS. Each primary is otherwise reported individually under §8 wording.

**3. Most of the declared secondaries are not implemented: this is a forking path.**
- `confirm_analysis.py` produces CIs only. It computes no p-values and no BH correction, no exact-state accuracy, no twin consistency, and no directional P test.
- Twins only get e2e generation (`run_arm` builds chains from `p["candidates"]`), so twin consistency for chain cannot exist.
- **Failure scenario:** these analyses get written after unsealing, with free choice of metric and test.
- **Fix:** implement every secondary with exact metric, direction and test before freeze. Otherwise label them "exploratory, untested". Drop chain-twin consistency or define it as e2e-only.

**4. The freeze record is incomplete.**
- `confirm_analysis.py` imports `scripts/analyze.py` (`load_gen`), but that file is not hashed.
- B3, S and B2 are fit on unhashed pilot artefacts: the `bank_pilot.json` sha and `runs/bank_pilot/sparse/{last.npy,index.json}`.
- `heads.py` sets the torch thread count from `RCWM_THREADS`. A different thread count can change float reductions, and with them early stopping and the selected learning rate.
- **Fix:** hash all of the above, and fix `RCWM_THREADS=8` in §6.

## SHOULD

**5. Add a primary C4: G_chain(Titan) − G_e2e(Titan).** This is the most valuable LLM claim available: same model, same rules, same φ; only the inference procedure differs.
- **Pilot k=4:**
  - L: −0.340 (SD 0.316)
  - R: −0.438 (SD 0.191)
- **It is not a prompt-frame artefact:** chain_m1, which uses the multi-step rule one event at a time, matches chain_one (Δ = −0.002 L, −0.005 R).
- **Power:** ample; at about 0.02 SE, Holm over four costs almost nothing.

**6. Pre-declare floor and ablation contrasts, and disclose them.**
- **Exact first-event-only simulator (`F_first_only`) at k=4:** L 0.595, R 0.803. It beats B3 (0.813) and S (0.746) on L.
- **G_chain − F_first on L:** only −0.088 (SD 0.38). That is below the −0.10 margin.
- **G_chain truncated after Titan's own step 1:** L −0.273, R −0.441. Truncated after step 3: L −0.089, R −0.164. So the later steps do add value.
- **Fix:** add secondaries with stated direction:
  - G_chain − F_first_only < 0
  - G_chain(4 steps) − G_chain(step-1 state) < 0

  Without these, C1 and C2 can be passed against comparators that are themselves beaten by a one-step heuristic.

**7. C2 wording overstates the comparison.**
- S is an end-to-end k=4 MLP trained on about 1.1k labels. It is not a matched one-step learner run in a chain.
- Beating it does not show "LLM > no-LLM simulator".
- **Fix:** reword the C2 claim as "beats an end-to-end symbolic MLP trained on the same pilot k=4 labels". Optionally add an S_chain secondary: one-step symbolic MLP, chained.

**8. The pooled result can be carried by a single family.**
- On the pilot, R effects are about 2× L effects with about half the SD.
- **Fix:** pre-declare that "holds in both families" requires each family's point estimate ≤ −0.10 and a one-sided p < 0.05 (intersection-union test, no extra correction). If that fails, the claim is reported as "pooled; driven by {family}".

**9. Layer wording does not match the code.**
- The code picks the layer per (family, k), by ridge DEV score on the Δ change mask. B3 is then an MLP on φ at that layer.
- **Fix:** state exactly that.

**10. Make the P prediction directional.**
- **Pilot:** G_chain − B3 at P k=4 is +0.14.
- **Chain vs e2e exact state at P k=4:** 0.000 vs 0.143.
- **Fix:** declare that at P k=4, G_chain is not better than B3 (predicted ≥ 0) and that chain exact-state accuracy is below e2e.
- **Note:** confirmation P parents require ≥ 4 distinct final states and pilot P parents do not, so P is not like-for-like with the pilot.

**11. Minimum effect: −0.10 is sensible, but say what it is.**
- It is a threshold on the point estimate, not a margin under test.
- Pilot effects are −0.24 to −0.61, so −0.10 leaves room for winner's-curse shrinkage. The pilot selected chain_one, k=4 and L/R after seeing the data.
- B3 at R k=4 is 0.965, essentially random, so C1 on R amounts to "beats near-random".
- **Fix:**
  - Add a "strong pass" when the Holm-adjusted bound is < −0.10.
  - Always report G_chain's absolute regret and the share of goals where the optimal plan is chosen. `regret_table` computes the latter, but `run()` drops it.

**12. Pre-declare a deviation policy, and test the confirm path.**
- The dry run never exercises the confirm-mode `ArmUnion` path.
- A missing generation record is silently treated as unparseable (`_gen_state`).
- **Fix:**
  - Assert that every candidate ID has a generation record.
  - Report imputation rates per arm.
  - Test `ArmUnion` on a fake split.
  - Allow only logged, non-statistical crash fixes, with a diff.

## NOTE

**13. Exact-state accuracy works as a key secondary, not a primary.** At k=4 it is at floor (L 0.024, R 0.087).
- **Chain vs e2e at k=2:**

  | Family | Chain | e2e |
  |---|---|---|
  | L | 0.202 | 0.072 |
  | R | 0.361 | 0.091 |

- **A compounding check is worth confirming:** chain accuracy at k=4 is close to (one-step accuracy)⁴.

  | Family | Chain k=4 | acc₁⁴ | e2e k=4 |
  |---|---|---|---|
  | L | 0.024 | 0.023 | 0.000 |
  | R | 0.087 | 0.138 | 0.005 |

  Externalising each step limits errors to per-step compounding; in-context composition does worse than compounding. That result speaks directly to "multi-step reasoning at floor".

**14. Do not promote B3 − B2 (out-of-span).** It holds largely by construction: a scalar evaluator never trained on out-of-span goal directions cannot rank them. R k=4 is already n.s. Keep it as a design check.

**15. Do not imply the fine-tuning was free.** PREREG-1 says nothing about forgetting, so the claim must not suggest the delta had no cost.

**16. Holm-adjusted p-values are not monotonised.** This affects reporting only; the pass/fail decisions are correct.

## Recommended claim wording (§8)

**All primaries pass (C1–C4):**
> On 300 fresh ledger and register-machine parents, choosing among 8 four-step plans by chaining Titan-Sparse's own one-step predictions, with each answer fed back as the next state, and scoring the predicted final state with a linear goal readout, gave normalised regret X (0 = oracle, 1 = random) over 256 unseen goals. That is at least 0.10 lower (Holm one-sided p < .05) than each of:
> - an MLP consequence head on Titan-Sparse's frozen answer-slot states (C1)
> - an end-to-end symbolic MLP trained on the same pilot labels (C2)
> - the best generative procedure of base Supa 1.7 (C3)
> - Titan-Sparse's own single-shot four-step answer (C4)
>
> Limits: horizon 4 only; families L and R only. On P, where Titan applies inapplicable actions, no advantage was predicted. An exact simulator of the first event alone reached Y. Nothing here is claimed about off-task forgetting.

**Partial outcomes:**
- **C1 or C4 pass, C3 fails:** "Explicit stepwise rollout beats latent and single-shot readouts, but the advantage is not attributable to the 384-example delta."
- **C3 passes, C1 fails:** "Fine-tuning installed a chainable one-step skill, but chained rollout does not beat a latent consequence head on the same states."
- **C4 fails, C1 passes:** "Explicit generation beats the latent head, but stepwise decomposition is not the source."
- **C2 fails:** "No decision advantage over a small symbolic learner given the same labels."
- **Significant but point estimate above −0.10:** "Directionally confirmed, below the pre-set practical threshold."
- **Pooled pass but one family fails item 8:** "Pooled effect, driven by {family}; not shown for {other}."
- **G_chain − F_first not significant on a family:** add to the claim, "On {family}, not better than exactly simulating the first event alone."