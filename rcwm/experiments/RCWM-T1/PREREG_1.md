# RCWM-T1 PREREG-1: a frozen LM's one-step skill used as a serial consequence simulator for goal-conditioned decisions

**Status:** FROZEN (see §9). This version incorporates the pre-freeze review in `docs/prereg_review/AMEND.md`. The status changes to FROZEN only when §9 is filled in, and that must happen before any model touches `data/bank_confirm.json`.

## 1. Claim under test

The claim:
> At decision horizon 4, choosing among precommitted plans by **chaining Titan-Sparse's own one-step predictions** gives lower goal-conditioned regret than single-pass alternatives at an equal label budget. Each predicted state is fed back as the next `before` in Titan's trained one-step frame. The predicted final state becomes consequence features φ, scored with a linear goal readout w·φ (RCWM "B4-roll").

The pilot evidence is exploratory: `RESULTS_PILOT.md` R6–R7 and the dry run in `runs/bank_pilot/analysis/PREREG_1_DRYRUN.json`. Values come from the code-validation dry run on pilot TEST (`PREREG_1_DRYRUN.json`), which used the pilot generations. Titan's outputs are unaffected by the instrument changes; Supa's are not. The regenerated pilot run under the frozen instrument is non-binding and runs after the confirmation jobs, so C3 has no valid pilot estimate.

| Method (pilot TEST, k=4, normalised regret) | L | R |
|---|---|---|
| G_chain(Titan) | 0.507 | 0.386 |
| F_touched (answer-free) | 0.599 | 0.683 |
| B3 (consequence head, frozen Titan states, ~1.1k labels; DEV-selected) | 0.780 | 0.839 |
| S (symbolic learner, same labels; DEV-selected) | 0.765 | 0.832 |
| G(Supa) best mode | no valid pilot estimate | no valid pilot estimate |
| G_e2e(Titan) | 0.842 | 0.818 |

The earlier Supa values (0.835 / 0.994) are void. They used a 56-token cap and strict chain parsing (DECISIONS D-12): 0 of 208 R chains reached step 4, so the gap was an instrument artefact.

## 2. Data

- **Confirmation bank:** `data/bank_confirm.json`, sha256 `8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16`.
  - 150 TEST parents per family, K = 8 precommitted plans, H = 4.
  - P parents have ≥ 4 distinct final states.
  - Parents are deduplicated against the pilot. No transition collides with the 6,464 historical payloads or with the pilot's transitions.
  - Order twins (`-t0`) are generated, e2e and chained, for the twin-consistency secondary only. They are never part of a decision set.
- **Fitting and selection:** `data/bank_pilot.json`. TRAIN is used for fitting and DEV for all selection.
- **S@abundant secondary:** `data/bank_s4000.json` (sha256 `af1336c2…22bb4ce`), 4,000 simulator-labelled parents per family for L and R, seed 777001, deduplicated against the confirmation parents.

## 3. Subjects and instrument

- **Titan-Sparse:** `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, parameter fingerprint prefix `818be9c6b6a3`.
- **Supa 1.7:** fingerprint prefix `a9952c55f5a1`.
- **Runtime:**
  - FP32, TF32 off.
  - torch 2.6.0+cu124 and transformers 5.14.1 (hermes venv).
  - Greedy decoding, `enable_thinking=False`, EOS ids [151645, 151643].
- **Generation cap:** **160 new tokens** for every family and arm. Titan's pilot L/R answers are ≤ 34 characters, so its outputs are unaffected.
- **Parsing:** `env.parse_state(..., lenient=True)` is applied identically to every arm. It handles:
  - compact JSON;
  - unwrapping `{"after": …}`;
  - `"K": a op b = c` → `c`;
  - the last complete `"after"` value in otherwise invalid text.
- **Throttling:** GPU throttling (DECISIONS I-10…I-15) changes wall time only.

## 4. Methods (all scored on the same confirmation parents and the same 256 test goals, seed 1001)

| Code | Definition |
|---|---|
| **G_chain(Titan)** | `generation.chain(mode="chain_one")`: Titan's verbatim one-step frame, applied event by event with its own previous answer as `before`. Final state → φ (`data.phi_from_state`) → U = w·φ. If a candidate is unparseable it gets its parent's mean predicted utility (primary). The last parsed chain state is used as a sensitivity analysis. |
| **G_e2e(Titan)** | A single greedy answer to the k-event prompt, parsed the same way. |
| **G(Supa)** | Supa e2e, chain_one and chain_m1. The **C3 comparator is, per family, the Supa mode with the lowest mean regret on the confirmation parents**. Choosing it this way favours Supa, so it is conservative for C3. |
| **F_touched** (answer-free floor) | φ(s0) with every "changed" feature set to 1 for each entity named by an applied event. For L only posted events count; for R every event counts (`fidelity.touched_delta`). |
| **F_copy_s0** | φ of the start state. |
| **B3** (equal label budget) | A consequence head on frozen Titan-Sparse answer-slot states of the k-event prompt, fitted on pilot TRAIN (~1.1k candidates per family). The layer (≠ 0) and the head (Ridge, clipped ±5, or MLP) are chosen **jointly per (family, k) by lowest pilot-DEV normalised regret under selection goals (seed 1002)**. MLP heads are averaged per parent over seeds 17–21. U = w·φ̂. |
| **S** (equal label budget) | The same, using the exact symbolic plan encoding (`decide.symbolic`) instead of LM states, with the same labels and selection. |
| **S@abundant** (secondary) | S, configured as selected above, but fitted on `bank_s4000` (k=4, L/R). |
| **B2** (secondary) | `heads.ScalarHead("bilinear")` on B3's chosen layer, trained on pilot-TRAIN utilities under 16 goals: "full16" (seed 2002) and "sub4" (seed 3003). |

The "changed" flags make up 4 of 12 φ dimensions for L and 3 of 9 for R.

## 5. Endpoint, primary contrasts and decision rule

- **Endpoint (per parent):** normalised regret = Σ_g (U* − U_sel) / Σ_g (U* − Ū) over the 256 test goals. Ties in predicted utility are broken at random with seed 17. A parent with a zero denominator is excluded.
- **Primary analysis set:** L and R at k = 4, 150 parents each.

**Primary contrasts:** a claim < 0 for each, all at k = 4 on L and R.

| | Contrast |
|---|---|
| **C0** | G_chain(Titan) − F_touched |
| **C1** | G_chain(Titan) − B3 |
| **C2** | G_chain(Titan) − S |
| **C3** | G_chain(Titan) − G(Supa)_best-mode |
| **C4** | G_chain(Titan) − G_e2e(Titan) |

**Test.**
- The pooled estimate is the mean of the two family means of paired per-parent differences.
- The one-sided p is (#{pooled* ≥ 0} + 1) / (10,000 + 1). Parents are resampled within family, seed 17.
- Holm step-down (monotone) is applied across C0–C4.

**Decision rules.**
- **PASS.** Cj passes iff its Holm-adjusted p < 0.05 **and** its pooled point estimate ≤ −0.10. The −0.10 is a threshold on the point estimate, not a tested margin. A "strong pass" additionally has a one-sided 95% bootstrap upper bound ≤ −0.10.
- **Holds in both families.** This is an intersection–union test with no further correction: each family's point estimate must be ≤ −0.10 **and** have an unadjusted one-sided p < 0.05. Otherwise Cj is reported as "pooled; driven by {family}".
- **Floor qualifier.** If B3 (or S) has a pooled mean regret ≥ F_touched's, a C1 (or C2) pass is reported as "beats a comparator at or below the answer-free floor". In that case it does not support superiority over latent or symbolic readouts in general.
- **Claim confirmed** iff C0–C4 all PASS. Otherwise each contrast is reported individually.

**Power.**
- Pilot pooled effects: C0 −0.19, C1 ≈ −0.38 to −0.44, C2 −0.35, C4 −0.39.
- The per-parent SD is about 0.25–0.35, so the pooled SE is about 0.015–0.02.
- C0 at L alone was −0.09 in the pilot, so the per-family rule may fail for L even if the pooled test passes.

## 6. Secondary analyses (descriptive; BH q = 0.10 across all of them)

For every family (L, P, R) and every k ∈ {1, 2, 4}:
- G_chain − B3
- G_chain − S
- G_chain − F_touched
- G_chain − G_e2e
- G_chain − G(Supa)_best
- G_chain − S@abundant (k = 4, L and R only)
- G_chain − F_first_only. This is a partial true simulator, not a floor.
- B3 − B2 on out-of-span goals (sub4). This holds largely by construction and is a design check only.
- B3 − B2 on in-span goals (full16)
- The fill-rule sensitivity, last parsed state in both arms: G_chain(Titan) − G_chain(Supa)

Descriptives:
- Exact-state accuracy and parse rates per (arm, family, k, mode).
- B3's best-on-test configuration, reported as an upper bound only.
- Twin consistency for e2e and chain: P(both of c0 and t0 exact at k=4), and P(predictions differ).
- B3/S selection traces.

Pre-declared directional prediction: for **P at k = 4**, G_chain − B3 ≥ 0 and chain exact-state accuracy < e2e. This follows from the precondition shortcut seen in the pilot. P's confirmation parents differ from the pilot's because they require ≥ 4 distinct final states.

Not implemented, and therefore exploratory and untested here:
- S-roll (a symbolic one-step learner, chained)
- B3@abundant
- latent-chained B3
- format-free likelihood ranking on the confirmation bank

## 7. Procedure and integrity

1. Freeze: fill in §9.
2. GPU work on the confirmation bank:
   - Titan-Sparse: `scripts/run_arm.py --bank data/bank_confirm.json --arm sparse --gen --gen-splits test --no-offtask`. This writes the capture, e2e and chains (twins included).
   - Supa: the same command plus `--gen-only`. Supa's hidden states enter no registered analysis, so capture is skipped. It still writes index.json, provenance.json and DONE.json, which the integrity checks require.
3. Run `scripts/confirm_analysis.py` once. It asserts:
   - the bank hash;
   - `DONE.json` for both arms, and that `index.json` refers to the confirmation bank;
   - both fingerprints;
   - complete e2e and chain coverage with 5 states per chain;
   - ≥ 10 non-NaN parents per family in every primary contrast.

   It writes `runs/bank_confirm/analysis/PREREG_1_RESULTS.json` once (create-only) and `PREREG_1_RESULTS_PER_PARENT.npz`.
4. Report in `RESULTS_CONFIRM.md` exactly as computed.

**Deviation policy.**
- A crash before the results file is written allows an identical rerun. A fix is allowed only for a non-statistical crash, and must be logged in DECISIONS.md with a diff.
- Partial `gen_*.jsonl` files left by a crashed GPU job are deleted and regenerated. Analysis requires `DONE.json`.
- The thread count is fixed at `RCWM_THREADS=8`.

## 8. Wording

**If every primary passes:**
> On 300 fresh ledger and register-machine parents, Titan-Sparse chose among 8 four-step plans by chaining its own one-step predictions and scoring the predicted final state with a linear goal readout. Its normalised regret over 256 unseen goals was X (0 = oracle, 1 = random). That is at least 0.10 lower (Holm one-sided p < .05) than each of the following:
> - an answer-free touched-entity rule;
> - the best of the ridge and MLP consequence heads on its frozen answer-slot states;
> - a symbolic-encoding learner given the same ~1.1k labels per family;
> - base Supa 1.7's best generative procedure;
> - its own single-shot four-step answer.

**Limits that always accompany it:**
- horizon 4 only;
- L and R only, with no advantage predicted for P;
- the advantage over learned readouts holds at equal label budget, and S@abundant (4,000 parents) reached Y;
- no claim is made about off-task forgetting.

**Partial outcomes:**

| Outcome | Report |
|---|---|
| C0 fails | "Not better than a trivial touched-entity rule; C1–C3 passes reflect weak comparators." |
| C4 fails, C1 passes | "Explicit generation beats the latent head, but stepwise decomposition is not the source." |
| C3 fails | "The advantage is not attributable to the 384-example delta." |
| C2 fails | "No decision advantage over a small symbolic learner given the same labels." |
| Significant, but point > −0.10 | "Directionally confirmed, below the pre-set practical threshold." |
| Pooled pass, per-family rule fails | "Pooled effect, driven by {family}; not shown for {other}." |

## 9. Freeze record

Frozen at 2026-10-06 16:36:30 -0500. The SHA-256 of every frozen input:

- `data/bank_confirm.json` `8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16`
- `data/bank_pilot.json` `7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a`
- `data/bank_s4000.json` `af1336c261375524a4a6eaf92cc48dc76b1159df093ce2705d279688f22bb4ce`
- `rcwmt/__init__.py` `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `rcwmt/bank.py` `b08b0229c566f9f5b50c7c0c5738f7fff90b768a827e7e32c9880ff69352592f`
- `rcwmt/capture.py` `7b0a19ee4aefda183f22336521fa39d5d3694bf1d60b5b950d769fa942ce255a`
- `rcwmt/controls.py` `08a9e4479fe621d5bb2d2a6e9d970eef122ae6850e7b894206b6a7c57002a702`
- `rcwmt/data.py` `f77270ee04d64985e5b6fa9a88f989a04b0b7ed8cc75f69a4984ca7e7593beb4`
- `rcwmt/decide.py` `148e98ab528dcd283f6896f07f01135ede658e124d09d08d293e7d3645463643`
- `rcwmt/decisions.py` `bd6a4531622d2479f1b7fe35fcb11338d831c1ab2e9402b515f9a35ef5237188`
- `rcwmt/env.py` `92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2`
- `rcwmt/fidelity.py` `da03bbca4962c05b13eb13dfa0414578344bbe2674015dc93070465ca4e1bb0e`
- `rcwmt/generation.py` `ef0250bb6aab764d647692fcdc367e4beb9017af13228670de29146ed93fd80a`
- `rcwmt/heads.py` `b3bbb95696fc40fa69cb1d3f60e2547f2fde8305862af2e5b934b073170d58b6`
- `rcwmt/lease_vendored.py` `d1576bb9296c6bd033dd677c0f8cbe18c9bc24939f75bba30d740429b2ec36c9`
- `rcwmt/offtask.py` `5fcc70500d21b9043ff93a054b53bdcf29fb2e052eeb722a99fb388fdd41ba0e`
- `rcwmt/prereg1.py` `1068d66dab6e1c87a255440eae376c1a884ac7dfc3cedac85478e30775cacd35`
- `rcwmt/pvr.py` `f7b1717fb276252900e75c3a6bf2f835d772498f9b75762aa82faea799e94b94`
- `rcwmt/runtime.py` `9360541286dc5648e6fd71c217135914b80fd4cef665364c47688cb07f0a12ca`
- `runs/bank_pilot/sparse/index.json` `7b869e08a100263e5716048838032b9ab3c350d90f0b4344f7ba9f281991ffa3`
- `runs/bank_pilot/sparse/last.npy` `3f9ba1d3d74f8685620c6d4b29feb28dddda19f6acaa6a6734e153cbd92bb56f`
- `scripts/analyze.py` `242c2f3974361ff3d454b8992de3632915010e1c0351a8794a6ec82778856a90`
- `scripts/confirm_analysis.py` `f006965b078798f7f457f589a04ba95474c640c6c4ee46ca5bd92a0f082fb9ec`
- `scripts/freeze_prereg1.py` `7e93274f039ea7611eae6773ee4ed3d2518cf2a32edb4843880d65d2023810f1`
- `scripts/run_arm.py` `7b8bf359bcdc962dc1339171105b758c44534d7664087968b2c0bad8d469e868`

The SHA-256 of this file as frozen is recorded in `PREREG_1_FREEZE.json`.
