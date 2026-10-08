# PREREG-1 merge: lead-methodologist ruling

## 0. Verification of the three reviews

I checked these claims myself, on CPU and pilot data only. No model touched `bank_confirm`.

**Confirmed**
- **Supa pilot chains are invalid as a C3 baseline.**
  - The supa `gen_*.jsonl` files were written 12:56–13:01. That was before I-09 (cap raised from 56 to 64) and before D-12 (lenient chain parsing at 14:30).
  - At R, 0 of 208 chains reach step 4. Only 116 of 208 first steps parse even leniently, and 50 are unclosed.
  - The truncated answer is exactly 56 tokens. The full compact wrapped answer is 62 tokens, so the current cap of 64 is marginal, and spaced JSON goes over it.
  - At L, 124 of 208 chains reach step 4.
  - Titan's longest L/R step answer is 33 characters.
- **Supa e2e parses fine but is near random.** It parses 95% (L) and 100% (R) at k=4, with regret 0.910 (L) and 0.936 (R).
- **F_touched beats both learned heads** (pilot TEST, k=4, n = 26 per family):

  | Method | L | R |
  |---|---|---|
  | F_touched | 0.599 | 0.683 |
  | G_chain(Titan) | 0.507 | 0.386 |
  | B3_mlp | 0.813 | 0.965 |
  | S | 0.746 | 0.841 |

  - G − F_touched at L is −0.092 [−0.213, +0.033]; at R it is −0.298.
  - Pooled with equal family weight: −0.193 [−0.263, −0.122].
- **S depends on label budget.** The reviewer's `a4.json` gives S at L: 0.746 → 0.684 → 0.607 → 0.203 with 133, 500, 1,000 and 4,000 parents. My independent re-run (new bank seed 424242, MLP seed 18) gives S@4000 L = 0.243, against G at 0.507.
- **Titan chain vs Titan e2e (C4 candidate):** L −0.340, R −0.436, pooled −0.388 [−0.457, −0.317].
- **The best Supa mode on the pilot** is chain_one at L and e2e at R. Pooled C3 against it is −0.438.
- **Holm adjustment is not monotone.** With p = {0.02, 0.03, 0.04}, C gets p_holm 0.04 but `holm_significant` is False.
- **`strat_boot` with an all-NaN family** returns p = 0.0 and point = NaN.
- **Twins:** the confirmation bank has 450 test parents, each with a `twin`. `run_arm` chains only `p["candidates"]`, so twins get e2e outputs only.
- **The bank sha256 matches** the value in the prereg.
- **Secondaries:** BH, exact-state accuracy and twin consistency are not implemented.
- **Dry run** matches R7: C1 −0.441, C2 −0.346, C3 −0.467.

**Corrections to the reviews**
- **Line references.**
  - `paired_bootstrap` is at `decide.py:87-93`, not 260-266.
  - In `run_arm.py`: e2e create-only open is at `:100`, the chain list at `:108-109`, the chain open at `:113`, and off-task capture at `:120-123`. The reviewer cited 292/300/305/312.
- **Missing files.** A missing whole `gen_*` file crashes with a KeyError at `confirm_analysis.py:101`. Only a partial file (cut at a line boundary) is silently imputed.
- **Layer selection criterion.** It is `Ridge.dev_score`, the mean per-feature binary accuracy on Δ (`heads.py:33-42`), not exact match.
- **B3 head choice.** The registered B3 is the weaker of two heads that are already computed. B3_ridge scores L 0.754 and R 0.895, against B3_mlp at 0.813 and 0.965 (`PREREG_1_DRYRUN.json`).
- **Partial simulators are not floors.** F_ignore_status (L k4 = 0.099) and F_first_only run the true transition function, so they must not be called floors. Only F_copy_s0 and F_touched are answer-free.
- **Supa fingerprint** is `a9952c55f5a1` (`runs/bank_pilot/supa/provenance.json`). §3 does not register it.

## 1. MUST-FIX before freeze

**M1. Generation cap (C3 measures format).**
- `rcwmt/generation.py:150-153`: replace the body with `return 160` for every family and arm. Update the comment to reference this ruling.
- `PREREG_1.md:36`: replace with "Generation cap: 160 new tokens for every family and arm. Titan's pilot L/R answers are ≤ 34 chars, so its greedy outputs are unaffected apart from float noise from batch composition."

**M2. Redo the pilot basis for C3 (GPU, pilot TEST only, before freeze).**
- `scripts/run_arm.py`:
  - At `:31-37`, add `--out-stem` and `--gen-only`.
  - At `:40`, use `out = ROOT/"runs"/(a.out_stem or bank_path.stem)/a.arm`.
  - Skip `:76-92` (capture) and `:120-123` (off-task) when `--gen-only` is set.
- Run `run_arm.py --bank data/bank_pilot.json --arm supa --gen --gen-splits test --gen-only --out-stem bank_pilot_regen`.
- Run the same for `sparse`. This doubles as a GPU-path test of the frozen instrument and should show Titan's texts essentially unchanged.
- Replace the Supa row in §1 (`PREREG_1.md:15`) with the regenerated values. Add a footnote: "earlier 0.835/0.994 used a 56-token cap and strict chain parsing (D-12); R chains 0/208 reached step 4; instrument artefact."
- If no GPU is available before freeze: delete the row and state "C3 has no valid pilot estimate."

**M3. C3 comparator = Supa's best mode.**
- `confirm_analysis.py`, after line 90:
  ```python
  best = min(("Gsupa_e2e", "Gsupa_chain_one", "Gsupa_chain_m1"), key=lambda m: np.nanmean(pp[m]))
  pp["Gsupa_best"] = pp[best]
  ```
  Store `best` in `results["per_family"][...]["supa_best_mode"]`.
- `PREREG_1.md:45`: "G(Supa): e2e, chain_one, chain_m1. The C3 comparator is, per family, the Supa mode with the lowest mean regret on the confirmation parents. This selection favours Supa, so it is conservative for C3."
- Add a line to §4: "Parse/imputation rate is reported per (arm, family, k, mode)."

**M4. Answer-free floor F_touched and contrast C0.**
- `rcwmt/decisions.py`, after line 66. Use its own RNG so the dry run still reproduces R7:
  ```python
  if fam in ("ledger_replay", "register_machine"):
      ch = slice(4, 8) if fam == "ledger_replay" else slice(6, 9)
      ft = []
      for r in te:
          ph = data.phi_from_state(fam, r["s0"], r["events"], r["s0"]).copy()
          ph[ch] = fidelity.touched_delta(fam, r["s0"], r["events"])[ch]
          ft.append(ph)
      t = decide.regret_table(U_true, _cube(pids, K, k, pos, np.stack(ft) @ Gte.T), np.random.default_rng(17))
      per_parent["F_touched"] = t
      summary["F_touched"] = {"nregret": decide.bootstrap_mean(t["nregret"]), "optimal": decide.bootstrap_mean(t["optimal"])}
  ```
- §4: add a row: "F_touched: φ(s0) with the 'changed' features set to 1 for every entity named by an applied event (L: posted events only; R: every event); `fidelity.touched_delta`."
- §4 line 50: add "The 'changed' flags are 4/12 (L) and 3/9 (R) of φ."
- §5: add C0 as a primary (see §3 below) and the floor qualifier from M7.

**M5. C2/B3 claims are tied to label budget.**
- Reword:
  - `PREREG_1.md:13`: "B3: consequence head on frozen Titan states, pilot TRAIN labels (~1.1k k=4 candidates/family)".
  - `PREREG_1.md:14`: "S: symbolic-encoding learner, same pilot TRAIN labels (equal label budget)".
  - `PREREG_1.md:47`: same wording as line 14.
- Register a secondary **S@4000**:
  - S fit on `bank.build(n_parents=4000, k_cand=8, horizon=4, seed=777001, exclusions=None, exclude_parent_keys=<confirm parent keys>)`, k=4, L/R.
  - Selection on pilot DEV, as for S.
  - Pre-stated reading: "If G_chain − S@4000 ≥ 0 in a family, C2 there is a statement about equal label budget only." The pilot predicts this at L.
- Drop "no-LLM learner" from any headline.

**M6. Select B3 and S on the endpoint, on DEV.**
- `confirm_analysis.py:81-84`: replace the ridge-Δ layer selection with a helper.
  - It fits `heads.Ridge` (predictions clipped to ±5) and `heads.MLPHead(seed=17)` on pilot TRAIN at every layer except 0.
  - It scores mean normalised regret on pilot DEV parents under 256 selection goals `decide.goals(dim, 256, 1002)`.
  - It picks the (layer, head) with the lowest score, then passes the layer to `decisions.run` and uses the key `B3_ridge` or `B3_mlp`.
- `decisions.py:101-104`: add `S_ridge` (`heads.Ridge` on `sym(tr)`, clipped to ±5). Choose S between `S_ridge` and `S_mlp` the same way.
- `PREREG_1.md:46-47`: "Layer (≠0) and head ∈ {Ridge, MLP} chosen jointly per (family, k) by lowest pilot-DEV normalised regret under selection goals (seed 1002); fit on pilot TRAIN."
- This also fixes the wording mismatch: the layer is currently chosen per (family, k), not per family.

**M7. Decision rule, claim mapping and per-family rule.**
- Replace `PREREG_1.md:56` and `:64` with the text in §3 below.
- `confirm_analysis.py:43-50`, monotone Holm:
  ```python
  prev = 0.0
  for i, k in enumerate(order):
      prev = max(prev, min(1.0, ps[k] * (m - i)))
      out[k] = {"p_holm": prev, "holm_significant": prev < alpha}
  ```
- `confirm_analysis.py:96-98`: primaries are C0–C4 (see §3).
- After `:106`, add:
  - a per-family `iut_both_families` flag: each family's point ≤ −0.10 and its own one-sided p < 0.05;
  - a `comparator_below_floor` flag for C1/C2: the comparator's pooled mean regret ≥ F_touched's.

**M8. Secondaries: implement now or label them exploratory.**
- `decide.py:93`: add `"p_one_sided": float((bs >= 0).mean())`.
- `confirm_analysis.py:110-118`: enumerate exactly the list in §3, compute one-sided p, and apply BH at q = 0.10 across that list.
- Add descriptive outputs computed from the generation files, using lenient parsing:
  - exact-state accuracy (e2e, chain_one) at k = 1, 2, 4;
  - parse/imputation rates;
  - absolute regret and the share of goals where the optimal plan is chosen.
- Twin consistency:
  - `run_arm.py:108-109`: iterate `p["candidates"] + ([p["twin"]] if "twin" in p else [])`.
  - Define the metrics as P(exact(c0,k4) ∧ exact(t0,k4)) and P(pred(c0) ≠ pred(t0)), for e2e and chain.
- Anything not implemented by freeze moves to "exploratory, untested" in `PREREG_1.md:66-73`.

**M9. Integrity guards.**
- `confirm_analysis.py`, after `:74`, when not a dry run, assert:
  - bank sha256 == `8bc6e2f8…a16`;
  - for both arms, `DONE.json` exists and `index.json["bank_sha256"]` equals that hash;
  - the sparse `provenance.json` fingerprint starts with `818be9c6b6a3`, and the supa one with `a9952c55f5a1`.
- Always assert, for both arms:
  - `{r["cand_id"] for r in conf_rows} ⊆ gen[arm]["chain_one"]` and `⊆ gen[arm]["chain_m1"]`;
  - every chain has `len(states) == 5`;
  - `{r["item_id"] for r in conf_rows} ⊆ gen[arm]["e2e"]`.
- `confirm_analysis.py:32-39`: assert every family has at least 10 non-NaN parents and `np.isfinite(bs).all()`.
- `PREREG_1.md:32`: add the Supa fingerprint prefix `a9952c55f5a1`.

**M10. Results must be auditable and written once.**
- `confirm_analysis.py:122`: open with `open(..., "x")`.
- Also write `PREREG_1_PER_PARENT.npz` with pids and every per-parent regret vector per (family, k).
- Record the sha256 of all code and data inputs, torch and numpy versions, `torch.get_num_threads()`, and a timestamp.

**M11. Freeze record and deviation policy.**
- `PREREG_1.md:79`: also hash:
  - `scripts/analyze.py`;
  - `data/bank_pilot.json` and `data/bank_confirm.json`;
  - `runs/bank_pilot/sparse/{index.json,last.npy}`.
- Fix `RCWM_THREADS=8`.
- `PREREG_1.md:83`, add:
  - "A crash before output is written allows an identical rerun."
  - "Fixes are allowed only for non-statistical crashes, logged in DECISIONS.md with a diff."
  - "Partial `gen_*.jsonl` files from a crashed GPU job are deleted and regenerated (`run_arm` writes create-only); analysis requires `DONE.json`."
- `PREREG_1.md:75`: update the power statement.
  - The C0 per-parent SD at L is about 0.32, and the pilot L effect is −0.09.
  - So the per-family rule for "holds in both families" may fail at L, while the pooled test has ample power (SE about 0.015).

## 2. SHOULD

1. **B2 memory.** At `decisions.py:106-112`, predict B2 in chunks of about 64 test items. At confirmation scale, `Hte` is about 1.26 GB plus copies, roughly 4–5 GB peak, against 2–4 GB that is typically free.
2. **Parser.** At `env.py:326-327`, under `lenient`, take the last complete `"after"` object and the right-hand side of `a op b = c` inside values. This applies identically to both arms, fixes about 20/208 Supa L first steps, and has no effect on Titan. Freeze it before the GPU run, because chains parse at generation time (`generation.py:188`).
3. **Seeds.** Average B3 and S over 5 seeds (17–21). Note that `confirm_analysis.py:87` currently keeps only the last tag per name; switch it to a per-parent average across tags.
4. **More secondaries:**
   - fill-rule sensitivity using the last parsed chain state;
   - C3 restricted to parents where both arms parse all 8 candidates;
   - G_chain(4 steps) − G_chain(Titan step-1 state) < 0;
   - G_chain − F_first_only, labelled a partial true simulator, not a floor;
   - S-roll: a one-step symbolic learner trained on pilot one-step transitions and chained 4 times.
5. **B3@abundant.** Capture ≥ 1,000 extra k=4 parents per family (capture only, no generation). Otherwise the claim keeps the "~1.1k labels" restriction.
6. **Latent-chained B3** (forward passes only): a one-step-frame head decodes the full next state, which is re-rendered into the next prompt, four times. This is the matched control for "serial vs single-pass".
7. **P prediction.** Make it directional: at P k=4, G_chain − B3 ≥ 0 and chain exact-state < e2e. Note that confirmation P parents require ≥ 4 distinct final states and pilot P parents do not.
8. **Reporting:**
   - add a "strong pass" when the Holm-level one-sided bootstrap bound is ≤ −0.10;
   - report p as (c+1)/(N+1);
   - always report G_chain's absolute regret and optimal-goal share.
9. **`run_arm`:** add `--no-offtask` for the confirmation run. In `confirm_analysis.py:69-73`, `arms["supa"]` is never used but forces a Supa capture on the confirmation bank; drop it or document it.
10. **Format-free companion for C3:** the D-09 likelihood ranking on the confirmation bank, if GPU time allows.
11. **Scope notes in §8:**
    - B3 − B2 is a design check only; it holds largely by construction.
    - No claim is made about off-task forgetting.

## 3. Final primary contrasts and wording

**Analysis set.** Families L and R at k = 4, 150 parents each, equal family weight, 256 goals (seed 1001). Each contrast uses paired per-parent differences and 10,000 within-family bootstrap resamples (seed 17).

| | Contrast (< 0 claimed) | Pilot pooled (L / R) |
|---|---|---|
| C0 | G_chain(Titan) − F_touched | −0.193 (−0.092 / −0.295) |
| C1 | G_chain(Titan) − B3* (DEV-selected layer and head) | B3_mlp: −0.441. B3_ridge (L 0.754 / R 0.895) would give about −0.377. |
| C2 | G_chain(Titan) − S* (equal label budget) | −0.346 |
| C3 | G_chain(Titan) − G(Supa)_best-mode | −0.438 (Supa chains pending regeneration per M2) |
| C4 | G_chain(Titan) − G_e2e(Titan) | −0.388 (−0.340 / −0.436) |

**Decision rule (replaces `PREREG_1.md:56` and `:64`)**
> **Test.** For each Cj, the pooled estimate is the mean of the two family means. The one-sided p is the share of 10,000 resampled pooled means ≥ 0, with parents resampled within family. Holm step-down (monotonised) is applied across C0–C4.
>
> **PASS.** Cj PASSES iff its Holm-adjusted p < 0.05 AND its pooled point estimate ≤ −0.10. The −0.10 is a threshold on the point estimate, not a tested margin.
>
> **Holds in both families.** Cj holds in both families iff each family's point estimate ≤ −0.10 and its unadjusted one-sided p < 0.05 (intersection–union test, no further correction). Otherwise Cj is reported as "pooled; driven by {family}".
>
> **Floor qualifier.** If B3* (or S*) has pooled mean regret ≥ F_touched, a C1 (or C2) PASS is reported as "beats a comparator at or below the answer-free floor". It does not then support superiority over latent or symbolic readouts in general.
>
> **Claim.** The §1 claim is confirmed iff C0–C4 all PASS. Otherwise each Cj is reported individually.

**§8 wording if all primaries pass**
> On 300 fresh ledger and register-machine parents, choosing among 8 four-step plans by chaining Titan-Sparse's own one-step predictions (each answer fed back as the next state) and scoring the predicted final state with a linear goal readout gave normalised regret X over 256 unseen goals (0 = oracle, 1 = random). Its regret was at least 0.10 lower (Holm one-sided p < .05) than each of:
> - an answer-free touched-entity rule;
> - the best of ridge and MLP consequence heads on Titan-Sparse's frozen answer-slot states (C1);
> - a symbolic-encoding learner trained on the same ~1.1k pilot labels per family (C2);
> - the best generative procedure of base Supa 1.7 (C3);
> - Titan-Sparse's own single-shot four-step answer (C4).
>
> **Limits:**
> - Horizon 4 only; families L and R only.
> - The advantage over learned heads holds at equal label budget. With 4,000 simulator-labelled parents, S reached Y.
> - On P, no advantage was predicted.
> - Nothing is claimed about off-task forgetting.

**Partial outcomes**
- **C0 fails:** "Not better than a trivial touched-entity rule; C1–C3 passes reflect weak comparators."
- **C4 fails, C1 passes:** "Explicit generation beats the latent head, but stepwise decomposition is not the source."
- **C3 fails:** "The advantage is not attributable to the 384-example delta."
- **C2 fails:** "No decision advantage over a small symbolic learner given the same labels."
- **Significant but point estimate > −0.10:** "Directionally confirmed, below the pre-set practical threshold."
- **Pooled pass but per-family rule fails:** "Pooled effect, driven by {family}; not shown for {other}."

**Files**
- Verification scripts (scratch):
  - `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/v1.py` through `v8.py`
- Files to edit:
  - `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/PREREG_1.md`
  - `.../RCWM-T1/scripts/confirm_analysis.py`
  - `.../RCWM-T1/scripts/run_arm.py`
  - `.../RCWM-T1/rcwmt/{generation,decisions,decide,env}.py`