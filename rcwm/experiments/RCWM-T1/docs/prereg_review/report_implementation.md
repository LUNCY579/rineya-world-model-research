# PREREG-1 implementation review: confirm_analysis.py and the code it calls

The primary path is correct, and I found no leakage, misalignment or ID bugs. The dry run finished and matches RESULTS_PILOT R7 to three decimals in every cell. Two problems must be fixed before freezing: the frozen script doesn't implement several of the registered secondary analyses, and it never checks that its inputs are complete or authentic. Missing generations are silently scored as unparseable.

## Verified correct (on CPU, no model run on the sealed bank)
- **Splits:**
  - `pilot_rows` contains only train and dev rows. No pilot TEST item ID appears in it.
  - The confirmation bank gives 10,800 rows, all `test`, so the assert holds.
  - Parent and candidate IDs don't overlap between the two banks.
  - The `bank_confirm.json` sha256 matches the value in the prereg.
- **Twins:**
  - Each of the 450 parents stores its twin under `p["twin"]`, so `candidate_table` produces no twin rows.
  - `bank.items` produces 450 twin capture items (`{pid}-t0-k4`). None of them collide with other IDs.
- **K and _cube:** K = 8 for every family and k, and every key of the form `{pid}-c{c}-k{k}` resolves. One parent has a zero denominator (P, k=1); none do at L/R k=4.
- **ArmUnion:**
  - The dry run uses a plain `Arm`, so it never exercises this path. I tested it with synthetic arms (`docs/prereg_review/impl_check_armunion.py`).
  - Mixed lists of pilot and confirmation IDs resolve to the right part, row and layer, and order is preserved.
- **strat_boot:**
  - Families get equal weight. With n = 150 vs 50, the point estimate is the mean of family means (−0.460), not the pooled mean (−0.370).
  - Parents are resampled within family.
  - p = P(bootstrap mean ≥ 0), which is the percentile one-sided p.
- **Alignment:**
  - `run()` and `g_only()` build `pids` the same way from the same rows, and `confirm_analysis.py:89` asserts they are equal.
  - All per-parent arrays share the same U_true and the same NaN mask, so the differences are properly paired.
- **load_gen:** `run_arm.py` writes `runs/<bank stem>/<arm>/gen_{e2e,chain_one,chain_m1}.jsonl` keyed by `item_id` / `cand_id`, which is exactly what `confirm_analysis` reads for `bank_confirm`. Chain states are padded to H+1 = 5.
- **Determinism and the dry run vs R7:**

  | Cell | G_chain(Titan) | B3 | S | G_chain(Supa) |
  |---|---|---|---|---|
  | L, k=4 | 0.507 | 0.813 | 0.746 | 0.835 |
  | R, k=4 | 0.388 | 0.965 | 0.841 | 0.994 |

  - B3 − B2 (sub4) at L k=4 is −0.132 [−0.192, −0.073], identical to R7.
  - Primary results: C1 −0.441 [−0.519, −0.364], C2 −0.346, C3 −0.467. All have p = 0 and PASS, with n = 26 per family.

## Findings

**1. MUST-FIX: registered secondary analyses are not implemented.**
- **Where:** `PREREG_1.md:66-73` vs `confirm_analysis.py:110-118`.
- **What's missing:**
  - **BH q = 0.10:** this needs p-values, but `decide.paired_bootstrap` (`decide.py:260-266`) returns only a mean and a CI.
  - **Titan exact-state accuracy** (e2e and chain, k = 1, 2, 4): not computed.
  - **Twin-pair consistency:** not computed. Also, `run_arm.py:300-301` builds chains only from `p["candidates"]`. Twins will get e2e outputs only (`-t0-k4`), so chain twin consistency can't be computed after the GPU run.
- **Failure:** the frozen script cannot produce analyses the prereg promises, so they become post-hoc.
- **Fix:**
  - Add a one-sided p to the secondary bootstrap.
  - Enumerate the secondary family explicitly and apply BH.
  - Compute exact-state accuracy and parse rate from the generation files.
  - Implement twin consistency on e2e, or add twins to the chain generation in `run_arm` before the GPU job.
  - Otherwise, relabel these analyses as exploratory in the prereg.

**2. MUST-FIX: no input integrity or coverage checks, and missing generations are silently imputed.**
- **Where:** `fidelity.py:262-267` returns None when an ID is missing. `decisions.py:73-76` and `:168-171` then give that candidate its parent's mean utility.
- **Failure:** a `gen_*.jsonl` file cut off cleanly at a line boundary, or a missing Supa mode, pushes G regret toward random without any trace. Given the reboot history (I-04, I-08, Kernel-Power 41), this is plausible.
  - If Supa's file is the incomplete one, C3 is inflated in Titan's favour.
  - Parse rates are never reported.
  - Nothing checks that the captures came from the sealed bank or the registered model.
- **Fix:** at the start of `main()`, assert:
  - `DONE.json` exists for both confirmation arms;
  - the bank sha256 is `8bc6…` and equals `bank_sha256` in each `index.json`;
  - the Sparse `provenance.json` fingerprint starts with `818be9c6b6a3`;
  - every confirmation `cand_id` is present in each chain file and every `item_id` in `gen["e2e"]`.

  Also write per-(arm, family, k, mode) parse rates into the results.

**3. SHOULD: Holm-adjusted p is not monotone.**
- **Where:** `confirm_analysis.py:43-50`.
- **Status:** the significance flags are correct, because `ok = ok and adj < alpha` stops at the first failure. The reported p values are wrong.
- **Shown:**
  - p = {0.01, 0.02, 0.021} gives p_holm = 0.03, 0.04, 0.021.
  - p = {0.02, 0.03, 0.04} gives C a reported p_holm of 0.04 (< 0.05) while `holm_significant` is False.
- **Fix:** `prev = max(prev, min(1, p*(m-i)))`, report `prev`, and set significant = `prev < alpha`.

**4. SHOULD: the decision rule's wording doesn't match the code.**
- **Mismatch:** the prereg rule is "Holm-adjusted one-sided 95% upper bound < 0". The code tests p_holm < 0.05 and reports only a two-sided 2.5–97.5% CI (line 39).
- **Effect:** the two agree to within one bootstrap resample at the boundary, but a reader won't find an upper bound in the JSON.
- **Fix:** reword to "Holm-adjusted one-sided percentile-bootstrap p < 0.05", or also output `quantile(bs, 1 − α/(m − rank))`. Report p = 0 as `< 1e-4` (or (c+1)/(N+1)).

**5. SHOULD: strat_boot returns p = 0 when a family is entirely NaN.**
- **Where:** `confirm_analysis.py:32-39`.
- **Verified:** the family mean becomes NaN, `bs` becomes all NaN, and `(bs >= 0).mean()` = 0, so `holm_significant` = True.
- **Effect:**
  - PASS is only avoided because `NaN <= -0.10` is False.
  - The empty contrast still takes the first (strictest) Holm slot with p = 0, which loosens the thresholds for the other contrasts.
- **Fix:** assert that each family has non-NaN parents and that `np.isfinite(bs).all()`.

**6. SHOULD: B2 prediction may run out of memory at confirmation scale.**
- **Where:** `decisions.py:107-112`, `heads.py:207-210`.
- **Cause:**
  - `Hte` repeats each test item once per goal: 1,200 items × 256 goals × 1,024 float32 = 1.26 GB.
  - Standardization then creates two temporaries.
  - The torch copy adds another, and the MLP variant adds a concatenation.
- **Measured:** I measured at 128- and 384-dimensional width (`docs/prereg_review/impl_check_b2_memory.py`) and extrapolated to 1,024: about +4.2 GB at peak (about 5.7 GB for the process). The dry run needed only about 0.2 GB per copy (26 parents), so it couldn't reveal this.
- **Risk:** your notes say 2–4 GB of RAM is typically free (7.8 GB is free now). The registered single run could hit a MemoryError after hours of work.
- **Fix:**
  - Predict in chunks of about 100 items.
  - For the bilinear head, compute `m(h) @ Gte.T` directly instead of repeating the inputs.
  - Add a prereg clause allowing an identical rerun after a crash that happens before the output is written.

**7. SHOULD: the results file can't be audited and can be overwritten.**
- **Where:** `confirm_analysis.py:119-123`.
- **Problems:**
  - Per-parent arrays are not saved, so any later audit (secondary, twin, BH) means retraining the heads.
  - The results file is silently overwritten, so "run once" is not enforced.
- **Fix:**
  - Save an `.npz` with the pids and every per-parent regret array for each family and k.
  - Open the results file in create-only mode (`"x"`).
  - Record the sha256 of the code files and the bank, the torch and numpy versions, `torch.get_num_threads()` / `RCWM_THREADS`, and a timestamp.

**8. NOTE: the C3 pilot baseline is mostly a parsing artefact.**
- **Cause:** the pilot Supa chains were generated under strict parsing (D-12).
- **Effect on parse rates:**
  - At k=4, the share of chains that parsed is R 0.00, L 0.60 and P 0.00.
  - Supa R k=4 regret is 0.994, the same as random.
  - With lenient parsing, the first step would parse for R 0.56 and L 0.90.
- **Consequence:** the Supa column in PREREG §1 (0.835 / 0.994) inherits this. The confirmation Supa chains will be parsed leniently, so expect a C3 effect well short of −0.467. Add a footnote to §1.

**9. NOTE: wording and determinism.**
- **Layer:** the layer is chosen per (family, k) using a change-mask ridge, not per family as §4 says.
- **Seed:** B3 and S use a single seed (17).
- **Tie-breaking:**
  - In `run()`, one random generator is consumed in a fixed order (oracle, random, F_*, G_e2e, G_chain_one, …); `g_only()` starts a fresh one.
  - Both are deterministic, and the dry run reproduced every pilot cell exactly.
- **Threads:** `RCWM_THREADS` changes the floating-point reduction order, so record it.

**10. NOTE: run_arm restarts and extra work.**
- **Restarts:** the generation files are opened create-only (`run_arm.py:292`, `:305`). A restart after a partial generation run raises FileExistsError, so the partial files must be deleted by hand; document this procedure.
- **Extra GPU time:** the off-task capture (`run_arm.py:312`) also runs on the confirmation bank.

The check scripts are in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/prereg_review/`:
- `impl_check_stats_and_bank.py`
- `impl_check_armunion.py`
- `impl_check_b2_memory.py`