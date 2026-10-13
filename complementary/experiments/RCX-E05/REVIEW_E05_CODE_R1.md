# Adversarial review of the E05 pilot code, round 1 (workflow wf_4fbe8bbb-382, 2026-10-12 ~22:10)

**Verdict: NOT READY FOR T2.** Three items block the pilot freeze (#1–#3), and #4–#6 should be fixed at the same time. All three are in T2-digested files, so changing them later means re-recording the digest. #7 does not block T2 but blocks T3.

**Tests.** I ran each E05 test file once; the lane was free before every run and all pass:

| File | Result | Time |
|---|---|---|
| `test_e05_oracle` | 15/15 | 1 s |
| `test_e05_families` | 15/15 | 1 s |
| `test_e05_enum` | 21/21 | 4 s |
| `test_e05_learners` | 21/21 | 8 s |
| `test_e05_lane` | 36/36 | 1 s |
| `test_e05_run` | 14/14 | 17 s |
| `test_e05_analyze` | 53/53 | 5 s |

Working-tree digests now: T2 `7d99b38fabdcde6514228b8c3cc4e332b98d681d7d9e45f7dd56c7b7d723b4be`, T3 `94bff6d2334e0a1d3a8c5da7901b3a4e367528207398246fc522303260e34638`. Pins hold, no files are missing, and no bytecode or `runs/` output was written.

**Rule slip:** for the `code-digest` check I piped `python` into `python -c`, so two Python processes ran at once. It was read-only and wrote nothing.

## Findings (most severe first)

**1. BLOCKER, needs a decision.**
- **Where:** `C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E05/enum_e05.py:270-281`; claim at `PREREG_E05_DRAFT.md:331-335`.
- **Problem:** ĥ_MIS overestimates h* by up to 1 on targets that contain `clear(y)`. The block resting on y can be unstacked last and kept in the hand, but `misplaced_compact` still charges it a placement.
  - I confirmed the counterexample by hand: n = 5, tower 4-0-2-1 with 3 on the table, G = {clear(4), on(3,2)} gives ĥ = 8 while h* = 7.
  - Such targets come from every on-menu (the `clear(b)` falsifier) and from H-SPAN, so the prereg's admissibility claim is false.
  - `tests/test_e05_enum.py:186-224` checks admissibility only on clear-free targets, which is narrower than the §2.1 test-table requirement.
- **Fix:** before T2, either (a) narrow the §5.2 claim and the test-table wording and keep the formula, or (b) switch `POLICY_KINDS["MIS"]` and `ARMS["ENUM-MIS"]` to `mis_adm` and amend the §5.2 formula. Record the choice.

**2. BLOCKER.**
- **Where:** `experiments/RCX-E05/run_e05.py:1561-1580` (`unit_bench`), `:1676-1682` (`project_confirmation.cell` defaults to `{"mean": 1.0}`), `lane_e05.py:541-552`.
- **Problem:** ENUM-H*± and GNN-ENUM± never run on DEV, so G-P5 has no measured cost for them.
  - The projection and the M2 unit check assume 1 s per cell. Cut items 4 and 6 therefore cannot be triggered by measurement, although `DESIGN.md:1030` says "G-P5 measures it; cut items 4 and 6 absorb it".
  - GNN-ENUM± does no pruning: one GNN prediction over all targets (up to 900 on H-NEG) per successor state. Its cells are likely tens of seconds each and cannot be interrupted, which breaks the 30-s unit rule and the 150-min budget at TEST.
  - A smaller mismatch: `lane_e05.py:533` names the arms "FT-UVFA" and "SF-FT", but run cells use "FT", so the measured FT maximum is ignored.
- **Fix:** have `unit_bench` time ENUM-H*± and GNN-ENUM± cells on a few DEV-ID, DEV-OOD H-K and DEV-OOD H-NEG parents, kept as bench data rather than DEV results, and store them as `arm@family`. Reject or split any unit over 30 s, and align the arm names.

**3. BLOCKER, needs a decision.**
- **Where:** `run_e05.py:2000-2017` versus §4.4 item 4 (`PREREG_E05_DRAFT.md:286`).
- **Problem:** E11b has 32 n = 7 instances in `test_id` and 36 in `ood_small`. Both are E11 evaluation splits (E11 `DESIGN.md:364-365, 391`). The Q8 n = 7 check passes only because it counts the E11b `exec` split alone, so the prereg claim "no RCX evaluation set contains an n = 7 instance" is false.
- **Fix:** before T2, either amend §4.4 item 4 (and DESIGN §4.5) with a stated rationale, or exclude those 68 instances from E05 rows, starts and banks. The second option changes what `pilot-a` draws, which is why it has to be settled before T2.

**4. MEDIUM, fix before T2.**
- **Where:** `run_e05.py:1433`; grids at `:152-161`; docstring claim at `:67-70`.
- **Problem:** the pilot per-goal seed includes the grid name, `f"{kind}:{grid}:{k}:{key}"`. So `w64s500` is an independent fit, not the 500-step checkpoint of `w64s2000`, and FT `s100` is not a checkpoint of `s400`. This contradicts `DESIGN.md:516-517` and §5.3, and the docstring's equivalence claim is wrong.
- **Fix:** drop the step count from the seed tag (for example `PGV:w<width>:lr<lr>:<k>:<key>`). At equal seed and B, learners already make the shorter run equal to the checkpoint.

**5. MEDIUM.**
- **Where:** `run_e05.py:848` together with `lane_e05.py:377-384`.
- **Problem:** the L passed to the start check is the slice budget, min(15, deadline − now), not the projected work. The §10.2 condition "start + L ≤ next window − 10 min" therefore can never fail while any room is left. Slices run up to the deadline and then overshoot by up to one check gap (≤ 90 s), so E05 will often still be running inside the last 10 minutes before a window.
- **Fix:** start with L = min(projected need, budget), and end the slice once now ≥ t0 + L − (60 s + largest unit).

**6. MEDIUM.**
- **Where:** `run_e05.py:1044-1047`, `:2526-2579`.
- **Problem:** `Pilot` refuses a directory holding outputs from another digest and tells the operator to use "a fresh --pilot-dir", but no `--pilot-dir` option exists. The TEST-side commands also hard-code `pilot` / `pilot-n6`. The G-P1 audit path (fix the bug, re-record T2, re-run the phase) has no supported route.
- **Fix:** add `--pilot-dir` to the pilot commands and carry it into the TEST commands.

**7. HIGH, blocks T3 (not T2).**
- **Where:** `analyze_e05.py:1133-1135`, against `run_e05.py:2028` and `:2455-2457`.
- **Problem:** `run_e05` writes both `<attempt>/cells.jsonl` and `<attempt>/evals/*.json`, so `analyze_e05.main` always refuses with "the eval source is ambiguous".
  - The fallback reader `native_arm` (`analyze_e05.py:972-988`) is also wrong for real output. Run writes tags like `PGV_400:<cfg>|k400`, `kcurve:k1600|k1600` and `FT:<cfg>:s<seed>|k400`; the reader expects `<cfg>|k400`, `k1600|k1600` and `<cfg>|s<seed>|k400`. PGV_400 would be reported missing and FT records refused.
  - The tests miss this: `tests/test_e05_run.py:621-626` calls `AN.load_evals` directly, and the native tests in `test_e05_analyze` use synthetic tags.
- **Fix:** make `evals/*.json` the only input contract, fix or remove `native_arm`, and add an end-to-end test that calls `AN.main` on the pipeline's actual output.

**8. MEDIUM-LOW.**
- **Where:** `lane_e05.py:527-552`, used by `run_e05.py:1566-1568` and `:1835-1838`.
- **Problem:** the undigested operational file decides which configurations G-P5 benchmarks and whether its unit check passes. That feeds the cuts, caps and sizing in the PILOT FILL, which contradicts M1 (the lane file decides only *when* a phase runs, never *what* it computes).
- **Fix:** move `STEP_UNITS`, `MODEL_PARAMS`, `plan_costs` and `validate_units` into `run_e05`.

**9. LOW-MEDIUM.**
- **Where:** `run_e05.py:1349`, `:1558`, `:1996`, `:2133`.
- **Problem:** several one-off units have no internal safe point: the pilot draw (est. 180 s), the bench (120 s), `draw-test` (300 s) and the oracle-test report (120 s). At n = 7 these can exceed the ≤ 2-min overlap bound in §10.2, and their real durations are unmeasured.
- **Fix:** measure the draw time at the pilot, and add lane checks between families.

**10. LOW.**
- **Where:** `enum_e05.py:438`.
- **Problem:** `load_gnn` imports torch before `rcx.bw.models`. Eval runs GNN-ENUM± before learners is imported, so on that path nothing in the code sets `CUDA_VISIBLE_DEVICES=""` before torch loads. `models.py` still patches `torch.cuda`.
- **Fix:** set the variable at the top of `run_e05.py`.

**11. LOW, a reading to record.**
- **Where:** `run_e05.py:2288` (`NO_BANK`).
- **Problem:** ENUM-H*± and GNN-ENUM± (and RAND, BEST-SINGLE) get no one-step bank. §3 says the bank is "identical for every arm" and §10.3 protects "the one-step banks".
- **Fix:** record this as a reading, or bank ENUM-H*± and GNN-ENUM± on H-ID and H-K.

**12. LOW, ambiguities to list in the T2 entry.** Each is a defensible reading.
- **RAND stream:** one per cell, `20261244:rand:<split>:<key>:<s0>` (`run_e05.py:688`), where App. A says `20261244:rand`.
- **Revisit rule:** the move into an already-visited state is executed and counted (`soft.py:347-351`).
- **H-CONJ STOP:** non-ORACLE arms score STOP on the linearization.
- **H-DYN banks:** bank states are not restricted to the height limit.

**13. LOW.**
- **Where:** `families_e05.py:180-190`.
- **Problem:** the H-CONJ key carries rtype `conj`, so it can never collide with a TRAIN key. If its linearization equals a TRAIN goal exactly (e.g. |G| = 3 gives w = 1_G at c = 0.25), the learned arms receive a training code. This follows §4.1 literally.
- **Fix (optional):** also check the linearized key against TRAIN.

## Verified correct
- **Oracle:**
  - Value iteration, the STOP-first then lowest-action-code tie rule, pointer doubling, nreg and clipping, renaming and the height mask are all correct.
- **ENUM:**
  - Menus are complete (14 facts at n = 7), and the B1 h_hold term is right.
  - The persistence rule is right, and storing one entry per G gives the same maximum as the literal enumeration.
  - The branch-and-bound is exact.
- **Learners:**
  - Parameter counts match §5.1 exactly, and joint renaming and stabilizer augmentation are exact.
  - FT step 0 is a candidate, with ties going to the earlier step, and validation uses the first 40 rows.
- **Lane:**
  - It uses `try_acquire` only and never writes a waiting marker, GPU file or release file.
  - Day and night-tail modes match §10.2, and GPU liveness covers pid and worker_pids.
  - The E01R refusal regex matches `rcx.lane`'s actual message, and the safe-point yields are correct.
- **Custody:**
  - The digest formula and scope are right, and DECISIONS is read at HEAD only.
  - VOID is written only in `oracle-test`, and the first-learned gate works.
- **Analysis:**
  - The §7.1 determinations and the §7.2 order are right; the IUT uses p = max, and the Holm family is fixed at the freeze.
  - The NI p-value is right, and the bootstrap draw order equals `rcx.stats`.
- **Leakage:**
  - The M6 key is scale-canonical, signed supports work, and the H-ID-U blocking is correct.
  - The exclusion adaptor is exact; `towers_of` returns sorted tower heights, so there are 30 classes or fewer.
  - n = 7 has 9 / 37 / 182 support classes at K = 2 / 3 / 4, so the unseen-class cap does not bind.
- **Earlier issues now fixed:**
  - The GNN-ENUM± `KeyError` is fixed.
  - The rev 1 and rev 2 ENUM controls now run on H-NEG.
  - SF-FT_400 is produced in extras.
