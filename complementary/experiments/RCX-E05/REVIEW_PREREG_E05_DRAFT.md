# Adversarial review of PREREG_E05_DRAFT.md (workflow wf_317f243c-6eb, 2026-10-12)

**Verdict: NOT READY FOR CODE.** Items B1–B4 are HIGH and block code. Items M1–M4 shape what goes into `run_e05.py`, `enum_e05.py` and `analyze_e05.py`, so they must also be settled in the text before the code is written. Everything else can be fixed before T1 or T3.

All of F1–F12 are addressed in the text. F5 and F4 are not actually resolved in substance (B1, B2). The power tables, the corrected n_ID lookup, the 1.15 SD factor, rev 1's 0.060/0.063/0.066 figures and the DESIGN sha `8e5e91bcff89…` all check out. The decision rule's order is consistent: STOP-D and ADVANCE cannot both hold. Tiering is correct: mechanism tier B, with no Rineya-ability claim.

**HIGH (blocking)**

**B1. ENUM-MIS± is degenerate on H-NEG, so F5 is not really fixed** (DESIGN §5.3, prereg §5.2)
- Problem:
  - `misplaced_compact` returns a bound of 0 for any goal that contains a `holding` fact (it uses `g.has_holding` for the whole goal; `rcx/bw/space.py`).
  - The falsifier menus offer holding(a) for every negative on(a,b) and ontable(a). So whenever a is not constrained by P, the target P ∪ {holding(a)} is satisfiable.
  - That target scores U = Σw(P), with no penalty charged, minus c′·(1 + 0). The score is the same for every successor s′, and it beats every other target.
  - Result: all moves tie, the tie rule takes the lowest action code (pick-up then put-down), the state is revisited and a STOP is forced. nreg is about 1, clipped.
  - On H-NEG the guard therefore behaves like STOP. G1 becomes almost automatic and STOP-D becomes almost unreachable. DESIGN prediction 4 is likely false.
- Fix:
  - For ENUM-MIS±, compute ĥ as `misplaced_compact(s′, G \ {holding facts})`. This is admissible, because every state that satisfies G also satisfies G minus its holding facts.
  - Add a test in `test_e05_enum.py` that uses ENUM-MIS± itself: a hand case with a penalty that a holding target can falsify. Also assert that ENUM-MIS± scores are not all tied across successors on random H-NEG weights.
  - Also check ENUM-MIS± against GREEDY on DEV-OOD H-NEG at the pilot.

**B2. GREEDY is exactly STOP on all positive on/ontable families, so a VOID trap exists and the F4 competence gate is hollow** (DESIGN §3 "Starts", §5.2, App. B Q7/G-P3; prereg §3, §5.3, §7.1)
- Problem:
  - Rev 2 pins hand-empty starts. From an empty hand every move (pick-up, unstack) only removes on/ontable facts; it can add only `clear` facts, which carry zero weight here. So GREEDY STOPs at s0, and its nreg is exactly 1 on DEV-ID, H-ID, H-K, H-COST, H-DYN and H-CONJ.
  - Consequences:
    - Q7 ("RAND and GREEDY strictly between, on DEV and TEST") fails by construction on DEV-ID, and on any per-family check. That makes G-P1 or VOID fire for reasons unrelated to the hypothesis (the same kind of defect as E04's F3). Only H-NEG keeps the TEST primary macro below 1.
    - "Competent" (nreg ≤ GREEDY − 0.20) just means nreg ≤ 0.80 on DEV-ID, which is nearly floor level. So the P2 comparator can still be a strawman.
- Fix:
  - Q7/VOID: require ORACLE = 0 and STOP = 1 exactly, and RAND < 1 on the primary macro only. Drop GREEDY from the gate and disclose that GREEDY = STOP on positive families.
  - Re-anchor competence to a non-degenerate bar fixed now: for example DEV-ID nreg ≤ 0.50, or ≤ BEST-SINGLE(L) or ENUM-MIS± on DEV-ID.
  - Include step 0 (the zero-shot UVFA) in FT-UVFA_400's early-stopping candidates.
  - Correct the rationale "expected to stop early on most starts".

**B3. The decision rule is not immune to degenerate comparators** (prereg §5.3, §6.2, §7; G-P4b)
- Problem:
  - G-P4b says "else one UVFA extension" and defines nothing if the extension also fails. A near-constant UVFA would let P1 pass against a strawman, giving REVISE, or ADVANCE together with P2 and G1.
  - Competence of PG400 and UVFA is checked only on DEV-ID. Nothing checks it on TEST: a comparator that collapses on TEST lets P2 or P1 pass.
  - ADVANCE has no absolute floor on SF-GPI. With competence at ≤ 0.80 and δ_NI = 0.05, SF-GPI could reach ADVANCE at nreg of about 0.85.
- Fix:
  - G-P4b terminal branch: if UVFA still fails after the extension, P1 is declared comparator-invalid before the freeze. ADVANCE becomes unavailable and P1 is descriptive.
  - Add TEST validity conditions that are evaluated before the rule:
    - P2 counts only if PG400's TEST macro passes the competence bar;
    - P1 counts only if UVFA's H-ID nreg passes it.
  - Add "SF-GPI primary macro nreg ≤ fixed bar, e.g. 0.50, in every primary family" to ADVANCE.
  - Add `test_e05_analyze` cases showing that a copy-state SF (= GREEDY) and a constant UVFA cannot reach ADVANCE.

**B4. After T4, voiding an attempt can depend on outcomes** (prereg §7.1, §10.4, §11)
- Problem:
  - §11 says "after T4, a change that could affect TEST outcomes voids the attempt", and VOID leads to "fix and redraw" (up to two times).
  - Nothing says which attempt decides, or whether a VOID may be declared after learned arms have been seen on TEST. That is an outcome-driven do-over.
- Fix:
  - Move every VOID check (Q3/Q4/Q7/Q8, disjointness, exclusion, manifest/digest) into `oracle-test`, before any learned-arm TEST evaluation. VOID may only be declared there.
  - Any defect found after the first learned TEST evaluation gives INCONCLUSIVE plus a design review, with no redraw and full reporting of that attempt.

**MEDIUM (settle before code)**

**M1. Lane rules are stale relative to CX D-44 (16:24)** (prereg §2.1, §10.2; DESIGN §9.3)
- Problem:
  - The pinned `rcx/lane.py` sha `03f93d59…` is wrong; the file on disk is `4a1bc447…`, changed by D-44.
  - The nights are hard-coded as 10-12/13/14. BWM accepted (16:40) that two more nights (10-15/16 and 10-16/17) may be added for the E01R SEALED evaluations.
  - Safe points can be 6–11 min apart (see M2), so an E05 phase could delay an E01R window block.
- Fix:
  - Re-pin `lane.py`.
  - At every phase start, read `runs/RCX-E01R/quiet_windows.json` plus any later D-entry windows, block from 10 min before each window opens, and refuse if the file cannot be read.
  - Re-state at T1 that the RAM < 11 GB rule may starve E05 in the daytime, now that E01R acquires the lane only inside windows.

**M2. "Safe points ≤ about 2 min apart" does not match the batch sizes** (prereg §10.2, DESIGN §9.3)
- Problem: a batch of 64 PGV_400 width-256 fits is about 7.4 TFLOP, roughly 6.5 min at 19 GFLOP/s. A 16-parent PGV_6400 batch is roughly 11 min. Yield latency to a dsv3wm or E01R waiting marker breaks the promise.
- Fix: size batches from the G-P5 rate so each takes ≤ 2 min, or checkpoint `BatchedMLP` every N steps. Add a test in `test_e05_lane.py`.

**M3. The bootstrap equivalence test cannot pass as written** (prereg §2.1 and §6.4)
- Problem: `rcx.stats` resamples with `random.Random(seed).randrange`. The prereg specifies a numpy PCG64 index stream. "Equals `seed_parent_bootstrap_test` on one stratum" cannot hold exactly.
- Fix: either use `random.Random(derive(20261250, "boot:<stratum>"))` with the same draw order, or define the test as equality given an injected index matrix.

**M4. G-P3's "otherwise PGV training is fixed before the freeze" is open-ended** (prereg §9.1, DESIGN App. B)
- Problem: it allows an unconstrained change to the comparator after DEV D2 is visible.
- Fix: pre-specify a single escalation (for example the G-P3 extension config, applied identically for all k). If it still fails, k\* is descriptive only.

**M5. P2's planning SD comes from DEV-ID only** (prereg §8 steps 1 and 4)
- Problem: D2 is a macro over H-ID, H-K and H-NEG, but PG400 is never run on DEV-OOD. P2 power on the OOD families may be overstated.
- Fix: run the selected PG400 config on the 24 DEV-OOD H-K/H-NEG parents in pilot B, and plan with max(σ̂₂,ID, σ̂₂,OOD).

**M6. Weighted-class disjointness ignores scale equivalence** (prereg §4.4.1)
- Problem: (w, c) and (λw, λc) are the same task with identical nreg. For example, H-COST all-1 weights at c = 0.125 equal all-2 weights at c = 0.25, and all-3 weights at c = 0.75 equal all-1 at c = 0.25. These can duplicate TRAIN or DEV tasks.
- Fix: canonicalize the parent key on the primitive integer vector of w̄ in units of 1/8, and assert disjointness on that.

**LOW**

- **L1 (§5.2 ENUM±).** M(on(a,b)) is not a complete falsifier set: it misses on(a,x), on(x,b) and holding(b). Disclose this, or extend the menu.
- **L2 (§8).** The planned effect of 0.05 is above the predicted H-ID gap of 0.00–0.04. P1-ID power at 0.03 is only 0.22–0.35. State this.
- **L3 (§6.4).** The parent bootstrap ignores seed-level variance. State that inference is conditional on the 3 replicates. Consider applying condition 4 to H-ID as well.
- **L4 (§7).** Two rule gaps:
  - STOP-F's "P2 did not pass" should say whether "dropped" counts.
  - STOP-M overrides a P2-only pass, which contradicts the stated rationale for placing STOP-M after ADVANCE. Justify this or narrow STOP-M.
- **L5 (§2.2).** `eval` should refuse unless the TEST manifest sha is in the committed `DECISIONS.md` (`git show HEAD:`).
- **L6 (§4.4.3).** Run the exclusion check on per-goal label rows and one-step bank states too. Also say what happens if `load_exclusions` fails closed after T3.
- **L7 (§9.1, §1).** Two fallback gaps:
  - If the pilot shows a large DEV contrast and E05 is then abandoned, `PILOT_REPORT` should still be published.
  - The G-P4 fallback to n = 6 should carry over into the H-E05 wording.
- **L8 (consistency).**
  - RSS is "< 1 GB" in DESIGN §5 but "< 1.5 GB" elsewhere.
  - DESIGN §6.6 step 5 says "Power above the cap"; it should say "below 0.80".
  - DESIGN §6.4 uses "stream per family"; the prereg uses "per stratum".
  - The F11 wording differs between the table and the qualifier.
  - The copy-floor acc½ is "share of STOP-optimal pairs", but the bank excludes those, so it is 0.
  - "Stab = 1 for asymmetric goals" is wrong: the blocks a goal does not mention give |Stab| ≥ 2 for most goals.
  - Protected "ENUM-MIS± on every family" conflicts with cut items 3 and 7.
  - DESIGN §9.3's E01R interaction paragraph is obsolete after D-44.
- **L9 (lane/thermal).** Two lane details:
  - Confirm with BWM at T1 that 4-thread E05 slices beside GPU bundles are acceptable under D-84; tz01 idles at 94–96 °C.
  - Add a lower free-RAM floor.
- **L10 (open items, as the reviser noted).**
  - The T1 CX D-entry is still to be recorded.
  - `experiments/RCX-E04/DESIGN.md` lines 231–235 still list 20261242–48.
  - The B4 fresh-set re-pin is pending (`EXTRA_RESERVED_FILES` already includes `instances_seeds_fresh.json`).

**Blocking list:** B1, B2, B3, B4, plus M1, M2, M3 and M4 before any code. M5 and M6 must be in the text before T1. The L items can be fixed before T3.

Files reviewed:
- `C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E05/PREREG_E05_DRAFT.md` (sha256 `ddaa23b8…`)
- `C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E05/DESIGN.md` (`8e5e91bcff89…`)
- `C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E05/CRITIQUE_RVM_DESIGNS.md`
- Evidence for B1–B2: `rcx/bw/space.py`. For M1: `rcx/lane.py` and `runs/RCX-E01R/quiet_windows.json`. Also checked: `rcx/stats.py`, `rcx/bw/excl.py`, `DECISIONS.md` D-43/D-44 and `COLLABORATION.md`.
