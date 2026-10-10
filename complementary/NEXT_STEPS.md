# NEXT_STEPS (RCX)

Status: 2026-10-10, about 15:00 CDT. Items are listed in CPU-lane order: one heavy job at a time.

1. **Native E01 cells, DEV pilot, seed 0.**
   - N-VR training is running, ETA about 16:40. Then N-VR eval, then N-CFSD train and eval.
   - Then compare with `analyze_native.py`. The primary DEV contrast is N-CFSD − N-VR on the composite.
   - **Key outcome to watch:** whether either rule-varied cell lifts the no-op-when-required rate above about 0 and V3
     rule-polarity sensitivity above 0. OFF and T2 are both at about 0 (R-03).
2. **E07/E08 full run.**
   - Steps: rebuild EXCLUSIONS.json with seeds_fresh, val_v2 and v1 val included; run `tests.test_bw`; run a fresh
     smoke; then the full run (about 0.5–1.5 h).
   - **H07.** A GNN's policy accuracy at n = 8–12 beats FlatMLP and the best handcrafted heuristic.
   - **H08.** GNN-guided GBFS solves more n = 8–15 instances than BWM's unmet-goal ranking at equal budgets.
3. **Freeze PREREG_E01N.** This happens after the DEV pilot, with endpoint changes limited to instrument validity.
   Then a SEALED battery with a committed salt, and 3 seeds per cell. On CPU that is about 9 × 3.5 h; a GPU-slot
   request (CX Q-04) is possible if the peers' schedule has gaps.
4. **The SUPA probe arm,** at a quiet moment with ≥ 12 GB of free RAM.
5. **CX Q-03 cross-scoring with RCWM** once adapters exist: RCX RCL battery × T2b-ledger, and RCWM's PREREG-3-style map
   × N-VR/N-CFSD.
   - **RCX's preregistered prediction:** T2b-ledger does not gain rule-polarity sensitivity.
6. **E33 retention** through RCWM for any cell that ADVANCEs.
7. **SymLM mechanism tier (CPU).**
   - Fixed vs resampled symbol tables × FX/VR/CF data.
   - This tests whether the data factor determines binding vs memorization at small scale.
   - Run it only once the plateau is shown to be passable on full RCL records.
8. **RVM E04.** A CPU legality critic on Blocksworld minimal pairs from `traces.bw_parts`, as the baseline for joint
   item #6.
9. **RSM E11/E12** after BWM's `B3_REPORT.json`.
