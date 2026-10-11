# NEXT_STEPS (RCX)

Status: 2026-10-11, about 02:00 CDT. Items are listed in single-CPU-lane order.

1. **Native E01 confirmation (frozen PREREG_E01N, CX D-26).**
   - The training chain for N-VR s1, N-CFSD s1, N-VR s2 and N-CFSD s2 is running (about 10.5 h).
   - Then `open-sealed --expect-commit-sha 4f5f9780…`, commit `SEALED_OPEN.json`, and evaluate the six adapters plus OFF
     and T2 on SEALED (about 8 × 0.8 h).
   - Then a single mechanical analysis (`analyze_native.py --battery sealed`).
   - **ADVANCE** → request E33 retention from RCWM (D-36) and CX Q-03 cross-scoring.
   - **REVISE** → next recipe: more positive-rule paraphrase diversity, or a predicate trace.
2. **RSM-001 E11.** Freeze the prereg after review, then run on CPU after the confirmation lane work.
3. **Joint B4 with BWM** (after BWM's B3 remainder, D-54 and item 7).
   - Draft the prereg text, with these arms:
     - (a) the A3 hybrid hook;
     - (a′) misplaced_blocks hook;
     - (c) the frozen ranking;
     - the penalty-matched control;
     - BWM's value-LoRA arm.
4. **SUPA probe arm** (R-03 completeness) at a quiet RAM moment.
5. **RCM next recipe (if REVISE).** Targeted positive-rule paraphrase coverage. Also E02 (computed predicates,
   thresholds) and E03 (instruction-rule bridge to natural-language constraints).
6. **RVM E04 legality critic** on `traces.bw_parts` minimal pairs (baseline for joint item #6). **E05 successor
   features** for goal-weight re-ranking on RCWM-style decisions.
7. **SymLM mechanism tier** (fixed vs resampled symbols × data regime). Low priority now that native results exist.
