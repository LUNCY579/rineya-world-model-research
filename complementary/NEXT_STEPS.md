# NEXT_STEPS (RCX)

Status: 2026-10-09 22:40 CDT.

1. **RCX-E01 instrument.**
   - Write the rule-conditioned ledger (RCL) spec.
   - Build a generator with parent-first splits and disjoint vocabulary, renderer and composition pools.
   - Build an exact interpreter plus an *independently written* second oracle and a rule-text parser.
   - Qualify the interpreter against RCWM `env.step` on the posted-rule subset.
2. **Floor validation (T10), before any training.** Score the shortcut floors on DEV:
   - COPY
   - ALWAYS-APPLY
   - TRIGGER
   - CONTAINS-POSTED
   - STATUS-LOOKUP
   - RULE-WORD-MATCH
   - MAJORITY

   Also score the interpreter ceiling. Redesign the composite if the floors cannot separate RULE from TRIGGER.
3. **Small-model factorial (CPU).** Arms:
   - fixed-rule (T2 analog)
   - fixed-rule surface-diversified (T2b analog)
   - varied-rule ordinary × {answer, trace}
   - varied-rule counterfactual × {answer, trace}
   - filler control
   - canonicalized-input control

   The engineering seed comes first, then 3 seeds for the preregistered comparison.
4. **Native kit.** Render the instrument in Titan's chat frame for a tier-A probe and LoRA cells. Draft the
   preregistration. Native execution waits for a GPU lease, operator approval and peer agreement.
5. **RGM-001 E07** on fresh Blocksworld instances (BWM simulator read-only).
