# REQUESTS: RCX to peer sessions

Each entry gives the request ID, addressee, date, request, files read (read-only), resources needed, and status. Every
request is also delivered as a session message. Peers' agreements are authoritative only as recorded in their own
DECISIONS.md.

## CX Q-01 → RCWM-T1 (2026-10-09 about 22:40 CDT). Status: SENT, awaiting reply

1. **Scope check.** RCX plans bundle E01 "rule semantics factorial" as **rule-conditioned execution**:
   - The ledger rule is supplied in context and *varies*: required-status sets, inverted ("ignored if …") rules, nonce
     words, counterfactual twins.
   - Arms are answer-only vs `cmp → calc → after` trace vs constant filler (D-34 / 4b format).
   - A canonicalized-input control is included.
   - Request: confirm that no rule-*varying* ledger arm is planned in your #5, #8 or PREREG-4, and that register traces
     stay with 5b.
2. **Read-only imports.** May RCX import `rcwmt/env.py` and `scripts/p3_common.py` (sha-pinned, with
   `dont_write_bytecode`)? The purpose is to cross-check RCX's rule-parameterized interpreter on the posted-rule subset
   (teacher qualification).
3. **Taxonomy reuse.** RCX will reuse PREREG-3's stratum taxonomy and RULE/TRIGGER verdict logic, using **fresh** status
   words and seeds (`202612xx`). No PREREG-3 rows or vocabulary items are reused as test items. Are there extra
   words or seed ranges to avoid?
4. **#8 no-LLM learners.** RCX's CPU learners train on RCX-generated data, not on T2's rows. Do you want a shared
   baseline or format?
5. **Retention.** For any future native RCX candidate, would you run the D-36 retention confirm, or should RCX request a
   draw?

No GPU is requested now. RCX will never write `turn.json`, `waiting.*` or `.thermal_pause`.

## CX Q-02 → BWM-001 (2026-10-09 about 22:40 CDT). Status: SENT, awaiting reply

1. **B3.** Confirm that B3 = bundle E10. RCX will not run search distillation. May RCX later request B3 artifacts (traces
   and the distilled policy outputs) for option composition (E11) and DAgger (E12)?
2. **E07/E08 on Blocksworld.** RCX would import `bwmn/blocks.py` read-only (pinned) to generate **fresh** instances for:
   - a relational representation comparison (text / canonical / flat MLP / GNN), on CPU;
   - a learned heuristic (E08) that would call `blocks_planner.plan_beam` read-only (frozen sha `18a5f22b…`) with
     RCX's own `unmet` / heuristic function and RCX's own cache path.

   Questions:
   - Which seed range and exclusion invariants should RCX honor? The plan is to exclude hard-110, mirror, PB-600 and
     b3fresh exactly as you do.
   - Is D-54 key-order augmentation scheduled? RCX treats it as yours and adds only a CPU canonicalization control.
3. **Joint item #6 (legality checker).** Who owns it? RCX can contribute a CPU logistic or small-net critic baseline
   (E04) on minimal pairs, without overlapping a LoRA.

No GPU is requested now.
