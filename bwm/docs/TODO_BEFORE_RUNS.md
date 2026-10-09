# Must-fix before running these tools (from adversarial reviews, 2026-10-06)

## scripts/planbench_qwen.py (local Qwen3 comparators on PlanBench)
- [ ] extract_plan: drop actions from deeper-indented commentary under an action-bearing parent; unit tests.
- [ ] Second scorer = PlanBench's own text_to_plan_blocksworld; report per-variant VAL; markdown tables, '1)' lists.
- [ ] Plan-heading detection must not capture action lines ("... to complete the plan: stack ...").
- [ ] Exclusive run lock; refuse 4B run while other native_v1 python jobs run (unless --force); re-check RAM per instance.
- [ ] KV-reuse fallback: narrow exceptions, record them, count re-prefill FLOPs, abort if first instance falls back.
- [ ] FLOPs: add attention-score term (prefill + decode); same formula for the BWM planner.
- [ ] Resume config: script sha, threads, snapshot id, versions, prompt sha; truncate partial last line.
- [ ] Decide/record PlanBench system prompt ("You are the planner assistant ..."), --max-new >= 1024 or justify.
- [ ] Move extractor into bwmn/planbench_extract.py (no env side effects on import); stratified subset generator.

## bwmn/mystery.py, scripts/mystery_data.py
- [ ] Suffix-row signature guard (as in blocks_data.py B-02) + assertion.
- [ ] NameGen substring filter (deceptive words, 4+ letter Blocksworld words); reject single-token English words.
- [ ] 30-50% single-letter object names; report eval by block count.
- [ ] Containment-based Blocksworld-fragment exclusion (>= 3 predicates & 3 actions).
- [ ] Token budget enforced at generation; per-instance seeding, streaming, resume, no silent overwrite.
- [ ] test_signatures include blocksworld_hard (<= 7 objects); noisy-row duplicate resampling; write_domain + VAL check.
- [ ] Eval: accuracy per instance and de-duplicated by signature (603 -> 262).
