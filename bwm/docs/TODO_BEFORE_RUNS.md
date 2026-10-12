# Must-fix before running these tools (from adversarial reviews, 2026-10-06)

## scripts/planbench_qwen.py (local Qwen3 comparators on PlanBench)
Ticked 2026-10-11 (track D, D-75.4), no model loaded. Evidence: `python -B scripts/planbench_qwen.py --selftest` rc 0;
`python -B -m pytest tests/test_planbench_qwen.py -q -p no:cacheprovider` 78 passed (75 + 3 verifier regression cases); `--dry-run` on hard (Q4N 110,
Q4T-iso 40) and PB-600 without importing torch/transformers. Plan: `research/QWEN_COMPARATOR_PLAN.md`.
- [x] extract_plan: drop actions from deeper-indented commentary under an action-bearing parent; unit tests.
  `bwmn/planbench_extract.py` `_select_actions` (group headers, wrapped lines); `UNIT_CASES` 17/17
  (`test_nl_extractor_unit_cases`); selftest: VAL-verdict agreement with the published extraction GPT-4 599/600,
  o1-preview 596/600, simulator vs VAL 0 mismatches. Verifier fix (track D review): a one-line bold step title (or a
  title ending in a Markdown hard break '\') no longer swallows the indented "*Action:*" commentary under it as a
  wrapped line (it double-counted every action in 11 published responses: R1 8, o1-preview 2, gemini-2.0-flash-thinking
  1; all 11 now match the published VAL verdict, no other extraction changed); `test_nl_bold_title_commentary_*`.
- [x] Second scorer = PlanBench's own text_to_plan_blocksworld; report per-variant VAL; markdown tables, '1)' lists.
  Port reproduces every rule-based published extraction (9 models 600/600, llama-3.1-405b 98/98); rows carry
  `pb_val_valid` and `pb_md_val_valid` (markdown-normalised). Hard (PDDL answers): second scorer = upstream's
  `save_gpt3_response` text checked by VAL as is; reproduces all 330 published step-0 `llm_plan`s and all 330 VAL
  verdicts (R1 59, o1-preview 26, o1-mini 1).
- [x] Plan-heading detection must not capture action lines ("... to complete the plan: stack ...").
  `_PLAN_LABEL` (bare label before the colon only); unit case "Finally, to complete the plan: stack ..." passes.
- [x] Exclusive run lock; refuse 4B run while other native_v1 python jobs run (unless --force); re-check RAM per instance.
  `runs/planbench_qwen/run.lock` (O_EXCL, stale -> `.stale-<pid>.bak`) plus the shared CPU-lane lock of D-76
  (`%TEMP%/rineya-cpu-lane.json`, RCX protocol; 11 lane tests incl. interop with RCX's `rcx/lane.py`); peer gate at
  start, before every instance and every 256 generated tokens (always-on orchestrate/gpu_run/health_log wrappers
  excepted, `test_native_jobs_filter`); RAM >= 12000 MB before any load, >= 4000 MB before every instance,
  >= 2000 MB during generation.
- [x] KV-reuse fallback: narrow exceptions, record them, count re-prefill FLOPs, abort if first instance falls back.
  `_is_cache_error` (memory errors never; RuntimeError only for shape/cache messages; `test_cache_error_narrowing`);
  per-row `cache_fallback_errors`, `reprefill_tokens` counted by `flops_breakdown`; `check_fallback` aborts on the
  first instance. Not exercised against a real model (no model may load in this track).
- [x] FLOPs: add attention-score term (prefill + decode); same formula for the BWM planner.
  `bwmn/flops.py` (dense + 4 L H d_head x causal pairs + LM head; closed form = per-token sum,
  `test_flops_closed_form_vs_per_token_sum`); BWM side through the same functions (`shared_prefix_breakdown`,
  `beam_ledger`), reproducing F-09's 22,895 (H2) and 20,530 (B2) unique F keys and E-01(b)'s 186.9 keys/instance.
- [x] Resume config: script sha, threads, snapshot id, versions, prompt sha; truncate partial last line.
  `run_config` (+ set, subset ids sha, budget rule, upstream file shas) and `read_rows`; `test_read_rows_resume`,
  `test_run_config_records_budget_and_subset`.
- [x] Decide/record PlanBench system prompt ("You are the planner assistant ..."), --max-new >= 1024 or justify.
  `--system-prompt auto`: PB-600 -> PlanBench's chat system message; hard -> none, because upstream's
  send_query_with_feedback comments it out for o1-*/deepseek-r1 and every hard file records a single user turn (the
  prereg may override). --max-new and the iso answer reserve >= 1024 unless --allow-short.
- [x] Move extractor into bwmn/planbench_extract.py (no env side effects on import); stratified subset generator.
  `stratified_subset` (PB-600) and `systematic_subset` (hard; 40-id sha f952f532...); importing the script no longer
  sets CUDA_VISIBLE_DEVICES/BWM_DEVICE (`test_import_has_no_env_side_effects_and_no_torch`); bwmn.lm is imported
  before transformers at model load (D-35 patch; the old loader skipped it for qwen3-4b).

## bwmn/mystery.py, scripts/mystery_data.py
- [ ] Suffix-row signature guard (as in blocks_data.py B-02) + assertion.
- [ ] NameGen substring filter (deceptive words, 4+ letter Blocksworld words); reject single-token English words.
- [ ] 30-50% single-letter object names; report eval by block count.
- [ ] Containment-based Blocksworld-fragment exclusion (>= 3 predicates & 3 actions).
- [ ] Token budget enforced at generation; per-instance seeding, streaming, resume, no silent overwrite.
- [ ] test_signatures include blocksworld_hard (<= 7 objects); noisy-row duplicate resampling; write_domain + VAL check.
- [ ] Eval: accuracy per instance and de-duplicated by signature (603 -> 262).
