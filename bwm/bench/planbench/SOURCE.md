# PlanBench assets (vendored for evaluation only)

Source: https://github.com/karthikv792/LLMs-Planning (MIT, see LICENSE_LLMs-Planning_MIT.txt), commit
fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284 (2025-09-17), cloned 2026-10-06.
VAL (planner_tools/VAL) is LGPL-3 per upstream; used unmodified as an external validator binary.
Contents: Blocksworld / Blocksworld-Hard / unsolvable instances, Mystery instances, VAL, and published per-instance
results for Task 1 (plan generation) and Task 7 (plan execution) for the models listed under results/.
These instances are TEST ONLY: never used for training, selection or prompt construction.

## Logistics (added 2026-10-10 for roadmap item 9; same upstream commit fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284, MIT)
- `logistics/generated_domain.pddl`, `logistics/generated_basic/` (286 instances, t1 set), `logistics/generated/` (286,
  t5 generalization set), `logistics/logistics.yaml`: from `plan-bench/instances/logistics` and `plan-bench/configs`.
- `logistics/prompts/task_1_plan_generation.json`: PlanBench t1 queries and Fast Downward astar(lmcut) ground-truth plans.
- `logistics/generator/logistics.c`, `README.txt`: from `plan-bench/pddlgenerators/logistics` (source only; the ELF binary is not copied).
- `results/logistics/`: published LLM result files (gpt-3.5-turbo-instruct, gpt-4_chat, llama-3.1-405b_aws,
  o1-preview_chat) from `llm_planning_analysis/results/logistics`.
Copied from a local clone of the upstream repository; contents unchanged.
