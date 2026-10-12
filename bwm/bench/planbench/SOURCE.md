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

## Blocksworld-hard published per-instance results (added 2026-10-11, DECISIONS D-75.3; same upstream commit fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284, MIT)
- Upstream directory: `llm_planning_analysis/results_backprompting/blocksworld_hard/<engine>/`. Each engine has one file,
  `task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json`,
  vendored unchanged under `results/blocksworld_hard/<engine>/` with the same file name:

  | Engine dir (model) | Bytes | sha256 | git blob sha1 at fc638a1a |
  |---|---|---|---|
  | `deepseek-r1-api` (DeepSeek-R1) | 3,741,291 | `332b0f8ec03f337b4f320e4986b3d758c0fbb6efd223c7cfef05534e8202740e` | `22ca8ced06058f04532cfe4a74f57459e1014046` |
  | `o1-preview_chat` (o1-preview) | 2,263,485 | `b8fdc9ba2b7cbd3f1f26a3e4aabe2be5ac108a1499916ef91809a88b1a51478c` | `87ccabe011fabdcb752af3a1142f96b83b24cc79` |
  | `o1-mini_chat` (o1-mini) | 3,054,739 | `928497901fc3e4be5b78fe26a5afdfa2939dec6c5b6e291c5cd67211fd71bc44` | `7f2e7e2bf825e1dc4a1e43ab0dd4f6e25f6eb208` |

- How the files were obtained: copied byte-for-byte from the H2 leak/claims audit download,
  `research/audit_evidence/bwm_h2_scratch/dl_upstream_hard/{deepseek-r1-api,o1-preview_chat,o1-mini_chat}.json`; the
  sha256 values are identical. Provenance check: the git blob sha1 of each local file equals the tree entry of the same
  path at fc638a1a (`git ls-tree -r fc638a1a` in the blob-less partial clone
  `research/audit_evidence/bwm_item9_scratch/llmsp`; listing saved as `runs/HARD_PUBLISHED_PAIRED/upstream_tree_hard.txt`).
  The same listing shows that the local `blocksworld_hard/generated/instance-*.pddl` files are byte-identical to
  upstream's (110/110 blob matches).
- Use: ONLY step 0 of each record's `step_wise_dict` (the first answer to the zero-shot PDDL prompt, before any VAL
  feedback) is compared, re-scored by the local VAL. Later steps are the verifier-in-loop (backprompting) regime and are
  not compared. Script: `scripts/hard_published_paired.py`. It refuses to run unless both the sha256 and the git blob
  match the pins above. Output: `runs/HARD_PUBLISHED_PAIRED/`. The comparison is POST HOC and in a DIFFERENT REGIME and
  must never be worded "beats".
- Upstream extractor sources used for the port: `llm_planning_analysis/utils/llm_utils.py` (blob 62f36232…,
  `save_gpt3_response`), `back_prompting_parallel.py` (blob 2a1e302c…) and `utils/text_to_pddl.py` (blob ba3c66b2…).
  Copies are in `runs/HARD_PUBLISHED_PAIRED/upstream_src/`, with blob ids matching the fc638a1a tree.
- These result files are reference outputs only. Like the instances, they are TEST ONLY: never used for training,
  selection or prompt construction.
