# Titan state-transition tasks: code, data, evaluation harnesses, runtimes, and multi-step coverage

Everything below was read-only. No files were modified, no weights were loaded and nothing was trained.

## 0. Summary

There are two separate code lineages, and both use the same three families and the same system prompts.

- **Lineage A: Forge RL v2, then process labels, then SELECT-VIEWS-001 and the RRP-015 training data.**
  - Scored through the RAB-001 "DenseReference" harness. That harness is pinned to transformers 4.51.3.
  - This lineage produced Titan-Experimental and Titan-Sparse (234/232 out of 384), the arithmetic /150 score and the long-context /256 score.
- **Lineage B: a standalone generator and oracles (FRESH-DUAL-SKILL), producing T_BANK_V2/V3X/V4/V5.**
  - Scored with plain HF `generate`.
  - This lineage produced CONFIRM-002 (Titan-Transition 434/864) and the boundary-abstention bank (250/336).

**Multi-step capability has barely been measured, and where it has, the models are at floor.**
- The only model-scored multi-step item is the DMRF scout's 2-event `multi_step` stratum.
- Supa scored 1/32 and Titan-Sparse scored 1/32. With 3 events, both scored 0/6 on the calibration split.
- Everything else is one-step.

## 1. Exact paths

### Lineage A (repo `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification`, branch `codex/dataset-retention-audit-v1`)

**World generator and solver:** `rineya_reasoning_forge_rl_v2/forge_rl/tasks.py`. Standard library only.
- `make_world(family, rng, tier, transfer)` builds a world.
- `ledger_step()` and `register_step()` apply one event.
- `plan()` is a Dijkstra search over bitmask states. `applicable()` and `apply_action()` handle actions.
- `solve(w)` returns `(answer, steps)`. Each step is `{tick, operation, before, after, event}`.

**Independent checker:** `forge_rl/oracle.py`.
- `replay_transition(f, before, event)` checks a single transition.
- `audit_answer()` and `verify_solution(w, obj, cert)` check whole tasks. `verify_solution` accepts alternative optimal plans, checks feasibility and checks the certificate.
- `verify_trace()` checks a trace.

**Corpus build:** `forge_rl/build.py` (`build_data`, `_build`).
- CLI: `python scripts/forge.py --output <new> --seed 20260914 [--scale 0.002]`.
- It cross-checks every trace step with `replay_transition` (build.py around lines 113–116).
- It writes the following process rows (build.py around lines 160–180):
  - for each training parent, up to 4 steps at indices `round(j*(len-1)/min(3,len-1))`;
  - a valid row for each, plus an invalid row (a perturbed `proposed_after`, ±1 or ±3, or a flipped bit).

**Release data** (root `rineya_reasoning_forge_rl_v2/`):
- `process/train_transitions.jsonl.gz`: 36,752 rows.
  - ledger_replay: 8,000 valid and 8,000 invalid.
  - register_machine: 8,000 valid and 8,000 invalid.
  - prerequisite_plan: 2,376 valid and 2,376 invalid.
- `metadata/{train,validation,test_*}.jsonl.gz`: full worlds plus `trace_steps`, which are the complete per-tick before/after chains. These are the natural source for multi-step data.
  - In validation, `trace_steps` lengths are: ledger 13/28/126 (min/median/max), register 12/24/96, prerequisite_plan 0/1/5 (80 of 200 have 0 steps because the answer is IMPOSSIBLE).
- `data/rl/*.jsonl.gz`: full-task prompts of 512 to 4,112 words.
- `episodes/*.jsonl.gz`: continuous_replan, 4 teacher-forced stages.
- `forge_rl/environment.py`: `PlanningEnvironment` (see section 5).
- `forge_rl/evaluate.py`, `rewards.py` (`score_response`) and `process_eval.py` (`evaluate_process`) are the full-task and valid/invalid scorers.
- The release is pinned in `experiments/reasoning_posttraining_v1/sources.py`.
  - Archive sha `d3bcd3d8…`.
  - The binding file is `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/a1-prepared-003/SOURCE_BINDING.json`.

**One-step view and training-row builder:** `experiments/reasoning_posttraining_v1/process_arm.py`.
- `verified_positives()` re-derives every label with `replay_transition` and keeps one positive step per parent, chosen by salted hash.
- `select()` draws parents. `compile_training()` builds the prompt as `json.dumps({"before","event"}, sort_keys=True, separators=(",",":"))`.
- System rules are read verbatim from the frozen views by `rules_from_views()`.
- `serialization.compile_row()` handles the chat template, the label mask and the single EOS.

**Instrument builder:** `experiments/reasoning_posttraining_v1/select_views.py`.
- Salt `SELECT-VIEWS-001-acquisition-gate`.
- 128 views per family, disjoint from the historical probe's 384 parents.
- `floors()` computes `copy_before` and `constant_majority`.

**Instruments (data):** under `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/`.
- `select-views-001/SELECT_VIEWS.jsonl`: sha `7dd53f2a751b54c3…`, which I verified on disk. The same directory holds `FLOORS.json`, `BUILD.json`, `FAMILY_DECOMPOSITION.json` and `STRUCTURAL_FLOOR_PROBE.json`.
- `process-transition-probe-001/CANDIDATE_VIEWS.jsonl`: the historical 384-view probe, sha `45674251e38e…`, marked "never for selection".

**Evaluation harness:** `experiments/rab001_exploration/process_probe.py`.
- It imports `experiments/architecture_benchmark_001/run.py` (`_load`, `_load_tokenizer`, `_eos_ids`) and `transfer.import_qwen_state`.
- Supervised launch wrapper: `docs/model-quality/audit-2026-09-17/supervised_probe.py`.
- RRP-015 preparation and summary: `prepare_rrp015.py`, `summarize_rrp015.py` and `RRP015_PREREG.md` in the same `docs/model-quality/audit-2026-09-17/` directory.
- Measured evidence: `docs/model-quality/audit-2026-09-17/rrp015-evidence/` (`PROCESS_*`, `LONGCTX_*`, `RETENTION_*`).
- Training CLI: `python -m experiments.reasoning_posttraining_v1.controller train --source <Supa> --prepared … --plan … --grant … --journal … --max-sequence 2048 --experiment RRP-008-PROCESS-DOSE` (full command in `rineya-dataset-retention-records/rep001-20260917/master_rep.sh`).

**REP-001 and SPARSE-001 records:**
- `rineya-dataset-retention-records/rep001-20260917/` contains `run_battery.sh`, `master_rep.sh`, `prepare_rep001.py` and `rep_build.py`.
- `rineya-dataset-retention-records/sparse001-20260917/` contains `build_sparse.py`, which also exists in the published core's `evidence/`.
- `rineya-dataset-retention-records/rrp015-20260917/` holds the RRP-015 records.
- The battery's working directory was `ChatBubble-retention-pilot-20260917`, which **no longer exists**.
- Its recorded `process_probe.py` sha `d89f86ab…` does not match any copy on disk.
- The current copy in ced-native-loader is modified and uncommitted (it adds `weight_identity()`). Its scoring logic looks the same.
- The `long_context_probe.py` sha `43b96ca7…` matches exactly.

### Lineage B (repo `C:/Users/clclc/BusinessProject/ChatBubble-titan-model-soup-mini-v1`, branch `research/titan-opallo-exact-endpoint-v3`; also copied into `ChatBubble-titan-clare-sparse-v1`)

**Oracles:** `experiments/fresh_dual_skill_instrument_v1/oracles.py`, 66 lines.
- `ORACLES[fam](u)` returns a canonical JSON string.
- It was reimplemented from the prompts and validated 384/384 against the historical probe.

**Generators:**
- `generate.py`: v1 sampler. It also holds the `SYSTEM` dict (the exact prompts), `harvest_exclusions()`, `operative_key()` and `content_keys()`.
- `generate_v2.py`: `load_marginals()` and `GENS`, which resample event parameters from the historical probe's empirical marginals.

**Banks:**
- `composition_confirmation_v2/make_t_bank_v4.py` produces `T_BANK_V4.json`.
  - 864 rows, 288 per family, seed 20260926, sha `82f0a642…`.
  - **Spent** by CONFIRM-002.
- `titan_third_skill_v1/make_t_bank_v5.py` produces `T_BANK_V5.json`.
  - Seed 20260927, 864 rows.
  - Its AMENDMENT_1 says no Stage A/B row has been read, so it **appears unread**. Confirm before relying on that.
- Earlier banks:
  - `fresh_dual_skill_instrument_v1/T_BANK.json` and `T_BANK_V2.json`;
  - `composition_confirmation_v1/T_BANK_V3.json` and `T_BANK_V3X.json`.

**Scorers:**
- `fresh_dual_skill_instrument_v1/score_arms.py`: transitions.
- `composition_confirmation_v2/score_boundary.py`: boundary and answerable rows.
- `run_stage_b.py`, `analyse_stage_b.py` and `audit_confirm002.py` run and audit CONFIRM-002.

**Boundary bank:**
- `composition_confirmation_v2/HSKILL_SOUP_V3.jsonl`: 445 rows, sha `686df45b…`, **spent**.
  - Built by `make_hskill_soup_v3.py`.
  - 7 boundary classes × 48 = 336 rows, plus 109 `arithmetic_short_answer` answerable rows.
- Rule module: `rineya-dataset-retention-records/hskill-views-002/boundary_rule.py` (sha prefix `416cfb6a`). It defines `abstains()`, `score_row()` and `ABSTAIN_PATTERNS`.
- A sealed, never-read bank exists: `hskill-views-002/HSKILL_CONFIRM_V2.jsonl`. `score_boundary.py` refuses it by name.

## 2. Task instances

**State and action per family:**

| Family | State | One action (event) |
|---|---|---|
| ledger_replay | dict over accounts A–D, signed integers | `kind` ∈ {add, scale, transfer}, plus `account`, `other`, `amount`, `status` ∈ {posted, void, …} and `tick`. Non-posted events are no-ops. `transfer`: account −= amount, other += amount. |
| register_machine | dict over X, Y, Z, integers mod 97 | `kind` ∈ {add, mix, swap, conditional}, plus `register`, `other`, `value`, `threshold` and `tick`. `mix`: r += 2·other + value. `conditional`: ± value depending on `other > threshold`. |
| prerequisite_plan | integer bitmask, ≤ 8 bits | action `{pre_on, pre_off, delete, add, cost, name}`. Delete is applied before add. |

**Prompt format.**
- Messages are `[{system: rule}, {user: compact JSON {"before","event"}}]`.
- The reference is compact sorted JSON of the next state, or a bare integer for prerequisite_plan.

**Lineage differences:**
- Forge `replay_transition` raises an error on an inapplicable action, so positives cover applicable actions only.
- The FDS oracle returns the state unchanged for an inapplicable action. The v1 FDS generator made 20% of rows inapplicable; v2 makes 100% applicable.

**Full-world parameters (Forge).**
- Ledger: 16/32/64/128 events for tiers 0–3; 256/512 in the length-stress split. Balances start in [−20, 40]. Amounts are in [−9, 12]; scale factors are ∈ {−2, −1, 1, 2}. 12% of events are void.
- Register: 12–96 instructions, values in [−8, 11].
- Prerequisite_plan: 5–7 bits (8 in the transfer split) and 2n+2 actions. The query is a goal mask; the solver finds the minimum-cost plan.

**Concrete SELECT-VIEWS-001 examples.** These are verbatim, and I checked each gold answer by hand.

- **ledger_replay** (`RRP-select:RRFR2-31d93def8457af0c530a281c-step0-valid`)
  - SYSTEM: "Apply one event to the supplied account balances. If status is not posted, leave all balances unchanged. add: add amount to account. scale: multiply account by amount. transfer: subtract amount from account and add it to other, simultaneously. Return only the complete resulting balances as a JSON object."
  - USER: `{"before":{"A":-9,"B":16,"C":3,"D":-3},"event":{"account":"C","amount":-6,"kind":"transfer","other":"A","status":"posted","tick":1}}`
  - GOLD: `{"A":-15,"B":16,"C":9,"D":-3}`

- **prerequisite_plan** (`…RRFR2-a4cfeafea2cb879d5b0d45ea-step2-valid`)
  - SYSTEM: "The state is an 8-bit flag mask. A supplied applicable action requires all pre_on bits set and all pre_off bits clear. Apply delete bits first, then add bits. Return only the resulting state as a JSON integer. Names and costs do not change the state."
  - USER: `{"before":56,"event":{"add":2,"cost":3,"delete":0,"kind":"action","name":"A3","pre_off":0,"pre_on":32}}`
  - GOLD: `58`

- **register_machine** (`…RRFR2-258ddcc36c624d34628032d5-step8-valid`)
  - SYSTEM: "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap exchanges register and other simultaneously. add adds value to register. mix adds (2 times other plus value) to register. conditional adds value if other is greater than threshold, otherwise subtracts value. Reduce updated numeric registers modulo 97; preserve other registers. Return only the complete resulting registers as a JSON object."
  - USER: `{"before":{"X":71,"Y":94,"Z":45},"event":{"kind":"swap","other":"Z","register":"Y","threshold":83,"tick":9,"value":10}}`
  - GOLD: `{"X":71,"Y":45,"Z":94}`

**T_BANK_V4 example** (ledger): before `{"A":20,"B":18,"C":6,"D":115}`, transfer A to B of 1, posted, gives `{"A":19,"B":19,"C":6,"D":115}`.
- V4 rejects rows that collide with any of 6,464 historical payloads, by full payload, operative key or (before, answer) pair.
- It drops 75% of rows whose answer equals the input; 59 copy-baseline rows remain.

**Floors** (from `select-views-001/FLOORS.json`, published before any model call):

| Family | Binding floor | Predictor |
|---|---:|---|
| ledger | 30/128 = 0.2344 | copy_before |
| prerequisite_plan | 10/128 = 0.0781 | constant_majority |
| register | 9/128 = 0.0703 | copy_before |

For comparison, the historical probe's floors are 31, 9 and 7.

## 3. How models were evaluated

### A. `process_probe.py`, used for SELECT-VIEWS-001

```
cd ChatBubble-ced-native-loader-qualification
python experiments/rab001_exploration/process_probe.py --model <dir> --label <L> --instrument select-views-001 [--checkpoint <MQ ckpt>] [--max-new-tokens 96]
```

- **Rendering:** `tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)`. With Qwen3 this ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- **Model:** the HF state dict is converted by `import_qwen_state` into an FP32 `DenseReference`. TF32 is forced off and matmul precision is "highest".
- **Decoding:** a manual greedy argmax loop with KV cache, `max_new_tokens` 96. It stops on the full EOS list from `generation_config.json`, `[151645, 151643]`.
- **Output:** decoded with `skip_special_tokens=True` and stripped.
- **Scoring:**
  - `strict`: the text equals the reference exactly.
  - `canonical`: `json.loads` and a sorted compact dump equal the reference.
  - A family clears if the two-sided Clopper–Pearson 95% lower bound on the canonical count is above the binding floor rate.
- **Guards and output:**
  - `hostbudget.require(3.0)` GiB.
  - Output goes create-only to `experiments/architecture_benchmark_001/evidence/retention-design-v1/PROCESS_<label>.json`.
- **Timing:** 290 s for a trained arm over 384 views; the Supa baseline took 642 s.
- **Gotcha:** `run._load` raises `GateClosed` unless transformers == 4.51.3 (pinned in `protocol.json`).

### B. `score_arms.py`, used for T_BANK_V2–V5

- **Rendering:** a hand-built string, `<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. It is self-tested byte-for-byte against `rineya-dataset-retention-records/qwen001-20260918/render-process/prompts.jsonl`, with a negative control.
- **Model and decoding:** `AutoModelForCausalLM` in float32 on cuda, `generate(do_sample=False, max_new_tokens=48, eos_token_id=tok.eos_token_id)`. Note this passes a single EOS id.
- **Scoring:** strict exact match after strip. The registered primary rule also strips at most one trailing ".".
- **Timing:** about 430–490 s per arm over 864 rows.

### Boundary scoring (`score_boundary.py`)

- Same setup as B, but rendering uses `apply_chat_template(..., enable_thinking=False)`, generation uses the EOS list, and TF32 is off.
- Boundary rows are credited when `abstains()` fires on the first line.
- Answerable rows are scored by canonical exact match.

### Shared caveat

`generation_config.json` for every Titan core says `do_sample: true, temperature: 0.6`. Any new harness must force greedy decoding explicitly.

### Training-row serialization (for consistency with any new data)

- `compile_row(prompt, final, tok)` requires the prefix to end in `<think>\n\n</think>\n\n`.
- Labels cover the assistant span plus exactly one EOS.
- No truncation is allowed; the window is 2048 tokens.
- RRP-015 rows averaged about 141 processed tokens; the maximum was 162.
- Training data comes from Forge TRAIN-split parents disjoint from all 768 scored parents.

### Baseline numbers (SELECT-VIEWS-001, canonical correct)

| Model | ledger | prerequisite_plan | register | Total /384 |
|---|---:|---:|---:|---:|
| Supa | 14 | 0 | 0 | 14 |
| Titan-Experimental | 54 | 95 | 85 | 234 |

**Titan-Experimental error profile:**
- ledger: 57 rows with 1 value wrong, 17 with 2 wrong.
- register: 43 rows with 1 value wrong, often a missing mod (for example Y = 105).
- prerequisite_plan: 33 wrong integers.

**T_BANK_V4 per family** (ledger / prerequisite_plan / register, out of 288 each):
- Titan-Sparse: 136 / 126 / 168, total 430/864.
- BASE: 5 / 0 / 0.

**Best recorded SELECT-VIEWS-001 arm overall:** `normline001_stack2` at 298/384 (119 / 96 / 83). It is not published.

## 4. Python environments

| Purpose | Interpreter | Versions | Status |
|---|---|---|---|
| Lineage A eval (process / retention / long context) | `C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe` | Py 3.11.15, torch 2.10.0+cu126, **transformers 4.51.3** | **Deleted.** Last used 2026-09-23. Needs rebuilding to reproduce the numbers. |
| Training (MQ / RRP controller) | `C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` | Py 3.11.15, torch 2.10.0+cu126, transformers 4.57.1, safetensors 0.7.0 | Exists. Pins are in `experiments/model_quality_gpu_v1/native.py` (`PINS`). |
| Lineage B / CONFIRM-002 | `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe` | Py 3.11.15, torch 2.6.0+cu124, **transformers 5.14.1**, accelerate 1.14.0 | Exists. This is the default `python` on PATH. |
| Other | `ChatBubble-rineya-spikingbrain-hybrid-memory-v1/experiments/rineya_spikingbrain_hybrid_memory_v1/.venv-cuda` | Py 3.10.11, torch 2.6.0+cu124, transformers 4.51.3 | Exists. The only surviving 4.51.3 install with CUDA, but a different Python and torch. |

## 5. Multi-step and plan-level evaluation: what exists

1. **DMRF scout `multi_step` stratum.**
   - Code: `ChatBubble-titan-tangent-task-arithmetic-v1/experiments/delta_multi_reasoning_v1/scout/build_scout.py` (`item_multi_step`, `MULTI_STEP_EVENTS = 2`, `RULES_MULTI`).
   - Each item is 2 chained add/scale events on **one** ledger account, with user content `{"before", "events": [...]}`.
   - Results (`DMRF001_MR_RESULTS.md`): Supa 1/32 and Titan-Sparse 1/32 under every inference condition. With 3 events, both scored 0/6 on the calibration split while producing well-formed JSON.
   - With events spread across accounts, 3 of 4 balances were often right but exact match was 0.
   - This is the only model-scored multi-step transition data.
2. **Forge v2 `PlanningEnvironment`** (`forge_rl/environment.py`).
   - A real `reset()`/`step(action)` interface, **prerequisite_plan only**.
   - Actions are a name, `STOP` or `IMPOSSIBLE`. Reward is sparse and terminal: 1 only for an optimal-cost goal.
   - Never run with a model.
3. **Plan-level verifier:** `oracle.verify_solution()` accepts any optimal plan and checks feasibility. Used by `forge_rl/evaluate.py` and `scripts/score_predictions.py`.
4. **Forge v2 full tasks:** prompts of 512–4,112 words. RRP006 records that the v2 reasoning view keeps 0 of 2,000 ledger rows within the 2048 training window. No Titan model has been scored on them.
5. **Other related evidence:**
   - `episodes/`: continuous_replan, 4 causal teacher-forced stages.
   - `experiments/rab001_exploration/forge_probe.py`: Forge v1 full tasks, about 1,400-token prompts, reasoning nulls.
   - `PUZZLE_CEILING.md`: Supa scored 102/450, below the 119 needed to proceed. hanoi_planning and code_breaking were at literal zero.

**Implication for multi-step data.** `metadata/*.jsonl.gz` `trace_steps` provides oracle-verified chains of 12–126 consecutive (before, event, after) steps for ledger and register. Prerequisite_plan chains are only 1–5 actions long, and 40% of worlds are IMPOSSIBLE.

You can build k-step consequence rows directly from these chains. Two requirements:
- Use TRAIN parents not among the 768 scored parents (`process_arm.verified_exclusions`).
- Use `replay_transition` or `ORACLES` as gold.

Alternatively, `generate_v2.GENS` can be iterated for fresh synthetic chains, with the `harvest_exclusions()` and `operative_key` collision checks.

## 6. Long-context /256 and arithmetic /150 instruments

### Long context /256

- **Data:** `ChatBubble-ced-native-loader-qualification/Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl` and `all_answer_key.jsonl` (also present in `Chat-Bubble/`).
  - 256 rows: 192 stability rows and 64 showcase rows; 64 of the 256 are unanswerable.
  - 8 task types × 32.
  - Input lengths are 1,119 / 1,340 / 1,556 tokens (min / median / max).
- **Harness:** `experiments/rab001_exploration/long_context_probe.py --model <dir> --label <L>`.
  - Greedy, `max_new_tokens` 24, `max_context` 4096. Overflowing rows are skipped, never truncated.
  - Scored by normalized first-line exact match against `accepted_answers`.
  - Literal UNKNOWN on the 64 unanswerable rows is counted separately.
  - Output: `LONGCTX_<label>.json`.
- **Status:** the suite is **SPENT**, report-only. Comparator values under transformers 4.51.3: Supa 161, Titan-Experimental 162 (157 for its FP32 parent), Titan-Sparse 164.

### Arithmetic /150

- **Data:** `experiments/architecture_benchmark_001/data/monitor_probe.jsonl`, sha `5f6df1c14aa0…`, verified. It has 50 rows each of percentage_reasoning, ratio_proportion and time_duration, some multi-turn.
- **Harness:** `python experiments/rab001_exploration/retention_probe.py <label> hf:<dir> cuda`.
  - Rendering: `apply_chat_template(messages, enable_thinking=False)` with `add_special_tokens=False`.
  - Greedy, `max_new_tokens` 24, `max_context` 512 (`protocol.json` "decoding").
  - Scoring: `retention.score_prediction` typed grammar under bare policies `permissive` and `strict`.
- **Source reference:**
  - Re-scored from `experiments/architecture_benchmark_001/outputs/retention-a329eb7b/B28/retention.json` (47 permissive / 41 strict); this file exists.
  - The harness refuses to run if `implementation_digest()[:8]` is not `a329eb7b`.
  - Label overlay: `evidence/retention-design-v1/PROBE_LABEL_AUDIT.json`.
- **Guard:** `hostbudget.require(5.0)` GiB.

### Resource note for this machine

With about 2 GB of host RAM free, the hostbudget guards refuse to start: 3 GiB for process, 5 GiB for retention, and 3.5 GiB for the CONFIRM-002 preflight.

Model files:
- Titan-Sparse and Titan-Transition are FP32 (2.38 GB) and must not be converted to BF16.
- Titan-Experimental is BF16 (1.19 GB).
- All three are Qwen3ForCausalLM: 28 layers, hidden size 1024, 596,049,920 parameters.