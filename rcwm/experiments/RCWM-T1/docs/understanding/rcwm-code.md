# RCWM reference package: technical read-out

Bundle root: `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05`. I changed no files and loaded no weights. I ran pytest with `-p no:cacheprovider` so it would not write `.pytest_cache`.

## 0. Test run

- **Interpreter on PATH:** `/c/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python`.
- **Versions:** Python 3.11.15, torch 2.6.0+cu124 (`torch.cuda.is_available()` = True), numpy 2.4.6, pytest 9.1.1, jsonschema 4.26.0, PyYAML 6.0.3.
- **Command:** `cd reference && python -m pytest -q -p no:cacheprovider`
- **Result:** `60 passed in 3.75s`, all on CPU. This matches the bundle's recorded `validation/pytest_output.txt` (`60 passed in 2.10s`). That record came from Linux, Python 3.13.5, torch 2.10.0+cpu, with no GPU tested.
- **Packages:** no `pip install -e` is needed, because `pyproject.toml` sets `pythonpath=["."]` and `testpaths=["tests"]`.
- **Dependency outside the package:** `tests/test_resource_contracts.py` loads `../tools/validate_resources.py` by path. It needs `jsonschema` and `yaml`, and both are present.

## 1. Modules: classes, functions, signatures

The package is `reference/rcwm/`, 420 lines in total. Its `__version__` is "0.1.0". The only stated dependencies are `numpy>=1.26,<3` and `torch>=2.2,<3`; test extras are `pytest`, `jsonschema` and `PyYAML`.

**`interfaces.py`**
- `class FrozenHistoryEncoder(Protocol)`
  - Attributes: `checkpoint_sha256: str`, `capture_version: str`.
  - Method: `encode_available_history(self, histories: list[dict]) -> torch.Tensor`, which returns `[batch, slots, width]` built from observed history only.
- This is a typing Protocol with no implementation. It is the intended hook for an LLM backbone.
- Its docstring requires a pinned tokenizer, mask and precision, parity tests, read-only use, cache provenance, and no target leakage.

**`synthetic.py`**
- Constants: `ACTIONS=("wait","take_key","open_door","collect","shortcut")` and `FEATURES=("key_gained","door_opened","object_gained","damage","action_cost")`.
- `transition(state, action:int) -> (next_state float32[4], features float32[5])`
- `rollout(initial, actions) -> (features [H,5], states [H,4])`
- `make_dataset(n:int, horizon:int, seed:int=0) -> (states [n,4] f32, actions [n,H] int64, targets [n,H,5] f32)`

**`model.py`**: `class FeatureTrajectoryTransformer(nn.Module)`. Details in section 3.

**`masks.py`** (boolean masks, where **True = BLOCKED**, following the `nn.Transformer` convention)
- `causal_mask(length:int, *, device=None) -> BoolTensor[L,L]` returns `ones.triu(1)`, so the diagonal is allowed.
- `prefix_plan_mask(prefix:int, actions:int, *, device=None) -> BoolTensor[n,n]`. History rows cannot see plan columns; plan rows see everything, i.e. bidirectional over a precommitted plan. The model does not use it.
- `validate_dag(parents: list[list[int]]) -> None` requires a topologically sorted DAG with unique parent indices, each earlier than its child.
- `@dataclass(frozen=True) ObservedEvent(available_at: float, observed: bool=True)`
- `available_observation_mask(query_times, events) -> BoolTensor[Q,E]` blocks a key if `not observed or available_at > t`.

**`codebook.py`**: `class TrainOnlyKMeans(clusters:int, seed:int=0, iterations:int=30)`
- `.fit(x, *, split:str)` raises an error unless `split=="train"`. It uses Lloyd's algorithm with random initial centres and stops at `atol=1e-10`.
- `.predict(x) -> labels` returns nearest-centre labels and never refits.
- It is a diagnostic, not a VQ layer.

**`data.py`**
- `split_for_parent(parent_id:str, seed:int=0) -> "train"|"dev"|"test"`. It hashes `sha256(f"{seed}:{parent_id}")[:8]/2**64` and splits at 0.70 / 0.85, giving a 70/15/15 parent-stable split.
- `validate_open_loop(record:dict) -> None` checks:
  - the required keys `parent_id, split, plan_origin, decision_time, observations, actions`;
  - `plan_origin=="open_loop_precommitted"`;
  - every observation has `kind=="observed"` and a finite `available_at <= decision_time`;
  - every action has a non-empty `operator` and `executor_version`;
  - no action contains `observed_outcome` or `observed_reward`.

**`returns.py`**
- `discounted_features(features[T,F], gamma=1.0, *, durations=None, terminated=None, truncated=None) -> np[F]`
  - It sums `x_t * gamma**elapsed_before_t` up to and including the first terminated or truncated step.
  - Durations give a continuous-time discount; without them each step counts as 1.
  - It does not bootstrap and does not treat truncation as a zero continuation value.
- `linear_utility(features[...,F], weights[F]) -> features @ weights`. This is the goal readout.

**`metrics.py`**
- `candidate_regret(actual_utilities, selected:int) -> max − actual[selected]`
- `brier_score(probabilities, targets) -> float`
- `paired_parent_bootstrap(parent_ids, baseline, candidate, *, seed=0, draws=2000, confidence=.95) -> dict`
  - It computes per-parent mean differences, weighting each parent equally, then a percentile bootstrap CI over parents.
  - It requires at least 2 parents and at least 100 draws.
  - It returns `difference, ci_low, ci_high, n_parents, draws, confidence, scope="conditional_on_fixed_models"`.

**`demo.py`**: `main()`, run with `python -m rcwm.demo --out DIR --steps 250 --seed 17`. Details in section 4.

**`tools/validate_resources.py`** (outside the package, used by tests): `validate_record(record)` does the following:
- validates against Draft 2020-12 `trajectory.schema.json`;
- runs `validate_open_loop`;
- requires unique `action_id`s and parents that refer to earlier IDs;
- requires each outcome's `action_id` to exist and be unique, with non-overlapping, increasing times (`start >= previous_end`, `end >= start`);
- requires `len(features)==len(feature_valid)` and `len(goal_weights)==len(features)`.

`main()` validates the schemas, examples, configs and sources, and writes `validation/resource_validation.json`. I did not run it.

## 2. Synthetic environment (`synthetic.py`)

**State:** binary `[key, door, object, damage]`. `transition` raises an error for any other shape or non-binary value.

**Actions and dynamics** (deterministic):

| id | name | effect | cost |
|---|---|---|---|
| 0 | wait | none | 0.1 |
| 1 | take_key | key=1 | 1.0 |
| 2 | open_door | door=1 if key, else nothing | 1.0 |
| 3 | collect | object=1 if door, else nothing | 1.0 |
| 4 | shortcut | if no object: object=1 and damage=1 | 1.0 |

- **Per-step consequence features (5):** `s_next − s_before` (4 deltas, each 0 or 1), plus `action_cost`, which is 0.1 for wait and 1.0 otherwise.
- **Success paid once:** a feature can be gained at most once, because the state latches.
- **Failed precondition:** the step's features are all-zero deltas plus the cost.

**Dataset generation:**
- Initial state: start from zeros and apply 0–3 random actions.
- Plan: `horizon` uniform random actions.
- Targets: per-step features from `rollout`.
- There are no goals inside the dataset.

**Plan structure:** a flat open-loop sequence of action IDs of fixed length H. There is no DAG, no arguments, and no variable length within a batch.

**Goals:** linear weights over the 5 features, defined only in the demo:
- `avoid_damage = [0,0,10,-20,-1]`
- `prefer_speed = [0,0,10,-1,-2]`

**Statistics I measured** with `make_dataset(1024,4,17)` for train and `(256,4,18)` for check, numpy only:
- **Reachable states:** only **7** in total, and all 7 appear as initial states. The zero state accounts for 567/1024.
- **Possible (state, plan) pairs:** 7 × 5^4 = 4,375. **106/256 (41%) of check pairs are exact duplicates of train pairs**, so the check set is not held out.
- **object_gained:** 454 events in train, of which **418 come from the shortcut and only 36 from the key→door→collect chain**.
- **The demo's safe chain:** only 4 zero-start train rows gain the object without damage, and only 2 begin with the exact plan prefix (1,2,3).
- **Per-step non-zero rates:** key 0.113, door 0.064, object 0.111, damage 0.102, cost 1.0.
- **Baseline:** predicting the per-feature mean gives check MSE 0.0954.

## 3. `model.py` architecture

Constructor: `FeatureTrajectoryTransformer(state_dim=4, actions=5, features=5, width=48, heads=4, layers=2, latent=24, max_horizon=16)`. The demo uses `max_horizon=8` and keeps the other defaults.

**Input tokens:** `[state_token, a_1 … a_H]`, length H+1.
- `state_token = Linear(state_dim, width)(observed_initial)`
- `a_t = Embedding(actions, width)`
- Plus a learned `position = Embedding(max_horizon+1, width)` and `type_embedding = Embedding(2, width)`, where 0 is state and 1 is action.

**Trunk:**
- `nn.TransformerEncoder` of `layers` × `TransformerEncoderLayer(width, heads, dim_feedforward=2*width, dropout=0, batch_first=True, norm_first=True)`, i.e. pre-LN with ReLU, then a final `LayerNorm(width)`, with `enable_nested_tensor=False`.

**Mask:**
- `causal_mask(H+1)`, so position t sees the state and actions 1..t.
- The output at action t predicts the consequences of action t.
- There is no padding mask, so every sequence in a batch has the same H.

**Heads:**
- `encoded[:,1:]` → `latent = Sequential(Linear(width,latent), Tanh())`, a per-step bottleneck z ∈ [−1,1]^24.
- `decode = Linear(latent, features)`.
- Return value: `{"features": [B,H,F], "latent": [B,H,latent]}`.

**Forward signature:** exactly `(observed_initial [B,state], candidate_actions [B,H] long)`. A test enforces that no goal or target argument is passed.

**Input guards:**
- `ndim`, matching batch size, `1 ≤ H ≤ max_horizon`, `long` dtype, finite state, and action ID in range.

**Losses:** none in the module. The demo uses plain `F.mse_loss` over all B×H×5 targets. That loss is unweighted and unnormalised, ignores `feature_valid`, uses no BCE for the binary events, and has no latent, contrastive or VQ terms.

**Parameter count:** 40,325 with the demo config. My breakdown:

| component | params |
|---|---|
| state | 240 |
| action embedding | 240 |
| position embedding | 432 |
| type embedding | 96 |
| 2 layers × 18,960 | 37,920 |
| final norm | 96 |
| latent | 1,176 |
| decode | 125 |

## 4. Demo: what was measured, and the numbers

Sources: `validation/demo/demo_result.json`, `demo_stdout.txt`, `demo_quality_assessment.json`.

**Setup:**
- Seed 17, `torch.set_num_threads(2)`, `torch.use_deterministic_algorithms(True)`.
- Train set `make_dataset(1024,4,17)`; check set `make_dataset(256,4,18)`.
- AdamW, lr 2e-3, wd 1e-4, batch 64 sampled with replacement, 250 steps, grad-clip 1.0, CPU.

**Results:**
- **Check MSE:** initial **0.54092**, final **0.0021860**. That is about 44× better than the mean baseline of 0.0954 I computed, but on a check set with heavy train overlap.
- **Train batch MSE trace:** 0.5436 (step 1), 0.0480 (26), 0.0342 (51), 0.0236 (76), 0.0139 (101), 0.00617 (126), 0.00238 (151), 0.00345 (176), 0.00312 (201), 0.00379 (226), 0.00624 (250). The trace is noisy and not monotone after step 151.
- **Training time:** 1.27 s. Parameters: 40,325.

**Revaluation probe:** two plans scored from the zero state; cumulative features are the per-step predictions summed over 3 steps.

| plan | predicted | actual |
|---|---|---|
| safe `[1,2,3]` | [0.985, 1.033, **0.0638**, 0.0551, 2.996] | [1, 1, **1**, 0, 3] |
| fast `[4,0,0]` | [−0.0115, 0.0113, 1.009, 1.018, 1.219] | [0, 0, 1, 1, 1.2] |

- **Per-feature errors:** safe plan [−0.015, +0.033, **−0.936**, +0.055, −0.004]; fast plan all within 0.02.

| goal | predicted scores | actual scores | chosen plan |
|---|---|---|---|
| avoid_damage | [−3.461, −11.496] | [7.0, −11.2] | correct (safe) |
| prefer_speed | [−5.411, 6.635] | [4.0, 6.6] | correct (fast) |

**Quality assessment** (`demo_quality_assessment.json`):
- `native_qualified=false`, `per_plan_fidelity_qualified=false`.
- Reason: "Critical prerequisite-chain consequence underprediction despite low aggregate MSE."

**My extension of the probe:** I swept goal weights w=[0,0,10,−d,−1] using the reported predictions.
- In reality the safe plan wins for d > 1.8.
- The model only ranks it first for d > 11.66.
- So for **d ∈ (1.8, 11.66) the model picks the wrong plan**, with regret d − 1.8: 3.2 at d=5, 8.2 at d=10, 9.8 at d=11.6.
- The two goals in the demo sit outside this band, which is why the ranking looked correct.
- The root cause matches the data imbalance in section 2: the model learned "object ≈ shortcut", and the low aggregate MSE hides this rare-event failure in the composed chain.

## 5. Schemas (`schemas/*.json`, Draft 2020-12)

**`trajectory.schema.json`** ("precommitted trajectory v1", `additionalProperties:false`)
- `schema_version`: const "1.0".
- `parent_id`: string, minLength 1.
- `counterfactual_family_id`, `episode_id`: strings.
- `split`: train / dev / test.
- `environment`: `{name, version, snapshot_sha256: string|null}`; no extra keys.
- `plan_origin`: const "open_loop_precommitted".
- `decision_time`: number.
- `observations[]`: `{kind: const "observed", available_at: number, value: string|object|array|number|boolean}`.
- `actions[]` (minItems 1):
  - `{action_id (minLen 1), operator (minLen 1), arguments: object, executor_version (minLen 1), parents: string[] unique}`
  - optional `budget: object` and `proposed_start: number`.
- `actual_outcomes[]`:
  - `{action_id, start, end, status ∈ {success, invalid_precondition, timeout, tool_error, refused, noop}, features: number[] (≥1), feature_valid: bool[] (≥1), terminated: bool, truncated: bool}`
  - optional `label_source: string`.
- Optional at top level: `goal_weights: number[]`, `feature_schema: string`, `capture: object|null`.

**Cross-field rules** live in `validate_resources.validate_record` (section 1), not in the schema.

**Examples:**
- `examples/trajectory.json` is toy-parent-001-safe, split test: take_key → open_door → collect, features per step, last step `truncated=true`, `label_source "exact_simulator"`.
- `examples/trajectories.jsonl` has 4 records: parents 001 (test) and 002 (train), each with a safe and a fast plan sharing `counterfactual_family_id`.

**`prediction.schema.json`** (`additionalProperties:false`)
- Required: `parent_id`, `candidate_set_id`, `model_id`, `goal_id`, `scores: number[] (≥1)`, `selected_index: int≥0|null`, `observed_utility: number|null`, `costs: object`.
- Optional: `fallback_reason: string|null`, `support: number[]`.
- Example scores: [7, −11.2], selected 0, `observed_utility` null, costs `{"units":"not_measured"}`, support [1, 1].

**`run_manifest.schema.json`** (`additionalProperties:true`)
- Required: `run_id`, `experiment_id`, `scope ∈ {synthetic_mechanics, native_capture, native_training, evaluation}`, `status ∈ {proposed, blocked, running, completed, failed}`, `seed: int`, `code_revision|null`, `checkpoint_sha256|null`, `data_manifest_sha256|null`, `hardware`, `software`, `artifacts: string[]`, `claims: string[]`.

## 6. Reuse with a frozen 0.6B Qwen3-architecture backbone, and what is missing

**Relevant planned design** (from `configs/` and the report):
- Experiment arms (`report/07_experiments.md`):
  - B0: random or heuristic.
  - B1: frozen LLM acting directly.
  - B2: scalar trajectory Transformer.
  - B3: measured features plus goal readout. This is what `model.py` plus `linear_utility` implements.
  - **B4: learned continuous consequences plus readout. This is the central arm.**
  - B5: clustering or VQ.
  - B6: successor features.
- `configs/pilot_sidecar.yaml`:
  - frozen encoder, `history_mode: available_prefix_only`, `max_text_tokens: 512`, `cache_action_boundaries_only: true`;
  - sidecar width 256, 3 layers, 4 heads, latent 64, `max_action_horizon` 4, `prospective_causal` mask;
  - 2000/300/500 train/dev/test parents; AdamW with lr ∈ {1e-4, 3e-4, 1e-3}; microbatch 1; effective batch 32; clip 1.0;
  - loss weights: grounded features 1.0, everything else 0, enabled one at a time.
- `configs/native_adapter.yaml`: LoRA ranks {4, 8, 16}, sequence length 512, bf16, proposed peak GPU admission 6.5 GiB. It is marked deferred.
- Native action registry (`configs/action_registry.yaml`): RETRIEVE, CALCULATE, VERIFY, ANSWER, plus RECUR, which is deferred.
- Native features (`configs/feature_registry.yaml`): terminal_success, satisfied_prerequisite_delta, elapsed_ms, generated_tokens, tool_failure.
- Primary comparison (`evaluation_gates.yaml`): B4 − matched B2. Seeds 17, 29, 43; bootstrap 10,000 draws. The endpoint and margins are null, i.e. not yet pre-registered.

**Reusable as-is:**
- `returns.discounted_features` and `linear_utility`, as the goal readout and return accounting.
- `metrics.candidate_regret`, `paired_parent_bootstrap` and `brier_score`, as the evaluation core.
- `data.split_for_parent` and `validate_open_loop`, for leak-safe 70/15/15 parent splits.
- `masks.causal_mask`, `prefix_plan_mask` and `available_observation_mask`. Note the True=blocked convention: HF/Qwen attention masks use the opposite convention (1/True = attend), so these must be inverted before passing them.
- `codebook.TrainOnlyKMeans`, for a B5 diagnostic on latents.
- `tools/validate_resources.validate_record` and the three schemas, as the data contract and JSONL format.
- The 60 tests, as regression tests.

**Reusable with small changes:**
- `FeatureTrajectoryTransformer` as the sidecar head:
  - Set `state_dim` to the backbone hidden size, which is 1024 for standard Qwen3-0.6B; I did not check the Titan config.
  - Feed a pooled or last-token hidden state, ideally from a mid or late layer, as `observed_initial`.
  - For multi-slot history (`FrozenHistoryEncoder` returns `[B,slots,width]`), the model must accept K prefix tokens instead of one. That means changing the token concatenation and position embedding size (`max_horizon+1` → `K+max_horizon`), and either `causal_mask(K+H)` or `prefix_plan_mask`.
  - A bigger change is to embed actions as text through the backbone, so that operators carry arguments.
- `synthetic.py` as a controlled pilot, but only after these fixes:
  - break the 7-state support with held-out compositional families;
  - rebalance the chain against the shortcut;
  - render state and actions as text so the LLM actually encodes them.

**Missing (must be built):**
1. A concrete `FrozenHistoryEncoder`: HF Qwen3 loading, hidden-state capture at action boundaries, an fp16/bf16 activation cache on disk, and checkpoint/tokenizer SHA pinning. This is a Protocol only today.
2. A loader from trajectory JSON(L) to tensors: an operator vocabulary built from `action_registry`, argument encoding, padding, `key_padding_mask`, `feature_valid` masks, and variable horizons in one batch.
3. A real loss: masked per-feature loss, with BCE for event features and MSE or Huber for continuous ones, plus train-split normalisers and rare-event weighting. The demo's failure is exactly what this would fix.
4. A B2 scalar-utility baseline with a matched budget, and a B1 direct-LLM baseline.
5. An end-to-end evaluation loop: candidate sets per parent, goal sweeps, regret and decision accuracy reported per parent, and output in the prediction schema. Today only the primitives exist.
6. B4 components: a self-supervised target encoder for learned latents, latent prediction or contrastive objectives, uncertainty or calibration heads, VQ, and successor features.
7. Any real environment or task with verifiable consequences (the native RETRIEVE/CALCULATE/VERIFY/ANSWER executors), and a goal-language mapping, since goals are currently hand-written weight vectors.
8. A GPU path. On CUDA, `use_deterministic_algorithms(True)` needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The bundle also has no AMP, device selection or batching of the backbone forward pass.

**Hardware fit:**
- A 0.6B model in bf16 is about 1.2 GB of weights. Frozen, inference-only feature capture at 512 tokens fits easily in 8 GB.
- The 40k–2M-parameter sidecar trains trivially on cached features.
- The binding constraint is host RAM, which is currently about 2 GB free. Load with `low_cpu_mem_usage` / safetensors mmap and stream the cache to disk.

**Bottom line:** the most transferable research signal in the bundle is the demo's own failure. Aggregate MSE of 0.0022 hid a 0.94 underprediction on a rare compositional chain, and that produces wrong decisions across a whole band of goal weights (regret up to 9.8). The demo was evaluated only at two goal weights, both outside that band. Any LLM-backbone version needs per-plan, rare-event and goal-sweep regret metrics, not aggregate MSE.