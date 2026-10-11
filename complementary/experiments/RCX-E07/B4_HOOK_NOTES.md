# B4 arm (a): the RCX GNN as BWM's `unmet=` hook (notes for the co-signed prereg)

**Code:** `rcx/bw/b4_hook.py` (hook, consistency check, sanitiser, model loader, planner-usage check).
**Tests:** `tests/test_b4_hook.py` (`python -B -m unittest tests.test_b4_hook -v`, 11 tests, about 0.2 s after the torch
import). They use tiny randomly initialised RelGNNs only.
**Context:** COLLABORATION.md, 2026-10-10 about 20:00 (BWM-001, joint B4 accepted in principle). PREREG_E07B.md says
E07b's winning variant goes into arm (a).

## 1. How BWM's frozen planner calls the hook (extracted read-only)

The source is `Rineya_BWM_001_Research_Bundle/native_v1/bwmn/blocks_planner.py`, sha256
`18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` (B2's 18a5f22b; the release copy under
`release/.../code/bwmn/` has the same lines).

To regenerate the report, run `python -B -m rcx.bw.b4_hook`, or call `b4_hook.planner_hook_usage()`. It reads the file
as text and parses it with `ast`; it never imports the file and does not need torch. It also checks the file's sha256
against the pin above.

| Item | Fact (line numbers in the pinned file) |
|---|---|
| Signature | `plan_beam(instances, proposer, fwd, *, eval_seed=0, width=4, max_expansions=48, max_depth=80, max_rollout=0, log=None, goal_met=None, unmet=None)`. `unmet` is **keyword-only** and defaults to `None`. |
| Contract (docstring, l. 277) | `unmet(state, goal) -> number (lower is closer)` |
| Binding (l. 280) | `unmet = unmet or (lambda st, g: len(g - st))`. The hook is selected by **truthiness**, so a falsy hook object (for example one with `__len__() == 0`) would be silently replaced by the default. `LearnedUnmet.__bool__` returns True, and the class has no `__len__`. |
| Call site 1 (l. 286) | `unmet(it["init"], it["goal"])`: once per instance, on the **true** initial state, before search. The result seeds `e["best"] = (u, 0, init, [])`. |
| Call site 2 (l. 319) | `um = [unmet(st, e["goal"]) for st in traj]`: on **every imagined state** of every rolled-out candidate. `traj` holds F's predicted states after each step F called legal; it stops at the first illegal or unparseable step. Candidates that reach the goal are scored too. |
| Call form | Both sites call it **positionally with two arguments** (state, goal), with no keywords and **one state per call**. There is no batch call, so `LearnedUnmet.batch` is a convenience only. |
| Arguments | `state` is a frozenset of tuples parsed from F's output and may be inconsistent. The state-format parser (`B.state_from_json`) can produce non-str names (JSON numbers or null). The delta-format parser (`tuple(fact.split())`) can produce unknown predicates, wrong arities and the empty tuple. `goal` is the instance's goal frozenset; BWM goals are `("on", x, y)` facts only. |
| Use of the value | `t_best = min(range(len(traj)), key=lambda t: (um[t], t))` (l. 320) takes the earliest best state. Children are keyed `(u, prefix_len, rank, ...)` and sorted by `x[:3]` (l. 321, 329, 332). The best node is replaced when `(u, ln) < e["best"][:2]` (l. 335). `best` is popped before return (l. 345). |
| Lower = better | **Yes**: the planner applies `min` and ascending sorts. |
| int or float | The default returns an **int**. The value is only **compared** (`min`, sort keys, tuple `<`); the ast check finds no arithmetic, `int()`, `round()` or serialisation applied to it. A **float is accepted**, but it **must not be NaN**, because NaN makes the min/sort order undefined. |
| Solution detection | The hook is **not** used. Solutions are detected by `goal_met`, which defaults to `B.goal_met`, i.e. `goal <= state`. The hook only ranks beam children and picks the best prefix that unsolved instances submit. |

## 2. API

```python
from rcx.bw.b4_hook import LearnedUnmet, is_consistent, sanitise, planner_hook_usage
hook = LearnedUnmet.from_file("runs/RCX-E07/main/models/gnn_s20261276.pt",   # or LearnedUnmet(model)
                              invalid_policy="fallback", hybrid_lambda=0.0, round_to=None,
                              cache_size=65536, instances=None)
hook(state, goal) -> float          # never raises; always finite
hook.batch(states, goal) -> list    # == [hook(s, goal) for s in states], exactly
hook.counters, hook.reasons         # see below; hook.reset_counters()
hook.describe() -> dict             # provenance (JSON-serialisable)
is_consistent(state, goal=None, objs=None) -> (bool, [reason, ...])
plan_beam(..., unmet=hook)          # BWM side; the RCX worktree root must be on sys.path
```

- **Loading.** `from_file` follows `run_e07.load_model`: it builds a new RelGNN, runs
  `load_state_dict(torch.load(path, weights_only=True))` and calls `eval()`. Strict loading is checked on the real
  `gnn_s20261276.pt`, and the weights are identical to the run_e07 path.
  - The architecture comes from the sidecar `<stem>.json` (`"config"`, as written by `phase_train`). If there is no
    sidecar, the default `RelGNNConfig()` is used.
  - `map_location="cpu"` is added.
  - A recorded `n_params` is checked.
  - Only `kind == "gnn"` is accepted. FlatMLP caps n at 12 by construction.
- **`describe()`** returns:
  - the model file, its sha256 and the sidecar's sha256;
  - a sha256 of the weights (state_dict);
  - `config_dict` and the policy settings;
  - the constants;
  - the cache size;
  - the instance registry size;
  - counters and reason counts;
  - the sha256 of `b4_hook.py`, `encode.py`, `models.py` and `space.py`;
  - the torch and Python versions;
  - the planner pin.
- **Counters.** All counters except `calls` and `cache_hits` count per evaluation (cache miss). With no evictions, an
  evaluation is a distinct (state, goal) pair. The counters are:
  - `calls`, `cache_hits`, `evaluations`, `model_evals`;
  - `invalid_states`, `fallbacks`;
  - `encode_errors`, `model_errors`, `nonfinite`, `clipped`;
  - `failsafe`, `analysis_errors`;
  - `bad_goal_facts`, `goal_non_on`.

  `reasons` counts reason codes over the invalid evaluations.

## 3. Value definition and constants

- **Objects.** `objs` is the sorted union of block names in the well-formed facts of the state and the goal, plus the
  instance's blocks if a registry is given. For a consistent state this equals `B.blocks_of(state)`, so the value is
  E07/E08's `LearnedH` value: `models.predict` on `encode.graph(state, goal, objs)`. It can differ from LearnedH only
  in float rounding, because LearnedH batches and runs with more threads.
- **Value by case:**

  | Case | Value |
  |---|---|
  | Consistent state | Model value + `hybrid_lambda` × misplaced_blocks, clipped to ±`MODEL_CLIP` = 500, and rounded to a multiple of `round_to` if set |
  | Inconsistent state, `invalid_policy="fallback"` (default) | `PENALTY` (1000) + misplaced_blocks(sanitised copy, goal) |
  | Inconsistent state, `invalid_policy="raw"` | The model on the state's well-formed facts as they are, clipped and rounded as above |
  | Encode or model error, or a NaN/inf output | The fallback (`PENALTY` + misplaced_blocks on the sanitised copy) |
  | The fallback itself fails (for example, a non-iterable state) | `FAILSAFE` = 1e6 |

- **Ordering guarantee.** Model values satisfy |v| ≤ 500 < 1000 ≤ fallback < 1e6, so every model-scored state ranks
  strictly before every fallback state, and every fallback state before every failsafe state. On real inputs the bound
  of 500 never binds: h\* ≤ 4n, and the trained GNN gives 32–52 on random 15-block states with a one-tower goal.
- **Reasons** (`is_consistent`):

  | Code | Meaning |
  |---|---|
  | `malformed_fact` | Not a tuple, unknown predicate, wrong arity, or a non-str or empty name |
  | `self_on` | on(x, x) |
  | `multiple_positions` | A block has more than one of on/ontable/holding |
  | `no_position` | A block of the state or goal has none of on/ontable/holding |
  | `shared_support` | Two blocks on one block |
  | `multiple_held` | More than one held block |
  | `holding_and_handempty` | Both holding and handempty |
  | `hand_unspecified` | Neither holding nor handempty |
  | `on_held_block` | A block stands on the held block |
  | `cycle` | A cycle in `on` |
  | `missing_clear` | An uncovered, unheld block without `clear` |
  | `wrong_clear` | `clear` on a covered or held block |
  | `unknown_block` | A name outside the instance's blocks; only checked when the `instances=` registry is given |

  `is_consistent` is checked against `space.from_facts` on all 22 states of the 3-block space and all 418 one-fact
  toggles of them. The two agree except that `from_facts` accepts a block standing on the held block. That state is
  unreachable under BW dynamics, and the hook flags it.
- **Block names not in the goal are NOT flagged without a registry.** Partial goals (PB-600 style) leave real blocks
  out of the goal. So do on-only full goals: a one-block goal tower has no on-fact. Pass
  `instances=[{"goal", "objs"}, ...]` (plan_beam's instances) to flag hallucinated names and missing blocks.
- **Sanitised copy** (deterministic, always a valid state, and the state itself when the state is consistent):
  1. Keep the well-formed facts only.
  2. Keep at most one held block: the smallest held name.
  3. Every other block, in sorted name order, takes its smallest on-support that is not held, not already taken and
     does not close a cycle; otherwise it goes on the table.
- **Host safety.** The hook runs inside BWM's process, next to the GPU forward model F. torch and `rcx.bw.models` are
  imported lazily, and the import has no lasting side effects:
  - `models.py` patches `torch.cuda.is_available` and `device_count` and may set `CUDA_VISIBLE_DEVICES=''`; both are
    undone.
  - `sys.dont_write_bytecode`, set by `bwm_import`, is restored.
  - The hook's forward pass runs at `HOOK_THREADS = 1` and restores the caller's thread count afterwards. Measured: no
    overhead, and identical values at host thread counts 1 and 2.
  - Model construction runs under `torch.random.fork_rng(devices=[])`, so the host's CPU RNG is not consumed and the
    CUDA RNG is never touched.
  - BWM code is never imported. `bwm_import.load()` is never called; it would refuse anyway next to BWM's own `bwmn`.
- **Cost.** On this laptop with 1 thread, with the trained 124k-parameter GNN on 15-block states, a distinct state
  costs about 1.0 ms and a cache hit about 1 µs.

## 4. Decisions the B4 prereg must fix (the hook supports each; defaults in bold)

1. **Policy for inconsistent imagined states.** The options are **`fallback`** and `raw`.
   - With `fallback`, arm (a) also gets a *consistency preference*: an inconsistent state always ranks after a
     consistent one. Arm (c), B2's default `len(goal - state)`, has no such preference and even rewards some
     hallucinations, such as an added goal on-fact without the matching delete.
   - To separate "learned value" from "consistency filter", consider a control arm: the B2 ranking plus the same
     penalty. In terms of this module, that is `len(goal - state)` if `is_consistent(state)[0]`, else
     `PENALTY + len(goal - state)`.
   - Report `invalid_states`, `fallbacks` and `reasons` per arm.
2. **Ties.** The default ranking is integer, so ties are frequent and plan_beam's tie-breaks act: the earliest state in
   the trajectory, then the shorter prefix, then the lower candidate rank. GNN floats almost never tie, so those
   tie-breaks become nearly inert. `round_to` (**None**; for example 1.0) restores integer-like ties if wanted.
3. **Model and variant.**
   - The model file and its sha256, from `describe()`.
   - If E07b's A3 hybrid wins, pass `hybrid_lambda=λ` with the A0 seed-1 model; misplaced_blocks is computed exactly
     as `heuristics.misplaced_blocks` for consistent states.
   - A1 or A2 models load by `from_file` if their sidecar carries `"config"` (or pass `config=`).
4. **Registry.** Decide whether to use `instances=` (**off**), which enables `unknown_block` and missing-block
   detection.
5. **Reporting.** Report the hook's own `describe()` next to `bw_select`-style F queries and expansions. The hook's
   cache is RCX's own, as agreed; it changes no value, only cost.

## 5. Limitations

- **Untrained goal relations.** E07 training goals contain only `on` facts (from `B.generate`). The relations for goal
  `ontable`, `clear`, `holding` and `handempty` are encoded, as `encode.graph` does, but untrained. Values for such
  goals are off-distribution; the `goal_non_on` counter shows them. This does not arise for BWM goals.
- **Values on inconsistent states.** In `raw` mode, the GNN's value on an inconsistent state is unvalidated, because
  E07 trained on valid states only. In `fallback` mode, misplaced_blocks on the sanitised copy is a heuristic proxy,
  not a cost-to-go of the imagined state.
- **Non-str names.** Names that are not str (possible from JSON) make the fact malformed. It is dropped for encoding
  and for the sanitised copy.
- **Thread safety.** The hook is not thread-safe; use one hook object per thread. The torch thread count is
  process-wide, and the hook changes it only for the duration of its own forward pass.
- **E08 equality is not bit-exact.** Values equal E08's `LearnedH` only up to float rounding, because LearnedH batched
  predictions at 4 threads. The hook's own values are bit-reproducible: one state per forward pass, at 1 thread.
- **Integration not yet run.** The hook has not been run inside plan_beam, or next to BWM's GPU F. That happens on a BWM
  turn, under the prereg.
