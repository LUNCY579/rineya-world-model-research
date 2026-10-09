# Ledger beam test, RCWM side: the interface for BWM (joint roadmap item 3)

Date: 2026-10-09. Written by RCWM-T1 for BWM-001. **Revision 3.** Revision 2 (08:30) answered the adversarial review of revision 1. Revision 3 applies the constraints that follow from PREREG-3's verdict.

Everything here ran on the CPU with 2 threads at below-normal priority. Nothing touched the GPU or the lease. No real model was loaded: the model tests used a tiny random Qwen3 + LoRA.

**The question.** Put the T2 seed-0 LoRA as the world model F inside BWM's `plan_beam`, and ask whether its near-oracle one-step prediction (PREREG-2) turns into goal reaching.

**Pinned BWM code.** The tests imported `bwmn/blocks_planner.py` read-only, sha256 `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` (D-53 hooks). No bytecode was written into either bundle.

**Binding constraints.**
- **PREREG-3: STATUS: TRIGGER (REPORT §3c).** T2 is reliable only inside its trained surface: accounts A–D, statuses posted/void, |amount| ≤ 12, |balance| ≤ 98, canonical sorted keys. Outside it, T2 breaks: amounts 13–490 score 0.14–0.25, reordered keys 0.37–0.47, 4–5-digit balances 0.50–0.60.
- **DECISIONS D-34(c).** Joint item 3 problems must stay inside T2's verified range. Out of range is reported only.
- **DECISIONS D-35.** Every claim gets a surface-form audit alongside the out-of-range audit.
- **What revision 3 enforces:**
  1. The **primary** problem distribution stays strictly inside that range at **every state reachable by every vocabulary action, along any plan up to the registered depth**. This is enforced in the generator and checked by BFS over all reachable states.
  2. `LedgerF` **always** renders the canonical frame.
  3. Out-of-range problems are a **separate, descriptive** set only.

## 1. Files

| file | sha256 | role |
|---|---|---|
| `scripts/lbeam_f.py` | `9a76562c2739df3254a650154acde41fe2052173b2f35556e5b75a7728680b6f` | `LedgerF` (canonical frame only; arms T2 / OFF / ORACLE / IGNORE_STATUS), `RangeGuardF` + `guarded()`, `BudgetF`, `SurfaceAuditF` (D-35, report only), `SimulatedF` (sizing only), `SharedLoader`, `load_registered_t2()` |
| `scripts/lbeam_problems.py` | `2353e2fc846fac5ced11cfc5420bd60b6e9e50cf3399e9299974b3921a09465e` | Generator v2: canonical forms, the hooks `goal_met` and `unmet`, reachable-state `closure`, `registered_max_depth`, oracle BFS, the fixed-candidate view, freshness, blind `planner_instances`, answer-free floors, `execute` and `score` |
| `scripts/lbeam_proposer.py` | `6c555d2e46db8a279541839a54b82d6e765f1e8bc438f735beb3ef30b9564881` | `EnumerateProposer` (recommended), `RandomProposer`, `FixedCandidatesProposer`, `OracleProposer` (harness only), `select_fixed`, **`beam_kwargs`** (the registered `plan_beam` keywords) |
| `scripts/lbeam_test.py` | `1f21b65f9ca92d76448c077b3f67e9c95711e342d7ebdfcdacea0b23a30a47c4` | CPU tests: `--part oracle` (38 checks), `--part tiny` (25), `--part fresh` (10) |
| `docs/ledger_beam/dev2_primary.jsonl` | `bb63aef48262f4643e8bc5bfc97990ee71f43ff8bee4a047e0603b04c1f10d05` | **PRIMARY dev set**: 120 in-range instances (24 per d* = 2..6), closure depth 8, seed 20261142 |
| `docs/ledger_beam/dev2_primary.SOURCE.json` | `0e8c2620bec3de022c6a2eebc68581497dec42dc3c298ba8129af44ac45c7b3f` | Build record (12:24): spec, rejections, freshness sources, union size, describe, verify |
| `docs/ledger_beam/dev2_oor.jsonl` | `2d497c88fbb133318921529bf2d4327cf10617a14b5bd0ec3790fce166427439` | **DESCRIPTIVE out-of-range set**: 30 instances (6 per d*), seed 20261143. Report only. |
| `docs/ledger_beam/dev2_oor.SOURCE.json` | `fea43680021d1167676426143c6425d9f6d66cf7a9df7f90bf82ec6826ecd1c0` | Build record |
| `docs/ledger_beam/TEST_ORACLE.json` | `18fa14c7b543ba9ae3b48452aecf19731c67626b423f8300d8d2764a526d60c3` | 38/38 checks; all numbers in §9–§10 |
| `docs/ledger_beam/TEST_TINY.json` | `c1b74d2a3af1f2bbffb745f7c79d8f6bd7e2f7efca76e122e5f9b2031904ee02` | 25/25 checks |
| `docs/ledger_beam/TEST_FRESH.json` | `c8b89c8addfaad3112164cc44479812d0da85f04f814fb89a972d63be179d176` | 10/10 checks, against the union at 12:27 |
| `docs/ledger_beam/dev_set.jsonl`, `dev_set.SOURCE.json` | `a64f1424…`, `8b4d0322…` | **SUPERSEDED** v1 dev set: generator v1, seed 20261141, no closure. 103 of its 120 in-range instances fail the closure. Kept unchanged as a record; do not use it. |

Each TEST record stores the four code sha256s and both set sha256s above, plus the BWM planner sha. All three were written by the final code.

## 2. What changed

### 2.1 Revision 3: the PREREG-3 constraints

| requirement | implementation |
|---|---|
| Primary strictly in range at every reachable state | **Generator v2** (`lbeam-problems-v2`). An in-range draw is accepted only if **every state reachable from init by any sequence of ≤ 8 vocabulary actions** has \|b\| ≤ 98. This is `P.closure`, a BFS over all reachable states (all orders, with repetition). Every vocabulary action also has a trained amount (−9..12) and status posted/void. 1,404 draws were rejected for the closure. **Tests:** the closure is re-derived independently with `env.step` on dicts (603,187 reachable states, max \|b\| = 98, 0 violations), and also under the rule-blind dynamics, which reach the same state set. |
| "along any plan up to max_depth" | `plan_beam`'s default `max_depth=80` lets long searches imagine states deeper than any finite closure. **`PR.beam_kwargs(proposer, W, E, R)`** sets `max_depth = D_CLOSE − sim_len`, where sim_len is the simulated candidate length. Every F input (depth ≤ max_depth + sim_len − 1) and every exact F output (depth ≤ max_depth + sim_len) is then a closure state. **Tests:** across 47 guarded ORACLE / IGNORE_STATUS runs (820,576 distinct keys summed over runs), 0 inputs and 0 outputs fell outside the range and 0 keys were refused. With the default `max_depth=80`, enum W1 E16 queried 56 keys outside the range. |
| `LedgerF` always renders the canonical frame | `LedgerF` has **no surface parameter**: passing one is a `TypeError`. Before every generation call it asserts that each prompt is byte-identical to `env.one_step_messages` (`LF.is_canonical_frame`: Titan's system text, compact sorted-key JSON, accounts A–D, posted/void). **Tests:** a swapped non-canonical renderer raises before any generation, and every one of 6,850 prompts sent in the text-path runs was canonical. |
| Out of range: a separate descriptive stratum only | **`dev2_oor.jsonl`** is its own set with its own seed (20261143). It is never pooled with the primary and never guarded; every instance leaves the verified range. The primary set contains only `in` instances. |
| D-35 audit vs canonical-only F | The audit renderings moved out of `LedgerF` into a separate, report-only **`SurfaceAuditF`**, which is not a `LedgerF` subclass. |

### 2.2 The revision-1 review items (status after revision 3)

| review item | status |
|---|---|
| **MUST-FIX:** in-range search queried F at \|b\| > 98 | **Applied, and now holds by construction.** The closure plus the registered `max_depth` keep every exact-F search in range. `RangeGuardF` stays registered for every arm: under an exact F it is inert (0 refusals), and it refuses only the wrong states a model F imagines beyond the bound (SimulatedF: ≤ 0.1% of keys). |
| **SHOULD:** INTERFACE stale on PREREG-3 | **Applied.** The verdict and D-34(c) are stated above. The oor stratum is a separate descriptive set. |
| **SHOULD:** D-35 surface audit | **Applied as paired, report-only audit arms** (`SurfaceAuditF`, 7 surfaces). **Declined:** a separate surface-audit *problem stratum*. The audit arms render the same primary instances, so they are paired instance by instance, and `LedgerF` must stay canonical. Claim scope is Q7. |
| **SHOULD:** findings rest on a weak random proposer | **Applied.** `EnumerateProposer` is the recommended default. Finding 1 is stated at exactly matched calls, with the 1.5× / 2.0× curve. |
| **SHOULD:** GPU estimates assume near-oracle T2 | **Applied.** `SimulatedF` presets size T2 and OFF (§10). |
| **SHOULD:** `planner_instances` hands BWM the answer key | **Applied.** Blind by default: id, init, goal and objs only. |
| **NOTE:** freshness record stale | **Applied.** Both v2 sets were rechecked against the union at 12:27 (§5). |
| **NOTE:** answer-free floors | **Applied** (§5). **Declined:** enriching needs-lookahead instances in the generator. It changes the draw distribution, so it is left to the joint prereg (Q6). `score()` reports the stratum. |
| **NOTE:** tiny coverage gap | **Applied.** The text-path run asserts `n_generate_fn_calls == n_predict_new` (18 = 18) and `model_queries == len(requested) == len(memo)`. |

## 3. Quick start (CPU, ORACLE F)

```python
import sys; sys.dont_write_bytecode = True          # never write bytecode into either bundle
sys.path[:0] = [RCWM_T1, RCWM_T1 + "/scripts", BWM_NATIVE_V1]
import lbeam_problems as P, lbeam_proposer as PR, lbeam_f as LF
from bwmn import blocks_planner as BP

insts = P.load_set(RCWM_T1 + "/docs/ledger_beam/dev2_primary.jsonl")  # RCWM records (answer key inside): RCWM side only
PI    = P.planner_instances(insts)       # BLIND: [{"id", "init", "goal", "objs": vocab tuple}] - what BWM receives
own   = P.owner_of_action(insts)         # action tuple -> instance id (every action has exactly one owner)
inner = LF.LedgerF("ORACLE")             # or "T2" / "OFF" with model + tok (section 4)
F     = LF.guarded(inner, insts)         # = RangeGuardF(inner, applies_to=P.in_range_actions(insts)); REGISTERED form
prop  = PR.EnumerateProposer()
eps   = BP.plan_beam(PI, prop, F, eval_seed=0, **PR.beam_kwargs(prop, 2, 12))   # W2 E12; max_depth 7 + the hooks
summary, rows = P.score(insts, eps, inner.calls_by_owner(own))   # use this, NOT BP.score (that one is Blocksworld)
F.stats()                                                         # exposure log: refused keys (0 for an exact F)
```

**Always pass `**PR.beam_kwargs(...)`.** It supplies `goal_met`, `unmet` and the registered `max_depth`. `plan_beam` forwards only `id`, `objs`, `goal`, `state` and `round` to the proposer; the vocabulary travels as `objs`.

## 4. F (`scripts/lbeam_f.py`)

### 4.1 `LedgerF`: canonical frame only

```python
LF.LedgerF(arm, model=None, tok=None, loader=None, adapter_name=None, generate_fn=None,
           max_new_tokens=None, cache=None, identity=None)            # no surface parameter
```

| arm | what it is |
|---|---|
| `T2` | Titan-Sparse FP32 (fingerprint `818be9c6b6a3`) plus the T2 s0 LoRA (`runs/t2/t2_s0/adapter`, sha `92bc7c74…`), unmerged PEFT. Never dense-merged (D-28): every call asserts that no LoRA layer is merged. Before each generation, `set_adapter(adapter_name)` is called and PEFT's LoRA-layer state is checked. |
| `OFF` | The same PeftModel inside `model.disable_adapter()` (= Titan-Sparse), in the same process. |
| `ORACLE` | `env.step`, no model. |
| `IGNORE_STATUS` | Diagnostic floor: every status treated as posted (PREREG-2's F_ignore_status). |

**`predict(keys)`.** This is exactly BWM's interface.
- Malformed keys raise `ValueError`; they are programming errors, not "unsupported".
- The distinct keys that are not yet memoised go to the model in **one** `rcwmt.generation.generate_texts` call, which splits long lists into the registered sub-batches.

**Frame and decoding.** These are byte-identical to PREREG-2's ledger `chain_one` step; a test checks the texts against `generation.chain(..., "chain_one")`.
- Prompt: `env.one_step_messages`, the only rendering `LedgerF` has. It is asserted with `LF.is_canonical_frame` before every generation call.
- Decoding: greedy; EOS [151645, 151643]; thinking off; FP32; 160 new tokens.
- Batching: ≤ 32 prompts and batch × (prompt + 160) ≤ 8,000 tokens per `model.generate`.
- Parser: `env.parse_state(..., lenient=True)`.

**Legality.** The ledger has no illegal events: `env.step` applies every event to every state, and a non-posted status is a no-op. So every parsed answer is returned as `(next_state, True)`. A parse failure, or an answer whose account set differs from the input's, gives `None` (memoised).

**Accounting.**
- `.model_queries`: new distinct keys computed.
- `.generated_tokens`: exact new tokens before EOS.
- `.requested`: distinct keys since `reset_accounting()`. This is the budget measure, independent of the cache.
- `.calls_by_owner(own)`: calls per instance.
- Diagnostics: `n_predict`, `n_predict_new`, `n_generate_fn_calls`, `n_model_generate_calls`, `prompts_per_model_generate`, `decode_steps`, `texts`.

**`cache=path`.** A JSONL file of raw answers, fsynced after each batch. Its header records class, arm, frame and identity (schema `lbeam-f-cache-v3`). A file from another arm, or from an audit surface, refuses to load.

**Loading is injected** through `model=`/`tok=` or `SharedLoader(fn)`. `load_registered_t2()` is GPU-only: call it inside `runtime.gpu_lease(...)`; it refuses on CPU.

```python
with runtime.gpu_lease(label="rcwm-t1:lbeam", job="RCWM-T1 ledger beam (T2/OFF)"):
    model, tok, prov = LF.load_registered_t2()
    f_t2  = LF.guarded(LF.LedgerF("T2",  model=model, tok=tok, adapter_name="t2", cache=OUT/"f_t2.jsonl",  identity=prov), insts)
    f_off = LF.guarded(LF.LedgerF("OFF", model=model, tok=tok,                     cache=OUT/"f_off.jsonl", identity=prov), insts)
```

### 4.2 `RangeGuardF`: the registered input-range guard

```python
LF.RangeGuardF(inner, applies_to=P.in_range_actions(insts), bound=98)    # or LF.guarded(inner, insts)
```

- **Rule.** A key is refused when its input lies outside T2's verified range (`P.in_verified_range`): any \|balance\| > 98, an untrained amount, or a status other than posted/void.
- **Effect.** A refused key returns `None` (unsupported: BWM's candidate stops). It never reaches the inner F and is not a model call.
- **Same for every arm**, and model-free: it reads only F's own imagined input.
- **On a v2 primary set with `beam_kwargs`, it is inert for an exact F.** ORACLE and IGNORE_STATUS gave 0 refusals in every run. It refuses only wrong states a model F imagines beyond the bound. Simulated T2-like F: 0–11 keys per configuration (≤ 0.03%). Simulated OFF: 8–25 keys (≤ 0.1%). Tiny noisy-text model: 257 of 3,693 keys.
- **Exposure log:** `.seen`, `.refused`, `.oor_key_share()`, `.exposure(own, range_of_id)`, `.refused_by_owner(own)`, `.stats()`.
- **Wrap order for hard budgets:** `RangeGuardF(BudgetF(LedgerF(...)))`, so refused keys are never charged.

### 4.3 `SurfaceAuditF`: the D-35 audit arms (report only)

```python
LF.SurfaceAuditF(arm, surface, model=..., tok=..., adapter_name="t2", cache=OUT/f"f_t2@{surface}.jsonl")   # T2 / OFF only
```

This class renders the **same** canonical (state, action) in a held-out surface, using PREREG-3's frame helpers (`p3_common.messages` / `parse_ledger`, imported read-only), and maps the answer back to A–D.
- It is a separate class, never the primary and never a comparator.
- The planner, proposer, hooks and problems do not change, so every audit arm is paired instance by instance with the canonical run. Use it under the same guard.

| surface | rendering | PREREG-3 analogue (T2 s0) |
|---|---|---|
| `keys_top_only` | `event` first, inner keys sorted | F_keyorder (even) |
| `keys_reversed` | `event` first, event and balance keys reverse-sorted | F_keyorder (0.37–0.47); D-36 held-out "reversed" |
| `names_letters` | accounts rendered E–H | A_renamed letters (0.82–0.90) |
| `names_words` | accounts rendered bank / card / cash / fees | A_renamed words (0.60–0.66) |
| `status_cancelled` | `void` rendered `cancelled` (posted events unchanged) | U_cancelled (unseen statuses 0.01–0.18) |
| `status_pending` | `void` rendered `pending` | U_pending |
| `heldout_combined` | keys_reversed + names_words + status_cancelled | D-36 combined held-out rendering |

### 4.4 `BudgetF` and `SimulatedF`

**`BudgetF(inner, own, cap)`.** Hard per-instance caps on distinct keys. New keys beyond an instance's cap return `None`; keys already charged stay free.

**`SimulatedF(preset)`.** A deterministic, model-free noisy F keyed by sha256 on (seed, state, action). It is **for sizing GPU turns only, never a comparator.**

| preset | void applied as posted | posted answer wrong (one account off by 1–9) | source |
|---|---|---|---|
| `SIM_T2` | 0 | 0.09 | PREREG-2: four-step exact 0.698 ≈ 0.914 per step; void 1.00 |
| `SIM_T2_P3` | 0 | 0.16 | PREREG-3 T_posted\|at s0 = 0.84 |
| `SIM_OFF` | 0.90 | 0.55 | PREREG-2 OFF: void 0.04–0.13, posted 0.30–0.57 |

## 5. Problems (`scripts/lbeam_problems.py`, generator v2)

No model, tokenizer or model output is used. Each instance's RNG is keyed on (version, seed, range, depth, slot, attempt).

**Domain.**
- **State.** `(("A", a), ("B", b), ("C", c), ("D", d))`, hashable.
- **Action.** `(kind, account, other, amount, status, tick)`. `P.action_to_event(a)` gives the exact `env.gen_event` dict. Every action tuple belongs to exactly one instance of a set.
- **Goal.** A sorted tuple of `(account, target)` pairs: all 4 accounts (76 of 120 primary instances) or 1–3 of them (44).
- **Hooks.**
  - `P.goal_met(state, goal)`: exact equality on the listed accounts.
  - `P.unmet(state, goal)`: the sum of \|balance − target\| over the listed accounts.

**Instance construction.**
- **Init.** Balances from the recorded pool, with \|b\| ≤ 98.
- **Vocabulary.** V = 8 actions:
  - 2–5 posted **core** actions; a hidden seed plan of length d uses each of them;
  - 1–2 **void twins**: a core action with status `void`, the same arithmetic, and a new tick;
  - posted **random** fill, with at least one action that moves away from the goal.
- **Oracle.** BFS gives the optimal depth d* ∈ 2..6, which must equal d. It also stores one optimal plan and the number of optimal plans.
- **Closure (primary).** Every state reachable within `d_close` = 8 vocabulary steps has \|b\| ≤ 98. Stored per instance: depth, bound, number of reachable states (median 4,206, max 18,662), and max \|b\|.

**Sets.**

| set | file | definition | n | status |
|---|---|---|---|---|
| primary | `dev2_primary.jsonl` (seed 20261142) | `in` only: init, goal, every vocabulary amount (−9..12) and status (posted/void) inside T2's trained surface; **reachable-state closure at depth 8** | 120 (24 per d*) | **The primary.** Run with `guarded()` and `beam_kwargs`. |
| oor | `dev2_oor.jsonl` (seed 20261143) | Amounts ±13..99 with probability 0.5 on posted add/transfer, init balances ±100..999 with probability 0.4 per account; the optimal plan must touch an out-of-range value | 30 (6 per d*) | **Descriptive only.** Never pooled, never guarded. 100% of its F keys lie outside the verified range. |

**What the closure does to the distribution.** All transitions stay inside T2's trained ranges, but the primary set is a *subset* of PREREG-2's event distribution:
- **Scale actions** have factor −1 (53 posted) or 0 (15 posted) only. A factor \|k\| ≥ 2 on an account that is non-zero in some reachable state escapes \|b\| ≤ 98 within 8 applications.
- **Kinds:** transfer 608, add 276, scale 76.
- **Init balances:** median \|b\| 16, p90 42, max 98.
- **Optimal plans:** median number by d* = 2..6 is 2, 3, 10, 60 and 60.

**Fixed-candidate view (PREREG-2 style).** K = 8 plans of length d*: the optimal plan plus distractors (void_swap 240, substitute 240, random 342, order_swap 18). Each distractor truly fails.

**Answer-free floors (primary).** These are computed by `P.answer_free_floors`, not stored.

| floor | value |
|---|---|
| fixed view, uniform pick over K | 0.125 |
| fixed view, uniform over **void-free** candidates | 0.233 |
| greedy one-step descent on the TRUE unmet hook solves | 77 / 120 (by d* = 2..6: 0.96, 0.71, 0.71, 0.46, 0.38) |

`score()` reports `success_by_lookahead`: `greedy` (77 instances) and `needs_lookahead` (43).

**Freshness (binding).** The binding transitions are init × vocabulary, every optimal-plan step and every fixed-candidate step.
- **Harvest.** Name-skipping: `HSKILL_CONFIRM_V2*` is never opened or hashed, and `rcwmt.bank.load_exclusions()` is never called. The union also covers PREREG-3's bank and every other `data/*.json[l]` payload and multi-event transition.
- **At build (12:24).** The union was 312,996 payloads from 28 sources: the 9 `data/t2br_*` files were absent while the 5b kit rebuilt them. They are register-family rows and cannot collide with ledger payloads. There were 0 overlaps and 0 freshness rejections.
- **Against the current union (12:27, 314,901 payloads / 307,043 operative, 37 sources; the 9 t2br files back).** Primary: 3,145 binding transitions, 0 / 0 overlaps. Oor: 775, 0 / 0. Depth-2 balls (7,152 and 1,792 transitions, descriptive): 0 / 0. Multi-event prompts in the records tree (1,068 transitions): 0 / 0. `SurfaceAuditF` renderings of the primary: 0 payload overlaps in all 7 surfaces. Regenerating with the union reproduces both sets byte for byte.

## 6. Proposers, the baseline and the registered beam keywords (`scripts/lbeam_proposer.py`)

All proposers here are model-free, with **`n_tokens = 0`**, so every budget is an F budget.

- **`EnumerateProposer(depth=1)`** (recommended default).
  - Proposes every vocabulary action as a one-step candidate, in vocabulary order; `depth=2` gives all 64 pairs.
  - Deterministic and independent of the batch and of `eval_seed`.
  - Its per-instance cost has a hard ceiling of (E + W − 1) · V F calls.
- **`RandomProposer(K, length=6)`.** K uniform random sequences from the item's `objs`, with CRN seeds keyed on (eval_seed, id, round, j).
- **`FixedCandidatesProposer(records)`.** The stored K candidates at round 0. Use it with W1 E1. It needs the RCWM records.
- **`OracleProposer(K=1)`.** BFS-shortest plans. Harness only.
- **`select_fixed(records, F)`.** The PREREG-2-style selection baseline. It needs the RCWM records; a candidate F cannot support ranks last.
- **`beam_kwargs(proposer, W, E, R=0)`** returns `{width, max_expansions, max_rollout, max_depth, goal_met, unmet}`, with `max_depth = 8 − sim_len`:

  | proposer | max_depth |
  |---|---|
  | enum depth 1 | 7 |
  | enum depth 2 | 6 |
  | random L6 R3 | 5 |
  | random one-shot | 2 |
  | fixed / oracle | 2 |

**Budget note.** `max_expansions` E counts node expansions, not rounds. The bound is E + W − 1 expansions.

## 7. Scoring (`P.score(insts, eps, calls_by_id)`)

**Success** means that executing the submitted plan from init under `env.step` meets the goal exactly. The summary reports:
- verified rate and verification precision;
- optimal-length rate;
- `nregret`;
- success by depth, by range and by lookahead;
- F calls per instance (mean, median, p90, max).

## 8. Registered form of a run (proposal)

- **Instances:** the blind `planner_instances(insts)` of the **primary** confirmation set (generator v2, closure depth 8).
- **F:** `LF.guarded(LedgerF(arm), insts)` for every arm: T2, OFF, and ORACLE as the ceiling. IGNORE_STATUS is a diagnostic.
- **Planner call:** `plan_beam(..., **beam_kwargs(prop, W, E, R))`, which fixes `max_depth`.
- **Calls:** the distinct keys requested per instance, `inner.calls_by_owner(own)`. The exposure log is `F.stats()` for each arm.
- **Fixed baseline:** `select_fixed(records, F)` on the RCWM side.
- **Report-only arms:**
  - audit arms `SurfaceAuditF("T2", surface)` under the same guard;
  - the oor set, T2 only, unguarded.

## 9. Test results (all on CPU)

All checks pass: oracle 38/38, tiny 25/25, fresh 10/10. Peak RAM was 559, 836 and 365 MB.

**Closure (primary, 120 instances).**
- An independent BFS with `env.step` on dicts covered 603,187 reachable states within 8 steps. The max \|b\| was 98, with 0 violations and 0 count mismatches against the generator's records.
- Every vocabulary amount is a trained value (\|amount\| ≤ 12), and every status is posted or void.
- The rule-blind dynamics reach the same state set on 120/120 instances.
- 1,404 draws were rejected for the closure. The superseded v1 set fails the closure on 103/120 of its in-range instances.
- All 30 oor instances leave the verified range.

**Harness and baseline (ORACLE F, guarded, `beam_kwargs`).**

| setup | result |
|---|---|
| `OracleProposer` + plan_beam W1 E1 | success, verified and precision 1.000; every plan of optimal length; nothing refused |
| Fixed view through plan_beam (W1 E1) | success 1.000 |
| `select_fixed` | success 1.000, regret 0. F calls per instance 22.8 (12.0, 16.8, 22.5, 28.0 and 34.6 for d* = 2 to 6) |
| `select_fixed`, IGNORE_STATUS F | success 0.567; it picked a void_swap distractor 50 times |

**Range exposure.**
- **Exact F with `beam_kwargs`:** 47 guarded ORACLE / IGNORE_STATUS runs gave 0 refused keys, 0 out-of-range inputs and 0 out-of-range outputs.
- **`plan_beam` default `max_depth=80`, ORACLE, unguarded:**

  | config | keys outside range | success, default → registered | calls, default → registered |
  |---|---|---|---|
  | enum W1E16 | **56 of 6,312** (24 refused when guarded) | 0.792 → 0.767 | 52.6 → 38.9 |
  | random W4K8R3E32 | 0 of 25,774 | 0.808 → 0.758 | 214.8 → 168.3 |
  | random W8K64R3E64 | 0 of 70,827 | 0.975 → 0.975 | 590.2 → 548.7 |
  | enum W8E96 | 0 of 24,560 | 0.967 → 0.967 | 204.7 → 192.9 |

  The registered `max_depth` costs a few ORACLE successes only in long, deep searches. The recommended enum W2E12, W4E24 and W8E48 lose nothing.
- **Noisy F (SimulatedF, guarded):** refused keys for enum W2E12 / enum W4E24 / random W4K32R3E8 were 0 / 0 / 3 (SIM_T2), 0 / 0 / 11 (SIM_T2_P3) and 8 / 16 / 25 (SIM_OFF). No refused key reached the inner F, and every admitted input was in range.

**Success against budget, ORACLE F, primary.** "Greedy" and "needs-lookahead" are success rates on those subsets (77 and 43 instances). Precision is 1.000 in every row, and success never falls as E grows.

| config | E | max_depth | success | greedy / needs-lookahead | calls: mean (p90, max) |
|---|---|---|---|---|---|
| enum W1 | 6 | 7 | 0.742 | 1.000 / 0.279 | 35 (48, 48) |
| enum W1 | 8–16 | 7 | 0.767 | 1.000 / 0.349 | 39 (64, 64) |
| enum W2 | 12 | 7 | **0.850** | 1.000 / 0.581 | **62** (104, 104) |
| enum W4 | 24 | 7 | 0.900 | 1.000 / 0.721 | 111 (168, 200) |
| enum W8 | 48 | 7 | **0.967** | 1.000 / 0.907 | 192 (320, 440) |
| random one-shot K32 | 1 | 2 | 0.342 | 0.468 / 0.116 | 139 (150, 158) |
| random one-shot K256 | 1 | 2 | 0.700 | 0.818 / 0.488 | 662 (784, 831) |
| random W4K8R3 | 8 | 5 | 0.617 | 0.740 / 0.395 | 126 (169, 174) |
| random W4K8R3 | 32 | 5 | 0.758 | 0.831 / 0.628 | 168 (291, 395) |
| random W4K32R3 | 8 | 5 | 0.883 | 0.948 / 0.767 | 262 (434, 490) |
| random W4K32R3 | 16 | 5 | 0.917 | 0.987 / 0.791 | 291 (529, 839) |
| random W8K64R3 | 32 | 5 | **0.975** | 1.000 / 0.930 | 546 (975, 2035) |

The full sweep is in `TEST_ORACLE.json` → `numbers.oracle_sweep`.

**Rule-blind F (IGNORE_STATUS) in the beam.**

| config | success | verified | precision |
|---|---|---|---|
| random W4K8R3E32 | 0.250 | 0.875 | 0.286 |
| random W4K32R3E32 | 0.275 | 0.958 | 0.287 |
| enum W2E12 | 0.283 | 0.850 | 0.333 |
| enum W4E24 | 0.300 | 0.900 | 0.333 |

**Descriptive oor set (ORACLE F, unguarded).** Enum W2E12 reaches 0.667 at 68 calls, enum W4E24 0.800 at 116, random W4K8R3E8 0.467 at 139, and `select_fixed` 1.000 at 21.9. Every F key lies outside the verified range, which is exactly where PREREG-3 measured T2 breaking.

**Hard matched budgets.** These use `RangeGuardF(BudgetF(ORACLE))` with caps at m × each instance's fixed-candidate calls (22.8 on average), against `select_fixed` at 1.000.

| config | 1.0× | 1.5× | 2.0× |
|---|---|---|---|
| enum W1E64 | 0.100 | 0.717 | 0.750 |
| enum W2E64 | 0.075 | 0.208 | 0.475 |
| enum W4E64 | 0.075 | 0.192 | 0.192 |
| random W2K4R2E64 | 0.142 | 0.233 | 0.283 |
| random W4K8R3E64 | 0.058 | 0.150 | 0.225 |
| random one-shot K8 | 0.058 | 0.108 | 0.125 |

With an unbinding cap, `BudgetF` reproduces the unbudgeted episodes exactly.

**Tiny model** (2-block Qwen3 with the real tokenizer, r4 LoRA saved and loaded as `t2`).
- **Canonical only.** `LedgerF` has no surface parameter, and the base class cannot be constructed. `SurfaceAuditF` refuses the canonical name and is not a `LedgerF`. `is_canonical_frame` holds on every vocabulary key of 30 primary and 5 oor instances. A swapped non-canonical renderer raises before generation.
- The texts and states equal `generation.chain('chain_one')`.
- **Batching:** 10 keys → 1 generation call and 1 `model.generate`; a mixed list → [10, 4]; 60 keys → 1 call with sub-batches [32, 28].
- `generated_tokens` is exact (120). Garbage answers give `None`; every parser path behaves as intended; the decode cap is 160.
- OFF equals the base model alone, and T2 differs from it.
- **Rule-following stub.** In the T2 arm it reproduces ORACLE's `plan_beam` episodes exactly, and the status-blind stub in the OFF arm reproduces IGNORE_STATUS's. That run had 18 predict calls, 18 generation calls and 4,178 keys, and all 6,850 prompts were canonical.
- **Audit.** Renderings equal `p3_common.messages`, and answers map back to A–D. The audit discriminates: a rule-level F reproduces ORACLE under all 7 surfaces. A trigger-level F reproduces ORACLE in the canonical frame and under the key-order and naming surfaces, but reproduces IGNORE_STATUS under `status_cancelled`, `status_pending` and `heldout_combined`.
- **Guard on a model arm.** A noisy text generator imagined states beyond the bound. 257 of 3,693 keys were refused, and no prompt with \|b\| > 98 reached generation.
- `plan_beam` and `select_fixed` run on the real tiny model. The cache refuses another arm's or an audit surface's file. `SharedLoader` is called once, and `load_registered_t2()` refuses on CPU. Malformed keys raise.
- **SimulatedF** is deterministic per key: the SIM_T2 posted-error rate was 0.078 (778 steps) and the SIM_OFF void-applied rate 0.940 (182 steps).

**Determinism.**
- The same seed gives the same sha, and a different seed a different one.
- Slots are independent, and serialisation round-trips exactly.
- `vstep` equals `env.step` on 20,000 transitions, and `env.step` equals the FDS oracle on 300.
- Both set files equal their SOURCE records and their in-memory regeneration.

## 10. GPU-call estimates (primary, guarded, `beam_kwargs`)

Calls are unique F calls per instance. **SIM_T2_P3 sizes T2 and SIM_OFF sizes OFF**; ORACLE gives the floor. Rates are 3.8 gens/s (observed, PREREG-2 ledger chain_one) and 2.7 (planning).

**Mean calls per instance, with success in brackets.**

| config | ORACLE | SIM_T2 | SIM_T2_P3 | SIM_OFF | per-instance ceiling |
|---|---|---|---|---|---|
| `select_fixed` K8 | 22.8 (1.000) | 22.9 (0.808) | 22.9 (0.717) | 24.3 (0.383) | K·d* (mean 32, max 48) |
| enum W2E12 | 62.3 (0.850) | 66.3 (0.767) | 70.4 (0.675) | 77.3 (0.158) | 104 |
| enum W4E24 | 111.5 (0.900) | 116.0 (0.800) | 118.7 (0.758) | 132.4 (0.158) | 216 |
| random one-shot K32 | 138.8 (0.342) | 140.0 (0.292) | 140.9 (0.250) | 157.5 (0.075) | 192 |
| random W4K8R3E8 | 125.9 (0.617) | 131.0 (0.533) | 132.3 (0.500) | 157.1 (0.133) | 264 |
| random W4K32R3E8 | 262.3 (0.883) | 276.2 (0.750) | 292.8 (0.717) | 404.7 (0.142) | 1056 |
| all six, with a shared per-arm cache | 445 | 482 | 511 | 709 | – |

Enum W2E12 verification precision is 1.000 (ORACLE), 0.978 (SIM_T2), 0.976 (SIM_T2_P3) and 0.271 (SIM_OFF).

**GPU-h at 3.8 gens/s**, for T2 (SIM_T2_P3) + OFF (SIM_OFF) with per-arm caches. Multiply by 1.41 for 2.7 gens/s, and add 0.15 h per turn for the A0 anchor and the model load.

| bundle | GPU-h per 100 | 120 instances (+0.15 h) |
|---|---|---|
| `select_fixed` + enum W2E12 | **1.31** | ≈ 1.7 h, one 4.5 h turn |
| + enum W4E24 | 2.20 | ≈ 2.8 h, one turn |
| all six configs | 8.92 | ≈ 10.9 h, three turns |
| one D-35 audit arm (T2 at one surface; fixed + enum W2E12; SIM_OFF as the proxy) | 0.69 per surface | ≈ 0.83 h per surface |
| oor descriptive (30 instances, T2 only, fixed + enum W2E12; SIM_OFF as the proxy) | – | ≈ 0.25 h |

**Worst case.** Enum W2E12 cannot exceed 104 calls per instance. So T2 + OFF + `select_fixed` + enum W2E12 is at most about 2.0 GPU-h per 100 instances at 3.8 gens/s (2.2 h if every fixed candidate set hit 48).

## 11. Findings that bear on the design

1. **At exactly matched calls, fixed-candidate selection wins, because its candidate set contains the answer.**
   - With ORACLE F, `select_fixed` reaches 1.000 at 22.8 calls. Every model-free beam hard-capped at those calls reaches 0.06–0.14.
   - At 1.5× the calls the best beam reaches 0.717 (enum W1). At 2.0× it reaches 0.750, which is enum W1's ceiling under the registered `max_depth`.
   - With a T2-like noisy F, fixed selection (0.72–0.81 at 23 calls) still beats enum W2E12 (0.68–0.77 at 66–70 calls).
   - So "beam versus fixed at matched calls" measures the candidate set, not F. Q1 proposes contrasts that measure F.
2. **The enumerating proposer dominates the random one.**
   - Enum W2E12 reaches 0.850 at 62 calls, against 0.617 at 126 for random W4K8R3E8.
   - Enum W8E48 reaches 0.967 at 192 calls, against 0.975 at 546 for random W8K64R3E32.
3. **64% of primary instances (77/120) are solved by greedy descent on the true hook.**
   - The planning signal lives in the 43 needs-lookahead instances. ORACLE success there is 0.28 (enum W1E6), 0.58 (W2E12), 0.72 (W4E24) and 0.91 (W8E48).
4. **The problems punish a rule-blind F.**
   - IGNORE_STATUS drops `select_fixed` to 0.567 and verifies falsely in the beam, with precision 0.29–0.33.
   - Simulated OFF reaches 0.08–0.38 with precision 0.22–0.39, against 0.25–0.81 and 0.91–1.00 for the T2-like presets.
5. **Range safety is now a property of the problems plus one planner keyword, not of a filter.**
   - With the closure and `beam_kwargs`, no exact-F search touches a state outside T2's trained surface.
   - The guard remains only for a model F's own arithmetic errors.
   - The cost is that the primary covers a subset of PREREG-2's event distribution: scale factors are −1 or 0 only, and balances are moderate (median \|b\| 16). A planning claim is scoped to that subset.
6. **The D-35 audit is wired (`SurfaceAuditF`) and discriminates rule-level from trigger-level F.** PREREG-3 predicts that T2 s0 is trigger-level on status words, key order and word names.

## 12. Open questions for the joint preregistration

1. **Primary outcome and contrasts.** RCWM proposes the following, all on the primary confirmation set under `guarded()` + `beam_kwargs`:
   - (a) T2 against OFF, paired, in one pre-set beam configuration; outcome = exact goal success.
   - (b) T2 against the ORACLE ceiling at the same budget, as success_T2 / success_ORACLE.
   - (c) The same two contrasts on the needs-lookahead stratum, as a co-primary or the primary.
   - (d) As secondaries: `select_fixed` (T2 / OFF / ORACLE / IGNORE_STATUS) as the PREREG-2 replication, and beam against one-shot at matched calls.
   - We advise against "beam vs fixed candidates at matched calls" as a primary (finding 1).
2. **Budget.** Proposal: enum W2E12 as the primary (ORACLE 0.850, needs-lookahead 0.581; at most 104 calls per instance), with enum W4E24 as a secondary. One registered budget, or a pre-set curve?
3. **Closure and depth registration.**
   - Register `D_CLOSE = 8`, \|b\| ≤ 98, and `max_depth = D_CLOSE − sim_len` through `beam_kwargs`?
   - A larger `D_CLOSE` would allow deeper searches but shrink the distribution further.
   - Accept the distribution consequence (scale −1 / 0 only)? The alternative is a separate scale-focused primary with smaller closure depth, which would change the planning horizon.
4. **Matched-call rule.** The cost unit is distinct keys per instance, cache-independent; refused keys are free.
   - Hard caps (`BudgetF`), or matching on the mean?
   - Caps set from ORACLE, or from each arm's realised calls? Noisy arms cost more (§10).
5. **Proposer.** `EnumerateProposer` (deterministic, 0 tokens, a hard call ceiling) as the default? Or should BWM build an LM proposer for the ledger? None exists yet. BWM receives only blind instances either way.
6. **Instance set.**
   - Size: power for T2 − OFF and T2/ORACLE, with an expected needs-lookahead share of about 36%.
   - Should the confirmation build enrich needs-lookahead instances? That needs a greedy filter, which changes the draws.
   - Draw a fresh confirmation seed at the freeze, with generator v2. Dev seeds 20261141 (v1), 20261142 and 20261143 are spent.
   - Rebuild the freshness union at the freeze: the t2bp/t2br data is still being rewritten (the t2br files were absent at 12:24 and back by 12:27).
7. **D-35 surface audit and claim scope.**
   - D-35 says joint item 3 confirmations use held-out-surface primaries. PREREG-3 predicts T2 s0 fails those (trigger-level), and R16 says frozen planner results that always render the training format are unaffected.
   - Option (i): register T2 s0's planning claim as trigger-level / training-surface, with a canonical-frame primary and `SurfaceAuditF` arms reported only.
   - Option (ii): defer the confirmation to a surface-diversified T2b adapter (D-36), with a held-out-surface primary.
   - Which audit surfaces? RCWM suggests `heldout_combined` + `status_cancelled`, at about 0.83 GPU-h each for 120 instances.
8. **Out-of-range set.** Keep a small T2-only descriptive oor arm (≈ 0.25 GPU-h for 30 instances), or drop it from the confirmation run?
9. **Guard registration.** Confirm the guard: bound 98, trained amounts, posted/void, applied identically to every arm, refused key = `None` (the candidate stops), and refused keys reported per arm. It is inert for an exact F.
10. **`None` handling in `select_fixed`.** Rank last (current), or use the last parsed state? Inside the beam, `None` ends a candidate, by BWM's interface.
11. **Batch invariance.** Measure at the GPU anchor (batch = 1 against batched, on about 500 keys). Register "F = the first computed (cached) answer", with per-arm and per-surface caches.
12. **Instrument identity.** Rerun PREREG-2's A0 anchor before any planning generation, and on every resumed turn.
13. **What is submitted, and what counts.** The submission is the verified plan, or the best node's prefix, and success means exact goal success. Should an unverified submission count? Secondaries: precision, optimal-length rate, and the exposure log.
14. **Statistics.** Paired bootstrap over instances; thresholds (for example T2 − OFF ≥ +0.20, T2/ORACLE ≥ 0.8); Holm across the primaries.
15. **Runner and roles.**
    - `scripts/lbeam_run.py` is not yet written. It would run T2 and OFF (and audit arms) in one process under `rcwm-t1:lbeam`, with stage and resume logic like `p3_run`'s.
    - Does BWM run `plan_beam` itself, importing `LedgerF` read-only, or receive cached F answers?
    - Either way, the fixed baseline and scoring stay on the RCWM side, because they hold the answer key.
16. **Decode cap.** Keep PREREG-2's 160 new tokens?
