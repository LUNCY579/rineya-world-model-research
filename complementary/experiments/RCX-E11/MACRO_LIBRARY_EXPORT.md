# E11 macro libraries: export for BWM's B5 candidate (macro arm in `plan_beam`)

Requested by BWM (X-07, 2026-10-11) after R-09. This is read-only material. Nothing here changes the E11 results, and
nothing in BWM's tree is touched.

## 1. Files and shas (branch `research/rineya-complementary-intelligence-v1`, commit `bddfcda` or later)

| File | sha256 | Content |
|---|---|---|
| `complementary/runs/RCX-E11/main/a/libraries.json` | `b61b4a05682c9ab1c40a643a84463f926d0df7ac457b7a18df07c611df13bd79` | Libraries M, R, S, G (`libraries`); per-step-key validation records (`validation`, `validation_summary`); selection logs; mining statistics; teacher custody (`custody`); M/S/R overlap checks (`overlap`); seeds |
| `complementary/rcx/bw/macros.py` | `e644f6b15df2f260bc9434b61cf9750653ab757c9b57c42352c5e7838c930f59` | `Macro` (load, bindings, ground, replay, symbolic effect), `validate`, and the macro search used in E11 |
| `complementary/rcx/bw/search.py` | `10a1b499446e732ff860f479f30639270296a8ac2443abb4024ed50def9b758f` | Base GBFS/beam (macro search equals these with an empty library) |
| `complementary/rcx/bw/space.py` | `29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4` | State spaces for validation (numpy/stdlib only) |
| `complementary/rcx/bw/bwm_import.py` | `84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e` | Read-only, sha-pinned loader of your `bwmn/blocks.py` (b1e28ed5…) |
| `complementary/runs/RCX-E11/main/REPORT_E11.md` | committed in `bddfcda` | Results (R-09) |

## 2. Library contents (K = 8 each; G has 2)

| Library | Role in E11 | Source |
|---|---|---|
| **M** | Treatment | Mined (L = 2–4, ≤ 4 variables) from the 211 F-verified B3 plans, ranked by support |
| **R** | Matched control | Random sound macros matched to M by length L and goal size G, and by variable count where possible |
| **S** | Matched control | Mined and ranked the same way from the same plans with each plan's actions shuffled. Disjoint from M by step signature. |
| **G** | Descriptive | Two goal-gated single moves (`pick-up ?0; stack ?0 ?1 \| on(?0,?1)`, `unstack ?0 ?1; stack ?0 ?2 \| on(?0,?2)`) |

`libraries.json["libraries"]["M"]` is a list of `Macro.to_json()` records:
- `steps`: for example `[["pick-up",0],["stack",0,1],["pick-up",2],["stack",2,0]]`. Variables are numbered by first
  appearance.
- `goal_on`: for example `[[0,1],[2,0]]`, meaning `on(?0,?1)` and `on(?2,?0)` must be **goal** facts.
- `pre`, `add`, `dele`: from STRIPS regression over your `blocks.py` semantics.
- `L`, `nvars`, `key`, `meta`: `meta` holds support, tier and rank.

No-op-containing candidates are excluded from M, R and S (`exclude_noop_macros: true`).

## 3. Semantics (`macros.py` module docstring)

- **Initiation.** A binding β is **injective** (distinct variables bind distinct blocks). The macro applies when
  pre[β] ⊆ s **and** goal_on[β] ⊆ goal. `Macro.bindings(StateIndex(s), goal_index(goal))` returns the sorted
  bindings.
- **Policy.** Execute the ground steps `Macro.ground(β)` open-loop.
- **Termination.** After the last step. The search stops mid-macro if an intermediate state already satisfies the goal.
- **Effect.** For consistent state s, executing the steps gives (s − dele[β]) ∪ add[β], whenever pre[β] ⊆ s.
- **Validation record** (`validation[step_key]`). Each macro was checked exhaustively against `B.applicable`/`B.step`
  on every valid state of n_check blocks (n_check = min(V + 2, 6), at least 3). The checks:
  - **sound:** the replay reaches the symbolic effect;
  - **complete:** pre is exactly the executability condition;
  - **non-vacuous** and **non-identity**.

  Every macro in M, R, S and G is `accepted: true`. Pattern totals: 706 patterns checked; 22 accepted, 505
  inconsistent, 179 vacuous.
- **Loading.** Use `Macro.from_json(d)`. It re-derives pre/add/dele from `steps` and refuses a record whose stored
  effects differ, so a corrupted file fails closed.

## 4. Notes for an LM-F beam arm (suggestions only; the B5 design is BWM's)

1. **Inconsistent imagined states.** In `plan_beam` the state is F-imagined, so it may be inconsistent. Binding
   enumeration assumes a consistent state; with an inconsistent one it can yield nonsensical bindings. Reusing the
   B4 hook's `is_consistent` / `sanitise` check (`rcx/bw/b4_hook.py`) before enumerating bindings keeps the arm
   comparable with B4's c+pen.
2. **Expansion.** Expanding a macro as k primitive steps, each predicted by F (as you proposed), makes F-query cost
   explicit. E11's compute-matched view counted `h_calls` and `prim_steps`. The B5 analogue is F-queries per success.
3. **Controls.** R and S are matched to M by length and goal size. With misplaced_blocks in E11, S helped as much as M
   (90 vs 85 at 200). The specificity of M held only with the A3 value. A B5 arm under the `unmet` key may therefore
   see the same "any library helps" pattern. Pre-registering both controls would separate generic macro-ness from
   mined structure.
4. **Instance hygiene.** E11a's 120 and E11b's 581 instances are already in your B4 exclusions. The macros were mined
   from the B3 `solutions.jsonl` problems (distill-240), which are already in your recipe.
