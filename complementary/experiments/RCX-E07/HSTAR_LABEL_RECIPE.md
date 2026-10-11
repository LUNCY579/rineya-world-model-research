# Exact h\* labelling recipe (for BWM's B4 arm b-h\*; RCX E07/E07b definition)

This is exported at BWM's request (BWM D-70, B4 §8 item 6). BWM generates its **own** instances in its own seed range
under the agreed exclusions, so no RCX-generated instance enters BWM's training. This file gives the exact label
definition and code, so that both programmes share one definition of h\*.

## 1. Definition

h\*(s, g) is the length of the shortest action sequence from state `s` to any state satisfying goal `g`, under PlanBench
4-op Blocksworld semantics (`bwmn/blocks.py`: pick-up, put-down, stack, unstack; inapplicable = illegal). Goals are
frozensets of `("on", x, y)` facts. "Satisfying" means `g <= s` (`B.goal_met`).

h\* is exact (not a bound). Blocksworld has no dead ends, so h\* is finite for every consistent goal.

## 2. Code (RCX branch `research/rineya-complementary-intelligence-v1`)

`complementary/rcx/bw/space.py` is **self-contained**: it uses only `numpy`, `heapq`, `math` and `functools`, and does
not import BWM code. Pin it by sha256: `space.py` is sha256 `29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4`, unchanged
since R-04. Record your copy's sha in D-70.

| Function | Use |
|---|---|
| `GoalLabels(objs, goal)` | n ≤ 8. Builds the full state space for `len(objs)` blocks and runs a backward BFS from all goal-satisfying states. `.h(s)` gives h\* for **every** state; `.opt_actions(s)` gives the optimal action set; `.hstar_array()` gives all states. Cost: n = 7 builds in about 0.25 s plus 9 ms per goal; n = 8 in about 2.7 s plus 0.27 s per goal (about 180 MB). |
| `astar_hstar(objs, s, goal, max_expansions=300_000)` | Any n. Exact h\* for one state by A\* with the admissible misplaced bound (checked admissible on 121,688 states). Returns `(h*, expansions)`, or `(None, expansions)` if over budget; such states must be **recorded as missing, never dropped silently**. n = 12 start states take 0.1–0.9 s. |
| `astar_optimal(objs, s, goal, max_expansions=300_000)` | An optimal plan, when one is needed. |

**State format.** `s` is a frozenset of BWM fact tuples: `("on", x, y)`, `("ontable", x)`, `("clear", x)`,
`("holding", x)`, `("handempty",)`. `objs` is the list of block names. States must be consistent; F-imagined states are
**not** labelled, because the training labels come from the simulator.

**Correctness evidence** (RCX tests, `tests/test_bw.py`):
- h\* equals `len(B.bfs_plan)` on 200 random instances with n = 3–6;
- A\* agrees with the full-space h\*;
- the optimal action sets match brute force;
- the transition function equals `B.step` exhaustively for n = 3–5.

## 3. Sampling regime used by RCX (E07)

| Item | RCX E07 setting |
|---|---|
| Instance generator | `B.generate(rng, n, full_goal_frac=0.5)`, mixing full-configuration and partial goals, as BWM recommends |
| Train sizes | n = 4, 5, 6, 7 |
| States per instance | all states for n ≤ 6; a uniform sample of 10,000 states for n = 7 (E07; 3,000 in the pre-review design) |
| Split unit | the goal class (renaming-canonical `goal_key`), never single states, because instances of one goal class share every state's label |
| Exclusions | BWM's recipe plus selection sets, applied to instances and also to rows (the (state, goal) pair read as an instance) |
| Target | h\* as a float. RCX trains with Huber loss. |

For BWM's hard-style B4 set (full goals, 6–15 blocks), we suggest including full-configuration goals at n = 6–8, where
full-space labelling is cheap. A\* labels at larger n are exact, but cost more per state.

## 4. Minimal example (runs in about 1 s on one thread)

```bash
python -B experiments/RCX-E07/hstar_recipe_example.py
```

The script prints h\* for a 5-block instance from the full-space BFS and from A\*, and asserts they are equal.
