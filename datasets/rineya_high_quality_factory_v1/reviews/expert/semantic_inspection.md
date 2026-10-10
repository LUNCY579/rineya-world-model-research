# Independent blind expert review: pilot v1

Reviewer execution: `/root/expert_reviewer`. Role: semantic Expert Reviewer.

I reviewed all 240 canonical candidate rows (RCM 120, RGM 60, RVM 60) and all six blind calibration rows. Every row was independently recomputed from the JSON in its student-visible prompt, with public `schemas/SEMANTICS.md` as the contract. The prompt payload was compared with the canonical semantic payload. The verifier was authored in this reviewer execution; it imports no generator, reference implementation or formal oracle. I did not read calibration expectations, adversarial receipts, adjudication records or another reviewer’s verdicts. This is an actual separate agent execution, with the same underlying model family as other agents; correlated model errors remain possible.

`independent_review.py` uses a strict eager AST interpreter and separate state updates, exhaustive breadth-first searches over graph `(node,fuel)` states and legal block on-maps, and an independent critic ranking calculation. Search has no timeout-to-unsolvable conversion. Each row's receipt contains the candidate record hash, independently calculated evidence, eight rubric scores, defects, actionable issues and a decision. The verifier checks target value and JSON type equality, including trace steps and ordered critic evaluations. The candidate target is used only for comparison after the result has been computed.

Programmatic recomputation is supplemented by substantive semantic inspection by this agent. I inspected the 30 RCM parent forward tasks, both twins at the diagnostic cases below, all 30 RGM parent summaries and the complete tasks/targets spanning each plan mode, and complete critic worlds/candidate lists for all six modes and their development variants. I read every distinct visible contract prefix. Representative reverse JSON renderings were explicitly read for `RGM-001-pilot-v1-p0015-base-reverse` and `RVM-001-pilot-v1-p0004-base-reverse`; per-record prompt parsing checks all remaining renderings. This scope is not a claim that a human manually solved every rendering separately.

## Causal mechanisms inspected in detail

| Mechanism | Exact representative IDs | Independent semantic finding |
|---|---|---|
| Literal status mapping and rule inversion | `RCM-001-pilot-v1-p0000-twin0-forward`, `RCM-001-pilot-v1-p0000-twin1-forward`, `RCM-001-pilot-v1-p0001-twin0-forward` | `posted` is an exact literal, not a normative status. The base p0000 adds 8 to x=15, producing 23; its inverted rule leaves 15. `void` with `ne` is false. |
| Threshold and compound gates | `RCM-001-pilot-v1-p0002-twin0-forward`, `RCM-001-pilot-v1-p0003-twin0-forward`, `RCM-001-pilot-v1-p0006-twin0-forward`, `RCM-001-pilot-v1-p0015-twin0-forward` | `amount >= 9` includes equality; x>37 at x=37 is false. In p0003, the status/amount conjunction is true for the first event and false for the second; swap is simultaneous and the final map is x27/y32. |
| Same answer under different predicates | `RCM-001-pilot-v1-p0006-twin0-reverse`, `RCM-001-pilot-v1-p0006-twin1-reverse`, `RCM-001-pilot-v1-p0024-twin0-forward`, `RCM-001-pilot-v1-p0027-twin0-forward` | False preconditions can suppress both rule twins. Final-state equality is valid but does not by itself identify the evaluated rule. Trace comparison is needed for this competency. |
| Applicable identity versus inapplicable no-op | `RCM-001-pilot-v1-p0026-twin0-reverse`, `RCM-001-pilot-v1-p0026-twin1-reverse` | Scaling x23 by 1 executes `scale` despite no state change. Its inverted-rule twin executes `identity`. Both end x23/y18; the trace correctly distinguishes applicability. |
| State-dependent multistep execution | `RCM-001-pilot-v1-p0004-twin0-forward`, `RCM-001-pilot-v1-p0017-twin0-forward`, `RCM-001-pilot-v1-p0022-twin0-forward`, `RCM-001-pilot-v1-p0029-twin0-forward` | p0022's swap raises y to135, causing the following `state.y<2` branch to become false. p0029 scales y3400 by501 to1703400, swaps the fields, then suppresses the final add through the changed x predicate. |
| Signed balances and unfamiliar vocabulary | `RCM-001-pilot-v1-p0019-twin1-forward`, `RCM-001-pilot-v1-p0019-twin1-reverse` | The inverted `ne quom` gate accepts the two `quom` events. Swap gives x400/y1000 and transfer702 gives x-302/y1702. There is no balance precondition, so the negative result is legal. |

All causal targets match. The pilot contains only two negative final-state renderings, both of the same semantic row; negative-balance breadth is therefore small. In the 24 initial training families, top-level `not` occurs in 28 forward rows, 19 of which execute no event. That visible syntactic imbalance is a modest shortcut risk. Copying an unchanged input state also succeeds on some answer-only controls; corresponding traces should be evaluated jointly.

## Graph and block plan modes inspected in detail

Each ID below has the full prefix `RGM-001-pilot-v1-` and suffix `-base-forward`.

| Plan mode | Exact middle ID | Semantic finding |
|---|---|---|
| Lexicographic tie | `p0015` | Both `Aster,Dune` and `Fjord,Dune` take two moves. The lexicographic winner is `Aster,Dune`, though it uses more fuel. This checks the declared objective precisely. `p0000` is a single chain and its tie skill tag is inaccurate. |
| Fuel-constrained route | `p0001` | The direct cost7 edge is infeasible with fuel6. The three-move chain costs6 and is the fresh optimum. The submitted plan stops at its first direct move. |
| Assemble stack | `p0002` | Place dahlia on cobalt before amber on dahlia; otherwise the destination is blocked. The two-move target preserves all other supports. |
| Unreachable goal | `p0003` | The chain costs11 but initial fuel is10; the direct edge also costs11. The fourth submitted move is illegal, and the fresh optimum is null. |
| Reverse stack with invalid tail | `p0004` | The four legal moves satisfy the goal. A repeated no-change fifth move is illegal, so reaches_goal is false although the final on-map satisfies the supports. |
| Cycle exhaustion | `p0005`, `p0020` | Fuel decreases on every positive-cost cycle. Stop at the first unaffordable action; do not evaluate the long submitted tail. |
| Missing edge | `p0006`, `p0021` | A self-destination move has no edge and fails immediately; later legal-looking destinations do not rescue the plan. |
| Blocked source | `p0007`, `p0022` | Moving a support while another block is on it fails. The fresh plan clears it first and then moves it. |
| Valid detour | `p0008`, `p0023` | Returning to start consumes fuel and is valid, but has not achieved the terminal goal. A fresh optimum is computed from initial fuel rather than the detour's fuel. |
| Clear partial goal | `p0009`, `p0024` | Repeating the first TABLE move is illegal because destination equals current support. The fresh optimum instead clears each dependency and then places the goal block. |
| Initially satisfied goal | `p0010`, `p0025` | Empty submitted and fresh plans are valid and goal-reaching. Irrelevant remaining edges need no traversal. |
| Full chain | `p0011`, `p0026` | Every step depends on the previous node and fuel; the complete chain reaches the goal. |
| Legal non-goal block detour | `p0012`, `p0027` | A clear distractor block can move onto a block and return to TABLE. This is valid but does not accomplish the separate partial goal. |
| Partial chain | `p0013`, `p0028` | The one-step submitted prefix is valid but not goal-reaching; shortest_plan supplies the complete chain. |
| Initially satisfied partial block goal | `p0014`, `p0029` | Initial supports satisfy the partial goal, so the fresh optimum is empty. The legal submitted move removes that goal support, so reaches_goal becomes false. |

The block legality and goal maps are well-formed. All graph/block targets match, including fresh plans, ties and stop positions. The visible contract makes fuel feasibility separate from the move-count objective. Node names and renderer order do not change semantic equality.

## Critic modes inspected in detail

Each ID below has the full prefix `RVM-001-pilot-v1-` and suffix `-base-forward`.

| Mode | Exact middle IDs | Semantic finding |
|---|---|---|
| Equal-length good/bad claims | `p0000`, `p0018` | Identical legal plans with different fuel claims separate validity from claim correctness. In p0000, c22428 legally reaches Dune but claims fuel9 instead of7: progress2/usefulness2 remain true, first_error is `claim`, and it ranks below valid correctly claimed candidates. |
| Fuel precondition | `p0001`, `p0019` | In p0001, the direct goal edge costs10 against fuel9. A legal return-to-start detour leaves fuel5 and loses reachability; remaining_moves/progress become null while usefulness is0. The seven-step p0019 route costs24 exactly. |
| Legal detour | `p0002`, `p0008` | A valid detour can have progress0; a longer valid goal plan is ranked below a shorter equally correct one. The objective does not prefer the higher remaining fuel route once tuple fields tie. |
| Initially unreachable | `p0003`, `p0009` | Remaining distances and fresh replacements are null. Valid candidates still rank ahead of invalid candidates according to the tuple. p0003's chain fails at zero-based step4, not at the first earlier legal move. |
| Initial goal | `p0004`, `p0028` | The empty plan has usefulness2 and progress0. Leaving the goal gives progress-1 and usefulness0. Returning legally reaches it again but has a longer submitted plan. Empty-plan fuel overclaims leave goal achievement unchanged but set first_error to `claim`. |
| Fresh replacement counterexample | `p0005`, `p0029` | Every candidate receives the same lexicographic minimal fresh replacement from the initial world. In p0005 that is `Aster,Fjord` even though `Ember,Fjord` uses less fuel. It is not an edit-distance patch to the submitted candidate. |

Candidate lists are shuffled and IDs have no stated quality meaning. Exact duplicates create required tie groups. Invalid plans may correctly claim the stopped state; those claims do not turn invalidity into validity. A wrong claim on a legal goal trajectory does not change its progress/usefulness. All evaluations and rankings match the independent calculation.

Recurring self-node failures, repeated intermediate-node failures and positive fuel-claim offsets make the corruption set narrow. Matched equal-length claims and longer valid/shorter invalid plans help, but this pilot alone does not show robust criticism of arbitrary corruption producers or generic successor-feature learning.

## Difficulty and decisions

At this initial candidate snapshot, targets are correct for every pilot row. There are 192 ACCEPT decisions and 48 REVISE decisions. Every REVISE concerns the level-7 difficulty claim, scored as a medium quality defect rather than a high correctness defect:

- RCM's six development families share predicate tree shapes and both renderers with training; held-out magnitudes/vocabulary do not satisfy the curriculum's crossed predicate-tree/magnitude/renderer requirement.
- RGM's six development families increase object counts but reuse graph/block modes and both renderers; object range alone does not demonstrate the declared motif/renderer holdout.
- RVM's six development families reuse all six plan modes, the objective and corruption signatures; larger worlds do not demonstrate held-out corruption producers or goal functions.

Relabel structurally and retain exposed `dev_challenge` status: causal rows generally level4, relational rows level4 except initially satisfied partial-goal rows around level3, critic rows level5 or4 depending on constraints. The curricula already distinguish broad research aspirations from these bounded tasks. Generalization/diversity scores are moderate, not perfect. Low issues may remain on accepted rows as documented limitations.

The six blind calibration receipts were fixed before any other reviewer verdict was received: 00,02,04 ACCEPT; 01,03,05 REVISE. Their independent final states respectively require x23 rather than24, x27 rather than28, and x14 rather than15. The traces already contained the correct states in all three faulty rows. These are high target-correctness defects with direct fixes. `calibration_initial.jsonl` is immutable in the verifier.

`pilot_initial.jsonl` preserves the first programmatic checkpoint, before completion of RGM/RVM's detailed manual difficulty inspection. The final independent initial expert verdicts are in `pilot-v1.jsonl`. This reviewer does not approve a root release. Changed candidates require matching-hash review receipts.
