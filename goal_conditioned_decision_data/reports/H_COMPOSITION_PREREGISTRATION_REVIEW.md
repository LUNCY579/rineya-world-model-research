# H composition preregistration review

Reviewed 12 October 2026. **PASS for the bounded, inactive mechanical DEV experiment described below.** This verdict does not approve a production profile, publication, a model-performance claim, or a pilot outcome. The actual proposal freeze, attempt marker and result were absent at review. The full programme/source freeze and independent postmeasurement result review remain separate gates.

The reviewer owns only `tests/test_h_composition_review.py` and this report. The prototype and specification were changed by the root author in response to review findings. No approved production source/configuration, accepted record, runtime, automation, remote object or sealed label was changed or read by this reviewer. The earlier operation/composition reports were authenticated through bounded archive reads into RAM.

## Reviewed snapshot and evidence

| File | SHA-256 |
| --- | --- |
| `generators/h_composition_experiment.py` | `18751b15853905d79ed644c6ced6ce4e21a46300a189defc54465b564d3621f0` |
| `configs/h_composition_dev_experiment.json` | `dc2a17a16860c5087ca3a8a583bd7fd3a3c6abfb7c8b08ebc697252102ce7ea9` |
| `tests/test_h_composition_review.py` | `e9d1f9de1e0d75b387cacca91511a04af15f370d6cbef842cdff97cac51bca7f` |
| Author tests `tests/test_h_composition_experiment.py` | `78c514c7d6b7e4eda8c677881d9e43b8d2cdccfa8a540939408e006c534eda34` |

The combined 24 tests passed in 0.319 seconds: ten independent tests and fourteen author tests. Independent semantic executions use developmental seed 113/index 17 only. Custody challenges use synthetic temporary receipts and cached developmental inputs; selected-seed constructors, either oracle and baseline generation do not execute during those replay challenges. Author tests also use developmental witnesses 111/index 0 and 112/index 31, and temporary pure-construction freeze fixtures. None is the actual confirmation attempt.

The reviewed v1/v2 generators, environment semantics, both oracles, auditor and v1/v2 registry/config files are byte-for-byte identical to approved production. The immutable historical coverage receipt is `OPERATION_COMPOSITION_COVERAGE_20261011.json`, SHA-256 `3fdc62d5b434b335257da6910007c55c4bef3f6b71985246765e5bcbac90a374`; its Markdown summary is SHA-256 `5758fdab28895bebb714afd2cfeca92c2330bf0f11b1958348caf33f949f51d5`. Its measured scope is 1,936 nonsealed v1 records, not the subsequently enlarged corpus. It records a goal/rule/constraint-blind H selector optimal on 264/264 rows, and 66 register H rows supplying add/noop only, one-step plans, one active valid register and one goal atom. V2 code preserves those H dynamics.

## Frozen hypothesis and decision thresholds

The proposed confirmation uses exactly 32 DEV seed identities, 430000–430031, with index 0–31. No parent rejection, replacement seed, adaptive proposal search, model training or canonical publication is permitted. There are 96 goal-arm cases, 128 constraint-arm cases, 32 current-v2 H/register comparison cases and 544 individual goal-atom ablations: **800 evaluated cases**. One worker has a 300-second wall budget and a one-MiB research-result limit. Baseline generation additionally constructs complete v2 groups, including unused family records and their oracle work; the result separately discloses that expansion. The 800-case count does not claim to include every internally generated baseline row.

PASS requires all 32 complete parents; zero exact dual-oracle/label or reproduction mismatches; fixed candidate/state/rule inventories under the stated interventions; all 96 full goal-arm winners executing three operations; all 32 conjunction cases preserving their exact two-way tie; and all 544 atom ablations producing a nonempty optimum disjoint from the original best set. Any contradiction, illegal supplied action, redundant atom, interrupted attempt, changed source/spec/input commitment, task/time/result-bound failure or incomplete evidence is failure evidence. There is no reduced-denominator success rule.

All prototype candidates and their IDs/order are fixed within a bundle before selecting the goal or constraint variant. Goal triplets change only `goal`; constraint quartets change only `constraints`. Opaque IDs/order exclude goals, constraints, expected labels and record metadata. The prototype uses bare validated register tasks and has no allowlisted canonical generation profile or exporter admission route.

## Goal arm: operation order and simultaneous goals

Let `x=2+i%16`, `y=21+i`, `m=2+i//16`, with initial registers `(a,b,c)=(x,y,0)`. The equal-cost, three-operation plans have endpoints:

- P: swap a/b; multiply a by m; copy a to c → `(my,x,my)`.
- Q: multiply a by m; swap a/b; copy a to c → `(y,mx,y)`.
- R: copy a to c; swap a/b; multiply a by m → `(my,x,x)`.

P/Q reverse the first two operations; R changes the timing of the copy. The opcode multiset stays fixed. The endpoints are distinct, and every atom of every full goal differs from the initial value. The corrected `y=21+i` avoids the former last-index coincidence `mx=y`. Developmental action-deletion tests independently establish that removing any step from the selected three-step plan prevents that full goal.

Set costs one in this arm. For each goal and omitted atom, a two-set candidate establishes the other two goal values. Nine proposed partial plans deduplicate to eight: P/R omit-c produce the same a/b assignments. Deduplication precedes opaque IDs, leaving eleven supplied candidates. No partial plan can achieve a full three-atom goal because its untouched register differs from every goal's corresponding value. After its atom is omitted, a partial plan has utility `(1,2,-2)` and strictly beats the original winner's `(1,2,-3)`. This corrects the earlier single-goal-atom shortcut; merely counting three goal atoms would not have established their necessity.

The full triplet has distinct singleton best sets P/Q/R. Any selector fixed while ignoring the goal can be optimal on at most one of each three cases: at most 32/96. The specified first/shortest/longest/lowest-ID/add heuristics are diagnostics under this bound, not model evaluations or an enumeration of all possible shortcuts.

## Constraint arm: binding conjunction, full rankings and ties

Let initial `(a,b,c)=(2+i%8,3+i//8,6+i)`, `A=2c-2`, `B=2c`; the goal requires `a=A,b=B`. Set costs four here, while add/multiply/copy/swap cost one and noop costs zero. Rules are fixed across the four siblings; the cost difference between arms is declared.

P overshoots a to B, copies a to b, then subtracts two: cost 3. R doubles c, copies c to b, then sets a to A: cost 6. Q and its exact reversal set a/b independently: cost 8. F is P's two-step prefix. Additional candidates are noop, a one-add a-only plan, R's two-step b-only prefix, and a one-set b-only plan. They achieve only one full goal atom, so the full-goal winners remain unchanged.

Aliases below are audit metadata, not model input. Every displayed tier is exact, including infeasible candidates; feasibility outranks goal count and cost.

| Constraints | Ordered ranking tiers |
| --- | --- |
| none | P; R; {Q,Q_reverse}; partial_a; {F,partial_b_maximum}; partial_b_conjunction; N |
| maximum a=A | R; {Q,Q_reverse}; partial_a; partial_b_maximum; partial_b_conjunction; N; P; F |
| protect c | P; {Q,Q_reverse}; partial_a; F; partial_b_conjunction; N; R; partial_b_maximum |
| both | {Q,Q_reverse}; partial_a; partial_b_conjunction; N; P; R; {F,partial_b_maximum} |

Initial constraints are satisfied. P/F violate the maximum at steps 1 and 2 even though P restores a=A at its terminal state. R and its prefix change protected c; successful legal execution does not make them feasible. Under both constraints, removing the maximum yields P and removing protection yields R. Each conjunct is therefore indispensable to the joint best set. None/protection sharing P is an intended negative winner-change control. A fixed constraint-blind P selector and the specified terminal-only diagnostic are correct on 64/128 full cases; there is no claimed one-quarter bound.

Dropping goal a gives {F,R-prefix} without constraints, R-prefix under the maximum, F under protection, and the b-only set under both. The unconstrained case is an exact cost-two tie. Dropping goal b gives the one-add a-only candidate in every constraint variant. All these optima exclude the old full-goal best set. Membership and complete-set recovery must remain separate metrics: selecting only Q is not exact recovery of {Q,Q_reverse}.

No cost-cap success-flip hypothesis is made. With deterministic fixed candidate costs, a max-cost-only change cannot remove the cheapest full-goal winner while retaining a more expensive feasible full-goal alternative.

## Custody corrections and chronology

Before actual freeze, independent review identified insufficient completed-PASS receipt validation, an unbound STARTED body, missing per-atom receipts and an unbounded completed-result read. The author corrected them. Current controls have bounded nonsymlink JSON reads, duplicate-key/nonfinite refusal, exact STARTED version/freeze/source/time fields, a whole-result integrity digest, complete parent/task inventories, frozen input commitments, expected best sets/plan lengths, candidate summaries, 544 per-atom hash/best-set receipts and baseline commitments. Pure reconstruction verifies replay custody; it does not rerun either oracle or the baseline. The independent malformed-PASS, rehashed-wrong-STARTED and oversize-result refusal tests pass with recomputed result integrity fields, so they do not rely merely on a missing self-hash.

Input commitments are constructed without oracle/baseline execution. Source/configuration and test files must stop changing before the actual freeze; report Markdown is excluded from executable source hashes. An immutable attempt marker precedes confirmation execution. A prior interrupted attempt cannot be silently repeated, and a completed result is observed without re-execution. The CLI preserves bounded FAIL evidence on attempt failure. The result's self-digest and structural checks are consistency checks, not independent authenticity. The final independent review must pin the exact freeze/result file hashes before any later promotion or claim of a measured PASS.

## Limits that survive a successful pilot

Numeric worlds depend on index; seed changes IDs and candidate order only. Developmental indices 0/31/17 overlap confirmation dynamics. These are 32 fixed parameter settings and seed identities, not an unseen semantic holdout, random sample or IID evidence. Rows, ablations and candidate views share parents.

This tests three-register, fully observed, reject-overflow tasks with literal a/b/c, supplied candidates and a particular maximum/protection conjunction. It exercises all six operations across the two arms, but noop is a control and X's conjunction winners use two set operations. It does not establish reserved vocabulary transfer, ledger-H coverage, uncertainty/overflow generalization, arbitrary operation combinations, global action-space optimality, TRAIN/OOD generalization, real-world performance or learned reasoning. Goal-aware/formula-aware shortcuts can remain. Baseline/prototype comparisons change goal cardinality, candidates, operations and costs; only the within-bundle interventions isolate the stated mechanisms. Any future v3 canonical profile, broader split design or production activation needs its own reviewed compatibility and admission work; existing v1/v2 records and the sealed instrument remain immutable.
