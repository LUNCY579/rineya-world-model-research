# Planning gap V2: bounded prospective correction

Actual author: `/root/research_architect`, research-design role only. The new public interface is `schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md`. Preserve `RGM-NEXT-GAP-PROSPECTIVE.md`, both original planning schema documents, SEMANTICS, V1 source/candidate/reviews and earlier approved releases. This is a revised prospective 48-family pilot, not additional family growth on top of V1.

## Measurements from permitted frozen public inputs

The input-only V1 view is `reviews/input_only/planning-gap-v1.jsonl`, SHA256 `2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654`; prior approved public inputs are `research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl`, SHA256 `896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d`. These omit targets and judgments. The new actual audit source/report are `research/research_gap_analysis/PLANNING_GAP_V2_PUBLIC_INPUT_AUDIT.py` and `.json`. Replay/finite graph search was derived directly from the public contract; no stored answer comparison or production approval was performed.

The 384 V1 rendered inputs collapse to 192 semantic inputs / 48 lineage families. All six A1 fresh fuel-infeasible inputs are train; finite fuel-lengthened A1 has four train and both dev families. All twelve C1 first failures are source-not-clear; first-error positions 0/1/2 each occur four times. In this particular public view all twelve C1 failures independently pass the other five guards and expose source-clear omission. Among all 96 block semantic inputs, independently executed no-op-allowed and destination-clear-ignored replay mutants each differ from strict replay zero times; source-clear omission differs twelve times. A reason count in general is not selective ablation evidence: both the full guard vector and actual mutated transition must be verified.

Trying every adjacent swap of each public C0 plan finds destination-clear failure in only families 207, 210, 212 and 213; each available destination case is selective, with the other five guards true and actual acceptance after disabling destination clearance. Neither dev C0 (208/209) admits one. All twelve C0s offer some selective source-clear swap. This means future destination coverage requires revised base configurations/canonical plans, not merely a different C1 annotation. A legal non-goal C0 prefix with an isolated TABLE no-op audit probe exists for all twelve current C0s, including prefix lengths 1, 2 and 3 in every case. These probes are executed measurements on existing public inputs, not supplied future C4 records or future labels.

Root reported the original public `precondition` value ambiguity. This author confirms the frozen public property/encoding documents do not specify its enum. The new amendment publishes exact typed values/selection/indices before any V2 implementation. No conclusion about historical V1 metadata legality or final reviewer decisions follows from this prospective clarification.

## Prospective allocation and honest units

Reserve the same source indices 180–227, subject to fresh active-lineage verification. A/B/C/D each retain twelve families; split is `dev_challenge` for index `%10` in `{8,9}` and `train` otherwise. Every world/control/renderer inside a family shares that split. V2 revisions and V1 predecessors count once in any active independent-family total, never twice.

Retain 4 contexts each for A/B/D and add C4 to all 12 C families: **204 semantic inputs ×2 actual field/list-order styles =408 correlated rows**, comprising 340 train/68 exposed-dev rows from 40 train/8 dev families. The two additional renderer rows per C4 and its alternative prefix are no new structural families or action mechanisms. If only the fully reviewed V2 cohort becomes active, it adds 48 to the previously measured 180 RGM families, giving 228. The current active count must be recounted before release; this arithmetic is not a release claim.

Use this exact public A allocation; `I` means A1 topologically reachable but fuel-infeasible, and `L` means A1 finite but its fresh minimum is longer than the topology-only minimum:

| Source index | Objects | A subtype | Split |
|---:|---:|---|---|
| 180 | 5 | I | train |
| 181 | 5 | L | train |
| 182 | 5 | I | train |
| 183 | 5 | L | train |
| 184 | 6 | I | train |
| 185 | 6 | L | train |
| 186 | 7 | I | train |
| 187 | 6 | L | train |
| 188 | 6 | I | dev_challenge |
| 189 | 7 | L | dev_challenge |
| 190 | 7 | I | train |
| 191 | 7 | L | train |

Each subtype has two families of every graph size 5/6/7 and five train/one dev family. A has four of every size overall. Its dev sizes are 6/7; all train sizes remain supported.

For B indices 192–203, object counts respectively are `[5,6,7,5,6,7,5,6,7,5,6,7]`. Its dev indices 198/199 supply sizes 5/6. A/B together therefore support all three sizes in each split and each graph stratum has four families per size. Two dev families per stratum cannot individually cover three sizes: this residual limitation is explicit.

For both C204–215 and D216–227, use five blocks for even indices and six for odd indices. Each has six of each size, with five of each in train and one of each in exposed dev. C208/209 and D218/219 supply those dev pairs.

C1's destination-not-clear family indices are **207,209,210,212,213,214**; the remaining C indices **204,205,206,208,211,215** require source-not-clear. Each reason has three five-/three six-block families and five train/one dev family. Each first failure must be selective: the named guard alone is false, the other five are true, and its single-guard ablation accepts and executes that action. This preserves the four existing opportunities as feasible design starting points and requires new suitable C0s at least for 209 and 214. It does not require preserving other V1 worlds, plans or targets; every actual V2 base still must independently pass all freshness, canonical-minimum and dependency gates. The allocation is verified arithmetically; future novelty and C0 feasibility remain measured admission criteria.

For C4, set `k=1+(source_index-204)%3`. Thus failure positions 1/2/3 each occur in four families; dev208/209 use 2/3. Select a non-goal legal canonical C0 prefix of exactly that length with a clear TABLE block. C3 supplies that prefix and C4 appends its isolated unchanged-support failure. If a redesigned C0 lacks this witness, redesign it before freezing; do not change k or call a multiply invalid action isolated. C1 must have at least three actual first-error positions overall and two within each clearance reason, including a nonempty legal-prefix failure within each class.

## Feasibility reasoning and unchanged admission gates

Source dependencies occur when an earlier action removes the block covering a later source. Destination dependencies occur when it removes a block covering a later destination, while the later source is already clear. Both are legal dependencies under existing block dynamics. A minimal ordering can remove an obstruction and then put another clear block on that newly clear destination; exchanging these actions fails destination clearance. Other independent goal obligations can put this dependent pair at different canonical plan positions and keep a minimum horizon within 4–8. This establishes the local dependency pattern, not six future novel accepted worlds. Current public C0 inputs already demonstrate four instances; six revised instances and required lexical ordering must be searched and witnessed, not assumed from this reasoning.

The isolated no-op after a proper prefix requires only a clear TABLE block at that state. Moving it to TABLE passes all other guards and changes no relation, so the contract rejects it solely for unchanged support. No new primitive, state field or repair target is needed. Actual permitted input probes confirm availability at all requested prefix lengths before redesign. Redesigned inputs must demonstrate it again.

All original bounds and novelty gates stay in force: graph sizes 5–7/fuel at most30, block sizes5/6, submitted plans at most12, complete finite search, 24 distinct fresh weighted directed start/goal base classes absent from prior approved bases with at least16 new unweighted graph topologies; 24 distinct fresh joint initial/goal block classes absent from prior approved bases, twelve5-/twelve6-block bases, at least8 distinct six-block initial support shapes. Preserve all A fuel-only boundaries, B edge-only controls and at least6 B0 minimum-route ties, C0 4–8 canonical minimum horizons and at least4 required temporary goal-displacement witnesses, C2 legal goal detours, D valid partial/full goal extension monotonicity and detours. A/B/D original four contexts and C original four remain present.

The old prospective design's shortage was controlled observations, not family count: the earlier measured active RGM already had180 source families within the requested150–250 range. This revised pilot grows relational state/structure/control coverage under the two existing action grammars. It introduces neither inventory transfer/replenishment, logistics prerequisites, minimum edit-distance repair nor a new resource grammar. Object count, long filler plans or additional renderer rows do not establish higher difficulty. A C4 failure isolated after one to three legal moves can remain a lower-level legality task; its C0 family may separately have a dependent4–8 move minimum. No efficacy claim follows.

## Required measured completion before release consideration

Require explicit version/source/spec hash receipts, schema read-back under both styles, exact finite solver agreement, supplied pair field differences and ID bindings, all truthful row witnesses, context coverage, finite-search completion/frontier counts, lineage/split/novelty measurements, and all-family independent expert/red first predictions frozen before targets followed by completed final reviews. Public-only formal fixtures must reject wrong witness strings/types, false code selection, false enabling moves and compound-invalid purported isolated no-ops; use strict integer checks excluding bools.

Measure all three defined single-guard transition mutants on actual supplied C1/C4 inputs, with no compensating guard. Require all6 source C1, all6 destination C1 and all12 C4 semantic inputs to expose their respective mutants, with independently recomputed full six-guard vectors, actual executed changed steps and split/size support as specified in the amendment. All12 C4 must remain unchanged by either clearance omission, while no-op omission changes validity. Preserve first failing machine/reviewer reports; failed or unfinished search is UNKNOWN, not infeasible.

Histograms must cross family/stratum/context/split/style with size, submitted length, fresh feasibility/length, valid/goal status, first-failure reason/index, dependency/enabling position, fuel subtype/boundary, row tags and detour length. Add actual C1 reason×size×split and C4 failure-position×split tables. Run frozen input-only size/length/fuel/edge/token and type baselines. Position variation alone does not establish conditional balance. C4 adds illegal shorter plans and a twelve-family context; mixed inputs do not automatically remove every length/token cue, class imbalance or lookup shortcut. Disclose observed associations rather than asserting universal shortcut removal or structural OOD.

Missing required controls or cohort gates cannot be bypassed by publishing only easy target-correct members. Any later redesign of an infeasible obligation requires another new prospective version, before generation. This document generates no candidate, defines no future gold, approves no data and makes no model/training claims. The author was guided by root's concrete measured concerns and independently checked public inputs, authored V1, previously adjudicated RGM, and may later adjudicate after completed independent first reviews; same-model and design/adjudicator overlap remain explicit.
