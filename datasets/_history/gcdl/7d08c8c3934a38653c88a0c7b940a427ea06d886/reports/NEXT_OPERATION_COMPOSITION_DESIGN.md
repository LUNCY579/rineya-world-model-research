# Proposed generator v3: register operation composition

Status: DESIGN ONLY, 11 October 2026. No source/configuration change, task generation, publication, model evaluation or training is authorized by this document. A later approved code/spec freeze must precede the pilot. Preserve the exact reviewed v2 digest `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467` and confirmed branch backup `94a0952eb04674c6e28097c48d1682938a5856d2`.

## Observed weakness and falsifiable hypothesis

The fresh nonsealed audit [OPERATION_COMPOSITION_COVERAGE_20261011.md](OPERATION_COMPOSITION_COVERAGE_20261011.md) reports 66 H/register rows: add 198, noop 66, and multiply/set/copy/swap zero despite enabled rules. All 264 candidates are one-step, with one active valid register and one goal atom per row. Across all 264 audited H rows, singleton settle or positive valid add recovers every winner without reading goals/constraints. The reserved OOD constraints do not bind these choices. This is coverage evidence, not evidence of model performance. The completed audit receipt/hash must accompany any later freeze; preliminary counts must not be silently restated as new measurements.

**H-v3:** a bounded register-specific generator can produce verified decisions requiring (a) different primitive operations, (b) noncommuting read-after-write effects across at least two registers, and (c) goal/constraint-dependent choices. It will fail promotion if merely listing opcodes, appending noops, touching idle registers, naming a held-out template, or shuffling options accounts for the claimed gain.

Keep this improvement focused on new H/register bundles. Ledger H limitations remain open; aligned additive ledger/register probes do not become compositional transfer evidence. Do not rewrite existing H rows or call the proposed records generated/research-ready.

## Exact semantic scope and concrete witnesses

Retain `gcdl-1.0`, `environments-1.0` and current oracle semantics: integer register add/multiply/set/copy/swap/noop; overflow reject/clamp/wrap; successful-action costs; initial and every intermediate constraint check; illegal execution stops at its first illegal step. Utility remains exact lexicographic `(feasible, satisfied_goal_atoms, -cost)`. No distance reward or new confidence semantics.

For each new bundle declare 2–4 registers, public bounds/overflow/costs, at most six candidate plans and plan horizon at most three. Primary claim is only best among these candidates. At least two actual register values must influence a composed plan's outcome; an unused extra register is a distractor, not interaction coverage.

These are hand-specified design witnesses, not generated/tested examples:

- **Order/goal:** initial `(x,y)=(2,3)`. P=`copy y->x; multiply y by 2` gives `(3,6)`; Q=`multiply y by 2; copy y->x` gives `(6,6)`. The operation multiset, costs, rules and candidate list are identical; goals `(x=3,y=6)` and `(x=6,y=6)` reverse the correct choice. Neither positive-add preference nor option order resolves the pair.
- **Intermediate constraint:** initial `(x,y)=(2,3)`, goal `(x=2,y=6)`. P=`multiply x by 3; copy x->y; set x=2`, with costs 1+1+4, gives `(2,6)` through `x=6`. Q=`set y=6`, cost 4, is initially cheaper; therefore this cost assignment cannot demonstrate a constraint flip. Freeze instead costs multiply=1, copy=1, set=3: P costs 5 and Q costs 3, still unsuitable. The valid construction uses P=`multiply x by 3; copy x->y; multiply x by 0; add x by 2` only if horizon four is authorized, which this pilot does not allow. **Use the following horizon-three witness instead:** initial `(x,y)=(2,3)`, goal `(x=3,y=6)`, P=`swap x,y; multiply y by 3` costs 2, Q=`set x=3; set y=6` costs 6 with set=3. P transiently/finally changes x to 3, so `protected_registers=[x]` excludes both and is an all-infeasible control, not a winner flip. For an actual bound flip, P's intermediate/final y=6 violates `maximum_values={y:5}`, again excluding both successes. **Implementers must construct and freeze a valid constraint-flip witness from the recipe below; these failed sketches are exclusion examples, not accepted tasks.**

Constraint recipe: find two legal plans under the fixed six-candidate library with identical goal-atom count, differing cost, and differing intermediate constraint history. The cheaper plan must violate a proposed protected/minimum/maximum constraint while the more expensive plan satisfies it; the initial state must satisfy both settings. Independently verify an unchanged-goal twin with disjoint feasible best sets. Exclude failures without repairing gold. Cost-cap twins may instead change the feasible optimum from full success to a cheaper partial-goal plan; explicitly label `impossible_in_candidate_set` and distinguish this from no feasible candidate. Do not claim every constraint type can create the same kind of reversal.

## Fixed construction and matched controls

Before execution, freeze deterministic templates, operand ranges, seed list, retry order and exclusion reasons. Generate from syntax/structural predicates, not from sealed outcomes. A bounded proposal may execute its public candidates to certify the advertised property; disclose this conditioning and preserve excluded proposals.

Use three libraries on matched initial states/rules/goals/constraints and six slots: B0 add/noop/invalid control resembling the old H limitation; B1 diverse one-step primitives; B2 dependent two/three-step compositions. Keep candidate availability identical within all goal/constraint twins and within opcode-order controls. B0/B1/B2 deliberately change candidate availability; their coverage comparison is not a matched model-success comparison. Zero-cost noop padding cannot count toward nontrivial horizon or causal depth.

For B2, include a plan and its same-operation-multiset order control. Require a read-after-write edge from an earlier write to a later copy/swap/arithmetic read, and an exact terminal/ranking difference when reordered. Merely observing different terminals is insufficient: verify that the changed value reaches a goal atom or a binding intermediate constraint. Report actual distinct registers read/written, nonnoop steps, noncommuting pairs, and success/utility witnesses. All operation counts come from executed successful and attempted traces separately.

Candidate IDs, order, namespaces and framing use goal/constraint-independent bundle identity. Every goal/constraint twin shares the same candidate surface. A spelling-only control must retain exact labels; a governing rule/order change requires fresh oracle labels. Preserve complete tied-best sets and all infeasible/unsuccessful labels; do not force a singleton for the experiment.

## Reservations and immutable compatibility

| Partition | Proposed semantic reservation, fixed before sampling |
|---|---|
| TRAIN | Two-register primitive support; dependency motifs add->copy and multiply->swap; reject overflow; each constraint primitive separately |
| DEV pilot | Three-register worlds; motifs set->copy and swap->add; independently bounded magnitudes; constraint twins; no OOD motif/conjunction |
| OOD, future instrument | Four-register worlds; copy->swap->add chains, combined protected+cost constraints and held-out overflow boundary regimes |

Whole dependency-graph/motif, world/rule skeleton and constraint composition assign custody, not names or random seeds. No root parent, goal/constraint twin, order control, namespace variant or trajectory suffix crosses partitions. Keep a persistent cross-version normalized fingerprint registry; version prefixes must not defeat duplication. Primitive operation sharing is intentional; new joint structure is the holdout. The proposed pilot loads DEV only and neither builds nor reads OOD or SEALED. Existing v1/v2 sealed instruments remain untouched.

Add a separately allowlisted v3 profile and archived curriculum only after approval, without altering v1/v2 archives, generator code, ID algorithm, normalizer or seals. V3 metadata may authenticate its own motif/group invariants; schema additions require a separate contract decision. Revalidate exact archived nonsealed IDs/fingerprints, previous profile resolution and resumed-job ownership. New jobs pin v3 explicitly; existing enqueued jobs/manifests remain their original profile. Accepted v1/v2 bytes, shards, manifests and receipts remain immutable. Cross-version collisions quarantine the new proposal, never overwrite an old record.

## Predeclared pilot and decision rule

Maximum **32 independent DEV parents**, selected by a frozen seed list: 8 primitive goal pairs, 8 dependent-order goal pairs, 8 constraint twins, and 8 edge parents (2 tied, 2 illegal-prefix, 2 no-success-with-feasible-candidates, 2 all-infeasible). Goal/constraint twins are variants, not extra independent parents. At most four v3 variants per parent and one B0/B1 control each: 192 task evaluations maximum. At most eight deterministic proposals per intended parent; failure reduces the valid count, never expands the parent budget or replaces seeds after seeing results.

One CPU process, no model/GPU/API, hard wall cap 300 seconds, process RSS cap 512 MiB, serialized pilot artifacts cap 8 MiB; stop/checkpoint on cap or fewer than the normal 10 GiB free-disk floor. No production enqueue/publication. Optional exhaustive checks must use a frozen alphabet of at most six actions through horizon three: at most `1+6+36+216=259` plans per checked parent. Count all attempts and checks; this finite proof never establishes unbounded optimality.

Freeze analysis and thresholds before pilot outcomes. ADVANCE requires:

1. Every proposed record passes both executable implementations, schema/reproducibility and protected custody; zero unresolved disagreement. Failed parent groups are excluded/quarantined together.
2. At least 24/32 intended parents valid, including at least 6/8 dependent-order and 6/8 constraint parents. Each add/multiply/set/copy/swap executes nontrivially in at least four valid parents; noop is separately counted. At least four parents have feasible best sets requiring a non-add primitive. Listing enabled operations earns no credit.
3. At least six dependent parents show verified noncommuting, goal-relevant read-after-write effects across two registers; at least six goal pairs and six constraint twins have unchanged candidate universes and disjoint nonempty feasible best sets. For balanced unique-choice twins, any deterministic selector ignoring the changed goal/constraint has at most 1/2 pair accuracy by construction; an all-candidate output does not qualify as a choice.
4. All eight edge slots retain their intended exact label semantics, or the corresponding stratum fails. The original positive-add/settle shortcut and frozen first-option, cheapest-feasible and single-register-only floors are scored separately, with reasons. A floor's failure is descriptive; it does not rule out all shortcuts or show model efficacy.
5. B2 materially expands witnessed primitive/dependency/constraint coverage over B0/B1 while preserving all frozen compatibility tests. No promotion based on volume, option entropy or opcode frequency alone.

Otherwise REVISE with named failures, or INCONCLUSIVE for resource exhaustion; never relax thresholds after inspection. Report parent counts, exclusions, per-operation strata and matched-control differences without statistical efficacy claims from 32 parents. A passing pilot supports generator design only. Further code review, fresh confirmation design, resource gates and explicit promotion are needed before any v3 production activation.
