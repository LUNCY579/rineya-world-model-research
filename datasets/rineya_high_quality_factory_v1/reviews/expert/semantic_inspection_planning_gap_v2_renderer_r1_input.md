# Expert renderer-r1 input-stage inspection

Actual author/execution: `/root/expert_reviewer`. This is the same-task renderer revision input stage, completed before access to the fresh canonical candidate, gold, control manifest, producer code/fixtures, formal implementation/tests, or other private verdicts. Original V2 canonical exposure and own previous semantic review knowledge are disclosed. The public revision policy summarizes earlier audit/review findings; its cited private files were not read. My solver, handmade fixtures, and original pre-gold expectations are correlated with my earlier work and share the underlying model/public design context. This is not a claim of blind new semantic review or model independence.

All 408 fresh input rows joined uniquely to my original stripped rows by source family, split, exact typed task content, and observed payload rendering style. IDs and record-hash strings were carried through as custody keys; the full fresh canonical record hash cannot be recomputed from this stripped view. My original pre-gold predictions were reused only after these joins. I also recomputed full typed targets for every one of the 204 normalized tasks with my own public-contract solver, replayed all 408 submitted plans, and checked every prefix state against my frozen original input-derived facts. There were no input discrepancies.

The new authored wrapper was archived before any import, calibration, or execution, SHA `9798e4881de9894f53229abddd9282643b9cb1fca8bf43b55495b44bbd4c711b`. Actual execution exited 0 in 5.107 seconds. All 830 checks from my own frozen public V2 preparation and six new handmade rendering/type calibration checks passed. These are checks of my actual code, not a target-model performance evaluation.

## Actual input and rendering scope

408 rows represent 204 normalized tasks, 300 raw typed JSON variants, and 48 existing source families. The split is 340 train rows and 68 dev-challenge rows. There are zero new tasks or source families relative to the original V2 view. Graph node/edge list presentation is normalized for the task count; the submitted action list is never sorted or reversed. Exact original-to-new joins additionally preserve the raw typed semantic lists, so normalization does not conceal semantic edits.

All 408 visible prompts were parsed with duplicate-key rejection and checked against their corresponding typed semantic tasks. Every prompt lead is byte-identical to its original own input-only lead. There are exactly two leads, the graph and block executable contracts. The graph lead explicitly defines directed edges, positive integer fuel costs, first-illegal stopping with unchanged state, initially satisfied/empty behavior, fresh search from the initial state, move-count minimization, and complete destination-sequence lexicographic tie breaking. The block lead explicitly defines clearance, destination legality, distinctness, changed-support/no-op rejection, first-illegal stopping, partial goals, fresh minimum-move planning, and full `(block,to)` sequence tie breaking. These rules make the expected answer unique under the stated contract. No recognized private field-label leak was found in any lead.

There are 192 graph and 216 block rendered rows. All block forward goal-map keys are lexicographically ascending; all reverse keys are descending. Other top-level, on-map and action-object key ordering is unchanged. Reconstructing each new prompt from the original prompt by changing only goal-map presentation reproduced all 408 new prompts byte for byte. Exactly 22 prompts differ: the two D0 renderings for eleven families. The remaining 386 prompts, including every D1 full-goal prompt, are unchanged. Family 225 D0 was already ordered correctly in both styles. All 24 D0/D1 same-style partial/full-goal pairs preserve the order of shared goal keys, eliminating the previous shared-key-order confound while leaving values and action chronology intact.

## Every changed rendering inspected

The table gives exact fresh sample IDs and their original-to-new goal-key sequences. Complete original/fresh record hashes and per-row evidence are in the immutable input analysis and predictions.

| Fresh sample ID | Goal keys before → after | Own fresh minimum moves |
|---|---|---:|
| RGM-001-planning-gap-v2-renderer-r1-p0216-D0-forward | ember, pearl, nacre → ember, nacre, pearl | 5 |
| RGM-001-planning-gap-v2-renderer-r1-p0216-D0-reverse | nacre, pearl, ember → pearl, nacre, ember | 5 |
| RGM-001-planning-gap-v2-renderer-r1-p0217-D0-forward | larch, quartz, amber → amber, larch, quartz | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0217-D0-reverse | amber, quartz, larch → quartz, larch, amber | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0218-D0-forward | spruce, nacre, grove → grove, nacre, spruce | 5 |
| RGM-001-planning-gap-v2-renderer-r1-p0218-D0-reverse | grove, nacre, spruce → spruce, nacre, grove | 5 |
| RGM-001-planning-gap-v2-renderer-r1-p0219-D0-forward | cedar, maple, larch → cedar, larch, maple | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0219-D0-reverse | larch, maple, cedar → maple, larch, cedar | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0220-D0-forward | vessel, nacre, willow → nacre, vessel, willow | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0220-D0-reverse | willow, nacre, vessel → willow, vessel, nacre | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0221-D0-forward | maple, jade → jade, maple | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0221-D0-reverse | jade, maple → maple, jade | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0222-D0-forward | birch, amber, ember → amber, birch, ember | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0222-D0-reverse | ember, amber, birch → ember, birch, amber | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0223-D0-forward | willow, yarrow, fjord → fjord, willow, yarrow | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0223-D0-reverse | fjord, yarrow, willow → yarrow, willow, fjord | 4 |
| RGM-001-planning-gap-v2-renderer-r1-p0224-D0-forward | hazel, dune, rivet → dune, hazel, rivet | 2 |
| RGM-001-planning-gap-v2-renderer-r1-p0224-D0-reverse | rivet, dune, hazel → rivet, hazel, dune | 2 |
| RGM-001-planning-gap-v2-renderer-r1-p0226-D0-forward | olive, willow, umber → olive, umber, willow | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0226-D0-reverse | umber, willow, olive → willow, umber, olive | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0227-D0-forward | willow, vessel → vessel, willow | 3 |
| RGM-001-planning-gap-v2-renderer-r1-p0227-D0-reverse | vessel, willow → willow, vessel | 3 |

## Substantive semantic inspection

I inspected both renderings for every listed D0 change and the corresponding same-style D1 shared-key projection in the actual input evidence. The changes are object-map presentation, not reordered goals in an ordered instruction language. The public block contract defines goals as a partial map; goal-map insertion order does not change satisfaction or the lexicographic ordering of action sequences.

For `RGM-001-planning-gap-v2-renderer-r1-p0218-D0-forward` (fresh record hash `cf2da675a05b95d7c1b01fa6339fc3fa9b3bf57f7eeb016ebb5d67d2df7b9416`), the old keys `spruce,nacre,grove` become `grove,nacre,spruce`. The initial stacks include nacre on fjord on grove, and spruce on dune. The plan temporarily moves nacre to TABLE, clears fjord from grove, restores nacre on fjord, moves spruce to nacre, then grove to dune. All five actions are legal and satisfy the three partial goal entries. My own fresh minimum is that five-action sequence. The matching D1 prompt adds dune-on-spruce and fjord-on-TABLE, preserves all three shared entries and their presentation order, and retains the same submitted plan. The submitted plan remains valid but fails the added dune-on-spruce goal; a fresh minimum takes six moves, inserting dune-to-spruce before grove-to-dune. Thus the rendering repair preserves the decisive partial/full-goal distinction and does not change the semantics into an easier or different task.

For `RGM-001-planning-gap-v2-renderer-r1-p0225-D0-forward`, the two-move olive-to-maple, nacre-to-olive plan is legal from six separate TABLE roots and satisfies the partial goals. The keys maple,nacre,olive were already ascending; its prompt therefore correctly remains unchanged. This case guards against equating all D0 rows with changed renderings or counting changed aliases as new semantic data.

I also examined unchanged mechanisms directly from the fresh visible prompts. At `p0209-C1-reverse`, the first action clears dune from cedar; the next cedar-to-jade move fails because rivet is still on jade. The illegal step leaves the map unchanged and later moves are not evaluated; the fresh minimum still starts by clearing dune then rivet. At `p0198-B2-forward`, pearl-to-birch is legal and consumes two fuel, but birch-to-pearl lacks an edge; the answer stops at birch with fuel 28 while the fresh minimum is birch,amber,spruce. At `p0188-A1-forward`, the submitted four legal moves leave fuel 2 at umber and the fifth needs 3 fuel; the fresh minimum is null because reaching vessel needs total fuel 9 and the initial fuel is 8. These checks distinguish replay, feasibility, and fresh optimal planning and confirm the renderer did not reorder action chronology.

The independent executable family audits passed for all 48 families and their public context relations. The renderer repair changes no mechanism class. The earlier narrow-family and conditional shortcut limitations remain: fixed context proposal lengths in the B stratum, consistent missing-edge failure positions in several contexts, and legal D-stratum proposals can support partial-label shortcuts. The corrected goal presentation supplies a cleaner controlled comparison; it does not establish generalization, remove every shortcut, add mechanism classes, or justify raising difficulty. Per-input generalization/diversity/difficulty scores stay moderate, and target correctness is left unscored until fresh canonical comparison.

## Frozen result and remaining scope

All 408 immutable first input judgments are `INPUT_CLEAR_PENDING_FRESH_CANONICAL_COMPARISON`; none is a dataset ACCEPT. Predicted targets SHA `1d4decd9539d08e1e6e53bc894b5a915c72d38c161e28a2501dd09e753024d7a`; first-input judgments SHA `30672302b9f9f3ad05e0edc5a628f32132c35a9a2b7c97b8f6c1cd493cff39a8`; input freeze SHA `d36b5562d8b570bb4c0941ccb31cdf8e65af03269147e8b12f221e556aa1204c`.

Fresh full-record hash recomputation, canonical gold comparison, tags/difficulty/property witnesses and renderer-specific manifest declarations remain unassessed. I will perform that separate comparison stage only after root authorization. Original review/source bytes and release state were not edited; no publication or release approval is issued.
