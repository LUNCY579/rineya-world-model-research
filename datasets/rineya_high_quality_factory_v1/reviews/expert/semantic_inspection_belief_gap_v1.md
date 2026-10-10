# Actual expert semantic inspection — belief-gap-v1

Actual reviewer/execution: `/root/expert_reviewer`. Initial sample review, guided by my previously published prospective RBM gap diagnosis and the shared public design. I authored a new implementation from the public contract; I read no producer/reference/oracle implementation/tests or reports, formal/root mutant results, or another reviewer's private judgments. Separate agents share an underlying model and may make correlated errors. These receipts are expert decisions, not adjudication or release approval.

I accepted 1,296 exact-bound records after independent typed recomputation and substantive inspection. No target, contract, property-locator, profile, family-control, split-custody or rendering blocker remains. The quality scores retain material limitations: four independent source tables, repeated motifs, cyclic row patterns, simple empty/repeated substeps, and no evidence for Bayesian inference, chosen information gathering, utility, sealed OOD performance or model efficacy.

## Prediction-before-gold sequence

The root supplied an eight-field input-only view, SHA256 `86a07b5b6e4bf89706e9491ed41415e3facc8a744acdf14ad551cbade8529ac6`, retaining original IDs/hashes and visible tasks/prompts but excluding gold, tags, role maps, locators, family claims and reviews. My prepared program explicitly rejects fields outside that whitelist and nested gold fields. Its typed incremental filter and separately recomputed direct prefix conjunction agree on every full answer and prefix set. On 2026-10-10 at 02:31:19 UTC, I froze all 1,296 own expected outputs, supports, prefixes, property facts and metamorphic results in `belief-gap-v1.input_only_predictions.json`, SHA256 `f732c60d2ae3646bb3cc92bc166dc05a867c94bbc3b364558630e6e1f647d4fe`. No canonical gold or producer metadata had been read at that point.

A separately invoked comparison subsequently read the original canonical batch, SHA256 `423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e`. All 1,296 full targets match using canonical typed JSON equality: true/1 and false/0 differ in nested structures as well as scalar filters. Every record SHA256 independently recomputes after excluding its own record_hash field. Every canonical semantic task agrees with the student-visible Task JSON. There is one complete, unchanged contract prefix; metadata/property tags and stored target fields are absent from all prompts. The contract fixes survivor sorting, scalar-bin sorting, query order, prefix order, empty flags and the different interpretations of guaranteed=null; these make the complete answer unambiguous.

Five authored toy fixtures and 42 preparation probes precede any candidate read. On the actual batch, I executed 9,072 invariance probes and 972 typed-equals probes; 648 typed probes qualify as nonempty, different changed-prefix support controls. World-list reversal, world-ID bijections, query reversal/ID renaming, consistent fact renaming, observation reversal with independently recomputed traces, and repeated-observation final invariance all pass. Probe executions create zero corpus rows/families.

## Substantive table and context inspection

I read the complete four representative visible world tables and the outputs/trace/property facts for all six contexts, plus each family's A-s26-reverse task and role/locator mapping. Programmatic recomputation covers every individual row; this is not a claim that I manually read each of 1,296 renderings.

Source 12 has 12 distinct payload worlds, six occupied observation cells, six incidence vertices, cycle rank one and uniform cell weights two. In `RBM-001-belief-gap-v1-p0012-A-s00-forward`, ribbon=reed then copper=linen gives prefix sizes [6,2], survivors w13956/w78775. Rivet is an unobserved invariant null; basket has bins null:1,true:1, while ribbon is directly entailed reed. Omitting the first constraint leaves rivet opal:2,null:2; omitting the second leaves opal:4,null:2. Both constraints are necessary for the invariant. The matched B changes only ribbon=reed to fern, still retains two worlds, and all three offered queries become redundant. Basket entails the literal string unknown, which is an ordinary scalar; canvas still varies east/west, so the two survivors are genuinely different. C has sizes [4,2] with the same full final answer, D [6,2,2], E [2,0], and F [2,0,0]. The final empty/null outputs differ from A's entailed-null and ambiguous-null through counts/flags. This family supplies the complete null contrast.

Source 13 has eight distinct worlds and a connected acyclic six-vertex/five-edge incidence graph with a degree-three branch. In `RBM-001-belief-gap-v1-p0013-A-s00-forward`, copper=boolean false then vessel=linen gives [5,2]. Ribbon entails null over w22298/w51037; panel has separate integer-0 and boolean-false bins, in canonical order 0 then false. Before the first filter, both false and integer 0 are active on copper; exact false excludes all integer-0 worlds. A/B toggles only that typed equals value, with different nonempty changed-prefix supports and equally sized final supports. B sizes are [2,2]: its second filter is redundant, and I do not count it as a second strict reduction or claim a conjunction-required invariant there. C [4,2] differs from A's trace; D [5,2,2], E [1,0], F [1,0,0]. The string unknown in the world/contradiction control has no special status.

Source 18 has 12 distinct worlds, the same unweighted cyclic incidence motif as source 12, but unequal occupied weights 3,3,2,2,1,1 and a different full partition structure. In `RBM-001-belief-gap-v1-p0018-A-s00-forward`, column=boolean true then basket=linen gives [5,3]. Panel entails null over w39633/w89662/w95963. Rivet varies with null:2,true:1, so skew is unweighted multiplicity, not posterior probability or information gain. Removing either observation makes panel vary. B changes true to integer 1, retains three different worlds, and all offered queries become redundant while socket varies. C [6,3], D [5,3,3], E [2,0], F [2,0,0]. It supplies another full entailed-null/ambiguous-null/empty-null contrast in exposed dev.

Source 19 has eight distinct worlds and a connected six-vertex/five-edge path with maximum degree two, distinct from source 13's branch. In `RBM-001-belief-gap-v1-p0019-A-s00-forward`, socket=boolean true then ribbon=linen gives [3,2]. Vessel entails null over w34094/w92071; canvas partitions integer 0 and boolean false into two separate bins. True versus integer 1 is an active observation separator; false versus integer 0 is an active final-query partition separator. B changes only true to integer 1 and keeps final support two, making every offered query redundant. C [4,2], D [3,2,2], E [1,0], F [1,0,0]. All empty E/F queries have empty count lists, distinct_values=0, both flags false and guaranteed=null.

Each E's two different-fact observations are individually supported in the full table but jointly impossible. Each F appends a later observation individually supported in that full table; empty support stays empty. Each D repeats the first observation at nonempty support. A duplicate constraint makes single-observation omission unable to prove that both original constraints remain individually necessary, so D carries no conjunction_required_invariant tag. This is a useful check against blanket family tagging.

## Property and family evidence

All attached per-row property names are public identifiers, and every locator's query IDs, observation indices and prefix indices identifies a recomputed witness in that same row. Actual property counts equal attached tag counts: multiworld_unobserved_invariant 864; conjunction_required_invariant 594; useful_after_conjunctive_evidence 648; multiworld_all_queries_redundant 216; multiworld_entailed_null 648; ambiguous_null_partition 324; cross_fact_inconsistent_conjunction 432; post_empty_absorption 216; nonempty_repeated_observation 216; typed_observation_separator 648; typed_query_partition_separator 324; skewed_possible_value_partition 162. No undefined aliases or per-family controls appear as row skill properties.

The supplied A/B, A/C, A/D, A/E and E/F relationships preserve their specified worlds/queries and change precisely the allowed observation content/order. There are 162 supplied typed-equals A/B pairs across sources 13,18,19. Optional producer evidence pointers were never read or credited as witnesses. Audit-scoped world-order and world-ID claims are supported by my actually executed same-family transformations, with input/equivariant output evidence. Family scope denotes relationship existence, so a claim on an E/F descendant is not a claim that the empty row itself is a typed corpus pair.

My initial post-freeze program emitted 972 diagnostic warnings because its family-control registry implemented typed-equals relationships only as probes, overlooking supplied A/B relationships. I independently reconstructed those exact pairs and preserved the original diagnostic output. All 972 warnings are resolved as an implementation coverage limitation. This correction occurred before any per-row judgment was issued; no prediction, target, candidate or first verdict was changed.

All 648 forward/reverse pairs have equal prompts before Task JSON, identical typed parsed tasks/list orders, and actual recursive JSON object-key reversal. They are different prompt strings with identical full answers; observation order is changed only in the separate C context. All rows have difficulty and structural_difficulty 3; A/C have decisive two-constraint filtering, B/D/empty contexts have the stated simpler component metrics. No level4–7 is attached.

## Crossing, independence and limits

The achieved counts are four source families, 24 base contexts, 648 rotation tasks and 1,296 prompt strings/field-order records. There are 12 literal world tables due to three queried-name rotations, four decoded world tables and 3,888 rendered query observations. Families 12/13 retain train across all 648 descendants; 18/19 retain exposed dev across all 648 descendants. No duplicate complete world payload is used to manufacture multiworld uncertainty.

For every one of 48 family/context/field-style groups, all 27 schedule indices occur once, every stable role occupies every actual ID/position/name triple once, and every marginal/pair/triple full typed-query-payload plus prefix/survivor-provenance histogram is identical. All 336 measured axis checks pass. Actual consistent name mapping includes worlds, observations and queries. The A-s26-reverse inspections confirm shifted names/IDs/positions and locators refer to the new visible facts rather than stale slot names. This removes the specific three-way role-plane defect of the earlier nine-schedule proposal.

The four new full incidence structures are mutually nonisomorphic after arbitrary world/fact names and scalar substitutions; exact comparisons establish this, with WL fingerprints used only as prefilters. My frozen prior own analysis measured only four-/six-world tables. New eight-/12-world tables cannot be isomorphic to those because the number of world vertices differs. Uniform/weighted families still share an unweighted motif; four tables are four source families, not 1,296 mechanisms. The task uses one deterministic filtering/partition mechanism.

The earlier incomplete useful/entailed heuristics each fail on 1,080/3,888 query flags and on 864/1,296 complete tasks. Every A contains an unobserved invariant that breaks both, and every A/B pair changes usefulness at equal support cardinality with the same observed-fact membership. This supports the targeted discrimination gain, not model performance. Query categories are 1,944 entailed, 648 ambiguous and 1,296 empty; guaranteed-null occurs in 648 entailed-null, 648 ambiguous and 1,296 empty queries.

Residual patterns remain. An empirical support-only joint-flag majority fits 3,240/3,888 observations (5/6), while ID alone, position alone, name alone or their nuisance triple each fits only 1,944/3,888 joint flags. Decoded semantic role plus context fits all flag pairs, but those are internal audit labels, not student-visible fields, and that statistic is not a demonstrated solver or leakage. Null/opal invariant columns and linen auxiliary values recur; cyclic shifts retain row cyclic order. Broader all-fact role, arbitrary permutation, world-ID/order-style independence and unseen-mechanism generalization were not supplied or claimed. Full counts and survivor/prefix provenance require more than a majority flag prediction.

The batch usefully tests the requested finite gap, with quality scores3 for generalization/diversity/shortcut resistance. E/F's empty-query substep and D's repeated step remain simple; component difficulty scores reflect that. An ACCEPT receipt means this row is correct and suitable within its declared finite scope, not that its 27 schedules are independent evidence or that broader belief/action capabilities have been established.

## Exact substantive inspection IDs

The actual-agent inspected representatives are:

- RBM-001-belief-gap-v1-p0012-A-s00-forward
- RBM-001-belief-gap-v1-p0012-B-s00-forward
- RBM-001-belief-gap-v1-p0012-C-s00-forward
- RBM-001-belief-gap-v1-p0012-D-s00-forward
- RBM-001-belief-gap-v1-p0012-E-s00-forward
- RBM-001-belief-gap-v1-p0012-F-s00-forward
- RBM-001-belief-gap-v1-p0013-A-s00-forward
- RBM-001-belief-gap-v1-p0013-B-s00-forward
- RBM-001-belief-gap-v1-p0013-C-s00-forward
- RBM-001-belief-gap-v1-p0013-D-s00-forward
- RBM-001-belief-gap-v1-p0013-E-s00-forward
- RBM-001-belief-gap-v1-p0013-F-s00-forward
- RBM-001-belief-gap-v1-p0018-A-s00-forward
- RBM-001-belief-gap-v1-p0018-B-s00-forward
- RBM-001-belief-gap-v1-p0018-C-s00-forward
- RBM-001-belief-gap-v1-p0018-D-s00-forward
- RBM-001-belief-gap-v1-p0018-E-s00-forward
- RBM-001-belief-gap-v1-p0018-F-s00-forward
- RBM-001-belief-gap-v1-p0019-A-s00-forward
- RBM-001-belief-gap-v1-p0019-B-s00-forward
- RBM-001-belief-gap-v1-p0019-C-s00-forward
- RBM-001-belief-gap-v1-p0019-D-s00-forward
- RBM-001-belief-gap-v1-p0019-E-s00-forward
- RBM-001-belief-gap-v1-p0019-F-s00-forward
- RBM-001-belief-gap-v1-p0012-A-s26-reverse
- RBM-001-belief-gap-v1-p0013-A-s26-reverse
- RBM-001-belief-gap-v1-p0018-A-s26-reverse
- RBM-001-belief-gap-v1-p0019-A-s26-reverse

`verification_summary_belief_gap_v1.json`, the completed execution receipt, the input-only freeze receipt and the first/final exact-hash JSONL bind the actual computation. The original 42,222,663-byte post-freeze comparison is retained locally; its deterministic gzip is 1,133,682 bytes, SHA256 `01e5c457bc2f15e1942c1d49620fc3477ae4fd6e99a4533c0fe0e8706f379fed`, with verified decompression byte equality and decoded SHA256 `3f84bd5289a79a8e07d109d84d8dc72e7a07973049c833fc68c467765c2f1937`. Other review/release and REM artifacts remain unchanged.
