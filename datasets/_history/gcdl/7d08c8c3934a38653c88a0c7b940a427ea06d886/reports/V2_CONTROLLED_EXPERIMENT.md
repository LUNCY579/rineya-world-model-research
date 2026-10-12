# Frozen generator-v2 DEV experiment

The frozen experiment passed its mechanical construction, oracle, reproduction and vocabulary-custody criteria. All 32 C parents were valid, with zero exclusions; all 32 vocabulary pairs were valid. The primary goal-blind relationship selector chose an exact best candidate for one goal in every C triplet: **32/96 goal tasks**. This is the expected within-bundle one-third bound from three disjoint best sets, not a trained-model performance result.

The source, specification, profile and lexical pools were frozen before the official measurement. This report summarizes the committed [result JSON](V2_CONTROLLED_EXPERIMENT.json), [freeze receipt](../manifests/V2_EXPERIMENT_FREEZE.json) and [independent review](INDEPENDENT_V2_REVIEW.md). It does not rerun the official seeds or change frozen Python, configuration or tests.

## Protocol and denominators

Arm C used exactly DEV seeds **420000–420031**, one parent per seed, four blocks, horizon four and at most 24 legal-path proposals per parent. The first three distinct non-initial terminal forests defined three equally weighted complete goals over the same ordered six candidates: three endpoint paths, the first path's one-step-shorter prefix, its exact action-order reverse and the empty plan. Candidate IDs and order were fixed independently of goals and gold. Within each triplet, only the inference-input goal changed.

Arm V used exactly DEV seeds **421000–421031**, one ledger parent per seed and two presentations: a separate familiar-vocabulary control using TRAIN pools and a reserved-DEV bijection. There were no replacement seeds. The run used one CPU worker, verified **160 v2 experimental cases in memory**—96 C goal tasks plus 64 V presentation tasks—and completed in **36.249368249991676 seconds**, below the 300-second cap. The separate v1 C baseline supplied 32 observations on the same seed range.

There are **32 C parents, not 96 independent C trials**. The three goals are coupled observations of one fixed bundle. The 288 rendered goal-only checks are repeated presentations of those same 96 tasks, not additional independent parents. The vocabulary arm likewise contains 32 paired parents, not 64 independent worlds. No task-level IID significance or confidence claim is inferred from these repetitions.

## Goal-blind diagnostics

The preregistered primary heuristic selects the lowest opaque candidate ID among plans having both a supplied one-step-shorter prefix and their exact reverse; if none qualifies, it selects the lowest ID. It reads the supplied candidate bundle, not the goal, state, rules or labels. The secondary selectors were also fixed before measurement.

| Fixed selector | V2 best-membership correctness / 96 | Singleton exact-best-set recovery / 96 |
|---|---:|---:|
| Prefix/reverse relationship, then opaque ID | 32 | 29 |
| First option | 19 | 17 |
| Shortest plan, then opaque ID | 0 | 0 |
| Longest plan, then opaque ID | 25 | 23 |
| Lowest opaque ID | 16 | 15 |

Best membership means the one selected ID belongs to the complete gold best set. Singleton exact recovery requires that the complete best set equal that one predicted ID; selecting one member of a tie is not complete set recovery. Gold retains every tied best candidate.

The v1 relationship baseline was best-membership correct for **26/32** tasks and recovered the complete singleton best set for **21/32**, with **five tied baseline tasks**. V1 supplies **five candidates**, whereas v2 supplies **six**, with different candidate-path and goal construction. The baseline is matched by seed and structural reservation, not by candidate universe or complete decision task. Consequently, 26/32 versus 32/96 cannot isolate a single across-version treatment effect, and it is not a model-performance comparison. The result JSON's `matched_baseline_*` fields have this limited matching meaning.

The clean goal-only contrast is **within each v2 triplet**. Its initial state, rules, constraints, instruction, utility definition and ordered candidate IDs/plans stay identical. The three full goals are distinct and non-initial, each has a feasible successful alternative, and the full exact best sets are nonempty and pairwise disjoint. A deterministic selector that reads only this fixed bundle makes the same choice for all three goals; a single chosen ID can therefore belong to at most one best set. The primary selector hit exactly one of three goals for each of the 32 valid parents, reaching the structural one-third bound. This does not constrain a selector returning all candidates or a goal-aware model.

V2 had **93 singleton best sets and three tied best sets of size two**. The ties occurred at goal index zero for seeds 420002, 420012 and 420024. All ties remained in the gold and denominator, explaining why primary best membership is 32 while singleton exact recovery is 29. No prefix winner, reverse tie or illegal reverse outcome was filtered to force an intended path winner.

All 32 C parents passed the preregistered minimum of 24 valid parents. Exclusions were **0/32**, and the exclusion list is empty. Actual proposal counts were three for 17 parents, four for 11, five for two, six for one and seven for one; none exceeded the declared cap of 24. These observations establish feasibility for this bounded DEV sample, not for every possible forest or future split.

## Vocabulary contrast

All **32 paired ledger controls** preserved exact decision labels and restored namespace dynamics/fingerprints. Their reserved-DEV members exercised **48 distinct complete namespace strings**, with **zero overlap against the TRAIN pools**. This arm exercises account, entry and status strings, including invalid references that remain invalid. Familiar controls remain a separate DEV stratum and never count as lexical holdout.

The declaration contains 384 globally distinct eight-letter strings across six namespaces and four split pools, with the same grammar and length distribution and no structural/opcode collisions. Status role keys and action opcodes retain their meanings; the bijections change namespace surfaces consistently. Complete-string non-overlap does not establish unseen tokenizer pieces, unfamiliar natural-language meanings, learned lexical transfer or a model's OOD performance. This pilot's ledger arm alone does not measure block/table/register lexical generalization.

Raw oracle trajectories and state-name strings change with the bijection, so complete target JSON hashes need not be identical across a pair. The exact decision projections and restored executable dynamics match. The parent receipts contain **128 distinct semantic task fingerprints**: 96 C tasks plus 32 lexical parents, whose two presentations intentionally share a fingerprint. There were **zero semantic duplicates across different parents** in the reviewed receipts.

## Verification and physical publication

| Frozen verification measure | Result |
|---|---:|
| Differential-oracle or stored-label mismatches | 0 |
| Deterministic-reproduction mismatches | 0 |
| Valid C parents / attempted parents | 32 / 32 |
| Valid vocabulary pairs | 32 |
| V2 cases verified in memory | 160 |
| Rendered goal-only checks in memory | 288 |
| Physically published v2 canonical records | 0 |
| Physically published v2 rendered records | 0 |

No model training, model inference, paid API or sealed-gold access occurred. The 288 rendered checks were in-memory verification, not dataset publication. Metrics and hash commitments were saved as evidence; canonical v2 tasks and rendered training shards were not physically published. The production storage/free-space guard remains unchanged, and the independent PASS does not authorize bypassing its pause.

The approved **42-file source/configuration digest** is `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`. The root's complete suite passed **189 tests in 75.309 seconds**; the fresh independent reviewer's complete run passed the same **189 tests in 136.504 seconds**. The reviewer separately checked all 64 experimental parent receipts and recomputed membership, ties, singleton recovery and lexical-overlap accounting from the committed evidence, without rerunning or tuning the official seeds. Its `FINAL_VERDICT: PASS` applies to the reviewed implementation and frozen evidence, not dataset readiness or learned planning.

## Commitments and limits

| Committed artifact | SHA-256 |
|---|---|
| Freeze receipt | `b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0` |
| Result JSON | `ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3` |
| Experiment specification file | `19d88bd2da381806c481b88ade73dedf411ebcc07d154054b94d8733ed4b5b3a` |
| V2 canonical curriculum | `1fa733236552daae035fc844be2b09381005d6343aa732483e19185f61ec92d6` |
| V2 curriculum file bytes | `90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd` |
| Canonical lexical pools | `5913a57e252ea84e249fdede5fd6ba3dc89b3dd1b2a025c2c26edf6322bf8024` |

The experiment supplies finite candidate universes, small synthetic forests, endpoint-derived goals and explicitly declared symbol grammars. Goal-aware terminal-structure shortcuts may remain. Separately implemented oracles still share a specification and engineering ancestry; agents share one host/account rather than cryptographic isolation. Existing v1 bytes, identities and semantic fingerprints remain preserved through the archived profile boundary. These results support the bounded mechanical contrast and complete-string vocabulary custody, without establishing learned generalization, causal transfer, universal shortcut resistance or unbounded planning optimality.
