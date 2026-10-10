# REM delayed history, conflicts and relevance controls

REM-GAP-003 design, 2026-10-09. Actual author: `/root/research_architect`, Dataset Research Architect. Status: specification only; no sample generation, target production or approval. Public factual property definitions are in `schemas/MEMORY_CONTROL_PROPERTIES.md`. The existing episodic contract remains unchanged.

## Why this gap is specific

The adjudicated REM scope contains 120 records from 12 source families. Matched query rotations removed the earlier deterministic outcome-by-ID/position pattern, but remain correlated views of small repeated ledgers. The extra development event at 19 is not queried before supersession; its presence therefore does not demonstrate an additional decisive history dependency. `reviews/adjudication/ADJUDICATION_EXTENDED_V3.md` reports these exact limitations. The next-priority note `research/research_gap_analysis/PRODUCTION_MILESTONE.md` recommends decisive updates and relevance/conflict controls, rather than context-length inflation.

This pilot addresses those gaps using visible finite ledgers. Relevance means exact key and time eligibility. Conflict resolution means the supplied greatest `(version,at,input_index)` policy. The broader original `research/curricula/REM-001.md` proposals for source authority, unresolved conflict, procedure memory and multi-record reasoning remain future design hypotheses; they are not added by this pilot. No semantic matching, trust score, uncertainty output or cross-query dependency may be inferred from the current contract.

## Planned family and query design

The production proposal is 12 new source families at producer indices 12..23, each with four controlled histories, eight query-role/ID rotations and two JSON field orders: **768 correlated records**. The12 indices must be checked against actual prior source-family/world lineage. Each family and all its descendants remain in one split. Historical approved bytes remain frozen.

Declare eight base query roles before rendering. The following allocation makes the intended checks concrete; the producer may use different role names only with an auditable mapping to the same properties:

| Query role | What the task must demonstrate |
|---|---|
| delayed-before | Exact same-key cutoff T-1 immediately before an event at T. Supply the matched delayed-at query, and ensure their semantic answers differ. The future event must be excluded despite being a contender if cutoff were ignored. |
| delayed-at | Cutoff T includes the newly decisive event. Its winning status or typed value changes from T-1. A version/index-only change is insufficient. |
| later-lower-version | A more recent eligible lower-version put conflicts with an older eligible higher-version winner; latest-time-only retrieval would answer differently. An appended irrelevant lower version is additionally an invariance control only for named already-answered protected queries. |
| lifecycle-present-null | The winning operation is put with null value. Contrast a deleted/absent-null query, so value=null cannot predict status. |
| lifecycle-delete | A winning tombstone dominates at least one eligible lower-version put. If claimed as a deletion boundary, include same-key before/at cutoffs with status change. If claimed as null lifecycle counterfactual, change only the winning op while holding null value and all other fields fixed. |
| same-version-time | An older and newer eligible event at the maximum version conflict in status/typed value, and time determines the winner. The competitor must be active at the queried horizon. |
| same-version-time-index | Conflicting eligible events tie at the maximum `(version,at)`; input index determines the winner. An active pair swap must change semantic answer if that counterfactual is claimed. JSON object field-order reversal must preserve it. |
| absent | No eligible exact-key event. Report whether the key is wholly missing or future-only. When used as the future-only control, the excluded event would otherwise win and change status/value. |

Roles can share a key and reuse the same event for several legitimate checks, but each property needs its own witness and each claimed query signature must be counted accurately. Different roles must not be declared distinct semantic queries merely because their IDs differ. All histories should include present, deleted and absent answers, so all reused query IDs and positions see each status after a complete eight-rotation schedule.

Use four equally styled, arbitrary key strings with four declared semantic key groups. Require an exact measured within-family crossing schedule, identified by history/rotation/field-order indices. A four-history crossing can place each name in each group once; alternatively, an eight-rotation crossing within each history can place every key/group pair twice per field order. Either schedule qualifies if events and queries are consistently renamed and every key/group pair has equal support over the declared complete set. The latter uses the existing eight variants and adds no count multiplier. Names cannot encode recency, lifecycle or conflict roles. Prefer semantic isomorphs for the key crossing, where complete key-wise outcome histograms can be checked for equality. If some histories instead alter control mechanisms, provide matched controls separately and report actual per-key outcome differences. Query rotation alone does not reassign event keys. Do not describe random key/query shuffling as evidence of shortcut removal.

The two field orders preserve parsed values, event list order, query list order and scalar types. List order has a semantic role for tied events, so event permutation is not a neutral renderer operation. Event ordering variants that change an active tie belong to a counterfactual pair, with newly derived targets and retained lineage.

## Observable counterfactual and relevance checks

The required coverage is explicit matched evidence, not an unseen intention in the generator. Every history should witness decisive before/at retrieval, version over recency, queried conflict resolution, present-null versus absent/deleted-null, and contract-defined absence. Report per-record property truth independently; a family can exercise additional controls in siblings without falsely attaching them to every row.

For the key-name control, require exact role crossing over its declared history/rotation schedule. For the eight query rotations, every base role/signature, after decoding consistent key renames, occupies every ID and slot once for each field order. Confirm role tables and actual status/type/present-null histograms per ID/position. If a role has two comparable variants in the corpus, preserve its explicit pair IDs and delta description. Consistently renamed variants alone do not supply a version-change, input-swap or op-change counterfactual.

Each family must additionally have independently executed relevance and conflict probes, reported separately from training records when they are audit-only:

- Hold protected query/key events fixed and edit another key; semantic answer and selected event identity remain unchanged. Include an other-key high-tuple conflicting event that makes a key-agnostic resolver fail, and mark the row only if its visible task actually contains that witness.
- Hold the protected cutoff and eligible key events fixed and edit future events; semantic answer and selected event identity remain unchanged. Use a future event that would otherwise dominate to demonstrate why cutoff filtering is necessary.
- Append a later, eligible lower version below the protected winner; preserve old indices and exact output for that query. Do not require invariance for an earlier absent query that the new event would make answerable.
- Raise an eligible conflicting competitor's version above the current eligible maximum; only that version changes, and semantic answer must change. This establishes the supplied priority rule as causally decisive.
- Swap active conflicting top `(version,at)` events; input order must change semantic answer. Also compare the neutral object-field-order rendering, which must not change any exact output.
- At a null-valued winner, toggle only put/delete; status changes while the value stays null. This defeats treating null as absence or deletion.

Index shifting under event insertion/removal requires recomputing numeric chosen_event. Compare decoded event identity where appropriate; do not demand a stale numeric index. No invariance applies indiscriminately to every query in a row. Each witness names its protected query signatures and eligibility conditions.

If a new transient delayed-winner claim is added, the public definition requires queries at T-1, T, U-1, U with U >= T+2, a winner at T surviving until U-1, and semantic answer changes at both boundaries. The present eight-role pilot need not claim that extra property. It must not assert coverage from an unqueried intermediate event.

## Honest size and difficulty

Eight output queries increase answer length, not dependency depth: each is an independent filtered argmax followed by an op/status interpretation. A long numerical delay does not add reasoning steps. History size is a context/scan stressor only when decisive evidence and competing events remain identifiable; filler alone is not a higher-level mechanism.

| Level | Supported interpretation for this pilot |
|---|---|
| 1 | A control query retrieves one eligible exact-key event or identifies no eligible event. Record which component is simple, even inside a harder mixed row. |
| 2 | Consistent arbitrary renaming, query rotation and JSON field-order changes preserve the supplied policy. These are nuisance transformations, not new memory mechanisms. |
| 3 | Active time eligibility, competing versions, time/index tie resolution and put/delete/null interpretation require following the declared policy. Certify actual conflicting witnesses and answer changes. |
| 4-7 | Not established by independent query rows, more events, eight rotations, longer delays or a larger rendering count. Multi-record joins, dependent queries, stored procedures, new relevance rules and mechanism-held-out generalization require separate semantics/evidence. |

Track per-history event count, per-query eligible same-key event count, maximum simultaneously competitive top-version events, distinct decisive boundaries, status changes, typed-value changes, active time/index disputes, distractor count/ratio and the time gap. Also report number of queries and renderer views. Changes only to chosen version/index are counted as provenance changes, not semantic-answer changes. Do not assign a complexity level solely from source-family index or development split. Actual task difficulty remains bounded finite retrieval.

The plan has 48 base controlled histories before any key-role renaming within rotations. With eight distinct base signatures in each after decoding those renames, it has 384 base history/query observations, 3,072 rotation-level observations and 6,144 rendered query observations. All are correlated within 12 source families. Eight rotations, consistently renamed key views and two renderers do not supply additional independent histories or mechanisms. Verify actual unique signatures rather than automatically repeating planned totals.

## Review and expansion criteria

Before approval, independently bind each complete visible task and target to its exact record hash; recompute all outputs under the unchanged public contract; check schema, typed values and provenance; and certify every tag with an in-record witness. Audit complete per-family rotation/key tables and status/type histograms at both deduplicated-control and full-rendered scopes. Report any remaining label/signature/time/key correlations, missing controls and split-level family counts. Keep audit probes out of achieved training-record counts.

Initial expert and adversarial reviews must derive their own targets without producer expectations or other reviewer verdicts. This public design is appropriate common input; it is not an adjudication decision. Adjudication occurs only after both final exact-hash review receipts and formal evidence exist. Preserve disagreements and previous versions. A gold-correct row with a false property claim remains a metadata defect rather than silently receiving a competency approval.

Within this pilot, meaningful next expansion would first vary which queries expose decisive updates/conflicts, use new source histories and cross alternative typed values/operations with the same controls. Genuine multi-record relevance, unresolved conflict/abstention, procedure retrieval and dependent read/write tasks would need a new public contract and independent oracle. More filler or random shuffles should not substitute for these mechanisms. Development has exposed familiar retrieval rules; any later held-out-mechanism or learned-memory claim requires separate authorized evaluation. No efficacy, forgetting reduction or architectural benefit follows from data acceptance.

## Frozen evidence and execution boundary

Paths below are relative to `High-Quality_Dataset/`, with SHA256 at design execution:

| Evidence | SHA256 |
|---|---|
| `schemas/EXTENDED_SEMANTICS.md` | `2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23` |
| `reviews/adjudication/ADJUDICATION_EXTENDED_V3.md` | `6b6bcba90e022dd693a3fe14aabf14af6f1030844a8cddb27eb46fa78584effd` |
| `research/research_gap_analysis/PRODUCTION_MILESTONE.md` | `6150a5cfd47713fc0a76170b5390904c4b2ae75700432db04f970a2ead9aab4c` |
| `research/curricula/REM-001.md` | `b0ce82726d4dfc6726b2331ce82a7fd4a238bb1ee8908119a9a3d5d60d0fb112` |

The author read those public/prior evidence files and the user's original directive. No producer/reference/oracle implementation, new candidate, unfinished target or new reviewer verdict was read for this design. No source research, established schema, candidate or release was changed. The actual identity previously served as architect and later adjudicator, with no sample generation or first-review authorship; future adjudication must disclose this design role. Separate executions still share an underlying model and can have correlated errors.
