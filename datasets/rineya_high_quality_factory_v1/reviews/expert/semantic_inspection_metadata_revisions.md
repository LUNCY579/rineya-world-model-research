# Independent fresh metadata revision inspections

Execution `/root/expert_reviewer` independently recomputed every visible prompt and full target, exact record hash, canonical schema and relevant factual tag in RGM expansion-v3 and RVM pilot-v4. All420 candidates pass and receive ACCEPT:360 RGM expansion-v3 rows and60 RVM pilot-v4 rows. Original prompt/task/gold/split/renderer/format/difficulty are exactly identical to their independently reviewed v1/v2 baselines. These batches add zero independent semantic parents. First checkpoints and completed exact-hash receipts are preserved separately for each batch.

Blinding limitation: before this revision, parent communicated a supplemental reviewer concern/count about ambiguous earlier block-stack tags. An adjudicator later sent aggregate prior expansion decision counts. No other reviewer verdict files or producer/reference/oracle implementation were read, and no sample-specific RVM verdict was supplied. New label facts and all targets were independently computed from actual prompt/on/goal/world data. This limitation is disclosed rather than presenting this followup as fully blind.

## RGM expansion-v3

All360 targets are freshly recomputed using expert-owned graph and block semantics. Graph labels distinguish184 genuinely branching rendered worlds from26 single-successor rendered worlds. Node count alone is not used as evidence of branching. Block labels are literal and independently counted:118 rows have at least two initial TABLE roots,18 have one initial root but at least two explicit goal TABLE roots, and14 require only the generic `block_goal_composition` tag. Valid/invalid/goal and unreachable tags are also checked against independent replay/search.176 rows change tags from v1; no world or target changes.

Detailed actual examples:

- `RGM-001-expansion-v3-p0030-base-forward`: Cedar has outgoing Dune and Beryl edges, and several later nodes branch. The branching tag is factual. Submitted Dune,Aster reaches goal with11 fuel; fresh Beryl,Aster wins the two-move lexicographic tie.
- `p0040-base-forward`: Aster,Dune,Cedar form a directed cycle with exactly one successor per node. The corrected `single_successor_resource_graph` tag is literal; empty start-at-goal plan remains valid/goal-reaching with shortest[].
- `p0032-base-forward`: birch and dahlia are both on TABLE initially, so `multiple_initial_table_stacks` is factual even though the submitted plan is empty and already satisfies the goal. It does not claim that the row requires nontrivial stack construction.
- `p0034-base-forward`: only amber is initially on TABLE; only birch is explicitly on TABLE in the full goal. The new generic `block_goal_composition` tag avoids a multi-stack assertion. Moving amber to TABLE remains illegal because it already lies there and supports dahlia. The exact six-move fresh solution is unchanged.
- `p0054-base-forward`: initially only dahlia is on TABLE, supporting birch,amber,cobalt. The partial goal explicitly puts amber,birch,dahlia on TABLE, three explicit goal roots; `multiple_explicit_goal_table_roots` describes that fact without claiming multiple initial stacks. The proposed amber→birch is unchanged/blocked and illegal. Fresh solution clears cobalt,amber,birch to TABLE. Goal omits cobalt, but the solver still respects its support constraints when planning.

Earlier detailed dependent search, seven-step resource exhaustion, partial-goal and terminal-goal-with-invalid-last-action examples remain in the initial v1 report. Exact prompt/task/target equality plus fresh per-record recomputation establishes that those semantics remain applicable. The earlier same-model/proxy/difficulty/shortcut limitations remain attached; metadata correction does not change curriculum mechanism coverage. RGM expansion-v2 is diagnostic and is not approved by the completed v2 receipt file.

## RVM pilot-v4

All60 targets/hashes are freshly checked against visible prompts, and exact dynamics/gold equality is established against pilot-v2. Ten rendered equal-route mode tags change: six correctly become `no_equal_length_goal_routes` and four become `equal_length_goal_routes`. The latter identifies actual feasible equal-length routes; exact path counting checks all fuel-bounded move lengths. The immutable released six-record metadata erratum remains preserved and no release bytes are rewritten.

Detailed examples use `RVM-001-pilot-v4-p0000` through `p0006`, `base-forward`:

- p0000: Fjord→Beryl2, Beryl→Dune3, Beryl→Fjord1, fuel12. Goal paths have2/4/6 moves and5/8/11 cost, one sequence per length. `no_equal_length_goal_routes` is factual. Correct shortest repair Beryl,Dune and duplicate candidate ranking ties remain unchanged.
- p0006: Cedar→Dune→Aster costs8 and Cedar→Fjord→Aster7, both two moves within21. `equal_length_goal_routes` is factual. Fresh Dune,Aster wins the lexicographic tie despite higher fuel cost.
- p0001: direct Ember→Cedar costs10 with fuel9, so the direct move fails; the three-move Aster,Dune,Cedar costs exactly9 and is feasible. Fuel-precondition semantics and ranking remain exact.
- p0002: a valid return detour does not alter the fresh INITIAL repair objective; optimum remains Ember,Cedar, not an edit or suffix completion.
- p0003: chain/direct goal cost14 with fuel13; fresh repair null. All labels distinguish null progress from numeric zero and illegal-stop state from a wrong goal claim.
- p0004: start Cedar equals goal; correct empty candidates tie best and every repair[]. A start self-move is still illegal even though executed node remains goal.
- p0005: Aster,Fjord versus Ember,Fjord are two-move feasible routes; Aster wins full lexicographic sequence despite spending more fuel. Fresh repair applies to every submitted candidate, including a correct one.

Source/receipt/script hashes, factual check counts and exact example IDs are in `verification_summary_expansion-v3.json` and `verification_summary_pilot-v4.json`. The script is `independent_metadata_revisions.py`, using expert-owned public-prompt semantics and path counters only. Acceptance applies to these exact candidates; release approval remains separate.
