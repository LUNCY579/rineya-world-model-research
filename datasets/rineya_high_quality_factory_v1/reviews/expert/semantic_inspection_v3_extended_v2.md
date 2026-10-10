# Independent expert review: RGM pilot-v3 and extended-pilot-v2

Actual reviewer execution: `/root/expert_reviewer`. I reviewed every new canonical rendering:72 relational rows and168 extended rows, representing36 relational task variants and84 extended semantic parents. Receipts are bound to each candidate's exact `record_hash` in `pilot-v3.jsonl` and `extended-pilot-v2.jsonl`. All240 decisions are ACCEPT with documented low limitations. This is no root release approval.

I read public `SEMANTICS.md`, `EXTENDED_SEMANTICS.md` and the relevant public curricula. I read no producer private/reference implementation, calibration expectation, adversarial or adjudication verdict. Correctness was computed using this execution's own prompt-derived implementations, not an external independent oracle. `independent_extended.py` imports only my earlier independent graph/block verifier and shared receipt/type-equality utilities. Each visible Task JSON is parsed, checked against the canonical task, executed, and compared with the entire target using type-sensitive equality. All extended prompts contain the complete public mechanism contract plus an explicit common scalar-typing rule. Every new target matches.

The expert semantic inspection covered every mechanism and edge-case mode, with detailed examples below. I inspected the12 forward tasks in each extended track, supplemented by direct target inspection for the diagnostic representatives listed below. Programmatic recomputation covers all reverse renderings as well. This is actual-agent inspection in a separate execution, not a claim of120 separate human solutions. The shared underlying model family remains a correlated-error limitation. Clarity, required competency, canonical alternative handling, difficulty, proxy scope and shortcuts were assessed in addition to target parity.

## Relational pilot-v3

All36 forward task variants were summarized and all six new matched task payloads/targets were inspected in detail. The original graph/block modes remain independently covered by the earlier documented p0001 fuel constraint, p0002 assembly, p0003 unreachable, p0004 invalid tail, p0005 cycle, p0006 missing edge, p0007 blocked source, p0010 initial goal, p0011 full chain, p0012 block detour, p0013 partial chain, p0014 initially satisfied partial goal, and p0015 lexicographic tie. Their exact v3 IDs have prefix `RGM-001-pilot-v3-` and suffix `-base-forward`.

Exact new representative IDs and independent findings:

- `RGM-001-pilot-v3-p0008-matched-forward`: the six chain moves consume16 fuel from20, end at Nacre with4, and are both valid and goal-reaching. Its base counterpart legally returns to Quasar with16 and does not reach the goal.
- `RGM-001-pilot-v3-p0009-matched-forward`: clearing onyx, sapphire, pearl and quartz in dependency order makes ruby clear; placing ruby on onyx satisfies the partial goal. Removing the repeated no-change move avoids the base counterpart's early stop.
- `RGM-001-pilot-v3-p0018-matched-forward`: initial fuel41 permits the direct cost20 edge from Sirocco to Vesper. A one-move fresh plan is optimal despite a feasible longer chain, and the final fuel is21. The base counterpart has fuel19 and no feasible goal plan.
- `RGM-001-pilot-v3-p0019-matched-forward`: the five stack-reversal moves satisfy every goal support. The base counterpart's sixth repeated support move is illegal, even though its final on-map satisfies the goal.
- `RGM-001-pilot-v3-p0028-matched-forward`: costs1+2+4+4+1+3 total15, exactly the initial fuel. The six-move plan ends at Umber with0. Its one-move base counterpart is valid but unfinished.
- `RGM-001-pilot-v3-p0029-matched-forward`: the initial partial goal already holds, so empty supplied and fresh plans are valid and goal-reaching. The base move legally destroys the goal support.

Each family's base/matched/rendered variants stay in one split. Among12 forward development variants, six reach a goal and six do not: matched positive outcomes remove the earlier universal non-goal outcome associated with development names/size. Goal satisfaction is still checked separately from legality and from shortest-plan optimality. Planning-horizon difficulty labels now account for required fresh plans, including the six-move p0028 task. No difficulty7 label is present. The matched p0029 empty-plan control is still labeled3 despite trivial execution; its difficulty_quality is3 with a low boundary note. Repeated motif scope remains moderate.

## RSM: supplied reusable procedures

Detailed exact representatives: `RSM-001-extended-pilot-v2-p0000-base-forward`, `...-p0002-base-forward`, `...-p0005-base-forward`, `...-p0008-base-forward`, and `...-p0009-base-forward` (the ellipses here mean the same `RSM-001-extended-pilot-v2` prefix).

In p0000, bump is entered twice and executes two additions, giving x1/y4. p0002 executes gate(false), fold(scale-by-one then simultaneous swap), gate(false), giving x-6/y-3 with four primitives. p0005's nested bump entries are recorded: call_order is gate,bump,gate,bump,bump; only three arithmetic primitives count. In p0008, threshold initially passes at x2, sprout increases x to3, weave swaps to x1/y3, then two threshold tests fail and reduce y. Final x1/y1, conditions true/false/false and primitive count7 are correct. In p0009 the main y<0 test runs first; weave swaps the negated y into x3, both later x<3 tests fail, and cascade ultimately produces x8/y3 with six primitives.

Calls share state and return before the next instruction; call/if do not count as primitives. All procedure-call graphs, including unused branch calls, are validated as acyclic. Only the chosen branch executes. This measures supplied finite procedure composition, not learned options, argument discovery or interruption recovery. State-dependent multi-condition rows receive moderate difficulty-boundary notes where label3 understates a richer composition.

## RMM: capability routing and protected writes

Exact detailed representatives: `RMM-001-extended-pilot-v2-p0000-base-forward`, `RMM-001-extended-pilot-v2-p0001-base-forward`, `RMM-001-extended-pilot-v2-p0008-base-forward`, and `RMM-001-extended-pilot-v2-p0009-base-forward`.

In p0000, arithmetic admits alpha/beta and priority/ID chooses alpha. Unsupported ocean selects null and attempts no writes, yielding blocked_writes[], even though its proposed writes include protected keys. Empty needs admits every module and chooses gamma with priority0. Protected seal/budget writes at indices0,2,3 are blocked, and note becomes2. In p0001, budget is unprotected and the final empty-needs request sets it102. In development p0008, pixels chooses nova over orion; empty needs chooses lumen over nova by ID at equal priority; ledger+index chooses meridian over orion. Protection preserves anchor1008/reserve7 while memo becomes3. p0009 correctly allows reserve103.

The prompt specifies all eligibility, priority, abstention and sequential-write behavior. This is declared competence, not measured competence or retention. The repeated unsupported middle request, empty-needs suffix, fixed write pattern and initial protected anchor values make diversity and shortcut resistance2. Exact world uniqueness does not remove that limitation.

## REM: versioned as-of retrieval

Exact detailed representatives: `REM-001-extended-pilot-v2-p0000-base-forward`, `REM-001-extended-pilot-v2-p0001-base-forward`, `REM-001-extended-pilot-v2-p0003-base-forward`, `REM-001-extended-pilot-v2-p0008-base-forward`, and `REM-001-extended-pilot-v2-p0009-base-forward`.

For p0000 at as_of10, version2 at5 defeats the later version1 at9. At as_of14, version3/time12/delete/input-index4 defeats the same-version/time put at index2. In p0001 the put is the later same-version/time input entry and wins instead. p0003's version3 put at13 has value null: status is present, not deleted or absent. At21, version4/time20 wins; the missing exact key stays absent. Future and other-key records do not enter a query's eligible set.

All outputs include the decisive version and input event index. The development histories contain8 or10 events rather than5–7, with held-out key names. However the added queried-key version2/time19 event is excluded by as_of14 and superseded by version4/time20 for as_of21; added other-key events likewise never affect the supplied answers. This is a valid longer visible history, with weak new decisive dependency. Development generalization_value is2 and the receipts recommend an as_of19 query. Repeated cutoffs/version/value patterns yield diversity/shortcut resistance2. No claim of persistent or learned memory is supported.

## RBM: exact finite beliefs

Exact detailed representatives: `RBM-001-extended-pilot-v2-p0000-base-forward`, `RBM-001-extended-pilot-v2-p0001-base-forward`, `RBM-001-extended-pilot-v2-p0002-base-forward`, `RBM-001-extended-pilot-v2-p0004-base-forward`, and `RBM-001-extended-pilot-v2-p0009-base-forward`.

p0000 keeps all four worlds and reports color values in canonical scalar-JSON order: "blue",1,null,true. In p0001, equality to integer1 keeps w0, excluding boolean true. Consecutive equality to1 then true in p0002 gives belief_sizes[1,0]; the empty belief is not vacuously entailed. p0004 keeps w3 and entails a present null color, distinguished by counts[{value:null,count:1}] and entailed:true despite guaranteed:null. Development p0009's portal=open observation keeps scenario1/3/5; hue counts null1/true2 and sigil counts known1/unknown2 both permit distinguishing queries, while portal is entailed open.

Canonical scalar ordering and explicit empty/entailed-null distinctions make targets unique. No posterior probability, entropy, noisy observation policy or information-cost objective is asked for. This is deterministic possible-world filtering and count-based query separability.

## RJP: exact predictive abstraction

Exact detailed representatives: `RJP-001-extended-pilot-v2-p0000-base-forward`, `RJP-001-extended-pilot-v2-p0002-base-forward`, `RJP-001-extended-pilot-v2-p0003-base-forward`, `RJP-001-extended-pilot-v2-p0004-base-forward`, `RJP-001-extended-pilot-v2-p0008-base-forward`, and `RJP-001-extended-pilot-v2-p0009-base-forward`.

All p0000 observables are0, so every finite action sequence observes0 and every state belongs to one class despite differing transitions. In p0002 the lone observable1 at b distinguishes a from c from d by one/two/three steps; iterative refinement yields four singleton classes. p0003's canonical classes are [a,d] and [b,c]. p0004 preserves two classes for true and integer1 despite Python's ordinary bool/int equality. Development p0008 refines six states to six singletons. In p0009, initial zero-observable states split into [u,z] and [v,w], while [x,y] remains together; canonical class IDs and pulse/rest/turn rows follow the declared ordering.

Full transition maps are total and deterministic. Canonicalization removes equally named alternative partitions, and sorted action output removes serialization ambiguity. Supplied finite equivalence is correct; no latent encoder/JEPA/energy training is measured. State-list shuffles and typed/iterated cases give shortcut_resistance4 while generalization/diversity remain3.

## RDP: canonical assignment repair and greedy diversity

Exact detailed representatives: `RDP-001-extended-pilot-v2-p0000-base-forward`, `RDP-001-extended-pilot-v2-p0001-base-forward`, `RDP-001-extended-pilot-v2-p0002-base-forward`, `RDP-001-extended-pilot-v2-p0003-base-forward`, `RDP-001-extended-pilot-v2-p0005-base-forward`, and `RDP-001-extended-pilot-v2-p0008-base-forward`.

The independent implementation enumerates every assignment (eight train combinations,81 development combinations before constraints). p0000 has all eight feasible and min_distance0 permits duplicate repairs; all four candidates are selected in ID order. p0001 has three feasible assignments; a fully birch candidate needs one edit to birch,birch,ash. Greedy diversity accepts the two original-valid distinct assignments first and suppresses duplicate repaired outcomes. p0002 has two feasible assignments and min_distance2 allows only the first valid assignment because the alternatives differ by one position. p0003's empty allowed set makes feasible_count0, all repairs/distances null and selection empty. p0005 has two feasible assignments; equal-distance repairs choose ash,ash,birch lexicographically, and min_distance4 exceeds slots3, so only the first accepted candidate survives. Development p0008 has18 feasible assignments and selects c8920,c4322,c8425 under the stated greedy order.

Both repair minimality and canonical ties are exact. The contract explicitly distinguishes greedy diversity from maximum packing; alternative equal-distance repairs are resolved lexicographically. This is static assignment repair, not learned diffusion or plan-prefix repair. IDs/order changes can create exact unique worlds without creating a different underlying constraint mechanism.

## ROM: observation-conditioned agent beliefs

Exact detailed representatives: `ROM-001-extended-pilot-v2-p0000-base-forward`, `ROM-001-extended-pilot-v2-p0002-base-forward`, `ROM-001-extended-pilot-v2-p0004-base-forward`, `ROM-001-extended-pilot-v2-p0005-base-forward`, `ROM-001-extended-pilot-v2-p0008-base-forward`, and `ROM-001-extended-pilot-v2-p0009-base-forward`.

In p0000 both agents observe all moves and have correct final beliefs, yet Bex abstains because its priority list is empty. In p0002 no event is observed: Ada's key belief drawer is false against actual shelf, while Bex's initially false shelf belief becomes correct when the actual key moves there. Unknown book is not labeled a false belief. p0004's all-null Ada abstains; Bex also abstains with an empty priority list despite a known key belief. In p0005 Ada misses the final private book move and falsely believes drawer while Bex correctly believes desk. Development p0008 leaves all Cora's known beliefs false and all Eli's beliefs unknown; Dax abstains because its priority list is empty. In p0009, Cora updates coin, Dax updates map, Eli updates vase; each policy acts from its own final belief without omniscient location leakage.

All policy actions are uniquely specified. Empty priorities, initially false beliefs, unknowns and late/private correction are meaningful controls. The repeated movement scaffold and explicit first-known-object policy limit diversity2; intentions/emotions/cooperation or second-order beliefs are outside this proxy.

## Coverage, uniqueness and scope

Independent checks found12 exactly unique semantic worlds and12 exactly unique full targets per extended track. The two field-order renderings in each parent intentionally share the semantic task and one split. All relational family splits are consistent. Challenge size axes are supported: procedure libraries3→4; declared modules3→4; possible worlds4→6; automata states4→6 and actions2→3; assignment slots3→4 and symbols2→3; social agents2→3, objects2→3 and events3→5; episodic histories5–7→8–10. Changed names are visible limited shift claims, not mechanism novelty. No difficulty7 or sealed evaluation claim appears in these batches.

Clarity and target correctness are5 after independent verification. Generalization is generally3, diversity2–3 depending on repeated templates, and shortcut scores2–4. All receipts contain both individual independent target evidence and actionable low issues. These limitations constrain downstream conclusions while leaving the finite tasks valid for the explicitly stated competencies.
