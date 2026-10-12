# PREREG_FREPLAY: registered CPU replay of every logged world-model (F) query (F-09)

Written 2026-10-11 ~16:15 CDT. At the time of writing, the registered replay script (`scripts/freplay.py`) does not
exist and `runs/FREPLAY/` does not exist. The sha256 of this file is recorded in `runs/FREPLAY/PREREG_HASH.txt`
before the registered replay runs, and it is checked again by the script at run time.

CPU only. No model is loaded. The inputs are existing outputs, read only. The ground truth is the exact simulator
`bwmn.blocks.step`, and the planner's own parser is `bwmn.blocks.parse_forward_delta`.

## 0. Disclosure: exploratory numbers already exist (not blind)

F-09 (DECISIONS) and `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` report unregistered scratch replays of the same
caches. The scripts are kept in `research/audit_evidence/bwm_next_scratch/` (`freplay.py`, `legality.py`,
`forest.py`, `fcache_replay.py`, `unparsed*.py`). The numbers already known are:

- **B2-hard:** 20,530 unique keys; 0 illegal actions accepted; 0 wrong next states; 6 false-illegal, all in
  hard/instance-102.
- **H2-hard:** 22,895 keys; 5 false-illegal, also in instance-102.
- **F's `legal` flag says "legal" on illegal keys:** 798/4,066 (19.6%) in B2-hard; 657/2,363 (27.8%) in H2-hard;
  1,078/4,498 (24.0%) in the mirror; 10/77 in PB-602; about 22% in the B3 distillation.

This replay is therefore NOT a blind test. What it registers, before the registered run, is the definitions, the
unit, the strata, the per-instance rule, the readings and the wording consequences. The numbers that enter claims then
come from one fixed procedure, and that procedure applies unchanged to later runs (§9).

**Known differences from the scratch scripts.** Any difference in the registered numbers is reported, not reconciled.

- (a) The scratch scripts lumped every unparseable answer together. Here, the support check, malformed answers and the
  F flag are separated.
- (b) `legality.py` read the flag as `json.loads(t).get("legal")`, with no schema check.
- (c) Duplicate keys across shards were resolved by glob order.
- (d) `forest.py` used the union of all shards' caches.

**Name clash.** The readings R1 and R2 below are not the item-7 arms R0/R1/R2.

## 1. Inputs (read only; every sha256 is pinned; a mismatch aborts the run)

The F is identical in all five sets: `runs/BW_Fdelta_s0/adapter` (sha 2ed0bb00…), delta format.

- `FREEZE_B2_beam.json` (412a6ee5…) defines it.
- `FREEZE_B3_H2.json` (a433e499…) changes only `pa`.
- `scripts/bw_distill.py search` loads the `FREEZE_B2_beam` F.

**F caches.** One cache per shard; for the distillation, one cache for all 240 problems.

| File | Lines | sha256 |
|---|---|---|
| `runs/BW_TEST_B2/hard_beam/shard_0of8/fcache.jsonl` | 2417 | `c84563c20fec05028b1a28a271848a6081b98c54f919bf379c242d44c07283f1` |
| `runs/BW_TEST_B2/hard_beam/shard_1of8/fcache.jsonl` | 2551 | `8931869e07489758ebc2df55855691367ac6690de6dc5562922ff7a06c41a73c` |
| `runs/BW_TEST_B2/hard_beam/shard_2of8/fcache.jsonl` | 2534 | `54db54c9ab01fd753b494376959b630d8bec7fb1464052c9b8dcc5a43d3ad170` |
| `runs/BW_TEST_B2/hard_beam/shard_3of8/fcache.jsonl` | 2493 | `ba4868f2b19fb708457e1bda95858f7ddc380b1accb39049f918e15d43ea7123` |
| `runs/BW_TEST_B2/hard_beam/shard_4of8/fcache.jsonl` | 2685 | `18b5135e49756cd46c92aef130797f2727c76d3a0aa6eb72cccca3647b5776c3` |
| `runs/BW_TEST_B2/hard_beam/shard_5of8/fcache.jsonl` | 3149 | `5d2f833fa7e535294df0c8f5f26457cfdbbee80c82fc350f9aeb2defefb15f00` |
| `runs/BW_TEST_B2/hard_beam/shard_6of8/fcache.jsonl` | 2130 | `7baf775dbe026ac1675a9eecbcbdcc718151fbaee860824c931572e411ec7098` |
| `runs/BW_TEST_B2/hard_beam/shard_7of8/fcache.jsonl` | 2595 | `8481815e1735f2f01b9565eb4aafa552f8c2b39f7fa2edaf597913d2606b6ce1` |
| `runs/BW_TEST_B2/mirror_beam/shard_0of8/fcache.jsonl` | 3014 | `54a71e26ecc30963ba38a26444c4adfaa95aed317845632adf85e77fa9342df3` |
| `runs/BW_TEST_B2/mirror_beam/shard_1of8/fcache.jsonl` | 2647 | `0d3380fb00276c27c842605821a72e57b70bc9c51334b5d8962e8223c5142e41` |
| `runs/BW_TEST_B2/mirror_beam/shard_2of8/fcache.jsonl` | 2356 | `1b5568e43421f0d636b9b402d937345df9e8d28ab9f8453c7302334d88da59c0` |
| `runs/BW_TEST_B2/mirror_beam/shard_3of8/fcache.jsonl` | 2747 | `e70a17d9d058f06d6b73d3378cffb956efdcd832a1be95474a3f70878d52c7d7` |
| `runs/BW_TEST_B2/mirror_beam/shard_4of8/fcache.jsonl` | 3686 | `99416006d1f298fc29fd9cbe72fecd48c97771437e7869acc3fed22b08261a08` |
| `runs/BW_TEST_B2/mirror_beam/shard_5of8/fcache.jsonl` | 3082 | `0fdfe929af2ec772909da5c66d26bdfddfca5d68f01a00b31e411950dfec7421` |
| `runs/BW_TEST_B2/mirror_beam/shard_6of8/fcache.jsonl` | 3378 | `6a2e8f086c94f7b5fa83007c89fe7b52687b4b0d644388f1e03d8ce2dd5812c7` |
| `runs/BW_TEST_B2/mirror_beam/shard_7of8/fcache.jsonl` | 2842 | `97563de12c847669849be906b93db169ad9da6961751c85f20f9a1b3654ad600` |
| `runs/BW_TEST_B2/planbench_beam/shard_0of4/fcache.jsonl` | 466 | `2c9cb1f6553423ecf613ec7fe225190ba687b92ad38a44912ccdd99f86f033c4` |
| `runs/BW_TEST_B2/planbench_beam/shard_1of4/fcache.jsonl` | 424 | `1fd852d9e16d0c23a2d54cea1cd8ae23d4f3ca121ad835ab034a3edf1baddfbf` |
| `runs/BW_TEST_B2/planbench_beam/shard_2of4/fcache.jsonl` | 454 | `7c115df45577c3222cb7235a66857f450cf6b8515f988770172d6b4c96cbcc75` |
| `runs/BW_TEST_B2/planbench_beam/shard_3of4/fcache.jsonl` | 442 | `f8ca5d7190c6da98aa08b0aa753090140ea2dfc69a88852003aec162b7c51287` |
| `runs/BW_B3_HARD/h2_beam/shard_0of8/fcache.jsonl` | 2642 | `c7b9dfb605868e53f1d0b6bac6068a80e5bbc87779bf53f44ec89e51b809e58a` |
| `runs/BW_B3_HARD/h2_beam/shard_1of8/fcache.jsonl` | 3036 | `28c5e6ee0d08ebd4a0895add7e2a88f5a6d98dce5ccc87edca17042242b8ca4b` |
| `runs/BW_B3_HARD/h2_beam/shard_2of8/fcache.jsonl` | 2988 | `13ccb217f8a54a56bb15c7035cb9e643a55ca0290d38885b162ae657cb2c2717` |
| `runs/BW_B3_HARD/h2_beam/shard_3of8/fcache.jsonl` | 2890 | `f1f2bf098263f283bde582aec94290a0f4bee0ab3f04852884b72f87a710054a` |
| `runs/BW_B3_HARD/h2_beam/shard_4of8/fcache.jsonl` | 2919 | `52b312b8722e2e1476cf4a82fdc8093c1895687229d4c50076859d8d0753bb4f` |
| `runs/BW_B3_HARD/h2_beam/shard_5of8/fcache.jsonl` | 3057 | `39c361eed0eff68fa78b3820f2f1cad957a6465f2e2db16401e6427b17275cc0` |
| `runs/BW_B3_HARD/h2_beam/shard_6of8/fcache.jsonl` | 2395 | `5b5418bd565da7f4036ace70bda362188cf28fad2cd918b6ec66f56809dd5bab` |
| `runs/BW_B3_HARD/h2_beam/shard_7of8/fcache.jsonl` | 2983 | `2c354964f8f58556313482fdf96d9708486d7121659db2150aeadbf7cf6bd998` |
| `runs/BW_DISTILL_r1/fcache.jsonl` | 15030 | `055d3b667f74e1675f92c0ea9855d300b08fbd99f67c6acb40ac62b0e7b78b46` |

**Instance outcomes and initial states** (merged plan files; the per-shard `plans.jsonl` give the shard of each
instance and are hashed at run time):

| Set | Outcomes | Initial states |
|---|---|---|
| `b2_hard` | `runs/BW_TEST_B2/hard_beam/plans.jsonl` 85812972… | `bench/planbench/blocksworld_hard/generated/*.pddl` (110 files, digest d7b7e446…) |
| `b2_mirror` | `runs/BW_TEST_B2/mirror_beam/plans.jsonl` 22147f07… | `runs/BW_val_large/instances_mirror_hard.json` 2f61e38b… |
| `b2_pb602` | `runs/BW_TEST_B2/planbench_beam/plans.jsonl` 839142ab… | `bench/planbench/blocksworld/generated_basic_3/*.pddl` (101, 46aceac7…) and `generated_basic/*.pddl` (501, 4807eb8b…) |
| `h2_hard` | `runs/BW_B3_HARD/h2_beam/plans.jsonl` 3cf3cd01… | as `b2_hard` |
| `distill_r1` | `runs/BW_DISTILL_r1/solutions.jsonl` fdf80151… | the same file (`init`) |

- **Directory digest:** sha256 over the lines `"<file name> <sha256 of file>\n"`, sorted by file name.
- **Code:** `bwmn/blocks.py` b1e28ed5… The pinned `bwmn/blocks_planner.py` (18a5f22b…) is the semantics reference
  but is not imported, so no `bwmn.lm` or torch import happens.

## 2. Unit of analysis

The unit is the **unique (state, action) key within a set**.

- **Key:** state = `frozenset(tuple(f) for f in r["s"])`, action = `tuple(r["a"])`, text = `r["text"]`.
- **Reading a file:** each file is read exactly as `LMForwardBW` loads its cache. A line that fails `json.loads` is
  skipped and counted. Within a file, the last line for a key wins (the planner's memo semantics).
- **Duplicates across files of a set:**
  - If every occurrence has the same text, the key is one unit.
  - If texts differ, the key is **conflicting**. It is classified under each distinct text, and the set-level unit
    takes the most severe class, in the order false_legal_accepted > wrong_state > false_illegal >
    false_legal_rejected_by_support_check > malformed > ok.
  - The number of conflicting keys and their class pairs are reported.
- **Occurrence-level counts** (one per file × key) are reported as a secondary.

## 3. Classification (mutually exclusive and exhaustive)

**Definitions.**

- **Truth:** `(ns, legal) = B.step(s, a)`.
- **Planner parse:** `p = B.parse_forward_delta(text, s)`, which gives `(next_state, flag)` or None.
- **Schema parse `q`:** the same steps as `parse_forward_delta`, without the support condition `dele <= s`. These
  steps are `json.loads(text.strip())`, the key set exactly `{add, del, legal}`, `legal` a bool, and each of `add` and
  `del` mapped through `tuple(x.split())`, with any exception giving None. The result is `(add, dele, flag)` or None.
- **Consistency check (asserted for every key):**
  - q None ⇒ p None;
  - q not None ⇒ (p None ⇔ not dele ⊆ s);
  - p not None ⇒ p = ((s − dele) ∪ add, flag).

| Truth | Answer | Class (subtype) |
|---|---|---|
| legal | p not None, flag true, p state = ns | **ok** (`ok_legal_exact`) |
| legal | p not None, flag true, p state ≠ ns | **wrong_state** |
| legal | q not None, flag false (whether or not the deletions hold) | **false_illegal** (`fi_flag`) |
| legal | q not None, flag true, p None (deletion not in s) | **false_illegal** (`fi_support`) |
| legal | q None | **false_illegal** (`fi_malformed`) |
| illegal | p not None, flag true | **false_legal_accepted** |
| illegal | p not None, flag false | **ok** (`ok_illegal_flag`) |
| illegal | q not None, flag false, p None | **ok** (`ok_illegal_flag_unsupported`: F's flag already says illegal) |
| illegal | q not None, flag true, p None | **false_legal_rejected_by_support_check** |
| illegal | q None | **malformed** |

**Harmful** = false_illegal ∪ false_legal_accepted ∪ wrong_state.

**Second code path (asserted for every key).** The planner-effective outcome is "stop" if p is None or the flag is
false, else ("go", p state). ExactF's outcome is "stop" if not legal, else ("go", ns). A key is harmful ⇔ the two
outcomes differ. `plan_beam`/`rollout_many` treat None and flag false identically (stop), and ExactF returns
(s, False) on illegal actions, which is also a stop.

**Secondary per-key metrics** (reported, no reading attached):

- **F-flag accuracy:** among keys with q not None, flag = legal. It is reported separately from the parser checks
  (RCWM's adopted practice, D-75 F-09 follow-up).
- **Delta-exact rate:** q's (add, del) equal the true (ns − s, s − ns), and flag = legal.

## 4. Strata and statistics

Each set is reported in total and by stratum:

- block count `n = len(B.blocks_of(s))`;
- action type `a[0]` (pick-up, put-down, stack, unstack);
- `n` × action type.

Every stratum reports counts by class and subtype, and these rates with exact two-sided Clopper-Pearson 95% CIs
(α = 0.05; null when the denominator is 0):

| Rate | Numerator / denominator |
|---|---|
| harmful | harmful / keys |
| false_illegal | false_illegal / legal keys |
| wrong_state | wrong_state / legal keys |
| false_legal_accepted | false_legal_accepted / illegal keys |
| **support_share (R2)** | false_legal_rejected_by_support_check / illegal keys |
| flag_false_legal | (false_legal_accepted + false_legal_rejected_by_support_check) / illegal keys |
| malformed | malformed / keys |

- **Pooled set `all5`** (secondary): the union of the five sets' keys, with the same conflict rule.
- **Rounding:** rates and CI bounds to 6 decimals.

## 5. Per-instance imagined search forest

- **Instance shard:** the shard whose `shard_*/plans.jsonl` lists the instance id. For `distill_r1`, it is the single
  cache. The cache used is that shard's file, read as in §2 (last line wins), i.e. exactly the memo the planner had.
- **Forest:** a search from the instance's initial state.
  - For every visited state s, each cached key (s, a) of that cache is in the forest.
  - If `parse_forward_delta(text, s)` is not None and its flag is true, the predicted state is visited.
  - Every key the planner queried for this instance is in its forest. `plan_beam` reaches states only through
    F-accepted predictions from `init`, and all of its F queries go through the memoised cache. The forest can also
    hold keys that another instance of the same shard queried from a coinciding state, so it is a conservative
    over-approximation.
- **Coverage check** (reported; failures are listed, not dropped):
  - every action of the submitted plan, walked from `init` through the cache's F-accepted predictions, is a cached
    key in the forest;
  - for verified plans, the walk ends in a state that satisfies the goal.
- **Flagged instance:** its forest holds ≥ 1 harmful key. The record lists the harmful keys by class.
- **Support-exposed instance** (secondary): its forest holds ≥ 1 false_legal_rejected_by_support_check key. Without
  the check, an illegal action would have been accepted at a state the search reached.
- **Also reported per instance:** set, id, block count, success (executor), verified, fail reason, forest states,
  forest keys, and class counts within the forest.

## 6. Registered reading R1 (oracle-equivalence)

**Statement.** An unflagged instance's exact-F run is identical to the logged run, by construction. On every key it
could query, the learned F's planner-effective outcome equals ExactF's (§3, second code path). By induction over the
rounds, `plan_beam` therefore makes the same queries and the same choices.

- **Assumption (D-50):** proposer outputs are a function of (state, goal, id, round label, eval_seed) and do not
  depend on batch or shard composition (per-instance CRN seeds). The round labels coincide when the beams coincide.

**Per set, report:**

- **Lower bound on exact-F successes:** L = successes among unflagged instances.
- **Upper bound:** U = L + the number of flagged instances.
- If every flagged instance was solved, U equals the observed success, and **the exact-F ceiling is at most the
  observed success.** Whether it is exactly equal needs an exact-F re-run of the flagged instances only, which is GPU
  work and not part of this replay.

**Consequence (advisory, for the B4 draft, D-70 answer 5):** if U equals the observed success on a set, an exact-F
oracle arm can show no headroom there. B4 should then drop the oracle arm, or restrict it to flagged instances.

## 7. Registered reading R2 (support-check dependence)

**Statement.** R2 = false_legal_rejected_by_support_check / illegal keys, per set. It is the share of illegal queried
actions whose "illegal" outcome came from the support check rather than from F's own flag.

**Wording consequence, registered in advance.** The threshold is a point estimate above 5%, applied to each claim's
own set:

| Claim | Set |
|---|---|
| P-08/P-10 hard: "every plan S-beam verified in imagination was valid" | `b2_hard` |
| P-10 mirror and PB-602 verification statements | `b2_mirror`, `b2_pb602` |
| P-14: "every plan the beam verified in imagination was valid (88/88)" | `h2_hard` |
| P-11 / B3: label precision 1.000 (211/211), "no simulator labels" | `distill_r1` |

- **If R2 > 5% in a claim's set:** the claim must say the plan was verified "by the learned F plus a domain-agnostic
  delta-consistency check (a predicted deletion must be a fact of the state)".
- **If R2 > 5% in any set:** the phrase "the domain rules are known only to the learned F" is withdrawn from every
  document that states or quotes it. The pinned `blocks_planner.py` docstring itself is not edited; its sha is pinned.
- **If R2 ≤ 5% in a set:** that set's claims keep their wording, with R2 reported beside them.

## 8. Not claimed; limits

- **The forest is an over-approximation.** Proposals were not logged, so the exact per-instance query set cannot be
  rebuilt. The exact-F outcome of flagged instances is not computed (GPU).
- **Only queried keys are measured.** The caches hold only the keys the planner queried. F on unqueried keys is not
  measured, and these are not F accuracy estimates for any other distribution.
- **Not blind** (§0).
- **No independent implementation in this run.** An independent second implementation from this text alone, compared
  byte for byte, is the D-61/D-71 practice. It is still to be done.
- **Byte-compare target:** the `core` block of `FREPLAY.json`, serialised as `json.dumps(core, sort_keys=True,
  separators=(",", ":"), ensure_ascii=True)`. Its sha256 is recorded in `FREPLAY.json` and `FREPLAY.md`.
- **Logistics is out of scope.** F-09 records why the check does not transfer there.

## 9. Outputs and later inputs

- **Outputs:** `runs/FREPLAY/FREPLAY.json` (`core`, `meta` with every input sha and the prereg sha, the per-instance
  table, conflicts and checks), `runs/FREPLAY/FREPLAY.md`, `runs/FREPLAY/RUN_LOG.txt` (every invocation, including any
  failed one), and `runs/FREPLAY/PREREG_HASH.txt`.
- **Later inputs** (item-7 R0/R1/R2, B4 and later runs): each set is added by an amendment that names its files and
  shas before any of their outputs are read. The definitions in §§2-7 stay unchanged.
