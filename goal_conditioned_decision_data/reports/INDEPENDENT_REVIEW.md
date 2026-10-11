# Independent adversarial and integrity review — GCDL-001

Reviewer: E/F (`/root/independent_audit`), 11 October 2026. This reviewer authored only this report and `tests/test_adversarial_review.py`, and authored no environment, oracle, generator, label, integrity, or operations implementation. Findings were returned to the responsible authors and the corrected implementation was tested again. This is a scoped review of an experimental synthetic instrument, not automatic approval based on agent agreement.

FINAL_VERDICT: PASS

Scope: the source/configuration snapshot identified below, the 568-record bootstrap inventory, and promotion of the measured DEV option-position correction. This permits bounded synthetic production using the reviewed code and its gates. It does not certify learned decision intelligence, real-world generalization, calibration, retention, unbounded optimality, live remote backup, or indefinite host availability. Source/configuration changes require revalidation; future records still require every executable gate. A fresh agent audit of future releases is separate work.

## Independence and custody

The primary and differential checkers share their engineer and specification. The differential checker imports neither the primary oracle nor environment transition/scoring helpers: it uses ledger tuples/deltas, Blocksworld stack lists, separate register arithmetic, and separately written goal/constraint/exact-fraction aggregation. This provides differential evidence, not independent authorship.

The reviewer independently specified and hand-calculated semantic tasks outside the standard generator vocabulary. Their expected outcomes do not come from the generator. Integrity mutation tests may start from a valid executable baseline to test rejection of corrupted disclosures; these are not additional independent mathematical proofs.

No sealed target JSONL was parsed or inspected by this reviewer, and no sealed answers were used to change a generator. Sealed checks were limited to file hashes, freeze commitments, counts and protected indices. Same-user filesystem custody is procedural, not security isolation. The coordinator's explicit mechanical release-auditor path separately rechecks sealed targets.

## Final executable evidence

```sh
python3 -m unittest goal_conditioned_decision_data.tests.test_adversarial_review -v
python3 -m unittest discover -s goal_conditioned_decision_data/tests -v
```

The independent suite passes **31/31 tests**: 14 semantic, 8 integrity and 9 operations tests. The final complete suite passes **56/56 tests**, with no skipped cases or expected failures hiding a failing gate.

Independent tests verify reverse/retain semantics; literal `void` meaning settled while another token means void; signed entries; unchanged dynamics under goal reversal; uniform three-world success probability 1/3; duplicate/contradictory world rejection; invalid booleans/bounds/statuses/cycles/branching; genuine ties; all-infeasible abstention even when the final state repairs an initial violation; unrepaired historical constraint violations; illegal-prefix termination and successful-action-only cost; Blocksworld clearance; register add/multiply/set/copy/swap and reject/clamp/wrap controls.

They also verify option/identifier invariance; namespace canonicalization with entity names `goal`, `amount` and `rules`; finite-world order invariance; false provenance/utility/uncertainty rejection; matched-search universe/budget checks; finite-proof scope/witness checks; explicit quarantine export refusal; nested oracle-key leakage refusal; immutable publication replay/conflict; resource reservation; metadata-only recovery/conflict; descendant watchdog cleanup; Git scope guards before credentials; error-token redaction; and failed all-quarantine sealing refusing to mint a freeze.

Other tests add randomized ledger/register differential checks, exhaustive three-block forests/two-step actions, deliberate transition/scoring/cost mutants, JSON persistence invariance, paused no-op behaviour, three unproductive batches checkpointing then pausing, and mid-batch source-drift refusal.

## Bootstrap inventory and separate sample inspection

The reviewer mechanically re-audited **all 352 nonsealed accepted records** for schema, both executable results, evidence hashes, decision/search/proof labels, option order and input/target separation: **zero errors**. Counts are TRAIN 240, DEV 54, OOD 58, from 12 seed clusters and 48 protected parents. Family counts are A 36, B 24, C 12, D 88, E 72, F 24, G 48, H 48. These demonstrate implemented instruments, not sufficient coverage of every difficulty/rule combination.

All 20 bootstrap manifests pass hash/count and protected cross-split index checks. The complete inventory contains **568 unique accepted IDs**, including **216 sealed records** in eight committed seed shards, and 80 protected parents. No ID, parent, template or normalized task fingerprint crossed splits. The sealed rows were hash/index checked here, not semantically inspected. The complete manifest-hash-map digest is `ed2afe0853387b1bcb13a0bfdaf1f493d13adaa00c910512202f96c804d49666`; sealed freeze SHA-256 is `7f3e2d8cbea2ca79b61b9f4e762760be23e53eeeb40c347d9851a6f350ec2f39`.

Separate inspection used `random.Random(610711)`, sorted seed clusters, cyclic ledger/Blocksworld/register strata, and a random record within each assigned environment. This inspected 12 records from 12 distinct seed clusters, four per environment. Visible plans, endpoints, exact utility, legality, costs, constraints and complete best sets were checked. Zero errors were observed.

| Seed | Record | Independently checked evidence |
|---|---|---|
| 1000000 | `gcdl_27b3efc6a890a4a4e2f8e2b9e783bd0b` | Ledger 14+4=18; transfer 10/11; missing entry fails at step one with zero cost |
| 1000001 | `gcdl_c797d1f42e7811975450ed2a4ddafd42` | Identical three-move alternatives correctly tie at (1,3,-3); prefix satisfies two atoms |
| 1000002 | `gcdl_4f4763512b12799812423ea086e28f57` | Register 26+3=29 versus 23; unknown register is illegal |
| 1000003 | `gcdl_f783f89f64124f76a5a5639ec4b80e7f` | Settle satisfies goal but violates forbidden status; feasible noop outranks it |
| 1000004 | `gcdl_d37774f1a25e58b0a51d7a92bb961f93` | Six moves satisfy six atoms at cost 12; reversed first move is already at destination |
| 1000005 | `gcdl_f22aee8e2a48382f30b19387e4729b0e` | Register 18+10=28 versus 8; noop stays 18 |
| 1000006 | `gcdl_75cc618ded54b18a3b9e31cf6c20b86b` | Ledger 85+12=97; transfer 73/53; settle wins expanded selection |
| 1000007 | `gcdl_3940138d0ec33f96c56f9668c69656a8` | Same-destination move fails; legal three-move plan reaches goal at cost 3 |
| 1000008 | `gcdl_d4b3abf16f0ce506c0ed398d032509c3` | Register 73+11=84 versus 62; unknown target preserves initial state |
| 1000009 | `gcdl_ad33446061394104b94da65c3e569f47` | Minimum 3736 exceeds available balances; initial violations require abstention |
| 1000010 | `gcdl_0fea51b3b8a41c67f496c0143fabfc6a` | Reversed prefix reaches goal then fails at step three; (0,6,-12) loses to legal (1,6,-36) |
| 1000011 | `gcdl_c0e2f4e29087e9e0f8589dcf4bae77c6` | Register 100+19=119 versus 81; correct cost/legality ordering |

Variants/parents sharing a seed are not independent error observations. Zero failures in 12 independent representative observations gives the illustrative one-sided 95% binomial upper bound `1 - 0.05**(1/12) = 22.09%`. This small stratified, template-heavy sample is not representative IID sampling of future data, so this is **not a dataset-wide error guarantee**. Exhaustive mechanical agreement cannot bound shared specification mistakes.

## Corrected findings

All critical findings raised by this reviewer have been corrected and retested:

1. All-infeasible sets retain rankings but recommend no candidate; initial/intermediate violations cannot be repaired away.
2. Uncertainty, abstention, utility/provenance, matched-search universe/budgets, proof scope and witness bounds are verified rather than trusted.
3. Semantic role names were removed from inference IDs. Status/entity normalization preserves namespaces. The generator author also corrected in-memory aliasing and added JSON persistence tests.
4. Rule reversal includes a settled-entry pair that changes best choice in TRAIN/DEV. Restrictive sealed/OOD constraints legitimately alter it and the tags scope that difference.
5. The watchdog controls the process group; publication reserves storage; recovery rejects conflicting metadata; repeated paused invocations are no-ops.
6. Reviewed source hashes are checked at startup/between batches; mid-batch changes halt acceptance. Unproductive batches/oracle disagreements pause with evidence. Empty/failed sealed shards cannot freeze.
7. Backup checks branch/directory scope before credentials, uses an ephemeral environment, redacts known tokens/credential URLs, and bounds transport retry/backoff under a shared deadline.

Git and watchdog fault tests use mocks and temporary state. Launchd and backup code were inspected; live restart/login/sleep recovery and a real push must be separately demonstrated by operations. Process-interruption replay does not establish arbitrary power-loss/filesystem durability.

## Controlled improvement decision

PROMOTION_VERDICT: PASS

The independently rerun DEV comparison uses 32 paired seed groups starting at 410000. Winner positions change from 32/0/0/0 to 8/8/8/8, maximum position fraction 1.0 to 0.25 and entropy 0 to 2 bits, with zero normalized semantic changes and zero oracle label changes. Opaque-ID checks pass; the lowest-lexical-ID heuristic succeeds 9/32 (28.125%), a limited diagnostic rather than proof against arbitrary shortcuts. The stored measured-promotion receipt names the already frozen sealed commitment.

Promotion is supported **only for removing this option-position correlation in ordinary decision records**. Prefix-search examples retain order because their computation policy depends on it. No model was trained and no learning/generalization gain was measured.

## Reviewed snapshot and limits

SHA-256 digest of the canonical `source_hashes()` map covering 25 Python/configuration files:

`9492bb7f5163d45fba93c3bf89e5fb5d063b0c27bf9b6b6f6a2b9ee2dcb666be`

Operations must retain the matching full map in `manifests/VALIDATED_CODE.json`. Key pins:

| Artifact | SHA-256 |
|---|---|
| `oracles/primary.py` | `0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19` |
| `oracles/independent.py` | `0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae` |
| `environments/core.py` | `b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d` |
| `generators/build.py` | `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575` |
| `auditors/integrity.py` | `1ce9a68a04d2eb71c3d3bcb8188dd3d392af69cda105b0caa408be9c6c53a62d` |
| `pipelines/worker.py` | `4130b27b7332c6fba9d0c58decd4e950d240817bc044a9823f96d1f8e40ee076` |
| `tests/test_adversarial_review.py` | `807a1f3161ea6c7f58f33ec9a5f2def5e448ca853f2412947f56384238abd5bd` |
| `DATA_CONTRACT.md` | `e92b9e720aa70cd46589eb1c2c8c0ee7d7523f36e338d3f1581ab20fba7f5762` |
| `manifests/schema-1.0.json` | `2a0e5540ccbb444cd7d4083874ac0ce94050e2f7a6ac10ecfb8a9fa3e321afa0` |

The generator remains narrow: additive ledger/register primitives, small Blocksworld forests, closed renderers, finite candidates and search bounds. Non-additive register controls are tested but sparse in production. OOD combines structural shifts, so future scores cannot isolate one causal factor without additional matched strata. Fingerprints do not decide arbitrary logical equivalence; a finite shortcut battery cannot prove absence of every shortcut.

Uniform-world probabilities are exact conditional facts, not empirical calibration for an unknown distribution. Search operation counts are not model FLOPs; runtime evidence stays separate from targets. Active-batch time divided by accepted records measures combined generation/verification/persistence cost, not isolated audit execution time.

No training, GPU experiment, paid API, model download, or other-programme modification occurred in this review. The worker may run while the logged-in host is awake; it does not imply an indefinitely active AI research team. These limits are part of the PASS scope.
