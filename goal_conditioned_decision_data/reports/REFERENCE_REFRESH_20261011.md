# Bounded reference refresh — 2026-10-11 local

The original RCWM/BWM main reference has no newer remote commit. The complementary branch has one newer commit, which adds an E01R protocol-freeze and verifier-calibration receipt. This update does not supply new transfer, planning, retention, or model-generalization results. The GCDL v2 source/configuration freeze and registered DEV experiment remain unchanged.

Inspection completed at `2026-10-12T01:30:48Z` (`2026-10-11 20:30:48 America/Chicago`). This report supplements the immutable [reference snapshot](reference_snapshot.json), written at `2026-10-11T23:28:39.824780+00:00`; its SHA-256 remains `0117ad20d0ab3eceefcae242c7c7d88c9b65a3a53d28421fa5b9bc957b81ec0e`.

## Reference heads and scope

| Reference | Snapshot head | Remote head at refresh | Finding |
|---|---|---|---|
| `main` (RCWM/BWM) | `ece566dfa31e1e49f93efb264505c3c622c5dad9` | Same | GitHub comparison: identical, ahead 0, behind 0, zero changed files. |
| `research/rineya-complementary-intelligence-v1` | `11981b66aba399be927109a74639f7c73f8232e5` | `1bb6c321eff8a2c710c6c4a382940842a67ad9c0` | One descendant commit; 11 changed/added paths in metadata. |
| External Oct 9 complementary research bundle | Historical repository pin `a223884d55ff47baf3890c7d4a82054d07223215` | Local document bundle | All eight snapshot-listed source files have identical SHA-256 and byte lengths. No `.md`/`.json` document among the 46 locally inventoried documents has a modification time newer than the snapshot. |

The native local repository is `/Users/chengli/Downloads/Rineya_World_Model/rineya-world-model-research`. Its local `main` checkout still points to the earlier Oct 9 commit `a223884d55ff47baf3890c7d4a82054d07223215`; its recorded remote refs still point to the original snapshot heads. Consequently, local refs alone would miss the new complementary commit. Connected GitHub GET requests supplied the current heads and comparisons; no fetch, checkout, pull, push, download, or local ref change was performed. An initial unauthenticated `git ls-remote` returned “Repository not found”; connected GitHub reads succeeded.

A bounded local `git show` hash comparison covered all 25 repository source files listed in the snapshot (22 main and three complementary), totaling 1,181,107 bytes. All match their original recorded SHA-256 values at the locally recorded snapshot heads. The remote main head is identical, so the RCWM/BWM source content is still identical. The remote complementary comparison changes only `complementary/DECISIONS.md` among the three snapshot-listed complementary documents; its README and EXPERIMENT_RESULTS remain unchanged. Selected safe native report/document paths have no local tracked or untracked changes.

## One new complementary receipt

The new [complementary commit](https://github.com/LUNCY579/rineya-world-model-research/commit/1bb6c321eff8a2c710c6c4a382940842a67ad9c0) has parent `11981b66aba399be927109a74639f7c73f8232e5`, commit time `2026-10-12T01:22:03Z` (`2026-10-11 20:22:03 America/Chicago`), and records CX D-37. Only the safe [DECISIONS.md receipt](https://github.com/LUNCY579/rineya-world-model-research/blob/1bb6c321eff8a2c710c6c4a382940842a67ad9c0/complementary/DECISIONS.md#L497) was read for scientific interpretation. The changed file has:

| Receipt | Value |
|---|---|
| UTF-8 bytes | 38,220 |
| Git blob SHA-1 | `13bd796ede806ea501cc2f165ad4e89f475255bc` |
| SHA-256 | `faf6a4ec61de679f1965a35735690ea509a3b50887b3d1538ada0267dd0d7bc0` |
| Relation to snapshot | Exact original 35,634-byte prefix, followed by 36 added lines. |

The receipt states that E01R arm-choice section 2 and R1/R2/R3 recipe specifications were frozen before any Step A output. At that freeze, the chain had zero START lines and no DEV2 prediction file; it was waiting for RAM. The broader protocol sections 4–11 remained drafts, R2 mirrored-pair implementation remained to be added, and remaining review items were to be resolved before the full freeze. These are recorded custody/planning assertions; this refresh did not independently reconstruct run timing or audit those implementation artifacts.

The added receipt also reports one label-blind calibration using three verifiers: 12/12 planted ambiguous wordings rejected, 11/11 clear decoys passed, 23/23 P2 retests passed, and 141/141 training candidates passed. This changes the narrow previous limitation that verifier ambiguity sensitivity had not been measured: the source now reports a finite calibration. The calibration rows, wordings, labels, verifier outputs, and statistical generalization were not independently checked. It does not establish sensitivity to arbitrary future ambiguity, heldout transfer performance, or successful E01R training/evaluation.

The compare metadata also lists new preregistration/recipe/trace/test files and pool/calibration files, plus a changed chain log. Their filenames and change counts were observed solely as comparison metadata. Their contents were not opened. In particular, E01R wording pools, sealed labels, predictions, chain-log contents, and model weights were excluded.

## Effect on the dataset work

The original source findings and their qualification remain in force: BWM 73/110 versus 5/110 and the later 13/110 and 88/110 results are stored-output evidence rather than fresh model/VAL reruns; RCWM ledger gains and token/surface failures remain scoped to their recorded experiments; assisted search/verification and matched-control limitations remain material. The new complementary calibration receipt does not replace any of those measurements or justify a new dataset-generalization claim.

No GCDL source, config, tests, accepted records, sealed records, experiment seeds, thresholds, or frozen hypotheses were changed by this refresh. The parent supplied final suite receipt is 189 tests PASS in 75.309 seconds against frozen 42-file source digest `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`; it is recorded in [PROFILE_COMPATIBILITY_DESIGN.md](PROFILE_COMPATIBILITY_DESIGN.md). The official once-only DEV measurement and its independent review are separate work and are not evaluated in this report. Resource-floor publishing remains paused.

This is a reference-status audit, not a native experiment, external reproducibility study, or approval of a generated upgrade. Reads were limited to snapshot-listed safe source blobs, safe report status, external document hashes/metadata, remote branch/commit/comparison metadata, and the changed public-in-repository decision receipt. No model execution, training/API inference, weights, sealed gold, or unrelated dataset contents were read.
