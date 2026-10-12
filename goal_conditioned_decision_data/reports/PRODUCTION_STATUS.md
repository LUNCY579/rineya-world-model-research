# GCDL-001 production status

Snapshot: 2026-10-12T01:39:19.585806+00:00. The canonical worker remains **resource-paused**. The reviewed v2 implementation is promoted, but **all 2,152 accepted records are still v1**. The frozen v2 experiment verified 160 cases in memory; no v2 canonical records or real rendered datasets were published. No model was trained or evaluated.

| Split | Accepted | Groups | Natural production quarantines |
|---|---:|---:|---:|
| train | 1320 | 44 | 0 |
| dev | 297 | 11 | 0 |
| ood | 319 | 11 | 0 |
| sealed | 216 | 8 | 0 |

Four deliberately corrupted demonstration records remain quarantined separately and excluded from training and natural rejection rates. All 66 nonsealed shards, eight sealed shards, 74 batch receipts and the sealed freeze were byte-preserved: 149 artifacts in [V2_PROMOTION.json](../manifests/V2_PROMOTION.json). Prior v4 approval and quality evidence remain under `manifests/validation_history/` and `reports/review_history/`.

## Reviewed generator improvement

The exact 42-file approved source/configuration digest is `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`. The root proposal suite, fresh independent suite and canonical post-promotion suite each passed **189 tests**, with unchanged source maps. The canonical run took 128.950 seconds. [INDEPENDENT_V2_REVIEW.md](INDEPENDENT_V2_REVIEW.md) contains the scope, adversarial checks, exact source map and mandatory frozen evidence commitments.

V2 supplies three full, distinct non-initial Blocksworld goals over the same six ordered candidates, with nonempty disjoint exact best sets and a feasible successful alternative for each goal. Candidate and nested-key rendering order share a goal-independent bundle identity. Complete entity/status strings come from split-reserved pools, with explicit familiar DEV controls. Archived v1 curricula and exact profile custody preserve historical records, IDs, targets and fingerprints. V2 admission quarantines the complete generated batch if any row or parent check fails; legacy v1 retains its accepted-subset policy. Source, review and frozen experiment evidence must all match before unattended publication.

The once-only frozen DEV experiment used 32 C parents (96 goals) and 32 lexical pairs (64 presentations), with zero exclusions or oracle/label/reproduction mismatches. The tested goal-blind relationship selector selected a best-set member in 26/32 v1 tasks and 32/96 v2 tasks; singleton exact-set recovery was 21/32 and 29/96. V1 has five candidates and v2 six, so the across-version counts include changed construction. Within each v2 triplet, a fixed single choice can succeed for at most one of three disjoint best sets. The reserved DEV arm used 48 distinct complete strings with zero TRAIN-pool overlap and preserved exact decision projections. [V2_CONTROLLED_EXPERIMENT.md](V2_CONTROLLED_EXPERIMENT.md) records denominators, all ties, uncertainty and limits. These are mechanical coverage results; learned generalization and goal-aware terminal shortcuts remain untested.

## Release and publication gates

The full explicit release-auditor path rechecked all **2,152 accepted records**, including the unchanged sealed gold, with zero errors and every release gate passing. Sealed answers were used only for this audit, never generator adaptation. [QUALITY_REPORT.json](QUALITY_REPORT.json) contains actual counts, family/difficulty coverage and sampling limitations. Its PASS verifies implemented evidence/integrity checks on the existing synthetic pool, not complete shortcut resistance or model performance.

The real publication guard and new export CLI refused publication because free space was below 10 GiB; no data files were created. [V2_LIVE_PUBLICATION_GATE.json](V2_LIVE_PUBLICATION_GATE.json) records the actual refusal without simulated disk space. Successful publication tests used bounded temporary fixtures with explicit simulated healthy space. At this snapshot, free space was **1.43 GiB**; programme storage was 30,944,239 bytes under its 512 MiB ceiling. Free space varies with other host activity. The original disk pause cause is preserved, and accepted data is never deleted to meet a quota.

## Host recovery and backup

The canonical isolated worktree is `/Users/chengli/.codex/gcdl001-host/worktree` on `research/goal-conditioned-decision-data-v1`. Other-agent and native RCWM/BWM/complementary directories remain untouched. The read-only [reference refresh](REFERENCE_REFRESH_20261011.md) found unchanged RCWM/BWM main and one complementary preregistration/calibration receipt, with no new model-evaluation evidence.

Checkpoint remains train seed **1000065**. The canonical generation queue is empty; all 74 completed jobs resolve to v1. **66 source groups await rendered backfill** under the new reviewed specification; 66 older pending specifications are marked superseded, with old evidence preserved. Zero rendered datasets are complete. New jobs will select v2 after the resource floor and every review/integrity/storage gate pass; old jobs keep their stored profile. Derived presentations count separately from canonical worlds.

Launchd `com.rineya.gcdl001` remains registered at five-minute intervals; paused invocations are no-ops. The quiet two-hour Codex follow-up rechecks a disk-only pause and may resume the authorized worker when real space and all gates pass. Operator STOP and unresolved quality/backup pauses remain binding. [README.md](../README.md) contains exact start/stop/status/resume and bounded render-only commands. Operation requires the host to be logged in and awake; AI-service availability and future uptime are not guaranteed.

Git backups remain restricted to this programme and its dedicated branch, with bounded retries and remote/local HEAD confirmation in `state/backup.json`. No global authentication changes, force pushes, paid APIs, model weights or training were used.
