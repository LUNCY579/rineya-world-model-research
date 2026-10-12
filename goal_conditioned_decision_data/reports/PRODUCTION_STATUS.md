# GCDL-001 production status

Snapshot: 2026-10-12T00:49:36.430810+00:00. The canonical worker is **resource-paused**, with its checkpoint and registered launchd schedule preserved. This remains an experimental verified synthetic pilot; no model was trained or evaluated.

| Split | Accepted | Groups | Natural production quarantines |
|---|---:|---:|---:|
| train | 1320 | 44 | 0 |
| dev | 297 | 11 | 0 |
| sealed | 216 | 8 | 0 |
| ood | 319 | 11 | 0 |

Total accepted: **2,152**, up from the earlier 920-record status snapshot. Four deliberately corrupted demonstration records remain quarantined separately; they are excluded from training and natural rejection rates. All 66 nonsealed accepted shards and all eight frozen sealed shards were byte-preserved during promotion.

The full canonical release audit rechecked all 2,152 accepted records, including sealed gold through the explicit auditor path, with zero errors, all A–H families present, and every release gate passing. This validates executable evidence and the implemented split/fingerprint checks; it does not establish model generalization or complete shortcut resistance. Sealed answers were not used for adaptation. QUALITY_REPORT.json contains the current complete result, actual counts and sampling limitations.

## Reviewed training-view improvement

The promoted source digest is `3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12`. All **114 regression tests** pass in the canonical host checkout; 58 preserve the original core, 16 test the renderer, nine test integration, and 31 were authored by the fresh independent reviewer. The independent scope is code, regression and mechanical evidence. The original core receipt/review and historical quality report remain byte-preserved under `manifests/validation_history/` and `reports/review_history/`.

The versioned exporter provides pointwise, pairwise, listwise, computation and distillation views with separate `input_text` and `target` fields. Only `input_text` enters the model. Three actual JSON key/layout/framing variants survive outer JSONL serialization. Gold-independent candidate permutation rechecks pair signs and exact labels; search-prefix records retain their policy order. Computation/distillation expose the previously omitted exact search/proof targets, with their finite optimality scope preserved.

The automatic queue publishes listwise, computation and distillation views for at most twelve nonsealed source groups per bounded cycle. Accepted-subset batches remain usable when other candidates are quarantined; zero-eligible auxiliary views have explicit zero-row manifests. Immutable canonical receipts, derived manifests, source hashes and per-batch completion receipts support replay after interruption and reject corrupt/symlinked paths. Counts distinguish rendered presentations from new reasoning worlds.

A fresh independent comparison used the actual production stride-six DEV/OOD shards: **616 accepted source records -> 1,848 serialized/reparsed listwise presentations**, with zero parse, oracle, label or mutation mismatches and three surviving key orders per source. In the actual close-call samples, DEV's fixed slots 0/11 become 18/15; OOD's 11/0 become 13/20 across three presentations per source. A separate 32-group-per-split reviewer experiment corroborates reduced position concentration. Individual variants still have imbalance; exact balance is not claimed.

The shared entity/status vocabulary and goal-blind C plan-relationship heuristic remain unresolved. In actual DEV, that C heuristic still selects a best-set member in 11/11 sources and 33/33 renderings. These findings and a future generator-version hypothesis are documented, with no model-performance claim. See CONTROLLED_SURFACE_COMPARISON.md, SURFACE_RENDERING_DESIGN.md and INDEPENDENT_REVIEW.md.

## Current host and recovery state

The canonical isolated worktree is `/Users/chengli/.codex/gcdl001-host/worktree`, backed by its own sparse repository on `research/goal-conditioned-decision-data-v1`. Native RCWM/BWM/complementary and other-agent folders remain untouched. The retired initial checkout's 99-file byte-verified archive remains preserved.

Checkpoint: train seed **1000065**. The canonical generation queue is empty. **66 nonsealed source groups are queued for rendered backfill; zero real rendered datasets have been published** because the resource gate remains closed. The real new export CLI independently refused publication under the current free-space condition. Successful export tests used small temporary fixtures with simulated healthy free space and dedicated refusal checks; they are not evidence of live production publication.

At this snapshot, free space was **9.02 GiB**, below the required **10 GiB**; programme storage was 30,409,483 bytes under its 512 MiB ceiling. The pause cause is retained in state/PAUSED.json. Accepted data is never deleted to meet a quota. Launchd `com.rineya.gcdl001` remains registered at five-minute intervals, and paused invocations are no-ops.

The quiet two-hour Codex follow-up now rechecks a disk-only pause and may resume once real space and all other review/resource/integrity gates pass. It preserves operator STOP and unresolved quality/backup pauses. After recovery, a bounded `python3 -m goal_conditioned_decision_data.pipelines.worker --render-only` cycle can catch up historical views. README.md contains exact control/export commands. The host must be logged in and awake; neither the AI service nor future uptime is assumed.

Git backups remain restricted to this programme and its dedicated branch, with scoped changes, bounded transport retries and confirmed remote/local commit equality. The final backup receipt is in state/backup.json. No global authentication changes, paid APIs, model weights or training were used.
