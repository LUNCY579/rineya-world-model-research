# Rineya Goal-Conditioned Decision Learning — GCDL-001

This self-contained programme produces reproducible synthetic decision records with executable outcome evidence. It never fine-tunes a model, uses paid inference, or downloads model weights. All implementation and records live inside this directory on `research/goal-conditioned-decision-data-v1` in an isolated worktree.

The current release is an **experimental verified synthetic pilot**. Coverage of A–H is an instrument design, not evidence that Rineya has acquired general decision intelligence. The current scientific context, including confirmed results and shortcut failures, is pinned in [the architecture report](reports/RESEARCH_ARCHITECTURE.md) and [source snapshot](reports/reference_snapshot.json). [The research plan](RESEARCH_PLAN.md), [data contract](DATA_CONTRACT.md) and [role/custody protocol](AGENT_ROLES.md) define scope and claim limits.

Three deterministic environments implement ledger accounting, Blocksworld and integer registers. Every accepted candidate is executed by a primary oracle and a separately implemented checker; the pipeline independently reconstructs utility, rankings, finite-world probabilities, search cost and bounded proofs. The checker implementations share their engineer and specification. A separate reviewer authored adversarial, hand-calculated tests and challenged the integration; repeated LLM agreement is not ground truth.

Canonical records separate `input` from `targets`, provenance and audit evidence. Candidate IDs are opaque. Related goals, counterfactuals and surface variants stay in protected world/template groups. Reserved world sizes and constraint combinations separate TRAIN, DEV, SEALED and OOD. Normalized entity/status/option-order fingerprints suppress duplicates across parents. The finite normalization is not a universal semantic-equivalence solver.

`manifests/schema-1.0.json` is the interchange schema; the dependency-free authoritative structural/semantic validator is `auditors/integrity.py`. Immutable JSONL shards, SHA-256 manifests and exact rational labels preserve evidence. Rejected records are quarantined and excluded from training exports. Accepted shards stay below 8 MiB (repository limit: 25 MB). Reproducible candidate caches are excluded from Git and removed only after their hashes, seeds and configs are recorded. Accepted data and evidence are never rotated.

## Commands on this host

The initial development checkout was `/Users/chengli/Downloads/Rineya_World_Model/.rineya-gcdl-worktree`. Its programme/state were byte-verified into `state/development_archive.tar.gz` (manifested), then this self-created redundant checkout was retired to reclaim 729 MiB. Existing main/other-agent research folders remain intact. For host persistence, the canonical runner worktree is `/Users/chengli/.codex/gcdl001-host/worktree`; run the commands there. It uses an isolated shallow, sparse clone of the same confirmed branch so a daemon need not access Downloads or another programme’s Git metadata. No model weights or native experiment folders are fetched.

```sh
python3 -m unittest discover -s goal_conditioned_decision_data/tests -v
python3 -m goal_conditioned_decision_data.auditors.release --include-sealed
python3 goal_conditioned_decision_data/scripts/control.py start
python3 goal_conditioned_decision_data/scripts/control.py status
python3 goal_conditioned_decision_data/scripts/control.py stop
python3 goal_conditioned_decision_data/scripts/control.py resume
python3 goal_conditioned_decision_data/scripts/control.py once
python3 goal_conditioned_decision_data/scripts/backup.py
python3 goal_conditioned_decision_data/scripts/export.py --split train --view listwise
python3 goal_conditioned_decision_data/scripts/export_training.py --split train --view listwise --max-records 100 --variant-budget 3
python3 -m goal_conditioned_decision_data.pipelines.worker --render-only
```

After changing the runner environment or interval, use `stop` followed by `resume` to reload launchd’s cached configuration.

`resume` clears the durable pause/stop markers, then rechecks disk, storage, reviewed source hashes and all other gates. Resolve the reported cause first. It does not relax a resource or quality gate. `once` processes a short resumable cycle. `start` requires frozen evaluation and a validated source/config receipt. `control.py seal` is the explicit auditor-only bootstrap path; it creates the sealed pool once and never replaces an existing freeze. Normal production recovery reads only sealed hashes and protected manifest indices, never its target labels. All agents use the same user account; sealed access isolation is procedural.

## Persistence and resource bounds

The launchd user agent `com.rineya.gcdl001` invokes one bounded cycle every five minutes. SQLite stores the queue, protected identity index, checkpoints, production ledger and error events. An atomic heartbeat, metrics and backup receipt live under ignored `state/`. Each worker holds an exclusive lock; the watchdog terminates its process group on timeout and cleans up command descendants. Interrupted jobs replay from stored seeds. A complete immutable shard + manifest is authoritative; recovery rejects conflicts rather than silently replacing bytes or labels.

`configs/production.json` limits one CPU worker, six seed groups per cycle, 180 seconds of production plus bounded Git backup, 512 MiB programme storage, 12,000 accepted records, and a minimum 10 GiB of free disk. Publication reserves storage before acceptance. Quotas, corrupt evidence, oracle disagreement or backup errors checkpoint and pause. Repeated launchd invocations while paused are no-ops. No accepted data is deleted to satisfy a quota. Keep candidate diversity and independent quality above volume.

Every completed nonsealed batch also enters a resumable rendered-view queue. At most twelve source groups per cycle produce listwise, computation and distillation views with three actual prompt variants. The queue prioritizes recent batches and uses remaining capacity for historical accepted batches. `--render-only` performs one bounded backfill cycle. Immutable `manifests/render-batch-*.json` receipts define completed exports; incomplete directories are excluded. Recovery verifies canonical, export-manifest and derived-shard commitments before trusting a completed receipt. Changing the approved executable source creates a new derived specification, so versions must be counted separately. Canonical records and frozen evaluation bytes never change.

The runner operates while this macOS user session is logged in and the computer is awake. It resumes at login and after sleep; it does not keep the machine awake. The unattended worker is a deterministic programme, not an indefinitely active AI agent. Fresh research interpretation, code improvements and independent agent review require an available agent session; unreviewed code/config changes stop production. The controlled DEV comparison demonstrates an audited option-position correction, not autonomous model learning.

A Codex heartbeat in this chat is scheduled every two hours for quiet health checks and bounded, independently reviewed improvements. It respects stop markers and reports only actionable changes. Its service availability is separate from the local CPU worker.

Git backup commits only this programme and pushes only the dedicated branch, with an explicit rebase before pushing. It uses the already configured repository-owner credential in a subprocess environment, never changes global authentication and never persists secrets. Confirmed remote/local commit equality is recorded. Global repository config and other programmes remain untouched.

## Evidence and future experiments

`reports/QUALITY_REPORT.json`, `reports/INDEPENDENT_REVIEW.md` and `reports/PRODUCTION_STATUS.md` report actual counts, tests, gates, known limits and operating status. `reports/IMPROVEMENT_EXPERIMENT.json` records a frozen DEV before/after comparison. `manifests/SEALED_FREEZE.json` and `manifests/VALIDATED_CODE.json` pin evaluation and approved production sources. Pointwise, pairwise and listwise exports use separate input/target fields and no audit metadata; export does not train anything.

For model-visible surface variation, use the versioned rendered exporter rather than the legacy structured-input exporter. Rendered examples have separate `input_text` and `target` fields. Only `input_text` is a model input; IDs, split, provenance, rendering metadata and verification are nonmodel fields. Outer JSON sorting preserves the embedded prompt's actual nested key order and whitespace. Ordinary comparisons use candidate permutations independent of gold labels; candidate-prefix search keeps its declared order. Computation and distillation include their exact stored auxiliary targets with finite proof scope. Pointwise and pairwise exports are available on demand. See [the rendering design](reports/SURFACE_RENDERING_DESIGN.md) and [actual-schedule controlled comparison](reports/CONTROLLED_SURFACE_COMPARISON.md).

Rendered row counts measure repeated surface presentations and supervision views, not new reasoning worlds. The comparison supports format preservation and reduced answer-position concentration. The goal-blind C heuristic and shared entity/status vocabulary remain known limitations; no model-performance claim follows from this change.

Future 0.6B decision-head experiments should compare direct choice, matched candidate consequence scoring and distillation under the same available plans. Evaluate goal reversal, lexical controls, held-out rule conjunctions, larger worlds and horizons separately. Report exact success/regret, compute counts, finite-world probability scores, abstention and retention of existing skills. Use fresh model predictions to study world-model errors. A future model's sealed results must not feed generator optimization; once used that way, freeze a new independent instrument. Full training requires separate authorization.

The [next generator-version design](reports/NEXT_GENERATOR_VERSION_DESIGN.md) proposes fixed candidate bundles under three non-initial goals, separate lexical controls and compatible archived profiles. It is unimplemented; complete experiment specifications, bounded DEV evidence and fresh independent review are required before promotion.
