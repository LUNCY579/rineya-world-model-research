# GCDL-001 production status

Snapshot: 2026-10-11T18:44:26-05:00. Experimental verified synthetic pilot; no model training or learned-transfer claim.

| Split | Accepted | Groups | Production quarantines |
|---|---:|---:|---:|
| train | 240 | 8 | 0 |
| dev | 54 | 2 | 0 |
| sealed | 216 | 8 | 0 |
| ood | 58 | 2 | 0 |

Total accepted: 568. Four separate deliberate corruptions were quarantined to demonstrate failed-outcome, ranking, leakage and uncertainty gates; they are excluded from training exports and natural production failure rates.

The final suite passes 56 tests, including 31 independently authored adversarial/semantic/operations tests. Oracle properties cover 4,272 plans and detect deliberate transition/scoring/cost errors. The release auditor rechecks every accepted row; independent review reads nonsealed labels only and separately verifies sealed manifest hashes and protected groups. All 20 bootstrap groups reproduce exactly from stored seeds/configurations.

Families A–H are represented. Dataset coverage, difficulty, normalized uniqueness, uncertainty categories, source versions, option-position distributions and release gates are in QUALITY_REPORT.json. Independent manual review sampled 12 distinct seed clusters, four cases per environment, with no observed issues. Its conditional one-sided 95% IID error-rate upper bound is about 22.1%; shared specification errors remain outside that estimate.

The DEV option-position comparison contains 32 paired cases: 32/0/0/0 before versus 8/8/8/8 after; zero semantic or decision-label changes. This supports the narrow correction. ID-only lexical choice was 9/32; no broader shortcut or model-performance claim follows.

Bootstrap active generation/audit time: 17.001 seconds. Active-time throughput is distinct from wall-clock production. Runtime manifests record separate cheap/expanded simulator nanosecond samples and exact attempted-transition/candidate counts; these CPU timings do not represent future language-model inference cost. No paid APIs or model inference are used.

Resource settings: one worker; six groups every five minutes; 180-second production budget plus bounded Git backup; 512 MiB programme ceiling; 12,000 accepted-record ceiling; minimum 10 GiB free disk. Three consecutive unproductive batches, mismatched oracle evidence, source drift, unsafe publication, or a failed bounded backup preserve a checkpoint and pause. Paused invocations do not grow logs.

## Host activation evidence

Pending final receipt, remote confirmation and live launchd cycle. The CPU runner executes while logged in and awake. The separate two-hour Codex follow-up is scheduled and respects operator stops; neither guarantees AI-service availability.

Exact lifecycle and recovery commands are in README.md. State/heartbeat.json, metrics.json, checkpoint.json, queue.sqlite and backup.json retain current operational evidence; shards and manifests retain remote-recoverable scientific evidence.
