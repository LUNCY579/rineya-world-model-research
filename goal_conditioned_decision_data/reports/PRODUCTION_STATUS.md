# GCDL-001 production status

Snapshot: 2026-10-11T19:01:23-05:00. The canonical host runner is healthy. This is an experimental verified synthetic pilot; no model training or learned-transfer claim.

| Split | Accepted | Groups | Production quarantines |
|---|---:|---:|---:|
| train | 480 | 16 | 0 |
| dev | 108 | 4 | 0 |
| sealed | 216 | 8 | 0 |
| ood | 116 | 4 | 0 |

Total accepted: 920. Four separate deliberately corrupted records were quarantined and excluded from training exports and natural production failure rates.

All 58 regression tests pass, including 33 independently authored adversarial/semantic/operations tests. Oracle properties cover 4,272 plans and detect deliberate transition/scoring/cost errors. The release auditor rechecks accepted labels/evidence and protected splits; its independent-review gate binds the exact current source hashes and independent review receipt bytes. The independent reviewer did not inspect sealed gold. Its initial hand-calculated audit sampled 12 distinct seed clusters, with zero issues and a conditional one-sided 95% IID error upper bound of 22.1%; this does not bound correlated specification errors.

All A–H families are present. QUALITY_REPORT.json contains current counts, difficulty, normalized uniqueness, uncertainty, option-position coverage and release gates. Independent review approves the code and the narrow DEV position correction: 32/0/0/0 winners before, 8/8/8/8 after, zero semantic/label changes. ID-only lexical choice was 9/32; no broad shortcut or model-performance claim follows.

Active generation/audit time recorded: 43.789 seconds. This differs from wall-clock throughput. Each search manifest retains cheap/expanded runtime samples plus exact attempted-transition and candidate execution counts. These CPU simulator costs do not predict future language-model inference cost. No paid APIs, inference service or model training are used.

## Live host evidence

The canonical isolated worktree is `/Users/chengli/.codex/gcdl001-host/worktree`, backed by its own 9 MiB shallow sparse repository. Launchd `com.rineya.gcdl001` completed a real generation/checkpoint/Git-backup cycle. Checkpoint advanced from 1000011 / 568 accepted to 1000017 / 744; a later verified cycle reached 1000023 / 920. The daemon itself confirmed local and remote commit equality. HOST_DEPLOYMENT.json preserves that observed evidence. Long-term uptime is not inferred from two successful cycles.

A startup PATH omission was corrected and tested. The original daemon stalled opening its script under Downloads, so the host runner uses an unprotected isolated checkout and its own Git metadata. Its redundant initial development checkout was byte-verified into a 441,667-byte archive (99 files, SHA-256 manifest), then retired to reclaim 729 MiB. Existing native research/other-agent workspaces remain intact. Accepted shards are preserved both locally and remotely.

The host's free space briefly fell below the 10 GiB floor and the runner correctly paused. Free space recovered; the guarded runner resumed. Current free space at this snapshot: 13.50 GiB. The guard is unchanged.

One CPU worker runs six groups every five minutes, with 180 seconds of production plus bounded Git backup, a 512 MiB programme ceiling and 12,000 accepted-record ceiling. Three unproductive batches, oracle disagreements, source drift, unsafe publication or failed bounded backup preserve a checkpoint and pause. Paused launchd invocations are no-ops. Accepted data is never deleted to meet a quota.

The launchd user session must remain logged in and awake. A quiet two-hour Codex follow-up monitors actionable failures and can run bounded independently reviewed improvements when the agent service is available. The AI agent itself is not active continuously. Stop markers are preserved. README.md provides exact start/stop/status/resume/export commands. State heartbeat, metrics, SQLite ledger, checkpoint and backup receipts hold live evidence; manifests and immutable shards provide remote-recoverable scientific evidence.
