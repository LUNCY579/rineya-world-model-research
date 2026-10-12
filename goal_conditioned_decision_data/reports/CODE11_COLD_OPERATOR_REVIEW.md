# CODE11 cold recovery verifier review

FINAL_VERDICT: PASS

This operator-only approval covers recovery verification with the exact helper below. It certifies the reviewed procedure and execution boundary. The helper has not been executed by this reviewer; no successful cold-recovery result or production readiness is claimed here.

The frozen helper is `manifests/operators/code11/verify_cold_recovery.txt` in `/Users/chengli/.codex/gcdl001-host/recovery-proposal-worktree/goal_conditioned_decision_data`: **13,838 bytes**, SHA-256 **`d25df3f1563bf5f0f0939d77c19220b50c0b085410cd70441e60fc5c424abcc5`**.

The approved execution command, from `/Users/chengli/.codex/gcdl001-host/worktree`, is:

```sh
/usr/bin/nice -n 10 /opt/homebrew/opt/python@3.12/bin/python3.12 -B - < /Users/chengli/.codex/gcdl001-host/recovery-proposal-worktree/goal_conditioned_decision_data/manifests/operators/code11/verify_cold_recovery.txt
```

Use a fresh nonoptimized process and the exact frozen helper bytes. The operator checks those interpreter flags, an empty read session/raw cache, and the fixed canonical programme root. It must exit successfully and return the expected stdout JSON before its verification is accepted. A timeout, exception, incomplete output or nonzero process exit provides no successful verification.

## Approval and timing boundary

The operator requires live source69 digest `f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c`, the deployed `validated-code-11` receipt, original independent approval SHA-256 `14ff351071e1e66f879075c713df701369199218d568ebcc740f810d4b81c4ab`, both 511/511 complete-suite records and zero errors/failures/skips. Initial and final current independent review checks use the same deadline and preserve the existing five scientific evidence commitments. No scientific experiment or full suite is rerun.

Its 144-second deadline starts after standard-library imports and before programme-package imports, approval, custody/control validation, Git enumeration and both recovery phases. A SIGALRM and explicit monotonic checks enforce this inclusive operator boundary. One bounded read session and the physical Git reader share the deadline. This is recovery-only measurement; it does not exercise the complete production startup/generation/publication/backup cycle or relax any runtime budget, publication floor, programme ceiling or watchdog.

## Recovery and preservation assessment

The live queue is opened with SQLite `mode=ro` and `query_only=ON`. Each copy is made into `:memory:` with `temp_store=MEMORY` verified, and positive page-count/page-size metadata capped at the existing 8 MiB archive metadata bound before backup. Every backup chunk rechecks the total byte bound and deadline. Recovery calls are the actual deployed `worker.recover_inventory(memory)` and `rendered.recover(memory, R)`. Their SQL setup, upserts and commits target the supplied RAM connection. The helper never calls the production connection/run/process/drain path, backup, compaction, publication, pause clearing or source deployment.

Before recovery, after RAM recovery and after a second read-only live snapshot, it checks the three frozen logical tables: `jobs`, `seen` and `rendered_jobs`. Required inventory is 105 completed canonical groups, 3,124 accepted records, zero quarantined/pending canonical jobs, split totals dev 16/442, OOD 16/474, sealed 8/216 and train 65/1,992. Protected manifest IDs must exactly match SQL and must have no semantic/parent/template overlap across splits. Render status counts remain 42 done, 87 pending and 268 superseded, with 42 receipts and 5,250 derived rows.

The original operational freeze is pinned to `453735f3046c4569a4fa90638e3251861ec757926f97e9337da2d75aa12a2d59`, and the archive index to `c3c0d2851b182a75bb70292f1cb27b87d94b513cd7248463b7b99e09b630a434`. The 203 frozen control commitments have no QUALITY/status bindings. Only the two deliberately promoted deployment controls are substituted: the current validated receipt is captured and kept stable, while the active independent review must retain the exact approved hash. Source, controls, STOP and original PAUSED bytes are checked again after the final blocking approval call.

The accepted/sealed/rendered body-read wrappers reject those payload paths and all corresponding historical blob OIDs. Their attempt counters must remain zero. Catalog metadata, protected identities and scientific approval evidence remain permitted. Metadata-only sealed checks do not authorize fetching sealed payload bodies. Indexed payload paths must remain absent locally.

The approved Git helper enumerates physically present object names with network protocols and lazy fetching disabled. It never probes missing cold OIDs individually. The validated catalog union must contain exactly 443 historical cold blob OIDs; none may be present before or after, and the complete physically enumerated object set must remain equal. The helper performs no Git-store mutation or remote write.

## Verification and scope limits

I read the full final helper and its called recovery/Git-reader paths, and compiled the helper without execution. Compilation passed. A fresh byte comparison confirmed that the live 69-file map equals its receipt and still has digest `f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c`; the active review remains the exact original `14ff351...` approval. The live owned PAUSED bytes retain SHA-256 `afe3173b64c2d0f0525519f630df949e85b7059620911def2ebd27a7fbfd7fe0`.

`live_queue_unchanged` is assessed through the three specified frozen logical tables. Read-only SQLite may use ordinary locks/read marks in existing WAL/SHM files; this review does not certify byte-for-byte sidecar immutability or every unrelated database table. The helper's main and temporary recovery databases are RAM-only, and it issues no DML/DDL against the live connection.

This helper leaves the original pause in place and emits a recovery-only result. A successful run can establish its stated metadata recovery, inventory, deadline and cold-storage checks. It does not establish new-record production, rendering publication, archive deletion, central-ref migration or permission to resume by itself. Any change to the frozen helper requires another operator review.

Only this supplementary Markdown report was written by this review. The helper, deployed source, original approvals, live queue/runtime state, remote refs, dataset bodies and sealed labels were not executed or modified by this reviewer.
