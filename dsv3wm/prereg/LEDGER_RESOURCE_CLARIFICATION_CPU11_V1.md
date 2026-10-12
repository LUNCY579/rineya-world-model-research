# Prospective Ledger CPU11 admission extension

2026-10-12, before any new Ledger or paired R/P v2 outcome under this extension.
The shared GPU owner's host-memory use leaves approximately 11.8 GiB physical
availability, below the original 12 GiB loader admission floor. Preserve every
original loader, Ledger and R/P evaluator, protocol, freeze and failed attempt.
This declaration extends the existing CPU11 operational profile to Ledger only;
it makes no scientific qualification or retention claim.

Use the new copied `train_ledger_recovery_cpu11.py`,
`evaluate_ledger_recovery_cpu11.py` and `evaluate_rcwm_retention_v2_cpu11.py`.
Ledger B0 qualification uses the same Ledger evaluation CLI with
`--baseline-only`; adapted DEV uses that CLI without the flag. Changes are
limited to the loader import, original/new source pins and resource-admission
metadata. Exact-copy and AST tests protect the optimizer, TRAIN draws, causal
MTP and multihorizon objectives, loss weights, dropout seeds, FP32 precision,
unmerged T2 adapters, teacher-forced CE, own-state rollouts, generation budgets,
strict parsing and delayed offline scoring.

`dsv3wm.native_cpu11.load` requires CPU and exactly four threads, with both
physical and commit availability at least 11 GiB after BWM/torch imports and
before core hashing/loading. Record its constant profile and actual admission
readings even when load refuses. The existing CPU lease and four-GiB work
boundary remain active. `BWM_CPU_BATCH_TOKENS` stays unset or `8000`, and
`BWM_RAM_FLOOR_GB` stays unset or equivalent to 8 GiB. The inherited sampler's
8000 parameter is a grouping heuristic, not a hard padded-position ceiling.
The prior teacher CPU11 run's 11.766 GiB physical/19.535 GiB commit admission
and 5.148 GB sampled tree peak support attempting this operational profile;
they do not guarantee Ledger's training peak or future concurrent availability.
Stop and preserve partial artifacts if the existing resource or priority guard
refuses work. Do not load a model during another owner's reserved CPU slot.

Before each phase, run `freeze_ledger_cpu11.py --freeze ORIGINAL --out NEW`.
It verifies every original member, retains all original metadata and digests,
pins the actual original freeze bytes, and appends both original and copied
entrypoints, the CPU11 loader, this declaration and the shared execution
dependency closure. It accepts only the existing Ledger development and R/P v2
baseline/adapted instruments with closed confirmation access. An existing
extension/output cannot be overwritten. The resulting fresh freeze, rather
than the old 12 GiB freeze, must be passed to the CPU11 entrypoint. No freeze
or native outcome is created as part of implementation/testing this amendment.

Underlying B0/data/T2/source pins and qualification gates are unchanged.
Ledger training still requires the actual qualified B0 result and traces, with
their original execution-freeze reference retained in a later training freeze.
Adapted R/P readout still requires its separately validated completed T2 B0
and trained Ledger lineage; this helper cannot substitute for those freezers.
Their source-membership checks accept the additional CPU11 execution hashes.
R/P v2 remains the same paired 12 R/12 P fixtures and 512/64 greedy caps, with
the declared P one-step-frame limitation and no broad retention pass. The six
original retention fixtures remain a separate registered requirement.

This extension does not authorize changing Blocksworld, historical programmes,
private confirmation access, data or numerical contracts, and does not relax
any gain, malformed-output or retention gate. No model run is authorized until
the parent has reviewed the new files and created the new prospective freezes.
