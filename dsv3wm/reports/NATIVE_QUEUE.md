# Native continuation queue

The full research goal remains active. Native recovery A0/A1/A2/A5 dynamics,
planning and six-case behavior are complete and published in `aa17d958`.
Neither predictive arm passes the fixed RegisterWorld gain gate. No larger
training/confirmation or integration is qualified by that study.

RDW E06's first10-step slot finished and its helper released. RCX then acquired
the shared CPU lane (live ownerPID49976, job `RCX E01R step A (4 DEV2 evals + analysis)`
at2026-10-12 06:27UTC, nominal6600s). Its estimate is not permission to expire a
live owner. Check actual process/lease state before another model load. GPU
reclaim remains forbidden by the nonempty-worker record. No paid/cloud work.

The first H3 teacher launch failed at lease acquisition, before model load or
prediction. Private failed manifest is
`artifacts/recovery-h3-v1/teacher_B0/manifest.json`; resource evidence is
`results/recovery-v1/H3_teacher_B0.resources.json`. Preserve both. A same-input
bounded retry uses NEW paths after release; do not reinterpret resource failure
as failed label quality or redraw the frozen32-parent pool.
The retry uses V3's additional shared execution dependency pins; its model,
configuration, task pool, prediction and scoring code are unchanged. Ledger
B0 and Block G1 likewise use new extended V2 freezes. Original freezes remain
unchanged; see `prereg/EXECUTION_SOURCE_CLOSURE_V1.md`.

Run one native process at a time from this worktree:

```powershell
$env:BWM_DEVICE='cpu'
$env:PYTHONPATH='dsv3wm/src'
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-v1/H3_teacher_B0_retry01.resources.json -- dsv3wm/scripts/collect_recovery_teacher.py --config dsv3wm/configs/recovery_A0.json --adapter ../Rineya_BWM_001_Research_Bundle/native_v1/runs/F_titan_main/adapter --tasks dsv3wm/datasets/recovery-h3-v1/planning_train.jsonl --train dsv3wm/datasets/recovery-v1/train.jsonl --freeze dsv3wm/prereg/RECOVERY_H3_TEACHER_FREEZE_V3.json --out dsv3wm/artifacts/recovery-h3-v1/teacher_B0_retry01
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/B0.resources.json -- dsv3wm/scripts/evaluate_ledger_recovery.py --config dsv3wm/configs/ledger_A0.json --data dsv3wm/datasets/ledger-recovery-v1/dev.jsonl --freeze dsv3wm/prereg/LEDGER_B0_FREEZE_V2.json --out dsv3wm/results/ledger-recovery-v1/B0.json --baseline-only
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2.resources.json -- dsv3wm/scripts/evaluate_rcwm_retention_v2.py --config dsv3wm/configs/rcwm_retention_v1.json --adapter rcwm/experiments/RCWM-T1/runs/t2/t2_s0/adapter --fixtures dsv3wm/datasets/rcwm-retention-v2/fixtures.json --freeze dsv3wm/prereg/RCWM_RETENTION_B0_FREEZE_V2.json --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2.json --model-label T2B0_v2
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/block-g1-v1/B0.resources.json -- dsv3wm/scripts/evaluate_block_g1.py --config dsv3wm/datasets/block-g1-v1/config.json --cases dsv3wm/datasets/block-g1-v1/cases.jsonl --manifest dsv3wm/datasets/block-g1-v1/manifest.json --freeze dsv3wm/prereg/BLOCK_G1_FREEZE_V2.json --out dsv3wm/results/block-g1-v1/B0.json
```

Execute these independently and stop on failures; the block is a queue, not
authorization to overlap jobs or proceed past a failed gate.

After a qualified H3 corpus, validate/freeze actual shared labels and final A0/A2
warm artifacts with `freeze_recovery_student.py`, then run A3/A4 and matched
nominal-forward C3/C4 controls. Explicit warm-parent proposer readouts, identical
direct/pass@4 and optional fixed-B0 learned reranking are required. Trained
outputs need a new readout freeze; never reuse a warm-only training freeze.
Extend subsequent student and Ledger training/readout freezes with the same
`freeze_execution_dependencies.py` before execution. Preserve their originals
and retain the actual B0 qualification-freeze reference.

After qualified Ledger B0, create a new training freeze binding its result/traces
and original qualification-freeze reference, then run the fixed four-arm recipes.
Measure paired R/P behavior on the frozen24-case V2 supplement using
`freeze_rcwm_retention_adapted_v2.py` for adapted-run lineage qualification.
V2 restores the actual T2 R event/message surface; P retains the source's original
one-step frame, while T2 P TRAIN uses a single-event multi-step frame. Measure B0
behavior first; a case B0 fails cannot certify retention of that capability.
V1's altered R event surface and all prior sources/freezes remain unchanged and
unexecuted. Block G1 qualifies the original interface only; larger
Block training needs its own finite-parent/object-count/split protocol.

Keep the original confirmation vault closed. Successful feasible pilots still
need multi-seed replication and a complete fresh sealed co-primary registration
before confirmation. Remaining causal ablations, shortcut checks, retention,
H4/H5 follow-ups and G7 recommendations must reflect actual executed evidence.
