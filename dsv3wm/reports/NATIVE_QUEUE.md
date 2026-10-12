# Native continuation queue

The full research goal remains active. Native recovery A0/A1/A2/A5 dynamics,
planning and six-case behavior are complete and published in `aa17d958`.
Neither predictive arm passes the fixed RegisterWorld gain gate. No larger
training/confirmation or integration is qualified by that study.

RDW E06 slot002 is now live after the coordinated teacher handoff. Its reserved
window is expected to finish around2026-10-12 09:35–09:40UTC. No other dsv3wm
native model is active. Check actual process/lease release before loading;
an estimate is not permission to expire a live owner. The BWM GPU training
process and its lease remain untouched. No paid/cloud work.

The first H3 teacher launch failed at lease acquisition, before model load or
prediction. Private failed manifest is
`artifacts/recovery-h3-v1/teacher_B0/manifest.json`; resource evidence is
`results/recovery-v1/H3_teacher_B0.resources.json`. Preserve both. A same-input
bounded retry uses NEW paths after release; do not reinterpret resource failure
as failed label quality or redraw the frozen32-parent pool.
RCX subsequently exited and released the lane. A new BWM GPU training process
uses host RAM. The completed H3-only retry used the separate CPU11 loader/
entrypoint and V4 CPU11 freeze in
`RECOVERY_RESOURCE_CLARIFICATION_3.md`: physical and commit floors11 GiB after
imports, unchanged four-GiB boundary, model/math/sampler/data/gates. Both old
loaders/collectors/freezes remain unchanged. Its V3 shared dependency pins are
retained; original source history still verifies. Ledger
B0 and Block G1 likewise use new extended V2 freezes. Original freezes remain
unchanged; see `prereg/EXECUTION_SOURCE_CLOSURE_V1.md`.

The retry completed successfully in841.880s:27 accepted TRAIN parents from the
fixed32,20 distinct plans and three limits. Corpus SHA256 is
`3a2a25a33b7dc983b422b67fc1a1977d7315896e42c72132a454012c27f1841b`.
Private corpus: `artifacts/recovery-h3-v1/teacher_B0_retry01/`; public audit:
[H3 teacher qualification](H3_TEACHER_QUALIFICATION_2026_10_12.md).
This is TRAIN corpus qualification, not a student/native DEV result. Preserve
the successful outputs and the first failed attempt; no repeat teacher run is
queued. All predictive gain gates remain unchanged.

After the live peer releases, run one native process at a time from this worktree.
Next are warm H3 guards, warm proposer references, then A3/A4/C3/C4. CPU11 is
separately preregistered for these fixed H3 phases by
`prereg/RECOVERY_RESOURCE_CLARIFICATION_4.md`. Ledger/Blocksworld retain their
original resource paths and independent qualification gates.

```powershell
$env:BWM_DEVICE='cpu'
$env:PYTHONPATH='dsv3wm/src'
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
```

Execute these independently and stop on failures; the block is a queue, not
authorization to overlap jobs or proceed past a failed gate.

The actual shared27-parent corpus and final A0/A2 warm artifacts are already
bound by `RECOVERY_H3_STUDENT_TRAIN_FREEZE_V3_CPU11.json`. Its four fixed recipes
include the matched nominal-forward C3/C4 controls. Trained outputs still need
a new output/readout freeze; never reuse the training freeze for later weights.

The H3 dynamics/behavior guard evaluator is a separate new instrument accepting
the full StudentConfig. Its unchanged warm-parent guard inputs are already
bound by `RECOVERY_H3_WARM_GUARDS_FREEZE_V3_CPU11.json`. Run both warm references
before accepting student guard comparisons:

```powershell
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-h3-v1/warm_A0_guards.resources.json -- dsv3wm/scripts/evaluate_recovery_student_guards_cpu11.py --config dsv3wm/configs/recovery_student_A4.json --adapter dsv3wm/artifacts/recovery-v1/A0_s0/adapter --run dsv3wm/artifacts/recovery-v1/A0_s0/run.json --data dsv3wm/datasets/recovery-v1/dev.jsonl --fixtures dsv3wm/results/retention_core_v1.fixtures.json --freeze dsv3wm/prereg/RECOVERY_H3_WARM_GUARDS_FREEZE_V3_CPU11.json --out dsv3wm/results/recovery-h3-v1/warm_A0_guards.json --warm-baseline --train dsv3wm/datasets/recovery-v1/train.jsonl
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-h3-v1/warm_A2_guards.resources.json -- dsv3wm/scripts/evaluate_recovery_student_guards_cpu11.py --config dsv3wm/configs/recovery_student_A3.json --adapter dsv3wm/artifacts/recovery-v1/A2_s0/adapter --run dsv3wm/artifacts/recovery-v1/A2_s0/run.json --data dsv3wm/datasets/recovery-v1/dev.jsonl --fixtures dsv3wm/results/retention_core_v1.fixtures.json --freeze dsv3wm/prereg/RECOVERY_H3_WARM_GUARDS_FREEZE_V3_CPU11.json --out dsv3wm/results/recovery-h3-v1/warm_A2_guards.json --warm-baseline --train dsv3wm/datasets/recovery-v1/train.jsonl
```

Warm proposer references are now prospectively bound by
`RECOVERY_H3_WARM_READOUT_FREEZE_V3_CPU11.json`, SHA256
`4170f6ad5ffbabcf526109fe3c9a7e69991ad8cf268363afb166576fe0142f2f`.
This retains the actual A0/A2 artifacts and fixed12 planning parents, with CPU11
and shared execution-source extensions. Each readout reports fixed greedy top1,
offline pass@4 coverage and matched fixed-B0 learned reranking on the same
completed candidate set; exact scoring follows all generation/ranking:

```powershell
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-h3-v1/warm_A0_proposer.resources.json -- dsv3wm/scripts/evaluate_recovery_student_cpu11.py --config dsv3wm/configs/recovery_student_A4.json --adapter dsv3wm/artifacts/recovery-v1/A0_s0/adapter --run dsv3wm/artifacts/recovery-v1/A0_s0/run.json --tasks dsv3wm/datasets/recovery-v1/planning_dev.jsonl --freeze dsv3wm/prereg/RECOVERY_H3_WARM_READOUT_FREEZE_V3_CPU11.json --out dsv3wm/results/recovery-h3-v1/warm_A0_proposer.json --warm-baseline --train dsv3wm/datasets/recovery-v1/train.jsonl --rerank
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-h3-v1/warm_A2_proposer.resources.json -- dsv3wm/scripts/evaluate_recovery_student_cpu11.py --config dsv3wm/configs/recovery_student_A3.json --adapter dsv3wm/artifacts/recovery-v1/A2_s0/adapter --run dsv3wm/artifacts/recovery-v1/A2_s0/run.json --tasks dsv3wm/datasets/recovery-v1/planning_dev.jsonl --freeze dsv3wm/prereg/RECOVERY_H3_WARM_READOUT_FREEZE_V3_CPU11.json --out dsv3wm/results/recovery-h3-v1/warm_A2_proposer.json --warm-baseline --train dsv3wm/datasets/recovery-v1/train.jsonl --rerank
```

Then run the already frozen continuation recipes, with no outcome-driven corpus
or config selection:

```powershell
foreach ($kind in @('A3','A4','C3','C4')) {
  $warmArm = if ($kind -in @('A3','C3')) { 'A2' } else { 'A0' }
  & $nativePython dsv3wm/scripts/run_measured.py --out "dsv3wm/results/recovery-h3-v1/${kind}_s0.resources.json" -- dsv3wm/scripts/train_recovery_student_cpu11.py --config "dsv3wm/configs/recovery_student_${kind}.json" --train dsv3wm/datasets/recovery-v1/train.jsonl --tasks dsv3wm/datasets/recovery-h3-v1/planning_train.jsonl --teacher-dir dsv3wm/artifacts/recovery-h3-v1/teacher_B0_retry01 --warm-dir "dsv3wm/artifacts/recovery-v1/${warmArm}_s0" --qualification dsv3wm/results/recovery-v1/B0_qualification.json --freeze dsv3wm/prereg/RECOVERY_H3_STUDENT_TRAIN_FREEZE_V3_CPU11.json --out "dsv3wm/artifacts/recovery-h3-v1/${kind}_s0"
  if ($LASTEXITCODE -ne 0) { throw 'H3 continuation failed; preserve artifacts and stop' }
}
```

After completed A3/A4/C3/C4 training, `freeze_recovery_student_guards.py`
requires each actual `--student-dir` and its own `--baseline` warm guard record.
Extend each new guard/proposer freeze with the CPU11 phase helper and execution
closure before use; pass the same warm baseline to
`evaluate_recovery_student_guards_cpu11.py`. Never substitute an old RecoveryConfig readout for this new
lineage/trace/counter instrument or infer retention from a warm-baseline failure.
After genuine proposer/rerank readouts, run `analyze_recovery_student_costs.py`
with each `--input` and a new `--out`; use the registered `--tokenizer-json` for
tokenization-only verification of forward input/position totals. Keep pure,
offline pass@4 coverage and learned-rerank costs separate. Physical counters
absent from the original proposal instrument remain unknown.

Independent Ledger/R/P and Block qualifications remain in the later queue:

```powershell
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/B0.resources.json -- dsv3wm/scripts/evaluate_ledger_recovery.py --config dsv3wm/configs/ledger_A0.json --data dsv3wm/datasets/ledger-recovery-v1/dev.jsonl --freeze dsv3wm/prereg/LEDGER_B0_FREEZE_V2.json --out dsv3wm/results/ledger-recovery-v1/B0.json --baseline-only
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2.resources.json -- dsv3wm/scripts/evaluate_rcwm_retention_v2.py --config dsv3wm/configs/rcwm_retention_v1.json --adapter rcwm/experiments/RCWM-T1/runs/t2/t2_s0/adapter --fixtures dsv3wm/datasets/rcwm-retention-v2/fixtures.json --freeze dsv3wm/prereg/RCWM_RETENTION_B0_FREEZE_V2.json --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2.json --model-label T2B0_v2
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/block-g1-v1/B0.resources.json -- dsv3wm/scripts/evaluate_block_g1.py --config dsv3wm/datasets/block-g1-v1/config.json --cases dsv3wm/datasets/block-g1-v1/cases.jsonl --manifest dsv3wm/datasets/block-g1-v1/manifest.json --freeze dsv3wm/prereg/BLOCK_G1_FREEZE_V2.json --out dsv3wm/results/block-g1-v1/B0.json
```

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
