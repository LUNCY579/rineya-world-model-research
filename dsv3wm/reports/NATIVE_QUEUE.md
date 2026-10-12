# Native continuation queue

All four registered H3 continuations completed32 updates: A3_s0,
A4_s0_retry01, C3_s0 and C4_s0. Actual public run/resource records and costs are
in [the training checkpoint](H3_TRAINING_CHECKPOINT_2026_10_12.md).
A4_s0's initial zero-update CPU11 preload refusal remains preserved. Its bounded
retry and C3/C4 used the separately frozen CPU10.5 training-only profile; A3
used CPU11. No student planning, dynamics or retention effect is known yet.

The [warm references](H3_WARM_REFERENCES_2026_10_12.md) remain unchanged: both
parents score24/24 own-state h4 but0/48 valid complete-plan proposals and0/12
proposer top1/pass@4/hybrid successes. The TRAIN-only teacher qualified27/32
parents,20 plans and three limits; see
[its qualification](H3_TEACHER_QUALIFICATION_2026_10_12.md). All completed output
paths and the first failed teacher/A4 attempts remain archived, not pending
commands to rerun. Neither predictive arm passes the old RegisterWorld gain gate.

Peer slot004 exited1 after a bounded wait; helper session14808 exited/released.
This is not a completed E06 training claim. No dsv3wm native model is active.
Check actual process/lease state before every new load; estimates cannot expire
live owners. BWM's GPU job/lease and owner files remain untouched. No paid/cloud
work is authorized by this queue.

Checkpoint14 passed290 tests and compilation. The final offline gate reducer
checks exact control parent/cohort and six-fixture maps plus deterministic
sampling from hash-bound128 TRAIN rows/27 positives. Pure tests are not native
student efficacy evidence.

## Actual output binding and current resource blocker

Completed artifacts have already passed actual-output admission and prospective
V1 -> V2_CPU11 -> V3_CPU11 binding:

- `RECOVERY_H3_STUDENT_READOUT_FREEZE_V3_CPU11.json`: SHA256
  `10a88c0bb31cab3e413e7af95c98968a6524dd5efff77a68e938f22404250546`.
- `RECOVERY_H3_STUDENT_GUARDS_FREEZE_V3_CPU11.json`: SHA256
  `18323febfab44c1f78e13699fd43b7d00c4df12f2efdb332784a9a1423b5b4d3`.

A source CPU11 no-model admission probe observed10.917744GiB physical /
18.396648GiB commit after imports and refused before core load. No readout
result followed. The training-only CPU10.5 profile cannot be reused as an
inference loader.

Singleton readout source review and prospective V4 CPU10.5 warm/proposer freezes
are complete under `RECOVERY_RESOURCE_CLARIFICATION_6.md`. Warm A0 completed
with byte-identical raw traces and passed parity. Warm A2's first attempt refused
before core load at10.067127GiB physical. See
[the actual singleton checkpoint](H3_SINGLETON_CHECKPOINT_2026_10_12.md).
The remaining declared workflow is:

1. Preserve completed A0 and failed A2 paths. After actual post-import physical
   and commit admission both reach10.5GiB, permit one same-input A2 retry at fresh
   `warm_A2_guards_cpu105_singleton_retry01.json`; retain its V4 freeze and recipe.
2. Complete A2 under the unchanged24 DEV/six-fixture contract, then
   require exact raw reply/token/EOS/conditioning/tag parity with the preserved
   broad warm guards. A mismatch preserves both and blocks student guards.
3. Bind both qualified old/new warm references into the actual student guard
   extension. Keep all earlier freezes and broad results unchanged.
4. Run each actual A3/A4/C3/C4 guard and proposer readout once in its fresh path,
   using its full actual configuration, weights and corresponding warm parent.
   Proposer comparisons retain greedy top1, offline pass@4 and fixed-B0 learned
   reranking at the registered budgets; exact scoring follows all generation.

A0's raw parity qualifies only that warm artifact; no A2/student result follows.
No operator should execute a completed-path training/warm command again. All
CLIs refuse overwrites and preserve failures. A3/C3 warm from final A2; A4/C4
warm from final A0. Actual A4 output lives in `A4_s0_retry01`, not its failed
`A4_s0` directory. New trained-output freezes bind that distinction.

After genuine student readouts, use `analyze_recovery_student_costs.py` with the
pinned tokenizer for tokenization-only cost checks, and
`analyze_recovery_student_gates.py --student-proposer ... --warm-proposer ...
--student-guards ... --warm-guards ... --out NEW` for paired operational
thresholds. Optional actual `--control-proposer`/`--control-guards` contrasts
must preserve the same source/corpus/draw order. Pass@4 is offline coverage,
not verifier-assisted deployment efficacy. Unknown physical counters remain
unknown, and every baseline-correct behavior loss requires explanation.

## Exact pending H3 invocation sequence

These commands are not executed results. Check current post-import physical and
commit admission before a load; every CLI retains its own10.5GiB checks and the
shared CPU lane. Completed A0, failed A2 and all historical paths are immutable.
The source-bound DEV eligibility audit confirms published optimal nonfinish
distances3/4/5 for6/5/1 tasks: all12 fit the unchanged six-action cap including
final `finish`. See `results/recovery-h3-v1/DEV_ELIGIBILITY_AUDIT_01.json`.
No model/solver was executed and no task or denominator was changed.
Execute the single A2 retry only in its fresh registered destination:

```powershell
$env:BWM_DEVICE='cpu'
$env:PYTHONPATH='dsv3wm/src'
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/recovery-h3-v1/warm_A2_guards_cpu105_singleton_retry01.resources.json -- dsv3wm/scripts/evaluate_recovery_student_guards_cpu105_readout.py --config dsv3wm/configs/recovery_student_A3.json --adapter dsv3wm/artifacts/recovery-v1/A2_s0/adapter --run dsv3wm/artifacts/recovery-v1/A2_s0/run.json --data dsv3wm/datasets/recovery-v1/dev.jsonl --fixtures dsv3wm/results/retention_core_v1.fixtures.json --freeze dsv3wm/prereg/RECOVERY_H3_WARM_GUARDS_FREEZE_V4_CPU105_SINGLETON.json --out dsv3wm/results/recovery-h3-v1/warm_A2_guards_cpu105_singleton_retry01.json --warm-baseline --train dsv3wm/datasets/recovery-v1/train.jsonl
if ($LASTEXITCODE -ne 0) { throw 'Stop and preserve the A2 retry' }
```

Only a completed A2 readout with exact raw parity permits the following guard
freeze. The helper checks raw old/new A0 and A2 traces and cross-binds them to
every actual student/control warm artifact. A mismatch stops before any guard
freeze or student generation; do not repair, redraw or rerun a completed path.

```powershell
& $nativePython dsv3wm/scripts/freeze_recovery_student_cpu105_readout.py --freeze dsv3wm/prereg/RECOVERY_H3_STUDENT_GUARDS_FREEZE_V3_CPU11.json --out dsv3wm/prereg/RECOVERY_H3_STUDENT_GUARDS_FREEZE_V4_CPU105_SINGLETON.json --old-warm-guard dsv3wm/results/recovery-h3-v1/warm_A0_guards.json --new-warm-guard dsv3wm/results/recovery-h3-v1/warm_A0_guards_cpu105_singleton.json --old-warm-guard dsv3wm/results/recovery-h3-v1/warm_A2_guards.json --new-warm-guard dsv3wm/results/recovery-h3-v1/warm_A2_guards_cpu105_singleton_retry01.json
if ($LASTEXITCODE -ne 0) { throw 'Stop: warm raw parity or lineage did not qualify' }
```

After the actual guard freeze exists and resource admission qualifies each
process, use the full trained configs and fresh paths below. All proposer calls
include `--rerank`, required by the registered hybrid gate. Actual A4 lives in
`A4_s0_retry01`; the failed `A4_s0` directory is never selected. Student calls
omit warm-only flags. Own warm guard baselines are A2 for A3/C3 and A0 for A4/C4.

```powershell
foreach ($arm in @('A3','A4','C3','C4')) {
  $taskDirectory=if ($arm -eq 'A4') { 'A4_s0_retry01' } else { "${arm}_s0" }
  $taskBaseline=if ($arm -in @('A3','C3')) { 'warm_A2_guards_cpu105_singleton_retry01.json' } else { 'warm_A0_guards_cpu105_singleton.json' }
  & $nativePython dsv3wm/scripts/run_measured.py --out "dsv3wm/results/recovery-h3-v1/${arm}_cpu105_singleton_guards.resources.json" -- dsv3wm/scripts/evaluate_recovery_student_guards_cpu105_readout.py --config "dsv3wm/configs/recovery_student_${arm}.json" --adapter "dsv3wm/artifacts/recovery-h3-v1/${taskDirectory}/adapter" --run "dsv3wm/artifacts/recovery-h3-v1/${taskDirectory}/run.json" --data dsv3wm/datasets/recovery-v1/dev.jsonl --fixtures dsv3wm/results/retention_core_v1.fixtures.json --freeze dsv3wm/prereg/RECOVERY_H3_STUDENT_GUARDS_FREEZE_V4_CPU105_SINGLETON.json --baseline "dsv3wm/results/recovery-h3-v1/${taskBaseline}" --out "dsv3wm/results/recovery-h3-v1/${arm}_cpu105_singleton_guards.json"
  if ($LASTEXITCODE -ne 0) { throw "Stop and preserve ${arm} guards" }
  & $nativePython dsv3wm/scripts/run_measured.py --out "dsv3wm/results/recovery-h3-v1/${arm}_cpu105_singleton_proposer.resources.json" -- dsv3wm/scripts/evaluate_recovery_student_cpu105_readout.py --config "dsv3wm/configs/recovery_student_${arm}.json" --adapter "dsv3wm/artifacts/recovery-h3-v1/${taskDirectory}/adapter" --run "dsv3wm/artifacts/recovery-h3-v1/${taskDirectory}/run.json" --tasks dsv3wm/datasets/recovery-v1/planning_dev.jsonl --freeze dsv3wm/prereg/RECOVERY_H3_STUDENT_READOUT_FREEZE_V4_CPU105_SINGLETON.json --out "dsv3wm/results/recovery-h3-v1/${arm}_cpu105_singleton_proposer.json" --rerank
  if ($LASTEXITCODE -ne 0) { throw "Stop and preserve ${arm} proposer" }
}
```

Generate all actual output before cost/gate reduction. Use each own warm proposer
(`warm_A2_proposer.json` or `warm_A0_proposer.json`) and the paired singleton guard.
Add C3 to A3 and C4 to A4 through both `--control-proposer` and `--control-guards`;
neither control can masquerade as a planning-trained student. Reduce costs with
the pinned tokenizer and preserve unknown physical counters. These commands do
not open confirmation or certify a successful pilot.

## Independent domain qualification queue

An additive [conditional-legality calibration diagnostic](LEGAL_CALIBRATION_PREPARATION_2026_10_12.md)
is now independently reviewed, tested and prospectively frozen. It has no
native probability outcome and retains ordinary CPU11 admission. Its report
contains fresh-path commands; the H3 CPU10.5 profiles cannot admit it. Keep
primary student/domain qualification priorities and execute one model at a time.
One [exploratory shuffled-MTP2 control](SHUFFLED_AUX_PREPARATION_2026_10_12.md)
also has a fresh86-member registration and one-boolean config change. It remains
untrained and uses the original12GiB physical floor. It cannot reopen failed G3
or use the H3 resource profiles; execute its pending command only after ordinary
admission, with separate actual-output/readout namespaces.

The separate Ledger/RP and Block CPU11 copies and V3 input freezes remain
qualified implementation paths without native outcomes. They are outside the
H3 CPU10.5 profiles and retain their independent baseline gates. Execute one
native process at a time only after resource/lease admission:

```powershell
$env:BWM_DEVICE='cpu'
$env:PYTHONPATH='dsv3wm/src'
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/B0_cpu11.resources.json -- dsv3wm/scripts/evaluate_ledger_recovery_cpu11.py --config dsv3wm/configs/ledger_A0.json --data dsv3wm/datasets/ledger-recovery-v1/dev.jsonl --freeze dsv3wm/prereg/LEDGER_B0_FREEZE_V3_CPU11.json --out dsv3wm/results/ledger-recovery-v1/B0_cpu11.json --baseline-only
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2_cpu11.resources.json -- dsv3wm/scripts/evaluate_rcwm_retention_v2_cpu11.py --config dsv3wm/configs/rcwm_retention_v1.json --adapter rcwm/experiments/RCWM-T1/runs/t2/t2_s0/adapter --fixtures dsv3wm/datasets/rcwm-retention-v2/fixtures.json --freeze dsv3wm/prereg/RCWM_RETENTION_B0_FREEZE_V3_CPU11.json --out dsv3wm/results/ledger-recovery-v1/T2_RP_B0_v2_cpu11.json --model-label T2B0_v2_cpu11
& $nativePython dsv3wm/scripts/run_measured.py --out dsv3wm/results/block-g1-v1/B0_cpu11.resources.json -- dsv3wm/scripts/evaluate_block_g1_cpu11.py --config dsv3wm/datasets/block-g1-v1/config.json --cases dsv3wm/datasets/block-g1-v1/cases.jsonl --manifest dsv3wm/datasets/block-g1-v1/manifest.json --freeze dsv3wm/prereg/BLOCK_G1_FREEZE_V3_CPU11.json --out dsv3wm/results/block-g1-v1/B0_cpu11.json
```

Stop on failures; this is a queue, not permission to overlap native jobs. Ledger
adaptation needs qualified B0 and its own result-bound training freeze. Adapted
R/P readouts need their actual lineage and preserved baseline; failed B0 cases
cannot certify retained skill. V2 restores R's actual T2 event surface while P
uses the source's original one-step frame, different from its trained m=1 frame.
Block G1 qualifies only its original interface; larger Block training needs a
separate finite-parent/object-count/split protocol.

The sealed confirmation vault remains closed. Successful feasible H3 readouts
would still need a larger preregistered development/seed study and a complete
fresh sealed co-primary/hard-planning registration before confirmation. Training
completion, resource admission and synthetic tests do not qualify G4/G5/G7,
broad retention, calibration or model/adapter/skill promotion.
