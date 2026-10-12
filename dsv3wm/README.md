# DeepSeek-inspired Rineya world-model research

Status: **active research; native-CPU pilots, efficacy inconclusive**.
The initial cold A0/A1/A2/A5 screen returned state 0/2 and planning 0/1.
Recovery reuses the registered unmerged rank-16 forward adapter: inherited B0
has 24/24 exact four-step development rollouts, with all intermediate states
correct. Adapted A0/A1/A2/A5 scored24/22/24/23; direct four-step accuracy is0/24
for each. Neither predictive arm meets the fixed gain gate. Planning scored
12/12 for B0/A0/A1/A2 and11/12 for A5; A1/A2 also lose a retained boundary
legality fixture. The fixed TRAIN-only H3 teacher qualified27/32 parents, with20
distinct plans across three limits. Warm proposer references have0/48 valid
plans and0/12 top1/pass@4/hybrid successes. A3/A4/C3/C4 completed32 native
updates each; student readout effects remain unknown. Next are resource-qualified
H3 readouts after completing singleton warm A2 parity, followed by independent
Ledger/Blocksworld qualification. G4–G7 remain unqualified;
confirmation labels remain sealed. See `reports/RECOVERY_STATUS.md`,
`reports/GATE_STATUS.md` and the exact pending commands in `reports/NATIVE_QUEUE.md`.
Original test evidence remains in `results/TESTS_FINAL.json`; new records use
separate paths. Checkpoint14 passed290 tests and compilation; test evidence
is in `results/recovery-v1/TESTS_CHECKPOINT_14.txt`. Checkpoint12 preserves
the preceding domain/CPU11 preparation and checkpoint11 preserves
the preceding teacher/student preparation evidence. The consolidated native
evidence and current integration decisions are in
`reports/WARM_RESEARCH_REPORT_2026_10_12.md`; the executed MTP architecture is
documented in `research/IMPLEMENTATION_EVIDENCE_2026_10_12.md`.
The later [H3 teacher qualification](reports/H3_TEACHER_QUALIFICATION_2026_10_12.md)
records the native TRAIN collection and its failures. The
[warm references](reports/H3_WARM_REFERENCES_2026_10_12.md) and
[completed training checkpoint](reports/H3_TRAINING_CHECKPOINT_2026_10_12.md)
separate measured artifacts from unknown efficacy. A4's preserved preload
refusal was followed by its bounded CPU10.5 training retry; C3/C4 used the same
training-only profile. Student output freezes and reviewed singleton V4 warm/
proposer extensions exist. The singleton A0 warm guard completed with exact raw
parity; A2 refused before loading at10.067GiB, below its10.5GiB floor. See
[the singleton checkpoint](reports/H3_SINGLETON_CHECKPOINT_2026_10_12.md).
Student guards require both warm parity passes; their effects remain unknown.
Peer slot004 exited1 after a bounded wait; no own native model is active.
Separate Ledger/RP and Block CPU11 copies and
V3 input freezes are now verified; their native outcomes remain pending.
The [full-assignment audit](reports/COMPLETION_AUDIT_2026_10_12.md) preserves
every required section/gate/deliverable and its remaining evidence. An additive
[conditional-legality calibration diagnostic](reports/LEGAL_CALIBRATION_PREPARATION_2026_10_12.md)
passed18 semantic tests, independent review and actual tokenizer/lineage audits;
its72-member CPU11 freeze is registered. No probabilities or calibration results
have been measured, and it does not certify generated-state/planning confidence.

Dedicated experiment, isolated from BWM/RCWM. No core/model release or promotion.
Parent repository baseline: `aeb19ab17128d884c4457ce59db6e926ceb13a7a`.
See `research/DEEPSEEK_TRANSFER.md`, `reports/BASELINE_INVENTORY.md`, and
`prereg/PILOT_V1.md`. The interim report is in `reports/REPORT.md`.

From repository root with the existing native Python environment, reproduce
without replacing any original output:

```powershell
$env:BWM_DEVICE = 'cpu'
$env:PYTHONPATH = 'dsv3wm/src'
python -m unittest discover -s dsv3wm/tests -v
python dsv3wm/scripts/build_data.py --seed 202610111 --rows 24 --output dsv3wm/datasets/replica-01 --vault ../dsv3wm-vault-replica-01
python dsv3wm/scripts/freeze_pilot.py --dataset dsv3wm/datasets/replica-01 --out dsv3wm/prereg/PILOT_REPLICA_01.json
python dsv3wm/scripts/native_probe.py --device cpu --out dsv3wm/results/replica_native_probe.json
foreach ($arm in @('A0','A1','A2','A5')) {
  python dsv3wm/scripts/train.py --config "dsv3wm/configs/${arm}_pilot.json" --train dsv3wm/datasets/replica-01/train_register.jsonl --out "dsv3wm/artifacts/replica-01/${arm}_s0" --freeze dsv3wm/prereg/PILOT_REPLICA_01.json
  if ($LASTEXITCODE -ne 0) { throw 'Native replica failed; preserve run record' }
}
python dsv3wm/scripts/cache_probe.py --adapter dsv3wm/artifacts/replica-01/A0_s0/adapter --out dsv3wm/results/replica_cache_probe.json
python dsv3wm/scripts/freeze_replica.py --dataset dsv3wm/datasets/replica-01 --run A0=dsv3wm/artifacts/replica-01/A0_s0 --run A1=dsv3wm/artifacts/replica-01/A1_s0 --run A2=dsv3wm/artifacts/replica-01/A2_s0 --run A5=dsv3wm/artifacts/replica-01/A5_s0 --out dsv3wm/prereg/CACHE_REPLICA_01.json --probe dsv3wm/results/replica_cache_probe.json
foreach ($arm in @('A0','A1','A2','A5')) {
  python dsv3wm/scripts/cached_pilot.py --config "dsv3wm/configs/${arm}_pilot.json" --adapter "dsv3wm/artifacts/replica-01/${arm}_s0/adapter" --data dsv3wm/datasets/replica-01/dev.jsonl --out "dsv3wm/results/${arm}_replica_cached_dev.json" --freeze dsv3wm/prereg/CACHE_REPLICA_01.json --probe dsv3wm/results/replica_cache_probe.json
  if ($LASTEXITCODE -ne 0) { throw 'Native evaluation failed; preserve run record' }
}
```

Dataset/run commands refuse overwriting previous artifacts. Reproduction must
use a new output/vault path; hashes bind the original inputs. The register-only
training projection is produced by `scripts/freeze_pilot.py` after data generation,
before training. No existing training data or frozen test labels are read.

The full four-arm replay requires all A0/A1/A2/A5 training before freezing, because
the freeze binds every trained artifact. `freeze_replica.py` supports new paths;
never reuse the original freeze for new timestamped run JSON. Original v1/v2 replay is pinned
to commit 108631d. Original cached measurements bind `CACHE_FREEZE_V3.json` and source
checkpoint 58ba757. Current replay uses v4 input-binding gates; archived records
are unchanged. The example reproduces training and dynamics; optional planning,
retention and student CLIs are listed below. Exact
recorded invocation arrays, configs, versions and hashes are in `results/`.

`train.py` implements A0/A1/A2/A3/A4/A5. A3/A4 require independently re-verified
TRAIN-only trajectory labels; use `scripts/distill.py --help`. Dedicated direct
student inference is `scripts/evaluate_student.py --help`: fixed top1 and offline
pass@k, no verifier reranking. Actual warm H3 A3 training used
`train_recovery_student_cpu11.py` and `RECOVERY_H3_STUDENT_TRAIN_FREEZE_V3_CPU11.json`.
Actual A4 retry/C3/C4 used `train_recovery_student_cpu105.py` and the separate
`RECOVERY_H3_STUDENT_TRAIN_FREEZE_V4_CPU105_TRAINING.json`. Both bind the shared
27-parent corpus and actual A0/A2 parents. Original CPU11 readout entrypoints are
`evaluate_recovery_student_cpu11.py` and `evaluate_recovery_student_guards_cpu11.py`;
the new singleton entrypoints use the separately reviewed CPU10.5 readout profile.
The fixed warm guards use `RECOVERY_H3_WARM_GUARDS_FREEZE_V3_CPU11.json`.
Warm proposer references use `RECOVERY_H3_WARM_READOUT_FREEZE_V3_CPU11.json`.
A3/A4/C3/C4 have completed native training; their student readout outcomes remain
unknown. Actual output readout/guard V3 CPU11 freezes bind the four new artifacts.
Any singleton readout requires its separate prospective resource/parity extension.
Stored
adapters remain unmerged. MTP/legality training heads are omitted in inference.
The original no-cache instrument is preserved; the explicit cached instrument
requires a passed native parity probe and complete source/config/data freeze.

Native loss-only retention: `scripts/audit_retention.py --help`. Inherited BWM
forward behavior smoke: `scripts/registered_forward_probe.py --help`. Analysis:
`scripts/summarize_results.py`, with no substitution of missing results by zero.

Tiny native pilots do not establish efficacy. Confirmation CLI rejects all
nondevelopment data until a complete resource-qualified G5 protocol is frozen.
Specialist routing and MLA/cache conversion remain conditional follow-ups.
Large artifacts, fresh datasets and the external sealed vault stay outside Git;
`data_manifests/` and `results/EXCLUDED_ARTIFACTS_V1.json` record hashes.
