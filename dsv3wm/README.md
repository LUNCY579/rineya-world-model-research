# DeepSeek-inspired Rineya world-model research

Status: **active research; native-CPU pilots, efficacy inconclusive**.
The initial cold A0/A1/A2/A5 screen returned state 0/2 and planning 0/1.
Recovery reuses the registered unmerged rank-16 forward adapter: inherited B0
has 24/24 exact four-step development rollouts, with all intermediate states
correct. Adapted A0/A1/A2/A5 scored24/22/24/23; direct four-step accuracy is0/24
for each. Neither predictive arm meets the fixed gain gate. Planning scored
12/12 for B0/A0/A1/A2 and11/12 for A5; A1/A2 also lose a retained boundary
legality fixture. H3/Ledger/Blocksworld execution is next. G4–G7 remain unqualified;
confirmation labels remain sealed. See `reports/RECOVERY_STATUS.md`,
`reports/GATE_STATUS.md` and the exact pending commands in `reports/NATIVE_QUEUE.md`.
Original test evidence remains in `results/TESTS_FINAL.json`; new records use
separate paths. The current package passes221 tests and compilation, recorded
in `results/recovery-v1/PREPARATION_CHECKPOINT_10.json`. The consolidated native
evidence and current integration decisions are in
`reports/WARM_RESEARCH_REPORT_2026_10_12.md`; the executed MTP architecture is
documented in `research/IMPLEMENTATION_EVIDENCE_2026_10_12.md`.

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
pass@k, no verifier reranking. A3/A4 have no native training outcome yet. Stored
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
