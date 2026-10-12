# DeepSeek-inspired Rineya world-model research

Dedicated experiment, isolated from BWM/RCWM. No core/model release or promotion.
Parent repository baseline: `aeb19ab17128d884c4457ce59db6e926ceb13a7a`.
See `research/DEEPSEEK_TRANSFER.md`, `reports/BASELINE_INVENTORY.md`, and
`prereg/PILOT_V1.md`. Final executed status is in `reports/REPORT.md`.

From repository root with the existing native Python environment:

```powershell
$env:BWM_DEVICE = 'cpu'
$env:PYTHONPATH = 'dsv3wm/src'
python -m unittest discover -s dsv3wm/tests -v
python dsv3wm/scripts/build_data.py --seed 202610111 --rows 24 --output dsv3wm/datasets/pilot-v1 --vault ../dsv3wm-vault-v1
python dsv3wm/scripts/native_probe.py --device cpu --out dsv3wm/results/native_probe_cpu.json
python dsv3wm/scripts/train.py --config dsv3wm/configs/A0_pilot.json --train dsv3wm/datasets/pilot-v1/train_register.jsonl --out dsv3wm/artifacts/A0_s0
python dsv3wm/scripts/evaluate.py --config dsv3wm/configs/A0_pilot.json --data dsv3wm/datasets/pilot-v1/dev.jsonl --domain register --limit 2 --adapter dsv3wm/artifacts/A0_s0/adapter --out dsv3wm/results/A0_dev.json
```

Dataset/run commands refuse overwriting previous artifacts. Reproduction must
use a new output/vault path; hashes bind the original inputs. The register-only
training projection is produced by `scripts/freeze_pilot.py` after data generation,
before training. No existing training data or frozen test labels are read.

`train.py` implements A0/A1/A2/A3/A4/A5. A3/A4 require independently re-verified
TRAIN-only trajectory labels; use `scripts/distill.py --help`. Stored adapters
remain unmerged. Prediction uses only the trained one-step/direct decoder;
MTP/legality training heads are omitted in inference. Generation uses no cache
until numerical cache parity and safe native memory qualification are performed.

Tiny native pilots do not establish efficacy. Confirmation CLI rejects all
nondevelopment data until a complete resource-qualified G5 protocol is frozen.
Specialist routing and MLA/cache conversion remain conditional follow-ups.
Large artifacts, fresh datasets and the external sealed vault stay outside Git;
`data_manifests/` and `results/EXCLUDED_ARTIFACTS.json` record hashes.
