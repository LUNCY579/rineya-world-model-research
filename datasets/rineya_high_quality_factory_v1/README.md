# Rineya high-quality dataset factory

Original synthetic CPU-only data for ten complementary research tracks. The factory follows local RCWM/BWM findings and keeps the original research directories read-only. No model training or efficacy evaluation is performed.

Semantic families are assigned to train or exposed development challenges before twins and renderings. Canonical records contain privileged provenance and gold labels; student exports contain only the prompt and supervised answer. Independent oracle, expert, adversarial and adjudication receipts are joined by exact candidate hash. Missing or stale approval fails closed. Review executions use the same underlying model; correlated error remains a limitation.

Run from this directory with Python 3.12 and `jsonschema` 4.x:

```sh
/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'
/opt/anaconda3/bin/python3 backup_logs/test_checkpoint_backup.py
/opt/anaconda3/bin/python3 -m factory.catalog
/opt/anaconda3/bin/python3 -m factory.status
```

Load active approved canonical rows through `factory.catalog.catalog()`. Historical releases, including superseded versions and exact-hash errata, remain available for audit. Concatenating every release directory would mix obsolete records with approved replacements.

Generation writes pending candidates. Use a fresh batch name and record the seed, offset and count before generating. Audit with `-m factory.pipeline audit --batch NAME`. Release assembly requires actual independent reviews and adjudication; passing an oracle does not approve a sample. Release directories are immutable. Corrected data requires a new release version and explicit supersession record.

Protected test sets and weights are excluded. Structural template overlap is measured and disclosed; the development partition is not a sealed generalization benchmark. Dataset usefulness remains a hypothesis until a separately authorized training experiment.

Git backup uses a separate dedicated worktree of the existing private repository. Each backup receipt certifies only its exact file inventory. Consult PROJECT_STATUS.md and NEXT_SESSION_HANDOFF.md before continuing.
