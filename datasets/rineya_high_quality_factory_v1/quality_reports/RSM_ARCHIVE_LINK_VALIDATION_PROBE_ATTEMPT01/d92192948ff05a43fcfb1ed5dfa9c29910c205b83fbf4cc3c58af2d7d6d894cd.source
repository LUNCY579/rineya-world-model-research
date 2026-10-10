"""Actual HAND archive-validation link attacks; no production approval."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE = 'quality_reports/RSM_CORE_INSTALLATION_ATTEMPT01/INSTALLED_RUNTIME_SOURCE_FREEZE.json'
EXPECTED = 'e8db5c7569076e03d2a67f2616fe2e0e4db47c35ac8a8050f1ff80210204ec60'


def verify():
    raw = (ROOT / FREEZE).read_bytes()
    assert sha256(raw).hexdigest() == EXPECTED
    freeze = json.loads(raw)
    assert len(freeze['bindings']) == 463
    for relative, binding in freeze['bindings'].items():
        data = (ROOT / relative).read_bytes()
        assert sha256(data).hexdigest() == binding['sha256'] and len(data) == binding['bytes']
        assert data == (ROOT / binding['archive']).read_bytes()


def main():
    verify()  # Complete closure before first shared core/HAND test import.
    sys.path.insert(0, str(ROOT))
    path = ROOT / 'tests/rsm_core_integration_test.py'
    spec = importlib.util.spec_from_file_location('root_frozen_hand_link_probe_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module.RSMCoreIntegrationTests()
    case.setUpClass()
    results = []
    with case.sandbox() as (root, rows):
        report = module.R.require_rsm_controls(rows, module.R.BATCH)
        ledger = module.R.preserve_rsm_bytes(report)
        manifest = {'source_archive': 'reproducibility/source_blobs/{sha256}.source',
                    'rsm_source_and_evidence': ledger,
                    'reproducibility_code_sha256': {p: b['sha256'] for p, b in ledger.items()}}
        module.R.validate_rsm_release_bindings(report, manifest)
        directory = root / 'reproducibility/source_blobs'
        blob = directory / (ledger[module.R.TASK_SOLVER_PATH]['sha256'] + '.source')
        held = root / 'held-official-source'
        blob.rename(held)
        blob.symlink_to(held)
        try:
            module.R.validate_rsm_release_bindings(report, manifest)
        except OSError:
            results.append({'attack': 'required-official-archive-leaf-symlink', 'rejected': True})
        else:
            raise AssertionError('Archive leaf symlink was accepted')
        blob.unlink()
        held.rename(blob)
        held_directory = root / 'held-archive-directory'
        directory.rename(held_directory)
        directory.symlink_to(held_directory, target_is_directory=True)
        try:
            module.R.validate_rsm_release_bindings(report, manifest)
        except OSError:
            results.append({'attack': 'archive-parent-symlink', 'rejected': True})
        else:
            raise AssertionError('Archive parent symlink was accepted')
        directory.unlink()
        held_directory.rename(directory)
        module.R.validate_rsm_release_bindings(report, manifest)
    verify()
    receipt = {'status': 'PASS_TWO_ACTUAL_HAND_ARCHIVE_LINK_REJECTIONS',
               'at_utc': datetime.now(timezone.utc).isoformat(), 'actual_runner_execution': '/root',
               'runtime_freeze_sha256': EXPECTED, 'all463_runtime_bindings_unchanged': True,
               'cases': results, 'actual_production_data_read': False, 'data_approval': False}
    with (HERE / 'EXECUTION.json').open('x') as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
