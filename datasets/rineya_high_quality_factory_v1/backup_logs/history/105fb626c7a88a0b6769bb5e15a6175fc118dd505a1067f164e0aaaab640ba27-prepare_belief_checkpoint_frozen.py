"""Defer concurrently authored RGM files from a completed belief checkpoint.

The unchanged manager-authored inventory builder still validates every included
byte, secret scan, size limit and lossless representation. The root-authored
filter only defers the named new RGM wave, never an immutable release.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backup_logs import prepare_checkpoint_inventory as inventory


def deferred_rgm(relative):
    return (any(name in relative for name in ('planning-gap-v1', 'planning_gap', 'rgm_planning'))
            or relative.startswith('reviews/oracle/RGM-'))


def main():
    arguments = sys.argv[1:]
    checkpoint = arguments[arguments.index('--checkpoint-id') + 1]
    original = Path.rglob
    deferred = []

    def filtered(path, *args, **kwargs):
        for candidate in original(path, *args, **kwargs):
            if path.resolve() == ROOT and candidate.is_file():
                relative = candidate.relative_to(ROOT).as_posix()
                if deferred_rgm(relative):
                    assert 'releases' not in candidate.parts, 'Never defer a release artifact'
                    deferred.append({'path': relative, 'reason': 'Concurrent new RGM wave; local until frozen next checkpoint'})
                    continue
            yield candidate

    Path.rglob = filtered
    try:
        inventory.main()
    finally:
        Path.rglob = original
    spec_path = ROOT / 'backup_logs' / f'{checkpoint}_SPEC.json'
    spec = json.loads(spec_path.read_text())
    report_path = ROOT / 'backup_logs' / f'{checkpoint}_FROZEN_SCOPE.json'
    report = {'actual_author_execution': '/root', 'actual_runner_execution': '/root',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_id': checkpoint,
              'scope': 'Completed belief evidence and prior factory; new concurrent RGM production/oracle files deferred',
              'deferred_local_files': sorted(deferred, key=lambda item: item['path']),
              'all_release_artifacts_in_scope': True, 'grants_data_approval': False,
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'unchanged_inventory_builder_sha256': hashlib.sha256(Path(inventory.__file__).read_bytes()).hexdigest()}
    inventory.atomic_json(report_path, report)
    entry, _ = inventory.snapshot(report_path)
    spec['files'] = [item for item in spec['files'] if item['path'] != entry['path']] + [entry]
    spec['files'].sort(key=lambda item: item['path'])
    spec['frozen_scope_report'] = entry
    spec['frozen_scope_filter_author_execution'] = '/root'
    spec['scope_note'] = report['scope'] + '; all included original validation and lossless checks retained; no data approval'
    inventory.atomic_json(spec_path, spec)
    print(json.dumps({'frozen_files': len(spec['files']), 'bytes': sum(item['size'] for item in spec['files']),
                      'deferred_local_files': len(deferred), 'validation_status': spec['validation_status']}))


if __name__ == '__main__':
    main()
