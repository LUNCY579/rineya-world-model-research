"""Create a target-free legacy-input view; this grants no data approval."""
from pathlib import Path
import argparse
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('sample_id', 'semantic_parent_id', 'semantic_family_id', 'track_id',
          'semantic_task', 'rendered_input', 'split', 'record_hash')


def proof(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    freeze_path = ROOT / args.freeze
    assert proof(freeze_path)['sha256'] == args.expected_freeze_sha256
    frozen = json.loads(freeze_path.read_bytes())
    assert proof(Path(__file__)) == frozen['source']
    original = ROOT / frozen['input_path']
    assert proof(original) == frozen['input']
    destination = ROOT / frozen['view_path']
    manifest_path = ROOT / frozen['manifest_path']
    assert not destination.exists() and not manifest_path.exists()
    original_rows = [json.loads(line) for line in original.read_bytes().splitlines()]
    assert len(original_rows) == len({row['sample_id'] for row in original_rows}) == 24
    assert all(row['track_id'] == 'RSM-001' and row['semantic_task']['kind'] == 'skills'
               for row in original_rows)
    rows = []
    for original_row in original_rows:
        row = {field: original_row[field] for field in FIELDS}
        visible_task = json.loads(row['rendered_input'].split('Task JSON:\n', 1)[1])
        assert canonical(visible_task) == canonical(row['semantic_task'])
        assert set(row) == set(FIELDS) and 'verified_target' not in row
        rows.append(row)
    with destination.open('xb') as out:
        for row in rows:
            out.write((canonical(row) + '\n').encode('utf-8'))
    emitted = [json.loads(line) for line in destination.read_bytes().splitlines()]
    assert canonical(emitted) == canonical(rows)
    assert proof(original) == frozen['input']
    assert proof(Path(__file__)) == frozen['source']
    manifest = {
        'actual_author_execution': '/root', 'actual_runner_execution': '/root',
        'status': 'PASS_EXACT_TARGET_FREE_LEGACY_INPUT_VIEW',
        'source': frozen['source'], 'source_freeze_sha256': args.expected_freeze_sha256,
        'original_path': frozen['input_path'], 'original': frozen['input'],
        'view_path': frozen['view_path'], 'view': proof(destination),
        'fields': list(FIELDS), 'records': len(rows),
        'source_families': len({row['semantic_family_id'] for row in rows}),
        'splits': dict(collections.Counter(row['split'] for row in rows)),
        'typed_task_and_raw_prompt_preserved': True,
        'targets_and_internal_controls_removed': True,
        'full_record_hash_is_an_opaque_original_custody_identifier': True,
        'purpose': 'Input grammar and normalized graph comparison; no new gold or novelty verdict.',
        'data_approval': False, 'remote_backup_claim': False,
    }
    with manifest_path.open('x') as out:
        json.dump(manifest, out, sort_keys=True, indent=2)
        out.write('\n')
    print(json.dumps({'records': 24, 'view': manifest['view'], 'manifest': proof(manifest_path)}))


if __name__ == '__main__':
    main()
