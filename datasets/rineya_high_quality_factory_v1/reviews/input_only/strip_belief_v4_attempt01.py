"""Produce an input-only view after a separately frozen V4 production receipt.

This is custody and input-lineage work, never a prediction or data approval.
No target, producer control, reviewer judgment, or privileged metadata is exported.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import traceback

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    'record_hash', 'rendered_input', 'sample_id', 'semantic_family_id',
    'semantic_parent_id', 'semantic_task', 'split', 'track_id',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        out.write(data)


def path_in_root(name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Expected a factory-relative path')
    path = ROOT / rel
    if any(p.is_symlink() for p in [path, *path.parents] if p != ROOT.parent):
        raise ValueError('Symlink path forbidden in custody view')
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Path escaped factory root')
    return path


def check_bindings(freeze):
    for name, binding in freeze['bindings'].items():
        data = path_in_root(name).read_bytes()
        if len(data) != binding['bytes'] or sha(data) != binding['sha256']:
            raise ValueError('Frozen byte binding failed: ' + name)
    if freeze['bindings'][str(Path(__file__).relative_to(ROOT))]['sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('Wrong frozen view source')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    fb = path_in_root(args.freeze).read_bytes()
    if sha(fb) != args.expected_freeze_sha256:
        raise ValueError('Freeze checksum failed')
    freeze = json.loads(fb)
    check_bindings(freeze)
    try:
        candidate_bytes = path_in_root(freeze['candidate']).read_bytes()
        candidate = [json.loads(line) for line in candidate_bytes.splitlines()]
        prior = [json.loads(line) for line in path_in_root(freeze['prior_input_view']).read_bytes().splitlines()]
        if len(candidate) != 3456 or len(prior) != 3456:
            raise ValueError('Expected complete original and fresh presentation cohorts')
        old_prefix, new_prefix = freeze['prior_id_prefix'], freeze['fresh_id_prefix']

        def coordinate(row, prefix):
            value = row['sample_id']
            if not isinstance(value, str) or not value.startswith(prefix):
                raise ValueError('Unexpected input-lineage sample prefix')
            return value[len(prefix):]

        old = {coordinate(row, old_prefix): row for row in prior}
        fresh = {coordinate(row, new_prefix): row for row in candidate}
        if len(old) != 3456 or len(fresh) != 3456 or old.keys() != fresh.keys():
            raise ValueError('Input-lineage coordinate set mismatch')
        joins = []
        for key, row in fresh.items():
            previous = old[key]
            for name in ['rendered_input', 'semantic_task', 'split', 'track_id']:
                if type(row[name]) != type(previous[name]) or canonical(row[name]) != canonical(previous[name]):
                    raise ValueError('Original student input changed: ' + name + ':' + key)
            for name in ['semantic_parent_id', 'semantic_family_id']:
                if row[name] != previous[name].replace(old_prefix, new_prefix, 1):
                    raise ValueError('Source-family input-lineage mismatch: ' + key)
            if row['review_status'] != 'PENDING' or row['review_receipt_ids']:
                raise ValueError('View requires fresh unreviewed producer candidate')
            if row['track_id'] != 'RBM-001':
                raise ValueError('Unexpected track')
            joins.append({'fresh_sample_id': row['sample_id'],
                          'prior_sample_id': previous['sample_id'],
                          'input_sha256': sha(canonical({name: row[name] for name in ['rendered_input', 'semantic_task', 'split', 'track_id']})),
                          'same_typed_student_input_and_split': True})
        view_rows = [{name: row[name] for name in FIELDS} for row in candidate]
        view_bytes = b''.join(canonical(row) + b'\n' for row in view_rows)
        view_name = freeze['output_view']
        join_name = freeze['output_input_lineage']
        manifest_name = freeze['output_manifest']
        for name in [view_name, join_name, manifest_name]:
            if path_in_root(name).exists():
                raise ValueError('Refusing to overwrite input-only evidence: ' + name)
        check_bindings(freeze)
        write_new(path_in_root(view_name), view_bytes)
        join_bytes = canonical({'scope': 'Exact typed input/split lineage only; no target or control validation, predictions, verdicts, or approval.',
                                'fresh_records': 3456, 'prior_records': 3456,
                                'all_complete_coordinates_joined': True,
                                'joins': joins}) + b'\n'
        write_new(path_in_root(join_name), join_bytes)
        check_bindings(freeze)
        receipt = {
            'view_author_execution': '/root', 'view_runner_execution': '/root',
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'COMPLETE_FROZEN_INPUT_ONLY_VIEW_NO_DATA_APPROVAL',
            'source_freeze': args.freeze, 'source_freeze_sha256': sha(fb),
            'bindings': freeze['bindings'], 'source_sha256': sha(Path(__file__).read_bytes()),
            'view': view_name, 'view_bytes': len(view_bytes), 'view_sha256': sha(view_bytes),
            'input_lineage': join_name, 'input_lineage_sha256': sha(join_bytes),
            'records': 3456, 'source_families': len({row['semantic_family_id'] for row in candidate}),
            'view_fields': list(FIELDS),
            'removed_fields': sorted(set().union(*(row.keys() for row in candidate)) - set(FIELDS)),
            'same_original_typed_student_inputs_and_splits': True,
            'scope': 'Student-visible inputs plus record/family/split join keys only. Stored targets, producer property/role/locator claims and review evidence absent. All fresh own predictions and first judgments must be frozen before canonical access.',
            'record_hash_note': 'Complete original candidate hash, not a hash of the stripped row.',
            'independence_disclosure': 'Same prior V3 inputs, prior gold exposure and reused own solvers remain disclosed; this does not establish unseen blind semantics.',
            'data_approval': False,
        }
        write_new(path_in_root(manifest_name), json.dumps(receipt, indent=2, sort_keys=True).encode() + b'\n')
        print(json.dumps({k: receipt[k] for k in ['status', 'records', 'source_families', 'view_sha256', 'input_lineage_sha256']}))
    except Exception as exc:
        failure = {'actual_execution': '/root', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'source_sha256': sha(Path(__file__).read_bytes()), 'freeze_sha256': sha(fb),
                   'exception_type': type(exc).__name__, 'exception': str(exc),
                   'traceback': traceback.format_exc(), 'data_approval': False}
        write_new(path_in_root(freeze['failure_receipt']), json.dumps(failure, indent=2, sort_keys=True).encode() + b'\n')
        raise


if __name__ == '__main__':
    main()
