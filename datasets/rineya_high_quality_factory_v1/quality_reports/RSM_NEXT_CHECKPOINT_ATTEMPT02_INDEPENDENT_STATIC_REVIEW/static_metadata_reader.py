"""Independent stdlib metadata/hash readback only; no target import or phase."""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FACTORY = Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset')
PREP = FACTORY / 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT02'
OWN = FACTORY / 'quality_reports/RSM_NEXT_CHECKPOINT_ATTEMPT02_INDEPENDENT_STATIC_REVIEW'
FREEZE_SHA = '4d7bebfdb6e1d353504db18c6294735a1168a809fc88775bacc8c20984d02b00'
READY_SHA = '7fde44cbcceaaac4c887e4c322a2f0b3ec70afc1a83ca07815467885f4a66699'
READS = {}
ERRORS = []


def digest(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Absolute literal file identity required')
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise ValueError('Symlink byte-read identity rejected')
    before = path.stat()
    if not path.is_file():
        raise ValueError('Regular byte input required')
    h = hashlib.sha256()
    count = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('Input replaced before opening')
        while block := stream.read(65536):
            count += len(block)
            h.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if tuple(getattr(before, k) for k in fields) != tuple(getattr(after, k) for k in fields) or count != before.st_size:
        raise ValueError('Static byte input changed while reading')
    actual = {'path': str(path), 'size': count, 'sha256': h.hexdigest()}
    READS[str(path)] = actual
    return actual


def bound(row, absolute=False):
    if type(row) is not dict or set(row) != {'path', 'size', 'sha256'}:
        raise ValueError('Closed three-field metadata tuple required')
    if type(row['size']) is not int or row['size'] < 0:
        raise ValueError('Integer byte count required')
    value = row['path']
    p = Path(value)
    if type(value) is not str or str(p) != value or '\\' in value or '\x00' in value or any(k in ('.', '..') for k in p.parts):
        raise ValueError('Literal byte path required')
    if not absolute and (p.is_absolute() or '.git' in p.parts):
        raise ValueError('Factory-relative metadata tuple required')
    path = p if absolute else FACTORY / p
    actual = digest(path)
    if (actual['size'], actual['sha256']) != (row['size'], row['sha256']):
        raise ValueError('Declared bytes differ at ' + value)
    return path


def metadata(path, expected=None):
    actual = digest(path)
    if expected is not None and actual['sha256'] != expected:
        raise ValueError('Externally bound metadata changed')
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def check(label, action):
    try:
        return action()
    except Exception as error:
        ERRORS.append({'check': label, 'error_type': type(error).__name__, 'error': str(error)})
        return None


own_sha = sys.argv[1]
if digest(Path(__file__).absolute())['sha256'] != own_sha or digest(OWN / 'source_blobs' / (own_sha + '.source'))['sha256'] != own_sha:
    raise ValueError('Independent reader and archived source must match before metadata work')
freeze = metadata(PREP / 'SOURCE_PREPARATION_FREEZE.json', FREEZE_SHA)
ready = metadata(PREP / 'SOURCE_READY.json', READY_SHA)
runtime = metadata(PREP / 'RUNTIME_SOURCE_FREEZE.json')
policy = metadata(PREP / 'ADMISSION_POLICY.json')
rows = freeze['frozen_inputs']
check('preparation_count_and_uniqueness', lambda: None if len(rows) == 277 and len({r['path'] for r in rows}) == 277 else (_ for _ in ()).throw(ValueError('Expected 277 unique preparation paths')))
for row in rows:
    check('preparation:' + row['path'], lambda row=row: bound(row))
for row in runtime['runtime_bindings']:
    check('runtime_current:' + row['current']['path'], lambda row=row: bound(row['current'], True))
    check('runtime_raw:' + row['raw_archive']['path'], lambda row=row: bound(row['raw_archive']))
    check('runtime_pair:' + row['current']['path'], lambda row=row: None if (row['current']['size'], row['current']['sha256']) == (row['raw_archive']['size'], row['raw_archive']['sha256']) else (_ for _ in ()).throw(ValueError('Current/raw tuple mismatch')))
for row in runtime['source_code_bindings']:
    check('collected_source:' + row['path'], lambda row=row: bound(row))
for label in ('collector_source', 'collector_config', 'collector_preexecution'):
    check(label, lambda label=label: bound(runtime[label]))
for row in freeze['raw_source_archives']:
    original = check('own_original:' + row['original_path'], lambda row=row: digest(FACTORY / row['original_path']))
    archived = check('own_archive:' + row['archive_path'], lambda row=row: digest(FACTORY / row['archive_path']))
    if original and archived and (original['size'], original['sha256']) != (archived['size'], archived['sha256']):
        ERRORS.append({'check': 'own_original_raw_equality', 'path': row['original_path']})

# Read admission metadata identities/counts only; do not open producer code or output payloads.
core = metadata(bound(policy['core_result_fixed']['completion']))
core_execution = metadata(bound(policy['core_result_fixed']['execution']))
core_runtime = metadata(bound(policy['core_fixed']['runtime_freeze']))
producer = {k: metadata(bound(v)) for k, v in policy['producer_fixed'].items()}
living = metadata(bound(policy['living_transition_fixed']['completion']))
prior = metadata(bound(next(r for r in rows if r['path'] == 'backup_logs/belief-v4-approved-release-r1_SPEC.json')))
report = {
    'schema': 'independent-static-metadata-byte-readback-v1',
    'actual_reviewer_execution': '/root/rsm_core_integration_auditor',
    'status': 'MATCHED_STATIC_BINDINGS' if not ERRORS else 'STATIC_BYTE_MISMATCHES',
    'target_imports_tests_compilations_phases_inventory_git_network_packing': 0,
    'data_approval': False,
    'reader_source_sha256': own_sha,
    'source_preparation_sha256': FREEZE_SHA,
    'source_ready_sha256': READY_SHA,
    'preparation_binding_count': len(rows),
    'runtime_current_raw_pair_count': len(runtime['runtime_bindings']),
    'collected_source_count': len(runtime['source_code_bindings']),
    'own_source_archive_count': len(freeze['raw_source_archives']),
    'operative_membership_matches_collected_source': freeze['operative_source_paths'] == [r['path'] for r in runtime['source_code_bindings']],
    'core_metadata_only': {'status': core.get('status'), 'tests_run': core_execution.get('tests_run'), 'module_count': len(core_execution.get('actual_test_modules', [])), 'old_runtime_pair_count': len(core_runtime['bindings'])},
    'producer_metadata_only': {'final_status': producer['final'].get('status'), 'artifact_count': len(producer['final']['all_owned_preparation_execution_output_inventory_excluding_this_receipt']), 'prepared_pair_count': len(producer['prepared_freeze']['bindings']), 'runtime_pair_count': len(producer['runtime_freeze']['bindings']), 'output_count': len(producer['final']['eight_core_artifacts'])},
    'living_metadata_only': {'status': living.get('status'), 'old_count': len(living.get('old', {})), 'new_count': len(living.get('new', {}))},
    'prior_metadata_only': {'capped_path_count': len(prior['files']), 'release_union_count': len({r['path'] for r in [*prior['files'], *prior['raw_size_exclusions']] if '/releases/' in '/' + r['path']})},
    'errors': ERRORS,
    'actual_bytes_hashed': sorted(READS.values(), key=lambda r: r['path']),
    'boundary': 'Metadata and full hash bytes only; no original producer implementation/output semantic inspection or approved sample vote.',
}
with (OWN / 'STATIC_METADATA_BYTE_READBACK.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps({k: v for k, v in report.items() if k not in ('actual_bytes_hashed', 'core_metadata_only', 'producer_metadata_only', 'living_metadata_only')}, sort_keys=True))
