"""Execute a source-bound independent decision; preserve every prior release.

This source remains unexecuted until its configuration, independent final
evidence, exact review joins, and completed backup proof have been frozen.
It supplies no dataset-quality judgment of its own.
"""
from pathlib import Path
import collections
import contextlib
import datetime
import hashlib
import json
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'BELIEF_V4_RELEASE_ATTEMPT01'
BATCH = 'belief-gap-v4'
DEST = ROOT / 'data/RBM-001/releases/v0002'
CONFIG = ROOT / 'quality_reports' / (PREFIX + '_CONFIG.json')
FREEZE = ROOT / 'quality_reports' / (PREFIX + '_SOURCE_FREEZE.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(65536):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resolve(name):
    path = ROOT / name
    assert isinstance(name, str) and not Path(name).is_absolute()
    assert '..' not in Path(name).parts and path.resolve().is_relative_to(ROOT)
    assert path.is_file() and not path.is_symlink(), name
    return path


def new(name, value):
    with (ROOT / 'quality_reports' / name).open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write('\n')


def check_bindings(bindings):
    assert isinstance(bindings, list) and bindings
    assert len({v['path'] for v in bindings}) == len(bindings)
    for value in bindings:
        path = resolve(value['path'])
        assert path.stat().st_size == value['bytes'], value['path']
        assert sha(path) == value['sha256'], value['path']
    return len(bindings)


def code_paths():
    paths = list((ROOT / 'factory').glob('*.py'))
    paths += list((ROOT / 'oracles').glob('*.py'))
    paths += list((ROOT / 'schemas').glob('*.md'))
    paths += list((ROOT / 'schemas').rglob('*.schema.json'))
    paths += list((ROOT / 'generators').rglob('*.py'))
    return sorted(p.relative_to(ROOT).as_posix() for p in paths)


def assert_gate(value, assertions):
    # Only the independently issued, frozen evidence values authorize execution.
    for assertion in assertions:
        actual = value
        for key in assertion['keys']:
            actual = actual[key]
        expected = assertion['expected']
        assert type(actual) is type(expected) and actual == expected, assertion['keys']


def preserve_reports(names):
    preserved = {}
    history = ROOT / 'handoff/report_history/belief-v4-publication-attempt01'
    for name in names:
        path = resolve(name)
        digest = sha(path)
        archive = history / (digest + '-' + path.name)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            assert archive.stat().st_size == path.stat().st_size and sha(archive) == digest
        else:
            with path.open('rb') as source, archive.open('xb') as output:
                while block := source.read(65536):
                    output.write(block)
        assert sha(path) == digest and sha(archive) == digest
        preserved[name] = {'sha256': digest, 'bytes': path.stat().st_size,
                           'archive': archive.relative_to(ROOT).as_posix()}
    return preserved


def main():
    freeze = json.loads(FREEZE.read_bytes())
    assert freeze['runner_execution'] == '/root'
    assert freeze['source_sha256'] == sha(Path(__file__))
    assert freeze['configuration_sha256'] == sha(CONFIG)
    config = json.loads(CONFIG.read_bytes())
    assert config['batch'] == BATCH and config['release'] == DEST.relative_to(ROOT).as_posix()
    assert config['authoritative_adjudicator_execution'] == '/root/research_architect'
    assert config['prior_counts'] == {'records': 3748, 'source_families': 696}
    assert config['result_counts'] == {'records': 7204, 'source_families': 700}
    assert not DEST.exists(), 'Immutable destination already exists'
    checked = check_bindings(config['input_bindings'])
    for gate in config['independent_gates']:
        assert_gate(json.loads(resolve(gate['path']).read_bytes()), gate['assertions'])
    for name, digest in freeze['historical_release_bindings'].items():
        assert sha(resolve(name)) == digest, name
    assert len(freeze['historical_release_bindings']) == 210
    current_names = code_paths()
    assert current_names == sorted(freeze['code_sha256'])
    for name, digest in freeze['code_sha256'].items():
        assert sha(resolve(name)) == digest, name
    # Imports occur only after the exact implementation and inputs are verified.
    from factory.common import load_jsonl, record_hash
    from factory.pipeline import release, validate_release, approval_evidence, APPROVAL_ACTORS
    from factory.catalog import catalog, write_catalog
    candidate = ROOT / 'data/RBM-001/candidate_renderings' / (BATCH + '.jsonl')
    rows = load_jsonl(candidate)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 3456
    assert all(r['review_status'] == 'PENDING' and r['record_hash'] == record_hash(r) for r in rows)
    assert dict(collections.Counter(r['split'] for r in rows)) == {'train': 1728, 'dev_challenge': 1728}
    evidence = {}
    ids = []
    for role, actor in APPROVAL_ACTORS.items():
        reviews = load_jsonl(ROOT / 'reviews' / role / (BATCH + '.jsonl'))
        assert len(reviews) == len({r['sample_id'] for r in reviews}) == 3456
        assert all(r['reviewer_execution_id'] == actor for r in reviews)
        evidence[role] = {r['sample_id']: [r] for r in reviews}
        ids.extend(r['receipt_id'] for r in reviews)
    assert len(ids) == len(set(ids)) == 10368
    for row in rows:
        receipts, failures = approval_evidence(row, evidence)
        assert not failures and len(receipts) == 3, row['sample_id']
    prior_rows, prior_catalog = catalog()
    assert len(prior_rows) == 3748 and prior_catalog['active_source_families'] == 696
    historical = freeze['historical_release_bindings']
    preserved = preserve_reports(config['reports_to_preserve'])
    pre = {'at_utc': now(), 'runner_execution': '/root', 'source_sha256': sha(Path(__file__)),
           'configuration_sha256': sha(CONFIG), 'source_freeze_sha256': sha(FREEZE),
           'authoritative_adjudicator_execution': '/root/research_architect',
           'batch': BATCH, 'release': DEST.relative_to(ROOT).as_posix(),
           'independent_input_binding_count': checked, 'actual_approval_joins': 10368,
           'exact_input_bindings': config['input_bindings'], 'prior_reports_preserved': preserved,
           'historical_release_bindings': historical, 'code_sha256': freeze['code_sha256'],
           'approval_boundary': 'Root executes frozen independent adjudication; root supplies no data-quality verdict.',
           'same_task_revision': {'records': 3456, 'source_families': 4,
                                  'new_semantic_tasks_relative_to_v3_quarantine': 0,
                                  'new_source_families_relative_to_v3_quarantine': 0},
           'remote_boundary': 'New release bytes require their own subsequent actual remote-byte proof.'}
    new(PREFIX + '_PREEXECUTION.json', pre)
    started = time.monotonic()
    status, stages, result = 'FAILED_PRESERVED', [], {}
    log_path = ROOT / 'quality_reports' / (PREFIX + '_OUTPUT.txt')
    try:
        with log_path.open('x', encoding='utf-8') as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            created = release(BATCH, 'v0002', ['RBM-001'])
            assert created == [DEST.relative_to(ROOT).as_posix()]
            stages.append('new_immutable_release_written')
            validation = validate_release(DEST)
            assert validation['pass'] is True and validation['records'] == 3456
            stages.append('new_release_targets_reviews_controls_exports_checksums_validated')
            accepted = load_jsonl(DEST / 'canonical.jsonl')
            assert all(r['review_status'] == 'ACCEPTED' for r in accepted)
            assert len(accepted) == len({r['reviewed_candidate_hash'] for r in accepted}) == 3456
            assert not load_jsonl(ROOT / 'data/RBM-001/quarantine' / (BATCH + '.jsonl'))
            check_bindings(config['input_bindings'])
            assert all(sha(resolve(name)) == digest for name, digest in historical.items())
            assert code_paths() == current_names
            assert all(sha(resolve(name)) == digest for name, digest in freeze['code_sha256'].items())
            active, updated = write_catalog()
            assert len(active) == 7204 and updated['active_source_families'] == 700
            stages.append('active_catalog_updated_after_full_validation')
            result = {'created': created, 'validation': validation,
                      'active_catalog_counts': {'records': len(active), 'source_families': updated['active_source_families']},
                      'active_track_records': dict(collections.Counter(r['track_id'] for r in active)),
                      'historical_releases': len(updated['releases']),
                      'active_releases': sum(e['status'].startswith('ACTIVE') for e in updated['releases']),
                      'new_release_files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                            for p in sorted(DEST.iterdir()) if p.is_file()},
                      'active_catalog_sha256': sha(ROOT / 'manifests/ACTIVE_RELEASE_CATALOG.json')}
            status = 'COMPLETED_VALIDATED_LOCAL_RELEASE'
            print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as error:
        with log_path.open('a', encoding='utf-8') as output:
            traceback.print_exc(file=output)
        result = {'error_type': type(error).__name__, 'error': str(error), 'release_exists': DEST.exists()}
    receipt = {**pre, 'finished_at_utc': now(), 'status': status, 'completed_stages': stages,
               'wall_seconds': time.monotonic() - started, 'result': result,
               'exact_inputs_unchanged': all(sha(resolve(v['path'])) == v['sha256'] for v in config['input_bindings']),
               'historical_release_files_unchanged': all(sha(resolve(n)) == h for n, h in historical.items()),
               'release_code_set_unchanged': code_paths() == current_names,
               'output_path': log_path.relative_to(ROOT).as_posix(), 'output_sha256': sha(log_path)}
    new(PREFIX + '_EXECUTION.json', receipt)
    print(json.dumps({'status': status, 'receipt_sha256': sha(ROOT / 'quality_reports' / (PREFIX + '_EXECUTION.json')), 'result': result}))
    assert status == 'COMPLETED_VALIDATED_LOCAL_RELEASE'
    assert receipt['exact_inputs_unchanged'] and receipt['historical_release_files_unchanged'] and receipt['release_code_set_unchanged']


if __name__ == '__main__':
    main()
