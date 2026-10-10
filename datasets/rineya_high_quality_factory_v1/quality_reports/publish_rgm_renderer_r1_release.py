"""Publish only the independently approved renderer-r1 bytes, preserving history."""
from pathlib import Path
import collections
import contextlib
import datetime
import hashlib
import json
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'RGM_RENDERER_R1_RELEASE'
BATCH = 'planning-gap-v2-renderer-r1'
DEST = ROOT / 'data/RGM-001/releases/v0003'
EXPECTED = {
    'factory/pipeline.py': 'd3d328e4b604f85fdd0c88d5f39aee2773689626412f72f7aeae9768429c5b62',
    'factory/planning_controls.py': '352f5fc0247b339293206958144e25f7d4437477f4a60702d926b805f25c18c3',
    'data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl': 'b306f67b2208406027d26d95094c649913e590291857764a9bf17edb9bcd923c',
    'data/RGM-001/manifests/planning-gap-v2-renderer-r1.family_controls.json': 'e69a9c130b40aebd0dfdefa9b0f96ec075d3a383073aa963e63446e084bf5c63',
    'reviews/expert/planning-gap-v2-renderer-r1.jsonl': '9505e5921c117aff287d12634020ff33e6f6c41fb9e3b0af3617b661201bcc89',
    'reviews/adversarial/planning-gap-v2-renderer-r1.jsonl': '6b58be3a5b738b3ff01c7f6de936e0eddef249e04d46ac6de15df60c7aa071c9',
    'reviews/adjudication/planning-gap-v2-renderer-r1.jsonl': '52187f0a3287d544de28d816337ac196844073a25783fe8930cabdb11c019474',
    'reviews/adjudication/planning-gap-v2-renderer-r1.audit.json': '7c33a3c3810820202af7ada2cd52b3da89566cd2ef0f224902d9f874d21ff981',
    'reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json': '73d0864b489d8e2dea8d818eef25dc429808f580c0cad7802ca585ba535d228a',
    'quality_reports/RGM_RENDERER_R1_FROZEN_REVIEW_EVIDENCE_JOIN.json': 'dbdd35b3e6575f26ce9722750f74f4925a596a3b2c5d35cea508a250620d2ff8',
    'quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json': '34732f8a7a5cab9addd30d113ae36f1e3db1c5d26ecab9bf785c706dcc78fe16',
    'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl': '57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23',
    'data/RGM-001/quarantine/planning-gap-v2.jsonl': '61f3575f32144abf0f2bb1a28c84e85ef41ef988ca9feb5a7c8713357b64b043',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new(name, value):
    with (ROOT / 'quality_reports' / name).open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.write('\n')


def code_paths():
    paths = list((ROOT / 'factory').glob('*.py'))
    paths += list((ROOT / 'oracles').glob('*.py'))
    paths += list((ROOT / 'schemas').glob('*.md'))
    paths += list((ROOT / 'schemas').rglob('*.schema.json'))
    paths += list((ROOT / 'generators').rglob('*.py'))
    return sorted(p.relative_to(ROOT).as_posix() for p in paths)


def check_bindings(value):
    bindings = value['bindings']
    records = list(bindings.values()) if isinstance(bindings, dict) else bindings
    assert isinstance(records, list) and records
    for record in records:
        path = ROOT / record['path']
        assert path.is_file() and sha(path) == record['sha256'], record['path']
        assert path.stat().st_size == record['bytes'], record['path']
    return len(records)


def main():
    from factory.common import load_jsonl, record_hash
    from factory.pipeline import release, validate_release, approval_evidence, APPROVAL_ACTORS
    from factory.catalog import catalog, write_catalog

    freeze = json.loads((ROOT / 'quality_reports' / (PREFIX + '_SOURCE_FREEZE.json')).read_text())
    assert sha(Path(__file__)) == freeze['source_sha256']
    assert not DEST.exists(), 'Immutable release already exists'
    for name, expected in EXPECTED.items():
        assert sha(ROOT / name) == expected, name
    final = json.loads((ROOT / 'reviews/adjudication' / (BATCH + '.final_execution_receipt.json')).read_text())
    adjud = json.loads((ROOT / 'reviews/adjudication' / (BATCH + '.audit.json')).read_text())
    assert final['publication_eligible'] is True and final['completed_execution'] is True
    assert final['actual_decision_counts'] == {'ACCEPT': 408}
    assert adjud['publication_eligible'] is True and adjud['decision_counts'] == {'ACCEPT': 408}
    checked_bindings = check_bindings(final) + check_bindings(adjud)
    rows = load_jsonl(ROOT / 'data/RGM-001/candidate_renderings' / (BATCH + '.jsonl'))
    assert len(rows) == len({r['sample_id'] for r in rows}) == 408
    assert all(r['review_status'] == 'PENDING' and r['record_hash'] == record_hash(r) for r in rows)
    assert dict(collections.Counter(r['split'] for r in rows)) == {'train': 340, 'dev_challenge': 68}
    evidence = {}
    receipt_ids = []
    for role, actor in APPROVAL_ACTORS.items():
        reviews = load_jsonl(ROOT / 'reviews' / role / (BATCH + '.jsonl'))
        assert len(reviews) == len({r['sample_id'] for r in reviews}) == 408
        assert all(r['reviewer_execution_id'] == actor for r in reviews)
        evidence[role] = {r['sample_id']: [r] for r in reviews}
        receipt_ids.extend(r['receipt_id'] for r in reviews)
    assert len(receipt_ids) == len(set(receipt_ids)) == 1224
    for row in rows:
        receipts, failures = approval_evidence(row, evidence)
        assert not failures and len(receipts) == 3, row['sample_id']
    prior_rows, prior_catalog = catalog()
    assert len(prior_rows) == 3340 and prior_catalog['active_source_families'] == 648
    historical = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.glob('data/*/releases/*/*') if p.is_file()}
    assert len(historical) == 198
    code_names = code_paths()
    all_bound = dict(EXPECTED)
    all_bound.update({name: sha(ROOT / name) for name in code_names})
    # Existing reports are archived before the release's mandatory deterministic audit.
    preserved = {}
    for name in ['manifests/ACTIVE_RELEASE_CATALOG.json',
                 'reviews/oracle/' + BATCH + '.jsonl',
                 'audits/leakage/' + BATCH + '.json',
                 'audits/diversity/' + BATCH + '-planning-controls.json']:
        path = ROOT / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dest = ROOT / 'handoff/report_history/rgm-renderer-r1-publication' / (digest + '-' + path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
        preserved[name] = {'sha256': digest, 'archive': str(dest.relative_to(ROOT)), 'bytes': len(raw)}
    pre = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'runner_execution': '/root', 'source_sha256': sha(Path(__file__)),
           'authoritative_adjudicator_execution': '/root/research_architect',
           'batch': BATCH, 'release': 'data/RGM-001/releases/v0003',
           'exact_input_bindings': all_bound, 'historical_release_bindings': historical,
           'prior_reports_preserved': preserved, 'all_actual_review_joins': 1224,
           'independent_final_binding_count': checked_bindings,
           'prior_catalog_counts': {'records': 3340, 'source_families': 648},
           'approval_boundary': 'Root executes independent adjudication; root does not supply a data-quality verdict.',
           'same_task_revision': {'records': 408, 'semantic_tasks': 204, 'source_families': 48, 'new_semantics_relative_to_quarantine': 0},
           'remote_boundary': 'New local release bytes are not remotely verified by this execution.'}
    new(PREFIX + '_PREEXECUTION.json', pre)
    start = time.monotonic()
    stages = []
    status = 'FAILED'
    result = {}
    log_path = ROOT / 'quality_reports' / (PREFIX + '_OUTPUT.txt')
    try:
        with log_path.open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            created = release(BATCH, 'v0003', ['RGM-001'])
            stages.append('immutable_release_written')
            assert created == ['data/RGM-001/releases/v0003'], created
            validation = validate_release(DEST)
            stages.append('new_release_fully_validated')
            assert validation['pass'] is True and validation['records'] == 408
            accepted = load_jsonl(DEST / 'canonical.jsonl')
            assert all(r['review_status'] == 'ACCEPTED' for r in accepted)
            assert len({r['reviewed_candidate_hash'] for r in accepted}) == 408
            assert not load_jsonl(ROOT / 'data/RGM-001/quarantine' / (BATCH + '.jsonl'))
            assert all(sha(ROOT / name) == digest for name, digest in historical.items())
            assert all(sha(ROOT / name) == digest for name, digest in all_bound.items())
            assert code_paths() == code_names, 'Release code set changed during execution'
            active, updated = write_catalog()
            assert len(active) == 3748 and updated['active_source_families'] == 696
            stages.append('active_catalog_updated_after_validation')
            result = {'created': created, 'validation': validation,
                      'active_catalog_counts': {'records': len(active), 'source_families': updated['active_source_families']},
                      'active_track_records': dict(collections.Counter(r['track_id'] for r in active)),
                      'historical_releases': len(updated['releases']),
                      'active_releases': sum(e['status'].startswith('ACTIVE') for e in updated['releases']),
                      'new_release_files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(DEST.iterdir()) if p.is_file()},
                      'active_catalog_sha256': sha(ROOT / 'manifests/ACTIVE_RELEASE_CATALOG.json')}
            status = 'COMPLETED_VALIDATED_LOCAL_RELEASE'
            print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as error:
        with log_path.open('a') as log:
            traceback.print_exc(file=log)
        result = {'error_type': type(error).__name__, 'error': str(error), 'release_exists': DEST.exists()}
    after = {name: sha(ROOT / name) for name in all_bound}
    receipt = {**pre, 'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': status, 'completed_stages': stages, 'wall_seconds': time.monotonic() - start,
               'result': result, 'exact_inputs_unchanged': after == all_bound,
               'historical_release_files_unchanged': all(sha(ROOT / name) == digest for name, digest in historical.items()),
               'release_code_set_unchanged': code_paths() == code_names,
               'output_path': str(log_path.relative_to(ROOT)), 'output_sha256': sha(log_path)}
    new(PREFIX + '_EXECUTION.json', receipt)
    print(json.dumps({'status': status, 'wall_seconds': receipt['wall_seconds'], 'receipt_sha256': sha(ROOT / 'quality_reports' / (PREFIX + '_EXECUTION.json')), 'result': result}))
    assert status == 'COMPLETED_VALIDATED_LOCAL_RELEASE' and receipt['exact_inputs_unchanged'] and receipt['historical_release_files_unchanged'] and receipt['release_code_set_unchanged']


if __name__ == '__main__':
    main()
