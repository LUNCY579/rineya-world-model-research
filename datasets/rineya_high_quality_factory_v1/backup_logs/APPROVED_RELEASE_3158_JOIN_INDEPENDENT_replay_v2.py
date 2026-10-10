#!/usr/bin/env python3
"""Independent bounded frozen direct-function replay. Never calls main/Git/network.
All corruptions are in-memory metadata copies; no opaque producer bytes are copied.
"""
import argparse, ast, copy, datetime, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
FACTORY = Path(__file__).resolve().parent.parent
LOGS = FACTORY / 'backup_logs'
PREFIX = 'APPROVED_RELEASE_3158_JOIN_INDEPENDENT_'
FREEZE = LOGS / (PREFIX + 'replay_v2_SOURCE_FREEZE.json')
OUTPUT = LOGS / (PREFIX + 'replay_v2_EXECUTION.json')
TARGET = 'backup_logs/join_belief_v4_approved_release_r1_custody_v2.py'
TARGET_SHA = 'aa8237fc0dc771b843246e9dc712454543582a828071c846381a03c32deea76d'
TARGET_FREEZE = 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_V2_SOURCE_FREEZE.json'
TARGET_FREEZE_SHA = '2d24eb2a106bcf7cd3c9dcbab1564be92d75fde9227250d3a56f01f0ac621928'

def artifact(name):
    p = FACTORY / name
    assert not p.is_symlink() and p.is_file()
    data = p.read_bytes()
    return {'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def read(name):
    return json.loads((FACTORY / name).read_bytes())

def lines(name):
    return [json.loads(v) for v in (FACTORY / name).read_text().splitlines()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expected-freeze-sha256', required=True)
    a = p.parse_args()
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == a.expected_freeze_sha256
    frozen = json.loads(FREEZE.read_bytes())
    assert frozen['frozen_before_first_target_import_and_execution'] is True
    assert artifact(Path(__file__).relative_to(FACTORY).as_posix()) == frozen['source']
    for row in frozen['frozen_inputs']:
        assert artifact(row['path']) == row
    assert artifact(TARGET)['sha256'] == TARGET_SHA
    assert artifact(TARGET_FREEZE)['sha256'] == TARGET_FREEZE_SHA
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_ast = ast.parse((FACTORY / TARGET).read_text())
    target_main = next(v for v in source_ast.body if isinstance(v, ast.FunctionDef) and v.name == 'main')
    calls = [(n.lineno, n.func.id) for n in ast.walk(target_main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert min(n for n, f in calls if f == 'validate_preparation_freeze') < min(n for n, f in calls if f == 'load_dependencies')
    sp = importlib.util.spec_from_file_location('independent_frozen_join_v2', FACTORY / TARGET)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    freeze = read(TARGET_FREEZE)
    assert m.B is None and m.M is None and m.G is None
    m.validate_preparation_freeze(freeze)
    m.load_dependencies()
    forbidden_calls = []
    def forbidden(*args, **kwargs):
        forbidden_calls.append('main/Git/network/process called')
        raise AssertionError('Bounded replay prohibits main/Git/network/process calls')
    m.main = forbidden
    m.verify_current = forbidden
    m.bounded_commit_diff = forbidden
    for module, names in ((m.B, ('api', 'process_stream', 'full_checkpoint_guards')), (m.M, ('git', 'text', 'run'))):
        for name in names:
            if hasattr(module, name):
                setattr(module, name, forbidden)
    m.subprocess.Popen = forbidden
    m.subprocess.run = forbidden
    spec, stage, local, scope, ec = [read(n) for n in (m.SPEC, m.STAGE, m.LOCAL, m.CONFIG, m.EXECUTION_CONFIG)]
    bindings = ec['bindings']
    full, gz = read(bindings['full']['path']), read(bindings['gzip']['path'])
    sl, fl, gl = lines(m.STAGE_LEDGER), lines(bindings['full_ledger']['path']), lines(bindings['gzip_ledger']['path'])
    full_sha = bindings['full']['sha256']
    tree = full['remote_verification']['remote_tree_sha']
    results = []
    def positive(name, fn):
        try:
            fn()
            results.append({'case': name, 'expected': 'PASS', 'actual': 'PASS'})
        except Exception as e:
            results.append({'case': name, 'expected': 'PASS', 'actual': 'FAILED', 'error_type': type(e).__name__, 'error': str(e)})
    def reject(name, fn):
        try:
            fn()
            results.append({'case': name, 'expected': 'REJECT', 'actual': 'ACCEPTED_CORRUPTION'})
        except Exception as e:
            results.append({'case': name, 'expected': 'REJECT', 'actual': 'REJECT', 'error_type': type(e).__name__, 'error': str(e)})
    def corrupt(value, mutate, fn):
        bad = copy.deepcopy(value)
        mutate(bad)
        return fn(bad)
    def proofs(f=None, g=None, s=None):
        return m.validate_proofs(spec, stage if s is None else s, full if f is None else f, gz if g is None else g, full_sha, tree)
    positive('complete55 preparation before dependency loading', lambda: m.validate_preparation_freeze(freeze))
    mandatory = set(m.REQUIRED_PREPARATION_HASHES) | {freeze['source']['path'], freeze['tests']['path']}
    assert len(mandatory) == 55
    for name in sorted(mandatory):
        reject('mandatory omission ' + name, lambda name=name: corrupt(freeze, lambda d: d.update(frozen_inputs=[v for v in d['frozen_inputs'] if v['path'] != name]), m.validate_preparation_freeze))
    for key in ('source', 'tests', 'scope_config'):
        reject('declared artifact omitted ' + key, lambda key=key: corrupt(freeze, lambda d: d.pop(key), m.validate_preparation_freeze))
        reject('declared artifact SHA substituted ' + key, lambda key=key: corrupt(freeze, lambda d: d[key].update(sha256='0'*64), m.validate_preparation_freeze))
    for label, mutate in (
        ('empty list', lambda d: d.update(frozen_inputs=[])),
        ('duplicate row', lambda d: d['frozen_inputs'].append(d['frozen_inputs'][0])),
        ('bool size', lambda d: d['frozen_inputs'][0].update(size=True)),
        ('wrong integer size', lambda d: d['frozen_inputs'][0].update(size=d['frozen_inputs'][0]['size'] + 1)),
        ('uppercase hash', lambda d: d['frozen_inputs'][0].update(sha256='F'*64)),
        ('extra row field', lambda d: d['frozen_inputs'][0].update(unbound=True)),
        ('missing row hash', lambda d: d['frozen_inputs'][0].pop('sha256')),
        ('wrong author', lambda d: d.update(author_execution='/other')),
        ('false preparation', lambda d: d.update(frozen_before_first_import_test_execution=False))):
        reject('preparation ' + label, lambda mutate=mutate: corrupt(freeze, mutate, m.validate_preparation_freeze))
    for name in ('/outside', 'backup_logs/../outside', 'backup_logs//outside', 'backup_logs\\outside', 'backup_logs/./outside'):
        reject('literal alias ' + name, lambda name=name: m.literal_path(name))
    positive('exact stage/local3158 paths2474 IDs', lambda: m.stage_coverage(spec, stage, sl, local))
    positive('actual full9fac gzip0ad1 complete crossjoin', proofs)
    stage_paths = m.stage_coverage(spec, stage, sl, local)
    remote_paths = proofs()
    positive('stage/full3158 actual payload groups crossjoin', lambda: m.crossjoin(stage_paths, remote_paths))
    positive('actual full/gzip ledgers complete', lambda: m.validate_ledgers(full, gz, fl, gl))
    positive('scope26 deferred29 local', lambda: m.validate_scope(scope, spec))
    positive('execution four completed immutable bindings', lambda: m.validate_execution_config(ec))
    for row in bindings.values():
        positive('actual evidence file hash ' + row['path'], lambda row=row: m.require(m.artifact(row['path'], row['sha256']) == row, 'Evidence row mismatch'))
        reject('actual evidence wrong expected hash ' + row['path'], lambda row=row: m.artifact(row['path'], '0'*64))
    for key, value in (('runner_execution', '/other'), ('expected_parent_sha', '0'*40), ('commit_sha', '0'*40), ('spec_sha256', '0'*64), ('actual_exit_code', 0), ('warning_count', 5), ('strict_diff_check', 'PASS'), ('restricted_changed_path_count', 481), ('grants_data_approval', True)):
        reject('stage ' + key, lambda key=key, value=value: corrupt(stage, lambda d: d.update({key: value}), lambda d: m.validate_stage(spec, d)))
    for label, mutate in (
        ('source omitted', lambda a: a.remove(next(e for e in a if e['phase'] == 'source_target_verified'))),
        ('source hash changed', lambda a: next(e for e in a if e['phase'] == 'source_target_verified').update(sha256='0'*64)),
        ('source duplicate', lambda a: a.insert(0, copy.deepcopy(next(e for e in a if e['phase'] == 'source_target_verified')))),
        ('index omitted', lambda a: a.remove(next(e for e in a if e['phase'] == 'actual_index_blob_verified'))),
        ('index duplicate ID', lambda a: a.insert(0, copy.deepcopy(next(e for e in a if e['phase'] == 'actual_index_blob_verified')))),
        ('index duplicate path', lambda a: next(e for e in a if e['phase'] == 'actual_index_blob_verified')['paths'].append(next(e for e in a if e['phase'] == 'actual_index_blob_verified')['paths'][0])),
        ('terminal wrong commit', lambda a: a[-1].update(commit_sha='0'*40))):
        reject('stage ledger ' + label, lambda mutate=mutate: corrupt(sl, mutate, lambda d: m.stage_coverage(spec, stage, d, local)))
    for key, value in (('status', 'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING'), ('remote_commit_sha', m.COMMIT), ('actual_exit_code', 0)):
        reject('local preserved ' + key, lambda key=key, value=value: corrupt(local, lambda d: d.update({key: value}), lambda d: m.stage_coverage(spec, stage, sl, d)))
    for key, value in (('status','PENDING'), ('files',3157), ('local_git_verified_unique_blobs',2473), ('current_source_matching_files',3157), ('runner_execution','/other'), ('source_freeze_sha256','0'*64), ('verification_source_sha256','0'*64), ('metadata_scanner_source_sha256','0'*64), ('metadata_exception_config_sha256','0'*64), ('current_source_evolution',[{'path':'changed'}]), ('current_source_evolution_at_finish',[{'path':'changed'}]), ('local_raw_representation_decode_iterations',31), ('local_raw_representation_decoded_bytes',1724728576)):
        reject('full ' + key, lambda key=key, value=value: corrupt(full, lambda d: d.update({key:value}), lambda d: proofs(f=d)))
    report_keys = ('capped_source_secret_scan_before','capped_source_secret_scan_after','actual_local_git_metadata_exception_scan','actual_remote_metadata_exception_scan')
    for key in report_keys:
        reject('metadata whole report missing ' + key, lambda key=key: corrupt(full, lambda d: d.pop(key), lambda d: proofs(f=d)))
        reject('metadata exact four list missing ' + key, lambda key=key: corrupt(full, lambda d: d[key].pop('four_exception_files'), lambda d: proofs(f=d)))
        reject('metadata exact four list empty ' + key, lambda key=key: corrupt(full, lambda d: d[key].update(four_exception_files=[]), lambda d: proofs(f=d)))
        for i in range(4):
            for field, value in (('path','unbound/path'), ('size',0), ('sha256','0'*64), ('matches',[])):
                reject('metadata row ' + key + '/' + str(i) + '/' + field, lambda key=key, i=i, field=field, value=value: corrupt(full, lambda d: d[key]['four_exception_files'][i].update({field:value}), lambda d: proofs(f=d)))
            if key.startswith('actual_'):
                for field, value in (('git_blob',None), ('git_blob',False), ('git_blob','0'*40), ('full_size_sha256_gitsha1_verified_before_exception_acceptance',False)):
                    reject('metadata stream row ' + key + '/' + str(i) + '/' + field + '/' + str(value), lambda key=key, i=i, field=field, value=value: corrupt(full, lambda d: d[key]['four_exception_files'][i].update({field:value}), lambda d: proofs(f=d)))
                reject('metadata stream Git ID omitted ' + key + '/' + str(i), lambda key=key, i=i: corrupt(full, lambda d: d[key]['four_exception_files'][i].pop('git_blob'), lambda d: proofs(f=d)))
        if key.startswith('actual_'):
            for value in (None, False, 'wrong', 'local_git' if key == 'actual_remote_metadata_exception_scan' else 'actual_remote'):
                reject('metadata context ' + key + '/' + str(value), lambda key=key, value=value: corrupt(full, lambda d: d[key].update(context=value), lambda d: proofs(f=d)))
            reject('metadata context omitted ' + key, lambda key=key: corrupt(full, lambda d: d[key].pop('context'), lambda d: proofs(f=d)))
    for key in ('checkpoint_custody_guards_before', 'checkpoint_custody_guards_after'):
        for field in full[key]:
            reject('full custody guard omitted ' + key + '/' + field, lambda key=key, field=field: corrupt(full, lambda d: d[key].pop(field), lambda d: proofs(f=d)))
    reject('full private ADMIN changed', lambda: corrupt(full, lambda d: d['private_repository_evidence'].update(viewerPermission='WRITE'), lambda d: proofs(f=d)))
    for i in range(3):
        reject('prior evidence SHA ' + str(i), lambda i=i: corrupt(full, lambda d: d['prior_evidence'][i].update(sha256='0'*64), lambda d: proofs(f=d)))
    for field, value in (('http_accept','cached'), ('cache_policy','inherited'), ('remote_tree_sha','0'*40), ('verified_unique_blobs',2473)):
        reject('full remote ' + field, lambda field=field, value=value: corrupt(full, lambda d: d['remote_verification'].update({field:value}), lambda d: proofs(f=d)))
    for label, mutate in (
        ('wrong provenance', lambda a: a[0].update(verification_source='cache')),
        ('wrong SHA', lambda a: a[0].update(sha256='0'*64)),
        ('empty coverage', lambda a: a[0].update(paths=[])),
        ('unowned path', lambda a: a[0]['paths'].append('outside')),
        ('duplicate path', lambda a: a[0]['paths'].append(a[0]['paths'][0])),
        ('duplicate group', lambda a: a.__setitem__(1,copy.deepcopy(a[0])))):
        reject('full blob ' + label, lambda mutate=mutate: corrupt(full, lambda d: mutate(d['remote_verification']['blob_receipts']), lambda d: proofs(f=d)))
    for key, value in (('actual_remote_gzip_fetches',32), ('unique_manifest_path_count',32), ('frozen_representation_reference_count',31), ('unique_raw_gzip_pair_count',32), ('raw_bytes_exactly_recovered',1766951239), ('actual_decoded_bytes_including_multiple_representations',1766951239), ('local_fullproof_decode_iterations',31), ('status','PENDING'), ('source_sha256','0'*64), ('dependency_sha256','0'*64), ('source_freeze_sha256','0'*64), ('verified_inventory_receipt_sha256','0'*64), ('actual_runner_execution','/root'), ('grants_data_approval',True), ('stream_chunk_bytes',131072), ('bounded_spool_memory_bytes',16777216)):
        reject('gzip ' + key, lambda key=key, value=value: corrupt(gz, lambda d: d.update({key:value}), lambda d: proofs(g=d)))
    for key, value in (('actual_remote_fetch_this_run',False), ('decoded_byte_for_byte_equal_to_preserved_raw',False), ('remote_git_blob','0'*40), ('remote_gzip_sha256','0'*64), ('raw_sha256','0'*64)):
        reject('gzip pair ' + key, lambda key=key, value=value: corrupt(gz, lambda d: d['actual_remote_gzip_proofs'][0].update({key:value}), lambda d: proofs(g=d)))
    reject('gzip duplicate raw/gzip pair', lambda: corrupt(gz, lambda d: d['actual_remote_gzip_proofs'].__setitem__(1,copy.deepcopy(d['actual_remote_gzip_proofs'][0])), lambda d: proofs(g=d)))
    for i in range(32):
        reject('manifest perraw reference omitted ' + str(i), lambda i=i: corrupt(gz, lambda d: d['full_frozen_lossless_manifest_reference_coverage'].pop(i), lambda d: proofs(g=d)))
    reject('manifest actual Git ID crosswired', lambda: corrupt(gz, lambda d: d['full_frozen_lossless_manifest_reference_coverage'][0].update(actual_remote_git_blob='0'*40), lambda d: proofs(g=d)))
    reject('stage/full/current Git ID crosswired', lambda: corrupt(stage_paths, lambda d: d[next(iter(d))].update(git_blob='0'*40), lambda d: m.crossjoin(remote_paths,d)))
    for label, mutate in (
        ('no terminal',lambda a:a.pop()),
        ('remote hash crosswired',lambda a:next(e for e in a if e['phase']=='actual_remote_blob_verified').update(sha256='0'*64)),
        ('local omitted',lambda a:a.remove(next(e for e in a if e['phase']=='local_git_blob_verified'))),
        ('local duplicate replacing different ID',lambda a:a.__setitem__(next(i for i,e in enumerate(a) if e['phase']=='local_git_blob_verified' and e['git_blob']!=next(v for v in a if v['phase']=='local_git_blob_verified')['git_blob']),copy.deepcopy(next(v for v in a if v['phase']=='local_git_blob_verified')))),
        ('local extra duplicate',lambda a:a.insert(0,copy.deepcopy(next(e for e in a if e['phase']=='local_git_blob_verified'))))):
        reject('full ledger ' + label, lambda mutate=mutate: corrupt(fl,mutate,lambda d:m.validate_ledgers(full,gz,d,gl)))
    for label, mutate in (
        ('no terminal',lambda a:a.pop()),
        ('Git ID crosswired',lambda a:next(e for e in a if e['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF').update(remote_git_blob='0'*40)),
        ('duplicate pair',lambda a:a.insert(0,copy.deepcopy(next(e for e in a if e['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF'))))):
        reject('gzip ledger ' + label, lambda mutate=mutate: corrupt(gl,mutate,lambda d:m.validate_ledgers(full,gz,fl,d)))
    for key, value in (('postcut_completed_adjudication_files',28), ('files',True), ('grants_data_approval',True), ('manifest_references',31), ('remote_raw_gzip_pairs',32)):
        reject('scope ' + key,lambda key=key,value=value:corrupt(scope,lambda d:d.update({key:value}),lambda d:m.validate_scope(d,spec)))
    reject('scope historical26 omission',lambda:corrupt(scope,lambda d:d['historical_observed_pending_atcut'].pop(),lambda d:m.validate_scope(d,spec)))
    reject('scope postcut insertion into frozen inventory',lambda:corrupt(scope,lambda d:d['completed_postcut_adjudication_artifacts'][0].update(path=spec['files'][0]['path']),lambda d:m.validate_scope(d,spec)))
    for label, mutate in (
        ('missing gzip ledger',lambda d:d['bindings'].pop('gzip_ledger')),
        ('bool size',lambda d:d['bindings']['full'].update(size=True)),
        ('invalid hash',lambda d:d['bindings']['full'].update(sha256='bad')),
        ('false completion',lambda d:d.update(actual_completed_proofs_bound=False)),
        ('wrong runner',lambda d:d.update(actual_runner_execution='/root')),
        ('duplicate evidence path',lambda d:d['bindings']['gzip'].update(path=d['bindings']['full']['path']))):
        reject('execution config ' + label,lambda mutate=mutate:corrupt(ec,mutate,m.validate_execution_config))
    for row in frozen['frozen_inputs']:
        assert artifact(row['path']) == row
    assert artifact(Path(__file__).relative_to(FACTORY).as_posix()) == frozen['source']
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == a.expected_freeze_sha256
    bad = [v for v in results if v['actual'] not in ('PASS','REJECT')]
    report = {'schema':'independent-approved-release3158-join-v2-direct-replay','author_execution':'/root/backup_manager/approved_release_join_review',
      'status':'PASS' if not bad and not forbidden_calls else 'FAILED','started_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source':frozen['source'],'source_freeze_sha256':a.expected_freeze_sha256,'reviewed_source_sha256':TARGET_SHA,'reviewed_source_freeze_sha256':TARGET_FREEZE_SHA,
      'actual_proof_bindings':bindings,'mandatory_paths_omission_replays':len(mandatory),'cases':len(results),'positive_cases':sum(v['expected']=='PASS' for v in results),
      'rejected_corruptions':sum(v['actual']=='REJECT' for v in results),'accepted_corruptions':[v for v in bad if v['actual']=='ACCEPTED_CORRUPTION'],
      'failed_positives':[v for v in bad if v['actual']=='FAILED'],'forbidden_calls':forbidden_calls,'main_Git_network_calls':0,'metadata_fullcontents_copied_to_newpaths':0,
      'all_frozen_inputs_unchanged_at_finish':True,'grants_data_approval':False,'results':results}
    with OUTPUT.open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':report['status'],'cases':len(results),'accepted_corruptions':len(report['accepted_corruptions']),'failed_positives':len(report['failed_positives']),'execution':artifact(OUTPUT.relative_to(FACTORY).as_posix())}))
    if bad or forbidden_calls:raise SystemExit(1)

if __name__ == '__main__':main()
