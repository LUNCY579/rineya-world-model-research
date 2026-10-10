#!/usr/bin/env python3
"""Frozen independent direct-function replay. No target main, Git or network.
Only four authorized opaque metadata files are read; they are never copied.
All remote-like fixtures stay in memory and are not actual remote proof.
"""
import concurrent.futures
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
FACTORY = LOGS.parent
FREEZE = LOGS / 'REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_replay_freeze.json'
OUT = LOGS / 'REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_replay_execution.json'
RESULTS = []
GAPS = []

def sha(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        while chunk := handle.read(65536): h.update(chunk)
    return h.hexdigest()

def oid(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

freeze = json.loads(FREEZE.read_bytes())
assert freeze['source']['sha256'] == sha(Path(__file__))
for row in freeze['frozen_inputs']:
    assert sha(FACTORY / row['path']) == row['sha256']

def load(name, filename):
    path = LOGS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

r = load('independent_remote_v2', 'recover_remote_checkpoint_exact_public_metadata_v2.py')
g = load('independent_gzip_v2', 'verify_archival_remote_lossless_exact_public_metadata_v2.py')
h = r.metadata_helper()

def forbidden(*args, **kwargs):
    raise AssertionError('Git, network, subprocess, or target main is forbidden in replay')

r.git = r.api = r.command = r.process_stream = r.main = forbidden
g.main = forbidden
payloads = {row['path']: (FACTORY / row['path']).read_bytes() for row in h.EXPECTED_FILES}

def case(name, function):
    function()
    RESULTS.append({'case': name, 'status': 'PASS'})

def reject(function):
    try: function()
    except (ValueError, TypeError, KeyError, gzip.BadGzipFile, EOFError): return
    raise AssertionError('Expected rejection did not occur')

def observe_gap(name, function):
    try: function()
    except (ValueError, TypeError, KeyError):
        RESULTS.append({'case': name, 'status': 'REJECTED'})
        return
    GAPS.append({'case': name, 'status': 'ACCEPTED_CORRUPTION_REQUIRES_REVIEW'})

def stream(row, context, names=None, data=None, blob=None):
    data = payloads[row['path']] if data is None else data
    return r.hash_stream(io.BytesIO(data), row['size'], oid(data) if blob is None else blob,
                         True, [row['path']] if names is None else names, context)

def complete(context):
    for row in h.EXPECTED_FILES: stream(row, context)
    report = r.complete_metadata_stream_report(context)
    r.validate_metadata_report(report, True)
    return report

case('all_four_source_exact_scans', lambda: [r.check_source(FACTORY, {k: row[k] for k in ('path','size','sha256')}, False) for row in h.EXPECTED_FILES])
source_report = h.report(require_complete=True)
local_report = complete('local_git')
remote_report = complete('actual_remote')
case('both_complete_stream_contexts', lambda: [r.validate_metadata_report(x,True) for x in (local_report,remote_report)])
row = h.EXPECTED_FILES[0]
data = payloads[row['path']]
for name, value in [('truncated',data[:-1]),('overlong',data+b'X'),('full_hash_changed',data[:-1]+bytes([data[-1]^1]))]:
    case('metadata_'+name, lambda value=value: reject(lambda: stream(row,'actual_remote',data=value)))
case('wrong_git_sha1', lambda: reject(lambda: stream(row,'local_git',blob='0'*40)))
case('duplicate_context_paths', lambda: reject(lambda: stream(row,'local_git',names=[row['path'],row['path']])))
case('mixed_known_unknown_paths', lambda: reject(lambda: stream(row,'local_git',names=[row['path'],'backup_logs/ordinary.txt'])))
case('unknown_alias', lambda: reject(lambda: stream(row,'actual_remote',names=[row['path']+'.copy'])))
case('invalid_stream_context', lambda: reject(lambda: stream(row,'cached_remote')))
case('shared_exact_two_original_paths', lambda: stream(h.EXPECTED_FILES[2],'actual_remote',names=[x['path'] for x in h.EXPECTED_FILES[2:]]))
ordinary = b'ordinary prose with a harmless short value\n'
case('ordinary_stream', lambda: r.hash_stream(io.BytesIO(ordinary),len(ordinary),oid(ordinary),True,['backup_logs/ordinary.txt'],'actual_remote'))
secret = b'sk' + b'-' + b'A'*34
case('ordinary_secret_rejected', lambda: reject(lambda: r.hash_stream(io.BytesIO(secret),len(secret),oid(secret),True,['backup_logs/ordinary.txt'],'actual_remote')))
for field,value in [('size',True),('sha256','0'*64),('matches',[]),('full_size_sha256_gitsha1_verified_before_exception_acceptance',False)]:
    bad = copy.deepcopy(remote_report);bad['four_exception_files'][0][field]=value
    case('stream_report_'+field, lambda bad=bad: reject(lambda: r.validate_metadata_report(bad,True)))
bad = copy.deepcopy(remote_report);bad['four_exception_files'][0]['matches'][0]['pattern_index']=True
case('typed_span_index',lambda: reject(lambda:r.validate_metadata_report(bad,True)))
bad = copy.deepcopy(remote_report);bad['four_exception_files'][0]['git_blob']='0'*40
observe_gap('stream_report_unbound_valid_sha1',lambda:r.validate_metadata_report(bad,True))
bad = copy.deepcopy(remote_report);bad['context']='cached_remote'
observe_gap('stream_report_invalid_context',lambda:r.validate_metadata_report(bad,True))

with r._LOCK:
    r._STREAM_REPORTS['actual_remote'].clear()
groups = [(h.EXPECTED_FILES[0],[h.EXPECTED_FILES[0]['path']]),(h.EXPECTED_FILES[1],[h.EXPECTED_FILES[1]['path']]),(h.EXPECTED_FILES[2],[x['path'] for x in h.EXPECTED_FILES[2:]])]
def threaded():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda pair: stream(pair[0],'actual_remote',names=pair[1]),groups))
    r.validate_metadata_report(r.complete_metadata_stream_report('actual_remote'),True)
case('two_worker_complete_stream_reporting',threaded)

spec = json.loads((LOGS/'belief-v4-approved-release-r1_SPEC.json').read_bytes())
case('actual_exact_scope_metadata',lambda:r.validate_exact_scope(spec))
bad = copy.deepcopy(spec);bad['raw_size_exclusions'].pop()
case('scope_missing_raw',lambda:reject(lambda:r.validate_exact_scope(bad)))
stage = json.loads((LOGS/'belief-v4-approved-release-r1_STAGE.json').read_bytes())
stage_module = load('independent_stage_diagnostics_only','run_provisional_archival_checkpoint_v4_attempt02.py')
stage_module.git = stage_module.run = stage_module.main = forbidden
case('actual_stage_diagnostics_schema',lambda:r.validate_stage_observation(stage,stage_module))
bad_stage = copy.deepcopy(stage);bad_stage['actual_exit_code']=0
case('stage_unrestricted_false_pass',lambda:reject(lambda:r.validate_stage_observation(bad_stage,stage_module)))

# In-memory synthetic complete proof assembled only from custody ledger metadata.
# Never publish this fixture or claim its provenance as actual remote verification.
ledger = [json.loads(line) for line in (LOGS/'belief-v4-approved-release-r1_STAGE_LEDGER.jsonl').read_text().splitlines()]
blobs = [{**{k:v[k] for k in ('git_blob','size','sha256','paths')},'verification_source':'actual_remote_raw_GitHub_blob'} for v in ledger if v['phase']=='actual_index_blob_verified']
verified = {'status':'BACKED_UP','commit_sha':r.EXPECTED_COMMIT,'remote_commit_sha':r.EXPECTED_COMMIT,
    'spec_sha256':r.EXPECTED_SPEC_SHA,'inventory':spec['files'],'files':3158,'total_bytes':1263090770,
    'verification_source_sha256':g.DEPENDENCY_SHA,'remote_verification':{'verified_files':3158,'verified_unique_blobs':len(blobs),'blob_receipts':blobs},
    'oversized_raw_local_verification':[{k:e[k] for k in ('path','size','sha256')}|{'lossless_local_roundtrip':True} for e in spec['raw_size_exclusions']],
    'capped_source_secret_scan_before':source_report,'capped_source_secret_scan_after':source_report,
    'actual_local_git_metadata_exception_scan':local_report,'actual_remote_metadata_exception_scan':remote_report}
def bindings(v):return g.validate_bindings(spec,stage,v,r.EXPECTED_COMMIT,r.EXPECTED_SPEC_SHA,r)
case('synthetic_full_binding_only_no_remote_authority',lambda:bindings(verified))
inventory,actual=bindings(verified)
paths={r.PREFIX+'/'+name:{'type':'blob','mode':'100644','size':entry['size'],'sha':actual[name]['git_blob']} for name,entry in inventory.items()}
work=g.plan_representations(spec,inventory,actual,paths,r.PREFIX,31,r)
refs=g.manifest_reference_coverage(spec,inventory,actual,paths,r.PREFIX)
case('exact31_pairs32_refs31_manifest_paths',lambda: (len(work)==31 and len(refs)==32 and len({x['manifest']['path'] for x in refs})==31) or (_ for _ in ()).throw(AssertionError('Wrong exact plan')))
bad_paths=copy.deepcopy(paths);bad_paths[r.PREFIX+'/'+refs[0]['manifest']['path']]['mode']='120000'
case('manifest_remote_mode_corruption',lambda:reject(lambda:g.manifest_reference_coverage(spec,inventory,actual,bad_paths,r.PREFIX)))
bad_verified=copy.deepcopy(verified);bad_verified['remote_verification']['blob_receipts'][0]['verification_source']='cached_remote'
case('full_blob_cached_provenance_rejected',lambda:reject(lambda:bindings(bad_verified)))
bad_verified=copy.deepcopy(verified);bad_verified['actual_remote_metadata_exception_scan']=copy.deepcopy(local_report)
observe_gap('gzip_accepts_local_report_as_remote_report',lambda:bindings(bad_verified))
bad_verified=copy.deepcopy(verified);bad_verified['actual_remote_metadata_exception_scan']['four_exception_files'][0]['git_blob']='0'*40
observe_gap('gzip_accepts_metadata_sha1_not_bound_to_remote_blob',lambda:bindings(bad_verified))

# Synthetic ordinary bytes only, never original metadata content.
with tempfile.TemporaryDirectory(prefix='REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_ordinary_',dir=LOGS) as temporary:
    temp=Path(temporary);raw_data=b'ordinary synthetic evidence\n'*101
    raw_path=temp/'raw.txt';raw_path.write_bytes(raw_data)
    packed_data=gzip.compress(raw_data,mtime=0)
    raw={'path':'synthetic/raw.txt','size':len(raw_data),'sha256':hashlib.sha256(raw_data).hexdigest()}
    packed={'path':'synthetic/raw.txt.gz','size':len(packed_data),'sha256':hashlib.sha256(packed_data).hexdigest()}
    blob={'sha':oid(packed_data)}
    def consume(payload=packed_data,p=packed,b=blob):return g.consume_remote(io.BytesIO(payload),raw_path,raw,p,b,temp)
    case('ordinary_gzip_direct_byte_compare',consume)
    case('gzip_truncated',lambda:reject(lambda:consume(packed_data[:-1])))
    case('gzip_wrong_blob',lambda:reject(lambda:consume(b={'sha':'0'*40})))
    corrupt=bytearray(packed_data);corrupt[-8]^=1;corrupt=bytes(corrupt)
    corrupt_meta={'path':packed['path'],'size':len(corrupt),'sha256':hashlib.sha256(corrupt).hexdigest()}
    case('gzip_crc_after_correct_compressed_binding',lambda:reject(lambda:consume(corrupt,corrupt_meta,{'sha':oid(corrupt)})))
    calls=[]
    class Transport:
        REPOSITORY=r.REPOSITORY
        source_path=staticmethod(lambda factory,name:raw_path)
        @staticmethod
        def process_stream(args,consume):calls.append(args);return consume(io.BytesIO(packed_data[:-1]))
    case('transport_corruption_has_no_local_fallback',lambda:reject(lambda:g.fetch_one(Transport,temp,raw,packed,blob)))
    case('transport_exact_raw_no_cache_command',lambda: (len(calls)==1 and calls[0][:2]==['gh','api'] and 'Accept: application/vnd.github.raw+json' in calls[0] and 'Cache-Control: no-cache' in calls[0] and '--cache' not in calls[0]) or (_ for _ in ()).throw(AssertionError('Transport changed')))

for row in freeze['frozen_inputs']:assert sha(FACTORY/row['path'])==row['sha256']
assert freeze['source']['sha256']==sha(Path(__file__))
report={'status':'BOUNDED_DIRECT_FUNCTION_REPLAY_COMPLETE_WITH_REPORT_VALIDATION_GAPS' if GAPS else 'BOUNDED_DIRECT_FUNCTION_REPLAY_PASS',
    'author_execution':'/root/backup_manager/approved_release_join_review','source_sha256':sha(Path(__file__)),
    'source_freeze_sha256':sha(FREEZE),'passed_cases':len(RESULTS),'results':RESULTS,'accepted_corruptions':GAPS,
    'target_mains_executed':False,'git_commands':0,'network_calls':0,'actual_remote_payload_fetches':0,
    'metadata_content_copied_to_new_paths':False,'synthetic_full_fixture_published':False,'grants_data_approval':False,
    'remote_pairs':len(work),'manifest_reference_records':len(refs),'unique_manifest_paths':len({x['manifest']['path'] for x in refs}),
    'unique_raw_bytes':sum(e['size'] for e in spec['raw_size_exclusions']),
    'local_all_representation_comparison_bytes':sum(e['size']*len(e['lossless_representations']) for e in spec['raw_size_exclusions'])}
with OUT.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
print(json.dumps({k:report[k] for k in ('status','passed_cases','accepted_corruptions','network_calls','git_commands','actual_remote_payload_fetches')}))
