#!/usr/bin/env python3
"""Additional frozen direct API truth probes; no main/Git/network or metadata copies."""
import copy,hashlib,importlib.util,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
L=Path(__file__).resolve().parent
F=L.parent
FREEZE=L/'REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_supplement_freeze.json'
OUT=L/'REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_supplement_execution.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  while b:=s.read(65536):h.update(b)
 return h.hexdigest()
fr=json.loads(FREEZE.read_bytes())
assert fr['source']['sha256']==sha(Path(__file__))
for x in fr['frozen_inputs']:assert sha(F/x['path'])==x['sha256']
def load(name,file):
 s=importlib.util.spec_from_file_location(name,L/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('independent_supplement_remote','recover_remote_checkpoint_exact_public_metadata_v2.py')
g=load('independent_supplement_gzip','verify_archival_remote_lossless_exact_public_metadata_v2.py')
h=r.metadata_helper()
def forbidden(*a,**kw):raise AssertionError('Main/Git/network forbidden')
r.git=r.api=r.process_stream=r.command=r.main=g.main=forbidden
results=[]
def observe(name,call):
 try:call()
 except (ValueError,TypeError,KeyError):results.append({'case':name,'result':'REJECTED'});return
 results.append({'case':name,'result':'ACCEPTED'})
row=h.EXPECTED_FILES[0]
payload=(F/row['path']).read_bytes()
def stream(blob):return r.exact_secret_stream(io.BytesIO(payload),row['size'],blob,[row['path']],'local_git')
for blob in (None,False,123,[],{},'', 'bad', '0'*40):
 observe('expected_git_blob_'+type(blob).__name__+'_'+('None' if blob is None else 'invalid'),lambda blob=blob:stream(blob))
none_record=copy.deepcopy(r._STREAM_REPORTS['local_git'].get(row['path']))
observe('complete_report_missing_three_paths',lambda:r.complete_metadata_stream_report('local_git'))
for entry in h.EXPECTED_FILES:
 data=(F/entry['path']).read_bytes();blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
 r.exact_secret_stream(io.BytesIO(data),entry['size'],blob,[entry['path']],'local_git')
valid=r.complete_metadata_stream_report('local_git')
for field,value in [('allowed_exact_match_detections',True),('unbound_secret_detections',False),('scanner_source_sha256','0'*64),('configuration_sha256','0'*64),('existing_regexes_unchanged',1)]:
 q=copy.deepcopy(valid);q[field]=value
 observe('report_typed_'+field,lambda q=q:r.validate_metadata_report(q,True))
q=copy.deepcopy(valid);del q['context']
observe('report_missing_context',lambda:r.validate_metadata_report(q,True))
q=copy.deepcopy(valid);q['context']='actual_remote'
observe('report_context_substitution',lambda:r.validate_metadata_report(q,True))

minimal={'author_execution':g.AUTHOR,'frozen_before_first_import_test_execution':True,
 'runner_source':{'path':Path(g.__file__).relative_to(F).as_posix(),**r.hash_file(Path(g.__file__))},'frozen_inputs':[]}
observe('gzip_freeze_empty_dependencies',lambda:g.validate_preparation_freeze(minimal,r))
q=copy.deepcopy(minimal);q['frozen_inputs']=[{'path':'backup_logs/recover_remote_checkpoint_exact_public_metadata_v2.py','size':0,'sha256':'0'*64}]
observe('gzip_freeze_changed_dependency',lambda:g.validate_preparation_freeze(q,r))
with tempfile.TemporaryDirectory(prefix='REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_synthetic_freeze_',dir=L) as t:
 path=Path(t)/'empty_dependencies.json'
 minimal_recovery={'author_execution':r.AUTHOR,'frozen_before_first_import_test_execution':True,
  'source':{'path':Path(r.__file__).relative_to(F).as_posix(),**r.hash_file(Path(r.__file__))},'frozen_inputs':[]}
 with path.open('x') as s:json.dump(minimal_recovery,s)
 args=SimpleNamespace(workers=2,allow_evolving_source=False,runner='/root',commit=r.EXPECTED_COMMIT,
  expected_spec_sha256=r.EXPECTED_SPEC_SHA,metadata_config=L/'EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json',
  expected_metadata_config_sha256=r.METADATA_CONFIG_SHA,source_freeze=path,expected_freeze_sha256=sha(path),
  prior_evidence=[L/'belief-v4-approved-release-r1_STAGE.json',L/'belief-v4-approved-release-r1_LOCAL_COMMIT.json',L/'belief-v4-approved-release-r1_STAGE_LEDGER.jsonl'])
 observe('recovery_freeze_empty_dependencies',lambda:r.check_new_source_freeze(args))
for x in fr['frozen_inputs']:assert sha(F/x['path'])==x['sha256']
assert fr['source']['sha256']==sha(Path(__file__))
report={'status':'BOUNDED_SUPPLEMENTAL_API_TRUTH_PROBES_COMPLETE','author_execution':'/root/backup_manager/approved_release_join_review',
 'source_sha256':sha(Path(__file__)),'source_freeze_sha256':sha(FREEZE),'results':results,
 'missing_expected_git_id_recorded_verified_flag':none_record is not None and none_record.get('full_size_sha256_gitsha1_verified_before_exception_acceptance') is True,
 'target_mains_executed':False,'git_commands':0,'network_calls':0,'actual_remote_payload_fetches':0,
 'metadata_content_copied_to_new_paths':False,'grants_data_approval':False,
 'interpretation':'Current main always supplies actual blob IDs and root-specified authentic freeze hashes. Accepted None and omitted dependency manifests are pure-API proof-truth/completeness gaps; altered report provenance is a downstream validator gap. No actual remote persistence conclusion follows.'}
with OUT.open('x') as s:json.dump(report,s,indent=2);s.write('\n')
print(json.dumps(report))
