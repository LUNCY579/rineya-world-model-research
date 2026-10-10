"""Static custody/runtime collection ONLY; no new reviewer/solver/test import.
Python bootstrap and declared stdlib import boundary is explicitly recorded.
Opaque private source bytes are archived/hash-bound, never parsed as code.
"""
import argparse
import ast
import collections
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
import linecache
import locale
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import tokenize
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
source_bytes=Path(__file__).read_bytes();source_hash=sha(source_bytes)
cfg=json.loads((HERE/'config.json').read_text())
# The following metadata objects are permitted custody/source assertions; no
# candidate target parsing, producer diagnostic result parsing or source import.
generation_path=ROOT/'manifests/option-resumption-pilot-v1_GENERATION.json'
generation=json.loads(generation_path.read_text())
manifest_path=ROOT/cfg['manifest'];manifest=json.loads(manifest_path.read_text())
archive=HERE/'raw-archives';archive.mkdir()
closure_archive=HERE/'runtime-archives';closure_archive.mkdir()
paths={str((HERE/n).relative_to(ROOT)):'OWN_STATIC_SOURCE_NOT_IMPORTED' for n in ['comparison_review.py','comparison_review.static-draft01.py','own_normalization.py','new_calibrations.py','config.json','PROCEDURE.md','collect_source_runtime_freeze.py']}
for k in ['candidate','manifest','parents','legacy','input','first_predictions','first_judgments','first_facts','own_solver','own_calibration','own_hand_fixtures','canonical_schema']:
 paths[cfg[k]]='CANONICAL_GOLD_RAW_NOT_COMPARED' if k=='candidate' else ('OWN_SOURCE_OR_FIRST_EVIDENCE' if k.startswith(('first_','own_')) else 'AUTHORIZED_PUBLIC_OR_SUPPLIED_RAW_INPUT')
for n in cfg['public_sources']:paths[n]='OPERATIVE_PUBLIC_SOURCE'
paths['research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R2_ROOT_FREEZE.json']='PUBLIC_ROOT_FREEZE'
paths['factory/common.py']='PUBLIC_COMMON_HASH_SOURCE_READ_NOT_IMPORTED'
paths['manifests/option-resumption-pilot-v1_GENERATION.json']='GENERATION_CUSTODY_ASSERTIONS_ONLY'
old='reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/'
for n in ['FIRST.json','INPUT-FIRST-FINAL.json','PREEXECUTION.json','EXECUTION.json','input_review_runner.py','input-only-relations.json','compact-summary.json','supplement-measurements.json','SEMANTIC-INSPECTION.md']:
 paths[old+n]='OWN_IMMUTABLE_FIRST_SOURCE_EVIDENCE'
for n in cfg['opaque_provenance_paths']:paths[n]='OPAQUE_NOT_SEMANTICALLY_INSPECTED_NOT_IMPORTED'
for n in manifest['source_bindings']:
 if n not in paths:raise ValueError('asserted source binding not explicitly authorized/configured: '+n)
# Only whitelisted asserted producer/reference/config/fixture paths; no
# formal/root-mutant/other-reviewer payloads are followed or copied.
failures=[];bindings=[]
expected={cfg[k]:cfg.get(k+'_sha256') for k in ['candidate','manifest','parents','legacy','input','first_predictions','first_judgments']}
expected.update(manifest['source_bindings'])
expected.update({x['path']:x['sha256'] for x in generation['core_artifacts']})
expected[old+'FIRST.json']='728e01c8b0553f7230caef6af4bc6730afa8f50d49a3f0014d8ed2ecf0d9e42c'
expected[old+'INPUT-FIRST-FINAL.json']='154c2c2ca070da19e443ce195f7d2669d60e24276dd50aa9393eee45ffc49f11'
for name,scope in sorted(paths.items()):
 p=ROOT/name
 if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe asserted path')
 walk=ROOT
 for part in Path(name).parts:
  walk/=part
  if walk.is_symlink():raise ValueError('symlink asserted path '+name)
 try:b=p.read_bytes()
 except OSError as exc:failures.append({'path':name,'error':str(exc),'status':'MISSING_REQUIRED_BYTES'});continue
 h=sha(b);ap=archive/(h+'.raw')
 if not ap.exists():ap.write_bytes(b)
 assert ap.read_bytes()==b
 row={'path':name,'absolute_path':str(p.resolve()),'sha256':h,'bytes':len(b),'archive_path':str(ap.relative_to(HERE)),'scope':scope}
 if expected.get(name) is not None:
  row['asserted_sha256']=expected[name];row['assertion_matches']=h==expected[name]
  if not row['assertion_matches']:failures.append({'path':name,'actual':h,'asserted':expected[name],'status':'SOURCE_BYTE_BINDING_MISMATCH'})
 bindings.append(row)
# Enumerate the actual stdlib collector import closure only, plus executable;
# collect source and on-disk cache because -B disables cache writes, not reads.
# No source from site-packages or unrelated application modules is admitted.
module_rows=[];runtime_paths={Path(sys.executable).resolve()}
for name,module in sorted(sys.modules.items()):
 filename=getattr(module,'__file__',None)
 if not filename:
  module_rows.append({'module':name,'kind':'BUILTIN_OR_FROZEN','origin':getattr(getattr(module,'__spec__',None),'origin',None)});continue
 p=Path(filename).resolve()
 if p==Path(__file__).resolve():
  module_rows.append({'module':name,'kind':'STDLIB_ONLY_CUSTODY_BOOTSTRAP','source_sha256':source_hash});continue
 if '/site-packages/' in str(p):raise ValueError('unexpected ambient site-package import '+name)
 if not str(p).startswith('/opt/anaconda3/lib/python3.12/') and not str(p).startswith('/opt/anaconda3/bin/'):
  raise ValueError('unexpected module outside declared runtime '+name+':'+str(p))
 runtime_paths.add(p);cache=getattr(module,'__cached__',None)
 if cache and Path(cache).is_file():runtime_paths.add(Path(cache).resolve())
 module_rows.append({'module':name,'kind':'FILE_BACKED_STDLIB','absolute_path':str(p),'cached_path':cache})
# Record and archive local dynamic libraries linked by interpreter/extensions.
# Apple system shared-cache/OS libraries remain a disclosed system dependency.
links=[];pending=list(runtime_paths);done=set()
while pending:
 p=pending.pop()
 if p in done:continue
 done.add(p)
 if p.suffix not in ('.so','.dylib') and p!=Path(sys.executable).resolve():continue
 result=subprocess.run(['/usr/bin/otool','-L',str(p)],capture_output=True,text=True,check=False)
 links.append({'binary':str(p),'exit_code':result.returncode,'output':result.stdout,'stderr':result.stderr})
 if result.returncode:failures.append({'path':str(p),'status':'LINK_DEPENDENCY_INSPECTION_FAILED'});continue
 for line in result.stdout.splitlines()[1:]:
  raw=line.strip().split(' (',1)[0];q=None
  if raw.startswith('/opt/anaconda3/'):q=Path(raw)
  elif raw.startswith('@rpath/'):
   found=Path('/opt/anaconda3/lib')/raw[len('@rpath/'):]
   if found.exists():q=found
  elif raw.startswith('@loader_path/'):
   found=p.parent/raw[len('@loader_path/'):]
   if found.exists():q=found
  if q is not None:
   q=q.resolve()
   if q.is_file() and q not in runtime_paths:runtime_paths.add(q);pending.append(q)
runtime_bindings=[]
for p in sorted(runtime_paths):
 b=p.read_bytes();h=sha(b);a=closure_archive/(h+'.runtime.raw')
 if not a.exists():a.write_bytes(b)
 assert a.read_bytes()==b
 runtime_bindings.append({'absolute_path':str(p),'sha256':h,'bytes':len(b),'archive_path':str(a.relative_to(HERE))})
# Runtime module discovery happens again after linker inspection to ensure no
# new file-backed import escaped the recorded closure.
for name,module in sys.modules.items():
 filename=getattr(module,'__file__',None)
 if filename and Path(filename).resolve()!=Path(__file__).resolve():
  if Path(filename).resolve() not in runtime_paths:raise AssertionError('late stdlib import escaped closure '+name)
freeze={'actual_author_execution_id':'/root/expert_reviewer','status':'STATIC_PREPARATION_FROZEN_NOT_EXECUTED' if not failures else 'STATIC_PREPARATION_WITH_BINDING_FAILURES_NO_EXECUTION',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'binding_failures':failures,
 'runtime':{'python_executable':str(Path(sys.executable).resolve()),'python_executable_sha256':sha(Path(sys.executable).resolve().read_bytes()),'sys_version':sys.version,'sys_platform':sys.platform,'uname':list(os.uname()),'file_bindings':runtime_bindings,'module_closure':module_rows,'linked_dependencies':links,'stdlib_bootstrap_boundary':'Declared stdlib imports and metadata/hash/archive custody execute to discover closure; NO new reviewer/solver/normalizer/calibration import or execution occurs. This is not a pre-bootstrap OS-runtime freeze.','system_library_limitation':'Apple shared-cache/OS libraries are recorded by linker output, not extracted or byte-archived. File-backed stdlib/local dependencies and executable are archived.'},
 'actual_freeze_command':'/opt/anaconda3/bin/python3 -I -B -S reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt01/collect_source_runtime_freeze.py',
 'future_planned_comparison_command':'/opt/anaconda3/bin/python3 -I -B -S reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt01/comparison_review.py compare',
 'no_comparison_tests_or_predictions_executed':True,'no_dataset_verdicts_written':True,'original_first_unchanged':True,
 'opaque_sources_not_parsed_as_code_or_imported':True,'disclosures':cfg['disclosures'],
 'read_discovery_errors_preserved':['schemas/canonical.schema.json absent; operative path is schemas/canonical/v1.schema.json','schemas/common absent; authorized helper is factory/common.py','jq unavailable; only shell metadata excerpts used for candidate/parent preview'],
 'private_filename_inventory_only':'One overbroad rg --files enumeration exposed other-review filenames without reading their bodies; no private verdict/prediction/code content consulted.'}
p=HERE/'SOURCE-PREPARATION-FREEZE.json'
with p.open('x') as f:json.dump(freeze,f,sort_keys=True,indent=2);f.write('\n')
final={'actual_author_execution_id':'/root/expert_reviewer','status':freeze['status'],'scope':'SOURCE_PREPARATION_ONLY_AWAITING_ROOT_READBACK_AND_EXECUTION_GRANT','dataset_approval':False,'comparison_executed':False,'calibration_executed':False,'future_final_receipts_written':False,
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze':{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size},
 'source_bindings':[x for x in bindings if x['scope']=='OWN_STATIC_SOURCE_NOT_IMPORTED'],'runtime_file_count':len(runtime_bindings),'raw_binding_count':len(bindings),'opaque_binding_count':sum(x['scope']=='OPAQUE_NOT_SEMANTICALLY_INSPECTED_NOT_IMPORTED' for x in bindings),'binding_failures':failures,'original_first_receipt_sha256':expected[old+'FIRST.json'],'original_input_final_sha256':expected[old+'INPUT-FIRST-FINAL.json'],'first_predictions_sha256':cfg['first_predictions_sha256'],'first_judgments_sha256':cfg['first_judgments_sha256'],'designated_future_final_path':'reviews/expert/option-resumption-pilot-v1.jsonl','all_raw_archives_verified_byte_equal':True,'stop_after_this_source_freeze':True}
q=HERE/'SOURCE-PREPARATION-FINAL.json'
with q.open('x') as f:json.dump(final,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'final':str(q.relative_to(ROOT)),'final_sha256':sha(q.read_bytes()),'freeze_sha256':sha(p.read_bytes()),'own_source_sha256':sha((HERE/'comparison_review.py').read_bytes()),'runtime_files':len(runtime_bindings),'raw_bindings':len(bindings),'binding_failures':failures,'no_reviewer_import_or_execution':True}))
