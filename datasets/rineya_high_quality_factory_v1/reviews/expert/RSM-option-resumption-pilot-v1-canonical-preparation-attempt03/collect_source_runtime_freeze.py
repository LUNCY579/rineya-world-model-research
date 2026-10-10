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
custody_parser=argparse.ArgumentParser(description='Expert stdlib-only source/runtime custody preparation')
custody_parser.add_argument('--expected-collector-source-sha256',required=True)
custody_parser.add_argument('--expected-config-sha256',required=True)
custody_args=custody_parser.parse_args()
config_bytes=(HERE/'config.json').read_bytes()
for value in [custody_args.expected_collector_source_sha256,custody_args.expected_config_sha256]:
 if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('external collector/config exact SHA256 required')
if source_hash!=custody_args.expected_collector_source_sha256 or sha(config_bytes)!=custody_args.expected_config_sha256:
 raise ValueError('external collector/config SHA mismatch')
if (HERE/'COLLECTOR-SOURCE-PREEXECUTION.raw').read_bytes()!=source_bytes or (HERE/'CONFIG-PREEXECUTION.raw').read_bytes()!=config_bytes:
 raise ValueError('exact external preexecution collector/config raw archive mismatch')
cfg=json.loads(config_bytes)

# Actual CLI shape copied from the exact own comparator, not imported from it.
# Benign placeholders exercise startup paths only; no compare/emitter invocation.
runtime_parser=argparse.ArgumentParser();runtime_parser.add_argument('phase',choices=['compare','receipts']);runtime_parser.add_argument('--quality-map')
runtime_parser.add_argument('--expected-config-sha256',required=True);runtime_parser.add_argument('--expected-source-freeze-sha256',required=True)
runtime_parser.add_argument('--expected-quality-map-sha256');runtime_parser.add_argument('--expected-quality-freeze-sha256')
benign_compare=runtime_parser.parse_args(['compare','--expected-config-sha256',custody_args.expected_config_sha256,'--expected-source-freeze-sha256','0'*64])
benign_receipts=runtime_parser.parse_args(['receipts','--quality-map','NEVER_OPENED_BENIGN_RUNTIME_PLACEHOLDER.json','--expected-config-sha256',custody_args.expected_config_sha256,'--expected-source-freeze-sha256','0'*64,'--expected-quality-map-sha256','0'*64,'--expected-quality-freeze-sha256','0'*64])
runtime_help=runtime_parser.format_help()
import shutil
import tempfile
import bz2
import lzma
try:raise RuntimeError('benign stdlib exception-format startup materialization; no semantic test')
except RuntimeError:startup_exception_text=traceback.format_exc()
startup={'scope':'BENIGN_STDLIB_STARTUP_PATH_MATERIALIZATION_NOT_SEMANTIC_CALIBRATION_OR_COMPARISON',
 'actual_comparator_cli_shape_constructed':True,'actual_compare_args_parsed':vars(benign_compare),'actual_receipts_args_parsed':vars(benign_receipts),
 'placeholder_hashes_are_not_target_expectations_or_execution_authority':True,'help_text_sha256':sha(runtime_help.encode()),
 'declared_lazy_stdlib_imports':['shutil','tempfile','bz2','lzma'],'exception_format_path_materialized':True,
 'exception_text_sha256':sha(startup_exception_text.encode()),'own_semantic_module_imported':False,
 'external_source_SHA':custody_args.expected_collector_source_sha256,'external_config_SHA':custody_args.expected_config_sha256}
# The following metadata objects are permitted custody/source assertions; no
# candidate target parsing, producer diagnostic result parsing or source import.
generation_path=ROOT/'manifests/option-resumption-pilot-v1_GENERATION.json'
generation=json.loads(generation_path.read_text())
manifest_path=ROOT/cfg['manifest'];manifest=json.loads(manifest_path.read_text())
archive=HERE/'raw-archives';archive.mkdir()
closure_archive=HERE/'runtime-archives';closure_archive.mkdir()
paths={str((HERE/n).relative_to(ROOT)):'OWN_STATIC_SOURCE_NOT_IMPORTED' for n in ['comparison_review.py','own_normalization.py','new_calibrations.py','config.json','PROCEDURE.md','collect_source_runtime_freeze.py','custody_guard.source-fragment.txt','prepare_attempt03.py','STATIC-SEMANTIC-PRESERVATION.json','PROTECTED-ORIGINAL-INVENTORY.json','COLLECTOR-SOURCE-PREEXECUTION.raw','CONFIG-PREEXECUTION.raw','CUSTODY-COLLECTOR-PREEXECUTION.json']}
for k in ['candidate','manifest','parents','legacy','input','first_predictions','first_judgments','first_facts','own_solver','own_calibration','own_hand_fixtures','canonical_schema']:
 paths[cfg[k]]='CANONICAL_GOLD_RAW_NOT_COMPARED' if k=='candidate' else ('OWN_SOURCE_OR_FIRST_EVIDENCE' if k.startswith(('first_','own_')) else 'AUTHORIZED_PUBLIC_OR_SUPPLIED_RAW_INPUT')
for n in cfg['public_sources']:paths[n]='OPERATIVE_PUBLIC_SOURCE'
paths['research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R2_ROOT_FREEZE.json']='PUBLIC_ROOT_FREEZE'
paths['factory/common.py']='PUBLIC_COMMON_HASH_SOURCE_READ_NOT_IMPORTED'
paths['manifests/option-resumption-pilot-v1_GENERATION.json']='GENERATION_CUSTODY_ASSERTIONS_ONLY'
old='reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/'
for n in ['FIRST.json','INPUT-FIRST-FINAL.json','PREEXECUTION.json','EXECUTION.json','input_review_runner.py','input-only-relations.json','compact-summary.json','supplement-measurements.json','SEMANTIC-INSPECTION.md']:
 paths[old+n]='OWN_IMMUTABLE_FIRST_SOURCE_EVIDENCE'
prior='reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt01/'
for n in ['comparison_review.py','comparison_review.static-draft01.py','own_normalization.py','new_calibrations.py','config.json','PROCEDURE.md','collect_source_runtime_freeze.py','SOURCE-PREPARATION-FREEZE.json','SOURCE-PREPARATION-FINAL.json']:
 paths[prior+n]='OWN_PROTECTED_ATTEMPT01_BYTES_NOT_IMPORTED'
failed='reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt02/'
for n in ['comparison_review.py','config.json','collect_source_runtime_freeze.py','SOURCE-PREPARATION-FREEZE.json','SOURCE-PREPARATION-FINAL.json','ACTUAL-COMPARE-STDOUT.txt','ACTUAL-COMPARE-STDERR.txt','FAILURE-compare.txt','ACTUAL-COMPARE-FAILED-EXECUTION.json','ACTUAL-COMPARE-FAILED-FIRST.json']:
 paths[failed+n]='OWN_PROTECTED_ATTEMPT02_AND_ACTUAL_STARTUP_FAILURE_NOT_IMPORTED'
for own in [ROOT/cfg['own_solver'],ROOT/cfg['own_calibration'],HERE/'comparison_review.py',HERE/'own_normalization.py',HERE/'new_calibrations.py']:
 cache=Path(importlib.util.cache_from_source(str(own)))
 if cache.is_file():paths[str(cache.relative_to(ROOT))]='OWN_EXISTING_CACHE_NOT_IMPORTED'
for n in cfg['opaque_provenance_paths']:paths[n]='OPAQUE_NOT_SEMANTICALLY_INSPECTED_NOT_IMPORTED'
for n in manifest['source_bindings']:
 if n not in paths:raise ValueError('asserted source binding not explicitly authorized/configured: '+n)
# Only whitelisted asserted producer/reference/config/fixture paths; no
# formal/root-mutant/other-reviewer payloads are followed or copied.
def store_exact_archive(destination,raw):
 if destination.exists():
  if destination.read_bytes()!=raw:raise ValueError('existing archive does not equal exact current bytes')
  return
 for earlier in ['RSM-option-resumption-pilot-v1-canonical-preparation-attempt02','RSM-option-resumption-pilot-v1-canonical-preparation-attempt01']:
  possible=HERE.parent/earlier/destination.parent.name/destination.name
  if possible.is_file() and not possible.is_symlink() and possible.read_bytes()==raw:
   os.link(possible,destination)
   if destination.read_bytes()!=raw:raise ValueError('hardlinked exact archive changed during reuse')
   return
 with destination.open('xb') as f:f.write(raw)
 if destination.read_bytes()!=raw:raise ValueError('new exact raw archive write differs')
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
 store_exact_archive(ap,b)
 assert ap.read_bytes()==b
 row={'path':name,'absolute_path':str(p.resolve()),'sha256':h,'bytes':len(b),'archive_path':str(ap.relative_to(HERE)),'scope':scope}
 if expected.get(name) is not None:
  row['asserted_sha256']=expected[name];row['assertion_matches']=h==expected[name]
  if not row['assertion_matches']:failures.append({'path':name,'actual':h,'asserted':expected[name],'status':'SOURCE_BYTE_BINDING_MISMATCH'})
 bindings.append(row)
# Enumerate the actual stdlib collector import closure only, plus executable;
# collect source and on-disk cache because -B disables cache writes, not reads.
# No source from site-packages or unrelated application modules is admitted.
def discover_modules():
 rows=[];found={Path(sys.executable).resolve()}
 for name,module in sorted(list(sys.modules.items())):
  if module is None:raise ValueError('non-auditable module sentinel '+name)
  filename=getattr(module,'__file__',None);origin=getattr(getattr(module,'__spec__',None),'origin',None)
  if filename is None:
   if origin not in ('built-in','frozen'):raise ValueError('unrecognized non-file module origin '+name+':'+str(origin))
   rows.append({'module':name,'kind':'BUILTIN' if origin=='built-in' else 'FROZEN_WITHOUT_FILE','origin':origin,'file_backed':False});continue
  if type(filename) is not str:raise ValueError('untyped module filename '+name)
  p=Path(filename).resolve()
  if p==Path(__file__).resolve():
   rows.append({'module':name,'kind':'STDLIB_ONLY_CUSTODY_BOOTSTRAP','origin':origin,'source_sha256':source_hash,'absolute_path':str(p),'file_backed':True});continue
  if '/site-packages/' in str(p):raise ValueError('unexpected ambient site-package import '+name)
  if not str(p).startswith('/opt/anaconda3/lib/python3.12/') and not str(p).startswith('/opt/anaconda3/bin/'):
   raise ValueError('unexpected module outside declared runtime '+name+':'+str(p))
  found.add(p);cache=getattr(module,'__cached__',None);cache_path=None
  if cache is not None:
   if type(cache) is not str:raise ValueError('untyped cache filename '+name)
   cache_path=str(Path(cache).resolve())
   if Path(cache_path).is_file():found.add(Path(cache_path))
  rows.append({'module':name,'kind':'FROZEN_WITH_FILE' if origin=='frozen' else 'FILE_BACKED_STDLIB',
   'origin':origin,'absolute_path':str(p),'cached_path':cache_path,'cached_file_exists':cache_path is not None and Path(cache_path).is_file(),'file_backed':True})
 return rows,found
module_rows,runtime_paths=discover_modules()
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
 store_exact_archive(a,b)
 assert a.read_bytes()==b
 runtime_bindings.append({'absolute_path':str(p),'sha256':h,'bytes':len(b),'archive_path':str(a.relative_to(HERE))})
# Re-enumerate actual origins/paths after linker inspection, preserving exact
# module declarations instead of a misleading BUILTIN_OR_FROZEN catch-all.
module_rows,late_paths=discover_modules()
if not late_paths<=runtime_paths:raise ValueError('late runtime file import escaped archived closure')
for item in bindings+runtime_bindings:
 current=Path(item['absolute_path']).read_bytes();raw=(HERE/item['archive_path']).read_bytes()
 if type(item['bytes']) is not int or type(item['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',item['sha256']) is None:
  raise ValueError('untyped file descriptor in custody collection')
 if len(current)!=item['bytes'] or sha(current)!=item['sha256'] or len(raw)!=item['bytes'] or sha(raw)!=item['sha256'] or current!=raw:
  raise ValueError('post-collection current/raw SHA/size/bytes mismatch '+item['absolute_path'])
protected=json.loads((HERE/'PROTECTED-ORIGINAL-INVENTORY.json').read_text())
for item in protected['files']:
 raw=Path(item['absolute_path']).read_bytes()
 if len(raw)!=item['bytes'] or sha(raw)!=item['sha256']:raise ValueError('protected original changed '+item['path'])
freeze={'actual_author_execution_id':'/root/expert_reviewer','status':'STATIC_PREPARATION_FROZEN_NOT_EXECUTED' if not failures else 'STATIC_PREPARATION_WITH_BINDING_FAILURES_NO_EXECUTION',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'binding_failures':failures,
 'startup_materialization':startup,'runtime':{'python_executable':str(Path(sys.executable).resolve()),'python_executable_sha256':sha(Path(sys.executable).resolve().read_bytes()),'python_executable_bytes':Path(sys.executable).resolve().stat().st_size,'sys_version':sys.version,'sys_platform':sys.platform,'uname':list(os.uname()),'file_bindings':runtime_bindings,'module_closure':module_rows,'linked_dependencies':links,'stdlib_bootstrap_boundary':'Declared stdlib imports and metadata/hash/archive custody execute to discover closure; NO new reviewer/solver/normalizer/calibration import or execution occurs. This is not a pre-bootstrap OS-runtime freeze.','system_library_limitation':'Apple shared-cache/OS libraries are recorded by linker output, not extracted or byte-archived. File-backed stdlib/local dependencies and executable are archived.'},
 'actual_freeze_command':'/opt/anaconda3/bin/python3 -I -B -S '+str(Path(__file__).resolve().relative_to(ROOT))+' '+' '.join(sys.argv[1:]),
 'future_planned_comparison_command':'/opt/anaconda3/bin/python3 -I -B -S reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt03/comparison_review.py compare --expected-config-sha256 <EXTERNALLY_SUPPLIED_CONFIG_SHA256> --expected-source-freeze-sha256 <EXTERNALLY_SUPPLIED_SOURCE_FREEZE_SHA256>',
 'no_comparison_tests_or_predictions_executed':True,'no_dataset_verdicts_written':True,'original_first_unchanged':True,'protected_attempt01_and_input_first_file_count':len(protected['files']),'protected_all_exact_unchanged':True,
 'opaque_sources_not_parsed_as_code_or_imported':True,'disclosures':cfg['disclosures'],
 'read_discovery_errors_preserved':['schemas/canonical.schema.json absent; operative path is schemas/canonical/v1.schema.json','schemas/common absent; authorized helper is factory/common.py','jq unavailable; only shell metadata excerpts used for candidate/parent preview'],
 'private_filename_inventory_only':'One overbroad rg --files enumeration exposed other-review filenames without reading their bodies; no private verdict/prediction/code content consulted.'}
p=HERE/'SOURCE-PREPARATION-FREEZE.json'
with p.open('x') as f:json.dump(freeze,f,sort_keys=True,indent=2);f.write('\n')
final={'actual_author_execution_id':'/root/expert_reviewer','status':freeze['status'],'scope':'ATTEMPT03_SOURCE_ONLY_REAL_CLI_RUNTIME_CLOSURE_REPAIR_AWAITING_ROOT_READBACK_AND_EXECUTION_GRANT','dataset_approval':False,'comparison_executed':False,'calibration_executed':False,'future_final_receipts_written':False,
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze':{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size},
 'source_bindings':[x for x in bindings if x['scope']=='OWN_STATIC_SOURCE_NOT_IMPORTED'],'runtime_file_count':len(runtime_bindings),'raw_binding_count':len(bindings),'opaque_binding_count':sum(x['scope']=='OPAQUE_NOT_SEMANTICALLY_INSPECTED_NOT_IMPORTED' for x in bindings),'binding_failures':failures,'original_first_receipt_sha256':expected[old+'FIRST.json'],'original_input_final_sha256':expected[old+'INPUT-FIRST-FINAL.json'],'first_predictions_sha256':cfg['first_predictions_sha256'],'first_judgments_sha256':cfg['first_judgments_sha256'],'designated_future_final_path':'reviews/expert/option-resumption-pilot-v1.jsonl','all_raw_archives_verified_byte_equal':True,'typed_SHA_size_runtime_raw_archives_verified':True,'protected_original_files_verified_unchanged':len(protected['files']),'guard_operational_tests_executed':False,'semantic_preservation':json.loads((HERE/'STATIC-SEMANTIC-PRESERVATION.json').read_text()),'startup_materialization':startup,'exact_archive_hardlink_reuse_permitted':True,'attempt02_failure_recorded_preserved_no_retry':True,'stop_after_this_source_freeze':True}
q=HERE/'SOURCE-PREPARATION-FINAL.json'
with q.open('x') as f:json.dump(final,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'final':str(q.relative_to(ROOT)),'final_sha256':sha(q.read_bytes()),'freeze_sha256':sha(p.read_bytes()),'own_source_sha256':sha((HERE/'comparison_review.py').read_bytes()),'runtime_files':len(runtime_bindings),'raw_bindings':len(bindings),'binding_failures':failures,'no_reviewer_import_or_execution':True}))
