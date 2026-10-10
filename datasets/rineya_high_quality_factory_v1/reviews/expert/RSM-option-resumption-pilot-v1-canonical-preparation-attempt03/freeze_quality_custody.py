"""Own stdlib-only custody collector, never target/quality inference or emitter.
External source SHA/raw barrier and the already frozen resolved runtime precede
all reading of authored material. Preserve exclusive outputs and failures.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ACTOR='/root/expert_reviewer'
SF='91e91429d941a58790db4d9e716e85f332d66b29a9887547efa59530a8df6936'
CFG='a3dbb827bdb386c4e54660c1d14b5a8d0173dfdc3b4fcdf2a2959841a9621d94'
def sha(b):return hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['pre','seal'])
parser.add_argument('--expected-source-sha256',required=True)
args=parser.parse_args()
started=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter()

def desc(p,archive=None):
 p=Path(p).resolve();b=p.read_bytes()
 x={'path':str(p.relative_to(ROOT)),'absolute_path':str(p),'sha256':sha(b),'bytes':len(b)}
 if archive is not None:x['archive_path']=str(archive.relative_to(HERE))
 return x
def put(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,ensure_ascii=False);f.write('\n')
 return desc(p)
def barrier():
 own=Path(__file__).resolve();b=own.read_bytes()
 if sha(b)!=args.expected_source_sha256 or b!=(HERE/'freeze_quality_custody.py.before-execution.raw').read_bytes():raise ValueError('external own-source/raw barrier')
 sfbytes=(HERE/'SOURCE-PREPARATION-FREEZE.json').read_bytes()
 if sha(sfbytes)!=SF or sha((HERE/'config.json').read_bytes())!=CFG:raise ValueError('comparison source/config freeze')
 sf=json.loads(sfbytes)
 for item in sf['bindings']+sf['runtime']['file_bindings']:
  b=Path(item['absolute_path']).read_bytes();a=(HERE/item['archive_path']).read_bytes()
  if type(item['bytes']) is not int or len(b)!=item['bytes'] or sha(b)!=item['sha256'] or b!=a:raise ValueError('existing current/raw descriptor changed')
 runtime=sf['runtime'];exe=Path(sys.executable).resolve()
 if str(exe)!=runtime['python_executable'] or exe.stat().st_size!=runtime['python_executable_bytes'] or sha(exe.read_bytes())!=runtime['python_executable_sha256']:raise ValueError('resolved Python mismatch')
 declarations={x['module']:x for x in runtime['module_closure']};paths={x['absolute_path'] for x in runtime['file_bindings']};mods=[]
 for name,m in sorted(list(sys.modules.items())):
  origin=getattr(getattr(m,'__spec__',None),'origin',None);filename=getattr(m,'__file__',None)
  if filename is None:
   if origin not in ('built-in','frozen') or name not in declarations or declarations[name]['origin']!=origin:raise ValueError('unbound builtin/frozen '+name)
   mods.append({'module':name,'origin':origin,'file_backed':False});continue
  p=str(Path(filename).resolve())
  if name=='__main__':
   if p!=str(own):raise ValueError('collector main changed')
  elif name not in declarations or declarations[name]['absolute_path']!=p or declarations[name]['origin']!=origin or p not in paths:raise ValueError('unbound runtime '+name)
  cache=getattr(m,'__cached__',None)
  if cache and Path(cache).exists() and str(Path(cache).resolve()) not in paths:raise ValueError('unbound cache')
  mods.append({'module':name,'origin':origin,'absolute_path':p,'file_backed':True})
 return sf,{'existing_primary':len(sf['bindings']),'runtime_count':len(paths),'current_raw_equality':True,'resolved_python':str(exe),'modules':mods}

def bind_all(paths,sf):
 archive_dir=HERE/'quality-author-raw';archive_dir.mkdir(exist_ok=True)
 known={x['absolute_path']:HERE/x['archive_path'] for x in sf['bindings']+sf['runtime']['file_bindings']}
 original_scopes={x['absolute_path']:x.get('scope','EXISTING_OWN_OR_PUBLIC_BOUND_INPUT') for x in sf['bindings']}
 out=[];seen=set()
 for p in paths:
  p=Path(p).resolve()
  if str(p) in seen:continue
  seen.add(str(p));b=p.read_bytes();ap=archive_dir/(sha(b)+'.raw')
  if not ap.exists():
   if str(p) in known and known[str(p)].read_bytes()==b:os.link(known[str(p)],ap)
   else:
    with ap.open('xb') as f:f.write(b)
  if ap.read_bytes()!=b:raise ValueError('exact quality archive equality')
  item=desc(p,ap)
  item['scope']=original_scopes.get(str(p),'OWN_NEW_ACTUAL_COMPARISON_OR_AUTHORED_QUALITY_EVIDENCE')
  out.append(item)
 return out

try:
 sf,before=barrier()
 comparison_inputs=[Path(x['absolute_path']) for x in sf['bindings']]
 named=['config.json','SOURCE-PREPARATION-FREEZE.json','SOURCE-PREPARATION-FINAL.json','CUSTODY-COLLECTOR-EXECUTION.json',
  'COMPARISON-EXECUTION.json','COMPARISON-FACTUAL-FIRST.json','ACTUAL-COMPARE-STDOUT.txt','ACTUAL-COMPARE-STDERR.txt',
  'canonical-comparison.first.jsonl','canonical-check-ledger.json','actual-control-evidence.json','new-calibration-results.json','own-novelty-evidence.json',
  'inspect_metadata_v2.py','inspect_metadata_v2.py.before-execution.raw','INSPECTION-V2-PREEXECUTION.json','inspection-panels.v2.json',
  'metadata-preview-v1.failed-source.txt','metadata-preview-v1.failure.json',
  'MANUAL-QUALITY-AUTHORSHIP.json','CANONICAL-QUALITY-INSPECTION.md','quality-map.config.json','serialize_authored_quality.py',
  'freeze_quality_custody.py','freeze_quality_custody.py.before-execution.raw','QUALITY-CUSTODY-COLLECTOR-BARRIER.json']
 paths=comparison_inputs+[HERE/n for n in named]
 if args.phase=='pre':
  binds=bind_all(paths,sf)
  _,after=barrier()
  freeze=put(HERE/'QUALITY-AUTHORSHIP-PREEXECUTION.json',{'actual_author_execution_id':ACTOR,
   'scope':'SOURCE_CONFIG_ACTUAL_AGENT_MATERIAL_NARRATIVE_AND_CURRENT_COMPARISON_BEFORE_QUALITY_SERIALIZATION',
   'source_version':'expert-rsm-quality-authorship-1.0.0','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'bindings':binds,'existing_runtime_source_freeze_sha256':SF,'resolved_runtime_raw_verified':True,
   'authored_decisions_are_not_machine_PASS_inferred':True,'expected_execution':'/opt/anaconda3/bin/python3 -I -B -S '+str(HERE/'serialize_authored_quality.py')+' --expected-preexecution-sha256 <EXTERNAL_ACTUAL_FREEZE_SHA>',
   'before':before,'after':after,'no_emitter_or_publication':True})
  result=put(HERE/'QUALITY-CUSTODY-COLLECTOR-PRE-EXECUTION.json',{'actual_author_execution_id':ACTOR,'started_utc':started,
   'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-timer,'argv':sys.argv,'exit_code':0,
   'own_source_sha256':args.expected_source_sha256,'output':freeze,'no_semantic_helper_import_target_computation_or_emitter':True})
 else:
  pre=json.loads((HERE/'QUALITY-AUTHORSHIP-PREEXECUTION.json').read_text())
  for x in pre['bindings']:
   b=Path(x['absolute_path']).read_bytes()
   if len(b)!=x['bytes'] or sha(b)!=x['sha256'] or b!=(HERE/x['archive_path']).read_bytes():raise ValueError('quality preexecution input changed')
  quality=json.loads((HERE/'quality-map.first.json').read_text());first=json.loads((HERE/'QUALITY-FIRST.json').read_text())
  if quality['actual_author_execution_id']!=ACTOR or len(quality['per_sample'])!=216 or not first['all216_complete'] or not first['no_emitter_run']:raise ValueError('own quality completion incomplete')
  extra=['QUALITY-AUTHORSHIP-PREEXECUTION.json','QUALITY-CUSTODY-COLLECTOR-PRE-EXECUTION.json','quality-map.first.json','QUALITY-AUTHORSHIP-EXECUTION.json','QUALITY-FIRST.json','ACTUAL-QUALITY-STDOUT.txt','ACTUAL-QUALITY-STDERR.txt']
  binds=bind_all(paths+[HERE/n for n in extra],sf);_,after=barrier()
  freeze=put(HERE/'QUALITY-MAP-FREEZE.json',{'actual_author_execution_id':ACTOR,'scope':'ACTUAL_AGENT_QUALITY_MAP_AND_CURRENT_COMPARISON_CUSTODY',
   'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_config_sha256':CFG,'expected_source_freeze_sha256':SF,
   'publication_input_bindings':binds,'own_quality_first_sha256':sha((HERE/'QUALITY-FIRST.json').read_bytes()),
   'runtime_equality_and_membership_before':before,'runtime_equality_and_membership_after':after,'emitter_run':False,'publication_approval':False})
  result=put(HERE/'QUALITY-CUSTODY-COLLECTOR-SEAL-EXECUTION.json',{'actual_author_execution_id':ACTOR,'started_utc':started,
   'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-timer,'argv':sys.argv,'exit_code':0,
   'own_source_sha256':args.expected_source_sha256,'output':freeze,'no_semantic_helper_import_target_computation_or_emitter':True})
 print(json.dumps({'freeze':freeze,'execution':result,'stop_before_emitter':True}))
except BaseException:
 with (HERE/('FAILURE-QUALITY-CUSTODY-'+args.phase+'.txt')).open('x') as f:f.write(traceback.format_exc())
 raise
