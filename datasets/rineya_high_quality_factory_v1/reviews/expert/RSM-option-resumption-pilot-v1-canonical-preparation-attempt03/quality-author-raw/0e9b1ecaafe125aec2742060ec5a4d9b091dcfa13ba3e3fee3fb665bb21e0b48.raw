"""Serialize actual agent's already authored/frozen quality assessment only.
No solver/oracle import, target recomputation, score inference from PASS, emitter,
designated receipts, production edits or publication. Preserve failures.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];ACTOR='/root/expert_reviewer'
def sha(b):return hashlib.sha256(b).hexdigest()
def put_json(p,value):
 with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=False);f.write('\n')
 return {'path':str(p.relative_to(ROOT)),'absolute_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
parser=argparse.ArgumentParser();parser.add_argument('--expected-preexecution-sha256',required=True);args=parser.parse_args()
start=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter()

def current_guard():
 raw=(HERE/'QUALITY-AUTHORSHIP-PREEXECUTION.json').read_bytes()
 if type(args.expected_preexecution_sha256) is not str or re.fullmatch('[0-9a-f]{64}',args.expected_preexecution_sha256) is None or sha(raw)!=args.expected_preexecution_sha256:
  raise ValueError('external quality-authorship preexecution SHA mismatch')
 pre=json.loads(raw);cfg=json.loads((HERE/'quality-map.config.json').read_text())
 if pre['actual_author_execution_id']!=ACTOR or cfg['actual_author_execution_id']!=ACTOR:raise ValueError('actual quality actor mismatch')
 paths={}
 for item in pre['bindings']:
  if type(item['bytes']) is not int or type(item['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',item['sha256']) is None:raise ValueError('untyped quality descriptor')
  p=Path(item['absolute_path']);ap=HERE/item['archive_path'];b=p.read_bytes();a=ap.read_bytes()
  if str(p.resolve())!=item['absolute_path'] or len(b)!=item['bytes'] or sha(b)!=item['sha256'] or len(a)!=item['bytes'] or sha(a)!=item['sha256'] or b!=a:raise ValueError('quality current/raw SHA/length/equality mismatch')
  paths[str(p.resolve())]=item
 if str(Path(__file__).resolve()) not in paths or str((HERE/'quality-map.config.json').resolve()) not in paths:raise ValueError('own serializer/config omitted from freeze')
 source_freeze=json.loads((HERE/'SOURCE-PREPARATION-FREEZE.json').read_text())
 if sha((HERE/'SOURCE-PREPARATION-FREEZE.json').read_bytes())!=cfg['comparison_source_freeze_sha256']:raise ValueError('comparison source freeze changed')
 for item in source_freeze['bindings']+source_freeze['runtime']['file_bindings']:
  b=Path(item['absolute_path']).read_bytes();a=(HERE/item['archive_path']).read_bytes()
  if type(item['bytes']) is not int or len(b)!=item['bytes'] or sha(b)!=item['sha256'] or len(a)!=item['bytes'] or sha(a)!=item['sha256'] or b!=a:raise ValueError('comparison source/runtime current/raw changed during quality serialization')
 runtime=source_freeze['runtime'];exe=Path(sys.executable).resolve()
 if str(exe)!=runtime['python_executable'] or sha(exe.read_bytes())!=runtime['python_executable_sha256'] or exe.stat().st_size!=runtime['python_executable_bytes']:raise ValueError('resolved Python changed')
 declarations={m['module']:m for m in runtime['module_closure']};runtimepaths={x['absolute_path'] for x in runtime['file_bindings']};module_rows=[]
 for name,module in sorted(list(sys.modules.items())):
  origin=getattr(getattr(module,'__spec__',None),'origin',None);filename=getattr(module,'__file__',None)
  if filename is None:
   if origin not in ('built-in','frozen') or name not in declarations or declarations[name]['origin']!=origin:raise ValueError('unknown quality nonfile module')
   module_rows.append({'module':name,'origin':origin,'file_backed':False});continue
  p=str(Path(filename).resolve())
  if name=='__main__':
   if p!=str(Path(__file__).resolve()) or p not in paths:raise ValueError('unbound quality serializer')
  elif p not in runtimepaths or name not in declarations or declarations[name]['absolute_path']!=p or declarations[name]['origin']!=origin:raise ValueError('quality file module escaped frozen membership '+name)
  cache=getattr(module,'__cached__',None)
  if cache and Path(cache).exists() and str(Path(cache).resolve()) not in runtimepaths:raise ValueError('quality existing cache escaped frozen runtime')
  module_rows.append({'module':name,'origin':origin,'absolute_path':p,'file_backed':True})
 return pre,cfg,{'preexecution_sha256':sha(raw),'binding_count':len(paths),'source_runtime_raw_equality':True,'resolved_python':str(exe),'loaded_module_membership':module_rows}

try:
 pre,cfg,before=current_guard()
 material=json.loads((HERE/cfg['authored_material']).read_text())
 compared=[json.loads(x) for x in (HERE/'canonical-comparison.first.jsonl').read_text().splitlines()]
 candidate=[json.loads(x) for x in (ROOT/cfg['candidate']).read_text().splitlines()]
 if len(candidate)!=216 or len(compared)!=216 or material['actual_author_execution_id']!=ACTOR:raise ValueError('scope or authorship mismatch')
 comparisons={r['sample_id']:r for r in compared};per_sample={};notes=[]
 for row in candidate:
  sid=row['sample_id'];context=row['controlled_context'];fam=str(row['source_family_index']);c=comparisons[sid]
  if sid in per_sample or row['record_hash']!=c['record_hash'] or c['actual_author_execution_id']!=ACTOR:raise ValueError('exact row/hash/actor join mismatch')
  # Decisions and scores come from the pre-frozen authored material. This is
  # validation of that assessment, not automatic selection from a PASS ledger.
  authored=material['context_assessments'][context];scores=dict(zip(material['rubric_order'],authored['scores']))
  scores.update(material['family_score_overrides'].get(fam,{}))
  decision=material['all_sample_decision_authored']
  if decision=='ACCEPT' and (c['failures'] or not c['target_equal_to_immutable_first']):raise ValueError('factual contradiction to an authored acceptance; stop, never rewrite score/decision')
  issues=[dict(material['moderate_issues_authored'][name],issue_id=name) for name in material['context_specific_issue_selection'][context]]
  y=row['verified_target'];evidence={'supported_scope':material['supported_scope'],'sample_id':sid,'record_hash':row['record_hash'],
   'family_program_inspection':material['family_assessments'][fam],
   'actual_competency':authored['competency'],'actual_difficulty_assessment':authored['difficulty_assessment'],
   'specific_narrow_acceptance_reason':authored['narrow_acceptance_reason'],
   'actual_difficulty':row['difficulty'],'actual_difficulty_explanation':row['source_provenance']['curriculum_difficulty_explanation'],
   'actual_status':y['status'],'actual_new_event_count':len(y['new_events']),'actual_new_effect_count':y['new_effects'],'actual_previous_effect_count':y['previous_effects'],
   'actual_end_state':None if y['checkpoint'] is None else y['checkpoint']['state'],
   'actual_renderer':row['renderer'],'split':row['split'],'field_style_is_correlated_view_not_new_mechanism':True,
   'actual_skill_descriptors':row['skill_tags'],'supplied_property_witnesses':row['property_witnesses'],
   'all_required_factual_checks_hold':not c['failures'],'immutable_first_target_matches':c['target_equal_to_immutable_first'],
   'comparison_first_sha256':sha((HERE/'canonical-comparison.first.jsonl').read_bytes()),
   'narrative_path':str((HERE/cfg['narrative']).relative_to(ROOT)),
   'manual_authorship_sha256':sha((HERE/cfg['authored_material']).read_bytes())}
  per_sample[sid]={'sample_id':sid,'record_hash':row['record_hash'],'scores':scores,'decision':decision,'confidence':authored['confidence'],'critical_defects':[],'issues':issues,'evidence':evidence}
  notes.append({'sample_id':sid,'record_hash':row['record_hash'],'context':context,'renderer':row['renderer'],'scores':scores,'decision':decision,'confidence':authored['confidence']})
 if set(per_sample)!=set(comparisons) or len(per_sample)!=216:raise ValueError('quality scope incomplete')
 quality={'actual_author_execution_id':ACTOR,'source_version':'expert-rsm-quality-authorship-1.0.0','candidate_sha256':cfg['candidate_sha256'],
  'comparison_first_sha256':sha((HERE/'canonical-comparison.first.jsonl').read_bytes()),'quality_authorship_preexecution_sha256':args.expected_preexecution_sha256,
  'supported_scope':material['supported_scope'],'disclosures':material['disclosures'],'per_sample':per_sample,
  'scores_authored_after_actual_semantic_inspection_not_machine_derived':True,'emitter_or_designated_receipts_run':False,'publication_approval':False}
 _,_,after=current_guard()
 output=put_json(HERE/'quality-map.first.json',quality)
 _,_,postwrite=current_guard()
 execution=put_json(HERE/'QUALITY-AUTHORSHIP-EXECUTION.json',{'actual_author_execution_id':ACTOR,'scope':'SERIALIZATION_OF_PREFROZEN_ACTUAL_AGENT_ASSESSMENT_NOT_NEW_TARGET_COMPARISON_OR_EMITTER',
  'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-timer,'argv':sys.argv,'exit_code':0,
  'before':before,'after':after,'postwrite':postwrite,'output':output,'decisions':dict(collections.Counter(x['decision'] for x in notes)),
  'all216_exact_original_hash_joins':notes,'dataset_publication_approval':False,'original_FIRST_and_actual_comparison_unchanged':True})
 first=put_json(HERE/'QUALITY-FIRST.json',{'actual_author_execution_id':ACTOR,'scope':'IMMUTABLE_ACTUAL_QUALITY_FIRST_BEFORE_OTHER_REVIEW_VERDICTS_OR_EMITTER','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'quality_map':output,'actual_authorship_execution':execution,'authored_material_sha256':sha((HERE/cfg['authored_material']).read_bytes()),'narrative_sha256':sha((HERE/cfg['narrative']).read_bytes()),
  'authorship_preexecution_sha256':args.expected_preexecution_sha256,'all216_complete':True,'decisions':dict(collections.Counter(x['decision'] for x in notes)),
  'no_other_reviewer_verdict_read':True,'no_emitter_run':True,'publication_approval':False})
 print(json.dumps({'quality_map':output,'execution':execution,'quality_FIRST':first,'stop_before_emitter':True}))
except BaseException:
 with (HERE/'FAILURE-QUALITY-AUTHORSHIP.txt').open('x') as f:f.write(traceback.format_exc())
 raise
