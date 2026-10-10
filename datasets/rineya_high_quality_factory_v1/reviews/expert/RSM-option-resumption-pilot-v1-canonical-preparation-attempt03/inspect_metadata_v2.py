"""Read-only actual-agent inspection panels; no new solver/oracle execution.
Frozen source before execution. Metadata-preview-v1 tuple-key formatting failed
and is separately preserved. This corrects formatting only, not evidence.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--expected-source-sha256',required=True);args=parser.parse_args()
if sha(Path(__file__).read_bytes())!=args.expected_source_sha256 or (HERE/'inspect_metadata_v2.py.before-execution.raw').read_bytes()!=Path(__file__).read_bytes():raise ValueError('inspection source external/raw barrier mismatch')
rows=[json.loads(x) for x in (ROOT/'data/RSM-001/candidate_renderings/option-resumption-pilot-v1.jsonl').read_text().splitlines()]
facts={x['sample_id']:x for x in map(json.loads,(ROOT/'reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/independent-facts.first.jsonl').read_text().splitlines())}
initial={x['sample_id']:x for x in map(json.loads,(ROOT/'reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/initial-judgments.first.jsonl').read_text().splitlines())}
families=collections.defaultdict(list)
for r in rows:families[r['source_family_key']].append(r)
panels=[]
for fam,rs in families.items():
 fs=next(r for r in rs if r['controlled_context']=='F' and r['renderer']=='forward')
 contexts=[]
 for r in rs:
  if r['renderer']!='forward':continue
  y=r['verified_target'];other=next(v for v in rs if v['controlled_context']==r['controlled_context'] and v['renderer']=='reverse')
  contexts.append({'context':r['controlled_context'],'sample_ids':[r['sample_id'],other['sample_id']],
   'record_hashes':[r['record_hash'],other['record_hash']],'difficulty':r['difficulty'],
   'explanation':r['source_provenance']['curriculum_difficulty_explanation'],'split':r['split'],
   'status':y['status'],'new_events':len(y['new_events']),'new_effects':y['new_effects'],
   'previous_effects':y['previous_effects'],'end_state':None if y['checkpoint'] is None else y['checkpoint']['state'],
   'witnesses':r['property_witnesses'],'tags':r['skill_tags'],
   'actual_first_input_judgments':[initial[r['sample_id']],initial[other['sample_id']]],
   'new_events_full':y['new_events'],'next_event':y['next_event']})
 panels.append({'family':fam,'source_family_index':fs['source_family_index'],'split':fs['split'],
  'F_task':fs['semantic_task'],'F_input_metrics':facts[fs['sample_id']]['metrics'],'contexts':contexts})
result={'scope':'ACTUAL_POST_FACTUAL_FIRST_METADATA_PANELS_NOT_NEW_EXPECTED_OUTPUT_EXECUTION_OR_AUTOMATIC_QUALITY','actual_author_execution_id':'/root/expert_reviewer','rows':len(rows),'panels':panels,
 'difficulty_histogram':dict(collections.Counter(str(r['difficulty']) for r in rows)),
 'context_difficulty_histogram':dict(collections.Counter(r['controlled_context']+str(r['difficulty']) for r in rows)),
 'skill_tag_histogram':dict(collections.Counter('|'.join(r['skill_tags']) for r in rows))}
with (HERE/'inspection-panels.v2.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'panels_file':str(HERE/'inspection-panels.v2.json'),'sha256':sha((HERE/'inspection-panels.v2.json').read_bytes()),'rows':len(rows),'families':len(panels),'context_difficulty_histogram':result['context_difficulty_histogram']}))
