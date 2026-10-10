"""Own input-only aggregate inspection, after immutable FIRST, before gold.
Reads only authorized packet and own frozen outputs. No interpreter import.
"""
from pathlib import Path
import collections,datetime,hashlib,json,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
def key(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.perf_counter()
freeze=json.loads((HERE/'SUPPLEMENT-PREEXECUTION.json').read_text())
for r in freeze['bindings']:assert sha((ROOT/r['path']).read_bytes())==r['sha256']
rows=[json.loads(x) for x in (ROOT/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl').read_text().splitlines()]
pred={x['sample_id']:x for x in map(json.loads,(HERE/'predictions.first.jsonl').read_text().splitlines())}
facts={x['sample_id']:x for x in map(json.loads,(HERE/'independent-facts.first.jsonl').read_text().splitlines())}
fs=[r for r in rows if r['sample_id'].endswith('-F-forward')]
traces=[facts[r['sample_id']]['original_full_run']['events'] for r in fs]
prefix=0
for group in zip(*traces):
 if len({key(e) for e in group})!=1:break
 prefix+=1
validnew=[e for r in rows for e in pred[r['sample_id']]['expected_output']['new_events']]
branch=collections.Counter(str(e['condition']) for e in validnew if e['event']=='branch')
returns=collections.Counter(('root' if e['instance_id']==0 else 'child')+':'+str(e['success_ok']) for e in validnew if e['event']=='return')
initiations=collections.Counter(('root' if e['instance_id']==0 else 'child')+':'+str(e['initiation_ok']) for e in validnew if e['event']=='enter')
res=[r for r in rows if r['sample_id'].endswith('-R-forward')]
paused=[r for r in rows if r['sample_id'].endswith('-P-forward')]
invalid=[r for r in rows if pred[r['sample_id']]['expected_output']['status']=='invalid']
hist=lambda vals:dict(collections.Counter(str(x) for x in vals))
summary={'actual_author_execution_id':'/root/expert_reviewer','first_receipt_sha256':sha((HERE/'FIRST.json').read_bytes()),
 'input_only':True,'canonical_gold_read':False,'dataset_approval':False,
 'all216_initial_states_histogram':hist(key(r['semantic_task']['initial_state']) for r in rows),
 'all12_F_identical_committed_prefix_events':prefix,'common_F_prefix':traces[0][:prefix],
 'distinct_R_history_values':len({key(r['semantic_task']['history']) for r in res}),
 'distinct_R_checkpoint_values':len({key(r['semantic_task']['checkpoint']) for r in res}),
 'distinct_P_full_expected_outputs':len({key(pred[r['sample_id']]['expected_output']) for r in paused}),
 'new_committed_branch_outcomes_all216':dict(branch),'new_returns_root_vs_child_all216':dict(returns),
 'new_entry_outcomes_root_vs_child_all216':dict(initiations),
 'invalid_checkpoint_mismatch_paths_all24':hist(key(d['path']) for r in invalid for d in facts[r['sample_id']]['checkpoint_mismatch_paths']),
 'F_actual_range':{'events':[min(len(t) for t in traces),max(len(t) for t in traces)],
   'effects':[min(sum(e['event']=='effect' for e in t) for t in traces),max(sum(e['event']=='effect' for e in t) for t in traces)]},
 'actual_valid_run_abs_max':max(f['cohort_bounds_observations']['run_abs_max'] for f in facts.values()),
 'per_family_F_results':[{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'events':len(pred[r['sample_id']]['expected_output']['new_events']),
   'effects':pred[r['sample_id']]['expected_output']['new_effects'],'state':pred[r['sample_id']]['expected_output']['checkpoint']['state'],
   'resume_state':pred[r['sample_id'].replace('-F-forward','-R-forward')]['expected_output']['checkpoint']['state'],
   'restart_state':pred[r['sample_id'].replace('-F-forward','-T-forward')]['expected_output']['checkpoint']['state']} for r in fs],
 'scope':'Own actual predictions and tasks only. Counts include field-order duplication; they are observations rather than model-performance or novelty acceptance.'}
p=HERE/'supplement-measurements.json'
with p.open('x') as f:json.dump(summary,f,sort_keys=True,indent=2);f.write('\n')
after={r['path']:sha((ROOT/r['path']).read_bytes()) for r in freeze['bindings']}
assert all(after[r['path']]==r['sha256'] for r in freeze['bindings'])
execution={'actual_author_execution_id':'/root/expert_reviewer','started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'elapsed_seconds':time.perf_counter()-clock,'argv':sys.argv,'python':sys.executable,'exit_code':0,'all_bound_bytes_unchanged':True,
 'preexecution_sha256':sha((HERE/'SUPPLEMENT-PREEXECUTION.json').read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),
 'output':{'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}}
p=HERE/'SUPPLEMENT-EXECUTION.json'
with p.open('x') as f:json.dump(execution,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'execution_sha256':sha(p.read_bytes()),'measurements':summary}))
