"""Frozen readback/final custody only. No new prediction or canonical access."""
from pathlib import Path
import datetime,hashlib,json,re,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(b):return hashlib.sha256(b).hexdigest()
def descriptor(path):
 b=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'sha256':sha(b),'bytes':len(b)}
start=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter()
f=json.loads((HERE/'FINALIZATION-PREEXECUTION.json').read_text())
for item in f['bindings']:assert sha((ROOT/item['path']).read_bytes())==item['sha256']
inputpath=ROOT/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl';rs=[json.loads(x) for x in inputpath.read_text().splitlines()]
pred=[json.loads(x) for x in (HERE/'predictions.first.jsonl').read_text().splitlines()]
js=[json.loads(x) for x in (HERE/'initial-judgments.first.jsonl').read_text().splitlines()]
assert len(rs)==len(pred)==len(js)==216
assert [r['sample_id'] for r in rs]==[r['sample_id'] for r in pred]==[r['sample_id'] for r in js]
assert len({r['sample_id'] for r in rs})==216 and len({r['record_hash'] for r in rs})==216
for r,p,j in zip(rs,pred,js):
 assert re.fullmatch('[0-9a-f]{64}',r['record_hash'])
 assert r['record_hash']==p['record_hash']==j['record_hash']
 assert p['reviewer_execution_id']==j['reviewer_execution_id']=='/root/expert_reviewer'
 assert p['input_packet_sha256']==j['evidence']['input_packet_sha256']==sha(inputpath.read_bytes())
 assert j['target_correctness']['status']=='UNKNOWN' and j['scores']['target_correctness'] is None
 assert j['decision']=='PENDING_CANONICAL' and not j['critical_defects']
 assert set(p['expected_output'])=={'status','error','checkpoint','new_events','previous_effects','new_effects','next_event'}
execution=json.loads((HERE/'EXECUTION.json').read_text())
assert execution['exit_code']==0 and execution['all_bindings_unchanged'] and execution['canonical_gold_access'] is False
first=json.loads((HERE/'FIRST.json').read_text())
for item in first['outputs'].values():assert sha((ROOT/item['path']).read_bytes())==item['sha256']
allfiles=[p for p in HERE.rglob('*') if p.is_file()]
assert max(p.stat().st_size for p in allfiles)<25*1024*1024
outputs={p.name:descriptor(p) for p in HERE.iterdir() if p.is_file() and p.name not in ('INPUT-FIRST-FINAL.json','FINALIZATION-EXECUTION.json')}
final={'actual_author_execution_id':'/root/expert_reviewer','status':'COMPLETE_INPUT_ONLY_FINAL_FROZEN_AWAITING_SEPARATE_GOLD_AUTHORIZATION',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':216,'input_packet':descriptor(inputpath),
 'first_receipt':descriptor(HERE/'FIRST.json'),'predictions':descriptor(HERE/'predictions.first.jsonl'),
 'initial_judgments':descriptor(HERE/'initial-judgments.first.jsonl'),'inspection':descriptor(HERE/'SEMANTIC-INSPECTION.md'),
 'source':descriptor(HERE/'input_review_runner.py'),'preexecution':descriptor(HERE/'PREEXECUTION.json'),
 'main_execution':descriptor(HERE/'EXECUTION.json'),'supplement_execution':descriptor(HERE/'SUPPLEMENT-EXECUTION.json'),
 'outputs':outputs,'all216_exact_join_rows_readback_verified':True,'all_first_bytes_unchanged':True,
 'target_correctness':'UNKNOWN','initial_decision':'PENDING_CANONICAL','dataset_acceptances':0,'candidate_gold_or_other_initial_verdict_read':False,
 'no_new_file_exceeds_25MiB':True,'largest_file_bytes':max(p.stat().st_size for p in allfiles),
 'actual_tool_execution':{'main_exec_tool_chunk':'2ff69c','main_unified_exec_session_id':1661,'completion_poll_chunk':'349c26','actual_exit_code':0,
  'main_command':'/opt/anaconda3/bin/python3 -B reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/input_review_runner.py',
  'source_freeze_tool_chunk':'9af742','supplement_tool_chunk':'bc4315'},
 'scope_disclosures':json.loads((HERE/'config.json').read_text())['disclosures'],
 'next_authorization_required':'Root separately grants canonical comparison after independent frozen-FIRST verification; no approval or target assessment before that.'}
p=HERE/'INPUT-FIRST-FINAL.json'
with p.open('x') as out:json.dump(final,out,sort_keys=True,indent=2);out.write('\n')
ex={'actual_author_execution_id':'/root/expert_reviewer','started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'elapsed_seconds':time.perf_counter()-timer,'argv':sys.argv,'exit_code':0,'all_first_outputs_unchanged':True,
 'finalization_preexecution_sha256':sha((HERE/'FINALIZATION-PREEXECUTION.json').read_bytes()),'final':descriptor(p)}
q=HERE/'FINALIZATION-EXECUTION.json'
with q.open('x') as out:json.dump(ex,out,sort_keys=True,indent=2);out.write('\n')
print(json.dumps({'final':descriptor(p),'finalization_execution':descriptor(q),'inspection':final['inspection'],'main_started_utc':execution['started_utc'],'main_finished_utc':execution['finished_utc'],'records':216,'target_correctness':'UNKNOWN','decision':'PENDING_CANONICAL'}))
