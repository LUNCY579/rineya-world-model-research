"""New completion custody verification only; no new semantic adjudication."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json
BASE=Path(__file__).resolve().parents[2]
P='reviews/adjudication/belief-gap-v4-'
def bind(p):
 h=hashlib.sha256();n=0
 with (BASE/p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block);n+=len(block)
 return {'path':p,'sha256':h.hexdigest(),'bytes':n}
def load(p):return json.loads((BASE/p).read_text())
def check():
 pre=load(P+'final-readback-preexecution-r1.json')
 for b in pre['bindings']:assert bind(b['path'])==b,b['path']
 final=load(P+'FINAL-r1.json');ex=load(P+'execution-r1.json');audit=load(P+'audit-r1.json')
 assert ex['completed_execution'] is True and ex['exit_code']==0 and ex['actual_author_and_execution_id']=='/root/research_architect'
 assert ex['decision_counts']==final['decision_counts']=={'ACCEPT':3456}
 assert ex['publication_eligible'] is True and ex['whole_cohort_publication_eligible'] is True and audit['whole_cohort_publication_eligible'] is True
 assert ex['bindings_before']==ex['bindings_after'] and ex['source_and_inputs_unchanged'] is True
 for b in ex['bindings_after']:assert bind(b['path'])==b,b['path']
 for b in final['files']:assert bind(b['path'])==b and b['bytes']<25*1024*1024,b['path']
 candidate={r['sample_id']:r['record_hash'] for r in map(json.loads,(BASE/'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl').open())}
 rows=[json.loads(line) for line in (BASE/'reviews/adjudication/belief-gap-v4.jsonl').open()]
 assert len(rows)==len({r['sample_id'] for r in rows})==3456 and Counter(r['decision'] for r in rows)=={'ACCEPT':3456}
 for r in rows:assert r['record_hash']==candidate[r['sample_id']] and r['reviewer_role']=='independent_adjudicator' and r['reviewer_execution_id']=='/root/research_architect' and not r['critical_defects']
 assert bind('reviews/adjudication/belief-gap-v4.jsonl')['sha256']==bind(P+'adjudication-r1.jsonl')['sha256']
 result={'status':'ACTUAL_COMPLETED_FINAL_READBACK','actual_execution_id':'/root/research_architect','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_adjudication_tool_execution':{'command':'/opt/anaconda3/bin/python3 -B reviews/adjudication/belief-gap-v4-adjudicate-r1.py','session_id':16242,'start_chunk_id':'8659d0','progress_chunk_id':'a69432','completion_chunk_id':'fb1086','exit_code':0},'preexecution':bind(P+'preexecution-r1.json'),'actual_adjudicator_final':bind(P+'FINAL-r1.json'),'actual_completed_execution':bind(P+'execution-r1.json'),'designated_judgments':bind('reviews/adjudication/belief-gap-v4.jsonl'),'all3456_unique_identity_role_hash_joins':True,'all126_original_preexecution_bindings_unchanged':True,'all_final_output_bindings_rechecked':True,'decision_counts':{'ACCEPT':3456},'whole_cohort_publication_eligible':True,'publication_eligible':True,'publication_performed':False,'frozen_source_or_output_changes':False,'disclosures':final['disclosures'],'profile_scope_note':'Four distinct declared weighted/type profiles. Cyclic uniform and cyclic weighted share an unweighted cyclic topology; profiles/families are not four unrelated high-level reasoning mechanisms.','read_only_discovery_failures_preserved_as_tool_history':[{'chunk':'d6c227','exit_code':1,'reason':'command -v jq unavailable; chained hash command not run'},{'chunk':'671ce8','exit_code':1,'reason':'rg indented-key pattern missed minified manifest; chained hash command not run'}],'source_draft_corrections_before_first_import':['Actual adversarial role spelling','Actual expert source binding shape','Publication schema aliases/residual-priority ticket','Predeclared designated identical judgment alias'],'actual_adjudication_source_run_failures':[],'next_priority':'Distinct source/mechanism family growth with fresh model-independent review and separately designed genuinely held-out evaluation; labels/length and finite-frequency lookup remain fitted correlations.'}
 p=BASE/(P+'completion-receipt-r1.json');assert not p.exists()
 with p.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'completion_receipt':bind(str(p.relative_to(BASE))),'decision_counts':result['decision_counts'],'unchanged_inputs':126,'files_readback':len(final['files'])}))
if __name__=='__main__':check()
