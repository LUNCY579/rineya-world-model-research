"""Custody-only completion of own actual input-first; no semantic imports/gold reads."""
import datetime,hashlib,json,pathlib,sys
D=pathlib.Path(__file__).resolve().parent;B=D.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def new(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
def main():
 pre_raw=(D/'finalization-preexecution-r2.json').read_bytes();assert sha(pre_raw)==sys.argv[1];pre=json.loads(pre_raw)
 for v in pre['bindings']:
  path=pathlib.Path(v['path']) if v['path'].startswith('/') else B/v['path'];raw=path.read_bytes();assert sha(raw)==v['sha256'] and len(raw)==v['bytes'] and raw==(B/v['archive']).read_bytes()
 exe=json.loads((D/'execution-r2.json').read_bytes());assert exe['completed_execution'] and exe['rows_predicted']==exe['initial_judgments']==exe['independent_full_output_agreements']==216 and exe['frozen_bindings_unchanged'] and not exe['gold_or_other_first_verdicts_read']
 pred=[json.loads(l) for l in (D/'predictions-r2.jsonl').read_bytes().splitlines()];initial=[json.loads(l) for l in (D/'initial-judgments-r2.jsonl').read_bytes().splitlines()];assert len(pred)==len(initial)==216;assert len({p['sample_id'] for p in pred})==216
 for p,j in zip(pred,initial):
  # Engine prediction order and INITIAL family order differ; exact ID/hash join below.
  pass
 pj={p['sample_id']:p for p in pred};jj={j['sample_id']:j for j in initial};assert set(pj)==set(jj)
 for sid,j in jj.items():assert j['record_hash']==pj[sid]['record_hash'] and j['target_correctness']=='UNKNOWN' and not j['data_approval'] and j['decision']=='PENDING_CANONICAL_COMPARISON' and j['reviewer_execution_id']=='/root/research_architect'
 own_files=sorted(p for p in D.rglob('*') if p.is_file());bindings=[]
 for p in own_files:
  raw=p.read_bytes();assert len(raw)<25*1024*1024;bindings.append({'path':str(p.relative_to(B)),'sha256':sha(raw),'bytes':len(raw),'scope':'owned-local-input-first-evidence'})
 for v in pre['bindings']:
  if not v['path'].startswith(str(D.relative_to(B))+'/'):bindings.append({'path':v['path'],'sha256':v['sha256'],'bytes':v['bytes'],'scope':'authorized-external-public-own-helper-or-runtime','archive':v['archive']})
 def bind(name):
  raw=(D/name).read_bytes();return {'path':str((D/name).relative_to(B)),'sha256':sha(raw),'bytes':len(raw)}
 at=datetime.datetime.now(datetime.timezone.utc).isoformat();first={'status':'FIRST_FROZEN_INPUT_ONLY','actual_execution_id':'/root/research_architect','reviewer_role':'independent_adjudicator_input_first','completed_execution':True,'completed_agent_input_inspection':True,'at_utc':at,'rows_predicted':216,'initial_judgments':216,'prediction':bind('predictions-r2.jsonl'),'initial':bind('initial-judgments-r2.jsonl'),'execution':bind('execution-r2.json'),'agent_inspection':bind('AGENT_INPUT_INSPECTION-r2.md'),'source_freeze':bind('source-freeze-r2.json'),'first_failed_attempt_freeze':bind('source-freeze-r1.json'),'first_failed_attempt':bind('failure-execution-r1.json'),'first_tool_output':bind('TOOL_OUTPUT-attempt01.json'),'retry_comparison':bind('retry-comparison-r2.json'),'finalization_preexecution_sha256':sha(pre_raw),'bindings':bindings,'prediction_status_counts':json.loads((D/'input-audit-r2.json').read_bytes())['prediction_status_counts'],'gold_target_correctness':'UNKNOWN','canonical_comparison_completed':False,'gold_or_other_initial_final_verdicts_read':False,'private_producer_or_formal_implementation_read':False,'data_approval':False,'publication_eligible':None,'all_writes_new_authorized_prefix':True,'first_r1_initial_semantics_preserved':True,'first_r2_draft_preserved':True,'disclosures':['original public task/curriculum author','prior public metadata review and legacy/multi-track exposure','possible later adjudicator overlap','root guidance/common underlying model','own earlier solver/simulator reuse, shared parser','finite supplied programs/correlated views/exposed dev'],'future_canonical_comparison_requires_separate_root_grant':True}
 new(D/'FIRST.json',first);first_binding=bind('FIRST.json')
 final={'status':'FINAL_INPUT_FIRST_COMPLETE_NO_APPROVAL','actual_execution_id':'/root/research_architect','completed_execution':True,'at_utc':at,'FIRST':first_binding,'prediction':first['prediction'],'initial':first['initial'],'execution':first['execution'],'agent_inspection':first['agent_inspection'],'source_freeze':first['source_freeze'],'bindings':bindings+[dict(first_binding,scope='owned-immutable-FIRST')],'counts':{'rows':216,'initials':216,'independent_full_output_agreements':216,'observed_family_ids':12,'semantic_parent_ids':108,'field_view_pairs':108,'source_independence_verified':None},'actual_tool_execution':{'r1':'7b32e7 exit1 preserved runtime custody failure','r2':'3c3057 exit0 full recomputation/runtime closure','retry_comparison':'20f2ed exit0','agent_display':'bcec86 exit0; f26cd5 exit0 full output preservation'},'first_failure_preserved':True,'canonical_gold_correctness':'UNKNOWN','data_approval':False,'publication_eligible':None,'independent_source_novelty_or_manifest_reviewed':False,'future_gold_and_other_review_access_requires_separate_grant':True,'backup_status':'LOCAL_ONLY_UNLESS_SEPARATE_BACKUP_PROOF','writes_finished_after_completion_receipt':True}
 new(D/'FINAL.json',final);fb=bind('FINAL.json');completion={'actual_execution_id':'/root/research_architect','status':'FINAL','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'FIRST':first_binding,'FINAL':fb,'completed_input_first':True,'data_approval':False,'writes_finished':True}
 new(D/'completion-receipt.json',completion)
 print(json.dumps({'FIRST':first_binding,'FINAL':fb,'completion':bind('completion-receipt.json'),'bindings':len(bindings),'new_files_including_completion':len(own_files)+3,'writes_finished':True}))
if __name__=='__main__':main()
