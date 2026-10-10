#!/opt/anaconda3/bin/python3
"""Same-task renderer revision input review, actual /root/expert_reviewer.
Original canonical exposure disclosed. No fresh candidate/gold/control access.
Only own pre-gold expectations are reusable, after independently proved joins.
"""
import collections,copy,datetime,hashlib,importlib.util,json,pathlib,re,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
sha=lambda b:hashlib.sha256(b).hexdigest()
BINDINGS={
 'reviews/input_only/planning-gap-v2-renderer-r1.jsonl':'5f0b3490720933caa7895f3223244e5e72ed8ca5fb89a97f2eeb516c151f58e3',
 'reviews/input_only/planning-gap-v2-renderer-r1.freeze.json':'808769c0134e05dd9b9b311bebebc3f961a698d907b5fc652b5b5040b5d36eac',
 'reviews/input_only/planning-gap-v2.jsonl':'00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c',
 'reviews/expert/planning-gap-v2.expert_input_predictions.jsonl':'ef9f9dc812f68eaa1442a17cbb7c957f74fe0adebc68e3ea569cdc54c83d2255',
 'reviews/expert/prepare_rgm_planning_gap_v2.py':'7c74fcebba1057bcdb0313c7c658b57a0752d041558f82534fb44a573d98fc53',
 'reviews/expert/review_rgm_planning_gap_v2_input_stage.py':'c85157b7920d3b6070bc8059f73f1c1d7184f7e09bea2df564dbd3306b39e102',
 'schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md':'6e3759ebe9e3e1f801257d30df610b7dea091eb3324f0a76b59ad9520eac5e56',
}
for p,h in BINDINGS.items():assert sha((ROOT/p).read_bytes())==h,p
s=importlib.util.spec_from_file_location('own_frozen_public_v2_prep',ROOT/'reviews/expert/prepare_rgm_planning_gap_v2.py');prep=importlib.util.module_from_spec(s);s.loader.exec_module(prep)
s=importlib.util.spec_from_file_location('own_frozen_input_source',ROOT/'reviews/expert/review_rgm_planning_gap_v2_input_stage.py');stage=importlib.util.module_from_spec(s);s.loader.exec_module(stage)
v1=prep.v1;eq=prep.eq;canon=v1.canon
GUIDANCE='Same204-task/48-family renderer revision, zero task/family growth. Original canonical exposure and original own V2 semantic review knowledge disclosed. Own pre-gold predictions reused only after exact typed semantic joins; same underlying model/public design overlap. Public renderer policy contains prior audit/review summaries; cited private files unread. Not an unseen or fully blind new semantic review.'

def bind(p):
 b=(ROOT/p).read_bytes();return {'path':p,'sha256':sha(b),'bytes':len(b)}
def strict_json(text):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON object key: '+k)
   d[k]=v
  return d
 return json.loads(text,object_pairs_hook=unique)
def payload(row):
 assert row['rendered_input'].count('Task JSON:\n')==1
 lead,text=row['rendered_input'].split('Task JSON:\n',1)
 parsed=strict_json(text)
 assert eq(parsed,row['semantic_task'])
 for label in prep.PRIVILEGED_LABELS:
  assert not re.search(r'(?<!\w)[\"\']?'+re.escape(label)+r'[\"\']?\s*[:=]',lead)
 v1.validate(parsed)
 return lead,parsed
def style(task):
 keys=list(task)
 if keys[0]=='kind' and keys[-1]=='plan':return 'forward'
 if keys[0]=='plan' and keys[-1]=='kind':return 'reverse'
 raise ValueError('no distinct actual top-key-order renderer witness')
def presentation_copy(task,renderer):
 t=copy.deepcopy(task)
 if t['kind']=='blocks':t['goal_on']={k:t['goal_on'][k] for k in sorted(t['goal_on'],reverse=renderer=='reverse')}
 return t
def fresh_fixture_calibration():
 on={'a':'TABLE','b':'TABLE','c':'TABLE'};goal={'c':'a','a':'b'}
 old=v1.b(on,goal,[('a','b'),('c','a')]);new=presentation_copy(old,'forward')
 check=[]
 def okay(name,value):assert value,name;check.append({'name':name,'pass':True})
 okay('goal key sorting preserves typed semantics',eq(old,new) and list(new['goal_on'])==['a','c'])
 okay('goal key sorting preserves complete own target',eq(v1.solve(old)[0],v1.solve(new)[0]))
 reverse=copy.deepcopy(old);reverse={k:reverse[k] for k in reversed(list(reverse))};reverse=presentation_copy(reverse,'reverse')
 okay('reverse presentation separate style and same values',style(reverse)=='reverse' and eq(old,reverse) and list(reverse['goal_on'])==['c','a'])
 wrong=copy.deepcopy(old);wrong['plan'].reverse()
 okay('action order change rejected by typed join',not eq(old,wrong))
 okay('boolean integer distinction in join',not eq({'fuel':True},{'fuel':1}))
 failed=False
 try:strict_json('{"goal_on":{"a":"TABLE","a":"b"}}')
 except ValueError:failed=True
 okay('duplicate object keys rejected',failed)
 return check

def run():
 started=time.monotonic();sources_before={p:bind(p) for p in BINDINGS};own_source=bind(str(pathlib.Path(__file__).relative_to(ROOT)))
 calibration=fresh_fixture_calibration();prep_result=prep.preparation()
 old=[strict_json(l) for l in (ROOT/'reviews/input_only/planning-gap-v2.jsonl').read_text().splitlines()]
 new=[strict_json(l) for l in (ROOT/'reviews/input_only/planning-gap-v2-renderer-r1.jsonl').read_text().splitlines()]
 oldpred={r['sample_id']:r for r in (strict_json(l) for l in (ROOT/'reviews/expert/planning-gap-v2.expert_input_predictions.jsonl').read_text().splitlines())}
 expected_fields={'sample_id','record_hash','source_family_key','split','semantic_task','rendered_input'}
 assert len(old)==len(new)==len(oldpred)==408 and all(set(r)==expected_fields for r in new)
 assert len({r['sample_id'] for r in new})==len({r['record_hash'] for r in new})==408
 assert not ({r['sample_id'] for r in old}&{r['sample_id'] for r in new})
 old_index=collections.defaultdict(list);old_style={}
 for r in old:
  lead,p=payload(r);renderer=style(p);old_style[r['sample_id']]=renderer
  old_index[(r['source_family_key'],r['split'],canon(r['semantic_task']),renderer)].append(r)
  assert oldpred[r['sample_id']]['record_hash']==r['record_hash']
 predictions=[];first=[];joins=[];changed=[];failures=[];fresh_cache={};family_items=collections.defaultdict(dict);goal_order_hist=collections.Counter();all_leads=set()
 for r in new:
  sid=r['sample_id'];errors=[];pr=None;oldrow=None;fresh=None;record_facts={}
  try:
   lead,p=payload(r);renderer=style(p);all_leads.add(lead)
   key=(r['source_family_key'],r['split'],canon(r['semantic_task']),renderer)
   matches=old_index.get(key,[]);assert len(matches)==1,'exact typed semantics/style original join not unique'
   oldrow=matches[0];pr=oldpred[oldrow['sample_id']]
   old_lead,old_payload=payload(oldrow)
   assert eq(r['semantic_task'],oldrow['semantic_task']) and r['split']==oldrow['split'] and r['source_family_key']==oldrow['source_family_key']
   assert lead==old_lead
   expected=presentation_copy(old_payload,renderer)
   assert list(p)==list(old_payload),'top key ordering changed'
   assert eq(p,old_payload),'typed semantic payload changed'
   if p['kind']=='blocks':
    assert list(p['on'])==list(old_payload['on']),'on-map presentation changed'
    assert [list(a) for a in p['plan']]==[list(a) for a in old_payload['plan']],'action object key presentation changed'
    assert list(p['goal_on'])==sorted(p['goal_on'],reverse=renderer=='reverse'),'goal key order not policy style'
    goal_order_hist[(renderer,tuple(p['goal_on']))]+=1
   # Exact text witness: rewrite only the goal-key presentation in original.
   assert r['rendered_input']==old_lead+'Task JSON:\n'+json.dumps(expected,ensure_ascii=False,separators=(',',':')),'visible change exceeds goal-key-only presentation'
   raw_changed=r['rendered_input']!=oldrow['rendered_input']
   nkey=stage.signature(r['semantic_task'])
   if nkey not in fresh_cache:fresh_cache[nkey]=prep.derive_visible(r)
   fresh=fresh_cache[nkey]
   assert eq(fresh['target'],pr['expected_target']),'fresh input-only solver differs from frozen own target'
   replay,error,states=v1.replay(p)
   assert eq(replay,{k:pr['expected_target'][k] for k in replay})
   assert eq(states,pr['independent_input_facts']['measurements']['prefix_states'])
   record_facts={'renderer_inferred_from_actual_payload':renderer,'original_sample_id':oldrow['sample_id'],'original_record_hash':oldrow['record_hash'],'exact_typed_semantic_join':True,'split_family_join':True,'visible_changed':raw_changed,'target_fresh_input_solver_matches_reused_expectation':True,'actual_top_keys':list(p),'old_goal_keys':list(old_payload.get('goal_on',{})),'new_goal_keys':list(p.get('goal_on',{})),'goal_key_only_text_reconstruction':True,'prompt_lead_unchanged':True,'fresh_minimum_moves':len(fresh['target']['shortest_plan']) if fresh['target']['shortest_plan'] is not None else None}
   if raw_changed:changed.append({'sample_id':sid,'record_hash':r['record_hash'],**record_facts})
   joins.append({'sample_id':sid,'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],**record_facts})
   item=family_items[r['source_family_key']].setdefault(nkey,{'task':stage.normalized(r['semantic_task']),'facts':fresh,'rows':[]});item['rows'].append(r)
  except (AssertionError,ValueError,TypeError,KeyError) as error:
   errors.append({'category':'renderer_revision_input_or_reuse_failure','severity':'high','evidence':type(error).__name__+': '+str(error),'fix':'Preserve current bytes and repair the concrete rendering/join or own solver failure in a fresh revision before comparison.'});failures.append({'sample_id':sid,'errors':errors})
  predictions.append({'sample_id':sid,'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],'reviewer_execution_id':ROLE,'stage':'SAME_TASK_RENDERER_INPUT_EXPECTATION_BEFORE_FRESH_GOLD','expected_target':pr['expected_target'] if pr is not None and not errors else None,'independent_input_facts':fresh if fresh is not None and not errors else None,'input_task_sha256':sha(canon(r['semantic_task']).encode()),'rendered_input_sha256':sha(r['rendered_input'].encode()),'reuse_join':record_facts,'original_prediction_file_sha256':BINDINGS['reviews/expert/planning-gap-v2.expert_input_predictions.jsonl'],'prior_original_canonical_exposure':True,'fresh_r1_gold_read':False,'guidance':GUIDANCE,'actual_errors':errors})
  issues=errors+[{'category':'same_task_correlated_reuse','severity':'moderate','evidence':'204 existing tasks and48 existing source families; own original pre-gold expectations reused after exact joins and checked by fresh own solver. Original canonical targets were previously seen, and common-model errors remain possible.','fix':'Disclose correlation and reuse; count zero new tasks/families and no unseen semantic review.'},{'category':'pending_fresh_target_control_metadata_comparison','severity':'low','evidence':'Fresh canonical target, full record hash, tags/witnesses/difficulty and renderer-specific manifest declarations are intentionally unread in this input stage.','fix':'Compare only after the immutable all408 expectation/first-judgment freeze and explicit root authorization.'}]
  first.append({'receipt_id':'expert-renderer-r1-input-'+sha(canon([sid,r['record_hash'],ROLE]).encode())[:24],'sample_id':sid,'record_hash':r['record_hash'],'reviewer_role':'expert_reviewer','reviewer_execution_id':ROLE,'review_status':'FIRST_RENDERER_INPUT_JUDGMENT_BEFORE_FRESH_GOLD','decision':'REVISE_INPUT_PENDING_COMPARISON' if errors else 'INPUT_CLEAR_PENDING_FRESH_CANONICAL_COMPARISON','scores':{'semantic_correctness':5 if not errors else 1,'target_correctness':None,'instruction_clarity':5 if not errors else 1,'reasoning_validity':5 if not errors else 1,'generalization_value':3,'difficulty_quality':3,'diversity':3,'shortcut_resistance':3},'scores_scope':'Fresh input stage only; canonical correctness/metadata unassessed. Prior same-task canonical exposure disclosed. No dataset ACCEPT.','critical_defects':errors,'issues':issues,'confidence':0.98 if not errors else 0.95,'evidence':record_facts,'guidance':GUIDANCE})
 family_audits=[];shared_goal_orders=[]
 for family,items in sorted(family_items.items()):
  index=stage.family_index(family);contexts,checks=stage.audit_family(list(items.values()),index)
  family_audits.append({'source_family_key':family,'context_sample_ids':{k:[r['sample_id'] for r in v['rows']] for k,v in contexts.items()},'input_context_checks':checks})
  if stage.stratum(index)=='D':
   for renderer in ('forward','reverse'):
    def select(ctx):
     options=[r for r in contexts[ctx]['rows'] if style(payload(r)[1])==renderer]
     assert len(options)==1
     return options[0]
    left,right=select('D0'),select('D1');a,b=payload(left)[1],payload(right)[1]
    shared=[k for k in b['goal_on'] if k in a['goal_on']]
    okay=shared==list(a['goal_on']);assert okay
    shared_goal_orders.append({'source_family_key':family,'renderer':renderer,'D0_sample_id':left['sample_id'],'D1_sample_id':right['sample_id'],'D0_goal_keys':list(a['goal_on']),'D1_shared_goal_keys':shared,'preserved':okay})
 assert len(predictions)==len(first)==408
 sources_after={p:bind(p) for p in BINDINGS};assert sources_before==sources_after and bind(own_source['path'])==own_source
 raw_variants=len({canon(r['semantic_task']) for r in new});all_context_checks=all(c['pass'] for f in family_audits for c in f['input_context_checks'])
 def jsonl_write(name,values):
  path='reviews/expert/'+name
  with (ROOT/path).open('x') as f:
   for value in values:f.write(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n')
  return bind(path)
 artifact_predictions=jsonl_write('planning-gap-v2-renderer-r1.expert_input_predictions.jsonl',predictions)
 artifact_first=jsonl_write('planning-gap-v2-renderer-r1.expert_first_input_judgments.jsonl',first)
 analysis={'actual_author':ROLE,'stage':'SAME_TASK_RENDERER_R1_INPUT_ONLY_COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':own_source,'sources_before':sources_before,'sources_after':sources_after,'actual_rows':408,'exact_typed_semantic_style_joins':len(joins),'fresh_own_target_recomputations':len(fresh_cache),'reused_own_predictions':len(joins),'normalized_tasks':len(fresh_cache),'raw_typed_JSON_variants':raw_variants,'source_families':len(family_items),'new_tasks':0,'new_families':0,'changed_visible_rows':len(changed),'changed_rows':changed,'joins':joins,'family_input_audits':family_audits,'all_context_input_requirements_pass':all_context_checks,'shared_goal_extension_orders':shared_goal_orders,'shared_order_checks':len(shared_goal_orders),'distinct_contract_leads':len(all_leads),'fresh_fixture_calibration':calibration,'own_V2_preparation_calibration_rerun_count':prep_result['actual_check_entries'],'input_failures':failures,'prior_original_canonical_exposure':True,'fresh_r1_gold_manifest_other_private_results_read':False,'public_policy_prior_result_summary_exposure':True,'guidance':GUIDANCE,'expected_predictions':artifact_predictions,'first_judgments':artifact_first,'actual_elapsed_seconds':time.monotonic()-started,'actual_exit_code':0,'no_dataset_approval':True}
 path='reviews/expert/planning-gap-v2-renderer-r1.expert_input_analysis.json'
 with (ROOT/path).open('x') as f:json.dump(analysis,f,indent=2);f.write('\n')
 execution={'actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'ALL408_REUSED_AND_RECOMPUTED_EXPECTATIONS_FIRST_JUDGMENTS_BEFORE_FRESH_GOLD','actual_command':[sys.executable,'-B',str(pathlib.Path(__file__).relative_to(ROOT))],'actual_exit_code':0,'own_source':own_source,'sources_before_after_unchanged':True,'actual_rows':408,'exact_typed_semantic_style_joins':len(joins),'complete_expectation_rows':sum(p['expected_target'] is not None for p in predictions),'first_input_judgments':408,'changed_visible_rows':len(changed),'input_failure_rows':len(failures),'artifacts':[artifact_predictions,artifact_first,bind(path)],'prior_canonical_exposure':True,'fresh_gold_manifest_other_private_results_read':False,'guidance':GUIDANCE,'actual_elapsed_seconds':analysis['actual_elapsed_seconds'],'no_dataset_or_publication_approval':True}
 path='reviews/expert/planning-gap-v2-renderer-r1.expert_input_execution.json'
 with (ROOT/path).open('x') as f:json.dump(execution,f,indent=2);f.write('\n')
 print(json.dumps({'execution':bind(path),'predictions':artifact_predictions,'first':artifact_first,'joins':len(joins),'fresh_recomputed_tasks':len(fresh_cache),'changed_visible_rows':len(changed),'input_failure_rows':len(failures),'elapsed_seconds':analysis['actual_elapsed_seconds']}))

if __name__=='__main__':run()
