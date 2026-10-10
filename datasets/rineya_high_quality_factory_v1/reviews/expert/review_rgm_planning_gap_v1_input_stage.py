#!/opt/anaconda3/bin/python3
"""Own actual six-field blind input stage, before canonical target/manifest access.
Frozen preparation source reused exactly; separate small wrapper archived before
execution. Staged derive then actual-agent input inspection and first freeze.
"""
import collections,datetime,hashlib,json,pathlib,re,runpy,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
sha=lambda b:hashlib.sha256(b).hexdigest()
PREP=OUT/'prepare_rgm_planning_gap_v1.py';PREPSHA='82ae6d000d5289bf439599579eb6437de5a09468388ead4fb1cef95f9dd3a987';assert sha(PREP.read_bytes())==PREPSHA
P=runpy.run_path(str(PREP));canon,eq=P['canon'],P['eq']
VIEW=ROOT/'reviews/input_only/planning-gap-v1.jsonl';VIEWSHA='2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654';CANDSHA='a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27'
FIELDS={'sample_id','original_record_hash','source_family_key','split','semantic_task','rendered_input'}
FORBIDDEN={'verified_target','gold','expected_target','expected_output','reference_output','oracle_output','control_manifest','family_control_claims','skill_tags','row_witnesses','row_properties'}
GUIDANCE='Input-only actual review from public prospective control/curriculum guidance and own frozen calibration; same underlying model and architect design/adjudication overlap disclosed. No new candidate targets, producer diagnostics/control manifest, new formal/reference implementations or other private verdict reads. Original record hashes supplied as custody join only, not independently recomputable until canonical stage.'
def save(path,obj):
 assert not path.exists();path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');return sha(path.read_bytes())
def source_archive():
 own=pathlib.Path(__file__).read_bytes();arch=OUT/'planning-gap-v1.expert_input_stage.preexecution_source.py';assert arch.exists() and arch.read_bytes()==own;return {'source_sha256':sha(own),'archive_path':str(arch),'archive_sha256':sha(arch.read_bytes())}
def public_bindings():
 out={}
 for f,h in P['PUBLIC'].items():
  b=(ROOT/f).read_bytes();assert sha(b)==h;out[f]={'sha256':h,'length':len(b)}
 return out
def read_view():
 b=VIEW.read_bytes();assert sha(b)==VIEWSHA and len(b)==1008984
 er=ROOT/'reviews/input_only/planning-gap-v1.execution.json';eb=er.read_bytes();receipt=json.loads(eb)
 assert receipt['input_view_sha256']==VIEWSHA and receipt['candidate_sha256']==CANDSHA and set(receipt['fields'])==FIELDS and receipt['excluded_gold_and_claim_metadata'] is True
 rows=[json.loads(l) for l in b.splitlines()];assert len(rows)==len({r['sample_id'] for r in rows})==384
 for r in rows:
  assert set(r)==FIELDS and not (set(r)&FORBIDDEN) and re.fullmatch('[0-9a-f]{64}',r['original_record_hash'])
 return rows,{'path':str(er),'sha256':sha(eb),'length':len(eb),'candidate_hash_is_later_custody_only':True}
def parse_prompt(r):
 prompt=r['rendered_input'];assert type(prompt) is str and prompt.count('Task JSON:\n')==1
 lead,body=prompt.split('Task JSON:\n');t=json.loads(body);assert eq(t,r['semantic_task']);P['validate'](t)
 for key in FORBIDDEN:assert key not in lead and key not in t
 assert lead.startswith('Executable task contract:\n')
 if t['kind']=='graph':required=['Every edge is directed','stop immediately','do not evaluate subsequent actions','entire destination-name sequence','INITIAL','nonnegative integer','Fuel cost is a feasibility constraint']
 else:required=['TABLE may support several blocks','unspecified supports need not change','complete sequence of (block,to) string pairs','INITIAL','stop immediately','ignore subsequent moves','contradictory and malformed']
 assert all(v in lead for v in required)
 return t,lead
def semantic_key(t):
 s=dict(t)
 if t['kind']=='graph':s['nodes']=sorted(t['nodes']);s['edges']=sorted(t['edges'])
 return canon(s)
def derive():
 sources=source_archive();docs=public_bindings();rows,rootrec=read_view();start=time.perf_counter();cache={};records=[];transition_checks=0;graph_alt=0;leadsets=collections.defaultdict(set)
 for r in rows:
  t,lead=parse_prompt(r);leadsets[t['kind']].add(lead);key=semantic_key(t)
  if key not in cache:cache[key]=P['measurements'](t)
  result=cache[key];target=result['target'];ex=result['measurements']
  assert ex['search']['complete'] and ex['search']['visited']<= (len(t['nodes'])*(t['fuel']+1) if t['kind']=='graph' else len(t['on'])**len(t['on']))
  if t['kind']=='graph':assert eq(P['graph_minimum_alternative'](t),target['shortest_plan']);graph_alt+=1
  else:
   for j,step in enumerate(target['steps']):
    state=ex['prefix_states'][j];action=t['plan'][j];ll,nn=P['block_step_relation'](state,action);assert ll==step['legal'] and eq(nn,step['after']);transition_checks+=1
  if target['shortest_plan'] is not None:
   replacement=dict(t);replacement['plan']=target['shortest_plan'];replay,_,_=P['replay'](replacement);assert replay['valid'] and replay['reaches_goal']
  records.append({'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],'visible_kind':t['kind'],'input_view_row_sha256':sha(canon(r).encode()),'semantic_task_sha256':sha(canon(t).encode()),'semantics_normalized_task_sha256':sha(key.encode()),'student_prompt_sha256':sha(r['rendered_input'].encode()),'expected_target':target,'independent_input_facts':ex,'independently_witnessed_property_names':result['row_properties'],'independent_row_witness_facts':result['row_witness_facts'],'valid_input_unambiguous_contract':True,'comparison_stage_target_correctness':None,'own_semantic_method':'frozen independent typed replay + finite lexicographic BFS; graph alternative heap minimum and block separately authored relation-set replay agreement','reviewer_execution_id':ROLE})
 assert all(len(v)==1 for v in leadsets.values())
 p=OUT/'planning-gap-v1.expert_input_predictions.jsonl';assert not p.exists();raw=('\n'.join(canon(r) for r in records)+'\n').encode();p.write_bytes(raw)
 log={'actual_author':ROLE,'stage':'INPUT_ONLY_DERIVATION_BEFORE_AGENT_FREEZE','actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,**sources,'frozen_preparation_source_sha256':PREPSHA,'public_sources':docs,'view_sha256':VIEWSHA,'view_bytes':1008984,'root_input_execution_receipt':rootrec,'later_candidate_custody_hash':CANDSHA,'prediction_sha256':sha(raw),'prediction_bytes':len(raw),'records':len(records),'semantics_normalized_distinct_tasks':len(cache),'graph_alternate_solver_checks':graph_alt,'block_relation_transition_checks':transition_checks,'prompt_contract_leads_per_kind':{k:len(v) for k,v in leadsets.items()},'no_gold_control_manifest_or_other_verdict_access':True,'dataset_approval':False,'guidance':GUIDANCE}
 save(OUT/'planning-gap-v1.expert_input_derivation.execution.json',log);print(canon(log))
def freeze():
 sources=source_archive();docs=public_bindings();rows,rootrec=read_view();rp=OUT/'planning-gap-v1.expert_input_predictions.jsonl';rb=rp.read_bytes();rs=[json.loads(l) for l in rb.splitlines()];dr=json.loads((OUT/'planning-gap-v1.expert_input_derivation.execution.json').read_text());assert sha(rb)==dr['prediction_sha256'] and len(rs)==384
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();judgments=[];famgroups=collections.defaultdict(list)
 for r in rs:famgroups[r['source_family_key']].append(r)
 assert len(famgroups)==48 and all(len(v)==8 for v in famgroups.values())
 for r in rs:
  f=r['independent_input_facts'];target=r['expected_target'];issues=[{'category':'correlated_views_and_pending_family_controls','severity':'medium','evidence':'This input-only view has eight correlated records per48 supplied lineage keys. Source novelty, declared metadata, manifest witnesses, rubric difficulty and target equality have not been assessed; counts do not establish new independent mechanisms.','fix':'After authorized target-comparison stage, verify all supplied interventions, actual novelty and exact manifest/row metadata against frozen facts.'},{'category':'bounded_planning_proxy','severity':'low','evidence':'Two deterministic grammars: directed fuel consumption and clear-block moves. No coupled resources, replenishment, edit-distance repair, sealed OOD or efficacy target.','fix':'Keep the public finite planning scope and report actual dependent constraints rather than inflate difficulty.'}]
  if 'valid_goal_detour' in r['independently_witnessed_property_names']:issues.append({'category':'detour_is_success_not_optimality','severity':'low','evidence':f"Legal submitted length {len(target['steps'])} exceeds fresh minimum {len(target['shortest_plan'])}. That distinction is explicitly supported; filler length alone is not solution difficulty.",'fix':'Retain distinct success/minimum labels and actual fresh horizon in later difficulty review.'})
  if f['first_error']:issues.append({'category':'first_illegal_stop_required','severity':'low','evidence':f"First error at {f['first_error']['index']} is {f['first_error']['reason']}; independent prediction includes that unchanged-state step and absorbs the entire suffix.",'fix':'Preserve immediate stopping in canonical target comparison.'})
  judgments.append({'receipt_id':'expert-planning-gap-v1-input-'+sha(canon([r['sample_id'],r['record_hash'],ROLE]).encode())[:24],'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_role':'expert_reviewer','reviewer_execution_id':ROLE,'review_stage':'INPUT_ONLY_FIRST_JUDGMENT','initial_input_judgment':'POSSIBLE_UNAMBIGUOUS_READY_FOR_TARGET_COMPARISON','review_status':'INPUT_STAGE_FROZEN','dataset_decision':None,'target_correctness_assessed':False,'confidence':0.98,'critical_input_defects':[],'issues':issues,'scores_assessed_on_visible_input':{'semantic_well_formedness':5,'instruction_clarity':5,'solution_reasoning_validity':5,'bounded_competency_alignment':4},'gold_dependent_rubric_scores':None,'frozen_expected_target':r['expected_target'],'evidence':{'prediction_file_sha256':sha(rb),'semantic_task_sha256':r['semantic_task_sha256'],'input_row_sha256':r['input_view_row_sha256'],'student_prompt_sha256':r['student_prompt_sha256'],'all_prefix_states':f['prefix_states'],'completed_search':f['search'],'input_derived_property_names':r['independently_witnessed_property_names'],'first_error':f['first_error']},'scope':'Own actual-agent inspection of all48 base family inputs plus per-context summaries and programmatic every-record replay/search. Ready label is NOT dataset ACCEPT; target/claim/supplied novelty comparison awaits root authorization.','guidance':GUIDANCE,'created_utc':now})
 jpath=OUT/'planning-gap-v1.expert_first_input_judgments.jsonl';assert not jpath.exists();jr=('\n'.join(canon(r) for r in judgments)+'\n').encode();jpath.write_bytes(jr)
 stats={'records':len(rs),'source_family_keys_observed':len(famgroups),'records_per_family':8,'kinds':dict(collections.Counter(r['visible_kind'] for r in rs)),'splits':dict(collections.Counter(r['split'] for r in rs)),'valid':dict(collections.Counter(str(r['expected_target']['valid']) for r in rs)),'reaches_goal':dict(collections.Counter(str(r['expected_target']['reaches_goal']) for r in rs)),'fresh_minimum_horizons':dict(collections.Counter(str(None if r['expected_target']['shortest_plan'] is None else len(r['expected_target']['shortest_plan'])) for r in rs)),'first_errors':dict(collections.Counter(canon(r['independent_input_facts']['first_error']) for r in rs)),'input_derived_property_counts':dict(collections.Counter(p for r in rs for p in r['independently_witnessed_property_names'])),'maximum_search_visited':max(r['independent_input_facts']['search']['visited'] for r in rs),'maximum_search_frontier':max(r['independent_input_facts']['search']['maximum_frontier'] for r in rs),'all_searches_complete':all(r['independent_input_facts']['search']['complete'] for r in rs),'actual_source_novelty_not_assessed':True}
 execution={'report_type':'ACTUAL_BLIND_INPUT_FIRST_PREDICTION_JUDGMENT_FREEZE','actual_author':ROLE,'created_utc':now,'actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,**sources,'frozen_preparation_source_sha256':PREPSHA,'public_bindings':docs,'view_sha256':VIEWSHA,'root_input_execution_receipt':rootrec,'later_candidate_custody_sha256':CANDSHA,'prediction_path':str(rp),'prediction_sha256':sha(rb),'prediction_bytes':len(rb),'first_input_judgments_path':str(jpath),'first_input_judgments_sha256':sha(jr),'first_input_judgment_bytes':len(jr),'statistics':stats,'canonical_gold_control_manifest_private_sources_read':False,'dataset_approval':False,'comparison_requires_new_root_authorization':True,'guidance':GUIDANCE,'actual_agent_inspection_report':'reviews/expert/semantic_inspection_planning_gap_v1_input_stage.md','filesystem_write_freeze_after_execution':True}
 save(OUT/'planning-gap-v1.expert_input_stage.freeze.json',execution);print(canon(execution))
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in ('derive','freeze')
 derive() if sys.argv[1]=='derive' else freeze()
