#!/opt/anaconda3/bin/python3
"""NEW guided corrective gate review, actual functions and real canonical schema.
No AST substitution, mocks, schema/checker stubs or ROOT override. Own synthetic
hand-target/hash fixtures exist in memory only, never in the candidate corpus.
"""
import collections,copy,datetime,hashlib,json,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
BIND={
 'factory/pipeline.py':'4767d4bb51b84ab8df22852a966c9d5293e0c59f9cfd6c86e1c3d89efdca3f21',
 'factory/planning_controls.py':'be85989ef8480c01056a55f1329de80b47d2eec2d1496852218890d430165668',
 'tests/test_planning_review_corrections.py':'f24edf6f68366c127b24a20e12f71d5935eca3ecdaa41568bddfadd5079bbb5d',
 'tests/test_approval_gate.py':'0f764bf25b47b20a4029e08f0611462cdee07086b6c4bb562754e920d3aa052e',
 'schemas/canonical/v1.schema.json':'f4d8276fe000ceafb0f70c69dbfd1608b823767487d7c0a79623f82cf0daf25f',
 'quality_reports/RGM_V1_GATE_REVIEW_REGRESSION_EXECUTION.json':'75f3c2f7745dc311becb7c63217a0580785dad247f9f1817d738fd48fb19e3e6',
 'reviews/expert/planning-gap-v1.expert_gate_CODE_REVIEW.initial.json':'6b79c17682f85b829b0769a002263f5c6dca4b183a0361c846a5e851f843fb5b',
 'reviews/expert/planning-gap-v1.expert_gate_CODE_REVIEW.guided.json':'5530f9e3a8147b033de04cc1cd67502e7db67021ef438f736de86596bb4f16b7',
 'reviews/expert/planning-gap-v1.expert_final_outputs.freeze.json':'5e9a89713cbf8e197ccd509dae0b380d2765cdd424091919f5f497007bd72a74',
 'reviews/expert/planning-gap-v1.first.jsonl':'9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72',
 'reviews/expert/planning-gap-v1.jsonl':'9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72',
 'oracles/rgm_planning_controls_independent.py':'cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3'}
def bindings():
 d={}
 for p,h in BIND.items():
  b=(ROOT/p).read_bytes();assert sha(b)==h,(p,sha(b));d[p]={'sha256':h,'bytes':len(b)}
 return d
own=pathlib.Path(__file__).read_bytes();arch=OUT/'planning-gap-v1.expert_gate_corrective.preexecution_source.py';assert arch.read_bytes()==own
before=bindings();start=time.perf_counter();tests=[];residual=[]
def put(name,passed,evidence):tests.append({'name':name,'passed':bool(passed),'evidence':evidence})
def main():
 from factory import pipeline as pl
 from factory import planning_controls as pc
 assert pathlib.Path(pl.__file__).resolve()==ROOT/'factory/pipeline.py' and pathlib.Path(pc.__file__).resolve()==ROOT/'factory/planning_controls.py'
 assert pl.planning_control_rows is pc.planning_control_rows
 task={'kind':'graph','nodes':['alpha','beta'],'edges':[['alpha','beta',1]],'start':'alpha','goal':'beta','fuel':2,'plan':['beta']}
 target={'steps':[{'legal':True,'after':{'node':'beta','fuel':1}}],'after':{'node':'beta','fuel':1},'valid':True,'reaches_goal':True,'shortest_plan':['beta']}
 row={'sample_id':'EXPERT-TEMP-CODE-GRAPH','semantic_parent_id':'EXPERT-TEMP-PARENT','semantic_family_id':'EXPERT-TEMP-PARENT','track_id':'RGM-001','skill_tags':['relational_planning'],'difficulty':2,'generator_version':'OWN_GUIDED_CODE_FIXTURE','oracle_version':'HAND_DERIVED_ONLY','generation_seed':20261010,'source_provenance':{'origin':'own_temporary_hand_fixture','sealed_evaluation_used':False},'semantic_task':task,'rendered_input':'Graph moves are directed and consume fuel; first illegal action stops with unchanged state. A fresh plan minimizes moves, full destination sequence breaks lexical ties.\nTask JSON:\n'+canon(task),'verified_target':target,'target_format':'structured','split':'train','review_status':'PENDING','review_receipt_ids':[],'record_hash':''}
 row['record_hash']=pl.record_hash(row);pl.schema_validate(row)
 put('actual real schema/hash baseline accepted',True,{'fixture_hash':row['record_hash'],'factory_file':str(pl.__file__),'schema_ROOT_unmodified':str(pl.ROOT)})
 put('ordinary unmarked RGM row retains old routing behavior',pc.planning_control_rows([row],'ORDINARY-BATCH')==[],{'ordinary_version':row['generator_version']})
 for st in 'ABCD':
  for i in range(4):
   x=copy.deepcopy(row);x['controlled_context']=st+str(i);x['record_hash']=pl.record_hash(x)
   put('actual context-only route '+st+str(i),pc.planning_control_rows([x],'RENAMED-BATCH')==[x],{'context':st+str(i),'nonprefix_version':x['generator_version'],'no_property_tag':True,'primary_stratum_absent':True})
 for st in 'ABCD':
  x=copy.deepcopy(row);x['primary_stratum']=st;x['record_hash']=pl.record_hash(x)
  put('actual stratum-only route '+st,pc.planning_control_rows([x],'RENAMED-BATCH')==[x],{'primary_stratum':st,'controlled_context_absent':True})
 for batch in ['planning-gap-v1','planning-gap-v2','planning-gap-v999']:
  put('actual batch-prefix route '+batch,pc.planning_control_rows([row],batch)==[row],{'batch':batch,'all_optional_control_metadata_absent':True})
 x=copy.deepcopy(row);x['generator_version']='rgm-planning-gap-1.0.0';put('existing version route preserved',pc.planning_control_rows([x],'RENAMED-BATCH')==[x],{})
 x=copy.deepcopy(row);x['skill_tags'].append('prefix:legal_prefix_then_first_failure:suffix');put('existing property component route preserved',pc.planning_control_rows([x],'RENAMED-BATCH')==[x],{'scope':'Selector evidence only; fixture property is not a corpus factual claim.'})
 x=copy.deepcopy(row);x['track_id']='RVM-001';x['controlled_context']='A0';put('other track excluded despite context marker',pc.planning_control_rows([x],'planning-gap-v2')==[],{})
 x=copy.deepcopy(row);x['controlled_context']='Z9';put('unknown context does not spuriously route ordinary batch',pc.planning_control_rows([x],'ORDINARY-BATCH')==[],{})
 for key in ['sample_id','semantic_parent_id','record_hash','review_receipt_ids','verified_target']:
  x=copy.deepcopy(row);x['rendered_input']=key+': '+canon(x[key])+'\n'+x['rendered_input'];x['record_hash']=pl.record_hash(x)
  error=None
  try:pl.schema_validate(x)
  except AssertionError as exc:error=str(exc)
  put('actual rehashed complete-lead guard '+key,error=='privileged metadata in student input',{'actual_assertion':error,'mutant_hash':x['record_hash'],'payload_task_unchanged':True,'schema_validator_oracle_not_stubbed':True})
 x=copy.deepcopy(row);x['rendered_input']='Return only evaluated steps; read the actual starting state.\n'+x['rendered_input'];x['record_hash']=pl.record_hash(x);pl.schema_validate(x);put('benign extra contract lead still accepted',True,{})
 x=copy.deepcopy(row);render_task=copy.deepcopy(task);render_task['fuel']=True;x['semantic_task']['fuel']=1;x['rendered_input']=row['rendered_input'].split('Task JSON:\n')[0]+'Task JSON:\n'+canon(render_task);x['record_hash']=pl.record_hash(x)
 error=None
 try:pl.schema_validate(x)
 except AssertionError as exc:error=str(exc)
 put('actual typed payload equality still rejects bool/integer confusion',error=='rendering semantic mismatch',{'actual_assertion':error})
 actors={'expert':'/root/expert_reviewer','adversarial':'/root/adversarial_auditor','adjudication':'/root/research_architect'};put('published actual role identities fixed',pl.APPROVAL_ACTORS==actors,{'actual_actor_map':pl.APPROVAL_ACTORS})
 def ev():return {role:{row['sample_id']:[{'receipt_id':'OWN-TOY-'+role,'sample_id':row['sample_id'],'record_hash':row['record_hash'],'decision':'ACCEPT','reviewer_execution_id':actor,'critical_defects':[],'issues':[]}]} for role,actor in actors.items()}
 put('actual correct role-specific receipts accepted',pl.approval_evidence(row,ev())[1]==[],{})
 for mode in ['unknown','swapped','duplicate','root','missing_role','extra_role','stale','latest_reject','high_issue']:
  e=ev()
  if mode=='unknown':
   for role in e:e[role][row['sample_id']][0]['reviewer_execution_id']='/root/other_'+role
  elif mode=='swapped':
   e['expert'][row['sample_id']][0]['reviewer_execution_id']=actors['adversarial'];e['adversarial'][row['sample_id']][0]['reviewer_execution_id']=actors['expert']
  elif mode=='duplicate':e['expert'][row['sample_id']][0]['reviewer_execution_id']=actors['adversarial']
  elif mode=='root':e['expert'][row['sample_id']][0]['reviewer_execution_id']='/root'
  elif mode=='missing_role':e.pop('adjudication')
  elif mode=='extra_role':e['extra']=copy.deepcopy(e['expert'])
  elif mode=='stale':e['expert'][row['sample_id']][0]['record_hash']='STALE'
  elif mode=='latest_reject':
   x=copy.deepcopy(e['expert'][row['sample_id']][0]);x['decision']='REVISE';e['expert'][row['sample_id']].append(x)
  elif mode=='high_issue':e['expert'][row['sample_id']][0]['issues']=[{'severity':'high'}]
  failure=pl.approval_evidence(row,e)[1];put('actual evidence rejection '+mode,bool(failure),{'actual_failures':failure,'temporary_fixture_not_corpus':True})
 # The correction covers the original five named fields. Public semantics also
 # excludes provenance/difficulty/splits/reviews: do not overclaim full closure.
 for key in ['semantic_family_id','source_provenance','difficulty','split','review_status']:
  x=copy.deepcopy(row);x['rendered_input']=key+': '+canon(x[key])+'\n'+x['rendered_input'];x['record_hash']=pl.record_hash(x);accepted=True;err=None
  try:pl.schema_validate(x)
  except Exception as exc:accepted=False;err=type(exc).__name__+': '+str(exc)
  residual.append({'key':key,'local_schema_accepts_rehashed_lead':accepted,'error':err,'mutant_hash':x['record_hash'],'scope':'Actual schema_validate function only. No publication invocation/downstream checker bypass claim and no actual corpus leak; these broader public internal metadata categories remain outside its five-string guard.'})
 command=[sys.executable,'-B','-m','unittest','tests.test_planning_review_corrections','tests.test_approval_gate','-v']
 result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=30)
 put('authorized root actual correction/approval tests independently rerun',result.returncode==0,{'actual_command':command,'actual_exit_code':result.returncode,'actual_stdout':result.stdout,'actual_stderr':result.stderr,'scope':'Only authorized two test modules, not root31/full320 suite.'})
 return {'actual_functions':{'pipeline_path':str(pl.__file__),'planning_controls_path':str(pc.__file__),'root_unmodified':str(pl.ROOT)},'original_three_local_findings_fixed':all(t['passed'] for t in tests),'actual_root_test_subset_exit_code':result.returncode,'temporary_hand_target':target,'new_fixture_canonical_hash':row['record_hash']}
failure=None;facts={}
try:facts=main()
except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
after=bindings();report={'actual_author':ROLE,'review_stage':'GUIDED_CORRECTIVE_CODE_ACTUAL_FUNCTION_EXECUTION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_argv':sys.argv,'python_executable':sys.executable,'own_source_sha256':sha(own),'preexecution_archive_sha256':sha(arch.read_bytes()),'source_bindings_before':before,'source_bindings_after':after,'bound_sources_unchanged':before==after,'elapsed_seconds':time.perf_counter()-start,'exit_code':0 if failure is None and all(t['passed'] for t in tests) else 1,'actual_failure':failure,'actual_tests':tests,'test_count':len(tests),'residual_metadata_observations':residual,**facts,'scope':'Actual unmodified factory function imports and real JSONSchema/canonical-hash helpers. No AST extraction, namespace/function replacement, mocks or stubs. Only synthetic own hand-derived graph/receipt/hash fixtures in memory. Existing Python dependency modules load normally; reference/generator/private V2 implementations are not inspected or invoked to solve data. No candidate/gold/V2 corpus read, full publication call, source/status/release edit or original expert artifact change.','guidance':'Root-guided corrective followup after own prior code findings, with shared underlying model. Root31-test report is disclosed external evidence; own independent regression invocation recorded separately. Full320 run was not awaited or claimed as own.'}
out=OUT/'planning-gap-v1.expert_gate_corrective.actual_execution.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(canon({'exit_code':report['exit_code'],'tests':len(tests),'all_expected_regressions_pass':all(t['passed'] for t in tests),'residual_metadata_observations':residual,'execution_sha256':sha(out.read_bytes()),'execution_bytes':out.stat().st_size,'source_sha256':sha(own)}));sys.exit(report['exit_code'])
