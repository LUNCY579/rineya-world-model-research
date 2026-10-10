#!/opt/anaconda3/bin/python3
"""Separate post-dataset-freeze code tests; never imports private implementations.
Extracts only the authorized wrapper/selected pipeline AST functions into an
isolated namespace, with explicit own synthetic checker/validator stubs.
These tests assess integration control flow, not the unread checker semantics.
"""
import ast,collections,copy,datetime,hashlib,json,pathlib,sys,tempfile,types
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert'
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
bindings={'factory/planning_controls.py':'ff9d904be78b428d0cd60e53c7e026318a6fd7ed62932487e2a2d22a1f8c5566','factory/pipeline.py':'703df4685c0e7e598d6bd4debc2932dd4fd04c84d50f61622ac375e3783d14dd'}
sources={}
for name,h in bindings.items():
 b=(ROOT/name).read_bytes();assert sha(b)==h;sources[name]=b.decode()
own=pathlib.Path(__file__).read_bytes();arch=OUT/'planning-gap-v1.expert_gate_code.preexecution_source.py';assert arch.read_bytes()==own
freeze=OUT/'planning-gap-v1.expert_comparison.freeze.json';fr=json.loads(freeze.read_bytes());assert fr['first_receipts']['sha256']=='9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72' and fr['decisions']=={'ACCEPT':384}
rows=[json.loads(l) for l in (ROOT/'data/RGM-001/candidate_renderings/planning-gap-v1.jsonl').read_bytes().splitlines()]
tests=[]
def record(name,passed,evidence):tests.append({'name':name,'passed':bool(passed),'evidence':evidence})
with tempfile.TemporaryDirectory(prefix='expert-planning-gate-') as td:
 r=pathlib.Path(td);checker_path=r/'own_stub_checker.py';checker_path.write_text('# Own checker control-flow stub; no oracle code.\n')
 checker=types.SimpleNamespace(__file__=str(checker_path),SEMANTICS_SHA256='OWN_FIXTURE',ENCODING_SUPPLEMENT_SHA256='OWN_FIXTURE',PROPERTIES={'legal_prefix_then_first_failure','fuel_lengthens_fresh_minimum_moves','fuel_blocks_topologically_reachable_goal','valid_goal_detour','later_action_requires_changed_block_clearance','partial_block_goal','required_temporary_goal_displacement'})
 ns={'checker':checker}
 tree=ast.parse(sources['factory/planning_controls.py']);tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='oracles')];exec(compile(tree,'factory/planning_controls.py AST (isolated own stub)','exec'),ns)
 sel=ns['planning_control_rows'];rep=ns['planning_control_report']
 record('actual384 selected under exact batch',len(sel(rows,'planning-gap-v1'))==384,{'selected':len(sel(rows,'planning-gap-v1'))})
 record('actual384 version route selected under other batch',len(sel(rows,'OWN_OTHER_BATCH'))==384,{'selected':len(sel(rows,'OWN_OTHER_BATCH'))})
 tagless=next(row for row in rows if row['controlled_context']=='A0' and set(row['skill_tags'])=={'finite_synthetic_proxy','relational_planning'})
 renamed=copy.deepcopy(tagless);renamed['generator_version']='rgm-1.4.0'
 record('published context/stratum routing counterexample observed',sel([renamed],'planning-gap-next')==[],{'sample_id':tagless['sample_id'],'original_record_hash':tagless['record_hash'],'changes':'Only own hypothetical metadata version label and invocation batch changed; world/prompt/context/family preserved. No actual candidate edit or new dataset approval.','controlled_context':renamed['controlled_context'],'primary_stratum':renamed['primary_stratum'],'selected_count':len(sel([renamed],'planning-gap-next')),'scope':'Local selection function skips preserved public planning context markers under a nonmatching version/batch name. Does not bypass existing exact-hash reviewer joins; actual frozen v1 remains selected.'})
 # Fixture bindings substitute own tiny source/prior/docs, because the actual
 # private checker implementation is deliberately unread and unimported.
 ns['CHECKER_SHA256']=sha(checker_path.read_bytes());ns['PUBLIC_BINDINGS']={'own_public_spec.txt':sha(b'public fixture\n')};(r/'own_public_spec.txt').write_bytes(b'public fixture\n')
 prior=r/ns['PRIOR_PATH'];prior.parent.mkdir(parents=True);prior.write_bytes(b'[]\n');ns['PRIOR_SHA256']=sha(prior.read_bytes())
 mp=r/'data/RGM-001/manifests/planning-gap-v1.family_controls.json';mp.parent.mkdir(parents=True);mp.write_text('{}\n')
 calls=[]
 def check_batch(selected,manifest,prior_inputs):calls.append(len(selected));return {'errors':[],'counts':{'rows':len(selected),'complete_families':len(selected)//8}}
 checker.check_batch=check_batch
 report=rep(rows,'planning-gap-v1',r);record('fixture complete report invokes checker',not report['errors'] and calls==[384],{'counts':report['counts'],'evidence_entries':len(report['evidence_sha256']),'stub_checker_semantics_not_validated':True})
 mp.unlink();report=rep(rows,'planning-gap-v1',r);record('missing manifest fail closed',bool(report['errors']) and calls==[384],{'errors':report['errors']});mp.write_text('{BAD\n');report=rep(rows,'planning-gap-v1',r);record('malformed manifest fail closed',bool(report['errors']) and calls==[384],{'errors':report['errors']});mp.write_text('{}\n')
 prior.write_bytes(b'{"changed":true}\n');report=rep(rows,'planning-gap-v1',r);record('changed prior hash fail closed',any('binding mismatch' in e for e in report['errors']),{'errors':report['errors']});prior.write_bytes(b'[]\n')
 (r/'own_public_spec.txt').write_bytes(b'changed\n');report=rep(rows,'planning-gap-v1',r);record('changed public spec hash fail closed',any('source binding mismatch' in e for e in report['errors']),{'errors':report['errors']});(r/'own_public_spec.txt').write_bytes(b'public fixture\n')
 checker_path.write_bytes(b'changed checker\n');report=rep(rows,'planning-gap-v1',r);record('changed checker source hash fail closed',any('source binding mismatch' in e for e in report['errors']),{'errors':report['errors']});checker_path.write_bytes(b'# Own checker control-flow stub; no oracle code.\n')
 def changing_checker(selected,manifest,prior_inputs):mp.write_text('{"changed_during_check":true}\n');return {'errors':[],'counts':{'rows':len(selected)}}
 checker.check_batch=changing_checker;report=rep(rows,'planning-gap-v1',r);record('evidence mutation during invocation fail closed',any('changed during audit' in e for e in report['errors']),{'errors':report['errors']});mp.write_text('{}\n')
 def subset_checker(selected,manifest,prior_inputs):return {'errors':[] if len(selected)==384 else ['own fixture: incomplete family controls'],'counts':{'rows':len(selected)}}
 checker.check_batch=subset_checker;report=rep(rows[:-1],'planning-gap-v1',r);record('subset still routed to family checker',bool(report['errors']),{'rows':383,'errors':report['errors']})
 # Isolated pipeline approval function: no production imports or filesystem writes.
 pt=ast.parse(sources['factory/pipeline.py']);fun=next(n for n in pt.body if isinstance(n,ast.FunctionDef) and n.name=='approval_evidence');pn={};exec(compile(ast.Module(body=[fun],type_ignores=[]),'isolated approval_evidence','exec'),pn)
 row={'sample_id':'own-toy','record_hash':'OWN_HASH'}
 def evidence(identities):return {role:{'own-toy':[{'sample_id':'own-toy','record_hash':'OWN_HASH','decision':'ACCEPT','reviewer_execution_id':identity,'critical_defects':[],'issues':[]}]} for role,identity in zip(['expert','adversarial','adjudication'],identities)}
 ids=['/root/expert_reviewer','/root/adversarial_auditor','/root/research_architect'];approved,errs=pn['approval_evidence'](row,evidence(ids));record('exact intended identities accepted locally',not errs,{'identities':ids,'errors':errs})
 ev=evidence(ids);ev['expert']['own-toy'][0]['record_hash']='STALE';_,errs=pn['approval_evidence'](row,ev);record('stale candidate receipt rejected',errs==['expert'],{'errors':errs})
 ev=evidence(ids);ev['expert']['own-toy'][0]['issues']=[{'severity':'high'}];_,errs=pn['approval_evidence'](row,ev);record('high unresolved issue rejected','unresolved_serious_issue' in errs,{'errors':errs})
 _,errs=pn['approval_evidence'](row,evidence(['/root',ids[1],ids[2]]));record('root self-approval rejected','nonindependent_approval' in errs,{'errors':errs})
 altids=['/root/other_expert','/root/other_red','/root/other_adjudicator'];_,errs=pn['approval_evidence'](row,evidence(altids));record('release/validation identity inconsistency counterexample observed',not errs,{'local_approval_errors':errs,'identities':altids,'later_validate_release_required_identities':ids,'scope':'Actual release approval function admits any three distinct nonblacklisted IDs; later validate_release requires exact three fixed IDs. Hypothetical exact-hash toy receipts only, not existing actual approvals.'})
 # Isolate the prompt validator with an explicit no-op schema validator stub;
 # demonstrates only its own prompt checks, never full-pipeline oracle behavior.
 fun=next(n for n in pt.body if isinstance(n,ast.FunctionDef) and n.name=='schema_validate');fun=copy.deepcopy(fun);fun.body=[n for n in fun.body if not isinstance(n,ast.ImportFrom)];sn={'ROOT':r,'json':json,'Draft202012Validator':lambda s:types.SimpleNamespace(validate=lambda row:None),'REQUIRED':set(),'record_hash':lambda row:row['record_hash'],'canonical':canon};sp=r/'schemas/canonical/v1.schema.json';sp.parent.mkdir(parents=True);sp.write_text('{}\n');exec(compile(ast.Module(body=[fun],type_ignores=[]),'isolated schema_validate with explicit own schema stub','exec'),sn)
 probe=copy.deepcopy(tagless);probe['rendered_input']='sample_id: toy hidden answer verified_target: TRUE\n'+probe['rendered_input'];sn['schema_validate'](probe);record('contract-lead metadata leak local validator counterexample observed',True,{'scope':'schema_validate parses/checks only Task JSON payload and accepts privileged metadata in pre-Task contract lead under own no-op canonical schema stub; unread downstream checker might add checks. No claim of full pipeline bypass or actual candidate leak.','student_lead_added':'sample_id: toy hidden answer verified_target: TRUE'})
assert all(t['passed'] for t in tests)
report={'actual_author':'/root/expert_reviewer','review_stage':'POST_DATASET_FREEZE_ISOLATED_CODE_REVIEW_TEST_EXECUTION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'artifact_hashes':bindings,'own_test_source_sha256':sha(own),'preexecution_archive_sha256':sha(arch.read_bytes()),'dataset_first_final_sha256':fr['first_receipts']['sha256'],'dataset_freeze_report_sha256':sha(freeze.read_bytes()),'actual_tests':tests,'test_count':len(tests),'private_checker_reference_generator_or_other_verdict_read':False,'production_pipeline_imported_or_run':False,'no_release_pipeline_status_or_candidate_writes':True,'scope':'Read authorized two root integration sources only after first dataset verdict freeze. Exact wrapper AST under own tiny fixture bindings/stub checker; selected pipeline AST functions under own explicit stubs. Does not inspect/approve private checker correctness or run publication. Counterexamples are hypothetical local-function robustness findings, not defects in current canonical rows or approvals.'}
op=OUT/'planning-gap-v1.expert_gate_code.execution.json';assert not op.exists();op.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(canon({'test_count':len(tests),'all_test_expectations_observed':True,'source_sha256':report['own_test_source_sha256'],'output_sha256':sha(op.read_bytes()),'output_bytes':op.stat().st_size}))
