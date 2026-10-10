#!/usr/bin/env python3
"""Guided actual-function publication review; actual /root/adversarial_auditor.
Manual semantic targets and expectations freeze before production function calls.
No audit/release, reference solver or future v2 checker/producer invocation.
"""
import collections,copy,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
BINDINGS={'factory/pipeline.py':'fc68a08d276abeb5bf4c73727fa9524d18798b1861ade2f701e89f23d4a7fe74',
          'factory/planning_controls.py':'be85989ef8480c01056a55f1329de80b47d2eec2d1496852218890d430165668',
          'factory/common.py':'8502429a270c0a2b0ee74ce89118b28c780b7242c58d8d9c6e858ebcd43ef7c8',
          'schemas/canonical/v1.schema.json':'f4d8276fe000ceafb0f70c69dbfd1608b823767487d7c0a79623f82cf0daf25f',
          'tests/test_planning_review_corrections.py':'cd86eee2c8e764249a764f014a421f7c5dbe6d1aa9d042fca4ae5fc12a34dff2',
          'schemas/SEMANTICS.md':'718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd',
          'quality_reports/RGM_V1_STUDENT_BOUNDARY_REGRESSION_EXECUTION.json':'e3d8de861f42c3b201e9f2e74df02f24aa0937e4d5c896234e2f4ee48212e979'}
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
def jb(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def own_hash(r):return hashlib.sha256(canon({k:v for k,v in r.items() if k!='record_hash'}).encode()).hexdigest()
def row(kind='graph',names=None):
    if kind=='graph':
        a,g=names or ['north','harbor'];t={'kind':'graph','nodes':[a,g],'edges':[[a,g,1]],'start':a,'goal':g,'fuel':1,'plan':[g]};end={'node':g,'fuel':0};target={'steps':[{'legal':True,'after':end}],'after':end,'valid':True,'reaches_goal':True,'shortest_plan':[g]}
        contract='Every directed move needs its edge and enough fuel; subtract cost. Stop at first illegal move and retain state. Fresh initial minimum plan minimizes moves then full destination sequence lexicographically.'
    else:
        a,b=names or ['A','B'];t={'kind':'blocks','on':{a:'TABLE',b:'TABLE'},'goal_on':{a:b},'plan':[{'block':a,'to':b}]};end={a:b,b:'TABLE'};target={'steps':[{'legal':True,'after':end}],'after':end,'valid':True,'reaches_goal':True,'shortest_plan':[{'block':a,'to':b}]}
        contract='A move needs a clear source and TABLE or a distinct clear destination, changing its support. Stop at first illegal action. Goal map is partial. Fresh initial minimum plan minimizes moves then the full (block,to) sequence lexicographically.'
    r={'sample_id':'RED-PUBLICATION-HAND-'+kind,'semantic_parent_id':'RED-PARENT-'+kind,'semantic_family_id':'RED-FAMILY-'+kind,'track_id':'RGM-001',
       'skill_tags':[],'difficulty':2,'generator_version':'manual-red-calibration','oracle_version':'hand-derived-no-oracle','generation_seed':0,
       'source_provenance':{'origin':'reviewer_hand_fixture','license':'original_synthetic','sealed_evaluation_used':False},'semantic_task':t,
       'rendered_input':contract+'\nTask JSON:\n'+canon(t),'verified_target':target,'target_format':'structured','split':'train','review_status':'PENDING','review_receipt_ids':[]}
    r['record_hash']=own_hash(r);return r

def fixtures():
    tests=[]
    def add(name,r,expect,group):tests.append({'name':name,'row':r,'expected_schema_accept':expect,'group':group})
    for kind in ('graph','blocks'):
        base=row(kind);add('baseline_'+kind,base,True,'legitimate')
        for n,prose in enumerate(['Consider difficulty as routes split into alternatives.','The words difficulty and split have ordinary meanings here.']):
            r=copy.deepcopy(base);r['rendered_input']=prose+'\n'+r['rendered_input'];r['record_hash']=own_hash(r);add(f'ordinary_prose_{kind}_{n}',r,True,'ordinary_prose')
    # Canonical/private interface names, plus optional fields actually present in
    # the previously reviewed frozen v1 rows. No target-derived expectation.
    private=['sample_id','semantic_parent_id','semantic_family_id','track_id','skill_tags','difficulty','generator_version','oracle_version','generation_seed',
             'source_provenance','target_format','split','review_status','review_receipt_ids','record_hash','verified_target',
             'source_family_key','source_index','controlled_context','controlled_variant','primary_stratum','property_witnesses','family_control_claims',
             'structural_family_profile','fact_role_assignments','query_role_assignments','auxiliary_scalar_variant','auxiliary_scalar_control','reviewed_candidate_hash','renderer',
             'producer_agent_id','producer_role','source_family_index','structural_difficulty']
    styles={'bare_colon':lambda k:k+': PRIVATE_SENTINEL','upper_equals':lambda k:k.upper()+' = PRIVATE_SENTINEL',
            'quoted_json':lambda k:'"'+k+'": "PRIVATE_SENTINEL"','bold':lambda k:'**'+k+'**: PRIVATE_SENTINEL',
            'inline_code':lambda k:'`'+k+'` = PRIVATE_SENTINEL','single_quote':lambda k:"'"+k+"': PRIVATE_SENTINEL",
            'mixed_case_bold':lambda k:'**'+k.title()+'** = PRIVATE_SENTINEL'}
    for k in private:
        for style,render in styles.items():
            for pos in ('beginning','middle'):
                r=row();label=render(k)
                if pos=='beginning':r['rendered_input']=label+'\n'+r['rendered_input']
                else:r['rendered_input']=r['rendered_input'].replace('\nTask JSON:\n','\n'+label+'\nTask JSON:\n')
                r['record_hash']=own_hash(r);add(f'private:{k}:{style}:{pos}',r,False,'private_label')
    for k in ('sample_id','semantic_parent_id','record_hash','review_receipt_ids','verified_target'):
        r=row();r['rendered_input']='Internal label without a delimiter '+k+'\n'+r['rendered_input'];r['record_hash']=own_hash(r);add('legacy_undelimited:'+k,r,False,'legacy_strings')
    for kind in ('graph','blocks'):
        for words in [('difficulty','split'),('sample_id','ordinary')]:add('public_names:'+kind+':'+','.join(words),row(kind,list(words)),True,'public_names')
    # Schema, hash and read-back must really run; corruptions aren't hidden by a stale hash.
    r=row();r['difficulty']=True;r['record_hash']=own_hash(r);add('typed_metadata_bool_difficulty',r,False,'typed_schema')
    r=row();r['split']='unknown';r['record_hash']=own_hash(r);add('typed_metadata_unknown_split',r,False,'typed_schema')
    r=row();r['record_hash']='0'*64;add('stale_record_hash',r,False,'real_hash')
    r=row();payload=copy.deepcopy(r['semantic_task']);payload['fuel']=True;r['rendered_input']=r['rendered_input'].split('Task JSON:\n')[0]+'Task JSON:\n'+canon(payload);r['record_hash']=own_hash(r);add('typed_payload_bool_vs_integer',r,False,'typed_readback')
    # Fresh boundary probes extend, rather than replace, all495 prior fixtures.
    extras=['red_private_stamp','RedAuditHint','red.control[stamp]']
    for kind in ('graph','blocks'):
        for key in extras:
            base=row(kind);base[key]={'stamp':[1,True,None]};base['record_hash']=own_hash(base);add('dynamic_unexposed:'+kind+':'+key,base,True,'dynamic_unexposed')
            for style in ('bare_colon','quoted_json','bold','inline_code','upper_equals'):
                r=copy.deepcopy(base);r['rendered_input']=styles[style](key)+'\n'+r['rendered_input'];r['record_hash']=own_hash(r);add('dynamic_lead:'+kind+':'+key+':'+style,r,False,'dynamic_private_lead')
            r=copy.deepcopy(base);r['semantic_task'][key]=base[key];r['rendered_input']=r['rendered_input'].split('Task JSON:\n')[0]+'Task JSON:\n'+canon(r['semantic_task']);r['record_hash']=own_hash(r);add('dynamic_outer:'+kind+':'+key,r,False,'dynamic_private_outer')
        for key in private:
            r=row(kind);r['semantic_task'][key]='PRIVATE_SENTINEL';r['rendered_input']=r['rendered_input'].split('Task JSON:\n')[0]+'Task JSON:\n'+canon(r['semantic_task']);r['record_hash']=own_hash(r);add('known_outer:'+kind+':'+key,r,False,'known_private_outer')
            add('public_payload_names:'+kind+':'+key,row(kind,[key,'public_peer']),True,'public_names_extended')
        for key in ('source_family_index','producer_role','difficulty'):
            for alias in (key.upper(),key.title()):
                r=row(kind);r[key]=180 if key=='source_family_index' else 'PRIVATE_SENTINEL' if key=='producer_role' else 2
                r['semantic_task'][alias]=r[key];r['rendered_input']=r['rendered_input'].split('Task JSON:\n')[0]+'Task JSON:\n'+canon(r['semantic_task']);r['record_hash']=own_hash(r);add('case_outer:'+kind+':'+alias,r,False,'case_private_outer')
        r=row(kind);r['rendered_input']+='\nproducer_role: PRIVATE_SENTINEL';r['record_hash']=own_hash(r);add('trailing_metadata:'+kind,r,False,'trailing_payload')
    return tests

def main():
    for path,h in BINDINGS.items():assert sha(ROOT/path)==h,path
    source=pathlib.Path(__file__);source_hash=sha(source);freeze(ROOT/f'audits/history/planning_publication_code_adversarial_v2-{source_hash}.py',source.read_bytes())
    assert sha(ROOT/'audits/planning_publication_code_adversarial_v1.py')=='08f954e018822a7684b22640558c48d72b472bec716325a99a4e7508704c1213'
    assert sha(ROOT/'reviews/adversarial/planning-publication-code-v1.json')=='68a18f0891ff5dda611e43d4d192769f5dd5f85fb82ce3560a6192508dd0b04e'
    tests=fixtures();expected_roles={'expert':'/root/expert_reviewer','adversarial':'/root/adversarial_auditor','adjudication':'/root/research_architect'}
    pre={'actual_author_execution':EXEC,'stage':'PREEXEC_EXPECTATION_FREEZE','source_sha256':source_hash,'bound_sources':BINDINGS,
         'manual_fixture_target_count':2,'schema_tests':tests,'role_expectations':expected_roles,
         'expectation_scope':'Known explicit private labels are excluded; normal prose and public string names remain valid under SEMANTICS. Schema/hash/read-back failures reject. Routing persistence and exact actor roles tested separately.',
         'review_mode':'Guided corrective review, reusing all495 own frozen v1 fixtures plus new boundary cases; root code/tests/report guidance read, no expert corrective verdict. Original REVISE_CODE unchanged; no dataset approval.'}
    prepath=ROOT/'audits/execution/planning-publication-code-adversarial-v2.preexec.json';freeze(prepath,jb(pre))
    print(canon({'preexec_sha256':sha(prepath),'source_sha256':source_hash,'schema_test_count':len(tests)}),flush=True)
    # Direct actual production functions, with their real JSON Schema and real
    # canonical hash helper. No mocks/stubs, no audit/release/solver call.
    sys.path.insert(0,str(ROOT))
    from factory.pipeline import schema_validate,approval_evidence,APPROVAL_ACTORS
    from factory.common import record_hash
    from factory.planning_controls import planning_control_rows
    assert APPROVAL_ACTORS==expected_roles
    results=[];unexpected=[];started=time.perf_counter()
    for t in tests:
        r=t['row'];exception=None
        try:schema_validate(r);accepted=True
        except Exception as e:accepted=False;exception={'type':type(e).__name__,'message':str(e).splitlines()[0]}
        out={'name':t['name'],'group':t['group'],'expected_accept':t['expected_schema_accept'],'actual_accept':accepted,'passed':accepted==t['expected_schema_accept'],'exception':exception,'record_hash_matches_real_helper':r['record_hash']==record_hash(r)}
        results.append(out)
        if not out['passed']:unexpected.append({'result':out,'actual_hand_row':r})
    # Real role join helper. Each corruption starts from a complete valid join.
    base=row();evidence={role:{base['sample_id']:[{'receipt_id':'RED-HAND-'+role,'record_hash':base['record_hash'],'decision':'ACCEPT','reviewer_execution_id':actor,'critical_defects':[],'issues':[]}]} for role,actor in expected_roles.items()}
    role_cases=[('intended',evidence,True)]
    for role in expected_roles:
        e=copy.deepcopy(evidence);e[role][base['sample_id']][0]['reviewer_execution_id']='/root/unrecognized_'+role;role_cases.append(('unknown_'+role,e,False))
    e=copy.deepcopy(evidence);e['expert'][base['sample_id']][0]['reviewer_execution_id']=expected_roles['adversarial'];e['adversarial'][base['sample_id']][0]['reviewer_execution_id']=expected_roles['expert'];role_cases.append(('swapped_expert_red',e,False))
    e=copy.deepcopy(evidence);e['adjudication'][base['sample_id']][0]['reviewer_execution_id']=expected_roles['expert'];role_cases.append(('duplicate_actor',e,False))
    for role in expected_roles:
        e=copy.deepcopy(evidence);e.pop(role);role_cases.append(('missing_role_'+role,e,False))
    e=copy.deepcopy(evidence);e['observer']={base['sample_id']:[copy.deepcopy(e['expert'][base['sample_id']][0])]};role_cases.append(('extra_role',e,False))
    e=copy.deepcopy(evidence);e['expert'][base['sample_id']][0]['record_hash']='0'*64;role_cases.append(('stale_hash',e,False))
    e=copy.deepcopy(evidence);e['expert'][base['sample_id']][0]['decision']='REVISE';role_cases.append(('revise_decision',e,False))
    e=copy.deepcopy(evidence);e['expert'][base['sample_id']][0]['issues']=[{'severity':'high'}];role_cases.append(('high_issue',e,False))
    role_results=[]
    for name,e,expect in role_cases:
        receipts,fail=approval_evidence(base,e);out={'name':name,'expected_accept':expect,'actual_accept':not fail,'failure_codes':fail,'passed':bool(not fail)==expect};role_results.append(out)
        if not out['passed']:unexpected.append({'role_result':out,'actual_evidence':e})
    # Keep each actual public planning marker independently, while other labels are renamed.
    route_cases=[('generic_legacy',row(),'RENAMED_BATCH',False),('planning_batch',row(),'planning-gap-v91',True)]
    for context in [s+str(i) for s in 'ABCD' for i in range(4)]+['C4']:
        r=row();r['controlled_context']=context;route_cases.append(('context_'+context,r,'RENAMED_BATCH',True))
    for s in 'ABCD':r=row();r['primary_stratum']=s;route_cases.append(('stratum_'+s,r,'RENAMED_BATCH',True))
    for tag in ['legal_prefix_then_first_failure','fuel_lengthens_fresh_minimum_moves','fuel_blocks_topologically_reachable_goal','valid_goal_detour','later_action_requires_changed_block_clearance','partial_block_goal','required_temporary_goal_displacement']:
        r=row();r['skill_tags']=['scope:'+tag];route_cases.append(('compound_tag_'+tag,r,'RENAMED_BATCH',True))
    r=row();r['generator_version']='rgm-planning-gap-renamed';route_cases.append(('version_marker',r,'RENAMED_BATCH',True))
    r=row();r['track_id']='RCM-001';r['controlled_context']='A0';route_cases.append(('non_RGM_excluded',r,'planning-gap-v91',False))
    routing=[]
    for name,r,batch,expect in route_cases:
        selected=planning_control_rows([r],batch);out={'name':name,'expected_selected':expect,'actual_selected':bool(selected),'passed':bool(selected)==expect};routing.append(out)
        if not out['passed']:unexpected.append({'routing_result':out,'actual_row':r,'batch':batch})
    for path,h in BINDINGS.items():assert sha(ROOT/path)==h,'reviewed bytes changed during execution'
    counts={'schema':len(results),'roles':len(role_results),'routing':len(routing),'unexpected':len(unexpected),
            'schema_groups':dict(collections.Counter(r['group'] for r in results)),'unexpected_groups':dict(collections.Counter(e.get('result',{}).get('group','role_or_routing') for e in unexpected))}
    execution={'actual_author_execution':EXEC,'completed':True,'review_mode':'GUIDED_ACTUAL_FUNCTION_CODE_PRIVACY_REVIEW','source_sha256':source_hash,'preexec_sha256':sha(prepath),
               'bound_sources_before_and_after':BINDINGS,'actual_functions':['factory.pipeline.schema_validate','factory.pipeline.approval_evidence','factory.planning_controls.planning_control_rows','factory.common.record_hash'],
               'real_json_schema_loaded':True,'no_mock_or_stub_function_closure':True,'manual_targets_no_oracle_call':True,'duration_seconds':time.perf_counter()-started,
               'counts':counts,'schema_results':results,'role_results':role_results,'routing_results':routing,'actual_unexpected_cases':unexpected,
               'import_scope':'Actual production modules transitively load their existing dependencies; no reference/planning/other target solver, audit or release function invoked. No prospective v2 private source/input read.',
               'limits':['Guard inspection is bounded to explicit known field labels/styles and legacy strings; arbitrary encoded metadata and all publication paths are not certified.',
                         'Canonical schema allows additional metadata fields; privacy membership must cover actual internal fields rather than assume REQUIRED is exhaustive.',
                         'Root report/tests read as guided context; no expert corrective verdict read. Same underlying model/common-public-design correlation remains.']}
    ex=ROOT/'audits/execution/planning-publication-code-adversarial-v2.json';freeze(ex,jb(execution))
    missed=sorted({e['result']['name'].split(':')[1] for e in unexpected if e.get('result',{}).get('group')=='private_label'});falsepublic=[e['result']['name'] for e in unexpected if e.get('result',{}).get('group')=='public_names']
    case_bypasses=[e for e in unexpected if e.get('result',{}).get('group')=='case_private_outer']
    receipt={'receipt_id':'RED-PLANNING-PUBLICATION-CODE-'+source_hash[:20],'reviewer_execution_id':EXEC,'reviewer_role':'adversarial_code_privacy','decision':'REVISE_CODE' if unexpected else 'ACCEPT_CODE_WITH_LIMITS',
             'source_sha256':source_hash,'code_sha256':{k:v for k,v in BINDINGS.items() if k.startswith('factory/')},'schema_sha256':BINDINGS['schemas/canonical/v1.schema.json'],'preexec_sha256':sha(prepath),'execution_sha256':sha(ex),
             'findings':[{'severity':'high','code':'recognized_internal_metadata_labels_missing_from_guard','actual_fields':missed,'evidence':'Independently authored valid hand rows were rehashed with explicit private labels in full contract text. Actual JSON Schema/hash/read-back passed and privacy guard accepted these labels. Fields are optional metadata observed in frozen v1 canonical records, not arbitrary encoded aliases.','fix':'Explicitly cover these actual private canonical metadata fields and rerun exact actual-function regressions; preserve current code receipt.'}] if missed else [],
             'scope_findings':[{'severity':'low','code':'public_name_false_positive','actual_cases':falsepublic,'evidence':'SEMANTICS permits distinct block/node strings beyond reserved TABLE. The full-prompt text guard can reject legitimate payload names that coincide with private labels.','fix':'Keep public payload semantics distinct from actual metadata labels or state an explicit prospective naming boundary; measure rather than infer broad validity.'}] if falsepublic else [],
             'roles_and_routing_correct':all(r['passed'] for r in role_results+routing),'no_original_data_review_rewrites':True,'dataset_approval':False,'release_approval':False,
             'review_mode':'Guided root-requested code review, independent actual fixtures/execution, after frozen original384 data reviews; not a new blind data review.',
             'confidence':0.99,'publication_scope':'Only these exact functions/code bytes and fixture controls assessed. No approval of entire pipeline or prospective v2.'}
    if case_bypasses:receipt['findings'].append({'severity':'high','code':'case_variant_private_outer_key_bypass','evidence':'Actual schema/hash/read-back accepts case variants of explicit known private outer task keys while lead labels are case-insensitive. Full rows and exact mutations preserved in execution evidence.','cases':[e['result']['name'] for e in case_bypasses],'fix':'Align recognized private outer-key normalization with case-insensitive label detection, preserving public names inside typed payload maps; freeze a fresh correction.'})
    receipt['corrective_reuse_disclosure']='All495 original fixtures,13 actor checks and32 routing checks replayed. Root guided corrected bytes. Original v1 REVISE_CODE retained.'
    receipt['previous_code_receipt_sha256']='68a18f0891ff5dda611e43d4d192769f5dd5f85fb82ce3560a6192508dd0b04e'
    first=ROOT/'reviews/adversarial/planning-publication-code-v2.first.json';final=ROOT/'reviews/adversarial/planning-publication-code-v2.json';freeze(first,jb(receipt));freeze(final,jb(receipt))
    print(canon({'decision':receipt['decision'],'code_review_sha256':sha(final),'first_sha256':sha(first),'execution_sha256':sha(ex),'source_sha256':source_hash,'counts':counts,'missed_private_fields':missed,'public_name_false_positives':falsepublic}),flush=True)
if __name__=='__main__':main()
