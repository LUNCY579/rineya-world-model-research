#!/usr/bin/env python3
"""Canonical-stage independent adversarial review, /root/adversarial_auditor.
Only frozen own predictions/public documents/candidate/manifest/prior public inputs.
No producer, new formal, expert, reference, root gate or other verdict imports.
"""
import collections,copy,hashlib,importlib.util,itertools,json,pathlib,re,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
FILES={'candidate':('data/RGM-001/candidate_renderings/planning-gap-v2.jsonl','57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23'),
       'manifest':('data/RGM-001/manifests/planning-gap-v2.family_controls.json','9c4497cf61e1d77cea24d808eca965c00e10e975cf235bd7d1aa2667e259da17'),
       'prior':('research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl','896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d'),
       'predictions':('reviews/adversarial/planning-gap-v2.input_predictions.jsonl','ee63bdc9f62c1780326969cb05dbb8410897b176d6fe24fa67b72f257c480c1b'),
       'first_input_judgments':('reviews/adversarial/planning-gap-v2.input_first_judgments.jsonl','5b66ecfd1b48c82aebadd4c1e1f3be89537384ec06c0ab04ca7ea752703ec18a')}
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data,'immutable artifact differs '+str(p)
    else:p.write_bytes(data)
def jsbytes(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def lines(xs):return ''.join(canon(x)+'\n' for x in xs).encode()
STAGE=ROOT/'audits/rgm_planning_gap_v2_adversarial_input.py'
assert sha(STAGE)=='c1f27c9f2aeddfa0b30b77e3e55afa4e96a23d4137df228fccb9b44b7d0778e0'
spec=importlib.util.spec_from_file_location('own_frozen_input_stage',STAGE);I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I);P=I.P
PROPS={'legal_prefix_then_first_failure','fuel_lengthens_fresh_minimum_moves','fuel_blocks_topologically_reachable_goal','valid_goal_detour','later_action_requires_changed_block_clearance','partial_block_goal','required_temporary_goal_displacement'}

def witness_errors(r,p,w):
    errors=[];name=w.get('property');e=w.get('evidence');t=r['semantic_task'];a=p['expected_target'];info=p['search_evidence'];truth=p['independent_row_properties']
    if set(w)!={'sample_id','property','scope','evidence'}:errors.append('witness shape')
    if w.get('scope')!='row' or w.get('sample_id')!=r['sample_id']:errors.append('witness row binding')
    if name not in PROPS or name not in truth:errors.append('false or unknown row property');return errors
    if type(e) is not dict or not e:errors.append('nonempty witness evidence object');return errors
    def scalar(k,v):
        if k not in e or type(e[k]) is not type(v) or e[k]!=v:errors.append('false typed evidence '+k)
    if name=='legal_prefix_then_first_failure':scalar('first_error_index',info['first_failure']['index'])
    elif name=='fuel_lengthens_fresh_minimum_moves':scalar('minimum_moves',info['fresh_minimum_moves']);scalar('topology_distance',info['topology_minimum_moves'])
    elif name=='fuel_blocks_topologically_reachable_goal':scalar('cheapest_fuel',info['cheapest_required_fuel']);scalar('topology_distance',info['topology_minimum_moves'])
    elif name in ('valid_goal_detour','required_temporary_goal_displacement'):scalar('minimum_moves',info['fresh_minimum_moves'])
    elif name=='partial_block_goal':scalar('block_count',len(t['on']));scalar('goal_relation_count',len(t['goal_on']))
    elif name=='later_action_requires_changed_block_clearance':
        idx=e.get('action_index');changed=e.get('changed_by_action_index');pre=e.get('precondition')
        if type(idx) is not int or type(changed) is not int or pre not in ('source','destination') or not 0<=changed<idx<len(a['steps']):errors.append('clearance indices/precondition');return errors
        action=t['plan'][idx];block=action['block'] if pre=='source' else action['to'];initial=t['on'];before=initial if changed==0 else a['steps'][changed-1]['after'];after=a['steps'][changed]['after']
        if not a['steps'][idx]['legal'] or block=='TABLE' or block not in initial.values():errors.append('relevant initial clearance does not fail')
        if block not in before.values() or block in after.values() or not a['steps'][changed]['legal']:errors.append('preceding action does not change relevant clearance')
    return errors

def control_errors(c,rs,preds):
    errors=[]
    if set(c)!={'control_id','name','scope','sample_ids','changed_task_fields','provenance'}:errors.append('control shape')
    ids=c.get('sample_ids',[])
    if type(ids) is not list or len(ids)!=2 or len(set(ids))!=2 or not all(i in rs for i in ids):return errors+['supplied pair binding']
    a,b=[rs[i] for i in ids];ta,tb=a['semantic_task'],b['semantic_task'];pa,pb=[preds[i] for i in ids];changed=sorted(k for k in ta if canon(ta[k])!=canon(tb[k]))
    if c.get('scope')!='corpus_variant' or c.get('provenance')!='supplied_tasks':errors.append('control scope/provenance')
    if a['source_family_key']!=b['source_family_key'] or a['split']!=b['split']:errors.append('pair custody')
    if sorted(c.get('changed_task_fields',[]))!=changed:errors.append('false changed_task_fields')
    name=c.get('name')
    if name=='fuel_only_counterfactual':
        if changed!=['fuel'] or ta['kind']!='graph':errors.append('fuel-only input relationship')
        if pa['search_evidence']['fresh_minimum_moves']==pb['search_evidence']['fresh_minimum_moves']:errors.append('fuel control does not alter fresh feasibility/minimum')
    elif name=='directed_edge_deletion_counterfactual':
        ea={canon(e) for e in ta['edges']};eb={canon(e) for e in tb['edges']}
        if changed!=['edges'] or not ((ea<eb or eb<ea) and len(ea^eb)==1):errors.append('one directed edge deletion')
        if not pa['expected_target']['valid'] or pb['expected_target']['valid'] or pb['search_evidence']['first_failure']['index']<1:errors.append('edge deletion actual plan effect')
        if pa['expected_target']['shortest_plan'] is None or pb['expected_target']['shortest_plan'] is None:errors.append('edge deletion collapses fresh feasibility')
    elif name=='adjacent_action_order_counterfactual':
        x,y=ta['plan'],tb['plan'];ds=[i for i,(a,b) in enumerate(zip(x,y)) if a!=b]
        if changed!=['plan'] or len(x)!=len(y) or len(ds)!=2 or ds[1]!=ds[0]+1 or x[ds[0]]!=y[ds[1]] or x[ds[1]]!=y[ds[0]]:errors.append('adjacent-only action swap')
        if not pa['expected_target']['valid'] or pb['expected_target']['valid'] or not set(pb['search_evidence']['first_failure']['reasons'])&{'source_not_clear','destination_not_clear'}:errors.append('swap has no required clearance failure')
    elif name=='partial_goal_extension_counterfactual':
        if changed!=['goal_on'] or not set(ta['goal_on'].items())<set(tb['goal_on'].items()):errors.append('goal-only valid extension')
        if not pa['expected_target']['reaches_goal'] or not pb['expected_target']['valid'] or pb['expected_target']['reaches_goal'] or not pb['search_evidence']['fresh_minimum_moves']>pa['search_evidence']['fresh_minimum_moves']:errors.append('goal extension outcome/minimum effects')
    elif name=='same_world_valid_goal_detour':
        if changed!=['plan'] or not pa['expected_target']['reaches_goal'] or not pb['expected_target']['reaches_goal'] or not len(tb['plan'])>len(ta['plan']):errors.append('same-world longer valid goal plan')
    else:errors.append('unknown control')
    return errors


FILES['parent_input']=('reviews/input_only/planning-gap-v1.jsonl','2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654')
FILES['current_input']=('reviews/input_only/planning-gap-v2.jsonl','00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c')
PROPS.add('isolated_unchanged_support_failure_after_legal_prefix')
PAIR_CONTEXTS={'fuel_only_counterfactual':{('A0','A1')},'directed_edge_deletion_counterfactual':{('B0','B1')},'adjacent_action_order_counterfactual':{('C0','C1')},'partial_goal_extension_counterfactual':{('D0','D1')},'same_world_valid_goal_detour':{('B0','B3'),('C0','C2'),('D2','D3')},'isolated_noop_append_counterfactual':{('C3','C4')}}

def witness_errors(r,p,w):
    errors=[];name=w.get('property');e=w.get('evidence');t=r['semantic_task'];a=p['expected_target'];info=p['search_evidence']
    if not {'sample_id','property','scope','evidence'}<=set(w):errors.append('required witness fields')
    if w.get('scope')!='row' or w.get('sample_id')!=r['sample_id']:errors.append('exact witness row binding')
    if name not in PROPS or name not in p['independent_row_properties']:return errors+['false or unknown row property']
    if type(e) is not dict or not e:return errors+['nonempty evidence object']
    def scalar(k,v):
        if k not in e or type(e[k]) is not type(v) or e[k]!=v:errors.append('false typed evidence '+k)
    if name=='legal_prefix_then_first_failure':scalar('first_error_index',info['first_failure']['index'])
    elif name=='fuel_lengthens_fresh_minimum_moves':scalar('minimum_moves',info['fresh_minimum_moves']);scalar('topology_distance',info['topology_minimum_moves'])
    elif name=='fuel_blocks_topologically_reachable_goal':scalar('cheapest_fuel',info['cheapest_required_fuel']);scalar('topology_distance',info['topology_minimum_moves'])
    elif name in ('valid_goal_detour','required_temporary_goal_displacement'):scalar('minimum_moves',info['fresh_minimum_moves'])
    elif name=='partial_block_goal':scalar('block_count',len(t['on']));scalar('goal_relation_count',len(t['goal_on']))
    elif name=='later_action_requires_changed_block_clearance':
        try:P.validate_clearance_v2(t,e)
        except (ValueError,KeyError,TypeError) as exc:errors.append('V2 clearance evidence: '+str(exc))
    elif name=='isolated_unchanged_support_failure_after_legal_prefix':
        idx=p['input_derived_six_guard_trace']['first_failure_index'];scalar('action_index',idx);scalar('block',t['plan'][idx]['block']);scalar('current_support','TABLE')
        if type(idx) is not int or idx<1 or idx!=len(t['plan'])-1 or p['input_derived_six_guard_trace']['sole_failure_guard']!='changed_support':errors.append('complete isolated TABLE no-op condition')
    return errors

def control_errors(c,rs,preds):
    errors=[];ids=c.get('sample_ids');name=c.get('name')
    if not {'control_id','name','scope','sample_ids','changed_task_fields','provenance'}<=set(c):errors.append('required control fields')
    if type(c.get('control_id')) is not str or not c['control_id']:errors.append('nonempty control ID')
    if type(ids) is not list or len(ids)!=2 or any(type(x) is not str or not x for x in ids) or len(set(ids))!=2 or not all(x in rs for x in ids):return errors+['two distinct existing exact sample IDs']
    left,right=[rs[x] for x in ids];a,b=left['semantic_task'],right['semantic_task'];pa,pb=[preds[x] for x in ids]
    if left['source_family_key']!=right['source_family_key'] or left['split']!=right['split'] or left['renderer']!=right['renderer']:errors.append('same family/split/actual renderer')
    if name not in PAIR_CONTEXTS:return errors+['unknown public control name']
    if (left['controlled_context'],right['controlled_context']) not in PAIR_CONTEXTS[name]:errors.append('ordered logical intervention context binding')
    if c.get('scope')!='corpus_variant' or c.get('provenance')!='supplied_tasks':errors.append('supplied corpus control scope/provenance')
    changed=sorted(k for k in a if canon(a[k])!=canon(b[k]));fields=c.get('changed_task_fields')
    if type(fields) is not list or not all(type(x) is str for x in fields) or len(fields)!=len(set(fields)) or sorted(fields)!=changed:errors.append('actual typed field differences')
    if name=='fuel_only_counterfactual':
        if changed!=['fuel'] or a['kind']!='graph' or not pa['expected_target']['reaches_goal'] or pb['expected_target']['valid'] or a['fuel']<=b['fuel']:errors.append('decisive directed fuel-only relationship')
        if pb['search_evidence']['topology_minimum_moves'] is None or pa['search_evidence']['fresh_minimum_moves']==pb['search_evidence']['fresh_minimum_moves']:errors.append('fuel intervention lacks fresh-plan effect')
    elif name=='directed_edge_deletion_counterfactual':
        ea={tuple(e) for e in a['edges']};eb={tuple(e) for e in b['edges']}
        if changed!=['edges'] or not eb<ea or len(ea-eb)!=1:errors.append('exact oriented one-edge deletion')
        fail=pb['search_evidence']['first_failure']
        if not pa['expected_target']['reaches_goal'] or pb['expected_target']['valid'] or fail is None or fail['index']<1 or fail['reasons']!=['missing_directed_edge']:errors.append('deleted used edge after legal prefix')
        if pa['expected_target']['shortest_plan'] is None or pb['expected_target']['shortest_plan'] is None:errors.append('both worlds must retain fresh feasible goals')
    elif name=='adjacent_action_order_counterfactual':
        if changed!=['plan']:errors.append('only plan differs')
        try:P.validate_selective_pair(a,b,c.get('evidence'))
        except (ValueError,KeyError,TypeError,IndexError) as exc:errors.append('actual selective C1 guard/mutation: '+str(exc))
    elif name=='partial_goal_extension_counterfactual':
        if changed!=['goal_on'] or not set(a['goal_on'].items())<set(b['goal_on'].items()):errors.append('proper goal-only obligation extension')
        if not pa['expected_target']['reaches_goal'] or not pb['expected_target']['valid'] or pb['expected_target']['reaches_goal'] or not pb['search_evidence']['fresh_minimum_moves']>pa['search_evidence']['fresh_minimum_moves']:errors.append('goal extension satisfaction/minimum contrast')
    elif name=='same_world_valid_goal_detour':
        if changed!=['plan'] or not pa['expected_target']['reaches_goal'] or not pb['expected_target']['reaches_goal'] or len(b['plan'])<=len(a['plan']) or len(b['plan'])<=pb['search_evidence']['fresh_minimum_moves']:errors.append('same-world legal longer goal detour')
    else:
        bases=[r for r in rs.values() if r['source_family_key']==left['source_family_key'] and r['controlled_context']=='C0' and r['renderer']==left['renderer']]
        if changed!=['plan'] or len(bases)!=1:errors.append('actual C0/no-op pair binding')
        else:
            try:P.validate_noop_pair(bases[0]['semantic_task'],a,b,c.get('evidence'))
            except (ValueError,KeyError,TypeError,IndexError) as exc:errors.append('actual isolated no-op append: '+str(exc))
    return errors

def control_cohort_errors(controls,rs,context_samples):
    errors=[];seen=set();coverage=collections.Counter()
    for c in controls:
        cid=c.get('control_id')
        if type(cid) is not str or not cid or cid in seen:errors.append('global duplicate/invalid control_id')
        seen.add(cid)
        ids=c.get('sample_ids',[])
        if len(ids)==2 and all(x in rs for x in ids):
            l,r=[rs[x] for x in ids];coverage[(l['source_family_key'],c.get('name'),l['controlled_context'],r['controlled_context'],l['renderer'])]+=1
    for family,cs in context_samples.items():
        styles={rs[x]['renderer'] for ids in cs.values() for x in ids}
        stratum=next(iter(cs))[0]
        names={'A':['fuel_only_counterfactual'],'B':['directed_edge_deletion_counterfactual','same_world_valid_goal_detour'],'C':['adjacent_action_order_counterfactual','same_world_valid_goal_detour','isolated_noop_append_counterfactual'],'D':['partial_goal_extension_counterfactual','same_world_valid_goal_detour']}[stratum]
        for name in names:
            pair=next(p for p in PAIR_CONTEXTS[name] if p[0] in cs and p[1] in cs)
            for style in styles:
                if coverage[(family,name,*pair,style)]<1:errors.append('missing required declaration '+canon([family,name,pair,style]))
    return errors

def v2_header_errors(manifest,public):
    errors=[];expected={'version':'rgm-planning-controls-2.0.0','batch_name':'planning-gap-v2','track_id':'RGM-001','semantics_sha256':public['schemas/SEMANTICS.md'],'revision_of_batch':'planning-gap-v1','parent_input_view_sha256':FILES['parent_input'][1]}
    for key,value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key]!=value:errors.append('exact V2 header '+key)
    bindings=manifest.get('public_specification_bindings');actual={}
    if type(bindings) is list and all(type(b) is dict and set(b)=={'path','sha256'} and type(b['path']) is type(b['sha256']) is str for b in bindings):actual={b['path']:b['sha256'] for b in bindings}
    if actual!=public or len(actual)!=len(bindings or []):errors.append('seven distinct exact public source bindings')
    return errors

def main():
    started=time.perf_counter()
    for role,(name,h) in FILES.items():assert sha(ROOT/name)==h,role+' exact bytes'
    source=pathlib.Path(__file__);source_hash=sha(source);freeze(ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_comparison-{source_hash}.py',source.read_bytes())
    public=json.loads((ROOT/P.ROOT_FREEZE).read_text())['public_specification_bindings']
    assert sha(ROOT/P.ROOT_FREEZE)==P.ROOT_FREEZE_SHA
    for rel,h in public.items():assert sha(ROOT/rel)==h
    pre={'actual_execution_id':EXEC,'stage':'SOURCE_FREEZE_BEFORE_CANONICAL_COMPARISON_EXECUTION','source_sha256':source_hash,'files_sha256':FILES,'own_input_source_sha256':sha(STAGE),'public_sources_sha256':public,'scope':'Gold-access comparison after ALL408 own typed targets/first input judgments and audits frozen. No producer/private formal/test/rootgate/other reviewer implementation or verdict read.','correlation_disclosure':'Same underlying model, public-guided corrective design, own prior V1/V2 solver/fixture reuse; separate execution is not independent model evidence.'}
    prepath=ROOT/'audits/execution/planning-gap-v2-adversarial-comparison.preexec.json';freeze(prepath,jsbytes(pre));print(canon({'stage':'canonical_source_frozen','source_sha256':source_hash,'preexec_sha256':sha(prepath)}),flush=True)
    rows=[json.loads(l) for l in (ROOT/FILES['candidate'][0]).read_text().splitlines()];rs={r['sample_id']:r for r in rows};preds={r['sample_id']:r for r in (json.loads(l) for l in (ROOT/FILES['predictions'][0]).read_text().splitlines())};views={r['sample_id']:r for r in (json.loads(l) for l in (ROOT/FILES['current_input'][0]).read_text().splitlines())};manifest=json.loads((ROOT/FILES['manifest'][0]).read_text())
    assert len(rs)==len(rows)==len(preds)==len(views)==408
    defects=collections.defaultdict(list);low=collections.defaultdict(list);manifest_errors=[];families=[];contexts=[];controls=[];witnesses=[];first_witness={};seen_samples=collections.Counter();context_samples={};base_rows=[];regressions=[]
    def issue(ids,code,evidence,severity='high'):
        target=low if severity=='low' else defects
        for sid in ids:target[sid].append({'severity':severity,'code':code,'evidence':evidence,'fix':'Create a fresh corrected version for material defects; preserve frozen current/first evidence. Preserve bounded scope disclosures for residual limits.'})
    required={'sample_id','semantic_parent_id','semantic_family_id','track_id','skill_tags','difficulty','generator_version','oracle_version','generation_seed','source_provenance','semantic_task','rendered_input','verified_target','target_format','split','review_status','review_receipt_ids','record_hash','source_family_key','source_family_index','controlled_context','primary_stratum','renderer','producer_agent_id','producer_role','structural_difficulty'}
    row_measurements=[];fresh_dep_cache={}
    for r in rows:
        sid=r['sample_id'];p=preds[sid];t=r['semantic_task'];errs=[]
        if not required<=set(r):errs.append('canonical required field coverage')
        if r['record_hash']!=hashlib.sha256(canon({k:v for k,v in r.items() if k!='record_hash'}).encode()).hexdigest():issue([sid],'record_hash_mismatch',r['record_hash'])
        if canon(r['verified_target'])!=canon(p['expected_target']):issue([sid],'target_mismatch',{'frozen_own_target':p['expected_target'],'actual_target':r['verified_target']})
        if any(canon(r[k])!=canon(views[sid][k]) for k in I.WHITELIST):issue([sid],'stripped_input_join_mismatch','Exact6 fields differ from frozen input-only view')
        for k in ['sample_id','semantic_parent_id','semantic_family_id','generator_version','oracle_version','source_family_key','controlled_context','renderer','producer_agent_id','producer_role','record_hash']:
            if type(r.get(k)) is not str or not r[k]:errs.append('nonempty canonical string '+k)
        if r.get('track_id')!='RGM-001' or r.get('target_format')!='structured' or r.get('split') not in ('train','dev_challenge'):errs.append('track/format/split')
        if type(r['difficulty']) is not int or not 1<=r['difficulty']<=6 or type(r['structural_difficulty']) is not int or r['difficulty']!=r['structural_difficulty']:errs.append('difficulty typed bounds/equality')
        if type(r['source_family_index']) is not int or type(r['generation_seed']) is not int:errs.append('typed source index/seed')
        if type(r['skill_tags']) is not list or any(type(x) is not str for x in r['skill_tags']) or len(set(r['skill_tags']))!=len(r['skill_tags']):errs.append('distinct string tags')
        if type(r['source_provenance']) is not dict or r['source_provenance'].get('sealed_evaluation_used') is not False or r['source_provenance'].get('origin')!='original_synthetic':errs.append('synthetic exposed provenance')
        if r['review_status']!='PENDING' or r['review_receipt_ids']!=[]:errs.append('candidate pre-review status')
        lead,payload=r['rendered_input'].split('Task JSON:\n')
        private=set(r)-{'semantic_task','rendered_input'}
        pattern=re.compile(r'(?<![A-Za-z0-9_])(?:'+'|'.join(re.escape(k) for k in sorted(private))+r')(?![A-Za-z0-9_])["\x27`*_]*\s*[:=]',re.I)
        if pattern.search(lead) or private.intersection(t):errs.append('actual row-private labels/outer task keys exposed')
        if canon(json.loads(payload))!=canon(t):errs.append('typed prompt readback')
        if r['producer_agent_id']==EXEC:errs.append('reviewer falsely bound as producer')
        if errs:issue([sid],'canonical_row_interface',errs)
        fresh=p['expected_target']['shortest_plan'];h=p['search_evidence']['fresh_minimum_moves'];dep=0
        if t['kind']=='blocks' and fresh is not None:
            key=canon({k:v for k,v in t.items() if k!='plan'})
            if key not in fresh_dep_cache:
                u=copy.deepcopy(t);u['plan']=fresh;trace=P.diagnostic_replay(u)[0];fresh_answer={**trace,'shortest_plan':fresh};fresh_dep_cache[key]=len(I.canonical_clearance_witnesses(u,fresh_answer))
            dep=fresh_dep_cache[key]
        resource=p['search_evidence'].get('topology_minimum_moves') is not None and h is not None and h>p['search_evidence'].get('topology_minimum_moves',h)
        if r['difficulty']>=5:issue([sid],'unsupported_Level5_or6','Public prospective scope does not establish new Level5/6 labels')
        if r['difficulty']==4 and not(h is not None and h>=4 and (dep or resource)):issue([sid],'unsupported_Level4_dependency',{'fresh_minimum':h,'fresh_changed_clearance':dep,'resource_lengthening':resource},'moderate')
        if r['difficulty']==2 and h is not None and h>=4:issue([sid],'low_difficulty_for_joint_fresh_plan_target',{'declared':2,'fresh_minimum_moves':h,'fresh_changed_clearance_witnesses':dep,'submitted_first_failure':p['search_evidence']['first_failure']},'low')
        row_measurements.append({'sample_id':sid,'record_hash':r['record_hash'],'declared_difficulty':r['difficulty'],'fresh_minimum_moves':h,'fresh_canonical_changed_clearance_witness_count':dep,'resource_lengthening':resource})
    top={'version':'rgm-planning-controls-2.0.0','batch_name':'planning-gap-v2','track_id':'RGM-001','semantics_sha256':public['schemas/SEMANTICS.md'],'revision_of_batch':'planning-gap-v1','parent_input_view_sha256':FILES['parent_input'][1]}
    manifest_errors+=v2_header_errors(manifest,public)
    if type(manifest.get('families')) is not list or len(manifest['families'])!=48:manifest_errors.append('48 actual family entries')
    indices=set();familykeys=set();all_controls=[]
    for f in manifest['families']:
        fk=f['source_family_key'];index=f['source_index'];stratum=f['primary_stratum'];family_errors=[];ctx={c['context_id']:c for c in f['contexts']};ids=[sid for c in f['contexts'] for sid in c['sample_ids']]
        if fk in familykeys or index in indices:family_errors.append('duplicate family/index');familykeys.add(fk);indices.add(index)
        else:familykeys.add(fk);indices.add(index)
        expected_stratum='A' if 180<=index<192 else 'B' if 192<=index<204 else 'C' if 204<=index<216 else 'D' if 216<=index<228 else None
        if stratum!=expected_stratum or type(index) is not int:family_errors.append('actual public stratum/index allocation')
        if f['split']!=('dev_challenge' if index%10 in (8,9) else 'train'):family_errors.append('family public split assignment')
        expected_contexts={stratum+str(i) for i in range(5 if stratum=='C' else 4)}
        if set(ctx)!=expected_contexts or len(ctx)!=len(f['contexts']):family_errors.append('required distinct contexts')
        if not all(sid in rs for sid in ids):issue([r['sample_id'] for r in rows if r['source_family_key']==fk],'missing_context_sample',ids);continue
        if f['base_sample_id'] not in ctx[stratum+'0']['sample_ids']:family_errors.append('base context binding')
        base=rs[f['base_sample_id']];base_rows.append(base);context_samples[fk]={k:c['sample_ids'] for k,c in ctx.items()}
        for c in f['contexts']:
            ci=c['sample_ids'];names=c['row_properties'];seen_samples.update(ci)
            if len(ci)!=2 or len(set(ci))!=2 or len({rs[s]['renderer'] for s in ci})!=2:family_errors.append('two actual renderer views')
            if type(names) is not list or any(type(n) is not str or n not in PROPS for n in names) or len(names)!=len(set(names)):family_errors.append('distinct public row properties')
            coverage=collections.Counter((w.get('sample_id'),w.get('property')) for w in c['row_witnesses'])
            if set(coverage)!={(s,n) for s in ci for n in names} or any(n!=1 for n in coverage.values()):family_errors.append('exact per-row/property witness coverage')
            if canon(I.normalized_task(rs[ci[0]]['semantic_task']))!=canon(I.normalized_task(rs[ci[1]]['semantic_task'])) or canon(preds[ci[0]]['expected_target'])!=canon(preds[ci[1]]['expected_target']):family_errors.append('normalized renderer semantic equality')
            if rs[ci[0]]['rendered_input']==rs[ci[1]]['rendered_input']:family_errors.append('renderer only relabeled rather than actual presentation change')
            for sid in ci:
                r=rs[sid];p=preds[sid]
                if r['source_family_key']!=fk or r['source_family_index']!=index or r['split']!=f['split'] or r['primary_stratum']!=stratum or r['controlled_context']!=c['context_id']:family_errors.append('row family/context metadata binding')
                if r['semantic_family_id']!=base['semantic_family_id'] or r['semantic_parent_id']!=base['semantic_parent_id']:family_errors.append('parent/family custody')
                if set(r['skill_tags'])&PROPS!=set(names):issue([sid],'row_tags_context_properties_disagree',{'actual_tags':r['skill_tags'],'declared_properties':names})
                if any(n not in p['independent_row_properties'] for n in names):issue([sid],'false_row_property',{'declared':names,'input_truth':list(p['independent_row_properties'])})
            for w in c['row_witnesses']:
                sid=w['sample_id'];err=witness_errors(rs[sid],preds[sid],w);witnesses.append({'sample_id':sid,'record_hash':rs[sid]['record_hash'],'witness':w,'verified':not err,'errors':err});first_witness.setdefault(w['property'],(rs[sid],preds[sid],w))
                if err:issue([sid],'false_or_malformed_row_witness',{'witness':w,'errors':err})
            r=rs[ci[0]];p=preds[r['sample_id']];t=r['semantic_task'];a=p['expected_target'];info=p['search_evidence'];context=c['context_id'];err=[]
            if context in ('A0','B0','C0','D0') and not a['reaches_goal']:err.append('legal goal-reaching base required')
            if context=='B0' and len(t['plan'])<3:err.append('B0 plan at least3')
            if context in ('C0','D0','D2') and canon(t['plan'])!=canon(a['shortest_plan']):err.append('full canonical minimum plan required')
            if context=='C0' and not(4<=info['fresh_minimum_moves']<=8 and p['input_derived_v2_clearance_witnesses'] and 'later_action_requires_changed_block_clearance' in names):err.append('C0 minimum/dependency witness obligation')
            if context=='D0' and not(0<len(t['goal_on'])<len(t['on']) and info['fresh_minimum_moves']>=2):err.append('D0 proper partial goal/minimum')
            if context in ('A2','C3') and not(t['plan'] and a['valid'] and not a['reaches_goal']):err.append('nonempty legal non-goal prefix')
            if context in ('A3','B2'):
                fail=info['first_failure']
                if fail is None or fail['index']<1 or fail['reasons']!=['missing_directed_edge']:err.append('missing directed edge after legal prefix')
                elif context=='B2':
                    i=fail['index'];node=a['steps'][i-1]['after']['node'];dest=t['plan'][i]
                    if not any(x==dest and y==node for x,y,cost in t['edges']):err.append('actual attempted reverse of existing edge')
            if context=='D2' and not(a['reaches_goal'] and canon(t['goal_on'])==canon(rs[ctx['D1']['sample_ids'][0]]['semantic_task']['goal_on'])):err.append('D2 strengthened-goal optimal plan')
            if context in ('B3','C2','D3') and 'valid_goal_detour' not in p['independent_row_properties']:err.append('actual legal goal detour')
            if context=='C4' and not {'isolated_unchanged_support_failure_after_legal_prefix','legal_prefix_then_first_failure'}<=set(names):err.append('mandatory C4 tags/witnesses')
            if context=='C3' and len(t['plan'])!=1+(index-204)%3:err.append('public fixed C3 prefix k')
            if err:issue(ci,'context_public_obligation',err)
            contexts.append({'source_family_key':fk,'context':context,'sample_ids':ci,'record_hashes':[rs[s]['record_hash'] for s in ci],'minimum_moves':info['fresh_minimum_moves'],'actual_properties':list(p['independent_row_properties']),'errors':err})
        # Unpaired context worlds cannot silently change their graph/block initial facts.
        for name in ctx:
            r=rs[ctx[name]['sample_ids'][0]];t=r['semantic_task'];bt=base['semantic_task'];fields=('nodes','edges','start','goal','fuel') if stratum in ('A','B') else ('on','goal_on')
            exempt={'A1':{'fuel'},'B1':{'edges'},'D1':{'goal_on'},'D2':{'goal_on'},'D3':{'goal_on'}}.get(name,set())
            if any(canon(I.normalized_task(t)[k])!=canon(I.normalized_task(bt)[k]) for k in fields if k not in exempt):family_errors.append('unpaired context initial world differs '+name)
        for c in f['controls']:
            err=control_errors(c,rs,preds);controls.append({'source_family_key':fk,'control':c,'verified':not err,'errors':err});all_controls.append(c)
            if err:issue([x for x in c.get('sample_ids',[]) if x in rs],'false_or_malformed_supplied_control',{'control':c,'errors':err})
        if family_errors:issue(ids,'family_manifest_obligation',family_errors)
        families.append({'source_family_key':fk,'source_index':index,'stratum':stratum,'samples':len(ids),'errors':family_errors})
    if set(seen_samples)!=set(rs) or any(n!=1 for n in seen_samples.values()):manifest_errors.append('exact once manifest sample coverage')
    if indices!=set(range(180,228)):manifest_errors.append('all48 revised source indices')
    manifest_errors+=control_cohort_errors(all_controls,rs,context_samples)
    if len(all_controls)<192:manifest_errors.append('at least192 per-renderer required declarations')
    if manifest_errors:issue(list(rs),'V2_manifest_interface_or_custody',manifest_errors)
    # Explicit public allocation, truth and collective gates already input-derived, now bind actual context IDs.
    cohort=json.loads((ROOT/'audits/diversity/planning-gap-v2-input-only-cohort.json').read_text())
    if cohort['public_cohort_errors']:issue(list(rs),'input_public_cohort_errors',cohort['public_cohort_errors'])
    C0=[r for r in base_rows if r['primary_stratum']=='C'];B0=[r for r in base_rows if r['primary_stratum']=='B']
    if sum(preds[r['sample_id']]['search_evidence']['required_temporary_goal_displacement'] for r in C0)<4:issue([r['sample_id'] for r in rows if r['primary_stratum']=='C'],'too_few_required_temporary_displacements','ActualC0 fewer4')
    if sum(preds[r['sample_id']]['search_evidence']['minimum_route_multiplicity']['route_count']>=2 for r in B0)<6:issue([r['sample_id'] for r in rows if r['primary_stratum']=='B'],'too_few_B0_minimum_route_ties','ActualB0 fewer6')
    # Actual rejected corruptions target publication-facing declarations, not production examples.
    for name,(r,p,w) in first_witness.items():
        key=next(k for k,v in w['evidence'].items() if type(v) is int)
        false=copy.deepcopy(w);false['evidence'][key]=True;err=witness_errors(r,p,false);assert err;regressions.append({'name':'typed_bool_index_or_scalar:'+name,'sample_id':r['sample_id'],'mutation':false,'rejected':err})
        false=copy.deepcopy(w);false['scope']='corpus_variant';err=witness_errors(r,p,false);assert err;regressions.append({'name':'wrong_scope:'+name,'sample_id':r['sample_id'],'rejected':err})
    for name in PAIR_CONTEXTS:
        c=next(c for c in all_controls if c['name']==name)
        for label,change in [('false_fields',lambda x:x.update(changed_task_fields=['start'])),('four_IDs',lambda x:x.update(sample_ids=x['sample_ids']*2)),('nonexistent_ID',lambda x:x.update(sample_ids=[x['sample_ids'][0],'RED-NONEXISTENT']))]:
            false=copy.deepcopy(c);change(false);err=control_errors(false,rs,preds);assert err;regressions.append({'name':label+':'+name,'actual_control_id':c['control_id'],'mutation':false,'rejected':err})
    false=copy.deepcopy(all_controls);false[1]['control_id']=false[0]['control_id'];err=control_cohort_errors(false,rs,context_samples);assert err;regressions.append({'name':'globally_duplicate_control_ID','rejected':err})
    false=copy.deepcopy(all_controls);false.pop();err=control_cohort_errors(false,rs,context_samples);assert err;regressions.append({'name':'missing_one_required_style','rejected':err})
    c=copy.deepcopy(next(c for c in all_controls if c['name']=='adjacent_action_order_counterfactual'));right=rs[c['sample_ids'][1]];other=next(r for r in rows if r['source_family_key']==right['source_family_key'] and r['controlled_context']==right['controlled_context'] and r['renderer']!=right['renderer']);c['sample_ids'][1]=other['sample_id'];err=control_errors(c,rs,preds);assert err;regressions.append({'name':'mixed_renderer_same_context_pair','rejected':err})
    for key,mut in [('version',2),('parent_input_view_sha256','0'*64),('revision_of_batch','wrong')]:
        false=copy.deepcopy(manifest);false[key]=mut;err=v2_header_errors(false,public);assert err;regressions.append({'name':'header_corruption_'+key,'expected':top[key],'mutation':mut,'rejected':err})
    lexical=[]
    for r in base_rows:
        t=r['semantic_task'];p=preds[r['sample_id']]
        if t['kind']!='graph' or p['search_evidence']['minimum_route_multiplicity']['route_count']<2:continue
        names=sorted(t['nodes']);mp={n:'z'+str(len(names)-i).zfill(2) for i,n in enumerate(names)};u=copy.deepcopy(t);u['nodes']=[mp[n] for n in t['nodes']];u['edges']=[[mp[a],mp[b],c] for a,b,c in t['edges']];u.update(start=mp[t['start']],goal=mp[t['goal']],plan=[mp[n] for n in t['plan']]);a,info=P.solve(u);alternate=I.independent_simple_graph(u)[0]
        assert canon(a['shortest_plan'])==canon(alternate) and a['valid']==p['expected_target']['valid'] and a['reaches_goal']==p['expected_target']['reaches_goal'] and len(a['shortest_plan'])==p['search_evidence']['fresh_minimum_moves']
        lexical.append({'origin_sample_id':r['sample_id'],'origin_record_hash':r['record_hash'],'task':u,'input_derived_exact_target':a,'minimum_route_differs_from_renamed_original':a['shortest_plan']!=[mp[n] for n in p['expected_target']['shortest_plan']]})
    # Freshness compared to frozen prior ACTIVE public inputs, while V1 is revised lineage.
    prior=[json.loads(l) for l in (ROOT/FILES['prior'][0]).read_text().splitlines()];parent=[json.loads(l) for l in (ROOT/FILES['parent_input'][0]).read_text().splitlines()]
    prior_units={canon(I.normalized_task(r['semantic_task'])):r for r in prior};parent_units={canon(I.normalized_task(r['semantic_task'])):r for r in parent};prior_signatures=collections.defaultdict(set);new_signatures=collections.defaultdict(dict);prior_guard_counts=collections.Counter();prior_solutions={}
    for key,r in prior_units.items():
        t=r['semantic_task'];a,info=P.solve(t);prior_solutions[key]=(a,info)
        if t['kind']=='graph':prior_signatures['weighted_graph'].add(I.graph_signature(t));prior_signatures['unweighted_graph'].add(I.graph_signature(t,False,False))
        else:
            prior_signatures['joint_blocks'].add(I.blocks_signature(t));prior_signatures['initial_blocks'].add(I.blocks_signature(t,False));_,guards,fail=P.diagnostic_replay(t)
            if fail is not None:
                for name,v in guards[fail].items():
                    if not v:prior_guard_counts[name]+=1
    prior_families={r['source_family_key'] for r in prior};parent_families={r['source_family_key'] for r in parent};new_families={r['source_family_key'] for r in rows};collisions=[]
    if new_families&prior_families:issue(list(rs),'active_source_family_collision',sorted(new_families&prior_families))
    if new_families!=parent_families:issue(list(rs),'revision_family_lineage_mismatch',{'missing_parent':sorted(parent_families-new_families),'additional':sorted(new_families-parent_families)})
    for r in base_rows:
        t=r['semantic_task'];sid=r['sample_id']
        if t['kind']=='graph':new_signatures['weighted_graph'][sid]=I.graph_signature(t);new_signatures['unweighted_graph'][sid]=I.graph_signature(t,False,False)
        else:new_signatures['joint_blocks'][sid]=I.blocks_signature(t);new_signatures['initial_blocks'][sid]=I.blocks_signature(t,False)
    for kind in ('weighted_graph','joint_blocks'):
        gs=collections.defaultdict(list)
        for sid,sig in new_signatures[kind].items():gs[sig].append(sid)
        for sig,sids in gs.items():
            if len(sids)>1 or sig in prior_signatures[kind]:
                facts={'kind':kind,'base_ids':sids,'prior_active_collision':sig in prior_signatures[kind]};collisions.append(facts);issue([r['sample_id'] for r in rows if r['source_family_key'] in {rs[s]['source_family_key'] for s in sids}],'required_base_structural_novelty_failure',facts)
    novelty={k:len(set(v.values())-prior_signatures[k]) for k,v in new_signatures.items()};sixshape={I.blocks_signature(r['semantic_task'],False) for r in base_rows if r['semantic_task']['kind']=='blocks' and len(r['semantic_task']['on'])==6}
    if novelty['unweighted_graph']<16:issue([r['sample_id'] for r in rows if r['semantic_task']['kind']=='graph'],'insufficient_new_unweighted_topologies',novelty['unweighted_graph'])
    if len(sixshape)<8:issue([r['sample_id'] for r in rows if r['semantic_task']['kind']=='blocks'],'insufficient_six_block_initial_shapes',len(sixshape))
    parent_shared_tasks=set(parent_units)&{canon(I.normalized_task(r['semantic_task'])) for r in rows}
    parent_world={fk:set() for fk in parent_families};new_world={fk:set() for fk in new_families}
    for r in parent_units.values():parent_world[r['source_family_key']].add(canon({k:v for k,v in I.normalized_task(r['semantic_task']).items() if k not in ('plan','fuel')}))
    for r in rows:new_world[r['source_family_key']].add(canon({k:v for k,v in I.normalized_task(r['semantic_task']).items() if k not in ('plan','fuel')}))
    for role,(name,h) in FILES.items():assert sha(ROOT/name)==h,'bound source changed after execution '+role
    for rel,h in public.items():assert sha(ROOT/rel)==h,'public bytes changed after execution'
    report={'actual_execution_id':EXEC,'completed':True,'stage':'GOLD_AND_PUBLIC_METADATA_COMPARISON_AFTER_FROZEN_INPUT_FIRSTS','source_sha256':source_hash,'preexec_sha256':sha(prepath),'files_sha256_before_after':FILES,'public_sources_sha256_before_after':public,'rows':408,'normalized_contexts':204,'families':48,'exact_typed_targets':sum(canon(r['verified_target'])==canon(preds[r['sample_id']]['expected_target']) for r in rows),'witnesses_checked':len(witnesses),'controls_checked':len(controls),'distinct_control_IDs':len({c['control_id'] for c in all_controls}),'witness_checks':witnesses,'control_checks':controls,'family_checks':families,'context_checks':contexts,'manifest_errors':manifest_errors,'row_difficulty_measurements':row_measurements,'material_defect_rows':len(defects),'material_defects':dict(defects),'low_scope_issues':dict(low),'executed_rejected_corruptions':regressions,'lexical_tie_actual_renaming_probes':lexical,'prior_public_inputs':len(prior),'prior_semantic_inputs_independently_solved':len(prior_units),'prior_guard_failures_independent_six_predicates':dict(prior_guard_counts),'prior_active_family_intersection':len(new_families&prior_families),'parent_v1_family_intersection':len(new_families&parent_families),'additional_families_relative_parent':len(new_families-parent_families),'shared_normalized_semantic_tasks_with_parent':len(parent_shared_tasks),'families_with_identical_world_set_to_parent':sum(parent_world[f]==new_world[f] for f in new_families&parent_families),'new_base_signature_counts':{k:len(set(v.values())) for k,v in new_signatures.items()},'absent_from_prior_active_signature_counts':novelty,'six_block_initial_shapes':len(sixshape),'novelty_collisions':collisions,'actual_duration_seconds':time.perf_counter()-started,'read_scope':pre['scope'],'review_mode':pre['correlation_disclosure'],'residual_scopes':['408 views /204 semantic inputs /48 revised source families; zero growth versus pendingV1. Two existing action grammars, not204 independent mechanisms.','C1 source/destination split balance is5train/1dev each; one dev infeasible A1 remains. Fuel-only lookup34/34 versus majority33/34 is a small exposed-dev association, with mixed same-fuel training cases.','Six shared block initial-shape groups remain; current joint base goals are distinct. No sealedOOD or efficacy claim.','Difficulty2–4 is bounded heuristic metadata; some C4 Level2 rows still require complex fresh minimum planning.','Actual producer private source/execution and root production schema/gate implementation were outside scope; declared generator/oracle versions/provenance are bound exact strings, not independently certified source execution.','No expert/adjudication/private formal/producer expectations read; common-model, public-guided design and own V1/V2 reuse correlation disclosed.']}
    audit=ROOT/'audits/diversity/planning-gap-v2-adversarial-canonical.json';freeze(audit,jsbytes(report))
    receipts=[]
    for r in rows:
        sid=r['sample_id'];p=preds[sid];errs=defects.get(sid,[]);limited=low.get(sid,[]);decision='REVISE' if errs else 'ACCEPT'
        scope=[{'severity':'low','code':'bounded_correlated_public_guided_proxy','evidence':'48 revised lineages,204 semantic inputs,408 presentations; independently frozen exact targets plus every actual witness/control, counterfactuals and prior-active structural comparison. Shared underlying model/common public design and exposed development remain.','fix':'Keep independent-family/semantic/presentation units separate; future efficacy needs separate evaluation custody.'}]
        if r['split']=='dev_challenge':scope.append({'severity':'low','code':'small_development_negative_count','evidence':'Only1 dev infeasible semantic task. Fuel-only lookup34/34 versus majority33/34; same-fuel train cases have opposite feasibility.','fix':'Disclose tiny negative cohort; no generalized resource-planning or efficacy inference.'})
        scores={'semantic_correctness':5 if not errs else 2,'target_correctness':5 if canon(r['verified_target'])==canon(p['expected_target']) else 1,'instruction_clarity':5,'reasoning_validity':5 if not errs else 2,'generalization_value':3,'difficulty_quality':3 if limited else 4,'diversity':4,'shortcut_resistance':3}
        receipts.append({'sample_id':sid,'record_hash':r['record_hash'],'receipt_id':'red-planning-gap-v2-'+hashlib.sha256((sid+r['record_hash']+EXEC).encode()).hexdigest()[:20],'reviewer_role':'adversarial','reviewer_execution_id':EXEC,'producer_execution_id':r['producer_agent_id'],'score_scale':'1–5 higher is better','scores':scores,'critical_defects':errs,'issues':errs+limited+scope,'decision':decision,'confidence':0.98,'evidence':{'prediction_sha256':FILES['predictions'][1],'canonical_audit_sha256':sha(audit),'candidate_sha256':FILES['candidate'][1],'manifest_sha256':FILES['manifest'][1],'canonical_comparison_source_sha256':source_hash,'target_check':'Exact typed canonical JSON against own pre-gold408 frozen predictions, not a producer/formal oracle approval.','fresh_minimum_moves':p['search_evidence']['fresh_minimum_moves'],'complete_finite_search':p['search_evidence']['search']['complete'],'row_properties_independently_supported':list(p['independent_row_properties'])},'review_scope':'Dataset correctness/control quality only on exact current bytes. Public-guided same-model own V1/V2 solver/fixture reuse disclosed. No publication/adjudication authority, independent-model or efficacy claim.'})
    first=ROOT/'reviews/adversarial/planning-gap-v2.first.jsonl';final=ROOT/'reviews/adversarial/planning-gap-v2.jsonl';freeze(first,lines(receipts));freeze(final,lines(receipts))
    print(canon({'first_canonical_verdicts_frozen':True,'first_sha256':sha(first),'final_sha256':sha(final),'audit_sha256':sha(audit),'decisions':dict(collections.Counter(p['decision'] for p in receipts)),'exact_targets':report['exact_typed_targets'],'witnesses':len(witnesses),'controls':len(controls),'material_defect_rows':len(defects),'novelty':novelty,'six_block_shapes':len(sixshape)}),flush=True)
    ex={'actual_execution_id':EXEC,'completed':True,'source_sha256':source_hash,'preexec_sha256':sha(prepath),'files_sha256_before_after':FILES,'public_sources_sha256_before_after':public,'first_canonical_judgment_sha256':sha(first),'final_judgment_sha256':sha(final),'canonical_audit_sha256':sha(audit),'rows':408,'decisions':dict(collections.Counter(p['decision'] for p in receipts)),'exact_targets':report['exact_typed_targets'],'witnesses':len(witnesses),'controls':len(controls),'executed_corruption_rejections':len(regressions),'actual_lexical_renaming_probes':len(lexical),'prior_public_semantic_inputs_solved':len(prior_units),'no_other_verdict_or_private_producer_formal_rootgate_read':True,'common_model_public_guided_own_reuse_disclosed':True,'dataset_review_only_not_publication_approval':True,'residual_scopes':report['residual_scopes']}
    out=ROOT/'audits/execution/planning-gap-v2-adversarial.json';freeze(out,jsbytes(ex));print(canon({'completed':True,'source_sha256':source_hash,'execution_sha256':sha(out),'first_sha256':sha(first),'final_sha256':sha(final),'rejected_corruptions':len(regressions),'lexical_probes':len(lexical),'duration_seconds':report['actual_duration_seconds']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        source=pathlib.Path(__file__);path=ROOT/f'audits/execution/planning-gap-v2-adversarial-comparison.failure-{sha(source)[:16]}.json';freeze(path,jsbytes({'actual_execution_id':EXEC,'source_sha256':sha(source),'completed':False,'failure_type':type(e).__name__,'failure':str(e),'candidate_and_previous_evidence_not_modified':True}));print(canon({'failure_retained':str(path.relative_to(ROOT)),'failure_sha256':sha(path),'error':str(e)}),flush=True);raise
