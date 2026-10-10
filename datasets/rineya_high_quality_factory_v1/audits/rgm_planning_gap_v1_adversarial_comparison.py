#!/usr/bin/env python3
"""Canonical-stage independent adversarial review, /root/adversarial_auditor.
Only frozen own predictions/public documents/candidate/manifest/prior public inputs.
No producer, new formal, expert, reference, root gate or other verdict imports.
"""
import collections,copy,hashlib,importlib.util,itertools,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
FILES={'candidate':('data/RGM-001/candidate_renderings/planning-gap-v1.jsonl','a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27'),
       'manifest':('data/RGM-001/manifests/planning-gap-v1.family_controls.json','405e97f28c81e791e8614b369e8af89fa08b74da49a66114315a217bcf083233'),
       'prior':('research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl','896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d'),
       'predictions':('reviews/adversarial/planning-gap-v1.input_predictions.jsonl','273fbdcc8005c034b1f443f5f4ff3309193c899edb461dfd33d69b2f8fda2ec3'),
       'first_input_judgments':('reviews/adversarial/planning-gap-v1.input_first_judgments.jsonl','57919815ecd74f047427e545a704a28bf490b9790ddd256a4847f31570988112')}
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data,'immutable artifact differs '+str(p)
    else:p.write_bytes(data)
def jsbytes(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def lines(xs):return ''.join(canon(x)+'\n' for x in xs).encode()
STAGE=ROOT/'audits/rgm_planning_gap_v1_adversarial_input.py'
assert sha(STAGE)=='9331eec2c8c894c206e49e46a0f086755690f91624e181ef3464a43ef59ac00f'
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

def main():
    for role,(name,h) in FILES.items():assert sha(ROOT/name)==h,role+' exact SHA'
    source=pathlib.Path(__file__);source_hash=sha(source);freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_comparison-{source_hash}.py',source.read_bytes())
    rows=[json.loads(s) for s in (ROOT/FILES['candidate'][0]).open()];rs={r['sample_id']:r for r in rows};preds={r['sample_id']:r for r in map(json.loads,(ROOT/FILES['predictions'][0]).open())};manifest=json.loads((ROOT/FILES['manifest'][0]).read_text());prior=[json.loads(s) for s in (ROOT/FILES['prior'][0]).open()]
    assert len(rs)==len(rows)==len(preds)==384
    defects=collections.defaultdict(list);manifest_errors=[];witness_count=0;controls=[];families=[];context_evidence=[];regressions=[];first_witnesses={}
    def issue(ids,code,detail,severity='high'):
        for sid in ids:defects[sid].append({'severity':severity,'code':code,'evidence':detail,'fix':'Create a fresh corrected artifact; preserve frozen candidates/first evidence.'})
    for r in rows:
        p=preds[r['sample_id']]
        if r['record_hash']!=hashlib.sha256(canon({k:v for k,v in r.items() if k!='record_hash'}).encode()).hexdigest():issue([r['sample_id']],'record_hash_mismatch','Canonical complete record except record_hash SHA256 differs.')
        if r['record_hash']!=p['record_hash'] or r['source_family_key']!=p['source_family_key'] or r['split']!=p['split']:issue([r['sample_id']],'input_join_mismatch','Frozen input prediction lineage/hash differs.')
        if canon(r['verified_target'])!=canon(p['expected_target']):issue([r['sample_id']],'incorrect_typed_target',{'expected':p['expected_target'],'actual':r['verified_target']})
        if r['structural_difficulty']!=r['difficulty'] or type(r['difficulty']) is not int or not 2<=r['difficulty']<=4:issue([r['sample_id']],'unsupported_difficulty_range','Pilot public difficulty bounds2–4 or typed structural correspondence fails.')
        if set(r['skill_tags'])-({'finite_synthetic_proxy','relational_planning'}|PROPS):issue([r['sample_id']],'unknown_row_tag',r['skill_tags'])
        if (set(r['skill_tags'])&PROPS)-set(p['independent_row_properties']):issue([r['sample_id']],'false_factual_tag',{'tags':r['skill_tags'],'independent':list(p['independent_row_properties'])})
        # Independence is actual execution, not declared oracle_version/producer metadata.
    if set(manifest)!={'batch_name','families','semantics_sha256','track_id','version'} or manifest['semantics_sha256']!=sha(ROOT/'schemas/SEMANTICS.md') or manifest['track_id']!='RGM-001' or manifest['batch_name']!='planning-gap-v1':manifest_errors.append('manifest top interface/hash')
    v=manifest['version']
    if not(type(v) is str and v or type(v) is int and v>0):manifest_errors.append('manifest typed version')
    seen_samples=collections.Counter();seen_keys=set();seen_controls=set();base_rows=[]
    for f in manifest['families']:
        fk=f['source_family_key'];ids=[i for c in f['contexts'] for i in c['sample_ids']];family_errors=[]
        if set(f)!={'source_family_key','source_index','split','primary_stratum','base_sample_id','contexts','controls'}:family_errors.append('family shape')
        if fk in seen_keys:family_errors.append('duplicate family key')
        seen_keys.add(fk);stratum=f['primary_stratum'];index=f['source_index'];ctx={c['context_id']:c for c in f['contexts']}
        if set(ctx)!={stratum+str(i) for i in range(4)} or len(f['contexts'])!=4 or len(ids)!=8 or len(set(ids))!=8:family_errors.append('four contexts/eight views')
        if not 180<=index<=227 or f['split']!=('dev_challenge' if index%10 in (8,9) else 'train'):family_errors.append('source index/split policy')
        if f['base_sample_id'] not in ctx[stratum+'0']['sample_ids']:family_errors.append('base context binding')
        if not all(i in rs for i in ids):family_errors.append('missing supplied sample');continue
        base=rs[f['base_sample_id']];base_rows.append(base)
        for c in f['contexts']:
            ci=c['sample_ids'];names=c['row_properties'];seen_samples.update(ci)
            if set(c)!={'context_id','sample_ids','row_properties','row_witnesses'} or len(ci)!=2 or len(set(ci))!=2 or set(rs[i]['renderer'] for i in ci)!={'forward','reverse'}:family_errors.append('context/renderers shape')
            if type(names) is not list or any(type(n) is not str or n not in PROPS for n in names) or len(names)!=len(set(names)):family_errors.append('context typed properties')
            bindings=collections.Counter((w.get('sample_id'),w.get('property')) for w in c['row_witnesses'])
            if set(bindings)!={(sid,n) for sid in ci for n in names} or any(v!=1 for v in bindings.values()):family_errors.append('per-row per-tag witness coverage')
            for sid in ci:
                r=rs[sid]
                if r['source_family_key']!=fk or r['source_family_index']!=index or r['split']!=f['split'] or r['primary_stratum']!=stratum or r['controlled_context']!=c['context_id']:family_errors.append('family/context lineage')
                if (set(r['skill_tags'])&PROPS)!=set(names):issue([sid],'context_row_tag_disagreement',{'context':names,'actual':r['skill_tags']})
                if r['semantic_family_id']!=base['semantic_family_id'] or r['semantic_parent_id']!=base['semantic_parent_id']:family_errors.append('parent/family custody')
            if canon(rs[ci[0]]['semantic_task'])!=canon(rs[ci[1]]['semantic_task']):family_errors.append('renderer semantic equality')
            for w in c['row_witnesses']:
                witness_count+=1;sid=w['sample_id'];errors=witness_errors(rs[sid],preds[sid],w)
                if errors:issue([sid],'false_row_witness',{'witness':w,'errors':errors})
                first_witnesses.setdefault(w['property'],(rs[sid],preds[sid],w))
            r=rs[ci[0]];p=preds[r['sample_id']];a=p['expected_target'];info=p['search_evidence'];task=r['semantic_task'];context=c['context_id'];err=[]
            if context in ('A0','B0','C0','D0') and not a['reaches_goal']:err.append('base is not legal goal-reaching')
            if context=='B0' and len(task['plan'])<3:err.append('B0 horizon below3')
            if context=='C0' and not (4<=info['fresh_minimum_moves']<=8 and len(task['plan'])==info['fresh_minimum_moves'] and 'later_action_requires_changed_block_clearance' in p['independent_row_properties']):err.append('C0 minimum/dependency obligation')
            if context=='D0' and not('partial_block_goal' in p['independent_row_properties'] and info['fresh_minimum_moves']>=2 and len(task['plan'])==info['fresh_minimum_moves']):err.append('D0 proper partial minimum obligation')
            if context in ('A2','C3') and not(task['plan'] and a['valid'] and not a['reaches_goal']):err.append('nonempty legal non-goal prefix')
            if context in ('A3','B2'):
                if info['first_failure'] is None or info['first_failure']['index']<1 or 'missing_directed_edge' not in info['first_failure']['reasons']:err.append('missing-edge failure after legal prefix')
                if context=='B2' and info['first_failure'] is not None:
                    i=info['first_failure']['index'];n=task['start'] if i==0 else a['steps'][i-1]['after']['node'];dest=task['plan'][i]
                    if not any(x==dest and y==n for x,y,c in task['edges']):err.append('no original edge whose reverse is attempted')
            if context=='D2' and not(a['reaches_goal'] and len(task['plan'])==info['fresh_minimum_moves']):err.append('strengthened-goal canonical length')
            if context in ('B3','C2','D3') and 'valid_goal_detour' not in p['independent_row_properties']:err.append('valid longer goal detour obligation')
            if err:issue(ci,'context_obligation_failure',err)
            context_evidence.append({'family':fk,'context':context,'sample_ids':ci,'valid':a['valid'],'reaches_goal':a['reaches_goal'],'minimum_moves':info['fresh_minimum_moves'],'first_failure':info['first_failure'],'properties':list(p['independent_row_properties']),'errors':err})
        for c in f['controls']:
            if c['control_id'] in seen_controls:family_errors.append('duplicate control ID')
            seen_controls.add(c['control_id']);errors=control_errors(c,rs,preds)
            if errors:issue(c['sample_ids'],'false_supplied_control',{'control':c,'errors':errors})
            controls.append({'family':fk,'control':c,'verified':not errors,'errors':errors})
        required_names={'A':{'fuel_only_counterfactual'},'B':{'directed_edge_deletion_counterfactual','same_world_valid_goal_detour'},'C':{'adjacent_action_order_counterfactual','same_world_valid_goal_detour'},'D':{'partial_goal_extension_counterfactual','same_world_valid_goal_detour'}}[stratum]
        if {c['name'] for c in f['controls']}!=required_names or len(f['controls'])!=len(required_names):family_errors.append('required supplied family controls')
        if family_errors:issue(ids,'family_manifest_obligation',family_errors)
        families.append({'source_family_key':fk,'stratum':stratum,'split':f['split'],'index':index,'samples':len(ids),'errors':family_errors})
    if set(seen_samples)!=set(rs) or any(n!=1 for n in seen_samples.values()) or len(manifest['families'])!=48:manifest_errors.append('manifest exact sample/family coverage')
    if manifest_errors:issue(list(rs),'manifest_interface',manifest_errors)
    # Prospective allocation and topology novelty are proven from inputs, not key declarations.
    prior_units={canon(r['semantic_task']):r for r in prior};prior_families={r['source_family_key'] for r in prior}
    signatures=collections.defaultdict(dict);oldsignatures=collections.defaultdict(set);oldguards=collections.Counter();oldguardexamples={};prior_graphs=[];prior_blocks=[]
    for r in prior_units.values():
        t=r['semantic_task'];a,info=P.solve(t)
        if info['first_failure']:
            for reason in info['first_failure']['reasons']:
                oldguards[reason]+=1;oldguardexamples.setdefault(reason,{'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'task':t,'independent_first_failure':info['first_failure']})
        if t['kind']=='graph':prior_graphs.append(r)
        else:prior_blocks.append(r)
    oldcache={}
    for r in prior_graphs:
        t=r['semantic_task'];world=canon({k:v for k,v in t.items() if k not in ('plan','fuel')})
        if world not in oldcache:oldcache[world]=(I.graph_signature(t),I.graph_signature(t,False,False))
        weighted,unweighted=oldcache[world];oldsignatures['weighted_graph'].add(weighted);oldsignatures['unweighted_graph'].add(unweighted)
    for r in prior_blocks:oldsignatures['joint_blocks'].add(I.blocks_signature(r['semantic_task']));oldsignatures['initial_blocks'].add(I.blocks_signature(r['semantic_task'],False))
    for r in base_rows:
        t=r['semantic_task'];sid=r['sample_id'];fk=r['source_family_key']
        if fk in prior_families:issue([x['sample_id'] for x in rows if x['source_family_key']==fk],'reused_source_family_key',fk)
        if t['kind']=='graph':signatures['weighted_graph'][sid]=I.graph_signature(t);signatures['unweighted_graph'][sid]=I.graph_signature(t,False,False)
        else:signatures['joint_blocks'][sid]=I.blocks_signature(t);signatures['initial_blocks'][sid]=I.blocks_signature(t,False)
    novel={k:len(set(v.values())-oldsignatures[k]) for k,v in signatures.items()}
    novelty_counts={k:len(set(v.values())) for k,v in signatures.items()}
    collision_evidence=[]
    for kind in ('weighted_graph','joint_blocks'):
        groups=collections.defaultdict(list)
        for sid,s in signatures[kind].items():groups[s].append(sid)
        for sig,ids in groups.items():
            if len(ids)>1 or sig in oldsignatures[kind]:
                affected=[r['sample_id'] for r in rows if r['source_family_key'] in {rs[sid]['source_family_key'] for sid in ids}]
                issue(affected,'false_prospective_structural_novelty',{'kind':kind,'base_ids':ids,'prior_collision':sig in oldsignatures[kind]});collision_evidence.append({'kind':kind,'base_ids':ids,'prior_collision':sig in oldsignatures[kind]})
    if novel['unweighted_graph']<16:issue([r['sample_id'] for r in rows if r['semantic_task']['kind']=='graph'],'insufficient_new_unweighted_topologies',novel['unweighted_graph'])
    sixbases=[r for r in base_rows if r['semantic_task']['kind']=='blocks' and len(r['semantic_task']['on'])==6];sixshapes={I.blocks_signature(r['semantic_task'],False) for r in sixbases}
    if len(sixshapes)<8:issue([r['sample_id'] for r in rows if r['semantic_task']['kind']=='blocks'],'insufficient_six_block_initial_shapes',len(sixshapes))
    totals=collections.Counter(f['primary_stratum'] for f in manifest['families']);allocation=collections.Counter((f['primary_stratum'],f['split'],len(rs[f['base_sample_id']]['semantic_task'].get('nodes',rs[f['base_sample_id']]['semantic_task'].get('on',{})))) for f in manifest['families'])
    if totals!={'A':12,'B':12,'C':12,'D':12}:issue(list(rs),'stratum_family_count',dict(totals))
    subtype=collections.Counter('blocked' if preds[c['sample_ids'][0]]['expected_target']['shortest_plan'] is None else 'lengthened' for f in manifest['families'] if f['primary_stratum']=='A' for c in f['contexts'] if c['context_id']=='A1')
    if subtype!={'blocked':6,'lengthened':6}:issue([r['sample_id'] for r in rows if r['primary_stratum']=='A'],'fuel_subtype_allocation',dict(subtype))
    c0_displace=sum('required_temporary_goal_displacement' in preds[f['base_sample_id']]['independent_row_properties'] for f in manifest['families'] if f['primary_stratum']=='C')
    b0_ties=sum(preds[f['base_sample_id']]['search_evidence']['minimum_route_multiplicity']['route_count']>=2 for f in manifest['families'] if f['primary_stratum']=='B')
    if c0_displace<4:issue([r['sample_id'] for r in rows if r['primary_stratum']=='C'],'too_few_temporary_displacement_bases',c0_displace)
    if b0_ties<6:issue([r['sample_id'] for r in rows if r['primary_stratum']=='B'],'too_few_minimum_route_tie_bases',b0_ties)
    c1_indices=collections.Counter(preds[c['sample_ids'][0]]['search_evidence']['first_failure']['index'] for f in manifest['families'] if f['primary_stratum']=='C' for c in f['contexts'] if c['context_id']=='C1')
    if len(c1_indices)<2:issue([r['sample_id'] for r in rows if r['primary_stratum']=='C'],'fixed_clearance_failure_index',dict(c1_indices))
    # Independently executed witness/pair corruptions and lexical tie renaming probes.
    for name,(r,p,w) in first_witnesses.items():
        false=copy.deepcopy(w);key=next(k for k,v in false['evidence'].items() if type(v) is int);false['evidence'][key]+=100
        errors=witness_errors(r,p,false);assert errors;regressions.append({'name':'false_scalar_or_index:'+name,'sample_id':r['sample_id'],'mutation':false,'rejected_errors':errors})
        false=copy.deepcopy(w);false['scope']='corpus_variant';errors=witness_errors(r,p,false);assert errors;regressions.append({'name':'wrong_scope:'+name,'sample_id':r['sample_id'],'rejected_errors':errors})
    for entry in controls[:1]+[next(c for c in controls if c['control']['name']==n) for n in ('directed_edge_deletion_counterfactual','adjacent_action_order_counterfactual','partial_goal_extension_counterfactual','same_world_valid_goal_detour')]:
        false=copy.deepcopy(entry['control']);false['changed_task_fields']=['start'];errors=control_errors(false,rs,preds);assert errors;regressions.append({'name':'false_changed_fields:'+false['name'],'control_id':false['control_id'],'rejected_errors':errors})
    lexical=[]
    for r in base_rows:
        t=r['semantic_task'];p=preds[r['sample_id']]
        if t['kind']!='graph' or p['search_evidence']['minimum_route_multiplicity']['route_count']<2:continue
        names=sorted(t['nodes']);rename={n:f'z{len(names)-i:02}' for i,n in enumerate(names)};u=copy.deepcopy(t);u['nodes']=[rename[n] for n in t['nodes']];u['edges']=[[rename[a],rename[b],c] for a,b,c in t['edges']];u.update(start=rename[t['start']],goal=rename[t['goal']],plan=[rename[n] for n in t['plan']]);a,info=P.solve(u)
        assert a['valid']==p['expected_target']['valid'] and a['reaches_goal']==p['expected_target']['reaches_goal'] and len(a['shortest_plan'])==p['search_evidence']['fresh_minimum_moves']
        lexical.append({'origin_sample_id':r['sample_id'],'origin_record_hash':r['record_hash'],'scope':'audit_probe','task':u,'expected_target':a,'recomputed_shortest_differs_from_renamed_old':a['shortest_plan']!=[rename[n] for n in p['expected_target']['shortest_plan']]})
    booltarget=copy.deepcopy(rows[0]['verified_target']);booltarget['valid']=1;assert canon(booltarget)!=canon(preds[rows[0]['sample_id']]['expected_target']);regressions.append({'name':'target_bool_vs_integer','rejected':True})
    report={'actual_author_execution':EXEC,'completed':True,'candidate_sha256':FILES['candidate'][1],'manifest_sha256':FILES['manifest'][1],'prediction_sha256':FILES['predictions'][1],'source_sha256':source_hash,
            'read_scope':{role:{'path':p,'sha256':h} for role,(p,h) in FILES.items()},'public_sources':[{'path':p,'sha256':sha(ROOT/p)} for p in P.PUBLIC],
            'no_producer_reference_newformal_rootgate_expert_implementation_or_other_verdict_read':True,'same_model_public_guided_prior_exposure_disclosed':True,
            'rows':384,'source_families':48,'semantic_contexts':192,'exact_typed_target_matches':sum(canon(r['verified_target'])==canon(preds[r['sample_id']]['expected_target']) for r in rows),
            'row_witnesses_checked':witness_count,'supplied_controls_checked':len(controls),'controls':controls,'family_checks':families,'context_checks':context_evidence,
            'prior_public_rows':len(prior),'prior_semantic_units':len(prior_units),'prior_guard_failure_counts':dict(oldguards),'prior_guard_actual_examples':oldguardexamples,
            'prior_signature_counts':{k:len(v) for k,v in oldsignatures.items()},'new_base_signature_counts':novelty_counts,'new_absent_from_prior_signature_counts':novel,'six_block_initial_shapes':len(sixshapes),
            'base_signatures':dict(signatures),'novelty_collisions':collision_evidence,'source_family_intersection_with_prior':len({r['source_family_key'] for r in rows}&prior_families),
            'family_allocation_by_stratum_split_size':[{'stratum':s,'split':sp,'size':n,'families':v} for (s,sp,n),v in sorted(allocation.items())],
            'fuel_subtypes':dict(subtype),'temporary_displacement_C0_bases':c0_displace,'minimum_route_tie_B0_bases':b0_ties,'C1_first_failure_index_counts':dict(c1_indices),
            'executed_rejected_corruptions':regressions,'lexical_tie_actual_renaming_probes':lexical,
            'detailed_issues':dict(defects),'critical_defect_rows':len(defects),
            'scope_limits':['384 correlated views are48 source instances and192 semantic contexts, not384 independent worlds or mechanisms. Two existing action grammars.',
                            'Same-model/common-design/public guidance/prior reviewer exposure; separate executions do not establish independent error distributions.',
                            'All32 development semantic inputs have feasible fresh goals; constant-feasible baseline32/32. This limits dev infeasibility assessment, not correct targets or manifest claims.',
                            'Actual new no-op/destination-clear mutants have0 observed failures, while prior public inputs independently demonstrate both guard failures. This pilot focuses supplied controls, not every guard.',
                            'Input-only coarse validity/goal lookups top24/32 and23/32 on dev. Finite small-set associations do not establish efficacy or future robustness.',
                            'No sealed/OOD/learned-planning/efficacy claim. Levels2–4 reflect bounded horizons/controls; object count or detour length alone is not difficulty.']}
    audit=ROOT/'audits/diversity/planning-gap-v1-adversarial-canonical.json';freeze(audit,jsbytes(report))
    receipts=[]
    for r in rows:
        sid=r['sample_id'];p=preds[sid];errs=defects[sid];decision='REVISE' if errs else 'ACCEPT';scope=[{'severity':'low','code':'bounded_public_guided_proxy_scope','evidence':'Independent exact recomputation, all supplied controls, mutation/metamorphic checks and historical structural comparison; common underlying model and48 correlated source families limit inference.','fix':'Preserve scope disclosures; future efficacy needs separate evaluation custody.'}]
        if r['split']=='dev_challenge':scope.append({'severity':'low','code':'development_infeasibility_coverage_limit','evidence':'All32 development semantic contexts have feasible fresh plans; no dev null shortest_plan examples.','fix':'Use a fresh stratified future batch to assess infeasibility generalization; no change to frozen targets.'})
        if r['semantic_task']['kind']=='blocks':scope.append({'severity':'low','code':'new_batch_guard_coverage_limit','evidence':'Actual new no-op/destination-clear mutants have zero observed errors; earlier public inputs independently exercise these guards.','fix':'Keep aggregate prior coverage distinct from this batch; add matched guard controls in future work if desired.'})
        scores={'semantic_correctness':5 if not errs else 2,'target_correctness':5 if canon(r['verified_target'])==canon(p['expected_target']) else 1,
                'instruction_clarity':5,'reasoning_validity':5 if not errs else 2,'generalization_value':3,'difficulty_quality':4,'diversity':4,'shortcut_resistance':3}
        receipts.append({'sample_id':sid,'record_hash':r['record_hash'],'receipt_id':'red-planning-gap-v1-'+hashlib.sha256((sid+r['record_hash']+EXEC).encode()).hexdigest()[:20],
            'reviewer_role':'adversarial','reviewer_execution_id':EXEC,'producer_execution_id':r['producer_agent_id'],'score_scale':'1–5, higher is better',
            'scores':scores,'critical_defects':errs,'issues':errs+scope,'decision':decision,'confidence':0.98,
            'evidence':{'prediction_sha256':FILES['predictions'][1],'canonical_audit_sha256':sha(audit),'candidate_sha256':FILES['candidate'][1],'manifest_sha256':FILES['manifest'][1],
                        'target_comparison':'Exact canonical JSON, including bool/int/null distinctions, against own input-only frozen expected target.',
                        'independent_row_properties':list(p['independent_row_properties']),'fresh_minimum_moves':p['search_evidence']['fresh_minimum_moves'],'row_search_complete':p['search_evidence']['search']['complete']},
            'review_scope':'Actual same-model public-guided independent execution; no other verdicts/newformal/producer code read. ACCEPT is bounded data correctness/control-quality judgment, no release/adjudication authority or efficacy claim.'})
    first=ROOT/'reviews/adversarial/planning-gap-v1.first.jsonl';final=ROOT/'reviews/adversarial/planning-gap-v1.jsonl';freeze(first,lines(receipts));freeze(final,lines(receipts))
    print(canon({'first_canonical_judgments_frozen':True,'rows':384,'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'first_sha256':sha(first),'final_sha256':sha(final),'audit_sha256':sha(audit),'source_sha256':source_hash}),flush=True)
    execution={'actual_author_execution':EXEC,'completed':True,'source_sha256':source_hash,'candidate_sha256':FILES['candidate'][1],'manifest_sha256':FILES['manifest'][1],
               'first_prediction_sha256':FILES['predictions'][1],'first_input_judgment_sha256':FILES['first_input_judgments'][1],
               'first_canonical_judgment_sha256':sha(first),'final_judgment_sha256':sha(final),'canonical_audit_sha256':sha(audit),'rows':384,
               'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'exact_targets':report['exact_typed_target_matches'],'witnesses':witness_count,'supplied_controls':len(controls),
               'rejected_corruptions':len(regressions),'lexical_tie_renaming_probes':len(lexical),'prior_semantic_units_independently_solved':len(prior_units),'scope_limits':report['scope_limits'],
               'independence':'Actual own preparation and first expectations preceded gold/metadata; no producer/reference/newformal/rootgate/expert code or other verdict read. Same underlying model and public-guided context disclosed.'}
    dest=ROOT/'audits/execution/planning-gap-v1-adversarial.json';freeze(dest,jsbytes(execution));print(canon({'completed':True,'execution_sha256':sha(dest),'novelty':novel,'prior_guard_counts':dict(oldguards),'first_sha256':sha(first),'final_sha256':sha(final)}),flush=True)
if __name__=='__main__':main()
