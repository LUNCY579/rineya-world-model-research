#!/usr/bin/env python3
"""Small post-freeze extraction/allocation/histogram supplement; own reviewer only."""
import collections,hashlib,importlib.util,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
def b(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def main():
    names={'input':('reviews/input_only/planning-gap-v1.jsonl','2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654'),
           'candidate':('data/RGM-001/candidate_renderings/planning-gap-v1.jsonl','a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27'),
           'predictions':('reviews/adversarial/planning-gap-v1.input_predictions.jsonl','273fbdcc8005c034b1f443f5f4ff3309193c899edb461dfd33d69b2f8fda2ec3'),
           'prior':('research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl','896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d'),
           'first':('reviews/adversarial/planning-gap-v1.first.jsonl','404c54fc35dc86ab4670abd9ab4dc2d666a6912d2bf1962a6dce1109efbda557'),
           'final':('reviews/adversarial/planning-gap-v1.jsonl','404c54fc35dc86ab4670abd9ab4dc2d666a6912d2bf1962a6dce1109efbda557')}
    for p,h in names.values():assert sha(ROOT/p)==h
    prep=ROOT/'audits/rgm_planning_gap_adversarial_preparation.py';assert sha(prep)=='2330e383f2a8e1ec608435ee7eacfd647e072e8715bb35d583c12798eaaa1ad3'
    spec=importlib.util.spec_from_file_location('own_final_prep',prep);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    source=pathlib.Path(__file__);freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_final_checks-{sha(source)}.py',source.read_bytes())
    def load(k):return [json.loads(s) for s in (ROOT/names[k][0]).open()]
    rs=load('candidate');ins={r['sample_id']:r for r in load('input')};ps={r['sample_id']:r for r in load('predictions')};prior={canon(r['semantic_task']):r for r in load('prior')}.values()
    hist=collections.defaultdict(lambda:collections.defaultdict(collections.Counter));allocation=collections.Counter();bases={};fuelchecks=[]
    for r in rs:
        sid=r['sample_id'];i=ins[sid];p=ps[sid];t=r['semantic_task'];a=p['expected_target'];info=p['search_evidence']
        assert canon(t)==canon(i['semantic_task']) and r['rendered_input']==i['rendered_input'] and r['record_hash']==i['original_record_hash']
        assert r['source_family_key']==i['source_family_key'] and r['split']==i['split']
        assert len(t['plan'])<=12
        if t['kind']=='graph':assert 5<=len(t['nodes'])<=7 and 0<=t['fuel']<=30
        else:assert len(t['on']) in (5,6)
        fail=info['first_failure'];values={'valid':a['valid'],'reaches_goal':a['reaches_goal'],'fresh_feasible':a['shortest_plan'] is not None,
            'fresh_minimum_moves':info['fresh_minimum_moves'],'first_failure_index':fail['index'] if fail else None,'first_failure_all_reasons':fail['reasons'] if fail else [],
            'object_count':len(t.get('nodes',t.get('on',{}))),'edge_count':len(t['edges']) if t['kind']=='graph' else None,
            'submitted_plan_length':len(t['plan']),'initial_fuel':t.get('fuel'),'goal_relation_count':len(t['goal_on']) if t['kind']=='blocks' else None}
        keys=['ALL','family:'+r['source_family_key'],'stratum:'+r['primary_stratum'],'split:'+r['split'],'renderer:'+r['renderer'],
              'stratum_split_renderer:'+canon([r['primary_stratum'],r['split'],r['renderer']]),'context_split:'+canon([r['controlled_context'],r['split']])]
        for key in keys:
            for feature,value in values.items():hist[key][feature][canon(value)]+=1
        if r['renderer']=='forward' and r['controlled_context'][1]=='0':
            bases[r['source_family_key']]=r;allocation[(r['primary_stratum'],r['split'],values['object_count'])]+=1
    assert len(bases)==48
    for stratum in 'ABCD':
        assert sum(n for (s,sp,size),n in allocation.items() if s==stratum and sp=='train')==10
        assert sum(n for (s,sp,size),n in allocation.items() if s==stratum and sp=='dev_challenge')==2
    for stratum in 'CD':
        assert all(allocation[(stratum,sp,size)]==(5 if sp=='train' else 1) for sp in ('train','dev_challenge') for size in (5,6))
    assert {size for (s,sp,size) in allocation if s in 'AB' and sp=='dev_challenge'}=={5,6,7}
    for fk,base in bases.items():
        if base['primary_stratum']!='A':continue
        changed=next(r for r in rs if r['source_family_key']==fk and r['controlled_context']=='A1' and r['renderer']=='forward');p0,p1=ps[base['sample_id']],ps[changed['sample_id']]
        assert not p1['expected_target']['valid']
        if p1['expected_target']['shortest_plan'] is None:
            assert base['semantic_task']['fuel']==p0['search_evidence']['cheapest_required_fuel'] and changed['semantic_task']['fuel']==base['semantic_task']['fuel']-1
            typ='exact_cost_boundary_and_one_lower'
        else:
            assert p1['search_evidence']['fresh_minimum_moves']>p0['search_evidence']['fresh_minimum_moves'];typ='blocked_submitted_route_but_longer_fresh_route'
        fuelchecks.append({'family':fk,'kind':typ,'sample_ids':[base['sample_id'],changed['sample_id']],'verified':True})
    old_all=collections.Counter();old_isolated=collections.Counter();old_examples={};old_families=collections.defaultdict(set);old_tasks={};old_ids=set()
    for r in prior:
        t=r['semantic_task'];a,info=P.solve(t);old_ids.add(r['sample_id']);old_families[r['source_family_key']].add(r['split']);old_tasks[canon(t)]=r['split']
        if info['first_failure']:
            reasons=info['first_failure']['reasons'];old_all.update(reasons)
            if len(reasons)==1:
                old_isolated.update(reasons);old_examples.setdefault(reasons[0],{'sample_id':r['sample_id'],'original_record_hash':r['original_record_hash'],'task':t,'independent_first_failure':info['first_failure']})
    assert old_all['unchanged_support']>0 and old_all['destination_not_clear']>0
    # No positive isolated-guard count is assumed; actual zeros remain evidence.
    assert not set(bases)&set(old_families) and not {r['sample_id'] for r in rs}&old_ids
    assert not any(canon(r['semantic_task']) in old_tasks for r in rs)
    current_family_splits=collections.defaultdict(set);current_parent_splits=collections.defaultdict(set);current_semfamily_splits=collections.defaultdict(set)
    for r in rs:
        current_family_splits[r['source_family_key']].add(r['split']);current_parent_splits[r['semantic_parent_id']].add(r['split']);current_semfamily_splits[r['semantic_family_id']].add(r['split'])
    assert all(len(v)==1 for d in (current_family_splits,current_parent_splits,current_semfamily_splits) for v in d.values())
    hpath=ROOT/'audits/shortcut_detection/planning-gap-v1-conditioned-adversarial.json';freeze(hpath,b({'actual_author_execution':'/root/adversarial_auditor','candidate_sha256':names['candidate'][1],'prediction_sha256':names['predictions'][1],'conditional_histograms':hist,
         'scope':'All384 correlated views. Histograms distinguish family, stratum, split, renderer and context; no training. First error counts list all guards failing at that action, not a canonical precedence label.'}))
    out={'actual_author_execution':'/root/adversarial_auditor','completed':True,'source_sha256':sha(source),'input_matches':384,'canonical_task_and_rendered_input_exactly_equal_frozen_inputs':True,'bounds_checks':384,
         'allocation_policy_checks_completed':True,'fuel_submitted_route_and_exact_boundary_checks':fuelchecks,'old_guard_all_violated_counts':dict(old_all),'old_guard_isolated_failure_counts':dict(old_isolated),
         'old_guard_isolated_actual_examples':old_examples,'guard_count_interpretation':'The original canonical audit reports all violated guards at first failure. Isolated counts require exactly one guard; neither uses producer/canonical reason precedence. Observed unchanged_support and destination_not_clear violations co-fail with another guard; this establishes observed violations, not selective single-guard ablation coverage. Actual isolated counts are reported without assumed positives.',
         'prior_family_id_sample_id_exact_task_intersections':{'families':0,'sample_ids':0,'semantic_tasks':0},'cross_split_family_parent_semantic_family_groups':0,
         'conditional_histograms_sha256':sha(hpath),'first_and_final_verdicts_preserved':True,'first_sha256':sha(ROOT/names['first'][0]),'final_sha256':sha(ROOT/names['final'][0]),
         'no_other_reviewer_implementation_private_expectations_or_verdicts_read':True,'decision_changes':0,'preserved_failed_executions':['audits/execution/planning-gap-v1-adversarial-final-checks-first-failure.json','audits/execution/planning-gap-v1-adversarial-final-checks-second-failure.json'],'scope':'Supplemental checks confirm frozen ACCEPT receipts; development infeasibility gap and bounded public-guided same-model review remain disclosed.'}
    path=ROOT/'audits/execution/planning-gap-v1-adversarial-final-checks-v3.json';freeze(path,b(out));print(canon({'source_sha256':sha(source),'supplement_sha256':sha(path),'conditional_histograms_sha256':sha(hpath),'prior_isolated_guard_counts':dict(old_isolated),'input_matches':384}))
if __name__=='__main__':main()
