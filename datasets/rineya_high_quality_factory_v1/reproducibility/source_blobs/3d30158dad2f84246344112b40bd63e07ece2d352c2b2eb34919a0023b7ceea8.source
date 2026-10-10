"""Independent targeted claim review from public visible prompts only."""
import collections
import copy
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from independent_expansion import BASE,OUT,canonical,compute
from independent_review import critic,graph_replay,same
from expansion_semantic_audit import graph_dp,distinct_equal_feasible_routes

SOURCE=BASE/'data/RVM-001/candidate_renderings/claim-gap-v2.jsonl'

def own_signed_claims(task,computed):
    labels={e['candidate_id']:e for e in computed['evaluations']}
    grouped=collections.defaultdict(list)
    all_candidates=[]
    for c in task['candidates']:
        replay=graph_replay(task['world'],c['plan']);after=replay['after'];e=labels[c['candidate_id']]
        delta=c['claimed_after']['fuel']-after['fuel'];node_ok=c['claimed_after']['node']==after['node']
        data={'candidate_id':c['candidate_id'],'plan':c['plan'],'actual_after':after,'claimed_after':c['claimed_after'],
              'fuel_delta':delta,'location_correct':node_ok,'valid':e['valid'],'claim_correct':e['claim_correct'],'first_error':e['first_error'],
              'progress':e['progress'],'usefulness':e['usefulness'],'remaining_moves':e['remaining_moves'],'reaches_goal':e['reaches_goal']}
        grouped[canonical(c['plan'])].append(data);all_candidates.append(data)
    controls={}
    for planlen in (0,1):
        possible=[]
        for key,g in grouped.items():
            if len(json.loads(key))!=planlen:continue
            types={('correct' if x['claim_correct'] else 'wrong_location' if not x['location_correct'] and x['fuel_delta']==0 else 'over' if x['location_correct'] and x['fuel_delta']>0 else 'under' if x['location_correct'] and x['fuel_delta']<0 else 'other') for x in g if x['valid']}
            if {'correct','over','under','wrong_location'}.issubset(types):possible.append({'plan':json.loads(key),'cases':g})
        controls[str(planlen)]=possible
    return all_candidates,controls

def run():
    rows=[json.loads(l) for l in SOURCE.read_text().splitlines()]
    validator=Draft202012Validator(json.loads((BASE/'schemas/canonical/v1.schema.json').read_text()))
    prior_paths=[BASE/'data/RVM-001/candidate_renderings/expansion-v2.jsonl',BASE/'data/RVM-001/candidate_renderings/pilot-v4.jsonl']
    old=[json.loads(l) for p in prior_paths for l in p.read_text().splitlines()]
    prefixes={'RVM-001':{r['rendered_input'].split('Task JSON:',1)[0] for r in old}}
    first=[];final=[];counts=collections.Counter();family_groups=collections.defaultdict(list);details={};new_worlds={}
    old_worlds={canonical({k:(sorted(v) if k in ('nodes','edges') else v) for k,v in r['semantic_task']['world'].items() if k not in ('plan','kind')}):r['sample_id'] for r in old}
    for r in rows:
        rc=compute(r,validator,prefixes);task=json.loads(r['rendered_input'].split('Task JSON:',1)[1]);computed=critic(task)
        signed,controls=own_signed_claims(task,computed);mode=next(tag.split(':')[-1] for tag in r['skill_tags'] if tag.startswith('branched_resource_graph:'))
        world=task['world'];dp=graph_dp(world);equal=distinct_equal_feasible_routes(world);degree=collections.Counter(a for a,b,c in world['edges'])
        facts={'branched_resource_graph':max(degree.values(),default=0)>1,
               'empty_matched_signed_and_location_controls':bool(controls['0']),
               'one_step_matched_signed_and_location_controls':bool(controls['1'])}
        if mode=='initial_goal_reachable':facts[mode]=dp['shortest'] is not None
        elif mode=='initial_goal_unreachable':facts[mode]=dp['shortest'] is None
        elif mode=='initial_goal':facts[mode]=world['start']==world['goal']
        elif mode=='equal_length_goal_routes':facts[mode]=equal
        elif mode=='no_equal_length_goal_routes':facts[mode]=not equal
        elif mode=='fuel_precondition':facts[mode]=any(a==world['start'] and b==world['goal'] and cost>world['fuel'] for a,b,cost in world['edges'])
        # Full ranking tie checks and signed direction consistency on controls.
        rank={cid:i for i,g in enumerate(computed['ranking']) for cid in g}
        for length,sets in controls.items():
            for group in sets:
                correct=[x for x in group['cases'] if x['claim_correct']]
                wrong=[x for x in group['cases'] if not x['claim_correct']]
                assert all(rank[a['candidate_id']]<rank[b['candidate_id']] for a in correct for b in wrong)
                assert len({rank[x['candidate_id']] for x in wrong})==1
                independent=['valid','remaining_moves','progress','usefulness','reaches_goal']
                assert all(all(same(g[k],group['cases'][0][k]) for k in independent) for g in group['cases'])
        for name,ok in facts.items():
            if not ok:rc['issues'].append({'category':'false_tag_or_missing_control','severity':'moderate','evidence':name,'fix':'Correct literal tags or add complete valid matched claims.'});rc['decision']='REVISE';rc['review_status']='REVISE'
        # Every candidate remaining distance checked with a second algorithm.
        for s in signed:
            assert s['remaining_moves']==graph_dp(world,s['actual_after'])['distance']
            counts['candidate_DP_distances']+=1
        # Reverse opaque ID order and submission order; evaluation follows new
        # order, while each ranking tie remains alphabetically ordered by ID.
        shuffled=copy.deepcopy(task);mapping={c['candidate_id']:f'renamed-{len(task["candidates"])-i:03d}' for i,c in enumerate(task['candidates'])}
        for c in shuffled['candidates']:c['candidate_id']=mapping[c['candidate_id']]
        shuffled['candidates'].reverse();expect=copy.deepcopy(computed)
        for e in expect['evaluations']:e['candidate_id']=mapping[e['candidate_id']]
        expect['evaluations'].reverse();expect['ranking']=[sorted(mapping[c] for c in group) for group in expect['ranking']]
        assert same(critic(shuffled),expect)
        counts['ID_and_candidate_order_metamorphics']+=1
        # Lex-order-preserving graph renaming and node/edge serialization reversal.
        gm={n:f'node{i:02d}' for i,n in enumerate(sorted(world['nodes']))};renamed=copy.deepcopy(task)
        rw=renamed['world'];rw['nodes']=[gm[n] for n in reversed(world['nodes'])];rw['edges']=[[gm[a],gm[b],c] for a,b,c in reversed(world['edges'])]
        rw['start']=gm[world['start']];rw['goal']=gm[world['goal']];rw['plan']=[gm[n] for n in world['plan']]
        for c in renamed['candidates']:c['plan']=[gm[n] for n in c['plan']];c['claimed_after']['node']=gm[c['claimed_after']['node']]
        exp=copy.deepcopy(computed)
        for e in exp['evaluations']:
            if e['repair_plan'] is not None:e['repair_plan']=[gm[n] for n in e['repair_plan']]
        assert same(critic(renamed),exp);counts['graph_rename_serialization_metamorphics']+=1
        family_groups[r['semantic_family_id']].append(r)
        rc['evidence'].update({'factual_tag_checks':facts,'mode':mode,'own_initial_path_count':dp,'has_equal_length_goal_routes':equal,
            'signed_claim_candidates':signed,'matched_control_plan_lengths':[int(k) for k,v in controls.items() if v],
            'matched_control_sets':controls,'ID_order_metamorphic':True,'graph_renaming_metamorphic':True,'second_distance_algorithm_match':True})
        first.append(copy.deepcopy(rc))
        # Preserve quality scope honestly: this fills a specific signed/location
        # gap but repeats a finite corruption matrix and the existing graph rules.
        rc['scores'].update({'generalization_value':3,'diversity':3,'shortcut_resistance':4,'difficulty_quality':4})
        rc['issues'].append({'category':'targeted_proxy_scope','severity':'low','evidence':'Matched empty/one-step claim controls add signed fuel and legal wrong-location errors; worlds retain the existing explicit graph dynamics, objective and recurring illegal plan patterns. Twelve sourcefamilies do not imply new mechanism classes or measured model benefit.','fix':'Describe signed/location claim-gap coverage and measure any model benefit separately; add independent later-error/novel-objective tests for broader critic generalization.'})
        if r['difficulty']==3 and max(len(c['plan']) for c in task['candidates'])>2:
            rc['scores']['difficulty_quality']=3
            rc['issues'].append({'category':'difficulty_boundary','severity':'low','evidence':{'label':r['difficulty'],'candidate_max_horizon':max(len(c['plan']) for c in task['candidates'])},'fix':'Calibrate conservative labels against the entire candidate evaluation/ranking demand.'})
        rc['review_phase']='completed_initial_blind_claim_gap_review';rc['confidence']=0.99
        rc['review_method']='Independent visible-prompt recomputation of every row and candidate; exact hash/schema/contract/payload/gold and literal tags; exhaustive signed matched-control/ranking checks, own second-algorithm distance computation and ID/graph metamorphics; actual-agent all12-family semantic inspection. No claim-gap-v1, producer/reference/oracle or other verdicts read.'
        rc['evidence']['actual_semantic_inspection_report']='reviews/expert/semantic_inspection_claim_gap_v2.md';final.append(rc)
        if r['renderer']=='forward':
            details[r['sample_id']]={'world':world,'candidate_count':len(task['candidates']),'signed_claims':signed,'controls':controls,'ranking':computed['ranking'],'mode':mode,'initial_optimum':dp}
            for s in signed:
                if s['valid'] and s['location_correct']:
                    counts['valid_same_location_sign_'+str((s['fuel_delta']>0)-(s['fuel_delta']<0))]+=1
                elif s['valid'] and not s['location_correct']:counts['valid_wrong_location']+=1
                else:counts['invalid']+=1
            worldkey=canonical({k:(sorted(v) if k in ('nodes','edges') else v) for k,v in world.items() if k not in ('plan','kind')})
            new_worlds[r['semantic_parent_id']]={'key':worldkey,'prior_exact_world_match':old_worlds.get(worldkey)}
    for fam,group in family_groups.items():
        assert len(group)==2 and {r['renderer'] for r in group}=={'forward','reverse'}
        assert len({r['split'] for r in group})==1 and len({r['source_family_key'] for r in group})==1
        assert same(group[0]['semantic_task'],group[1]['semantic_task'])
    for name,records in [('claim-gap-v2.first_judgments.jsonl',first),('claim-gap-v2.jsonl',final)]:
        p=OUT/name
        if p.exists():raise FileExistsError('Immutable judgments already exist.')
        p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    report={'reviewer_execution_id':'/root/expert_reviewer','rows':len(rows),'sourcefamilies':len(family_groups),'unique_new_graph_worlds':len({v['key'] for v in new_worlds.values()}),
            'prior_exact_graph_world_matches':{k:v['prior_exact_world_match'] for k,v in new_worlds.items() if v['prior_exact_world_match']},
            'source_family_keys':sorted({r['source_family_key'] for r in rows}),'splits':dict(collections.Counter(r['split'] for r in rows)),
            'decisions':dict(collections.Counter(r['decision'] for r in final)),'counts':dict(counts),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'candidate_failures':[r['sample_id'] for r in final if r['decision']!='ACCEPT'],'family_details':details,
            'own_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_model_performance_or_release_approval':True}
    (OUT/'verification_summary_claim_gap_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='family_details'},indent=2))

if __name__=='__main__':run()
