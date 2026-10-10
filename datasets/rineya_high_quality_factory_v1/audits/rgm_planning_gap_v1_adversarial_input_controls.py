#!/usr/bin/env python3
"""Input-only pair/field-order audit; actual author /root/adversarial_auditor."""
import collections,hashlib,itertools,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
def main():
    inp=ROOT/'reviews/input_only/planning-gap-v1.jsonl';predp=ROOT/'reviews/adversarial/planning-gap-v1.input_predictions.jsonl'
    assert sha(inp)=='2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654'
    assert sha(predp)=='273fbdcc8005c034b1f443f5f4ff3309193c899edb461dfd33d69b2f8fda2ec3'
    rs=[json.loads(s) for s in inp.open()];preds={r['sample_id']:r for r in map(json.loads,predp.open())}
    source=pathlib.Path(__file__);freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_input_controls-{sha(source)}.py',source.read_bytes())
    tasks=collections.defaultdict(list);fams=collections.defaultdict(list)
    for r in rs:tasks[canon(r['semantic_task'])].append(r)
    top_orders=collections.Counter();nested_orders=collections.Counter();field_pairs=[]
    for twins in tasks.values():
        assert len(twins)==2 and len({x['split'] for x in twins})==1 and len({x['source_family_key'] for x in twins})==1
        payloads=[json.loads(x['rendered_input'].split('Task JSON:\n')[1]) for x in twins]
        orders=[list(x) for x in payloads];assert orders[0]==list(reversed(orders[1]))
        for p in payloads:top_orders[canon(list(p))]+=1
        t=twins[0]['semantic_task']
        if t['kind']=='blocks':
            for k in ('on','goal_on'):
                assert list(payloads[0][k])==list(reversed(list(payloads[1][k])))
                nested_orders[k]+=1
        assert canon(preds[twins[0]['sample_id']]['expected_target'])==canon(preds[twins[1]['sample_id']]['expected_target'])
        fams[twins[0]['source_family_key']].append(twins[0]);field_pairs.append([x['sample_id'] for x in twins])
    relationships=[];diffcounts=collections.Counter();split_sizes=collections.Counter();context_stats=collections.defaultdict(collections.Counter)
    for fam,units in fams.items():
        assert len(units)==4
        for r in units:
            t=r['semantic_task'];context=r['sample_id'].rsplit('-',2)[1];p=preds[r['sample_id']];a=p['expected_target'];info=p['search_evidence']
            split_sizes[(context[0],r['split'],len(t.get('nodes',t.get('on',{}))))]+=1
            context_stats[context].update({'rows':1,'valid':int(a['valid']),'reaches_goal':int(a['reaches_goal']),
                                            'fresh_feasible':int(a['shortest_plan'] is not None),
                                            'at_least_two_graph_min_routes':int(t['kind']=='graph' and info['minimum_route_multiplicity']['route_count']>=2)})
        for left,right in itertools.combinations(units,2):
            a,b=left['semantic_task'],right['semantic_task'];changed=sorted(k for k in a if canon(a[k])!=canon(b[k]));diffcounts[','.join(changed)]+=1
            ca,cb=preds[left['sample_id']],preds[right['sample_id']];relationship=[]
            if changed==['fuel']:relationship.append('actual_only_fuel_changes')
            if changed==['edges']:
                ea={canon(e) for e in a['edges']};eb={canon(e) for e in b['edges']}
                if (ea<eb or eb<ea) and len(ea^eb)==1:relationship.append('actual_one_directed_edge_deletion')
            if changed==['plan'] and a['kind']=='blocks' and len(a['plan'])==len(b['plan']):
                diffs=[i for i,(x,y) in enumerate(zip(a['plan'],b['plan'])) if x!=y]
                if len(diffs)==2 and diffs[1]==diffs[0]+1 and a['plan'][diffs[0]]==b['plan'][diffs[1]] and a['plan'][diffs[1]]==b['plan'][diffs[0]]:relationship.append('actual_one_adjacent_action_swap')
            if changed==['goal_on']:
                ga=set(a['goal_on'].items());gb=set(b['goal_on'].items())
                if ga<gb or gb<ga:relationship.append('actual_only_partial_goal_extension')
            if changed==['plan'] and ca['expected_target']['reaches_goal'] and cb['expected_target']['reaches_goal'] and len(a['plan'])!=len(b['plan']):relationship.append('actual_same_world_legal_goal_detour')
            if relationship:relationships.append({'family':fam,'sample_ids':[left['sample_id'],right['sample_id']],'changed_task_fields':changed,'input_derived_relationships':relationship,
                'valid_pair':[ca['expected_target']['valid'],cb['expected_target']['valid']],
                'reaches_pair':[ca['expected_target']['reaches_goal'],cb['expected_target']['reaches_goal']],
                'fresh_minimum_moves_pair':[ca['search_evidence']['fresh_minimum_moves'],cb['search_evidence']['fresh_minimum_moves']]})
    out={'actual_author_execution':'/root/adversarial_auditor','stage':'INPUT_ONLY','input_sha256':sha(inp),'prediction_sha256':sha(predp),'source_sha256':sha(source),
         'views':384,'semantic_units':192,'families':48,'field_order_pairs':192,'top_key_orders':dict(top_orders),'nested_reverse_pair_counts':dict(nested_orders),
         'field_pair_exact_targets_invariant':True,'field_order_pair_sample_ids':field_pairs,'context_stats_from_join_ids_not_predictor_features':dict(context_stats),
         'stratum_split_size_counts':[{'stratum':k[0],'split':k[1],'size':k[2],'semantic_units':v} for k,v in sorted(split_sizes.items())],
         'within_family_pair_changed_field_counts':dict(diffcounts),'actual_input_relationship_counts':dict(collections.Counter(k for r in relationships for k in r['input_derived_relationships'])),
         'actual_input_relationships':relationships,'scope':'No control manifest read. Pair relationships here prove supplied input field relationships, not canonical metadata claims, exclusive controls, historical source novelty or dataset acceptance.'}
    dest=ROOT/'audits/diversity/planning-gap-v1-input-controls.json';freeze(dest,(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode());print(canon({'source_sha256':sha(source),'artifact_sha256':sha(dest),'relationship_counts':out['actual_input_relationship_counts'],'field_order_pairs':192}))
if __name__=='__main__':main()
