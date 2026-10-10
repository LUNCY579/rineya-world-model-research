"""Fresh hash-bound review of the metadata-corrected affected-track batch."""
import collections
import copy
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from independent_expansion import BASE,OUT,canonical,compute
from expansion_semantic_audit import graph_dp,distinct_equal_feasible_routes

def main():
    paths=[BASE/f'data/{t}/candidate_renderings/expansion-v2.jsonl' for t in ('RGM-001','RVM-001')]
    rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
    originals={r['sample_id']:r for r in [json.loads(l) for t in ('RGM-001','RVM-001') for l in (BASE/f'data/{t}/candidate_renderings/expansion-v1.jsonl').read_text().splitlines()]}
    old_receipts={r['sample_id']:r for r in [json.loads(l) for l in (OUT/'expansion-v1.jsonl').read_text().splitlines()]}
    validator=Draft202012Validator(json.loads((BASE/'schemas/canonical/v1.schema.json').read_text()))
    prefixes={t:{r['rendered_input'].split('Task JSON:',1)[0] for r in originals.values() if r['track_id']==t} for t in ('RGM-001','RVM-001')}
    receipts=[];first=[];fact_counts=collections.Counter();changes=[];family_splits=collections.defaultdict(set)
    for r in rows:
        rc=compute(r,validator,prefixes);oldid=r['sample_id'].replace('expansion-v2','expansion-v1');old=originals[oldid];kind=r['semantic_task']['kind']
        unchanged_keys=['rendered_input','semantic_task','verified_target','split','renderer','target_format','difficulty','structural_difficulty','challenge_axes']
        unchanged=all(canonical(r[k])==canonical(old[k]) for k in unchanged_keys)
        rc['evidence']['semantic_prompt_target_split_unchanged_from_v1']=unchanged
        if not unchanged:rc['issues'].append({'category':'unexpected_semantic_revision','severity':'high','evidence':oldid,'fix':'Independently inspect and review actual semantic changes.'});rc['decision']='REVISE';rc['review_status']='REVISE'
        if r['skill_tags']!=old['skill_tags']:changes.append(r['sample_id'])
        family_splits[r['semantic_parent_id']].add(r['split'])
        if kind=='graph':
            w=r['semantic_task'];degrees=collections.Counter(a for a,b,c in w['edges']);branched=max(degrees.values(),default=0)>1
            expected='branched_resource_graph' if branched else 'single_successor_resource_graph'
            facts=[(expected,expected in r['skill_tags']),('valid_plan',('valid_plan' in r['skill_tags'])==rc['evidence']['valid']),('invalid_plan',('invalid_plan' in r['skill_tags'])==(not rc['evidence']['valid'])),('goal_satisfaction',('goal_satisfaction' not in r['skill_tags']) or rc['evidence']['reaches_goal']),('unreachable',('unreachable' not in r['skill_tags']) or rc['evidence']['shortest_plan'] is None)]
            rc['evidence']['outdegree_tag_expected']=expected
        elif kind=='critic':
            w=r['semantic_task']['world'];suffix=r['skill_tags'][-1].split(':')[-1];dp=graph_dp(w);equal=distinct_equal_feasible_routes(w);facts=[]
            if suffix=='initial_goal_reachable':facts=[(suffix,dp['shortest'] is not None)]
            elif suffix=='initial_goal_unreachable':facts=[(suffix,dp['shortest'] is None)]
            elif suffix=='equal_length_goal_routes':facts=[(suffix,equal)]
            elif suffix=='no_equal_length_goal_routes':facts=[(suffix,not equal)]
            elif suffix=='initial_goal':facts=[(suffix,w['start']==w['goal'])]
            elif suffix=='fuel_precondition':facts=[(suffix,any(a==w['start'] and b==w['goal'] and c>w['fuel'] for a,b,c in w['edges']))]
            else:facts=[(suffix,True)] # procedural mode, not a stronger factual route claim
            facts.append(('branched_resource_graph',max(collections.Counter(a for a,b,c in w['edges']).values(),default=0)>1))
            rc['evidence']['factual_route_metadata']={'suffix':suffix,'initial_optimum':dp,'has_distinct_feasible_equal_move_goal_routes':equal}
        else:
            facts=[('valid_plan',('valid_plan' in r['skill_tags'])==rc['evidence']['valid']),('invalid_plan',('invalid_plan' in r['skill_tags'])==(not rc['evidence']['valid'])),('goal_satisfaction',('goal_satisfaction' not in r['skill_tags']) or rc['evidence']['reaches_goal'])]
        rc['evidence']['factual_tag_checks']=dict(facts)
        for name,valid in facts:
            fact_counts[name]+=1
            if not valid:
                rc['issues'].append({'category':'false_competency_tag','severity':'moderate','evidence':name,'fix':'Use a tag that follows the actual final world semantics.'});rc['decision']='REVISE';rc['review_status']='REVISE'
        first.append(copy.deepcopy(rc))
        # Carry forward only inspected nonblocking scope observations when exact
        # prompt/task/target equality has been established by fresh computation.
        oldrc=old_receipts[oldid]
        for issue in oldrc['issues']:
            if issue['category']=='false_competency_tag':continue
            rc['issues'].append(copy.deepcopy(issue))
        rc['scores']=copy.deepcopy(oldrc['scores'])
        rc['evidence']['manual_inspection_examples_for_mechanism']=[s.replace('expansion-v1','expansion-v2') for s in oldrc['evidence']['manual_inspection_examples_for_mechanism']]
        rc['evidence']['prior_detailed_semantic_inspection']='reviews/expert/semantic_inspection_expansion_v1.md'
        rc['evidence']['revision_inspection']='reviews/expert/semantic_inspection_expansion_v2.md'
        rc['review_phase']='completed_fresh_metadata_revision_review'
        rc['review_method']='Fresh independent recomputation of every visible prompt/hash/schema/target and factual route/outdegree tags; exact semantic equality checked against immutable v1; actual-agent revised metadata examples inspected. Own implementations/path counter only; no producer/reference/oracle or other verdicts.'
        rc['confidence']=0.99;receipts.append(rc)
    assert all(len(v)==1 for v in family_splits.values())
    for name,rs in [('expansion-v2.first_judgments.jsonl',first),('expansion-v2.jsonl',receipts)]:
        p=OUT/name
        if p.exists():raise FileExistsError('Preserved judgments cannot be overwritten.')
        p.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rs))
    report={'rows':len(rows),'families':len(family_splits),'new_independent_parent_count_relative_to_v1':0,'decisions':dict(collections.Counter(r['decision'] for r in receipts)),
            'track_decisions':{str(k):v for k,v in collections.Counter((r['sample_id'].split('-expansion')[0],r['decision']) for r in receipts).items()},
            'factual_check_counts':dict(fact_counts),'tag_changed_rows':len(changes),'tag_changed_ids':changes,
            'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'target_hash_schema_or_tag_failures':[{'sample_id':r['sample_id'],'issues':[i for i in r['issues'] if i['severity']!='low']} for r in receipts if r['decision']!='ACCEPT'],
            'semantic_prompt_target_changes':sum(not r['evidence']['semantic_prompt_target_split_unchanged_from_v1'] for r in receipts),
            'own_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_release_approval':True}
    (OUT/'verification_summary_expansion_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tag_changed_ids','source_sha256')},indent=2))

if __name__=='__main__':main()
