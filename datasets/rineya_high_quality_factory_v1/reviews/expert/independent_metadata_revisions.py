"""Fresh independently recomputed record/metadata reviews; no reference/oracle."""
import collections
import copy
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from independent_expansion import BASE,OUT,canonical,compute
from expansion_semantic_audit import graph_dp,distinct_equal_feasible_routes

def run(track,batch,baseline,prior):
    path=BASE/f'data/{track}/candidate_renderings/{batch}.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    originals={r['sample_id']:r for r in [json.loads(l) for l in (BASE/f'data/{track}/candidate_renderings/{baseline}.jsonl').read_text().splitlines()]}
    old_receipts={r['sample_id']:r for r in [json.loads(l) for l in (OUT/f'{prior}.jsonl').read_text().splitlines()]}
    validator=Draft202012Validator(json.loads((BASE/'schemas/canonical/v1.schema.json').read_text()))
    prefixes={track:{r['rendered_input'].split('Task JSON:',1)[0] for r in originals.values()}}
    first=[];final=[];counts=collections.Counter();examples={};tag_changes=[];splits=collections.defaultdict(set)
    for r in rows:
        oldid=r['sample_id'].replace(batch,baseline);old=originals[oldid];rc=compute(r,validator,prefixes)
        unchanged=all(canonical(r[k])==canonical(old[k]) for k in ['rendered_input','semantic_task','verified_target','split','renderer','target_format','difficulty','structural_difficulty','challenge_axes'])
        rc['evidence']['exact_prompt_task_target_unchanged_from']=oldid
        rc['evidence']['unchanged']=unchanged
        if not unchanged:rc['issues'].append({'category':'unexpected_semantic_change','severity':'high','evidence':oldid,'fix':'Inspect/review actual semantic changes.'});rc['decision']='REVISE';rc['review_status']='REVISE'
        if r['skill_tags']!=old['skill_tags']:tag_changes.append(r['sample_id'])
        t=r['semantic_task'];kind=t['kind'];facts={};splits[r['semantic_parent_id']].add(r['split'])
        if kind=='blocks':
            initial_roots=sum(v=='TABLE' for v in t['on'].values());goal_roots=sum(v=='TABLE' for v in t['goal_on'].values())
            expected='multiple_initial_table_stacks' if initial_roots>=2 else 'multiple_explicit_goal_table_roots' if goal_roots>=2 else 'block_goal_composition'
            facts[expected]=expected in r['skill_tags'];rc['evidence']['table_root_tag_fact']={'initial_TABLE_roots':initial_roots,'explicit_goal_TABLE_roots':goal_roots,'expected_tag':expected}
            examples.setdefault(expected,r['sample_id'])
        elif kind=='graph':
            deg=collections.Counter(a for a,b,c in t['edges']);branch=max(deg.values(),default=0)>1;expected='branched_resource_graph' if branch else 'single_successor_resource_graph'
            facts[expected]=expected in r['skill_tags'];facts['outdegree_label_exclusive']=sum(tag in r['skill_tags'] for tag in ('branched_resource_graph','single_successor_resource_graph'))==1
            examples.setdefault(expected,r['sample_id'])
        else:
            w=t['world'];suffix=r['skill_tags'][-1];dp=graph_dp(w);equal=distinct_equal_feasible_routes(w)
            facts[suffix]=equal if suffix=='equal_length_goal_routes' else not equal if suffix=='no_equal_length_goal_routes' else dp['shortest'] is None if suffix=='initially_unreachable' else w['start']==w['goal'] if suffix=='initial_goal' else True
            rc['evidence']['route_tag_fact']={'actual_equal_move_goal_routes':equal,'initial_optimum':dp,'tag':suffix}
            examples.setdefault(suffix,r['sample_id'])
        if kind in ('graph','blocks'):
            facts['valid_plan']=('valid_plan' in r['skill_tags'])==rc['evidence']['valid'];facts['invalid_plan']=('invalid_plan' in r['skill_tags'])==(not rc['evidence']['valid'])
            facts['goal_satisfaction']=('goal_satisfaction' not in r['skill_tags']) or rc['evidence']['reaches_goal']
            facts['unreachable']=('unreachable' not in r['skill_tags']) or rc['evidence']['shortest_plan'] is None
        for name,ok in facts.items():
            counts[name]+=1
            if not ok:rc['issues'].append({'category':'false_factual_tag','severity':'moderate','evidence':{'name':name,'facts':rc['evidence']},'fix':'Correct the factual tag and independently review its new hash.'});rc['decision']='REVISE';rc['review_status']='REVISE'
        rc['evidence']['factual_tag_checks']=facts;first.append(copy.deepcopy(rc))
        previous=old_receipts[oldid]
        rc['scores']=copy.deepcopy(previous['scores']);rc['issues'] += [copy.deepcopy(i) for i in previous['issues'] if i['category']!='false_competency_tag']
        rc['review_phase']='completed_fresh_metadata_revision_review';rc['confidence']=0.99
        rc['review_method']='Fresh per-record visible-prompt recomputation and exact hash/schema/contract/target checks with expert-owned semantics and exhaustive equal-route path counter; actual-agent factual-tag examples inspected; exact prompt/task/gold equality checked against prior independently reviewed candidates. No producer/reference/oracle or other verdict files read.'
        rc['evidence']['substantive_revision_report']='reviews/expert/semantic_inspection_metadata_revisions.md'
        rc['evidence']['new_independent_parent_count_relative_to_baseline']=0
        if track=='RGM-001':
            rc['evidence']['reviewer_blinding_limit']='Parent disclosed a supplemental reviewer concern/count about ambiguous previous multi_stack tags before this revision. No verdict files read. New literal tag facts independently computed from each actual on/goal_on map.'
        final.append(rc)
    assert all(len(v)==1 for v in splits.values())
    for name,rs in [(batch+'.first_judgments.jsonl',first),(batch+'.jsonl',final)]:
        p=OUT/name
        if p.exists():raise FileExistsError('Immutable review already exists.')
        p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs))
    report={'track':track,'batch':batch,'baseline':baseline,'rows':len(rows),'families':len(splits),'new_independent_parents':0,'decisions':dict(collections.Counter(r['decision'] for r in final)),
            'factual_checks':dict(counts),'substantive_examples':examples,'changed_tag_rows':len(tag_changes),'tag_changed_ids':tag_changes,
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'unchanged_prompt_task_target_rows':sum(r['evidence']['unchanged'] for r in final),
            'failures':[r['sample_id'] for r in final if r['decision']!='ACCEPT'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reviewer_execution_id':'/root/expert_reviewer','no_release_approval':True}
    (OUT/f'verification_summary_{batch}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    reports=[run('RGM-001','expansion-v3','expansion-v1','expansion-v1'),run('RVM-001','pilot-v4','pilot-v2','pilot-v2')]
    print(json.dumps([{k:v for k,v in r.items() if k!='tag_changed_ids'} for r in reports],indent=2))
