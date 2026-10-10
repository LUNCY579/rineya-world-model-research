"""Blind expansion recomputation. Imports only this reviewer's own semantics."""
import collections
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from independent_review import causal, graph, blocks, critic, same, SCORE_KEYS, normalized_tree

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'reviews/expert'

def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def sha(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()

def depth(n):
    return 0 if 'op' not in n else 1+max([depth(a) for a in n['args']] or [0])

def features(t,result):
    kind=t['kind']
    if kind=='causal':
        return {'horizon':len(t['events']),'rule_depth':depth(t['rule']),'precondition_depth':depth(t['precondition']),
                'root':t['rule'].get('op','literal'),'gate_values':[s['cmp'] for s in result['steps']],
                'executed_operations':[s['calc']['operation'] for s in result['steps']],
                'independent_after':result['after'],'unchanged':same(t['state'],result['after'])}
    if kind in ('graph','blocks'):
        f={'supplied_plan_length':len(t['plan']),'legal_steps':[s['legal'] for s in result['steps']],
           'independent_after':result['after'],'valid':result['valid'],'reaches_goal':result['reaches_goal'],
           'shortest_plan':result['shortest_plan'],'optimal_distance':None if result['shortest_plan'] is None else len(result['shortest_plan'])}
        if kind=='graph':
            deg=collections.Counter(e[0] for e in t['edges'])
            f.update({'nodes':len(t['nodes']),'edges':len(t['edges']),'branch_nodes':sum(v>1 for v in deg.values()),'outdegree_multiset':sorted(deg.get(n,0) for n in t['nodes'])})
        else:
            f.update({'objects':len(t['on']),'initial_table_roots':sum(v=='TABLE' for v in t['on'].values()),'goal_terms':len(t['goal_on'])})
        return f
    ev=result['evaluations']
    return {'candidate_count':len(t['candidates']),'nodes':len(t['world']['nodes']),'edges':len(t['world']['edges']),
            'independent_evaluations':ev,'independent_ranking':result['ranking'],
            'first_error_positions':[e['first_error'] for e in ev],
            'goal_candidates':sum(e['reaches_goal'] for e in ev),'wrong_claim_candidates':sum(not e['claim_correct'] for e in ev),
            'invalid_candidates':sum(not e['valid'] for e in ev),'exact_tie_groups':sum(len(g)>1 for g in result['ranking'])}

def compute(r,validator,old_prefixes):
    issues=[];scores=dict(zip(SCORE_KEYS,(5,5,5,5,3,4,3,3)));evidence={}
    def issue(cat,severity,ev,fix):issues.append({'category':cat,'severity':severity,'evidence':ev,'fix':fix})
    calculated_hash=sha({k:v for k,v in r.items() if k!='record_hash'})
    evidence['hash_match']=calculated_hash==r['record_hash']
    if not evidence['hash_match']:issue('record_hash','high',calculated_hash,'Restore exact candidate content/hash binding.')
    errors=list(validator.iter_errors(r));evidence['schema_match']=not errors
    if errors:issue('canonical_schema','high',[e.message for e in errors],'Repair canonical shape.')
    try:
        prefix,payload=r['rendered_input'].split('Task JSON:',1)
        evidence['contract_matches_previously_reviewed_public_prompt']=prefix in old_prefixes[r['track_id']]
        task=json.loads(payload);evidence['prompt_task_match']=same(task,r['semantic_task'])
        if not evidence['prompt_task_match']:issue('prompt_payload_mismatch','high','Visible Task JSON differs from semantic_task.','Make task and student-visible payload identical.')
        result={'causal':causal,'graph':graph,'blocks':blocks,'critic':critic}[task['kind']](task)
        target={'after':result['after']} if r['target_format']=='answer_only' else result
        evidence.update(features(task,result));evidence['target_match']=same(target,r['verified_target'])
        if not evidence['target_match']:
            issue('target_correctness','high',{'independent_target':target,'candidate_target':r['verified_target']},'Regenerate correct target and independently review changed record.');scores['target_correctness']=1
        evidence['difficulty']=r['difficulty'];evidence['challenge_axes']=r.get('challenge_axes',[])
        if r.get('structural_difficulty')!=r['difficulty']:issue('difficulty_metadata','medium','difficulty and structural_difficulty differ.','Use one honest structural difficulty label.')
        if r['difficulty']==7:issue('difficulty_inflation','medium','No held-out renderer or objective/corruption-function crossing established.','Use structural complexity level and separately describe development shifts.')
    except Exception as exc:
        issue('malformed_or_underspecified_task','high',str(exc),'Repair task/contract then independently recompute.');scores['semantic_correctness']=1;scores['target_correctness']=1
    blocked=any(i['severity'] in ('medium','high','critical') for i in issues)
    return {'receipt_id':'expert-'+r['sample_id']+'-'+r['record_hash'][:12],
            'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_role':'expert_reviewer',
            'reviewer_execution_id':'/root/expert_reviewer','review_status':'REVISE' if blocked else 'ACCEPTED',
            'scores':scores,'critical_defects':[i for i in issues if i['severity'] in ('high','critical')],
            'issues':issues,'decision':'REVISE' if blocked else 'ACCEPT','confidence':0.98,
            'evidence':evidence,'review_method':'Independent per-record recomputation from visible prompt using expert-owned implementations; first checkpoint before substantive structural inspection.',
            'review_phase':'programmatic_first_judgment'}

def run():
    paths=[BASE/f'data/{t}/candidate_renderings/expansion-v1.jsonl' for t in ('RCM-001','RGM-001','RVM-001')]
    rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
    validator=Draft202012Validator(json.loads((BASE/'schemas/canonical/v1.schema.json').read_text()))
    old_prefixes={t:{json.loads(l)['rendered_input'].split('Task JSON:',1)[0] for l in (BASE/f'data/{t}/candidate_renderings/pilot-v2.jsonl').read_text().splitlines()} for t in ('RCM-001','RGM-001','RVM-001')}
    receipts=[compute(r,validator,old_prefixes) for r in rows]
    initial=OUT/'expansion-v1.first_judgments.jsonl'
    if initial.exists():raise FileExistsError('First judgments are immutable.')
    initial.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in receipts))
    report={'rows':len(rows),'hash_mismatches':sum(not r['evidence']['hash_match'] for r in receipts),
            'schema_mismatches':sum(not r['evidence']['schema_match'] for r in receipts),
            'target_mismatches':sum(not r['evidence'].get('target_match') for r in receipts),
            'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'counts':dict(collections.Counter(r['decision'] for r in receipts)),
            'issues':[{'sample_id':r['sample_id'],'issues':r['issues']} for r in receipts if r['issues']],
            'contract_prefix_mismatch_ids':[r['sample_id'] for r in receipts if not r['evidence'].get('contract_matches_previously_reviewed_public_prompt')]}
    (OUT/'expansion-v1.first_judgments_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('issues','source_sha256')},indent=2))

if __name__=='__main__':run()
