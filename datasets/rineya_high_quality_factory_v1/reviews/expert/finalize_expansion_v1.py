"""Finalize first blind expansion inspection, retaining hash-bound first checkpoint."""
import collections
import copy
import hashlib
import json
from pathlib import Path
from independent_expansion import BASE, OUT, canonical

EXAMPLES={
 'causal':['RCM-001-expansion-v1-p0030-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0031-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0032-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0033-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0034-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0035-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0038-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0039-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0048-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0049-twin1-explicit_trace-forward','RCM-001-expansion-v1-p0058-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0059-twin0-explicit_trace-forward','RCM-001-expansion-v1-p0063-twin0-explicit_trace-reverse'],
 'graph':['RGM-001-expansion-v1-p0030-base-forward','RGM-001-expansion-v1-p0031-base-forward','RGM-001-expansion-v1-p0033-base-forward','RGM-001-expansion-v1-p0038-base-forward','RGM-001-expansion-v1-p0038-matched-forward','RGM-001-expansion-v1-p0040-base-forward','RGM-001-expansion-v1-p0058-base-reverse'],
 'blocks':['RGM-001-expansion-v1-p0032-base-forward','RGM-001-expansion-v1-p0034-base-forward','RGM-001-expansion-v1-p0037-base-forward','RGM-001-expansion-v1-p0039-base-forward','RGM-001-expansion-v1-p0039-matched-forward','RGM-001-expansion-v1-p0179-base-reverse'],
 'critic':[f'RVM-001-expansion-v1-p{i:04d}-base-forward' for i in range(30,37)]+['RVM-001-expansion-v1-p0038-base-forward','RVM-001-expansion-v1-p0039-base-forward','RVM-001-expansion-v1-p0051-base-reverse']}

def main():
    first=[json.loads(l) for l in (OUT/'expansion-v1.first_judgments.jsonl').read_text().splitlines()]
    first_report=json.loads((OUT/'expansion-v1.first_judgments_summary.json').read_text())
    for p,expected in first_report['source_sha256'].items():assert hashlib.sha256((BASE/p).read_bytes()).hexdigest()==expected,'Original batch mutated'
    rows={r['sample_id']:r for r in [json.loads(l) for t in ('RCM-001','RGM-001','RVM-001') for l in (BASE/f'data/{t}/candidate_renderings/expansion-v1.jsonl').read_text().splitlines()]}
    audit=json.loads((OUT/'expansion-v1.semantic_audit.json').read_text())
    receipts=[]
    for initial in first:
        receipt=copy.deepcopy(initial);r=rows[receipt['sample_id']];task=r['semantic_task'];kind=task['kind'];e=receipt['evidence'];scores=receipt['scores']
        def issue(category,severity,evidence,fix):receipt['issues'].append({'category':category,'severity':severity,'evidence':evidence,'fix':fix})
        receipt['review_phase']='completed_initial_blind_semantic_review'
        receipt['review_method']='Independent visible-prompt per-record recomputation, exact hash/schema/contract checks, structural/metamorphic tests and substantive actual-agent inspection of documented representatives. Uses only expert-owned implementations, not independent oracle/reference or other reviewer verdicts.'
        e['manual_inspection_examples_for_mechanism']=EXAMPLES[kind]
        e['semantic_inspection_report']='reviews/expert/semantic_inspection_expansion_v1.md'
        e['metamorphic_audit']='reviews/expert/expansion-v1.semantic_audit.json'
        if kind=='causal':
            issue('bounded_template_scope','low','150 families instantiate six base predicate families plus optional added guards. Numeric/status changes are new instances, not new arithmetic or causal mechanisms; state remains x/y and AST is supplied.','Report supplied-rule execution scope and the actual shape/behavior counts; do not infer hidden causal discovery or open-ended rule learning.')
            if e['unchanged'] and r['target_format']=='answer_only':
                scores['shortcut_resistance']=2
                issue('answer_copy_control','low','Final equals initial state; copying input balances passes this answer-only row without gate reasoning. Linked trace and rule-only twin provide diagnostics.','Measure linked gate/trace and paired-rule performance, not this answer-only row alone.')
            if len(task['events'])>1 and all(op=='identity' for op in e['executed_operations']):
                scores['difficulty_quality']=3
                issue('state_dependency_coverage','low','All events are gated off, so this multistep instance does not exercise changing intermediate state.','Retain as a no-op control; separately evaluate rows with executed updates and changed applicability.')
            if task['rule'].get('op')=='not' and r['split']=='train':
                scores['shortcut_resistance']=min(scores['shortcut_resistance'],3)
                issue('visible_root_imbalance','low','Among138 distinct training twin tasks with top-level not,88 execute no update; this creates a partial syntactic no-op cue despite rule-only twins.','Stratify evaluation and future balancing by visible AST root and apply/no-op outcome.')
            if r['split']=='dev_challenge':
                guard_key=next(k for k in audit['guards'] if k.split('-twin')[0]==r['sample_id'].split('-twin')[0] and ('-twin1-' in k)==('-twin1-' in r['sample_id']))
                g=audit['guards'][guard_key];e['dev_guard_semantics']=g
                if g['guard_equals_amount']:
                    scores['generalization_value']=2
                    issue('composition_coverage','low',{'own_guard_trace':g,'cohort':'41/60 distinct dev twin guards constant;32/60 always true;54/60 equal the amount-only clause. New AST shapes are syntactic holdouts, while the state clause seldom affects decisions.'},'Add matched events where each OR branch is independently decisive before claiming robust held-out state-dependent composition.')
        else:
            issue('bounded_dynamics_scope','low','Explicit finite graph/block dynamics and fixed goal/rank objective are fully supplied. These verified instances do not establish generic planning, learned relational inference or successor-feature effectiveness.','Keep bounded development scope and report model effectiveness only after separate evaluation.')
            if kind=='graph':scores['diversity']=4;scores['shortcut_resistance']=4
            if kind=='graph' and r['semantic_parent_id'] in audit['false_competency_tag_families'].get('nonbranching_branched_tag',[]):
                issue('false_competency_tag','moderate',{'tag':'branched_resource_graph','outdegree_multiset':e['outdegree_multiset'],'branch_nodes':e['branch_nodes']},'Replace branched tag with a factual nonbranching-chain/control label in a fresh immutable candidate revision.')
                receipt['decision']='REVISE';receipt['review_status']='REVISE'
            if kind=='blocks':scores['diversity']=4
            if kind in ('graph','blocks') and e.get('optimal_distance',0) is not None and e['optimal_distance']>2 and r['difficulty']<=3:
                scores['difficulty_quality']=3
                issue('difficulty_boundary','low',{'label':r['difficulty'],'required_fresh_optimal_moves':e['optimal_distance']},'Calibrate against the required fresh plan search as well as submitted-plan replay, and record shortest_goal_distance explicitly.')
            if kind=='critic':
                scores['diversity']=2
                issue('corruption_shortcut_scope','low','All286 incorrect fuel-only claims across150 distinct tasks are positive overclaims; each task includes a repeated intermediate-node invalid plan and a start self-move. New branched worlds do not create new corruption classes.','Add matched underclaims, wrong-node legal claims, and varied later non-self missing-edge errors before asserting held-out corruption robustness.')
                max_len=max(len(c['plan']) for c in task['candidates'])
                if r['difficulty']==3 and max_len>2:
                    scores['difficulty_quality']=3
                    issue('difficulty_boundary','low',{'label':3,'longest_submitted_plan':max_len,'first_errors':e['first_error_positions']},'Record candidate horizon/error position and calibrate conservative labels against full critic output demands.')
                if any(tag.endswith(':initially_unreachable') for tag in r['skill_tags']) and e['independent_evaluations'][0]['repair_plan'] is not None:
                    issue('false_competency_tag','moderate',{'tag':'branched_resource_graph:initially_unreachable','independent_feasible_fresh_plan':e['independent_evaluations'][0]['repair_plan']},'Replace the false tag with a factual reachable motif label in a new immutable candidate revision, or change the world and independently regenerate targets.')
                    receipt['decision']='REVISE';receipt['review_status']='REVISE';scores['diversity']=2
                if r['semantic_parent_id'] in audit['false_competency_tag_families'].get('no_equal_routes_equal_routes_tag',[]):
                    issue('false_competency_tag','moderate',{'tag':'branched_resource_graph:equal_length_routes','distinct_equal_move_feasible_goal_routes':False,'check':'Exact-depth path counts for all horizons up to initial fuel / minimum positive edge cost.'},'Use factual single-route/branch motif labels in a fresh immutable candidate revision; do not report equal-route competency without distinct feasible equal-length routes.')
                    receipt['decision']='REVISE';receipt['review_status']='REVISE'
        receipt['confidence']=0.99
        receipts.append(receipt)
    path=OUT/'expansion-v1.jsonl'
    if path.exists():raise FileExistsError('Completed initial verdicts are immutable.')
    path.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in receipts))
    counts=collections.Counter((rows[r['sample_id']]['track_id'],r['decision']) for r in receipts)
    report={'reviewer_execution_id':'/root/expert_reviewer','rows':len(receipts),'decisions':{str(k):v for k,v in counts.items()},'source_sha256':first_report['source_sha256'],'first_judgments_preserved':'reviews/expert/expansion-v1.first_judgments.jsonl','hash_schema_contract_target_mismatches':0,'metamorphic_counts':audit['metamorphic_counts'],'cross_parent_semantic_duplicates':audit['cross_parent_semantic_duplicates'],'cross_split_exact_graph_worlds':audit['cross_split_exact_graph_worlds'],'guard_summary':audit['guard_summary'],'manual_examples':EXAMPLES,'revise_ids':[r['sample_id'] for r in receipts if r['decision']=='REVISE'],'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'independent_review.py',OUT/'independent_expansion.py',OUT/'expansion_semantic_audit.py',OUT/'finalize_expansion_v1.py']},'scope':'Actual-agent Expert Review of every canonical row with independent programmatic recomputation; substantive manual mechanism representatives; bounded proxy limitations scored. No release approval.'}
    (OUT/'verification_summary_expansion_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rows':len(receipts),'counts':report['decisions'],'revisions':len(report['revise_ids'])},indent=2))

if __name__=='__main__':main()
