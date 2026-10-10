#!/opt/anaconda3/bin/python3
"""Own compact same-task r1 semantic/shortcut support, source frozen before execution."""
import pathlib,json,hashlib,collections,datetime,importlib.util,time
ROOT=pathlib.Path(__file__).resolve().parents[2];ROLE='/root/expert_reviewer'
SOURCE='reviews/expert/review_rgm_planning_gap_v2_renderer_r1_comparison.py'
assert hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()=='041338a81b7ded6d2d1c36115043b224784d0deaa1dcc8ee45e9669b31c11bc4'
sp=importlib.util.spec_from_file_location('own_comparison',ROOT/SOURCE);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
started=time.monotonic();before={p:c.bind(p) for p in c.FILES}
rows=c.load('data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl')
pred={x['sample_id']:x for x in c.load('reviews/expert/planning-gap-v2-renderer-r1.expert_input_predictions.jsonl')}
analysis_path='reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.analysis.json'
a=json.loads((ROOT/analysis_path).read_text());units={};family_table=[]
for r in rows:
 key=c.stage.signature(r['semantic_task'])
 units.setdefault(key,{'task':c.stage.normalized(r['semantic_task']),'facts':pred[r['sample_id']]['independent_input_facts'],'rows':[]})['rows'].append(r)
for index in range(180,228):
 members=[r for r in rows if r['source_family_index']==index and r['renderer']=='forward'];contexts=[]
 for r in members:
  target=pred[r['sample_id']]['expected_target'];facts=pred[r['sample_id']]['independent_input_facts']
  contexts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'context':r['controlled_context'],'minimum_moves':len(target['shortest_plan']) if target['shortest_plan'] is not None else None,'submitted_moves':len(r['semantic_task']['plan']),'valid':target['valid'],'reaches_goal':target['reaches_goal'],'first_error':facts['measurements']['first_error']})
 family_table.append({'source_family_key':members[0]['source_family_key'],'split':members[0]['split'],'contexts':contexts})
c1=[r for r in rows if r['controlled_context']=='C1' and r['renderer']=='forward'];c4=[r for r in rows if r['controlled_context']=='C4' and r['renderer']=='forward']
reason_classes={}
for reason in ('source_not_clear','destination_not_clear'):
 group=[r for r in c1 if pred[r['sample_id']]['independent_input_facts']['measurements']['first_error']['reason']==reason]
 hist=lambda key:dict(collections.Counter(key(r) for r in group))
 reason_classes[reason]={'semantic_tasks':len(group),'block_sizes':hist(lambda r:len(r['semantic_task']['on'])),'splits':hist(lambda r:r['split']),'first_failure_indices':hist(lambda r:pred[r['sample_id']]['independent_input_facts']['measurements']['first_error']['index'])}
checks=[]
def check(name,truth,evidence):checks.append({'name':name,'pass':bool(truth),'evidence':evidence})
for reason,group in reason_classes.items():
 check(reason+':exact6_size3plus3_split5plus1',group['semantic_tasks']==6 and group['block_sizes']=={5:3,6:3} and group['splits']=={'train':5,'dev_challenge':1},group)
 check(reason+':varying_indices_and_nonempty_prefix',len(group['first_failure_indices'])>=2 and any(i>=1 for i in group['first_failure_indices']),group['first_failure_indices'])
check('C1_three_actual_indices',len({pred[r['sample_id']]['independent_input_facts']['measurements']['first_error']['index'] for r in c1})>=3,[pred[r['sample_id']]['independent_input_facts']['measurements']['first_error'] for r in c1])
kcounts=dict(collections.Counter(pred[r['sample_id']]['independent_input_facts']['v2_isolated_noop_witness']['action_index'] for r in c4));check('C4_four_each_prefix1_2_3',kcounts=={1:4,2:4,3:4},kcounts)
temporary=[r['sample_id'] for r in rows if r['controlled_context']=='C0' and r['renderer']=='forward' and 'required_temporary_goal_displacement' in pred[r['sample_id']]['independent_input_facts']['row_properties']];check('C0_atleast4_required_temporary_displacements',len(temporary)>=4,temporary)
ties=[r['sample_id'] for r in rows if r['controlled_context']=='B0' and r['renderer']=='forward' and pred[r['sample_id']]['independent_input_facts']['measurements']['minimum_feasible_route_count']>=2];check('B0_atleast6_actual_minimum_route_ties',len(ties)>=6,ties)
for r in c1+c4:
 facts=pred[r['sample_id']]['independent_input_facts'];target=facts['target'];idx=facts['measurements']['first_error']['index'];strict={k:target[k] for k in ('steps','after','valid','reaches_goal')}
 if r['controlled_context']=='C1':
  reason=facts['measurements']['first_error']['reason'];omission='source_clear_ignored' if reason=='source_not_clear' else 'destination_clear_ignored';mutant=facts['diagnostic_mutant_outputs'][omission]
  check('C1:'+r['sample_id']+':single_guard_kill',mutant['steps'][idx]['legal'] and not c.eq(mutant,strict),facts['v2_guard_facts']['guard_vectors'][idx])
 else:
  mutations=facts['diagnostic_mutant_outputs'];check('C4:'+r['sample_id']+':noop_only_kill',mutations['no_op_allowed']['valid'] and mutations['no_op_allowed']['steps'][idx]['legal'] and all(c.eq(strict,mutations[m]) for m in ('source_clear_ignored','destination_clear_ignored')),facts['v2_guard_facts']['guard_vectors'][idx])
after={p:c.bind(p) for p in c.FILES};assert before==after
baselines=c.stage.baselines(list(units.values()))
report={'actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'R1_OWN_COMPACT_SEMANTIC_AND_SHORTCUT_SUPPORT','source':c.bind(str(pathlib.Path(__file__).relative_to(ROOT))),'comparison_source':c.bind(SOURCE),'comparison_analysis':c.bind(analysis_path),'actual_exit_code':0,'bindings_before_after_unchanged':True,'records':408,'normalized_semantic_tasks':len(units),'source_families':48,'new_tasks_families':0,'family_table':family_table,'cohort_checks':checks,'cohort_checks_all_pass':all(x['pass'] for x in checks),'C1_reason_size_split_index_classes':reason_classes,'C4_prefix_counts':kcounts,'C0_required_temporary_goal_displacement_tasks':temporary,'B0_minimum_route_tie_tasks':ties,'negative_control_probes':len(a['own_control_mutation_calibration']),'negative_probe_failures':[x for x in a['own_control_mutation_calibration'] if not x['pass']],'negative_probe_types':dict(collections.Counter(x['mutation'] for x in a['own_control_mutation_calibration'])),'difficulty_counts':dict(collections.Counter(r['difficulty'] for r in rows)),'lookup_component_baselines':baselines,'baseline_limit':'Train-only finite-feature lookup on own prompt-derived typed component truths, one forward row per semantic task, majority fallback; not model/full-target performance or an unpublished acceptance gate. Correlated family contexts and small dev set limit inference.','novelty':{k:v for k,v in a['novelty'].items() if k!='base_details'},'guidance':c.GUIDANCE,'no_dataset_or_publication_approval':True,'actual_elapsed_seconds':time.monotonic()-started}
path='reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.summary.json'
with (ROOT/path).open('x') as output:json.dump(report,output,indent=2);output.write('\n')
print(json.dumps({'summary':c.bind(path),'cohort_checks':len(checks),'all_cohort_checks_pass':report['cohort_checks_all_pass'],'negative_tests':report['negative_control_probes'],'lookup_baselines':len(baselines),'elapsed':report['actual_elapsed_seconds']}))
