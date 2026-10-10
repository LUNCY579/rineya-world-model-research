#!/usr/bin/env python3
"""Input-only cohort supplement from already frozen own408 predictions; no gold."""
import collections,hashlib,json,pathlib,re,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
BINDINGS={'reviews/input_only/planning-gap-v2.jsonl':'00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c','reviews/adversarial/planning-gap-v2.input_predictions.jsonl':'ee63bdc9f62c1780326969cb05dbb8410897b176d6fe24fa67b72f257c480c1b','reviews/adversarial/planning-gap-v2.input_first_judgments.jsonl':'5b66ecfd1b48c82aebadd4c1e1f3be89537384ec06c0ab04ca7ea752703ec18a'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def freeze(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
def jb(x):return (json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
def main():
 for p,h in BINDINGS.items():assert sha(ROOT/p)==h
 source=pathlib.Path(__file__);sh=sha(source);freeze(ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_input_cohort-{sh}.py',source.read_bytes())
 pre=ROOT/'audits/execution/planning-gap-v2-adversarial-input-cohort.preexec.json'
 freeze(pre,jb({'actual_execution_id':EXEC,'source_sha256':sh,'bound_own_first_predictions_and_input':BINDINGS,'stage':'SOURCE_FROZEN_BEFORE_INPUT_COHORT_EXECUTION','gold_control_manifest_and_other_reviewer_not_read':True,'expectations':'Public A allocation6infeasible6lengthened including1dev each; C1 selective6source6destination (3eachsize and1dev each), at least3failure indices global2each,12C4 isolatednoops k1/2/3foureach; C0 canonical horizons4–8, at least4required temporary displacement, both style views.','join_scope':'Family source index parsed only for public cohort allocation/split, never for solving or student shortcut predictors.'}))
 print(canon({'stage':'cohort_source_frozen','source_sha256':sh,'preexec_sha256':sha(pre)}),flush=True)
 start=time.perf_counter();rows=[json.loads(l) for l in (ROOT/'reviews/input_only/planning-gap-v2.jsonl').read_text().splitlines()]
 preds={p['sample_id']:p for p in (json.loads(l) for l in (ROOT/'reviews/adversarial/planning-gap-v2.input_predictions.jsonl').read_text().splitlines())}
 cp=ROOT/'audits/diversity/planning-gap-v2-input-derived-controls.json';controls=json.loads(cp.read_text())
 pairs=controls['actual_input_derived_semantic_pairs'];rows_by_id={r['sample_id']:r for r in rows};family_pairs=collections.defaultdict(list)
 for c in pairs:family_pairs[c['source_family_key']].append(c)
 errors=[];measurements=[];reason_hist=collections.Counter();fail_positions=collections.defaultdict(set);noop_hist=collections.Counter();temporary=0
 def issue(code,evidence):errors.append({'code':code,'evidence':evidence,'scope':'Input-derived prospective control requirement; no manifest/gold verdict.'})
 source_dest={207,209,210,212,213,214};asizes=[5,5,5,5,6,6,7,6,6,7,7,7];A=[];C=[];B=[];D=[]
 for family,ps in sorted(family_pairs.items()):
  match=re.search(r'p(\d+)$',family);assert match,family;i=int(match.group(1));split='dev_challenge' if i%10 in (8,9) else 'train'
  stratum='A' if i<192 else 'B' if i<204 else 'C' if i<216 else 'D'
  named=collections.defaultdict(list)
  for p in ps:named[p['name']].append(p)
  entry={'source_family_key':family,'source_index':i,'stratum':stratum,'split':split,'inferred_pair_names':dict(collections.Counter(p['name'] for p in ps))}
  for p in ps:
   if p['split']!=split:issue('family_split_public_allocation_mismatch',p)
  if stratum=='A':
   pp=named['fuel_only_counterfactual'];assert len(pp)==1
   p=pp[0];hi,lo=[rows_by_id[x] for x in p['sample_ids']];hp,lp=[preds[x] for x in p['sample_ids']];size=len(hi['semantic_task']['nodes']);sub='I' if lp['expected_target']['shortest_plan'] is None else 'L'
   expected='I' if i%2==0 else 'L'
   entry.update(size=size,actual_subtype=sub,low_fuel=lo['semantic_task']['fuel'],high_fuel=hi['semantic_task']['fuel'],low_minimum=lp['search_evidence']['fresh_minimum_moves'],high_minimum=hp['search_evidence']['fresh_minimum_moves'],topology_minimum=lp['search_evidence']['topology_minimum_moves'],minimum_required_fuel=lp['search_evidence']['cheapest_required_fuel'])
   if sub!=expected or size!=asizes[i-180]:issue('A_public_size_or_subtype_mismatch',entry)
   if sub=='I' and not (lo['semantic_task']['fuel']==lp['search_evidence']['cheapest_required_fuel']-1 and hi['semantic_task']['fuel']==lp['search_evidence']['cheapest_required_fuel']):issue('A_exact_fuel_boundary_missing',entry)
   if sub=='L' and not(lp['search_evidence']['fresh_minimum_moves']>lp['search_evidence']['topology_minimum_moves'] and lp['search_evidence']['fresh_minimum_moves']>hp['search_evidence']['fresh_minimum_moves']):issue('A_actual_finite_lengthening_missing',entry)
   A.append(entry)
  elif stratum=='B':
   pp=named['directed_edge_deletion_counterfactual'];assert len(pp)==1
   base=rows_by_id[pp[0]['sample_ids'][0]];bp=preds[base['sample_id']];size=len(base['semantic_task']['nodes']);entry.update(size=size,base_submitted_length=len(base['semantic_task']['plan']),minimum_route_multiplicity=bp['search_evidence']['minimum_route_multiplicity']['route_count'],removed_edge=pp[0]['input_derived_evidence']['removed_edge'],both_fresh_feasible=pp[0]['input_derived_evidence']['both_fresh_feasible'])
   if size!=[5,6,7][(i-192)%3] or len(base['semantic_task']['plan'])<3:issue('B_size_or_base_horizon_mismatch',entry)
   B.append(entry)
  elif stratum=='C':
   order=named['adjacent_action_order_counterfactual'];noop=named['isolated_noop_append_counterfactual'];assert len(order)==len(noop)==1
   c=order[0];base,bad=[rows_by_id[x] for x in c['sample_ids']];bp=preds[base['sample_id']];badp=preds[bad['sample_id']];size=len(base['semantic_task']['on']);reason=c['input_derived_evidence']['first_failure_reason'];idx=c['input_derived_evidence']['first_failure_index'];expected='destination_not_clear' if i in source_dest else 'source_not_clear';k=1+(i-204)%3
   entry.update(size=size,actual_reason=reason,first_failure_index=idx,swap_index=c['input_derived_evidence']['swap_index'],guard_truth=c['input_derived_evidence']['guard_truth'],canonical_base_plan=canon(base['semantic_task']['plan'])==canon(bp['expected_target']['shortest_plan']),base_minimum=bp['search_evidence']['fresh_minimum_moves'],base_changed_clearance_witness_count=len(bp['input_derived_v2_clearance_witnesses']),required_temporary_goal_displacement=bp['search_evidence']['required_temporary_goal_displacement'],actual_noop_prefix=noop[0]['input_derived_evidence']['prefix_length'],expected_noop_prefix=k)
   if reason!=expected or size!=(5 if i%2==0 else 6):issue('C_public_size_or_selective_reason_mismatch',entry)
   if not entry['canonical_base_plan'] or not(4<=entry['base_minimum']<=8) or not entry['base_changed_clearance_witness_count']:issue('C0_canonical_minimum_or_dependency_missing',entry)
   if entry['actual_noop_prefix']!=k:issue('C4_fixed_k_mismatch',entry)
   reason_hist[(reason,size,split)]+=1;fail_positions[reason].add(idx);noop_hist[(k,split)]+=1;temporary+=entry['required_temporary_goal_displacement'];C.append(entry)
  else:
   goals=named['partial_goal_extension_counterfactual'];assert len(goals)==1
   weak,strong=[rows_by_id[x] for x in goals[0]['sample_ids']];wp=preds[weak['sample_id']];sp=preds[strong['sample_id']];size=len(weak['semantic_task']['on']);entry.update(size=size,weak_partial_goal=0<len(weak['semantic_task']['goal_on'])<size,weak_minimum=wp['search_evidence']['fresh_minimum_moves'],strong_minimum=sp['search_evidence']['fresh_minimum_moves'],strong_reaches_goal=sp['expected_target']['reaches_goal'])
   if size!=(5 if i%2==0 else 6) or not entry['weak_partial_goal'] or entry['strong_minimum']<=entry['weak_minimum']:issue('D_actual_size_or_goal_obligation_extension_missing',entry)
   D.append(entry)
  measurements.append(entry)
 if len(B)!=12 or sum(e['minimum_route_multiplicity']>=2 for e in B)<6:issue('B_minimum_move_tie_family_coverage',B)
 if len(C)!=12 or temporary<4:issue('C_required_temporary_displacement_family_coverage',{'actual':temporary})
 if len(set().union(*fail_positions.values()))<3 or any(len(v)<2 or not any(i>=1 for i in v) for v in fail_positions.values()):issue('C_first_error_position_diversity', {r:sorted(v) for r,v in fail_positions.items()})
 raw_leads=collections.defaultdict(set)
 for r in rows:raw_leads[r['semantic_task']['kind']].add(r['rendered_input'].split('Task JSON:\n')[0])
 prompt_lead_evidence={k:[{'sha256':hashlib.sha256(s.encode()).hexdigest(),'characters':len(s),'text':s} for s in v] for k,v in raw_leads.items()}
 if any(len(v)!=1 for v in raw_leads.values()):issue('varying_contract_leads_require_individual_semantics_review',prompt_lead_evidence)
 # Conditioned input-only label distributions, with each normalized semantic task once.
 seen=set();units=[]
 for r in rows:
  t=r['semantic_task'];n=json.loads(json.dumps(t))
  if t['kind']=='graph':n['nodes']=sorted(t['nodes']);n['edges']=sorted(t['edges'])
  key=canon(n)
  if key not in seen:seen.add(key);units.append(r)
 conditioned=collections.defaultdict(collections.Counter)
 for r in units:
  p=preds[r['sample_id']];t=r['semantic_task'];idx=int(re.search(r'p(\d+)$',r['source_family_key']).group(1));stratum='A' if idx<192 else 'B' if idx<204 else 'C' if idx<216 else 'D'
  for scope in [(stratum,r['split']), (t['kind'],r['split'],len(t['plan'])),(stratum,r['split'],len(t.get('nodes',t.get('on',{}))))]:
   conditioned[canon(scope)][canon({'valid':p['expected_target']['valid'],'reaches_goal':p['expected_target']['reaches_goal'],'fresh_feasible':p['expected_target']['shortest_plan'] is not None})]+=1
 for p,h in BINDINGS.items():assert sha(ROOT/p)==h
 out=ROOT/'audits/diversity/planning-gap-v2-input-only-cohort.json'
 value={'actual_execution_id':EXEC,'source_sha256':sh,'preexec_sha256':sha(pre),'input_and_first_prediction_bindings':BINDINGS,'input_derived_control_file_sha256':sha(cp),'completed':True,'actual_duration_seconds':time.perf_counter()-start,'public_cohort_errors':errors,'actual_family_measurements':measurements,'A_counts':{canon(k):v for k,v in collections.Counter((e['actual_subtype'],e['size'],e['split']) for e in A).items()},'C_reason_size_split_histogram':{canon(k):v for k,v in reason_hist.items()},'C_failure_position_sets':{r:sorted(v) for r,v in fail_positions.items()},'C_noop_k_split_histogram':{canon(k):v for k,v in noop_hist.items()},'required_temporary_displacement_C0_families':temporary,'B_actual_minimum_route_tie_families':sum(e['minimum_route_multiplicity']>=2 for e in B),'actual_prompt_contract_leads':prompt_lead_evidence,'conditioned_labels_normalized_semantic_units':{k:dict(v) for k,v in conditioned.items()},'same_model_public_guided_own_solver_reuse_disclosed':True,'canonical_gold_manifest_and_other_verdict_not_read':True,'dataset_approval':False,'scope_limits':['Public cohort allocation checked using custody source indices only; input solutions do not depend on them.','Actual current family controls inferred from tasks;192supplied declarations/control IDs remain unobserved until comparison authorization.','Small exposed-dev counts (one dev family per C failure type) remain; no sealed or efficacy claim.']}
 freeze(out,jb(value));ex=ROOT/'audits/execution/planning-gap-v2-adversarial-input-cohort.json';freeze(ex,jb({'actual_execution_id':EXEC,'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'output_sha256':sha(out),'input_and_first_prediction_bindings_before_after':BINDINGS,'errors':len(errors),'actual_family_count':len(measurements),'dataset_approval':False}))
 print(canon({'completed':True,'execution_sha256':sha(ex),'source_sha256':sh,'cohort_sha256':sha(out),'errors':errors,'C_positions':{r:sorted(v) for r,v in fail_positions.items()},'C_temporary':temporary,'B_ties':sum(e['minimum_route_multiplicity']>=2 for e in B)}),flush=True)
if __name__=='__main__':main()
