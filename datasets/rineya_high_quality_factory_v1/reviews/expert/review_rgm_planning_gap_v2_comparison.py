#!/opt/anaconda3/bin/python3
"""Own actual expert canonical comparison after frozen input-only predictions.
No producer/formal/root private implementations or other reviewer verdicts read.
"""
import collections,copy,datetime,hashlib,importlib.util,itertools,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
sha=lambda b:hashlib.sha256(b).hexdigest()
FILES={
 'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl':'57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23',
 'data/RGM-001/manifests/planning-gap-v2.family_controls.json':'9c4497cf61e1d77cea24d808eca965c00e10e975cf235bd7d1aa2667e259da17',
 'reviews/expert/planning-gap-v2.expert_input_predictions.jsonl':'ef9f9dc812f68eaa1442a17cbb7c957f74fe0adebc68e3ea569cdc54c83d2255',
 'reviews/expert/planning-gap-v2.expert_first_input_judgments.jsonl':'16717fe11f20d4ad5c797ad99d792621094193685ddd9797d51f410396fef77f',
 'reviews/expert/planning-gap-v2.expert_input.freeze.json':'86ce199557ccb7b8353a51964e892f10569ddb94814c19f4c411f035a426804e',
 'reviews/expert/planning-gap-v2.expert_input_analysis.json':'81e1d89190270b9b6fa8c041d31ea18fda201a7e276ca430deac22fe1be0dbc1',
 'reviews/input_only/planning-gap-v2.jsonl':'00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c',
 'reviews/input_only/planning-gap-v1.jsonl':'2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654',
 'research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl':'896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d',
 'reviews/expert/prepare_rgm_planning_gap_v2.py':'7c74fcebba1057bcdb0313c7c658b57a0752d041558f82534fb44a573d98fc53',
 'reviews/expert/review_rgm_planning_gap_v2_input_stage.py':'c85157b7920d3b6070bc8059f73f1c1d7184f7e09bea2df564dbd3306b39e102',
}
spec=importlib.util.spec_from_file_location('own_frozen_v2_prep',ROOT/'reviews/expert/prepare_rgm_planning_gap_v2.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
v1=prep.v1;canon=v1.canon;eq=v1.eq
spec=importlib.util.spec_from_file_location('own_input_stage',ROOT/'reviews/expert/review_rgm_planning_gap_v2_input_stage.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
PROPERTY_NAMES={'legal_prefix_then_first_failure','fuel_lengthens_fresh_minimum_moves','fuel_blocks_topologically_reachable_goal','valid_goal_detour','later_action_requires_changed_block_clearance','partial_block_goal','required_temporary_goal_displacement','isolated_unchanged_support_failure_after_legal_prefix'}
GUIDANCE='Canonical access authorized only after immutable all408 input-derived predictions/first judgments. Public-guided corrective same48-family cohort, own frozen V1 solver reused and same underlying model; no fully blind design, new-family growth relative V1 or model independence claim.'

def bind(path):
 b=(ROOT/path).read_bytes();return {'path':path,'sha256':sha(b),'bytes':len(b)}
def load(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines()]
def json_write(path,x):
 with (ROOT/path).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def graph_key(task,weighted=True):
 """Exact isomorphism canonicalization within invariant degree/cost classes.
 Partitions improve finite enumeration only; no random hashes replace equality.
 Includes isolated-node count, and start/goal roles for weighted base worlds.
 """
 nodes=sorted(task['nodes']);groups=collections.defaultdict(list)
 for n in nodes:
  incoming=[c for a,b,c in task['edges'] if b==n];outgoing=[c for a,b,c in task['edges'] if a==n]
  role=(n==task['start'],n==task['goal']) if weighted else ()
  invariant=(role,len(incoming),len(outgoing),tuple(sorted(incoming)) if weighted else (),tuple(sorted(outgoing)) if weighted else ())
  groups[invariant].append(n)
 classes=sorted(groups);options=[itertools.permutations(groups[k]) for k in classes];best=None
 for permutations in itertools.product(*options):
  order=[n for group in permutations for n in group];ids={n:i for i,n in enumerate(order)}
  edges=tuple(sorted((ids[a],ids[b],c) if weighted else (ids[a],ids[b]) for a,b,c in task['edges']))
  value=(len(nodes),tuple(classes),edges)
  if best is None or value<best:best=value
 return repr(best)
def property_truth(facts,name):
 if name=='later_action_requires_changed_block_clearance':return bool(facts.get('v2_canonical_clearance_witnesses'))
 if name=='isolated_unchanged_support_failure_after_legal_prefix':return facts.get('v2_isolated_noop_witness') is not None
 return name in facts['row_properties']
def witness_truth(task,facts,name,evidence):
 if type(evidence) is not dict:return False
 if not property_truth(facts,name):return False
 target=facts['target'];m=facts['measurements'];minimum=len(target['shortest_plan']) if target['shortest_plan'] is not None else None
 if name=='later_action_requires_changed_block_clearance':return prep.check_clearance_witness(task,evidence)
 if name=='isolated_unchanged_support_failure_after_legal_prefix':return prep.check_noop_witness(task,evidence)
 expected={'first_error_index':m['first_error']['index'] if m['first_error'] else None,'minimum_moves':minimum,'topology_distance':len(m['topology_shortest_plan']) if m.get('topology_shortest_plan') is not None else None,'cheapest_fuel':m.get('cheapest_required_fuel'),'block_count':len(task.get('on',{})),'goal_relation_count':len(task.get('goal_on',{})),'submitted_moves':len(task['plan'])}
 if not any(k in evidence for k in expected):return False
 for k,value in evidence.items():
  if k in expected:
   if type(value) is not int or not eq(value,expected[k]):return False
 return True

def derive():
 started=time.monotonic();before={p:bind(p) for p in FILES}
 for p,h in FILES.items():assert before[p]['sha256']==h,p
 own_source=bind(str(pathlib.Path(__file__).relative_to(ROOT)))
 rows=load('data/RGM-001/candidate_renderings/planning-gap-v2.jsonl');preds=load('reviews/expert/planning-gap-v2.expert_input_predictions.jsonl');view=load('reviews/input_only/planning-gap-v2.jsonl')
 by={r['sample_id']:r for r in rows};pred={r['sample_id']:r for r in preds};visible={r['sample_id']:r for r in view}
 assert len(rows)==len(by)==len(pred)==len(visible)==408 and set(by)==set(pred)==set(visible)
 initial_analysis=json.loads((ROOT/'reviews/expert/planning-gap-v2.expert_input_analysis.json').read_text())
 own_context={sid:context for f in initial_analysis['family_audits'] for context,c in f['contexts'].items() for sid in c['sample_ids']}
 manifest=json.loads((ROOT/'data/RGM-001/manifests/planning-gap-v2.family_controls.json').read_text())
 public=json.loads((ROOT/'reviews/expert/planning-gap-v2.expert_preparation.freeze.json').read_text())['public_specifications']
 per={sid:{'sample_id':sid,'record_hash':r['record_hash'],'checks':{},'failures':[],'witnesses':[],'control_ids':[]} for sid,r in by.items()}
 def check(ids,name,truth,evidence,fix='Fresh versioned correction of the exact affected canonical rows/declarations; preserve current bytes.'):
  for sid in ids:
   per[sid]['checks'][name]=bool(truth)
   if not truth:per[sid]['failures'].append({'category':name,'severity':'high','evidence':evidence,'fix':fix})
 allids=list(by)
 check(allids,'exact_v2_interface_version',type(manifest.get('version')) is str and manifest['version']=='rgm-planning-controls-2.0.0' and manifest.get('batch_name')=='planning-gap-v2' and manifest.get('track_id')=='RGM-001','Prospective public V2 version/batch/track interface')
 check(allids,'parent_and_semantics_binding',manifest.get('revision_of_batch')=='planning-gap-v1' and manifest.get('parent_input_view_sha256')==FILES['reviews/input_only/planning-gap-v1.jsonl'] and manifest.get('semantics_sha256')==public['schemas/SEMANTICS.md']['sha256'],'Published parent input/unchanged semantics hashes')
 bindings=manifest.get('public_specification_bindings',[])
 actual_public={b.get('path'):b.get('sha256') for b in bindings if type(b) is dict}
 expected_public={p:b['sha256'] for p,b in public.items()}
 public_ok=len(bindings)==len(actual_public)==7 and actual_public==expected_public
 for p,h in expected_public.items():public_ok=public_ok and bind(p)['sha256']==h
 check(allids,'seven_public_document_bindings',public_ok,{'actual':actual_public,'expected':expected_public})
 block_alt=0;graph_alt=0
 for sid,r in by.items():
  pr=pred[sid];task=r['semantic_task'];facts=pr['independent_input_facts'];target=pr['expected_target']
  computed_hash=sha(canon({k:v for k,v in r.items() if k!='record_hash'}).encode())
  check([sid],'full_record_hash_exact',computed_hash==r['record_hash'],{'actual_hash':r['record_hash'],'own_recomputed_hash':computed_hash})
  check([sid],'frozen_view_exact',eq({k:r[k] for k in visible[sid]},visible[sid]) and r['record_hash']==pr['record_hash'], 'Full six-field canonical join against frozen stripped view and independent prediction')
  check([sid],'typed_target_exact',eq(r['verified_target'],target),{'expected_sha256':sha(canon(target).encode()),'actual_sha256':sha(canon(r['verified_target']).encode())})
  check([sid],'declared_context_matches_input_inference',r['controlled_context']==own_context[sid] and r['primary_stratum']==own_context[sid][0],'Context inferred from visible field changes and strict replay, not ID spelling')
  check([sid],'public_family_split_metadata',type(r['source_family_index']) is int and r['source_family_index']==stage.family_index(r['source_family_key']) and r['split']==('dev_challenge' if r['source_family_index']%10 in (8,9) else 'train'),'Published index/split allocation')
  check([sid],'nonsealed_original_scope',r['source_provenance'].get('sealed_evaluation_used') is False and r['source_provenance'].get('origin')=='original_synthetic' and r['target_format']=='structured' and r['track_id']=='RGM-001','Original exposed synthetic planning inputs')
  check([sid],'difficulty_within_supported_structural_range',type(r['difficulty']) is int and r['difficulty']==r['structural_difficulty'] and 2<=r['difficulty']<=4,'No level7/inflated sealed OOD label; fuller semantic assessment is in own inspection')
  check([sid],'renderer_declared',r['renderer'] in ('forward','reverse'),'Two public renderer styles')
  tags=r['skill_tags'];properties=set(tags)&PROPERTY_NAMES
  check([sid],'property_tags_truthful',type(tags) is list and all(type(tag) is str for tag in tags) and len(tags)==len(set(tags)) and all(property_truth(facts,name) for name in properties),{'declared':sorted(properties),'own_true':[p for p in sorted(PROPERTY_NAMES) if property_truth(facts,p)]})
  if own_context[sid]=='C4':check([sid],'C4_required_noop_property_present','isolated_unchanged_support_failure_after_legal_prefix' in properties,'Prospective V2 required row property')
 # Extra independent lexicographic relation-set/heap check on every semantic input.
 altcache={}
 for sid,r in by.items():
  key=stage.signature(r['semantic_task'])
  if key not in altcache:
   if r['semantic_task']['kind']=='blocks':altcache[key]=v1.block_minimum_alternative(r['semantic_task']);block_alt+=1
   else:altcache[key]=v1.graph_minimum_alternative(r['semantic_task']);graph_alt+=1
  check([sid],'alternative_exact_fresh_minimum',eq(altcache[key],pred[sid]['expected_target']['shortest_plan']),'Separate complete-sequence heap search; blocks use independently authored relation-set transitions')
 records={sid:{'task':r['semantic_task'],'context':own_context[sid],'family':r['source_family_key'],'split':r['split'],'style':r['renderer']} for sid,r in by.items()}
 families=manifest.get('families',[]);seen_family=set();seen_context_samples=[];all_controls=[];witness_count=0;control_details=[]
 expected_controls_by_stratum={'A':[('fuel_only_counterfactual','A0','A1')],'B':[('directed_edge_deletion_counterfactual','B0','B1'),('same_world_valid_goal_detour','B0','B3')],'C':[('adjacent_action_order_counterfactual','C0','C1'),('same_world_valid_goal_detour','C0','C2'),('isolated_noop_append_counterfactual','C3','C4')],'D':[('partial_goal_extension_counterfactual','D0','D1'),('same_world_valid_goal_detour','D2','D3')]}
 for family in families:
  key=family['source_family_key'];sids=[sid for sid,r in by.items() if r['source_family_key']==key];index=stage.family_index(key);st=stage.stratum(index)
  check(sids,'manifest_family_identity',key not in seen_family and type(family['source_index']) is int and family['source_index']==index and family['primary_stratum']==st and family['split']==by[sids[0]]['split'],'Manifest exact source family/index/stratum/split');seen_family.add(key)
  check(sids,'base_sample_actual_context',family['base_sample_id'] in by and family['base_sample_id'] in sids and own_context[family['base_sample_id']]==st+'0','Base sample may use either actual style; no private ID format required')
  context_rows={};context_ids=[]
  for c in family['contexts']:
   context=c['context_id'];context_ids.append(context);ids=c['sample_ids'];seen_context_samples+=ids
   exists=type(ids) is list and len(ids)==2 and len(set(ids))==2 and all(sid in by for sid in ids)
   check(sids,'manifest_context_binding:'+context,exists and all(sid in sids and own_context[sid]==context for sid in ids) and {by[sid]['renderer'] for sid in ids}=={'forward','reverse'},'Each declared context covers exactly its two inferred equal-task renderer views')
   if not exists:continue
   check(ids,'renderer_task_target_equivalence',stage.signature(by[ids[0]]['semantic_task'])==stage.signature(by[ids[1]]['semantic_task']) and eq(pred[ids[0]]['expected_target'],pred[ids[1]]['expected_target']),'List/key order is surface variation; no new semantic task/family')
   context_rows[context]={by[sid]['renderer']:sid for sid in ids}
   props=c['row_properties'];props_ok=type(props) is list and all(type(p) is str and p in PROPERTY_NAMES for p in props) and len(props)==len(set(props))
   check(ids,'context_properties_valid',props_ok,props)
   check(ids,'row_context_property_declarations_agree',all(set(by[sid]['skill_tags'])&PROPERTY_NAMES==set(props) for sid in ids),'Declared context properties match actual per-row factual tags; optional properties need not enumerate all true facts')
   expected_witnesses={(sid,p) for sid in ids for p in props};seen_witnesses=[]
   for w in c['row_witnesses']:
    sid=w.get('sample_id');name=w.get('property');okay=sid in ids and name in props and w.get('scope')=='row'
    if sid in by and name in PROPERTY_NAMES:okay=okay and witness_truth(by[sid]['semantic_task'],pred[sid]['independent_input_facts'],name,w.get('evidence'))
    else:okay=False
    check(ids if sid not in by else [sid],'witness_truth:'+str(name)+':'+str(sid),okay,w)
    seen_witnesses.append((sid,name));witness_count+=1
    if sid in per:per[sid]['witnesses'].append({'property':name,'evidence':w.get('evidence'),'verified':okay})
   check(ids,'one_bound_witness_per_tagged_sample_property',len(seen_witnesses)==len(set(seen_witnesses)) and set(seen_witnesses)==expected_witnesses,'No private evidence-ID naming or extra closed evidence shape required')
  expected_contexts={st+str(i) for i in range(5 if st=='C' else 4)}
  check(sids,'manifest_exact_required_contexts',len(context_ids)==len(set(context_ids)) and set(context_ids)==expected_contexts,'Prospective required four/five context cohort')
  controls=family['controls'];all_controls+=controls
  expected_pairs={(name,context_rows[left][style],context_rows[right][style]) for name,left,right in expected_controls_by_stratum[st] for style in ('forward','reverse')}
  actual_pairs=[]
  for c in controls:
   ids=c.get('sample_ids',[]);okay=False;actual_error=None
   try:okay=prep.compare_renderer_control(c,records)
   except (AssertionError,ValueError,KeyError,TypeError) as error:actual_error=type(error).__name__+': '+str(error)
   check(sids,'supplied_control:'+str(c.get('control_id')),okay and all(sid in sids for sid in ids),{'control':c,'actual_error':actual_error})
   if type(ids) is list and len(ids)==2:actual_pairs.append((c.get('name'),ids[0],ids[1]))
   control_details.append({'source_family_key':key,'control_id':c.get('control_id'),'name':c.get('name'),'sample_ids':ids,'pass':okay,'actual_error':actual_error})
   for sid in ids:
    if sid in per:per[sid]['control_ids'].append(c.get('control_id'))
  check(sids,'every_required_per_renderer_control_pair',expected_pairs<=set(actual_pairs),'All exact required logical left/right sample pairs, both styles; names may repeat')
  check(sids,'per_renderer_control_group_encoding',all(len(c.get('sample_ids',[]))==2 for c in controls),'Combined four-ID declaration is not the public two-ID V2 form')
 check(allids,'all_manifest_families_covered',len(families)==len(seen_family)==48 and seen_family=={r['source_family_key'] for r in rows},'Exact family coverage')
 check(allids,'all408_manifest_context_rows_once',len(seen_context_samples)==408 and len(set(seen_context_samples))==408 and set(seen_context_samples)==set(by),'Each actual sample bound exactly once by context')
 control_ids=[c.get('control_id') for c in all_controls]
 check(allids,'globally_unique_nonempty_control_ids',len(control_ids)==len(set(control_ids)) and all(type(c) is str and c for c in control_ids),'Explicit public V2 global declaration custody uniqueness')
 # Exact novelty against authorized exposed prior active inventory; parent is a
 # revised-cohort comparison, not a forbidden prior-collision gate.
 prior=load('research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl');parent=load('reviews/input_only/planning-gap-v1.jsonl')
 graph_cache={};block_cache={}
 def shape(task,weighted=True):
  key=canon({k:v for k,v in stage.normalized(task).items() if k not in ('plan','fuel')})
  cache=graph_cache if task['kind']=='graph' else block_cache
  ck=(key,weighted)
  if ck not in cache:cache[ck]=graph_key(task,weighted) if task['kind']=='graph' else stage.block_shape(task,weighted)
  return cache[ck]
 calibration=v1.g(['s','a','g'],[['s','a',1],['a','g',2]],3,[])
 renamed=v1.rename_graph(calibration,{'s':'omega','a':'alpha','g':'zeta'})
 assert shape(calibration)==shape(renamed)
 extra=copy.deepcopy(calibration);extra['nodes'].append('isolated');assert shape(calibration)!=shape(extra)
 cost=copy.deepcopy(calibration);cost['edges'][0][2]=2;assert shape(calibration)!=shape(cost) and shape(calibration,False)==shape(cost,False)
 prior_weighted={shape(r['semantic_task']) for r in prior if r['semantic_task']['kind']=='graph'}
 prior_unweighted={shape(r['semantic_task'],False) for r in prior if r['semantic_task']['kind']=='graph'}
 prior_blocks={shape(r['semantic_task']) for r in prior if r['semantic_task']['kind']=='blocks'}
 parent_bases={r['source_family_key']:r['semantic_task'] for r in parent if re_base(r.get('sample_id',''))}
 base_weighted=[];base_unweighted=[];base_blocks=[];new_unweighted=[];parent_equal=[];base_details=[]
 for family in families:
  row=by[family['base_sample_id']];task=row['semantic_task'];key=row['source_family_key'];sids=[sid for sid,r in by.items() if r['source_family_key']==key]
  own_shape=shape(task)
  if task['kind']=='graph':
   base_weighted.append(own_shape);unweighted=shape(task,False);base_unweighted.append(unweighted)
   if unweighted not in prior_unweighted:new_unweighted.append(key)
   absent=own_shape not in prior_weighted
  else:base_blocks.append(own_shape);absent=own_shape not in prior_blocks
  check(sids,'base_absent_from_prior_active_inputs',absent,{'source_family_key':key,'own_exact_shape_sha256':sha(own_shape.encode())})
  unchanged=key in parent_bases and shape(parent_bases[key])==own_shape
  if unchanged:parent_equal.append(key)
  base_details.append({'source_family_key':key,'kind':task['kind'],'shape_sha256':sha(own_shape.encode()),'absent_prior':absent,'same_shape_as_parent_V1':unchanged})
 check(allids,'24distinct_weighted_graph_baseworlds',len(set(base_weighted))==24,len(set(base_weighted)))
 check(allids,'24distinct_joint_block_baseworlds',len(set(base_blocks))==24,len(set(base_blocks)))
 check(allids,'atleast16new_unweighted_graph_topologies',len({value for value in base_unweighted if value not in prior_unweighted})>=16,len({value for value in base_unweighted if value not in prior_unweighted}))
 after={p:bind(p) for p in FILES};assert before==after and bind(own_source['path'])==own_source
 failures=collections.Counter(f['category'] for p in per.values() for f in p['failures'])
 result={'actual_author':ROLE,'review_stage':'V2_OWN_CANONICAL_COMPARISON_DERIVATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':own_source,'sources_before':before,'sources_after':after,'sources_unchanged':True,'guidance':GUIDANCE,'records':len(rows),'typed_target_matches':sum(p['checks']['typed_target_exact'] for p in per.values()),'exact_hash_matches':sum(p['checks']['full_record_hash_exact'] for p in per.values()),'frozen_view_matches':sum(p['checks']['frozen_view_exact'] for p in per.values()),'independent_alternative_minimum_checks':{'graph_semantic_inputs':graph_alt,'block_semantic_inputs':block_alt},'families':len(families),'manifest_row_witnesses':witness_count,'supplied_control_declarations':len(all_controls),'control_details':control_details,'per_record':list(per.values()),'failure_counts':dict(failures),'rows_with_high_failures':sum(bool(p['failures']) for p in per.values()),'novelty':{'prior_input_rows':len(prior),'prior_source_families':len({r['source_family_key'] for r in prior}),'prior_weighted_graph_classes':len(prior_weighted),'prior_unweighted_graph_classes':len(prior_unweighted),'prior_joint_block_classes':len(prior_blocks),'current_weighted_graph_classes':len(set(base_weighted)),'current_unweighted_graph_classes':len(set(base_unweighted)),'current_new_unweighted_classes':len({value for value in base_unweighted if value not in prior_unweighted}),'current_joint_block_classes':len(set(base_blocks)),'unchanged_parent_base_shapes':len(parent_equal),'base_details':base_details,'independent_shape_calibration':'Renaming invariant, isolated-node count preserved, weighted-cost intervention distinct while unweighted shape same. Parent V1 overlaps are revised-cohort descent, not48 additional new families.'},'input_analysis_sha256':FILES['reviews/expert/planning-gap-v2.expert_input_analysis.json'],'actual_elapsed_seconds':time.monotonic()-started,'actual_exit_code':0,'other_private_reviews_formal_producer_root_diagnostics_read':False,'scope':'Actual every-row target/hash/view/metadata comparison against own frozen independent predictions; public renderer controls and row witnesses executed. Final judgments require own actual-agent semantic inspection, never publication approval.'}
 path='reviews/expert/planning-gap-v2.expert_comparison.analysis.json';json_write(path,result)
 print(json.dumps({'analysis':bind(path),'rows':len(rows),'typed_matches':result['typed_target_matches'],'witnesses':witness_count,'controls':len(all_controls),'high_failure_rows':result['rows_with_high_failures'],'elapsed_seconds':result['actual_elapsed_seconds']}))

def re_base(sample):
 # Parent view exposes context-bearing custody IDs. This selects only previous
 # base inputs for descent statistics; no label or new truth inferred from IDs.
 return any('-'+s+'0-' in sample for s in 'ABCD')

def finalize():
 analysis_path='reviews/expert/planning-gap-v2.expert_comparison.analysis.json';a=json.loads((ROOT/analysis_path).read_text());ab=bind(analysis_path)
 inspection='reviews/expert/semantic_inspection_planning_gap_v2_comparison.md';ib=bind(inspection)
 rows=load('data/RGM-001/candidate_renderings/planning-gap-v2.jsonl');pred={r['sample_id']:r for r in load('reviews/expert/planning-gap-v2.expert_input_predictions.jsonl')};per={r['sample_id']:r for r in a['per_record']};now=datetime.datetime.now(datetime.timezone.utc).isoformat();receipts=[]
 for r in rows:
  sid=r['sample_id'];p=per[sid];pr=pred[sid];facts=pr['independent_input_facts'];minimum=len(pr['expected_target']['shortest_plan']) if pr['expected_target']['shortest_plan'] is not None else None
  issues=list(p['failures'])
  issues.append({'category':'bounded_development_proxy_and_correlated_views','severity':'moderate','evidence':'48 revised relational source families,204 semantic inputs and408 renderer rows across two existing grammars. Same-family interventions/field order do not establish additional mechanism classes, independent model errors, sealed OOD or model improvement.','fix':'Keep the public scope, parent lineage and measured family counts; evaluate model effects separately.'})
  if r['primary_stratum']=='B':issues.append({'category':'conditional_length_and_failure_position_shortcut','severity':'moderate','evidence':'In all12 B families, B0/B1 submitted length3, B2 length2, B3 length5; all B1/B2 missing-edge failures are index1. These conditional cues remain despite true directed-edge controls and ties.','fix':'For a separately designed future wave, cross longer/shorter legal and illegal routes and failure positions without weakening the present public requirements.'})
  elif r['controlled_context']=='A3':issues.append({'category':'fixed_missing_edge_failure_position','severity':'moderate','evidence':'All12 A3 semantic inputs fail missing directed edge at index1. No broader first-error-position diversity is established for this context.','fix':'Disclose the conditional support; consider future position-crossed controls.'})
  elif r['primary_stratum']=='D':issues.append({'category':'goal_obligation_subset_scope','severity':'low','evidence':'Every D submitted trajectory is legal; D1 varies only the goal and changes goal achievement. The subset is a clean goal control but does not supply illegal D trajectories.','fix':'Keep goal-versus-legality scope explicit; measure the whole corpus rather than claim this subset covers both.'})
  issues.append({'category':'difficulty_is_structural_not_model_calibrated','severity':'low','evidence':{'difficulty':r['difficulty'],'fresh_minimum_moves':minimum,'submitted_moves':len(r['semantic_task']['plan']),'first_error':facts['measurements']['first_error']},'fix':'Retain bounded2–4 labels; object count and detour filler are not empirical difficulty or OOD evidence.'})
  failures=p['failures'];decision='QUARANTINE' if any(f['category'] in ('full_record_hash_exact','frozen_view_exact') for f in failures) else 'REVISE' if failures else 'ACCEPT'
  receipts.append({'receipt_id':'expert-planning-gap-v2-'+sha(canon([sid,r['record_hash'],ROLE]).encode())[:24],'sample_id':sid,'record_hash':r['record_hash'],'reviewer_role':'expert_reviewer','reviewer_execution_id':ROLE,'review_stage':'CANONICAL_COMPARISON_FIRST_FROZEN','review_status':'ACCEPTED' if decision=='ACCEPT' else 'NEEDS_REVISION','scores':{'semantic_correctness':5 if p['checks']['frozen_view_exact'] else 1,'target_correctness':5 if p['checks']['typed_target_exact'] else 1,'instruction_clarity':5,'reasoning_validity':5 if p['checks']['alternative_exact_fresh_minimum'] else 1,'generalization_value':3,'difficulty_quality':3,'diversity':3,'shortcut_resistance':2 if r['primary_stratum']=='B' else 3},'critical_defects':failures,'issues':issues,'decision':decision,'confidence':0.985 if decision=='ACCEPT' else 0.98,'evidence':{'prediction_file_sha256':FILES['reviews/expert/planning-gap-v2.expert_input_predictions.jsonl'],'own_input_prediction_target_sha256':sha(canon(pr['expected_target']).encode()),'exact_candidate_target_match':p['checks']['typed_target_exact'],'exact_record_hash_verified':p['checks']['full_record_hash_exact'],'frozen_six_field_view_verified':p['checks']['frozen_view_exact'],'completed_independent_input_search':facts['measurements']['search'],'manifest_sha256':FILES['data/RGM-001/manifests/planning-gap-v2.family_controls.json'],'bound_verified_witnesses':p['witnesses'],'verified_control_ids':p['control_ids'],'per_record_checks':p['checks'],'analysis_sha256':ab['sha256'],'actual_agent_inspection_sha256':ib['sha256']},'guidance':GUIDANCE,'scope':a['scope'],'created_utc':now})
 raw=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in receipts).encode()
 first='reviews/expert/planning-gap-v2.first.jsonl';final='reviews/expert/planning-gap-v2.jsonl'
 for path in (first,final):
  with (ROOT/path).open('xb') as f:f.write(raw)
 summary={'actual_author':ROLE,'stage':'V2_ACTUAL_FIRST_AND_FINAL_COMPARISON_RECEIPTS_FROZEN','created_utc':now,'decision_counts':dict(collections.Counter(r['decision'] for r in receipts)),'records':408,'target_matches':a['typed_target_matches'],'hash_matches':a['exact_hash_matches'],'frozen_view_matches':a['frozen_view_matches'],'manifest_witnesses':a['manifest_row_witnesses'],'renderer_controls':a['supplied_control_declarations'],'first':bind(first),'final':bind(final),'first_final_byte_identical':True,'analysis':ab,'actual_agent_inspection':ib,'source':bind(str(pathlib.Path(__file__).relative_to(ROOT))),'input_predictions_and_initial_judgments_unchanged':all(bind(p)['sha256']==h for p,h in FILES.items()),'other_private_verdicts_read_before_first_freeze':False,'guidance':GUIDANCE,'no_dataset_publication_approval':True,'actual_exit_code':0}
 path='reviews/expert/planning-gap-v2.expert_comparison.execution.json';json_write(path,summary)
 print(json.dumps({'execution':bind(path),'first':bind(first),'final':bind(final),'decisions':summary['decision_counts']}))

if __name__=='__main__':
 if sys.argv[1:] == ['derive']:derive()
 elif sys.argv[1:] == ['finalize']:finalize()
 else:raise ValueError('explicit derive/finalize stage required')
