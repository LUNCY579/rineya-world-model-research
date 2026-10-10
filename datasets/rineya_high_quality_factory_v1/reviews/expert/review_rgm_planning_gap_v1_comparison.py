#!/opt/anaconda3/bin/python3
"""Actual expert comparison, using immutable own pre-gold predictions.
No producer/checker/reference/private verdict imports. Derive evidence first;
actual-agent inspection and immutable first/final receipts use a separate mode.
"""
import collections,copy,datetime,hashlib,itertools,json,pathlib,re,runpy,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
sha=lambda b:hashlib.sha256(b).hexdigest()
PREPSHA='82ae6d000d5289bf439599579eb6437de5a09468388ead4fb1cef95f9dd3a987'
PREP=OUT/'prepare_rgm_planning_gap_v1.py'
assert sha(PREP.read_bytes())==PREPSHA
P=runpy.run_path(str(PREP));canon,eq=P['canon'],P['eq']
BINDINGS={
 'data/RGM-001/candidate_renderings/planning-gap-v1.jsonl':'a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27',
 'data/RGM-001/manifests/planning-gap-v1.family_controls.json':'405e97f28c81e791e8614b369e8af89fa08b74da49a66114315a217bcf083233',
 'research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl':'896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d',
 'reviews/expert/planning-gap-v1.expert_input_predictions.jsonl':'5fe3101f6ca6b6dcda5518b92fe659ffe254bf7862992609bccf9e7fc018f5b3',
 'reviews/expert/planning-gap-v1.expert_first_input_judgments.jsonl':'d6b66b8906628ba7d3d86d7cbc6efe3209343dd0856463ed2ba08ee6ec3209bb',
 'reviews/expert/planning-gap-v1.expert_input_stage.preexecution_source.py':'ae3242d5edbf90c1ce3f897d64acaa5eabf95cdc22980b48626bfff2a47fc839',
 'reviews/expert/prepare_rgm_planning_gap_v1.py':PREPSHA,
 **P['PUBLIC']}
GUIDANCE=('Public prospective guidance and own prior semantics knowledge; common underlying model and architect design/adjudication overlap disclosed. Own exact full targets/prefix/support facts frozen before canonical gold and controls. No new producer/reference/formal/checker, root diagnostics or other private reviewer verdict read before these first comparison judgments. Different executions do not establish independent error distributions.')
PROPERTIES={'legal_prefix_then_first_failure','fuel_lengthens_fresh_minimum_moves','fuel_blocks_topologically_reachable_goal','valid_goal_detour','later_action_requires_changed_block_clearance','partial_block_goal','required_temporary_goal_displacement'}
def save(path,value):
 assert not path.exists(),str(path)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 return {'path':str(path),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
def archive():
 b=pathlib.Path(__file__).read_bytes();ap=OUT/'planning-gap-v1.expert_comparison.preexecution_source.py'
 assert ap.exists() and ap.read_bytes()==b,'Own source must be archived BEFORE execution'
 return {'source_sha256':sha(b),'preexecution_archive_path':str(ap),'preexecution_archive_sha256':sha(ap.read_bytes())}
def bound_reads():
 sources={};loaded={}
 for f,h in BINDINGS.items():
  b=(ROOT/f).read_bytes();assert sha(b)==h,(f,sha(b));sources[f]={'sha256':h,'bytes':len(b)}
  if f.endswith('.jsonl'):loaded[f]=[json.loads(l) for l in b.splitlines()]
  elif f.endswith('.json'):loaded[f]=json.loads(b)
 return sources,loaded
def norm(t):
 x=copy.deepcopy(t)
 if x['kind']=='graph':x['nodes'].sort();x['edges'].sort()
 return x
def task_changes(a,b):return sorted(k for k in set(a)|set(b) if not eq(a.get(k),b.get(k)))
def graph_sig(t,weighted=True):
 # Exact renaming invariant. Terminals fixed to 0/1 for weighted roles;
 # topology-only signature exhausts every permutation, ignores cost/roles.
 ns=sorted(t['nodes']);n=len(ns)
 if weighted:
  roles=[t['start']]+([] if t['start']==t['goal'] else [t['goal']])
  rest=[v for v in ns if v not in roles];perms=(roles+list(p) for p in itertools.permutations(rest))
 else:perms=itertools.permutations(ns)
 best=None
 for order in perms:
  ren={v:i for i,v in enumerate(order)}
  es=tuple(sorted((ren[a],ren[b],c) if weighted else (ren[a],ren[b]) for a,b,c in t['edges']))
  if best is None or es<best:best=es
 return (n, int(t['start']==t['goal']) if weighted else -1,best)
def block_sig(t):
 ns=sorted(t['on']);n=len(ns);best=None
 for order in itertools.permutations(ns):
  ren={v:i for i,v in enumerate(order)}
  on=tuple(n if t['on'][v]=='TABLE' else ren[t['on'][v]] for v in order)
  go=tuple(-1 if v not in t['goal_on'] else n if t['goal_on'][v]=='TABLE' else ren[t['goal_on'][v]] for v in order)
  sig=(on,go)
  if best is None or sig<best:best=sig
 return (n,best)
def block_shape(t):
 above={to:b for b,to in t['on'].items() if to!='TABLE'};lengths=[]
 for b,to in t['on'].items():
  if to=='TABLE':
   count=1
   while b in above:count+=1;b=above[b]
   lengths.append(count)
 return tuple(sorted(lengths))
def sig_calibration():
 g={'kind':'graph','nodes':['s','a','g'],'edges':[['s','a',2],['a','g',3]],'start':'s','goal':'g','fuel':5,'plan':['a','g']}
 rn=P['rename_graph'](g,{'s':'z','a':'x','g':'y'});rn['nodes'].reverse();rn['edges'].reverse()
 assert graph_sig(g)==graph_sig(rn) and graph_sig(g,False)==graph_sig(rn,False)
 different=copy.deepcopy(g);different['edges'][0][2]=4
 assert graph_sig(g)!=graph_sig(different) and graph_sig(g,False)==graph_sig(different,False)
 different=copy.deepcopy(g);different['start']='a';assert graph_sig(g)!=graph_sig(different)
 b={'kind':'blocks','on':{'a':'TABLE','b':'a','c':'TABLE'},'goal_on':{'c':'b'},'plan':[]}
 rb={'kind':'blocks','on':{'y':'z','z':'TABLE','x':'TABLE'},'goal_on':{'x':'y'},'plan':[]}
 assert block_sig(b)==block_sig(rb) and block_shape(b)==block_shape(rb)
 eb=copy.deepcopy(b);eb['goal_on']['a']='TABLE';assert block_sig(b)!=block_sig(eb)
 return {'graph_order_rename_invariance':True,'cost_changes_weighted_only':True,'terminal_roles_preserved':True,'block_joint_rename_invariance':True,'partial_goal_presence_preserved':True,'fixtures':'Own authored tiny hand-derived invariance cases in frozen comparison source; no new semantic solver or gold-trained calibration.'}
def derive():
 start=time.perf_counter();own=archive();sources,loaded=bound_reads();cal=sig_calibration()
 rows=loaded[next(f for f in BINDINGS if 'candidate_renderings' in f)];manifest=loaded[next(f for f in BINDINGS if 'family_controls' in f)];prior=loaded[next(f for f in BINDINGS if 'PRIOR_ACTIVE' in f)]
 preds=loaded['reviews/expert/planning-gap-v1.expert_input_predictions.jsonl'];pred={r['sample_id']:r for r in preds};by={r['sample_id']:r for r in rows}
 assert len(rows)==len(by)==len(pred)==384 and set(pred)==set(by)
 per={sid:{'sample_id':sid,'record_hash':r['record_hash'],'failures':[],'checks':{},'manifest_witnesses':[]} for sid,r in by.items()}
 def fail(sids,category,evidence):
  for sid in sids:per[sid]['failures'].append({'category':category,'severity':'high','evidence':evidence,'fix':'Regenerate a new version correcting this exact public-contract or metadata defect; preserve original candidate and initial review bytes.'})
 def check(sids,name,ok,evidence):
  for sid in sids:per[sid]['checks'][name]=bool(ok)
  if not ok:fail(sids,name,evidence)
 for sid,r in by.items():
  pr=pred[sid];t=r['semantic_task'];P['validate'](t)
  check([sid],'record_hash_exact',sha(canon({k:v for k,v in r.items() if k!='record_hash'}).encode())==r['record_hash'],'Full typed canonical record hash differs')
  check([sid],'frozen_input_exact',r['record_hash']==pr['record_hash'] and r['source_family_key']==pr['source_family_key'] and r['split']==pr['split'] and sha(canon(t).encode())==pr['semantic_task_sha256'] and sha(r['rendered_input'].encode())==pr['student_prompt_sha256'],'Canonical input or supplied hash differs from frozen pre-gold input')
  check([sid],'typed_target_exact',eq(r['verified_target'],pr['expected_target']),{'frozen_prediction':pr['expected_target'],'canonical_target':r['verified_target']})
  check([sid],'family_identity',r['source_family_key']==f"RGM-001-seed20261009-p{r['source_family_index']:04d}" and r['semantic_family_id']==r['semantic_parent_id']==f"RGM-001-planning-gap-v1-p{r['source_family_index']:04d}",'Incorrect family key/ID/index custody')
  check([sid],'source_split_rule',r['split']==('dev' if r['source_family_index']%10>=8 else 'train'),'Split differs from public rule')
  check([sid],'bounded_public_difficulty',type(r['difficulty']) is int and r['difficulty']==r['structural_difficulty'] and 2<=r['difficulty']<=4,'Difficulty not supported bounded 2–4 integer declaration')
  tags=r['skill_tags'];check([sid],'optional_property_tags_true',isinstance(tags,list) and len(set(tags))==len(tags) and set(tags)-{'finite_synthetic_proxy','relational_planning'}<=PROPERTIES and set(tags)&PROPERTIES<=set(pr['independently_witnessed_property_names']),'False, undefined or duplicated property tag')
  per[sid]['minimum_moves']=None if pr['expected_target']['shortest_plan'] is None else len(pr['expected_target']['shortest_plan'])
  per[sid]['first_error']=pr['independent_input_facts']['first_error']
  per[sid]['input_derived_properties']=pr['independently_witnessed_property_names']
 assert set(manifest)=={'version','track_id','batch_name','semantics_sha256','families'}
 assert manifest['track_id']=='RGM-001' and manifest['batch_name']=='planning-gap-v1' and manifest['semantics_sha256']==P['PUBLIC']['schemas/SEMANTICS.md']
 assert (type(manifest['version']) is str and bool(manifest['version'])) or (type(manifest['version']) is int and manifest['version']>0)
 families=manifest['families'];assert len(families)==len({f['source_family_key'] for f in families})==48
 claimed_sids=[];familyfacts=[];witness_count=0;control_count=0
 for f in families:
  key=f['source_family_key'];sids=[r['sample_id'] for r in rows if r['source_family_key']==key];st=f['primary_stratum'];cs={c['context_id']:c for c in f['contexts']}
  check(sids,'manifest_family_identity',len(sids)==8 and st in 'ABCD' and f['source_index']==by[sids[0]]['source_family_index'] and f['split']==by[sids[0]]['split'] and f['base_sample_id'] in sids,'Manifest family fields do not bind actual family')
  check(sids,'four_contexts',set(cs)=={st+str(i) for i in range(4)} and len(f['contexts'])==4,'Missing/duplicated context')
  cx={}
  for cid,c in cs.items():
   ids=c['sample_ids'];claimed_sids+=ids
   check(ids,'manifest_context_custody',len(ids)==len(set(ids))==2 and all(i in sids and by[i]['controlled_context']==cid and by[i]['primary_stratum']==st for i in ids),'Context sample IDs/custody differ')
   check(ids,'renderer_equivalence',set(by[i]['renderer'] for i in ids)=={'forward','reverse'} and eq(norm(by[ids[0]]['semantic_task']),norm(by[ids[1]]['semantic_task'])) and eq(pred[ids[0]]['expected_target'],pred[ids[1]]['expected_target']),'Declared renderer pair is not semantically identical')
   cx[cid]=next(i for i in ids if by[i]['renderer']=='forward')
   ps=c['row_properties'];check(ids,'manifest_property_set',type(ps) is list and len(set(ps))==len(ps) and set(ps)<=PROPERTIES and all(set(ps)==set(by[i]['skill_tags'])&PROPERTIES for i in ids),'Context property list differs from optional per-row tags or public names')
   ws=c['row_witnesses'];pairs=[]
   for w in ws:
    witness_count+=1;i=w['sample_id'];p=w['property'];e=w['evidence'];pairs.append((i,p));t=by[i]['semantic_task'];pr=pred[i];ft=pr['independent_input_facts'];target=pr['expected_target'];good=set(w)=={'sample_id','property','scope','evidence'} and i in ids and p in ps and w['scope']=='row' and type(e) is dict and bool(e)
    if p=='legal_prefix_then_first_failure':good=good and ft['first_error'] is not None and eq(e,{'first_error_index':ft['first_error']['index']}) and ft['first_error']['index']>=1
    elif p=='fuel_lengthens_fresh_minimum_moves':good=good and eq(e,{'minimum_moves':len(target['shortest_plan']),'topology_distance':len(ft['topology_shortest_plan'])})
    elif p=='fuel_blocks_topologically_reachable_goal':good=good and eq(e,{'cheapest_fuel':ft['cheapest_required_fuel'],'topology_distance':len(ft['topology_shortest_plan'])})
    elif p in {'valid_goal_detour','required_temporary_goal_displacement'}:good=good and eq(e,{'minimum_moves':len(target['shortest_plan'])})
    elif p=='partial_block_goal':good=good and eq(e,{'block_count':len(t['on']),'goal_relation_count':len(t['goal_on'])})
    elif p=='later_action_requires_changed_block_clearance':
     ai=e.get('action_index');j=e.get('changed_by_action_index');pre=e.get('precondition');valid_index=type(ai) is int and type(j) is int and 0<=j<ai<len(target['steps']) and target['steps'][ai]['legal'] and pre in {'source','destination'}
     if valid_index:
      action=t['plan'][ai];obj=action['block'] if pre=='source' else action['to'];states=ft['prefix_states'];blocker=t['plan'][j]['block']
      initial_blocked=obj!='TABLE' and obj in t['on'].values();clear_before=obj not in states[j].values();clear_after=obj not in states[j+1].values();clear_at_action=obj not in states[ai].values()
      relevant=states[j].get(blocker)==obj and states[j+1].get(blocker)!=obj
      good=good and set(e)=={'action_index','changed_by_action_index','precondition'} and initial_blocked and not clear_before and clear_after and clear_at_action and relevant
     else:good=False
    else:good=False
    good=good and p in pr['independently_witnessed_property_names']
    check([i],f'manifest_witness:{p}',good,{'witness':w,'independent_facts':pr['independent_row_witness_facts'],'first_error':ft['first_error']})
    per[i]['manifest_witnesses'].append({'property':p,'evidence':e,'independently_checked':bool(good)})
   check(ids,'witness_coverage',collections.Counter(pairs)==collections.Counter((i,p) for i in ids for p in ps),'Each tagged sample/property must have exactly its own witness')
  def target(cid):return pred[cx[st+str(cid)]]['expected_target']
  def facts(cid):return pred[cx[st+str(cid)]]['independent_input_facts']
  def task(cid):return by[cx[st+str(cid)]]['semantic_task']
  def minimum(cid):return len(target(cid)['shortest_plan']) if target(cid)['shortest_plan'] is not None else None
  def contextcheck(name,good,e):check(sids,'family_'+name,good,e)
  contextcheck('base_valid_goal',target(0)['valid'] and target(0)['reaches_goal'],'Base submitted plan must be legal goal reaching')
  if st=='A':
   changes=task_changes(task(0),task(1));mode='infeasible_boundary' if minimum(1) is None else 'longer_feasible'
   a1good=changes==['fuel'] and not target(1)['valid'] and ((minimum(1) is None and facts(1)['topology_shortest_plan'] is not None and task(0)['fuel']==facts(0)['cheapest_required_fuel'] and task(1)['fuel']==task(0)['fuel']-1) or (minimum(1) is not None and minimum(1)>minimum(0)))
   contextcheck('A1_decisive_fuel',a1good,{'changed_fields':changes,'mode':mode})
   contextcheck('A2_legal_non_goal_prefix',task_changes(task(0),task(2))==['plan'] and len(task(2)['plan'])>=1 and target(2)['valid'] and not target(2)['reaches_goal'],'A2 must keep A0 world and be legal non-goal, no base-route-copy requirement')
   contextcheck('A3_missing_edge_after_prefix',task_changes(task(0),task(3))==['plan'] and facts(3)['first_error'] is not None and facts(3)['first_error']['index']>=1 and facts(3)['first_error']['reason']=='missing_edge','A3 missing directed edge after legal prefix')
  elif st=='B':
   es0={tuple(x) for x in task(0)['edges']};es1={tuple(x) for x in task(1)['edges']};deleted=es0-es1;er=facts(1)['first_error'];i=er['index'] if er else -1
   prior_node=task(1)['start'] if i==0 else task(1)['plan'][i-1] if i>0 else None;dest=task(1)['plan'][i] if i>=0 else None
   contextcheck('B0_horizon',len(task(0)['plan'])>=3,'B0 submitted plan length below three')
   contextcheck('B1_edge_only_decisive',task_changes(task(0),task(1))==['edges'] and len(deleted)==1 and not(es1-es0) and i>=1 and er['reason']=='missing_edge' and any(a==prior_node and b==dest for a,b,c in deleted) and minimum(0) is not None and minimum(1) is not None,{'deleted_edges':list(deleted),'first_error':er})
   er=facts(2)['first_error'];i=er['index'] if er else -1;edges={(a,b) for a,b,c in task(2)['edges']};a=task(2)['plan'][i-1] if i>0 else task(2)['start'];b=task(2)['plan'][i] if i>=0 else None
   contextcheck('B2_reverse_missing',i>=1 and er['reason']=='missing_edge' and (b,a) in edges and (a,b) not in edges,{'attempted_edge':[a,b],'first_error':er})
   contextcheck('B3_detour','valid_goal_detour' in pred[cx['B3']]['independently_witnessed_property_names'] and task_changes(task(0),task(3))==['plan'],'B3 must be same world legal goal detour')
  elif st=='C':
   contextcheck('C0_clearance_minimum',4<=minimum(0)<=8 and len(task(0)['plan'])==minimum(0) and 'later_action_requires_changed_block_clearance' in pred[cx['C0']]['independently_witnessed_property_names'],'C0 fresh minimum4–8 with actual changed clearance')
   er=facts(1)['first_error'];a=task(0)['plan'];b=task(1)['plan'];swap=[j for j in range(len(a)-1) if eq(a[:j]+[a[j+1],a[j]]+a[j+2:],b)]
   contextcheck('C1_adjacent_clearance_swap',task_changes(task(0),task(1))==['plan'] and bool(swap) and er is not None and er['reason'] in {'source_not_clear','destination_not_clear'}, {'adjacent_swap_indices':swap,'first_error':er})
   contextcheck('C2_detour',task_changes(task(0),task(2))==['plan'] and 'valid_goal_detour' in pred[cx['C2']]['independently_witnessed_property_names'],'C2 same world valid goal detour')
   contextcheck('C3_legal_non_goal_prefix',task_changes(task(0),task(3))==['plan'] and len(task(3)['plan'])>=1 and target(3)['valid'] and not target(3)['reaches_goal'],'C3 same world legal non-goal prefix')
  else:
   g0=task(0)['goal_on'];g1=task(1)['goal_on'];extension=len(g1)>len(g0) and all(g1.get(b)==v for b,v in g0.items())
   contextcheck('D0_partial_minimum',0<len(g0)<len(task(0)['on']) and minimum(0)>=2 and len(task(0)['plan'])==minimum(0),'D0 nonempty proper partial goal with minimum>=2')
   contextcheck('D1_only_goal_extension',task_changes(task(0),task(1))==['goal_on'] and extension and target(1)['valid'] and not target(1)['reaches_goal'] and minimum(1)>minimum(0),'D1 valid goal-only extension same plan, strictly longer fresh minimum')
   contextcheck('D2_new_goal_minimum',task_changes(task(1),task(2))==['plan'] and target(2)['valid'] and target(2)['reaches_goal'] and len(task(2)['plan'])==minimum(2),'D2 same extended world minimum plan')
   contextcheck('D3_new_goal_detour',task_changes(task(2),task(3))==['plan'] and 'valid_goal_detour' in pred[cx['D3']]['independently_witnessed_property_names'],'D3 same extended world legal goal detour')
  names=[]
  for co in f['controls']:
   control_count+=1;ids=co['sample_ids'];names.append(co['name']);good=set(co)=={'control_id','name','scope','sample_ids','changed_task_fields','provenance'} and type(co['control_id']) is str and bool(co['control_id']) and len(ids)==2 and all(i in sids for i in ids) and co['scope']=='corpus_variant' and co['provenance']=='supplied_tasks'
   a,b=(by[i]['semantic_task'] for i in ids);ta,tb=(pred[i]['expected_target'] for i in ids);diff=task_changes(a,b)
   good=good and eq(sorted(co['changed_task_fields']),diff)
   name=co['name']
   if name=='fuel_only_counterfactual':good=good and diff==['fuel']
   elif name=='directed_edge_deletion_counterfactual':good=good and diff==['edges'] and len({tuple(e) for e in a['edges']}-{tuple(e) for e in b['edges']})==1 and not({tuple(e) for e in b['edges']}-{tuple(e) for e in a['edges']})
   elif name=='adjacent_action_order_counterfactual':good=good and diff==['plan'] and any(eq(a['plan'][:j]+[a['plan'][j+1],a['plan'][j]]+a['plan'][j+2:],b['plan']) for j in range(len(a['plan'])-1))
   elif name=='partial_goal_extension_counterfactual':good=good and diff==['goal_on'] and len(b['goal_on'])>len(a['goal_on']) and all(b['goal_on'].get(k)==v for k,v in a['goal_on'].items())
   elif name=='same_world_valid_goal_detour':good=good and diff==['plan'] and ta['valid'] and ta['reaches_goal'] and tb['valid'] and tb['reaches_goal'] and len(b['plan'])>len(a['plan']) and 'valid_goal_detour' in pred[ids[1]]['independently_witnessed_property_names']
   else:good=False
   check(ids,'supplied_control:'+name,good,{'control':co,'actual_changed_task_fields':diff})
  required={'A':['fuel_only_counterfactual'],'B':['directed_edge_deletion_counterfactual','same_world_valid_goal_detour'],'C':['adjacent_action_order_counterfactual','same_world_valid_goal_detour'],'D':['partial_goal_extension_counterfactual','same_world_valid_goal_detour']}[st]
  contextcheck('required_supplied_controls',collections.Counter(names)==collections.Counter(required),{'actual_names':names,'required_names':required})
  familyfacts.append({'source_family_key':key,'source_index':f['source_index'],'split':f['split'],'stratum':st,'base_sample_id':f['base_sample_id'],'sample_ids':sids,'base_task':task(0),'A_mode':mode if st=='A' else None,'contexts':[{'context':st+str(i),'sample_id':cx[st+str(i)],'submitted_length':len(task(i)['plan']),'minimum_moves':minimum(i),'valid':target(i)['valid'],'goal':target(i)['reaches_goal'],'first_error':facts(i)['first_error'],'route_count':facts(i).get('minimum_feasible_route_count'),'properties':pred[cx[st+str(i)]]['independently_witnessed_property_names'],'difficulty':by[cx[st+str(i)]]['difficulty']} for i in range(4)]})
 assert collections.Counter(claimed_sids)==collections.Counter(by)
 allids=list(by);globalchecks=[]
 def globalcheck(name,good,details,ids=allids):
  globalchecks.append({'check':name,'pass':bool(good),'evidence':details})
  if not good:fail(ids,'cohort_'+name,details)
 strata=collections.Counter(f['stratum'] for f in familyfacts);splits=collections.Counter(f['split'] for f in familyfacts)
 globalcheck('48_new_family_allocation',strata==dict.fromkeys('ABCD',12) and splits=={'train':40,'dev':8} and {f['source_index'] for f in familyfacts}==set(range(180,228)),{'strata':strata,'split_families':splits})
 af=[f for f in familyfacts if f['stratum']=='A'];modes=collections.Counter(f['A_mode'] for f in af)
 a2=collections.Counter(f['contexts'][2]['submitted_length'] for f in af);a3=collections.Counter(f['contexts'][3]['submitted_length'] for f in af)
 globalcheck('A_fuel_modes_and_length_matching',modes=={'infeasible_boundary':6,'longer_feasible':6} and a2==a3,{'fuel_modes':modes,'A2_lengths':a2,'A3_lengths':a3})
 bf=[f for f in familyfacts if f['stratum']=='B'];tied=sum(f['contexts'][0]['route_count']>=2 for f in bf)
 globalcheck('B_lexical_tie_coverage',tied>=6,{'base_worlds_with_multiple_minimum_routes':tied,'counts':[(f['base_sample_id'],f['contexts'][0]['route_count']) for f in bf]})
 cf=[f for f in familyfacts if f['stratum']=='C'];errs=collections.Counter(f['contexts'][1]['first_error']['index'] for f in cf);temp=sum('required_temporary_goal_displacement' in f['contexts'][0]['properties'] for f in cf)
 globalcheck('C_clearance_and_displacement_variation',len(errs)>=2 and temp>=4,{'C1_error_indices':errs,'C0_required_displacement_families':temp})
 # Novelty uses prior input tasks only: no target or historical verdict is loaded.
 assert all(set(r)=={'sample_id','original_record_hash','source_family_key','split','semantic_task'} for r in prior)
 priorgraph={};priorblock={};unweightedprior={};priorcached={};current_sigs=[]
 for r in prior:
  t=r['semantic_task'];P['validate'](t)
  if t['kind']=='graph':
   raw=canon([sorted(t['nodes']),sorted(t['edges']),t['start'],t['goal']]);key=('g',raw)
   if key not in priorcached:priorcached[key]=graph_sig(t)
   priorgraph.setdefault(priorcached[key],[]).append(r['sample_id'])
   raw=canon([sorted(t['nodes']),sorted((a,b) for a,b,c in t['edges'])]);key=('u',raw)
   if key not in priorcached:priorcached[key]=graph_sig(t,False)
   unweightedprior.setdefault(priorcached[key],[]).append(r['sample_id'])
  else:
   raw=canon([t['on'],t['goal_on']]);key=('b',raw)
   if key not in priorcached:priorcached[key]=block_sig(t)
   priorblock.setdefault(priorcached[key],[]).append(r['sample_id'])
 weightedgroups=collections.defaultdict(list);blockgroups=collections.defaultdict(list);ugroups=collections.defaultdict(list);sizes=collections.Counter();sixshapes=set();novelty=[]
 for f in familyfacts:
  t=f['base_task'];n=len(t['nodes']) if t['kind']=='graph' else len(t['on']);sizes[(f['stratum'],f['split'],n)]+=1
  if t['kind']=='graph':
   sig=graph_sig(t);usig=graph_sig(t,False);weightedgroups[sig].append(f);ugroups[usig].append(f);collision=priorgraph.get(sig,[]);uprior=unweightedprior.get(usig,[])
   entry={'base_sample_id':f['base_sample_id'],'source_family_key':f['source_family_key'],'kind':'graph','objects':n,'weighted_terminal_signature_sha256':sha(canon(sig).encode()),'unweighted_topology_signature_sha256':sha(canon(usig).encode()),'prior_weighted_collision_ids':collision,'prior_unweighted_collision_ids':uprior}
  else:
   sig=block_sig(t);blockgroups[sig].append(f);collision=priorblock.get(sig,[]);shape=block_shape(t)
   if n==6:sixshapes.add(shape)
   entry={'base_sample_id':f['base_sample_id'],'source_family_key':f['source_family_key'],'kind':'blocks','objects':n,'joint_support_signature_sha256':sha(canon(sig).encode()),'initial_support_shape':shape,'prior_joint_collision_ids':collision}
  check(f['sample_ids'],'absent_prior_base_isomorphism',not collision,{'prior_collision_sample_ids':collision,'signature':entry});novelty.append(entry)
 for kind,groups in [('graph',weightedgroups),('blocks',blockgroups)]:
  for sig,fs in groups.items():
   if len(fs)>1:
    ids=[i for f in fs for i in f['sample_ids']];fail(ids,'fresh_base_isomorphism_collision',{'kind':kind,'base_ids':[f['base_sample_id'] for f in fs]})
 newu=sum(s not in unweightedprior for s in ugroups)
 globalcheck('graph_unweighted_novelty',newu>=16,{'distinct_current_unweighted_classes':len(ugroups),'new_unweighted_classes_absent_all_prior_tasks':newu})
 globalcheck('block_sizes_and_six_support_shapes',sum(f['base_task']['kind']=='blocks' and len(f['base_task']['on'])==5 for f in familyfacts)==12 and sum(f['base_task']['kind']=='blocks' and len(f['base_task']['on'])==6 for f in familyfacts)==12 and len(sixshapes)>=8,{'six_block_shape_classes':len(sixshapes),'six_block_shapes':sorted(sixshapes)})
 for st in 'CD':
  globalcheck('block_size_split_'+st,all(sizes[(st,sp,n)]==count for sp,count in [('train',5),('dev',1)] for n in [5,6]),{str(k):v for k,v in sizes.items() if k[0]==st})
 for st in 'AB':
  globalcheck('graph_train_size_'+st,{n for s,sp,n in sizes if s==st and sp=='train'}=={5,6,7},{str(k):v for k,v in sizes.items() if k[0]==st})
 globalcheck('graph_dev_sizes_across_strata',{n for s,sp,n in sizes if s in 'AB' and sp=='dev'}=={5,6,7},{str(k):v for k,v in sizes.items() if k[0] in 'AB'})
 # Component shortcut diagnostics: label-free visible features, fit train only.
 def features(r):
  t=r['semantic_task'];kind=t['kind'];n=len(t['nodes']) if kind=='graph' else len(t['on']);f={'kind':kind,'object_count':n,'submitted_length':len(t['plan']),'prompt_character_length':len(r['rendered_input']),'task_character_length':len(canon(t)),'distinct_plan_actions':len({canon(a) for a in t['plan']})}
  if kind=='graph':f.update({'edge_count':len(t['edges']),'fuel':t['fuel'],'total_edge_cost':sum(e[2] for e in t['edges']),'plan_last_is_goal':bool(t['plan']) and t['plan'][-1]==t['goal']})
  else:f.update({'goal_relation_count':len(t['goal_on']),'initial_table_roots':sum(v=='TABLE' for v in t['on'].values()),'goal_table_roots':sum(v=='TABLE' for v in t['goal_on'].values()),'submitted_TABLE_count':sum(a['to']=='TABLE' for a in t['plan'])})
  return f
 feats={r['sample_id']:features(r) for r in rows};train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];baselines=[]
 labels={'valid':lambda p:p['expected_target']['valid'],'reaches_goal':lambda p:p['expected_target']['reaches_goal'],'minimum_feasible':lambda p:p['expected_target']['shortest_plan'] is not None,'status':lambda p:(p['expected_target']['valid'],p['expected_target']['reaches_goal']),'first_error_index':lambda p:None if p['independent_input_facts']['first_error'] is None else p['independent_input_facts']['first_error']['index']}
 for lname,label in labels.items():
  fallback=collections.Counter(canon(label(pred[r['sample_id']])) for r in train).most_common(1)[0][0]
  names=sorted({k for d in feats.values() for k in d});feature_sets=[[]]+[[k] for k in names]+[['kind','submitted_length'],['kind','object_count','submitted_length'],['kind','object_count','submitted_length','edge_count','fuel','goal_relation_count','submitted_TABLE_count'],['kind','prompt_character_length'],['kind','task_character_length']]
  for fs in feature_sets:
   key=lambda r:canon([feats[r['sample_id']].get(k,'ABSENT') for k in fs]);groups=collections.defaultdict(collections.Counter)
   for r in train:groups[key(r)][canon(label(pred[r['sample_id']]))]+=1
   predictor={k:c.most_common(1)[0][0] for k,c in groups.items()};correct=lambda r:predictor.get(key(r),fallback)==canon(label(pred[r['sample_id']]))
   bad=[r['sample_id'] for r in dev if not correct(r)]
   baselines.append({'target_component':lname,'features':fs or ['train_majority_constant'],'train_correct':sum(correct(r) for r in train),'train_n':len(train),'exposed_dev_correct':len(dev)-len(bad),'exposed_dev_n':len(dev),'dev_unseen_feature_keys':sum(key(r) not in predictor for r in dev),'first_dev_counterexample':bad[0] if bad else None,'scope':'Categorical empirical train majority per visible feature key, deterministic train-first tie, train-majority fallback. Component prediction only, not full-output task performance.'})
 # Leakage-free task/world table lookup: development families disjoint by exact
 # invariant, so this does not estimate extrapolation from lexical name overlap.
 traing={graph_sig(r['semantic_task']) for r in train if r['semantic_task']['kind']=='graph'};trainb={block_sig(r['semantic_task']) for r in train if r['semantic_task']['kind']=='blocks'}
 devworldhits=[r['sample_id'] for r in dev if (graph_sig(r['semantic_task']) in traing if r['semantic_task']['kind']=='graph' else block_sig(r['semantic_task']) in trainb)]
 conditional=[]
 for st in 'ABCD':
  for cid in range(4):
   rs=[r for r in rows if r['controlled_context']==st+str(cid)];conditional.append({'context':st+str(cid),'rows':len(rs),'status_counts':dict(collections.Counter(canon(labels['status'](pred[r['sample_id']])) for r in rs)),'first_error_counts':dict(collections.Counter(canon(pred[r['sample_id']]['independent_input_facts']['first_error']) for r in rs)),'size_by_split':dict(collections.Counter(canon([r['split'],feats[r['sample_id']]['object_count']]) for r in rs)),'submitted_lengths':dict(collections.Counter(str(len(r['semantic_task']['plan'])) for r in rs)),'minimum_horizons':dict(collections.Counter(str(per[r['sample_id']]['minimum_moves']) for r in rs)),'difficulty_counts':dict(collections.Counter(str(r['difficulty']) for r in rs))})
 failure_counts=collections.Counter(x['category'] for p in per.values() for x in p['failures']);report={'actual_author':ROLE,'stage':'OWN_POST_FREEZE_COMPARISON_DERIVATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guidance':GUIDANCE,'source_bindings':sources,**own,'own_new_isomorphism_calibration':cal,'records':len(rows),'families':len(familyfacts),'manifest_row_witness_count':witness_count,'supplied_control_count':control_count,'typed_target_matches':sum(p['checks']['typed_target_exact'] for p in per.values()),'exact_record_hash_matches':sum(p['checks']['record_hash_exact'] for p in per.values()),'frozen_input_matches':sum(p['checks']['frozen_input_exact'] for p in per.values()),'failure_counts':failure_counts,'rows_with_high_failures':sum(bool(p['failures']) for p in per.values()),'per_record':list(per.values()),'family_context_facts':familyfacts,'cohort_checks':globalchecks,'novelty':novelty,'novelty_summary':{'prior_rows':len(prior),'prior_families':len({r['source_family_key'] for r in prior}),'prior_distinct_weighted_terminal_worlds':len(priorgraph),'prior_unweighted_classes':len(unweightedprior),'prior_joint_block_worlds':len(priorblock),'current_distinct_weighted_terminal_worlds':len(weightedgroups),'current_distinct_unweighted_classes':len(ugroups),'current_new_unweighted_classes':newu,'current_distinct_joint_block_worlds':len(blockgroups),'current_six_block_initial_shapes':len(sixshapes),'sizes':{str(k):v for k,v in sizes.items()}},'shortcut_component_baselines':baselines,'conditional_contexts':conditional,'dev_isomorphic_train_world_hits':devworldhits,'scope':'Independent programmatic every-row comparison to own pre-gold exact typed replay/BFS and substantive actual-agent all48 base input inspection; manifest/control/novelty inspection requires separate final agent report before judgments. No other reviewer verdict or checker read. No release approval.'}
 evidence=save(OUT/'planning-gap-v1.expert_comparison.analysis.json',report)
 execution={'actual_author':ROLE,'stage':'OWN_COMPARISON_DERIVE_ACTUAL_EXECUTION','created_utc':report['created_utc'],'python_executable':sys.executable,'actual_argv':sys.argv,'exit_code':0,'elapsed_seconds':time.perf_counter()-start,**own,'analysis_artifact':evidence,'source_bindings':sources,'records':384,'typed_target_matches':report['typed_target_matches'],'rows_with_high_failures':report['rows_with_high_failures'],'no_other_reviewer_new_source_or_gate_reads':True,'dataset_judgments_not_yet_written':True,'guidance':GUIDANCE}
 save(OUT/'planning-gap-v1.expert_comparison.derive.execution.json',execution);print(canon({k:v for k,v in execution.items() if k!='source_bindings'}))
def freeze():
 own=archive();sources,loaded=bound_reads();analysispath=OUT/'planning-gap-v1.expert_comparison.analysis.json';b=analysispath.read_bytes();a=json.loads(b);ex=json.loads((OUT/'planning-gap-v1.expert_comparison.derive.execution.json').read_text());assert sha(b)==ex['analysis_artifact']['sha256']
 inspection=OUT/'semantic_inspection_planning_gap_v1_comparison.md';assert inspection.exists();ib=inspection.read_bytes();assert len(ib)>1000
 rows=loaded[next(f for f in BINDINGS if 'candidate_renderings' in f)];by={r['sample_id']:r for r in rows};pred={r['sample_id']:r for r in loaded['reviews/expert/planning-gap-v1.expert_input_predictions.jsonl']};now=datetime.datetime.now(datetime.timezone.utc).isoformat();receipts=[]
 for p in a['per_record']:
  sid=p['sample_id'];r=by[sid];pr=pred[sid];st=r['primary_stratum'];er=p['first_error'];failed=p['failures'];issues=list(failed)
  issues.append({'category':'bounded_generalization_and_correlated_contexts','severity':'medium','evidence':f"This exact row belongs to one of48 new base worlds and four supplied contexts with two order views:384 rows,192 semantic tasks, two action grammars. Novelty is structural within the public proxy, not48 new mechanisms, model efficacy, or sealed OOD.",'fix':'Retain family-level split/counts, finite proxy scope and measured fresh topology/support novelty; assess model learning separately.'})
  issues.append({'category':'conditional_template_shortcuts','severity':'medium','evidence':f"Published internal {r['controlled_context']} role predicts component outcomes but is excluded from the prompt. Actual B1/B2 missing-edge cases all fail at index1; C1 errors vary0/1/2; four-context layouts remain correlated. Visible-feature train/dev component baselines and counterexamples are separately measured; none establishes full-output competence.",'fix':'Retain diagnostics and diversify error positions, outcome/length combinations and plan construction before broader competency claims.'})
  issues.append({'category':'difficulty_is_supported_proxy_work_not_empirical_model_score','severity':'low','evidence':f"Declared level{r['difficulty']}; independently fresh minimum{p['minimum_moves']}, submitted moves{len(r['semantic_task']['plan'])}, first error{canon(er)}. Actual clearance/fuel/partial-goal constraints support bounded levels2–4; filler detour length and object count alone do not prove increased difficulty.",'fix':'Keep bounded labels and actual horizons/dependencies; calibrate learning difficulty only with separately collected model measurements.'})
  scores={'semantic_correctness':5 if p['checks']['frozen_input_exact'] else 1,'target_correctness':5 if p['checks']['typed_target_exact'] else 1,'instruction_clarity':5,'reasoning_validity':5 if p['checks']['typed_target_exact'] else 1,'generalization_value':3,'difficulty_quality':3,'diversity':3,'shortcut_resistance':2 if st=='B' else 3}
  if failed and any('witness' in x['category'] or 'property' in x['category'] for x in failed):scores['semantic_correctness']=2
  decision='REVISE' if failed else 'ACCEPT'
  receipts.append({'receipt_id':'expert-planning-gap-v1-'+sha(canon([sid,r['record_hash'],ROLE]).encode())[:24],'sample_id':sid,'record_hash':r['record_hash'],'reviewer_role':'expert_reviewer','reviewer_execution_id':ROLE,'review_stage':'CANONICAL_COMPARISON_FIRST_FROZEN','review_status':'ACCEPTED' if decision=='ACCEPT' else 'NEEDS_REVISION','scores':scores,'critical_defects':failed,'issues':issues,'decision':decision,'confidence':0.985 if decision=='ACCEPT' else 0.98,'evidence':{'frozen_prediction_file_sha256':BINDINGS['reviews/expert/planning-gap-v1.expert_input_predictions.jsonl'],'frozen_prediction_target_sha256':sha(canon(pr['expected_target']).encode()),'exact_canonical_target_type_sensitive_match':p['checks']['typed_target_exact'],'full_record_hash_verified':p['checks']['record_hash_exact'],'frozen_input_verified':p['checks']['frozen_input_exact'],'manifest_sha256':BINDINGS['data/RGM-001/manifests/planning-gap-v1.family_controls.json'],'manifest_witnesses':p['manifest_witnesses'],'all_independent_input_properties':p['input_derived_properties'],'minimum_moves':p['minimum_moves'],'first_error':er,'completed_search':pr['independent_input_facts']['search'],'per_record_checks':p['checks'],'analysis_sha256':sha(b),'actual_agent_comparison_inspection_sha256':sha(ib)},'scope':a['scope'],'guidance':GUIDANCE,'created_utc':now})
 raw=('\n'.join(canon(r) for r in receipts)+'\n').encode();fp=OUT/'planning-gap-v1.first.jsonl';final=OUT/'planning-gap-v1.jsonl';assert not fp.exists() and not final.exists();fp.write_bytes(raw);final.write_bytes(raw)
 summary={'actual_author':ROLE,'stage':'ACTUAL_FIRST_COMPARISON_AND_FINAL_RECEIPTS_FREEZE','created_utc':now,'guidance':GUIDANCE,'python_executable':sys.executable,'actual_argv':sys.argv,'exit_code':0,**own,'source_bindings':sources,'first_receipts':{'path':str(fp),'sha256':sha(raw),'bytes':len(raw),'records':len(receipts)},'final_receipts':{'path':str(final),'sha256':sha(final.read_bytes()),'bytes':final.stat().st_size,'byte_identical_to_actual_first':True},'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'critical_defect_categories':a['failure_counts'],'target_matches':a['typed_target_matches'],'exact_record_hash_matches':a['exact_record_hash_matches'],'frozen_input_matches':a['frozen_input_matches'],'manifest_row_witnesses':a['manifest_row_witness_count'],'supplied_controls':a['supplied_control_count'],'novelty_summary':a['novelty_summary'],'cohort_checks':a['cohort_checks'],'analysis_binding':{'path':str(analysispath),'sha256':sha(b),'bytes':len(b)},'actual_agent_inspection':{'path':str(inspection),'sha256':sha(ib),'bytes':len(ib)},'previous_input_prediction_and_first_input_judgments_unchanged':True,'other_verdict_checker_pipeline_reads_before_first_freeze':False,'no_release_approval':True,'no_candidate_edits':True,'quality_limitations':['Only two executable grammars;48 worlds and192 semantic tasks, not384 independent samples/mechanisms.','Deterministic cohort conditional cues remain (notably B first missing edgeindex1); component baselines are not full-output competence.','Levels2–4 are reasonable structural judgments, not model-calibrated difficulty or sealed OOD.','Public-guided, common-model review and architect design/adjudication overlap disclosed.']}
 save(OUT/'planning-gap-v1.expert_comparison.freeze.json',summary);print(canon({'decisions':summary['decisions'],'first':summary['first_receipts'],'final':summary['final_receipts'],'target_matches':summary['target_matches'],'novelty':summary['novelty_summary']}))
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in {'derive','freeze'}
 derive() if sys.argv[1]=='derive' else freeze()
