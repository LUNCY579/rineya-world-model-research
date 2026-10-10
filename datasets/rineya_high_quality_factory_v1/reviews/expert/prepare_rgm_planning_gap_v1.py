#!/opt/anaconda3/bin/python3
"""Own public-only RGM planning reviewer preparation; no candidate/gold reads.
Fresh graph/blocks transition and search implementations, hand-derived fixtures.
Actual author /root/expert_reviewer. Small bounded CPU evidence only.
"""
import collections,copy,datetime,hashlib,heapq,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
PUBLIC={'schemas/SEMANTICS.md':'718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd','schemas/RGM_PLANNING_CONTROL_PROPERTIES.md':'f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a','schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md':'02613aa2333f401117def89da224f169d564b527bacc554a0b872877aaac42fe','research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md':'6100d43fe1cf83d798bb34799fac70569f54d33f66692150d188581d9c940761'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def eq(a,b):return canon(a)==canon(b)
def fail(message):raise ValueError(message)
def integer(v):return type(v) is int
class UnknownSearch(RuntimeError):pass
def validate(t):
 if not isinstance(t,dict) or t.get('kind') not in ('graph','blocks'):fail('wrong planning kind')
 if t['kind']=='graph':
  if set(t)!={'kind','nodes','edges','start','goal','fuel','plan'}:fail('graph task keys')
  ns=t['nodes']
  if not isinstance(ns,list) or any(type(n) is not str for n in ns) or len(set(ns))!=len(ns) or not ns:fail('distinct graph nodes')
  if t['start'] not in ns or t['goal'] not in ns or type(t['start']) is not str or type(t['goal']) is not str:fail('unknown graph terminal')
  if not integer(t['fuel']) or t['fuel']<0:fail('nonnegative integer fuel excluding bool')
  if not isinstance(t['edges'],list):fail('edge list')
  pairs=set()
  for e in t['edges']:
   if not isinstance(e,list) or len(e)!=3 or type(e[0]) is not str or type(e[1]) is not str or e[0] not in ns or e[1] not in ns or not integer(e[2]) or e[2]<=0:fail('malformed graph edge')
   if (e[0],e[1]) in pairs:fail('duplicate ordered edge')
   pairs.add((e[0],e[1]))
  if not isinstance(t['plan'],list) or any(type(x) is not str for x in t['plan']):fail('destination string plan')
 else:
  if set(t)!={'kind','on','goal_on','plan'}:fail('block task keys')
  on,go=t['on'],t['goal_on']
  if not isinstance(on,dict) or not on or any(type(k) is not str or k=='TABLE' for k in on):fail('block identities')
  if not isinstance(go,dict) or any(k not in on for k in go):fail('partial goal domain')
  for relation in [on,go]:
   used=set()
   for b,to in relation.items():
    if type(to) is not str or to not in set(on)|{'TABLE'} or to==b:fail('unknown/self support')
    if to!='TABLE' and to in used:fail('multiply occupied support')
    used.add(to)
   for b in relation:
    seen=set();cur=b
    while cur in relation:
     if cur in seen:fail('support cycle')
     seen.add(cur);cur=relation[cur]
  if not isinstance(t['plan'],list):fail('block plan list')
  for a in t['plan']:
   if not isinstance(a,dict) or set(a)!={'block','to'} or type(a['block']) is not str or type(a['to']) is not str:fail('block action shape')
 return t
def graph_step(t,state,dest):
 matching=[c for a,b,c in t['edges'] if a==state['node'] and b==dest]
 if not matching:return False,dict(state),'missing_edge'
 if state['fuel']<matching[0]:return False,dict(state),'insufficient_fuel'
 return True,{'node':dest,'fuel':state['fuel']-matching[0]},None
def block_step(on,action):
 b,to=action['block'],action['to'];occupied=set(on.values())-{'TABLE'}
 if b not in on:return False,dict(on),'unknown_source'
 if b in occupied:return False,dict(on),'source_not_clear'
 if to!='TABLE' and to not in on:return False,dict(on),'unknown_destination'
 if to==b:return False,dict(on),'self_destination'
 if to!='TABLE' and to in occupied:return False,dict(on),'destination_not_clear'
 if on[b]==to:return False,dict(on),'unchanged_support'
 new=dict(on);new[b]=to;return True,new,None
def block_step_relation(on,action):
 """Separately authored edge-set implementation checked on every toy action."""
 edges=set(on.items());b,d=action['block'],action['to'];vertices={a for a,_ in edges};blockers={support for _,support in edges}
 okay=b in vertices and not any(parent==b for child,parent in edges) and d in vertices|{'TABLE'} and b!=d and (d=='TABLE' or not any(parent==d for child,parent in edges)) and (b,d) not in edges
 if not okay:return False,dict(on)
 old=next(e for e in edges if e[0]==b);edges.remove(old);edges.add((b,d));return True,dict(edges)
def block_goal(on,goal):return all(on[b]==v for b,v in goal.items())
def block_actions(on):
 for b in sorted(on):
  for to in sorted(set(on)|{'TABLE'}):
   a={'block':b,'to':to};legal,new,_=block_step(on,a)
   if legal:yield (b,to),new
def block_minimum(t,preserve=(),limit=50000):
 validate(t);initial=t['on'];q=collections.deque([(initial,())]);seen={tuple(sorted(initial.items()))};maximum=1;expanded=0
 while q:
  on,seq=q.popleft();expanded+=1
  if block_goal(on,t['goal_on']):return [dict(block=a,to=b) for a,b in seq],{'complete':True,'expanded':expanded,'visited':len(seen),'maximum_frontier':maximum}
  for action,new in block_actions(on):
   if any(new[b]!=to for b,to in preserve):continue
   key=tuple(sorted(new.items()))
   if key in seen:continue
   seen.add(key);q.append((new,seq+(action,)));maximum=max(maximum,len(q))
   if len(seen)>limit:raise UnknownSearch('block state bound exceeded; never infer infeasible')
 return None,{'complete':True,'expanded':expanded,'visited':len(seen),'maximum_frontier':maximum}
def block_minimum_alternative(t):
 """Heap complete-sequence enumeration with independent edge-set transition."""
 ns=sorted(t['on']);initial=tuple(t['on'][b] for b in ns);q=[(0,(),initial)];seen=set()
 while q:
  n,seq,s=heapq.heappop(q)
  if s in seen:continue
  seen.add(s);on=dict(zip(ns,s))
  if block_goal(on,t['goal_on']):return [{'block':b,'to':to} for b,to in seq]
  for b in ns:
   for to in sorted(ns+['TABLE']):
    legal,new=block_step_relation(on,{'block':b,'to':to})
    if legal:heapq.heappush(q,(n+1,seq+((b,to),),tuple(new[x] for x in ns)))
 return None
def graph_minimum(t,ignore_fuel=False):
 validate(t);adj=collections.defaultdict(list)
 for a,b,c in t['edges']:adj[a].append((b,c))
 q=collections.deque([(t['start'],t['fuel'],())]);seen={(t['start'],None if ignore_fuel else t['fuel'])};maximum=1;expanded=0
 while q:
  node,fuel,seq=q.popleft();expanded+=1
  if node==t['goal']:return list(seq),{'complete':True,'expanded':expanded,'visited':len(seen),'maximum_frontier':maximum}
  for dest,cost in sorted(adj[node]):
   if not ignore_fuel and cost>fuel:continue
   remaining=fuel if ignore_fuel else fuel-cost;key=(dest,None if ignore_fuel else remaining)
   if key in seen:continue
   seen.add(key);q.append((dest,remaining,seq+(dest,)));maximum=max(maximum,len(q))
 return None,{'complete':True,'expanded':expanded,'visited':len(seen),'maximum_frontier':maximum}
def graph_minimum_alternative(t):
 q=[(0,(),t['start'],t['fuel'])];seen=set()
 while q:
  count,seq,node,fuel=heapq.heappop(q)
  if (node,fuel) in seen:continue
  seen.add((node,fuel))
  if node==t['goal']:return list(seq)
  for a,b,c in t['edges']:
   if a==node and c<=fuel:heapq.heappush(q,(count+1,seq+(b,),b,fuel-c))
 return None
def cheapest_fuel(t):
 q=[(0,t['start'])];done={}
 while q:
  used,node=heapq.heappop(q)
  if node in done:continue
  done[node]=used
  if node==t['goal']:return used
  for a,b,c in t['edges']:
   if a==node and b not in done:heapq.heappush(q,(used+c,b))
 return None
def feasible_route_multiplicity(t,minimum):
 if minimum is None:return 0
 layer={(t['start'],t['fuel']):1}
 for _ in range(len(minimum)):
  next_layer=collections.Counter()
  for (node,fuel),n in layer.items():
   for a,b,c in t['edges']:
    if a==node and c<=fuel:next_layer[(b,fuel-c)]+=n
  layer=next_layer
 return sum(n for (node,fuel),n in layer.items() if node==t['goal'])
def replay(t):
 validate(t);state={'node':t['start'],'fuel':t['fuel']} if t['kind']=='graph' else dict(t['on']);steps=[];error=None;valid=True;states=[copy.deepcopy(state)]
 for i,a in enumerate(t['plan']):
  legal,new,reason=graph_step(t,state,a) if t['kind']=='graph' else block_step(state,a)
  steps.append({'legal':legal,'after':copy.deepcopy(new)});states.append(copy.deepcopy(new));state=new
  if not legal:valid=False;error={'index':i,'reason':reason};break
 reaches=valid and (state['node']==t['goal'] if t['kind']=='graph' else block_goal(state,t['goal_on']))
 return {'steps':steps,'after':state,'valid':valid,'reaches_goal':reaches},error,states
def solve(t):
 r,error,states=replay(t);path,search=graph_minimum(t) if t['kind']=='graph' else block_minimum(t);r['shortest_plan']=path
 return r,{'first_error':error,'prefix_states':states,'search':search}
def measurements(t):
 target,extra=solve(t);path=target['shortest_plan'];tags=[];witnesses={}
 if extra['first_error'] and extra['first_error']['index']>=1:tags.append('legal_prefix_then_first_failure');witnesses[tags[-1]]=extra['first_error']
 if target['valid'] and target['reaches_goal'] and path is not None and len(t['plan'])>len(path):tags.append('valid_goal_detour');witnesses[tags[-1]]={'submitted_moves':len(t['plan']),'minimum_moves':len(path)}
 if t['kind']=='graph':
  topo,ts=graph_minimum(t,True);extra.update({'topology_shortest_plan':topo,'topology_search':ts,'cheapest_required_fuel':cheapest_fuel(t),'minimum_feasible_route_count':feasible_route_multiplicity(t,path)})
  if topo is not None and path is None:tags.append('fuel_blocks_topologically_reachable_goal')
  if path is not None and topo is not None and len(path)>len(topo):tags.append('fuel_lengthens_fresh_minimum_moves')
 else:
  if 0<len(t['goal_on'])<len(t['on']):tags.append('partial_block_goal')
  clearance=[]
  for i,a in enumerate(t['plan'][:len(target['steps'])]):
   if i<1 or not target['steps'][i]['legal']:continue
   initial_legal,_,initial_reason=block_step(t['on'],a)
   if not initial_legal and initial_reason in ('source_not_clear','destination_not_clear'):
    relevant=[]
    for j,prior in enumerate(t['plan'][:i]):
     if t['on'].get(prior['block']) in {a['block'],a['to']} and extra['prefix_states'][j].get(prior['block']) in {a['block'],a['to']} and extra['prefix_states'][j+1].get(prior['block']) not in {a['block'],a['to']}:relevant.append(j)
    clearance.append({'index':i,'initial_reason':initial_reason,'preceding_clearance_moves':relevant,'action':a})
  if clearance:tags.append('later_action_requires_changed_block_clearance');witnesses[tags[-1]]=clearance
  fixed=[(b,to) for b,to in t['goal_on'].items() if t['on'][b]==to];restricted,rs=block_minimum(t,preserve=fixed);extra.update({'initially_true_goal_relations':fixed,'preserve_initial_goal_minimum':restricted,'preserve_search':rs})
  if fixed and path is not None and restricted is None:tags.append('required_temporary_goal_displacement')
 return {'target':target,'measurements':extra,'row_properties':sorted(tags),'row_witness_facts':witnesses}
def g(nodes,edges,fuel,plan,start='s',goal='g'):return {'kind':'graph','nodes':nodes,'edges':edges,'start':start,'goal':goal,'fuel':fuel,'plan':plan}
def b(on,goal,plan):return {'kind':'blocks','on':on,'goal_on':goal,'plan':[{'block':x,'to':y} for x,y in plan]}
def fixtures():
 out=[]
 def add(name,t,shortest,valid,reaches,after,error_index=None,reason=None,properties=()):out.append({'name':name,'task':t,'hand_expected':{'shortest_plan':shortest,'valid':valid,'reaches_goal':reaches,'after':after,'first_error_index':error_index,'first_error_reason':reason,'required_properties':list(properties)}})
 edges=[['s','g',9],['s','a',2],['a','z',2],['z','g',2]];ns=['s','a','z','g']
 add('moves_first_more_expensive_than_cheapest',g(ns,edges,9,['g']),['g'],True,True,{'node':'g','fuel':0})
 add('fuel_changes_fresh_horizon',g(ns,edges,8,['g']),['a','z','g'],False,False,{'node':'s','fuel':8},0,'insufficient_fuel',['fuel_lengthens_fresh_minimum_moves'])
 add('exact_required_fuel_boundary',g(ns,edges,6,['a','z','g']),['a','z','g'],True,True,{'node':'g','fuel':0})
 add('one_unit_below_required_fuel',g(ns,edges,5,['a','z','g']),None,False,False,{'node':'z','fuel':1},2,'insufficient_fuel',['legal_prefix_then_first_failure','fuel_blocks_topologically_reachable_goal'])
 add('cheap_long_valid_detour',g(ns,edges,9,['a','z','g']),['g'],True,True,{'node':'g','fuel':3},properties=['valid_goal_detour'])
 te=[['s','a',1],['a','z',1],['z','g',1],['a','m',1],['m','g',1],['s','b',1],['b','g',2]]
 add('full_sequence_lex_and_moves',g(['s','a','z','m','b','g'],te,3,['a','z','g']),['b','g'],True,True,{'node':'g','fuel':0},properties=['valid_goal_detour'])
 tie=[['s','a',1],['a','z',1],['z','g',1],['a','m',1],['m','g',1],['s','b',1],['b','z',1]]
 add('tie_beyond_shared_first_destination',g(['s','a','z','m','b','g'],tie,3,[]),['a','m','g'],True,False,{'node':'s','fuel':3})
 add('lex_tie_not_remaining_fuel',g(['s','a','b','g'],[['s','a',2],['a','g',1],['s','b',1],['b','g',1]],3,['a','g']),['a','g'],True,True,{'node':'g','fuel':0})
 add('directed_reverse_missing_suffix_ignored',g(['s','a','g'],[['s','a',1],['a','g',1]],3,['a','s','g']),['a','g'],False,False,{'node':'a','fuel':2},1,'missing_edge',['legal_prefix_then_first_failure'])
 add('empty_already_goal',g(['g','a'],[['g','a',1]],0,[],start='g'),[],True,True,{'node':'g','fuel':0})
 add('goal_visit_does_not_stop_proposed_plan',g(['s','g','a'],[['s','g',1],['g','a',1]],2,['g','a']),['g'],True,False,{'node':'a','fuel':0})
 on={'a':'TABLE','b':'a','c':'TABLE','d':'c'};goal={'a':'c','b':'a','c':'TABLE','d':'TABLE'};optimal=[('b','TABLE'),('d','TABLE'),('a','c'),('b','a')]
 add('required_temporary_satisfied_goal_displacement',b(on,goal,optimal),[{'block':x,'to':y} for x,y in optimal],True,True,goal,properties=['required_temporary_goal_displacement','later_action_requires_changed_block_clearance'])
 swapped=[('b','TABLE'),('a','c'),('d','TABLE'),('b','a')]
 add('adjacent_order_clear_destination_failure',b(on,goal,swapped),[{'block':x,'to':y} for x,y in optimal],False,False,{'a':'TABLE','b':'TABLE','c':'TABLE','d':'c'},1,'destination_not_clear',['legal_prefix_then_first_failure'])
 partial={'a':'c','b':'a'}
 add('proper_partial_goal_unspecified_supports',b(on,partial,optimal),[{'block':x,'to':y} for x,y in optimal],True,True,goal,properties=['partial_block_goal','required_temporary_goal_displacement'])
 init={'a':'TABLE','b':'a','c':'b'};og={'a':'c','b':'TABLE','c':'TABLE'};op=[('c','TABLE'),('b','TABLE'),('a','c')]
 add('source_clearance_order',b(init,og,op),[{'block':x,'to':y} for x,y in op],True,True,og,properties=['later_action_requires_changed_block_clearance'])
 add('source_blocked_after_prefix_suffix_ignored',b(init,og,[('c','TABLE'),('a','c'),('b','TABLE')]),[{'block':x,'to':y} for x,y in op],False,False,{'a':'TABLE','b':'a','c':'TABLE'},1,'source_not_clear',['legal_prefix_then_first_failure'])
 add('unchanged_support_is_illegal',b({'a':'TABLE','b':'TABLE'},{'a':'b'},[('a','TABLE'),('a','b')]),[{'block':'a','to':'b'}],False,False,{'a':'TABLE','b':'TABLE'},0,'unchanged_support')
 add('block_full_sequence_lex',b({'a':'TABLE','b':'TABLE','c':'TABLE'},{'a':'b','c':'TABLE'},[]),[{'block':'a','to':'b'}],True,False,{'a':'TABLE','b':'TABLE','c':'TABLE'},properties=['partial_block_goal'])
 add('block_valid_goal_detour',b({'a':'TABLE','b':'TABLE','c':'TABLE'},{'a':'b'},[('c','a'),('c','TABLE'),('a','b')]),[{'block':'a','to':'b'}],True,True,{'a':'b','b':'TABLE','c':'TABLE'},properties=['valid_goal_detour','partial_block_goal'])
 add('empty_partial_goal',b({'a':'TABLE','b':'a'},{},[]),[],True,True,{'a':'TABLE','b':'a'})
 return out
def malformed():
 base=g(['s','g'],[['s','g',1]],1,[]);cases=[]
 for key,value in [('fuel',True),('fuel',-1),('edges',[['s','g',True]]),('edges',[['s','g',0]]),('edges',[['s','g',1],['s','g',2]]),('edges',[['s','x',1]]),('plan',[1]),('nodes',['s','s','g'])]:
  t=copy.deepcopy(base);t[key]=value;cases.append(t)
 for on,go in [({'a':'a'},{}),({'a':'b','b':'a'},{}),({'a':'TABLE','b':'a','c':'a'},{}),({'a':'x'},{}),({'TABLE':'TABLE'},{}),({'a':'TABLE','b':'TABLE'},{'a':'b','b':'a'}),({'a':'TABLE','b':'TABLE','c':'TABLE'},{'b':'a','c':'a'}),({'a':'TABLE'},{'x':'TABLE'})]:cases.append(b(on,go,[]))
 return cases
def rename_graph(t,mapping):
 x=copy.deepcopy(t);x['nodes']=[mapping[n] for n in t['nodes']];x['start']=mapping[t['start']];x['goal']=mapping[t['goal']];x['edges']=[[mapping[a],mapping[b],c] for a,b,c in t['edges']];x['plan']=[mapping[n] for n in t['plan']];return x
def prepare():
 docs={}
 for p,h in PUBLIC.items():
  data=(ROOT/p).read_bytes();assert sha(data)==h;docs[p]={'sha256':h,'length':len(data)}
 reports=[];transition_checks=0;probes=[]
 for f in fixtures():
  t=f['task'];result=measurements(t);target=result['target'];hand=f['hand_expected'];er=result['measurements']['first_error']
  for k in ['shortest_plan','valid','reaches_goal','after']:assert eq(target[k],hand[k]),(f['name'],k,target[k],hand[k])
  assert (er['index'] if er else None)==hand['first_error_index'] and (er['reason'] if er else None)==hand['first_error_reason']
  assert set(hand['required_properties'])<=set(result['row_properties'])
  alt=graph_minimum_alternative(t) if t['kind']=='graph' else block_minimum_alternative(t);assert eq(alt,target['shortest_plan'])
  if t['kind']=='blocks':
   for state in result['measurements']['prefix_states']:
    for x in list(state)+['missing']:
     for y in list(state)+['TABLE','missing']:
      a={'block':x,'to':y};l,n,_=block_step(state,a);ll,nn=block_step_relation(state,a);assert l==ll and eq(n,nn);transition_checks+=1
  reverse=copy.deepcopy(t)
  if t['kind']=='graph':reverse['nodes'].reverse();reverse['edges'].reverse()
  else:reverse['on']=dict(reversed(list(reverse['on'].items())));reverse['goal_on']=dict(reversed(list(reverse['goal_on'].items())))
  assert eq(solve(t)[0],solve(reverse)[0]);probes.append({'fixture':f['name'],'probe':'node/edge/key input order invariance','pass':True})
  if er:
   stopped=copy.deepcopy(t);stopped['plan']=stopped['plan'][:er['index']+1]+stopped['plan']*2;assert eq(solve(stopped)[0],target);probes.append({'fixture':f['name'],'probe':'first illegal suffix absorption','pass':True})
  reports.append({**f,'own_execution':result,'alternative_minimum_agrees':True})
 for i,t in enumerate(malformed()):
  try:validate(t)
  except ValueError:continue
  raise AssertionError(('malformed accepted',i,t))
 tie=next(f['task'] for f in fixtures() if f['name']=='tie_beyond_shared_first_destination');mapping={n:n for n in tie['nodes']};mapping['a'],mapping['b']='b','a';renamed=rename_graph(tie,mapping);old=solve(tie)[0]['shortest_plan'];new=solve(renamed)[0]['shortest_plan'];assert old==['a','m','g'] and new==['a','z','g'];probes.append({'probe':'renaming preserves minimum length but changes full-sequence tie representative','old':old,'new':new,'pass':len(old)==len(new)})
 partial=b({'a':'TABLE','b':'TABLE','c':'TABLE'},{'a':'b'},[('a','b')]);extended=copy.deepcopy(partial);extended['goal_on']['c']='a';p=solve(partial)[0];e=solve(extended)[0];assert len(p['shortest_plan'])==1 and len(e['shortest_plan'])==2 and p['reaches_goal'] and e['valid'] and not e['reaches_goal'];probes.append({'probe':'valid goal-only extension strictly increases minimum and same legal plan fails stronger goal','old':p,'new':e,'pass':True})
 report={'report_type':'PUBLIC_ONLY_RGM_PLANNING_PREPARATION_FREEZE','actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guidance':'Public prospective gap/control specifications, same underlying model and architect design/adjudication overlap disclosed. No future candidate/gold/producer/formal/reference implementation or other private verdict read. Own past graph/block semantics knowledge; entirely new authored source/calibration.','public_sources':docs,'source_sha256':sha(pathlib.Path(__file__).read_bytes()),'actual_hand_derived_fixtures':reports,'hand_fixture_count':len(reports),'malformed_fixtures':malformed(),'malformed_rejection_count':len(malformed()),'independent_block_transition_checks':transition_checks,'metamorphic_probes':probes,'toy_probe_count':len(probes),'future_candidate_reads':0,'future_gold_reads':0,'future_expected_predictions_or_dataset_judgments':0,'search_policy':'Finite state BFS with sorted action/full-path traversal; independent heap full-sequence implementation on toys. Complete state counts/frontier retained. Bound exhaustion raises UNKNOWN; not infeasible. Graph route multiplicity uses depth layer counts; cheapest fuel is diagnostic only.','validation_scope_note':'Pilot valid domain uses nonnegative integer initial fuel (bool excluded), positive integer edge costs, exact task/action shapes; unknown string plan names yield illegal actions rather than initial-state malformed. Initial and partial-goal support cycles/multiple occupancy are rejected before search.','planned_later_work':'When root supplies immutable eight-field input-only planning view, derive all full typed targets/prefix states and search facts before canonical gold/control manifest metadata; freeze first predictions/hash then separately compare. No prospective count asserted achieved.'}
 out=OUT/'planning-gap-v1.expert_preparation.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 execution={'report_type':'ACTUAL_EXPERT_PUBLIC_ONLY_PREPARATION_EXECUTION','actual_author':ROLE,'python_executable':sys.executable,'actual_argv':sys.argv,'exit_code':0,'source_sha256':report['source_sha256'],'preparation_sha256':sha(out.read_bytes()),'preparation_length':out.stat().st_size,'hand_fixture_count':len(reports),'malformed_rejections':len(malformed()),'block_transition_checks':transition_checks,'public_source_bindings':docs,'no_candidate_gold_or_private_source_reads':True,'no_dataset_or_release_approval':True,'filesystem_write_freeze_after_execution':True}
 ep=OUT/'planning-gap-v1.expert_preparation.execution.json';assert not ep.exists();ep.write_text(json.dumps(execution,indent=2)+'\n');print(canon(execution))
if __name__=='__main__':prepare()
