#!/usr/bin/env python3
"""Public-only planning-gap reviewer preparation; /root/adversarial_auditor.
No candidate, generator, oracle, checker, private expectation or other verdict read.
"""
import collections,copy,hashlib,heapq,itertools,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXECUTION='/root/adversarial_auditor'
PUBLIC=['schemas/SEMANTICS.md','schemas/RGM_PLANNING_CONTROL_PROPERTIES.md','research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md','schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md']

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,msg):
    if not x:raise ValueError(msg)
def integer(x):return type(x) is int

def graph_validate(t):
    require(type(t) is dict and set(t)=={'kind','nodes','edges','start','goal','fuel','plan'} and t['kind']=='graph','graph exact shape')
    ns=t['nodes'];require(type(ns) is list and all(type(n) is str for n in ns) and len(set(ns))==len(ns),'distinct graph node strings')
    require(type(t['start']) is type(t['goal']) is str and t['start'] in ns and t['goal'] in ns,'existing start and goal')
    require(integer(t['fuel']) and t['fuel']>=0,'nonnegative integer fuel')
    require(type(t['edges']) is list and all(type(e) is list and len(e)==3 and type(e[0]) is type(e[1]) is str and e[0] in ns and e[1] in ns and integer(e[2]) and e[2]>0 for e in t['edges']),'typed positive graph edges')
    require(len({(e[0],e[1]) for e in t['edges']})==len(t['edges']),'unique directed pairs')
    require(type(t['plan']) is list and all(type(v) is str for v in t['plan']),'destination string actions')
    return {(x,y):c for x,y,c in t['edges']}

def graph_minimum(t,ignore_fuel=False):
    edges=graph_validate(t);adj=collections.defaultdict(list)
    for (x,y),c in edges.items():adj[x].append((y,c))
    for xs in adj.values():xs.sort()
    start=(t['start'],None if ignore_fuel else t['fuel']);queue=collections.deque([(start,[])]);seen={start};expanded=0;frontier=1
    while queue:
        (node,fuel),path=queue.popleft();expanded+=1
        if node==t['goal']:return path,{'complete':True,'goal_found':True,'expanded_states':expanded,'discovered_states':len(seen),'max_frontier':frontier,'finite_state_bound':len(t['nodes']) if ignore_fuel else len(t['nodes'])*(t['fuel']+1)}
        for dest,cost in adj[node]:
            if not ignore_fuel and cost>fuel:continue
            state=(dest,None if ignore_fuel else fuel-cost)
            if state in seen:continue
            seen.add(state);queue.append((state,path+[dest]))
        frontier=max(frontier,len(queue))
    return None,{'complete':True,'goal_found':False,'expanded_states':expanded,'discovered_states':len(seen),'max_frontier':frontier,'finite_state_bound':len(t['nodes']) if ignore_fuel else len(t['nodes'])*(t['fuel']+1)}

def graph_minimum_route_count(t):
    # Separate dynamic-programming layers retain multiplicities, not just a BFS representative.
    edges=graph_validate(t);adj=collections.defaultdict(list)
    for (x,y),cost in edges.items():adj[x].append((y,cost))
    layer={(t['start'],t['fuel']):1};earlier=set();depth=0
    while layer:
        goals=sum(ways for (n,f),ways in layer.items() if n==t['goal'])
        if goals:return {'minimum_moves':depth,'route_count':goals,'complete':True}
        earlier|=set(layer);nxt=collections.Counter()
        for (node,fuel),ways in layer.items():
            for dest,c in adj[node]:
                state=(dest,fuel-c)
                if c<=fuel and state not in earlier:nxt[state]+=ways
        layer=dict(nxt);depth+=1
    return {'minimum_moves':None,'route_count':0,'complete':True}

def graph_cheapest_fuel(t):
    edges=graph_validate(t);adj=collections.defaultdict(list)
    for (x,y),c in edges.items():adj[x].append((y,c))
    costs={t['start']:0};heap=[(0,t['start'])]
    while heap:
        cost,node=heapq.heappop(heap)
        if costs[node]!=cost:continue
        if node==t['goal']:return cost
        for dest,c in adj[node]:
            nc=cost+c
            if nc<costs.get(dest,float('inf')):costs[dest]=nc;heapq.heappush(heap,(nc,dest))
    return None

def graph_solve(t):
    edges=graph_validate(t);state={'node':t['start'],'fuel':t['fuel']};steps=[];failure=None
    for i,dest in enumerate(t['plan']):
        cost=edges.get((state['node'],dest));legal=cost is not None and cost<=state['fuel']
        if legal:state={'node':dest,'fuel':state['fuel']-cost}
        steps.append({'legal':legal,'after':copy.deepcopy(state)})
        if not legal:failure={'index':i,'reasons':['missing_directed_edge'] if cost is None else ['insufficient_fuel']};break
    path,search=graph_minimum(t);topo,toposearch=graph_minimum(t,True);valid=failure is None
    answer={'steps':steps,'after':state,'valid':valid,'reaches_goal':valid and state['node']==t['goal'],'shortest_plan':path}
    info={'first_failure':failure,'fresh_minimum_moves':None if path is None else len(path),'topology_minimum_moves':None if topo is None else len(topo),'cheapest_required_fuel':graph_cheapest_fuel(t),'minimum_route_multiplicity':graph_minimum_route_count(t),'search':search,'topology_search':toposearch}
    return answer,info

def support_valid(mapping,blocks):
    require(type(mapping) is dict and all(type(k) is type(v) is str and k in blocks and v in blocks|{'TABLE'} and k!=v for k,v in mapping.items()),'support/goal names')
    targets=[v for v in mapping.values() if v!='TABLE'];require(len(targets)==len(set(targets)),'at most one direct child')
    for start in mapping:
        seen=set();cur=start
        while cur in mapping:
            require(cur not in seen,'support cycle');seen.add(cur);cur=mapping[cur]

def blocks_validate(t):
    require(type(t) is dict and set(t)=={'kind','on','goal_on','plan'} and t['kind']=='blocks','block exact shape')
    require(type(t['on']) is dict and all(type(b) is str and b!='TABLE' for b in t['on']),'reserved block key');blocks=set(t['on']);support_valid(t['on'],blocks);support_valid(t['goal_on'],blocks)
    require(type(t['plan']) is list and all(type(m) is dict and set(m)=={'block','to'} and type(m['block']) is type(m['to']) is str for m in t['plan']),'typed block actions')
    return tuple(sorted(blocks))

def block_move(on,action):
    b,d=action['block'],action['to'];clear=set(on)-set(on.values());reasons=[]
    if b not in on:reasons.append('unknown_source')
    elif b not in clear:reasons.append('source_not_clear')
    if d!='TABLE':
        if d not in on:reasons.append('unknown_destination')
        elif d==b:reasons.append('self_destination')
        elif d not in clear:reasons.append('destination_not_clear')
    if b in on and on[b]==d:reasons.append('unchanged_support')
    if reasons:return False,copy.deepcopy(on),reasons
    after=copy.deepcopy(on);after[b]=d;return True,after,[]

def alternate_relation_move(relations,action):
    # Independently written relation-set transition: no map-clear helper shared.
    b,d=action['block'],action['to'];pairs=set(relations);names={src for src,dst in pairs}
    okay=b in names and not any(dst==b for src,dst in pairs) and (d=='TABLE' or d in names and d!=b and not any(dst==d for src,dst in pairs)) and (b,d) not in pairs
    if not okay:return False,frozenset(pairs)
    return True,frozenset({(src,dst) for src,dst in pairs if src!=b}|{(b,d)})

def blocks_minimum(t,preserve=()):
    blocks=blocks_validate(t);initial=tuple(t['on'][b] for b in blocks);goal=t['goal_on'];queue=collections.deque([(initial,[])]);seen={initial};expanded=0;frontier=1
    while queue:
        state,path=queue.popleft();expanded+=1;on=dict(zip(blocks,state))
        if all(on[b]==d for b,d in goal.items()):return path,{'complete':True,'goal_found':True,'expanded_states':expanded,'discovered_states':len(seen),'max_frontier':frontier,'raw_assignment_bound':len(blocks)**len(blocks)}
        for b,d in sorted(itertools.product(blocks,('TABLE',)+blocks)):
            move={'block':b,'to':d};legal,after,_=block_move(on,move)
            if not legal or any(after[b]!=d for b,d in preserve):continue
            nxt=tuple(after[b] for b in blocks)
            if nxt in seen:continue
            seen.add(nxt);queue.append((nxt,path+[move]))
        frontier=max(frontier,len(queue))
    return None,{'complete':True,'goal_found':False,'expanded_states':expanded,'discovered_states':len(seen),'max_frontier':frontier,'raw_assignment_bound':len(blocks)**len(blocks)}

def alternate_blocks_minimum(t):
    blocks=blocks_validate(t);goal=set(t['goal_on'].items());initial=frozenset(t['on'].items());layer={initial:[]};visited=set()
    while layer:
        paths=[p for state,p in layer.items() if goal<=set(state)]
        if paths:return min(paths,key=lambda p:tuple((m['block'],m['to']) for m in p))
        visited|=set(layer);nxt={}
        for state,path in layer.items():
            for b in sorted(blocks):
                for d in sorted(set(blocks)|{'TABLE'}):
                    move={'block':b,'to':d};okay,state2=alternate_relation_move(state,move)
                    if not okay or state2 in visited:continue
                    candidate=path+[move]
                    key=lambda p:tuple((m['block'],m['to']) for m in p)
                    if state2 not in nxt or key(candidate)<key(nxt[state2]):nxt[state2]=candidate
        layer=nxt
    return None

def blocks_solve(t):
    blocks_validate(t);state=copy.deepcopy(t['on']);steps=[];failure=None;clearance_witnesses=[]
    for i,action in enumerate(t['plan']):
        legal,after,reasons=block_move(state,action)
        if legal and i>=1:
            original_legal,_,oldreasons=block_move(t['on'],action);clearance=[x for x in oldreasons if x in ('source_not_clear','destination_not_clear')]
            if not original_legal and clearance:
                relevant={action['block'] if reason=='source_not_clear' else action['to'] for reason in clearance};changes=[]
                for j,step in enumerate(steps):
                    before=t['on'] if j==0 else steps[j-1]['after'];clear_before=set(before)-set(before.values());clear_after=set(step['after'])-set(step['after'].values())
                    for b in relevant:
                        if b not in clear_before and b in clear_after:changes.append({'action_index':j,'newly_clear_block':b,'action':t['plan'][j]})
                clearance_witnesses.append({'action_index':i,'action':action,'initial_clearance_failures':clearance,'prior_clearance_changes':changes})
        state=after;steps.append({'legal':legal,'after':copy.deepcopy(state)})
        if not legal:failure={'index':i,'reasons':reasons};break
    minimum,search=blocks_minimum(t);initialfacts=sorted((b,d) for b,d in t['goal_on'].items() if t['on'][b]==d)
    if initialfacts and minimum is not None:preserving,preserve_search=blocks_minimum(t,initialfacts)
    else:preserving,preserve_search=minimum,None
    valid=failure is None;answer={'steps':steps,'after':state,'valid':valid,'reaches_goal':valid and all(state[b]==d for b,d in t['goal_on'].items()),'shortest_plan':minimum}
    info={'first_failure':failure,'fresh_minimum_moves':None if minimum is None else len(minimum),'search':search,'changed_clearance_witnesses':clearance_witnesses,'initially_true_explicit_goal_relations':initialfacts,'minimum_preserving_all_initial_true_goal_relations':None if preserving is None else len(preserving),'preservation_search':preserve_search,'required_temporary_goal_displacement':bool(initialfacts) and minimum is not None and preserving is None,'partial_block_goal':0<len(t['goal_on'])<len(t['on'])}
    return answer,info

def solve(t):return graph_solve(t) if t.get('kind')=='graph' else blocks_solve(t) if t.get('kind')=='blocks' else (_ for _ in ()).throw(ValueError('unsupported planning kind'))

def factual_properties(t,answer,info):
    found={};fail=info['first_failure'];minimum=info['fresh_minimum_moves']
    if fail and fail['index']>=1:found['legal_prefix_then_first_failure']=fail
    if answer['valid'] and answer['reaches_goal'] and minimum is not None and len(t['plan'])>minimum:found['valid_goal_detour']={'submitted_length':len(t['plan']),'minimum_moves':minimum}
    if t['kind']=='graph':
        topo=info['topology_minimum_moves']
        if topo is not None and minimum is not None and minimum>topo:found['fuel_lengthens_fresh_minimum_moves']={'topology_moves':topo,'fuel_feasible_moves':minimum}
        if topo is not None and minimum is None:found['fuel_blocks_topologically_reachable_goal']={'topology_moves':topo,'completed_finite_search':info['search']}
    else:
        if info['changed_clearance_witnesses']:found['later_action_requires_changed_block_clearance']=info['changed_clearance_witnesses']
        if info['partial_block_goal']:found['partial_block_goal']={'explicit_goals':len(t['goal_on']),'blocks':len(t['on'])}
        if info['required_temporary_goal_displacement']:found['required_temporary_goal_displacement']={'initial_true_facts':info['initially_true_explicit_goal_relations'],'completed_preserving_search':info['preservation_search']}
    return found

def graph(nodes,edges,start,goal,fuel,plan):return {'kind':'graph','nodes':nodes,'edges':edges,'start':start,'goal':goal,'fuel':fuel,'plan':plan}
def blocks(on,goal,plan):return {'kind':'blocks','on':on,'goal_on':goal,'plan':plan}
def move(b,d):return {'block':b,'to':d}
def state(n,f):return {'node':n,'fuel':f}
def expected_graph(states,legal,after,valid,reaches,shortest):return {'steps':[{'legal':l,'after':s} for l,s in zip(legal,states)],'after':after,'valid':valid,'reaches_goal':reaches,'shortest_plan':shortest}
def expected_blocks(states,legal,after,valid,reaches,shortest):return expected_graph(states,legal,after,valid,reaches,shortest)

def calibration():
    fixtures=[]
    def check(name,t,expected,hand_proof):
        actual,info=solve(t);require(canon(actual)==canon(expected),'hand-derived calibration '+name);fixtures.append({'name':name,'task':t,'hand_derived_exact_target':expected,'actual_exact_target':actual,'hand_proof':hand_proof,'search_and_properties':info,'row_properties':factual_properties(t,actual,info),'passed':True})
    fuel_edges=[['S','A',3],['A','Z',3],['S','B',1],['B','C',1],['C','Z',1]]
    check('graph_min_moves_not_cheapest_fuel',graph(['S','A','Z','B','C'],fuel_edges,'S','Z',6,['A','Z']),expected_graph([state('A',3),state('Z',0)],[True,True],state('Z',0),True,True,['A','Z']),'Two-move A route costs6 and beats three-move B/C route costing3; move count is objective.')
    check('graph_fuel_lengthens_and_stops_after_legal_prefix',graph(['S','A','Z','B','C'],fuel_edges,'S','Z',3,['A','Z','S']),expected_graph([state('A',0),state('A',0)],[True,False],state('A',0),False,False,['B','C','Z']),'A costs3 then Z is unaffordable; suffix S is not executed. Fresh feasible route B/C/Z has3moves.')
    check('graph_exact_fuel_boundary_infeasible',graph(['S','A','Z','B','C'],fuel_edges,'S','Z',2,[]),expected_graph([],[],state('S',2),True,False,None),'Every S→Z route costs at least3; fuel2 makes finite search exhaust, topology remains reachable in2moves.')
    tie_edges=[['S','a',2],['a','G',1],['S','b',1],['b','G',1]]
    check('graph_complete_sequence_lex_tie',graph(['S','a','b','G'],tie_edges,'S','G',3,['b','G']),expected_graph([state('b',2),state('G',1)],[True,True],state('G',1),True,True,['a','G']),'Two feasible2move sequences; [a,G] lex smaller even though it costs more.')
    check('graph_tie_route_removed_by_fuel',graph(['S','a','b','G'],tie_edges,'S','G',2,[]),expected_graph([],[],state('S',2),True,False,['b','G']),'Only b route fits2fuel; lexical preference cannot override resource feasibility.')
    check('graph_empty_goal_and_departing_return',graph(['S','A'],[['S','A',1],['A','S',1]],'S','S',2,['A','S']),expected_graph([state('A',1),state('S',0)],[True,True],state('S',0),True,True,[]),'Initial goal needs0 fresh moves; submitted2move departure/return is legal goal detour.')
    check('graph_direction_is_not_symmetric',graph(['S','a','G'],[['S','a',1],['a','G',1]],'S','G',4,['a','S','G']),expected_graph([state('a',3),state('a',3)],[True,False],state('a',3),False,False,['a','G']),'S→a exists; a→S absent. First illegal reverse stops suffix.')
    check('graph_disconnected_goal',graph(['S','G'],[],'S','G',0,[]),expected_graph([],[],state('S',0),True,False,None),'Disconnected directed topology and finite resource graph both unreachable.')
    check('graph_unknown_action_name_is_illegal',graph(['S'],[],'S','S',0,['X']),expected_graph([state('S',0)],[False],state('S',0),False,False,[]),'Well-typed unknown destination has no edge; illegal plan prevents reaches_goal despite unchanged goal state.')
    on={'A':'B','B':'TABLE','C':'D','D':'TABLE'};goal={'A':'B','B':'C','C':'TABLE','D':'TABLE'};p=[move('A','TABLE'),move('C','TABLE'),move('B','C'),move('A','B')]
    s1={'A':'TABLE','B':'TABLE','C':'D','D':'TABLE'};s2={'A':'TABLE','B':'TABLE','C':'TABLE','D':'TABLE'};s3={'A':'TABLE','B':'C','C':'TABLE','D':'TABLE'};s4={'A':'B','B':'C','C':'TABLE','D':'TABLE'}
    check('blocks_changed_clearance_temporary_goal_displacement',blocks(on,goal,p),expected_blocks([s1,s2,s3,s4],[True]*4,s4,True,True,p),'To move B onto C, A must leave B; A returns to B. A requires2moves, C1 and B1: lower bound4. Lex minimum first A→TABLE, then C→TABLE; preserving A:B prevents all goal plans.')
    swapped=[move('A','TABLE'),move('B','C'),move('C','TABLE'),move('A','B')];s_bad={'A':'TABLE','B':'C','C':'D','D':'TABLE'}
    check('blocks_adjacent_order_fails_at_index2',blocks(on,goal,swapped),expected_blocks([s1,s_bad,s_bad],[True,True,False],s_bad,False,False,p),'B→C before C→TABLE is legal but blocks C, so first failure is index2; fresh minimum remains4.')
    detour=[move('A','TABLE'),move('C','TABLE'),move('A','D'),move('B','C'),move('A','B')];sd3={'A':'D','B':'TABLE','C':'TABLE','D':'TABLE'};sd4={'A':'D','B':'C','C':'TABLE','D':'TABLE'}
    check('blocks_legal_goal_detour_not_optimal',blocks(on,goal,detour),expected_blocks([s1,s2,sd3,sd4,s4],[True]*5,s4,True,True,p),'Extra A→D placement is legal and later A→B restores goal;5submitted moves versus4freshminimum.')
    alltable={b:'TABLE' for b in 'ABCD'};partial={'A':'B','C':'D'};weakp=[move('A','B'),move('C','D')];weak1={'A':'B','B':'TABLE','C':'TABLE','D':'TABLE'};weak2={'A':'B','B':'TABLE','C':'D','D':'TABLE'}
    check('blocks_proper_partial_goal_minimum2',blocks(alltable,partial,weakp),expected_blocks([weak1,weak2],[True,True],weak2,True,True,weakp),'Two explicit new supports each need one move; A-first is lex smaller. Unspecified B,D remainTABLE without being inferred goal obligations.')
    full={'A':'B','B':'C','C':'D','D':'TABLE'};fullp=[move('C','D'),move('B','C'),move('A','B')]
    check('blocks_goal_only_extension_same_plan_fails',blocks(alltable,full,weakp),expected_blocks([weak1,weak2],[True,True],weak2,True,False,fullp),'Goal extension retains A:B,C:D and adds B:C,D:TABLE. Same2move plan stays legal but misses B:C; min3 builds from bottom upward.')
    check('blocks_unspecified_support_not_TABLE',blocks({'A':'B','B':'C','C':'TABLE'},{'A':'B'},[]),expected_blocks([],[],{'A':'B','B':'C','C':'TABLE'},True,True,[]),'Goal constrains only A:B; B:C is allowed and initial goal already true.')
    check('blocks_unchanged_support_is_illegal',blocks({'A':'TABLE','B':'TABLE'},{'A':'B'},[move('A','TABLE'),move('B','A')]),expected_blocks([{'A':'TABLE','B':'TABLE'}],[False],{'A':'TABLE','B':'TABLE'},False,False,[move('A','B')]),'A→TABLE changes no support; illegal first action absorbs suffix.')
    check('blocks_unknown_source_is_illegal',blocks({'A':'TABLE'},{},[move('X','A')]),expected_blocks([{'A':'TABLE'}],[False],{'A':'TABLE'},False,False,[]),'Unknown but well-typed source is a failed action, not a fabricated block; valid flag gates reaches_goal even for empty goal.')
    check('blocks_empty_world_empty_goal',blocks({}, {}, []),expected_blocks([],[],{},True,True,[]),'Empty maps satisfy all finite well-formedness and goal constraints; empty plan is legal and minimal.')
    return fixtures


def malformed_probes():
    goodg=graph(['S','G'],[['S','G',1]],'S','G',1,[]);goodb=blocks({'A':'TABLE','B':'TABLE','C':'TABLE'},{'A':'B'},[]);cases=[]
    def add(name,t):
        try:solve(t)
        except ValueError as e:cases.append({'name':name,'rejected':True,'error':str(e),'task':t})
        else:raise ValueError('malformed accepted '+name)
    for name,field,value in [('duplicate_graph_nodes','nodes',['S','S','G']),('edge_bool_cost','edges',[['S','G',True]]),('edge_zero_cost','edges',[['S','G',0]]),('edge_duplicate_pair','edges',[['S','G',1],['S','G',2]]),('edge_unknown_endpoint','edges',[['S','X',1]]),('graph_fuel_bool','fuel',True),('graph_float_fuel','fuel',1.0),('unknown_initial_start','start','X'),('graph_nonstr_action','plan',[1])]:
        t=copy.deepcopy(goodg);t[field]=value;add(name,t)
    for name,field,value in [('initial_unknown_support','on',{'A':'X','B':'TABLE','C':'TABLE'}),('initial_two_children','on',{'A':'C','B':'C','C':'TABLE'}),('initial_cycle','on',{'A':'B','B':'A','C':'TABLE'}),('reserved_TABLE_block','on',{'TABLE':'A','A':'TABLE'}),('goal_unknown_block','goal_on',{'X':'TABLE'}),('goal_unknown_support','goal_on',{'A':'X'}),('goal_self_support','goal_on',{'A':'A'}),('goal_cycle','goal_on',{'A':'B','B':'A'}),('goal_two_children','goal_on',{'A':'C','B':'C'}),('malformed_block_action','plan',[{'block':'A'}])]:
        t=copy.deepcopy(goodb);t[field]=value;add(name,t)
    return cases


def independent_block_enumeration():
    valid_states={};transition_count=0;search_count=0;query_digest=hashlib.sha256()
    for n in (3,4):
        names=tuple('ABCD'[:n]);choices=[('TABLE',)+tuple(b for b in names if b!=x) for x in names];states=[]
        for assignment in itertools.product(*choices):
            on=dict(zip(names,assignment))
            try:support_valid(on,set(names))
            except ValueError:continue
            states.append(on)
        require(len(states)=={3:13,4:73}[n],'hand-counted labeled stack states')
        for on in states:
            for b,d in itertools.product(names,('TABLE',)+names):
                okay,after,_=block_move(on,move(b,d));altokay,altrel=alternate_relation_move(frozenset(on.items()),move(b,d));require(okay==altokay and set(after.items())==set(altrel),'independent transition disagreement');transition_count+=1
        valid_states[str(n)]=len(states)
        if n==3:
            goals=set()
            for on in states:
                for mask in range(1<<n):goals.add(tuple(sorted((b,on[b]) for i,b in enumerate(names) if mask&(1<<i))))
            require(len(goals)==44,'hand-counted wellformed partial goals n3')
            for on in states:
                for g in sorted(goals):
                    t=blocks(on,dict(g),[]);main,info=blocks_minimum(t);alternate=alternate_blocks_minimum(t);require(canon(main)==canon(alternate) and main is not None and info['complete'],'independent exact block search disagreement');search_count+=1;query_digest.update(canon([on,g,main]).encode())
    return {'valid_states_by_size':valid_states,'independent_map_relation_transition_comparisons':transition_count,'independent_map_relation_partial_goal_search_comparisons':search_count,'three_block_valid_partial_goals':44,'exact_search_comparison_digest':query_digest.hexdigest(),'all_complete':True}


def metamorphic_and_shortcut_probes(fixtures):
    checks=[];shortcuts=[]
    def record(name,condition,evidence):require(condition,name);checks.append({'name':name,'passed':True,'evidence':evidence})
    by={x['name']:x for x in fixtures};tie=by['graph_complete_sequence_lex_tie']['task'];mapping={'S':'start','G':'finish','a':'zeta','b':'beta'};ren=copy.deepcopy(tie);ren['nodes']=[mapping[n] for n in tie['nodes']];ren['edges']=[[mapping[x],mapping[y],c] for x,y,c in tie['edges']];ren['start']=mapping[tie['start']];ren['goal']=mapping[tie['goal']];ren['plan']=[mapping[n] for n in tie['plan']];answer,info=graph_solve(ren)
    record('lex_tie_changes_under_bijective_renaming',answer['shortest_plan']==['beta','finish'] and len(answer['shortest_plan'])==2 and answer['valid'] and answer['reaches_goal'],{'renamed_task':ren,'actual_target':answer,'incorrect_renamed_old_minimum':['zeta','finish']})
    for x in fixtures:
        t=x['task'];expected=x['actual_exact_target'];z=copy.deepcopy(t)
        if t['kind']=='graph':z['nodes'].reverse();z['edges'].reverse()
        else:z['on']=dict(reversed(list(z['on'].items())));z['goal_on']=dict(reversed(list(z['goal_on'].items())))
        actual,_=solve(z);record('input_list_key_order_invariance_'+x['name'],canon(actual)==canon(expected),{'task_hash':hashlib.sha256(canon(t).encode()).hexdigest()})
        if not expected['valid']:
            z=copy.deepcopy(t);z['plan']=t['plan']+[('S' if t['kind']=='graph' else move('A','TABLE'))];actual,_=solve(z);record('first_failure_suffix_absorption_'+x['name'],canon(actual)==canon(expected),{'original_evaluated_steps':len(expected['steps']),'appended_actions_not_evaluated':True})
    fuel=by['graph_min_moves_not_cheapest_fuel']['task'];minimum=graph_cheapest_fuel(fuel);lo=copy.deepcopy(fuel);hi=copy.deepcopy(fuel);lo['fuel']=minimum-1;hi['fuel']=minimum
    record('one_unit_minimum_required_fuel_boundary',graph_minimum(lo)[0] is None and graph_minimum(hi)[0] is not None,{'minimum_fuel':minimum,'below':lo['fuel'],'boundary':hi['fuel']})
    weak=by['blocks_proper_partial_goal_minimum2']['task'];strong=by['blocks_goal_only_extension_same_plan_fails']['task'];changed=[k for k in weak if canon(weak[k])!=canon(strong[k])]
    record('partial_goal_extension_only_monotonicity',changed==['goal_on'] and len(blocks_minimum(weak)[0])==2 and len(blocks_minimum(strong)[0])==3 and blocks_solve(weak)[0]['reaches_goal'] and not blocks_solve(strong)[0]['reaches_goal'],{'changed_task_fields':changed,'weak_minimum':2,'strong_minimum':3})
    before=by['blocks_changed_clearance_temporary_goal_displacement']['task'];swapped=by['blocks_adjacent_order_fails_at_index2']['task'];changed=[k for k in before if canon(before[k])!=canon(swapped[k])]
    record('adjacent_action_swap_only',changed==['plan'] and before['plan'][1:3]==swapped['plan'][1:3][::-1] and len(before['plan'])==len(swapped['plan']) and not blocks_solve(swapped)[0]['valid'],{'changed_task_fields':changed,'first_error_index':2})
    for name in ('graph_fuel_lengthens_and_stops_after_legal_prefix','graph_min_moves_not_cheapest_fuel','graph_complete_sequence_lex_tie','blocks_legal_goal_detour_not_optimal','blocks_unspecified_support_not_TABLE','graph_unknown_action_name_is_illegal'):
        x=by[name];shortcuts.append({'name':name,'task':x['task'],'exact_target':x['actual_exact_target'],'hand_counterexample_to':['minimize_fuel_instead_of_moves','submitted_length_equals_fresh_minimum','goal_fact_initially_true_implies_valid','all_unspecified_blocks_on_TABLE','no_op_or_copy_plan_bias'],'scope':'Hand-derived calibration/attack fixture only, not a future production label or source family.'})
    return {'executed_metamorphic_checks':checks,'actual_metamorphic_count':len(checks),'shortcut_attack_fixtures':shortcuts,'attack_plan_before_candidates':['CPU train→exposed-dev baselines for plan length,node/block count,fuel,edge count,token/name/error-index and joint features; independently derive labels first.','Measure success/validity/fresh-feasibility separately; required target-length differences do not prove input outcome leakage.','Conditional all four contexts per source/stratum/split/renderer; no family-count inflation from plans,budgets,edge deletion,goal extension or order views.','Canonical graph weighted start/goal topology and unweighted topology signatures; canonical joint block initial/goal relation signatures; prior public-input custody only after explicit later scope.','Verify actual row property witness versus family-control distinction, completed search bounds and exact supplied field differences.']}


def immutable(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_text()==text,'refuse frozen preparation overwrite '+str(path))
    else:path.write_text(text)

def prepare():
    public_hashes={rel:sha(ROOT/rel) for rel in PUBLIC};require(public_hashes['schemas/SEMANTICS.md']=='718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd','public unchanged semantics hash')
    fixtures=calibration();malformed=malformed_probes();cross=independent_block_enumeration();meta=metamorphic_and_shortcut_probes(fixtures)
    tests=ROOT/'reviews/adversarial/planning-gap-v1.preparation_calibration.jsonl';objects=[{'actual_execution_id':EXECUTION,'artifact_type':'hand_derived_calibration','fixtures':fixtures},{'actual_execution_id':EXECUTION,'artifact_type':'malformed_inputs','cases':malformed},{'actual_execution_id':EXECUTION,'artifact_type':'independent_enumeration','measurements':cross},{'actual_execution_id':EXECUTION,'artifact_type':'metamorphic_shortcut_preparation','evidence':meta}];immutable(tests,''.join(canon(o)+'\n' for o in objects))
    source=pathlib.Path(__file__);snapshot=ROOT/'audits/history'/('rgm_planning_gap_adversarial_preparation-'+sha(source)+'.py');immutable(snapshot,source.read_text())
    report={'actual_execution_id':EXECUTION,'reviewer_role':'Adversarial Red-Team and diversity/leakage auditor preparation','batch':'planning-gap-v1','stage':'PUBLIC_ONLY_SOLVER_AND_CALIBRATION_FROZEN_BEFORE_ANY_CANDIDATE_ACCESS','status':'completed','public_sources_sha256':public_hashes,'source_sha256':sha(source),'source_snapshot_path':str(snapshot.relative_to(ROOT)),'source_snapshot_sha256':sha(snapshot),'calibration_path':str(tests.relative_to(ROOT)),'calibration_sha256':sha(tests),'calibration_exact_target_cases':len(fixtures),'graph_calibration_cases':sum(x['task']['kind']=='graph' for x in fixtures),'block_calibration_cases':sum(x['task']['kind']=='blocks' for x in fixtures),'malformed_rejections':len(malformed),'independent_enumeration':cross,'metamorphic_checks':meta['actual_metamorphic_count'],'shortcut_attack_fixture_count':len(meta['shortcut_attack_fixtures']),'search_completion':all(x['search_and_properties']['search']['complete'] for x in fixtures),'candidate_or_gold_or_private_implementation_or_other_verdict_read':False,'dataset_approval_issued':False,'input_domain_interpretations':{'graph_budget':'nonnegative integer excluding bool; public pilot at most30, positive edge costs. No new action dynamics or resource replenishment.','well_typed_unknown_actions':'Unknown graph destination/block source strings produce illegal actions; nonsyntactic shapes/types are rejected input.','failure_reasons':'Own witness lists all violated block guards; canonical target has no reason-precedence field.','scope':'Only two unchanged action grammars; all finite searches complete, no timeout equated with infeasible.'},'review_scope':'Fresh public-only standalone CPU solver/calibration authored in actual /root/adversarial_auditor execution. Prior task participation and same underlying model/shared public design remain disclosed. No new planning candidate,gold,producer/formal/expert code,prior verdicts,private expectations or release contents read. No efficacy,sealed evaluation or independent-model claim.'}
    output=ROOT/'audits/execution/planning-gap-v1-adversarial-preparation.json';immutable(output,json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+'\n');print(canon({'execution_sha256':sha(output),'source_sha256':sha(source),'calibration_sha256':sha(tests),'exact_cases':len(fixtures),'malformed_cases':len(malformed),'metamorphics':meta['actual_metamorphic_count'],'enumeration':cross,'public_sources_sha256':public_hashes}),flush=True)

if __name__=='__main__':prepare()
