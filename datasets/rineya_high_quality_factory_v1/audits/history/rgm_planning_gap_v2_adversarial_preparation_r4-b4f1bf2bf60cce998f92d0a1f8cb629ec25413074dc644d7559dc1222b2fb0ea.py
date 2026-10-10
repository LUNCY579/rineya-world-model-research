#!/usr/bin/env python3
"""Public-only planning-gap reviewer preparation; /root/adversarial_auditor.
New public-guided V2 preparation; reuses this execution's own V1 solver and toys.
No V2 candidate, generator, oracle, checker, private expectation or other verdict read.
"""
import collections,copy,hashlib,heapq,itertools,json,pathlib,time
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


ROOT_FREEZE='research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json'
ROOT_FREEZE_SHA='7759940fcb0412f67f02c620cc40f52216ca7cbfb695e4f2f5690e99b12f2dbd'
OLD_SOURCE_SHA='2330e383f2a8e1ec608435ee7eacfd647e072e8715bb35d583c12798eaaa1ad3'
GUARDS=('source_exists','source_clear','destination_exists_or_table','source_destination_distinct','destination_clear_or_table','changed_support')
MUTANTS={'source_clear_ignored':'source_clear','destination_clear_ignored':'destination_clear_or_table','no_op_allowed':'changed_support'}

def guard_truth(on,a):
    b,d=a['block'],a['to']
    # Independent predicates, including cofailing checks; no short-circuit reason hides them.
    return {'source_exists':b in on,'source_clear':not any(v==b for v in on.values()),
            'destination_exists_or_table':d=='TABLE' or d in on,
            'source_destination_distinct':b!=d,
            'destination_clear_or_table':d=='TABLE' or not any(v==d for v in on.values()),
            'changed_support':b in on and on[b]!=d}

def diagnostic_replay(t,mutant=None):
    blocks_validate(t);require(mutant is None or mutant in MUTANTS,'named diagnostic mutant')
    on=copy.deepcopy(t['on']);steps=[];guards=[];failure=None
    for i,a in enumerate(t['plan']):
        truth=guard_truth(on,a);guards.append(truth)
        okay=all(v for k,v in truth.items() if mutant is None or k!=MUTANTS[mutant])
        if okay:on[a['block']]=a['to']
        steps.append({'legal':okay,'after':copy.deepcopy(on)})
        if not okay:failure=i;break
    # Deliberately no post-transition support validity repair: that would undo the mutant.
    return {'steps':steps,'after':on,'valid':failure is None,'reaches_goal':failure is None and all(on[b]==d for b,d in t['goal_on'].items())},guards,failure

def validate_clearance_v2(t,e):
    blocks_validate(t)
    require(type(e) is dict and {'action_index','precondition','enabling_action_index','obstructing_block'}<=set(e),'four required clearance fields')
    i,j=e['action_index'],e['enabling_action_index'];reason=e['precondition'];ob=e['obstructing_block']
    require(integer(i) and 1<=i<len(t['plan']) and integer(j) and 0<=j<i,'typed bounded clearance indices')
    require(type(reason) is str and reason in ('source_not_clear','destination_not_clear'),'canonical V2 clearance value')
    require(type(ob) is str and ob in t['on'],'existing obstructing block')
    a=t['plan'][i];initial=t['on'];b,d=a['block'],a['to'];truth=guard_truth(initial,a)
    canonical='source_not_clear' if not truth['source_clear'] else 'destination_not_clear' if d!='TABLE' and not truth['destination_clear_or_table'] else None
    require(reason==canonical and canonical is not None,'initial source-priority clearance selection')
    selected=b if reason=='source_not_clear' else d
    require(initial.get(ob)==selected,'initial direct obstruction')
    trace,_,failure=diagnostic_replay(t)
    require(failure is None or failure>i,'actually legal submitted prefix and selected action')
    before=initial if j==0 else trace['steps'][j-1]['after'];after=trace['steps'][j]['after']
    require(t['plan'][j]['block']==ob and any(v==selected for v in before.values()) and not any(v==selected for v in after.values()),'actual obstruction move clears selected object')
    require(all(not any(v==selected for v in trace['steps'][k]['after'].values()) for k in range(j,i)),'continuous clearance until selected action')
    return True

def validate_selective_pair(left,right,e):
    blocks_validate(left);blocks_validate(right)
    require(type(e) is dict and {'swap_index','first_failure_index','first_failure_reason','guard_truth'}<=set(e),'C1 evidence fields')
    s,i=e['swap_index'],e['first_failure_index'];reason=e['first_failure_reason'];truth=e['guard_truth']
    require(integer(s) and 0<=s<len(left['plan'])-1 and integer(i) and 0<=i<len(right['plan']),'typed C1 indices')
    require(type(reason) is str and reason in ('source_not_clear','destination_not_clear'),'C1 reason enum')
    require(type(truth) is dict and set(truth)==set(GUARDS) and all(type(v) is bool for v in truth.values()),'exact six boolean guards')
    require([k for k in left if canon(left[k])!=canon(right[k])]==['plan'],'plan-only C1 difference')
    swapped=copy.deepcopy(left['plan']);swapped[s],swapped[s+1]=swapped[s+1],swapped[s]
    require(canon(swapped)==canon(right['plan']),'actual adjacent swap')
    strict,actual,fail=diagnostic_replay(right)
    key='source_clear' if reason=='source_not_clear' else 'destination_clear_or_table'
    require(fail==i and actual[i]==truth and [k for k,v in truth.items() if not v]==[key],'selective actual first failure')
    mutant='source_clear_ignored' if key=='source_clear' else 'destination_clear_ignored'
    changed,_,_=diagnostic_replay(right,mutant)
    require(changed['steps'][i]['legal'] and canon(changed)!=canon(strict),'actual single-guard mutation changes complete output')
    return {'strict':strict,'mutant':changed,'first_failure_index':i,'guard_truth':truth,'mutant_name':mutant}

def validate_noop_pair(base,prefix,appended,e):
    for t in (base,prefix,appended):blocks_validate(t)
    require(type(e) is dict and {'prefix_length','block','current_support'}<=set(e),'noop evidence fields')
    k,b,d=e['prefix_length'],e['block'],e['current_support']
    require(integer(k) and k>=1 and type(b) is str and b in base['on'] and type(d) is str and d=='TABLE','typed noop evidence')
    require(canon(base['on'])==canon(prefix['on'])==canon(appended['on']) and canon(base['goal_on'])==canon(prefix['goal_on'])==canon(appended['goal_on']),'same noop world and goals')
    require(canon(prefix['plan'])==canon(base['plan'][:k]) and len(prefix['plan'])==k,'exact canonical C0 prefix')
    require(canon(appended['plan'])==canon(prefix['plan']+[move(b,'TABLE')]),'exact supplied no-op append')
    plain,_,fail=diagnostic_replay(prefix);strict,guards,idx=diagnostic_replay(appended)
    require(fail is None and not plain['reaches_goal'] and idx==k,'non-goal legal prefix and exact first noop index')
    require(plain['after'][b]=='TABLE' and not any(v==b for v in plain['after'].values()),'clear TABLE block')
    require([key for key,val in guards[k].items() if not val]==['changed_support'],'isolated unchanged support')
    require(canon(strict['after'])==canon(plain['after']),'illegal noop retains previous state')
    mutated,_,_=diagnostic_replay(appended,'no_op_allowed')
    require(mutated['valid'] and mutated['steps'][k]['legal'] and canon(mutated)!=canon(strict),'no-op mutant killed')
    for name in ('source_clear_ignored','destination_clear_ignored'):
        require(canon(diagnostic_replay(appended,name)[0])==canon(strict),'noop agrees under either clearance omission')
    return {'strict':strict,'mutant':mutated,'guard_truth':guards[k],'first_failure_index':k}

def manual_v2_fixtures():
    out=[]
    def add(name,t,states,legality,after,valid,reaches,shortest,proof):
        out.append({'name':name,'task':t,'hand_derived_exact_target':expected_blocks(states,legality,after,valid,reaches,shortest),'hand_proof':proof})
    source_on={'A':'B','B':'TABLE','C':'TABLE'};source_goal={'A':'B','B':'C'}
    source_plan=[move('A','TABLE'),move('B','C'),move('A','B')]
    s1={'A':'TABLE','B':'TABLE','C':'TABLE'};s2={'A':'TABLE','B':'C','C':'TABLE'};s3={'A':'B','B':'C','C':'TABLE'}
    add('sole_source_base',blocks(source_on,source_goal,source_plan),[s1,s2,s3],[True]*3,s3,True,True,source_plan,'A must leave B and return, while B moves once:3 moves. A first is forced; B then A meets goals.')
    add('sole_source_failure',blocks(source_on,source_goal,[source_plan[1],source_plan[0],source_plan[2]]),[source_on],[False],source_on,False,False,source_plan,'B is blocked only by A; C is clear, different and changed support.')
    dest_on={'A':'TABLE','B':'C','C':'TABLE'};dest_goal={'A':'C','B':'TABLE'};dest_plan=[move('B','TABLE'),move('A','C')]
    d1={'A':'TABLE','B':'TABLE','C':'TABLE'};d2={'A':'C','B':'TABLE','C':'TABLE'}
    add('sole_destination_base',blocks(dest_on,dest_goal,dest_plan),[d1,d2],[True]*2,d2,True,True,dest_plan,'B must move before A can use C; both new relations each need1 move.')
    add('sole_destination_failure',blocks(dest_on,dest_goal,dest_plan[::-1]),[dest_on],[False],dest_on,False,False,dest_plan,'A is clear; C alone is occupied by B. Destination omission executes a temporarily multiply occupied C.')
    on={'A':'B','B':'TABLE','C':'TABLE','Z':'TABLE'};goal={'A':'B','B':'C','Z':'A'};p=source_plan+[move('Z','A')]
    a1={**on,'A':'TABLE'};a2={**a1,'B':'C'};a3={**a2,'A':'B'};a4={**a3,'Z':'A'}
    add('four_move_base',blocks(on,goal,p),[a1,a2,a3,a4],[True]*4,a4,True,True,p,'A needs2 moves; B and Z need1 each. Canonical legal dependency order A,B,A,Z yields4; TABLE precedes spare Z lexically, while parking on C blocks the required B→C goal move.')
    add('source_failure_index_differs_from_swap',blocks(on,goal,[p[0],p[2],p[1],p[3]]),[a1,on,on],[True,True,False],on,False,False,p,'Swap index1, failure index2: restoring A onto B is legal but blocks B.')
    for k,b,st in [(1,'A',a1),(2,'A',a2),(3,'Z',a3)]:
        ts=blocks(on,goal,p[:k]+[move(b,'TABLE')]);states=[a1,a2,a3][:k]+[st]
        add('isolated_noop_prefix_'+str(k),ts,states,[True]*k+[False],st,False,False,p,'Legal non-goal prefix length'+str(k)+' then clear TABLE block '+b+' stays TABLE; only changed_support is false.')
    both={'A':'X','X':'TABLE','B':'Y','Y':'TABLE'};both_goal={'X':'Y','A':'TABLE','B':'TABLE'};both_p=[move('A','TABLE'),move('B','TABLE'),move('X','Y')]
    x1={**both,'A':'TABLE'};x2={**x1,'B':'TABLE'};x3={**x2,'X':'Y'}
    add('both_initial_clearance_false_source_priority',blocks(both,both_goal,both_p),[x1,x2,x3],[True]*3,x3,True,True,both_p,'Explicit A:TABLE and B:TABLE goals force those two obstruction removals. Initial both-blocked X→Y canonically selects source; exactly3 moves, A then B lexical.')
    reblock=both_p[:2]+[move('A','X'),move('A','TABLE'),move('X','Y')];r3={**x2,'A':'X'}
    add('continuous_enabling_requires_latest_clearance',blocks(both,both_goal,reblock),[x1,x2,r3,x2,x3],[True]*5,x3,True,True,both_p,'Initial obstruction A is cleared at0, restored at2, and cleared at3. Witness enabling0 is false; enabling3 is true. Goal detour remains legal.')
    return out

def v2_clearance_cases(fixtures):
    by={x['name']:x['task'] for x in fixtures};cases=[]
    def add(name,t,e,expect):cases.append({'name':name,'task':t,'evidence':e,'expected_valid':expect})
    source={'action_index':1,'precondition':'source_not_clear','enabling_action_index':0,'obstructing_block':'A'}
    destination={'action_index':1,'precondition':'destination_not_clear','enabling_action_index':0,'obstructing_block':'B'}
    priority={'action_index':2,'precondition':'source_not_clear','enabling_action_index':0,'obstructing_block':'A'}
    add('canonical_source',by['sole_source_base'],source,True);add('canonical_destination',by['sole_destination_base'],destination,True)
    add('both_blocked_source_priority',by['both_initial_clearance_false_source_priority'],priority,True)
    add('continuous_latest_enabling',by['continuous_enabling_requires_latest_clearance'],{**priority,'action_index':4,'enabling_action_index':3},True)
    add('extra_diagnostic_allowed',by['sole_source_base'],{**source,'diagnostic_note':'public-extra'},True)
    for n,val in enumerate(['source','destination','blocked_source','blocked_destination',0,1,True,False,None,[],{},'SOURCE_NOT_CLEAR']):
        add('wrong_precondition_type_or_alias_'+str(n),by['sole_source_base'],{**source,'precondition':val},False)
    for field,vals in [('action_index',[True,False,0,-1,3,1.0,'1',None]),('enabling_action_index',[True,False,-1,1,2,0.0,'0',None]),('obstructing_block',['X','B',True,None])]:
        for n,val in enumerate(vals):add('invalid_'+field+'_'+str(n),by['sole_source_base'],{**source,field:val},False)
    add('canonical_destination_cannot_override_blocked_source',by['both_initial_clearance_false_source_priority'],{**priority,'precondition':'destination_not_clear','enabling_action_index':1,'obstructing_block':'B'},False)
    add('earlier_clearance_undone_before_selected_action',by['continuous_enabling_requires_latest_clearance'],{**priority,'action_index':4},False)
    for key in source:add('missing_'+key,by['sole_source_base'],{k:v for k,v in source.items() if k!=key},False)
    add('TABLE_cannot_supply_destination_obstruction',by['both_initial_clearance_false_source_priority'],{**source,'action_index':1,'precondition':'destination_not_clear','enabling_action_index':0},False)
    return cases

def control_fixture_rows(fixtures):
    by={x['name']:x['task'] for x in fixtures};base=by['four_move_base'];detour=copy.deepcopy(base);detour['plan']=[move('A','TABLE'),move('B','Z'),move('B','C'),move('A','B'),move('Z','A')]
    contexts={'C0':base,'C1':by['source_failure_index_differs_from_swap'],'C2':detour,'C3':blocks(base['on'],base['goal_on'],base['plan'][:1]),'C4':by['isolated_noop_prefix_1']}
    rows={}
    for context,t in contexts.items():
        for style in ('key_order_forward','key_order_reverse'):
            id='RED-V2-TOY-'+context+'-'+style;task=copy.deepcopy(t)
            if style=='key_order_reverse':task=dict(reversed(list(task.items())));task['on']=dict(reversed(list(task['on'].items())));task['goal_on']=dict(reversed(list(task['goal_on'].items())))
            rows[id]={'sample_id':id,'family':'RED-TOY-FAMILY','split':'train','context':context,'renderer':style,'task':task}
    first=diagnostic_replay(contexts['C1']);e={'swap_index':1,'first_failure_index':2,'first_failure_reason':'source_not_clear','guard_truth':first[1][2]}
    controls=[]
    for style in ('key_order_forward','key_order_reverse'):
        for name,left,right,ev in [('adjacent_action_order_counterfactual','C0','C1',e),('valid_goal_detour_counterfactual','C0','C2',None),('isolated_noop_append_counterfactual','C3','C4',{'prefix_length':1,'block':'A','current_support':'TABLE'})]:
            c={'control_id':'RED-'+name+'-'+style,'name':name,'scope':'corpus_variant','provenance':'supplied_tasks','sample_ids':['RED-V2-TOY-'+left+'-'+style,'RED-V2-TOY-'+right+'-'+style],'changed_task_fields':['plan']}
            if ev is not None:c['evidence']=copy.deepcopy(ev)
            controls.append(c)
    return rows,controls

def validate_renderer_declarations(rows,controls):
    require(type(controls) is list,'control list');seen=set();covered=set()
    for c in controls:
        cid=c.get('control_id');ids=c.get('sample_ids')
        require(type(cid) is str and bool(cid) and cid not in seen,'globally distinct nonempty control_id');seen.add(cid)
        require(c.get('scope')=='corpus_variant' and c.get('provenance')=='supplied_tasks','supplied corpus control scope')
        require(type(ids) is list and len(ids)==2 and all(type(x) is str and x for x in ids) and len(set(ids))==2 and all(x in rows for x in ids),'exact two actual IDs')
        left,right=[rows[x] for x in ids]
        require(left['family']==right['family'] and left['split']==right['split'] and left['renderer']==right['renderer'],'same family/split/style')
        expected={'adjacent_action_order_counterfactual':('C0','C1'),'valid_goal_detour_counterfactual':('C0','C2'),'isolated_noop_append_counterfactual':('C3','C4')}
        require(c['name'] in expected and (left['context'],right['context'])==expected[c['name']],'ordered logical contexts')
        require(c.get('changed_task_fields')==['plan'] and [k for k in left['task'] if canon(left['task'][k])!=canon(right['task'][k])]==['plan'],'actual declared field differences')
        if c['name']=='adjacent_action_order_counterfactual':validate_selective_pair(left['task'],right['task'],c['evidence'])
        elif c['name']=='isolated_noop_append_counterfactual':
            base=next(r['task'] for r in rows.values() if r['context']=='C0' and r['renderer']==left['renderer']);validate_noop_pair(base,left['task'],right['task'],c['evidence'])
        else:
            a,_=solve(left['task']);b,_=solve(right['task']);require(a['valid'] and a['reaches_goal'] and b['valid'] and b['reaches_goal'] and len(right['task']['plan'])>len(b['shortest_plan']),'actual valid goal detour')
        covered.add((c['name'],left['renderer']))
    require(covered==set(itertools.product(['adjacent_action_order_counterfactual','valid_goal_detour_counterfactual','isolated_noop_append_counterfactual'],['key_order_forward','key_order_reverse'])),'each required semantic control under both styles')
    return True

def renderer_cases(rows,controls):
    out=[{'name':'complete_six_declarations','rows':rows,'controls':controls,'expected_valid':True}]
    def add(name,cs,rs=None):out.append({'name':name,'rows':rows if rs is None else rs,'controls':cs,'expected_valid':False})
    c=copy.deepcopy(controls);c.pop();add('missing_one_renderer_declaration',c)
    c=copy.deepcopy(controls);c[1]['control_id']=c[0]['control_id'];add('duplicate_control_ID_across_different_names',c)
    c=copy.deepcopy(controls);c[0]['control_id']='';add('empty_control_ID',c)
    c=copy.deepcopy(controls);c[0]['sample_ids']+=c[3]['sample_ids'];add('combined_four_IDs_is_not_two_per_style',c)
    c=copy.deepcopy(controls);c[0]['sample_ids'][1]=c[3]['sample_ids'][1];add('mixed_renderer_pair',c)
    c=copy.deepcopy(controls);c[0]['sample_ids'][1]='NONEXISTENT';add('nonexistent_actual_ID',c)
    c=copy.deepcopy(controls);c[0]['sample_ids'][1]=c[0]['sample_ids'][0];add('duplicate_pair_ID',c)
    c=copy.deepcopy(controls);c[0]['sample_ids'].reverse();add('reversed_logical_pair',c)
    c=copy.deepcopy(controls);c[0]['changed_task_fields']=['on'];add('false_changed_fields',c)
    c=copy.deepcopy(controls);c[0]['scope']='row';add('row_scope_cannot_replace_supplied_control',c)
    c=copy.deepcopy(controls);c[0]['evidence']['first_failure_index']=True;add('bool_is_not_integer_failure_index',c)
    c=copy.deepcopy(controls);c[0]['evidence']['guard_truth']['source_clear']=0;add('zero_is_not_boolean_guard',c)
    c=copy.deepcopy(controls);c[0]['evidence']['first_failure_reason']='destination_not_clear';add('false_clearance_reason',c)
    c=copy.deepcopy(controls);c[2]['evidence']['block']='B';add('wrong_noop_block',c)
    for field,val in [('family','OTHER'),('split','dev_challenge')]:
        rs=copy.deepcopy(rows);rs[controls[0]['sample_ids'][1]][field]=val;add('wrong_'+field,copy.deepcopy(controls),rs)
    # Same name is intentionally not globally unique, and truthful controls can share pairs.
    c=copy.deepcopy(controls);extra=copy.deepcopy(c[0]);extra['control_id']='RED-OPTIONAL-ADDITIONAL-TRUE';c.append(extra)
    out.append({'name':'truthful_additional_pair_not_collapsed_by_name','rows':rows,'controls':c,'expected_valid':True})
    return out

def input_features(t,prompt=''):
    numeric=t['fuel'] if t['kind']=='graph' else None
    objects=t['nodes'] if t['kind']=='graph' else list(t['on'])
    words=canon(t)
    return {'kind':t['kind'],'objects':len(objects),'edges':len(t['edges']) if t['kind']=='graph' else None,'fuel':numeric,'submitted_plan_length':len(t['plan']),'goal_facts':len(t['goal_on']) if t['kind']=='blocks' else None,'input_character_length':len(words),'prompt_character_length':len(prompt),'entity_characters':sum(map(len,objects)),'first_action':canon(t['plan'][:1]),'last_action':canon(t['plan'][-1:])}

def canonical_structure(t,weighted=True):
    # Permutation invariant; signatures are representation classes, not independent mechanisms.
    if t['kind']=='graph':
        names=sorted(t['nodes']);best=None
        for order in itertools.permutations(names):
            mp={n:i for i,n in enumerate(order)}
            data=(mp[t['start']],mp[t['goal']],tuple(sorted((mp[a],mp[b],c if weighted else 1) for a,b,c in t['edges'])))
            if best is None or data<best:best=data
        return canon(best)
    names=sorted(t['on']);best=None
    for order in itertools.permutations(names):
        mp={n:i for i,n in enumerate(order)}
        key=lambda x:-1 if x=='TABLE' else mp[x]
        data=(tuple(key(t['on'][n]) for n in order),tuple(sorted((mp[b],key(d)) for b,d in t['goal_on'].items())))
        if best is None or data<best:best=data
    return canon(best)

def public_allocation():
    types=['I','L','I','L','I','L','I','L','I','L','I','L'];sizes=[5,5,5,5,6,6,7,6,6,7,7,7]
    a=[{'source_index':i,'size':sizes[i-180],'subtype':types[i-180],'split':'dev_challenge' if i%10 in (8,9) else 'train'} for i in range(180,192)]
    c=[{'source_index':i,'size':5 if i%2==0 else 6,'reason':'destination_not_clear' if i in (207,209,210,212,213,214) else 'source_not_clear','k':1+(i-204)%3,'split':'dev_challenge' if i%10 in (8,9) else 'train'} for i in range(204,216)]
    require(collections.Counter(x['subtype'] for x in a)=={'I':6,'L':6},'A exact6/6 public allocation')
    require(collections.Counter((x['subtype'],x['size']) for x in a)=={(t,n):2 for t in ('I','L') for n in (5,6,7)},'A size-by-type')
    require(collections.Counter((x['subtype'],x['split']) for x in a)=={(t,s):n for t in ('I','L') for s,n in [('train',5),('dev_challenge',1)]},'A type split')
    require(collections.Counter((x['reason'],x['size']) for x in c)=={(r,n):3 for r in ('source_not_clear','destination_not_clear') for n in (5,6)},'C clearance size allocation')
    require(collections.Counter((x['reason'],x['split']) for x in c)=={(r,s):n for r in ('source_not_clear','destination_not_clear') for s,n in [('train',5),('dev_challenge',1)]},'C reason split allocation')
    require(collections.Counter(x['k'] for x in c)=={1:4,2:4,3:4},'C fixed k allocation')
    contexts=sum(5 if 204<=i<=215 else 4 for i in range(180,228));train=sum(2*(5 if 204<=i<=215 else 4) for i in range(180,228) if i%10 not in (8,9))
    require(contexts==204 and train==340 and contexts*2-train==68,'prospective count arithmetic')
    return {'A':a,'C':c,'semantic_inputs':contexts,'rendered_rows':contexts*2,'train_rows':train,'dev_rows':contexts*2-train,'semantic_required_pairs':12+12+12+12+36+12,'per_renderer_required_declarations':2*(12+12+12+12+36+12),'revised_families':48,'added_families_relative_v1':0,'status':'Public arithmetic only; no actual future cohort truth established.'}

def preparation_v2():
    freeze_path=ROOT/ROOT_FREEZE;require(sha(freeze_path)==ROOT_FREEZE_SHA,'exact root public freeze')
    bindings=json.loads(freeze_path.read_text())['public_specification_bindings']
    for rel,h in bindings.items():require(sha(ROOT/rel)==h,'public doc hash '+rel)
    require(sha(ROOT/'audits/rgm_planning_gap_adversarial_preparation.py')==OLD_SOURCE_SHA,'own frozen V1 source')
    source=pathlib.Path(__file__);source_hash=sha(source)
    snapshot=ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_preparation_r4-{source_hash}.py'
    immutable(snapshot,source.read_text())
    handmade=manual_v2_fixtures();clearance=v2_clearance_cases(handmade)
    # Only handwritten literal guards are expected at this point; actual recomputation occurs below.
    all_true={k:True for k in GUARDS};source_false={**all_true,'source_clear':False};dest_false={**all_true,'destination_clear_or_table':False}
    by={x['name']:x['task'] for x in handmade}
    selective=[{'name':'source_initial','left':by['sole_source_base'],'right':by['sole_source_failure'],'evidence':{'swap_index':0,'first_failure_index':0,'first_failure_reason':'source_not_clear','guard_truth':source_false}},
               {'name':'destination_initial','left':by['sole_destination_base'],'right':by['sole_destination_failure'],'evidence':{'swap_index':0,'first_failure_index':0,'first_failure_reason':'destination_not_clear','guard_truth':dest_false}},
               {'name':'source_failure_not_swap_index','left':by['four_move_base'],'right':by['source_failure_index_differs_from_swap'],'evidence':{'swap_index':1,'first_failure_index':2,'first_failure_reason':'source_not_clear','guard_truth':source_false}}]
    pre={'actual_execution_id':EXECUTION,'batch':'planning-gap-v2','stage':'PUBLIC_ONLY_SOURCE_AND_HAND_EXPECTATIONS_BEFORE_SOLVER_EXECUTION','public_source_bindings':bindings,'root_freeze_sha256':ROOT_FREEZE_SHA,'source_sha256':source_hash,'own_reused_v1_source_sha256':OLD_SOURCE_SHA,'manual_exact_targets':handmade,'clearance_expectation_cases':clearance,'selective_hand_guard_cases':selective,'read_scope':'Seven public documents/root freeze and own frozen V1 source. No V2 source/candidate/gold/manifests/formal/test/other reviewer private implementation or verdict.','correlation_disclosure':'Public-guided revision, same underlying model/common design, reuse of own V1 solvers/fixtures. No independent-model or blind-to-design claim.','candidate_approval':False,'prior_failed_own_source_sha256':'9b3951ccd4ae76caec4e60e64d77f3a8588909d08cb9f179d002027235787dce','prior_failed_execution_sha256':['2a56baac0f3f1850e1829e46a1f8b527f6ea2e4c1b15f5662bd3e28931764c93','69c02da26cccd784f75929413ae104d6d0182c645c2fd1e6069619db0cfed6dd','85cac7206b2cac169cf2d0a879df48b3cf7b21ef125e1a827a758fd65df2db93'],'correction':'Own hand lexical fixture had spare D parking lexically before TABLE. Fresh toy uses spare Z. Second own partial-goal toy permitted B→A parking lexically before B→TABLE, so new toy explicitly requires A:TABLE and B:TABLE. Third own malformed-evidence toy accidentally selected a real B destination rather than TABLE; new case actually selects the B→TABLE action. All first expectations/sources/failures preserved without editing.'}
    prepath=ROOT/'audits/execution/planning-gap-v2-adversarial-preparation-r4.preexec.json';immutable(prepath,json.dumps(pre,sort_keys=True,indent=2)+'\n')
    print(canon({'source_sha256':source_hash,'preexec_sha256':sha(prepath),'new_hand_target_count':len(handmade),'clearance_cases':len(clearance),'stage':'SOURCE_EXPECTATIONS_FROZEN_BEFORE_ACTUAL_EXECUTION'}),flush=True)
    started=time.perf_counter();execution={'actual_execution_id':EXECUTION,'source_sha256':source_hash,'preexec_sha256':sha(prepath),'root_freeze_sha256':ROOT_FREEZE_SHA,'public_source_bindings':bindings,'completed':False,'candidate_approval':False,'candidate_or_gold_or_v2_private_implementation_read':False,'prior_failed_own_source_sha256':'9b3951ccd4ae76caec4e60e64d77f3a8588909d08cb9f179d002027235787dce','prior_failed_execution_sha256':['2a56baac0f3f1850e1829e46a1f8b527f6ea2e4c1b15f5662bd3e28931764c93','69c02da26cccd784f75929413ae104d6d0182c645c2fd1e6069619db0cfed6dd','85cac7206b2cac169cf2d0a879df48b3cf7b21ef125e1a827a758fd65df2db93']}
    try:
        legacy=calibration();old_bad=malformed_probes();cross=independent_block_enumeration();legacy_meta=metamorphic_and_shortcut_probes(legacy)
        target_results=[];guard_results=[];metas=[];mutation_results=[]
        for toy in handmade:
            target,info=solve(toy['task']);require(canon(target)==canon(toy['hand_derived_exact_target']),'new hand target '+toy['name'])
            alt=alternate_blocks_minimum(toy['task']);require(canon(alt)==canon(target['shortest_plan']),'independent relation search '+toy['name'])
            strict,guards,failure=diagnostic_replay(toy['task']);require(canon(strict)==canon({k:v for k,v in target.items() if k!='shortest_plan'}),'independent all-guard replay '+toy['name'])
            target_results.append({**toy,'actual_target':target,'measurements':info,'full_guard_truth_by_evaluated_step':guards,'independent_relation_minimum':alt,'passed':True})
            reordered=copy.deepcopy(toy['task']);reordered['on']=dict(reversed(list(reordered['on'].items())));reordered['goal_on']=dict(reversed(list(reordered['goal_on'].items())))
            require(canon(solve(reordered)[0])==canon(target),'field order '+toy['name']);metas.append({'name':'key_order_'+toy['name'],'passed':True})
            if failure is not None:
                suffix=copy.deepcopy(toy['task']);suffix['plan']+=[move('UNKNOWN','TABLE')]
                require(canon(solve(suffix)[0])==canon(target),'suffix absorption '+toy['name']);metas.append({'name':'suffix_'+toy['name'],'passed':True})
            names=list(toy['task']['on']);mp={n:'renamed_'+str(len(names)-i) for i,n in enumerate(names)};ren=blocks({mp[n]:'TABLE' if d=='TABLE' else mp[d] for n,d in toy['task']['on'].items()},{mp[n]:'TABLE' if d=='TABLE' else mp[d] for n,d in toy['task']['goal_on'].items()},[move(mp[a['block']],'TABLE' if a['to']=='TABLE' else mp[a['to']]) for a in toy['task']['plan']])
            rt,_=solve(ren);require(rt['valid']==target['valid'] and rt['reaches_goal']==target['reaches_goal'] and len(rt['shortest_plan'])==len(target['shortest_plan']) and canon(rt['shortest_plan'])==canon(alternate_blocks_minimum(ren)),'renaming measures and actual lexical recomputation '+toy['name'])
            require(canonical_structure(ren)==canonical_structure(toy['task']),'rename canonical joint structure '+toy['name']);metas.append({'name':'rename_'+toy['name'],'passed':True})
        for c in clearance:
            try:validate_clearance_v2(c['task'],c['evidence']);valid=True;error=None
            except ValueError as e:valid=False;error=str(e)
            guard_results.append({'name':c['name'],'expected_valid':c['expected_valid'],'actual_valid':valid,'error':error});require(valid==c['expected_valid'],'clearance fixture '+c['name'])
        for c in selective:mutation_results.append({'name':c['name'],'evidence':c['evidence'],**validate_selective_pair(c['left'],c['right'],c['evidence'])})
        noops=[]
        for k,b in [(1,'A'),(2,'A'),(3,'Z')]:
            base=by['four_move_base'];prefix=blocks(base['on'],base['goal_on'],base['plan'][:k]);appended=by['isolated_noop_prefix_'+str(k)];noops.append({'prefix_length':k,'block':b,**validate_noop_pair(base,prefix,appended,{'prefix_length':k,'block':b,'current_support':'TABLE'})})
        rows,controls=control_fixture_rows(handmade);renderer=renderer_cases(rows,controls);renderer_results=[]
        for c in renderer:
            try:validate_renderer_declarations(c['rows'],c['controls']);valid=True;error=None
            except ValueError as e:valid=False;error=str(e)
            renderer_results.append({'name':c['name'],'expected_valid':c['expected_valid'],'actual_valid':valid,'error':error});require(valid==c['expected_valid'],'renderer fixture '+c['name'])
        # Exact typed rendering comparison must not use Python bool/int equality.
        task=graph(['S','G'],[['S','G',1]],'S','G',1,[]);bad=copy.deepcopy(task);bad['fuel']=True
        require(task==bad and canon(task)!=canon(bad),'JSON typed rendering adversarial bool1 mismatch');metas.append({'name':'typed_rendering_bool1_no_python_equality','passed':True})
        feature_results=[{'name':x['name'],'input_features':input_features(x['task']),'labels':{k:x['hand_derived_exact_target'][k] for k in ('valid','reaches_goal')},'fresh_minimum_moves':len(x['hand_derived_exact_target']['shortest_plan'])} for x in handmade]
        same_plan_different_guard=by['sole_source_failure']['plan'][0]['block']!=by['sole_destination_failure']['plan'][0]['block']
        require(same_plan_different_guard,'independent guard examples distinct actions')
        allocation=public_allocation()
        for rel,h in bindings.items():require(sha(ROOT/rel)==h,'public bytes changed during actual execution')
        execution.update({'completed':True,'status':'PREPARATION_ONLY_NO_DATASET_APPROVAL','new_exact_hand_targets':len(target_results),'reused_v1_exact_hand_targets':len(legacy),'reused_v1_malformed_rejections':len(old_bad),'canonical_clearance_validation_cases':len(guard_results),'canonical_clearance_acceptances':sum(c['actual_valid'] for c in guard_results),'canonical_clearance_rejections':sum(not c['actual_valid'] for c in guard_results),'selective_clearance_mutation_kills':len(mutation_results),'isolated_noop_mutation_kills':len(noops),'clearance_mutant_noop_agreements':2*len(noops),'renderer_encoding_cases':len(renderer_results),'renderer_encoding_rejections':sum(not c['actual_valid'] for c in renderer_results),'fresh_metamorphic_checks':len(metas),'reused_metamorphic_checks':legacy_meta['actual_metamorphic_count'],'independent_enumeration':cross,'target_results':target_results,'clearance_results':guard_results,'selective_mutation_evidence':mutation_results,'isolated_noop_evidence':noops,'renderer_cases':renderer_results,'new_metamorphic_results':metas,'input_feature_toy_results':feature_results,'public_allocation_arithmetic':allocation,'actual_duration_seconds':time.perf_counter()-started,'future_review_plan':['Freeze ALL408 input-derived exact targets/initial input judgments before candidate/gold/manifest read; recheck typed JSON readback, exact input hashes and per-row complete search.','Derive all six block guards; test source/destination/noop mutant traces on every actually affected C1/C4; require selective guard and proper prefix/witness, canonical initial source-priority, actual no-op k.','Measure successes/validity/feasibility/minimum length/first error separately by family,stratum,split,renderer,size,planlength,fuel,edgecount/degree,character/token/name/ID/rank and joint features. CPU train→dev majority/lookup probes, within-family opposite outcomes and conditioned results; no model training.','Measure full supplied 96semantic pairs/192per-style declarations independently without collapsing repeated names; count both renderer coverage, unique control_IDs, context row tags versus family claims, exact field differences and true locators.','Derive weighted/start-goal graph and unweighted graph permutation signatures plus block initial/joint-goal signatures; report crosssplit exact/structural/near duplicates and V1 shared48lineage honestly, no new independent family or sealedOOD claim.','Separate target-format/trace-length intent from input outcome leakage; exposed development, common public design and same model/own prior exposure remain disclosed.'], 'limits':['Hand toys are tests of this own reviewer implementation; no future candidate or target approved.','V2 revision reuses48 lineage families, two action grammars and exposed development data; no new independent mechanism or efficacy claim.','Preparation uses public prospective allocation and own V1 solvers/calibration; same-model/shared-design correlation remains.','Source/destination mutants deliberately permit resulting invalid support maps; no compensating validator silently restores omitted guard.']})
    except Exception as e:
        execution.update({'status':'ACTUAL_PREPARATION_FAILED','failure_type':type(e).__name__,'failure':str(e),'actual_duration_seconds':time.perf_counter()-started})
        path=ROOT/f'audits/execution/planning-gap-v2-adversarial-preparation.failure-{source_hash[:16]}.json';immutable(path,json.dumps(execution,sort_keys=True,indent=2)+'\n');print(canon({'failed':True,'failure_path':str(path.relative_to(ROOT)),'failure_sha256':sha(path),'failure':str(e)}),flush=True);raise
    ex=ROOT/'audits/execution/planning-gap-v2-adversarial-preparation-r4.json';immutable(ex,json.dumps(execution,sort_keys=True,indent=2)+'\n')
    print(canon({'completed':True,'source_sha256':source_hash,'preexec_sha256':sha(prepath),'execution_sha256':sha(ex),'new_exact_hand_targets':len(target_results),'clearance_cases':len(guard_results),'selective_mutation_kills':len(mutation_results),'isolated_noop_kills':len(noops),'renderer_cases':len(renderer_results),'fresh_metamorphics':len(metas),'public_sources_sha256':bindings,'independent_enumeration':cross}),flush=True)

if __name__=='__main__':preparation_v2()
