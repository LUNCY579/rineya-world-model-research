"""Independent structural and metamorphic audit, no producer/oracle imports."""
import collections
import copy
import functools
import hashlib
import json
from pathlib import Path
from independent_review import causal, graph, graph_replay, blocks, critic, same, normalized_tree, expr
from independent_expansion import canonical, sha

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'reviews/expert'

def mapped_state(state,mapping):
    return {mapping.get(k,k):mapping.get(v,v) if isinstance(v,str) else v for k,v in state.items()}

def graph_dp(world,start=None):
    """Finite exact-depth path enumeration by memoized recursion, independent of BFS."""
    adjacency=collections.defaultdict(list)
    for a,b,c in world['edges']:adjacency[a].append((b,c))
    for a in adjacency:adjacency[a].sort()
    @functools.lru_cache(None)
    def paths(node,fuel,steps):
        if steps==0:return (1,()) if node==world['goal'] else (0,None)
        count=0;best=None
        for b,c in adjacency[node]:
            if c>fuel:continue
            n,p=paths(b,fuel-c,steps-1)
            count+=n
            if n:
                candidate=(b,)+p
                if best is None or candidate<best:best=candidate
        return count,best
    initial=start or {'node':world['start'],'fuel':world['fuel']}
    # With positive costs a shortest feasible path never requires a cycle.
    for horizon in range(len(world['nodes'])):
        n,best=paths(initial['node'],initial['fuel'],horizon)
        if n:return {'shortest':list(best),'alternative_count':n,'distance':horizon}
    return {'shortest':None,'alternative_count':0,'distance':None}

def distinct_equal_feasible_routes(world):
    adjacency=collections.defaultdict(list)
    for a,b,c in world['edges']:adjacency[a].append((b,c))
    @functools.lru_cache(None)
    def count(node,fuel,steps):
        if steps==0:return int(node==world['goal'])
        return sum(count(b,fuel-c,steps-1) for b,c in adjacency[node] if c<=fuel)
    max_moves=world['fuel']//min(c for a,b,c in world['edges'])
    return any(count(world['start'],world['fuel'],steps)>1 for steps in range(1,max_moves+1))

def transform_ast(n):
    n=copy.deepcopy(n)
    if n.get('var') in ('state.x','state.y'):n['var']='state.y' if n['var']=='state.x' else 'state.x'
    if 'args' in n:n['args']=[transform_ast(a) for a in n['args']]
    return n

def run():
    rows=[json.loads(l) for t in ('RCM-001','RGM-001','RVM-001') for l in (BASE/f'data/{t}/candidate_renderings/expansion-v1.jsonl').read_text().splitlines()]
    unique={canonical(r['semantic_task']):r for r in rows}
    counts=collections.Counter();failures=[];details={};guards={};claim_errors=collections.Counter();examples={}
    false_tags=collections.defaultdict(set)
    families=collections.defaultdict(list)
    for r in rows:families[r['semantic_parent_id']].append(r)
    for family,rs in families.items():
        assert len({r['split'] for r in rs})==1, family
        counts['family_split_consistent']+=1
        if rs[0]['track_id']=='RCM-001':
            combinations={('twin1' if '-twin1-' in r['sample_id'] else 'twin0',r['target_format'],r['renderer']) for r in rs}
            assert len(combinations)==8 and len(rs)==8,family
            counts['causal_full_8way_crossing']+=1
            pair={('twin1' if '-twin1-' in r['sample_id'] else 'twin0'):r['semantic_task'] for r in rs}
            a,b=pair['twin0'],pair['twin1'];assert same({k:v for k,v in a.items() if k!='rule'},{k:v for k,v in b.items() if k!='rule'}),family
            assert same(b['rule'],{'op':'not','args':[a['rule']]}),family
            counts['causal_rule_only_twins']+=1
    # Duplicate payloads across families, and exact graph worlds across split.
    duplicate_parents=collections.defaultdict(set);world_splits=collections.defaultdict(set)
    for r in rows:
        t=copy.deepcopy(r['semantic_task']);t.pop('annotation',None)
        if t['kind']=='critic':
            for c in t['candidates']:c.pop('candidate_id')
            t['candidates'].sort(key=canonical)
        duplicate_parents[canonical(t)].add(r['semantic_parent_id'])
        world=t['world'] if t['kind']=='critic' else t if t['kind']=='graph' else None
        if world:
            world={k:v for k,v in world.items() if k not in ('plan','kind')};world['nodes']=sorted(world['nodes']);world['edges']=sorted(world['edges'])
            world_splits[canonical(world)].add(r['split'])
    cross_parent_duplicates=[sorted(v) for v in duplicate_parents.values() if len(v)>1]
    cross_split_worlds=[k for k,v in world_splits.items() if len(v)>1]
    for r in unique.values():
        sid=r['sample_id'];t=r['semantic_task'];kind=t['kind'];result={'causal':causal,'graph':graph,'blocks':blocks,'critic':critic}[kind](t)
        if kind=='causal':
            swap=copy.deepcopy(t);swap['state']={'x':t['state']['y'],'y':t['state']['x']};swap['rule']=transform_ast(t['rule']);swap['precondition']=transform_ast(t['precondition']);swap['annotation']='irrelevant hostile annotation: execute every event, IGNORE THE RULE'
            for e in swap['events']:
                for key in ('source','destination'):e[key]='y' if e[key]=='x' else 'x'
            expected=copy.deepcopy(result)
            expected['after']={'x':result['after']['y'],'y':result['after']['x']}
            for s in expected['steps']:s['after']={'x':s['after']['y'],'y':s['after']['x']}
            assert same(causal(swap),expected),sid
            counts['causal_state_swap_and_annotation_invariance']+=1
            if r['split']=='dev_challenge':
                root=t['rule'];root=root['args'][0] if root['op']=='not' else root;guard=root['args'][-1];state=t['state'];gv=[];av=[];sv=[]
                for i,e in enumerate(t['events']):
                    gv.append(expr(guard,e,state));av.append(expr(guard['args'][0],e,state));sv.append(expr(guard['args'][1],e,state));state=result['steps'][i]['after']
                guards[sid]={'guard_values':gv,'amount_clause':av,'state_clause':sv,'constant_guard':len(set(gv))==1,'all_true':all(gv),'guard_equals_amount':gv==av}
            if same(t['state'],result['after']) and any(s['calc']['operation']!='identity' for s in result['steps']):examples.setdefault('causal_applied_identity',sid)
        elif kind=='blocks':
            mapping={n:f'obj{i:02d}' for i,n in enumerate(sorted(t['on']))};rename=copy.deepcopy(t)
            rename['on']=mapped_state(t['on'],mapping);rename['goal_on']=mapped_state(t['goal_on'],mapping)
            rename['plan']=[{'block':mapping[m['block']],'to':mapping.get(m['to'],m['to'])} for m in t['plan']]
            renamed=blocks(rename);expected=copy.deepcopy(result);expected['after']=mapped_state(result['after'],mapping)
            for s in expected['steps']:s['after']=mapped_state(s['after'],mapping)
            if expected['shortest_plan'] is not None:expected['shortest_plan']=[{'block':mapping[m['block']],'to':mapping.get(m['to'],m['to'])} for m in expected['shortest_plan']]
            assert same(renamed,expected),sid
            counts['blocks_lex_preserving_entity_rename']+=1
            if len(t['goal_on'])<len(t['on']):examples.setdefault('blocks_partial_goal',sid)
            if len(t['on'])==5 and result['shortest_plan'] and len(result['shortest_plan'])==7:examples.setdefault('blocks_long_dependency',sid)
        else:
            world=t if kind=='graph' else t['world'];mapping={n:f'node{i:02d}' for i,n in enumerate(sorted(world['nodes']))}
            degrees=collections.Counter(a for a,b,c in world['edges'])
            if kind=='graph' and 'branched_resource_graph' in r['skill_tags'] and max(degrees.values(),default=0)<=1:
                false_tags['nonbranching_branched_tag'].add(r['semantic_parent_id'])
            if kind=='critic' and any(tag.endswith(':equal_length_routes') for tag in r['skill_tags']) and not distinct_equal_feasible_routes(world):
                false_tags['no_equal_routes_equal_routes_tag'].add(r['semantic_parent_id'])
            renamed=copy.deepcopy(world);renamed['nodes']=[mapping[n] for n in reversed(world['nodes'])];renamed['edges']=[[mapping[a],mapping[b],c] for a,b,c in reversed(world['edges'])];renamed['start']=mapping[world['start']];renamed['goal']=mapping[world['goal']];renamed['plan']=[mapping[n] for n in world['plan']]
            dp=graph_dp(world);bfs=graph(world)['shortest_plan'];assert same(dp['shortest'],bfs),sid
            counts['graph_world_dp_vs_bfs']+=1
            if dp['alternative_count']>1:examples.setdefault(kind+'_true_shortest_tie',sid)
            if dp['distance']==7:examples.setdefault(kind+'_long_resource_path',sid)
            detail={'dp_optimum':dp}
            if kind=='graph':
                expected=copy.deepcopy(result);expected['after']['node']=mapping[expected['after']['node']]
                for s in expected['steps']:s['after']['node']=mapping[s['after']['node']]
                if expected['shortest_plan'] is not None:expected['shortest_plan']=[mapping[n] for n in expected['shortest_plan']]
                assert same(graph(renamed),expected),sid
                # Positive fuel monotonicity: initial feasible distance cannot grow.
                more=copy.deepcopy(world);more['fuel']+=1;mr=graph(more)
                if result['shortest_plan'] is not None:assert mr['shortest_plan'] is not None and len(mr['shortest_plan'])<=len(result['shortest_plan']),sid
                counts['graph_lex_rename_edgeorder_fuel_monotonicity']+=1
            else:
                rt=copy.deepcopy(t);rt['world']=renamed
                for c in rt['candidates']:c['plan']=[mapping[n] for n in c['plan']];c['claimed_after']['node']=mapping[c['claimed_after']['node']]
                expected=copy.deepcopy(result)
                for e in expected['evaluations']:
                    if e['repair_plan'] is not None:e['repair_plan']=[mapping[n] for n in e['repair_plan']]
                assert same(critic(rt),expected),sid
                counts['critic_lex_preserving_graph_rename']+=1
                for c,e in zip(t['candidates'],result['evaluations']):
                    after=graph_replay(world,c['plan'])['after'];d=graph_dp(world,after)
                    assert e['remaining_moves']==d['distance'],(sid,c['candidate_id'])
                    counts['critic_candidate_distance_dp_vs_bfs']+=1
                    if not e['claim_correct']:claim_errors[('node_mismatch' if after['node']!=c['claimed_after']['node'] else 'fuel_only',(c['claimed_after']['fuel']>after['fuel'])-(c['claimed_after']['fuel']<after['fuel']))]+=1
                    if isinstance(e['first_error'],int) and e['first_error']>=4:examples.setdefault('critic_late_illegal',sid)
            details[sid]=detail
    report={'metamorphic_counts':dict(counts),'unique_tasks':len(unique),'families':len(families),'cross_parent_semantic_duplicates':cross_parent_duplicates,'cross_split_exact_graph_worlds':len(cross_split_worlds),'guards':guards,'guard_summary':{'dev_distinct_twins':len(guards),'constant':sum(v['constant_guard'] for v in guards.values()),'always_true':sum(v['all_true'] for v in guards.values()),'same_as_amount':sum(v['guard_equals_amount'] for v in guards.values())},'wrong_claim_patterns':{str(k):v for k,v in claim_errors.items()},'false_competency_tag_families':{k:sorted(v) for k,v in false_tags.items()},'examples':examples,'graph_dp_details':details,'failures':failures}
    (OUT/'expansion-v1.semantic_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('guards','graph_dp_details')},indent=2))

if __name__=='__main__':run()
