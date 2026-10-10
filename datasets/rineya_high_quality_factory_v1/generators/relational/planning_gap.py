"""RGM planning-gap producer; finite public-contract worlds and controls only."""
import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import heapq
from itertools import permutations, product
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record, record_hash, split_for, write_jsonl
from factory import reference
from generators.relational.generate import GRAPH_CONTRACT, BLOCK_CONTRACT, reorder

SEED=20261009
ACTOR="/root/relational_critic_producer"
VERSION="rgm-planning-gap-1.0.0"
PUBLIC={"schemas/SEMANTICS.md":"718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd",
        "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md":"f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a",
        "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md":"6100d43fe1cf83d798bb34799fac70569f54d33f66692150d188581d9c940761",
        "research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl":"896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d",
        "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md":"02613aa2333f401117def89da224f169d564b527bacc554a0b872877aaac42fe"}
FREEZE="research/curricula/RGM-NEXT-GAP-ROOT_FREEZE.json"
FREEZE_SHA="2a4af1615bd6e58a7ac71c97b615d405df3354235360c682486fb4f4c1f5f52f"
WORDS=("amber","birch","cedar","dune","ember","fjord","grove","hazel","iris","jade","kelp","larch","maple","nacre","olive","pearl","quartz","rivet","spruce","topaz","umber","vessel","willow","yarrow")
STATE_CACHE={};REFERENCE_CACHE={};SEARCH_COSTS=Counter();REFERENCE_INVOCATIONS=0;SPACE_OBSERVATIONS=[]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal(a,b):return canonical(a)==canonical(b)
def check_space(path,stage):
    location=Path(path).resolve()
    while not location.exists():location=location.parent
    free=shutil.disk_usage(location).free
    SPACE_OBSERVATIONS.append(dict(stage=stage,available_bytes=free))
    assert free>=32*1024*1024,"Storage under32MiB; preserve current outputs and stop"


def write_json(path,value):
    check_space(path,"json_output:"+Path(path).name)
    atomic(path,canonical(value)+"\n")


def search_result(kind,result):
    SEARCH_COSTS[kind+"_calls"]+=1
    for field in ("visited_states","expanded_states"):
        SEARCH_COSTS[kind+"_total_"+field]+=result[field]
    for field in ("visited_states","max_frontier"):
        SEARCH_COSTS[kind+"_maximum_"+field]=max(SEARCH_COSTS[kind+"_maximum_"+field],result[field])
    return result


def graph_search(t, topology=False):
    adjacency=defaultdict(list)
    for a,b,c in t["edges"]:adjacency[a].append((b,c))
    for edges in adjacency.values():edges.sort()
    start=(t["start"],0 if topology else t["fuel"])
    layer={start:(1,())};seen={start:0};depth=0;peak=1;popped=0
    while layer:
        goals=[(state,count,path) for state,(count,path) in layer.items() if state[0]==t["goal"]]
        if goals:
            return search_result("graph_topology" if topology else "graph",dict(plan=list(min(path for _,_,path in goals)),minimum_moves=depth,minimum_route_count=sum(count for _,count,_ in goals),
                        visited_states=len(seen),expanded_states=popped,max_frontier=peak,complete=True,completion="OPTIMUM_PROVED"))
        next_layer={}
        for (node,fuel),(count,path) in sorted(layer.items(),key=lambda item:item[1][1]):
            popped+=1
            for destination,cost in adjacency[node]:
                if not topology and cost>fuel:continue
                state=(destination,0 if topology else fuel-cost)
                if state in seen and seen[state]<depth+1:continue
                seen[state]=depth+1
                prior_count,prior_path=next_layer.get(state,(0,path+(destination,)))
                next_layer[state]=(prior_count+count,min(prior_path,path+(destination,)))
        layer=next_layer;depth+=1;peak=max(peak,len(layer))
    return search_result("graph_topology" if topology else "graph",dict(plan=None,minimum_moves=None,minimum_route_count=0,visited_states=len(seen),expanded_states=popped,max_frontier=peak,
                complete=True,completion="EXHAUSTED_INFEASIBLE"))


def cheapest_graph(t):
    adjacent=defaultdict(list)
    for a,b,c in t["edges"]:adjacent[a].append((b,c))
    queue=[(0,0,(),t["start"])];seen=set()
    while queue:
        cost,moves,path,node=heapq.heappop(queue)
        if node in seen:continue
        seen.add(node)
        if node==t["goal"]:return cost,list(path)
        for dest,c in sorted(adjacent[node]):heapq.heappush(queue,(cost+c,moves+1,path+(dest,),dest))
    return None,None


def valid_state(state):
    supports=[s for s in state if s!=-1]
    if len(set(supports))!=len(supports):return False
    for block in range(len(state)):
        seen=set();at=block
        while at!=-1:
            if at in seen:return False
            seen.add(at);at=state[at]
    return True


def shape(state):
    lengths=[]
    above={support:block for block,support in enumerate(state) if support!=-1}
    for bottom,support in enumerate(state):
        if support==-1:
            length=1;at=bottom
            while at in above:at=above[at];length+=1
            lengths.append(length)
    return tuple(sorted(lengths,reverse=True))


def states(size):
    if size not in STATE_CACHE:
        groups=defaultdict(list)
        for state in product(*[tuple([-1]+[j for j in range(size) if j!=i]) for i in range(size)]):
            if valid_state(state):groups[shape(state)].append(state)
        STATE_CACHE[size]=groups
    return STATE_CACHE[size]


def block_encoding(task):
    names=sorted(task["on"]);indices={name:i for i,name in enumerate(names)}
    initial=tuple(-1 if task["on"][name]=="TABLE" else indices[task["on"][name]] for name in names)
    goal={indices[block]:-1 if destination=="TABLE" else indices[destination] for block,destination in task["goal_on"].items()}
    return names,initial,goal


def block_search(task, preserve_initial_goal=False):
    names,initial,goal=block_encoding(task)
    protected={i:support for i,support in goal.items() if initial[i]==support} if preserve_initial_goal else {}
    queue=deque([initial]);parent={initial:None};moves={};peak=1;popped=0
    while queue:
        state=queue.popleft();popped+=1
        if all(state[block]==destination for block,destination in goal.items()):
            actions=[];at=state
            while parent[at] is not None:
                block,dest=moves[at];actions.append(dict(block=names[block],to="TABLE" if dest==-1 else names[dest]));at=parent[at]
            return search_result("blocks_preserving" if preserve_initial_goal else "blocks",dict(plan=list(reversed(actions)),minimum_moves=len(actions),visited_states=len(parent),expanded_states=popped,max_frontier=peak,
                        complete=True,completion="OPTIMUM_PROVED",protected_initial_goal_relations={names[i]:"TABLE" if dest==-1 else names[dest] for i,dest in protected.items()}))
        occupied={support for support in state if support!=-1}
        clear=[i for i in range(len(names)) if i not in occupied]
        for block in clear:
            for dest in [-1]+clear: # TABLE sorts before the lowercase synthetic names.
                if block==dest or state[block]==dest:continue
                after=list(state);after[block]=dest;after=tuple(after)
                if after in parent or any(after[i]!=support for i,support in protected.items()):continue
                parent[after]=state;moves[after]=(block,dest);queue.append(after)
        peak=max(peak,len(queue))
    return search_result("blocks_preserving" if preserve_initial_goal else "blocks",dict(plan=None,minimum_moves=None,visited_states=len(parent),expanded_states=popped,max_frontier=peak,complete=True,
                completion="EXHAUSTED_INFEASIBLE",protected_initial_goal_relations={names[i]:"TABLE" if dest==-1 else names[dest] for i,dest in protected.items()}))


def reference_target(task):
    # Existing producer reference supplies both replay and lexicographic fresh plan.
    global REFERENCE_INVOCATIONS
    REFERENCE_INVOCATIONS+=1
    key=digest({key:value for key,value in task.items() if key!="plan"})
    if key not in REFERENCE_CACHE:
        REFERENCE_CACHE[key]=reference.shortest_graph(task) if task["kind"]=="graph" else reference.shortest_blocks(task)
    if task["kind"]=="graph":out=reference.graph_run(task)
    else:
        state=dict(task["on"]);steps=[];valid=True
        for action in task["plan"]:
            legal,state=reference.blocks_move(state,action);steps.append(dict(legal=legal,after=dict(state)))
            if not legal:valid=False;break
        out=dict(steps=steps,after=state,valid=valid,reaches_goal=valid and all(state[b]==d for b,d in task["goal_on"].items()))
    return dict(**out,shortest_plan=REFERENCE_CACHE[key])


def graph_signature(task, weighted=True, terminals=True):
    nodes=task["nodes"];edges=task["edges"]
    incoming=defaultdict(list);outgoing=defaultdict(list)
    for a,b,c in edges:outgoing[a].append(c if weighted else 1);incoming[b].append(c if weighted else 1)
    colors={node:(int(terminals and node==task["start"]),int(terminals and node==task["goal"]),
                  tuple(sorted(incoming[node])),tuple(sorted(outgoing[node]))) for node in nodes}
    groups=defaultdict(list)
    for node in nodes:groups[colors[node]].append(node)
    ordered=[groups[color] for color in sorted(groups)]
    best=None
    for choices in product(*[list(permutations(group)) for group in ordered]):
        sequence=[node for group in choices for node in group];mapping={node:i for i,node in enumerate(sequence)}
        transformed=tuple(sorted((mapping[a],mapping[b],c) if weighted else (mapping[a],mapping[b]) for a,b,c in edges))
        candidate=(tuple(sorted(colors.values())),transformed)
        if best is None or candidate<best:best=candidate
    return digest(best)


def block_signature(task):
    names,initial,goal=block_encoding(task);size=len(names);best=None
    for order in permutations(range(size)):
        mapping={old:new for new,old in enumerate(order)}
        on=tuple(-1 if initial[old]==-1 else mapping[initial[old]] for old in order)
        desired=tuple(-2 if old not in goal else -1 if goal[old]==-1 else mapping[goal[old]] for old in order)
        candidate=(on,desired)
        if best is None or candidate<best:best=candidate
    return digest(best)


def block_task(initial,goal,names):
    return dict(kind="blocks",on={names[i]:"TABLE" if support==-1 else names[support] for i,support in enumerate(initial)},
                goal_on={names[i]:"TABLE" if support==-1 else names[support] for i,support in goal.items()},plan=[])


def first_failure(task,target):
    if target["valid"]:return None,None
    index=len(target["steps"])-1
    before=task["on"] if task["kind"]=="blocks" and index==0 else target["steps"][index-1]["after"] if index else {"node":task["start"],"fuel":task["fuel"]}
    action=task["plan"][index]
    if task["kind"]=="graph":
        edge={(a,b):cost for a,b,cost in task["edges"]}
        reason="missing_directed_edge" if (before["node"],action) not in edge else "insufficient_fuel"
    else:
        clear=set(before)-set(before.values())
        reason="source_not_clear" if action["block"] not in clear else "destination_not_clear" if action["to"]!="TABLE" and action["to"] not in clear else "unchanged_support"
    return index,reason


def clearance_witness(task,target):
    if task["kind"]!="blocks":return None
    initial=task["on"];initial_clear=set(initial)-set(initial.values())
    for index,step in enumerate(target["steps"]):
        if index<1 or not step["legal"]:continue
        action=task["plan"][index]
        blocked=[("source",action["block"])] if action["block"] not in initial_clear else []
        if action["to"]!="TABLE" and action["to"] not in initial_clear:blocked.append(("destination",action["to"]))
        for precondition,obj in blocked:
            prior=next((j for j in range(index) if target["steps"][j]["legal"] and initial.get(task["plan"][j]["block"])==obj),None)
            if prior is not None:return dict(action_index=index,clearance_precondition=precondition,object=obj,preceding_clearance_change_index=prior,
                                             initially_on_object=[b for b,support in initial.items() if support==obj])
    return None


def row_analysis(task,target):
    search=graph_search(task) if task["kind"]=="graph" else block_search(task)
    assert equal(search["plan"],target["shortest_plan"])
    failure,reason=first_failure(task,target);properties={}
    if failure is not None and failure>=1:properties["legal_prefix_then_first_failure"]=dict(first_illegal_action_index=failure,reason=reason,evaluated_steps=len(target["steps"]),submitted_steps=len(task["plan"]))
    if target["valid"] and target["reaches_goal"] and len(task["plan"])>search["minimum_moves"]:
        properties["valid_goal_detour"]=dict(submitted_moves=len(task["plan"]),fresh_minimum_moves=search["minimum_moves"])
    extra={}
    if task["kind"]=="graph":
        topology=graph_search(task,True);cheap,_=cheapest_graph(task)
        extra=dict(topology_minimum_moves=topology["minimum_moves"],minimum_required_fuel=cheap)
        if search["minimum_moves"] is None and topology["minimum_moves"] is not None:properties["fuel_blocks_topologically_reachable_goal"]=dict(topology_minimum_moves=topology["minimum_moves"],initial_fuel=task["fuel"],minimum_required_fuel=cheap,search=search)
        if search["minimum_moves"] is not None and search["minimum_moves"]>topology["minimum_moves"]:properties["fuel_lengthens_fresh_minimum_moves"]=dict(topology_minimum_moves=topology["minimum_moves"],fuel_feasible_minimum_moves=search["minimum_moves"],initial_fuel=task["fuel"])
    else:
        witness=clearance_witness(task,target)
        if witness:properties["later_action_requires_changed_block_clearance"]=witness
        if 0<len(task["goal_on"])<len(task["on"]):properties["partial_block_goal"]=dict(specified_blocks=sorted(task["goal_on"]),goal_relation_count=len(task["goal_on"]),block_count=len(task["on"]))
        constrained=block_search(task,True)
        if constrained["protected_initial_goal_relations"] and search["plan"] is not None and constrained["plan"] is None:
            properties["required_temporary_goal_displacement"]=dict(initially_satisfied_goal_relations=constrained["protected_initial_goal_relations"],preserving_search=constrained)
        extra=dict(initial_shape=list(shape(block_encoding(task)[1])),goal_relation_count=len(task["goal_on"]))
    level=4 if search["minimum_moves"] is not None and search["minimum_moves"]>=4 and ("later_action_requires_changed_block_clearance" in properties or "fuel_lengthens_fresh_minimum_moves" in properties) else 3 if properties or (search["minimum_moves"] or 0)>=3 else 2
    return properties,dict(kind=task["kind"],valid=target["valid"],reaches_goal=target["reaches_goal"],fresh_plan_feasible=search["plan"] is not None,
        fresh_plan_length=search["minimum_moves"],submitted_plan_length=len(task["plan"]),first_error_index=failure,first_error_reason=reason,
        object_count=len(task["nodes"] if task["kind"]=="graph" else task["on"]),edge_count=len(task["edges"]) if task["kind"]=="graph" else None,
        initial_fuel=task.get("fuel"),search=search,structural_difficulty=level,**extra)


def graph_family(index,ordinal,stratum,rng):
    # Both graph strata expose5/6/7 nodes in train; their two dev slots cover5/6 and6/7 respectively.
    size=(5+ordinal%3) if split_for(index)=="train" else ({"A":{188:5,189:6},"B":{198:6,199:7}}[stratum][index])
    for attempt in range(10000):
        names=rng.sample(WORDS,size);start,goal=names[0],names[-1]
        edges={(names[i],names[i+1]):rng.randint(1,3) for i in range(size-1)}
        if stratum=="A" and ordinal>=6:
            edges[start,names[2]]=7;edges[names[2],goal]=7
        if stratum=="B":
            edges={(start,names[1]):rng.randint(1,3),(names[1],names[2]):rng.randint(1,3),(names[2],goal):rng.randint(1,3),
                   (start,names[3]):rng.randint(1,3),(names[3],names[2]):rng.randint(1,3),(names[1],start):rng.randint(1,3)}
            for node in names[4:-1]:edges[names[3],node]=rng.randint(1,3);edges[node,names[2]]=rng.randint(1,3)
        candidates=[(a,b) for a in names for b in names if a!=b and (a,b) not in edges]
        rng.shuffle(candidates)
        for a,b in candidates[:rng.randint(1,max(2,size+1))]:edges[a,b]=rng.randint(1,7)
        world=dict(kind="graph",nodes=list(names),edges=[[a,b,c] for (a,b),c in edges.items()],start=start,goal=goal,fuel=30,plan=[])
        rng.shuffle(world["nodes"]);rng.shuffle(world["edges"])
        topology=graph_search(world,True)
        if stratum=="A" and ordinal<6:
            required,_=cheapest_graph(world)
            if required is None or required>30 or topology["minimum_moves"]<3:continue
            base={**world,"fuel":required};search=graph_search(base);plan=search["plan"]
            low={**base,"fuel":required-1,"plan":plan}
            if graph_search(low)["plan"] is not None:continue
        elif stratum=="A":
            base={**world,"fuel":14};search=graph_search(base);plan=search["plan"]
            required,_=cheapest_graph(base)
            low={**base,"fuel":required,"plan":plan};lowsearch=graph_search(low)
            if plan is None or lowsearch["plan"] is None or lowsearch["minimum_moves"]<=len(plan) or reference.graph_run(low)["valid"]:continue
        else:
            base=world;search=graph_search(base);plan=search["plan"]
            if plan is None or len(plan)<3 or (ordinal<6 and search["minimum_route_count"]<2):continue
            removed=(plan[0],plan[1]);altered=[edge for edge in world["edges"] if tuple(edge[:2])!=removed]
            low={**base,"edges":altered,"plan":plan}
            if graph_search(low)["plan"] is None:continue
        base={**base,"plan":plan}
        if stratum=="A":
            # Any legal non-goal2-move prefix is permitted; deliberately length-match A3.
            prefix=None
            for first,cost in sorted((b,c) for a,b,c in base["edges"] if a==start):
                for second,stepcost in sorted((b,c) for a,b,c in base["edges"] if a==first):
                    if second!=goal and cost+stepcost<=base["fuel"]:prefix=[first,second];break
                if prefix:break
            if prefix is None:continue
            missing=next((node for node in sorted(names) if (prefix[0],node) not in edges),None)
            if missing is None:continue
            tasks=[base,low,{**base,"plan":prefix},{**base,"plan":[prefix[0],missing]}]
        else:
            reverse=next(([b,a] for a,b,c in base["edges"] if a==start and (b,a) not in edges and c<=base["fuel"]),None)
            cycle=next(([b,start] for a,b,c in base["edges"] if a==start and (b,start) in edges and c+edges[b,start]+sum(edges[x,y] for x,y in zip([start]+plan,plan))<=30),None)
            if reverse is None or cycle is None:continue
            tasks=[base,low,{**base,"plan":reverse},{**base,"plan":cycle+plan}]
        targets=[reference_target(task) for task in tasks]
        if not(targets[0]["valid"] and targets[0]["reaches_goal"] and not targets[1]["valid"] and targets[2]["valid"]==(stratum=="A") and targets[3]["valid"]==(stratum=="B")):continue
        return tasks,dict(attempts=attempt+1,size=size)
    raise RuntimeError(f"Unable to construct public graph stratum{stratum} index{index}; no constraint weakened")


SIX_SHAPES=[(6,),(5,1),(4,2),(4,1,1),(3,3),(3,2,1),(3,1,1,1),(2,2,2),(2,2,1,1),(2,1,1,1,1),(1,1,1,1,1,1),(3,2,1)]
FIVE_SHAPES=[(5,),(4,1),(3,2),(3,1,1),(2,2,1),(2,1,1,1),(1,1,1,1,1)]


def block_detour(task,plan):
    for block in sorted(task["on"]):
        for dest in sorted(["TABLE"]+list(task["on"])):
            action=dict(block=block,to=dest);legal,after=reference.blocks_move(task["on"],action)
            if legal:
                inverse=dict(block=block,to=task["on"][block]);valid,restored=reference.blocks_move(after,inverse)
                if valid and restored==task["on"]:return [action,inverse]+plan
    raise RuntimeError("No legal reversible block detour")


def block_family(index,ordinal,stratum,rng):
    size=5 if index%2==0 else 6
    half=ordinal//2
    proposed_shape=SIX_SHAPES[half+(0 if stratum=="C" else 6)] if size==6 else FIVE_SHAPES[half%6 if stratum=="C" else (half+1)%7]
    pool=states(size);all_states=[state for group in pool.values() for state in group]
    for attempt in range(2000):
        names=sorted(rng.sample(WORDS,size));initial=rng.choice(pool[proposed_shape]);desired=rng.choice(all_states)
        if stratum=="C":goal=dict(enumerate(desired))
        else:
            count=rng.choice((2,3));chosen=rng.sample(range(size),count);goal={i:desired[i] for i in chosen}
        task=block_task(initial,goal,names);searched=block_search(task);plan=searched["plan"]
        if plan is None:continue
        if stratum=="C":
            if not 4<=len(plan)<=8:continue
            base={**task,"plan":plan};target=reference_target(base)
            if not clearance_witness(base,target):continue
            if ordinal<4 and block_search(task,True)["plan"] is not None:continue
            swaps=[]
            for j in range(len(plan)-1):
                variant=copy.deepcopy(plan);variant[j],variant[j+1]=variant[j+1],variant[j]
                swapped={**task,"plan":variant};label=reference_target(swapped)
                error,reason=first_failure(swapped,label)
                if not label["valid"] and reason in ("source_not_clear","destination_not_clear"):swaps.append((j,error,swapped))
            preferred=[item for item in swaps if item[1]==ordinal%3]
            if not preferred:continue
            _,_,swapped=preferred[0]
            tasks=[base,swapped,{**task,"plan":block_detour(task,plan)},{**task,"plan":plan[:max(1,len(plan)//2)]}]
        else:
            if len(plan)<2:continue
            full=block_task(initial,dict(enumerate(desired)),names);fullsearch=block_search(full);fullplan=fullsearch["plan"]
            if fullplan is None or len(fullplan)<=len(plan) or len(fullplan)>10:continue
            if reference_target({**full,"plan":plan})["reaches_goal"]:continue
            tasks=[{**task,"plan":plan},{**full,"plan":plan},{**full,"plan":fullplan},{**full,"plan":block_detour(full,fullplan)}]
        return tasks,dict(attempts=attempt+1,size=size,proposed_initial_shape=list(proposed_shape))
    raise RuntimeError(f"Unable to construct public block stratum{stratum} index{index}; no constraint weakened")


def prior_signatures():
    rows=load_jsonl(ROOT/"research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
    indices={int(row["source_family_key"].rsplit('-p',1)[1]) for row in rows}
    assert len(indices)==180 and not indices.intersection(range(180,228))
    # Read lineage metadata only, never prior labels. Ignore this exact batch's
    # already-frozen main output when executing its isolated deterministic replay.
    current_indices=set()
    metadata_paths=[]
    for directory in ("semantic_parents","candidate_renderings"):
        for path in sorted((ROOT/"data/RGM-001"/directory).glob("*.jsonl")):
            if path.name=="planning-gap-v1.jsonl":continue
            metadata_paths.append(str(path.relative_to(ROOT)))
            for line in path.read_text().splitlines():
                item=json.loads(line)
                if "source_family_index" in item:current_indices.add(item["source_family_index"])
                elif "source_family_key" in item:current_indices.add(int(item["source_family_key"].rsplit('-p',1)[1]))
    assert not current_indices.intersection(range(180,228)),"Reserved lineage is already occupied; no overwrite"
    unique={digest(row["semantic_task"]):row["semantic_task"] for row in rows}
    graphs=[task for task in unique.values() if task["kind"]=="graph"]
    blocks=[task for task in unique.values() if task["kind"]=="blocks"]
    return dict(occupied_source_indices=sorted(indices),current_metadata_source_indices=sorted(current_indices),metadata_paths=metadata_paths,
                active_families=len(indices),graph_weighted={graph_signature(task) for task in graphs},
                graph_unweighted={graph_signature(task,False,False) for task in graphs},blocks_joint={block_signature(task) for task in blocks})


def construct():
    prior=prior_signatures();families=[];weighted=set();unweighted=set();blocks=set();sixshapes=set();collisions=Counter()
    for index in range(180,228):
        stratum="ABCD"[(index-180)//12];ordinal=(index-180)%12;split=split_for(index)
        rng=random.Random(SEED+index*2027)
        for retry in range(300):
            tasks,construction=graph_family(index,ordinal,stratum,rng) if stratum in "AB" else block_family(index,ordinal,stratum,rng)
            base=tasks[0]
            signature=graph_signature(base) if base["kind"]=="graph" else block_signature(base)
            collision=signature in (prior["graph_weighted"]|weighted if base["kind"]=="graph" else prior["blocks_joint"]|blocks)
            if collision:collisions[stratum]+=1;continue
            if base["kind"]=="graph":weighted.add(signature);unweighted.add(graph_signature(base,False,False))
            else:
                blocks.add(signature)
                if len(base["on"])==6:sixshapes.add(shape(block_encoding(base)[1]))
            targets=[reference_target(task) for task in tasks]
            analyzed=[row_analysis(task,target) for task,target in zip(tasks,targets)]
            assert all(len(task["plan"])<=12 for task in tasks)
            assert all(metrics["search"]["complete"] for _,metrics in analyzed)
            families.append(dict(index=index,source_family_key=f"RGM-001-seed{SEED}-p{index:04d}",stratum=stratum,split=split,
                tasks=tasks,targets=targets,analyzed=analyzed,base_isomorphism_signature=signature,construction=construction));break
        else:raise RuntimeError(f"Novelty failed index{index}; no gate weakened")
    novel_unweighted=unweighted-prior["graph_unweighted"]
    assert len(weighted)==len(blocks)==24 and len(novel_unweighted)>=16 and len(sixshapes)>=8
    assert sum("required_temporary_goal_displacement" in family["analyzed"][0][0] for family in families if family["stratum"]=="C")>=4
    assert len({family["analyzed"][1][1]["first_error_index"] for family in families if family["stratum"]=="C"})>=3
    return families,dict(prior_active_families=prior["active_families"],reserved_indices_available=True,
        current_metadata_source_indices=prior["current_metadata_source_indices"],metadata_only_availability_paths=prior["metadata_paths"],new_weighted_graph_terminal_classes=len(weighted),
        new_unweighted_graph_topology_classes=len(novel_unweighted),new_joint_block_initial_goal_classes=len(blocks),distinct_six_block_initial_shapes=[list(s) for s in sorted(sixshapes)],
        collision_rejections=dict(collisions),four_strata_source_counts=dict(Counter(family["stratum"] for family in families)))


def manifest_witness(name,rich,metrics):
    """Use the prospective shared public scalar/locator vocabulary only."""
    if name=="legal_prefix_then_first_failure":return dict(first_error_index=metrics["first_error_index"])
    if name=="fuel_blocks_topologically_reachable_goal":return dict(topology_distance=metrics["topology_minimum_moves"],cheapest_fuel=metrics["minimum_required_fuel"])
    if name=="fuel_lengthens_fresh_minimum_moves":return dict(minimum_moves=metrics["fresh_plan_length"],topology_distance=metrics["topology_minimum_moves"])
    if name=="later_action_requires_changed_block_clearance":return dict(action_index=rich["action_index"],precondition=rich["clearance_precondition"],changed_by_action_index=rich["preceding_clearance_change_index"])
    if name=="partial_block_goal":return dict(goal_relation_count=rich["goal_relation_count"],block_count=rich["block_count"])
    return dict(minimum_moves=metrics["fresh_plan_length"])


def build(batch,families,novelty):
    rows,parents,manifest_families,measurements=[],[],[],[]
    for family in families:
        index=family["index"];stratum=family["stratum"];contexts=[];sample_ids=[]
        for number,(task,target,(properties,metrics)) in enumerate(zip(family["tasks"],family["targets"],family["analyzed"])):
            context=f"{stratum}{number}";ids_for_context=[];witnesses=[]
            for style in ("forward","reverse"):
                prompt="Executable task contract:\n"+(GRAPH_CONTRACT if task["kind"]=="graph" else BLOCK_CONTRACT)+"Task JSON:\n"+json.dumps(task if style=="forward" else reorder(task),ensure_ascii=False,separators=(",",":"))
                row=record("RGM-001",batch,index,context,task,prompt,target,["finite_synthetic_proxy","relational_planning"]+sorted(properties),metrics["structural_difficulty"],"structured",style)
                row.update(generator_version=VERSION,source_family_key=family["source_family_key"],source_family_index=index,primary_stratum=stratum,controlled_context=context,
                           structural_difficulty=metrics["structural_difficulty"],producer_agent_id=ACTOR,producer_role="relational_planning_producer")
                row["record_hash"]=record_hash(row);rows.append(row);ids_for_context.append(row["sample_id"])
                witnesses += [dict(sample_id=row["sample_id"],property=name,scope="row",evidence=manifest_witness(name,evidence,metrics)) for name,evidence in sorted(properties.items())]
                measurements.append(dict(sample_id=row["sample_id"],source_family_key=family["source_family_key"],stratum=stratum,context=context,split=family["split"],renderer=style,row_property_details=properties,**metrics))
            contexts.append(dict(context_id=context,sample_ids=ids_for_context,row_properties=sorted(properties),row_witnesses=witnesses));sample_ids.append(ids_for_context)
        changes={"A":("fuel_only_counterfactual","fuel"),"B":("directed_edge_deletion_counterfactual","edges"),"C":("adjacent_action_order_counterfactual","plan"),"D":("partial_goal_extension_counterfactual","goal_on")}
        name,field=changes[stratum]
        actual=[key for key in family["tasks"][0] if not equal(family["tasks"][0][key],family["tasks"][1][key])]
        assert actual==[field]
        controls=[dict(control_id=f"{index}-matched",name=name,scope="corpus_variant",sample_ids=[sample_ids[0][0],sample_ids[1][0]],changed_task_fields=[field],provenance="supplied_tasks")]
        if stratum in "BCD":
            left,right=(0,3) if stratum=="B" else (0,2) if stratum=="C" else (2,3)
            assert [key for key in family["tasks"][left] if not equal(family["tasks"][left][key],family["tasks"][right][key])]==["plan"]
            controls.append(dict(control_id=f"{index}-detour",name="same_world_valid_goal_detour",scope="corpus_variant",sample_ids=[sample_ids[left][0],sample_ids[right][0]],changed_task_fields=["plan"],provenance="supplied_tasks"))
        manifest_families.append(dict(source_family_key=family["source_family_key"],source_index=index,split=family["split"],primary_stratum=stratum,
            base_sample_id=sample_ids[0][0],contexts=contexts,controls=controls))
        parent=f"RGM-001-{batch}-p{index:04d}"
        parents.append(dict(semantic_parent_id=parent,semantic_family_id=parent,source_family_key=family["source_family_key"],source_family_index=index,
            split=family["split"],primary_stratum=stratum,tasks=family["tasks"],contexts=4,renderers=2,source_family_count=1,
            base_isomorphism_signature=family["base_isomorphism_signature"],construction=family["construction"]))
    return rows,parents,dict(version="rgm-planning-controls-1.0.0",track_id="RGM-001",batch_name=batch,semantics_sha256=PUBLIC["schemas/SEMANTICS.md"],families=manifest_families),measurements


def rename_value(value,mapping):
    if isinstance(value,dict):return {mapping.get(key,key):rename_value(item,mapping) for key,item in value.items()}
    if isinstance(value,list):return [rename_value(item,mapping) for item in value]
    return mapping.get(value,value) if isinstance(value,str) else value


def producer_probes(families):
    probes=[];control_results=[];lex_changes=0
    def completed(name,task,probe,comparison,details):
        assert comparison,name
        probes.append(dict(scope="audit_probe",name=name,actual_invocation=True,completed=True,input_sha256=digest(task),
                           probe_input_sha256=digest(probe),comparison_passed=comparison,details=details))
    for family in families:
        tasks=family["tasks"];targets=family["targets"];stratum=family["stratum"];metrics=[m for _,m in family["analyzed"]]
        assert targets[0]["valid"] and targets[0]["reaches_goal"]
        if stratum=="A":
            assert equal(tasks[0]["plan"],tasks[1]["plan"]) and not targets[1]["valid"]
            assert targets[2]["valid"] and not targets[2]["reaches_goal"] and len(tasks[2]["plan"])==len(tasks[3]["plan"])
            assert metrics[3]["first_error_index"]>=1 and metrics[3]["first_error_reason"]=="missing_directed_edge"
            if metrics[1]["fresh_plan_feasible"]:
                assert metrics[1]["fresh_plan_length"]>metrics[0]["fresh_plan_length"]
            else:assert tasks[0]["fuel"]==metrics[0]["minimum_required_fuel"]==tasks[1]["fuel"]+1
        elif stratum=="B":
            assert len(tasks[0]["plan"])>=3 and metrics[1]["fresh_plan_feasible"]
            removed=[e for e in tasks[0]["edges"] if e not in tasks[1]["edges"]]
            assert len(removed)==1 and len(tasks[0]["edges"])-len(tasks[1]["edges"])==1 and metrics[1]["first_error_index"]>=1
            a=tasks[0]["start"];b=tasks[2]["plan"][0];edges={tuple(e[:2]) for e in tasks[0]["edges"]}
            assert (a,b) in edges and (b,a) not in edges and tasks[2]["plan"]==[b,a] and metrics[2]["first_error_index"]==1
        elif stratum=="C":
            assert 4<=metrics[0]["fresh_plan_length"]<=8 and len(tasks[0]["plan"])==metrics[0]["fresh_plan_length"]
            assert "later_action_requires_changed_block_clearance" in family["analyzed"][0][0]
            differences=[i for i,(a,b) in enumerate(zip(tasks[0]["plan"],tasks[1]["plan"])) if a!=b]
            assert len(tasks[0]["plan"])==len(tasks[1]["plan"]) and len(differences)==2 and differences[1]==differences[0]+1
            left,right=differences;assert tasks[0]["plan"][left]==tasks[1]["plan"][right] and tasks[0]["plan"][right]==tasks[1]["plan"][left]
            assert metrics[1]["first_error_reason"] in ("source_not_clear","destination_not_clear")
            assert targets[3]["valid"] and not targets[3]["reaches_goal"] and tasks[3]["plan"]
        else:
            assert 0<len(tasks[0]["goal_on"])<len(tasks[0]["on"]) and metrics[0]["fresh_plan_length"]>=2
            assert all(tasks[1]["goal_on"][key]==value for key,value in tasks[0]["goal_on"].items())
            assert len(tasks[1]["goal_on"])>len(tasks[0]["goal_on"]) and targets[1]["valid"] and not targets[1]["reaches_goal"]
            assert metrics[1]["fresh_plan_length"]>metrics[0]["fresh_plan_length"] and len(tasks[2]["plan"])==metrics[1]["fresh_plan_length"]
        if stratum in "BCD":
            detour=3 if stratum in "BD" else 2
            assert targets[detour]["valid"] and targets[detour]["reaches_goal"] and len(tasks[detour]["plan"])>metrics[detour]["fresh_plan_length"]
        control_results.append(dict(source_family_key=family["source_family_key"],primary_stratum=stratum,all_required_contexts_checked=True,passed=True))
        for task,target in zip(tasks,targets):
            if not target["valid"]:
                suffix=[task["start"]] if task["kind"]=="graph" else [copy.deepcopy(task["plan"][0])]
                changed={**task,"plan":task["plan"]+suffix};out=reference_target(changed)
                completed("stop_at_first_illegal_suffix_absorption",task,changed,equal(target,out),dict(evaluated_steps=len(out["steps"]),submitted_steps=len(changed["plan"])))
            changed=reorder(copy.deepcopy(task))
            if task["kind"]=="graph":changed["nodes"]=list(reversed(changed["nodes"]));changed["edges"]=list(reversed(changed["edges"]))
            completed("object_list_and_key_order_invariance",task,changed,equal(reference_target(changed),target),dict(reversed_nodes_and_edges=task["kind"]=="graph",reversed_object_keys=True))
            names=sorted(task["nodes"] if task["kind"]=="graph" else task["on"]);mapping=dict(zip(names,reversed(names)));changed=rename_value(task,mapping)
            out=reference_target(changed);replay=rename_value(target,mapping)
            replay_equal=all(equal(out[key],replay[key]) for key in ("steps","after","valid","reaches_goal"))
            minimum_equal=(out["shortest_plan"] is None)==(target["shortest_plan"] is None) and (out["shortest_plan"] is None or len(out["shortest_plan"])==len(target["shortest_plan"]))
            lex_search=graph_search(changed) if task["kind"]=="graph" else block_search(changed)
            different=not equal(out["shortest_plan"],replay["shortest_plan"])
            if task is tasks[0] and stratum=="B" and different:lex_changes+=1
            completed("name_permutation_replay_minimum_and_full_sequence_lex",task,changed,replay_equal and minimum_equal and equal(lex_search["plan"],out["shortest_plan"]),
                      dict(replay_equivariant=replay_equal,minimum_length_invariant=minimum_equal,canonical_plan_changed_beyond_direct_renaming=different,search=lex_search))
    assert sum(f["analyzed"][0][1]["search"].get("minimum_route_count",0)>=2 for f in families if f["stratum"]=="B")>=6
    assert lex_changes>=1,"Renaming must actually exercise a lexical tie change"
    return dict(scope="audit_probe",does_not_add_families_or_replace_supplied_controls=True,probes=probes,
                actual_probe_count=len(probes),lexical_tie_changed_bases=lex_changes,required_context_regressions=control_results)


def feature_lookup(examples):
    all_table=defaultdict(Counter);train=defaultdict(Counter);dev=[];global_train=Counter()
    for feature,outcome,split in examples:
        all_table[feature][outcome]+=1
        if split=="train":train[feature][outcome]+=1;global_train[outcome]+=1
        else:dev.append((feature,outcome))
    majority=lambda counts:min(counts,key=lambda value:(-counts[value],value))
    fallback=majority(global_train)
    transfer=sum((majority(train[key]) if key in train else fallback)==value for key,value in dev)
    train_count=sum(sum(counts.values()) for counts in train.values())
    unknown=[key for key,_ in dev if key not in train]
    return dict(prediction="[valid,reaches_goal,fresh_plan_feasible]",all_rows=len(examples),distinct_all_buckets=len(all_table),distinct_train_buckets=len(train),
                in_sample_accuracy=sum(max(counts.values()) for counts in all_table.values())/len(examples),
                train_fit_accuracy=sum(max(counts.values()) for counts in train.values())/train_count,
                train_to_exposed_dev_accuracy=transfer/len(dev),dev_rows=len(dev),unknown_dev_rows=len(unknown),unknown_dev_buckets=len(set(unknown)),
                unknown_fallback=json.loads(fallback),fallback_rule="train global majority; canonical lexical tie",train_label_histogram=dict(sorted(global_train.items())),
                measured_exposed_split_transfer_only=True,fit_does_not_establish_model_or_sealed_generalization=True)


def diagnostics(rows,measurements,novelty,probes):
    hist=defaultdict(lambda:defaultdict(Counter));examples=defaultdict(list);property_counts=Counter()
    for row,metrics in zip(rows,measurements):
        outcome=canonical([metrics["valid"],metrics["reaches_goal"],metrics["fresh_plan_feasible"]])
        for group in ("overall",f"family:{row['source_family_key']}",f"stratum:{row['primary_stratum']}",f"context:{row['controlled_context']}",
                      f"split:{row['split']}",f"renderer:{row['renderer']}",f"stratum_split:{row['primary_stratum']}:{row['split']}",
                      f"stratum_split_renderer:{row['primary_stratum']}:{row['split']}:{row['renderer']}"):
            for field in ("valid","reaches_goal","fresh_plan_feasible","fresh_plan_length","first_error_index","first_error_reason","object_count","edge_count",
                          "submitted_plan_length","initial_fuel","structural_difficulty"):
                hist[group][field][canonical(metrics[field])]+=1
        property_counts.update(metrics["row_property_details"].keys())
        task=row["semantic_task"];tokens=Counter(re.findall(r"[A-Za-z_]+|-?\d+",row["rendered_input"].split("Task JSON:\n")[1]))
        shaped=Counter("ENTITY" if word in WORDS else word for word,count in tokens.items() for _ in range(count))
        features=dict(object_size=[task["kind"],metrics["object_count"]],submitted_length=[task["kind"],metrics["submitted_plan_length"]],
                      fuel=[task["kind"],metrics["initial_fuel"]],edge_count=[task["kind"],metrics["edge_count"]],
                      size_length_fuel_edges=[task["kind"],metrics["object_count"],metrics["edge_count"],metrics["submitted_plan_length"],metrics["initial_fuel"]],
                      token_bag=sorted(tokens.items()),token_bag_without_entity_identity=sorted(shaped.items()))
        for name,feature in features.items():examples[name].append((canonical(feature),outcome,row["split"]))
    baselines={name:feature_lookup(examples[name]) for name in sorted(examples)}
    selected=min(baselines,key=lambda name:(-baselines[name]["train_fit_accuracy"],name))
    unique_metrics=[m for m in measurements if m["renderer"]=="forward"]
    return dict(novelty=novelty,actual_rows=len(rows),actual_source_families=48,actual_splits=dict(Counter(row["split"] for row in rows)),
        source_split_counts=dict(Counter(row["split"] for row in rows if row["controlled_context"].endswith("0") and row["renderer"]=="forward")),
        conditional_histograms={group:{field:dict(sorted(counts.items())) for field,counts in sorted(fields.items())} for group,fields in sorted(hist.items())},
        actual_public_row_property_counts=dict(sorted(property_counts.items())),measurements=measurements,producer_audit_probes=probes,
        input_only_scalar_and_token_lookups=baselines,highest_train_fit_lookup=dict(name=selected,**baselines[selected]),
        finite_search_costs=dict(sorted(SEARCH_COSTS.items())),supplied_unique_context_searches=dict(actual_contexts=len(unique_metrics),
            all_completed=all(m["search"]["complete"] for m in unique_metrics),maximum_visited_states=max(m["search"]["visited_states"] for m in unique_metrics),
            maximum_frontier=max(m["search"]["max_frontier"] for m in unique_metrics)),
        limitations=["Two existing action grammars,48 new state/world families; contexts/renderers are correlated.",
            "Graph dev has only2 families per stratum; three sizes occur across both graph strata, not within each dev stratum.",
            "Context/cohort outcomes remain associated with submitted length, fuel, edge and token signatures; actual conditional histograms and fitted lookup results are reported.",
            "The strongest train-fit lookup is selected using train fit only; exposed-dev transfer includes its explicit majority fallback for unknown buckets.",
            "No learning efficacy, sealed OOD, edit repair or logistics primitive is established."])


def output_paths(output,batch):
    return [output/f"data/RGM-001/candidate_renderings/{batch}.jsonl",output/f"data/RGM-001/semantic_parents/{batch}.jsonl",output/f"manifests/{batch}_SOURCE.py",
            output/f"data/RGM-001/manifests/{batch}.family_controls.json",output/f"data/RGM-001/manifests/{batch}.producer_integrity.json"]


def generate(batch="planning-gap-v1",output=ROOT,worker=False):
    output=Path(output).resolve();started=time.perf_counter();started_cpu=time.process_time();paths=output_paths(output,batch);candidate,parent,snapshot,control_path,integrity=paths
    generation=output/f"manifests/{batch}_GENERATION.json"
    for path in paths+[generation]:
        if path.exists():raise ValueError(f"First run preserved at{path}; use a new batch")
    assert {name:sha(ROOT/name) for name in PUBLIC}==PUBLIC and sha(ROOT/FREEZE)==FREEZE_SHA
    source=Path(__file__).read_text();check_space(snapshot,"source_snapshot");atomic(snapshot,source)
    rows,parents=[],[];failure=None;status="PRODUCER_CHECK_FAILED";reproduction=None
    try:
        families,novelty=construct();rows,parents,controls,measurements=build(batch,families,novelty)
        probes=producer_probes(families);integrity_report=diagnostics(rows,measurements,novelty,probes)
        assert len(rows)==384 and len(parents)==48
        for row in rows:
            assert row["record_hash"]==record_hash(row) and row["review_status"]=="PENDING" and row["split"]==split_for(row["source_family_index"])
            assert equal(json.loads(row["rendered_input"].split("Task JSON:\n")[1]),row["semantic_task"])
            assert row["sample_id"] not in row["rendered_input"] and row["source_family_key"] not in row["rendered_input"]
        check_space(parent,"semantic_parent_output");write_jsonl(parent,parents)
        check_space(candidate,"candidate_output");write_jsonl(candidate,rows)
        write_json(control_path,controls);write_json(integrity,integrity_report)
        print("FROZEN",sha(candidate),len(rows),"CORE_BYTES",sum(path.stat().st_size for path in paths),flush=True)
        if not worker:
            scratch=output/f"reproducibility/{batch}/hashseed-971"
            assert not scratch.exists()
            environment={**os.environ,"PYTHONHASHSEED":"971"}
            check_space(output,"before_isolated_replay")
            command=["/opt/anaconda3/bin/python3","-m","generators.relational.planning_gap","--batch",batch,"--output-root",str(scratch),"--worker"]
            child=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
            matched=[dict(path=str(path.relative_to(output)),main_sha256=sha(path),replay_sha256=sha(scratch/path.relative_to(output)) if (scratch/path.relative_to(output)).exists() else None) for path in paths]
            passed=child.returncode==0 and all(item["main_sha256"]==item["replay_sha256"] for item in matched)
            reproduction=output/f"manifests/{batch}_REPRODUCTION.json"
            worker_manifest=scratch/f"manifests/{batch}_GENERATION.json"
            write_json(reproduction,dict(actual_execution_id=ACTOR,hash_seeds=[os.environ.get("PYTHONHASHSEED","process_default"),"971"],
                separate_processes=True,deterministic_artifacts=matched,passed=passed,stdout=child.stdout,stderr=child.stderr,
                excluded_measured_walltime_manifest=True,worker_run_manifest=json.loads(worker_manifest.read_text()) if worker_manifest.exists() else None))
            assert sum(path.stat().st_size for path in scratch.rglob('*') if path.is_file())<=150*1024*1024
            assert passed,"First failed replay preserved"
            shutil.rmtree(scratch) # Own ordinary successful scratch only; all evidence retained in receipt.
        status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW"
    except Exception:
        failure=traceback.format_exc()
        if not parent.exists():write_jsonl(parent,parents)
        if not candidate.exists():write_jsonl(candidate,rows)
        raise
    finally:
        artifacts=[path for path in paths+([reproduction] if reproduction else []) if path.exists()]
        write_json(generation,dict(batch=batch,track_id="RGM-001",generator_version=VERSION,status=status,actual_execution_id=ACTOR,actual_role="relational_planning_producer",
            public_source_sha256=PUBLIC,root_freeze_sha256=FREEZE_SHA,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            dependency_sha256={name:sha(ROOT/name) for name in ["factory/common.py","factory/reference.py","generators/relational/generate.py"]},
            seed=SEED,source_indices=list(range(180,228)),actual_counts=dict(rows=len(rows),source_families=len(parents),unique_reference_shortest_searches=len(REFERENCE_CACHE)),
            measured_wall_seconds=time.perf_counter()-started,measured_cpu_seconds=time.process_time()-started_cpu,
            actual_reference_invocations=REFERENCE_INVOCATIONS,finite_search_costs=dict(sorted(SEARCH_COSTS.items())),
            actual_python_executable=sys.executable,actual_python_version=sys.version,actual_process_id=os.getpid(),disk_space_observations=SPACE_OBSERVATIONS,
            artifact_bytes=sum(path.stat().st_size for path in artifacts),max_scratch_bytes=150*1024*1024,
            artifact_sha256={str(path.relative_to(output)):sha(path) for path in artifacts},initial_failure=failure,worker_reproduction=worker,
            reproduction="/opt/anaconda3/bin/python3 -m generators.relational.planning_gap --batch planning-gap-v1 (fresh output root)",no_approval_release_or_push=True))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--batch",default="planning-gap-v1");parser.add_argument("--output-root",default=str(ROOT));parser.add_argument("--worker",action="store_true");parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        families,novelty=construct();rows,parents,controls,measurements=build(args.batch,families,novelty)
        probes=producer_probes(families);report=diagnostics(rows,measurements,novelty,probes)
        print(canonical(dict(actual_families=len(families),actual_rows=len(rows),novelty=novelty,source_split_counts=report["source_split_counts"],
            actual_probe_count=probes["actual_probe_count"],public_property_counts=report["actual_public_row_property_counts"],
            highest_train_fit_lookup=report["highest_train_fit_lookup"],prospective_core_bytes=sum(len(canonical(item).encode())+1 for item in rows+parents)+len(canonical(controls).encode())+len(canonical(report).encode())+Path(__file__).stat().st_size)))
    else:print("Produced",len(generate(args.batch,args.output_root,args.worker)),"PENDING rows",flush=True)
