"""Structural diversity beyond pilot towers and chain graphs; generator only."""
from factory.reference import solve
import collections


def stack_state(names,rng):
    order=list(names);rng.shuffle(order)
    on={order[0]:"TABLE"}
    for previous,block in zip(order,order[1:]):
        on[block]=previous if rng.random()<0.6 else "TABLE"
    return on


def diversify(task,index,rng):
    if task["kind"]=="graph":
        edges={(a,b):c for a,b,c in task["edges"]}
        for a in task["nodes"]:
            for b in task["nodes"]:
                if a!=b and (a,b) not in edges and rng.random()<0.18:edges[a,b]=rng.randint(1,6)
        task["edges"]=[[a,b,c] for (a,b),c in edges.items()]
        successors=collections.Counter(a for a,b,c in task['edges'])
        return task,"branched_resource_graph" if any(n>1 for n in successors.values()) else "single_successor_resource_graph"
    names=list(task["on"])
    task["on"]=stack_state(names,rng)
    goal=stack_state(names,rng)
    if index%3==0:
        keys=rng.sample(names,max(1,len(names)-1));goal={k:goal[k] for k in keys}
    task["goal_on"]=goal;task["plan"]=[]
    optimal=solve(task)["shortest_plan"]
    mode=index%4
    if mode==0:task["plan"]=optimal
    elif mode==1:task["plan"]=optimal[:max(1,len(optimal)//2)]
    elif mode==2:task["plan"]=[{"block":min(names),"to":task["on"][min(names)]}]
    else:
        after=solve({**task,"plan":optimal})["after"]
        task["plan"]=optimal+[{"block":min(names),"to":after[min(names)]}]
    initial_roots=sum(support=='TABLE' for support in task['on'].values())
    explicit_goal_roots=sum(support=='TABLE' for support in task['goal_on'].values())
    if initial_roots>=2:mechanism='multiple_initial_table_stacks'
    elif explicit_goal_roots>=2:mechanism='multiple_explicit_goal_table_roots'
    else:mechanism='block_goal_composition'
    return task,mechanism
