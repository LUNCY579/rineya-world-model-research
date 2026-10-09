"""Generator-side reference computation, separate from independent oracle."""
from collections import deque


def evaluate(expr, state, event):
    if "const" in expr:
        return expr["const"]
    if "var" in expr:
        owner, name = expr["var"].split(".")
        return (state if owner == "state" else event)[name]
    args = [evaluate(a, state, event) for a in expr["args"]]
    op = expr["op"]
    if op == "all": return all(args)
    if op == "any": return any(args)
    if op == "not": return not args[0]
    a, b = args
    return {"eq": lambda: type(a) is type(b) and a == b, "ne": lambda: type(a) is not type(b) or a != b,
            "ge": lambda: a >= b, "gt": lambda: a > b,
            "le": lambda: a <= b, "lt": lambda: a < b}[op]()


def causal(task):
    state, steps = dict(task["state"]), []
    for e in task["events"]:
        rule = evaluate(task["rule"], state, e)
        pre = evaluate(task["precondition"], state, e)
        op, x, y, a = e["op"], e["source"], e["destination"], e["amount"]
        if rule and pre:
            if op == "add": state[x] += a
            elif op == "scale": state[x] *= a
            elif op == "transfer": state[x], state[y] = state[x] - a, state[y] + a
            elif op == "swap": state[x], state[y] = state[y], state[x]
        steps.append({"cmp": {"rule": rule, "precondition": pre},
                      "calc": {"operation": op if rule and pre else "identity"}, "after": dict(state)})
    return {"after": state, "steps": steps}


def graph_run(t, plan=None):
    plan = t["plan"] if plan is None else plan
    edge = {(a,b): c for a,b,c in t["edges"]}
    state, steps, valid = {"node": t["start"], "fuel": t["fuel"]}, [], True
    for dest in plan:
        cost = edge.get((state["node"], dest))
        legal = cost is not None and cost <= state["fuel"]
        if legal: state = {"node": dest, "fuel": state["fuel"] - cost}
        steps.append({"legal": legal, "after": dict(state)})
        if not legal:
            valid = False
            break
    return {"steps": steps, "after": state, "valid": valid,
            "reaches_goal": valid and state["node"] == t["goal"]}


def shortest_graph(t, state=None):
    s = state or {"node": t["start"], "fuel": t["fuel"]}
    queue = deque([(s["node"], s["fuel"], [])])
    seen = set()
    while queue:
        node, fuel, path = queue.popleft()
        if node == t["goal"]: return path
        if (node,fuel) in seen: continue
        seen.add((node,fuel))
        for a,b,c in sorted(t["edges"]):
            if a == node and c <= fuel: queue.append((b, fuel-c, path+[b]))
    return None


def blocks_move(on, action):
    x, dest = action["block"], action["to"]
    clear = set(on) - set(on.values())
    legal = x in clear and (dest == "TABLE" or dest in clear) and x != dest and on.get(x) != dest
    after = dict(on)
    if legal: after[x] = dest
    return legal, after


def shortest_blocks(t):
    queue = deque([(dict(t["on"]),[])])
    seen = set()
    while queue:
        on,path = queue.popleft()
        key = tuple(sorted(on.items()))
        if key in seen: continue
        seen.add(key)
        if all(on[b]==d for b,d in t["goal_on"].items()): return path
        for block in sorted(on):
            for dest in sorted(["TABLE"]+list(on)):
                a={"block":block,"to":dest}
                legal,after=blocks_move(on,a)
                if legal: queue.append((after,path+[a]))
    return None


def blocks(t):
    state,steps,valid=dict(t["on"]),[],True
    for a in t["plan"]:
        legal,state=blocks_move(state,a)
        steps.append({"legal":legal,"after":dict(state)})
        if not legal:
            valid=False
            break
    return {"steps":steps,"after":state,"valid":valid,
            "reaches_goal":valid and all(state[b]==d for b,d in t["goal_on"].items()),
            "shortest_plan":shortest_blocks(t)}


def critic(t):
    w=t["world"]
    optimum=shortest_graph(w)
    base=None if optimum is None else len(optimum)
    evaluations=[]
    keys={}
    for c in t["candidates"]:
        run=graph_run(w,c["plan"])
        tail=shortest_graph(w,run["after"])
        distance=None if tail is None else len(tail)
        progress=base-distance if run["valid"] and distance is not None and base is not None else None
        utility=-1 if not run["valid"] else 2 if run["reaches_goal"] else 1 if progress is not None and progress>0 else 0
        claim=c["claimed_after"]==run["after"]
        error=next((i for i,s in enumerate(run["steps"]) if not s["legal"]),None)
        if error is None and not claim:error="claim"
        cid=c["candidate_id"]
        evaluations.append(dict(candidate_id=cid,valid=run["valid"],claim_correct=claim,
                                reaches_goal=run["reaches_goal"],remaining_moves=distance,
                                progress=progress,usefulness=utility,first_error=error,repair_plan=optimum))
        keys[cid]=(run["valid"] and claim,run["reaches_goal"],utility,
                   -distance if distance is not None else float('-inf'),-len(c["plan"]))
    groups=[]
    for key in sorted(set(keys.values()),reverse=True):
        groups.append(sorted(cid for cid,k in keys.items() if k==key))
    return {"evaluations":evaluations,"ranking":groups}


def solve(t):
    if t["kind"]=="causal":return causal(t)
    if t["kind"]=="graph":return {**graph_run(t),"shortest_plan":shortest_graph(t)}
    if t["kind"]=="blocks":return blocks(t)
    if t["kind"]=="critic":return critic(t)
    raise ValueError(t["kind"])
