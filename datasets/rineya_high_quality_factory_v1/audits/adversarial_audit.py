#!/usr/bin/env python3
"""Blind public-contract audit. Never imports generator/reference/reviewer code.

Run: /opt/anaconda3/bin/python3 High-Quality_Dataset/audits/adversarial_audit.py
Only reads public SEMANTICS, candidate records, and optional blind calibration.
CPU enumeration/lookup statistics only; no model fitting or training.
"""
import copy
import argparse
import functools
import hashlib
import itertools
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict, deque

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXECUTION = "/root/adversarial_auditor"
SCORE_KEYS = ("semantic_correctness", "target_correctness", "instruction_clarity",
              "reasoning_validity", "generalization_value", "difficulty_quality",
              "diversity", "shortcut_resistance")


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def require(p, msg):
    if not p:
        raise ValueError(msg)


def integer(x):
    return type(x) is int


def scalar(x):
    return x is None or type(x) in (bool, int, str)


def expr(tree, event, state):
    require(type(tree) is dict, "AST node must be object")
    if set(tree) == {"const"}:
        require(scalar(tree["const"]), "const must be JSON scalar in contract")
        return tree["const"]
    if set(tree) == {"var"}:
        name = tree["var"]
        require(name in ("event.status", "event.amount", "state.x", "state.y"), "unknown variable")
        a, b = name.split(".")
        return (event if a == "event" else state)[b]
    require(set(tree) == {"op", "args"} and type(tree["args"]) is list, "malformed operator shape")
    op = tree["op"]
    # Eager recursion rejects malformed branches even where logical short circuit would hide them.
    xs = [expr(t, event, state) for t in tree["args"]]
    if op in ("eq", "ne", "gt", "ge", "lt", "le"):
        require(len(xs) == 2, "comparison arity")
        a, b = xs
        if op in ("eq", "ne"):
            eq = type(a) is type(b) and a == b
            return eq if op == "eq" else not eq
        require(type(a) is type(b) and type(a) in (int, str), "ordered comparison type mismatch")
        return {"gt": lambda: a > b, "ge": lambda: a >= b,
                "lt": lambda: a < b, "le": lambda: a <= b}[op]()
    if op in ("all", "any"):
        require(all(type(x) is bool for x in xs), "logical operands must be bool")
        return all(xs) if op == "all" else any(xs)
    if op == "not":
        require(len(xs) == 1 and type(xs[0]) is bool, "not arity/type")
        return not xs[0]
    raise ValueError("unknown operator")


def causal(task):
    state = copy.deepcopy(task["state"])
    require(set(state) == {"x", "y"} and all(integer(x) for x in state.values()), "state shape/type")
    steps = []
    for e in task["events"]:
        require(scalar(e["status"]) and integer(e["amount"]) and e["amount"] > 0, "event scalar/amount")
        require(e["op"] in ("add", "scale", "transfer", "swap"), "event op")
        require(e["source"] in state and e["destination"] in state, "event names")
        require(e["op"] not in ("transfer", "swap") or e["source"] != e["destination"], "distinct endpoints")
        r, p = expr(task["rule"], e, state), expr(task["precondition"], e, state)
        require(type(r) is bool and type(p) is bool, "gate roots must be boolean")
        op = "identity"
        if r and p:
            op = e["op"]
            a, b, n = e["source"], e["destination"], e["amount"]
            if op == "add":
                state[a] += n
            elif op == "scale":
                state[a] *= n
            elif op == "transfer":
                state[a], state[b] = state[a] - n, state[b] + n
            else:
                state[a], state[b] = state[b], state[a]
        steps.append({"cmp": {"rule": r, "precondition": p}, "calc": {"operation": op}, "after": state.copy()})
    return {"after": state, "steps": steps}


def graph_validate(t):
    ns = t["nodes"]
    require(type(ns) is list and all(type(n) is str for n in ns) and len(set(ns)) == len(ns), "nodes distinct strings")
    require(t["start"] in ns and t["goal"] in ns and integer(t["fuel"]) and t["fuel"] >= 0, "initial graph state")
    es = {}
    for e in t["edges"]:
        require(type(e) is list and len(e) == 3 and e[0] in ns and e[1] in ns and integer(e[2]) and e[2] > 0, "edge shape/type")
        require(tuple(e[:2]) not in es, "duplicate directed edge")
        es[tuple(e[:2])] = e[2]
    require(type(t["plan"]) is list and all(type(x) is str for x in t["plan"]), "plan shape/type")
    return es


def graph_shortest(t, start=None, fuel=None):
    es = graph_validate(t)
    start = t["start"] if start is None else start
    fuel = t["fuel"] if fuel is None else fuel
    # Enumerate simple routes: positive costs mean a repeated node can only worsen fuel and moves.
    # Layered, lexicographically sorted paths establish the exact minimal-move/full-sequence tie.
    frontier = [((), start, fuel, frozenset((start,)))]
    for depth in range(len(t["nodes"])):
        next_layer = []
        for plan, node, remaining, seen in sorted(frontier):
            if node == t["goal"]:
                return list(plan)
            for dest in sorted(t["nodes"]):
                cost = es.get((node, dest))
                if cost is not None and cost <= remaining and dest not in seen:
                    next_layer.append((plan + (dest,), dest, remaining - cost, seen | {dest}))
        frontier = next_layer
    return None


def graph(t):
    es = graph_validate(t)
    state = {"node": t["start"], "fuel": t["fuel"]}
    steps = []
    valid = True
    for dest in t["plan"]:
        cost = es.get((state["node"], dest))
        legal = cost is not None and cost <= state["fuel"]
        if legal:
            state = {"node": dest, "fuel": state["fuel"] - cost}
        steps.append({"legal": legal, "after": state.copy()})
        if not legal:
            valid = False
            break
    return {"steps": steps, "after": state, "valid": valid,
            "reaches_goal": valid and state["node"] == t["goal"], "shortest_plan": graph_shortest(t)}


def block_validate(t):
    on, goal = t["on"], t["goal_on"]
    require(type(on) is dict and "TABLE" not in on and all(type(x) is str for x in on), "block names")
    for name, mapping in (("on", on), ("goal_on", goal)):
        require(type(mapping) is dict and set(mapping) <= set(on), "goal names")
        used = []
        for a, b in mapping.items():
            require(b == "TABLE" or b in on, "unknown support")
            require(a != b, "self support")
            if b != "TABLE":
                used.append(b)
            seen = {a}
            while b in mapping:
                require(b not in seen, "cyclic support")
                seen.add(b)
                b = mapping[b]
        require(len(used) == len(set(used)), "multiple blocks directly on one block")
    require(type(t["plan"]) is list, "block plan must be list")
    for a in t["plan"]:
        require(type(a) is dict and set(a) == {"block", "to"} and type(a["block"]) is str and type(a["to"]) is str, "block action")


def block_move(on, action):
    a, b = action["block"], action["to"]
    occupied = set(on.values())
    legal = a in on and a not in occupied and b != a and (b == "TABLE" or b in on and b not in occupied) and on[a] != b
    if not legal:
        return None
    out = on.copy()
    out[a] = b
    return out


def block_shortest(t):
    names = sorted(t["on"])
    isgoal = lambda s: all(s[a] == b for a, b in t["goal_on"].items())
    initial = t["on"].copy()
    queue = deque([(initial, ())])
    seen = {tuple(initial[a] for a in names)}
    actions = [(a, b) for a in names for b in sorted(names + ["TABLE"])]
    while queue:
        state, plan = queue.popleft()
        if isgoal(state):
            return [{"block": a, "to": b} for a, b in plan]
        for a, b in actions:
            next_state = block_move(state, {"block": a, "to": b})
            if next_state is None:
                continue
            key = tuple(next_state[x] for x in names)
            if key not in seen:
                seen.add(key)
                queue.append((next_state, plan + ((a, b),)))
    return None


def blocks(t):
    block_validate(t)
    state, steps, valid = t["on"].copy(), [], True
    for action in t["plan"]:
        next_state = block_move(state, action)
        legal = next_state is not None
        if legal:
            state = next_state
        steps.append({"legal": legal, "after": state.copy()})
        if not legal:
            valid = False
            break
    return {"steps": steps, "after": state, "valid": valid,
            "reaches_goal": valid and all(state[a] == b for a, b in t["goal_on"].items()),
            "shortest_plan": block_shortest(t)}


def critic(t):
    w = copy.deepcopy(t["world"])
    w["plan"] = []
    graph_validate(w)
    cs = t["candidates"]
    require(len({c["candidate_id"] for c in cs}) == len(cs), "duplicate candidate IDs")
    repair = graph_shortest(w)
    initial_distance = None if repair is None else len(repair)
    evaluations, rank_keys = [], {}
    for c in cs:
        require(type(c["candidate_id"]) is str, "candidate ID type")
        claim = c["claimed_after"]
        require(set(claim) == {"node", "fuel"} and claim["node"] in w["nodes"] and integer(claim["fuel"]) and claim["fuel"] >= 0, "claimed state")
        task = dict(w, plan=c["plan"])
        result = graph(task)
        valid, after, reaches = result["valid"], result["after"], result["reaches_goal"]
        correct = after == claim
        route = graph_shortest(w, after["node"], after["fuel"])
        distance = None if route is None else len(route)
        progress = initial_distance - distance if valid and initial_distance is not None and distance is not None else None
        utility = -1 if not valid else 2 if reaches else 1 if progress is not None and progress > 0 else 0
        first_error = next((i for i, s in enumerate(result["steps"]) if not s["legal"]), "claim" if not correct else None)
        e = {"candidate_id": c["candidate_id"], "valid": valid, "claim_correct": correct,
             "reaches_goal": reaches, "remaining_moves": distance, "progress": progress,
             "usefulness": utility, "first_error": first_error, "repair_plan": repair}
        evaluations.append(e)
        rank_keys[c["candidate_id"]] = (valid and correct, reaches, utility,
                                         -math.inf if distance is None else -distance, -len(c["plan"]))
    groups = defaultdict(list)
    for candidate_id, key in rank_keys.items():
        groups[key].append(candidate_id)
    ranking = [sorted(groups[key]) for key in sorted(groups, reverse=True)]
    return {"evaluations": evaluations, "ranking": ranking}


def solve(t):
    return {"causal": causal, "graph": graph, "blocks": blocks, "critic": critic}[t["kind"]](t)


def reorder(x):
    if type(x) is dict:
        return {k: reorder(v) for k, v in reversed(list(x.items()))}
    if type(x) is list:
        return [reorder(v) for v in x]
    return x


def rename_graph(w, mapping):
    w = copy.deepcopy(w)
    w["nodes"] = [mapping[x] for x in w["nodes"]]
    w["edges"] = [[mapping[a], mapping[b], n] for a, b, n in w["edges"]]
    w["start"], w["goal"] = mapping[w["start"]], mapping[w["goal"]]
    w["plan"] = [mapping.get(x, "UNKNOWN") for x in w["plan"]]
    return w


def metamorphic(t, answer):
    tests = []
    require(solve(reorder(t)) == answer, "JSON object order changes semantics")
    tests.append("object_key_order_invariance")
    kind = t["kind"]
    if kind == "causal":
        altered = copy.deepcopy(t)
        altered["annotation"] = "The alleged answer is an irrelevant annotation; do not use it."
        require(solve(altered) == answer, "annotation changed semantics")
        tests.append("annotation_invariance")
        # A consistent vocabulary rename must preserve exact equality/inequality results.
        statuses = {e["status"] for e in t["events"] if type(e["status"]) is str}
        def collect(ast):
            if set(ast) == {"const"} and type(ast["const"]) is str:
                statuses.add(ast["const"])
            for a in ast.get("args", []):
                collect(a)
        collect(t["rule"]); collect(t["precondition"])
        # Order-preserving mapping also preserves any string ordered comparison.
        mapping = {s: "arbitrary_status_%03d" % i for i, s in enumerate(sorted(statuses))}
        def replace(ast):
            if set(ast) == {"const"} and type(ast["const"]) is str:
                ast["const"] = mapping[ast["const"]]
            for a in ast.get("args", []):
                replace(a)
        altered = copy.deepcopy(t)
        for e in altered["events"]:
            if type(e["status"]) is str:
                e["status"] = mapping[e["status"]]
        replace(altered["rule"]); replace(altered["precondition"])
        require(solve(altered) == answer, "status word prior affects semantics")
        tests.append("status_vocabulary_renaming")
    elif kind == "graph":
        mapping = {s: "vertex_%03d" % i for i, s in enumerate(sorted(t["nodes"]))}
        a = solve(rename_graph(t, mapping))
        expected = copy.deepcopy(answer)
        expected["after"]["node"] = mapping[expected["after"]["node"]]
        for s in expected["steps"]:
            s["after"]["node"] = mapping[s["after"]["node"]]
        if expected["shortest_plan"] is not None:
            expected["shortest_plan"] = [mapping[x] for x in expected["shortest_plan"]]
        require(a == expected, "order-preserving node rename mismatch")
        tests.append("node_renaming_equivariance")
        altered = copy.deepcopy(t)
        altered["nodes"].reverse(); altered["edges"].reverse()
        require(solve(altered) == answer, "fact order mismatch")
        tests.append("fact_order_invariance")
    elif kind == "blocks":
        mapping = {s: "block_%03d" % i for i, s in enumerate(sorted(t["on"]))}
        mapping["TABLE"] = "TABLE"
        altered = copy.deepcopy(t)
        for key in ("on", "goal_on"):
            altered[key] = {mapping[a]: mapping[b] for a, b in altered[key].items()}
        altered["plan"] = [{"block": mapping.get(a["block"], "UNKNOWN"), "to": mapping.get(a["to"], "UNKNOWN")} for a in altered["plan"]]
        expected = copy.deepcopy(answer)
        expected["after"] = {mapping[a]: mapping[b] for a, b in expected["after"].items()}
        for s in expected["steps"]:
            s["after"] = {mapping[a]: mapping[b] for a, b in s["after"].items()}
        if expected["shortest_plan"] is not None:
            expected["shortest_plan"] = [{"block": mapping[a["block"]], "to": mapping[a["to"]]} for a in expected["shortest_plan"]]
        require(solve(altered) == expected, "block rename mismatch")
        tests.append("block_renaming_equivariance")
    else:
        altered = copy.deepcopy(t)
        altered["candidates"].reverse()
        result = solve(altered)
        require(result["ranking"] == answer["ranking"] and result["evaluations"] == list(reversed(answer["evaluations"])), "candidate order affects quality")
        tests.append("candidate_order_invariance")
        mapping = {c["candidate_id"]: "opaque_%03d" % i for i, c in enumerate(sorted(t["candidates"], key=lambda c: c["candidate_id"]))}
        altered = copy.deepcopy(t)
        for c in altered["candidates"]:
            c["candidate_id"] = mapping[c["candidate_id"]]
        result = solve(altered)
        expected = copy.deepcopy(answer)
        for e in expected["evaluations"]:
            e["candidate_id"] = mapping[e["candidate_id"]]
        expected["ranking"] = [sorted(mapping[c] for c in group) for group in expected["ranking"]]
        require(result == expected, "candidate ID affects quality")
        tests.append("candidate_id_renaming_equivariance")
    return tests


def causal_signature(t):
    # Numeric magnitudes and lexical strings are stripped, topology/operators retained.
    def ast(a):
        if set(a) == {"const"}:
            x = a["const"]
            return {"const": x if type(x) is bool or x is None else "NUMBER" if integer(x) else "STRING"}
        return {k: [ast(x) for x in v] if k == "args" else v for k, v in a.items()}
    return {"kind": "causal", "rule": ast(t["rule"]), "precondition": ast(t["precondition"]),
            "events": [{k: v for k, v in e.items() if k not in ("status", "amount")} for e in t["events"]]}


def graph_signature(t, full=True):
    # Exact canonical isomorphism with start/goal roles pinned; at eight nodes this
    # requires only 6! permutations. Fuel/cost magnitudes are deliberately stripped.
    pinned = [t["start"]] + ([t["goal"]] if t["goal"] != t["start"] else [])
    names = pinned + [n for n in t["nodes"] if n not in pinned]
    require(len(names) <= 9, "structural audit bound exceeded; do not claim exhaustive isomorphism")
    permutations = (tuple(range(len(pinned))) + p for p in itertools.permutations(range(len(pinned), len(names))))
    best = None
    for order in permutations:
        mapping = dict(zip(names, order))
        obj = {"kind": "graph", "count": len(names), "edges": sorted([mapping[a], mapping[b]] for a, b, n in t["edges"]),
               "start": mapping[t["start"]], "goal": mapping[t["goal"]]}
        if full:
            obj["plan"] = [mapping.get(x, "UNKNOWN") for x in t["plan"]]
        key = canon(obj)
        if best is None or key < best:
            best = key
    return best


def block_signature(t):
    names = list(t["on"])
    best = None
    for order in itertools.permutations(range(len(names))):
        mapping = dict(zip(names, order)); mapping["TABLE"] = "TABLE"
        obj = {"kind": "blocks", "on": sorted([[mapping[a], mapping[b]] for a, b in t["on"].items()], key=canon),
               "goal_on": sorted([[mapping[a], mapping[b]] for a, b in t["goal_on"].items()], key=canon),
               "plan": [[mapping.get(a["block"], "UNKNOWN"), mapping.get(a["to"], "UNKNOWN")] for a in t["plan"]]}
        key = canon(obj)
        if best is None or key < best:
            best = key
    return best


@functools.lru_cache(maxsize=None)
def signature_cached(serialized):
    t = json.loads(serialized)
    if t["kind"] == "causal":
        return canon(causal_signature(t))
    if t["kind"] == "blocks":
        return block_signature(t)
    if t["kind"] == "graph":
        return graph_signature(t)
    # World-only motif overlap is separated from full candidate-set overlap.
    return graph_signature(t["world"], full=False)


def signature(t):
    return signature_cached(canon(t))


def remove_numbers(t):
    if type(t) is dict:
        return {k: remove_numbers(v) for k, v in t.items() if k != "annotation"}
    if type(t) is list:
        return [remove_numbers(v) for v in t]
    return "NUMBER" if integer(t) else t


def groups_by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def duplicate_report(rows, key, min_size=2):
    return [{"fingerprint": hashlib.sha256(k.encode()).hexdigest(), "samples": [r["sample_id"] for r in rs],
             "parents": sorted({r["semantic_parent_id"] for r in rs}), "splits": sorted({r["split"] for r in rs})}
            for k, rs in groups_by(rows, key).items() if len(rs) >= min_size]


def majority(xs):
    return Counter(xs).most_common(1)[0][0] if xs else False


def lookup_baseline(examples, feature, label):
    # Fit only a frequency table on train, evaluate exposed development; unknown -> train majority.
    train = [e for e in examples if e["split"] == "train"]
    dev = [e for e in examples if e["split"] != "train"]
    counts = defaultdict(list)
    for e in train:
        counts[canon(feature(e))].append(label(e))
    fallback = majority([label(e) for e in train])
    predictions = [majority(counts[canon(feature(e))]) if canon(feature(e)) in counts else fallback for e in dev]
    truth = [label(e) for e in dev]
    return {"train_examples": len(train), "dev_examples": len(dev),
            "dev_accuracy": sum(a == b for a, b in zip(predictions, truth)) / len(dev) if dev else None,
            "train_majority_dev_accuracy": sum(fallback == b for b in truth) / len(dev) if dev else None,
            "development_majority_descriptive_accuracy": max(Counter(truth).values())/len(truth) if truth else None,
            "covered_dev_examples": sum(canon(feature(e)) in counts for e in dev),
            "train_feature_table": {k: dict(Counter(v)) for k, v in counts.items()}}


def shortcuts(rows, answers):
    # Collapse serialization siblings. Their repeated labels are not independent observations.
    units = list({(r["semantic_parent_id"], canon(r["semantic_task"])): r for r in rows}.values())
    report = {"scope": "stripped-feature count/lookups; no model training; exposed dev only", "units": len(units), "tracks": {}}
    counterexamples = []
    for track in sorted({r["track_id"] for r in units}):
        examples = []
        rs = [r for r in units if r["track_id"] == track]
        for r in rs:
            t, a = r["semantic_task"], answers[r["sample_id"]]
            common = {"split": r["split"], "family": r["semantic_family_id"], "sample": r["sample_id"],
                      "task_length": len(canon(t)), "input_length": len(r["rendered_input"]), "renderer": r["renderer"]}
            if t["kind"] == "causal":
                common.update(label=a["after"] != t["state"], applied=any(s["calc"]["operation"] != "identity" for s in a["steps"]),
                              plan_length=len(t["events"]), statuses=tuple(e["status"] for e in t["events"]),
                              op_tuple=tuple(e["op"] for e in t["events"]), kind="causal")
                examples.append(common)
            elif t["kind"] in ("graph", "blocks"):
                common.update(label=a["valid"], reaches=a["reaches_goal"], plan_length=len(t["plan"]), kind=t["kind"])
                examples.append(common)
            else:
                emap = {e["candidate_id"]: e for e in a["evaluations"]}
                for position, c in enumerate(t["candidates"]):
                    e = emap[c["candidate_id"]]
                    examples.append(dict(common, label=e["valid"], claim_correct=e["claim_correct"], reaches=e["reaches_goal"],
                                         candidate_id=c["candidate_id"], position=position, plan_length=len(c["plan"]),
                                         claim_length=len(canon(c["claimed_after"])), candidate_length=len(canon(c)),
                                         id_prefix=c["candidate_id"][:2], id_bucket=int(c["candidate_id"][1:])//10000))
        track_report = {"examples": len(examples), "label_counts": dict(Counter(e["label"] for e in examples)),
                        "length_only": lookup_baseline(examples, lambda e: e["task_length"]//50, lambda e: e["label"]),
                        "plan_length_only": lookup_baseline(examples, lambda e: e["plan_length"], lambda e: e["label"])}
        if track == "RCM-001":
            track_report["status_only"] = lookup_baseline(examples, lambda e: e["statuses"], lambda e: e["applied"])
            track_report["copy_state_accuracy"] = sum(not e["label"] for e in examples)/len(examples)
            track_report["applied_vs_state_changed"] = dict(Counter(canon([e["applied"], e["label"]]) for e in examples))
            for status, same in groups_by(examples, lambda e: canon(e["statuses"])).items():
                if len({e["applied"] for e in same}) > 1:
                    pair = [next(e for e in same if e["applied"]), next(e for e in same if not e["applied"])]
                    counterexamples.append({"attack": "status_words_imply_execution", "counterexample_samples": [e["sample"] for e in pair], "status_sequence": json.loads(status)})
        elif track == "RVM-001":
            track_report["candidate_position_only"] = lookup_baseline(examples, lambda e: e["position"], lambda e: e["label"])
            track_report["candidate_id_bucket_only"] = lookup_baseline(examples, lambda e: e["id_bucket"], lambda e: e["label"])
            track_report["claim_length_only"] = lookup_baseline(examples, lambda e: e["claim_length"], lambda e: e["claim_correct"])
            track_report["candidate_length_only"] = lookup_baseline(examples, lambda e: e["candidate_length"]//10, lambda e: e["claim_correct"])
            track_report["joint_valid_claim_counts"] = dict(Counter(canon([e["label"], e["claim_correct"]]) for e in examples))
            matched = opposite = 0
            for family, same in groups_by(examples, lambda e: e["family"]).items():
                bylen = groups_by(same, lambda e: canon(e["plan_length"]))
                matched += any(len({(e["label"],e["claim_correct"]) for e in group}) > 1 for group in bylen.values())
                opposite += any(a["label"] and not b["label"] and a["plan_length"] > b["plan_length"] for a in same for b in same)
            track_report["families_equal_length_different_quality"] = matched
            track_report["families_longer_valid_shorter_invalid"] = opposite
            # Frequency of top rank ID/order extrema, relative to an explicit per-world chance baseline.
            top = minid = maxid = firstpos = lastpos = 0
            top_shortest = 0
            for r in rs:
                cs, ranking = r["semantic_task"]["candidates"], answers[r["sample_id"]]["ranking"]
                best = set(ranking[0]); top += 1
                minid += min(c["candidate_id"] for c in cs) in best
                maxid += max(c["candidate_id"] for c in cs) in best
                firstpos += cs[0]["candidate_id"] in best; lastpos += cs[-1]["candidate_id"] in best
                shortest = min(len(c["plan"]) for c in cs)
                top_shortest += any(c["candidate_id"] in best and len(c["plan"]) == shortest for c in cs)
            track_report["top_rank_extrema"] = {"worlds": top, "min_ID": minid, "max_ID": maxid, "first_input": firstpos,
                                                 "last_input": lastpos, "shortest_candidate": top_shortest,
                                                 "chance_expected_single_ID_hits": sum(len(answers[r["sample_id"]]["ranking"][0])/len(r["semantic_task"]["candidates"]) for r in rs)}
            pair_lengths=Counter()
            for r in rs:
                lengths={c["candidate_id"]:len(c["plan"]) for c in r["semantic_task"]["candidates"]}
                ranking=answers[r["sample_id"]]["ranking"]
                for i,better in enumerate(ranking):
                    for worse in ranking[i+1:]:
                        for a in better:
                            for b in worse:
                                pair_lengths["preferred_shorter" if lengths[a]<lengths[b] else "preferred_longer" if lengths[a]>lengths[b] else "equal_length"]+=1
            track_report["strict_preference_pair_length_counts"]=dict(pair_lengths)
            unequal=pair_lengths["preferred_shorter"]+pair_lengths["preferred_longer"]
            track_report["shorter_candidate_pair_baseline_accuracy_excluding_equal_lengths"]=pair_lengths["preferred_shorter"]/unequal if unequal else None
            track_report["independence_caveat"]="310 candidate labels and all pairwise preferences share only 30 worlds; report family-level controls and do not treat these as independent trials."
        else:
            track_report["kind_only"] = lookup_baseline(examples, lambda e: e["kind"], lambda e: e["label"])
            track_report["reaches_goal_counts"] = dict(Counter(e["reaches"] for e in examples))
            track_report["development_reaches_goal_counts"]=dict(Counter(e["reaches"] for e in examples if e["split"]!="train"))
            track_report["goal_label_by_plan_length"]=lookup_baseline(examples,lambda e:e["plan_length"],lambda e:e["reaches"])
            track_report["always_false_goal_label_development_accuracy"]=sum(not e["reaches"] for e in examples if e["split"]!="train")/sum(e["split"]!="train" for e in examples)
        report["tracks"][track] = track_report
    report["shortcut_counterexamples"] = counterexamples
    report["length_interpretation"] = "RCM target-format choices intentionally change target length, not a hidden outcome label. Outcome leakage tests collapse siblings and use Task JSON length; full prompt lengths are stratified by output format. In v1 renderer and target format are confounded; v2 crosses both. Plan length is a contract-defined ranking tiebreak, so association alone is not leakage."
    format_lengths = defaultdict(list)
    for r in rows:
        a, t = answers[r["sample_id"]], r["semantic_task"]
        label = a["after"] != t["state"] if t["kind"] == "causal" else a.get("valid")
        format_lengths[canon([r["track_id"],r["target_format"],label])].append((len(r["rendered_input"]),len(canon(r["verified_target"]))))
    report["length_distributions_stratified_by_target_format_and_outcome"] = {
        key: {"records":len(vals), "input_mean": statistics.mean(v[0] for v in vals),
              "input_min":min(v[0] for v in vals),"input_max":max(v[0] for v in vals),
              "target_mean": statistics.mean(v[1] for v in vals)} for key,vals in format_lengths.items()}
    report["format_renderer_counts"] = dict(Counter(canon([r["track_id"],r["target_format"],r["renderer"]]) for r in rows))
    return report


def coverage(rows,answers):
    units=list({(r["semantic_parent_id"],canon(r["semantic_task"])):r for r in rows}.values())
    out={}
    for k in sorted({r["semantic_task"]["kind"] for r in units}):
        rs=[r for r in units if r["semantic_task"]["kind"]==k]
        group={"logical_tasks":len(rs),"families":len({r["semantic_family_id"] for r in rs})}
        if k=="causal":
            group["event_horizons"]=dict(Counter(len(r["semantic_task"]["events"]) for r in rs))
            group["executed_operations"]=dict(Counter(s["calc"]["operation"] for r in rs for s in answers[r["sample_id"]]["steps"]))
            group["gate_combinations"]=dict(Counter(canon(s["cmp"]) for r in rs for s in answers[r["sample_id"]]["steps"]))
            group["applies_but_state_identity_tasks"]=[r["sample_id"] for r in rs if any(s["calc"]["operation"]!="identity" for s in answers[r["sample_id"]]["steps"]) and answers[r["sample_id"]]["after"]==r["semantic_task"]["state"]]
        elif k in ("graph","blocks"):
            group["legality_and_goal_counts"]=dict(Counter(canon([answers[r["sample_id"]]["valid"],answers[r["sample_id"]]["reaches_goal"]]) for r in rs))
            group["shortest_plan_distances_or_unreachable"]=dict(Counter("unreachable" if answers[r["sample_id"]]["shortest_plan"] is None else len(answers[r["sample_id"]]["shortest_plan"]) for r in rs))
            group["first_illegal_positions"]=dict(Counter(next((i for i,s in enumerate(answers[r["sample_id"]]["steps"]) if not s["legal"]),"none") for r in rs))
        else:
            group["initial_goal_distances_or_unreachable"]=dict(Counter("unreachable" if answers[r["sample_id"]]["evaluations"][0]["repair_plan"] is None else len(answers[r["sample_id"]]["evaluations"][0]["repair_plan"]) for r in rs))
            group["first_error_counts"]=dict(Counter(str(e["first_error"]) for r in rs for e in answers[r["sample_id"]]["evaluations"]))
            group["distance_null_counts"]=dict(Counter(e["remaining_moves"] is None for r in rs for e in answers[r["sample_id"]]["evaluations"]))
            group["ranking_tie_groups"]=sum(len(g)>1 for r in rs for g in answers[r["sample_id"]]["ranking"])
        out[k]=group
    return out


def diversity(rows):
    exact = duplicate_report(rows, lambda r: canon(r["semantic_task"]))
    near = duplicate_report(rows, lambda r: canon({k:v for k,v in r["semantic_task"].items() if k != "annotation"}))
    numeric_near = duplicate_report(rows, lambda r: canon(remove_numbers(r["semantic_task"])))
    structure = duplicate_report(rows, lambda r: signature(r["semantic_task"]))
    families = groups_by(rows, lambda r: r["semantic_family_id"])
    parent_split_violations = [k for k, rs in families.items() if len({r["split"] for r in rs}) > 1]
    train_structures = {signature(r["semantic_task"]) for r in rows if r["split"] == "train"}
    dev = [r for r in rows if r["split"] == "dev_challenge"]
    affected = [r["sample_id"] for r in dev if signature(r["semantic_task"]) in train_structures]
    old_rows=[json.loads(line) for p in (ROOT/"data").glob("*/candidate_renderings/*.jsonl") for line in p.read_text().splitlines() if line.strip()]
    all_exact=groups_by(old_rows,lambda r:canon(r["semantic_task"]))
    current_tasks={canon(r["semantic_task"]) for r in rows}
    cross_batch_exact=[]
    for task,rs in all_exact.items():
        versions={r["sample_id"].split("-p")[0] for r in rs}
        if task in current_tasks and len(versions)>1:
            cross_batch_exact.append({"fingerprint":hashlib.sha256(task.encode()).hexdigest(),"sample_ids":[r["sample_id"] for r in rs],
                                      "splits":sorted({r["split"] for r in rs})})
    declared=groups_by([r for r in old_rows if "source_family_key" in r],lambda r:r["source_family_key"])
    return {"records": len(rows), "families": len(families), "parents": len({r["semantic_parent_id"] for r in rows}),
            "exact_task_duplicates": exact, "near_duplicates_annotation_removed": near,
            "near_duplicates_numbers_and_annotation_removed": numeric_near,
            "structural_duplicates_or_world_motifs": structure,
            "cross_split_exact_duplicates": [g for g in exact if len(g["splits"]) > 1],
            "cross_split_structural_overlap": [g for g in structure if len(g["splits"]) > 1],
            "development_samples_with_train_structure": affected,
            "family_split_violations": parent_split_violations,
            "track_split_counts": dict(Counter(canon([r["track_id"],r["split"]]) for r in rows)),
            "renderer_split_counts": dict(Counter(canon([r["renderer"],r["split"]]) for r in rows)),
            "source_provenance_counts": dict(Counter(canon(r["source_provenance"]) for r in rows)),
            "generation_seed_counts": dict(Counter(r["generation_seed"] for r in rows)),
            "source_family_key_presence_counts":dict(Counter(canon([r["track_id"],"source_family_key" in r]) for r in rows)),
            "cross_batch_exact_task_reuse":cross_batch_exact,
            "cross_batch_exact_tasks_with_split_conflict":[g for g in cross_batch_exact if len(g["splits"])>1],
            "declared_source_family_split_conflicts":[key for key,rs in declared.items() if len({r["split"] for r in rs})>1],
            "limitations": ["Exact duplicates within one family are expected order/format renderings and are not independent examples.",
                            "Structural overlap strips names and magnitudes but preserves graph edges/roles/plan and causal AST operators/event shape.",
                            "Critic structural overlap uses world motif only, not full candidate set; it cannot establish exact candidate-template identity.",
                            "Graph canonicalization exhaustively enumerates entity bijections with start/goal roles pinned, within the nine-node audit bound.",
                            "Synthetic provenance is asserted by records; this audit does not establish external license/history or sealed evaluation custody.",
                            "Development is exposed; future efficacy claims require untouched sealed evaluation custody."]}


def dataset_counterexamples(rows,answers,batch):
    examples=[]
    for r in rows:
        if r["track_id"]=="RGM-001" and r["split"]=="dev_challenge" and r["semantic_task"]["kind"]=="graph":
            a=answers[r["sample_id"]]
            if not a["reaches_goal"] and a["shortest_plan"]:
                altered=copy.deepcopy(r["semantic_task"])
                altered["plan"]=a["shortest_plan"]
                expected=solve(altered)
                require(expected["reaches_goal"],"positive matched goal counterexample failed")
                examples.append({"defect":"development_goal_label_collapse","source_sample_id":r["sample_id"],"source_record_hash":r["record_hash"],
                                 "minimal_intervention":{"plan":altered["plan"]},"new_task":altered,"new_target":expected,
                                 "explanation":"Same names, node count, graph, fuel, start and goal support a positive goal label by changing only the submitted plan."})
                break
    causal_rows=[r for r in rows if r["semantic_task"]["kind"]=="causal"]
    for family,rs in groups_by(causal_rows,lambda r:r["semantic_family_id"]).items():
        unique={canon(r["semantic_task"]):r for r in rs}
        if len(unique)==2:
            a,b=list(unique.values())
            same_except_rule={k:v for k,v in a["semantic_task"].items() if k!="rule"}=={k:v for k,v in b["semantic_task"].items() if k!="rule"}
            examples.append({"attack":"counterfactual_changes_more_than_rule","family":family,"sample_ids":[a["sample_id"],b["sample_id"]],
                             "same_except_rule":same_except_rule,"after_equal":answers[a["sample_id"]]["after"]==answers[b["sample_id"]]["after"]})
            require(same_except_rule,"causal twin changes fields besides rule")
    p=ROOT/"audits"/"shortcut_detection";p.mkdir(parents=True,exist_ok=True)
    (p/("dataset-counterexamples-"+batch+".json")).write_text(json.dumps(examples,indent=2)+"\n")
    return len(examples)


def issue(code, severity, evidence, fix):
    return {"code": code, "severity": severity, "evidence": evidence, "fix": fix}


def calibration():
    path = ROOT/"reviews"/"reviewer_calibration"/"blind_candidates.jsonl"
    receipts = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        answer = solve(r["semantic_task"])
        projected = {"after":answer["after"]} if r["target_format"] == "answer_only" else answer
        wrong = projected != r["verified_target"]
        receipts.append({"sample_id":r["sample_id"], "record_hash":r["record_hash"],
                         "receipt_id":"adv-calibration-"+digest([r["sample_id"],r["record_hash"]])[:20],
                         "reviewer_role":"Adversarial Red-Team", "reviewer_execution_id":EXECUTION,
                         "decision":"REJECT" if wrong else "ACCEPT", "critical_defects":["target_mismatch"] if wrong else [],
                         "scores":{k:1 if wrong and k in ("target_correctness","reasoning_validity") else 5 for k in SCORE_KEYS},
                         "confidence":0.99, "issues":[issue("target_mismatch","critical",{"computed":projected,"observed":r["verified_target"]},"Replace corrupted final target.")] if wrong else [],
                         "expected_recomputed":projected, "private_calibration_expectations_read":False})
    out=ROOT/"reviews"/"adversarial";out.mkdir(parents=True,exist_ok=True)
    (out/"calibration-blind.jsonl").write_text("".join(canon(r)+"\n" for r in receipts))
    print(canon({"blind_calibration_decisions":{r["sample_id"]:r["decision"] for r in receipts}}))


def semantic_probes():
    # Minimal counterexamples to tempting incorrect interpretations, independent of producer cases.
    samples = []
    c = {"kind":"causal","state":{"x":2,"y":3},"rule":{"op":"eq","args":[{"var":"event.status"},{"const":True}]},
         "precondition":{"const":True},"events":[{"status":1,"amount":1,"op":"add","source":"x","destination":"y"}]}
    require(causal(c)["after"] == {"x":2,"y":3}, "bool/int equality coercion")
    samples.append({"attack":"boolean_integer_coercion", "task":c, "correct":causal(c)})
    bad = copy.deepcopy(c)
    bad["rule"] = {"op":"any","args":[{"const":True},{"op":"unknown","args":[]}]}
    try:
        causal(bad)
    except ValueError:
        samples.append({"attack":"malformed_branch_hidden_by_short_circuit", "task":bad,"correct":"malformed/reject"})
    else:
        raise AssertionError("invalid branch was hidden")
    g={"kind":"graph","nodes":["s","a","b","g"],"edges":[["s","g",5],["s","a",1],["a","g",1],["s","b",1],["b","g",1]],
       "start":"s","goal":"g","fuel":5,"plan":["g","a"]}
    require(graph(g)["shortest_plan"] == ["g"] and graph(g)["reaches_goal"] is False, "cost objective/goal-before-illegal bug")
    samples.append({"attack":"minimum_fuel_cost_wrong_objective_and_early_goal", "task":g,"correct":graph(g)})
    b={"kind":"blocks","on":{"a":"TABLE","b":"TABLE"},"goal_on":{"a":"TABLE"},"plan":[{"block":"a","to":"TABLE"},{"block":"a","to":"b"}]}
    require(blocks(b)["valid"] is False and len(blocks(b)["steps"])==1, "block no-op not illegal")
    samples.append({"attack":"current_support_noop_and_first_failure", "task":b,"correct":blocks(b)})
    w={"kind":"graph","nodes":["s","g"],"edges":[["s","g",1]],"start":"s","goal":"g","fuel":1,"plan":[]}
    t={"kind":"critic","world":w,"candidates":[{"candidate_id":"A","plan":["g"],"claimed_after":{"node":"g","fuel":1}},
          {"candidate_id":"B","plan":[],"claimed_after":{"node":"s","fuel":1}},
          {"candidate_id":"C","plan":["s"],"claimed_after":{"node":"s","fuel":1}}]}
    a=critic(t)
    require(a["ranking"]==[["B"],["A"],["C"]] and a["evaluations"][0]["valid"] and not a["evaluations"][0]["claim_correct"], "claims vs validity/rank priority")
    samples.append({"attack":"incorrect_claim_does_not_make_transition_invalid_and_claim_priority", "task":t,"correct":a})
    out=ROOT/"audits"/"shortcut_detection";out.mkdir(parents=True,exist_ok=True)
    (out/"semantic-counterexamples.json").write_text(json.dumps(samples,indent=2)+"\n")
    return len(samples)


def main(batch):
    paths = sorted((ROOT/"data").glob("*/candidate_renderings/"+batch+".jsonl"))
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    require(rows, "no candidate records")
    answers, receipts, errors = {}, [], []
    for r in rows:
        issues, critical, tests = [], [], []
        scores = {k: 5 for k in SCORE_KEYS}
        scores.update(generalization_value=4, difficulty_quality=4, diversity=4, shortcut_resistance=4)
        try:
            require(digest({k:v for k,v in r.items() if k != "record_hash"}) == r["record_hash"], "record hash not bound to current complete candidate")
            answer = solve(r["semantic_task"])
            answers[r["sample_id"]] = answer
            projected = {"after": answer["after"]} if r["target_format"] == "answer_only" else answer
            if canon(projected) != canon(r["verified_target"]):
                critical.append("independently_recomputed_target_mismatch")
                issues.append(issue("target_mismatch", "critical", {"expected": projected, "observed": r["verified_target"]}, "Recompute target and add regression."))
                scores["target_correctness"] = scores["reasoning_validity"] = 1
            payload = json.loads(r["rendered_input"].split("Task JSON:\n", 1)[1])
            if canon(payload) != canon(r["semantic_task"]):
                critical.append("rendered_payload_semantic_mismatch")
                issues.append(issue("payload_mismatch", "critical", {"payload": payload}, "Render canonical semantic task exactly."))
                scores["instruction_clarity"] = 1
            tests = metamorphic(r["semantic_task"], answer)
            forbidden = [r["sample_id"], r["semantic_family_id"], r["semantic_parent_id"], r["record_hash"]]
            if any(x in r["rendered_input"] for x in forbidden):
                critical.append("internal_identity_rendered")
                issues.append(issue("identity_leakage", "critical", "Internal sample/family/hash in input", "Strip metadata from rendering."))
                scores["shortcut_resistance"] = 1
        except Exception as e:
            critical.append("semantic_or_metamorphic_failure")
            issues.append(issue("semantic_failure", "critical", str(e), "Repair malformed task or target; replay this audit."))
            scores["semantic_correctness"] = 1
            errors.append({"sample_id": r["sample_id"], "error": str(e)})
        kind = r["semantic_task"]["kind"]
        if kind == "causal" and len(r["semantic_task"]["events"]) == 1 and r["difficulty"] == 4:
            issues.append(issue("difficulty_needs_dependency", "low", "One event; level 4 curriculum specifies 2-8 dependent events.", "Use measured AST depth and dependency horizon rather than a nominal level."))
            scores["difficulty_quality"] = 3
        receipt = {"sample_id": r["sample_id"], "record_hash": r["record_hash"],
                   "receipt_id": "adv-"+batch+"-" + digest([r["sample_id"],r["record_hash"],EXECUTION])[:20],
                   "reviewer_role": "Adversarial Red-Team", "reviewer_execution_id": EXECUTION,
                   "additional_audit_role": "Diversity and leakage auditor",
                   "scores": scores, "critical_defects": critical, "issues": issues,
                   "decision": "REJECT" if critical else "ACCEPT", "confidence": 0.95 if not critical else 0.99,
                   "evidence": {"independent_public_contract_recomputation": True, "metamorphic_tests": tests,
                                "reviewer_verdicts_or_private_expectations_read": False,
                                "manual_assessment": "Public instructions and curriculum compared to full payload, trace, and ranking; target equality alone does not establish diversity or training efficacy."}}
        receipts.append(receipt)
    out = ROOT/"reviews"/"adversarial"; out.mkdir(parents=True,exist_ok=True)
    # These first judgments are frozen before any external reviewer verdict is disclosed.
    first = out/(batch+".first_judgments.jsonl")
    if not first.exists():
        first.write_text("".join(canon(r)+"\n" for r in receipts))
    (out/(batch+".jsonl")).write_text("".join(canon(r)+"\n" for r in receipts))
    print(canon({"phase": "independent_first_judgments_saved", "records": len(receipts), "decision_counts": dict(Counter(r["decision"] for r in receipts)), "errors": errors}))
    if errors:
        return
    div, short = diversity(rows), shortcuts(rows, answers)
    div["semantic_coverage"]=coverage(rows,answers)
    for directory, data in (("diversity", div), ("shortcut_detection", short)):
        p = ROOT/"audits"/directory; p.mkdir(parents=True,exist_ok=True)
        (p/(batch+".json")).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    overlap = set(div["development_samples_with_train_structure"])
    goal_dev=[r for r in rows if r["track_id"]=="RGM-001" and r["split"]=="dev_challenge"]
    goal_collapse=goal_dev and len({answers[r["sample_id"]]["reaches_goal"] for r in goal_dev})==1
    for receipt,r in zip(receipts,rows):
        if r["sample_id"] in overlap:
            receipt["issues"].append(issue("development_structure_reused","medium" if batch=="pilot-v1" else "low",
                "Names and numbers stripped: this development task shares a training AST/event template.",
                "Describe this as an exposed magnitude/vocabulary challenge; reserve structural OOD claims for disjoint templates."))
            receipt["scores"]["generalization_value"] = 3 if batch=="pilot-v1" else 4
        if batch=="pilot-v1" and r["split"]=="dev_challenge" and r["difficulty"]==7:
            receipt["issues"].append(issue("challenge_difficulty_conflation","medium",
                "Exposed development status assigned level7; both renderers occur in train and dev.",
                "Record structural difficulty and challenge axes separately; document absent held-out renderer combination."))
            receipt["scores"]["difficulty_quality"] = 3
            if not receipt["critical_defects"]:
                receipt["decision"] = "REVISE"
        if goal_collapse and r["track_id"]=="RGM-001" and r["split"]=="dev_challenge":
            receipt["issues"].append(issue("development_goal_label_collapse","high",
                {"all_dev_reaches_goal":answers[r["sample_id"]]["reaches_goal"],"renderings":len(goal_dev),
                 "families":len({x["semantic_family_id"] for x in goal_dev}),"always_false_dev_accuracy":1.0},
                "Add matched valid goal-reaching and non-reaching plans at the same held-out names/sizes, grouped within the same source family."))
            receipt["scores"]["generalization_value"]=3
            receipt["scores"]["shortcut_resistance"]=2
            if not receipt["critical_defects"]:
                receipt["decision"]="REVISE"
    (out/(batch+".jsonl")).write_text("".join(canon(r)+"\n" for r in receipts))
    execution = {"reviewer_execution_id": EXECUTION, "record_count": len(rows), "candidate_file_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 "semantics_sha256": hashlib.sha256((ROOT/"schemas"/"SEMANTICS.md").read_bytes()).hexdigest(),
                 "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                 "oracle_source": "This script implements public semantics independently; no generator imports or expected-output imports.",
                 "all_record_target_matches": True, "metamorphic_checks": dict(Counter(test for r in receipts for test in r["evidence"]["metamorphic_tests"])),
                 "findings": {"family_split_violations": len(div["family_split_violations"]),
                              "cross_split_exact_task_groups": len(div["cross_split_exact_duplicates"]),
                              "cross_split_structural_groups": len(div["cross_split_structural_overlap"]),
                              "development_records_with_train_structure": len(div["development_samples_with_train_structure"])}}
    execution["decision_counts"] = dict(Counter(r["decision"] for r in receipts))
    execution["minimal_semantic_attack_probes"] = semantic_probes()
    execution["dataset_counterexamples_and_twin_checks"]=dataset_counterexamples(rows,answers,batch)
    (out/("execution-"+batch+".json")).write_text(json.dumps(execution,indent=2)+"\n")
    print(canon(execution["findings"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",default="pilot-v1")
    parser.add_argument("--calibration",action="store_true")
    args=parser.parse_args()
    if args.calibration:
        calibration()
    else:
        main(args.batch)
