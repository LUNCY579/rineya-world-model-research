"""Producer implementation of the three PUBLIC graph-normalization views.

SOURCE PREPARATION ONLY until a separate root generation grant.
No imports from official or private normalizers/checkers.
"""
from copy import deepcopy
from itertools import permutations
import json


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lift_legacy(task):
    """Lift legacy main/procedures to uniform syntax without executing them."""
    root = "__legacy_entry__"
    while root in task["procedures"]:
        root += "_"

    def body(instructions):
        translated = []
        for instruction in instructions:
            op = instruction["op"]
            if op in ("add", "scale"):
                translated.append({"op": op, "register": instruction["register"], "value": {"const": instruction["value"]}})
            elif op == "swap":
                translated.append({"op": "swap"})
            elif op == "call":
                translated.append({"op": "call", "option": instruction["name"], "args": {}})
            elif op == "if":
                test = instruction["test"]
                translated.append({"op": "if", "test": {"all": [{"left": {"reg": test["register"]}, "cmp": test["cmp"], "right": {"const": test["value"]}}]},
                                   "then": body(instruction["then"]), "else": body(instruction["else"])})
            else:
                raise ValueError("Unsupported legacy instruction")
        return translated

    options = {name: {"params": [], "initiation": {"all": []}, "success": {"all": []}, "body": body(instructions)}
               for name, instructions in task["procedures"].items()}
    options[root] = {"params": [], "initiation": {"all": []}, "success": {"all": []}, "body": body(task["main"])}
    return {"options": options, "entry": {"option": root, "args": {}}}


def _calls(body):
    result = set()
    for instruction in body:
        if instruction["op"] == "call":
            result.add(instruction["option"])
        elif instruction["op"] == "if":
            result |= _calls(instruction["then"])
            result |= _calls(instruction["else"])
    return result


def signature(task, view):
    """Canonical least encoding over option and register alpha-renamings.

    Ordered formal declaration positions define parameter alpha identities.
    Full view retains entry/root, initiation/success, args and parameter uses.
    Both coarse views erase those wrappers and root marking; scalar operands
    collapse constants and parameters, while live register reads survive.
    Lists/operators/comparisons/calls/branches/read-write roles survive.
    """
    if view not in ("full_option", "shared_primitive_declared", "shared_primitive_reachable"):
        raise ValueError(view)
    program = lift_legacy(task) if task.get("kind") == "skills" else task
    options = program["options"]
    names = set(options)
    if view == "shared_primitive_reachable":
        names = {program["entry"]["option"]}
        frontier = list(names)
        while frontier:
            for called in sorted(_calls(options[frontier.pop()]["body"])):
                if called not in names:
                    names.add(called)
                    frontier.append(called)
    full = view == "full_option"
    encodings = []
    # <=4 cohort options makes exhaustive alpha-canonicalization bounded.
    for ordering in permutations(sorted(names)):
        option_ids = {name: index for index, name in enumerate(ordering)}
        parameters = {name: {param: index for index, param in enumerate(options[name]["params"])} for name in ordering}
        for registers in (("x", "y"), ("y", "x")):
            register_ids = {name: index for index, name in enumerate(registers)}

            def expression(expr, owner):
                if "reg" in expr:
                    return ["register", register_ids[expr["reg"]]]
                if full and "param" in expr:
                    return ["parameter", parameters[owner][expr["param"]]]
                return ["scalar"]

            def predicate(pred, owner):
                return [[expression(atom["left"], owner), atom["cmp"], expression(atom["right"], owner)] for atom in pred["all"]]

            def arguments(args, callee, owner):
                return [[parameters[callee][key], expression(value, owner)]
                        for key, value in sorted(args.items(), key=lambda pair: parameters[callee][pair[0]])]

            def instructions(body, owner):
                result = []
                for instruction in body:
                    op = instruction["op"]
                    if op in ("add", "scale"):
                        result.append([op, register_ids[instruction["register"]], expression(instruction["value"], owner)])
                    elif op == "swap":
                        result.append(["swap", 0, 1])
                    elif op == "call":
                        call = ["call", option_ids[instruction["option"]]]
                        if full:
                            call.append(arguments(instruction["args"], instruction["option"], owner))
                        result.append(call)
                    else:
                        result.append(["if", predicate(instruction["test"], owner), instructions(instruction["then"], owner), instructions(instruction["else"], owner)])
                return result

            nodes = []
            for name in ordering:
                option = options[name]
                node = {"body": instructions(option["body"], name)}
                if full:
                    node.update(params=list(range(len(option["params"]))), initiation=predicate(option["initiation"], name), success=predicate(option["success"], name))
                nodes.append(node)
            graph = {"nodes": nodes}
            if full:
                root = program["entry"]["option"]
                graph["entry"] = [option_ids[root], arguments(program["entry"]["args"], root, root)]
            encodings.append(_json(graph))
    return min(encodings)


def compare_actual(programs, legacy_rows):
    """Fail closed on any actual mutual or legacy collision in any view."""
    evidence = {}
    for view in ("full_option", "shared_primitive_declared", "shared_primitive_reachable"):
        old = [signature(row["semantic_task"], view) for row in legacy_rows]
        current = {index: signature(task, view) for index, task in programs.items()}
        groups = {}
        for index, encoded in current.items():
            groups.setdefault(encoded, []).append(index)
        evidence[view] = {"legacy_rows": len(old), "legacy_unique_signatures": len(set(old)),
                          "legacy_source_lineages": len({row["semantic_family_id"] for row in legacy_rows}),
                          "new_unique_signatures": len(groups),
                          "new_signatures": current,
                          "mutual_collision_groups": [group for _, group in sorted(groups.items()) if len(group) > 1],
                          "legacy_colliding_indices": sorted(index for index, encoded in current.items() if encoded in set(old))}
    return evidence
