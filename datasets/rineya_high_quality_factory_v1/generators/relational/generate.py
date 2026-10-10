"""Original relational parents; only generator-side reference targets are used.

Producer: /root/relational_critic_producer. This module never approves records.
"""
import argparse
import copy
import json
import random

from factory.common import ROOT, load_jsonl, record, record_hash, split_for, write_jsonl
from factory.reference import blocks_move, solve


GRAPH_CONTRACT = """Graph contract: kind=graph. nodes contains distinct strings. edges contains [from,to,cost] triples with positive integer fuel costs and at most one edge per ordered pair. start and goal are nodes; fuel is a nonnegative integer. plan is an ordered list of destination node names. Every edge is directed. Start at {node:start,fuel:fuel}. An action is legal exactly when an edge from the current node to its destination exists and its cost is at most the remaining fuel. A legal move changes node to its destination and subtracts the edge cost. At the first illegal action, include that action's {legal:false,after:unchanged_state} step, stop immediately, and do not evaluate subsequent actions. An empty plan is valid. valid means every proposed action was legal. reaches_goal is true exactly when the whole plan is valid and its terminal node is goal, including an initially satisfied goal with an empty plan. shortest_plan is a fresh plan from the INITIAL state that reaches goal within initial fuel, minimizes the number of moves, and breaks ties by the lexicographic order of the entire destination-name sequence. It is [] for an initially satisfied goal and null if no feasible plan exists. Fuel cost is a feasibility constraint, not the optimization objective. Return exactly a JSON object with steps:[{legal:boolean,after:{node:string,fuel:integer}}], after:{node:string,fuel:integer}, valid:boolean, reaches_goal:boolean, shortest_plan:list_or_null. Include only evaluated steps.\n"""

BLOCK_CONTRACT = """Blocks contract: kind=blocks. on maps each distinct block name to a support block or TABLE. TABLE is reserved, supports must exist, and self-support, cycles, or more than one block directly on a block are malformed. TABLE may support several blocks. goal_on is a partial map using existing blocks or TABLE; self-support, cycles, and multiple blocks directly assigned to one block are contradictory and malformed. The initial state and goals here are well-formed. A block is clear when no block is directly on it. plan is an ordered list of {block,to} actions. A move is legal exactly when block exists and is clear, to is TABLE or an existing clear block distinct from block, and to differs from block's current support. A legal move changes only that block's support. At the first illegal move, include its {legal:false,after:unchanged_on_map} step, stop immediately, and ignore subsequent moves. An empty plan is valid. valid means every proposed move was legal. reaches_goal means the entire plan is valid and every goal_on entry equals the final on-map entry; unspecified supports need not change. shortest_plan is a fresh plan from the INITIAL on-map that satisfies every goal_on entry, minimizes the number of moves, and breaks ties by the lexicographic order of the complete sequence of (block,to) string pairs. It is [] when the initial state already satisfies the goal and null if unreachable. Return exactly a JSON object with steps:[{legal:boolean,after:on_map}], after:on_map, valid:boolean, reaches_goal:boolean, shortest_plan:list_or_null. Include only evaluated steps.\n"""


def reorder(value):
    if isinstance(value, dict):
        return {key: reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(item) for item in value]
    return value


def graph_parent(index, ordinal, split, rng):
    size = 7 + (index // 10) % 2 if split == "dev_challenge" else 3 + ordinal % 4
    vocabulary = (["Nacre", "Obsidian", "Quasar", "Sirocco", "Tundra", "Umber", "Vesper", "Zephyr"]
                  if split == "dev_challenge" else ["Aster", "Beryl", "Cedar", "Dune", "Ember", "Fjord"])
    nodes = rng.sample(vocabulary, size)
    start, goal = nodes[0], nodes[-1]
    costs = [rng.randint(1, 4) for _ in range(size - 1)]
    edges = {(nodes[j], nodes[j + 1]): costs[j] for j in range(size - 1)}
    budget = sum(costs)
    mode = ordinal % 9
    plan = nodes[1:]
    mechanism = ["lexicographic_tie", "fuel_constrained_route", "unreachable_goal", "cycle_exhaustion",
                 "missing_edge_stop", "valid_detour", "initial_goal", "valid_full_chain", "partial_chain"][mode]
    if mode == 0:
        budget += 7
        if size >= 4:
            for middle in nodes[1:3]:
                edges[start, middle] = rng.randint(1, 3)
                edges[middle, goal] = rng.randint(1, 3)
            plan = [max(nodes[1:3]), goal]
        else:
            mechanism = "single_chain"
    elif mode == 1:
        edges[start, goal] = budget + 1
        plan = [goal, nodes[1], goal]
    elif mode == 2:
        budget = max(0, budget - 1)
        edges[start, goal] = budget + 1
        plan = nodes[1:]
    elif mode == 3:
        edges[nodes[1], start] = 1
        budget = sum(costs)
        plan = [nodes[1], start] * (budget + 1) + [goal]
    elif mode == 4:
        # The absent initial self-edge must stop the later feasible chain.
        plan = [start] + nodes[1:]
    elif mode == 5:
        edges[nodes[1], start] = 1
        budget += 4
        plan = [nodes[1], start]
    elif mode == 6:
        goal = start
        plan = []
    elif mode == 7:
        budget += rng.randint(0, 3)
    else:
        plan = nodes[1:2]
    task = dict(kind="graph", nodes=nodes, edges=[[a, b, cost] for (a, b), cost in edges.items()],
                start=start, goal=goal, fuel=budget, plan=plan)
    return task, mechanism


def block_parent(index, ordinal, split, rng):
    size = 5 if split == "dev_challenge" else 3 + ordinal % 2
    vocabulary = (["onyx", "pearl", "quartz", "ruby", "sapphire"] if split == "dev_challenge"
                  else ["amber", "birch", "cobalt", "dahlia"])
    names = rng.sample(vocabulary, size)
    mode = ordinal % 6
    on = {name: "TABLE" for name in names}
    goal = {names[j]: names[j - 1] for j in range(1, size)}
    mechanism = ["assemble_stack", "reverse_stack_and_invalid_tail", "blocked_source", "clear_partial_goal",
                 "legal_non_goal_detour", "initial_partial_goal"][mode]
    if mode in (1, 3, 5):
        on = {names[0]: "TABLE", **{names[j]: names[j - 1] for j in range(1, size)}}
    if mode == 1:
        goal = {names[-1]: "TABLE", **{names[j]: names[j + 1] for j in range(size - 1)}}
    elif mode == 2:
        on[names[1]] = names[0]
        if size >= 4:
            on[names[3]] = names[2]
        goal = {names[0]: names[2]}
    elif mode == 3:
        goal = {names[0]: names[-1]}
    elif mode == 4:
        goal = {names[0]: names[1]}
    elif mode == 5:
        goal = {names[-1]: names[-2]}
    task = dict(kind="blocks", on=on, goal_on=goal, plan=[])
    optimum = solve(task)["shortest_plan"]
    if mode == 0:
        task["plan"] = optimum
    elif mode == 1:
        final = solve({**task, "plan": optimum})["after"]
        task["plan"] = optimum + [{"block": names[0], "to": final[names[0]]}]
    elif mode == 2:
        task["plan"] = [{"block": names[0], "to": names[2]}] + optimum
    elif mode == 3:
        prefix = optimum[:1]
        after = solve({**task, "plan": prefix})["after"]
        # A state-preserving action is illegal even when source/destination exist.
        invalid = {"block": names[-1], "to": after[names[-1]]}
        assert not blocks_move(after, invalid)[0]
        task["plan"] = prefix + [invalid] + optimum[1:]
    elif mode == 4:
        task["plan"] = [{"block": names[2], "to": names[0]}, {"block": names[2], "to": "TABLE"}]
    else:
        task["plan"] = [{"block": names[-1], "to": "TABLE"}]
    return task, mechanism


def preserve_batch(paths_and_rows):
    for path, rows in paths_and_rows:
        if path.exists() and load_jsonl(path) != rows:
            raise ValueError(f"Batch {path.name} already exists with different content; use a new batch version")
    for path, rows in paths_and_rows:
        write_jsonl(path, rows)


def generate(count=30, batch="pilot-v3", offset=0):
    if count < 0 or offset < 0 or not batch or "/" in batch or "\\" in batch:
        raise ValueError("Use nonnegative count/offset and a filename-safe batch")
    rows, parents = [], []
    for index in range(offset, offset + count):
        split = split_for(index)  # Family split is fixed before task/rendering creation.
        rng = random.Random(20261009 + 17 * index)
        position = index % 5
        if position in (0, 1, 3):
            ordinal = (index // 5) * 3 + {0: 0, 1: 1, 3: 2}[position]
            task, mechanism = graph_parent(index, ordinal, split, rng)
            contract = GRAPH_CONTRACT
        else:
            ordinal = (index // 5) * 2 + (position == 4)
            task, mechanism = block_parent(index, ordinal, split, rng)
            contract = BLOCK_CONTRACT
        if index >= 30:
            from generators.relational.expansion import diversify
            task,mechanism=diversify(task,index,rng)
        target = solve(task)
        tags = ["relational_planning", "state_transition", "preconditions", "canonical_shortest_plan", mechanism,
                task["kind"], "valid_plan" if target["valid"] else "invalid_plan"]
        if target["reaches_goal"]:
            tags.append("goal_satisfaction")
        if target["shortest_plan"] is None:
            tags.append("unreachable")
        structural_difficulty = (4 if max(len(task["plan"]),len(target["shortest_plan"] or [])) > 2 else
                                 1 if not task["plan"] and target["reaches_goal"] else
                                 2 if len(task["plan"]) <= 1 else
                                 4 if (not target["valid"] and len(target["steps"]) > 1) or
                                 (task["kind"] == "blocks" and len(task["goal_on"]) > 1) else 3)
        challenge_axes = ["heldout_names", "larger_graph_or_block_count"] if split == "dev_challenge" else []
        parent_id = f"RGM-001-{batch}-p{index:04d}"
        parents.append(dict(semantic_parent_id=parent_id, semantic_family_id=parent_id,
                            split=split, mechanism=mechanism, tasks=[task],
                            structural_difficulty=structural_difficulty, challenge_axes=challenge_axes))
        for renderer in ("forward", "reverse"):
            body = json.dumps(task if renderer == "forward" else reorder(task), ensure_ascii=False, separators=(",", ":"))
            prompt = contract + "Task JSON:\n" + body
            row = record("RGM-001", batch, index, "base", task, prompt, target, tags,
                         structural_difficulty, "structured", renderer)
            row.update(generator_version="1.3.0" if index>=30 else "1.2.0",structural_difficulty=structural_difficulty, challenge_axes=challenge_axes,
                       source_family_key=f"RGM-001-seed20261009-p{index:04d}")
            row["record_hash"] = record_hash(row)
            rows.append(row)
        if split == "dev_challenge":
            matched=copy.deepcopy(task)
            if target["reaches_goal"]:
                matched["plan"]=([matched["start"]] if task["kind"]=="graph" else
                                 [{"block":min(matched["on"]),"to":matched["on"][min(matched["on"])]}])
            else:
                optimal=target["shortest_plan"]
                if optimal is None and task["kind"]=="graph":
                    matched["fuel"]=sum(edge[2] for edge in matched["edges"])+1
                    optimal=solve(matched)["shortest_plan"]
                if optimal is None:raise ValueError("Cannot construct matched goal case")
                matched["plan"]=optimal
            matched_target=solve(matched)
            assert matched_target["reaches_goal"] != target["reaches_goal"]
            parents[-1]["tasks"].append(matched)
            level=4 if max(len(matched["plan"]),len(matched_target["shortest_plan"] or []))>2 else 3
            mtags=[t for t in tags if t not in ["valid_plan","invalid_plan","goal_satisfaction","unreachable"]]
            mtags += ["matched_goal_outcome", "valid_plan" if matched_target["valid"] else "invalid_plan"]
            for renderer in ("forward","reverse"):
                body=json.dumps(matched if renderer=="forward" else reorder(matched),ensure_ascii=False,separators=(",",":"))
                row=record("RGM-001",batch,index,"matched",matched,contract+"Task JSON:\n"+body,matched_target,mtags,level,"structured",renderer)
                row.update(generator_version="1.3.0" if index>=30 else "1.2.0",structural_difficulty=level,challenge_axes=challenge_axes,
                           source_family_key=f"RGM-001-seed20261009-p{index:04d}")
                row["record_hash"]=record_hash(row);rows.append(row)
    preserve_batch([(ROOT / f"data/RGM-001/semantic_parents/{batch}.jsonl", parents),
                    (ROOT / f"data/RGM-001/candidate_renderings/{batch}.jsonl", rows)])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--batch", default="pilot-v3")
    parser.add_argument("--offset", type=int, default=0)
    arguments = parser.parse_args()
    generate(arguments.count, arguments.batch, arguments.offset)
