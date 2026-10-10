"""Original graph critic parents, produced without oracle/reviewer access.

Producer: /root/relational_critic_producer. This module never approves records.
"""
import argparse
import json
import random

from factory.common import ROOT, load_jsonl, record, record_hash, split_for, write_jsonl
from factory.reference import graph_run, shortest_graph, solve


CONTRACT = """Critic contract: kind=critic. world describes a directed graph. nodes are distinct strings, edges are [from,to,positive_integer_fuel_cost] triples with at most one edge per ordered pair, start and goal are nodes, and fuel is a nonnegative integer. Ignore world's plan. Each candidate has a distinct candidate_id, an ordered plan of destination names, and claimed_after with exactly {node:world_node_string,fuel:nonnegative_integer}. Every candidate independently begins at {node:world.start,fuel:world.fuel}. A move is legal exactly when the directed edge exists and costs no more than remaining fuel. If legal, subtract cost and change node; at the first illegal move retain the previous state and stop without evaluating later moves. Empty plans are valid. Goal achievement requires the ENTIRE proposed plan to be valid and its executed terminal node to equal world.goal.
Evaluate every candidate in input order. Return exactly {evaluations:[{candidate_id,valid,claim_correct,reaches_goal,remaining_moves,progress,usefulness,first_error,repair_plan}],ranking:[[candidate_id,...],...]}. valid means every action is legal. claim_correct independently compares claimed_after with the actual executed final state, including the unchanged state after an illegal step. reaches_goal means valid and final node equals goal. remaining_moves is the minimum number of feasible moves to goal from the actual executed final state using its remaining fuel, or null if unreachable. Define initial_remaining_moves by the same search from the world's initial state. progress is initial_remaining_moves minus remaining_moves ONLY when valid and both distances are finite; otherwise null. usefulness is -1 for invalid candidates, 2 for valid candidates reaching goal, 1 for valid reachable candidates with progress>0, and 0 otherwise. Incorrect claims do not change validity, reachability, progress, or usefulness. first_error is the zero-based index of the first illegal action, otherwise the string "claim" if the legal candidate's claimed state is wrong, otherwise null.
repair_plan is an optimal FRESH replacement plan from the INITIAL world state for every candidate, even an already valid one: minimize moves within initial fuel, and break ties lexicographically by the full destination-name sequence. This is minimal fresh-plan replacement, not an edit-distance repair to the submitted candidate. It is [] if start already equals goal, null if no feasible initial plan exists. Fuel consumption is a constraint, not the optimization objective. Rank candidates descending by the tuple (valid AND claim_correct, reaches_goal, usefulness, -remaining_moves, -len(plan)), using the original full submitted plan length. Null distance is worse than every finite distance. Booleans order true above false. Return each exact tuple tie as one group of candidate IDs sorted lexicographically, and order groups best first. Candidate IDs and input order convey no quality information.\n"""


def reorder(value):
    if isinstance(value, dict):
        return {key: reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(item) for item in value]
    return value


def world_parent(index, split, rng):
    size = 7 + (index // 10) % 2 if split == "dev_challenge" else 3 + index % 4
    vocabulary = (["Nacre", "Obsidian", "Quasar", "Sirocco", "Tundra", "Umber", "Vesper", "Zephyr"]
                  if split == "dev_challenge" else ["Aster", "Beryl", "Cedar", "Dune", "Ember", "Fjord"])
    nodes = rng.sample(vocabulary, size)
    start, goal = nodes[0], nodes[-1]
    costs = [rng.randint(2, 4) for _ in range(size - 1)]
    edges = {(nodes[j], nodes[j + 1]): costs[j] for j in range(size - 1)}
    edges[nodes[1], start] = 1
    mode = index % 6
    fuel = sum(costs) + 7
    mechanism = ["equal_length_routes", "fuel_precondition", "legal_detour", "initially_unreachable",
                 "initial_goal", "fresh_replacement_counterexample"][mode]
    if mode in (0, 2, 5) and size >= 4:
        for middle in nodes[1:3]:
            edges[start, middle] = rng.randint(2, 4)
            edges[middle, goal] = rng.randint(2, 4)
    if mode == 1:
        fuel = sum(costs)
    if mode == 3:
        fuel = max(costs[0] + 1, sum(costs) - 1)
    if mode == 4:
        goal = start
    elif mode in (1, 3):
        # The superficially shortest route exists but violates available fuel.
        edges[start, goal] = fuel + 1
    world = dict(kind="graph", nodes=nodes, edges=[[a, b, cost] for (a, b), cost in edges.items()],
                 start=start, goal=goal, fuel=fuel, plan=[])
    return world, mechanism


def candidates_for(world, rng):
    optimum = shortest_graph(world)
    first = world["nodes"][1]
    goal, start = world["goal"], world["start"]
    prefix = [first]
    backtrack = [first, start]
    assert graph_run(world, prefix)["valid"] and graph_run(world, backtrack)["valid"]
    # The absent self-edge supplies a known one-step invalid candidate even
    # when the start already satisfies goal.
    short_invalid = [start]
    assert not graph_run(world, short_invalid)["valid"]
    initial = {"node": start, "fuel": world["fuel"]}
    entries = [([], "truth"), (prefix, "truth"), (backtrack, "truth"),
               (short_invalid, "truth"), (prefix, "arithmetic_error"), (prefix, "truth")]
    if optimum is not None:
        entries += [(optimum, "truth"), (optimum, "arithmetic_error")]
        # A longer legal goal-reaching plan, when fuel permits, outranks a
        # shorter illegal plan without relying on the plan-length feature.
        longer = backtrack + optimum
        if graph_run(world, longer)["valid"]:
            entries.append((longer, "truth"))
    else:
        entries.append((world["nodes"][1:], "truth"))
    # Failure after an executed prefix, with a subsequent action to be ignored.
    later_invalid = prefix + [first, goal]
    assert not graph_run(world, later_invalid)["valid"]
    entries.append((later_invalid, "false_terminal"))
    # Cost failures are distinct from absent edges where a direct edge exists.
    direct = [goal]
    if not graph_run(world, direct)["valid"] and direct != short_invalid:
        entries.append((direct, "truth"))
    candidates = []
    ids = rng.sample(range(10000, 99999), len(entries))
    for candidate_number, (plan, claim_type) in enumerate(entries):
        actual = graph_run(world, plan)["after"]
        claim = dict(actual)
        if claim_type == "arithmetic_error":
            claim["fuel"] += rng.randint(1, 3)
        elif claim_type == "false_terminal":
            claim = {"node": goal, "fuel": 0}
            if claim == actual:
                claim["fuel"] += 1
        candidates.append(dict(candidate_id=f"c{ids[candidate_number]}", plan=list(plan), claimed_after=claim))
    rng.shuffle(candidates)  # Independent of labels, lengths and quality.
    assert any(len(candidate["plan"]) == 1 and graph_run(world, candidate["plan"])["valid"] for candidate in candidates)
    assert any(len(candidate["plan"]) == 1 and not graph_run(world, candidate["plan"])["valid"] for candidate in candidates)
    assert any(len(candidate["plan"]) >= 2 and graph_run(world, candidate["plan"])["valid"] for candidate in candidates)
    assert initial == graph_run(world, [])["after"]
    return candidates


def generate(count=30, batch="pilot-v2", offset=0):
    if count < 0 or offset < 0 or not batch or "/" in batch or "\\" in batch:
        raise ValueError("Use nonnegative count/offset and a filename-safe batch")
    rows, parents = [], []
    for index in range(offset, offset + count):
        split = split_for(index)
        rng = random.Random(20261009 + 31 * index)
        world, mechanism = world_parent(index, split, rng)
        if index >= 30:
            from generators.relational.expansion import diversify
            world, topology = diversify(world, index, rng)
            mechanism = topology + ":" + mechanism
            if mechanism.endswith(':initially_unreachable'):
                mechanism=mechanism.rsplit(':',1)[0]+(':'+'initial_goal_unreachable' if shortest_graph(world) is None else ':'+'initial_goal_reachable')
        if mechanism.endswith('equal_length_routes'):
            from generators.critic.route_facts import equal_length_goal_routes
            mechanism=mechanism[:-len('equal_length_routes')]+('equal_length_goal_routes' if equal_length_goal_routes(world) else 'no_equal_length_goal_routes')
        candidates=candidates_for(world,rng)
        if index>=180:
            from generators.critic.claim_controls import augment
            candidates=augment(world,candidates,rng)
        task = dict(kind="critic", world=world, candidates=candidates)
        target = solve(task)
        tags = ["value_critic", "legality", "claim_arithmetic", "progress", "usefulness", "rank_with_ties",
                "minimal_fresh_plan_replacement", "equal_length_good_bad", "longer_valid_shorter_invalid", mechanism]
        structural_difficulty = 4 if any(graph_run(world, candidate["plan"])["valid"] and len(candidate["plan"]) > 2
                                         for candidate in task["candidates"]) else 3
        challenge_axes = ["heldout_names", "larger_graph_or_block_count"] if split == "dev_challenge" else []
        parent_id = f"RVM-001-{batch}-p{index:04d}"
        parents.append(dict(semantic_parent_id=parent_id, semantic_family_id=parent_id,
                            split=split, mechanism=mechanism, tasks=[task],
                            structural_difficulty=structural_difficulty, challenge_axes=challenge_axes))
        for renderer in ("forward", "reverse"):
            body = json.dumps(task if renderer == "forward" else reorder(task), ensure_ascii=False, separators=(",", ":"))
            prompt = CONTRACT + "Task JSON:\n" + body
            row = record("RVM-001", batch, index, "base", task, prompt, target, tags,
                         structural_difficulty, "structured", renderer)
            row.update(structural_difficulty=structural_difficulty, challenge_axes=challenge_axes)
            row.update(generator_version="1.1.2",source_family_key=f"RVM-001-seed20261009-p{index:04d}")
            if index>=180:
                row['generator_version']='1.2.1'
                row['skill_tags']=row['skill_tags']+['fuel_underclaim','fuel_overclaim','wrong_location_claim','matched_equal_plan_claims']
            row["record_hash"] = record_hash(row)
            rows.append(row)
    output = [(ROOT / f"data/RVM-001/semantic_parents/{batch}.jsonl", parents),
              (ROOT / f"data/RVM-001/candidate_renderings/{batch}.jsonl", rows)]
    for path, contents in output:
        if path.exists() and load_jsonl(path) != contents:
            raise ValueError(f"Batch {path.name} already exists with different content; use a new batch version")
    for path, contents in output:
        write_jsonl(path, contents)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--batch", default="pilot-v2")
    parser.add_argument("--offset", type=int, default=0)
    arguments = parser.parse_args()
    generate(arguments.count, arguments.batch, arguments.offset)
