"""Bounded public-contract audit of approved active P0 releases only."""
from pathlib import Path
from collections import Counter, defaultdict, deque
from itertools import permutations
import datetime
import hashlib
import heapq
import json

ROOT = Path(__file__).resolve().parents[2]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def packed(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def shortest_graph(task, ignore_fuel=False):
    adjacency = defaultdict(list)
    for a, b, cost in task["edges"]:
        adjacency[a].append((b, cost))
    queue = [(0, (), task["start"], task["fuel"])]
    seen = set()
    while queue:
        length, plan, node, fuel = heapq.heappop(queue)
        key = node if ignore_fuel else (node, fuel)
        if key in seen:
            continue
        seen.add(key)
        if node == task["goal"]:
            return list(plan)
        for destination, cost in adjacency[node]:
            if ignore_fuel or cost <= fuel:
                heapq.heappush(queue, (length + 1, plan + (destination,), destination,
                                       fuel if ignore_fuel else fuel - cost))
    return None

def cheapest_graph(task):
    adjacency = defaultdict(list)
    for a, b, cost in task["edges"]:
        adjacency[a].append((b, cost))
    heap = [(0, task["start"])]
    seen = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == task["goal"]:
            return cost
        for destination, edge_cost in adjacency[node]:
            heapq.heappush(heap, (cost + edge_cost, destination))
    return None

def shortest_graph_count(task):
    # Counts complete goal-ending destination sequences at the minimum feasible length.
    if task["start"] == task["goal"]:
        return 1
    level = {(task["start"], task["fuel"]): 1}
    while level:
        nxt = defaultdict(int)
        for (node, fuel), count in level.items():
            for a, b, cost in task["edges"]:
                if a == node and cost <= fuel:
                    nxt[b, fuel - cost] += count
        goal_count = sum(count for (node, _), count in nxt.items() if node == task["goal"])
        if goal_count:
            return goal_count
        level = nxt
    return 0

def graph_replay(task):
    node, fuel = task["start"], task["fuel"]
    edges = {(a, b): cost for a, b, cost in task["edges"]}
    steps, failure = [], None
    for idx, destination in enumerate(task["plan"]):
        cost = edges.get((node, destination))
        legal = cost is not None and cost <= fuel
        if legal:
            node, fuel = destination, fuel - cost
        else:
            failure = {"index": idx, "reason": "missing_directed_edge" if cost is None else "insufficient_fuel"}
        steps.append({"legal": legal, "after": {"node": node, "fuel": fuel}})
        if not legal:
            break
    return {"steps": steps, "after": {"node": node, "fuel": fuel},
            "valid": failure is None, "reaches_goal": failure is None and node == task["goal"],
            "shortest_plan": shortest_graph(task)}, failure

def block_actions(state):
    clear = set(state) - set(state.values())
    for block in sorted(clear):
        for destination in sorted(clear | {"TABLE"}):
            if destination != block and destination != state[block]:
                nxt = dict(state)
                nxt[block] = destination
                yield {"block": block, "to": destination}, nxt

def shortest_blocks(task):
    names = sorted(task["on"])
    key = lambda state: tuple(state[name] for name in names)
    queue = deque([(task["on"], [])])
    seen = {key(task["on"])}
    while queue:
        state, plan = queue.popleft()
        if all(state[b] == support for b, support in task["goal_on"].items()):
            return plan
        for action, nxt in block_actions(state):
            value = key(nxt)
            if value not in seen:
                seen.add(value)
                queue.append((nxt, plan + [action]))
    return None

def blocks_replay(task):
    state, steps, failure = dict(task["on"]), [], None
    for idx, action in enumerate(task["plan"]):
        block, destination = action["block"], action["to"]
        clear = set(state) - set(state.values())
        reason = ("unknown_source" if block not in state else
                  "blocked_source" if block not in clear else
                  "unknown_destination" if destination != "TABLE" and destination not in state else
                  "self_support" if destination == block else
                  "blocked_destination" if destination != "TABLE" and destination not in clear else
                  "unchanged_support" if destination == state[block] else None)
        if reason is None:
            state[block] = destination
        else:
            failure = {"index": idx, "reason": reason}
        steps.append({"legal": reason is None, "after": dict(state)})
        if reason is not None:
            break
    return {"steps": steps, "after": state, "valid": failure is None,
            "reaches_goal": failure is None and all(state[b] == s for b, s in task["goal_on"].items()),
            "shortest_plan": shortest_blocks(task)}, failure

def graph_fingerprint(task, weighted=False, terminals=False):
    names = task["nodes"]
    assert len(names) <= 7, "Bounded exhaustive isomorphism audit"
    values = []
    for ordering in permutations(names):
        index = {name: idx for idx, name in enumerate(ordering)}
        edges = tuple(sorted((index[a], index[b], cost) if weighted else (index[a], index[b])
                             for a, b, cost in task["edges"]))
        value = (edges, index[task["start"]], index[task["goal"]]) if terminals else (edges,)
        values.append(value)
    return repr(min(values))

def block_fingerprint(task, goals=False):
    names = sorted(task["on"])
    assert len(names) <= 7, "Bounded exhaustive isomorphism audit"
    values = []
    for ordering in permutations(names):
        index = {name: idx for idx, name in enumerate(ordering)}
        on = tuple(-1 if task["on"][name] == "TABLE" else index[task["on"][name]] for name in ordering)
        goal = tuple((index[b], -1 if s == "TABLE" else index[s]) for b, s in task["goal_on"].items())
        values.append((on, tuple(sorted(goal))) if goals else (on,))
    return repr(min(values))

bindings = []
def read(name):
    raw = (ROOT / name).read_bytes()
    bindings.append({"path": name, "sha256": sha(raw), "bytes": len(raw)})
    return raw

catalog = json.loads(read("manifests/ACTIVE_RELEASE_CATALOG.json"))
tracks = {track: [] for track in ["RCM-001", "RGM-001", "RVM-001"]}
for entry in catalog["releases"]:
    if entry["track"] not in tracks or entry["status"] != "ACTIVE":
        continue
    raw = read(entry["path"] + "/canonical.jsonl")
    assert sha(raw) == entry["canonical_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    rows = [row for row in rows if row["record_hash"] not in entry["excluded_record_hashes"]]
    assert len(rows) == entry["active_records"]
    tracks[entry["track"]].extend(rows)

counts = {}
for track, rows in tracks.items():
    families = defaultdict(set)
    for row in rows:
        families[row["source_family_key"]].add(row["split"])
    assert all(len(value) == 1 for value in families.values())
    counts[track] = {"active_records": len(rows), "source_families": len(families),
        "source_families_by_split": dict(Counter(next(iter(value)) for value in families.values())),
        "distinct_visible_semantic_tasks": len({packed(row["semantic_task"]) for row in rows}),
        "operational_target": [150, 250], "within_operational_target": 150 <= len(families) <= 250,
        "gap_to_lower_target": max(0, 150 - len(families)), "room_to_upper_target": 250 - len(families)}

rows = tracks["RGM-001"]
unique = {}
for row in rows:
    unique.setdefault(packed(row["semantic_task"]), row)
task_results = []
for row in unique.values():
    task = row["semantic_task"]
    result, failure = graph_replay(task) if task["kind"] == "graph" else blocks_replay(task)
    assert result == row["verified_target"], row["sample_id"]
    item = {"sample_id": row["sample_id"], "record_hash": row["record_hash"],
            "source_family_key": row["source_family_key"], "split": row["split"], "kind": task["kind"],
            "submitted_length": len(task["plan"]), "shortest_length": None if result["shortest_plan"] is None else len(result["shortest_plan"]),
            "valid": result["valid"], "reaches_goal": result["reaches_goal"], "failure": failure,
            "difficulty_metadata": row["structural_difficulty"]}
    if task["kind"] == "graph":
        no_fuel = shortest_graph(task, True)
        min_cost = cheapest_graph(task)
        item.update({"objects": len(task["nodes"]), "edges": len(task["edges"]), "fuel": task["fuel"],
            "topology_fingerprint": graph_fingerprint(task), "weighted_topology_fingerprint": graph_fingerprint(task, True),
            "terminal_topology_fingerprint": graph_fingerprint(task, False, True),
            "ignoring_fuel_shortest_length": None if no_fuel is None else len(no_fuel),
            "minimum_fuel_to_goal": min_cost,
            "resource_changes_fresh_feasibility": no_fuel is not None and result["shortest_plan"] is None,
            "resource_lengthens_fresh_shortest": no_fuel is not None and result["shortest_plan"] is not None and len(no_fuel) < len(result["shortest_plan"]),
            "shortest_move_routes": shortest_graph_count(task),
            "opposite_direction_missing": sum((b, a) not in {(x, y) for x, y, _ in task["edges"]} for a, b, _ in task["edges"])})
    else:
        item.update({"objects": len(task["on"]), "partial_goal": len(task["goal_on"]) < len(task["on"]),
            "initial_table_roots": list(task["on"].values()).count("TABLE"),
            "explicit_goal_table_roots": list(task["goal_on"].values()).count("TABLE"),
            "initial_arrangement_fingerprint": block_fingerprint(task),
            "initial_goal_fingerprint": block_fingerprint(task, True)})
    task_results.append(item)

summaries = {}
for kind in ["graph", "blocks"]:
    items = [item for item in task_results if item["kind"] == kind]
    summary = {"unique_tasks": len(items), "source_families": len({item["source_family_key"] for item in items}),
       "split_tasks": dict(Counter(item["split"] for item in items)),
       "objects": dict(Counter(str(item["objects"]) for item in items)),
       "submitted_lengths": dict(Counter(str(item["submitted_length"]) for item in items)),
       "shortest_lengths": dict(Counter(str(item["shortest_length"]) for item in items)),
       "valid_goal_joint": dict(Counter(str((item["valid"], item["reaches_goal"])) for item in items)),
       "first_error_reasons": dict(Counter(item["failure"]["reason"] for item in items if item["failure"])),
       "first_error_indices": dict(Counter(str(item["failure"]["index"]) for item in items if item["failure"])),
       "legal_prefix_then_failure": sum(item["failure"] is not None and item["failure"]["index"] >= 1 for item in items),
       "valid_detours_over_shortest": sum(item["valid"] and item["reaches_goal"] and item["shortest_length"] is not None and item["submitted_length"] > item["shortest_length"] for item in items)}
    if kind == "graph":
        for key in ["topology_fingerprint", "weighted_topology_fingerprint", "terminal_topology_fingerprint"]:
            distribution = Counter(item[key] for item in items)
            summary[key] = {"classes": len(distribution), "largest_class_tasks": max(distribution.values()),
                            "train_dev_shared_classes": len({item[key] for item in items if item["split"] == "train"} & {item[key] for item in items if item["split"] == "dev_challenge"})}
        for key in ["resource_changes_fresh_feasibility", "resource_lengthens_fresh_shortest"]:
            summary[key] = sum(item[key] for item in items)
        summary["multiple_minimum_move_routes"] = sum(item["shortest_move_routes"] > 1 for item in items)
    else:
        for key in ["initial_arrangement_fingerprint", "initial_goal_fingerprint"]:
            distribution = Counter(item[key] for item in items)
            summary[key] = {"classes": len(distribution), "largest_class_tasks": max(distribution.values()),
                            "train_dev_shared_classes": len({item[key] for item in items if item["split"] == "train"} & {item[key] for item in items if item["split"] == "dev_challenge"})}
        summary["partial_goals"] = sum(item["partial_goal"] for item in items)
        summary["initial_table_roots"] = dict(Counter(str(item["initial_table_roots"]) for item in items))
    summaries[kind] = summary

docs = ["schemas/SEMANTICS.md", "schemas/COMPETENCY_TAGS.md", "research/curricula/RGM-001.md",
        "research/curricula/RCM-001.md", "research/curricula/RVM-001.md", "research/research_gap_analysis/PRODUCTION_MILESTONE.md",
        "CURRICULUM_COVERAGE.md", "RESEARCH_GAPS.md"]
for doc in docs:
    read(doc)
report = {"actual_author_execution_id": "/root/research_architect",
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Approved active P0 canonical releases and public semantics/curriculum/research docs only; no candidate, implementation, unfinished review or protected instrument read.",
    "source_bindings": bindings, "actual_source_sha256": sha(Path(__file__).read_bytes()),
    "active_catalog_declared_total_records": catalog["active_records"], "active_catalog_declared_total_families": catalog["active_source_families"],
    "independently_measured_p0_counts": counts, "rgm": summaries, "rgm_task_measurements": task_results,
    "rgm_exact_public_recomputation_matches": len(task_results), "rgm_rendered_records": len(rows),
    "independent_mechanism_count_claim": "No statistical independence or unique mechanism count follows from source keys. RGM uses two explicit action grammars; isomorphism classes are measured representation invariants, not mechanisms.",
    "source_family_unit": "Declared source family, counted once across all twins, plans, renderers and active versions; no family crosses train/dev.",
    "approves_generation_or_data": False}
out = ROOT / "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.json"
assert not out.exists()
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"counts": counts, "rgm": summaries, "output_sha256": sha(out.read_bytes())}, indent=2))
