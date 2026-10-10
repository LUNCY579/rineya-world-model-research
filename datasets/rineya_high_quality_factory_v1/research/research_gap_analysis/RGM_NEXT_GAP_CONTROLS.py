"""Additional public release measurements; no production/oracle imports."""
from pathlib import Path
from collections import Counter, defaultdict, deque
import datetime
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
# Execute only definitions from our own audit source, without rerunning its output writer.
source_path = ROOT / "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.py"
source = source_path.read_text().split("bindings = []\n", 1)[0]
namespace = {"__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
rows = []
catalog = json.loads((ROOT / "manifests/ACTIVE_RELEASE_CATALOG.json").read_text())
for entry in catalog["releases"]:
    if entry["track"] == "RGM-001" and entry["status"] == "ACTIVE":
        rows.extend(json.loads(line) for line in (ROOT / entry["path"] / "canonical.jsonl").read_text().splitlines())
tasks = {}
families = defaultdict(dict)
for row in rows:
    key = namespace["packed"](row["semantic_task"])
    tasks.setdefault(key, row)
    families[row["source_family_key"]][key] = row
controls = Counter()
for members in families.values():
    values = list(members.values())
    if len(values) > 1:
        controls["families_with_multiple_semantic_tasks"] += 1
    for idx, left in enumerate(values):
        for right in values[idx + 1:]:
            a, b = left["semantic_task"], right["semantic_task"]
            changed = [key for key in a if a[key] != b[key]]
            controls["pair_changed_fields:" + ",".join(sorted(changed))] += 1

def retaining_goal_path(task):
    initial = task["on"]
    protected = {b: s for b, s in task["goal_on"].items() if initial[b] == s}
    if not protected:
        return False, 0
    names = sorted(initial)
    key = lambda state: tuple(state[name] for name in names)
    queue = deque([initial])
    seen = {key(initial)}
    while queue:
        state = queue.popleft()
        if all(state[b] == s for b, s in task["goal_on"].items()):
            return False, len(protected)
        for _, nxt in namespace["block_actions"](state):
            if any(nxt[b] != s for b, s in protected.items()):
                continue
            value = key(nxt)
            if value not in seen:
                seen.add(value)
                queue.append(nxt)
    return True, len(protected)

measurements = []
for row in tasks.values():
    task = row["semantic_task"]
    item = {"sample_id": row["sample_id"], "source_family_key": row["source_family_key"], "kind": task["kind"]}
    if task["kind"] == "graph":
        optimum = namespace["shortest_graph"](task)
        edges = {(a, b): cost for a, b, cost in task["edges"]}
        node, cost = task["start"], 0
        for destination in optimum or []:
            cost += edges[node, destination]
            node = destination
        cheapest = namespace["cheapest_graph"](task)
        item["canonical_minimum_moves_costs_more_than_minimum_fuel"] = optimum is not None and cost > cheapest
    else:
        necessary, protected = retaining_goal_path(task)
        optimum = namespace["shortest_blocks"](task)
        initially_illegal_later_optimal = 0
        for action in optimum:
            initial_probe = dict(task, plan=[action])
            if not namespace["blocks_replay"](initial_probe)[0]["valid"]:
                initially_illegal_later_optimal += 1
        swapped_failures = 0
        for idx in range(len(optimum) - 1):
            altered = list(optimum)
            altered[idx], altered[idx + 1] = altered[idx + 1], altered[idx]
            if not namespace["blocks_replay"](dict(task, plan=altered))[0]["valid"]:
                swapped_failures += 1
        item.update({"must_temporarily_break_initially_satisfied_goal_fact": necessary,
                     "initially_satisfied_explicit_goal_facts": protected,
                     "initially_illegal_actions_on_later_optimal_prefix": initially_illegal_later_optimal,
                     "adjacent_optimal_swaps_that_are_illegal": swapped_failures})
    measurements.append(item)

summary = {
    "graph_canonical_minimum_moves_costs_more_than_minimum_fuel": sum(item.get("canonical_minimum_moves_costs_more_than_minimum_fuel", False) for item in measurements),
    "block_tasks_requiring_temporary_break_of_an_initially_satisfied_goal_fact": sum(item.get("must_temporarily_break_initially_satisfied_goal_fact", False) for item in measurements),
    "block_tasks_with_initially_illegal_later_optimal_action": sum(item.get("initially_illegal_actions_on_later_optimal_prefix", 0) > 0 for item in measurements),
    "block_tasks_with_illegal_adjacent_optimal_swap": sum(item.get("adjacent_optimal_swaps_that_are_illegal", 0) > 0 for item in measurements),
}
out = ROOT / "research/research_gap_analysis/RGM_NEXT_GAP_CONTROLS.json"
assert not out.exists()
report = {"actual_author_execution_id": "/root/research_architect",
          "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "actual_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "own_prior_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
          "base_report_sha256": hashlib.sha256((ROOT / "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.json").read_bytes()).hexdigest(),
          "existing_supplied_family_control_pairs": dict(controls), "summary": summary,
          "measurements": measurements, "approves_generation_or_data": False}
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"controls": dict(controls), "summary": summary, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))
