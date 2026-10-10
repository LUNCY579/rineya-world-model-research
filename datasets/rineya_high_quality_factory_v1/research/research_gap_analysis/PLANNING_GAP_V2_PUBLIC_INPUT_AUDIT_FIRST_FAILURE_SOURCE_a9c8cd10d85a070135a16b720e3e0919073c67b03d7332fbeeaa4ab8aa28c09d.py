"""Prospective planning-v2 design: public input-only measurement, no target comparison."""
from pathlib import Path
from collections import Counter, defaultdict, deque
import datetime
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
START = datetime.datetime.now(datetime.timezone.utc).isoformat()
BINDINGS = []


def read_public(path):
    raw = (ROOT / path).read_bytes()
    BINDINGS.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    return raw


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def graph_distance(task, unlimited=False):
    q = deque([(task["start"], task["fuel"], 0)])
    seen = set()
    while q:
        node, fuel, dist = q.popleft()
        key = node if unlimited else (node, fuel)
        if key in seen:
            continue
        seen.add(key)
        if node == task["goal"]:
            return dist
        for source, dest, cost in task["edges"]:
            if source == node and (unlimited or fuel >= cost):
                q.append((dest, fuel if unlimited else fuel - cost, dist + 1))
    return None


def block_replay(task, ignore_destination_clear=False, allow_noop=False):
    state = dict(task["on"])
    trace, failure = [], None
    for index, action in enumerate(task["plan"]):
        block, destination = action["block"], action["to"]
        clear = set(state) - set(state.values())
        reason = ("unknown_source" if block not in state else
                  "source_not_clear" if block not in clear else
                  "unknown_destination" if destination != "TABLE" and destination not in state else
                  "self_support" if destination == block else
                  "destination_not_clear" if destination != "TABLE" and destination not in clear and not ignore_destination_clear else
                  "unchanged_support" if destination == state[block] and not allow_noop else None)
        if reason is None:
            state[block] = destination
        else:
            failure = {"index": index, "reason": reason}
        trace.append({"legal": reason is None, "after": dict(state)})
        if reason is not None:
            break
    return {"trace": trace, "after": state, "valid": failure is None,
            "goal_achieved": failure is None and all(state[b] == s for b, s in task["goal_on"].items()),
            "failure": failure}


for path in ["schemas/SEMANTICS.md", "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md",
             "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md",
             "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md"]:
    read_public(path)
rows = [json.loads(line) for line in read_public("reviews/input_only/planning-gap-v1.jsonl").splitlines()]
prior = [json.loads(line) for line in read_public("research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl").splitlines()]
assert len(rows) == 384 and len(prior) == 432
allowed = {"original_record_hash", "rendered_input", "sample_id", "semantic_task", "source_family_key", "split"}
assert all(set(row) == allowed for row in rows)
assert all(not any(key in row for key in ["verified_target", "expected", "reviews", "decision"]) for row in rows + prior)
families = defaultdict(dict)
for row in rows:
    match = re.fullmatch(r"RGM-001-planning-gap-v1-p(\d+)-([ABCD][0-3])-(forward|reverse)", row["sample_id"])
    assert match, "This regex describes this frozen input-view only, not the future manifest API."
    index, context, style = int(match[1]), match[2], match[3]
    assert row["split"] == ("dev" if index % 10 in [8, 9] else "train")
    if context in families[index]:
        assert packed(row["semantic_task"]) == packed(families[index][context]["semantic_task"])
    else:
        families[index][context] = row
assert set(families) == set(range(180, 228))
assert all(len(contexts) == 4 for contexts in families.values())

infeasible, lengthened = Counter(), Counter()
for index in range(180, 192):
    row = families[index]["A1"]
    task = row["semantic_task"]
    fresh, topology = graph_distance(task), graph_distance(task, True)
    assert topology is not None
    if fresh is None:
        infeasible[row["split"]] += 1
    elif fresh > topology:
        lengthened[row["split"]] += 1
    else:
        raise AssertionError("A1 is neither intended subtype")

block_rows = [row for contexts in families.values() for row in contexts.values() if row["semantic_task"]["kind"] == "blocks"]
mutant_differences = Counter()
c1_reasons, c1_positions, c0_swap_choices, c4_prefix_choices = Counter(), Counter(), [], []
for row in block_rows:
    task = row["semantic_task"]
    baseline = block_replay(task)
    for mutant, args in [("no_op_allowed", {"allow_noop": True}),
                         ("destination_clear_ignored", {"ignore_destination_clear": True})]:
        mutant_differences[mutant] += baseline != block_replay(task, **args)
for index in range(204, 216):
    c0 = families[index]["C0"]["semantic_task"]
    c1 = block_replay(families[index]["C1"]["semantic_task"])
    assert c1["failure"] is not None
    c1_reasons[c1["failure"]["reason"]] += 1
    c1_positions[str(c1["failure"]["index"])] += 1
    choices = Counter()
    for at in range(len(c0["plan"]) - 1):
        probe = dict(c0)
        probe["plan"] = list(c0["plan"])
        probe["plan"][at], probe["plan"][at + 1] = probe["plan"][at + 1], probe["plan"][at]
        fail = block_replay(probe)["failure"]
        if fail:
            choices[fail["reason"]] += 1
    c0_swap_choices.append({"source_index": index, "split": families[index]["C0"]["split"],
                            "adjacent_swap_failure_reason_counts": dict(choices)})
    prefix_choices = []
    for at in range(1, len(c0["plan"])):
        prefix = dict(c0, plan=c0["plan"][:at])
        result = block_replay(prefix)
        if not result["valid"] or result["goal_achieved"]:
            continue
        clear_table = [b for b, s in result["after"].items() if s == "TABLE" and b not in result["after"].values()]
        if clear_table:
            # This is an executed public-input audit probe, never a proposed production target.
            probe = dict(c0, plan=c0["plan"][:at] + [{"block": min(clear_table), "to": "TABLE"}])
            strict, noop, omit = block_replay(probe), block_replay(probe, allow_noop=True), block_replay(probe, ignore_destination_clear=True)
            assert strict["failure"] == {"index": at, "reason": "unchanged_support"}
            assert noop["valid"] and strict == omit
            prefix_choices.append(at)
    c4_prefix_choices.append({"source_index": index, "split": families[index]["C0"]["split"],
                              "legal_non_goal_prefix_lengths_with_isolated_table_noop": prefix_choices})

# Prospective allocation only. No world, production plan or label is instantiated.
a_sizes = {180: 5, 181: 5, 182: 5, 183: 5, 184: 6, 185: 6,
           186: 7, 187: 6, 188: 6, 189: 7, 190: 7, 191: 7}
b_sizes = {192: 5, 193: 6, 194: 7, 195: 5, 196: 6, 197: 7,
           198: 5, 199: 6, 200: 7, 201: 5, 202: 6, 203: 7}
allocation = []
for index in range(180, 228):
    stratum = "ABCD"[(index - 180) // 12]
    subtype = ("fresh_infeasible" if index % 2 == 0 else "fuel_lengthened") if stratum == "A" else None
    size = a_sizes[index] if stratum == "A" else b_sizes[index] if stratum == "B" else 5 if index % 2 == 0 else 6
    allocation.append({"source_index": index, "stratum": stratum, "split": "dev" if index % 10 in [8, 9] else "train",
                       "objects": size, "a_subtype": subtype,
                       "c1_required_first_failure": ("source_not_clear" if index % 2 == 0 else "destination_not_clear") if stratum == "C" else None,
                       "semantic_contexts": 5 if stratum == "C" else 4})
summary = {}
for stratum in "ABCD":
    selected = [item for item in allocation if item["stratum"] == stratum]
    summary[stratum] = {"families": len(selected), "split": dict(Counter(x["split"] for x in selected)),
                       "objects": dict(Counter(str(x["objects"]) for x in selected)),
                       "split_by_objects": dict(Counter(f'{x["split"]}:{x["objects"]}' for x in selected))}
    assert summary[stratum]["split"] == {"train": 10, "dev": 2}
assert Counter(x["a_subtype"] for x in allocation[:12]) == {"fresh_infeasible": 6, "fuel_lengthened": 6}
assert Counter(x["a_subtype"] for x in allocation[:12] if x["split"] == "dev") == {"fresh_infeasible": 1, "fuel_lengthened": 1}
assert all(Counter(x["objects"] for x in allocation[:12] if x["a_subtype"] == subtype) == {5: 2, 6: 2, 7: 2}
           for subtype in ["fresh_infeasible", "fuel_lengthened"])
assert summary["A"]["objects"] == {"5": 4, "6": 4, "7": 4}
assert summary["B"]["objects"] == {"5": 4, "6": 4, "7": 4}
assert all(summary[s]["objects"] == {"5": 6, "6": 6} for s in "CD")
semantic_count = sum(x["semantic_contexts"] for x in allocation)
assert semantic_count == 204
assert sum(x["semantic_contexts"] * 2 for x in allocation if x["split"] == "train") == 340
assert sum(x["semantic_contexts"] * 2 for x in allocation if x["split"] == "dev") == 68
report = {"author_execution_id": "/root/research_architect", "role": "prospective_public_design_input_audit",
          "started_utc": START, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "bindings": BINDINGS,
          "read_boundary": "Only the six named frozen public contracts/curriculum/input-view files; no candidate targets, producer, oracle, reviewer judgments.",
          "comparison_boundary": "Derived replay/feasibility from public inputs only; no stored target or candidate receipt compared or approved.",
          "v1_input_only_measurements": {"rendered_rows": len(rows), "semantic_inputs": len(block_rows) + 96,
              "families": len(families), "prior_public_input_rows_read": len(prior),
              "a_fresh_infeasible_by_split": dict(infeasible), "a_fuel_lengthened_by_split": dict(lengthened),
              "block_semantic_inputs": len(block_rows), "mutant_replay_difference_counts": dict(mutant_differences),
              "c1_first_reason_counts": dict(c1_reasons), "c1_first_error_position_counts": dict(c1_positions),
              "c0_all_adjacent_swap_choices": c0_swap_choices, "c0_isolated_noop_probe_prefix_choices": c4_prefix_choices,
              "destination_swap_feasible_in_current_c0_families": sum(bool(x["adjacent_swap_failure_reason_counts"].get("destination_not_clear")) for x in c0_swap_choices),
              "isolated_noop_feasible_in_current_c0_families": sum(bool(x["legal_non_goal_prefix_lengths_with_isolated_table_noop"]) for x in c4_prefix_choices)},
          "prospective_allocation": allocation, "prospective_allocation_summary": summary,
          "prospective_counts": {"revised_source_families": 48, "additional_families_relative_to_v1": 0,
              "semantic_inputs": semantic_count, "correlated_rendered_rows": semantic_count * 2,
              "train_families": 40, "dev_families": 8, "train_rows": 340, "dev_rows": 68},
          "limits": ["Allocation feasibility is arithmetic, not evidence that future fresh-world novelty and search gates pass.",
                     "Input-derived probes on v1 worlds are controls, not future samples or target approvals.",
                     "No implication of structural OOD, extra primitive mechanisms, training efficacy or error independence."]}
path = ROOT / "research/research_gap_analysis/PLANNING_GAP_V2_PUBLIC_INPUT_AUDIT.json"
assert not path.exists(), "Immutable evidence output already exists"
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "measured": report["v1_input_only_measurements"], "prospective_counts": report["prospective_counts"]}, sort_keys=True))
