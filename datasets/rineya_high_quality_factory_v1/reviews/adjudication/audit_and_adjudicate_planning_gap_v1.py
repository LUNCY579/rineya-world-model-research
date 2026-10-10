"""Actual V1 adjudication against original public wording; no private solver imports."""
from pathlib import Path
from collections import Counter, defaultdict, deque
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
START = datetime.datetime.now(datetime.timezone.utc).isoformat()
BINDINGS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read(name):
    raw = (ROOT / name).read_bytes()
    BINDINGS[name] = {"path": name, "sha256": sha(raw), "bytes": len(raw)}
    return raw


def load_json(name):
    return json.loads(read(name))


def load_rows(name):
    return [json.loads(line) for line in read(name).splitlines()]


def new_json(name, value):
    p = ROOT / name
    assert not p.exists(), name
    p.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    raw = p.read_bytes()
    return {"path": name, "sha256": sha(raw), "bytes": len(raw)}


def int_type(value):
    return type(value) is int


def valid_relations(relations, names, full=False):
    assert type(relations) is dict and (set(relations) == names if full else set(relations) <= names)
    supports = []
    for block, support in relations.items():
        assert type(support) is str and support in names | {"TABLE"} and support != block
        if support != "TABLE":
            supports.append(support)
        walked = {block}
        while support in relations:
            assert support not in walked
            walked.add(support)
            support = relations[support]
    assert len(supports) == len(set(supports))


def task_schema(task):
    assert type(task) is dict
    if task["kind"] == "graph":
        assert set(task) == {"kind", "nodes", "edges", "start", "goal", "fuel", "plan"}
        nodes = task["nodes"]
        assert type(nodes) is list and all(type(x) is str for x in nodes) and len(set(nodes)) == len(nodes)
        assert 5 <= len(nodes) <= 7 and task["start"] in nodes and task["goal"] in nodes
        assert int_type(task["fuel"]) and 0 <= task["fuel"] <= 30
        assert type(task["edges"]) is list
        edge_keys = []
        for edge in task["edges"]:
            assert type(edge) is list and len(edge) == 3
            a, b, cost = edge
            assert a in nodes and b in nodes and int_type(cost) and cost > 0
            edge_keys.append((a, b))
        assert len(edge_keys) == len(set(edge_keys))
        assert type(task["plan"]) is list and all(type(x) is str and x in nodes for x in task["plan"])
    else:
        assert task["kind"] == "blocks" and set(task) == {"kind", "on", "goal_on", "plan"}
        names = set(task["on"])
        assert len(names) in [5, 6] and "TABLE" not in names and all(type(x) is str for x in names)
        valid_relations(task["on"], names, True)
        valid_relations(task["goal_on"], names)
        assert type(task["plan"]) is list
        for action in task["plan"]:
            assert type(action) is dict and set(action) == {"block", "to"}
            assert type(action["block"]) is str and action["block"] in names
            assert type(action["to"]) is str and action["to"] in names | {"TABLE"}
    assert len(task["plan"]) <= 12


def clearance(state, block):
    return block in state and block not in state.values()


def restricted_block_search(task):
    initial_true = {b: s for b, s in task["goal_on"].items() if task["on"][b] == s}
    if not initial_true:
        return {"necessary": False, "initial_true": {}, "expanded": 0, "complete": True}
    names = sorted(task["on"])
    key = lambda state: tuple(state[b] for b in names)
    queue, seen = deque([dict(task["on"])]), {key(task["on"])}
    expanded = 0
    while queue:
        state = queue.popleft()
        expanded += 1
        if all(state[b] == s for b, s in task["goal_on"].items()):
            return {"necessary": False, "initial_true": initial_true, "expanded": expanded, "complete": True}
        for _, nxt in FN["block_actions"](state):
            if any(nxt[b] != s for b, s in initial_true.items()):
                continue
            value = key(nxt)
            if value not in seen:
                seen.add(value)
                queue.append(nxt)
    return {"necessary": True, "initial_true": initial_true, "expanded": expanded, "complete": True}


def property_check(task, target, failure, witness):
    prop, e = witness["property"], witness["evidence"]
    minimum = None if target["shortest_plan"] is None else len(target["shortest_plan"])
    details = {}
    if prop == "legal_prefix_then_first_failure":
        assert failure and failure["index"] >= 1
        assert int_type(e["first_error_index"]) and e["first_error_index"] == failure["index"]
    elif prop == "valid_goal_detour":
        assert target["valid"] and target["reaches_goal"] and minimum is not None and len(task["plan"]) > minimum
        assert int_type(e["minimum_moves"]) and e["minimum_moves"] == minimum
    elif prop == "partial_block_goal":
        assert task["kind"] == "blocks" and 0 < len(task["goal_on"]) < len(task["on"])
        assert int_type(e["block_count"]) and e["block_count"] == len(task["on"])
        assert int_type(e["goal_relation_count"]) and e["goal_relation_count"] == len(task["goal_on"])
    elif prop.startswith("fuel_"):
        assert task["kind"] == "graph"
        topology_plan = FN["shortest_graph"](task, True)
        assert topology_plan is not None
        topology_length = len(topology_plan)
        assert int_type(e["topology_distance"]) and e["topology_distance"] == topology_length
        if prop == "fuel_blocks_topologically_reachable_goal":
            assert minimum is None
            cost = FN["cheapest_graph"](task)
            assert int_type(e["cheapest_fuel"]) and e["cheapest_fuel"] == cost
        else:
            assert prop == "fuel_lengthens_fresh_minimum_moves" and minimum is not None and minimum > topology_length
            assert int_type(e["minimum_moves"]) and e["minimum_moves"] == minimum
    elif prop == "required_temporary_goal_displacement":
        assert task["kind"] == "blocks" and minimum is not None
        world_key = canonical({k: v for k, v in task.items() if k != "plan"})
        if world_key not in DISPLACEMENT:
            DISPLACEMENT[world_key] = restricted_block_search(task)
        assert DISPLACEMENT[world_key]["necessary"]
        assert int_type(e["minimum_moves"]) and e["minimum_moves"] == minimum
        details = DISPLACEMENT[world_key]
    else:
        assert prop == "later_action_requires_changed_block_clearance" and task["kind"] == "blocks"
        i, j, which = e["action_index"], e["changed_by_action_index"], e["precondition"]
        assert int_type(i) and int_type(j) and 0 <= j < i < len(task["plan"]) and i >= 1
        assert i < len(target["steps"]) and all(step["legal"] for step in target["steps"][:i + 1])
        # Original public wording permits either truthful predicate, with no exclusive enum/priority.
        assert which in ["source", "destination"]
        action = task["plan"][i]
        object_name = action["block"] if which == "source" else action["to"]
        assert object_name != "TABLE" and not clearance(task["on"], object_name)
        before_j = task["on"] if j == 0 else target["steps"][j - 1]["after"]
        after_j = target["steps"][j]["after"]
        before_i = target["steps"][i - 1]["after"]
        assert not clearance(before_j, object_name) and clearance(after_j, object_name) and clearance(before_i, object_name)
        details = {"selected_object": object_name, "initially_not_clear": True,
                   "actual_enabling_action": task["plan"][j], "before_enabler": before_j,
                   "after_enabler": after_j, "before_selected_action": before_i,
                   "source_initially_not_clear": not clearance(task["on"], action["block"]),
                   "destination_initially_not_clear": action["to"] != "TABLE" and not clearance(task["on"], action["to"]),
                   "public_v1_accepts_either_truthful_predicate": True}
    return details


def main():
    global FN, DISPLACEMENT
    own_helpers = "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.py"
    raw = read(own_helpers)
    assert sha(raw) == "b93c498295a988770f5384a7b3d9fdd66d441f95aafc22e286f109090bebe175"
    FN = {"__file__": str(ROOT / own_helpers)}
    exec(compile(raw.decode().split("\nbindings = []\n", 1)[0], str(ROOT / own_helpers), "exec"), FN)
    DISPLACEMENT = {}
    source = read("reviews/adjudication/audit_and_adjudicate_planning_gap_v1.py")
    for name in ["schemas/SEMANTICS.md", "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md",
                 "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md", "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md"]:
        read(name)
    path = "data/RGM-001/candidate_renderings/planning-gap-v1.jsonl"
    rows = load_rows(path)
    assert BINDINGS[path]["sha256"] == "a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27"
    manifest_path = "data/RGM-001/manifests/planning-gap-v1.family_controls.json"
    manifest = load_json(manifest_path)
    assert BINDINGS[manifest_path]["sha256"] == "405e97f28c81e791e8614b369e8af89fa08b74da49a66114315a217bcf083233"
    input_rows = load_rows("reviews/input_only/planning-gap-v1.jsonl")
    prior = load_rows("research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
    assert len(rows) == 384 and len(prior) == 432
    indexed, input_index = {r["sample_id"]: r for r in rows}, {r["sample_id"]: r for r in input_rows}
    assert len(indexed) == len(rows) and set(indexed) == set(input_index)
    receipts = {}
    for role, receipt_path in [("expert", "reviews/expert/planning-gap-v1.jsonl"),
                               ("expert_first", "reviews/expert/planning-gap-v1.first.jsonl"),
                               ("adversarial", "reviews/adversarial/planning-gap-v1.jsonl"),
                               ("adversarial_first", "reviews/adversarial/planning-gap-v1.first.jsonl"),
                               ("oracle", "reviews/oracle/planning-gap-v1.jsonl")]:
        reviewed = load_rows(receipt_path)
        receipts[role] = {r["sample_id"]: r for r in reviewed}
        assert len(reviewed) == len(receipts[role]) == 384 and set(receipts[role]) == set(indexed)
        assert all(r["record_hash"] == indexed[sid]["record_hash"] for sid, r in receipts[role].items())
    assert BINDINGS["reviews/expert/planning-gap-v1.jsonl"]["sha256"] == "9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72"
    assert BINDINGS["reviews/adversarial/planning-gap-v1.jsonl"]["sha256"] == "404c54fc35dc86ab4670abd9ab4dc2d666a6912d2bf1962a6dce1109efbda557"
    for role in ["expert", "adversarial"]:
        assert all(receipts[role][sid] == receipts[role + "_first"][sid] for sid in indexed)
        assert all(r["decision"] == "ACCEPT" for r in receipts[role].values())
    expert_freeze = load_json("reviews/expert/planning-gap-v1.expert_comparison.freeze.json")
    assert expert_freeze["exit_code"] == 0 and expert_freeze["stage"] == "ACTUAL_FIRST_COMPARISON_AND_FINAL_RECEIPTS_FREEZE"
    load_json("reviews/expert/planning-gap-v1.expert_comparison.derive.execution.v3.json")
    read("reviews/expert/semantic_inspection_planning_gap_v1_comparison.md")
    expert_guided = load_json("reviews/expert/planning-gap-v1.expert_witness_interface_guided_supplement.json")
    assert expert_guided["exit_code"] == 0 and not expert_guided["dataset_first_final_changed"]
    load_json("reviews/expert/planning-gap-v1.expert_gate_CODE_REVIEW.guided.json")
    load_json("reviews/expert/planning-gap-v1.expert_final_outputs.freeze.json")
    red_execution = load_json("audits/execution/planning-gap-v1-adversarial.json")
    assert red_execution["completed"]
    red_guided = load_json("audits/execution/planning-gap-v1-adversarial-interface-supplement.json")
    assert red_guided["completed"] and red_guided["original_first_final_receipts_unchanged"]
    load_json("audits/execution/planning-gap-v1-adversarial-final-checks-v3.json")
    read("audits/shortcut_detection/planning-gap-v1-conditioned-adversarial.json")
    read("audits/diversity/planning-gap-v1-adversarial-canonical.json")
    root_execution = load_json("quality_reports/RGM_PLANNING_V1_FIRST_AUDIT_EXECUTION.json")
    discrepancy = load_json("quality_reports/RGM_PLANNING_V1_WITNESS_INTERFACE_DISCREPANCY.json")
    machine = load_json("audits/diversity/planning-gap-v1-planning-controls.json")["report"]
    assert root_execution["result"]["oracle_failures"] == 0 and root_execution["result"]["planning_control_error_count"] == 30
    assert len(machine["errors"]) == 30
    assert discrepancy["witness_count"] == 434 and discrepancy["affected_source_families"] == 21
    assert discrepancy["witness_encoding_failures"] == 144
    cache, details = {}, []
    for row in rows:
        sid, task = row["sample_id"], row["semantic_task"]
        assert sha(canonical({k: v for k, v in row.items() if k != "record_hash"}).encode()) == row["record_hash"]
        task_schema(task)
        assert canonical(json.loads(row["rendered_input"].split("Task JSON:\n", 1)[1])) == canonical(task)
        prior_input = input_index[sid]
        assert row["record_hash"] == prior_input["original_record_hash"] and row["rendered_input"] == prior_input["rendered_input"]
        assert canonical(task) == canonical(prior_input["semantic_task"])
        assert row["split"] == prior_input["split"] and row["source_family_key"] == prior_input["source_family_key"]
        assert row["review_status"] == "PENDING"
        key = canonical(task)
        if key not in cache:
            cache[key] = FN["graph_replay"](task) if task["kind"] == "graph" else FN["blocks_replay"](task)
        target, fail = cache[key]
        assert canonical(target) == canonical(row["verified_target"]), sid
        assert type(target["valid"]) is bool and type(target["reaches_goal"]) is bool
        assert all(type(step["legal"]) is bool for step in target["steps"])
        details.append({"sample_id": sid, "record_hash": row["record_hash"], "source_family_key": row["source_family_key"],
                        "context": row["controlled_context"], "split": row["split"], "typed_target_match": True,
                        "hash_match": True, "input_view_exact_match": True, "failure": fail,
                        "minimum_moves": None if target["shortest_plan"] is None else len(target["shortest_plan"]),
                        "public_property_witnesses": []})
    by_detail = {d["sample_id"]: d for d in details}
    families, covered, controls, witness_counts = {}, [], [], Counter()
    a_subtypes, c1_indices, c1_reasons = Counter(), Counter(), Counter()
    c0_temp, base_b_routes, family_sizes = [], [], Counter()
    for family in manifest["families"]:
        index, stratum, split = family["source_index"], family["primary_stratum"], family["split"]
        assert int_type(index) and 180 <= index <= 227
        assert stratum == "ABCD"[(index - 180) // 12]
        assert split == ("dev_challenge" if index % 10 in [8, 9] else "train")
        context_rows = {}
        for context in family["contexts"]:
            context_id, sids = context["context_id"], context["sample_ids"]
            assert len(sids) == len(set(sids)) == 2
            assigned = [indexed[sid] for sid in sids]
            assert {r["renderer"] for r in assigned} == {"forward", "reverse"}
            assert all(r["controlled_context"] == context_id and r["primary_stratum"] == stratum for r in assigned)
            assert all(r["source_family_index"] == index and r["source_family_key"] == family["source_family_key"] and r["split"] == split for r in assigned)
            assert canonical(assigned[0]["semantic_task"]) == canonical(assigned[1]["semantic_task"])
            assert canonical(assigned[0]["verified_target"]) == canonical(assigned[1]["verified_target"])
            props = context["row_properties"]
            assert type(props) is list and len(props) == len(set(props)) and all(type(p) is str for p in props)
            assert all(set(props) <= set(r["skill_tags"]) for r in assigned)
            expected_witnesses = {(sid, prop) for sid in sids for prop in props}
            actual_witnesses = set()
            for witness in context["row_witnesses"]:
                sid = witness["sample_id"]
                assert witness["scope"] == "row" and sid in sids and witness["property"] in props
                assert (sid, witness["property"]) not in actual_witnesses
                actual_witnesses.add((sid, witness["property"]))
                row = indexed[sid]
                target, fail = cache[canonical(row["semantic_task"])]
                proof = property_check(row["semantic_task"], target, fail, witness)
                by_detail[sid]["public_property_witnesses"].append({"property": witness["property"], "evidence": witness["evidence"], "truth": True, "own_proof": proof})
                witness_counts[witness["property"]] += 1
            assert actual_witnesses == expected_witnesses
            context_rows[context_id] = assigned[0]
            covered.extend(sids)
        assert set(context_rows) == {stratum + str(i) for i in range(4)}
        assert family["base_sample_id"] in [r["sample_id"] for r in indexed.values() if r["source_family_key"] == family["source_family_key"] and r["controlled_context"] == stratum + "0"]
        base = context_rows[stratum + "0"]
        task = base["semantic_task"]
        size = len(task["nodes"] if task["kind"] == "graph" else task["on"])
        family_sizes[f"{stratum}:{split}:{size}"] += 1
        target0 = cache[canonical(task)][0]
        assert target0["valid"] and target0["reaches_goal"]
        if stratum == "A":
            a1 = context_rows["A1"]["semantic_task"]
            target1 = cache[canonical(a1)][0]
            assert not target1["valid"] and FN["shortest_graph"](a1, True) is not None
            if target1["shortest_plan"] is None:
                subtype = "fresh_infeasible"
                cost = FN["cheapest_graph"](task)
                assert task["fuel"] == cost and a1["fuel"] == cost - 1
            else:
                subtype = "fuel_lengthened"
                assert len(target1["shortest_plan"]) > len(target0["shortest_plan"])
            a_subtypes[f"{split}:{subtype}"] += 1
            for context_id in ["A2", "A3"]:
                other = context_rows[context_id]["semantic_task"]
                assert canonical({k: v for k, v in task.items() if k != "plan"}) == canonical({k: v for k, v in other.items() if k != "plan"})
            a2 = context_rows["A2"]["semantic_task"]
            t2, _ = cache[canonical(a2)]
            assert a2["plan"] and t2["valid"] and not t2["reaches_goal"]
            _, f3 = cache[canonical(context_rows["A3"]["semantic_task"])]
            assert f3["reason"] == "missing_directed_edge" and f3["index"] >= 1
        elif stratum == "B":
            assert len(task["plan"]) >= 3
            routes = FN["shortest_graph_count"](task)
            base_b_routes.append(routes)
            b1 = context_rows["B1"]["semantic_task"]
            t1, f1 = cache[canonical(b1)]
            assert t1["shortest_plan"] is not None and f1["index"] >= 1
            b2 = context_rows["B2"]["semantic_task"]
            t2, f2 = cache[canonical(b2)]
            assert f2["reason"] == "missing_directed_edge" and f2["index"] >= 1
            before = t2["steps"][f2["index"] - 1]["after"]["node"]
            destination = b2["plan"][f2["index"]]
            assert (destination, before) in {(a, b) for a, b, _ in b2["edges"]}
        elif stratum == "C":
            assert 4 <= len(task["plan"]) == len(target0["shortest_plan"]) <= 8
            assert any(w["property"] == "later_action_requires_changed_block_clearance" for w in by_detail[base["sample_id"]]["public_property_witnesses"])
            c1 = context_rows["C1"]["semantic_task"]
            _, f1 = cache[canonical(c1)]
            assert f1["reason"] in ["blocked_source", "blocked_destination"]
            c1_indices[str(f1["index"])] += 1
            c1_reasons[f1["reason"]] += 1
            c3 = context_rows["C3"]["semantic_task"]
            t3, _ = cache[canonical(c3)]
            assert c3["plan"] and t3["valid"] and not t3["reaches_goal"]
            restricted = restricted_block_search(task)
            if restricted["necessary"]:
                c0_temp.append(index)
        else:
            assert 0 < len(task["goal_on"]) < len(task["on"])
            assert len(task["plan"]) == len(target0["shortest_plan"]) >= 2
            d1 = context_rows["D1"]["semantic_task"]
            t1, _ = cache[canonical(d1)]
            assert t1["valid"] and not t1["reaches_goal"] and len(t1["shortest_plan"]) > len(target0["shortest_plan"])
            d2 = context_rows["D2"]["semantic_task"]
            t2, _ = cache[canonical(d2)]
            assert t2["valid"] and t2["reaches_goal"] and len(d2["plan"]) == len(t2["shortest_plan"])
            assert canonical({k: v for k, v in d1.items() if k != "plan"}) == canonical({k: v for k, v in d2.items() if k != "plan"})
        for control in family["controls"]:
            assert control["scope"] == "corpus_variant" and control["provenance"] == "supplied_tasks"
            sids = control["sample_ids"]
            assert len(sids) == 2 and len(set(sids)) == 2
            left, right = [indexed[sid] for sid in sids]
            assert all(r["source_family_key"] == family["source_family_key"] and r["split"] == split for r in [left, right])
            a, b = left["semantic_task"], right["semantic_task"]
            changed = sorted(k for k in a if canonical(a[k]) != canonical(b[k]))
            assert changed == sorted(control["changed_task_fields"])
            name = control["name"]
            if name == "fuel_only_counterfactual":
                assert changed == ["fuel"]
            elif name == "directed_edge_deletion_counterfactual":
                assert changed == ["edges"] and len(a["edges"]) == len(b["edges"]) + 1
                assert {tuple(e) for e in b["edges"]} < {tuple(e) for e in a["edges"]}
            elif name == "adjacent_action_order_counterfactual":
                assert changed == ["plan"] and len(a["plan"]) == len(b["plan"])
                different = [i for i, (x, y) in enumerate(zip(a["plan"], b["plan"])) if x != y]
                assert len(different) == 2 and different[1] == different[0] + 1
                i = different[0]
                assert a["plan"][i] == b["plan"][i + 1] and a["plan"][i + 1] == b["plan"][i]
            elif name == "partial_goal_extension_counterfactual":
                assert changed == ["goal_on"] and set(a["goal_on"]) < set(b["goal_on"])
                assert all(b["goal_on"][x] == support for x, support in a["goal_on"].items())
            else:
                assert name == "same_world_valid_goal_detour" and changed == ["plan"]
                tb = cache[canonical(b)][0]
                assert tb["valid"] and tb["reaches_goal"] and len(b["plan"]) > len(tb["shortest_plan"])
            controls.append({"control_id": control["control_id"], "name": name, "source_family_key": family["source_family_key"],
                             "sample_ids": sids, "record_hashes": [r["record_hash"] for r in [left, right]], "truth": True, "changed_fields": changed})
        families[index] = {"source_family_key": family["source_family_key"], "stratum": stratum, "split": split, "base": base,
                           "complete_original_contexts": True, "controls": [c["control_id"] for c in family["controls"]]}
    assert len(covered) == len(set(covered)) == 384 and set(covered) == set(indexed)
    assert len(families) == 48 and len(controls) == 84 and sum(witness_counts.values()) == 434
    assert len(c0_temp) >= 4 and len(c1_indices) > 1 and sum(x >= 2 for x in base_b_routes) >= 6
    assert a_subtypes["train:fresh_infeasible"] == 6
    # Full novelty recomputation from input-only prior; names/keys/renderers do not establish novelty.
    prior_unique = {canonical(row["semantic_task"]): row["semantic_task"] for row in prior}
    prior_graph = [t for t in prior_unique.values() if t["kind"] == "graph"]
    prior_blocks = [t for t in prior_unique.values() if t["kind"] == "blocks"]
    old_weighted = {FN["graph_fingerprint"](t, True, True) for t in prior_graph}
    old_unweighted = {FN["graph_fingerprint"](t) for t in prior_graph}
    old_joint = {FN["block_fingerprint"](t, True) for t in prior_blocks}
    new_graph = [f["base"]["semantic_task"] for f in families.values() if f["base"]["semantic_task"]["kind"] == "graph"]
    new_blocks = [f["base"]["semantic_task"] for f in families.values() if f["base"]["semantic_task"]["kind"] == "blocks"]
    weighted = {FN["graph_fingerprint"](t, True, True) for t in new_graph}
    unweighted = {FN["graph_fingerprint"](t) for t in new_graph}
    joint = {FN["block_fingerprint"](t, True) for t in new_blocks}
    six_shapes = {FN["block_fingerprint"](t) for t in new_blocks if len(t["on"]) == 6}
    assert len(weighted) == len(new_graph) == 24 and not weighted & old_weighted
    assert len(unweighted - old_unweighted) >= 16
    assert len(joint) == len(new_blocks) == 24 and not joint & old_joint
    assert len(six_shapes) >= 8
    assert Counter(len(t["on"]) for t in new_blocks) == {5: 12, 6: 12}
    prior_ids = {r["sample_id"] for r in prior}
    prior_families = {r["source_family_key"] for r in prior}
    assert not set(indexed) & prior_ids and not {r["source_family_key"] for r in rows} & prior_families
    assert not set(cache) & set(prior_unique)
    # Recompute supplied expert toy under the original OR definition, not V2 source priority.
    toy = expert_guided["actual_toy_both_initial_preconditions_blocked"]["task"]
    target_toy, _ = FN["blocks_replay"](toy)
    assert target_toy["valid"] and target_toy["reaches_goal"]
    assert not clearance(toy["on"], "a") and not clearance(toy["on"], "c")
    assert clearance(target_toy["steps"][0]["after"], "a") and clearance(target_toy["steps"][1]["after"], "c")
    assert canonical(target_toy) == canonical(expert_guided["actual_toy_both_initial_preconditions_blocked"]["actual_frozen_checker_target"])
    affected = {item["sample_id"] for item in discrepancy["failed_witnesses"]}
    affected_families = {indexed[sid]["source_family_key"] for sid in affected}
    assert len(affected) == 144 and len(affected_families) == 21
    assert affected == {sid for sid, d in by_detail.items() if any(w["property"] == "later_action_requires_changed_block_clearance" for w in d["public_property_witnesses"])}
    direct_errors = [e for e in machine["errors"] if ": incorrect clearance witness: precondition" in e]
    downstream_errors = [e for e in machine["errors"] if e not in direct_errors]
    assert len(direct_errors) == 21 and len(downstream_errors) == 9
    # Bind already frozen own prospective measurements as limitations, never V1 admission requirements.
    prospective_audit = load_json("research/research_gap_analysis/PLANNING_GAP_V2_PUBLIC_INPUT_AUDIT.json")
    v1_measured = prospective_audit["v1_input_only_measurements"]
    assert v1_measured["mutant_replay_difference_counts"]["destination_clear_ignored"] == 0
    assert v1_measured["mutant_replay_difference_counts"]["no_op_allowed"] == 0
    disagreements = sum(receipts["expert"][sid]["decision"] != receipts["adversarial"][sid]["decision"] for sid in indexed)
    assert disagreements == 0
    audit = {"actual_execution_id": "/root/research_architect", "started_utc": START,
             "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "bindings": BINDINGS,
             "candidate_rows": 384, "distinct_semantic_inputs": len(cache), "source_families": 48,
             "source_families_by_split": dict(Counter(f["split"] for f in families.values())),
             "full_hash_target_render_input_checks_passed": 384, "factual_property_witnesses": dict(witness_counts),
             "all_public_v1_witnesses_and_locators_truthful": True, "all_supplied_controls_truthful": True,
             "supplied_controls": controls, "per_record_checks": details,
             "novelty": {"weighted_terminal_graph_bases": len(weighted), "new_unweighted_graph_classes": len(unweighted - old_unweighted),
                         "joint_block_bases": len(joint), "six_block_initial_shapes": len(six_shapes), "prior_disjoint": True},
             "original_cohort_coverage": {"complete_families": 48, "family_sizes": dict(family_sizes), "c1_first_indices": dict(c1_indices),
                         "c1_first_reasons": dict(c1_reasons), "c0_required_temporary_displacement_indices": c0_temp,
                         "b0_multiple_minimum_route_families": sum(x >= 2 for x in base_b_routes), "a_subtypes_by_split": dict(a_subtypes)},
             "frozen_machine_gate": {"errors": machine["errors"], "direct_encoding_family_errors": 21,
                         "dependent_aggregate_errors": 9, "direct_affected_row_ids": sorted(affected),
                         "factual_false_witnesses_established": 0, "missing_families_established": 0,
                         "completed_root_executed_probe_count": len(machine["executed_audit_probes"]),
                         "complete_families_reported_before_aborts": machine["counts"]["complete_families"]},
             "guided_toy_recomputed": {"task": toy, "own_target": target_toy,
                         "source_initially_blocked_and_cleared_at": 0, "destination_initially_blocked_and_cleared_at": 1,
                         "both_public_v1_choices_truthful": True, "own_checker_function_executed": False,
                         "actual_expert_checker_destination_error": expert_guided["actual_toy_both_initial_preconditions_blocked"]["destination_error"]},
             "review_join": {"expert_first_and_final": "ACCEPT384, identical", "adversarial_first_and_final": "ACCEPT384, identical",
                         "reviewer_disagreements": disagreements, "guided_supplements_request_publication_interface_alignment": True,
                         "adjudicator_scope_difference": "QUARANTINE is a publication-interface hold, not reversal of the scoped factual ACCEPT judgments."},
             "decision_counts": {"QUARANTINE": 384}, "publication_eligible": False,
             "decision_basis": "Original public wording supports the factual source/destination witnesses; the required frozen machine gate remains incompatible/unresolved. Hold the complete publication cohort without declaring false targets or retroactively imposing V2 enum/priority/coverage requirements.",
             "limitations": ["Six fuel-infeasible A1 families are all train and no current dev task has an infeasible fresh goal.",
                         "New C1 negative controls cover only source clearance; no-op and destination-clear mutants each expose zero of96 new block semantic inputs.",
                         "These are measured limitations, not unmet V1 frozen requirements; V2 publishes prospective improvements.",
                         "Two finite deterministic grammars,48 source instances/192 semantic inputs/384 correlated views, not48 independent mechanisms or model/structural-OOD efficacy."],
             "independence": "Actual V1/V2 design author and prior RGM adjudicator, never producer/initial reviewer. Root-guided retrospective adjudication after completed first reviews; common underlying model. Only own frozen public-derived solver definitions reused, no producer/reference/oracle/reviewer implementation imported."}
    audit_binding = new_json("reviews/adjudication/planning-gap-v1.audit.json", audit)
    verdict_path = ROOT / "reviews/adjudication/planning-gap-v1.jsonl"
    assert not verdict_path.exists()
    verdicts = []
    for row in rows:
        sid = row["sample_id"]
        scope = "direct_unresolved_interface" if sid in affected else "whole_cohort_publication_dependency"
        issue = {"category": "frozen_public_machine_interface_unresolved", "severity": "high", "scope": scope,
                 "evidence": "Public V1 permits the truthful source/destination witness without a declared enum/priority; frozen checker rejects144 witnesses and aborts21 families. The complete cohort remains unadmitted.",
                 "fix": "Preserve exact V1 bytes/reviews and machine failure; use an explicitly versioned prospective interface and fresh reviewed batch. Do not relabel correct targets as wrong or silently relax the old gate."}
        verdicts.append({"receipt_id": "adj-planning-gap-v1-" + sha(canonical([sid, row["record_hash"], "/root/research_architect", scope]).encode())[:24],
            "sample_id": sid, "record_hash": row["record_hash"], "reviewer_role": "independent_adjudicator",
            "reviewer_execution_id": "/root/research_architect", "decision": "QUARANTINE", "confidence": 0.97,
            "scores": {"target_correctness": 5, "semantic_correctness": 5, "reasoning_validity": 5,
                       "instruction_clarity": 5, "difficulty_quality": 3, "diversity": 3,
                       "generalization_value": 3, "shortcut_resistance": 2 if row["primary_stratum"] == "B" else 3,
                       "metadata_machine_interface": 2},
            "critical_defects": ["unresolved_required_machine_admission"], "issues": [issue],
            "reason": "QUARANTINE the complete cohort pending an explicit public/machine interface resolution. This exact row's target, supplied input and attached factual witnesses pass; neither V1 false gold nor false property is established. V2 requirements are not applied retroactively.",
            "evidence": {"candidate_sha256": BINDINGS[path]["sha256"], "manifest_sha256": BINDINGS[manifest_path]["sha256"],
                         "audit": audit_binding, "adjudicator_source_sha256": sha(source),
                         "own_typed_target_match": True, "own_public_property_and_locator_match": True,
                         "own_original_context_control_coverage": True, "direct_checker_encoding_rejection": sid in affected,
                         "expert_receipt_id": receipts["expert"][sid]["receipt_id"], "expert_decision": "ACCEPT",
                         "adversarial_receipt_id": receipts["adversarial"][sid]["receipt_id"], "adversarial_decision": "ACCEPT",
                         "reviewer_disagreement": False, "scoped_factual_accept_preserved": True,
                         "frozen_gate_report_sha256": BINDINGS["audits/diversity/planning-gap-v1-planning-controls.json"]["sha256"],
                         "publication_eligible": False},
            "guidance": audit["independence"]})
    verdict_path.write_text("".join(canonical(v) + "\n" for v in verdicts))
    verdict_binding = {"path": str(verdict_path.relative_to(ROOT)), "sha256": sha(verdict_path.read_bytes()), "bytes": verdict_path.stat().st_size}
    ticket = {"ticket_id": "RGM-INTERFACE-007", "status": "OPEN_FROZEN_V1_PUBLICATION_HOLD", "actual_author_execution_id": "/root/research_architect",
              "candidate": BINDINGS[path], "manifest": BINDINGS[manifest_path], "audit": audit_binding, "verdicts": verdict_binding,
              "decision_counts": {"QUARANTINE": 384}, "whole_cohort_hold": True, "no_partial_publication": True,
              "original_public_witness_truth": "All434 witnesses truthful, including144 unambiguous source/destination choices. V1 declares neither exact enum nor source-first priority.",
              "machine_conflict": {"affected_rows": 144, "aborted_families": 21, "downstream_coverage_errors": 9,
                  "root_checker_sha256": "cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3",
                  "gate_report": BINDINGS["audits/diversity/planning-gap-v1-planning-controls.json"]},
              "public_author_responsibility": "This adjudicator authored the underspecified V1 public evidence wording. Its ambiguity is not assigned to the producer as a false factual label.",
              "action": ["Preserve V1 candidate/manifest/spec/checker and first/final/guided reviewer receipts unchanged.",
                  "Hold the complete V1 publication cohort; do not infer absent semantic controls from early-abort aggregate counts.",
                  "Use already frozen prospective V2 interface with fresh source/version/hash-bound data and independent review; no future approval is given here.",
                  "Generic code-test fixes are separate; they do not resolve the frozen V1 batch gate.",
                  "Treat dev infeasibility, selective destination/no-op negatives and conditional shortcuts as measured research gaps, not retroactive V1 contract defects."],
              "no_wrong_targets_or_false_properties_established": True, "role_overlap": audit["independence"]}
    ticket_binding = new_json("feedback/improvement_requests/RGM-INTERFACE-007.json", ticket)
    report_text = f'''# Actual planning-gap-v1 adjudication

Actual author/execution: `/root/research_architect`. Final publication decisions are **QUARANTINE384**, with no row eligible for publication. This is an unresolved frozen machine/public interface hold, not a wrong-target verdict. Original expert/red first and final judgments remain ACCEPT384 each and byte-identical; they agree on factual row quality. Their completed guided supplements separately require interface alignment before publication. The adjudicator's broader publication decision and original narrower ACCEPT scopes are both preserved.

Candidate SHA `{BINDINGS[path]["sha256"]}` and manifest SHA `{BINDINGS[manifest_path]["sha256"]}` bind every decision. Own audit SHA `{audit_binding["sha256"]}` recomputes all384 full record hashes, typed canonical targets, prompt/input-view equality, all434 attached property truths and their actual locators, all84 supplied controls and all48 original four-context families. There are192 distinct semantic inputs and40 train/8 exposed-dev lineage families. No target/renderer/lineage/split/control defect is established.

The original public property explicitly permits source **or** destination clearance and asks for the actual preceding clearing move. Neither original public schema nor encoding supplement specifies `precondition` strings, exclusive initial-failure priority, the V2 obstruction field or its stronger intervening-clearance rule. Actual producer `source`/`destination` witnesses are unambiguous and independently true. They therefore are not false public metadata merely because a private checker expects `source_not_clear`/`destination_not_clear`. This author accepts responsibility for underspecified V1 interface wording; the new V2 document is prospective and immutable, and was not used as an old admission rule.

Root's exact frozen checker report `{BINDINGS['audits/diversity/planning-gap-v1-planning-controls.json']['sha256']}` reports30 errors:21 family encoding aborts and9 aggregate coverage errors. The exact root discrepancy `{BINDINGS['quality_reports/RGM_PLANNING_V1_WITNESS_INTERFACE_DISCREPANCY.json']['sha256']}` locates144 affected witness rows across21 families. Own complete-manifest inspection establishes48 actual families,7 C0 worlds requiring temporary goal displacement, C1 first-error positions0/1/2,24 fresh joint block worlds and11 six-block initial shapes. Root's27 completed-family/one six-block-shape figures describe the early-aborted checker traversal, not the actual corpus; declaring missing worlds from them would be unsupported. Root's264 executed probes and zero typed-target failures are separately attributed and do not resolve that interface conflict.

The completed expert toy has both source and destination initially blocked, moves their obstructions at0 and1, then legally moves at2. Own public replay/search reproduces the same exact target and separately proves both choices. Expert's actual frozen-checker execution rejects the truthful destination antecedent when source-first priority is selected. This is an independent semantic interpretation plus own guided recomputation, not an assertion that this adjudicator executed the private checker. The toy is not a new source family or production target.

All24 weighted terminal-role graph base classes and24 joint initial/goal block base classes are distinct and absent from the432-row prior public input view;23 new unweighted graph classes and11 six-block initial shapes pass the original gates. All12 B0 bases have minimum-route ties and7 C0s require temporary goal displacement. Actual size/split support satisfies the original prose, which acknowledges two graph dev families cannot cover three sizes individually. V2's four-per-size allocations, six selective failures per guard, new C4 no-ops, exact enum and source priority are not retroactively imposed.

Measured limitations remain real: all six fresh-infeasible A1 families are train, so always-feasible matches all exposed-dev feasibility components; all C1 negative ordering cases are selective source-clearance failures, while destination-clear-ignored/no-op-allowed each expose zero of96 new block semantic inputs. Existing prior data has11 isolated unchanged-support failures, but zero isolated destination-clear failures under the red full-guard audit; total violated-guard counts cannot imply selective coverage. Visible train-fitted component baselines and exact conditional histograms are preserved in the completed reviewers' reports. Those are component/cohort observations, not full-output model scores or efficacy. These limitations motivate prospective V2 and do not independently violate V1's original frozen controls.

The complete publication cohort remains held because operational machine admission is unresolved; this adjudication neither silently aliases old metadata nor changes that frozen gate. Generic code reviews/fixes and passing suites are distinct from old batch admission. This execution writes only new adjudication/report/ticket evidence; candidate, manifest, original specs/reviews/checker/releases and V2 docs are unchanged. It imports only this author's previously frozen public-contract helper definitions, with no producer/reference/oracle/reviewer implementation imports.

This author designed V1/V2 and previously adjudicated RGM, but generated and initially reviewed none of these records. Root communicated findings before this retrospective adjudication, and completed reviewer supplements are explicitly guided. Common underlying-model and design/adjudicator overlap prevent a claim of independent error distributions. No sealed instrument, training outcome, structural OOD or future V2 approval is asserted.

Outputs: `reviews/adjudication/planning-gap-v1.jsonl` SHA `{verdict_binding["sha256"]}`, `planning-gap-v1.audit.json` SHA `{audit_binding["sha256"]}`, and `feedback/improvement_requests/RGM-INTERFACE-007.json` SHA `{ticket_binding["sha256"]}`. Exact inputs, completed reviewer/code-interface scopes and machine evidence are all bound in the audit and execution receipt.
'''
    report_path = ROOT / "reviews/adjudication/planning-gap-v1.report.md"
    assert not report_path.exists()
    report_path.write_text(report_text)
    report_binding = {"path": str(report_path.relative_to(ROOT)), "sha256": sha(report_path.read_bytes()), "bytes": report_path.stat().st_size}
    for name, b in BINDINGS.items():
        assert sha((ROOT / name).read_bytes()) == b["sha256"], ("binding changed", name)
    execution = {"actual_execution_id": "/root/research_architect", "reviewer_role": "independent_adjudicator",
                 "started_utc": START, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "completed_execution": True,
                 "actual_source_sha256": sha(source), "bindings": BINDINGS,
                 "outputs": [audit_binding, verdict_binding, ticket_binding, report_binding], "decision_counts": {"QUARANTINE": 384},
                 "exact_receipt_coverage": 384, "expert_adversarial_disagreements": 0,
                 "publication_eligible": False, "actual_final_bytes_unchanged": True,
                 "preserved_first_review_scopes": {"expert": "ACCEPT384 factual/original contract", "adversarial": "ACCEPT384 factual/original contract"},
                 "private_implementation_imports": [], "reused_own_public_helpers": BINDINGS[own_helpers],
                 "role_overlap_and_guidance": audit["independence"],
                 "supplement_scopes": "Completed original reviews versus guided checker/public-interface supplements separately preserved.",
                 "own_preview_limitation": "An earlier read-only summary command tried root errors at top level and raised KeyError; corrected to nested report.errors. No artifact, target or decision was written by that preview.",
                 "new_generation_or_candidate_release_edits": False}
    execution_binding = new_json("reviews/adjudication/planning-gap-v1.execution.json", execution)
    print(json.dumps({"outputs": execution["outputs"] + [execution_binding], "decisions": execution["decision_counts"],
                      "own_targets_hashes": 384, "own_witnesses": 434, "own_controls": 84,
                      "own_novelty": audit["novelty"], "publication_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
