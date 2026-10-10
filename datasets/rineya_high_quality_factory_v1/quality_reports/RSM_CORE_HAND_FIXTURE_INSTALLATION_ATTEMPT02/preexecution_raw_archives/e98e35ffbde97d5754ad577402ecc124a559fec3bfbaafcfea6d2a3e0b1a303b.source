"""Corrective REM producer with public in-row witnesses and scoped controls.

Frozen memory-gap-v1 is imported only for neutral formatting/vocabulary helpers.
No independent oracle or reviewer implementation is imported or read.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from factory.common import ROOT, canonical, digest, load_jsonl, record, record_hash, split_for, write_json, write_jsonl
from generators.memory.history_gap import SEED, TRACK, KEY_ROLES, TRAIN_KEYS, DEV_KEYS, TRAIN_VALUES, DEV_VALUES, event, reorder
from generators.other_tracks.reference import contracts, solve


QUERY_ROLES = ("delayed_before", "delayed_at", "delayed_later", "lifecycle_delete",
               "lifecycle_at", "lifecycle_present_null", "same_version_time_index", "absent")
VARIANTS = ("baseline", "version_priority_raise", "lifecycle_op_toggle", "index_tie_swap")
PROPERTIES = ("decisive_delayed_boundary", "version_over_recency", "equal_version_time_conflict",
              "equal_version_time_input_conflict", "authoritative_tombstone", "present_null_distinction",
              "future_event_filter_control", "cross_key_filter_control", "absence_control")
CORPUS_CONTROLS = ("query_role_id_position_crossing", "key_role_crossing", "field_order_invariance",
                   "version_priority_counterfactual", "active_input_tie_swap_counterfactual", "null_lifecycle_counterfactual")
PROBE_CONTROLS = ("lower_version_append_invariance", "other_key_edit_invariance", "future_event_edit_invariance")


def typed(value):
    return "null" if value is None else "boolean" if type(value) is bool else "integer" if type(value) is int else "string"


def event_answer(item):
    return dict(status="deleted" if item["op"] == "delete" else "present", value=item["value"])


def answer_signature(result):
    return canonical(dict(status=result["status"], value_type=typed(result["value"]), value=result["value"]))


def family_history(index, split):
    rng = random.Random(SEED + 97 * index)
    names = rng.sample(DEV_KEYS if split == "dev_challenge" else TRAIN_KEYS, 4)
    values = rng.sample(DEV_VALUES if split == "dev_challenge" else TRAIN_VALUES, 8)
    start = 7 + (index - 12) * 3
    delay = start + 8 + 3 * (index % 5)
    early_delete = delay + 2
    lifecycle = delay + 6 + index % 3
    null_time, tie, later = lifecycle + 1, delay - 3, delay + 2 + index % 4
    dv, lv, cv = 1 + index % 4, 2 + index % 3, 100 + index
    previous = (values[0], -index, bool(index % 2))[index % 3]
    delayed = (index + 20, bool(index % 2), values[1])[index % 3]
    tagged = [
        ("delayed_initial", event("delayed", dv + 1, start, "put", previous)),
        ("delayed_previous", event("delayed", dv + 1, delay - 2, "put", values[7])),
        ("delayed_update", event("delayed", dv + 2, delay, "put", delayed)),
        ("later_lower", event("delayed", dv + 1, later, "put", values[2])),
        ("lifecycle_initial", event("lifecycle", lv, start + 2, "put", values[3])),
        ("retained_delete", event("lifecycle", lv + 1, early_delete, "delete", None)),
        ("lifecycle_lower_later", event("lifecycle", lv, early_delete + 1, "put", values[4])),
        ("lifecycle_control", event("lifecycle", lv + 2, lifecycle, "delete", None)),
        ("put_null", event("lifecycle", lv + 3, null_time, "put", None)),
        ("tie_initial", event("index_tie", cv, start + 1, "put", values[4])),
        ("tie_left", event("index_tie", cv + 1, tie, "put", values[5])),
        ("tie_right", event("index_tie", cv + 1, tie, "put", values[6])),
    ]
    # Vary actual competing histories without claiming greater dependency depth.
    mode = index % 3
    if mode >= 1:
        tagged.append(("lifecycle_previous", event("lifecycle", lv, early_delete - 1, "put", index + 300)))
    if mode >= 2:
        tagged.append(("tie_previous", event("index_tie", cv, tie - 1, "put", -index - 300)))
    rng.shuffle(tagged)
    specifications = [("delayed", delay - 1), ("delayed", delay), ("delayed", later),
                      ("lifecycle", early_delete + 1), ("lifecycle", lifecycle), ("lifecycle", null_time),
                      ("index_tie", tie), ("absent", null_time + 9 + index % 5)]
    return names, [item for _, item in tagged], [label for label, _ in tagged], specifications, rng.sample(range(8), 8), mode


def controlled_events(original, roles, variant):
    result = copy.deepcopy(original)
    if variant == "version_priority_raise":
        result[roles.index("later_lower")]["version"] = result[roles.index("delayed_update")]["version"] + 1
    elif variant == "lifecycle_op_toggle":
        result[roles.index("lifecycle_control")]["op"] = "put"
    elif variant == "index_tie_swap":
        left, right = roles.index("tie_left"), roles.index("tie_right")
        result[left], result[right] = result[right], result[left]
    return result


def render_task(index, names, events, specifications, permutation, rotation):
    key_names = {role: names[(position + rotation % 4) % 4] for position, role in enumerate(KEY_ROLES)}
    order = permutation[rotation:] + permutation[:rotation]
    queries, assignments = [], []
    for position, role_number in enumerate(order):
        key_role, as_of = specifications[role_number]
        query_id = f"q{(position + 2 * rotation + index) % 8}"
        queries.append(dict(query_id=query_id, key=key_names[key_role], as_of=as_of))
        assignments.append(dict(query_id=query_id, position=position, role=QUERY_ROLES[role_number]))
    return dict(kind="episodic", events=[{**item, "key": key_names[item["key"]]} for item in events], queries=queries), assignments, key_names


def property_witnesses(task, target):
    """Derive witnesses from visible events/queries, rather than role aliases."""
    events = task["events"]
    gold = {item["query_id"]: item for item in target["results"]}
    witnesses, infos = {}, []

    def tuple_at(index):
        item = events[index]
        return [item["version"], item["at"], index]

    def best(indices):
        return max(indices, key=tuple_at) if indices else None

    def evidence(info, competitors=()):
        return dict(query_signature={key: info["query"][key] for key in ("key", "as_of")},
                    query_id=info["query"]["query_id"], winner_index=info["winner"],
                    winner_tuple=tuple_at(info["winner"]) if info["winner"] is not None else None,
                    answer=info["answer"], competitor_indices=list(competitors),
                    competitor_tuples=[tuple_at(index) for index in competitors],
                    competitor_answers=[event_answer(events[index]) for index in competitors])

    for query in task["queries"]:
        eligible = [i for i, item in enumerate(events) if item["key"] == query["key"] and item["at"] <= query["as_of"]]
        winner = best(eligible)
        answer = event_answer(events[winner]) if winner is not None else dict(status="absent", value=None)
        expected = dict(query_id=query["query_id"], key=query["key"], **answer,
                        chosen_version=events[winner]["version"] if winner is not None else None, chosen_event=winner)
        assert canonical(expected) == canonical(gold[query["query_id"]])
        info = dict(query=query, eligible=eligible, winner=winner, answer=answer)
        infos.append(info)
        if winner is None:
            witnesses.setdefault("absence_control", {**evidence(info), "absence_kind": "future_only" if any(item["key"] == query["key"] for item in events) else "missing_key"})
        else:
            win = events[winner]
            latest = max(eligible, key=lambda i: (events[i]["at"], i))
            if win["version"] > events[latest]["version"] and win["at"] < events[latest]["at"] and answer_signature(answer) != answer_signature(event_answer(events[latest])):
                witnesses.setdefault("version_over_recency", evidence(info, [latest]))
            time_conflicts = [i for i in eligible if events[i]["version"] == win["version"] and events[i]["at"] < win["at"]
                              and answer_signature(event_answer(events[i])) != answer_signature(answer)]
            if time_conflicts:
                witnesses.setdefault("equal_version_time_conflict", evidence(info, [time_conflicts[0]]))
            input_conflicts = [i for i in eligible if i != winner and events[i]["version"] == win["version"] and events[i]["at"] == win["at"]
                               and answer_signature(event_answer(events[i])) != answer_signature(answer)]
            if input_conflicts:
                witnesses.setdefault("equal_version_time_input_conflict", evidence(info, [input_conflicts[0]]))
            lower_puts = [i for i in eligible if events[i]["op"] == "put" and events[i]["version"] < win["version"]]
            if win["op"] == "delete" and lower_puts:
                witnesses.setdefault("authoritative_tombstone", evidence(info, [lower_puts[0]]))
        unfiltered_time = best([i for i, item in enumerate(events) if item["key"] == query["key"]])
        if unfiltered_time is not None and events[unfiltered_time]["at"] > query["as_of"] and answer_signature(event_answer(events[unfiltered_time])) != answer_signature(answer):
            witnesses.setdefault("future_event_filter_control", evidence(info, [unfiltered_time]))
        unfiltered_key = best([i for i, item in enumerate(events) if item["at"] <= query["as_of"]])
        if unfiltered_key is not None and events[unfiltered_key]["key"] != query["key"] and answer_signature(event_answer(events[unfiltered_key])) != answer_signature(answer):
            witnesses.setdefault("cross_key_filter_control", evidence(info, [unfiltered_key]))
    for at in infos:
        winner = at["winner"]
        if winner is not None and at["query"]["as_of"] >= 1 and events[winner]["at"] == at["query"]["as_of"]:
            before = next((item for item in infos if item["query"]["key"] == at["query"]["key"] and item["query"]["as_of"] == at["query"]["as_of"] - 1), None)
            if before and answer_signature(before["answer"]) != answer_signature(at["answer"]):
                witnesses.setdefault("decisive_delayed_boundary", dict(before=evidence(before), at=evidence(at), boundary=at["query"]["as_of"]))
        if at["answer"]["status"] == "present" and at["answer"]["value"] is None:
            contrast = next((item for item in infos if item["answer"]["status"] in ("absent", "deleted") and item["answer"]["value"] is None), None)
            if contrast:
                witnesses.setdefault("present_null_distinction", dict(present_null=evidence(at), null_contrast=evidence(contrast)))
    return witnesses


def histogram_add(table, result):
    table["status"][result["status"]] += 1
    table["value_type"][typed(result["value"])] += 1
    table["present_null"][str(result["status"] == "present" and result["value"] is None).lower()] += 1
    table["typed_answer"][answer_signature(result)] += 1
    table["provenance"][canonical(dict(chosen_version=result["chosen_version"], chosen_event=result["chosen_event"]))] += 1


def new_histograms():
    return {name: Counter() for name in ("status", "value_type", "present_null", "typed_answer", "provenance")}


def results_by_role(row):
    answers = {result["query_id"]: result for result in row["verified_target"]["results"]}
    return {item["role"]: answers[item["query_id"]] for item in row["query_role_assignments"]}


def verify(rows, parents):
    sets, twin_sets = defaultdict(list), defaultdict(list)
    properties, split_status, prompts, task_hashes = Counter(), defaultdict(Counter), set(), set()
    for row in rows:
        assert row["record_hash"] == record_hash(row) and row["review_status"] == "PENDING" and not row["review_receipt_ids"]
        assert row["split"] == split_for(row["source_family_index"])
        _, payload = row["rendered_input"].split("Task JSON:\n")
        assert canonical(json.loads(payload)) == canonical(row["semantic_task"])
        assert row["source_family_key"] not in row["rendered_input"] and row["sample_id"] not in row["rendered_input"]
        computed = property_witnesses(row["semantic_task"], row["verified_target"])
        assert canonical(computed) == canonical(row["property_witnesses"])
        assert set(row["skill_tags"]) == {"finite_synthetic_proxy", "versioned_retrieval", *computed}
        assert set(computed) == set(PROPERTIES)
        assert {result["status"] for result in row["verified_target"]["results"]} == {"present", "deleted", "absent"}
        properties.update(computed.keys())
        split_status[row["split"]].update(result["status"] for result in row["verified_target"]["results"])
        sets[row["source_family_key"], row["controlled_variant"], row["renderer"]].append(row)
        twin_sets[row["source_family_key"], row["variant_key"]].append(row)
        assert row["rendered_input"] not in prompts
        prompts.add(row["rendered_input"])
        task_hashes.add(digest(row["semantic_task"]))
    for twins in twin_sets.values():
        assert len(twins) == 2
        assert canonical(twins[0]["semantic_task"]) == canonical(twins[1]["semantic_task"])
        assert canonical(twins[0]["verified_target"]) == canonical(twins[1]["verified_target"])
    crossing_reports = []
    for (source, history, style), records in sorted(sets.items()):
        assert len(records) == 8
        id_roles, position_roles, key_groups = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        ids, positions, keys = defaultdict(new_histograms), defaultdict(new_histograms), defaultdict(new_histograms)
        assignment_table = []
        for row in sorted(records, key=lambda item: item["query_rotation"]):
            answers = {result["query_id"]: result for result in row["verified_target"]["results"]}
            assignment_table.append(dict(sample_id=row["sample_id"], query_rotation=row["query_rotation"], key_rotation=row["key_rotation"],
                                         queries=row["query_role_assignments"], key_groups=row["key_role_assignments"]))
            for assignment in row["query_role_assignments"]:
                qid, pos, role = assignment["query_id"], str(assignment["position"]), assignment["role"]
                result = answers[qid]
                id_roles[qid][role] += 1
                position_roles[pos][role] += 1
                histogram_add(ids[qid], result)
                histogram_add(positions[pos], result)
                histogram_add(keys[result["key"]], result)
            for role, name in row["key_role_assignments"].items():
                key_groups[name][role] += 1
        assert all(table == Counter(QUERY_ROLES) for table in id_roles.values())
        assert all(table == Counter(QUERY_ROLES) for table in position_roles.values())
        assert all(table == Counter({role: 2 for role in KEY_ROLES}) for table in key_groups.values())
        assert all(table == next(iter(ids.values())) for table in ids.values())
        assert all(table == next(iter(positions.values())) for table in positions.values())
        assert all(table == next(iter(keys.values())) for table in keys.values())
        assert all(set(table["status"]) == {"present", "deleted", "absent"} for table in ids.values())
        crossing_reports.append(dict(source_family_key=source, controlled_history=history, field_order=style,
                                     assignment_table=assignment_table, role_by_id=id_roles, role_by_position=position_roles,
                                     groups_by_key=key_groups, actual_histograms_by_id=ids, actual_histograms_by_position=positions,
                                     actual_histograms_by_key=keys, id_outcomes_equal=True, position_outcomes_equal=True,
                                     key_outcomes_equal=True, measured_residual_arbitrary_name_or_position_difference=0))
    return dict(producer_checks_passed=True, source_family_count=len(parents), rendered_record_count=len(rows),
                controlled_histories=len(parents) * 4, unique_correlated_task_instances=len(task_hashes), unique_prompts=len(prompts),
                rendered_query_observations=sum(len(row["semantic_task"]["queries"]) for row in rows),
                observed_property_row_counts=dict(properties), rendered_status_counts_by_split={key: dict(value) for key, value in split_status.items()},
                rows_with_present_deleted_absent=len(rows), exact_crossing_sets=len(crossing_reports),
                properties_from_in_row_witnesses_only=True, independent_review_or_approval_performed=False), crossing_reports


def paired_controls(rows):
    forward = {(row["source_family_key"], row["controlled_variant"], row["query_rotation"]): row for row in rows if row["renderer"] == "forward"}
    pairs = []
    for (source, variant, rotation), baseline in sorted(forward.items()):
        if variant != "baseline":
            continue
        for other_name, control, role in [("version_priority_raise", "version_priority_counterfactual", "delayed_later"),
                                          ("lifecycle_op_toggle", "null_lifecycle_counterfactual", "lifecycle_at"),
                                          ("index_tie_swap", "active_input_tie_swap_counterfactual", "same_version_time_index")]:
            other = forward[source, other_name, rotation]
            a, b = results_by_role(baseline)[role], results_by_role(other)[role]
            assert answer_signature(a) != answer_signature(b)
            before, after = baseline["semantic_task"], other["semantic_task"]
            assert before["queries"] == after["queries"]
            changes = [(i, key, old[key], new[key]) for i, (old, new) in enumerate(zip(before["events"], after["events"])) for key in old if canonical(old[key]) != canonical(new[key])]
            if control == "version_priority_counterfactual":
                assert len(changes) == 1 and changes[0][1] == "version"
                assert changes[0][2] < a["chosen_version"] < changes[0][3]
            elif control == "null_lifecycle_counterfactual":
                assert len(changes) == 1 and changes[0][1] == "op" and a["value"] is b["value"] is None
            else:
                indices = sorted({change[0] for change in changes})
                assert len(indices) == 2
                assert before["events"][indices[0]] == after["events"][indices[1]]
                assert before["events"][indices[1]] == after["events"][indices[0]]
            pairs.append(dict(control=control, scope="corpus_variant", source_family_key=source, query_rotation=rotation,
                              baseline_sample_id=baseline["sample_id"], counterfactual_sample_id=other["sample_id"],
                              protected_query_signature=next({key: query[key] for key in ("key", "as_of")} for query in before["queries"] if query["query_id"] == a["query_id"]),
                              event_field_changes=[dict(event_index=i, field=key, before=old, after=new) for i, key, old, new in changes],
                              before_result=a, after_result=b, semantic_answer_changed=True))
    return pairs


def audit_probes(rows):
    probes = []
    for baseline in rows:
        if baseline["controlled_variant"] != "baseline" or baseline["query_rotation"] != 0 or baseline["renderer"] != "forward":
            continue
        task, answers = baseline["semantic_task"], results_by_role(baseline)
        for control in PROBE_CONTROLS:
            role = "delayed_before" if control == "future_event_edit_invariance" else "delayed_later"
            original = answers[role]
            query = next(query for query in task["queries"] if query["query_id"] == original["query_id"])
            edited = copy.deepcopy(task)
            if control == "lower_version_append_invariance":
                edited["events"].append(event(query["key"], original["chosen_version"] - 1, query["as_of"], "put", -1000 - baseline["source_family_index"]))
            else:
                indices = [i for i, item in enumerate(edited["events"]) if (item["at"] > query["as_of"] if control == "future_event_edit_invariance" else item["key"] != query["key"])]
                assert indices
                edited["events"][indices[0]]["op"] = "put"
                edited["events"][indices[0]]["value"] = -2000 - baseline["source_family_index"]
            new_target = solve(edited)
            result = next(item for item in new_target["results"] if item["query_id"] == query["query_id"])
            assert canonical(result) == canonical(original)
            probes.append(dict(control=control, scope="audit_probe", source_family_key=baseline["source_family_key"],
                               source_sample_id=baseline["sample_id"], protected_query_signature={key: query[key] for key in ("key", "as_of")},
                               edited_task=edited, producer_reference_target=new_target, exact_protected_result_unchanged=True,
                               added_training_records=0, added_source_families=0))
    return probes


def frozen_files():
    candidates = [ROOT / "generators/memory/history_gap.py"]
    for name in ("semantic_parents", "candidate_renderings"):
        candidates.append(ROOT / f"data/{TRACK}/{name}/memory-gap-v1.jsonl")
    for suffix in ("generation", "producer_integrity"):
        candidates.append(ROOT / f"data/{TRACK}/manifests/memory-gap-v1.{suffix}.json")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}


def generate(count=12, batch="memory-gap-v2", offset=12):
    if count < 0 or offset < 12 or not batch or "/" in batch or "\\" in batch:
        raise ValueError("Use nonnegative count, source offset>=12 and a fresh filename-safe batch")
    before = frozen_files()
    prompt_contract = contracts()["episodic"]
    rows, parents = [], []
    for index in range(offset, offset + count):
        split = split_for(index)
        family_id, source = f"{TRACK}-{batch}-p{index:04d}", f"REM-seed{SEED}-p{index:04d}"
        names, original, event_roles, specifications, permutation, mode = family_history(index, split)
        tasks, keys = [], []
        for variant_number, variant in enumerate(VARIANTS):
            for rotation in range(8):
                task, assignments, key_roles = render_task(index, names, controlled_events(original, event_roles, variant), specifications, permutation, rotation)
                target = solve(task)
                witnesses = property_witnesses(task, target)
                variant_key = f"v{variant_number}-qrot{rotation}"
                tasks.append(task)
                keys.append(variant_key)
                for renderer in ("forward", "reverse"):
                    prompt = "Executable task contract:\n" + prompt_contract + "Task JSON:\n" + json.dumps(task if renderer == "forward" else reorder(task), ensure_ascii=False, separators=(",", ":"))
                    row = record(TRACK, batch, index, variant_key, task, prompt, target,
                                 ["finite_synthetic_proxy", "versioned_retrieval"] + sorted(witnesses), 3, "structured", renderer)
                    row.update(source_family_key=source, source_family_index=index, variant_key=variant_key,
                               controlled_variant=variant, counterfactual_baseline="baseline", query_rotation=rotation, key_rotation=rotation % 4,
                               structural_difficulty=3, challenge_axes=["heldout_key_and_value_vocabulary"] if split == "dev_challenge" else [],
                               query_role_assignments=assignments, key_role_assignments=key_roles, property_witnesses=witnesses,
                               family_control_claims=[dict(name=name, scope="corpus_variant", source_family_key=source,
                                                           evidence_manifest=f"data/{TRACK}/manifests/{batch}.family_controls.json") for name in CORPUS_CONTROLS] +
                                                     [dict(name=name, scope="audit_probe", source_family_key=source,
                                                           evidence_manifest=f"data/{TRACK}/manifests/{batch}.family_controls.json") for name in PROBE_CONTROLS],
                               producer_role="memory_producer", producer_agent_id="/root/relational_critic_producer",
                               scope_limitation="Finite supplied histories and correlated control views; learned-memory efficacy remains unsupported.")
                    row["record_hash"] = record_hash(row)
                    rows.append(row)
        parents.append(dict(semantic_parent_id=family_id, semantic_family_id=family_id, source_family_key=source, source_family_index=index,
                            split=split, history_template=mode, tasks=tasks, task_variant_keys=keys, controlled_histories_per_family=4,
                            correlated_query_key_rotations_per_history=8, field_order_renderers=2, structural_difficulty=3,
                            revises_pending_batch="memory-gap-v1", counts_as_new_source_family=False))
    report, crossing = verify(rows, parents)
    pairs, probes = paired_controls(rows), audit_probes(rows)
    prior_hashes = set()
    for prior_batch in ("extended-pilot-v1", "extended-pilot-v2"):
        path = ROOT / f"data/{TRACK}/semantic_parents/{prior_batch}.jsonl"
        if path.exists():
            prior_hashes.update(digest(task) for parent in load_jsonl(path) for task in parent["tasks"])
    assert not prior_hashes.intersection(digest(row["semantic_task"]) for row in rows)
    report.update(supplied_counterfactual_pairs=dict(Counter(pair["control"] for pair in pairs)),
                  audit_probe_counts=dict(Counter(probe["control"] for probe in probes)), frozen_v1_sha256=before,
                  exact_overlap_with_own_pre_gap_semantic_tasks=0, correlated_views_not_additional_families=True,
                  metadata_revision_guidance="Public property names replace aliases; every control history retains deleted answers through a separately queried tombstone. v1 stays frozen.",
                  limitations=["12 source families, 48 controlled histories, 384 query/key rotations and 768 renderings are correlated.",
                               "Each query is an independent finite argmax; eight queries and longer timestamps do not establish greater dependency depth.",
                               "Histories contain 12–14 events; persistent learned memory, long-context retention, semantic relevance and efficacy remain gaps."])
    source_paths = [Path(__file__), ROOT / "generators/memory/history_gap.py", ROOT / "generators/other_tracks/reference.py",
                    ROOT / "schemas/EXTENDED_SEMANTICS.md", ROOT / "schemas/MEMORY_CONTROL_PROPERTIES.md", ROOT / "research/curricula/REM-DELAYED-HISTORY.md"]
    config = dict(batch=batch, track_id=TRACK, seed=SEED, producer_role="memory_producer", producer_agent_id="/root/relational_critic_producer",
                  count=count, offset=offset, source_family_indices=list(range(offset, offset + count)), source_family_count=count,
                  pending_source_lineage="Same12 pending families as memory-gap-v1; no accepted source records modified", controlled_histories_per_family=4,
                  query_rotations_per_history=8, field_order_renderers=2, expected_record_count=count * 64,
                  query_roles=list(QUERY_ROLES), key_roles=list(KEY_ROLES), history_variants=list(VARIANTS),
                  split_assignment="factory.common.split_for(source_family_index), before variants", label_reference="generators.other_tracks.reference.solve",
                  input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                  command=f"/opt/anaconda3/bin/python3 -m generators.memory.history_gap_v2 --count {count} --batch {batch} --offset {offset}",
                  status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW")
    family_evidence = dict(crossings=crossing, supplied_counterfactual_pairs=pairs, audit_probes=probes,
                           field_order_pair_count=len(rows) // 2, audit_probes_add_training_records=0, audit_probes_add_source_families=0)
    output = [(ROOT / f"data/{TRACK}/semantic_parents/{batch}.jsonl", parents, True),
              (ROOT / f"data/{TRACK}/candidate_renderings/{batch}.jsonl", rows, True),
              (ROOT / f"data/{TRACK}/manifests/{batch}.generation.json", config, False),
              (ROOT / f"data/{TRACK}/manifests/{batch}.producer_integrity.json", report, False),
              (ROOT / f"data/{TRACK}/manifests/{batch}.family_controls.json", family_evidence, False)]
    for path, value, jsonl in output:
        if path.exists() and (load_jsonl(path) if jsonl else json.loads(path.read_text())) != value:
            raise ValueError(f"Refusing to change existing batch file {path}; choose a fresh version")
    for path, value, jsonl in output:
        (write_jsonl if jsonl else write_json)(path, value)
    assert before == frozen_files()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--batch", default="memory-gap-v2")
    parser.add_argument("--offset", type=int, default=12)
    args = parser.parse_args()
    result = generate(args.count, args.batch, args.offset)
    print(f"Produced {len(result)} provisional REM rows in {args.batch} from {args.count} existing pending source families")
