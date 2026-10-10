"""REM-only delayed-history production; never imports an independent oracle.

Four meaningful controlled histories share each source family. Eight query/key
rotations and two field orders are correlated renderings, not new families.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from factory.common import ROOT, canonical, digest, load_jsonl, record, record_hash, split_for, write_json, write_jsonl
from generators.other_tracks.reference import contracts, solve


SEED = 20261009
TRACK = "REM-001"
QUERY_ROLES = ("delayed_before", "delayed_at", "delayed_older_later", "lifecycle_before",
               "tombstone_at", "put_null_at", "index_tie_at", "absent_key")
KEY_ROLES = ("delayed", "lifecycle", "index_tie", "absent")
VARIANTS = ("baseline", "delayed_tombstone", "tombstone_to_put_null", "index_tie_swap")
TRAIN_KEYS = ("parcel", "signal", "cabinet", "badge", "ledger", "button", "canvas", "hinge", "vessel", "window",
              "beacon", "pocket", "bridge", "panel", "socket", "basket", "rivet", "column", "lantern", "ticket")
DEV_KEYS = ("nectar", "prism", "ribbon", "tile", "copper", "ripple", "arch", "cove", "fable", "petal",
            "willow", "mosaic", "coral", "dusk", "meadow", "fern", "orbit", "pebble", "tulip", "acorn")
TRAIN_VALUES = ("amber", "cobalt", "jade", "opal", "violet", "ivory", "moss", "sienna", "azure", "ochre")
DEV_VALUES = ("granite", "marble", "basalt", "garnet", "agate", "flint", "slate", "jasper", "topaz", "quartz")


def reorder(value):
    if isinstance(value, dict):
        return {key: reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(item) for item in value]
    return value


def event(key, version, at, op, value):
    return dict(key=key, version=version, at=at, op=op, value=value)


def family_history(index, split):
    rng = random.Random(SEED + 97 * index)
    names = rng.sample(DEV_KEYS if split == "dev_challenge" else TRAIN_KEYS, 4)
    values = rng.sample(DEV_VALUES if split == "dev_challenge" else TRAIN_VALUES, 8)
    start = 7 + (index - 12) * 3
    delay = start + 8 + 3 * (index % 5)
    tomb = delay + 5 + index % 3
    put_null, tie, older_later = tomb + 1, delay - 3, delay + 2 + index % 4
    dv, lv, cv = 1 + index % 4, 2 + index % 3, 100 + index
    before_value = (values[0], -index, bool(index % 2))[index % 3]
    delayed_value = (index + 20, bool(index % 2), values[1])[index % 3]
    tagged = [
        ("delayed_initial", event("delayed", dv, start, "put", before_value)),
        ("delayed_update", event("delayed", dv + 2, delay, "put", delayed_value)),
        ("older_later", event("delayed", dv + 1, older_later, "put", values[2])),
        ("lifecycle_initial", event("lifecycle", lv, start + 2, "put", values[3])),
        ("tombstone", event("lifecycle", lv + 1, tomb, "delete", None)),
        ("put_null", event("lifecycle", lv + 2, put_null, "put", None)),
        ("tie_initial", event("index_tie", cv, start + 1, "put", values[4])),
        ("tie_left", event("index_tie", cv + 1, tie, "put", values[5])),
        ("tie_right", event("index_tie", cv + 1, tie, "put", values[6])),
    ]
    mode = index % 4
    if mode >= 1:
        tagged.append(("delayed_previous_update", event("delayed", dv + 1, delay - 2, "put", values[7])))
    if mode >= 2:
        tagged.append(("lifecycle_previous_update", event("lifecycle", lv, tomb - 2, "put", index + 300)))
    if mode >= 3:
        tagged.append(("tie_previous_update", event("index_tie", cv, tie - 1, "put", -index - 300)))
    rng.shuffle(tagged)
    roles = [label for label, _ in tagged]
    events = [item for _, item in tagged]
    specifications = [
        ("delayed", delay - 1), ("delayed", delay), ("delayed", older_later),
        ("lifecycle", tomb - 1), ("lifecycle", tomb), ("lifecycle", put_null),
        ("index_tie", tie), ("absent", put_null + 9 + index % 5),
    ]
    query_permutation = rng.sample(range(len(QUERY_ROLES)), len(QUERY_ROLES))
    return names, events, roles, specifications, query_permutation, mode


def controlled_events(events, roles, variant):
    changed = copy.deepcopy(events)
    if variant == "delayed_tombstone":
        update = changed[roles.index("delayed_update")]
        update.update(op="delete", value=None)
    elif variant == "tombstone_to_put_null":
        changed[roles.index("tombstone")]["op"] = "put"
    elif variant == "index_tie_swap":
        left, right = roles.index("tie_left"), roles.index("tie_right")
        changed[left], changed[right] = changed[right], changed[left]
    return changed


def render_task(index, names, events, specifications, permutation, rotation):
    key_rotation = rotation % len(KEY_ROLES)
    key_names = {role: names[(position + key_rotation) % len(names)] for position, role in enumerate(KEY_ROLES)}
    rendered_events = [{**item, "key": key_names[item["key"]]} for item in events]
    order = permutation[rotation:] + permutation[:rotation]
    queries, assignments = [], []
    for position, role_number in enumerate(order):
        key_role, as_of = specifications[role_number]
        query_id = f"q{(position + 2 * rotation + index) % len(QUERY_ROLES)}"
        queries.append(dict(query_id=query_id, key=key_names[key_role], as_of=as_of))
        assignments.append(dict(query_id=query_id, position=position, role=QUERY_ROLES[role_number]))
    return dict(kind="episodic", events=rendered_events, queries=queries), assignments, key_names


def result_signature(result):
    return canonical({key: value for key, value in result.items() if key not in ("query_id", "key")})


def verify(rows, parents):
    semantic_hashes, prompts = set(), set()
    family_variants = defaultdict(list)
    by_split = defaultdict(Counter)
    for row in rows:
        assert row["record_hash"] == record_hash(row)
        assert row["review_status"] == "PENDING" and not row["review_receipt_ids"]
        assert row["difficulty"] == row["structural_difficulty"] == 3
        assert row["split"] == split_for(row["source_family_index"])
        prefix, payload = row["rendered_input"].split("Task JSON:\n")
        assert json.loads(payload) == row["semantic_task"]
        assert row["source_family_key"] not in row["rendered_input"]
        assert row["sample_id"] not in row["rendered_input"]
        assert row["semantic_parent_id"] not in row["rendered_input"]
        assert "greatest tuple (version,at,zero_based_input_event_index)" in prefix
        assert row["rendered_input"] not in prompts
        prompts.add(row["rendered_input"])
        semantic_hashes.add(digest(row["semantic_task"]))
        if row["renderer"] == "forward":
            family_variants[row["source_family_key"], row["controlled_variant"]].append(row)
            results = {item["query_id"]: item for item in row["verified_target"]["results"]}
            roles = {item["role"]: results[item["query_id"]] for item in row["query_role_assignments"]}
            assert (roles["delayed_before"]["status"], canonical(roles["delayed_before"]["value"])) != \
                   (roles["delayed_at"]["status"], canonical(roles["delayed_at"]["value"]))
            assert roles["delayed_at"] | {"query_id": "ignored"} == roles["delayed_older_later"] | {"query_id": "ignored"}
            assert roles["absent_key"]["status"] == "absent"
            assert roles["put_null_at"]["status"] == "present" and roles["put_null_at"]["value"] is None
            assert roles["tombstone_at"]["status"] == ("present" if row["controlled_variant"] == "tombstone_to_put_null" else "deleted")
            assert roles["index_tie_at"]["chosen_event"] == max(row["tie_event_indices"])
            # The selected high-version event for another queried key was already
            # eligible at all delayed queries, so filtering the key is necessary.
            assert roles["index_tie_at"]["chosen_version"] > roles["delayed_at"]["chosen_version"]
            assert row["semantic_task"]["events"][roles["index_tie_at"]["chosen_event"]]["at"] <= \
                   next(q["as_of"] for q in row["semantic_task"]["queries"]
                        if q["query_id"] == roles["delayed_before"]["query_id"])
            by_split[row["split"]].update(item["status"] for item in results.values())
    for (family, variant), records in family_variants.items():
        assert len(records) == len(QUERY_ROLES)
        id_role, position_role = defaultdict(Counter), defaultdict(Counter)
        id_labels, position_labels = defaultdict(Counter), defaultdict(Counter)
        key_roles = defaultdict(Counter)
        for row in records:
            results = {item["query_id"]: item for item in row["verified_target"]["results"]}
            for assignment in row["query_role_assignments"]:
                qid, position, role = assignment["query_id"], assignment["position"], assignment["role"]
                id_role[qid][role] += 1
                position_role[position][role] += 1
                signature = result_signature(results[qid])
                id_labels[qid][signature] += 1
                position_labels[position][signature] += 1
            for role, name in row["key_role_assignments"].items():
                key_roles[name][role] += 1
        assert all(table == Counter(QUERY_ROLES) for table in id_role.values())
        assert all(table == Counter(QUERY_ROLES) for table in position_role.values())
        assert all(table == next(iter(id_labels.values())) for table in id_labels.values())
        assert all(table == next(iter(position_labels.values())) for table in position_labels.values())
        assert all(table == Counter({role: 2 for role in KEY_ROLES}) for table in key_roles.values())
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["semantic_parent_id"], row["variant_key"]].append(row)
    for pair in pairs.values():
        assert len(pair) == 2 and pair[0]["semantic_task"] == pair[1]["semantic_task"]
        assert pair[0]["verified_target"] == pair[1]["verified_target"] and pair[0]["split"] == pair[1]["split"]
    # Counterfactual effects must change status or typed value; provenance alone
    # is insufficient. Compare like-for-like rotated histories.
    effects = Counter()
    for parent in parents:
        source = parent["source_family_key"]
        baseline = {r["query_rotation"]: r for r in family_variants[source, "baseline"]}
        for variant, role in [("delayed_tombstone", "delayed_at"), ("tombstone_to_put_null", "tombstone_at"), ("index_tie_swap", "index_tie_at")]:
            for row in family_variants[source, variant]:
                other = baseline[row["query_rotation"]]
                qid = next(item["query_id"] for item in row["query_role_assignments"] if item["role"] == role)
                a = next(item for item in row["verified_target"]["results"] if item["query_id"] == qid)
                b = next(item for item in other["verified_target"]["results"] if item["query_id"] == qid)
                assert (a["status"], canonical(a["value"])) != (b["status"], canonical(b["value"]))
                effects[variant] += 1
    return dict(producer_checks_passed=True, source_family_count=len(parents), rendered_record_count=len(rows),
                controlled_history_variants=len(parents) * len(VARIANTS), correlated_task_instances=len(semantic_hashes),
                unique_prompts=len(prompts), meaningful_counterfactual_pairs=dict(effects),
                forward_query_status_counts_by_split={key: dict(value) for key, value in by_split.items()},
                exact_query_role_crossing_every_id_and_position=True, exact_label_distribution_crossing_every_id_and_position=True,
                ordinary_key_role_crossing_every_visible_key=True, before_at_delayed_status_or_typed_value_change=True,
                first_party_label_computation_only=True, independent_review_or_approval_performed=False,
                scope_limitations=["Finite deterministic versioned retrieval; no persistent-memory learning or efficacy established.",
                                   "Query/key rotations and field-order renderings are correlated variants, never additional source families.",
                                   "Histories contain 9–12 events; long-context retention and resource-limited retrieval remain unsupported.",
                                   "Development holds out visible vocabulary; structural difficulty is 3 in both splits."])


def generate(count=12, batch="memory-gap-v1", offset=12):
    if count < 0 or offset < 12 or not batch or "/" in batch or "\\" in batch:
        raise ValueError("Use nonnegative count, source offset>=12, and a filename-safe fresh batch")
    prompt_contract = contracts()["episodic"]
    rows, parents = [], []
    for index in range(offset, offset + count):
        split = split_for(index)  # Assigned at source-family creation, before variants.
        family_id = f"{TRACK}-{batch}-p{index:04d}"
        source_key = f"REM-seed{SEED}-p{index:04d}"
        names, original_events, event_roles, specifications, permutation, mode = family_history(index, split)
        tasks, task_keys = [], []
        for variant_number, variant in enumerate(VARIANTS):
            events = controlled_events(original_events, event_roles, variant)
            for rotation in range(len(QUERY_ROLES)):
                task, query_roles, key_roles = render_task(index, names, events, specifications, permutation, rotation)
                target = solve(task)  # Producer's local reference, not an independent verifier.
                variant_key = f"v{variant_number}-qrot{rotation}"
                tasks.append(task)
                task_keys.append(variant_key)
                for renderer in ("forward", "reverse"):
                    payload = task if renderer == "forward" else reorder(task)
                    prompt = "Executable task contract:\n" + prompt_contract + "Task JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    row = record(TRACK, batch, index, variant_key, task, prompt, target,
                                 ["finite_synthetic_proxy", "versioned_retrieval", "delayed_dependency", "decisive_asof_boundary",
                                  "version_first", "tombstone_vs_put_null", "same_time_index_tie", "wrong_key_suppression",
                                  "absent_key_control", "exact_query_role_crossing"], 3, "structured", renderer)
                    row.update(source_family_key=source_key, source_family_index=index, variant_key=variant_key,
                               controlled_variant=variant, counterfactual_baseline="baseline", query_rotation=rotation,
                               key_rotation=rotation % 4, structural_difficulty=3,
                               challenge_axes=["heldout_key_and_value_vocabulary"] if split == "dev_challenge" else [],
                               query_role_assignments=query_roles, key_role_assignments=key_roles,
                               tie_event_indices=[event_roles.index("tie_left"), event_roles.index("tie_right")],
                               producer_role="memory_producer", producer_agent_id="/root/relational_critic_producer",
                               scope_limitation="Finite supplied histories; correlated controlled variants; no learned-memory efficacy claim.")
                    row["record_hash"] = record_hash(row)
                    rows.append(row)
        parents.append(dict(semantic_parent_id=family_id, semantic_family_id=family_id, source_family_key=source_key,
                            source_family_index=index, split=split, history_template=mode, tasks=tasks, task_variant_keys=task_keys,
                            controlled_history_variants_per_family=4, correlated_query_key_rotations_per_variant=8,
                            field_order_renderers_per_task=2, structural_difficulty=3))
    report = verify(rows, parents)
    # Compare only the producer's own older REM semantic parents, not reviewer files.
    prior_hashes = set()
    preserved = {}
    for prior_batch in ("extended-pilot-v1", "extended-pilot-v2"):
        for directory in ("semantic_parents", "candidate_renderings"):
            path = ROOT / f"data/{TRACK}/{directory}/{prior_batch}.jsonl"
            if path.exists():
                preserved[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                if directory == "semantic_parents":
                    prior_hashes.update(digest(task) for parent in load_jsonl(path) for task in parent["tasks"])
    assert not prior_hashes.intersection(digest(row["semantic_task"]) for row in rows)
    report.update(exact_task_overlap_with_own_prior_batches=0, preserved_prior_file_sha256=preserved)
    config = dict(batch=batch, track_id=TRACK, seed=SEED, producer_agent_id="/root/relational_critic_producer",
                  producer_role="memory_producer", count=count, offset=offset, source_family_indices=list(range(offset, offset + count)),
                  source_family_count=count, controlled_history_variants_per_family=4, query_rotations_per_variant=8,
                  field_order_renderers=2, expected_record_count=count * 64,
                  split_assignment="factory.common.split_for(source_family_index), before variants", query_roles=list(QUERY_ROLES),
                  key_roles=list(KEY_ROLES), variants=list(VARIANTS), source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  public_contract_sha256=hashlib.sha256((ROOT / "schemas/EXTENDED_SEMANTICS.md").read_bytes()).hexdigest(),
                  label_reference="generators.other_tracks.reference.solve", label_reference_sha256=hashlib.sha256((ROOT / "generators/other_tracks/reference.py").read_bytes()).hexdigest(),
                  command=f"/opt/anaconda3/bin/python3 -m generators.memory.history_gap --count {count} --batch {batch} --offset {offset}",
                  status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW")
    outputs = [(ROOT / f"data/{TRACK}/semantic_parents/{batch}.jsonl", parents, True),
               (ROOT / f"data/{TRACK}/candidate_renderings/{batch}.jsonl", rows, True),
               (ROOT / f"data/{TRACK}/manifests/{batch}.generation.json", config, False),
               (ROOT / f"data/{TRACK}/manifests/{batch}.producer_integrity.json", report, False)]
    for path, value, jsonl in outputs:
        if path.exists() and (load_jsonl(path) if jsonl else json.loads(path.read_text())) != value:
            raise ValueError(f"Refusing to alter existing batch file {path}; use a fresh version")
    for path, value, jsonl in outputs:
        (write_jsonl if jsonl else write_json)(path, value)
    for relative, previous_hash in preserved.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == previous_hash
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--batch", default="memory-gap-v1")
    parser.add_argument("--offset", type=int, default=12)
    args = parser.parse_args()
    generated = generate(args.count, args.batch, args.offset)
    print(f"Produced {len(generated)} provisional REM records from {args.count} source families")
