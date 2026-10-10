#!/opt/anaconda3/bin/python3
"""Expert V4 INPUT ONLY; own full recomputation before canonical access.

This wrapper reuses only the expert's own public-derived exact solver.  Root's
lineage is a join pointer, never evidence that the task/prompt actually match.
No producer, formal checker, candidate, custody or other reviewer is imported.
"""
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import pathlib
import platform
import resource
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/expert"
ROLE = "/root/expert_reviewer"
PREFIX = "belief-gap-v4-input-attempt01"
FIELDS = {"sample_id", "record_hash", "semantic_family_id", "semantic_parent_id",
          "split", "track_id", "semantic_task", "rendered_input"}
EXPOSURE = ["same underlying model as other roles; no model-independence claim",
            "public-guided corrective metadata repair of previously seen cases",
            "own V3 canonical gold exposure from earlier completed review",
            "reuse of own frozen V1 solver; every actual V4 task recomputed twice",
            "old own predictions crosschecked only after exact typed task, raw prompt, split and track join",
            "V4 canonical, producer metadata and all other reviewers unread in this stage",
            "public V4 documentation discloses historical outcome summaries indirectly"]
BINDINGS = {
    "reviews/input_only/belief-gap-v4.jsonl": "ad3c3f79a3547474476a6572c14d1e12bbe3f94e18468dd610638efdbe6d54b5",
    "reviews/input_only/belief-gap-v4.manifest.json": "dad2b33321ae1baca05120ccdddadce97d6f8fff9ae10d7736a12c1d065a0d39",
    "reviews/input_only/belief-gap-v4.input_lineage.json": "a8fe0ee79f395e6ed5ce8c5e10c47a1c79f9b63aa1ea360440b6ecda3c48ca0a",
    "reviews/input_only/belief-gap-v3.jsonl": "50b3f47bce29e8a639f298ece104744be1583d04c04d3041d940219d6998b694",
    "reviews/expert/belief-gap-v3.input_only_predictions.json": "73ea6e8fb04fc4790082988d6ab23a26e4aa2aeab93889d21c0d1c2d6cdc1d43",
    "reviews/expert/belief-gap-v3.first_judgments.jsonl": "c3d216ca74f3d332aa70bd25de21ea934c4e014e4f5934c24dfa90c8612cac0f",
    "reviews/expert/prepare_belief_gap_v1.py": "e55b7ee49b25b0c4857a3387c730501bfda4ad0572906ce13be9c23a9c11d784",
    "reviews/expert/belief-v4-prospective-interface-toys.py": "4480f30fef7bcad66f50b659d75f807fbc7b55dc42232df1dc37f9db6c9f9c8b",
    "reviews/expert/belief-v4-prospective-review-procedure.md": "ac8df46f90f8126a803f20020d7dd4c1e51ea6f44de259f485dd889f5805c700",
    "schemas/EXTENDED_SEMANTICS.md": "2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23",
    "schemas/BELIEF_CONTROL_PROPERTIES.md": "200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034",
    "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md": "46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f",
    "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md": "3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11",
    "research/curricula/RBM-EVIDENCE-GAP.md": "004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120",
    "schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md": "40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce",
    "research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json": "2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d",
    "schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md": "925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c",
    "research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json": "2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625",
}


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(v):
    return hashlib.sha256(v).hexdigest()


def hash_path(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "length_bytes": size}


def digest(v):
    return sha_bytes(canon(v).encode())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def strict_loads(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate JSON object key: " + k)
            out[k] = v
        return out
    def reject(v):
        raise ValueError("nonfinite JSON value: " + v)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        f.write(canon(value) + "\n")


def bindings():
    actual = {p: hash_path(ROOT / p) for p in BINDINGS}
    for p, expected in BINDINGS.items():
        if actual[p]["sha256"] != expected:
            raise AssertionError("changed authorized dependency: " + p)
    return actual


def read_view(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = strict_loads(line)
            assert set(row) == FIELDS, (line_no, "view whitelist")
            sid = row["sample_id"]
            assert isinstance(sid, str) and sid not in rows, (line_no, "duplicate sample ID")
            assert isinstance(row["record_hash"], str) and len(row["record_hash"]) == 64
            assert all(c in "0123456789abcdef" for c in row["record_hash"])
            assert all(isinstance(row[k], str) for k in FIELDS - {"semantic_task"})
            rows[sid] = row
    return rows


def input_subset(row):
    return {k: row[k] for k in ["semantic_task", "rendered_input", "split", "track_id"]}


def identity(row):
    return {k: row[k] for k in ["sample_id", "record_hash", "semantic_family_id", "semantic_parent_id", "split", "track_id"]}


def query(qid, counts, entailed, guaranteed):
    return {"query_id": qid, "counts": counts, "distinct_values": len(counts),
            "entailed": entailed, "guaranteed": guaranteed,
            "can_reduce_uncertainty": len(counts) > 1}


def calibration(own):
    # All exact expected outputs below were hand-derived before execution.
    base = {"kind": "belief", "worlds": [
        {"world_id": "z", "facts": {"x": True, "q": None}},
        {"world_id": "a", "facts": {"x": 1, "q": "null"}},
        {"world_id": "m", "facts": {"x": None, "q": None}}],
        "observations": [], "queries": [{"query_id": "qx", "fact": "x"}, {"query_id": "qq", "fact": "q"}]}
    tests = []
    def add(name, task, expected, prefixes):
        a, p = own.solve(task)
        b, q = own.solve_conjunction(task)
        assert canon(a) == canon(expected) == canon(b), name
        assert canon(p) == canon(prefixes) == canon(q), name + " prefixes"
        tests.append({"name": name, "passed": True, "hand_task": task,
                      "hand_expected": expected, "hand_prefixes": prefixes})
    add("typed bool/int/null scalar encoding and lexical world sorting", copy.deepcopy(base),
        {"surviving_worlds": ["a", "m", "z"], "belief_sizes": [], "queries": [
            query("qx", [{"value": 1, "count": 1}, {"value": None, "count": 1}, {"value": True, "count": 1}], False, None),
            query("qq", [{"value": "null", "count": 1}, {"value": None, "count": 2}], False, None)]}, [["a", "m", "z"]])
    for value, chosen, guaranteed in [(True, "z", None), (1, "a", "null"), (None, "m", None)]:
        t = copy.deepcopy(base)
        t["observations"] = [{"fact": "x", "equals": value}]
        add("typed filter " + canon(value), t,
            {"surviving_worlds": [chosen], "belief_sizes": [1], "queries": [
                query("qx", [{"value": value, "count": 1}], True, value),
                query("qq", [{"value": guaranteed, "count": 1}], True, guaranteed)]}, [["a", "m", "z"], [chosen]])
    t = copy.deepcopy(base)
    t["observations"] = [{"fact": "x", "equals": True}, {"fact": "x", "equals": 1}, {"fact": "x", "equals": None}]
    add("empty absorption and entailed-null distinction", t,
        {"surviving_worlds": [], "belief_sizes": [1, 0, 0], "queries": [query("qx", [], False, None), query("qq", [], False, None)]},
        [["a", "m", "z"], ["z"], [], []])
    t = copy.deepcopy(base)
    t["observations"] = [{"fact": "x", "equals": 1}, {"fact": "x", "equals": 1}]
    add("nonempty exact repetition", t,
        {"surviving_worlds": ["a"], "belief_sizes": [1, 1], "queries": [
            query("qx", [{"value": 1, "count": 1}], True, 1),
            query("qq", [{"value": "null", "count": 1}], True, "null")]}, [["a", "m", "z"], ["a"], ["a"]])
    return tests


def metamorphics(task, facts, own):
    results = []
    def check(name, changed, expectation):
        output, prefixes = own.solve(changed)
        other, other_prefixes = own.solve_conjunction(changed)
        assert canon(output) == canon(other) and canon(prefixes) == canon(other_prefixes), name
        assert expectation(output, prefixes), name
        results.append({"name": name, "passed": True, "changed_task_sha256": digest(changed), "output_sha256": digest(output), "prefix_sha256": digest(prefixes)})
    wanted, prefix = facts["expected"], facts["prefix_survivors"]
    t = copy.deepcopy(task)
    t["worlds"].reverse()
    check("world order invariant", t, lambda a, p: canon(a) == canon(wanted) and canon(p) == canon(prefix))
    t = copy.deepcopy(task)
    t["queries"].reverse()
    exp = copy.deepcopy(wanted)
    exp["queries"].reverse()
    check("query order equivariant", t, lambda a, p: canon(a) == canon(exp) and canon(p) == canon(prefix))
    t = copy.deepcopy(task)
    rename = {w["world_id"]: "new opaque " + str(i) for i, w in enumerate(task["worlds"])}
    for w in t["worlds"]:
        w["world_id"] = rename[w["world_id"]]
    exp = copy.deepcopy(wanted)
    exp["surviving_worlds"] = sorted(rename[w] for w in exp["surviving_worlds"])
    pref = [sorted(rename[w] for w in p) for p in prefix]
    check("coherent world ID rename", t, lambda a, p: canon(a) == canon(exp) and canon(p) == canon(pref))
    t = copy.deepcopy(task)
    t["observations"].reverse()
    check("observation reorder final invariant with independently recomputed trace", t,
          lambda a, p: canon(a["surviving_worlds"]) == canon(wanted["surviving_worlds"]) and canon(a["queries"]) == canon(wanted["queries"]))
    if task["observations"]:
        t = copy.deepcopy(task)
        t["observations"].append(copy.deepcopy(task["observations"][-1]))
        exp = copy.deepcopy(wanted)
        exp["belief_sizes"].append(len(prefix[-1]))
        pref = prefix + [prefix[-1]]
        check("last observation repetition / empty absorption", t, lambda a, p: canon(a) == canon(exp) and canon(p) == canon(pref))
    t = strict_loads(json.dumps(task, ensure_ascii=False, sort_keys=True, indent=1))
    check("JSON object serialization invariance", t, lambda a, p: canon(a) == canon(wanted) and canon(p) == canon(prefix))
    return results


def freeze():
    source = pathlib.Path(__file__).resolve()
    actual = bindings()
    archive = OUT / (PREFIX + ".source.preexecution.py")
    with archive.open("xb") as f:
        f.write(source.read_bytes())
    config = {"report_type": "EXPERT_INPUT_ONLY_PREEXECUTION", "actual_author": ROLE,
              "reviewer_execution_id": ROLE, "created_utc": utc(), "source": hash_path(source),
              "archive_source": hash_path(archive), "dependencies_before_import_and_execution": actual,
              "python_executable": str(pathlib.Path(sys.executable).resolve()), "python_version": sys.version,
              "python_executable_binding": hash_path(pathlib.Path(sys.executable).resolve()),
              "planned_views": 3456, "field_whitelist": sorted(FIELDS), "exposure": EXPOSURE,
              "stage": "input-only; no canonical gold or producer property/control access",
              "recompute": "each task via own incremental and separate full-prefix conjunction algorithm",
              "no_dataset_approval": True, "hash_only_prior_file": "reviews/expert/belief-gap-v3.first_judgments.jsonl"}
    exclusive_json(OUT / (PREFIX + ".preexecution.json"), config)
    print(canon({"preexecution": hash_path(OUT / (PREFIX + ".preexecution.json")), "source": config["source"]}))


def execute():
    started = utc()
    timer = time.monotonic()
    pre_path = OUT / (PREFIX + ".preexecution.json")
    pre = strict_loads(pre_path.read_text())
    source = pathlib.Path(__file__).resolve()
    assert hash_path(source) == pre["source"]
    assert hash_path(OUT / (PREFIX + ".source.preexecution.py")) == pre["archive_source"] == pre["source"]
    before = bindings()
    assert before == pre["dependencies_before_import_and_execution"]
    assert hash_path(pathlib.Path(sys.executable).resolve()) == pre["python_executable_binding"]
    spec = importlib.util.spec_from_file_location("expert_v1_owned_semantics_for_v4_input_only", ROOT / "reviews/expert/prepare_belief_gap_v1.py")
    own = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(own)
    handmade = calibration(own)
    exclusive_json(OUT / (PREFIX + ".handmade-calibration.json"), {"actual_author": ROLE, "results": handmade, "all_passed": True})
    fresh = read_view(ROOT / "reviews/input_only/belief-gap-v4.jsonl")
    prior = read_view(ROOT / "reviews/input_only/belief-gap-v3.jsonl")
    assert len(fresh) == len(prior) == 3456
    lineage = strict_loads((ROOT / "reviews/input_only/belief-gap-v4.input_lineage.json").read_text())
    joins = {j["fresh_sample_id"]: j for j in lineage["joins"]}
    assert len(joins) == len(lineage["joins"]) == 3456 and set(joins) == set(fresh)
    assert {j["prior_sample_id"] for j in joins.values()} == set(prior)
    old_full = strict_loads((ROOT / "reviews/expert/belief-gap-v3.input_only_predictions.json").read_text())
    assert old_full["reviewer_execution_id"] == ROLE
    old_predictions = {r["sample_id"]: {k: r[k] for k in ["record_hash", "visible_identity", "task", "task_sha256"]} |
                       {"expected": r["facts"]["expected"], "prefix_survivors": r["facts"]["prefix_survivors"]}
                       for r in old_full["records"]}
    assert len(old_predictions) == len(old_full["records"]) == 3456 and set(old_predictions) == set(prior)
    del old_full
    families, splits, tracks, target_status, query_status, support, obs_count = [collections.Counter() for _ in range(7)]
    property_counts = collections.Counter()
    family_map = collections.defaultdict(set)
    split_families = collections.defaultdict(set)
    semantic_task_groups = collections.defaultdict(list)
    full_input_groups = collections.defaultdict(list)
    representatives = {}
    heuristic_counts = collections.Counter()
    max_values = collections.Counter()
    scalar_counts = collections.Counter()
    conditionals = collections.defaultdict(lambda: collections.Counter())
    metamorphic_results = []
    rows_digest = hashlib.sha256()
    full_join_digest = hashlib.sha256()
    prediction_path = OUT / (PREFIX + ".predictions.jsonl")
    first_path = OUT / (PREFIX + ".first-input-judgments.jsonl")
    with prediction_path.open("x", encoding="utf-8") as predictions, first_path.open("x", encoding="utf-8") as judgments:
        for number, (sid, row) in enumerate(fresh.items(), 1):
            j = joins[sid]
            old_row = prior[j["prior_sample_id"]]
            old_prediction = old_predictions[j["prior_sample_id"]]
            assert canon(input_subset(row)) == canon(input_subset(old_row)), (sid, "typed task / raw prompt / split / track changed")
            assert row["rendered_input"] == old_row["rendered_input"] and row["split"] == old_row["split"] and row["track_id"] == old_row["track_id"]
            assert canon(old_prediction["task"]) == canon(old_row["semantic_task"])
            assert old_prediction["task_sha256"] == digest(old_row["semantic_task"])
            assert old_prediction["record_hash"] == old_row["record_hash"]
            assert old_prediction["visible_identity"]["split"] == old_row["split"] and old_prediction["visible_identity"]["track_id"] == old_row["track_id"]
            # strict parse duplicate keys before the own full-contract prompt check.
            prompt = row["rendered_input"]
            assert prompt.count("Task JSON:\n") == 1
            strict_task = strict_loads(prompt.split("Task JSON:\n", 1)[1])
            task = own.parse_prompt(row)
            assert canon(task) == canon(strict_task) == canon(row["semantic_task"]), (sid, "rendered task mismatch")
            facts = own.row_facts(task)  # Fresh execution for every row; no reused target or semantic cache.
            assert canon(facts["expected"]) == canon(old_prediction["expected"]), (sid, "new recomputation vs own old target")
            assert canon(facts["prefix_survivors"]) == canon(old_prediction["prefix_survivors"]), (sid, "new recomputation vs own old prefixes")
            fam, split, track = row["semantic_family_id"], row["split"], row["track_id"]
            families[fam] += 1
            splits[split] += 1
            tracks[track] += 1
            split_families[fam].add(split)
            family_map[old_row["semantic_family_id"]].add(fam)
            status = "nonempty" if facts["expected"]["surviving_worlds"] else "empty"
            target_status[status] += 1
            support[str(facts["metrics"]["final_support"])] += 1
            obs_count[str(len(task["observations"]))] += 1
            semantic_task_groups[digest({"task": task, "split": split, "track": track})].append(sid)
            full_input_groups[digest(input_subset(row))].append(sid)
            names = sorted(p for p, w in facts["properties"].items() if w)
            for p in names:
                property_counts[p] += 1
                representatives.setdefault("property:" + p, {"sample_id": sid, "record_hash": row["record_hash"], "task": task,
                    "expected": facts["expected"], "prefix_survivors": facts["prefix_survivors"], "witness": facts["properties"][p][0]})
            representatives.setdefault("family:" + fam, {"sample_id": sid, "record_hash": row["record_hash"], "task": task,
                "expected": facts["expected"], "prefix_survivors": facts["prefix_survivors"], "metrics": facts["metrics"]})
            for key, errors in facts["incomplete_heuristic_error_query_ids"].items():
                heuristic_counts[key + "_wrong_rows"] += bool(errors)
                heuristic_counts[key + "_wrong_queries"] += len(errors)
            for w in task["worlds"]:
                for v in w["facts"].values():
                    scalar_counts["null" if v is None else type(v).__name__] += 1
            max_values["maximum_worlds"] = max(max_values["maximum_worlds"], len(task["worlds"]))
            max_values["maximum_facts"] = max(max_values["maximum_facts"], len(task["worlds"][0]["facts"]))
            max_values["maximum_observations"] = max(max_values["maximum_observations"], len(task["observations"]))
            max_values["maximum_queries"] = max(max_values["maximum_queries"], len(task["queries"]))
            for pos, q in enumerate(facts["expected"]["queries"]):
                query_status["entailed_" + str(q["entailed"]).lower()] += 1
                query_status["useful_" + str(q["can_reduce_uncertainty"]).lower()] += 1
                query_status["entailed_null"] += q["entailed"] and q["guaranteed"] is None
                for key, val in [("position", str(pos)), ("query_id", q["query_id"]), ("fact_name", task["queries"][pos]["fact"])]:
                    conditionals[key + ":" + val]["useful_" + str(q["can_reduce_uncertainty"]).lower()] += 1
                    conditionals[key + ":" + val]["entailed_" + str(q["entailed"]).lower()] += 1
                    conditionals[key + ":" + val][status] += 1
            selection = (fam, len(task["observations"]), status)
            if not any(r["selection"] == list(selection) for r in metamorphic_results):
                probes = metamorphics(task, facts, own)
                metamorphic_results.append({"selection": list(selection), "sample_id": sid, "record_hash": row["record_hash"], "results": probes})
            join_evidence = {"prior_sample_id": old_row["sample_id"], "prior_record_hash": old_row["record_hash"],
                "typed_task_raw_prompt_split_track_exact": True, "own_full_input_sha256": digest(input_subset(row)),
                "root_lineage_input_sha256": j["input_sha256"], "old_own_target_and_prefix_agreement_after_join": True}
            compact_properties = {p: {"witness_count": len(w), "actual_witnesses_sha256": digest(w)} for p, w in sorted(facts["properties"].items())}
            record = {"report_type": "INPUT_DERIVED_EXPECTATION", "actual_author": ROLE, "reviewer_execution_id": ROLE,
                "identity": identity(row), "sample_id": sid, "record_hash": row["record_hash"], "input_view_sha256": BINDINGS["reviews/input_only/belief-gap-v4.jsonl"],
                "task_sha256": digest(task), "rendered_input_sha256": sha_bytes(prompt.encode()),
                "expected_target": facts["expected"], "prefix_survivors": facts["prefix_survivors"],
                "independently_computed_properties": compact_properties, "metrics": facts["metrics"],
                "incomplete_heuristic_error_query_ids": facts["incomplete_heuristic_error_query_ids"], "join_evidence": join_evidence,
                "canonical_target_not_read": True, "dataset_approval": False}
            predictions.write(canon(record) + "\n")
            issue = {"category": "generalization", "severity": "moderate", "evidence": "Exact old/new full student input join; same finite cohort and four existing families. Input-stage artifacts do not certify future producer metadata.",
                     "fix": "Keep same-case repair and finite noiseless proxy scope explicit; canonical metadata and control claims need a separate authorized review."}
            judgment = {"receipt_id": "expert-belief-v4-input-attempt01-" + sid, "sample_id": sid,
                "record_hash": row["record_hash"], "reviewer_role": "expert", "reviewer_execution_id": ROLE,
                "review_status": "FIRST_INPUT_JUDGMENT_FROZEN_BEFORE_CANONICAL_ACCESS", "identity": identity(row),
                "scores": {"semantic_correctness": 5, "target_correctness": None, "instruction_clarity": 5,
                    "reasoning_validity": 5, "generalization_value": 2, "difficulty_quality": 3, "diversity": 2, "shortcut_resistance": 2},
                "critical_defects": [], "issues": [issue], "decision": "INPUT_CLEAR_PENDING_CANONICAL_COMPARISON",
                "confidence": 0.99, "evidence": {"prompt_full_contract": True, "duplicate_key_strict_parse": True,
                    "typed_rendered_and_semantic_task_join": True, "independent_incremental_conjunction_agreement": True,
                    "fresh_complete_target_sha256": digest(facts["expected"]), "fresh_all_prefixes_sha256": digest(facts["prefix_survivors"]),
                    "property_witnesses": compact_properties, "join_evidence": join_evidence,
                    "canonical_hash_not_recomputable_from_stripped_view": True},
                "scope": "Input correctness, clarity and finite competency only; no dataset ACCEPT; target_correctness unscored until actual canonical comparison.",
                "exposure": EXPOSURE}
            judgments.write(canon(judgment) + "\n")
            rows_digest.update((sid + "\t" + row["record_hash"] + "\n").encode())
            full_join_digest.update((sid + "\t" + old_row["sample_id"] + "\t" + digest(input_subset(row)) + "\n").encode())
            if number % 576 == 0:
                print("completed input recomputation", number, flush=True)
    assert all(len(s) == 1 for s in split_families.values()), "family split overlap"
    assert all(len(s) == 1 for s in family_map.values()) and len(set.union(*family_map.values())) == len(family_map)
    aggregate = {"view_records": len(fresh), "all_exact_joins": len(fresh), "all_fresh_targets_recomputed": len(fresh),
        "all_fresh_prefixes_recomputed": len(fresh), "all_own_old_expectation_crosschecks_after_join": len(fresh),
        "families": dict(families), "splits": dict(splits), "tracks": dict(tracks),
        "old_to_fresh_family_mapping": {k: sorted(v) for k, v in family_map.items()}, "added_semantic_families": 0,
        "status_histogram": dict(target_status), "query_label_histogram": dict(query_status),
        "support_histogram": dict(support), "observation_length_histogram": dict(obs_count),
        "actual_property_support_rows": dict(property_counts), "input_world_scalar_type_counts": dict(scalar_counts),
        "input_limits": dict(max_values), "incomplete_heuristic_errors": dict(heuristic_counts),
        "distinct_object_key_normalized_typed_tasks_with_split_and_track": len(semantic_task_groups),
        "normalized_task_rendering_multiplicity_histogram": dict(collections.Counter(map(len, semantic_task_groups.values()))),
        "distinct_exact_full_inputs_with_split_and_track": len(full_input_groups),
        "full_input_multiplicity_histogram": dict(collections.Counter(map(len, full_input_groups.values()))),
        "actual_ids_record_hashes_ordered_sha256": rows_digest.hexdigest(), "actual_full_input_lineage_ordered_sha256": full_join_digest.hexdigest(),
        "query_id_position_fact_name_conditional_histograms": dict(conditionals),
        "counterfactual_sample_groups": len(metamorphic_results), "counterfactual_checks": sum(len(r["results"]) for r in metamorphic_results),
        "counterfactual_checks_all_pass": True,
        "normalization_definition": "Object key order normalized only; world/observation/query list order and opaque names preserved. These are typed tasks, not independent mechanisms or new families.",
        "pending": ["canonical target comparison", "full record hash recomputation", "all producer row tags and locators", "all cohort family controls, scalar/schedule/profile/probe claims", "published difficulty labels"]}
    analysis = {"report_type": "EXPERT_INPUT_ONLY_ANALYSIS", "actual_author": ROLE, "reviewer_execution_id": ROLE,
        "stage": "INPUT ONLY", "exposure": EXPOSURE, "aggregates": aggregate, "representatives": representatives,
        "handmade_calibration_count": len(handmade), "metamorphic_results": metamorphic_results,
        "no_dataset_approval": True, "canonical_or_other_private_files_read": False}
    exclusive_json(OUT / (PREFIX + ".analysis.json"), analysis)
    after = bindings()
    assert before == after and hash_path(source) == pre["source"]
    outputs = {p.name: hash_path(p) for p in [prediction_path, first_path,
        OUT / (PREFIX + ".analysis.json"), OUT / (PREFIX + ".handmade-calibration.json")]}
    execution = {"report_type": "EXPERT_INPUT_ONLY_ACTUAL_EXECUTION", "actual_author": ROLE, "reviewer_execution_id": ROLE,
        "start_utc": started, "finish_utc": utc(), "elapsed_seconds": round(time.monotonic() - timer, 6), "exit_code": 0,
        "source_before_after": hash_path(source), "preexecution": hash_path(pre_path), "bindings_before": before, "bindings_after": after,
        "outputs": outputs, "records": len(fresh), "handmade_tests_passed": len(handmade),
        "metamorphic_checks_passed": aggregate["counterfactual_checks"], "execution_platform": platform.platform(),
        "ru_maxrss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "ru_maxrss_unit_note": "macOS resource units are bytes; retained old full analysis not duplicated on disk",
        "only_owned_solver_imported": "reviews/expert/prepare_belief_gap_v1.py", "exposure": EXPOSURE,
        "all_fresh_predictions_and_first_input_judgments_completed": True, "canonical_comparison_authorized_or_started": False,
        "no_dataset_approval": True}
    exec_path = OUT / (PREFIX + ".execution.json")
    exclusive_json(exec_path, execution)
    exclusive_json(OUT / (PREFIX + ".freeze.json"), {"report_type": "EXPERT_INPUT_ONLY_FROZEN_OUTPUTS", "actual_author": ROLE,
        "reviewer_execution_id": ROLE, "created_utc": utc(), "source": hash_path(source), "preexecution": hash_path(pre_path),
        "execution": hash_path(exec_path), "outputs": outputs, "records": len(fresh), "stage": "all inputs complete; STOP before canonical access",
        "exposure": EXPOSURE, "no_dataset_approval": True})
    print(canon({"records": len(fresh), "predictions": outputs[prediction_path.name], "first_input_judgments": outputs[first_path.name],
        "execution": hash_path(exec_path), "analysis": outputs[PREFIX + ".analysis.json"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "execute"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        try:
            execute()
        except BaseException as error:
            failure = {"report_type": "PRESERVED_INPUT_STAGE_FAILURE", "actual_author": ROLE, "reviewer_execution_id": ROLE,
                       "created_utc": utc(), "source": hash_path(pathlib.Path(__file__).resolve()),
                       "error_type": type(error).__name__, "error": str(error), "no_dataset_approval": True}
            exclusive_json(OUT / (PREFIX + ".failure.json"), failure)
            raise
