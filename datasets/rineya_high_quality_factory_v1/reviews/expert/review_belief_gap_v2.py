#!/opt/anaconda3/bin/python3
"""Corrective v2 expert execution. Public guidance, independent typed predictions.
All first predictions precede canonical target/metadata access. Own v1 decisions
and evidence remain immutable. No producer/oracle/reference/checker reads.
"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import pathlib
import re
import runpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROLE = "/root/expert_reviewer"
OUT = ROOT / "reviews/expert"
SUPPORT_PATH = OUT / "belief_gap_v2_metadata_support.py"
SUPPORT_SHA = "54324adf2332c7d85651f6217dc51d1ac6f6610ddd68d1819f43416f9f1d9e84"
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(SUPPORT_PATH.read_bytes()) == SUPPORT_SHA
OWN = runpy.run_path(str(SUPPORT_PATH))
canon, equal = OWN["canon"], OWN["equal"]
INPUT_KEYS = {"sample_id", "record_hash", "semantic_parent_id", "semantic_family_id", "track_id", "split", "semantic_task", "rendered_input"}
PUBLIC = {
    "schemas/EXTENDED_SEMANTICS.md": "2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23",
    "schemas/BELIEF_CONTROL_PROPERTIES.md": "200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034",
    "research/curricula/RBM-EVIDENCE-GAP.md": "004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120",
    "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md": "46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f",
    "research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md": "c2eb841ed546848088db4e0499e8e8f7087aa3d213af333ff39237ae6558bd33",
    "schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json": "611656b37d7fe917d64eeb191d901d272a32d8d3c72c9f8865476c6889a3e51a"}


def public_bindings():
    result = {}
    for p, expected in PUBLIC.items():
        data = (ROOT / p).read_bytes()
        assert sha(data) == expected, "public source changed: " + p
        result[p] = {"sha256": sha(data), "length": len(data)}
    return result


def scalar_encoding(value):
    if value is not None and type(value) not in (bool, int, str):
        raise ValueError("scalar outside finite public contract")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def direct_solver(task):
    """New v2 implementation: direct world-by-prefix conjunction enumeration."""
    OWN["validate_task"](task)
    prefixes = []
    for k in range(len(task["observations"]) + 1):
        prefixes.append(sorted(w["world_id"] for w in task["worlds"] if all(
            scalar_encoding(w["facts"][o["fact"]]) == scalar_encoding(o["equals"])
            for o in task["observations"][:k])))
    selected = [w for w in task["worlds"] if w["world_id"] in prefixes[-1]]
    answers = []
    for query in task["queries"]:
        bins = collections.defaultdict(int)
        for world in selected:
            bins[scalar_encoding(world["facts"][query["fact"]])] += 1
        counts = [{"value": json.loads(v), "count": n} for v, n in sorted(bins.items())]
        entails = bool(selected) and len(counts) == 1
        answers.append({"query_id": query["query_id"], "counts": counts, "distinct_values": len(counts),
                        "entailed": entails, "guaranteed": counts[0]["value"] if entails else None,
                        "can_reduce_uncertainty": len(counts) >= 2})
    return {"surviving_worlds": prefixes[-1], "belief_sizes": [len(p) for p in prefixes[1:]], "queries": answers}, prefixes


def sequential_solver(task):
    """New alternative: explicit type/object equality and list bins, no encoded keys."""
    OWN["validate_task"](task)
    active = task["worlds"][:]
    prefixes = [sorted(w["world_id"] for w in active)]
    for observation in task["observations"]:
        keep = []
        for w in active:
            value, desired = w["facts"][observation["fact"]], observation["equals"]
            if type(value) is type(desired) and value == desired:
                keep.append(w)
        active = keep
        prefixes.append(sorted(w["world_id"] for w in active))
    answers = []
    for q in task["queries"]:
        bins = []
        for world in active:
            value = world["facts"][q["fact"]]
            match = next((b for b in bins if type(b["value"]) is type(value) and b["value"] == value), None)
            if match is None:
                bins.append({"value": value, "count": 1})
            else:
                match["count"] += 1
        bins.sort(key=lambda b: scalar_encoding(b["value"]))
        certain = len(bins) == 1
        answers.append({"query_id": q["query_id"], "counts": bins, "distinct_values": len(bins),
                        "entailed": certain, "guaranteed": bins[0]["value"] if certain else None,
                        "can_reduce_uncertainty": len(bins) > 1})
    return {"surviving_worlds": prefixes[-1], "belief_sizes": list(map(len, prefixes[1:])), "queries": answers}, prefixes


def facts(task):
    expected, prefixes = direct_solver(task)
    other, p2 = sequential_solver(task)
    assert equal(expected, other) and equal(prefixes, p2), "new solvers disagree"
    measured = OWN["row_facts"](task)
    assert equal(expected, measured["expected"]) and equal(prefixes, measured["prefix_survivors"])
    return measured


def diagnostics_from_input(task):
    """Rules frozen from public root disclosure; no final support/gold features."""
    vals = [o["equals"] for o in task["observations"]]
    marker = any(type(v) is str and v in {"stone", "unknown"} for v in vals)
    linen = any(type(v) is str and v == "linen" for v in vals)
    rules = {"v1_original_scalar_marker_empty": marker,
             "prospective_marker_and_auxiliary_token_empty": marker and linen}
    N, M = len(task["worlds"]), len(task["observations"])
    def core(active):
        if not active or N not in (8, 12) or M < 1:
            return None
        return {"surviving_worlds": [], "belief_sizes": [1 if N == 8 else 2] + [0] * (M - 1),
                "queries": [{"query_id": q["query_id"], "counts": [], "distinct_values": 0,
                             "entailed": False, "guaranteed": None, "can_reduce_uncertainty": False} for q in task["queries"]]}
    # The second full-output extension uses exactly the same disclosed core
    # construction with only the frozen prospective activation predicate.
    outputs = {name: core(active) for name, active in rules.items()}
    _, prefixes = direct_solver({**task, "observations": task["observations"][:1]})
    features = {"scalar_occurrences": sorted(set(scalar_encoding(v) for v in vals)),
                "position_values": [[j, scalar_encoding(v)] for j, v in enumerate(vals)],
                "position_fact_values": [[j, o["fact"], scalar_encoding(o["equals"])] for j, o in enumerate(task["observations"])],
                "world_count": N, "fact_count": len(task["worlds"][0]["facts"]), "query_count": len(task["queries"]),
                "observation_length": M, "first_prefix_survivors": prefixes[-1], "first_prefix_count": len(prefixes[-1])}
    return {"empty_predictions": rules, "abstaining_full_output_predictions": outputs, "input_features": features,
            "source": "schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json", "final_gold_or_final_support_used_as_predictor": False}


def fresh_toys():
    values = [("u1", "stone", "silk", False, "unknown", "east"), ("u2", "stone", "silk", 0, "unknown", "west"),
              ("u3", "reed", "linen", None, True, "east"), ("u4", "reed", "cotton", True, None, "west"),
              ("u5", "fern", "linen", "opal", "unknown", "east")]
    worlds = [{"world_id": i, "facts": {"x": x, "y": y, "latent": a, "control": b, "extra": c}} for i, x, y, a, b, c in values]
    queries = [{"query_id": "qa", "fact": "latent"}, {"query_id": "qb", "fact": "control"}, {"query_id": "qc", "fact": "x"}]
    observation_sets = [[], [{"fact": "latent", "equals": False}],
                        [{"fact": "x", "equals": "stone"}, {"fact": "y", "equals": "linen"}],
                        [{"fact": "x", "equals": "stone"}, {"fact": "y", "equals": "silk"}],
                        [{"fact": "x", "equals": "stone"}, {"fact": "y", "equals": "linen"}, {"fact": "x", "equals": "stone"}],
                        [{"fact": "x", "equals": "stone"}, {"fact": "y", "equals": "silk"}, {"fact": "x", "equals": "stone"}]]
    return [{"kind": "belief", "worlds": copy.deepcopy(worlds), "observations": obs, "queries": copy.deepcopy(queries)} for obs in observation_sets]


def compress_exact(path):
    raw = path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    cp = pathlib.Path(str(path) + ".gz")
    assert not cp.exists()
    cp.write_bytes(compressed)
    assert gzip.decompress(cp.read_bytes()) == raw and len(compressed) <= 25 * 1024 * 1024
    return {"raw_path": str(path), "raw_sha256": sha(raw), "raw_length": len(raw),
            "compressed_path": str(cp), "compressed_sha256": sha(compressed), "compressed_length": len(compressed),
            "gzip_mtime": 0, "decoded_sha256": sha(gzip.decompress(compressed)), "roundtrip_byte_equal": True}


def prepare():
    inputs = public_bindings()
    fixtures = []
    for i, task in enumerate(fresh_toys()):
        f = facts(task)
        probe = OWN["metamorphics"](task)
        fixtures.append({"fixture_index": i, "task": task, "facts": f, "probes": probe, "frozen_diagnostic_predictions": diagnostics_from_input(task)})
    assert fixtures[2]["facts"]["expected"]["belief_sizes"] == [2, 0]
    assert fixtures[3]["facts"]["expected"]["belief_sizes"] == [2, 2]
    assert fixtures[4]["facts"]["expected"]["belief_sizes"] == [2, 0, 0]
    assert fixtures[5]["facts"]["expected"]["belief_sizes"] == [2, 2, 2]
    assert fixtures[2]["facts"]["prefix_survivors"][1] == fixtures[3]["facts"]["prefix_survivors"][1]
    assert fixtures[4]["facts"]["prefix_survivors"][1] == fixtures[5]["facts"]["prefix_survivors"][1]
    assert fixtures[3]["facts"]["expected"]["queries"][0]["counts"] == [{"value": 0, "count": 1}, {"value": False, "count": 1}]
    assert fixtures[1]["facts"]["expected"]["surviving_worlds"] == ["u1"]
    assert fixtures[4]["facts"]["properties"]["post_empty_absorption"]
    assert fixtures[5]["facts"]["properties"]["nonempty_repeated_observation"]
    assert not equal(True, 1) and not equal(False, 0)
    for key in OWN["FORBIDDEN"]:
        try:
            OWN["refuse_targets"]({"nested": {key: None}})
            raise AssertionError("gold field accepted")
        except ValueError:
            pass
    report = {"report_type": "CORRECTIVE_V2_EXPERT_PREPARATION", "actual_author": ROLE, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "public_inputs": inputs, "source_sha256": sha(pathlib.Path(__file__).read_bytes()), "own_support_source_sha256": SUPPORT_SHA,
              "guidance": "Disclosed v1 marker defect and root prospective marker+linen diagnostic; corrective guided mode, not fully blind.",
              "new_solver_methods": ["direct per-world prefix conjunction using scalar encodings", "sequential explicit type-and-value filtering with list-based bins"],
              "own_v1_public_derived_utility_reuse": "Property/locator/incidence/probe routines adapted into new support file; no private producer/oracle code.",
              "future_candidate_reads": 0, "future_gold_reads": 0, "dataset_verdicts": 0,
              "new_toy_fixtures": len(fixtures), "executed_toy_probes": sum(len(f["probes"]["invariance_probes"]) + len(f["probes"]["typed_equals_counterfactual_probes"]) for f in fixtures),
              "fixtures": fixtures, "planned_records_not_measured": 1728,
              "stage_sequence": ["new program/toy preparation freeze", "eight-field input-only full typed predictions/supports/prefixes and frozen diagnostics", "prediction SHA freeze", "separate original target/property/locator/control/profile/crossing comparison", "actual first judgments freeze before private cross-review"],
              "required_additional_checks": ["E/G andF/H exact same first-prefix IDs, different final emptiness, only equals[1] changed", "F repeats first observation after empty;H after nonempty;G/H truthful component difficulty", "all scalar/position/name/first-prefix/coarse buckets byfamily/length/split", "frozen marker and prospectivejointmarker rules evaluated unchanged", "new perfect shortcut must be reported rather than call this complete shortcut removal"]}
    p = OUT / "belief-gap-v2.preparation.json"
    assert not p.exists()
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    receipt = {"actual_author": ROLE, "report_type": "PREPARATION_EXECUTION_FREEZE", "python_executable": "/opt/anaconda3/bin/python3",
               "actual_argv": ["/opt/anaconda3/bin/python3", "reviews/expert/review_belief_gap_v2.py", "prepare"], "exit_code": 0,
               "preparation_sha256": sha(p.read_bytes()), "source_sha256": report["source_sha256"], "own_support_source_sha256": SUPPORT_SHA,
               "public_inputs": inputs, "candidate_reads": 0, "gold_reads": 0, "no_verdict": True}
    rp = OUT / "belief-gap-v2.preparation_execution.json"
    assert not rp.exists()
    rp.write_text(json.dumps(receipt, indent=2) + "\n")
    print(canon({"path": str(p), "sha256": receipt["preparation_sha256"], "source_sha256": report["source_sha256"], "fixtures": len(fixtures), "toy_probes": report["executed_toy_probes"]}))


def predict(view, original_sha, out):
    inputs = public_bindings()
    data = pathlib.Path(view).read_bytes()
    assert sha(data) == "378007b6afe442764db7a1696cad217fa357959d97be50c603946d2251a25af6"
    assert original_sha == "589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df"
    records = []
    for line in data.splitlines():
        row = json.loads(line)
        assert set(row) == INPUT_KEYS
        OWN["refuse_targets"](row)
        task = OWN["parse_prompt"](row)
        assert equal(task, row["semantic_task"])
        f = facts(task)
        records.append({"sample_id": row["sample_id"], "record_hash": row["record_hash"], "visible_identity": {k: row[k] for k in INPUT_KEYS - {"rendered_input", "semantic_task"}},
                        "task": task, "task_sha256": sha(canon(task).encode()), "facts": f,
                        "probes": OWN["metamorphics"](task), "input_only_frozen_diagnostics": diagnostics_from_input(task)})
    assert len(records) == len({r["sample_id"] for r in records}) == 1728
    report = {"report_type": "CORRECTIVE_V2_INPUT_ONLY_PREDICTION_FREEZE", "actual_author": ROLE, "reviewer_execution_id": ROLE,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "public_inputs": inputs,
              "view_sha256": sha(data), "original_candidate_sha256": original_sha, "source_sha256": sha(pathlib.Path(__file__).read_bytes()),
              "own_support_sha256": SUPPORT_SHA, "metadata_and_gold_not_seen": True, "no_dataset_verdict": True,
              "guidance": "Public corrective design and disclosed root diagnostic hypotheses; not fully blind.", "records": records}
    p = pathlib.Path(out)
    assert not p.exists()
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compression = compress_exact(p)
    receipt = {"actual_author": ROLE, "created_utc": report["created_utc"], "report_type": "PREDICTION_FREEZE_RECEIPT",
               "record_count": len(records), "metadata_and_gold_not_seen": True, "no_dataset_verdict": True,
               "source_sha256": report["source_sha256"], "own_support_sha256": SUPPORT_SHA, "view_sha256": sha(data),
               "original_candidate_sha256": original_sha, "prediction_sha256": sha(p.read_bytes()), "prediction_length": p.stat().st_size,
               "compression": compression}
    rp = pathlib.Path(str(p) + ".freeze_receipt.json")
    assert not rp.exists()
    rp.write_text(json.dumps(receipt, indent=2) + "\n")
    print(canon(receipt))


def compare(frozen, original, out):
    p = pathlib.Path(frozen)
    raw = p.read_bytes()
    receipt = json.loads(pathlib.Path(str(p) + ".freeze_receipt.json").read_text())
    assert sha(raw) == receipt["prediction_sha256"]
    predictions = json.loads(raw)
    candidate_bytes = pathlib.Path(original).read_bytes()
    assert sha(candidate_bytes) == predictions["original_candidate_sha256"]
    candidates = {r["sample_id"]: r for r in map(json.loads, candidate_bytes.splitlines())}
    records = []
    for record in predictions["records"]:
        row = candidates[record["sample_id"]]
        assert row["record_hash"] == record["record_hash"]
        task = OWN["parse_prompt"](row)
        assert equal(task, record["task"]) and equal(row["semantic_task"], task)
        target_equal = equal(record["facts"]["expected"], row["verified_target"])
        payload = {k: v for k, v in row.items() if k != "record_hash"}
        hash_equal = sha(canon(payload).encode()) == row["record_hash"]
        md, errors = OWN["metadata_audit"](row, task, record["facts"])
        for claim in md.get("family_control_claims", []):
            if claim.get("name", claim.get("property")) in {"same_prefix_empty_nonempty_counterfactual", "contradiction_equals_mixed_outcomes"} and claim.get("scope") == "corpus_variant":
                if "pair_sample_ids" not in claim or not claim.get("evidence_manifest"):
                    errors.append("new supplied paired claim lacks exact pair IDs/length evidence pointer")
        records.append({**record, "metadata": md, "metadata_errors": errors, "target_equal": target_equal,
                        "original_hash_equal": hash_equal, "declared_difficulty": row["difficulty"], "structural_difficulty": row["structural_difficulty"]})
    audit = OWN["family_audits"](records)
    report = {"report_type": "POST_PREDICTION_FREEZE_V2_TARGET_METADATA_COMPARISON", "actual_author": ROLE,
              "prediction_sha256": sha(raw), "original_candidate_sha256": sha(candidate_bytes), "source_sha256": sha(pathlib.Path(__file__).read_bytes()),
              "records": records, "family_audits": audit, "target_mismatches": sum(not r["target_equal"] for r in records),
              "original_hash_mismatches": sum(not r["original_hash_equal"] for r in records), "metadata_error_rows": sum(bool(r["metadata_errors"]) for r in records),
              "no_judgment_yet": True}
    o = pathlib.Path(out)
    assert not o.exists()
    o.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compression = compress_exact(o)
    cr = pathlib.Path(str(o) + ".compression.json")
    assert not cr.exists()
    cr.write_text(json.dumps(compression, indent=2) + "\n")
    print(canon({"path": str(o), "sha256": sha(o.read_bytes()), "target_mismatches": report["target_mismatches"], "hash_mismatches": report["original_hash_mismatches"], "metadata_error_rows": report["metadata_error_rows"], "family_errors": collections.Counter(e["error"] for e in audit["errors"]), "compression": compression}))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["prepare", "predict", "compare"])
    a.add_argument("--view")
    a.add_argument("--original-sha")
    a.add_argument("--frozen")
    a.add_argument("--canonical")
    a.add_argument("--out")
    args = a.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "predict":
        predict(args.view, args.original_sha, args.out)
    else:
        compare(args.frozen, args.canonical, args.out)
