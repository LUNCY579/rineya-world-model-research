"""Owned actual V2 audit. Only public-derived own solvers are executed.

`audit` freezes conclusions; `finalize` additionally requires completed final custody.
Never rewrites candidates, other reviews, public specifications or release files.
"""
from pathlib import Path
from collections import Counter, defaultdict, deque
import datetime
import hashlib
import importlib.util
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "/root/research_architect"
BINDINGS = {}
BASE = "reviews/adjudication/planning-gap-v2"
CANDIDATE = "data/RGM-001/candidate_renderings/planning-gap-v2.jsonl"
MANIFEST = "data/RGM-001/manifests/planning-gap-v2.family_controls.json"
EXPECTED = {
    CANDIDATE: "57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23",
    MANIFEST: "9c4497cf61e1d77cea24d808eca965c00e10e975cf235bd7d1aa2667e259da17",
    "schemas/SEMANTICS.md": "718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd",
    "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md": "f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a",
    "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md": "02613aa2333f401117def89da224f169d564b527bacc554a0b872877aaac42fe",
    "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md": "6100d43fe1cf83d798bb34799fac70569f54d33f66692150d188581d9c940761",
    "schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md": "d59ce14dd66c6ebd644580f7736e714fbceda32dd132d8c9cb5cf61e78a5a258",
    "research/curricula/RGM-PLANNING-GAP-V2-PROSPECTIVE.md": "34a3c89302d4b965ee784cd2f52207b6e9cb0b4299b52b82bf97c675c2d77d2c",
    "schemas/RGM_PLANNING_V2_RENDERER_CONTROL_ENCODING_ADDENDUM.md": "48e1ebbe954944d8b6569866fa102962607c4ecf27a1d7133f1259f815eac603",
    "reviews/adjudication/planning-gap-v2-preparation.py": "77c8cda01e9498a18efe4a74fb94d0912eb271cde725c857568a94b0a6818af6",
    "reviews/adjudication/audit_and_adjudicate_planning_gap_v1.py": "c28fbde6721c8ad9eb32dc663745713d21080ccb0e4115c7e02e97337980ec89",
    "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.py": "b93c498295a988770f5384a7b3d9fdd66d441f95aafc22e286f109090bebe175",
    "reviews/input_only/planning-gap-v2.jsonl": "00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c",
    "reviews/input_only/planning-gap-v1.jsonl": "2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654",
    "research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl": "896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d",
    "reviews/expert/planning-gap-v2.jsonl": "86b78d2eafff250629b0b2074cd4bf058259d079835df0af1ca30c1c23aba884",
    "reviews/expert/planning-gap-v2.first.jsonl": "86b78d2eafff250629b0b2074cd4bf058259d079835df0af1ca30c1c23aba884",
    "reviews/adversarial/planning-gap-v2.jsonl": "a1e3867be69eb816d3a82c0c1bacc5903ece38513072d2a18e82755bfa15e882",
    "reviews/adversarial/planning-gap-v2.first.jsonl": "a1e3867be69eb816d3a82c0c1bacc5903ece38513072d2a18e82755bfa15e882",
    "quality_reports/RGM_V2_FROZEN_REVIEW_EVIDENCE_JOIN.json": "1fd3badf1a0fb29e635828a65461d19dc1cbeccaaae7353d2f8ca6a52b83a7df",
    "audits/leakage/planning-gap-v2.json": "38eee2ad025e5ec5e68495a408135fc6abcb451dadcfd1044d65612012876d53",
    "audits/diversity/planning-gap-v2-planning-controls.json": "20212a488c949b4f5d9cbc920399eec444c976d335d5aca637d302cb4527ba26",
    "quality_reports/RGM_V2_FULL_EXECUTION.json": "c95e3bc15dd6d3f1b01e0b718f127bf23dd444f690efc3eac69c71e9abeac506",
    "reviews/oracle/RGM-v2-renderer-public-contract-assessment-20261010T062719Z.md": "f8243aa9d895d2d4c0f825e18129d488ed899a307b59ce2e4779f6179d4bcef3",
    "reviews/oracle/RGM-v2-renderer-public-assessment-receipt-20261010T062719Z.json": "c52be7f3d6dbdfe5aa586cfd594b7205ac8d4a34fb6814a32a55673859f67aa5",
}
DISCLOSURE = (
    "Actual /root/research_architect authored the public V1/V2 designs and previously "
    "adjudicated V1; never produced these candidates or performed either first review. "
    "Root supplied failure localization and requested retrospective assessment after "
    "both input predictions/first reviews froze. Own frozen public-derived solvers reused. "
    "No producer private source/expected fixtures, private formal or reviewer implementation "
    "imported. Shared underlying model/public design means correlated errors remain possible. "
    "This is independent execution, not blind design or independent-model evidence."
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic_key(task):
    """Graph node/edge list permutations preserve world; action sequence never sorted."""
    normalized = dict(task)
    if task["kind"] == "graph":
        normalized["nodes"] = sorted(task["nodes"])
        normalized["edges"] = sorted(task["edges"])
    return canonical(normalized)


def read(path):
    raw = (ROOT / path).read_bytes()
    if path in EXPECTED:
        assert sha(raw) == EXPECTED[path], ("frozen byte mismatch", path)
    BINDINGS[path] = {"path": path, "sha256": sha(raw), "bytes": len(raw)}
    return raw


def obj(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines()]


def write_new(path, value, raw=False):
    p = ROOT / path
    assert not p.exists(), ("immutable output already exists", path)
    data = value if raw else json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    p.write_text(data)
    return {"path": path, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}


def setup():
    global P, V1, FN
    for name in ["reviews/adjudication/planning-gap-v2-preparation.py", "reviews/adjudication/audit_and_adjudicate_planning_gap_v1.py"]:
        read(name)
        spec = importlib.util.spec_from_file_location("owned_" + name.rsplit("/", 1)[1], ROOT / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if "preparation" in name:
            P = module
        else:
            V1 = module
    path = "research/research_gap_analysis/RGM_NEXT_GAP_MEASURE.py"
    raw = read(path)
    FN = {"__file__": str(ROOT / path)}
    exec(compile(raw.decode().split("\nbindings = []\n", 1)[0], str(ROOT / path), "exec"), FN)
    V1.FN, V1.DISPLACEMENT = FN, {}


def measured_shortest(task):
    """Exhaustive public finite-state BFS, sorted full action sequences; no cutoff."""
    if task["kind"] == "graph":
        initial = (task["start"], task["fuel"])
        key = lambda state: state
        goal = lambda state: state[0] == task["goal"]
        def actions(state):
            node, fuel = state
            for a, b, cost in sorted(task["edges"], key=lambda e: e[1]):
                if a == node and cost <= fuel:
                    yield b, (b, fuel - cost)
    else:
        names = sorted(task["on"])
        initial = dict(task["on"])
        key = lambda state: tuple(state[n] for n in names)
        goal = lambda state: all(state[b] == s for b, s in task["goal_on"].items())
        actions = FN["block_actions"]
    queue, seen = deque([(initial, [])]), {key(initial)}
    expanded, frontier = 0, 1
    while queue:
        state, plan = queue.popleft()
        expanded += 1
        if goal(state):
            return plan, {"complete": True, "expanded": expanded, "visited": len(seen), "maximum_frontier": frontier}
        for action, nxt in actions(state):
            k = key(nxt)
            if k not in seen:
                seen.add(k)
                queue.append((nxt, plan + [action]))
        frontier = max(frontier, len(queue))
    return None, {"complete": True, "expanded": expanded, "visited": len(seen), "maximum_frontier": frontier}


def renderer_projection(parsed, shared_goal_keys=None):
    """Owned input-order diagnostic, not an import/relaxation of the frozen gate."""
    result = {"top_keys": list(parsed)}
    if parsed["kind"] == "graph":
        result.update(nodes=list(parsed["nodes"]), edges=[e[:2] for e in parsed["edges"]])
    else:
        result.update(on_keys=list(parsed["on"]), action_object_keys=list(parsed["plan"][0]) if parsed["plan"] else [])
        if shared_goal_keys is not None:
            result["shared_goal_keys"] = [b for b in parsed["goal_on"] if b in shared_goal_keys]
    return result


def property_truth(task, target, failure, witness, source_index):
    name, evidence = witness["property"], witness["evidence"]
    if name == "later_action_requires_changed_block_clearance":
        return P.clearance_evidence_check(task, evidence)
    if name == "isolated_unchanged_support_failure_after_legal_prefix":
        return P.isolated_noop_check(task, evidence, source_index)
    return V1.property_check(task, target, failure, witness)


def control_truth(control, index, cache):
    left, right = [index[sid] for sid in control["sample_ids"]]
    a, b = left["semantic_task"], right["semantic_task"]
    changed = sorted(k for k in a if canonical(a[k]) != canonical(b[k]))
    assert changed == sorted(control["changed_task_fields"])
    name = control["name"]
    detail = {}
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
        assert control["evidence"]["swap_index"] == i
        detail = P.selective_failure_check(b, control["evidence"])
    elif name == "partial_goal_extension_counterfactual":
        assert changed == ["goal_on"] and set(a["goal_on"]) < set(b["goal_on"])
        assert all(b["goal_on"][block] == support for block, support in a["goal_on"].items())
    elif name == "same_world_valid_goal_detour":
        assert changed == ["plan"]
        target = cache[semantic_key(b)][0]
        assert target["valid"] and target["reaches_goal"] and len(b["plan"]) > len(target["shortest_plan"])
    else:
        assert name == "isolated_noop_append_counterfactual" and changed == ["plan"]
        e = control["evidence"]
        assert type(e["prefix_length"]) is int and e["prefix_length"] == len(a["plan"])
        assert b["plan"] == a["plan"] + [{"block": e["block"], "to": "TABLE"}] and e["current_support"] == "TABLE"
        detail = P.isolated_noop_check(b, {"action_index": e["prefix_length"], "block": e["block"], "current_support": e["current_support"]}, right["source_family_index"])
    return {"control_id": control["control_id"], "name": name, "sample_ids": control["sample_ids"],
            "record_hashes": [left["record_hash"], right["record_hash"]], "renderer": left["renderer"],
            "changed_fields": changed, "truth": True, "own_proof": detail}


def verify_freeze_artifacts(freeze):
    """Hashes only; never imports or uses another reviewer's implementation."""
    for item in freeze.get("artifacts", freeze.get("new_exact_artifacts", [])):
        assert sha(read(item["path"])) == item["sha256"], item["path"]


def review_custody(index):
    joined = {}
    for role, subdir, execution in [("expert_reviewer", "expert", "/root/expert_reviewer"), ("adversarial", "adversarial", "/root/adversarial_auditor")]:
        final = rows(f"reviews/{subdir}/planning-gap-v2.jsonl")
        first = read(f"reviews/{subdir}/planning-gap-v2.first.jsonl")
        assert first == read(f"reviews/{subdir}/planning-gap-v2.jsonl")
        joined[subdir] = P.exact_review_join(index, final, role, execution)
        prediction_path = "reviews/expert/planning-gap-v2.expert_input_predictions.jsonl" if subdir == "expert" else "reviews/adversarial/planning-gap-v2.input_predictions.jsonl"
        predicted = rows(prediction_path)
        assert len(predicted) == len(index) and {p["sample_id"] for p in predicted} == set(index)
        for prediction in predicted:
            row = index[prediction["sample_id"]]
            assert prediction["record_hash"] == row["record_hash"] and prediction["reviewer_execution_id"] == execution
            assert P.typed_equal(prediction["expected_target"], row["verified_target"])
        assert all(receipt["decision"] == "ACCEPT" for receipt in joined[subdir].values())
    expert_freeze = obj("reviews/expert/planning-gap-v2.expert_comparison.freeze.json")
    assert expert_freeze["actual_derive_exit_code"] == expert_freeze["actual_finalize_exit_code"] == 0
    assert expert_freeze["canonical_access_only_after_all_input_predictions_and_first_judgments_frozen"] is True
    verify_freeze_artifacts(expert_freeze)
    ef = obj("reviews/expert/planning-gap-v2.expert_input.freeze.json")
    verify_freeze_artifacts(ef)
    ep = obj("reviews/expert/planning-gap-v2.expert_comparison.preexecution.json")
    P.first_freeze_before_comparison(ef["created_utc"], ep["created_utc"])
    ex = obj("reviews/expert/planning-gap-v2.expert_comparison.execution.json")
    assert ex["actual_exit_code"] == 0 and ex["records"] == 408 and ex["first_final_byte_identical"] is True
    read(ex["actual_agent_inspection"]["path"])
    red = obj("audits/execution/planning-gap-v2-adversarial.json")
    assert red["completed"] is True and red["rows"] == 408 and red["actual_execution_id"] == "/root/adversarial_auditor"
    assert red["final_judgment_sha256"] == BINDINGS["reviews/adversarial/planning-gap-v2.jsonl"]["sha256"]
    for path, digest in red["files_sha256_before_after"].values():
        assert sha(read(path)) == digest
    rf = obj("reviews/adversarial/planning-gap-v2.input_freeze.json")
    assert rf["candidate_metadata_and_gold_not_observed"] is True
    for path, digest in rf["all_artifacts_sha256"].items():
        assert sha(read(path)) == digest
    rp = obj("audits/execution/planning-gap-v2-adversarial-comparison.preexec.json")
    # The actual red input completion receipt exposes its timestamp separately.
    rx = obj("audits/execution/planning-gap-v2-adversarial-input-freeze.json")
    read("audits/execution/planning-gap-v2-adversarial-preparation-failure-custody.json")
    read("reviews/expert/semantic_inspection_planning_gap_v2_input.md")
    read("audits/diversity/planning-gap-v2-adversarial-canonical.json")
    read("audits/diversity/planning-gap-v2-parent-invariant-overlap.json")
    return joined, {"expert_input_freeze_utc": ef["created_utc"], "expert_comparison_start_utc": ep["created_utc"],
                    "red_actual_input_freeze": rx, "red_actual_comparison_preexecution": rp,
                    "disagreements": sum(joined["expert"][sid]["decision"] != joined["adversarial"][sid]["decision"] for sid in index)}


def audit():
    start, started = now(), time.monotonic()
    setup()
    source = read("reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py")
    public = {path: sha(read(path)) for path in P.PUBLIC_DOCS}
    for p in ["research/curricula/RGM-PLANNING-GAP-V2-ROOT_FREEZE.json", "research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json", "reviews/adjudication/planning-gap-v2-preparation.execution.json"]:
        read(p)
    candidate, manifest = rows(CANDIDATE), obj(MANIFEST)
    index = P.pending_record_index(candidate, 408)
    P.public_binding_check(manifest, public)
    assert manifest["revision_of_batch"] == "planning-gap-v1" and manifest["parent_input_view_sha256"] == EXPECTED["reviews/input_only/planning-gap-v1.jsonl"]
    assert manifest["batch_name"] == "planning-gap-v2" and manifest["track_id"] == "RGM-001" and manifest["semantics_sha256"] == public["schemas/SEMANTICS.md"]
    inputs = rows("reviews/input_only/planning-gap-v2.jsonl")
    input_index = {r["sample_id"]: r for r in inputs}
    assert len(inputs) == len(input_index) == 408 and set(input_index) == set(index)
    prior = rows("research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
    parent = rows("reviews/input_only/planning-gap-v1.jsonl")
    reviews, custody = review_custody(index)
    root_join = obj("quality_reports/RGM_V2_FROZEN_REVIEW_EVIDENCE_JOIN.json")
    leakage = obj("audits/leakage/planning-gap-v2.json")
    machine = obj("audits/diversity/planning-gap-v2-planning-controls.json")["report"]
    full_suite = obj("quality_reports/RGM_V2_FULL_EXECUTION.json")
    assert full_suite["exit_code"] == 0 and full_suite["bound_sources_unchanged"] is True
    assert "358" in full_suite["actual_output_tail"]
    read("quality_reports/RGM_V2_AUDIT_EXECUTION.json")
    read("quality_reports/RGM_V2_RENDERER_CONTROL_FAILURE_DIAGNOSTIC.json")
    read("reviews/oracle/RGM-v2-renderer-public-contract-assessment-20261010T062719Z.md")
    formal = obj("reviews/oracle/RGM-v2-renderer-public-assessment-receipt-20261010T062719Z.json")
    assert formal["status"] == "COMPLETED_FROZEN_BOUNDED_ASSESSMENT"
    for artifact in formal["bindings"]:
        # Only the actual report and receipt are interpreted; probe source is custody hash only.
        assert sha(read(artifact["path"])) == artifact["sha256"]
    read("reviews/oracle/planning-gap-v2.jsonl")
    # Authorized custody manifests, without importing source or reading private expected fixtures.
    reproduction = obj("manifests/planning-gap-v2_REPRODUCTION.json")
    assert reproduction["passed"] is True and len(reproduction["full_deterministic_source_candidate_and_manifest_comparisons"]) == 8
    assert all(c["main_sha256"] == c["replay_sha256"] for c in reproduction["full_deterministic_source_candidate_and_manifest_comparisons"])
    cache, details, parsed_by_id = {}, {}, {}
    for row in candidate:
        sid, task = row["sample_id"], row["semantic_task"]
        V1.task_schema(task)
        parsed = json.loads(row["rendered_input"].split("Task JSON:\n", 1)[1])
        assert P.typed_equal(parsed, task)
        parsed_by_id[sid] = parsed
        frozen = input_index[sid]
        assert set(frozen) == {"record_hash", "rendered_input", "sample_id", "semantic_task", "source_family_key", "split"}
        assert all(P.typed_equal(frozen[k], row[k]) for k in frozen)
        assert row["track_id"] == "RGM-001" and row["producer_agent_id"] == "/root/relational_critic_producer"
        assert row["generator_version"] == "rgm-planning-gap-2.0.2" and row["oracle_version"] == "independent-1.0.0"
        assert row["source_provenance"]["sealed_evaluation_used"] is False
        assert type(row["difficulty"]) is int and 2 <= row["difficulty"] <= 4 and row["structural_difficulty"] == row["difficulty"]
        k = semantic_key(task)
        if k not in cache:
            target, failure = FN["graph_replay"](task) if task["kind"] == "graph" else FN["blocks_replay"](task)
            minimum, search = measured_shortest(task)
            assert P.typed_equal(minimum, target["shortest_plan"])
            cache[k] = (target, failure, search)
        target, failure, search = cache[k]
        assert P.typed_equal(target, row["verified_target"]), sid
        details[sid] = {"sample_id": sid, "record_hash": row["record_hash"], "source_family_key": row["source_family_key"],
                        "context": row["controlled_context"], "renderer": row["renderer"], "split": row["split"],
                        "own_typed_target_hash_prompt_view_match": True, "failure": failure,
                        "fresh_minimum_moves": None if target["shortest_plan"] is None else len(target["shortest_plan"]),
                        "complete_finite_search": search, "public_witnesses": []}
    assert len(cache) == 204
    pair_bindings = P.control_renderer_bindings(manifest["families"], index)
    assert pair_bindings["verified_pair_binding_declarations"] == 192
    covered, family_results, controls = [], {}, []
    properties, a_subtypes, sizes, c1_positions, c1_balances, c4_positions = [Counter() for _ in range(6)]
    c0_temp, b_ties, style_mismatches, style_evidence = [], [], [], []
    for family in manifest["families"]:
        n = family["source_index"]
        allocation = P.expected_allocation(n)
        s, split = allocation["stratum"], allocation["split"]
        assert family["primary_stratum"] == s and family["split"] == split
        assert family["source_family_key"] == f"RGM-001-seed20261009-p{n:04d}" and n not in family_results
        contexts, by_style = {}, {}
        for context in family["contexts"]:
            cid, ids, tags = context["context_id"], context["sample_ids"], context["row_properties"]
            assigned = [index[sid] for sid in ids]
            assert len(tags) == len(set(tags)) and all(type(tag) is str for tag in tags)
            assert all(row["controlled_context"] == cid and row["source_family_index"] == n and row["primary_stratum"] == s for row in assigned)
            assert semantic_key(assigned[0]["semantic_task"]) == semantic_key(assigned[1]["semantic_task"])
            assert P.typed_equal(assigned[0]["verified_target"], assigned[1]["verified_target"])
            assert {r["renderer"] for r in assigned} == {"forward", "reverse"}
            observed = set()
            for witness in context["row_witnesses"]:
                sid, prop = witness["sample_id"], witness["property"]
                assert witness["scope"] == "row" and sid in ids and prop in tags and (sid, prop) not in observed
                observed.add((sid, prop))
                task = index[sid]["semantic_task"]
                target, failure, _ = cache[semantic_key(task)]
                proof = property_truth(task, target, failure, witness, n)
                details[sid]["public_witnesses"].append({"property": prop, "evidence": witness["evidence"], "truth": True, "own_proof": proof})
                properties[prop] += 1
            assert observed == {(sid, prop) for sid in ids for prop in tags}
            for row in assigned:
                assert set(tags) <= set(row["skill_tags"])
                assert set(row["skill_tags"]) - {"finite_synthetic_proxy", "relational_planning"} == set(tags)
                rendered = parsed_by_id[row["sample_id"]]
                forward_order = ["kind", "nodes", "edges", "start", "goal", "fuel", "plan"] if rendered["kind"] == "graph" else ["kind", "on", "goal_on", "plan"]
                assert list(rendered) == (forward_order if row["renderer"] == "forward" else list(reversed(forward_order)))
                if rendered["kind"] == "blocks":
                    assert list(rendered["on"]) == sorted(rendered["on"], reverse=row["renderer"] == "reverse")
                    assert all(list(action) == (["block", "to"] if row["renderer"] == "forward" else ["to", "block"]) for action in rendered["plan"])
            left, right = [parsed_by_id[r["sample_id"]] for r in assigned]
            assert list(left) == list(reversed(list(right)))
            if left["kind"] == "graph":
                assert left["nodes"] == list(reversed(right["nodes"])) and left["edges"] == list(reversed(right["edges"]))
            else:
                assert list(left["on"]) == list(reversed(list(right["on"]))) and list(left["goal_on"]) == list(reversed(list(right["goal_on"])))
            covered.extend(ids)
            contexts[cid] = assigned[0]
            by_style[cid] = {r["renderer"]: r for r in assigned}
        assert set(contexts) == {s + str(i) for i in range(allocation["contexts"])}
        base = index[family["base_sample_id"]]
        assert base["controlled_context"] == s + "0" and base["source_family_index"] == n
        t0 = base["semantic_task"]
        minimum0 = cache[semantic_key(t0)][0]["shortest_plan"]
        out0 = cache[semantic_key(t0)][0]
        assert out0["valid"] and out0["reaches_goal"]
        size = len(t0["nodes"] if t0["kind"] == "graph" else t0["on"])
        assert size == allocation["objects"]
        sizes[f"{s}:{split}:{size}"] += 1
        if s == "A":
            t1 = contexts["A1"]["semantic_task"]
            o1 = cache[semantic_key(t1)][0]
            assert not o1["valid"] and FN["shortest_graph"](t1, True) is not None
            if o1["shortest_plan"] is None:
                subtype = "fresh_infeasible"
                cost = FN["cheapest_graph"](t0)
                assert t0["fuel"] == cost and t1["fuel"] == cost - 1
            else:
                subtype = "fuel_lengthened"
                assert len(o1["shortest_plan"]) > len(minimum0)
            assert subtype == allocation["a_subtype"]
            a_subtypes[f"{split}:{subtype}:{size}"] += 1
            for cid in ["A2", "A3"]:
                other = contexts[cid]["semantic_task"]
                assert P.typed_equal({k:v for k,v in t0.items() if k != "plan"}, {k:v for k,v in other.items() if k != "plan"})
            a2 = contexts["A2"]["semantic_task"]
            o2 = cache[semantic_key(a2)][0]
            assert a2["plan"] and o2["valid"] and not o2["reaches_goal"]
            f3 = cache[semantic_key(contexts["A3"]["semantic_task"])][1]
            assert f3["reason"] == "missing_directed_edge" and f3["index"] >= 1
        elif s == "B":
            assert len(t0["plan"]) >= 3
            b_ties.append(FN["shortest_graph_count"](t0))
            t1 = contexts["B1"]["semantic_task"]
            o1, f1, _ = cache[semantic_key(t1)]
            assert o1["shortest_plan"] is not None and f1["index"] >= 1 and f1["reason"] == "missing_directed_edge"
            t2 = contexts["B2"]["semantic_task"]
            o2, f2, _ = cache[semantic_key(t2)]
            assert f2["reason"] == "missing_directed_edge" and f2["index"] >= 1
            before = o2["steps"][f2["index"] - 1]["after"]["node"]
            assert (t2["plan"][f2["index"]], before) in {(a,b) for a,b,_ in t2["edges"]}
        elif s == "C":
            assert 4 <= len(t0["plan"]) == len(minimum0) <= 8
            assert "later_action_requires_changed_block_clearance" in {w["property"] for w in details[base["sample_id"]]["public_witnesses"]}
            c1 = contexts["C1"]["semantic_task"]
            _, failure, _ = cache[semantic_key(c1)]
            reason = "source_not_clear" if failure["reason"] == "blocked_source" else "destination_not_clear"
            assert reason == allocation["c1_reason"]
            c1_positions[f"{reason}:{failure['index']}"] += 1
            c1_balances[f"{reason}:{split}:{size}"] += 1
            c3 = contexts["C3"]["semantic_task"]
            o3 = cache[semantic_key(c3)][0]
            assert c3["plan"] == t0["plan"][:allocation["c4_prefix"]] and o3["valid"] and not o3["reaches_goal"]
            c4_positions[f"{split}:{allocation['c4_prefix']}"] += 1
            if V1.restricted_block_search(t0)["necessary"]:
                c0_temp.append(n)
        else:
            assert 0 < len(t0["goal_on"]) < len(t0["on"]) and len(t0["plan"]) == len(minimum0) >= 2
            t1 = contexts["D1"]["semantic_task"]
            o1 = cache[semantic_key(t1)][0]
            assert o1["valid"] and not o1["reaches_goal"] and len(o1["shortest_plan"]) > len(minimum0)
            t2 = contexts["D2"]["semantic_task"]
            o2 = cache[semantic_key(t2)][0]
            assert o2["valid"] and o2["reaches_goal"] and len(t2["plan"]) == len(o2["shortest_plan"])
            assert P.typed_equal({k:v for k,v in t1.items() if k != "plan"}, {k:v for k,v in t2.items() if k != "plan"})
            shared = set(t0["goal_on"])
            for cid, style_rows in by_style.items():
                for style, row in style_rows.items():
                    parsed = parsed_by_id[row["sample_id"]]
                    baseline = parsed_by_id[by_style["D0"][style]["sample_id"]]
                    ordinary = P.typed_equal(renderer_projection(parsed), renderer_projection(baseline))
                    full = P.typed_equal(renderer_projection(parsed, shared), renderer_projection(baseline, shared))
                    assert ordinary
                    if not full:
                        style_mismatches.append(row["sample_id"])
                    style_evidence.append({"sample_id": row["sample_id"], "record_hash": row["record_hash"],
                                          "context": cid, "renderer": style, "common_non_goal_style_projection_matches": ordinary,
                                          "shared_goal_order_matches": full, "actual_shared_goal_order": [b for b in parsed["goal_on"] if b in shared],
                                          "base_shared_goal_order": [b for b in baseline["goal_on"] if b in shared]})
        for control in family["controls"]:
            controls.append(control_truth(control, index, cache))
        family_results[n] = {"source_family_key": family["source_family_key"], "stratum": s, "split": split,
                             "base": base, "contexts": sorted(contexts), "all_public_semantic_controls_true": True}
    assert len(covered) == len(set(covered)) == 408 and set(covered) == set(index)
    assert len(family_results) == 48 and set(family_results) == set(range(180,228))
    assert sum(properties.values()) == 510 and len(controls) == 192
    assert len(c0_temp) >= 4 and sum(count >= 2 for count in b_ties) >= 6
    # Public novelty is measured on base state structures, not IDs/presentation counts.
    old_tasks = {semantic_key(r["semantic_task"]):r["semantic_task"] for r in prior}
    old_graph = [t for t in old_tasks.values() if t["kind"] == "graph"]
    old_blocks = [t for t in old_tasks.values() if t["kind"] == "blocks"]
    new_graph = [f["base"]["semantic_task"] for f in family_results.values() if f["stratum"] in "AB"]
    new_blocks = [f["base"]["semantic_task"] for f in family_results.values() if f["stratum"] in "CD"]
    old_weighted = {FN["graph_fingerprint"](t, True, True) for t in old_graph}
    old_unweighted = {FN["graph_fingerprint"](t) for t in old_graph}
    old_joint = {FN["block_fingerprint"](t, True) for t in old_blocks}
    weighted = {FN["graph_fingerprint"](t,True,True) for t in new_graph}
    unweighted = {FN["graph_fingerprint"](t) for t in new_graph}
    joint = {FN["block_fingerprint"](t,True) for t in new_blocks}
    six_shapes = {FN["block_fingerprint"](t) for t in new_blocks if len(t["on"]) == 6}
    initial_shapes = {FN["block_fingerprint"](t) for t in new_blocks}
    assert len(weighted) == 24 and not weighted & old_weighted
    assert len(unweighted - old_unweighted) >= 16
    assert len(joint) == 24 and not joint & old_joint and len(six_shapes) >= 8
    assert not set(index) & {r["sample_id"] for r in prior}
    assert not {r["source_family_key"] for r in candidate} & {r["source_family_key"] for r in prior}
    assert {r["source_family_key"] for r in candidate} == {r["source_family_key"] for r in parent}
    assert not set(cache) & set(old_tasks)
    lineage_splits = defaultdict(set)
    task_lineages = defaultdict(set)
    prompts = []
    for row in candidate:
        lineage_splits[row["source_family_key"]].add(row["split"])
        task_lineages[semantic_key(row["semantic_task"])].add(row["source_family_key"])
        prompts.append(row["rendered_input"])
    assert all(len(values) == 1 for values in lineage_splits.values())
    assert all(len(values) == 1 for values in task_lineages.values()) and len(prompts) == len(set(prompts))
    assert len(style_mismatches) == 66
    mismatch_families = {index[sid]["source_family_key"] for sid in style_mismatches}
    assert len(mismatch_families) == len(machine["errors"]) == 11
    assert mismatch_families == {e.split(": UNKNOWN:")[0] for e in machine["errors"]}
    assert leakage["oracle_failures"] == leakage["metadata_failures"] == leakage["validation_failures"] == 0
    assert leakage["all_machine_quality_gates_pass"] is False
    # Conditional histograms are independently counted once per semantic context, not twice per view.
    histograms = defaultdict(Counter)
    for row in candidate:
        if row["renderer"] != "forward":
            continue
        task = row["semantic_task"]
        target, failure, _ = cache[semantic_key(task)]
        values = {"valid": target["valid"], "reaches_goal": target["reaches_goal"], "fresh_feasible": target["shortest_plan"] is not None,
                  "minimum_moves": None if target["shortest_plan"] is None else len(target["shortest_plan"]),
                  "submitted_moves": len(task["plan"]), "first_error_index": None if failure is None else failure["index"],
                  "first_error_reason": None if failure is None else failure["reason"],
                  "objects": len(task["nodes"] if task["kind"] == "graph" else task["on"]),
                  "edges": len(task["edges"]) if task["kind"] == "graph" else None, "fuel": task.get("fuel")}
        for grouping in ["all", row["split"], row["primary_stratum"] + ":" + row["split"], row["controlled_context"] + ":" + row["split"]]:
            for key, value in values.items():
                histograms[grouping + ":" + key][canonical(value)] += 1
    # Actual input-only train-majority scalar-fuel baseline for fresh feasibility.
    train, dev = [], []
    for row in candidate:
        if row["renderer"] == "forward":
            label = cache[semantic_key(row["semantic_task"])][0]["shortest_plan"] is not None
            (train if row["split"] == "train" else dev).append((row["semantic_task"].get("fuel"), label, row["sample_id"]))
    buckets = defaultdict(Counter)
    for feature, label, sid in train:
        buckets[feature][label] += 1
    majority = Counter(label for _,label,_ in train).most_common(1)[0][0]
    lookup = {feature: sorted(counts, key=lambda label:(-counts[label],canonical(label)))[0] for feature,counts in buckets.items()}
    wrong = [sid for feature,label,sid in dev if lookup.get(feature,majority) != label]
    majority_correct = sum(label == majority for _,label,_ in dev)
    # Custody claims are not new generation executions by this adjudicator.
    reproduction_scope = {"declared_actual_execution_id": reproduction["actual_execution_id"], "all_eight_hash_comparisons_equal": True,
                          "hash_seeds": reproduction["hash_seeds"], "adjudicator_reran_producer": False,
                          "producer_source_or_private_fixture_contents_read": False}
    decision = "QUARANTINE"
    result = {"actual_execution_id": IDENTITY, "started_utc": start, "completed_utc": now(), "actual_wall_seconds": time.monotonic()-started,
              "source_sha256": sha(source), "bindings": BINDINGS, "records": 408, "semantic_inputs": len(cache), "source_families": 48,
              "source_family_growth_relative_pending_v1": 0, "actual_public_semantic_witnesses": dict(properties),
              "all510_witnesses_true": True, "all192_supplied_declarations_true": True, "semantic_control_pairs": 96,
              "controls": controls, "per_record": list(details.values()), "family_allocations": sizes,
              "source_families_by_split": dict(Counter(f["split"] for f in family_results.values())),
              "coverage": {"a_subtypes_by_size_split": a_subtypes, "C1_reason_size_split": c1_balances, "C1_reason_first_positions": c1_positions,
                           "C4_prefix_split": c4_positions, "C0_required_temporary_displacement_source_indices": c0_temp,
                           "B0_multiple_minimum_routes_families": sum(count>=2 for count in b_ties)},
              "novelty": {"weighted_terminal_graph_bases": len(weighted), "new_unweighted_graph_classes": len(unweighted-old_unweighted),
                          "joint_block_bases": len(joint), "six_block_initial_shapes": len(six_shapes), "all_block_initial_shapes": len(initial_shapes),
                          "prior_disjoint": True, "same_split_lineages": True, "cross_parent_semantic_duplicates": 0, "exact_prompt_duplicates": 0},
              "conditional_histograms_semantic_unit": {k:dict(v) for k,v in sorted(histograms.items())},
              "fuel_fresh_feasibility_baseline": {"train_semantic_inputs":len(train),"dev_semantic_inputs":len(dev),"dev_correct":len(dev)-len(wrong),
                       "actual_wrong_ids":wrong,"global_majority_dev_correct":majority_correct,"mixed_training_fuel_values":[k for k,v in buckets.items() if len(v)>1],
                       "no_model_or_sealed_ood_claim":True},
              "renderer_contract_assessment": {"explicit_same_actual_style_and_two_ID_pairs_mandatory":True,
                       "both_views_reverse_actual_keys_lists_without_values_or_action_sequence_change":True,
                       "shared_goal_relative_key_order_preservation_explicitly_specified":False,
                       "shared_goal_order_changes":style_evidence,"direct_full_projection_mismatch_ids":style_mismatches,
                       "direct_full_projection_mismatch_families":sorted(mismatch_families),"direct_mismatch_rows":66,
                       "no_false_target_or_false_property_established":True,
                       "common_non_goal_projection_is_diagnostic_not_substitute_acceptance_gate":True,
                       "actual_producer_rendering_function_provenance_independently_certified":False,
                       "ambiguity":"The public same-style obligation has no exact identity/fingerprint algorithm. It does not publish preservation of shared goal-key insertion order when semantic goal fields change. Declared labels alone are insufficient, but observed other key/list/action styles stay consistent and paired views truly reverse. The stronger frozen predicate is unresolved, not an established false gold/property."},
              "machine_gate": {"unchanged_error_count":11,"errors":machine["errors"],"completion_counts":machine["counts"],
                        "actual_present_declarations":192,"completed_gate_declarations_before_aborts":148,
                        "all_machine_quality_gates_pass":False,"passing358_suite_overrides_gate":False},
              "review_join": {"expert": "ACCEPT408 first/final identical", "adversarial":"ACCEPT408 first/final identical",
                       "reviewer_disagreements":custody["disagreements"],"scoped_factual_judgments_preserved":True,"custody":custody},
              "reproduction_custody_scope":reproduction_scope,"proposed_decision_counts":{decision:408},"publication_eligible":False,
              "whole_cohort_hold":True,"partial_release_breaks_required_family_context_renderer_control_coverage":True,
              "decision_reason":"Hold exact complete cohort for unresolved required frozen renderer-style admission. All targets and factual witnesses/pairs pass own public-derived computation. Additional shared-goal order predicate is not explicitly published; no retroactive criterion/gate relaxation or target rejection. Fresh-source/fresh-candidate remediation and full new review are required.",
              "limitations":["408 correlated rows/204 supplied contexts/48 revised source lineages under two existing deterministic grammars, not408 independent mechanisms or added48 versus V1.",
                    "Only one exposed-dev infeasible semantic task; scalar-fuel train lookup may exploit finite conditional associations, not a valid universal rule.",
                    "Block arrangement motifs and context outcome correlations remain; source/destination/no-op controls now selective but do not establish model acquisition.",
                    "Difficulty2–4 labels are structural heuristics; object count and detour length do not calibrate model difficulty.",
                    "Fresh replacement minimum plans are not edit-distance-minimal repairs; no coupled-resource/logistics/new action grammar."],
              "independence":DISCLOSURE,"producer_private_source_or_expectations_read_before_conclusion":False,
              "immutable_v1_quarantine_not_changed":True,"final_per_record_decisions_emitted_in_this_audit":False}
    for path, binding in list(BINDINGS.items()):
        assert sha((ROOT/path).read_bytes()) == binding["sha256"], ("evidence changed during audit",path)
    binding = write_new(BASE + ".audit.json", result)
    print(canonical({"audit":binding,"proposed_decisions":{decision:408},"targets":408,"witnesses":510,"declarations":192,"direct_renderer_mismatch_rows":66,"actual_wall_seconds":result["actual_wall_seconds"]}))


def finalize():
    """Finalization waits for NEW completed red final custody confirmation supplied by root."""
    audit_data = obj(BASE + ".audit.json")
    assert audit_data["publication_eligible"] is False and audit_data["proposed_decision_counts"] == {"QUARANTINE":408}
    for path,binding in audit_data["bindings"].items():
        assert sha((ROOT/path).read_bytes()) == binding["sha256"], ("frozen audit input changed",path)
    # Explicit command-line custody receipt path, no discovery of pending reviews.
    assert len(sys.argv) == 3, "finalize requires root-confirmed completed red custody receipt path"
    final_custody = obj(sys.argv[2])
    assert final_custody.get("completed") is True or final_custody.get("completed_execution") is True or final_custody.get("status", "").startswith("COMPLETED"), "final red custody confirmation incomplete"
    candidate = rows(CANDIDATE)
    expert = {r["sample_id"]:r for r in rows("reviews/expert/planning-gap-v2.jsonl")}
    red = {r["sample_id"]:r for r in rows("reviews/adversarial/planning-gap-v2.jsonl")}
    affected = set(audit_data["renderer_contract_assessment"]["direct_full_projection_mismatch_ids"])
    a_binding = BINDINGS[BASE + ".audit.json"]
    own_source = read("reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py")
    verdicts = []
    for row in candidate:
        sid = row["sample_id"]
        issue_scope = "direct_frozen_style_projection_UNKNOWN" if sid in affected else "whole_cohort_required_admission_dependency"
        verdicts.append({"receipt_id":"adj-planning-gap-v2-"+sha(canonical([sid,row["record_hash"],IDENTITY,issue_scope]).encode())[:24],
              "sample_id":sid,"record_hash":row["record_hash"],"reviewer_role":"independent_adjudicator","reviewer_execution_id":IDENTITY,
              "scores":{"target_correctness":5,"semantic_correctness":5,"reasoning_validity":5,"instruction_clarity":5,
                        "metadata_machine_interface":2,"difficulty_quality":3,"diversity":4,"generalization_value":3,"shortcut_resistance":3},
              "critical_defects":["unresolved_required_frozen_renderer_admission"],
              "issues":[{"category":"public_style_identity_and_frozen_machine_gate_unresolved","severity":"high","scope":issue_scope,
                    "evidence":"Own408 target/hash/input joins,510 witnesses and192 supplied per-renderer controls pass. Frozen gate aborts11 D families/66 projected rows under a shared-goal insertion-order assumption not explicitly published. No false gold/property established; same-style interpretation remains insufficiently specified for current admission.",
                    "fix":"Keep exact candidate/spec/checker/reviewer firsts unchanged. Freeze a fresh producer rendering source and candidate with documented stable goal-key style matching the unchanged gate, then rerun complete independent reviews; any alternative gate definition must be prospective and versioned."}],
              "decision":"QUARANTINE","confidence":0.97,"reason":audit_data["decision_reason"],
              "evidence":{"candidate_sha256":EXPECTED[CANDIDATE],"manifest_sha256":EXPECTED[MANIFEST],"own_audit":a_binding,
                    "adjudicator_source_sha256":sha(own_source),"own_typed_target_and_factual_witnesses_pass":True,
                    "all96_semantic_pairs_192_actual_style_declarations_pass":True,"direct_frozen_projection_mismatch":sid in affected,
                    "expert_receipt_id":expert[sid]["receipt_id"],"expert_decision":expert[sid]["decision"],
                    "adversarial_receipt_id":red[sid]["receipt_id"],"adversarial_decision":red[sid]["decision"],"reviewer_disagreement":False,
                    "scoped_first_factual_ACCEPTs_preserved":True,"publication_eligible":False,
                    "final_red_custody":BINDINGS[sys.argv[2]],"machine_report_sha256":EXPECTED["audits/diversity/planning-gap-v2-planning-controls.json"]},
              "guidance":DISCLOSURE})
    verdict_binding = write_new(BASE + ".jsonl", "".join(canonical(v)+"\n" for v in verdicts),raw=True)
    ticket = {"ticket_id":"RGM-RENDERER-008","status":"OPEN_FROZEN_V2_WHOLE_COHORT_PUBLICATION_HOLD","actual_author_execution_id":IDENTITY,
              "candidate":BINDINGS[CANDIDATE],"manifest":BINDINGS[MANIFEST],"audit":a_binding,"verdicts":verdict_binding,
              "decision_counts":{"QUARANTINE":408},"whole_cohort_hold":True,"no_partial_publication":True,
              "public_contract_boundary":"Same actual renderer style, per-style two-ID evidence and all192 controls are mandatory. No frozen public doc explicitly requires shared-goal relative insertion-order preservation under goal extension. Do not retroactively infer that added fingerprint criterion.",
              "actual_present_truth":{"typed_targets":408,"factual_witnesses":510,"per_renderer_declarations":192,"semantic_pairs":96,"complete_families":48},
              "actual_gate_failure":{"UNKNOWN_families":11,"direct_projection_mismatch_rows":66,"report":audit_data["bindings"]["audits/diversity/planning-gap-v2-planning-controls.json"],
                   "completed_declarations_before_aborts":148,"not_absent_controls":True,"not_false_gold":True},
              "prospective_remediation":{"preferred":"Freeze NEW producer source documenting stable goal-key ordering within each style, preserve shared-key projection and all original values/plans/controls. Generate fresh candidate/manifest under distinct prospective custody/batch/version handling and review all rows again against unchanged frozen gate.",
                   "alternative":"If style identity is intentionally different, first publish a NEW prospective public definition and freeze NEW checker/source/positive-negative tests before fresh generation. Reject mixed/missing styles and false/foreign pairs. Never silently delete the goal-order check to pass current candidate.",
                   "fresh_source_candidate_manifest_hashes_and_actual_execution_required":True,"old_bytes_and_FAIL_history_preserved":True,
                   "new_mechanisms_or_family_growth_from_rerendering":0,"future_candidate_not_generated_or_approved":True},
              "scoped_expert_red_ACCEPT_firsts_unchanged":True,"design_authorship_ambiguity_disclosed":True,"guidance":DISCLOSURE}
    ticket_binding = write_new("feedback/improvement_requests/RGM-RENDERER-008.json",ticket)
    report = """# Actual planning-gap-v2 adjudication\n\nAll **408 exact records are QUARANTINE**, with no publication eligibility or partial release. This is an unresolved required renderer-admission hold, not a claim that the stored answers or factual properties are false. The immutable V1 whole-cohort quarantine is unchanged.\n\nOwn public-derived replay and complete finite-state searches verify all408 typed targets, record hashes, parsed prompts and six-field frozen input joins. All510 row-property witnesses and192 supplied per-renderer declarations (96 semantic controls) are truthful and bound to48 complete revised source families/204 semantic tasks. A1 now has both infeasible and fuel-lengthened exposed-dev families. C1 selective source/destination controls have6 families each, including one dev of each; C4 supplies12 isolated no-op negatives after legal non-goal prefixes. Exact allocation, error-position, size, search and outcome histograms are in the audit.\n\nThe frozen machine report still has11 UNKNOWN family failures. Own input-order measurements localize66 D1/D2/D3 renderer rows whose shared goal-key order differs from D0. Actual key/list/action styles otherwise remain consistent, and each context's paired views truly reverse their field/list orders without changing semantics. The public addendum requires the same actual style and both two-ID declarations. It does **not** publish an exact style fingerprint or shared-goal insertion-order preservation under a goal-only extension. That omission was in this author's public design; the narrower diagnostic projection is not a replacement acceptance gate or proof of producer rendering-function custody. The stronger frozen gate remains unresolved. All192 controls exist; the machine's148 count is completion before11 aborts, not evidence of44 missing declarations.\n\nExpert and adversarial first/final judgments remain408 scoped ACCEPT each with zero disagreement. Their actual all408 input predictions froze before canonical comparison. Formal's completed public-only assessment agrees that the disputed shared-goal projection is an additional sufficient implementation inference, not an explicit published rule. This adjudicator independently assessed actual payloads and public wording. Passing358 tests and zero target/metadata/validation errors do not discharge the separate11 cohort admission errors. Releasing selected rows would break the required complete context/style/control cohort.\n\nPreferred remediation: preserve every current artifact, freeze a NEW producer source with a documented stable goal-key order for each renderer and a fresh candidate/manifest satisfying the unchanged gate, then require new complete review/custody. Alternatively publish a NEW prospective style definition and freeze a new checker with handmade positive and negative cases before fresh generation. Neither path retroactively approves V2, edits old bytes or creates new families/mechanisms. See `feedback/improvement_requests/RGM-RENDERER-008.json`.\n\nThe audit independently confirms prior-disjoint base-structure novelty, single-split lineages and no exact/cross-parent task duplication. These are48 revised lineages versus pending V1, with zero additional families from the correction. Correlated context/renderer views remain under two existing finite grammars. One dev semantic task is freshly infeasible; input-only scalar-fuel lookup and other conditional associations remain descriptive finite-corpus risks. Structural difficulty2–4 is not model-calibrated; minimum replacement planning is not edit-distance repair. No sealed OOD, learning, architecture or efficacy result is claimed.\n\n""" + DISCLOSURE + "\n\nActual source, before-execution archive, all input/output hashes, failures (if any) and tool completion are bound in the execution receipts. No producer private implementation or expected hand fixtures were read before this conclusion froze. No candidate, manifest, checker, existing specification, other review, root code or release was edited.\n"
    report_binding = write_new(BASE + ".report.md",report,raw=True)
    execution = {"actual_execution_id":IDENTITY,"completed_execution":True,"completed_utc":now(),"phase":"FINAL_ADJUDICATION_AFTER_COMPLETED_COMPARISON_CUSTODY",
                 "candidate":BINDINGS[CANDIDATE],"manifest":BINDINGS[MANIFEST],"audit":a_binding,"verdicts":verdict_binding,"ticket":ticket_binding,
                 "report":report_binding,"source_sha256":sha(own_source),"final_custody_receipt":BINDINGS[sys.argv[2]],
                 "all_bound_audit_inputs_unchanged":True,"decision_counts":{"QUARANTINE":408},"publication_eligible":False,"guidance":DISCLOSURE}
    execution_binding = write_new(BASE + ".execution.json",execution)
    print(canonical({"verdicts":verdict_binding,"execution":execution_binding,"report":report_binding,"ticket":ticket_binding,"publication_eligible":False}))


if __name__ == "__main__":
    try:
        {"audit":audit,"finalize":finalize}[sys.argv[1]]()
    except Exception:
        failure = {"actual_execution_id":IDENTITY,"at_utc":now(),"actual_command":sys.argv,"exception":traceback.format_exc(),
                   "source_sha256":sha(Path(__file__).read_bytes()),"bindings_observed_before_failure":BINDINGS,"no_approval_from_failure":True}
        path = BASE + ".owned-failure-" + sha(canonical(failure).encode())[:16] + ".json"
        write_new(path,failure)
        print(canonical({"owned_failure_preserved":path}),file=sys.stderr)
        raise
