#!/opt/anaconda3/bin/python3
"""New expert implementation, derived only from public belief semantics/design.

Preparation runs only authored toy fixtures. `predict` accepts a target-free view,
freezes typed expectations/facts/audits, and never issues dataset judgments.
`compare` is a separate post-freeze operation; no producer/oracle imports exist.
Actual author/execution: /root/expert_reviewer.
"""
import argparse
import collections
import copy
import datetime
import hashlib
import itertools
import json
import pathlib
import re

ROLE = "/root/expert_reviewer"
ROOT = pathlib.Path(__file__).resolve().parents[2]
PUBLIC = ["schemas/EXTENDED_SEMANTICS.md", "schemas/BELIEF_CONTROL_PROPERTIES.md",
          "research/curricula/RBM-EVIDENCE-GAP.md",
          "reviews/expert/RBM-gap-analysis-20261010T015614Z.md"]
FROZEN = {"schemas/EXTENDED_SEMANTICS.md": "2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23",
          "schemas/BELIEF_CONTROL_PROPERTIES.md": "200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034",
          "research/curricula/RBM-EVIDENCE-GAP.md": "004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120",
          "reviews/expert/RBM-gap-analysis-20261010T015614Z.md": "4fcac3c1836683462c670b8c85a859668a54a1158a324219f9da44ef5e37ed1f"}
PROPERTIES = set("multiworld_unobserved_invariant conjunction_required_invariant useful_after_conjunctive_evidence multiworld_all_queries_redundant multiworld_entailed_null ambiguous_null_partition cross_fact_inconsistent_conjunction post_empty_absorption nonempty_repeated_observation typed_observation_separator typed_query_partition_separator skewed_possible_value_partition".split())
FAMILY_CONTROLS = set("equal_support_usefulness_counterfactual observation_reorder_final_invariance nonempty_repetition_final_invariance cross_fact_contradiction_counterfactual post_empty_append_absorption typed_equals_counterfactual null_guarantee_contrast query_id_position_fact_crossing field_order_invariance world_order_invariance world_id_rename_equivariance".split())
SCOPE_TAGS = {"finite_synthetic_proxy", "finite_noiseless_filter", "possible_value_partition"}
CORE_ROLES = {"observation_control", "observation_auxiliary", "invariant_query", "variation_query", "unoffered_feature"}
QUERY_ROLES = ["invariant_query", "variation_query", "observed_control"]
PROFILE = {12: "cyclic_uniform", 13: "acyclic_branching", 18: "cyclic_weighted", 19: "typed_path"}
FORBIDDEN = {"verified_target", "gold", "expected_target", "expected_output", "reference_output", "oracle_output"}


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def equal(a, b):
    """Whole JSON equality that never treats true as integer 1."""
    return canon(a) == canon(b)


def scalar(value):
    if value is None or type(value) in (bool, int, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    raise ValueError("non-contract scalar: " + repr(value))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_bindings():
    bindings = {p: {"sha256": sha((ROOT / p).read_bytes()), "length": (ROOT / p).stat().st_size} for p in PUBLIC}
    assert all(bindings[p]["sha256"] == h for p, h in FROZEN.items()), "public source changed; do not silently adapt"
    return bindings


def nonnegative_int(v):
    return type(v) is int and v >= 0


def validate_task(task):
    assert isinstance(task, dict) and task.get("kind") == "belief", "kind=belief required"
    assert set(task) == {"kind", "worlds", "observations", "queries"}, "unexpected task keys"
    assert all(isinstance(task[k], list) for k in ("worlds", "observations", "queries"))
    ids, keys = set(), None
    for world in task["worlds"]:
        assert set(world) == {"world_id", "facts"} and isinstance(world["world_id"], str)
        assert world["world_id"] not in ids, "duplicate world ID"
        ids.add(world["world_id"])
        assert isinstance(world["facts"], dict) and all(isinstance(f, str) for f in world["facts"])
        if keys is None:
            keys = set(world["facts"])
        assert set(world["facts"]) == keys, "unequal fact sets"
        for v in world["facts"].values():
            scalar(v)
    assert keys is not None, "this pilot requires visible worlds"
    for obs in task["observations"]:
        assert set(obs) == {"fact", "equals"} and obs["fact"] in keys
        scalar(obs["equals"])
    qids = set()
    for query in task["queries"]:
        assert set(query) == {"query_id", "fact"} and isinstance(query["query_id"], str)
        assert query["query_id"] not in qids and query["fact"] in keys
        qids.add(query["query_id"])


def solve(task):
    """Incremental exact filtering and scalar-encoding partitions."""
    validate_task(task)
    survivors = task["worlds"][:]
    prefixes = [[w["world_id"] for w in survivors]]
    for obs in task["observations"]:
        key = scalar(obs["equals"])
        survivors = [w for w in survivors if scalar(w["facts"][obs["fact"]]) == key]
        prefixes.append([w["world_id"] for w in survivors])
    rows = []
    for query in task["queries"]:
        bins = collections.Counter(scalar(w["facts"][query["fact"]]) for w in survivors)
        counts = [{"value": json.loads(k), "count": bins[k]} for k in sorted(bins)]
        entailed = bool(survivors) and len(counts) == 1
        rows.append({"query_id": query["query_id"], "counts": counts, "distinct_values": len(counts),
                     "entailed": entailed, "guaranteed": counts[0]["value"] if entailed else None,
                     "can_reduce_uncertainty": len(counts) > 1})
    return {"surviving_worlds": sorted(w["world_id"] for w in survivors),
            "belief_sizes": [len(p) for p in prefixes[1:]], "queries": rows}, [sorted(p) for p in prefixes]


def solve_conjunction(task):
    """Independent direct prefix conjunction; no incremental survivor reuse."""
    validate_task(task)
    prefixes = []
    for count in range(len(task["observations"]) + 1):
        prefixes.append(sorted(w["world_id"] for w in task["worlds"] if all(
            type(w["facts"][o["fact"]]) is type(o["equals"]) and w["facts"][o["fact"]] == o["equals"]
            for o in task["observations"][:count])))
    selected = [w for w in task["worlds"] if w["world_id"] in prefixes[-1]]
    query_rows = []
    for q in task["queries"]:
        values = sorted({scalar(w["facts"][q["fact"]]) for w in selected})
        counts = [{"value": json.loads(v), "count": sum(scalar(w["facts"][q["fact"]]) == v for w in selected)} for v in values]
        entails = len(counts) == 1
        query_rows.append({"query_id": q["query_id"], "counts": counts, "distinct_values": len(counts),
                           "entailed": entails, "guaranteed": counts[0]["value"] if entails else None,
                           "can_reduce_uncertainty": len(counts) > 1})
    return {"surviving_worlds": prefixes[-1], "belief_sizes": list(map(len, prefixes[1:])), "queries": query_rows}, prefixes


def aliases(value):
    if type(value) is bool:
        return [int(value)]
    if type(value) is int and value in (0, 1):
        return [bool(value)]
    return []


def query_partition(task, ids, fact):
    bins = collections.Counter(scalar(w["facts"][fact]) for w in task["worlds"] if w["world_id"] in ids)
    return [{"value": json.loads(k), "count": bins[k]} for k in sorted(bins)]


def row_facts(task):
    expected, prefixes = solve(task)
    other, other_prefixes = solve_conjunction(task)
    assert equal(expected, other) and equal(prefixes, other_prefixes), "own algorithms disagree"
    observed = {o["fact"] for o in task["observations"]}
    final = prefixes[-1]
    worlds = {w["world_id"]: w for w in task["worlds"]}
    distinct_payloads = len({canon(worlds[i]["facts"]) for i in final})
    m = len(task["observations"])
    witnesses = {name: [] for name in PROPERTIES}

    def add(prop, qids=(), observations=(), pref=(), **detail):
        witnesses[prop].append({"query_ids": list(qids), "observation_indices": list(observations),
                                "prefix_indices": list(pref), **detail})

    strict = [j for j in range(m) if 0 < len(prefixes[j + 1]) < len(prefixes[j])]
    for q, out in zip(task["queries"], expected["queries"]):
        common = {"fact": q["fact"], "surviving_worlds": final, "counts": out["counts"]}
        qid = q["query_id"]
        if len(final) >= 2 and distinct_payloads >= 2 and q["fact"] not in observed and out["distinct_values"] == 1:
            add("multiworld_unobserved_invariant", [qid], pref=[m], **common)
            for a, b in itertools.combinations(range(m), 2):
                if task["observations"][a]["fact"] == task["observations"][b]["fact"]:
                    continue
                omissions = []
                for omitted in (a, b):
                    altered = copy.deepcopy(task)
                    altered["observations"].pop(omitted)
                    omitted_output, _ = solve(altered)
                    om_ids = omitted_output["surviving_worlds"]
                    omissions.append({"omitted_observation": omitted, "surviving_worlds": om_ids,
                                      "counts": query_partition(task, om_ids, q["fact"])})
                if all(len(x["counts"]) >= 2 for x in omissions):
                    add("conjunction_required_invariant", [qid], [a, b], [m], omissions=omissions, **common)
        if final and out["distinct_values"] > 1:
            for a, b in itertools.combinations(strict, 2):
                if task["observations"][a]["fact"] != task["observations"][b]["fact"]:
                    add("useful_after_conjunctive_evidence", [qid], [a, b], sorted({a, a + 1, b, b + 1, m}), **common)
        if len(final) >= 2 and distinct_payloads >= 2 and out["counts"] == [{"value": None, "count": len(final)}]:
            add("multiworld_entailed_null", [qid], pref=[m], **common)
        if out["distinct_values"] > 1 and any(x["value"] is None for x in out["counts"]):
            add("ambiguous_null_partition", [qid], pref=[m], **common)
        encoded_values = {scalar(x["value"]) for x in out["counts"]}
        pairs = [[v, a] for v in (False, True) for a in aliases(v) if scalar(v) in encoded_values and scalar(a) in encoded_values]
        if pairs:
            add("typed_query_partition_separator", [qid], pref=[m], aliases=pairs, **common)
        if out["distinct_values"] >= 2 and len({x["count"] for x in out["counts"]}) >= 2:
            add("skewed_possible_value_partition", [qid], pref=[m], **common)
    offered = {q["fact"] for q in task["queries"]}
    varying_unoffered = [f for f in next(iter(worlds.values()))["facts"] if f not in offered and len(query_partition(task, final, f)) > 1]
    if len(final) >= 2 and distinct_payloads >= 2 and all(q["distinct_values"] == 1 for q in expected["queries"]) and varying_unoffered:
        add("multiworld_all_queries_redundant", [q["query_id"] for q in task["queries"]], pref=[m],
            surviving_worlds=final, varying_unoffered={f: query_partition(task, final, f) for f in varying_unoffered})
    for a, b in itertools.combinations(range(m), 2):
        oa, ob = task["observations"][a], task["observations"][b]
        if oa["fact"] == ob["fact"]:
            continue
        ia = sorted(w["world_id"] for w in task["worlds"] if scalar(w["facts"][oa["fact"]]) == scalar(oa["equals"]))
        ib = sorted(w["world_id"] for w in task["worlds"] if scalar(w["facts"][ob["fact"]]) == scalar(ob["equals"]))
        if ia and ib and not set(ia).intersection(ib) and not final:
            add("cross_fact_inconsistent_conjunction", observations=[a, b], pref=[m], individual_supports=[ia, ib])
    empty_prefix = next((j for j, p in enumerate(prefixes) if not p), None)
    if empty_prefix is not None:
        for j in range(empty_prefix, m):
            o = task["observations"][j]
            standalone = sorted(w["world_id"] for w in task["worlds"] if scalar(w["facts"][o["fact"]]) == scalar(o["equals"]))
            if standalone and all(not p for p in prefixes[empty_prefix:]) and all(not q["counts"] for q in expected["queries"]):
                add("post_empty_absorption", observations=[empty_prefix - 1, j], pref=list(range(empty_prefix, m + 1)),
                    first_empty_prefix=empty_prefix, later_individual_support=standalone)
    for j, obs in enumerate(task["observations"]):
        for i in range(j):
            if equal(obs, task["observations"][i]) and prefixes[j] and equal(prefixes[j], prefixes[j + 1]):
                add("nonempty_repeated_observation", observations=[i, j], pref=[j, j + 1], unchanged_survivors=prefixes[j])
        for alias in aliases(obs["equals"]):
            exact = sorted(i for i in prefixes[j] if scalar(worlds[i]["facts"][obs["fact"]]) == scalar(obs["equals"]))
            aliased = sorted(i for i in prefixes[j] if scalar(worlds[i]["facts"][obs["fact"]]) == scalar(alias))
            if exact and aliased and equal(prefixes[j + 1], exact):
                add("typed_observation_separator", observations=[j], pref=[j, j + 1],
                    fact=obs["fact"], exact_value=obs["equals"], alias=alias, exact_survivors=exact, excluded_alias_survivors=aliased)
    heuristic = {"useful": [], "entailed": []}
    for q, out in zip(task["queries"], expected["queries"]):
        useful = len(final) > 1 and q["fact"] not in observed
        entailed = bool(final) and (len(final) == 1 or q["fact"] in observed)
        if useful != out["can_reduce_uncertainty"]:
            heuristic["useful"].append(q["query_id"])
        if entailed != out["entailed"]:
            heuristic["entailed"].append(q["query_id"])
    metrics = {"worlds": len(worlds), "facts": len(next(iter(worlds.values()))["facts"]), "queries": len(task["queries"]),
               "observations": m, "strict_nonempty_reduction_indices": strict,
               "final_support": len(final), "distinct_surviving_payloads": distinct_payloads,
               "duplicate_visible_payloads": len(worlds) - len({canon(w["facts"]) for w in task["worlds"]}),
               "first_empty_prefix": empty_prefix, "steps_after_first_empty": 0 if empty_prefix is None else m - empty_prefix,
               "conjunction_essential_constraint_counts": [2 for w in witnesses["conjunction_required_invariant"]],
               "active_typed_observation_separators": len(witnesses["typed_observation_separator"]),
               "query_partition_bin_counts": [q["distinct_values"] for q in expected["queries"]],
               "query_partition_count_vectors": [[c["count"] for c in q["counts"]] for q in expected["queries"]]}
    return {"expected": expected, "prefix_survivors": prefixes, "properties": witnesses,
            "metrics": metrics, "incomplete_heuristic_error_query_ids": heuristic}


def refuse_targets(value, path="$"):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN:
                raise ValueError("target-free stage encountered prohibited field " + path + "." + k)
            refuse_targets(v, path + "." + k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            refuse_targets(v, path + "[" + str(i) + "]")


def locate(row, key):
    """Metadata search, never prompt-derived targets; conflicting definitions fail."""
    values = []
    def walk(v):
        if isinstance(v, dict):
            if key in v:
                values.append(v[key])
            for k, item in v.items():
                if k not in {"rendered_input", "input", "task", "contract", "property_witnesses", "family_control_claims"}:
                    walk(item)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    walk(item)
    walk(row)
    if not values:
        raise ValueError("missing declared metadata " + key)
    if any(not equal(values[0], v) for v in values[1:]):
        raise ValueError("conflicting metadata " + key)
    return values[0]


def parse_prompt(row):
    prompt = row.get("rendered_input")
    if not isinstance(prompt, str):
        raise ValueError("rendered_input must be the complete student-visible string")
    marker = "Task JSON:\n"
    if prompt.count(marker) != 1:
        raise ValueError("one Task JSON marker required")
    lead, body = prompt.split(marker)
    semantics = (ROOT / "schemas/EXTENDED_SEMANTICS.md").read_text()
    belief_contract = semantics.split("## `kind=belief`\n", 1)[1].split("\n## ", 1)[0].strip()
    if belief_contract not in lead:
        raise ValueError("student prompt lacks the complete unchanged public belief contract")
    task = json.loads(body)
    validate_task(task)
    return task


def locator_errors(task, facts, prop, locator):
    errors = []
    required = {"query_ids", "observation_indices", "prefix_indices"}
    optional = {"supporting_world_ids", "note"}
    if not isinstance(locator, dict) or not required <= set(locator) or set(locator) - required - optional:
        return ["wrong locator shape"]
    m = len(task["observations"])
    qids = {q["query_id"] for q in task["queries"]}
    for key, allowed in [("query_ids", qids), ("observation_indices", set(range(m))), ("prefix_indices", set(range(m + 1)))]:
        values = locator[key]
        if not isinstance(values, list) or len({canon(v) for v in values}) != len(values):
            errors.append("duplicate/nonlist " + key)
        elif key != "query_ids" and any(type(v) is not int for v in values):
            errors.append("noninteger " + key)
        elif not set(values) <= allowed:
            errors.append("out-of-domain " + key)
    if "note" in locator and not isinstance(locator["note"], str):
        errors.append("nonstring note")
    if errors:
        return errors
    matches = [w for w in facts["properties"][prop] if all(set(w[k]) <= set(locator[k]) for k in required)]
    if not matches:
        errors.append("locator does not include the actual required query/observation/prefix witness")
    if "supporting_world_ids" in locator:
        supplied = locator["supporting_world_ids"]
        allids = {w["world_id"] for w in task["worlds"]}
        if not isinstance(supplied, list) or supplied != sorted(set(supplied)) or not set(supplied) <= allids:
            errors.append("invalid supporting_world_ids")
        else:
            relevant = set().union(*(set(facts["prefix_survivors"][i]) for i in locator["prefix_indices"]))
            def collect_support(v, field=""):
                if isinstance(v, dict):
                    for k, item in v.items():
                        collect_support(item, k)
                elif isinstance(v, list):
                    if v and all(isinstance(i, str) and i in allids for i in v) and any(s in field for s in ("support", "survivor")):
                        relevant.update(v)
                    else:
                        for item in v:
                            collect_support(item, field)
            for w in matches:
                collect_support(w)
            if not set(supplied) <= relevant:
                errors.append("supporting worlds outside reconstructed witness supports")
    return errors


def metadata_audit(row, task, facts):
    errors, out = [], {}
    keys = ["controlled_context", "structural_family_profile", "source_family_index", "source_family_key", "id_shift", "position_shift", "fact_name_shift", "schedule_index", "renderer", "query_id_pool", "queried_fact_name_pool", "fact_role_assignments", "query_role_assignments", "skill_tags", "property_witnesses", "family_control_claims"]
    for key in keys:
        try:
            out[key] = locate(row, key)
        except ValueError as e:
            errors.append(str(e))
    if len(out) != len(keys):
        return out, errors
    if out["controlled_context"] not in list("ABCDEF") or out["renderer"] not in ("forward", "reverse"):
        errors.append("invalid context/renderer")
    idx = out["source_family_index"]
    if type(idx) is not int or idx not in PROFILE or out["structural_family_profile"] != PROFILE.get(idx):
        errors.append("invalid family/profile association")
    shifts = [out[k] for k in ("id_shift", "position_shift", "fact_name_shift")]
    if any(type(v) is not int or v not in range(3) for v in shifts) or type(out["schedule_index"]) is not int:
        errors.append("invalid schedule shifts/index")
    elif out["schedule_index"] != 9 * shifts[0] + 3 * shifts[1] + shifts[2]:
        errors.append("wrong schedule_index")
    for key in ("query_id_pool", "queried_fact_name_pool"):
        pool = out[key]
        if not isinstance(pool, list) or len(pool) != 3 or len(set(pool)) != 3 or any(not isinstance(p, str) for p in pool):
            errors.append("invalid pool " + key)
    assignments = out["fact_role_assignments"]
    visible_keys = set(task["worlds"][0]["facts"])
    if not isinstance(assignments, dict) or not CORE_ROLES <= set(assignments):
        errors.append("missing core fact roles")
    elif any(not re.fullmatch(r"aux_extra_[0-9]+", k) for k in set(assignments) - CORE_ROLES) or len(set(assignments.values())) != len(assignments) or set(assignments.values()) != visible_keys:
        errors.append("fact roles not a bijection onto visible fact keys")
    roles = out["query_role_assignments"]
    if not isinstance(roles, list) or len(roles) != 3 or any(not isinstance(r, dict) for r in roles):
        errors.append("invalid query_role_assignments")
    elif {r.get("role") for r in roles} != set(QUERY_ROLES):
        errors.append("incorrect stable queried roles")
    elif not errors:
        u, v, w = shifts
        role_map = {"observed_control": "observation_control", "invariant_query": "invariant_query", "variation_query": "variation_query"}
        for r in roles:
            if not {"role", "position", "query_id"} <= set(r) or set(r) - {"role", "position", "query_id", "fact"}:
                errors.append("invalid query-role object keys")
                continue
            ri = QUERY_ROLES.index(r["role"])
            p, qid, fact = (ri + v) % 3, out["query_id_pool"][(ri + u) % 3], out["queried_fact_name_pool"][(ri + w) % 3]
            if type(r["position"]) is not int or r["position"] != p or r["query_id"] != qid or ("fact" in r and r["fact"] != fact):
                errors.append("declared schedule role mismatch " + r["role"])
            if task["queries"][p] != {"query_id": qid, "fact": fact} or assignments[role_map[r["role"]]] != fact:
                errors.append("visible query/fact renaming mismatch " + r["role"])
        if any(assignments[r] in out["queried_fact_name_pool"] for r in set(assignments) - {"invariant_query", "variation_query", "observation_control"}):
            errors.append("auxiliary roles overlap queried-name pool")
    tags = out["skill_tags"]
    if not isinstance(tags, list) or len(set(tags)) != len(tags):
        errors.append("duplicate/nonlist tags")
    else:
        if not SCOPE_TAGS <= set(tags) or set(tags) - SCOPE_TAGS - PROPERTIES:
            errors.append("missing domain tags or unknown/per-family skill tag")
        attached = set(tags) & PROPERTIES
        locators = out["property_witnesses"]
        if not isinstance(locators, dict) or set(locators) != attached:
            errors.append("property_witnesses keys differ from property tags")
        else:
            for prop in sorted(attached):
                if not facts["properties"][prop]:
                    errors.append("unwitnessed property " + prop)
                errors.extend(prop + ": " + e for e in locator_errors(task, facts, prop, locators[prop]))
    claims = out["family_control_claims"]
    allowed = {"name", "property", "scope", "source_family_key", "evidence_manifest", "control_id", "pair_sample_ids", "probe_id"}
    seen = {}
    if not isinstance(claims, list):
        errors.append("family_control_claims not list")
    else:
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) - allowed or ("name" in claim) == ("property" in claim):
                errors.append("invalid family claim shape")
                continue
            name = claim.get("name", claim.get("property"))
            scope = claim.get("scope")
            if name not in FAMILY_CONTROLS or scope not in ("corpus_variant", "audit_probe"):
                errors.append("unknown family identifier/scope")
            if name in seen and seen[name] != scope:
                errors.append("inconsistent duplicate family scope")
            seen[name] = scope
            if "source_family_key" in claim and claim["source_family_key"] != out["source_family_key"]:
                errors.append("mismatched family claim key")
            for key in ("evidence_manifest", "control_id", "probe_id"):
                if key in claim and (not isinstance(claim[key], str) or not claim[key]):
                    errors.append("invalid claim " + key)
            if "evidence_manifest" in claim and pathlib.Path(claim["evidence_manifest"]).is_absolute():
                errors.append("absolute evidence_manifest")
            if "pair_sample_ids" in claim and (scope != "corpus_variant" or not isinstance(claim["pair_sample_ids"], list) or len(claim["pair_sample_ids"]) != 2 or len(set(claim["pair_sample_ids"])) != 2):
                errors.append("invalid pair_sample_ids/scope")
            if "probe_id" in claim and scope != "audit_probe":
                errors.append("probe_id with corpus scope")
    return out, errors


def incidence(task, roles):
    f, g = roles["observation_control"], roles["observation_auxiliary"]
    edges = collections.Counter((scalar(w["facts"][f]), scalar(w["facts"][g])) for w in task["worlds"])
    neighbors = collections.defaultdict(set)
    for a, b in edges:
        neighbors[(0, a)].add((1, b))
        neighbors[(1, b)].add((0, a))
    components, unseen = [], set(neighbors)
    while unseen:
        seed, seen, todo = min(unseen), set(), [min(unseen)]
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo.extend(neighbors[x] - seen)
        unseen -= seen
        components.append(sorted([list(x) for x in seen]))
    degrees = sorted(len(v) for v in neighbors.values())
    cycle_rank = len(edges) - len(neighbors) + len(components)
    result = {"observation_facts": [f, g], "weighted_edges": [{"control": json.loads(a), "auxiliary": json.loads(b), "world_count": n} for (a, b), n in sorted(edges.items())],
              "vertices": len(neighbors), "occupied_cells": len(edges), "components": components,
              "degrees": degrees, "cycle_rank": cycle_rank, "uniform_occupied_weights": len(set(edges.values())) == 1,
              "connected": len(components) == 1, "max_degree": max(degrees, default=0),
              "unsupported_joint_cells": len({a for a, b in edges}) * len({b for a, b in edges}) - len(edges)}
    return result


def profile_errors(profile, graph, facts):
    errors = []
    if profile == "cyclic_uniform" and (graph["cycle_rank"] < 1 or not graph["uniform_occupied_weights"]):
        errors.append("cyclic_uniform requires a cycle and uniform positive cell weights")
    if profile == "acyclic_branching" and (not graph["connected"] or graph["cycle_rank"] != 0 or graph["max_degree"] < 3):
        errors.append("acyclic_branching requires connected tree and degree>=3")
    if profile == "cyclic_weighted" and (graph["cycle_rank"] < 1 or graph["uniform_occupied_weights"]):
        errors.append("cyclic_weighted requires a cycle and unequal occupied weights")
    if profile == "typed_path" and (not graph["connected"] or graph["cycle_rank"] != 0 or graph["max_degree"] != 2):
        errors.append("typed_path requires connected path with max degree2")
    return errors


def structural_graph(task):
    """Typed value names intentionally erased; fact/value incidence retained.

    Exact isomorphism compares worlds, facts and within-fact equality blocks,
    permitting arbitrary world/fact names and scalar substitutions.
    """
    colors, adj = {}, collections.defaultdict(set)
    def edge(a, b):
        adj[a].add(b)
        adj[b].add(a)
    for i, world in enumerate(task["worlds"]):
        node = ("world", i)
        colors[node] = "world"
        for fact, value in world["facts"].items():
            fn, vn = ("fact", fact), ("value", fact, scalar(value))
            colors[fn], colors[vn] = "fact", "value"
            edge(fn, vn)
            edge(node, vn)
    return colors, dict(adj)


def wl_signature(task):
    colors, adj = structural_graph(task)
    labels = colors.copy()
    for _ in range(len(colors)):
        new = {n: sha(canon([labels[n], sorted(labels[i] for i in adj[n])]).encode()) for n in colors}
        old_groups = {frozenset(n for n in colors if labels[n] == label) for label in set(labels.values())}
        new_groups = {frozenset(n for n in colors if new[n] == label) for label in set(new.values())}
        labels = new
        if old_groups == new_groups:
            break
    return sha(canon(sorted(labels.values())).encode())


def structurally_isomorphic(a, b, limit=1000000):
    ca, aa = structural_graph(a)
    cb, ab = structural_graph(b)
    if collections.Counter((ca[n], len(aa[n])) for n in ca) != collections.Counter((cb[n], len(ab[n])) for n in cb):
        return {"isomorphic": False, "search_nodes": 0}
    choices = {n: [j for j in cb if cb[j] == ca[n] and len(ab[j]) == len(aa[n])] for n in ca}
    assigned, used, visits = {}, set(), [0]
    def search():
        visits[0] += 1
        if visits[0] > limit:
            raise RuntimeError("isomorphism search limit; do not claim distinctness")
        if len(assigned) == len(ca):
            return True
        candidates = []
        for n in ca:
            if n not in assigned:
                possible = [j for j in choices[n] if j not in used and all((k in aa[n]) == (assigned[k] in ab[j]) for k in assigned)]
                candidates.append((len(possible), -sum(k in aa[n] for k in assigned), repr(n), n, possible))
        _, _, _, n, possible = min(candidates)
        for j in possible:
            assigned[n] = j
            used.add(j)
            if search():
                return True
            used.remove(j)
            del assigned[n]
        return False
    try:
        return {"isomorphic": search(), "search_nodes": visits[0]}
    except RuntimeError as e:
        return {"isomorphic": None, "search_nodes": visits[0], "error": str(e)}


def metamorphics(task):
    base, prefixes = solve(task)
    results = []
    def check(name, altered, expected):
        actual, altered_prefixes = solve(altered)
        oracle2, p2 = solve_conjunction(altered)
        assert equal(actual, expected) and equal(actual, oracle2) and equal(altered_prefixes, p2), name
        results.append({"name": name, "pass": True, "task_hash": sha(canon(altered).encode())})
    altered = copy.deepcopy(task)
    altered["worlds"].reverse()
    check("world_order_invariance", altered, base)
    altered = copy.deepcopy(task)
    mapping = {w["world_id"]: "renamed_" + str(len(task["worlds"]) - i) for i, w in enumerate(task["worlds"])}
    for w in altered["worlds"]:
        w["world_id"] = mapping[w["world_id"]]
    remapped = copy.deepcopy(base)
    remapped["surviving_worlds"] = sorted(mapping[i] for i in base["surviving_worlds"])
    check("world_id_rename_equivariance", altered, remapped)
    altered = copy.deepcopy(task)
    altered["queries"].reverse()
    wanted = copy.deepcopy(base)
    wanted["queries"].reverse()
    check("query_order_equivariance", altered, wanted)
    altered = copy.deepcopy(task)
    wanted = copy.deepcopy(base)
    for i, (query, answer) in enumerate(zip(altered["queries"], wanted["queries"])):
        query["query_id"] = answer["query_id"] = "renamed_query_" + str(i)
    check("query_id_rename_equivariance", altered, wanted)
    factmap = {f: "renamed_fact_" + str(i) for i, f in enumerate(sorted(task["worlds"][0]["facts"]))}
    altered = copy.deepcopy(task)
    for world in altered["worlds"]:
        world["facts"] = {factmap[f]: v for f, v in world["facts"].items()}
    for entry in altered["observations"] + altered["queries"]:
        entry["fact"] = factmap[entry["fact"]]
    check("consistent_fact_name_invariance", altered, base)
    altered = copy.deepcopy(task)
    altered["observations"].reverse()
    final, _ = solve_conjunction(altered)
    assert equal(final["surviving_worlds"], base["surviving_worlds"]) and equal(final["queries"], base["queries"])
    check("observation_order_final_invariance_recomputed_trace", altered, final)
    if task["observations"]:
        altered = copy.deepcopy(task)
        altered["observations"].append(copy.deepcopy(task["observations"][0]))
        wanted = copy.deepcopy(base)
        wanted["belief_sizes"].append(len(base["surviving_worlds"]))
        check("repeat_observation_final_invariance", altered, wanted)
    typed_probes = []
    for j, obs in enumerate(task["observations"]):
        for alias in aliases(obs["equals"]):
            altered = copy.deepcopy(task)
            altered["observations"][j]["equals"] = alias
            actual, pref = solve(altered)
            alt, pref2 = solve_conjunction(altered)
            assert equal(actual, alt) and equal(pref, pref2)
            qualifying = bool(prefixes[j + 1]) and bool(pref[j + 1]) and not equal(prefixes[j + 1], pref[j + 1])
            typed_probes.append({"observation_index": j, "old": obs["equals"], "new": alias,
                                 "qualifying_changed_prefix": qualifying,
                                 "old_prefix_survivors": prefixes[j + 1], "new_prefix_survivors": pref[j + 1],
                                 "old_final_support": base["surviving_worlds"], "new_final_support": actual["surviving_worlds"],
                                 "new_expected": actual})
    return {"invariance_probes": results, "typed_equals_counterfactual_probes": typed_probes}


def decoded_task(task, metadata):
    reverse = {v: k for k, v in metadata["fact_role_assignments"].items()}
    out = copy.deepcopy(task)
    for w in out["worlds"]:
        w["facts"] = {reverse[f]: v for f, v in w["facts"].items()}
    for obs in out["observations"]:
        obs["fact"] = reverse[obs["fact"]]
    out["queries"] = [{"query_id": r, "fact": "observation_control" if r == "observed_control" else r} for r in QUERY_ROLES]
    return out


def same_final(a, b):
    return equal(a["surviving_worlds"], b["surviving_worlds"]) and equal(a["queries"], b["queries"])


def single_equals_change(a, b):
    if len(a) != len(b):
        return None
    changed = [j for j in range(len(a)) if not equal(a[j], b[j])]
    if len(changed) == 1 and a[changed[0]]["fact"] == b[changed[0]]["fact"]:
        return changed[0]
    return None


def family_audits(records):
    """Only independently computed expectations/facts enter these audits."""
    errors, pairs, histograms, families = [], [], {}, {}
    groups = collections.defaultdict(list)
    for r in records:
        md = r["metadata"]
        if r["metadata_errors"]:
            continue
        groups[(md["source_family_key"], md["controlled_context"], md["renderer"])].append(r)
        families.setdefault(md["source_family_key"], []).append(r)
    for group, rows in sorted(groups.items()):
        if len(rows) != 27 or sorted(r["metadata"]["schedule_index"] for r in rows) != list(range(27)):
            errors.append({"group": group, "error": "27 schedules not present exactly once"})
        pools = {(canon(r["metadata"]["query_id_pool"]), canon(r["metadata"]["queried_fact_name_pool"])) for r in rows}
        decoded = {canon(decoded_task(r["task"], r["metadata"])) for r in rows}
        if len(pools) != 1 or len(decoded) != 1:
            errors.append({"group": group, "error": "unstable pools or inconsistent full renamed task"})
        tables = {label: collections.defaultdict(collections.Counter) for label in ("id", "position", "name", "id_position", "id_name", "position_name", "triple")}
        role_triples = {role: collections.Counter() for role in QUERY_ROLES}
        for row in rows:
            md, task, f = row["metadata"], row["task"], row["facts"]
            inverse = {v: k for k, v in md["fact_role_assignments"].items()}
            for pos, (query, payload) in enumerate(zip(task["queries"], f["expected"]["queries"])):
                role = inverse[query["fact"]]
                if role == "observation_control":
                    role = "observed_control"
                i = md["query_id_pool"].index(query["query_id"])
                n = md["queried_fact_name_pool"].index(query["fact"])
                role_triples[role][(i, pos, n)] += 1
                full = {k: v for k, v in payload.items() if k != "query_id"}
                category = "empty" if not payload["counts"] else "entailed" if payload["entailed"] else "ambiguous"
                signature = canon({"role": role, "full_typed_payload": full, "category": category,
                                   "guaranteed_null_context": category if payload["guaranteed"] is None else "non_null",
                                   "type_vector": [type(c["value"]).__name__ for c in payload["counts"]],
                                   "surviving_worlds": f["expected"]["surviving_worlds"], "prefix_survivors": f["prefix_survivors"]})
                cells = {"id": i, "position": pos, "name": n, "id_position": (i, pos), "id_name": (i, n), "position_name": (pos, n), "triple": (i, pos, n)}
                for label, cell in cells.items():
                    tables[label][canon(cell)][signature] += 1
        for role, counts in role_triples.items():
            if len(counts) != 27 or set(counts.values()) != {1}:
                errors.append({"group": group, "role": role, "error": "role lacks full uniform nuisance triple support"})
        result = {}
        for label, table in tables.items():
            expected_cells = 3 if label in ("id", "position", "name") else 27 if label == "triple" else 9
            payloads = [canon(dict(counter)) for counter in table.values()]
            identical = len(table) == expected_cells and len(set(payloads)) == 1
            if not identical:
                errors.append({"group": group, "axis": label, "error": "actual typed outcome/provenance histograms not equal"})
            result[label] = {"cells": {cell: dict(counter) for cell, counter in sorted(table.items())}, "identical": identical}
        histograms[canon(group)] = {"role_triple_counts": {role: {canon(k): v for k, v in sorted(c.items())} for role, c in role_triples.items()}, "outcome_histograms": result}
    profiles, structural_pairs, control_witnesses = {}, [], {}
    bysample = {r["sample_id"]: r for r in records}
    for family, rows in sorted(families.items()):
        lookup = {(r["metadata"]["controlled_context"], r["metadata"]["schedule_index"], r["metadata"]["renderer"]): r for r in rows}
        if len(rows) != 324 or len(lookup) != 324:
            errors.append({"family": family, "error": "family must have6 contexts*27 schedules*2 fields"})
        controls = collections.defaultdict(list)
        indices = {r["metadata"]["source_family_index"] for r in rows}
        if len(indices) != 1:
            errors.append({"family": family, "error": "family index conflict"})
        for schedule in range(27):
            for renderer in ("forward", "reverse"):
                block = {c: lookup.get((c, schedule, renderer)) for c in "ABCDEF"}
                if any(v is None for v in block.values()):
                    continue
                A, B, C, D, E, F = [block[c] for c in "ABCDEF"]
                tasks = [r["task"] for r in (A, B, C, D, E, F)]
                if any(not equal(t["worlds"], tasks[0]["worlds"]) or not equal(t["queries"], tasks[0]["queries"]) for t in tasks[1:]):
                    errors.append({"family": family, "schedule": schedule, "renderer": renderer, "error": "contexts changed worlds or ordered queries"})
                for name, X, Y in [("equal_support_usefulness_counterfactual", A, B), ("observation_reorder_final_invariance", A, C), ("nonempty_repetition_final_invariance", A, D), ("cross_fact_contradiction_counterfactual", A, E), ("post_empty_append_absorption", E, F)]:
                    x, y = X["task"]["observations"], Y["task"]["observations"]
                    xo, yo = X["facts"]["expected"], Y["facts"]["expected"]
                    change = single_equals_change(x, y)
                    if name == "equal_support_usefulness_counterfactual":
                        okay = change is not None and len(xo["surviving_worlds"]) == len(yo["surviving_worlds"]) >= 2 and any(a["can_reduce_uncertainty"] != b["can_reduce_uncertainty"] for a, b in zip(xo["queries"], yo["queries"])) and bool(Y["facts"]["properties"]["multiworld_all_queries_redundant"])
                    elif name == "observation_reorder_final_invariance":
                        okay = sorted(map(canon, x)) == sorted(map(canon, y)) and same_final(xo, yo) and not equal(xo["belief_sizes"], yo["belief_sizes"])
                    elif name == "nonempty_repetition_final_invariance":
                        okay = len(y) == len(x) + 1 and equal(y[:-1], x) and any(equal(y[-1], o) for o in x) and bool(xo["surviving_worlds"]) and same_final(xo, yo) and yo["belief_sizes"][-1] == len(xo["surviving_worlds"])
                    elif name == "cross_fact_contradiction_counterfactual":
                        okay = change is not None and bool(Y["facts"]["properties"]["cross_fact_inconsistent_conjunction"])
                    else:
                        okay = len(y) == len(x) + 1 and equal(y[:-1], x) and not xo["surviving_worlds"] and bool(Y["facts"]["properties"]["post_empty_absorption"]) and same_final(xo, yo) and yo["belief_sizes"][-1] == 0
                    detail = {"name": name, "pair_sample_ids": [X["sample_id"], Y["sample_id"]], "pass": okay,
                              "changed_observation": change, "old_expected": xo, "new_expected": yo}
                    pairs.append(detail)
                    if okay:
                        controls[name].append(detail)
                    else:
                        errors.append({"family": family, "pair": detail["pair_sample_ids"], "error": name + " relationship failed"})
                ap = A["facts"]["properties"]
                if len(tasks[0]["observations"]) != 2 or not ap["conjunction_required_invariant"] or not ap["useful_after_conjunctive_evidence"]:
                    errors.append({"sample_id": A["sample_id"], "error": "A lacks actual necessary conjunction invariant/useful control"})
                if not A["facts"]["incomplete_heuristic_error_query_ids"]["useful"] or not A["facts"]["incomplete_heuristic_error_query_ids"]["entailed"]:
                    errors.append({"sample_id": A["sample_id"], "error": "A does not break both specified incomplete heuristics"})
            for context in "ABCDEF":
                a, b = lookup.get((context, schedule, "forward")), lookup.get((context, schedule, "reverse"))
                if a is not None and b is not None:
                    okay = equal(a["task"], b["task"]) and equal(a["facts"]["expected"], b["facts"]["expected"])
                    if okay:
                        controls["field_order_invariance"].append({"pair_sample_ids": [a["sample_id"], b["sample_id"]]})
                    else:
                        errors.append({"family": family, "context": context, "schedule": schedule, "error": "field-order pair task/expectation mismatch"})
        representatives = {c: lookup.get((c, 0, "forward")) for c in "ABCDEF"}
        if all(representatives.values()):
            A = representatives["A"]
            graph = incidence(A["task"], A["metadata"]["fact_role_assignments"])
            perrors = profile_errors(A["metadata"]["structural_family_profile"], graph, A["facts"])
            errors.extend({"family": family, "error": e} for e in perrors)
            profiles[family] = {"profile": A["metadata"]["structural_family_profile"], "incidence": graph,
                                "name_scalar_invariant_wl_fingerprint": wl_signature(A["task"]),
                                "wl_is_only_a_prefilter": True, "representative": A["sample_id"]}
            null_types = {"entailed_null": [], "ambiguous_null": [], "empty": []}
            for r in rows:
                if r["facts"]["properties"]["multiworld_entailed_null"]:
                    null_types["entailed_null"].append(r["sample_id"])
                if r["facts"]["properties"]["ambiguous_null_partition"]:
                    null_types["ambiguous_null"].append(r["sample_id"])
                if not r["facts"]["expected"]["surviving_worlds"]:
                    null_types["empty"].append(r["sample_id"])
            if all(null_types.values()):
                controls["null_guarantee_contrast"].append({k: v[0] for k, v in null_types.items()})
        if all(histograms.get(canon((family, c, ren))) and not any(e.get("group") == (family, c, ren) for e in errors) for c in "ABCDEF" for ren in ("forward", "reverse")):
            controls["query_id_position_fact_crossing"].append({"verified_contexts": list("ABCDEF"), "renderers": ["forward", "reverse"]})
        for r in rows:
            for probe in r["probes"]["invariance_probes"]:
                if probe["name"] in FAMILY_CONTROLS:
                    controls[probe["name"]].append({"sample_id": r["sample_id"], "scope": "audit_probe", **probe})
            for probe in r["probes"]["typed_equals_counterfactual_probes"]:
                if probe["qualifying_changed_prefix"]:
                    controls["typed_equals_counterfactual"].append({"sample_id": r["sample_id"], "scope": "audit_probe", **probe})
        for r in rows:
            for claim in r["metadata"]["family_control_claims"]:
                if not isinstance(claim, dict):
                    continue
                name = claim.get("name", claim.get("property"))
                matching = [w for w in controls[name] if (w.get("scope", "corpus_variant") == claim.get("scope"))]
                if "pair_sample_ids" in claim:
                    pair = claim["pair_sample_ids"]
                    if any(i not in bysample or bysample[i]["metadata"].get("source_family_key") != family for i in pair):
                        matching = []
                    else:
                        matching = [w for w in matching if set(w.get("pair_sample_ids", [])) == set(pair)]
                if not matching:
                    errors.append({"sample_id": r["sample_id"], "error": "family claim has no reconstructed witness at declared scope: " + str(name)})
        control_witnesses[family] = dict(controls)
    reps = [(family, next(r for r in rows if r["metadata"]["controlled_context"] == "A")) for family, rows in sorted(families.items())]
    for (fa, a), (fb, b) in itertools.combinations(reps, 2):
        comparison = structurally_isomorphic(a["task"], b["task"])
        structural_pairs.append({"families": [fa, fb], **comparison})
        if comparison["isomorphic"] is not False:
            errors.append({"families": [fa, fb], "error": "structural distinction not established by exact incidence isomorphism"})
    return {"errors": errors, "context_control_pairs": pairs, "histograms": histograms,
            "structural_profiles": profiles, "exact_table_structure_comparisons": structural_pairs,
            "family_control_witnesses": control_witnesses,
            "remaining_limits": ["27 schedules are correlated views, not new mechanisms", "cyclic row-role order remains", "two auxiliary roles and world-ID styles are outside the specified crossing", "finite noiseless partitions provide no Bayes/utility/action/efficacy evidence"]}


def toy_fixtures():
    def world(i, a, b, q, t, z):
        return {"world_id": i, "facts": {"a": a, "b": b, "q": q, "t": t, "z": z}}
    # Toy-only fixtures, not future production worlds/targets or design answers.
    worlds = [world("z", 0, 0, None, False, "x"), world("a", 0, 0, None, 0, "y"),
              world("m", 0, 1, "unknown", True, "x"), world("n", 1, 0, "other", 1, "x"),
              world("p", 1, 1, "unknown", None, "y")]
    queries = [{"query_id": "qz", "fact": "q"}, {"query_id": "qa", "fact": "t"}, {"query_id": "qm", "fact": "a"}]
    base = {"kind": "belief", "worlds": worlds, "observations": [{"fact": "a", "equals": 0}, {"fact": "b", "equals": 0}], "queries": queries}
    empty = copy.deepcopy(base)
    empty["observations"] = [{"fact": "a", "equals": 0}, {"fact": "q", "equals": "other"}, {"fact": "b", "equals": 1}]
    typed = copy.deepcopy(base)
    typed["observations"] = [{"fact": "t", "equals": False}]
    repeat = copy.deepcopy(base)
    repeat["observations"].append(copy.deepcopy(base["observations"][0]))
    unfiltered = copy.deepcopy(base)
    unfiltered["observations"] = []
    return [base, empty, typed, repeat, unfiltered]


def preparation():
    bindings = source_bindings()
    reports, total = [], 0
    for index, fixture in enumerate(toy_fixtures()):
        facts = row_facts(fixture)
        probes = metamorphics(fixture)
        total += len(probes["invariance_probes"]) + len(probes["typed_equals_counterfactual_probes"])
        reports.append({"fixture_index": index, "fixture": fixture, "facts": facts, "probes": probes})
    assert reports[0]["facts"]["expected"]["surviving_worlds"] == ["a", "z"]
    assert reports[0]["facts"]["expected"]["belief_sizes"] == [3, 2]
    assert reports[0]["facts"]["expected"]["queries"][0] == {"query_id": "qz", "counts": [{"value": None, "count": 2}], "distinct_values": 1, "entailed": True, "guaranteed": None, "can_reduce_uncertainty": False}
    assert reports[0]["facts"]["expected"]["queries"][1]["counts"] == [{"value": 0, "count": 1}, {"value": False, "count": 1}]
    assert reports[0]["facts"]["properties"]["conjunction_required_invariant"]
    assert reports[1]["facts"]["expected"]["belief_sizes"] == [3, 0, 0]
    assert reports[1]["facts"]["properties"]["cross_fact_inconsistent_conjunction"] and reports[1]["facts"]["properties"]["post_empty_absorption"]
    assert reports[2]["facts"]["expected"]["surviving_worlds"] == ["z"]
    assert reports[2]["facts"]["properties"]["typed_observation_separator"]
    assert reports[3]["facts"]["properties"]["nonempty_repeated_observation"]
    assert reports[4]["facts"]["properties"]["ambiguous_null_partition"]
    assert reports[4]["facts"]["properties"]["typed_query_partition_separator"]
    assert not equal(True, 1) and not equal(False, 0) and not equal({"x": True}, {"x": 1})
    malformed = copy.deepcopy(toy_fixtures()[0])
    malformed["worlds"][0]["facts"]["a"] = 0.0
    try:
        solve(malformed)
        raise AssertionError("float fixture incorrectly accepted")
    except ValueError:
        pass
    for forbidden in FORBIDDEN:
        try:
            refuse_targets({"some_metadata": {forbidden: None}})
            raise AssertionError("target field accepted")
        except ValueError:
            pass
    transformed = copy.deepcopy(toy_fixtures()[0])
    transformed["worlds"].reverse()
    assert structurally_isomorphic(toy_fixtures()[0], transformed)["isomorphic"] is True
    transformed["worlds"].append({"world_id": "extra", "facts": copy.deepcopy(transformed["worlds"][0]["facts"])})
    assert structurally_isomorphic(toy_fixtures()[0], transformed)["isomorphic"] is False
    source = pathlib.Path(__file__).read_bytes()
    report = {"report_type": "EXPERT_PREPARATION", "actual_author": ROLE, "reviewer_execution_id": ROLE,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "guidance": "Own prospective gap analysis and shared public design; not a fresh unprompted blind gap discovery.",
              "candidate_reads": 0, "canonical_gold_reads": 0, "producer_reference_oracle_reads": 0,
              "dataset_decisions": 0, "source_inputs": bindings,
              "new_source": {"path": str(pathlib.Path(__file__)), "sha256": sha(source), "length": len(source)},
              "python_executable": "/opt/anaconda3/bin/python3", "toy_fixture_count": len(reports),
              "actual_metamorphic_probe_count": total, "fixtures": reports,
              "planned_candidate_counts_not_achieved": {"source_families": 4, "contexts_per_family": 6, "schedules_per_context": 27, "field_orders": 2, "records": 1296},
              "review_sequence": ["Root supplies frozen original candidate SHA and target-free input view", "Own expected outputs/facts/audits freeze with source/input hashes", "Only then canonical gold comparison", "Actual first per-row judgments freeze before other reviewer judgments", "Substantive examples and honest limitations finalize separately"],
              "preparation_limits": ["No candidate schema layout yet received; extraction may require documented input-view adaptation", "No achieved production property/coverage claimed", "Prior-table comparison requires authorized prior input evidence", "WL fingerprint is not a complete isomorphism proof; exact backtracking is separately available", "Family probe pointers require independently executed witness; producer report never read"]}
    path = ROOT / "reviews/expert/belief-gap-v1.preparation.json"
    assert not path.exists(), "preserve existing preparation"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(canon({"preparation_path": str(path), "sha256": sha(path.read_bytes()), "toy_fixtures": len(reports), "metamorphic_probes": total, "candidate_reads": 0}))


def predict(view, out, original_sha):
    source_bindings()
    assert re.fullmatch(r"[0-9a-f]{64}", original_sha), "root-frozen original file hash required"
    data = pathlib.Path(view).read_bytes()
    records = []
    for line_no, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        refuse_targets(row)
        allowed = {"sample_id", "record_hash", "semantic_parent_id", "semantic_family_id", "track_id", "split", "semantic_task", "rendered_input"}
        assert not set(row) - allowed, "input-only view contains non-whitelisted fields"
        task = parse_prompt(row)
        facts = row_facts(task)
        if isinstance(row.get("semantic_task"), dict):
            assert equal(row["semantic_task"], task), "visible semantic_task differs from prompt task"
        sample_id = row.get("sample_id")
        record_hash = row.get("record_hash")
        assert isinstance(sample_id, str) and re.fullmatch(r"[0-9a-f]{64}", record_hash), "original ID/hash required"
        records.append({"sample_id": sample_id, "record_hash": record_hash, "input_view_line": line_no,
                        "view_payload_hash": sha(line), "task": task, "task_hash": sha(canon(task).encode()),
                        "facts": facts, "probes": metamorphics(task),
                        "visible_identity": {k: row[k] for k in allowed - {"rendered_input", "semantic_task"} if k in row}})
    assert len({r["sample_id"] for r in records}) == len(records), "duplicate sample IDs"
    report = {"report_type": "EXPERT_INPUT_ONLY_PREDICTION_FREEZE", "actual_author": ROLE,
              "reviewer_execution_id": ROLE, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_inputs": source_bindings(), "source_sha256": sha(pathlib.Path(__file__).read_bytes()),
              "view": {"path": str(pathlib.Path(view).resolve()), "sha256": sha(data), "length": len(data)},
              "root_frozen_original_candidate_sha256": original_sha,
              "no_gold_seen": True, "no_dataset_verdict": True,
              "guidance": "Corrective prospective-gap/design guided initial review; independent expected outputs before gold comparison.",
              "records": records,
              "metadata_audits_deferred_until_post_freeze": True,
              "observed_counts": {"records": len(records), "unique_typed_tasks": len({r["task_hash"] for r in records}),
                                  "visible_semantic_families": len({r["visible_identity"].get("semantic_family_id") for r in records})}}
    path = pathlib.Path(out)
    assert not path.exists(), "freeze output already exists; never overwrite"
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    receipt = {"actual_author": ROLE, "report_type": "PREDICTION_FREEZE_RECEIPT", "created_utc": report["created_utc"],
               "prediction_path": str(path.resolve()), "prediction_sha256": sha(raw), "prediction_length": len(raw),
               "view_sha256": sha(data), "original_candidate_sha256": original_sha, "record_count": len(records),
               "source_sha256": report["source_sha256"], "source_inputs": report["source_inputs"], "no_gold_seen": True,
               "no_dataset_decision": True, "producer_metadata_seen": False, "family_audits_deferred": True}
    rp = pathlib.Path(str(path) + ".freeze_receipt.json")
    assert not rp.exists()
    rp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(canon(receipt))


def compare(frozen, canonical, out):
    """Explicit separate invocation after input-only expectations are frozen."""
    path = pathlib.Path(frozen)
    raw = path.read_bytes()
    receipt = json.loads(pathlib.Path(str(path) + ".freeze_receipt.json").read_text())
    assert sha(raw) == receipt["prediction_sha256"]
    expected = json.loads(raw)
    candidate_bytes = pathlib.Path(canonical).read_bytes()
    assert sha(candidate_bytes) == expected["root_frozen_original_candidate_sha256"], "original batch hash mismatch"
    rows = [json.loads(line) for line in candidate_bytes.splitlines() if line.strip()]
    lookup = {r["sample_id"]: r for r in rows}
    assert len(rows) == len(lookup) == len(expected["records"])
    comparisons = []
    audit_records = []
    for r in expected["records"]:
        actual = lookup[r["sample_id"]]
        assert actual["record_hash"] == r["record_hash"]
        assert equal(parse_prompt(actual), r["task"])
        candidate_copy = copy.deepcopy(actual)
        candidate_copy.pop("record_hash")
        bound = sha(canon(candidate_copy).encode())
        facts = r["facts"]
        metadata, metadata_errors = metadata_audit(actual, r["task"], facts)
        audit_records.append({**r, "metadata": metadata, "metadata_errors": metadata_errors})
        comparisons.append({"sample_id": r["sample_id"], "record_hash": r["record_hash"],
                            "type_sensitive_target_equal": equal(r["facts"]["expected"], actual["verified_target"]),
                            "own_expected": r["facts"]["expected"], "actual_verified_target": actual["verified_target"],
                            "canonical_payload_hash_without_record_hash": bound,
                            "original_record_hash_matches_without_hash_field": bound == r["record_hash"],
                            "metadata_errors": metadata_errors})
    audits = family_audits(audit_records)
    report = {"actual_author": ROLE, "report_type": "POST_FREEZE_TARGET_COMPARISON_NO_VERDICT",
              "prediction_sha256": sha(raw), "original_candidate_sha256": sha(candidate_bytes),
              "records": comparisons, "target_mismatch_count": sum(not r["type_sensitive_target_equal"] for r in comparisons),
              "metadata_error_rows": sum(bool(r["metadata_errors"]) for r in comparisons),
              "family_audits": audits, "post_freeze_audit_records": audit_records,
              "source_sha256": sha(pathlib.Path(__file__).read_bytes())}
    output = pathlib.Path(out)
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(canon({"path": str(output), "sha256": sha(output.read_bytes()), "record_count": len(comparisons), "target_mismatches": report["target_mismatch_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "predict", "compare"])
    parser.add_argument("--view")
    parser.add_argument("--original-sha")
    parser.add_argument("--frozen")
    parser.add_argument("--canonical")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "prepare":
        preparation()
    elif args.mode == "predict":
        predict(args.view, args.out, args.original_sha)
    else:
        compare(args.frozen, args.canonical, args.out)
