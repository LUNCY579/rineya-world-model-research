"""RBM-GAP-007 producer: finite tables, public locators, exact Z3^3 views.

Producer-only execution. No independent checker, test or reviewer is imported.
The first generated candidate and run manifest are retained even on failure.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import random
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record, record_hash, split_for, write_json, write_jsonl
from generators.other_tracks.reference import contracts, solve


SEED = 20261009
PROFILES = {12: "cyclic_uniform", 13: "acyclic_branching", 18: "cyclic_weighted", 19: "typed_path"}
FACT_ROLES = ("observation_control", "observation_auxiliary", "invariant_query", "variation_query", "unoffered_feature")
QUERY_ROLES = ("invariant_query", "variation_query", "observed_control")
PUBLIC_HASHES = {"schemas/BELIEF_CONTROL_PROPERTIES.md": "200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034",
                 "research/curricula/RBM-EVIDENCE-GAP.md": "004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120"}
WORDS = ("panel", "ribbon", "canvas", "sprig", "vessel", "rivet", "parcel", "copper", "beacon", "socket", "column", "basket")
PROPERTIES = ("multiworld_unobserved_invariant", "conjunction_required_invariant", "useful_after_conjunctive_evidence",
              "multiworld_all_queries_redundant", "multiworld_entailed_null", "ambiguous_null_partition",
              "cross_fact_inconsistent_conjunction", "post_empty_absorption", "nonempty_repeated_observation",
              "typed_observation_separator", "typed_query_partition_separator", "skewed_possible_value_partition")
CORE_CONTROLS = ("equal_support_usefulness_counterfactual", "observation_reorder_final_invariance", "nonempty_repetition_final_invariance",
                 "cross_fact_contradiction_counterfactual", "post_empty_append_absorption", "query_id_position_fact_crossing", "field_order_invariance")


def equal(a, b):
    return canonical(a) == canonical(b)


def prefixes(task, observations=None):
    observations = task["observations"] if observations is None else observations
    result = [list(task["worlds"])]
    for observation in observations:
        result.append([world for world in result[-1] if equal(world["facts"][observation["fact"]], observation["equals"])])
    return result


def ids(worlds):
    return sorted(world["world_id"] for world in worlds)


def bins(worlds, fact):
    return Counter(canonical(world["facts"][fact]) for world in worlds)


def reorder(value):
    if isinstance(value, dict):
        return {key: reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(item) for item in value]
    return value


def base_family(index):
    rng = random.Random(SEED + 101 * index)
    profile = PROFILES[index]
    if profile == "cyclic_uniform":
        cells = [(0, 0, 2), (1, 0, 2), (0, 1, 2), (1, 1, 2), (0, 2, 2), (2, 1, 2)]
        av = ["reed", "fern", "stone"]
        selected_values = [None, True]
    elif profile == "acyclic_branching":
        cells = [(0, 0, 2), (1, 0, 2), (0, 1, 1), (0, 2, 2), (2, 1, 1)]
        av = [False, 0, "unknown"]
        selected_values = [False, 0]
    elif profile == "cyclic_weighted":
        cells = [(0, 0, 3), (1, 0, 3), (0, 1, 1), (1, 1, 2), (0, 2, 1), (2, 1, 2)]
        av = [True, 1, "stone"]
        selected_values = [None, None, True]
    else:
        cells = [(0, 0, 2), (1, 0, 2), (0, 1, 1), (1, 2, 2), (2, 2, 1)]
        av = [True, 1, "stone"]
        selected_values = [False, 0]
    bv = ["linen", "silk", "cotton"]
    pool = rng.sample(WORDS, 5)
    worlds = []
    numbers = rng.sample(range(10000, 99999), sum(weight for _, _, weight in cells))
    for row, column, weight in cells:
        for within in range(weight):
            invariant = None if (row, column) == (0, 0) else "opal"
            varying = (selected_values[within] if (row, column) == (0, 0)
                       else "unknown" if (row, column) == (1, 0) else row + column)
            facts = dict(observation_control=av[row], observation_auxiliary=bv[column], invariant_query=invariant,
                         variation_query=varying, unoffered_feature=["east", "west", "north"][within])
            worlds.append(dict(world_id=f"w{numbers[len(worlds)]}", facts=facts))
    rng.shuffle(worlds)
    assert len({canonical(world["facts"]) for world in worlds}) == len(worlds)
    a = [dict(fact="observation_control", equals=av[0]), dict(fact="observation_auxiliary", equals=bv[0])]
    contexts = {"A": a,
                "B": [{**a[0], "equals": av[1]}, copy.deepcopy(a[1])],
                "C": list(reversed(copy.deepcopy(a))),
                "D": copy.deepcopy(a) + [copy.deepcopy(a[0])],
                "E": [{**a[0], "equals": av[2]}, copy.deepcopy(a[1])]}
    contexts["F"] = copy.deepcopy(contexts["E"]) + [copy.deepcopy(a[0])]
    return worlds, contexts, pool[:3], pool[3:], ["q0", "q1", "q2"]


def make_task(worlds, observations, names, auxiliaries, query_ids, u, v, w):
    role_names = {"invariant_query": names[w], "variation_query": names[(1 + w) % 3], "observation_control": names[(2 + w) % 3],
                  "observation_auxiliary": auxiliaries[0], "unoffered_feature": auxiliaries[1]}
    rendered_worlds = [dict(world_id=world["world_id"], facts={role_names[role]: value for role, value in world["facts"].items()}) for world in worlds]
    rendered_observations = [dict(fact=role_names[item["fact"]], equals=item["equals"]) for item in observations]
    assignments, queries = [], [None] * 3
    for role_number, role in enumerate(QUERY_ROLES):
        position, query_id = (role_number + v) % 3, query_ids[(role_number + u) % 3]
        fact = role_names["observation_control" if role == "observed_control" else role]
        queries[position] = dict(query_id=query_id, fact=fact)
        assignments.append(dict(role=role, position=position, query_id=query_id, fact=fact))
    return dict(kind="belief", worlds=rendered_worlds, observations=rendered_observations, queries=queries), role_names, assignments


def witness_locators(task):
    support = prefixes(task)
    final, m = support[-1], len(task["observations"])
    obs, result = task["observations"], {}
    query_bins = {query["query_id"]: bins(final, query["fact"]) for query in task["queries"]}
    distinct_payloads = len({canonical(world["facts"]) for world in final})
    def put(name, query_ids=(), observation_indices=(), prefix_indices=()):
        result.setdefault(name, dict(query_ids=list(query_ids), observation_indices=sorted(set(observation_indices)), prefix_indices=sorted(set(prefix_indices))))
    for query in task["queries"]:
        qid, fact, partition = query["query_id"], query["fact"], query_bins[query["query_id"]]
        if distinct_payloads >= 2 and len(partition) == 1 and not any(item["fact"] == fact for item in obs):
            put("multiworld_unobserved_invariant", [qid], [], [m])
            for i, j in combinations(range(m), 2):
                if obs[i]["fact"] == obs[j]["fact"]:
                    continue
                omissions = [prefixes(task, [item for k, item in enumerate(obs) if k != omitted])[-1] for omitted in (i, j)]
                if all(len(bins(worlds, fact)) >= 2 for worlds in omissions):
                    put("conjunction_required_invariant", [qid], [i, j], [0, m])
                    break
        reductions = [i for i in range(m) if 0 < len(support[i + 1]) < len(support[i])]
        if final and len(partition) > 1 and len({item["fact"] for item in obs}) >= 2 and len(reductions) >= 2:
            put("useful_after_conjunctive_evidence", [qid], reductions, [m] + [i for j in reductions for i in (j, j + 1)])
        if distinct_payloads >= 2 and partition == Counter({"null": len(final)}):
            put("multiworld_entailed_null", [qid], [], [m])
        if "null" in partition and len(partition) >= 2:
            put("ambiguous_null_partition", [qid], [], [m])
        if any(a in partition and b in partition for a, b in (("true", "1"), ("false", "0"))):
            put("typed_query_partition_separator", [qid], [], [m])
        if len(partition) >= 2 and len(set(partition.values())) >= 2:
            put("skewed_possible_value_partition", [qid], [], [m])
    if distinct_payloads >= 2 and all(len(partition) == 1 for partition in query_bins.values()):
        offered = {query["fact"] for query in task["queries"]}
        if any(len(bins(final, fact)) > 1 for fact in task["worlds"][0]["facts"] if fact not in offered):
            put("multiworld_all_queries_redundant", [query["query_id"] for query in task["queries"]], [], [m])
    for i, j in combinations(range(m), 2):
        if obs[i]["fact"] != obs[j]["fact"]:
            alone_i, alone_j = prefixes(task, [obs[i]])[-1], prefixes(task, [obs[j]])[-1]
            jointly = prefixes(task, [obs[i], obs[j]])[-1]
            if alone_i and alone_j and not jointly and not final:
                put("cross_fact_inconsistent_conjunction", [], [i, j], [0, i + 1, j + 1, m])
        if equal(obs[i], obs[j]) and support[j] and ids(support[j]) == ids(support[j + 1]):
            put("nonempty_repeated_observation", [], [i, j], [j, j + 1])
    first_empty = next((i for i, worlds in enumerate(support) if not worlds), None)
    if first_empty is not None and first_empty < m:
        later = next((i for i in range(first_empty, m) if prefixes(task, [obs[i]])[-1]), None)
        if later is not None and all(not worlds for worlds in support[first_empty:]):
            put("post_empty_absorption", [query["query_id"] for query in task["queries"]], [first_empty - 1, later], [first_empty - 1, first_empty, later, later + 1, m])
    for i, observation in enumerate(obs):
        value = canonical(observation["equals"])
        alias = {"true": "1", "1": "true", "false": "0", "0": "false"}.get(value)
        available = bins(support[i], observation["fact"])
        if alias and value in available and alias in available and support[i + 1]:
            put("typed_observation_separator", [], [i], [i, i + 1])
    return result


def incidence_profile(worlds):
    controls = sorted({canonical(world["facts"]["observation_control"]) for world in worlds})
    auxiliaries = sorted({canonical(world["facts"]["observation_auxiliary"]) for world in worlds})
    weights = Counter((canonical(world["facts"]["observation_control"]), canonical(world["facts"]["observation_auxiliary"])) for world in worlds)
    vertices = [("a", value) for value in controls] + [("b", value) for value in auxiliaries]
    adjacency = {vertex: set() for vertex in vertices}
    for a, b in weights:
        adjacency["a", a].add(("b", b)); adjacency["b", b].add(("a", a))
    components, unseen = 0, set(vertices)
    while unseen:
        components += 1
        frontier = [unseen.pop()]
        while frontier:
            vertex = frontier.pop()
            for neighbor in adjacency[vertex] & unseen:
                unseen.remove(neighbor); frontier.append(neighbor)
    matrices = []
    for rows in permutations(controls):
        for columns in permutations(auxiliaries):
            matrix = tuple(weights.get((a, b), 0) for a in rows for b in columns)
            matrices.append(matrix)
            matrices.append(tuple(weights.get((a, b), 0) for b in columns for a in rows))
    signature = dict(canonical_weighted_incidence=list(min(matrices)), world_count=len(worlds),
                     fact_partition_block_sizes=sorted(sorted(bins(worlds, fact).values()) for fact in FACT_ROLES))
    return dict(signature=signature, signature_sha256=digest(signature), occupied_cells=len(weights),
                positive_cell_multiplicities=sorted(weights.values()), vertex_degrees=sorted(len(adjacency[vertex]) for vertex in vertices),
                component_count=components, cycle_rank=len(weights) - len(vertices) + components,
                total_worlds=len(worlds), distinct_typed_payloads=len({canonical(world["facts"]) for world in worlds}))


def validate_base_contexts(worlds, contexts, profile):
    queries = [dict(query_id=f"q{i}", fact="observation_control" if role == "observed_control" else role) for i, role in enumerate(QUERY_ROLES)]
    tasks = {context: dict(kind="belief", worlds=worlds, observations=observations, queries=queries) for context, observations in contexts.items()}
    supports = {context: prefixes(task) for context, task in tasks.items()}
    a, b = supports["A"], supports["B"]
    assert 2 <= len(a[-1]) == len(b[-1])
    assert len(worlds) > len(a[1]) > len(a[2]) > 0
    assert len(a[1]) != len(supports["C"][1])
    assert ids(a[-1]) == ids(supports["C"][-1]) == ids(supports["D"][-1])
    assert len(supports["D"][-1]) == len(supports["D"][-2])
    assert not supports["E"][-1] and not supports["F"][-1] and len(supports["F"][-2]) == 0
    a_witnesses, b_witnesses = witness_locators(tasks["A"]), witness_locators(tasks["B"])
    assert "conjunction_required_invariant" in a_witnesses and "useful_after_conjunctive_evidence" in a_witnesses
    assert "multiworld_all_queries_redundant" in b_witnesses
    assert len(bins(a[-1], "variation_query")) > 1 and len(bins(b[-1], "variation_query")) == 1
    if profile in ("cyclic_uniform", "cyclic_weighted"):
        assert "multiworld_entailed_null" in a_witnesses and "ambiguous_null_partition" in a_witnesses
    graph = incidence_profile(worlds)
    assert graph["component_count"] == 1
    if profile == "cyclic_uniform":
        assert graph["cycle_rank"] > 0 and len(set(graph["positive_cell_multiplicities"])) == 1
    elif profile == "cyclic_weighted":
        assert graph["cycle_rank"] > 0 and len(set(graph["positive_cell_multiplicities"])) > 1 and len(a[-1]) >= 3
        assert "skewed_possible_value_partition" in a_witnesses
    elif profile == "acyclic_branching":
        assert graph["cycle_rank"] == 0 and max(graph["vertex_degrees"]) >= 3
    else:
        assert graph["cycle_rank"] == 0 and max(graph["vertex_degrees"]) == 2
    return graph


def typed_query_signature(row, result):
    support = prefixes(row["semantic_task"])
    category = "empty" if not support[-1] else "entailed" if result["entailed"] else "ambiguous"
    return canonical(dict(full_query_payload={key: value for key, value in result.items() if key != "query_id"},
                          audit_category=category, guaranteed_null_context=(category if result["guaranteed"] is None else "nonnull"),
                          value_types=["null" if item["value"] is None else "boolean" if type(item["value"]) is bool else "integer" if type(item["value"]) is int else "string" for item in result["counts"]],
                          final_survivors=ids(support[-1]), prefix_survivors=[ids(worlds) for worlds in support], prefix_counts=[len(worlds) for worlds in support]))


def crossing_report(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_family_key"], row["controlled_context"], row["renderer"]].append(row)
    reports = []
    for (source, context, style), views in sorted(grouped.items()):
        assert len(views) == 27 and {row["schedule_index"] for row in views} == set(range(27))
        roles, outcomes = defaultdict(Counter), defaultdict(Counter)
        for row in views:
            results = {result["query_id"]: result for result in row["verified_target"]["queries"]}
            for assignment in row["query_role_assignments"]:
                nuisance = (assignment["query_id"], str(assignment["position"]), assignment["fact"])
                signature = typed_query_signature(row, results[assignment["query_id"]])
                for axes in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
                    name = ",".join(("id", "position", "fact")[axis] for axis in axes)
                    bucket = canonical([nuisance[axis] for axis in axes])
                    roles[name, bucket][assignment["role"]] += 1
                    outcomes[name, bucket][signature] += 1
        histograms = {}
        for axes in ("id", "position", "fact", "id,position", "id,fact", "position,fact", "id,position,fact"):
            buckets = {bucket: dict(table) for (name, bucket), table in roles.items() if name == axes}
            outcome_buckets = {bucket: dict(table) for (name, bucket), table in outcomes.items() if name == axes}
            expected = 9 if axes.count(",") == 0 else 3 if axes.count(",") == 1 else 1
            assert all(table == dict(Counter({role: expected for role in QUERY_ROLES})) for table in buckets.values())
            assert all(table == next(iter(outcome_buckets.values())) for table in outcome_buckets.values())
            histograms[axes] = dict(actual_role_histograms=buckets, actual_full_typed_outcome_provenance_histograms=outcome_buckets)
        reports.append(dict(source_family_key=source, controlled_context=context, renderer=style, actual_schedule_indices=sorted(row["schedule_index"] for row in views),
                            histograms=histograms, complete_positive_triple_support=True, measured_bucket_histogram_difference=0))
    return reports


def family_evidence(rows):
    baseline = {(row["source_family_index"], row["controlled_context"]): row for row in rows if row["schedule_index"] == 0 and row["renderer"] == "forward"}
    pairs, probes = [], []
    for index in PROFILES:
        a = baseline[index, "A"]
        for context, name in (("B", "equal_support_usefulness_counterfactual"), ("C", "observation_reorder_final_invariance"),
                              ("D", "nonempty_repetition_final_invariance"), ("E", "cross_fact_contradiction_counterfactual")):
            other = baseline[index, context]
            pairs.append(dict(name=name, scope="corpus_variant", source_family_key=a["source_family_key"], pair_sample_ids=[a["sample_id"], other["sample_id"]]))
        pairs.append(dict(name="post_empty_append_absorption", scope="corpus_variant", source_family_key=a["source_family_key"], pair_sample_ids=[baseline[index, "E"]["sample_id"], baseline[index, "F"]["sample_id"]]))
        if index in (13, 18, 19):
            pairs.append(dict(name="typed_equals_counterfactual", scope="corpus_variant", source_family_key=a["source_family_key"], pair_sample_ids=[a["sample_id"], baseline[index, "B"]["sample_id"]]))
        if index in (12, 18):
            pairs.append(dict(name="null_guarantee_contrast", scope="corpus_variant", source_family_key=a["source_family_key"],
                              evidence_sample_ids=[a["sample_id"], baseline[index, "E"]["sample_id"]]))
        original = a["semantic_task"]
        reordered = copy.deepcopy(original); reordered["worlds"].reverse()
        reordered_target = solve(reordered)
        assert equal(reordered_target, a["verified_target"])
        rename = {world["world_id"]: f"p{position:02d}" for position, world in enumerate(reversed(original["worlds"]))}
        renamed = copy.deepcopy(original)
        for world in renamed["worlds"]:
            world["world_id"] = rename[world["world_id"]]
        renamed_target = solve(renamed)
        expected = {**a["verified_target"], "surviving_worlds": sorted(rename[value] for value in a["verified_target"]["surviving_worlds"])}
        assert equal(renamed_target, expected)
        for name, task, target in (("world_order_invariance", reordered, reordered_target), ("world_id_rename_equivariance", renamed, renamed_target)):
            probes.append(dict(name=name, scope="audit_probe", source_family_key=a["source_family_key"], probe_id=f"{index}-{name}",
                               source_sample_id=a["sample_id"], probe_task=task, producer_reference_target=target, added_records=0, added_families=0))
    return pairs, probes


def metrics(row):
    task, support = row["semantic_task"], prefixes(row["semantic_task"])
    final = support[-1]
    first_empty = next((i for i, worlds in enumerate(support) if not worlds), None)
    return dict(world_count=len(task["worlds"]), fact_count=len(task["worlds"][0]["facts"]), query_count=len(task["queries"]), observation_count=len(task["observations"]),
                prefix_counts=[len(worlds) for worlds in support], strict_nonempty_reductions=sum(0 < len(support[i + 1]) < len(support[i]) for i in range(len(support) - 1)),
                final_distinct_payloads=len({canonical(world["facts"]) for world in final}),
                final_query_bin_counts=[len(bins(final, query["fact"])) for query in task["queries"]],
                essential_constraint_count=2 if "conjunction_required_invariant" in row["property_witnesses"] else None,
                minimal_contradiction_witness_size=2 if "cross_fact_inconsistent_conjunction" in row["property_witnesses"] else None,
                steps_after_first_empty=len(task["observations"]) - first_empty if first_empty is not None else 0,
                complete_typed_query_answers=row["verified_target"]["queries"])


def validate_rows(rows):
    property_counts = Counter()
    per_context = defaultdict(Counter)
    twins = defaultdict(list)
    prompts, tasks = set(), set()
    for row in rows:
        assert row["review_status"] == "PENDING" and not row["review_receipt_ids"]
        assert row["record_hash"] == record_hash(row)
        assert row["split"] == split_for(row["source_family_index"])
        assert row["schedule_index"] == 9 * row["id_shift"] + 3 * row["position_shift"] + row["fact_name_shift"]
        assert set(row["skill_tags"]) == {"finite_synthetic_proxy", "finite_noiseless_filter", "possible_value_partition", *row["property_witnesses"]}
        assert equal(witness_locators(row["semantic_task"]), row["property_witnesses"])
        for locator in row["property_witnesses"].values():
            assert set(locator) == {"query_ids", "observation_indices", "prefix_indices"}
        assert set(row["fact_role_assignments"]) == set(FACT_ROLES)
        assert len(set(row["fact_role_assignments"].values())) == 5
        assert {assignment["role"] for assignment in row["query_role_assignments"]} == set(QUERY_ROLES)
        _, payload = row["rendered_input"].split("Task JSON:\n")
        assert equal(json.loads(payload), row["semantic_task"])
        assert row["source_family_key"] not in row["rendered_input"] and row["sample_id"] not in row["rendered_input"]
        assert row["rendered_input"] not in prompts
        prompts.add(row["rendered_input"]); tasks.add(digest(row["semantic_task"]))
        twins[row["source_family_key"], row["controlled_context"], row["schedule_index"]].append(row)
        property_counts.update(row["property_witnesses"].keys())
        per_context[row["controlled_context"]].update(row["property_witnesses"].keys())
    for pair in twins.values():
        assert len(pair) == 2 and equal(pair[0]["semantic_task"], pair[1]["semantic_task"])
        assert equal(pair[0]["verified_target"], pair[1]["verified_target"])
    return dict(rendered_records=len(rows), distinct_task_jsons=len(tasks), distinct_prompts=len(prompts), field_order_pairs=len(twins),
                observed_property_row_counts=dict(property_counts), property_counts_by_context={key: dict(value) for key, value in per_context.items()})


def generate(batch="belief-gap-v1", seed=SEED):
    if seed != SEED or not batch or "/" in batch or "\\" in batch:
        raise ValueError("This frozen pilot uses seed20261009 and a fresh filename-safe batch")
    start = time.perf_counter()
    track = "RBM-001"
    candidate_path = ROOT / f"data/{track}/candidate_renderings/{batch}.jsonl"
    parent_path = ROOT / f"data/{track}/semantic_parents/{batch}.jsonl"
    manifest_path = ROOT / f"manifests/{batch}_GENERATION.json"
    source_snapshot = ROOT / f"manifests/{batch}_SOURCE.py"
    report_path = ROOT / f"data/{track}/manifests/{batch}.producer_integrity.json"
    control_path = ROOT / f"data/{track}/manifests/{batch}.family_controls.json"
    for path in (candidate_path, parent_path, manifest_path, source_snapshot, report_path, control_path):
        if path.exists():
            raise ValueError(f"First run is frozen at {path}; use a new batch for any revision")
    public_actual = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in PUBLIC_HASHES}
    assert public_actual == PUBLIC_HASHES
    source = Path(__file__).read_text(encoding="utf-8")
    atomic(source_snapshot, source)
    rows, parents, graphs, base_metrics = [], [], {}, []
    reference_calls = 0
    status, failure = "PRODUCER_CHECK_FAILED", None
    try:
        prompt_contract = contracts()["belief"]
        for index, profile in PROFILES.items():
            split = split_for(index)  # Before worlds, contexts, schedules or rendering.
            worlds, contexts, name_pool, auxiliary_names, query_ids = base_family(index)
            graph = validate_base_contexts(worlds, contexts, profile)
            graphs[str(index)] = graph
            source_family = f"RBM-seed{SEED}-p{index:04d}"
            family_id = f"{track}-{batch}-p{index:04d}"
            family_tasks, variant_keys = [], []
            controls = list(CORE_CONTROLS)
            if index in (13, 18, 19):
                controls.append("typed_equals_counterfactual")
            if index in (12, 18):
                controls.append("null_guarantee_contrast")
            for context, observations in contexts.items():
                for u, v, w in product(range(3), repeat=3):
                    schedule = 9 * u + 3 * v + w
                    task, role_names, assignments = make_task(worlds, observations, name_pool, auxiliary_names, query_ids, u, v, w)
                    target = solve(task); reference_calls += 1
                    witnesses = witness_locators(task)
                    variant = f"{context}-s{schedule:02d}"
                    family_tasks.append(task); variant_keys.append(variant)
                    for renderer in ("forward", "reverse"):
                        prompt = "Executable task contract:\n" + prompt_contract + "Task JSON:\n" + json.dumps(task if renderer == "forward" else reorder(task), ensure_ascii=False, separators=(",", ":"))
                        row = record(track, batch, index, variant, task, prompt, target,
                                     ["finite_synthetic_proxy", "finite_noiseless_filter", "possible_value_partition"] + sorted(witnesses), 3, "structured", renderer)
                        row.update(controlled_context=context, structural_family_profile=profile, source_family_index=index, source_family_key=source_family,
                                   id_shift=u, position_shift=v, fact_name_shift=w, schedule_index=schedule,
                                   query_id_pool=query_ids, queried_fact_name_pool=name_pool, fact_role_assignments=role_names,
                                   query_role_assignments=assignments, property_witnesses=witnesses, structural_difficulty=3,
                                   family_control_claims=[dict(name=name, scope="corpus_variant", source_family_key=source_family,
                                                              evidence_manifest=f"data/{track}/manifests/{batch}.family_controls.json", control_id=f"{index}-{name}") for name in controls] +
                                                         [dict(name=name, scope="audit_probe", source_family_key=source_family,
                                                               evidence_manifest=f"data/{track}/manifests/{batch}.family_controls.json", probe_id=f"{index}-{name}")
                                                          for name in ("world_order_invariance", "world_id_rename_equivariance")],
                                   producer_role="belief_producer", producer_agent_id="/root/relational_critic_producer",
                                   scope_limitation="Finite noiseless typed filtering; correlated shifts and controls; no Bayesian/action/OOD/efficacy claim.")
                        row["record_hash"] = record_hash(row)
                        rows.append(row)
                        if schedule == 0 and renderer == "forward":
                            base_metrics.append(dict(source_family_key=source_family, context=context, metrics=metrics(row)))
            parents.append(dict(semantic_parent_id=family_id, semantic_family_id=family_id, source_family_key=source_family,
                                source_family_index=index, split=split, structural_family_profile=profile, tasks=family_tasks,
                                task_variant_keys=variant_keys, contexts=6, correlated_schedules_per_context=27, field_order_renderers=2,
                                source_family_count=1, structural_difficulty=3))
        # Freeze the actual first candidate before producer checks/reports.
        write_jsonl(parent_path, parents); write_jsonl(candidate_path, rows)
        assert len({graph["signature_sha256"] for graph in graphs.values()}) == 4
        report = validate_rows(rows)
        crossing = crossing_report(rows)
        pairs, probes = family_evidence(rows); reference_calls += len(probes)
        assert len(rows) == 1296 and len(parents) == 4 and len(crossing) == 48
        prior_tables, prior_counts = set(), set()
        for prior_batch in ("extended-pilot-v1", "extended-pilot-v2"):
            path = ROOT / f"data/{track}/semantic_parents/{prior_batch}.jsonl"
            if path.exists():
                for parent in load_jsonl(path):
                    for task in parent["tasks"]:
                        prior_tables.add(digest(sorted(canonical(world["facts"]) for world in task["worlds"])))
                        prior_counts.add(len(task["worlds"]))
        current_tables = {digest(sorted(canonical(world["facts"]) for world in row["semantic_task"]["worlds"])) for row in rows}
        assert not prior_tables & current_tables and all(graph["total_worlds"] not in prior_counts for graph in graphs.values())
        report.update(source_families=4, base_contexts=24, base_context_query_observations=sum(item["metrics"]["query_count"] for item in base_metrics),
                      rotation_tasks=len(rows) // 2, rotation_query_observations=sum(len(row["semantic_task"]["queries"]) for row in rows if row["renderer"] == "forward"),
                      rendered_query_observations=sum(len(row["semantic_task"]["queries"]) for row in rows),
                      split_records=dict(Counter(row["split"] for row in rows)), split_source_families=dict(Counter(parent["split"] for parent in parents)),
                      structural_profiles=graphs, base_context_metrics=base_metrics, exact_prior_own_world_table_overlap=0,
                      prior_own_world_counts=sorted(prior_counts), shape_freshness="Each new unique-payload world count exceeds own prior tables; four canonical incidence signatures differ.",
                      supplied_pair_counts=dict(Counter(item["name"] for item in pairs)), audit_probe_counts=dict(Counter(item["name"] for item in probes)),
                      complete_positive_triple_support=True, actual_histogram_report_sets=len(crossing), producer_checks_passed=True,
                      limitations=["Only4 independent source families; contexts/shifts/renderers remain correlated.",
                                   "Cyclic row orders, auxiliary fact roles and world-ID styling are not independently crossed.",
                                   "Exposed development contains new profiles, but no sealed OOD or model efficacy was measured.",
                                   "No noisy beliefs, Bayesian probability, expected inspection utility, actions or authority semantics."])
        write_json(report_path, report)
        write_json(control_path, dict(actual_crossing_histograms=crossing, supplied_pairs=pairs, executed_audit_probes=probes,
                                     field_order_pair_count=len(rows) // 2, probes_add_corpus_records=0, probes_add_source_families=0))
        status = "PROVISIONAL_PENDING_INDEPENDENT_REVIEW"
    except Exception:
        failure = traceback.format_exc()
        # Retain any partial first candidate; never silently regenerate it.
        if not parent_path.exists(): write_jsonl(parent_path, parents)
        if not candidate_path.exists(): write_jsonl(candidate_path, rows)
        raise
    finally:
        artifacts = [path for path in (candidate_path, parent_path, source_snapshot, report_path, control_path) if path.exists()]
        manifest = dict(batch=batch, track_id=track, status=status, seed=seed, source_family_indices=list(PROFILES),
                        producer_agent_id="/root/relational_critic_producer", producer_role="belief_producer",
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(), public_input_sha256=public_actual,
                        root_freeze_sha256=hashlib.sha256((ROOT / "research/curricula/RBM-EVIDENCE-GAP-ROOT_FREEZE.json").read_bytes()).hexdigest(),
                        reference_sha256=hashlib.sha256((ROOT / "generators/other_tracks/reference.py").read_bytes()).hexdigest(),
                        reproduction=f"/opt/anaconda3/bin/python3 -m generators.other_tracks.belief_gap --batch {batch}",
                        config=dict(profiles=PROFILES, contexts=list("ABCDEF"), id_shifts=[0, 1, 2], position_shifts=[0, 1, 2], fact_shifts=[0, 1, 2], field_orders=["forward", "reverse"]),
                        actual_counts=dict(source_families=len(parents), records=len(rows), reference_solve_calls=reference_calls),
                        cost=dict(wall_seconds=time.perf_counter() - start, world_observation_comparisons=sum(len(row["semantic_task"]["worlds"]) * len(row["semantic_task"]["observations"]) for row in rows if row["renderer"] == "forward"),
                                  output_bytes=sum(path.stat().st_size for path in artifacts)),
                        artifact_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
                        initial_run_failure=failure, no_approval_release_or_push=True)
        write_json(manifest_path, manifest)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="belief-gap-v1")
    arguments = parser.parse_args()
    produced = generate(arguments.batch)
    print(f"Produced {len(produced)} PENDING records in {arguments.batch}")
