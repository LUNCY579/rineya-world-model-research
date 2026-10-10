"""RBM-GAP-007 v2 producer; frozen public targeted value counterfactuals.

Imports only common storage, the producer reference, and frozen own-v1 helpers.
The first candidate/run is preserved even when a producer check fails.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record, record_hash, split_for, write_json, write_jsonl
from generators.other_tracks.reference import contracts, solve
from generators.other_tracks import belief_gap as own_v1

SEED = own_v1.SEED
PROFILES = own_v1.PROFILES
GENERATOR_VERSION = "rbm-belief-gap-2.0.0"
PUBLIC_HASHES = {**own_v1.PUBLIC_HASHES,
    "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md": "46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f",
    "research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md": "c2eb841ed546848088db4e0499e8e8f7087aa3d213af333ff39237ae6558bd33",
    "schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json": "611656b37d7fe917d64eeb191d901d272a32d8d3c72c9f8865476c6889a3e51a"}
ROOT_FREEZE = "research/curricula/RBM-EVIDENCE-GAP-REVISION-V2-ROOT_FREEZE.json"
ROOT_FREEZE_SHA = "a90cf0c6af1d11ca706a9be436f3c0010ae71a73195dec9b053863b7ff69a7c4"
V1_SOURCE_SHA = "f27373cd70b09d52880ab3c6f526347ccd440264f5b7918995c6ef8782694522"
V1_CANDIDATE_SHA = "423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e"
MARKER_RULE = dict(rule_id="original_communicated_stone_unknown_anywhere",
    rule="Predict empty iff at least one observation equals the typed JSON string stone or unknown; otherwise predict nonempty.",
    exact_typed_marker_values=["stone", "unknown"], input_fields=["observations[*].equals"],
    frozen_before_generation=True, actual_author="/root/relational_critic_producer",
    source="Parent-communicated whole-batch marker association; no reviewer file accessed.",
    old_candidate_sha256=V1_CANDIDATE_SHA,
    status_decoder="Exact frozen Boolean rule; no fitted threshold, labels, metadata or final-support input.",
    structured_decoder="When marker-positive, predict empty bins and survivors, compute only the first prefix from visible input, then zeros. Marker-negative abstains; abstentions count as incorrect task/query answers.")

equal = own_v1.equal
prefixes = own_v1.prefixes
ids = own_v1.ids
bins = own_v1.bins


def base_family(index):
    worlds, contexts, names, auxiliaries, query_ids = own_v1.base_family(index)
    e = contexts["E"]
    contexts["F"] = copy.deepcopy(e) + [copy.deepcopy(e[0])]
    supported = prefixes(dict(worlds=worlds, observations=[e[0]]))[-1]
    second_fact = e[1]["fact"]
    counts = bins(supported, second_fact)
    value_key = min(counts, key=lambda value: (-counts[value], value))
    alternative = json.loads(value_key)
    assert not equal(alternative, e[1]["equals"])
    contexts["G"] = [copy.deepcopy(e[0]), {**e[1], "equals": alternative}]
    contexts["H"] = copy.deepcopy(contexts["G"]) + [copy.deepcopy(e[0])]
    assert list(contexts) == list("ABCDEFGH")
    return worlds, contexts, names, auxiliaries, query_ids


def metrics(task):
    support = prefixes(task)
    final = support[-1]
    m = len(task["observations"])
    first_empty = next((i for i, worlds in enumerate(support) if not worlds), None)
    essential = [i for i in range(m) if ids(prefixes(task, [obs for j, obs in enumerate(task["observations"]) if j != i])[-1]) != ids(final)]
    repeated = [dict(earlier=i, repeated=j, incoming_count=len(support[j]), outgoing_count=len(support[j + 1]),
                     unchanged_support=ids(support[j]) == ids(support[j + 1]))
                for j in range(m) for i in range(j) if equal(task["observations"][i], task["observations"][j])]
    return dict(world_count=len(task["worlds"]), fact_count=len(task["worlds"][0]["facts"]), query_count=len(task["queries"]), observation_count=m,
                prefix_counts=[len(worlds) for worlds in support], prefix_survivor_ids=[ids(worlds) for worlds in support],
                strict_nonempty_filter_count=sum(0 < len(support[i + 1]) < len(support[i]) for i in range(m)),
                final_support_count=len(final), final_distinct_payload_count=len({canonical(world["facts"]) for world in final}),
                final_query_bin_counts=[len(bins(final, query["fact"])) for query in task["queries"]],
                essential_observation_indices=essential, essential_observation_count=len(essential),
                repeated_observations=repeated, first_empty_prefix_index=first_empty,
                steps_after_first_empty=m - first_empty if first_empty is not None else 0)


def difficulty(task, witnesses):
    measured = metrics(task)
    # Actual conjunctive constraints/contradiction or two strict nonempty filters.
    return 3 if ("conjunction_required_invariant" in witnesses or "cross_fact_inconsistent_conjunction" in witnesses
                 or measured["strict_nonempty_filter_count"] >= 2) else 2


def check_contexts(worlds, contexts, profile):
    graph = own_v1.validate_base_contexts(worlds, contexts, profile)
    for left, right in (("E", "G"), ("F", "H")):
        e, g = contexts[left], contexts[right]
        assert len(e) == len(g) and equal(e[0], g[0])
        assert [obs["fact"] for obs in e] == [obs["fact"] for obs in g]
        assert [i for i in range(len(e)) if not equal(e[i], g[i])] == [1]
        assert not equal(e[1]["equals"], g[1]["equals"])
        es = prefixes(dict(worlds=worlds, observations=e))
        gs = prefixes(dict(worlds=worlds, observations=g))
        assert ids(es[1]) == ids(gs[1]) and not es[-1] and gs[-1]
    assert equal(contexts["F"][0], contexts["F"][2]) and equal(contexts["H"][0], contexts["H"][2])
    return graph


def normalized_target(task, target):
    return dict(surviving_worlds=target["surviving_worlds"], belief_sizes=target["belief_sizes"],
                query_payloads_by_fact={query["fact"]: {k: v for k, v in answer.items() if k != "query_id"}
                                       for query, answer in zip(task["queries"], target["queries"])})


def decode_target(task, template):
    if template is None or set(template["query_payloads_by_fact"]) != {query["fact"] for query in task["queries"]}:
        return None
    return dict(surviving_worlds=template["surviving_worlds"], belief_sizes=template["belief_sizes"],
                queries=[dict(query_id=query["query_id"], **template["query_payloads_by_fact"][query["fact"]]) for query in task["queries"]])


def marker_predicts_empty(task):
    return any(canonical(obs["equals"]) in ('"stone"', '"unknown"') for obs in task["observations"])


def marker_structured_prediction(task):
    if not marker_predicts_empty(task):
        return None
    first_count = len(prefixes(task, task["observations"][:1])[-1])
    return dict(surviving_worlds=[], belief_sizes=[first_count] + [0] * (len(task["observations"]) - 1),
                queries=[dict(query_id=query["query_id"], counts=[], distinct_values=0, entailed=False,
                              guaranteed=None, can_reduce_uncertainty=False) for query in task["queries"]])


def marker_and_linen(task):
    return marker_predicts_empty(task) and any(type(obs["equals"]) is str and obs["equals"]=="linen" for obs in task["observations"])


def root_world_count_empty_core(task, predicate):
    # Root's separately frozen world-count rule; NO dynamic first-prefix filter.
    count=len(task["worlds"])
    if not predicate(task) or count not in (8,12):return None
    # Root prefix_counts includes S0; the task belief_sizes contract excludes S0.
    return dict(surviving_worlds=[],belief_sizes=[1 if count==8 else 2]+[0]*(len(task["observations"])-1),
                queries=[dict(query_id=query["query_id"],counts=[],distinct_values=0,entailed=False,
                              guaranteed=None,can_reduce_uncertainty=False) for query in task["queries"]])


def score_predictions(rows, status_predictions, target_predictions):
    count, status_ok, tasks_ok, queries_ok, queries, covered = len(rows), 0, 0, 0, 0, 0
    for row, prediction, target in zip(rows, status_predictions, target_predictions):
        actual = row["verified_target"]
        status_ok += prediction == (not actual["surviving_worlds"])
        tasks_ok += target is not None and equal(target, actual)
        covered += target is not None
        for i, query in enumerate(actual["queries"]):
            queries += 1
            queries_ok += target is not None and i < len(target["queries"]) and equal(target["queries"][i], query)
    return dict(records=count, status_correct=status_ok, status_errors=count-status_ok, status_accuracy=status_ok/count,
                complete_task_correct=tasks_ok, complete_task_accuracy=tasks_ok/count, query_answers_correct=queries_ok,
                query_answers=queries, query_accuracy=queries_ok/queries, structured_prediction_coverage=covered/count,
                covered_structured_records=covered,conditional_complete_task_accuracy=tasks_ok/covered if covered else None,
                conditional_query_accuracy=queries_ok/(covered*3) if covered else None)


def marker_audit(rows):
    groups = defaultdict(list)
    for row in rows:
        length = len(row["semantic_task"]["observations"])
        groups["overall"].append(row)
        groups[f"family:{row['source_family_key']}"].append(row)
        groups[f"family_length:{row['source_family_key']}:{length}"].append(row)
        groups[f"split:{row['split']}"].append(row)
        groups[f"split_length:{row['split']}:{length}"].append(row)
    return {key: score_predictions(group, [marker_predicts_empty(row["semantic_task"]) for row in group],
                                  [marker_structured_prediction(row["semantic_task"]) for row in group]) for key, group in groups.items()}


def root_rules_audit(rows):
    groups=defaultdict(list)
    for row in rows:
        length=len(row["semantic_task"]["observations"])
        for key in ("overall",f"family:{row['source_family_key']}",f"family_length:{row['source_family_key']}:{length}",
                    f"split:{row['split']}",f"split_length:{row['split']}:{length}"):
            groups[key].append(row)
    report={}
    for name,predicate in (("v1_original_scalar_marker_empty",marker_predicts_empty),
                           ("prospective_marker_and_auxiliary_token_empty",marker_and_linen)):
        report[name]={}
        for key,group in groups.items():
            status=[predicate(row["semantic_task"]) for row in group]
            result=score_predictions(group,status,[root_world_count_empty_core(row["semantic_task"],predicate) for row in group])
            result.update(false_positive_empty=sum(prediction and bool(row["verified_target"]["surviving_worlds"]) for row,prediction in zip(group,status)),
                          false_negative_empty=sum(not prediction and not row["verified_target"]["surviving_worlds"] for row,prediction in zip(group,status)))
            report[name][key]=result
    return dict(nonfitted_frozen_root_rules=report,
                structured_decoder_boundary="Root frozen N8=>1/N12=>2 first-prefix lookup, then zeros; S0 excluded from belief_sizes. Nonpositive rule cases abstain. Producer dynamic-first-prefix variant is separately named.")


def lookup_baseline(rows, feature, fit_rows=None):
    # Features read ONLY visible task. Gold is used only to fit explicitly in-sample lookup labels.
    status_tables, task_tables = defaultdict(Counter), defaultdict(Counter)
    fitted=rows if fit_rows is None else fit_rows
    overall_status=Counter(not row["verified_target"]["surviving_worlds"] for row in fitted)
    overall_tasks=Counter(canonical(normalized_target(row["semantic_task"],row["verified_target"])) for row in fitted)
    for row in fitted:
        key = canonical(feature(row["semantic_task"]))
        status_tables[key][not row["verified_target"]["surviving_worlds"]] += 1
        task_tables[key][canonical(normalized_target(row["semantic_task"], row["verified_target"]))] += 1
    statuses, targets = [], []
    unknown=0
    fallback_status=min(overall_status,key=lambda value:(-overall_status[value],value))
    fallback_template=min(overall_tasks,key=lambda value:(-overall_tasks[value],value))
    for row in rows:
        key = canonical(feature(row["semantic_task"]))
        if key not in status_tables:
            unknown+=1;winner=fallback_status;template=fallback_template
        else:
            winner = min(status_tables[key], key=lambda value: (-status_tables[key][value], value))
            template = min(task_tables[key], key=lambda value: (-task_tables[key][value], value))
        statuses.append(winner)
        targets.append(decode_target(row["semantic_task"], json.loads(template)))
    return dict(**score_predictions(rows, statuses, targets), distinct_input_feature_buckets=len(status_tables),
                mixed_status_buckets=sum(len(table)>1 for table in status_tables.values()),
                evaluation="label-fitted in-sample diagnostic; no held-out predictive validity" if fit_rows is None else "Train-derived lookup evaluated on exposed development; four families only, no sealed OOD/model efficacy.",
                fit_records=len(fitted),unknown_bucket_records=unknown,unknown_bucket_fraction=unknown/len(rows),
                unknown_status_fallback="global fitted majority, tie=>nonempty",unknown_status_prediction_empty=fallback_status,
                unknown_structured_fallback="global fitted majority exact template; abstain if queried visible fact-name set differs")


def coarse(task):
    first = prefixes(task, task["observations"][:1])[-1]
    return dict(world_count=len(task["worlds"]), fact_count=len(task["worlds"][0]["facts"]), query_count=len(task["queries"]),
                observation_length=len(task["observations"]), first_prefix_ids=ids(first), first_prefix_count=len(first))


def scalar_and_lookup_audit(rows):
    grouped = defaultdict(list)
    for row in rows:
        length = len(row["semantic_task"]["observations"])
        grouped["overall"].append(row)
        grouped[f"family_length:{row['source_family_key']}:{length}"].append(row)
        grouped[f"split_length:{row['split']}:{length}"].append(row)
    report = {}
    for group_name, group in grouped.items():
        histogram = defaultdict(lambda: defaultdict(Counter))
        for row in group:
            task = row["semantic_task"]
            status = "empty" if not row["verified_target"]["surviving_worlds"] else "nonempty"
            inverse = {name: role for role, name in row["fact_role_assignments"].items()}
            seen_values, seen_facts, seen_roles = set(), set(), set()
            for position, obs in enumerate(task["observations"]):
                scalar, role, name = canonical(obs["equals"]), inverse[obs["fact"]], obs["fact"]
                seen_values.add(scalar); seen_facts.add(name); seen_roles.add(role)
                for axis, feature in (("scalar_position", [position,scalar]), ("fact_role_position",[position,role]),
                                      ("fact_name_position",[position,name]), ("fact_role_position_scalar",[role,position,scalar]),
                                      ("fact_name_position_scalar",[name,position,scalar]), ("fact_role_name_position_scalar",[role,name,position,scalar])):
                    histogram[axis][canonical(feature)][status] += 1
            for axis, values in (("scalar_anywhere",seen_values), ("fact_name_anywhere",seen_facts), ("fact_role_anywhere",seen_roles)):
                for value in values: histogram[axis][canonical(value)][status] += 1
            features = coarse(task)
            for key,value in features.items(): histogram["coarse_"+key][canonical(value)][status] += 1
            histogram["coarse_joint"][canonical(features)][status] += 1
        values = sorted({canonical(obs["equals"]) for row in group for obs in row["semantic_task"]["observations"]})
        scalar_tests = []
        position_tests = []
        for value in values:
            result = lookup_baseline(group, lambda task,v=value: any(canonical(obs["equals"])==v for obs in task["observations"]))
            scalar_tests.append(dict(typed_value_json=value, **result))
            for position in range(max(len(row["semantic_task"]["observations"]) for row in group)):
                result = lookup_baseline(group, lambda task,v=value,p=position: p<len(task["observations"]) and canonical(task["observations"][p]["equals"])==v)
                position_tests.append(dict(typed_value_json=value, observation_index=position, **result))
        strongest = lambda tests: max(tests, key=lambda item:(item["status_accuracy"], item["complete_task_accuracy"], item["query_accuracy"]))
        report[group_name] = dict(histograms={axis:{key:dict(counts) for key,counts in table.items()} for axis,table in histogram.items()},
            tested_single_scalar_rules=scalar_tests, strongest_tested_single_scalar=strongest(scalar_tests),
            tested_scalar_position_rules=position_tests, strongest_tested_scalar_position=strongest(position_tests),
            first_prefix_lookup=lookup_baseline(group,lambda task: ids(prefixes(task, task["observations"][:1])[-1])),
            coarse_joint_lookup=lookup_baseline(group,coarse),
            joint_input_lookup=lookup_baseline(group,lambda task: dict(worlds=sorted(task["worlds"],key=lambda world:world["world_id"]), observations=task["observations"])),
            stripped_ordered_scalar_lookup=lookup_baseline(group,lambda task:[obs["equals"] for obs in task["observations"]]),
            stripped_marker_linen_lookup=lookup_baseline(group,lambda task:[marker_predicts_empty(task),any(type(obs["equals"]) is str and obs["equals"]=="linen" for obs in task["observations"])]),
            majority_status_baseline=lookup_baseline(group,lambda task: "constant"))
    return dict(groups=report, feature_boundary="Baselines read only visible task; no gold final support/flags, roles, context, split or property metadata as predictive input.",
                label_fit_boundary="In-sample majority labels/templates are explicitly fitted from this finite corpus. Exact conjunction/table input lookup may memorize it; no model or generalization conclusion.")


def split_transfer_audit(rows):
    train=[row for row in rows if row["split"]=="train"]
    dev=[row for row in rows if row["split"]=="dev_challenge"]
    scalar_values=sorted({canonical(obs["equals"]) for row in train for obs in row["semantic_task"]["observations"]})
    tests=[]
    for value in scalar_values:
        feature=lambda task,v=value:any(canonical(obs["equals"])==v for obs in task["observations"])
        tests.append(dict(name="single_scalar_anywhere",typed_value_json=value,train_fit=lookup_baseline(train,feature),
                          exposed_dev_transfer=lookup_baseline(dev,feature,train)))
        for position in range(3):
            feature=lambda task,v=value,p=position:p<len(task["observations"]) and canonical(task["observations"][p]["equals"])==v
            tests.append(dict(name="scalar_position",typed_value_json=value,observation_index=position,train_fit=lookup_baseline(train,feature),
                              exposed_dev_transfer=lookup_baseline(dev,feature,train)))
    for name,feature in (("first_prefix_exact_ids",lambda task:ids(prefixes(task,task["observations"][:1])[-1])),
                         ("first_prefix_count_and_coarse_counts",lambda task:{k:v for k,v in coarse(task).items() if k!="first_prefix_ids"}),
                         ("coarse_exact_ids_joint",coarse),
                         ("stripped_ordered_scalars",lambda task:[obs["equals"] for obs in task["observations"]]),
                         ("stripped_marker_linen_pair",lambda task:[marker_predicts_empty(task),any(type(obs["equals"]) is str and obs["equals"]=="linen" for obs in task["observations"])]),
                         ("joint_world_table_observations",lambda task:dict(worlds=sorted(task["worlds"],key=lambda world:world["world_id"]),observations=task["observations"]))):
        tests.append(dict(name=name,train_fit=lookup_baseline(train,feature),exposed_dev_transfer=lookup_baseline(dev,feature,train)))
    # Select using train accuracy only, avoiding selection on development outcomes.
    strongest=max(tests,key=lambda item:(item["train_fit"]["status_accuracy"],item["train_fit"]["complete_task_accuracy"],item["train_fit"]["query_accuracy"]))
    return dict(train_records=len(train),exposed_dev_records=len(dev),tested_lookups=tests,strongest_by_train_fit=strongest,
                selection_rule="Maximum train status accuracy, then train complete-task/query accuracy; no dev selection.",
                transfer_limit="Diagnostic split transfer across the same four public motifs, not sealed OOD or actual learned model efficacy.")


def pair_evidence(left, right, name, input_hash):
    a,b = left["semantic_task"],right["semantic_task"]
    assert equal(a["worlds"],b["worlds"]) and equal(a["queries"],b["queries"])
    assert len(a["observations"])==len(b["observations"]) and equal(a["observations"][0],b["observations"][0])
    differences=[]
    for index,(x,y) in enumerate(zip(a["observations"],b["observations"])):
        for key in ("fact","equals"):
            if not equal(x[key],y[key]):differences.append(f"observations[{index}].{key}")
    assert differences==["observations[1].equals"]
    ap,bp=prefixes(a),prefixes(b)
    assert ids(ap[1])==ids(bp[1]) and not ap[-1] and bp[-1]
    return dict(name=name,scope="corpus_variant",source_family_key=left["source_family_key"],
                control_id=f"{left['source_family_index']}-{left['controlled_context']}-{right['controlled_context']}-s{left['schedule_index']:02d}-{left['renderer']}",
                pair_sample_ids=[left["sample_id"],right["sample_id"]],exact_candidate_sha256=input_hash,
                changed_fields=differences,observation_length=len(a["observations"]),
                typed_first_observation=a["observations"][0],first_prefix_survivor_ids=ids(ap[1]),
                left_prefix_survivor_ids=[ids(worlds) for worlds in ap],right_prefix_survivor_ids=[ids(worlds) for worlds in bp],
                left_prefix_counts=[len(worlds) for worlds in ap],right_prefix_counts=[len(worlds) for worlds in bp],
                left_complete_query_payloads=left["verified_target"]["queries"],right_complete_query_payloads=right["verified_target"]["queries"])


def supplied_and_probes(rows,input_hash):
    pairs,probes=own_v1.family_evidence(rows)
    indexed={(row["source_family_index"],row["controlled_context"],row["schedule_index"],row["renderer"]):row for row in rows}
    matched=[]
    for index in PROFILES:
        for schedule,style in product(range(27),("forward","reverse")):
            for left,right in (("E","G"),("F","H")):
                a,b=indexed[index,left,schedule,style],indexed[index,right,schedule,style]
                for name in ("same_prefix_empty_nonempty_counterfactual","contradiction_equals_mixed_outcomes"):
                    matched.append(pair_evidence(a,b,name,input_hash))
        g,h=indexed[index,"G",0,"forward"],indexed[index,"H",0,"forward"]
        assert equal(g["semantic_task"]["observations"]+[g["semantic_task"]["observations"][0]],h["semantic_task"]["observations"])
        assert equal(g["verified_target"]["queries"],h["verified_target"]["queries"]) and equal(g["verified_target"]["surviving_worlds"],h["verified_target"]["surviving_worlds"])
        pairs.append(dict(name="nonempty_repetition_final_invariance",scope="corpus_variant",source_family_key=g["source_family_key"],
                          control_id=f"{index}-G-H-repetition",pair_sample_ids=[g["sample_id"],h["sample_id"]]))
    return pairs,probes,matched


def claims(index,source,context,schedule,style,batch):
    evidence=f"data/RBM-001/manifests/{batch}.family_controls.json"
    controls=list(own_v1.CORE_CONTROLS)
    if index in (13,18,19):controls.append("typed_equals_counterfactual")
    if index in (12,18):controls.append("null_guarantee_contrast")
    result=[dict(name=name,scope="corpus_variant",source_family_key=source,evidence_manifest=evidence,control_id=f"{index}-{name}") for name in controls]
    for left,right in (("E","G"),("F","H")):
        for name in ("same_prefix_empty_nonempty_counterfactual","contradiction_equals_mixed_outcomes"):
            pair=[f"RBM-001-{batch}-p{index:04d}-{c}-s{schedule:02d}-{style}" for c in (left,right)]
            result.append(dict(name=name,scope="corpus_variant",source_family_key=source,evidence_manifest=evidence,
                               control_id=f"{index}-{left}-{right}-s{schedule:02d}-{style}",pair_sample_ids=pair))
    result += [dict(name=name,scope="audit_probe",source_family_key=source,evidence_manifest=evidence,probe_id=f"{index}-{name}")
               for name in ("world_order_invariance","world_id_rename_equivariance")]
    return result


def build_rows(batch):
    rows,parents,graphs,base_metrics=[],[],{},[]
    calls=0
    for index,profile in PROFILES.items():
        split=split_for(index) # Assigned before contexts/views.
        worlds,contexts,names,auxiliaries,query_ids=base_family(index)
        graphs[str(index)]=check_contexts(worlds,contexts,profile)
        source=f"RBM-seed{SEED}-p{index:04d}"
        family=f"RBM-001-{batch}-p{index:04d}"
        tasks,keys=[],[]
        for context,observations in contexts.items():
            for u,v,w in product(range(3),repeat=3):
                schedule=9*u+3*v+w
                task,role_names,assignments=own_v1.make_task(worlds,observations,names,auxiliaries,query_ids,u,v,w)
                target=solve(task);calls+=1
                witnesses=own_v1.witness_locators(task)
                level=difficulty(task,witnesses)
                tasks.append(task);keys.append(f"{context}-s{schedule:02d}")
                for style in ("forward","reverse"):
                    payload=task if style=="forward" else own_v1.reorder(task)
                    prompt="Executable task contract:\n"+contracts()["belief"]+"Task JSON:\n"+json.dumps(payload,ensure_ascii=False,separators=(",",":"))
                    row=record("RBM-001",batch,index,f"{context}-s{schedule:02d}",task,prompt,target,
                               ["finite_synthetic_proxy","finite_noiseless_filter","possible_value_partition"]+sorted(witnesses),level,"structured",style)
                    row.update(generator_version=GENERATOR_VERSION,controlled_context=context,structural_family_profile=profile,source_family_index=index,
                               source_family_key=source,id_shift=u,position_shift=v,fact_name_shift=w,schedule_index=schedule,
                               query_id_pool=query_ids,queried_fact_name_pool=names,fact_role_assignments=role_names,query_role_assignments=assignments,
                               property_witnesses=witnesses,structural_difficulty=level,family_control_claims=claims(index,source,context,schedule,style,batch),
                               producer_role="belief_producer",producer_agent_id="/root/relational_critic_producer",
                               scope_limitation="Targeted contradiction scalar control; four existing independent families, correlated contexts/views; residual finite-input lookup possible; no OOD/efficacy.")
                    row["record_hash"]=record_hash(row);rows.append(row)
                    if schedule==0 and style=="forward":base_metrics.append(dict(source_family_key=source,context=context,metrics=metrics(task),difficulty=level))
        parents.append(dict(semantic_parent_id=family,semantic_family_id=family,source_family_key=source,source_family_index=index,split=split,
                            structural_family_profile=profile,tasks=tasks,task_variant_keys=keys,contexts=8,correlated_schedules_per_context=27,
                            field_order_renderers=2,source_family_count=1,new_source_families_relative_to_v1=0,generator_version=GENERATOR_VERSION))
    return rows,parents,graphs,base_metrics,calls


def frozen_marker_path(batch):
    return ROOT/f"manifests/{batch}_MARKER_RULE.json"


def freeze_marker(batch):
    path=frozen_marker_path(batch)
    if path.exists():
        assert equal(json.loads(path.read_text()),MARKER_RULE)
    else:write_json(path,MARKER_RULE)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(batch="belief-gap-v2",seed=SEED):
    if seed!=SEED or not batch or "/" in batch or "\\" in batch:raise ValueError("Use frozen seed and fresh filename-safe batch")
    started=time.perf_counter()
    candidate=ROOT/f"data/RBM-001/candidate_renderings/{batch}.jsonl"
    parent=ROOT/f"data/RBM-001/semantic_parents/{batch}.jsonl"
    manifest=ROOT/f"manifests/{batch}_GENERATION.json"
    snapshot=ROOT/f"manifests/{batch}_SOURCE.py"
    integrity=ROOT/f"data/RBM-001/manifests/{batch}.producer_integrity.json"
    controls=ROOT/f"data/RBM-001/manifests/{batch}.family_controls.json"
    shortcut=ROOT/f"data/RBM-001/manifests/{batch}.shortcut_audit.json"
    for path in (candidate,parent,manifest,snapshot,integrity,controls,shortcut):
        if path.exists():raise ValueError(f"First run retained at {path}; never regenerate this batch")
    public={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PUBLIC_HASHES}
    assert public==PUBLIC_HASHES and hashlib.sha256((ROOT/ROOT_FREEZE).read_bytes()).hexdigest()==ROOT_FREEZE_SHA
    assert hashlib.sha256(Path(own_v1.__file__).read_bytes()).hexdigest()==V1_SOURCE_SHA
    old=ROOT/"data/RBM-001/candidate_renderings/belief-gap-v1.jsonl"
    assert hashlib.sha256(old.read_bytes()).hexdigest()==V1_CANDIDATE_SHA
    marker=frozen_marker_path(batch)
    assert marker.exists(),"Communicate frozen marker rule hash before generating"
    assert equal(json.loads(marker.read_text()),MARKER_RULE)
    source=Path(__file__).read_text();atomic(snapshot,source)
    rows,parents,graphs,base_metrics,calls=[],[],{},[],0
    status,failure="PRODUCER_CHECK_FAILED",None
    try:
        rows,parents,graphs,base_metrics,calls=build_rows(batch)
        write_jsonl(parent,parents);write_jsonl(candidate,rows) # First actual candidate frozen before checks.
        candidate_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
        report=own_v1.validate_rows(rows)
        crossing=own_v1.crossing_report(rows)
        assert len(rows)==1728 and len(parents)==4 and len(crossing)==64
        assert len({graph["signature_sha256"] for graph in graphs.values()})==4
        for row in rows:
            assert row["generator_version"]==GENERATOR_VERSION
            assert row["difficulty"]==row["structural_difficulty"]==difficulty(row["semantic_task"],row["property_witnesses"])
        pairs,probes,matched=supplied_and_probes(rows,candidate_sha);calls+=len(probes)
        audits=scalar_and_lookup_audit(rows)
        transfer=split_transfer_audit(rows)
        root_rules=root_rules_audit(rows)
        old_rows=load_jsonl(old)
        old_marker,new_marker=marker_audit(old_rows),marker_audit(rows)
        assert old_marker["overall"]["status_errors"]==0
        assert new_marker["overall"]["status_errors"]>0
        for index in PROFILES:
            for length in (2,3):
                group=f"family_length:RBM-seed{SEED}-p{index:04d}:{length}"
                assert new_marker[group]["status_errors"]>0
                assert any(len(table)==2 for table in audits["groups"][group]["histograms"]["coarse_joint"].values())
        assert hashlib.sha256(old.read_bytes()).hexdigest()==V1_CANDIDATE_SHA
        report.update(source_families=4,new_source_families_relative_to_v1=0,base_contexts=32,base_query_observations=96,
                      rotation_tasks=864,rotation_query_observations=2592,rendered_query_observations=5184,
                      split_records=dict(Counter(row["split"] for row in rows)),split_source_families=dict(Counter(p["split"] for p in parents)),
                      status_counts=dict(Counter("empty" if not row["verified_target"]["surviving_worlds"] else "nonempty" for row in rows)),
                      difficulty_counts=dict(Counter(row["difficulty"] for row in rows)),structural_profiles=graphs,base_context_metrics=base_metrics,
                      actual_query_crossing_histogram_sets=len(crossing),matched_pair_evidence_count=len(matched),
                      supplied_pair_counts=dict(Counter(item["name"] for item in pairs+matched)),executed_probe_counts=dict(Counter(item["name"] for item in probes)),
                      actual_v1_input_changes="A-E unchanged; F observation index2 becomes exact repeat of its own first; G/H added; truthful difficulty/provenance recomputed.",
                      v1_candidate_sha256_preserved=V1_CANDIDATE_SHA,producer_checks_passed=True,
                      shortcut_diagnostic_stage="FAILED_RESIDUAL_FROZEN_MARKER_AND_LINEN_RULE" if root_rules["nonfitted_frozen_root_rules"]["prospective_marker_and_auxiliary_token_empty"]["overall"]["status_accuracy"]==1 else "MEASURED",
                      limitations=["Four original independent families; added G/H and all rotations/renderings are correlated, zero added families.",
                          "G/H second filter is non-strict; branching/path G/H are singleton controls, measured level2.",
                          "Targeted stone/unknown marker broken; class imbalance and finite joint-input lookup remain; no complete shortcut removal.",
                          "Cyclic query order, auxiliary names/world-ID styling not independently crossed; exposed development, no sealed OOD or efficacy."])
        write_json(integrity,report)
        write_json(controls,dict(exact_candidate_sha256=candidate_sha,actual_crossing_histograms=crossing,supplied_pairs=pairs,
                                matched_empty_nonempty_pair_evidence=matched,executed_audit_probes=probes,field_order_pair_count=864,
                                probes_add_corpus_records=0,probes_add_source_families=0))
        write_json(shortcut,dict(exact_candidate_sha256=candidate_sha,frozen_marker_rule_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                                producer_visible_first_prefix_marker_v1=old_marker,producer_visible_first_prefix_marker_v2=new_marker,
                                frozen_root_diagnostics_sha256=PUBLIC_HASHES["schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json"],
                                root_world_count_only_v1_results=root_rules_audit(old_rows),root_world_count_only_v2_results=root_rules,
                                train_derived_exposed_dev_transfer=transfer,**audits,
                                bounded_decision="Preserve actual first v2 PENDING evidence and failed residual shortcut stage; no approval or silent scalar/table adjustment. Further scalar balancing requires another public revision/batch."))
        status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW"
    except Exception:
        failure=traceback.format_exc()
        if not parent.exists():write_jsonl(parent,parents)
        if not candidate.exists():write_jsonl(candidate,rows)
        raise
    finally:
        artifacts=[p for p in (candidate,parent,snapshot,integrity,controls,shortcut,marker) if p.exists()]
        write_json(manifest,dict(batch=batch,track_id="RBM-001",generator_version=GENERATOR_VERSION,status=status,seed=seed,
            producer_agent_id="/root/relational_critic_producer",producer_role="belief_producer",source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            public_input_sha256=public,root_freeze_sha256=ROOT_FREEZE_SHA,
            dependency_sha256={"generators/other_tracks/belief_gap.py":V1_SOURCE_SHA,
                               "generators/other_tracks/reference.py":hashlib.sha256((ROOT/"generators/other_tracks/reference.py").read_bytes()).hexdigest(),
                               "factory/common.py":hashlib.sha256((ROOT/"factory/common.py").read_bytes()).hexdigest()},
            frozen_marker_rule_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
            reproduction=f"/opt/anaconda3/bin/python3 -m generators.other_tracks.belief_gap_v2 --freeze-marker --batch {batch}; /opt/anaconda3/bin/python3 -m generators.other_tracks.belief_gap_v2 --batch {batch}",
            config=dict(source_family_indices=list(PROFILES),profiles=PROFILES,contexts=list("ABCDEFGH"),id_shifts=[0,1,2],position_shifts=[0,1,2],fact_shifts=[0,1,2],field_orders=["forward","reverse"]),
            actual_counts=dict(source_families=len(parents),new_source_families=0,records=len(rows),reference_solve_calls=calls),
            cost=dict(wall_seconds=time.perf_counter()-started,world_observation_comparisons=sum(len(row["semantic_task"]["worlds"])*len(row["semantic_task"]["observations"]) for row in rows if row["renderer"]=="forward"),
                      output_bytes=sum(p.stat().st_size for p in artifacts)),
            artifact_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},initial_run_failure=failure,
            shortcut_stage="FAILED_RESIDUAL_FROZEN_MARKER_AND_LINEN_RULE" if shortcut.exists() and json.loads(shortcut.read_text())["root_world_count_only_v2_results"]["nonfitted_frozen_root_rules"]["prospective_marker_and_auxiliary_token_empty"]["overall"]["status_accuracy"]==1 else "UNKNOWN_OR_MEASURED",
            preserved_v1_candidate_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),no_approval_release_or_push=True))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",default="belief-gap-v2")
    parser.add_argument("--freeze-marker",action="store_true")
    args=parser.parse_args()
    if args.freeze_marker:print("Frozen original marker rule SHA256",freeze_marker(args.batch))
    else:print(f"Produced {len(generate(args.batch))} PENDING rows in {args.batch}")
