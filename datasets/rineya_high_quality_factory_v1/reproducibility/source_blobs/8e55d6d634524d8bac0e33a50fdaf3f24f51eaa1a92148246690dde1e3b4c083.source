"""Public V3 producer: fact-local auxiliary views; no reviewer/oracle imports."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record, record_hash, split_for, write_jsonl
from generators.other_tracks.reference import contracts, solve
from generators.other_tracks import belief_gap_v2 as own_v2

ACTUAL_EXECUTION = "/root/relational_critic_producer"
VERSION = "rbm-belief-gap-3.0.0"
PROFILES = own_v2.PROFILES
VARIANTS = ("identity", "transposed")
PUBLIC_HASHES = {**own_v2.PUBLIC_HASHES,
    "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md":"3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11",
    "research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md":"7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33",
    "research/curricula/RBM-EVIDENCE-GAP-REVISION-V3-EXECUTION.json":"1d27ef32e6f58cb5adb6fba5842e5bfc809341c7cf400cbd75207d8a75c575c0",
    "feedback/improvement_requests/RBM-REPRO-010.json":"6cd38819a69ef331c6ba32311c87f8b079e64f6e84f27d2d6380f8e377279643"}
ROOT_FREEZE = "research/curricula/RBM-EVIDENCE-GAP-REVISION-V3-ROOT_FREEZE.json"
ROOT_FREEZE_SHA = "56ea87bfb614360a9091d6c835f2a83fa918f1946de612f5b723548a02782769"
PRIOR_HASHES = {
    "generators/other_tracks/belief_gap.py":own_v2.V1_SOURCE_SHA,
    "generators/other_tracks/belief_gap_v2.py":"dbfc7d23b4f57f9c61286b3e0f9046c1304445f1a4a706148a635eb9f26e33a1",
    "data/RBM-001/candidate_renderings/belief-gap-v1.jsonl":own_v2.V1_CANDIDATE_SHA,
    "data/RBM-001/candidate_renderings/belief-gap-v2.jsonl":"589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df",
    "data/RBM-001/manifests/belief-gap-v2.shortcut_audit.json":"3990815f282938db69d7cda1c90794219322357e6c473c34639a024523cc218f"}
THRESHOLD = 25 * 1024 * 1024
equal, prefixes, ids, bins = own_v2.equal, own_v2.prefixes, own_v2.ids, own_v2.bins


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_json(path, value):
    # Canonical serialization sorts ALL keys, including set-derived histogram keys.
    atomic(path, canonical(value)+"\n")


def transposition(task, control):
    result = copy.deepcopy(task)
    fact, baseline, supported = (control[key] for key in ("fact", "baseline_value", "supported_value"))
    assert type(baseline) is str and type(supported) is str and baseline != supported
    assert fact not in {query["fact"] for query in task.get("queries", [])}
    def swap(value):
        return supported if equal(value, baseline) else baseline if equal(value, supported) else value
    for world in result["worlds"]:
        world["facts"][fact] = swap(world["facts"][fact])
    for obs in result["observations"]:
        if obs["fact"] == fact: obs["equals"] = swap(obs["equals"])
    return result


def sample_id(batch, index, context, variant, schedule, style):
    return f"RBM-001-{batch}-p{index:04d}-{context}-{variant}-s{schedule:02d}-{style}"


def family_claims(batch, index, source, context, variant, schedule, style):
    evidence = f"data/RBM-001/manifests/{batch}.family_controls.json"
    names = list(own_v2.own_v1.CORE_CONTROLS)
    if index in (13,18,19): names.append("typed_equals_counterfactual")
    if index in (12,18): names.append("null_guarantee_contrast")
    names += ["same_prefix_empty_nonempty_counterfactual", "contradiction_equals_mixed_outcomes",
              "context_auxiliary_scalar_variant_crossing"]
    claims = [dict(name=name,scope="corpus_variant",source_family_key=source,evidence_manifest=evidence,
                   control_id=f"{index}-{variant}-{name}") for name in names]
    claims.append(dict(name="auxiliary_scalar_transposition_invariance",scope="corpus_variant",source_family_key=source,
        evidence_manifest=evidence,control_id=f"{index}-{context}-s{schedule:02d}-{style}-scalar",
        pair_sample_ids=[sample_id(batch,index,context,v,schedule,style) for v in VARIANTS]))
    if context in "EFGH":
        negative,positive = ("E","G") if context in "EG" else ("F","H")
        negative_variant = variant if context in "EF" else ("transposed" if variant=="identity" else "identity")
        positive_variant = "transposed" if negative_variant=="identity" else "identity"
        pair = [sample_id(batch,index,negative,negative_variant,schedule,style),sample_id(batch,index,positive,positive_variant,schedule,style)]
        claims.append(dict(name="same_observations_auxiliary_table_counterfactual",scope="corpus_variant",source_family_key=source,
            evidence_manifest=evidence,control_id=f"{index}-{negative}-{negative_variant}-{positive}-{positive_variant}-s{schedule:02d}-{style}",pair_sample_ids=pair))
    else:
        claims.append(dict(name="same_observations_auxiliary_table_counterfactual",scope="corpus_variant",source_family_key=source,
                           evidence_manifest=evidence,control_id=f"{index}-all-four-cross-table-pair-types"))
    claims += [dict(name=name,scope="audit_probe",source_family_key=source,evidence_manifest=evidence,probe_id=f"{index}-{variant}-{name}")
               for name in ("world_order_invariance","world_id_rename_equivariance")]
    return claims


def build_rows(batch):
    rows, parents, profiles, base_metrics = [], [], {}, []
    calls = 0
    for index, profile in PROFILES.items():
        split = split_for(index) # Before view/context/render construction.
        worlds, contexts, names, auxiliaries, query_ids = own_v2.base_family(index)
        profiles[str(index)] = own_v2.check_contexts(worlds,contexts,profile)
        baseline,supported = contexts["E"][1]["equals"],contexts["G"][1]["equals"]
        assert baseline=="linen" and type(supported) is str and supported not in {"linen","stone","unknown"}
        decoded_control=dict(fact="observation_auxiliary",baseline_value=baseline,supported_value=supported)
        source=f"RBM-seed{own_v2.SEED}-p{index:04d}"
        family=f"RBM-001-{batch}-p{index:04d}"
        tasks,keys=[],[]
        for scalar_variant in VARIANTS:
            for context, observations in contexts.items():
                decoded=dict(kind="belief",worlds=worlds,observations=observations,queries=[])
                if scalar_variant=="transposed":decoded=transposition(decoded,decoded_control)
                for u,v,w in product(range(3),repeat=3):
                    schedule=9*u+3*v+w
                    task,role_names,assignments=own_v2.own_v1.make_task(decoded["worlds"],decoded["observations"],names,auxiliaries,query_ids,u,v,w)
                    target=solve(task);calls+=1
                    witnesses=own_v2.own_v1.witness_locators(task)
                    level=own_v2.difficulty(task,witnesses)
                    control=dict(fact=role_names["observation_auxiliary"],baseline_value=baseline,supported_value=supported)
                    variant_key=f"{context}-{scalar_variant}-s{schedule:02d}"
                    tasks.append(task);keys.append(variant_key)
                    for style in ("forward","reverse"):
                        payload=task if style=="forward" else own_v2.own_v1.reorder(task)
                        prompt="Executable task contract:\n"+contracts()["belief"]+"Task JSON:\n"+json.dumps(payload,ensure_ascii=False,separators=(",",":"))
                        row=record("RBM-001",batch,index,variant_key,task,prompt,target,
                            ["finite_synthetic_proxy","finite_noiseless_filter","possible_value_partition"]+sorted(witnesses),level,"structured",style)
                        row.update(generator_version=VERSION,controlled_context=context,auxiliary_scalar_variant=scalar_variant,
                            auxiliary_scalar_control=control,structural_family_profile=profile,source_family_index=index,source_family_key=source,
                            id_shift=u,position_shift=v,fact_name_shift=w,schedule_index=schedule,query_id_pool=query_ids,queried_fact_name_pool=names,
                            fact_role_assignments=role_names,query_role_assignments=assignments,property_witnesses=witnesses,structural_difficulty=level,
                            family_control_claims=family_claims(batch,index,source,context,scalar_variant,schedule,style),
                            producer_role="belief_producer",producer_agent_id=ACTUAL_EXECUTION,
                            scope_limitation="Four existing finite lineage families; correlated auxiliary-label/context/query/style views; no complete shortcut removal, sealed OOD or efficacy.")
                        row["record_hash"]=record_hash(row);rows.append(row)
                        if schedule==0 and style=="forward":base_metrics.append(dict(source_family_key=source,context=context,auxiliary_scalar_variant=scalar_variant,
                                                                                    difficulty=level,metrics=own_v2.metrics(task)))
        parents.append(dict(semantic_parent_id=family,semantic_family_id=family,source_family_key=source,source_family_index=index,split=split,
            structural_family_profile=profile,tasks=tasks,task_variant_keys=keys,contexts=8,auxiliary_scalar_variants=list(VARIANTS),
            scalar_view_cases=16,correlated_schedules_per_case=27,field_order_renderers=2,source_family_count=1,new_source_families=0,generator_version=VERSION))
    return rows,parents,profiles,base_metrics,calls


def conditional_crossing(rows):
    reports=[]
    for variant in VARIANTS:
        subset=[row for row in rows if row["auxiliary_scalar_variant"]==variant]
        measured=own_v2.own_v1.crossing_report(subset)
        reports.extend(dict(auxiliary_scalar_variant=variant,**report) for report in measured)
    assert len(reports)==128
    return reports


def validate_rows(rows):
    for variant in VARIANTS:own_v2.own_v1.validate_rows([row for row in rows if row["auxiliary_scalar_variant"]==variant])
    for row in rows:
        assert row["generator_version"]==VERSION
        control=row["auxiliary_scalar_control"]
        assert set(control)=={"fact","baseline_value","supported_value"}
        assert control["fact"]==row["fact_role_assignments"]["observation_auxiliary"]
        task=row["semantic_task"]
        assert control["fact"] not in {query["fact"] for query in task["queries"]}
        assert equal(transposition(transposition(task,control),control),task)
        assert row["difficulty"]==own_v2.difficulty(task,row["property_witnesses"])
        assert "auxiliary_scalar_variant" not in row["rendered_input"] and "auxiliary_scalar_control" not in row["rendered_input"]
    return dict(rendered_records=len(rows),distinct_task_jsons=len({digest(row["semantic_task"]) for row in rows}),
                distinct_prompts=len({row["rendered_input"] for row in rows}),field_order_pairs=len(rows)//2,
                property_row_counts=dict(Counter(name for row in rows for name in row["property_witnesses"])),
                status_counts=dict(Counter("empty" if not row["verified_target"]["surviving_worlds"] else "nonempty" for row in rows)),
                split_records=dict(Counter(row["split"] for row in rows)),difficulty_counts=dict(Counter(row["difficulty"] for row in rows)))


def field_pairs(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["source_family_key"],row["controlled_context"],row["auxiliary_scalar_variant"],row["schedule_index"]].append(row)
    result=[]
    for key,pair in sorted(groups.items()):
        a,b=sorted(pair,key=lambda row:row["renderer"])
        assert len(pair)==2 and equal(a["semantic_task"],b["semantic_task"]) and equal(a["verified_target"],b["verified_target"])
        result.append(dict(pair_sample_ids=[a["sample_id"],b["sample_id"]],source_family_key=key[0],context=key[1],auxiliary_scalar_variant=key[2],
            schedule_index=key[3],typed_task_sha256=digest(a["semantic_task"]),complete_target_sha256=digest(a["verified_target"])))
    assert len(result)==1728
    return result


def prefix_evidence(row):
    supports=prefixes(row["semantic_task"])
    return dict(prefix_survivor_ids=[ids(s) for s in supports],prefix_counts=[len(s) for s in supports],complete_target=row["verified_target"])


def pair_controls(rows,input_sha):
    indexed={(row["source_family_index"],row["controlled_context"],row["auxiliary_scalar_variant"],row["schedule_index"],row["renderer"]):row for row in rows}
    scalar,same_observations=[],[]
    for index,context,schedule,style in product(PROFILES,"ABCDEFGH",range(27),("forward","reverse")):
        left,right=(indexed[index,context,variant,schedule,style] for variant in VARIANTS)
        a,b=left["semantic_task"],right["semantic_task"]
        assert equal(transposition(a,left["auxiliary_scalar_control"]),b) and not equal(a,b)
        assert equal(left["verified_target"],right["verified_target"])
        assert [ids(s) for s in prefixes(a)]==[ids(s) for s in prefixes(b)]
        scalar.append(dict(name="auxiliary_scalar_transposition_invariance",scope="corpus_variant",source_family_key=left["source_family_key"],
            control_id=f"{index}-{context}-s{schedule:02d}-{style}-scalar",pair_sample_ids=[left["sample_id"],right["sample_id"]],
            candidate_sha256=input_sha,auxiliary_scalar_control=left["auxiliary_scalar_control"],involution_verified=True,
            exact_changed_world_column=left["auxiliary_scalar_control"]["fact"],prefix_and_target_evidence=prefix_evidence(left)))
    for index,schedule,style,negative_variant,(negative,positive) in product(PROFILES,range(27),("forward","reverse"),VARIANTS,(("E","G"),("F","H"))):
        positive_variant="transposed" if negative_variant=="identity" else "identity"
        left,right=indexed[index,negative,negative_variant,schedule,style],indexed[index,positive,positive_variant,schedule,style]
        a,b=left["semantic_task"],right["semantic_task"]
        assert equal(a["observations"],b["observations"]) and equal(a["queries"],b["queries"])
        # The world table, rather than observations, is transposed in these pairs.
        world_only=transposition(dict(kind="belief",worlds=a["worlds"],observations=[],queries=a["queries"]),left["auxiliary_scalar_control"])
        assert equal(world_only["worlds"],b["worlds"]) and not equal(a["worlds"],b["worlds"])
        assert ids(prefixes(a)[1])==ids(prefixes(b)[1])
        assert not left["verified_target"]["surviving_worlds"] and right["verified_target"]["surviving_worlds"]
        same_observations.append(dict(name="same_observations_auxiliary_table_counterfactual",scope="corpus_variant",source_family_key=left["source_family_key"],
            control_id=f"{index}-{negative}-{negative_variant}-{positive}-{positive_variant}-s{schedule:02d}-{style}",
            pair_sample_ids=[left["sample_id"],right["sample_id"]],candidate_sha256=input_sha,observation_length=len(a["observations"]),
            ordered_identical_observations=a["observations"],ordered_identical_queries=a["queries"],auxiliary_scalar_control=left["auxiliary_scalar_control"],
            left=prefix_evidence(left),right=prefix_evidence(right)))
    assert len(scalar)==1728 and len(same_observations)==864
    paired=[sample for pair in same_observations for sample in pair["pair_sample_ids"]]
    assert len(set(paired))==1728
    assert set(paired)=={row["sample_id"] for row in rows if row["controlled_context"] in "EFGH"}
    return scalar,same_observations


def original_controls(rows,input_sha):
    supplied,probes,matched=[],[],[]
    for variant in VARIANTS:
        subset=[row for row in rows if row["auxiliary_scalar_variant"]==variant]
        pairs,audits,matches=own_v2.supplied_and_probes(subset,input_sha)
        for item in pairs+matches:
            item["auxiliary_scalar_variant"]=variant
            item["control_id"]=f"{item['source_family_key']}-{variant}-{item.get('control_id',item['name'])}"
        for item in audits:
            item["auxiliary_scalar_variant"]=variant
            item["probe_id"]=f"{item['source_family_key'].split('-p')[-1].lstrip('0')}-{variant}-{item['name']}"
        supplied+=pairs;probes+=audits;matched+=matches
    return supplied,probes,matched


def grouping_keys(row):
    family,split,variant,context=row["source_family_key"],row["split"],row["auxiliary_scalar_variant"],row["controlled_context"]
    length=len(row["semantic_task"]["observations"])
    return ("overall",f"family:{family}",f"split:{split}",f"variant:{variant}",f"context:{context}",f"context_variant:{context}:{variant}",
            f"family_length:{family}:{length}",f"split_length:{split}:{length}",
            f"family_variant_length:{family}:{variant}:{length}",f"split_variant_length:{split}:{variant}:{length}")


def table_features(task):
    return dict(column_typed_frequencies={fact:dict(sorted(bins(task["worlds"],fact).items())) for fact in sorted(task["worlds"][0]["facts"])},
                column_scalar_character_lengths=sorted(sum(len(canonical(world["facts"][fact])) for world in task["worlds"]) for fact in task["worlds"][0]["facts"]),
                observation_scalar_character_lengths=[len(canonical(obs["equals"])) for obs in task["observations"]],
                observation_full_table_value_counts=[sum(equal(world["facts"][obs["fact"]],obs["equals"]) for world in task["worlds"]) for obs in task["observations"]])


def conditional_histograms(rows):
    tables=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    for row in rows:
        task=row["semantic_task"]
        outcome="empty" if not row["verified_target"]["surviving_worlds"] else "nonempty"
        inverse={name:role for role,name in row["fact_role_assignments"].items()}
        entries=[]
        values=sorted({canonical(obs["equals"]) for obs in task["observations"]})
        # New-v3 explicit sorting fixes the public RBM-REPRO-010 set-order defect.
        for value in values:entries.append(("scalar_anywhere",canonical(value)))
        for pair in combinations(values,2):entries.append(("scalar_pair_anywhere",canonical(pair)))
        for position,obs in enumerate(task["observations"]):
            name,role,value=obs["fact"],inverse[obs["fact"]],canonical(obs["equals"])
            for axis,feature in (("scalar_position",[position,value]),("fact_role_position",[role,position]),
                                 ("fact_name_position",[name,position]),("fact_role_position_scalar",[role,position,value]),
                                 ("fact_name_position_scalar",[name,position,value])):
                entries.append((axis,canonical(feature)))
        for axis,features in (("coarse_joint",own_v2.coarse(task)),("table_joint",table_features(task))):
            entries.append((axis,canonical(features)))
            for name,value in sorted(features.items()):entries.append((axis+"_"+name,canonical(value)))
        for group in grouping_keys(row):
            for axis,key in entries:tables[group][axis][key][outcome]+=1
    return {group:{axis:{key:dict(sorted(counts.items())) for key,counts in sorted(table.items())} for axis,table in sorted(axes.items())}
            for group,axes in sorted(tables.items())}


def root_rule_audit(rows):
    groups=defaultdict(list)
    for row in rows:
        for group in grouping_keys(row):groups[group].append(row)
    reports={}
    for name,predicate in (("v1_original_scalar_marker_empty",own_v2.marker_predicts_empty),
                           ("prospective_marker_and_auxiliary_token_empty",own_v2.marker_and_linen)):
        reports[name]={}
        for group,subset in sorted(groups.items()):
            predictions=[predicate(row["semantic_task"]) for row in subset]
            reports[name][group]=own_v2.score_predictions(subset,predictions,[own_v2.root_world_count_empty_core(row["semantic_task"],predicate) for row in subset])
    return reports


def tested_lookups(rows):
    train=[row for row in rows if row["split"]=="train"]
    dev=[row for row in rows if row["split"]=="dev_challenge"]
    features=[]
    values=sorted({canonical(obs["equals"]) for row in rows for obs in row["semantic_task"]["observations"]})
    for value in values:
        features.append(("single_scalar_anywhere:"+value,lambda task,v=value:any(canonical(obs["equals"])==v for obs in task["observations"])))
        for position in range(3):features.append((f"scalar_position:{position}:{value}",lambda task,v=value,p=position:p<len(task["observations"]) and canonical(task["observations"][p]["equals"])==v))
    features += [("majority",lambda task:0),("first_prefix_exact_ids",lambda task:ids(prefixes(task,task["observations"][:1])[-1])),
                 ("coarse_joint",own_v2.coarse),("ordered_scalars",lambda task:[obs["equals"] for obs in task["observations"]]),
                 ("marker_linen_pair",lambda task:[own_v2.marker_predicts_empty(task),any(equal(obs["equals"],"linen") for obs in task["observations"])]),
                 ("observation_full_table_value_counts",lambda task:table_features(task)["observation_full_table_value_counts"]),
                 ("observation_character_lengths",lambda task:table_features(task)["observation_scalar_character_lengths"]),
                 ("table_column_frequency_and_length",table_features),
                 ("joint_world_table_observations",lambda task:dict(worlds=sorted(task["worlds"],key=lambda world:world["world_id"]),observations=task["observations"]))]
    results=[]
    for name,feature in features:
        results.append(dict(name=name,in_sample=own_v2.lookup_baseline(rows,feature),train_fit=own_v2.lookup_baseline(train,feature),
                            exposed_dev_transfer=own_v2.lookup_baseline(dev,feature,train)))
    # Deterministic tie breaking; do not select on exposed-development outcomes.
    best=lambda key:max(results,key=lambda item:(item[key]["status_accuracy"],item[key]["complete_task_accuracy"],item[key]["query_accuracy"],item["name"]))
    return dict(tested=results,strongest_in_sample=best("in_sample"),strongest_by_train_fit=best("train_fit"),
                feature_boundary="Predictions read visible tasks only; labels/templates fit in-sample or on train. No final gold flags/support or role/context/variant/split metadata as input.",
                selection_boundary="Maximum train status then complete-task/query accuracy, canonical name tie break; no dev selection.",
                transfer_boundary="Four exposed related families, not sealed OOD/model efficacy; unknown buckets use explicit fitted majority fallback/structured abstention.")


def archives(output,paths,batch):
    items=[]
    for path in paths:
        raw=path.read_bytes()
        if len(raw)<=THRESHOLD:continue
        relative=str(path.relative_to(output))
        target=output/f"data/RBM-001/archives/{batch}/{relative.replace('/','__')}.gz"
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+".tmp")
        with tmp.open("wb") as file:
            with gzip.GzipFile(filename="",mode="wb",fileobj=file,mtime=0,compresslevel=6) as compressor:compressor.write(raw)
            file.flush();os.fsync(file.fileno())
        tmp.replace(target)
        decoded=gzip.decompress(target.read_bytes())
        assert decoded==raw
        items.append(dict(raw_path=relative,raw_bytes=len(raw),raw_sha256=sha(path),archive_path=str(target.relative_to(output)),archive_bytes=target.stat().st_size,
                          archive_sha256=sha(target),decoded_bytes=len(decoded),decoded_sha256=hashlib.sha256(decoded).hexdigest(),roundtrip_exact=True))
    receipt=output/f"manifests/{batch}_ARCHIVES.json"
    stable_json(receipt,dict(actual_execution_id=ACTUAL_EXECUTION,actual_role="belief_producer",threshold_bytes=THRESHOLD,raw_bytes_preserved=True,
        gzip_parameters=dict(filename="",mtime=0,compresslevel=6),oversized_artifacts=items))
    return [output/item["archive_path"] for item in items]+[receipt]


def core_paths(output,batch):
    return [output/f"data/RBM-001/candidate_renderings/{batch}.jsonl",output/f"data/RBM-001/semantic_parents/{batch}.jsonl",
            output/f"manifests/{batch}_SOURCE.py",output/f"data/RBM-001/manifests/{batch}.producer_integrity.json",
            output/f"data/RBM-001/manifests/{batch}.family_controls.json",output/f"data/RBM-001/manifests/{batch}.shortcut_audit.json",
            output/f"manifests/{batch}_MARKER_RULES.json"]


def prior_own_artifacts():
    # Explicit own producer outputs only; no reviewer/oracle artifacts enumerated/read.
    paths=[]
    for batch in ("belief-gap-v1","belief-gap-v2"):
        paths += [ROOT/f"data/RBM-001/candidate_renderings/{batch}.jsonl",ROOT/f"data/RBM-001/semantic_parents/{batch}.jsonl",
                  ROOT/f"data/RBM-001/manifests/{batch}.producer_integrity.json",ROOT/f"data/RBM-001/manifests/{batch}.family_controls.json",
                  ROOT/f"manifests/{batch}_SOURCE.py",ROOT/f"manifests/{batch}_GENERATION.json"]
        if batch=="belief-gap-v2":paths += [ROOT/f"data/RBM-001/manifests/{batch}.shortcut_audit.json",ROOT/f"manifests/{batch}_MARKER_RULE.json"]
    paths += [ROOT/"generators/other_tracks/belief_gap.py",ROOT/"generators/other_tracks/belief_gap_v2.py"]
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}


def verify_reproduction(output,batch,paths):
    scratch=output/f"reproducibility/{batch}"
    processes=[]
    for seed in (17,991):
        child=scratch/f"hashseed-{seed}"
        assert not child.exists(),"Preserve first reproduction tree; do not overwrite"
        environment={**os.environ,"PYTHONHASHSEED":str(seed)}
        command=["/opt/anaconda3/bin/python3","-m","generators.other_tracks.belief_gap_v3","--batch",batch,"--output-root",str(child),"--worker"]
        processes.append((seed,child,subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)))
    executions=[]
    for seed,child,process in processes:
        stdout,stderr=process.communicate()
        executions.append(dict(hash_seed=seed,output_root=str(child.relative_to(output)),returncode=process.returncode,stdout=stdout,stderr=stderr))
    hashes=[]
    for path in paths:
        relative=str(path.relative_to(output))
        samples={"main":sha(path)}
        for seed,child,_ in processes:samples[str(seed)]=sha(child/relative) if (child/relative).exists() else None
        hashes.append(dict(path=relative,sha256_by_process=samples,all_bytes_identical=len(set(samples.values()))==1 and None not in samples.values()))
    passed=all(e["returncode"]==0 for e in executions) and all(item["all_bytes_identical"] for item in hashes)
    receipt=output/f"manifests/{batch}_REPRODUCTION.json"
    stable_json(receipt,dict(actual_execution_id=ACTUAL_EXECUTION,actual_role="belief_producer",separate_python_processes=True,python_hash_seeds=[17,991],
        isolated_scratch_trees=True,artifacts=hashes,executions=executions,passed=passed,
        excluded_nondeterministic_artifacts=[f"manifests/{batch}_GENERATION.json (measured wall_seconds / worker execution purpose)"],
        independent_agent_execution=False,scope="Determinism regression by same producer, not independent oracle or review."))
    assert passed,"First failed reproduction evidence preserved"
    return receipt


def generate(batch="belief-gap-v3",output=ROOT,worker=False):
    output=Path(output).resolve()
    if not batch or "/" in batch or "\\" in batch:raise ValueError("Fresh filename-safe batch required")
    started=time.perf_counter()
    paths=core_paths(output,batch)
    candidate,parent,snapshot,integrity,controls,shortcut,marker=paths
    manifest=output/f"manifests/{batch}_GENERATION.json"
    for path in paths+[manifest]:
        if path.exists():raise ValueError(f"First candidate/run preserved at {path}; use a new batch")
    public={name:sha(ROOT/name) for name in PUBLIC_HASHES}
    assert public==PUBLIC_HASHES and sha(ROOT/ROOT_FREEZE)==ROOT_FREEZE_SHA
    assert {name:sha(ROOT/name) for name in PRIOR_HASHES}==PRIOR_HASHES
    own_prior=prior_own_artifacts()
    source=Path(__file__).read_text();atomic(snapshot,source)
    # Exact public predicates were frozen before all v2/v3 samples.
    stable_json(marker,dict(public_diagnostic_sha256=PUBLIC_HASHES["schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json"],
        public_diagnostic=json.loads((ROOT/"schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json").read_text()),
        actual_execution_id=ACTUAL_EXECUTION,evaluated_without_updating=True))
    rows,parents,profiles,base_metrics,calls=[],[],{},[],0
    status,failure="PRODUCER_CHECK_FAILED",None
    archive_paths=[];reproduction=None
    try:
        rows,parents,profiles,base_metrics,calls=build_rows(batch)
        write_jsonl(parent,parents);write_jsonl(candidate,rows)
        candidate_sha=sha(candidate)
        print("FROZEN first candidate",candidate_sha,len(rows),flush=True)
        report=validate_rows(rows)
        crossing=conditional_crossing(rows)
        scalar,same_obs=pair_controls(rows,candidate_sha)
        supplied,probes,matched=original_controls(rows,candidate_sha);calls+=len(probes)
        assert len(rows)==3456 and len(parents)==4 and report["distinct_task_jsons"]==1728
        rules=root_rule_audit(rows)
        for groups in rules.values():
            assert groups["overall"]["status_errors"]==864
            assert groups["split:train"]["status_errors"]==groups["split:dev_challenge"]["status_errors"]==432
        histograms=conditional_histograms(rows)
        lookups=tested_lookups(rows)
        visible_tables={digest(row["semantic_task"]["worlds"]) for row in rows}
        decoded_tables=set();identity_coordinate_tables=set()
        for row in rows:
            inverse={name:role for role,name in row["fact_role_assignments"].items()}
            decoded_tables.add(digest([dict(world_id=world["world_id"],facts={inverse[name]:value for name,value in world["facts"].items()}) for world in row["semantic_task"]["worlds"]]))
            task=transposition(row["semantic_task"],row["auxiliary_scalar_control"]) if row["auxiliary_scalar_variant"]=="transposed" else row["semantic_task"]
            identity_coordinate_tables.add(digest([dict(world_id=world["world_id"],facts={inverse[name]:value for name,value in world["facts"].items()}) for world in task["worlds"]]))
        assert len(visible_tables)==24 and len(decoded_tables)==8
        assert len(identity_coordinate_tables)==4 and len({profile["signature_sha256"] for profile in profiles.values()})==4
        report.update(source_families=4,new_source_families=0,semantic_base_contexts=32,scalar_variant_base_cases=64,base_query_observations=192,
            rotation_tasks=1728,rotation_query_observations=5184,rendered_query_observations=10368,conditional_triple_groups=len(crossing),
            within_context_scalar_pairs=len(scalar),opposite_status_same_observation_pairs=len(same_obs),matched_feature_status_error_lower_bound=len(same_obs),
            decoded_source_structures=len(identity_coordinate_tables),decoded_scalar_label_tables=len(decoded_tables),visible_tables=len(visible_tables),structural_profiles=profiles,
            base_case_metrics=base_metrics,supplied_control_counts=dict(Counter(item["name"] for item in supplied+matched+scalar+same_obs)),
            executed_probe_counts=dict(Counter(item["name"] for item in probes)),producer_checks_passed=True,
            frozen_predicate_status_errors={name:group["overall"]["status_errors"] for name,group in rules.items()},
            limitations=["Four same source families and one filtering rule; all extra views/contexts are correlated.",
                "75% nonempty majority; G/H second filters non-strict and two families have singleton G/H.",
                "Auxiliary views preserve outputs; world-table features are excluded from matched-feature lower bound.",
                "Positive-only literals, table frequencies/length, recurring motifs/cyclic orders and finite lookup remain possible.",
                "Exposed related development, no sealed OOD, probabilistic/authority/action utility or model efficacy claim."])
        stable_json(integrity,report)
        stable_json(controls,dict(exact_candidate_sha256=candidate_sha,actual_conditional_query_histograms=crossing,
            supplied_original_pairs=supplied,supplied_same_table_v2_pairs=matched,within_context_scalar_pairs=scalar,
            same_observation_opposite_status_pairs=same_obs,executed_audit_probes=probes,field_order_pairs=field_pairs(rows),
            context_auxiliary_scalar_variant_crossing=dict(source_families=4,contexts=list("ABCDEFGH"),variants=list(VARIANTS),conditional_triple_groups=128),
            probes_add_corpus_records=0,probes_add_source_families=0))
        stable_json(shortcut,dict(exact_candidate_sha256=candidate_sha,frozen_public_predicates_sha256=PUBLIC_HASHES["schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json"],
            actual_frozen_predicate_results=rules,actual_conditional_scalar_table_histograms=histograms,tested_input_lookups=lookups,
            actual_majority_status_accuracy=0.75,feature_boundary="Matched-feature pair theorem excludes the differing actual world fact table.",
            no_complete_shortcut_removal_claim=True))
        archive_paths=archives(output,paths,batch)
        if not worker:reproduction=verify_reproduction(output,batch,paths+archive_paths)
        assert {name:sha(ROOT/name) for name in PRIOR_HASHES}==PRIOR_HASHES
        assert prior_own_artifacts()==own_prior
        status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW"
    except Exception:
        failure=traceback.format_exc()
        if not parent.exists():write_jsonl(parent,parents)
        if not candidate.exists():write_jsonl(candidate,rows)
        raise
    finally:
        artifacts=[path for path in paths+archive_paths+([reproduction] if reproduction else []) if path.exists()]
        stable_json(manifest,dict(batch=batch,track_id="RBM-001",generator_version=VERSION,status=status,actual_execution_id=ACTUAL_EXECUTION,
            actual_role="belief_producer",worker_reproduction=worker,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_snapshot_path=str(snapshot.relative_to(output)),public_source_sha256=public,root_freeze_sha256=ROOT_FREEZE_SHA,
            preserved_prior_sha256=own_prior,dependency_sha256={name:sha(ROOT/name) for name in ["factory/common.py","generators/other_tracks/reference.py","generators/other_tracks/belief_gap.py","generators/other_tracks/belief_gap_v2.py"]},
            seed=own_v2.SEED,config=dict(source_indices=list(PROFILES),contexts=list("ABCDEFGH"),auxiliary_scalar_variants=list(VARIANTS),
                shifts=list(product(range(3),repeat=3)),renderers=["forward","reverse"],diagnostic_key_order="sorted feature sets and canonical JSON"),
            actual_counts=dict(records=len(rows),source_families=len(parents),new_source_families=0,producer_reference_calls=calls),
            measured_cost=dict(wall_seconds=time.perf_counter()-started,core_bytes=sum(path.stat().st_size for path in paths if path.exists()),
                artifact_bytes=sum(path.stat().st_size for path in artifacts),world_observation_comparisons=sum(len(row["semantic_task"]["worlds"])*len(row["semantic_task"]["observations"]) for row in rows if row["renderer"]=="forward")),
            deterministic_artifact_sha256={str(path.relative_to(output)):sha(path) for path in artifacts},
            reproduction_required=not worker,reproduction_receipt=str(reproduction.relative_to(output)) if reproduction else None,
            reproduction_command="/opt/anaconda3/bin/python3 -m generators.other_tracks.belief_gap_v3 --batch belief-gap-v3 (fresh output root only)",
            first_run_failure=failure,no_approval_release_or_push=True))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",default="belief-gap-v3")
    parser.add_argument("--output-root",default=str(ROOT))
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    print("Produced",len(generate(args.batch,args.output_root,args.worker)),"PENDING records",flush=True)
