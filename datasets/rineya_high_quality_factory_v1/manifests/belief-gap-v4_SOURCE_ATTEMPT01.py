"""Producer-only V4 metadata correction. Import only after external source freeze."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record_hash, split_for, write_jsonl
from generators.other_tracks.reference import solve
from generators.other_tracks import belief_gap_v3 as own

BATCH = "belief-gap-v4"
VERSION = "rbm-belief-gap-4.0.0"
ACTOR = "/root/relational_critic_producer"
SOURCE = "generators/other_tracks/belief_gap_v4_attempt01.py"
CONFIG = f"manifests/{BATCH}_CONFIG_ATTEMPT01.json"
FREEZE = f"manifests/{BATCH}_SOURCE_FREEZE_ATTEMPT01.json"
SNAPSHOT = f"manifests/{BATCH}_SOURCE_ATTEMPT01.py"
HAND = f"manifests/{BATCH}_HAND_PROBES_ATTEMPT01.json"
INHERITED = ("same_prefix_empty_nonempty_counterfactual", "contradiction_equals_mixed_outcomes")
MANIFEST = f"data/RBM-001/manifests/{BATCH}.family_controls.json"
MINIMUM = {"name", "scope", "source_family_key", "auxiliary_scalar_variant", "observation_length", "control_id", "pair_sample_ids", "exact_candidate_sha256"}
CLAIM_KEYS = {"name", "property", "scope", "source_family_key", "evidence_manifest", "control_id", "pair_sample_ids", "probe_id"}
CLAIM_NAMES = {*own.own_v2.own_v1.CORE_CONTROLS, "typed_equals_counterfactual", "null_guarantee_contrast", *INHERITED,
               "auxiliary_scalar_transposition_invariance", "same_observations_auxiliary_table_counterfactual", "context_auxiliary_scalar_variant_crossing",
               "world_order_invariance", "world_id_rename_equivariance"}
THRESHOLD = 25 * 1024 * 1024
SCRATCH_LIMIT = 150 * 1024 * 1024
SPACE = []
equal = own.equal


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def room(path, stage):
    path = Path(path)
    while not path.exists(): path = path.parent
    free = shutil.disk_usage(path).free
    SPACE.append(dict(stage=stage,available_bytes=free))
    assert free >= 256 * 1024 * 1024, "Preserve existing outputs; insufficient bounded output space"


def json_out(path, value):
    assert not Path(path).exists(), f"Never overwrite first bytes:{path}"
    room(Path(path).parent,"write:"+Path(path).name)
    atomic(path,canonical(value)+"\n")


def configuration():
    cfg = json.loads((ROOT / CONFIG).read_text())
    frozen = json.loads((ROOT / FREEZE).read_text())
    assert frozen["source_sha256"] == sha(ROOT / SOURCE) == sha(ROOT / SNAPSHOT) == sha(Path(__file__))
    assert frozen["config_sha256"] == sha(ROOT / CONFIG)
    for group in ("public_bindings", "dependency_bindings", "protected_original_artifacts"):
        assert all(sha(ROOT / path) == value for path,value in cfg[group].items()), group
    return cfg,frozen


def coordinate(row):
    return (row["source_family_key"],row["controlled_context"],row["auxiliary_scalar_variant"],row["schedule_index"],row["renderer"])


def identifier(row):
    return f"RBM-001-{BATCH}-p{row['source_family_index']:04d}-{row['controlled_context']}-{row['auxiliary_scalar_variant']}-s{row['schedule_index']:02d}-{row['renderer']}"


def pair_key(row):
    length = len(decoded(row)["observations"])
    return (row["source_family_key"],row["auxiliary_scalar_variant"],row["schedule_index"],row["renderer"],length)


def control_id(key):
    family,variant,schedule,style,length = key
    return f"v4-{family}-{variant}-s{schedule:02d}-{style}-length{length}"


def decoded(row):
    task = json.loads(row["rendered_input"].split("Task JSON:\n",1)[1])
    assert equal(task,row["semantic_task"])
    return task


def comparison(left,right):
    a,b = decoded(left),decoded(right)
    if not equal(a["worlds"],b["worlds"]) or not equal(a["queries"],b["queries"]): raise ValueError("different_worlds_or_ordered_queries")
    if len(a["observations"]) not in (2,3) or len(a["observations"]) != len(b["observations"]): raise ValueError("wrong_observation_length")
    if [x["fact"] for x in a["observations"]] != [x["fact"] for x in b["observations"]]: raise ValueError("different_observation_facts")
    if [i for i,(x,y) in enumerate(zip(a["observations"],b["observations"])) if not equal(x,y)] != [1]: raise ValueError("not_exact_second_equality_change")
    if set(a["observations"][1]) != {"fact","equals"} or not equal(a["observations"][0],b["observations"][0]): raise ValueError("different_first_observation")
    if a["observations"][0]["fact"] == a["observations"][1]["fact"]: raise ValueError("not_distinct_facts")
    if len(a["observations"])==3 and any(not equal(t["observations"][2],t["observations"][0]) for t in (a,b)): raise ValueError("not_F_H_first_observation_repeat")
    pa,pb = own.prefixes(a),own.prefixes(b)
    if own.ids(pa[1]) != own.ids(pb[1]) or not pa[1]: raise ValueError("different_first_prefix")
    if pa[-1] or not pb[-1]: raise ValueError("not_empty_then_nonempty")
    if not own.prefixes(a,[a["observations"][1]])[-1]: raise ValueError("unsupported_contradiction_equality")
    la,lb = solve(a),solve(b)
    if not equal(la,left["verified_target"]) or not equal(lb,right["verified_target"]): raise ValueError("target_mismatch")
    return dict(changed_fields=["observations[1].equals"],typed_first_observation=a["observations"][0],
                first_prefix_survivor_ids=own.ids(pa[1]),left_prefix_counts=[len(s) for s in pa],right_prefix_counts=[len(s) for s in pb],
                left_prefix_survivor_ids=[own.ids(s) for s in pa],right_prefix_survivor_ids=[own.ids(s) for s in pb],
                left_complete_query_payloads=la["queries"],right_complete_query_payloads=lb["queries"])


def closed_claim(claim):
    if set(claim)-CLAIM_KEYS or ("name" in claim)==("property" in claim): raise ValueError("closed_claim_shape")
    name = claim.get("name",claim.get("property"))
    if name not in CLAIM_NAMES or claim.get("scope") not in ("corpus_variant","audit_probe"): raise ValueError("unknown_name_or_scope")
    if "pair_sample_ids" in claim and claim["scope"] != "corpus_variant": raise ValueError("pair_probe_scope")
    if "probe_id" in claim and claim["scope"] != "audit_probe": raise ValueError("probe_corpus_scope")
    return name


def entry_index(manifest):
    result = {}
    for entry in manifest["supplied_same_table_v2_pairs"]:
        key = (entry.get("control_id"),entry.get("name"),tuple(entry.get("pair_sample_ids",[])))
        result.setdefault(key,[]).append(entry)
    return result


def resolve(owner,claim,rows,manifest,manifest_path,candidate_sha,index=None):
    name = closed_claim(claim)
    if name not in INHERITED or claim["scope"] != "corpus_variant": raise ValueError("not_inherited_supplied_claim")
    if any(k not in claim for k in ("pair_sample_ids","evidence_manifest","control_id")): raise ValueError("missing_inline_binding")
    if claim["evidence_manifest"] != manifest_path or not claim["control_id"]: raise ValueError("wrong_manifest_binding")
    if claim.get("source_family_key",owner["source_family_key"]) != owner["source_family_key"]: raise ValueError("wrong_claim_family")
    pair = claim["pair_sample_ids"]
    if not isinstance(pair,list) or len(pair)!=2 or len(set(pair))!=2 or any(s not in rows for s in pair): raise ValueError("missing_or_repeated_pair")
    left,right = (rows[s] for s in pair)
    for member in (left,right):
        if any(not equal(member[field],owner[field]) for field in ("source_family_key","auxiliary_scalar_variant","schedule_index","renderer")): raise ValueError("wrong_owner_coordinate")
    length = len(decoded(owner)["observations"])
    owner_task,left_task = decoded(owner),decoded(left)
    if not equal(owner_task["worlds"],left_task["worlds"]) or not equal(owner_task["queries"],left_task["queries"]): raise ValueError("owner_visible_worlds_or_queries_mismatch")
    if length not in (2,3) or any(len(decoded(m)["observations"])!=length for m in (left,right)): raise ValueError("wrong_owner_length")
    expected = ("E","G") if length==2 else ("F","H")
    if (left["controlled_context"],right["controlled_context"]) != expected: raise ValueError("wrong_context_or_pair_order")
    matches = (entry_index(manifest) if index is None else index).get((claim["control_id"],name,tuple(pair)),[])
    if len(matches)!=1: raise ValueError("nonunique_or_missing_named_pair")
    entry = matches[0]
    if not MINIMUM.issubset(entry): raise ValueError("missing_minimum_fields")
    if entry["scope"]!="corpus_variant" or entry["exact_candidate_sha256"]!=candidate_sha or manifest["exact_candidate_sha256"]!=candidate_sha: raise ValueError("wrong_candidate_or_scope")
    if entry["source_family_key"]!=owner["source_family_key"] or entry["auxiliary_scalar_variant"]!=owner["auxiliary_scalar_variant"] or type(entry["observation_length"]) is not int or entry["observation_length"]!=length: raise ValueError("wrong_entry_owner_or_length")
    evidence = comparison(left,right)
    for key,value in evidence.items():
        if key in entry and not equal(value,entry[key]): raise ValueError("incorrect_parsed_evidence:"+key)
    return evidence


def hand_probes():
    rows = {}; fixture_manifest = "hand/manifest.json"; candidate_sha = "a"*64
    for family in ("hand-alpha","hand-beta"):
        for variant in ("identity","transposed"):
            for style in ("forward","reverse"):
                worlds = [dict(world_id="u",facts=dict(x=0,y="silk",q=True)),dict(world_id="v",facts=dict(x=0,y="silk",q=1)),dict(world_id="z",facts=dict(x=1,y="linen",q=None))]
                baseline,supported = "linen","silk"
                if variant=="transposed":
                    for world in worlds: world["facts"]["y"] = "silk" if world["facts"]["y"]=="linen" else "linen"
                    baseline,supported = supported,baseline
                for context in ("A","D","E","F","G","H"):
                    obs = [dict(fact="x",equals=0),dict(fact="y",equals=baseline if context in "EF" else supported)]
                    if context in "DFH": obs.append(copy.deepcopy(obs[0]))
                    task = dict(kind="belief",worlds=worlds,observations=obs,queries=[dict(query_id="ask",fact="q")])
                    payload = task if style=="forward" else own.own_v2.own_v1.reorder(task)
                    sid = f"hand-{family}-{variant}-{style}-{context}"
                    rows[sid] = dict(sample_id=sid,source_family_key=family,auxiliary_scalar_variant=variant,schedule_index=0,renderer=style,controlled_context=context,semantic_task=task,
                                     rendered_input="Task JSON:\n"+json.dumps(payload),verified_target=solve(task))
    entries = []
    for owner in list(rows.values()):
        if owner["controlled_context"] not in "AD": continue
        pair_contexts = ("E","G") if len(decoded(owner)["observations"])==2 else ("F","H")
        pair = [next(r for r in rows.values() if all(r[k]==owner[k] for k in ("source_family_key","auxiliary_scalar_variant","renderer")) and r["controlled_context"]==c) for c in pair_contexts]
        cid = control_id(pair_key(owner)); facts=comparison(*pair)
        for name in INHERITED: entries.append(dict(name=name,scope="corpus_variant",source_family_key=owner["source_family_key"],auxiliary_scalar_variant=owner["auxiliary_scalar_variant"],observation_length=len(decoded(owner)["observations"]),control_id=cid,pair_sample_ids=[r["sample_id"] for r in pair],exact_candidate_sha256=candidate_sha,**facts))
    manifest = dict(exact_candidate_sha256=candidate_sha,supplied_same_table_v2_pairs=entries)
    positives=[]
    for owner in rows.values():
        if owner["controlled_context"] not in "AD":continue
        for name in INHERITED:
            entry=next(e for e in entries if e["name"]==name and e["control_id"]==control_id(pair_key(owner)))
            claim=dict(name=name,scope="corpus_variant",source_family_key=owner["source_family_key"],evidence_manifest=fixture_manifest,control_id=entry["control_id"],pair_sample_ids=entry["pair_sample_ids"])
            resolve(owner,claim,rows,manifest,fixture_manifest,candidate_sha)
            positives.append(dict(name=f"nonmember_{owner['source_family_key']}_{owner['auxiliary_scalar_variant']}_{owner['renderer']}_{owner['controlled_context']}_{name}",passed=True))
    owner=next(r for r in rows.values() if r["source_family_key"]=="hand-alpha" and r["auxiliary_scalar_variant"]=="identity" and r["renderer"]=="forward" and r["controlled_context"]=="A")
    entry=next(e for e in entries if e["name"]==INHERITED[0] and e["control_id"]==control_id(pair_key(owner)))
    base=dict(name=INHERITED[0],scope="corpus_variant",source_family_key=owner["source_family_key"],evidence_manifest=fixture_manifest,control_id=entry["control_id"],pair_sample_ids=entry["pair_sample_ids"])
    negatives=[]
    for mode in ("no_pair","no_manifest","no_control","foreign_family","foreign_pair","wrong_manifest","wrong_owner_variant","wrong_owner_schedule","wrong_owner_style","wrong_owner_length","wrong_owner_visible_queries","reversed_pair","duplicate_lookup","missing_named_entry","wrong_named_entry","wrong_hash","missing_minimum","incorrect_prefix","scope_probe","undeclared_key","typed_first_change"):
        o,c,rs,m=copy.deepcopy((owner,base,rows,manifest))
        if mode=="no_pair":c.pop("pair_sample_ids")
        elif mode=="no_manifest":c.pop("evidence_manifest")
        elif mode=="no_control":c.pop("control_id")
        elif mode=="foreign_family":o["source_family_key"]="hand-beta"
        elif mode=="foreign_pair":c["pair_sample_ids"]=[next(r["sample_id"] for r in rs.values() if r["source_family_key"]=="hand-beta" and r["auxiliary_scalar_variant"]=="identity" and r["renderer"]=="forward" and r["controlled_context"]==ctx) for ctx in ("E","G")]
        elif mode=="wrong_manifest":c["evidence_manifest"]="hand/foreign.json"
        elif mode=="wrong_owner_variant":o["auxiliary_scalar_variant"]="transposed"
        elif mode=="wrong_owner_schedule":o["schedule_index"]=1
        elif mode=="wrong_owner_style":o["renderer"]="reverse"
        elif mode=="wrong_owner_length":o["semantic_task"]["observations"].append(copy.deepcopy(o["semantic_task"]["observations"][0]));o["rendered_input"]="Task JSON:\n"+json.dumps(o["semantic_task"])
        elif mode=="wrong_owner_visible_queries":o["semantic_task"]["queries"][0]["fact"]="x";o["rendered_input"]="Task JSON:\n"+json.dumps(o["semantic_task"])
        elif mode=="reversed_pair":c["pair_sample_ids"].reverse()
        elif mode=="duplicate_lookup":m["supplied_same_table_v2_pairs"].append(copy.deepcopy(entry))
        elif mode=="missing_named_entry":m["supplied_same_table_v2_pairs"]=[e for e in m["supplied_same_table_v2_pairs"] if e!=entry]
        elif mode=="wrong_named_entry":next(e for e in m["supplied_same_table_v2_pairs"] if e==entry)["name"]="contradiction_equals_mixed_outcomes"
        elif mode=="wrong_hash":next(e for e in m["supplied_same_table_v2_pairs"] if e==entry)["exact_candidate_sha256"]="b"*64
        elif mode=="missing_minimum":next(e for e in m["supplied_same_table_v2_pairs"] if e==entry).pop("observation_length")
        elif mode=="incorrect_prefix":next(e for e in m["supplied_same_table_v2_pairs"] if e==entry)["first_prefix_survivor_ids"]=["z"]
        elif mode=="scope_probe":c["scope"]="audit_probe";c["probe_id"]="unexecuted"
        elif mode=="undeclared_key":c["private_alias"]=True
        elif mode=="typed_first_change":
            row=rs[c["pair_sample_ids"][1]];row["semantic_task"]["observations"][0]["equals"]=False;row["rendered_input"]="Task JSON:\n"+json.dumps(row["semantic_task"])
        try:resolve(o,c,rs,m,fixture_manifest,candidate_sha)
        except ValueError as error:negatives.append(dict(name=mode,rejected=True,actual_reason=str(error)))
        else:raise AssertionError("Hand negative accepted:"+mode)
    # Publicly permitted shared ID/two names and optional property spelling are accepted.
    alias={**base,"property":base["name"]};alias.pop("name");resolve(owner,alias,rows,manifest,fixture_manifest,candidate_sha)
    positives.append(dict(name="property_spelling_and_shared_control_id_two_names",passed=True))
    return dict(actual_executing_agent=ACTOR,scope="producer_hand_probe",hand_inputs_only=True,prior_private_fixtures_read=False,positive_count=len(positives),negative_count=len(negatives),positive_results=positives,negative_results=negatives,all_passed=True)


def revised_rows():
    original = ROOT/"data/RBM-001/candidate_renderings/belief-gap-v3.jsonl"
    mapping={}; group_ids={}
    with original.open() as file:
        for line in file:
            row=json.loads(line);mapping[row["sample_id"]]=identifier(row);group_ids[coordinate(row)]=identifier(row)
    assert len(mapping)==3456 and len(group_ids)==3456
    rows=[];diffs=Counter();evidence=[]
    allowed={"sample_id","semantic_parent_id","semantic_family_id","generator_version","family_control_claims","record_hash"}
    with original.open() as file:
        for line in file:
            old=json.loads(line);row=copy.deepcopy(old)
            row["sample_id"]=mapping[old["sample_id"]]
            row["semantic_parent_id"]=row["semantic_family_id"]=f"RBM-001-{BATCH}-p{row['source_family_index']:04d}"
            row["generator_version"]=VERSION
            key=pair_key(row);length=key[-1];assert length in (2,3)
            negative,positive=("E","G") if length==2 else ("F","H")
            pair=[group_ids[(key[0],c,key[1],key[2],key[3])] for c in (negative,positive)]
            for claim in row["family_control_claims"]:
                name=closed_claim(claim);claim["evidence_manifest"]=MANIFEST
                if "pair_sample_ids" in claim:claim["pair_sample_ids"]=[mapping[x] for x in claim["pair_sample_ids"]]
                if name in INHERITED:
                    assert claim["scope"]=="corpus_variant";claim["control_id"]=control_id(key);claim["pair_sample_ids"]=pair
            row["record_hash"]=record_hash(row)
            actual=sorted(k for k in set(old)|set(row) if not equal(old.get(k),row.get(k)))
            assert set(actual)==allowed
            diffs.update(actual)
            assert equal(row["semantic_task"],old["semantic_task"]) and row["rendered_input"]==old["rendered_input"] and equal(row["verified_target"],old["verified_target"])
            target=solve(decoded(row));assert equal(target,row["verified_target"])
            assert row["split"]==split_for(row["source_family_index"]) and row["review_status"]=="PENDING" and not row["review_receipt_ids"]
            evidence.append(dict(coordinate=list(coordinate(row)),original_sample_id=old["sample_id"],new_sample_id=row["sample_id"],typed_task_sha256=digest(row["semantic_task"]),rendered_input_sha256=hashlib.sha256(row["rendered_input"].encode()).hexdigest(),full_typed_target_sha256=digest(target),actual_changed_metadata_fields=actual))
            rows.append(row)
    assert len(rows)==3456 and Counter(r["split"] for r in rows)=={"train":1728,"dev_challenge":1728}
    parents=load_jsonl(ROOT/"data/RBM-001/semantic_parents/belief-gap-v3.jsonl")
    for p in parents:
        p["semantic_parent_id"]=p["semantic_family_id"]=f"RBM-001-{BATCH}-p{p['source_family_index']:04d}";p["generator_version"]=VERSION
    return rows,parents,dict(actual_comparison_rows=3456,full_typed_task_rendered_input_target_triples_equal=True,reference_recomputed_rows=3456,metadata_diff_field_counts=dict(sorted(diffs.items())),coordinate_evidence=evidence,new_semantic_tasks=0,new_source_families=0,prior_canonical_exposure=True)


def supplied_entries(rows,candidate_sha):
    indexed={coordinate(row):row for row in rows};entries=[]
    for family in sorted({r["source_family_key"] for r in rows}):
        for variant in ("identity","transposed"):
            for schedule in range(27):
                for style in ("forward","reverse"):
                    for length,contexts in ((2,("E","G")),(3,("F","H"))):
                        pair=[indexed[(family,c,variant,schedule,style)] for c in contexts];key=(family,variant,schedule,style,length)
                        facts=comparison(*pair)
                        for name in INHERITED:entries.append(dict(name=name,scope="corpus_variant",source_family_key=family,auxiliary_scalar_variant=variant,observation_length=length,control_id=control_id(key),pair_sample_ids=[r["sample_id"] for r in pair],exact_candidate_sha256=candidate_sha,schedule_index=schedule,renderer=style,**facts))
    assert len(entries)==1728
    return entries


def archive(output,paths):
    entries=[];files=[]
    for path in paths:
        if path.stat().st_size<=THRESHOLD:continue
        relative=str(path.relative_to(output));target=output/f"data/RBM-001/archives/{BATCH}/{relative.replace('/','__')}.gz"
        assert not target.exists();room(target.parent,"archive:"+path.name);target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.name+".tmp");assert not tmp.exists()
        with path.open("rb") as raw,tmp.open("wb") as destination:
            with gzip.GzipFile(filename="",mode="wb",fileobj=destination,mtime=0,compresslevel=6) as compressor:
                shutil.copyfileobj(raw,compressor,1024*1024)
            destination.flush();os.fsync(destination.fileno())
        tmp.replace(target);decoded_hash=hashlib.sha256();count=0
        with gzip.open(target,"rb") as compressed,path.open("rb") as raw:
            while True:
                a,b=compressed.read(1024*1024),raw.read(1024*1024)
                assert a==b
                if not a:break
                decoded_hash.update(a);count+=len(a)
        entries.append(dict(raw_path=relative,raw_bytes=path.stat().st_size,raw_sha256=sha(path),archive_path=str(target.relative_to(output)),archive_bytes=target.stat().st_size,archive_sha256=sha(target),decoded_bytes=count,decoded_sha256=decoded_hash.hexdigest(),roundtrip_exact=True));files.append(target)
    path=output/f"manifests/{BATCH}_ARCHIVES.json";json_out(path,dict(actual_execution_id=ACTOR,actual_role="belief_producer",threshold_bytes=THRESHOLD,raw_bytes_preserved=True,gzip_parameters=dict(filename="",mtime=0,compresslevel=6),oversized_artifacts=entries));return files+[path]


def core_paths(output):
    return [output/SNAPSHOT,output/FREEZE,output/CONFIG,output/HAND,output/f"data/RBM-001/candidate_renderings/{BATCH}.jsonl",output/f"data/RBM-001/semantic_parents/{BATCH}.jsonl",
            output/MANIFEST,output/f"data/RBM-001/manifests/{BATCH}.producer_integrity.json",output/f"data/RBM-001/manifests/{BATCH}.shortcut_audit.json",output/f"manifests/{BATCH}_MARKER_RULES.json"]


def generate(output=ROOT,worker=False):
    output=Path(output).resolve();start=time.perf_counter();cpu=time.process_time();cfg,frozen=configuration();core=core_paths(output)
    for p in core[3:]:assert not p.exists(),f"First attempt preserved:{p}"
    if output!=ROOT:
        for source,target in zip(core_paths(ROOT)[:3],core[:3]):
            assert not target.exists();room(target.parent,"replay_frozen_source_copy");atomic(target,source.read_text())
    archive_source=output/frozen["source_archive_path"]
    if output!=ROOT:
        assert not archive_source.exists();atomic(archive_source,(ROOT/frozen["source_archive_path"]).read_text())
    assert sha(archive_source)==frozen["source_sha256"]
    status="FAILED_PRODUCER_RUN_PRESERVED";failure=None;rows=[];parents=[];deterministic=[]
    execution=output/f"manifests/{BATCH}_EXECUTION_ATTEMPT01.json"
    try:
        hands=hand_probes();json_out(core[3],hands);print("HAND_PROBES_PASSED",hands["positive_count"],hands["negative_count"],flush=True)
        rows,parents,equivalence=revised_rows()
        for variant in own.VARIANTS:own.own_v2.own_v1.validate_rows([row for row in rows if row["auxiliary_scalar_variant"]==variant])
        room(core[4].parent,"candidate_after_all_metadata_finalized");write_jsonl(core[4],rows);candidate_sha=sha(core[4]);write_jsonl(core[5],parents)
        print("CANDIDATE_FROZEN",candidate_sha,"ROWS",len(rows),flush=True)
        crossing=own.conditional_crossing(rows);scalar,same_observation=own.pair_controls(rows,candidate_sha)
        supplied,probes,prior_matched=own.original_controls(rows,candidate_sha);matched=supplied_entries(rows,candidate_sha)
        # Derived semantic diagnostics must agree with old producer pair evidence modulo custody.
        pair_map={(e["name"],tuple(e["pair_sample_ids"])):e for e in prior_matched}
        for entry in matched:
            previous=pair_map[(entry["name"],tuple(entry["pair_sample_ids"]))]
            for k in comparison(*[next(r for r in rows if r["sample_id"]==sid) for sid in entry["pair_sample_ids"]]):assert equal(entry[k],previous[k])
        controls=dict(exact_candidate_sha256=candidate_sha,actual_conditional_query_histograms=crossing,supplied_original_pairs=supplied,supplied_same_table_v2_pairs=matched,
            within_context_scalar_pairs=scalar,same_observation_opposite_status_pairs=same_observation,executed_audit_probes=probes,field_order_pairs=own.field_pairs(rows),
            context_auxiliary_scalar_variant_crossing=dict(source_families=4,contexts=list("ABCDEFGH"),variants=list(own.VARIANTS),conditional_triple_groups=128),probes_add_corpus_records=0,probes_add_source_families=0)
        by_id={r["sample_id"]:r for r in rows};index=entry_index(controls);resolved=0
        for row in rows:
            for claim in row["family_control_claims"]:
                if closed_claim(claim) in INHERITED:resolve(row,claim,by_id,controls,MANIFEST,candidate_sha,index);resolved+=1
        assert resolved==6912 and len(crossing)==128 and len(scalar)==1728 and len(same_observation)==864 and len(probes)==16
        old_controls=json.loads((ROOT/"data/RBM-001/manifests/belief-gap-v3.family_controls.json").read_text());assert equal(crossing,old_controls["actual_conditional_query_histograms"])
        old_shortcut=json.loads((ROOT/"data/RBM-001/manifests/belief-gap-v3.shortcut_audit.json").read_text())
        rules=own.root_rule_audit(rows);histograms=own.conditional_histograms(rows)
        assert equal(rules,old_shortcut["actual_frozen_predicate_results"]) and equal(histograms,old_shortcut["actual_conditional_scalar_table_histograms"])
        shortcut=copy.deepcopy(old_shortcut);shortcut["exact_candidate_sha256"]=candidate_sha
        shortcut["diagnostic_reuse_custody"]=dict(original_shortcut_sha256=cfg["protected_original_artifacts"]["data/RBM-001/manifests/belief-gap-v3.shortcut_audit.json"],exact3456_input_target_triples_revalidated=True,frozen_rules_and_all_histograms_actually_recomputed=True,fitted_lookups_reused_after_exact_input_target_equality=True,no_new_model_run=True)
        integrity=json.loads((ROOT/"data/RBM-001/manifests/belief-gap-v3.producer_integrity.json").read_text())
        integrity.update(metadata_revision_equivalence=equivalence,inherited_claim_bindings=dict(actual_inline_claims=resolved,actual_named_entries=len(matched),actual_semantic_pairs=len(matched)//2,lookup_key="control_id + name + ordered pair_sample_ids",required_minimum_fields=sorted(MINIMUM),owner_length_histogram=dict(Counter(len(decoded(r)["observations"]) for r in rows))),executed_interface_hand_probes=hands,actual_reference_calls_for_fresh_rows=3456,executed_audit_probe_count=16,all_new_status="PENDING",no_new_semantic_tasks_or_families=True)
        integrity["limitations"].insert(0,"Metadata-only V4 reuses3456 exact input/full-target triples, four families and prior canonical exposure. Source and record bindings are fresh; no unseen blind semantic or model efficacy claim.")
        json_out(core[6],controls);json_out(core[7],integrity);json_out(core[8],shortcut)
        json_out(core[9],json.loads((ROOT/"manifests/belief-gap-v3_MARKER_RULES.json").read_text()))
        archives=archive(output,core);deterministic=core+[archive_source]+archives
        generation=output/f"manifests/{BATCH}_GENERATION.json"
        json_out(generation,dict(batch=BATCH,track_id="RBM-001",generator_version=VERSION,status="PENDING",actual_execution_id=ACTOR,actual_role="belief_producer",source_sha256=frozen["source_sha256"],source_freeze_sha256=sha(output/FREEZE),configuration_sha256=sha(output/CONFIG),source_archive_path=frozen["source_archive_path"],public_bindings=cfg["public_bindings"],dependency_bindings=cfg["dependency_bindings"],protected_original_artifacts=cfg["protected_original_artifacts"],official_checker_custody_from_root_only=cfg["official_checker_custody_from_root_only"],actual_counts=dict(records=3456,reused_source_families=4,new_source_families=0,reused_semantic_base_contexts=32,reused_scalar_cases=64,new_semantic_tasks=0,rotation_tasks=1728,train_records=1728,dev_records=1728,inherited_inline_claims=6912,inherited_named_entries=1728,inherited_length_specific_pairs=864,scalar_pairs=1728,opposite_status_same_observation_pairs=864,conditional_triple_groups=128,executed_audit_probes=16),artifact_sha256={str(p.relative_to(output)):sha(p) for p in deterministic},runtime_costs_separately_recorded=True,prior_canonical_exposure=True,no_approval_release_or_push=True,reproduction_command="PYTHONHASHSEED=17 /opt/anaconda3/bin/python3 -m generators.other_tracks.belief_gap_v4_attempt01 --output-root /NEW/EMPTY/ROOT --worker"))
        deterministic.append(generation);configuration();status="PROVISIONAL_PENDING_FRESH_INDEPENDENT_REVIEW"
        print("ALL_CORE_FROZEN",len(deterministic),sum(p.stat().st_size for p in deterministic),flush=True)
        if not worker:
            scratch=output/f"reproducibility/{BATCH}/hashseed-991";assert not scratch.exists();room(scratch,"before_separate_replay")
            command=["/opt/anaconda3/bin/python3","-m","generators.other_tracks.belief_gap_v4_attempt01","--output-root",str(scratch),"--worker"]
            process=subprocess.run(command,cwd=ROOT,env={**os.environ,"PYTHONHASHSEED":"991"},capture_output=True,text=True)
            comparisons=[dict(path=str(p.relative_to(output)),main_sha256=sha(p),replay_sha256=sha(scratch/p.relative_to(output)) if (scratch/p.relative_to(output)).exists() else None) for p in deterministic]
            size=sum(p.stat().st_size for p in scratch.rglob('*') if p.is_file());passed=process.returncode==0 and size<=SCRATCH_LIMIT and all(x["main_sha256"]==x["replay_sha256"] for x in comparisons)
            json_out(output/f"manifests/{BATCH}_REPRODUCTION.json",dict(actual_execution_id=ACTOR,separate_python_processes=True,hash_seeds=[os.environ.get("PYTHONHASHSEED"),"991"],artifacts=comparisons,passed=passed,scratch_bytes=size,scratch_limit_bytes=SCRATCH_LIMIT,worker_stdout=process.stdout,worker_stderr=process.stderr,worker_returncode=process.returncode,worker_execution=json.loads((scratch/f"manifests/{BATCH}_EXECUTION_ATTEMPT01.json").read_text()) if (scratch/f"manifests/{BATCH}_EXECUTION_ATTEMPT01.json").exists() else None))
            assert passed,"Preserve failed replay and all first bytes"
            shutil.rmtree(scratch)
    except Exception:
        failure=traceback.format_exc();status="FAILED_PRODUCER_RUN_PRESERVED";raise
    finally:
        json_out(execution,dict(actual_execution_id=ACTOR,actual_role="belief_producer",batch=BATCH,generator_version=VERSION,status=status,initial_failure=failure,source_sha256=frozen["source_sha256"],source_freeze_sha256=sha(ROOT/FREEZE),actual_python_executable=sys.executable,actual_python_version=sys.version,actual_process_id=os.getpid(),hash_seed=os.environ.get("PYTHONHASHSEED"),worker_reproduction=worker,CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actual_rows=len(rows),actual_source_parents=len(parents),deterministic_artifact_sha256={str(p.relative_to(output)):sha(p) for p in deterministic if p.exists()},deterministic_bytes=sum(p.stat().st_size for p in deterministic if p.exists()),disk_space_observations=SPACE,no_approval_release_or_push=True))
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output-root",default=str(ROOT));parser.add_argument("--worker",action="store_true");args=parser.parse_args()
    print("Produced",len(generate(args.output_root,args.worker)),"PENDING records",flush=True)
