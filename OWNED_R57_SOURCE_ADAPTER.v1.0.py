#!/usr/bin/env python3
"""Root-owned R57 human-source supplement adapter. No model execution or training.
It copies reviewed source text, checks strict v1 shape and semantic information flow,
and exposes scoped static packets. Original schema files remain unchanged.
"""
import argparse, copy, hashlib, json, os, pathlib, re, shutil, sys, zipfile
from collections import Counter
from itertools import product
import jsonschema
sys.dont_write_bytecode = True
UNITS = (16,27,59,67,136,265)
FAMILY = "DUDENEY_AIM_1917"
TEACHER_UNITS = (16,27,59,67,265)
EDUCATIONAL_UNITS = (16,27,67,265)

def sha(b):
    return hashlib.sha256(b).hexdigest()
def jsread(p):
    return json.loads(pathlib.Path(p).read_bytes())
def jwrite(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
def jlwrite(p,rows):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for r in rows))
def jlread(p):
    return [json.loads(b) for b in pathlib.Path(p).read_bytes().splitlines()]
def parent_id(n):
    return FAMILY+"_"+str(n).zfill(3)
def student_input(task):
    return {"task_id":task["id"],"student_task":task["prompt"]}
def teacher_input(task,packet):
    assert task["id"]==packet["student_task_id"]
    assert task["parent_id"]==packet["parent_id"] and task["family_id"]==packet["family_id"]
    assert task["role"]=="train" and packet["role"]=="train"
    return {"task_id":task["id"],"student_task":task["prompt"],
            "verified_privileged_demonstration":packet["teacher_only_demonstration"],
            "response_prefix":None,"prefix_provenance":None,"scoring_status":"NOT_COLLECTED"}
def evaluator_input(task,parents,certificate):
    parent=parents[task["parent_id"]]
    return {"task_id":task["id"],"student_task":task["prompt"],
            "response_semantics":parent["response_semantics"],"certificate":certificate,
            "reference_response":task["target"],"visibility":"EXPOSED_TRAIN_VERIFIER"}
def bridge_valid(s):
    try:
        lines=[x.strip() for x in s.splitlines() if "|" in x]
        assert len(lines)==11
        partners=Counter();opponents=Counter()
        for line in lines:
            tables=line.split("|");assert len(tables)==3;seen=[]
            for table in tables:
                m=re.fullmatch(r"\s*([A-L])\s+([A-L])\s+—\s+([A-L])\s+([A-L])\s*",table);assert m
                a,b,c,d=m.groups();assert len({a,b,c,d})==4
                seen += [a,b,c,d]
                for pair in [(a,b),(c,d)]:partners[tuple(sorted(pair))]+=1
                for pair in product([a,b],[c,d]):opponents[tuple(sorted(pair))]+=1
            assert len(seen)==12 and set(seen)==set("ABCDEFGHIJKL")
        assert len(partners)==66 and set(partners.values())=={1}
        assert len(opponents)==66 and set(opponents.values())=={2}
        return True
    except (AssertionError,AttributeError):
        return False
def invariant(tasks,parents,sources,packets,task_schema,knowledge,knowledge_schema):
    assert len(parents)==6 and set(parents)=={parent_id(n) for n in UNITS}
    assert len(tasks)==10 and len({t["id"] for t in tasks})==10
    assert len({(t["prompt"],t["target"]) for t in tasks})==10
    validator=jsonschema.Draft202012Validator(task_schema)
    for t in tasks:
        validator.validate(t)
        p=parents[t["parent_id"]]
        assert t["role"]==p["role"]=="train" and t["family_id"]==p["family_id"]==FAMILY
        assert all(x in sources for x in t["evidence_ids"])
        assert t["evidence_ids"]==[p["statement_source_id"]]
        assert t["prompt"]==p["source_statement"]
        assert t["target"] in p["eligible_original_targets"]
        assert t["target"] not in t["prompt"]
        assert student_input(t)=={"task_id":t["id"],"student_task":p["source_statement"]}
        assert p["rights"]["source_admission"]=="SOURCE_ADMITTED"
        assert p["rights"]["rule"]=="B" and p["rights"]["qualified_jurisdictions"]==["US","UK-named-author-text"]
        assert p["content_verification"]=="EXACT_RENDERED_TARGET_AND_SEPARATE_PIXEL_REVIEW_PASS"
        if p["source_number"]==136:assert t["view"]=="concise" if "view" in t else t["id"].endswith("_concise_v1")
        if p["source_number"]==265:assert bridge_valid(t["target"])
    assert len(packets)==5 and {p["parent_id"] for p in packets}=={parent_id(n) for n in TEACHER_UNITS}
    taskmap={t["id"]:t for t in tasks}
    for p in packets:
        assert p["teacher_only_demonstration"] in parents[p["parent_id"]]["eligible_original_targets"]
        assert p["teacher_source_id"] in sources
        assert p["teacher_only_demonstration"] not in taskmap[p["student_task_id"]]["prompt"]
        assert p["actual_student_generated_prefix"] is None and p["teacher_logits"] is None
        teacher_input(taskmap[p["student_task_id"]],p)
    assert len(knowledge)==8 and len({k["id"] for k in knowledge})==8
    kv=jsonschema.Draft202012Validator(knowledge_schema)
    for k in knowledge:
        kv.validate(k)
        assert k["status"]=="supported_in_scope" and all(s in sources for s in k["source_ids"])
        assert k["payload_sha256"]==sha(k["canonical_form"].encode())
    assert all(p["empirical_Rineya_difficulty"] is None for p in parents.values())
    return True
def build(w,out):
    w=pathlib.Path(w).resolve();out=pathlib.Path(out).resolve()
    assert out.is_relative_to(w/"output") and not out.exists()
    root=w/"output/revision57_research"
    gate=jsread(root/"root_supplement_admission/v1.0/ROOT_SCOPED_ADMISSION.DATA.json")
    assert gate["six_parent_scoped_train_admission"] is True and gate["family_role_assignment"]=="train"
    assert gate["unqualified_full_corpus_or_native_claims"] is False
    out.mkdir(parents=True)
    restored=root/"root_minimum_verified_restore/v1.0/recovered"
    for name in ["task_record.schema.json","knowledge_record.schema.json"]:
        dst=out/"schemas"/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(restored/"schemas"/name,dst)
    corpus=root/"root_human_scan_transcription/v1.0"
    idx=jsread(corpus/"TRANSCRIPTION_AND_ALL_EDITS.DATA.json")
    assert sha((corpus/"TRANSCRIPTION_AND_ALL_EDITS.DATA.json").read_bytes())==gate["transcription_index_sha256"]
    parents={};sources={};tasks=[];packets=[]
    signatures={16:"unique bounded base7 place-value decomposition",27:"finite multi-agent coin conservation/assignment",
                59:"conditional clock measurement/rational uniform-rate inference",67:"equal-distance total-time aggregation",
                136:"minimum sum-of-two-positive-squares representation multiplicity",265:"cyclic partner/opponent block schedule"}
    for item in idx["components"]:
        number=item["source_unit_number"];pid=item["id"]
        assert pid==parent_id(number) and item["family_id"]==FAMILY
        texts={}
        for kind,c in item["components"].items():
            src=pathlib.Path(c["path"])
            assert src.is_relative_to(corpus) and src.suffix==".txt"
            b=src.read_bytes();assert len(b)==c["bytes"] and sha(b)==c["sha256"]
            texts[kind]=b.decode().removesuffix("\n")
            source_id=pid+"_"+kind
            path="source_text/"+src.name
            dst=out/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
            sources[source_id]={"id":source_id,"family_id":FAMILY,"path":path,"bytes":len(b),"sha256":sha(b),
                                "component":kind,"role":"train","positive_eligible":kind!="derivation" or number!=136,
                                "original_scan_sha256":idx["scan_sha256"],"rights_record":"review_records/commons_rights_decision.json"}
        eligible=[texts["concise_answer"]]
        if number in TEACHER_UNITS and texts["derivation"] not in eligible:eligible.append(texts["derivation"])
        parents[pid]={"version":"r57.source-parent.v1.0","id":pid,"family_id":FAMILY,"role":"train","source_number":number,
                      "source_statement":texts["statement"],"statement_source_id":pid+"_statement",
                      "concise_source_id":pid+"_concise_answer","eligible_original_targets":eligible,
                      "rights":{"source_admission":"SOURCE_ADMITTED","rule":"B","qualified_jurisdictions":["US","UK-named-author-text"],
                                "worldwide_clearance":False,"new_AI_positive_prose":False},
                      "content_verification":"EXACT_RENDERED_TARGET_AND_SEPARATE_PIXEL_REVIEW_PASS",
                      "structural_signature":signatures[number],
                      "response_semantics":{"valid_solutions":"semantic constraints; original source response is an admissible reference",
                                            "requires_conditional_measurement_scope":number==59,"alternate_valid_schedules_allowed":number==265,
                                            "unique_gift_count_pattern_only":number==16,"positive_unordered_square_pairs":number==136},
                      "parent_dependencies":[],"empirical_Rineya_difficulty":None,"human_review":"PENDING",
                      "global_novelty":None,"protected_evaluator_contamination":"UNQUALIFIED_NOT_ACCESSED"}
        views=[("concise",texts["concise_answer"])]
        if number in EDUCATIONAL_UNITS:views.append(("educational",texts["derivation"]))
        for view,target in views:
            tasks.append({"schema_version":"1.0","id":pid+"_"+view+"_v1","parent_id":pid,"family_id":FAMILY,"role":"train",
                          "prompt":texts["statement"],"target":target,"evidence_ids":[pid+"_statement"],
                          "generator_revision":"Root_R57_faithful_source_adapter_v1.0",
                          "response_contract":{"open_ended":False,"required_facts":[],"forbidden_facts":[],
                                               "question_required":False,"question_forbidden":False}})
        if number in TEACHER_UNITS:
            packets.append({"version":"r57.privileged-source-packet.v1.0","id":pid+"_teacher_packet_v1","parent_id":pid,
                            "family_id":FAMILY,"role":"train","student_task_id":pid+"_concise_v1",
                            "teacher_source_id":pid+"_derivation","teacher_only_demonstration":texts["derivation"],
                            "pairing_rationale":"same parent, exact question constraints and authentic verified original solution; intentional training privilege",
                            "mismatched_context_allowed":False,"actual_student_generated_prefix":None,"teacher_logits":None,
                            "response_tokenizer":None,"future_collection":"Require actual checkpoint/tokenizer/generation policy and identical response-token prefixes before scoring",
                            "teacher_advantage":"UNMEASURED","not_on_policy":True})
    src8=restored/"output/revision56_research/root_atomic_source_intake/v1.2"
    knowledge=jlread(src8/"SOURCE_ADMITTED_KNOWLEDGE.v1.0.DATA.jsonl")
    science_lineage=jlread(src8/"PROSPECTIVE_LINEAGE_AND_STATE_TRANSITIONS.DATA.jsonl")
    for k,line in zip(knowledge,science_lineage):
        assert k["id"]==line["knowledge_id"] and line["training_admission"] is False and line["role"] is None
        sid=k["source_ids"][0];canonical=k["canonical_form"].encode();payload=json.loads(canonical)
        path="source_facts/"+payload["entity_id"]+".canonical_snak.json"
        jwrite(out/path,payload)
        sources[sid]={"id":sid,"family_id":"wikidata-structured-science","path":path,
                      "raw_canonical_form_sha256":sha(canonical),"role":None,"training_admission":False,
                      "original_record_id":k["id"],"backup_release":"rakdc-backup-20261012T064525Z-r56-closed",
                      "source_snapshot_sha256":line["raw_source_sha256"],"source_snapshot_bytes":line["raw_source_bytes"],
                      "source_value_pointer":line["value_pointer"],"positive_eligible":False}
    tasks.sort(key=lambda x:x["id"]);packets.sort(key=lambda x:x["id"])
    jlwrite(out/"data/canonical/parents.sidecar.jsonl",sorted(parents.values(),key=lambda x:x["id"]))
    jlwrite(out/"data/train/tasks.v1.jsonl",tasks)
    jlwrite(out/"data/canonical/knowledge.v1.jsonl",knowledge)
    jlwrite(out/"data/distillation_packets/source_packets.jsonl",packets)
    jlwrite(out/"manifests/sources.jsonl",sorted(sources.values(),key=lambda x:x["id"]))
    jlwrite(out/"data/train/supervised_examples.jsonl",[{"id":t["id"],"parent_id":t["parent_id"],"student_input":student_input(t),
        "accepted_original_human_target":t["target"],"response_token_mask":None,"tokenizer_identity":None,
        "mask_status":"RUNTIME_TOKENIZATION_REQUIRED; no invented token indices","empirical_model_origin":False} for t in tasks])
    jwrite(out/"manifests/family_roles.json",{"family_id":FAMILY,"role":"train","all_source_document_descendants_coupled":True,
                                            "qualified_scope":"six reviewed original units; other units not automatically admitted",
                                            "past_other_roles_or_HOLDs_rewritten":False})
    jwrite(out/"manifests/subsets.json",{"all_six_source_parents":sorted(parents),
        "matched_SFT_vs_privileged_text_five_parents":[parent_id(n) for n in TEACHER_UNITS],
        "source136_concise_only":parent_id(136),"tokens":None})
    copied={
        "review_records/commons_rights_decision.json":root/"r57_commons_original_text_route/v1.0/COMMONS_ORIGINAL_TEXT_COMPONENT_ROUTE_DECISION.FINAL.DATA.json",
        "review_records/lineage_overlap_decision.json":root/"r57_public_lineage_overlap/v1.0/FINAL_PUBLIC_OVERLAP_AND_FAMILY_REVIEW.DATA.json",
        "review_records/scan_and_target_readout.md":root/"r57_schema_lineage_requirements/v1.1/READOUT.DATA.md",
        "review_records/exact_math_certificates.json":root/"root_rendered_target_exact_verification/v1.0/EXACT_RENDERED_TARGET_CERTIFICATES.DATA.json",
        "review_records/transcription_and_edits.json":corpus/"TRANSCRIPTION_AND_ALL_EDITS.DATA.json",
        "review_records/coordinate_frame_and_replay.json":root/"root_transcription_coordinate_frame/v1.0/PROSPECTIVE_COORDINATE_FRAME_AND_FULL_REPLAY.DATA.json",
        "review_records/root_scoped_admission.json":root/"root_supplement_admission/v1.0/ROOT_SCOPED_ADMISSION.DATA.json",
        "review_records/science_admission.jsonl":src8/"SOURCE_ADMISSION_DECISIONS.DATA.jsonl",
        "review_records/science_verifications.jsonl":src8/"EXACT_COMPONENT_VERIFICATIONS.DATA.jsonl",
        "review_records/science_lineage.jsonl":src8/"PROSPECTIVE_LINEAGE_AND_STATE_TRANSITIONS.DATA.jsonl"}
    for name,src in copied.items():
        assert src.suffix not in {".py",".pyc",".so"} and src.is_relative_to(root)
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    own=out/"pipeline/root_human_source_adapter.py";own.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,own)
    docs={
        "README.md":"R57 source-derived training supplement for Rineya Titan 0.6B. Six reviewed source parents, ten unique supervised views, five prospective privileged-context packets, eight pre-existing source-only knowledge records. This is a bounded supplement, not the completed substantial balanced v1 corpus. Run pipeline/root_human_source_adapter.py validate --artifact PATH. No model execution or weight updates occur.",
        "DATASET_CARD.md":"Original human Dudeney text, faithful scan transcription with recorded edits. One source-document family. Content checked by separate inherited-model context and actual exact arithmetic/enumeration. Human review pending. All mathematical source descendants remain TRAIN. Eight Wikidata facts retain original335-set identities, null roles and no training admission.",
        "DATA_CONTRACT.md":"Task and knowledge JSONL preserve original strict v1 schemas. Parent, rights, richer answer semantics, teacher, exposure and qualification metadata are linked sidecars. Student builder exposes only the source statement. Teacher privilege is explicit. Response masks are null until the actual tokenizer is supplied. Exact valid alternate card schedules are accepted.",
        "RIGHTS_AND_PROVENANCE.md":"DefaultB scoped US public-domain and UK named-author text basis for exact original human scan components; other jurisdictions unqualified. Commons policy/community prose, institutional marks and source136 general exposition excluded from positive targets. All new model-generated replacement targets remain under the personal-ChatGPT intended-use HOLD. No new CC0 grant is asserted. Exact Commons scan SHA256 f633dba5e776fc0b891c48c54ae2f57a141733c6db2e5723176fc7879ea1bd2c is retained in verified private GitHub R56 release archives; original physical paths remain absent. Research receipts preserve historical route HOLDs unchanged.",
        "QUALITY_AND_DIVERSITY_REPORT.md":"Six distinct source-number problems cover positional decomposition, multi-agent conservation, conditional measurement, weighted rate, minimum representation multiplicity and cyclic scheduling. One source family is concentrated; this does not satisfy balanced scale. Source59 duplicate educational target is omitted. Source136 optional paragraph is not teacher/positive educational content. Other valid schedules are accepted, not rejected as preferences.",
        "LEAKAGE_AND_SPLIT_REPORT.md":"Six exact source-unit IDs and original family retained. Source-family descendants are TRAIN-only. Sixty newer public inputs across29 parents plus eight older science TRAIN inputs were reviewed; no same six Dudeney problem found in that subset. Global novelty and protected evaluator contamination remain unqualified. Five of the eight P1086 facts were explicitly present in older TRAIN corpus inputs; this is corpus exposure, not proof of model training. No protected answer was inspected.",
        "COMPATIBILITY_REPORT.md":"Strict v1 schemas, references, student/teacher separation and actual supplemental adapter validation are tested. The historical native loader, actual Titan0.6B tokenizer, token counts, masks, truncation behavior and numerical learner treatments remain unqualified. Do not use this supplement to claim native Rineya performance or fully qualified corpus-wide integration.",
        "COLLECTION_AND_REVIEW_COSTS.md":"Six source units selected from nine reviewed historical units;37 ambiguity,88 false explanatory lemma,273 general-n proof remain HOLD/rejected. Source16/27/59/67/136/265 have exact source review. Actual262? no fabricated operations: Root checks59049coin assignments,62870bounded square pairs and66partner/66opponent pair identities. Twelve page images independently reviewed,337transcription edits replayed. Model tokens/costs and human hours unmeasured. No paid API.",
        "LIMITATIONS_AND_PENDING_REVIEW.md":"Full300–500design/2000–5000pilot/20000–40000v1 balanced bands not achieved. Full historical semantic/protected evaluator screening, corpus integration, actual tokenizer/native qualification and stratified human review pending. Static source material is not on-policy/adaptive discovery. No empirical student prefixes, probabilities, logits, success or difficulty recorded.",
        "RESEARCH_TO_DATA_MAP.md":"RAKDC-000: exposed source truth, exact verifier, lineage and builder instruments; native teacher/numerical qualification still pending. RAKDC-001: six-parent ordinary source SFT supplement; five-parent matched ordinary/privileged cohort, not empirical verified self-training. RAKDC-002: no interactive discovery corpus supplied here. RAKDC-003: five prospective verified human privileged-context packets, no measured teacher advantage or token scoring. RAKDC-004: arithmetic/rate/constraint skill inventory only; no retention/transfer outcome. RAKDC-005: static source cohort, no adaptive curriculum evidence. RAKDC-006: later eligible static cohort only, no scored online episode. Full seven-study program remains active."}
    docs["COLLECTION_AND_REVIEW_COSTS.md"]=docs["COLLECTION_AND_REVIEW_COSTS.md"].replace("Actual262? no fabricated operations: ","")
    for name,value in docs.items():(out/name).write_text(value+"\n")
    jwrite(out/"human_review_packet/packet.json",{"status":"PENDING","reviewer_signature":None,
        "strata":{"rate_and_scope":[parent_id(59),parent_id(67)],"finite_constraints":[parent_id(27),parent_id(265)],
                  "representation_and_minimum":[parent_id(16),parent_id(136)]},"review_original_question_first":True})
    validate(out,write=True)
    return out

def validate(out,write=False):
    out=pathlib.Path(out);tasks=jlread(out/"data/train/tasks.v1.jsonl");parents={p["id"]:p for p in jlread(out/"data/canonical/parents.sidecar.jsonl")}
    sources={s["id"]:s for s in jlread(out/"manifests/sources.jsonl")};packets=jlread(out/"data/distillation_packets/source_packets.jsonl")
    knowledge=jlread(out/"data/canonical/knowledge.v1.jsonl");ts=jsread(out/"schemas/task_record.schema.json");ks=jsread(out/"schemas/knowledge_record.schema.json")
    invariant(tasks,parents,sources,packets,ts,knowledge,ks)
    for s in sources.values():
        p=out/s["path"];assert p.is_file()
        if "sha256" in s:assert len(p.read_bytes())==s["bytes"] and sha(p.read_bytes())==s["sha256"]
    verifs={v["id"] for v in jlread(out/"review_records/science_verifications.jsonl")}
    assert all(v in verifs for k in knowledge for v in k["verification_ids"])
    assert all(p in parents for sub in jsread(out/"manifests/subsets.json").values() if isinstance(sub,list) for p in sub)
    tests=[]
    def must_reject(name,mut):
        t,p,s,pk,k=copy.deepcopy((tasks,parents,sources,packets,knowledge));mut(t,p,s,pk,k)
        try:invariant(t,p,s,pk,ts,k,ks)
        except (AssertionError,jsonschema.ValidationError,KeyError):
            tests.append({"name":name,"PASS":True});return
        raise AssertionError(name+" did not reject")
    must_reject("gold_in_student",lambda t,p,s,pk,k:t[0].update(prompt=t[0]["prompt"]+"\n"+t[0]["target"]))
    must_reject("cross_role_descendant",lambda t,p,s,pk,k:t[0].update(role="dev"))
    must_reject("schema_extension",lambda t,p,s,pk,k:t[0].update(teacher_context="private"))
    must_reject("broken_source_reference",lambda t,p,s,pk,k:t[0].update(evidence_ids=["missing"]))
    must_reject("wrong_target",lambda t,p,s,pk,k:t[0].update(target="incorrect"))
    must_reject("unadmitted_rights",lambda t,p,s,pk,k:p[t[0]["parent_id"]]["rights"].update(source_admission="HOLD"))
    must_reject("duplicate_parent_different_ID",lambda t,p,s,pk,k:p.update({"newid":copy.deepcopy(next(iter(p.values())))}))
    must_reject("mismatched_teacher_parent",lambda t,p,s,pk,k:pk[0].update(parent_id=parent_id(136)))
    must_reject("fake_empirical_difficulty",lambda t,p,s,pk,k:p[t[0]["parent_id"]].update(empirical_Rineya_difficulty=0.99))
    must_reject("source136_unqualified_teacher",lambda t,p,s,pk,k:pk.append({"parent_id":parent_id(136)}))
    card=next(t["target"] for t in tasks if t["parent_id"]==parent_id(265) and t["id"].endswith("_concise_v1"))
    assert bridge_valid(card.translate(str.maketrans("AB","BA")))
    assert not bridge_valid(card.replace("K E — B F","K F — B F"))
    tests.extend([{"name":"valid_alternate_card_schedule","PASS":True},{"name":"wrong_card_schedule","PASS":True}])
    lengths=[{"task_id":t["id"],"prompt_utf8_bytes":len(t["prompt"].encode()),"target_utf8_bytes":len(t["target"].encode()),
              "prompt_unicode_chars":len(t["prompt"]),"target_unicode_chars":len(t["target"]),"actual_token_count":None,
              "truncation":"NOT_PERFORMED","tokenizer":None} for t in tasks]
    result={"parents":6,"unique_supervised_views":10,"teacher_packets":5,"preference_pairs":0,"empirical_episodes":0,
            "knowledge_source_only":8,"supervised_tokens":None,"native_training_runs":0,"Source_derived_positive_text_only":True,
            "structural_reference_semantic_builder_checks":"PASS","adversarial_tests":tests,"test_count":len(tests),
            "exact_token_or_native_Rineya_qualification":"NOT_AVAILABLE"}
    if write:
        jwrite(out/"validation/EXECUTED_VALIDATION.json",result);jwrite(out/"validation/CHARACTER_AUDIT_NO_TOKEN_CLAIM.json",lengths)
        entries=[]
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name!="SHA256SUMS.txt":entries.append(sha(p.read_bytes())+"  "+str(p.relative_to(out)))
        (out/"SHA256SUMS.txt").write_text("\n".join(entries)+"\n")
    else:
        for line in (out/"SHA256SUMS.txt").read_text().splitlines():
            h,name=line.split("  ",1);p=out/name;assert p.is_file() and sha(p.read_bytes())==h
        assert set(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())=={
            line.split("  ",1)[1] for line in (out/"SHA256SUMS.txt").read_text().splitlines()}|{"SHA256SUMS.txt"}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("build");b.add_argument("--workspace",required=True);b.add_argument("--output",required=True)
    v=sub.add_parser("validate");v.add_argument("--artifact",required=True)
    a=p.parse_args()
    if a.mode=="build":path=build(a.workspace,a.output);print(json.dumps(validate(path),sort_keys=True))
    else:print(json.dumps(validate(a.artifact),sort_keys=True))
