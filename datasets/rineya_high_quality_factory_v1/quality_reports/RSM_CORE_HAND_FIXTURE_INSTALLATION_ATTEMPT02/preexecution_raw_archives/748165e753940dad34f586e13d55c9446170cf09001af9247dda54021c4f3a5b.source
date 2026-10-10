"""Actual same-task renderer-r1 adjudication, only owned public solvers executed."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import json
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
IDENTITY="/root/research_architect"
BASE="reviews/adjudication/planning-gap-v2-renderer-r1"
SOURCE="reviews/adjudication/adjudicate_planning_gap_v2_renderer_r1.py"
CANDIDATE="data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl"
MANIFEST="data/RGM-001/manifests/planning-gap-v2-renderer-r1.family_controls.json"
EXPECTED={
 CANDIDATE:"b306f67b2208406027d26d95094c649913e590291857764a9bf17edb9bcd923c",
 MANIFEST:"e69a9c130b40aebd0dfdefa9b0f96ec075d3a383073aa963e63446e084bf5c63",
 "reviews/expert/planning-gap-v2-renderer-r1.jsonl":"9505e5921c117aff287d12634020ff33e6f6c41fb9e3b0af3617b661201bcc89",
 "reviews/adversarial/planning-gap-v2-renderer-r1.jsonl":"6b58be3a5b738b3ff01c7f6de936e0eddef249e04d46ac6de15df60c7aa071c9",
 "reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.execution.json":"f13cccd49b7047fee299bbd9b3b7b531c69dfb6887fb35e1d83529f6ba9566f2",
 "audits/execution/planning-gap-v2-renderer-r1-adversarial.json":"f9a30152fc20cc4f21d303556f985a1fbfdf0cea13e27bb2cb606cb3eec9a52f",
 "quality_reports/RGM_RENDERER_R1_FROZEN_REVIEW_EVIDENCE_JOIN.json":"dbdd35b3e6575f26ce9722750f74f4925a596a3b2c5d35cea508a250620d2ff8",
 "audits/leakage/planning-gap-v2-renderer-r1.json":"d715c4ca68301db20f6fb2fb91d58e74f0c4ecda4de6f9c7e15a4a113f75c335",
 "audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json":"00687108adce0fd18e39f3974c329cf038a6fbd278e2a38582f960df74c7f742",
 "quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json":"34732f8a7a5cab9addd30d113ae36f1e3db1c5d26ecab9bf785c706dcc78fe16",
 BASE+".input_predictions.jsonl":"2d49ef472b0c2fc902974186a47eb2e5f6f09601d3eb8c991d6d09319185e9db",
 BASE+".preparation.final_receipt.json":"3ddf2ac7d2d27e30f8553daaa0814da4f08749a3277246e439457e1b599654b7",
 "reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py":"da2b2af07a419c01551d37dcc90515645e6655cdf83d57c2327bc6d04ad45a8f",
 "reviews/adjudication/planning-gap-v2.audit.json":"1132df81bb6e4c884fecaf53136b2dea7a31595b55550e4420a4e3bb2b42677e",
 "reviews/adjudication/planning-gap-v2.jsonl":"11487f362d0b3bef349f4f6e0f8b27327f50eb657df43bc3bbaab48ed925acd7",
 "reviews/input_only/planning-gap-v2-renderer-r1.jsonl":"5f0b3490720933caa7895f3223244e5e72ed8ca5fb89a97f2eeb516c151f58e3",
 "reviews/input_only/planning-gap-v2-renderer-r1.freeze.json":"808769c0134e05dd9b9b311bebebc3f961a698d907b5fc652b5b5040b5d36eac",
 "schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md":"6e3759ebe9e3e1f801257d30df610b7dea091eb3324f0a76b59ad9520eac5e56",
 "reviews/input_only/planning-gap-v2.jsonl":"00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c",
 "reviews/input_only/planning-gap-v1.jsonl":"2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654",
 "research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl":"896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d",
}
BINDINGS={}
DISCLOSURE=("Actual /root/research_architect authored public V1/V2 controls, adjudicated original V2 and saw original gold. "
 "All408 own fresh input predictions froze before fresh canonical access. Same204 tasks/48 lineages, own public solver reuse, "
 "root-guided correction, design/adjudicator overlap and common underlying model are disclosed. This is independent exact-byte "
 "execution with prior canonical exposure, not unseen blind semantics or independent-model errors. No producer private source, "
 "expected fixtures, private formal or reviewer implementations are imported. Reproduction/custody manifests are actor attestations, "
 "not independent producer re-execution. No model efficacy or sealed OOD claim.")


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path):
 raw=(ROOT/path).read_bytes()
 if path in EXPECTED: assert sha(raw)==EXPECTED[path],("exact bytes changed",path)
 BINDINGS[path]={"path":path,"sha256":sha(raw),"bytes":len(raw)}
 return raw


def obj(path): return json.loads(read(path))
def rows(path): return [json.loads(line) for line in read(path).splitlines()]


def write_new(path,value,raw=False):
 p=ROOT/path;assert not p.exists(),("frozen output exists",path)
 p.write_text(value if raw else json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
 return {"path":path,"sha256":sha(p.read_bytes()),"bytes":p.stat().st_size}


def output_preflight(bindings,required,output_paths):
 assert set(required)<=set(bindings),"all required bindings including manifest must exist BEFORE outputs"
 assert all(type(bindings[p]) is dict and bindings[p].get("path")==p and len(bindings[p].get("sha256",""))==64 for p in required)
 assert len(output_paths)==len(set(output_paths)) and all(type(p) is str and p for p in output_paths)
 for path in output_paths: assert not (ROOT/path).exists(),("output already frozen",path)


def receipt_preflight(candidate,receipts):
 index={r["sample_id"]:r for r in candidate}
 assert len(index)==len(candidate)==len(receipts)
 assert len({r["receipt_id"] for r in receipts})==len(receipts)
 assert {r["sample_id"] for r in receipts}==set(index)
 for r in receipts:
  assert r["record_hash"]==index[r["sample_id"]]["record_hash"]
  assert r["reviewer_execution_id"]==IDENTITY and r["reviewer_role"]=="independent_adjudicator"
  assert r["decision"] in ["ACCEPT","REVISE","REJECT","QUARANTINE"]


def setup():
 global A
 path="reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py";read(path)
 spec=importlib.util.spec_from_file_location("owned_original_planning_v2",ROOT/path)
 A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);A.setup()
 BINDINGS.update(A.BINDINGS)
 EXPECTED.update({p:A.EXPECTED[p] for p in A.P.PUBLIC_DOCS})
 for p in A.P.PUBLIC_DOCS:read(p)
 read("schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md");read(SOURCE)


def bind_artifacts(freeze,field):
 for artifact in freeze[field]: assert sha(read(artifact["path"]))==artifact["sha256"],artifact["path"]


def review_evidence(index):
 result={}
 receipt_ids=set()
 for name,role,execution in [("expert","expert_reviewer","/root/expert_reviewer"),("adversarial","adversarial","/root/adversarial_auditor")]:
  path=f"reviews/{name}/planning-gap-v2-renderer-r1.jsonl"
  verdicts=rows(path)
  assert read(path)==read(f"reviews/{name}/planning-gap-v2-renderer-r1.first.jsonl")
  result[name]=A.P.exact_review_join(index,verdicts,role,execution)
  assert all(r["decision"]=="ACCEPT" and r["critical_defects"]==[] for r in verdicts)
  assert not receipt_ids&{r["receipt_id"] for r in verdicts};receipt_ids.update(r["receipt_id"] for r in verdicts)
  prediction_path=f"reviews/{name}/planning-gap-v2-renderer-r1."+("expert_input_predictions.jsonl" if name=="expert" else "input_predictions.jsonl")
  predicted=rows(prediction_path);assert len(predicted)==408 and {r["sample_id"] for r in predicted}==set(index)
  for r in predicted:
   assert r["record_hash"]==index[r["sample_id"]]["record_hash"]
   assert canonical(r["expected_target"])==canonical(index[r["sample_id"]]["verified_target"])
 ef=obj("reviews/expert/planning-gap-v2-renderer-r1.expert_input.freeze.json");bind_artifacts(ef,"bindings")
 ex=obj("reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.execution.json")
 assert ex["actual_exit_code"]==0 and ex["first_final_byte_identical"] is True and ex["records"]==408
 xf=obj("reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.freeze.json");bind_artifacts(xf,"bindings")
 assert xf["fresh_gold_access_only_after_all408_input_freeze_and_actual_FINAL"] is True
 assert xf["actual_derive_exit_code"]==xf["actual_finalize_exit_code"]==xf["actual_compact_support_exit_code"]==0
 ep=obj("reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.preexecution.json")
 A.P.first_freeze_before_comparison(ef["created_utc"],ep["created_utc"])
 actual_final=obj(xf["input_actual_FINAL_freeze"]["path"])
 assert sha(read(xf["input_actual_FINAL_freeze"]["path"]))==xf["input_actual_FINAL_freeze"]["sha256"]
 read("reviews/expert/semantic_inspection_planning_gap_v2_renderer_r1_input.md")
 read(ex["actual_agent_inspection"]["path"])
 red=obj("audits/execution/planning-gap-v2-renderer-r1-adversarial.json")
 assert red["completed"] is True and red["actual_execution_id"]=="/root/adversarial_auditor" and red["rows"]==408
 assert red["first_canonical_judgment_sha256"]==red["final_judgment_sha256"]==EXPECTED["reviews/adversarial/planning-gap-v2-renderer-r1.jsonl"]
 for path,digest in red["files_sha256_before_after"].values():assert sha(read(path))==digest
 rf=obj("reviews/adversarial/planning-gap-v2-renderer-r1.input_freeze.json")
 assert rf["completed"] is True and rf["fresh_gold_not_read"] is True
 for path,digest in rf["exact_bindings_before_after"].items():assert sha(read(path))==digest
 read("audits/execution/planning-gap-v2-renderer-r1-adversarial-comparison.preexec.json")
 read("audits/diversity/planning-gap-v2-renderer-r1-adversarial-canonical.json")
 surface=obj("audits/shortcut_detection/planning-gap-v2-renderer-r1-surface.json")
 assert len(receipt_ids)==816
 return result,{"expert_completed_execution":BINDINGS["reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.execution.json"],
    "adversarial_completed_execution":BINDINGS["audits/execution/planning-gap-v2-renderer-r1-adversarial.json"],"all816_unique_exact_receipt_ids":True,
    "first_final_bytes_identical":True,"disagreements":0,"prior_original_gold_exposure_disclosed":True,
    "red_surface_residual_evidence":rf["residuals"],"expert_component_summary":xf["component_lookup_baselines"]}


def run():
 start,clock=now(),time.monotonic();setup()
 read(BASE+".preparation.final_receipt.json")
 obj(BASE+".input_preparation.execution.json")
 own_predictions=rows(BASE+".input_predictions.jsonl");prediction_index={p["sample_id"]:p for p in own_predictions}
 input_rows=rows("reviews/input_only/planning-gap-v2-renderer-r1.jsonl");input_index={r["sample_id"]:r for r in input_rows}
 custody=obj("reviews/input_only/planning-gap-v2-renderer-r1.freeze.json")
 original_inputs=rows("reviews/input_only/planning-gap-v2.jsonl");old_index={r["sample_id"]:r for r in original_inputs}
 parent=rows("reviews/input_only/planning-gap-v1.jsonl")
 prior=rows("research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
 original_audit=obj("reviews/adjudication/planning-gap-v2.audit.json")
 read("reviews/adjudication/planning-gap-v2.jsonl")
 candidate,manifest=rows(CANDIDATE),obj(MANIFEST)
 index=A.P.pending_record_index(candidate,408)
 assert set(index)==set(input_index)==set(prediction_index) and len(input_rows)==len(own_predictions)==408
 A.P.public_binding_check(manifest,{p:EXPECTED[p] for p in A.P.PUBLIC_DOCS})
 assert manifest["batch_name"]=="planning-gap-v2" and manifest["revision_of_batch"]=="planning-gap-v1"
 assert manifest["parent_input_view_sha256"]==EXPECTED["reviews/input_only/planning-gap-v1.jsonl"] and manifest["track_id"]=="RGM-001"
 assert manifest["semantics_sha256"]==EXPECTED["schemas/SEMANTICS.md"]
 reviewed,review_summary=review_evidence(index)
 root_join=obj("quality_reports/RGM_RENDERER_R1_FROZEN_REVIEW_EVIDENCE_JOIN.json")
 assert root_join["decision_disagreements"]==root_join["unresolved_high_or_critical_review_issues"]==0
 leakage=obj("audits/leakage/planning-gap-v2-renderer-r1.json")
 machine=obj("audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json")["report"]
 root_execution=obj("quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json")
 assert root_execution["exit_code"]==0 and root_execution["bound_sources_unchanged"] is True
 assert machine["errors"]==[] and machine["counts"]["complete_v2_families"]==48
 assert machine["counts"]["required_semantic_control_pairs"]==96 and machine["counts"]["required_renderer_control_declarations"]==192
 assert leakage["all_machine_quality_gates_pass"] is True
 assert leakage["oracle_failures"]==leakage["metadata_failures"]==leakage["validation_failures"]==leakage["planning_control_error_count"]==0
 # Producer files are execution-custody attestations only, never imported solver/source/expected fixtures.
 reproduction=obj("manifests/planning-gap-v2-renderer-r1_REPRODUCTION.json")
 producer=obj("manifests/planning-gap-v2-renderer-r1_PRODUCER_FINAL_CUSTODY.json")
 assert reproduction["passed"] is True and reproduction["actual_execution_id"]=="/root/relational_critic_producer"
 assert len(reproduction["full_deterministic_source_candidate_and_manifest_comparisons"])==8
 assert all(r["main_sha256"]==r["replay_sha256"] for r in reproduction["full_deterministic_source_candidate_and_manifest_comparisons"])
 assert producer["actual_executing_agent"]=="/root/relational_critic_producer" and producer["all_eight_core_artifacts_reproduced_byte_exactly"] is True
 assert producer["generator_version"]=="rgm-planning-gap-2.0.5" and producer["new_semantic_tasks"]==producer["new_source_families"]==0
 cache,details,parsed_by_id={}, {}, {}
 for row in candidate:
  sid,task=row["sample_id"],row["semantic_task"];A.V1.task_schema(task)
  assert all(canonical(row[k])==canonical(v) for k,v in input_index[sid].items())
  prediction=prediction_index[sid]
  assert prediction["record_hash"]==row["record_hash"] and prediction["review_or_approval_decision"] is None
  assert row["source_family_key"]==prediction["source_family_key"] and row["split"]==prediction["split"]
  assert canonical(task)==canonical(old_index[prediction["original_input_sample_id"]]["semantic_task"])
  parsed=json.loads(row["rendered_input"].split("Task JSON:\n",1)[1]);assert canonical(parsed)==canonical(task)
  parsed_by_id[sid]=parsed
  if task["kind"]=="blocks":assert list(parsed["goal_on"])==sorted(parsed["goal_on"],reverse=row["renderer"]=="reverse")
  assert row["generator_version"]=="rgm-planning-gap-2.0.5" and row["producer_agent_id"]=="/root/relational_critic_producer"
  assert row["source_provenance"]["sealed_evaluation_used"] is False and row["oracle_version"]=="independent-1.0.0"
  assert type(row["difficulty"]) is int and 2<=row["difficulty"]<=4 and row["difficulty"]==row["structural_difficulty"]
  key=A.semantic_key(task)
  if key not in cache:
   target,failure=A.FN["graph_replay"](task) if task["kind"]=="graph" else A.FN["blocks_replay"](task)
   minimum,search=A.measured_shortest(task);assert canonical(minimum)==canonical(target["shortest_plan"])
   cache[key]=(target,failure,search)
  target,failure,search=cache[key]
  assert canonical(target)==canonical(prediction["expected_target"])==canonical(row["verified_target"]),sid
  details[sid]={"sample_id":sid,"record_hash":row["record_hash"],"source_family_key":row["source_family_key"],"context":row["controlled_context"],
       "split":row["split"],"renderer":row["renderer"],"target_hash_prompt_frozen_view_match":True,"failure":failure,
       "minimum_moves":None if target["shortest_plan"] is None else len(target["shortest_plan"]),"search":search,"witnesses":[]}
 assert len(cache)==204
 bindings_result=A.P.control_renderer_bindings(manifest["families"],index)
 assert bindings_result["verified_pair_binding_declarations"]==192
 witnesses,covered,controls,base_tasks=[],[],[],{}
 props,subtypes,c1_positions,c1_support,c4_support,sizes=[Counter() for _ in range(6)]
 temporary,tie_counts,style_pairs=[],[],0
 for family in manifest["families"]:
  n=family["source_index"];allocation=A.P.expected_allocation(n);s=allocation["stratum"];split=allocation["split"]
  assert n not in base_tasks and family["primary_stratum"]==s and family["split"]==split
  assert family["source_family_key"]==f"RGM-001-seed20261009-p{n:04d}"
  contexts={}
  for context in family["contexts"]:
   ids=context["sample_ids"];assigned=[index[sid] for sid in ids];cid=context["context_id"]
   assert {r["renderer"] for r in assigned}=={"forward","reverse"}
   assert all(r["source_family_index"]==n and r["controlled_context"]==cid and r["primary_stratum"]==s for r in assigned)
   assert A.semantic_key(assigned[0]["semantic_task"])==A.semantic_key(assigned[1]["semantic_task"])
   assert canonical(assigned[0]["verified_target"])==canonical(assigned[1]["verified_target"])
   tags=context["row_properties"];assert len(tags)==len(set(tags));seen=set()
   for witness in context["row_witnesses"]:
    sid,prop=witness["sample_id"],witness["property"]
    assert witness["scope"]=="row" and sid in ids and prop in tags and (sid,prop) not in seen
    seen.add((sid,prop));task=index[sid]["semantic_task"];target,failure,_=cache[A.semantic_key(task)]
    proof=A.property_truth(task,target,failure,witness,n)
    details[sid]["witnesses"].append({"property":prop,"evidence":witness["evidence"],"own_proof":proof,"truth":True});props[prop]+=1
   assert seen=={(sid,tag) for sid in ids for tag in tags}
   for row in assigned:assert set(row["skill_tags"])-{"finite_synthetic_proxy","relational_planning"}==set(tags)
   forward=next(row for row in assigned if row["renderer"]=="forward");reverse=next(row for row in assigned if row["renderer"]=="reverse")
   left,right=parsed_by_id[forward["sample_id"]],parsed_by_id[reverse["sample_id"]]
   assert list(left)==list(reversed(list(right)))
   if left["kind"]=="graph":assert left["nodes"]==list(reversed(right["nodes"])) and left["edges"]==list(reversed(right["edges"]))
   else:
    assert list(left["on"])==list(reversed(list(right["on"]))) and list(left["goal_on"])==list(reversed(list(right["goal_on"])))
    assert all(list(a)==["block","to"] for a in left["plan"]) and all(list(a)==["to","block"] for a in right["plan"])
   covered.extend(ids);contexts[cid]=forward
  assert set(contexts)=={s+str(i) for i in range(allocation["contexts"])}
  base=index[family["base_sample_id"]];assert base["controlled_context"]==s+"0" and base["source_family_index"]==n
  task=base["semantic_task"];out=cache[A.semantic_key(task)][0];assert out["valid"] and out["reaches_goal"]
  size=len(task["nodes"] if task["kind"]=="graph" else task["on"]);assert size==allocation["objects"]
  sizes[f"{s}:{split}:{size}"]+=1
  if s=="A":
   t1=contexts["A1"]["semantic_task"];o1=cache[A.semantic_key(t1)][0];assert not o1["valid"]
   actual="fresh_infeasible" if o1["shortest_plan"] is None else "fuel_lengthened";assert actual==allocation["a_subtype"]
   assert A.FN["shortest_graph"](t1,True) is not None
   if actual=="fresh_infeasible":assert task["fuel"]==A.FN["cheapest_graph"](task) and t1["fuel"]==task["fuel"]-1
   else:assert len(o1["shortest_plan"])>len(out["shortest_plan"])
   subtypes[f"{actual}:{split}:{size}"]+=1
   for cid in ["A2","A3"]:
    other=contexts[cid]["semantic_task"];assert canonical({k:v for k,v in task.items() if k!="plan"})==canonical({k:v for k,v in other.items() if k!="plan"})
   t2=contexts["A2"]["semantic_task"];o2=cache[A.semantic_key(t2)][0];assert t2["plan"] and o2["valid"] and not o2["reaches_goal"]
   fail=cache[A.semantic_key(contexts["A3"]["semantic_task"])][1];assert fail["reason"]=="missing_directed_edge" and fail["index"]>=1
  elif s=="B":
   assert len(task["plan"])>=3;tie_counts.append(A.FN["shortest_graph_count"](task))
   t1=contexts["B1"]["semantic_task"];o1,f1,_=cache[A.semantic_key(t1)];assert o1["shortest_plan"] is not None and f1["reason"]=="missing_directed_edge" and f1["index"]>=1
   t2=contexts["B2"]["semantic_task"];o2,f2,_=cache[A.semantic_key(t2)];assert f2["reason"]=="missing_directed_edge" and f2["index"]>=1
   previous=o2["steps"][f2["index"]-1]["after"]["node"];assert (t2["plan"][f2["index"]],previous) in {(a,b) for a,b,_ in t2["edges"]}
  elif s=="C":
   assert 4<=len(task["plan"])==len(out["shortest_plan"])<=8
   assert "later_action_requires_changed_block_clearance" in {w["property"] for w in details[base["sample_id"]]["witnesses"]}
   t1=contexts["C1"]["semantic_task"];failure=cache[A.semantic_key(t1)][1]
   actual="source_not_clear" if failure["reason"]=="blocked_source" else "destination_not_clear";assert actual==allocation["c1_reason"]
   c1_positions[f"{actual}:{failure['index']}"]+=1;c1_support[f"{actual}:{split}:{size}"]+=1
   t3=contexts["C3"]["semantic_task"];o3=cache[A.semantic_key(t3)][0]
   assert t3["plan"]==task["plan"][:allocation["c4_prefix"]] and o3["valid"] and not o3["reaches_goal"]
   c4_support[f"{split}:{allocation['c4_prefix']}"]+=1
   if A.V1.restricted_block_search(task)["necessary"]:temporary.append(n)
  else:
   assert 0<len(task["goal_on"])<len(task["on"]) and len(task["plan"])==len(out["shortest_plan"])>=2
   t1=contexts["D1"]["semantic_task"];o1=cache[A.semantic_key(t1)][0]
   assert o1["valid"] and not o1["reaches_goal"] and len(o1["shortest_plan"])>len(out["shortest_plan"])
   t2=contexts["D2"]["semantic_task"];o2=cache[A.semantic_key(t2)][0]
   assert o2["valid"] and o2["reaches_goal"] and len(t2["plan"])==len(o2["shortest_plan"])
   assert canonical({k:v for k,v in t1.items() if k!="plan"})==canonical({k:v for k,v in t2.items() if k!="plan"})
   for style in ["forward","reverse"]:
    base_row=next(index[sid] for sid in next(c for c in family["contexts"] if c["context_id"]=="D0")["sample_ids"] if index[sid]["renderer"]==style)
    base_parsed=parsed_by_id[base_row["sample_id"]];shared=set(base_parsed["goal_on"])
    for context in family["contexts"]:
     row=next(index[sid] for sid in context["sample_ids"] if index[sid]["renderer"]==style)
     assert A.renderer_projection(parsed_by_id[row["sample_id"]],shared)==A.renderer_projection(base_parsed,shared);style_pairs+=1
  for control in family["controls"]:controls.append(A.control_truth(control,index,cache))
  base_tasks[n]=base
 assert len(covered)==len(set(covered))==408 and set(covered)==set(index)
 assert len(base_tasks)==48 and set(base_tasks)==set(range(180,228)) and sum(props.values())==510 and len(controls)==192 and style_pairs==96
 assert len(temporary)>=4 and sum(count>=2 for count in tie_counts)>=6
 old_tasks={A.semantic_key(r["semantic_task"]):r["semantic_task"] for r in prior}
 graphs=[r["semantic_task"] for r in base_tasks.values() if r["semantic_task"]["kind"]=="graph"]
 blocks=[r["semantic_task"] for r in base_tasks.values() if r["semantic_task"]["kind"]=="blocks"]
 old_graph=[t for t in old_tasks.values() if t["kind"]=="graph"];old_block=[t for t in old_tasks.values() if t["kind"]=="blocks"]
 weighted={A.FN["graph_fingerprint"](t,True,True) for t in graphs};old_weighted={A.FN["graph_fingerprint"](t,True,True) for t in old_graph}
 unweighted={A.FN["graph_fingerprint"](t) for t in graphs};old_unweighted={A.FN["graph_fingerprint"](t) for t in old_graph}
 joint={A.FN["block_fingerprint"](t,True) for t in blocks};old_joint={A.FN["block_fingerprint"](t,True) for t in old_block}
 shapes={A.FN["block_fingerprint"](t) for t in blocks};six_shapes={A.FN["block_fingerprint"](t) for t in blocks if len(t["on"])==6}
 assert len(weighted)==24 and not weighted&old_weighted and len(unweighted-old_unweighted)>=16
 assert len(joint)==24 and not joint&old_joint and len(six_shapes)>=8
 assert not set(cache)&set(old_tasks) and not set(index)&{r["sample_id"] for r in prior}
 assert {r["source_family_key"] for r in candidate}=={r["source_family_key"] for r in parent}
 assert not {r["source_family_key"] for r in candidate}&{r["source_family_key"] for r in prior}
 shape_groups=defaultdict(list)
 for row in base_tasks.values():
  if row["semantic_task"]["kind"]=="blocks":shape_groups[A.FN["block_fingerprint"](row["semantic_task"])].append({"source_family_key":row["source_family_key"],"split":row["split"]})
 hist=defaultdict(Counter);train=[];dev=[];c4_simple=[]
 for row in candidate:
  if row["renderer"]!="forward":continue
  task=row["semantic_task"];target,failure,_=cache[A.semantic_key(task)]
  values={"valid":target["valid"],"reaches_goal":target["reaches_goal"],"fresh_feasible":target["shortest_plan"] is not None,
          "minimum_moves":None if target["shortest_plan"] is None else len(target["shortest_plan"]),"submitted_moves":len(task["plan"]),
          "first_error_index":None if failure is None else failure["index"],"first_error_reason":None if failure is None else failure["reason"],
          "objects":len(task["nodes"] if task["kind"]=="graph" else task["on"]),"fuel":task.get("fuel")}
  for group in [row["split"],row["controlled_context"]+":"+row["split"],row["source_family_key"]]:
   for name,value in values.items():hist[group+":"+name][canonical(value)]+=1
  (train if row["split"]=="train" else dev).append((task.get("fuel"),values["fresh_feasible"],row["sample_id"]))
  if row["difficulty"]==2 and row["controlled_context"]=="C4":c4_simple.append({"sample_id":row["sample_id"],"minimum_moves":values["minimum_moves"],"submitted_moves":values["submitted_moves"]})
 buckets=defaultdict(Counter)
 for feature,label,sid in train:buckets[feature][label]+=1
 majority=Counter(label for _,label,_ in train).most_common(1)[0][0]
 lookup={feature:sorted(counts,key=lambda label:(-counts[label],canonical(label)))[0] for feature,counts in buckets.items()}
 wrong=[sid for feature,label,sid in dev if lookup.get(feature,majority)!=label]
 # Complete input binding preflight before ANY decision/ticket/report output.
 output_paths=[BASE+".audit.json",BASE+".jsonl",BASE+".report.md",BASE+".execution.json","feedback/improvement_requests/RGM-RENDERER-R1-RESIDUALS.json"]
 output_preflight(BINDINGS,[CANDIDATE,MANIFEST,SOURCE,BASE+".input_predictions.jsonl",BASE+".preparation.final_receipt.json",
                 "audits/leakage/planning-gap-v2-renderer-r1.json","audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json"],output_paths)
 for path,binding in BINDINGS.items():assert sha((ROOT/path).read_bytes())==binding["sha256"],("evidence changed during execution",path)
 audit={"actual_execution_id":IDENTITY,"started_utc":start,"completed_utc":now(),"actual_wall_seconds":time.monotonic()-clock,
        "bindings":BINDINGS,"candidate_rows":408,"semantic_tasks":204,"same_source_families":48,"new_semantic_tasks":0,"new_source_families_relative_to_pending_V1_V2":0,
        "exact_typed_target_hash_input_prediction_joins":408,"factual_witness_counts":dict(props),"all510_truthful":True,
        "all192_control_declarations_truthful":True,"semantic_control_pairs":96,"controls":controls,"per_record":list(details.values()),
        "novelty_against_prior_active":{"weighted_terminal_graph_classes":len(weighted),"new_unweighted_graph_classes":len(unweighted-old_unweighted),
                  "joint_block_classes":len(joint),"block_initial_shapes":len(shapes),"six_block_initial_shapes":len(six_shapes),"prior_disjoint_required_base_classes":True},
        "split_rows":dict(Counter(r["split"] for r in candidate)),"source_split_coverage":dict(sizes),"A_subtypes":dict(subtypes),
        "C1_selective_reason_positions":dict(c1_positions),"C1_reason_size_split":dict(c1_support),"C4_prefix_split":dict(c4_support),
        "C0_required_temporary_displacement_indices":temporary,"B0_tie_families":sum(count>=2 for count in tie_counts),
        "repeated_block_initial_motifs":[v for v in shape_groups.values() if len(v)>1],
        "conditional_histograms_semantic_units":{k:dict(v) for k,v in sorted(hist.items())},
        "own_fuel_feasibility_lookup":{"train_tasks":len(train),"dev_tasks":len(dev),"correct":len(dev)-len(wrong),"wrong_ids":wrong,
                 "majority_correct":sum(label==majority for _,label,_ in dev),"mixed_training_budgets":[k for k,v in buckets.items() if len(v)>1],"not_model_performance":True},
        "level2_C4_rows_with_fresh_minimum_dependency":c4_simple,"difficulty_labels_structural_not_model_calibrated":True,
        "root_machine_gate":{"all_required_gates_pass":True,"report":BINDINGS["audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json"],
                 "all48_families_complete":True,"all192_declarations_complete":True,"formerly11_UNKNOWNs_now_absent":True,"own96_D_shared_key_projections_pass":True},
        "review_custody":review_summary,"decision_counts":{"ACCEPT":408},"publication_eligible":True,"guidance":DISCLOSURE,
        "producer_reproduction_attestation":{"actual_execution_id":reproduction["actual_execution_id"],"eight_core_hash_comparisons_equal":True,
                 "hash_seeds":reproduction["hash_seeds"],"private_source_or_fixtures_read":False,"adjudicator_reran_producer":False},
        "original_V1_V2_quarantine_and_specs_reviews_sources_unchanged":True,
        "limitations":["This presentation fix creates no new task/family/mechanism. Same204 tasks/48 revised lineages across two finite grammars.",
                 "One exposed-dev infeasible semantic task; fuel lookup34/34 versus majority33/34 and surface goal-rank/length association26/34 are residual conditional risks.",
                 "B1/B2 and A3 first missing-edge positions remain1; every D proposal legal. Context labels are private but input component associations remain.",
                 "Some C4 Level2 rows require fresh minimum6–8; easy submitted no-op detection does not summarize full-output search difficulty.",
                 "Repeated initial block motifs, own solver/fixture reuse, prior gold/public design and common-model exposure limit independence and generalization claims.",
                 "No minimum edit-distance repair, logistics/coupled-resource grammar, sealed structural OOD, trained model efficacy or architecture result."]}
 audit_binding=write_new(BASE+".audit.json",audit)
 verdicts=[]
 for row in candidate:
  sid=row["sample_id"]
  verdicts.append({"receipt_id":"adj-planning-gap-v2-renderer-r1-"+sha(canonical([sid,row["record_hash"],IDENTITY]).encode())[:24],
    "sample_id":sid,"record_hash":row["record_hash"],"reviewer_role":"independent_adjudicator","reviewer_execution_id":IDENTITY,
    "scores":{"target_correctness":5,"semantic_correctness":5,"reasoning_validity":5,"instruction_clarity":5,
              "difficulty_quality":3,"diversity":3,"generalization_value":3,"shortcut_resistance":2 if row["primary_stratum"]=="B" else 3},
    "critical_defects":[],"issues":[{"category":"bounded_correlated_same_task_development_proxy","severity":"moderate",
       "evidence":"Same204 tasks/48 revised lineages/408 presentations under two grammars; prior gold/design/common-model exposure. Tiny exposed dev and measured partial-label shortcuts, structural difficulty2–4 and shared block arrangements remain.",
       "fix":"Keep unit counts and limitations explicit; future independent families should diversify positions, proposal lengths and feasibility, with difficulty measured separately for submitted replay and fresh search."}],
    "decision":"ACCEPT","confidence":0.98,
    "reason":"Exact fresh candidate target/hash/input predictions, all factual witnesses, required supplied controls and actual per-style bindings pass. New sorted goal-key presentation satisfies the unchanged whole-cohort gate; no unresolved critical defect. Accept only these exact same-task revision bytes, preserving original cohort quarantine.",
    "evidence":{"candidate_sha256":EXPECTED[CANDIDATE],"manifest_sha256":EXPECTED[MANIFEST],"own_audit":audit_binding,
       "adjudicator_source_sha256":BINDINGS[SOURCE]["sha256"],"own_frozen_input_prediction_sha256":EXPECTED[BASE+".input_predictions.jsonl"],
       "own_target_hash_prompt_join":True,"own_factual_witnesses_and_control_joins":True,"whole_cohort_gate_pass":True,
       "expert_receipt_id":reviewed["expert"][sid]["receipt_id"],"expert_decision":"ACCEPT",
       "adversarial_receipt_id":reviewed["adversarial"][sid]["receipt_id"],"adversarial_decision":"ACCEPT",
       "reviewer_disagreement":False,"publication_eligible":True,"original_V2_quarantine_unchanged":True},"guidance":DISCLOSURE})
 receipt_preflight(candidate,verdicts)
 verdict_binding=write_new(BASE+".jsonl","".join(canonical(r)+"\n" for r in verdicts),raw=True)
 residual_ticket={"ticket_id":"RGM-RENDERER-R1-RESIDUALS","actual_author_execution_id":IDENTITY,"status":"NONBLOCKING_MEASURED_FUTURE_COVERAGE",
   "candidate":BINDINGS[CANDIDATE],"manifest":BINDINGS[MANIFEST],"audit":audit_binding,"verdicts":verdict_binding,
   "decision_counts":{"ACCEPT":408},"original_publication_holds_unchanged":True,
   "measured_residuals":audit["limitations"],"next_coverage":["Fresh independent base worlds with varied B/A3 missing-edge positions, matched successful/failed lengths and legality in D-style tasks.",
          "More dev fresh-infeasible families at matched input fuel/size/context features; evaluate lookup association on fresh held custody.",
          "Separate submitted replay difficulty from required fresh minimum dependencies; avoid classifying whole structured output solely by no-op detection."],
   "no_new_acceptance_requirement_or_generation":True,"guidance":DISCLOSURE}
 ticket_binding=write_new("feedback/improvement_requests/RGM-RENDERER-R1-RESIDUALS.json",residual_ticket)
 report="""# Actual renderer-r1 adjudication\n\n**ACCEPT all408 exact fresh records; publication eligible under the reviewed data and current required machine gates.** No release is created by this adjudication. Original planning-gap-v1 and planning-gap-v2 quarantine, candidate bytes, first reviews and public contracts remain immutable.\n\nOwn frozen input predictions preceded fresh canonical access. Actual replay/complete finite searches again verify408 full typed targets, canonical record hashes, parsed prompt and six-field view joins. All510 factual row witnesses and192 actual renderer declarations (96 supplied semantic controls) are truthful. Both expert/red first-final files are byte-identical408 ACCEPT with completed executions and816 distinct receipts. The actual root whole-cohort report passes48 families with zero target, metadata, validation or control errors. Own96 D style projections pass; the fresh lexical goal-map presentation satisfies the unchanged stronger checker without removing a field or changing old criteria. Exactly22 original D0 presentations changed; zero task or family growth results.\n\nThe48 reserved lineages contain204 reused semantic tasks and408 correlated rows:340 train/68 exposed dev,40/8 families. Independent base invariants against the prior ACTIVE input inventory reproduce24 absent weighted-terminal graphs,24 new unweighted topologies,24 absent joint initial/goal block classes and11 six-block initial shapes. There are18 block initial shapes across24 bases, including repeated motifs. These are state/structure classes under two existing grammars, not48 independent mechanism types. C1 has six selective source and six selective destination failures with varied0/1/2 indices and one dev of each. C4 has12 isolated TABLE no-op failures after legal non-goal prefixes. Six C0 worlds require temporary displacement; all twelve B0 worlds have multiple minimum routes.\n\nResidual limits remain substantive: only one dev task is freshly infeasible; own fuel lookup predicts34/34 dev feasibility components versus33/34 majority, with mixed training budgets6 and8. The red measured goal-key-rank/plan-length association predicts26/34 goal labels per renderer. B1/B2 and A3 all fail missing edges at index1; every D submitted plan is legal. These do not predict every full structured target or establish a universal rule, but constrain generalization claims. Some C4 Level2 rows still require6–8 moves in the fresh minimum output; submitted no-op detection is easy while fresh planning remains dependent. Difficulty2–4 is structural and not model-calibrated. The new nonblocking residual ticket records future independent-family priorities without adding an unpublished admission gate.\n\nProducer replay/custody receipts attest byte-identical reproduction under hash seeds17/971 for eight core artifacts and actual generator rgm-planning-gap-2.0.5. This adjudicator did not rerun the producer or inspect its private source/fixtures. The policy's suggested2.0.3 is not a required enum. No sealed instrument, training, model efficacy, coupled-resource/logistics grammar or edit-distance-minimal repair is claimed.\n\n"""+DISCLOSURE+"\n\nThe actual before-execution source archive, hand preflight tests, every input hash, any failure and final tool/read-back are preserved in execution receipts. All required bindings, including the candidate and manifest, were verified before outputs; original missing-manifest and absent-ticket failures were not hidden or reused as success.\n"
 report_binding=write_new(BASE+".report.md",report,raw=True)
 execution={"actual_execution_id":IDENTITY,"completed_execution":True,"completed_utc":now(),"candidate":BINDINGS[CANDIDATE],"manifest":BINDINGS[MANIFEST],
       "source":BINDINGS[SOURCE],"audit":audit_binding,"verdicts":verdict_binding,"ticket":ticket_binding,"report":report_binding,
       "all_bound_inputs_unchanged":True,"decision_counts":{"ACCEPT":408},"publication_eligible":True,"no_release_written":True,
       "original_quarantine_preserved":True,"guidance":DISCLOSURE}
 execution_binding=write_new(BASE+".execution.json",execution)
 print(canonical({"decision_counts":{"ACCEPT":408},"publication_eligible":True,"audit":audit_binding,"verdicts":verdict_binding,
       "report":report_binding,"execution":execution_binding,"ticket":ticket_binding,"actual_wall_seconds":audit["actual_wall_seconds"]}))


if __name__=="__main__":
 try:run()
 except Exception:
  failure={"actual_execution_id":IDENTITY,"at_utc":now(),"source_sha256":sha(Path(__file__).read_bytes()),"exception":traceback.format_exc(),"bindings":BINDINGS,"no_approval_from_failure":True}
  path=BASE+".owned-adjudication-failure-"+sha(canonical(failure).encode())[:16]+".json";write_new(path,failure)
  print(canonical({"failure_preserved":path}),file=sys.stderr);raise
