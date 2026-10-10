"""Same-task r1 input-only preparation, with prior original gold exposure disclosed.

No fresh candidate, manifest, gold, producer/formal source, test gold or fresh verdict read.
No data approval. Original immutable audit/adjudication and public helpers are reused.
"""
from pathlib import Path
from collections import Counter, defaultdict
import copy
import datetime
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
EXECUTION = "/root/research_architect"
BASE = "reviews/adjudication/planning-gap-v2-renderer-r1"
SOURCE = BASE + "-input-preparation.py"
INPUT = "reviews/input_only/planning-gap-v2-renderer-r1.jsonl"
FREEZE = "reviews/input_only/planning-gap-v2-renderer-r1.freeze.json"
PARENT = "reviews/input_only/planning-gap-v2.jsonl"
POLICY = "schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md"
EXPECTED = {
    INPUT:"5f0b3490720933caa7895f3223244e5e72ed8ca5fb89a97f2eeb516c151f58e3",
    FREEZE:"808769c0134e05dd9b9b311bebebc3f961a698d907b5fc652b5b5040b5d36eac",
    PARENT:"00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c",
    POLICY:"6e3759ebe9e3e1f801257d30df610b7dea091eb3324f0a76b59ad9520eac5e56",
    "reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py":"da2b2af07a419c01551d37dcc90515645e6655cdf83d57c2327bc6d04ad45a8f",
    "reviews/adjudication/planning-gap-v2.audit.json":"1132df81bb6e4c884fecaf53136b2dea7a31595b55550e4420a4e3bb2b42677e",
    "reviews/adjudication/planning-gap-v2.jsonl":"11487f362d0b3bef349f4f6e0f8b27327f50eb657df43bc3bbaab48ed925acd7",
    "reviews/input_only/planning-gap-v1.jsonl":"2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654",
    "research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl":"896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d",
}
BINDINGS = {}
DISCLOSURE = (
    "Actual /root/research_architect authored original V1/V2 public controls and adjudicated "
    "the original canonical V2 batch, including original gold. This fresh r1 preparation is "
    "same-task retrospective input verification, not unseen blind semantic review. Own original "
    "public-derived replay/search is reused; exact old/new input joins precede target reuse. "
    "Root guided renderer correction and authorized the original six-field view. Shared model, "
    "design/adjudicator overlap and prior gold exposure remain. No fresh canonical candidate, "
    "manifest, stored gold, producer/private formal implementation or fresh reviewer verdict read. "
    "Independent execution does not establish independent-model errors or efficacy."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"),ensure_ascii=False,allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    raw=(ROOT/path).read_bytes()
    if path in EXPECTED:
        assert sha(raw)==EXPECTED[path], ("immutable hash mismatch",path)
    BINDINGS[path]={"path":path,"sha256":sha(raw),"bytes":len(raw)}
    return raw


def obj(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines()]


def write_new(path,value,raw=False):
    p=ROOT/path
    assert not p.exists(), ("immutable output already exists",path)
    p.write_text(value if raw else json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+"\n")
    return {"path":path,"sha256":sha(p.read_bytes()),"bytes":p.stat().st_size}


def setup():
    global A
    path="reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py"
    read(path)
    spec=importlib.util.spec_from_file_location("owned_original_v2",ROOT/path)
    A=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    A.setup()  # Only its own frozen public-derived helper definitions; never main/audit/finalize.
    BINDINGS.update(A.BINDINGS)
    EXPECTED.update({p:A.EXPECTED[p] for p in A.P.PUBLIC_DOCS})
    for p in A.P.PUBLIC_DOCS:
        read(p)
    read(POLICY)
    read(SOURCE)


def presentation(task,reverse=False,stable_goal=True):
    """Hand renderer implementing the prospective policy; no production/source import."""
    task=copy.deepcopy(task)
    if task["kind"]=="blocks":
        if stable_goal:
            task["goal_on"]={key:task["goal_on"][key] for key in sorted(task["goal_on"])}
        if reverse:
            task["on"]={key:task["on"][key] for key in reversed(list(task["on"]))}
            task["goal_on"]={key:task["goal_on"][key] for key in reversed(list(task["goal_on"]))}
            task["plan"]=[{"to":a["to"],"block":a["block"]} for a in task["plan"]]
    elif reverse:
        task["nodes"].reverse();task["edges"].reverse()
    if reverse:
        task={key:task[key] for key in reversed(list(task))}
    return task


def stable_projection(parsed,goal_keys=None):
    return A.renderer_projection(parsed,goal_keys)


def hand():
    started=now();setup()
    # Both goal insertion orders are intentionally noncanonical prior to presentation.
    base={"kind":"blocks","on":{"a":"b","b":"TABLE","c":"TABLE","d":"TABLE","e":"TABLE"},
          "goal_on":{"d":"c","a":"TABLE"},"plan":[{"block":"a","to":"TABLE"},{"block":"d","to":"c"}]}
    extension=copy.deepcopy(base);extension["goal_on"]={"b":"e","d":"c","a":"TABLE"}
    complete=copy.deepcopy(extension);complete["plan"]=[{"block":"a","to":"TABLE"},{"block":"b","to":"e"},{"block":"d","to":"c"}]
    before=A.FN["blocks_replay"](base)[0];after=A.FN["blocks_replay"](extension)[0];done=A.FN["blocks_replay"](complete)[0]
    assert before["valid"] and before["reaches_goal"] and len(before["shortest_plan"])==2
    assert after["valid"] and not after["reaches_goal"] and len(after["shortest_plan"])==3
    assert done["valid"] and done["reaches_goal"] and len(done["shortest_plan"])==3
    checks=[]
    for reverse in [False,True]:
        x,y=presentation(base,reverse),presentation(extension,reverse)
        assert list(x["goal_on"])==sorted(base["goal_on"],reverse=reverse)
        assert list(y["goal_on"])==sorted(extension["goal_on"],reverse=reverse)
        assert stable_projection(x,set(base["goal_on"]))==stable_projection(y,set(base["goal_on"]))
        assert A.semantic_key(x)==A.semantic_key(base) and A.semantic_key(y)==A.semantic_key(extension)
        assert canonical(x["plan"])==canonical(base["plan"]) and canonical(y["plan"])==canonical(extension["plan"])
        legacy_x,legacy_y=presentation(base,reverse,False),presentation(extension,reverse,False)
        assert stable_projection(legacy_x,set(base["goal_on"]))!=stable_projection(legacy_y,set(base["goal_on"]))
        checks.append({"style":"reverse" if reverse else "forward","stable_base":x,"stable_extension":y,
                       "legacy_base":legacy_x,"legacy_extension":legacy_y,"stable_shared_projection_matches":True,
                       "legacy_changed_shared_projection_detected":True,"semantic_values_and_plan_sequence_unchanged":True})
    mutation=copy.deepcopy(presentation(base));mutation["goal_on"]["a"]="e"
    assert A.semantic_key(mutation)!=A.semantic_key(base)  # Sorting cannot hide altered facts.
    mutation=copy.deepcopy(presentation(base));mutation["plan"].reverse()
    assert A.semantic_key(mutation)!=A.semantic_key(base)  # Action order must never sort.
    prefix=copy.deepcopy(base);prefix["plan"]=[{"block":"a","to":"TABLE"}]
    noop=copy.deepcopy(prefix);noop["plan"].append({"block":"a","to":"TABLE"})
    strict=A.P.block_prefix_replay(noop)
    assert strict["valid"] is False and strict["steps"][-1]["guards"]["changed_support"] is False
    assert A.P.block_prefix_replay(noop,"changed_support")["valid"] is True
    for reverse in [False,True]:
        assert A.FN["blocks_replay"](presentation(noop,reverse))[0]==A.FN["blocks_replay"](noop)[0]
    graph={"kind":"graph","nodes":["a","b","c"],"edges":[["a","b",1],["b","c",1]],"start":"a","goal":"c","fuel":2,"plan":["b","c"]}
    assert A.semantic_key(presentation(graph,True))==A.semantic_key(graph)
    assert A.FN["graph_replay"](presentation(graph,True))[0]==A.FN["graph_replay"](graph)[0]
    output={"actual_execution_id":EXECUTION,"started_utc":started,"completed_utc":now(),"source_sha256":BINDINGS[SOURCE]["sha256"],
            "hand_examples_only":True,"fresh_inputs_or_candidate_read":False,"checks":checks,
            "public_hand_target_checks":{"base_minimum":2,"extension_minimum":3,"base_plan_fails_extended_goal":True},
            "explicit_mutations_rejected":["changed_goal_value","changed_action_sequence","legacy_shared_goal_key_order"],
            "isolated_noop_guard_replayed_both_styles":True,"graph_style_keeps_semantics":True,"independence":DISCLOSURE}
    print(canonical({"hand_examples":write_new(BASE+".hand_examples.json",output),"data_approval":False}))


def inputs():
    started,clock=now(),time.monotonic();setup()
    read(BASE+".hand_examples.json")
    fresh,old,freeze=rows(INPUT),rows(PARENT),obj(FREEZE)
    own_audit=obj("reviews/adjudication/planning-gap-v2.audit.json")
    own_verdicts=rows("reviews/adjudication/planning-gap-v2.jsonl")
    parent_v1=rows("reviews/input_only/planning-gap-v1.jsonl")
    prior=rows("research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
    expected_fields={"record_hash","rendered_input","sample_id","semantic_task","source_family_key","split"}
    old_index={r["sample_id"]:r for r in old};fresh_index={r["sample_id"]:r for r in fresh}
    assert len(old_index)==len(fresh_index)==len(old)==len(fresh)==408
    old_details={r["sample_id"]:r for r in own_audit["per_record"]}
    assert {r["sample_id"] for r in own_verdicts}==set(old_index) and all(r["decision"]=="QUARANTINE" for r in own_verdicts)
    joins={j["revision_sample_id"]:j for j in freeze["exact_custody_joins"]}
    assert len(joins)==408 and set(joins)==set(fresh_index)
    cache, predictions, proofs, family_rows = {}, [], [], defaultdict(dict)
    visibly_changed=[]
    for row in fresh:
        sid,task=row["sample_id"],row["semantic_task"]
        assert set(row)==expected_fields and type(row["record_hash"]) is str and re.fullmatch(r"[0-9a-f]{64}",row["record_hash"])
        match=re.fullmatch(r"RGM-001-planning-gap-v2-renderer-r1-p(\d{4})-([ABCD]\d)-(forward|reverse)",sid)
        assert match
        n,cid,style=int(match[1]),match[2],match[3]
        allocation=A.P.expected_allocation(n)
        old_id=f"RGM-001-planning-gap-v2-p{n:04d}-{cid}-{style}"
        original=old_index[old_id]
        assert canonical(row["semantic_task"])==canonical(original["semantic_task"])
        assert row["source_family_key"]==original["source_family_key"]==f"RGM-001-seed20261009-p{n:04d}"
        assert row["split"]==original["split"]==allocation["split"]
        assert cid in [allocation["stratum"]+str(i) for i in range(allocation["contexts"])]
        joined=joins[sid]
        assert joined["original_sample_id"]==old_id and joined["original_record_hash"]==original["record_hash"]
        assert joined["revision_record_hash"]==row["record_hash"] and joined["controlled_context"]==cid and joined["renderer"]==style
        assert joined["source_family_key"]==row["source_family_key"]
        A.V1.task_schema(task)
        parsed=json.loads(row["rendered_input"].split("Task JSON:\n",1)[1])
        before=json.loads(original["rendered_input"].split("Task JSON:\n",1)[1])
        assert canonical(parsed)==canonical(task) and canonical(before)==canonical(task)
        assert row["rendered_input"].split("Task JSON:\n",1)[0]==original["rendered_input"].split("Task JSON:\n",1)[0]
        changed=row["rendered_input"]!=original["rendered_input"]
        assert joined["visible_input_changed"] is changed
        if changed:
            visibly_changed.append(sid)
        if task["kind"]=="blocks":
            assert list(parsed["goal_on"])==sorted(parsed["goal_on"],reverse=style=="reverse")
            # Every presentation component outside goal key order is literally preserved.
            expected=copy.deepcopy(before)
            expected["goal_on"]={key:before["goal_on"][key] for key in sorted(before["goal_on"],reverse=style=="reverse")}
            assert json.dumps(parsed,separators=(",",":"),ensure_ascii=False)==json.dumps(expected,separators=(",",":"),ensure_ascii=False)
        else:
            assert row["rendered_input"]==original["rendered_input"]
        key=A.semantic_key(task)
        if key not in cache:
            target,failure=A.FN["graph_replay"](task) if task["kind"]=="graph" else A.FN["blocks_replay"](task)
            minimum,search=A.measured_shortest(task)
            assert canonical(minimum)==canonical(target["shortest_plan"])
            cache[key]=(target,failure,search)
        target,failure,search=cache[key]
        old_computed=A.FN["graph_replay"](original["semantic_task"])[0] if task["kind"]=="graph" else A.FN["blocks_replay"](original["semantic_task"])[0]
        assert canonical(old_computed)==canonical(target)
        legacy=old_details[old_id]
        assert canonical(legacy["failure"])==canonical(failure)
        assert legacy["fresh_minimum_moves"]==(None if target["shortest_plan"] is None else len(target["shortest_plan"]))
        prediction={"sample_id":sid,"record_hash":row["record_hash"],"source_family_key":row["source_family_key"],"split":row["split"],
                    "actual_execution_id":EXECUTION,"role":"adjudicator_input_preparation","expected_target":target,
                    "complete_finite_search":search,"original_input_sample_id":old_id,"original_input_record_hash":original["record_hash"],
                    "exact_original_typed_semantic_join":True,"visible_input_changed":changed,
                    "fresh_stored_gold_compared":False,"review_or_approval_decision":None,"prior_original_gold_exposure":True,
                    "target_source":"Own original public-derived solver recomputed on fresh input and original six-field view; no fresh target read.",
                    "guidance":DISCLOSURE}
        predictions.append(prediction)
        proofs.append({"sample_id":sid,"record_hash":row["record_hash"],"original_sample_id":old_id,"context":cid,"renderer":style,
                       "typed_semantics_exact":True,"input_contract_unchanged":True,"presentation_change_only_goal_key_order":True,
                       "root_join_bound_exactly":True,"visible_input_changed":changed})
        assert (cid,style) not in family_rows[n]
        family_rows[n][cid,style]=parsed
    assert len(cache)==204 and len(family_rows)==48 and len(visibly_changed)==22
    assert set(visibly_changed)==set(freeze["changed_sample_ids"])
    projected=0
    for n,values in family_rows.items():
        alloc=A.P.expected_allocation(n)
        assert set(values)=={(alloc["stratum"]+str(i),style) for i in range(alloc["contexts"]) for style in ["forward","reverse"]}
        if alloc["stratum"]=="D":
            for style in ["forward","reverse"]:
                base=values["D0",style]; shared=set(base["goal_on"])
                for context in ["D0","D1","D2","D3"]:
                    assert stable_projection(values[context,style],shared)==stable_projection(base,shared)
                    projected+=1
    assert projected==96
    prior_ids={r["sample_id"] for r in prior};prior_families={r["source_family_key"] for r in prior}
    assert not set(fresh_index)&prior_ids and not {r["source_family_key"] for r in fresh}&prior_families
    assert {r["source_family_key"] for r in fresh}=={r["source_family_key"] for r in parent_v1}
    pred_binding=write_new(BASE+".input_predictions.jsonl","".join(canonical(p)+"\n" for p in predictions),raw=True)
    proof_binding=write_new(BASE+".input_custody_analysis.json",{"actual_execution_id":EXECUTION,"rows":proofs,
              "exact_original_semantic_joins":408,"unchanged_payloads":386,"goal_key_only_changed_payloads":22,
              "actual_full_D_shared_goal_style_projections_checked":96,"fresh_candidate_hash_validated":False,
              "fresh_manifest_controls_or_stored_gold_reviewed":False,"data_approval":False,"scope":DISCLOSURE})
    summary={"actual_execution_id":EXECUTION,"started_utc":started,"completed_utc":now(),"actual_wall_seconds":time.monotonic()-clock,
             "completed_input_preparation":True,"stage":"FROZEN_ALL408_INPUT_DERIVED_PREDICTIONS_BEFORE_FRESH_CANONICAL_ACCESS",
             "source_sha256":BINDINGS[SOURCE]["sha256"],"bindings":BINDINGS,"predictions":pred_binding,"custody_analysis":proof_binding,
             "rows":408,"same_semantic_tasks":204,"same_source_families":48,"new_semantic_tasks":0,"new_source_families":0,
             "exact_typed_original_semantic_joins":408,"fresh_stored_gold_compared":False,
             "own_complete_solver_predictions":408,"actual_goal_key_only_changed_presentations":22,
             "old_disputed_full_D_shared_key_projections_now_consistent":96,
             "fresh_full_record_hashes_verified":False,"fresh_gate_or_manifest_inspected":False,
             "split_rows":dict(Counter(r["split"] for r in fresh)),"data_approval":False,"publication_approval":False,
             "later_adjudication_requires":["root authorization after completed new first/final reviews","new exact candidate and manifest hashes/full schema checks",
                      "all510 factual witness and192 per-renderer pair bindings","unchanged strict formal gate including all11 formerly aborting D families",
                      "actual source/candidate reproduction custody and completed role executions","residual finite-corpus shortcut/difficulty/lineage disclosure"],
             "guidance":DISCLOSURE}
    for path,binding in BINDINGS.items():
        assert sha((ROOT/path).read_bytes())==binding["sha256"],("preparation input changed",path)
    summary_binding=write_new(BASE+".input_preparation.execution.json",summary)
    print(canonical({"input_preparation":summary_binding,"predictions":pred_binding,"custody":proof_binding,"rows":408,"data_approval":False}))


if __name__=="__main__":
    try:
        {"hand":hand,"inputs":inputs}[sys.argv[1]]()
    except Exception:
        failure={"actual_execution_id":EXECUTION,"at_utc":now(),"command":sys.argv,"exception":traceback.format_exc(),
                 "source_sha256":sha(Path(__file__).read_bytes()),"bindings":BINDINGS,"no_approval":True}
        name=BASE+".owned-input-failure-"+sha(canonical(failure).encode())[:16]+".json"
        write_new(name,failure)
        print(canonical({"preserved_failure":name}),file=sys.stderr)
        raise
