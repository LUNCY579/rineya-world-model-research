"""Root runtime comparison of five separately authored frozen interpreters.
No production corpus, independent review verdict, acceptance or efficacy claim.
"""
from pathlib import Path
from copy import deepcopy
import datetime,hashlib,importlib.util,json,sys,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def raw_binding(path):
 b=path.read_bytes();return {"size":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def sha(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def reverse(v):
 if type(v) is dict:return {k:reverse(w) for k,w in reversed(list(v.items()))}
 if type(v) is list:return [reverse(w) for w in v]
 return v
def run():
 freeze=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
 for p,e in freeze["bindings"].items():
  assert raw_binding(ROOT/p)=={"size":e["size"],"sha256":e["sha256"]},p
  assert (ROOT/e["archive"]).read_bytes()==(ROOT/p).read_bytes(),p
 mods=[]
 for j,p in enumerate(freeze["solver_paths"]):
  spec=importlib.util.spec_from_file_location("rsm_root_hand_solver_"+str(j),ROOT/p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods.append(m)
 rows=[json.loads(line) for line in (ROOT/freeze["hand_candidate_path"]).read_bytes().splitlines() if line.strip()]
 assert len(rows)==216
 checks=[];failures=[]
 for row in rows:
  original=row["semantic_task"]
  for mode in ("original","reverse_objects","zero_budget","boolean_version","nonlist_history","wrong_checkpoint"):
   task=deepcopy(original)
   if mode=="reverse_objects":task=reverse(task)
   elif mode=="zero_budget":task["event_budget"]=0
   elif mode=="boolean_version":task["version"]=True
   elif mode=="nonlist_history":task["history"]=None
   elif mode=="wrong_checkpoint":task["checkpoint"]={}
   outputs=[]
   for solver in mods:
    try:outputs.append({"result":solver.solve(deepcopy(task))})
    except Exception as e:outputs.append({"exception_type":type(e).__name__,"exception":str(e)})
   equal=all(canonical(z)==canonical(outputs[0]) for z in outputs) and all("result" in z for z in outputs)
   expected_stage={"boolean_version":"invalid_task","nonlist_history":"invalid_history","wrong_checkpoint":"invalid_checkpoint"}.get(mode)
   stage_ok=expected_stage is None or all(z.get("result",{}).get("error")==expected_stage for z in outputs)
   item={"sample_id":row["sample_id"],"mutation":mode,"task_sha256":sha(task),"output_sha256":[sha(z) for z in outputs],"exact_typed_agreement":equal,"expected_stage":expected_stage,"expected_stage_pass":stage_ok}
   checks.append(item)
   if not (equal and stage_ok):failures.append({**item,"actual_outputs":outputs,"actual_task":task})
 for p,e in freeze["bindings"].items():assert raw_binding(ROOT/p)=={"size":e["size"],"sha256":e["sha256"]},p
 report={"schema":"root-rsm-five-frozen-interpreters-hand-crosscheck-v1","actual_runner_execution_id":"/root","at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_freeze_sha256":hashlib.sha256((HERE/"SOURCE_FREEZE.json").read_bytes()).hexdigest(),"status":"PASS" if not failures else "FAIL","checks":checks,"failures":failures,"records":216,"actual_checks":len(checks),"solver_source_paths":freeze["solver_paths"],"all_frozen_bindings_and_archives_unchanged":True,"production_data_approval":False,"scope":"Formal frozen handmade216 development rows only; stored hand verified_targets ignored. Root comparisons disclose access to all five completed implementations and formal hand tasks. Same-model correlation; no new production first judgments or blind review claim."}
 with (HERE/"ACTUAL_RESULT.json").open("x") as h:json.dump(report,h,indent=2,ensure_ascii=False,sort_keys=True);h.write("\n")
 print(json.dumps({"status":report["status"],"records":216,"actual_checks":len(checks),"failures":len(failures),"result_sha256":hashlib.sha256((HERE/"ACTUAL_RESULT.json").read_bytes()).hexdigest()}),flush=True)
 return 0 if not failures else 1
if __name__=="__main__":sys.exit(run())
