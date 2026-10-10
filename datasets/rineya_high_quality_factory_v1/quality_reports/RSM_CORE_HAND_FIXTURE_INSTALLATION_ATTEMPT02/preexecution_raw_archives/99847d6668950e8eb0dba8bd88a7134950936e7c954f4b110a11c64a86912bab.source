"""Cross request/query identity and position with outcomes within each family."""
import copy
import json
import random
from factory.common import ROOT,record,record_hash,write_jsonl,load_jsonl,split_for
from generators.memory.generate import make_modules,make_episodic
from generators.other_tracks.reference import contracts,solve
from generators.other_tracks.generate import reorder


def generate(count=12,batch="extended-pilot-v3",offset=0):
    output=[];all_rows=[];rules=contracts()
    for track,builder,field,prefix,track_number in [("RMM-001",make_modules,"requests","r",1),("REM-001",make_episodic,"queries","q",2)]:
        rows=[];parents=[]
        for index in range(offset,offset+count):
            split=split_for(index);rng=random.Random(20261009+10007*track_number+53*index)
            base,mechanism,level,limitation=builder(index,split,rng)
            tasks=[];n=len(base[field])
            for rotation in range(n):
                task=copy.deepcopy(base)
                original=task[field]
                task[field]=original[rotation:]+original[:rotation]
                for position,item in enumerate(task[field]):item["request_id" if field=="requests" else "query_id"]=f"{prefix}{position}"
                tasks.append(task);target=solve(task)
                for renderer in ["forward","reverse"]:
                    body=json.dumps(task if renderer=="forward" else reorder(task),ensure_ascii=False,separators=(",",":"))
                    prompt="Executable task contract:\n"+rules[task["kind"]]+"Task JSON:\n"+body
                    row=record(track,batch,index,f"rotation{rotation}",task,prompt,target,["finite_synthetic_proxy",task["kind"],mechanism,"identity_position_outcome_crossing"],level,"structured",renderer)
                    row.update(generator_version="1.1.0",structural_difficulty=level,
                               challenge_axes=["heldout_names","more_declared_modules" if field=="requests" else "longer_event_history"] if split=="dev_challenge" else [],
                               scope_limitation=limitation,producer_agent_id="/root",producer_role="improvement_engineer",source_family_key=f"{track}-seed20261009-p{index:04d}")
                    row["record_hash"]=record_hash(row);rows.append(row)
            pid=f"{track}-{batch}-p{index:04d}"
            parents.append({"semantic_parent_id":pid,"semantic_family_id":pid,"split":split,"tasks":tasks,"mechanism":mechanism,"counterfactual_axis":"request/query position and ID assignment; output recomputed"})
        output += [(ROOT/f"data/{track}/semantic_parents/{batch}.jsonl",parents),(ROOT/f"data/{track}/candidate_renderings/{batch}.jsonl",rows)]
        all_rows+=rows
    for path,rows in output:
        if path.exists() and load_jsonl(path)!=rows:raise ValueError("Immutable batch differs; use new version")
    for path,rows in output:write_jsonl(path,rows)
    return all_rows


if __name__=="__main__":generate()
