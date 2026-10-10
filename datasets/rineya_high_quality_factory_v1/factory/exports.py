"""Derived, separately typed exports from already verified canonical labels."""
import collections
from factory.common import canonical,write_jsonl

PREFERENCE_RULES = """Select the better submitted candidate using the explicit ranking policy. Each candidate starts at world's start with world's initial fuel. A directed move is legal iff its edge exists and cost is at most remaining fuel. On an illegal action, stop and retain the prior state. A candidate is valid iff its entire plan is legal. A claim is correct iff claimed_after equals the executed final state. Goal achievement requires validity and terminal node=goal. remaining_moves is the minimum number of feasible moves to goal from terminal state/fuel, or null if unreachable. Initial distance is the same minimum from the initial state. progress is initial distance minus remaining distance for valid candidates with both finite, otherwise null. usefulness=-1 for invalid, 2 for valid goal-reaching, 1 for valid positive progress, 0 otherwise. Rank descending by (valid AND claim_correct, reaches_goal, usefulness, -remaining_moves, -full_submitted_plan_length). Null distance is worse than finite; true sorts above false. Return exactly the selected candidate object, including candidate_id, plan and claimed_after. Options have no quality order.\n"""


def specialized(rows,split):
    selected=[r for r in rows if r["split"]==split]
    result={}
    if not rows:return result
    track=rows[0]["track_id"]
    if track=="RGM-001":
        result[f"{split}-trajectories.jsonl"]=[{"input":r["rendered_input"],"actions":r["semantic_task"]["plan"],"steps":r["verified_target"]["steps"],"after":r["verified_target"]["after"]} for r in selected if r["verified_target"]["valid"] and r["verified_target"]["reaches_goal"]]
    if track=="RVM-001":
        process=[];preferences=[]
        for r in selected:
            task=r["semantic_task"];labels={x["candidate_id"]:x for x in r["verified_target"]["evaluations"]}
            rank={cid:i for i,group in enumerate(r["verified_target"]["ranking"]) for cid in group}
            candidates=sorted(task["candidates"],key=lambda c:c["candidate_id"])
            for c in candidates:process.append({"input":r["rendered_input"],"candidate":c,"labels":labels[c["candidate_id"]]})
            buckets={}
            for good in candidates:
                for bad in candidates:
                    if rank[good["candidate_id"]]>=rank[bad["candidate_id"]]:continue
                    difference=len(good["plan"])-len(bad["plan"])
                    bucket=0 if difference==0 else 1 if difference>0 else -1
                    buckets.setdefault(bucket,(good,bad))
            for bucket,(good,bad) in sorted(buckets.items()):
                options=sorted([good,bad],key=lambda c:c["candidate_id"])
                if r["renderer"]=="reverse":options.reverse()
                prompt=PREFERENCE_RULES+"World and response options JSON:\n"+canonical({"world":task["world"],"options":options})
                preferences.append({"prompt":prompt,"chosen":canonical(good),"rejected":canonical(bad)})
        result[f"{split}-process_reward.jsonl"]=process
        result[f"{split}-preference.jsonl"]=preferences
    for tag,tracks in [("skill_demonstrations",["RSM-001"]),("memory_updates",["REM-001","RMM-001"]),("belief_updates",["RBM-001","ROM-001"])]:
        if track in tracks:result[f"{split}-{tag}.jsonl"]=[{"input":r["rendered_input"],"target":r["verified_target"]} for r in selected]
    return result


def write_specialized(path,rows):
    for split in ["train","dev_challenge"]:
        for name,records in specialized(rows,split).items():write_jsonl(path/name,records)
