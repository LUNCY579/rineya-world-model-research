import json, os, statistics as st
from collections import defaultdict
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
for d in ["BW_B3_HARD/h2_beam","BW_TEST_B2/hard_beam","BW_TEST_B2/mirror_beam","BW_SEEDS/R0_beam"]:
    rows=[json.loads(l) for l in open(os.path.join(R,d,"plans.jsonl")) if l.strip()]
    for lo,hi in [(6,10),(11,15),(12,15)]:
        sel=[r for r in rows if lo<=r["n_blocks"]<=hi]
        if not sel: continue
        print(d,lo,hi,len(sel),"succ",sum(r["success"] for r in sel),"tok",round(st.mean(r["gen_tokens"] for r in sel)),
              "rounds",round(st.mean(r["rounds"] for r in sel),2),
              "fail",dict((k,sum(1 for r in sel if r.get("fail")==k)) for k in ("budget","dead_end")))
    # by size
    by=defaultdict(list)
    for r in rows: by[r["n_blocks"]].append(r)
    print("  by n:", {n:(len(v),sum(r["success"] for r in v),round(st.mean(r["gen_tokens"] for r in v)/1000,1)) for n,v in sorted(by.items())})
# r1 solutions: plan lengths and verified
sol=[json.loads(l) for l in open(os.path.join(R,"BW_DISTILL_r1/solutions.jsonl"))]
print(len(sol), sum(s["verified"] for s in sol), max(len(s["plan"]) for s in sol if s["verified"]))
