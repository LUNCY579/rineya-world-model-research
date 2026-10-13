import json, glob, os, statistics as st
from collections import defaultdict
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
def shards(d):
    out=[]
    for sd in sorted(glob.glob(os.path.join(R,d,"shard_*"))):
        sp=os.path.join(sd,"summary.json"); pp=os.path.join(sd,"plans.jsonl")
        if not os.path.exists(sp): continue
        s=json.load(open(sp))["summary"]
        rows=[json.loads(l) for l in open(pp) if l.strip()]
        tok=sum(r["gen_tokens"] for r in rows)
        nb=[r["n_blocks"] for r in rows]
        big=sum(1 for b in nb if b>=11)
        out.append((d,os.path.basename(sd),len(rows),big,s["wall_s"],tok,s.get("f_model_queries"), sum(r["success"] for r in rows)))
    return out
allr=[]
for d in ["BW_B3_HARD/h2_beam","BW_TEST_B2/hard_beam","BW_TEST_B2/mirror_beam","BW_SEEDS/R0_beam","BW_SEEDS/R1_beam","BW_SEEDS/R2_beam"]:
    for r in shards(d):
        allr.append(r)
        d_,sd,n,big,w,tok,fq,succ=r
        print(f"{d_:24s} {sd:12s} n={n:2d} big={big:2d} succ={succ:2d} wall={w:7.0f} s/inst={w/n:6.0f} tok/inst={tok/n:7.0f} s_per_1k={w/(tok/1000):5.1f} fq={fq}")
# per-instance tokens by size and success for h2, mirror
for d in ["BW_B3_HARD/h2_beam","BW_TEST_B2/mirror_beam","BW_SEEDS/R0_beam","BW_TEST_B2/hard_beam"]:
    rows=[json.loads(l) for l in open(os.path.join(R,d,"plans.jsonl")) if l.strip()]
    by=defaultdict(list)
    for r in rows:
        if r["n_blocks"]>=11: by[r["success"]].append(r["gen_tokens"])
    print(d, {k:(len(v), round(st.mean(v))) for k,v in by.items()})
