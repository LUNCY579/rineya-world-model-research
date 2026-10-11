import sys, json, statistics as st
from fractions import Fraction
from collections import defaultdict
from pathlib import Path
ROOT=Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import importlib.util
spec=importlib.util.spec_from_file_location("g",str(ROOT/"scripts/bw_mirror_gap.py"))
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
B=g.B; VL=g.VL
hard=[]
for p in (B.BENCH/"blocksworld_hard"/"generated").glob("*.pddl"):
    o,i,gg=B.parse_problem(p); hard.append((f"hard/{p.stem}",o,i,gg))
mirror=[(f"mirror/{k}",r["objs"],frozenset(map(tuple,r["init"])),frozenset(map(tuple,r["goal"]))) for k,r in enumerate(json.loads((ROOT/"runs/BW_val_large/instances_mirror_hard.json").read_text()))]
succ={}
for t in ("hard","mirror"):
    for line in open(ROOT/f"runs/BW_TEST_B2/{t}_beam/plans.jsonl",encoding="utf-8"):
        r=json.loads(line); succ[r["id"]]=int(bool(r.get("val",r["success"])))
out={}; bins={}
for name,insts in (("hard",hard),("mirror",mirror)):
    lb,gmax,gc,imax,by=[],[],[],[],defaultdict(list)
    for iid,o,i,gg in insts:
        gt=[h for h in VL.goal_towers(gg,o) if h>1]; it=[h for h in VL.towers_of(i,o) if h>1]
        lb.append(2*g.must_move(o,i,gg)); gmax.append(max(gt) if gt else 1); gc.append(len(gt)); imax.append(max(it) if it else 1)
        by[g.bin_of(gmax[-1])].append(succ[iid])
    bins[name]=by
    print(name,len(insts),'sum gmax',sum(gmax),'mean exact',Fraction(sum(gmax),len(gmax)),float(Fraction(sum(gmax),len(gmax))),'ge10',sum(1 for x in gmax if x>=10))
    out[name]={"n":len(insts),"success":round(sum(succ[x[0]] for x in insts)/len(insts),4),"mean_move_lower_bound":round(st.mean(lb),2),"mean_max_goal_tower":round(st.mean(gmax),2),"mean_goal_towers":round(st.mean(gc),2),"mean_max_init_tower":round(st.mean(imax),2),"success_by_max_goal_tower":{k:f"{sum(v)}/{len(v)}" for k,v in sorted(by.items())}}
    print(' lb exact',st.mean(lb),'gmax',st.mean(gmax))
n_h=sum(len(v) for v in bins["hard"].values())
std=sum(len(bins["hard"][k])*sum(bins["mirror"][k])/len(bins["mirror"][k]) for k in bins["hard"])/n_h
out["mirror_standardized_to_hard_goal_tower_mix"]=round(std,4)
out["gap_pp_raw"]=round(100*(out["hard"]["success"]-out["mirror"]["success"]),1)
out["gap_pp_standardized"]=round(100*(out["hard"]["success"]-std),1)
stored=json.load(open(ROOT/"runs/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json"))
stored.pop("note")
print('equal to stored:',out==stored)
for k in out:
    if out[k]!=stored.get(k): print('DIFF',k,out[k],stored.get(k))
print('std raw',std)
