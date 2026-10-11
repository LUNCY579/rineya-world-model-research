import time, json, re, statistics as st
from pathlib import Path
src = Path("../lg_bfs.py").read_text().split("d = json.loads")[0]
exec(compile(src, "lgbfs", "exec"))
S = json.load(open("../inst_stats.json"))["generated_basic"]
ts = []; states = []; fails = 0
for i in range(2, 287):
    if S[str(i)]["packages"] != 3: continue
    objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
    t0 = time.time(); n, ns, na = bfs(init, goal, cap=400000, tmax=120); ts.append(time.time() - t0); states.append(ns)
    fails += n is None
print("3-pkg PB instances:", len(ts), "fails", fails, "time s median", round(st.median(ts), 2), "max", round(max(ts), 2), "states max", max(states))
