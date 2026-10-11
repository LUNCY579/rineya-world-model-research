import json, os, datetime
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
FAILED = ROOT/"runs"/"_failed"
jobs = json.loads((ROOT/"runs"/"jobs.json").read_text(encoding="utf-8"))
def mt(p):
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M") if p.exists() else "-"
print("idx name | marker exists | failed | after | after exists")
for i,j in enumerate(jobs):
    m = ROOT/j["marker"]; f = FAILED/f"{j['name']}.failed"
    a = j.get("after")
    print(i, j["name"], "| marker:", m.exists(), mt(m), "| failed:", f.exists(), "| after:", a, (ROOT/a).exists() if a else "")
    if j["cmd"][0].endswith("bundle.py"):
        spec = json.loads((ROOT/j["cmd"][1]).read_text(encoding="utf-8"))
        for s in spec["jobs"]:
            sm = ROOT/s["marker"]
            print("     sub", s["name"], "est", s.get("est_s"), "marker", sm.exists(), mt(sm))
pending = [j for j in jobs if not (ROOT/j["marker"]).exists() and not (FAILED/f"{j['name']}.failed").exists()]
runnable = [j for j in pending if not j.get("after") or (ROOT/j["after"]).exists()]
print("PENDING:", [j["name"] for j in pending])
print("RUNNABLE ORDER:", [j["name"] for j in runnable])
# script existence
for j in jobs:
    s = ROOT/j["cmd"][0]
    if not s.exists(): print("MISSING SCRIPT", j["name"], s)
    if j["cmd"][0].endswith("bundle.py"):
        spec = json.loads((ROOT/j["cmd"][1]).read_text(encoding="utf-8"))
        for sj in spec["jobs"]:
            if not (ROOT/sj["cmd"][0]).exists(): print("MISSING SUB SCRIPT", j["name"], sj["name"], sj["cmd"][0])
