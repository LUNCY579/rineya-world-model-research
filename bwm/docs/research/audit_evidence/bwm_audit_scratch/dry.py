import sys, subprocess, hashlib, importlib.util
from pathlib import Path
repo = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/gh_export")
spec = importlib.util.spec_from_file_location("export_bwm", repo/"tools/export_bwm.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
keep, excluded = m.plan(m.DEFAULT_SRC)
remote = subprocess.run(["git","-C",str(repo),"ls-tree","-r","origin/main","--name-only","bwm"],capture_output=True,text=True).stdout.split()
remote=set(remote)
new=[];changed=[]
for s,rel in keep:
    r="bwm/"+str(rel).replace("\\","/")
    if r not in remote: new.append((r, s.stat().st_size))
    else:
        blob = subprocess.run(["git","-C",str(repo),"rev-parse","origin/main:"+r],capture_output=True,text=True).stdout.strip()
        data=s.read_bytes()
        h=hashlib.sha1(b"blob %d\0"%len(data)+data).hexdigest()
        if h!=blob: changed.append(r)
keepset={"bwm/"+str(rel).replace("\\","/") for _,rel in keep}
gone=[r for r in remote if r not in keepset and not r.endswith(("README.md","EXCLUDED_LARGE_FILES.json"))]
print("NEW",len(new)); [print("  ",x) for x in new]
print("CHANGED",len(changed)); [print("  ",x) for x in changed]
print("REMOVED",len(gone)); [print("  ",x) for x in gone[:20]]
print("EXCL",len(excluded))
for p,why in excluded:
    if "DISTILL" in str(p) or "ko" in p.name: print("  ex",p,why)
print("PREREG_D54 in keep:", any("PREREG_D54" in str(s) for s,_ in keep))
