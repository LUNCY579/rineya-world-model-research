"""VAL's own reading of a raw plan text (from `validate -v`): kind 'bad' | 'ok' | 'other', [(time, action)], valid."""
import sys, subprocess, tempfile, re
from pathlib import Path
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
STOP = ("Plan Validation details", "Plan executed", "Plan failed", "Goal not satisfied", "Plan Repair", "Plan valid",
        "Plan invalid", "Failed plans", "Successful plans")

def val_read(problem, text, domain=L.DOMAIN_PDDL, raw=False):
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d) / "plan.txt"
        pp.write_bytes(text.encode("utf-8"))
        r = subprocess.run([str(L.VAL_EXE), "-v", str(domain), str(problem), str(pp)], capture_output=True, timeout=60)
    out = r.stdout.decode("utf-8", "replace")
    valid = r.returncode == 0 and "Plan valid" in out
    if raw:
        return out
    if "Bad plan description" in out:
        return "bad", None, valid, r.returncode
    m = re.search(r"Plan size: (\d+)", out)
    if not m:
        return "other", None, valid, r.returncode
    sec = out[m.end():]
    cut = min([sec.find(s) for s in STOP if sec.find(s) >= 0] + [len(sec)])
    sec = sec[:cut]
    acts, t = [], None
    for ln in sec.splitlines():
        ln = ln.strip()
        if re.fullmatch(r"-?[\d.]+:", ln):
            t = ln[:-1]
        elif ln.startswith("("):
            acts.append((t, tuple(ln.strip("()").split())))
    assert len(acts) >= int(m.group(1)), (m.group(1), acts)
    return "ok", acts, valid, r.returncode
