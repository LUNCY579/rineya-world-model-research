"""Supa e2e generative exact-state accuracy: strict parse vs lenient extraction of an "after" field
(Supa often answers {"before":..., "after":...}). Independent of rcwmt metric code."""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT / "docs" / "verification"))
ACC, REG = ("A", "B", "C", "D"), ("X", "Y", "Z")
bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
gen = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / "runs/bank_pilot/supa/gen_e2e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def stepL(s, e):
    s = dict(s)
    if e["status"] != "posted":
        return s
    a, o, k, m = e["account"], e["other"], e["kind"], e["amount"]
    if k == "add": s[a] += m
    elif k == "scale": s[a] *= m
    else: s[a] -= m; s[o] += m
    return s


def stepR(s, e):
    s = dict(s); k, r, o, v = e["kind"], e["register"], e["other"], e.get("value", 0)
    if k == "swap": s[r], s[o] = s[o], s[r]
    elif k == "add": s[r] = (s[r] + v) % 97
    elif k == "mix": s[r] = (s[r] + 2 * s[o] + v) % 97
    else: s[r] = (s[r] + (v if s[o] > e["threshold"] else -v)) % 97
    return s


def stepP(s, e):
    return (s & ~e["delete"]) | e["add"] if (s & e["pre_on"]) == e["pre_on"] and (s & e["pre_off"]) == 0 else s


ST = {"ledger_replay": stepL, "register_machine": stepR, "prerequisite_plan": stepP}


def valid(fam, v):
    if fam == "prerequisite_plan":
        return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 256
    keys = ACC if fam == "ledger_replay" else REG
    return isinstance(v, dict) and set(v) == set(keys) and all(isinstance(v[q], int) and not isinstance(v[q], bool) for q in keys)


def strict(fam, t):
    t = t.strip()
    t = t[:-1] if t.endswith(".") else t
    try:
        v = json.loads(t)
    except Exception:
        return None
    return v if valid(fam, v) else None


def lenient(fam, t):
    v = strict(fam, t)
    if v is not None:
        return v, "strict"
    m = re.search(r'"after"\s*:\s*(\{[^{}]*\}|-?\d+)', t)
    if m:
        try:
            v = json.loads(m.group(1))
            if valid(fam, v):
                # P: a bare integer cut by the 12-token budget may be truncated; flag if text ends right after it
                trunc = fam == "prerequisite_plan" and t.rstrip().endswith(m.group(1))
                return v, "after_trunc" if trunc else "after"
        except Exception:
            pass
    return None, "none"


for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    for frame in ("one", "k1"):
        for scope in ("test", "all"):
            n = ex_s = ex_l = 0
            kinds = Counter()
            for p in bank["parents"]:
                if p["family"] != fam or (scope == "test" and p["split"] != "test"):
                    continue
                for c in p["candidates"]:
                    g = gen.get(c["cand_id"] + ("-one" if frame == "one" else "-k1"))
                    if g is None:
                        kinds["MISSING"] += 1
                        continue
                    gold = ST[fam](p["s0"], c["events"][0])
                    s = strict(fam, g["text"])
                    v, how = lenient(fam, g["text"])
                    n += 1
                    ex_s += s == gold
                    ex_l += (v == gold) and how != "after_trunc"
                    kinds[how] += 1
            print(f"{fam[:8]:<9}{frame:<4}{scope:<5} n={n:<5} strict_exact={ex_s / n:.3f} lenient_after_exact={ex_l / n:.3f} parse_kinds={dict(kinds)}")
