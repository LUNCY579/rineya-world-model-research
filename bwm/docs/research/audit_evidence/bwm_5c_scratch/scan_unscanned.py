# Independent re-scan of the confirm draw against directories NOT in ret_gen.SCAN_SOURCES (read-only).
import json, sys, os, time, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
R = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
_Q = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u00a0": " "})
def norm(s): return " ".join(unicodedata.normalize("NFKC", str(s)).translate(_Q).casefold().split())
def split(n):
    out, st = [], 0
    for i, c in enumerate(n):
        if c in ".?!:" and (i + 1 == len(n) or n[i + 1] == " "):
            seg = n[st:i + 1].strip()
            if seg: out.append(seg)
            st = i + 1
    t = n[st:].strip()
    if t: out.append(t)
    return out
def utext(msgs): return "\n".join(m["content"] for m in msgs if m.get("role") == "user")
rows = [json.loads(l) for l in open(R / "docs/retention/draws/confirm_1708484032065101819.jsonl", encoding="utf-8")]
users = defaultdict(list); owners = defaultdict(set); allw = defaultdict(list)
for r in rows:
    u = norm(utext(r["messages"])); users[u].append(r["pid"])
    for s in set(split(u)):
        if len(s.split()) >= 8: owners[s].add(r["item"])
        if len(s.split()) >= 6: allw[s].append(r["pid"])
gate = {s for s, it in owners.items() if len(it) == 1}
report = {s for s in allw if s not in gate}
print("users", len(users), "gate", len(gate), "report", len(report))
roots = [Path(r"C:/Users/clclc/BusinessProject/WorldModel") / d for d in
         ("bwm-gh-export", "rineya-world-model-research", "rineya-complementary-worktree", "rineya_complementary_research_2026-10-09")]
SKIP = {".git", ".claude", ".venv", "venv", "node_modules", "__pycache__"}
hits = {"user": Counter(), "gate": Counter(), "report": Counter()}
byfile = defaultdict(Counter)
cur = [""]
def visit(s):
    n = norm(s)
    if n in users: hits["user"][n] += 1; byfile[cur[0]]["user"] += 1
    if len(n) < 30: return
    for sent in split(n):
        if sent in gate: hits["gate"][sent] += 1; byfile[cur[0]]["gate"] += 1
        elif sent in report: hits["report"][sent] += 1; byfile[cur[0]]["report"] += 1
def walk(doc):
    st = [doc]
    while st:
        c = st.pop()
        if isinstance(c, str):
            if len(c) >= 16: visit(c)
        elif isinstance(c, dict): st.extend(c.values())
        elif isinstance(c, list): st.extend(c)
nfiles = 0
for root in roots:
    for suf in ("*.jsonl", "*.json"):
        for p in root.rglob(suf):
            if any(part in SKIP for part in p.parts): continue
            if p.name.startswith("HSKILL_CONFIRM_V2"): print("reserved skip", p); continue
            cur[0] = str(p.relative_to(root.parent)); nfiles += 1
            try:
                if p.suffix == ".jsonl":
                    with open(p, encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            if line.strip():
                                try: walk(json.loads(line))
                                except json.JSONDecodeError: pass
                else:
                    try: walk(json.loads(p.read_text(encoding="utf-8", errors="ignore")))
                    except json.JSONDecodeError: pass
            except OSError: pass
print("files", nfiles)
print("F1 user hits", sum(hits["user"].values()), "F2 gate hits", sum(hits["gate"].values()), "F3 report hits", sum(hits["report"].values()), "distinct report", len(hits["report"]))
for f, c in sorted(byfile.items(), key=lambda kv: -sum(kv[1].values()))[:40]:
    print(dict(c), f)
print("report sentences hit (top 20):")
for s, c in hits["report"].most_common(20): print(c, s)
print("gate sentences hit (top 10):")
for s, c in hits["gate"].most_common(10): print(c, s)
