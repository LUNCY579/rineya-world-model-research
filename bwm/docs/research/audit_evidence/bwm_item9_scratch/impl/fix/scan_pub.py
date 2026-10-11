import sys, json, re
from collections import Counter
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
c = Counter(); ex = {}
for f in sorted(L.RESULTS_DIR.rglob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for x in d["instances"]:
        raw = x.get("extracted_llm_plan")
        raw = "\n".join(raw) if isinstance(raw, list) else (raw or "")
        t = re.sub(r";[^\n]*", "", raw)
        res = re.sub(r"\([^()]*\)", " ", t)
        res2 = res.strip()
        odd = set(re.sub(r"[A-Za-z0-9_\-\s]", "", raw))
        if odd: c["odd_chars:" + "".join(sorted(odd))] += 1
        if res2: c["residue"] += 1; ex.setdefault("residue", []).append((f.parent.name + "/" + f.stem, x["instance_id"], res2[:80]))
        if re.search(r"\d\s*:", raw): c["timestamps?"] += 1
        if t.count("(") != t.count(")"): c["unbalanced"] += 1
        if re.search(r"\d[a-zA-Z]", re.sub(r"[a-zA-Z_][\w\-]*", "", raw)): c["digit_letter"] += 1
        if "(" in raw and re.search(r"\([^()]*\(", raw): c["nested"] += 1
print(c)
for k, v in ex.items():
    for e in v[:20]: print(k, ascii(e))
