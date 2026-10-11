import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print(type(d), list(d.keys())[:20] if isinstance(d, dict) else len(d))
ins = d.get("instances", d)
print("n instances", len(ins))
x = ins[0]
def show(o, depth=0, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                print("  " * depth + f"{k}: {type(v).__name__} len={len(v)}"); 
                if depth < 2: show(v if isinstance(v, dict) else (v[0] if v else {}), depth + 1)
            else:
                s = str(v); print("  " * depth + f"{k}: {s[:150]!r}")
show(x)
