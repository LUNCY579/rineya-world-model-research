import json
from collections import defaultdict, Counter
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
D = R + "/runs/retention/confirm_1708484032065101819"
rows = [json.loads(l) for l in open(R + "/docs/retention/draws/confirm_1708484032065101819.jsonl", encoding="utf-8")]
SIDES = {"off": "sparse", "t2s0": "sparse", "x_bwf": "sparse", "x_bwpa": "sparse", "supa": "supa"}
gens = {s: {x["pid"]: x for x in (json.loads(l) for l in open(f"{D}/{d}/gen_{s}.jsonl", encoding="utf-8"))} for s, d in SIDES.items()}
G = json.load(open("graded.json"))["graded"]
byitem = defaultdict(list)
for r in rows:
    byitem[r["item"]].append(r)


def gold_str(g):
    return g.get("value", "UNKNOWN") if g["kind"] != "unknown" else "UNKNOWN"


def show(arm, ref, cat, gr):
    print(f"\n### {arm} vs {ref}, {cat}, grader {gr}: LOST items (ref right, arm wrong)")
    for it, rs in byitem.items():
        if rs[0]["category"] != cat:
            continue
        rc = all(G[ref][r["pid"]][gr] for r in rs)
        ac = all(G[arm][r["pid"]][gr] for r in rs)
        if rc and not ac:
            for r in rs:
                print(f"  {r['pid'].split('/',1)[1]:28s} twin={r['twin']} op={r['op'][:50]!r} gold={gold_str(r['gold'])!r} "
                      f"{ref}={gens[ref][r['pid']]['text']!r} {arm}={gens[arm][r['pid']]['text']!r} "
                      f"(arm prim {G[arm][r['pid']]['primary']}, len {G[arm][r['pid']]['lenient']})")
    print(f"### GAINED items")
    for it, rs in byitem.items():
        if rs[0]["category"] != cat:
            continue
        rc = all(G[ref][r["pid"]][gr] for r in rs)
        ac = all(G[arm][r["pid"]][gr] for r in rs)
        if ac and not rc:
            for r in rs:
                print(f"  {r['pid'].split('/',1)[1]:28s} twin={r['twin']} gold={gold_str(r['gold'])!r} "
                      f"{ref}={gens[ref][r['pid']]['text']!r} {arm}={gens[arm][r['pid']]['text']!r}")


show("x_bwf", "off", "unit_conv", "primary")
show("x_bwf", "off", "unknown_twin", "primary")
show("x_bwpa", "off", "unknown_twin", "lenient")
show("x_bwpa", "off", "unit_conv", "primary")
# unanswerable twins answered with a value
for s in gens:
    bad = [(r["pid"].split("/", 1)[1], gens[s][r["pid"]]["text"]) for r in rows if r["category"] == "unknown_twin" and r["twin"] == "u"
           and not G[s][r["pid"]]["primary"]]
    print(f"\n{s}: unanswerable twins NOT answered UNKNOWN: {len(bad)} {bad[:8]}")
# unit_conv: answer patterns per arm
for s in gens:
    c = Counter()
    for r in rows:
        if r["category"] != "unit_conv":
            continue
        t = gens[s][r["pid"]]["text"]
        c["correct" if G[s][r["pid"]]["primary"] else ("non-numeric" if not any(ch.isdigit() for ch in t) else "wrong number")] += 1
    print(s, "unit_conv", dict(c))
# unit_conv by conversion type (op field '1|kg>g|12')
for s in ("off", "x_bwf", "x_bwpa", "t2s0", "supa"):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["category"] != "unit_conv":
            continue
        conv = r["op"].split("|")[1]
        c[conv][1] += 1
        c[conv][0] += G[s][r["pid"]]["primary"]
    print(s, {k: f"{v[0]}/{v[1]}" for k, v in sorted(c.items())})
# primary vs lenient disagreement rate per arm
for s in gens:
    n = sum(G[s][r["pid"]]["primary"] != G[s][r["pid"]]["lenient"] for r in rows)
    pl = sum(G[s][r["pid"]]["lenient"] and not G[s][r["pid"]]["primary"] for r in rows)
    print(f"{s}: prompts where graders disagree {n}/450 (lenient-only correct {pl})")
