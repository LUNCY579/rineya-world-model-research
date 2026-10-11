# Independent recomputation of the registered 5c rule (RET_RESULTS) from the raw generations.
# Does NOT import ret_analysis. Read-only on RCWM files; writes only into this scratch dir.
import sys, json, math, re, unicodedata
from fractions import Fraction
from collections import defaultdict
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
D = R + "/runs/retention/confirm_1708484032065101819"
sys.path.insert(0, R + "/scripts")
import ret_grade as RG          # frozen grader (pure functions); analysis code is NOT imported
assert "ret_analysis" not in sys.modules and "torch" not in sys.modules
rows = [json.loads(l) for l in open(R + "/docs/retention/draws/confirm_1708484032065101819.jsonl", encoding="utf-8")]
SIDES = {"off": "sparse", "t2s0": "sparse", "x_bwf": "sparse", "x_bwpa": "sparse", "supa": "supa"}
gens = {s: [json.loads(l) for l in open(f"{D}/{d}/gen_{s}.jsonl", encoding="utf-8")] for s, d in SIDES.items()}
for s, g in gens.items():
    assert len(g) == 450 and [x["pid"] for x in g] == [r["pid"] for r in rows], s
    assert [x["lp_row"] for x in g] == list(range(450)), s
    print(s, "terminated", sum(x["terminated"] for x in g), "max n_new", max(x["n_new"] for x in g),
          "first_token==prefill", sum(x["first_token"] == x["first_token_prefill"] for x in g))
# identical-text check between arms (an adapter that did not load would give texts identical to OFF)
for s in ("t2s0", "x_bwf", "x_bwpa"):
    same = sum(a["text"] == b["text"] for a, b in zip(gens[s], gens["off"]))
    print(f"{s}: texts identical to off on {same}/450")
print("x_bwf vs x_bwpa identical texts:", sum(a["text"] == b["text"] for a, b in zip(gens["x_bwf"], gens["x_bwpa"])))

_Q = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-", "\u2013": "-",
                    "\u2014": "-", "\u00a0": " "})


def basic(t):
    return " ".join(unicodedata.normalize("NFKC", str(t or "")).translate(_Q).split())


PREF = sorted(("final answer:", "answer:", "the answer is", "it is", "it will be"), key=len, reverse=True)


def prep(t, cs):
    t = basic(t)
    if not cs:
        t = t.casefold()
    stripped = False
    if t.endswith("."):
        t, stripped = t[:-1].rstrip(), True
    for a, b in (("**", "**"), ("__", "__"), ("*", "*"), ("`", "`"), ('"', '"'), ("'", "'")):
        if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b) and t[len(a):len(t) - len(b)].strip():
            t = t[len(a):len(t) - len(b)].strip()
            if not stripped and t.endswith("."):
                t = t[:-1].rstrip()
            break
    low = t.casefold()
    for p in PREF:
        if low == p:
            return None
        if low.startswith(p + " "):
            t = t[len(p) + 1:].strip()
            break
        if p.endswith(":") and low.startswith(p):
            t = t[len(p):].strip()
            break
    return t or None


NUM = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def my_primary(g, text, term):
    if not term:
        return False
    k = g["kind"]
    cs = k == "unknown" or (k == "exact" and g.get("case"))
    b = prep(text, cs)
    if b is None:
        return False
    if k == "number":
        m = re.match(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?", b)
        if not m:
            return False
        tok, rest = m.group(0), b[m.end():].strip()
        if rest and rest not in g.get("units", []):
            return False
        if "/" in tok:
            n_, d_ = [x.strip() for x in tok.split("/")]
            if not (NUM.match(n_) and NUM.match(d_)) or "." in n_ + d_:
                return False
            ni, di = int(n_.replace(",", "")), int(d_.replace(",", ""))
            if di <= 0 or math.gcd(abs(ni), di) != 1 or Fraction(str(g["value"])).denominator == 1:
                return False
            v = Fraction(ni, di)
        else:
            if not NUM.match(tok) and not re.match(r"^[+-]?\.\d+$", tok):
                return False
            v = Fraction(tok.replace(",", ""))
        return v == Fraction(str(g["value"]))
    if k == "kv":
        return b == (str(g["value"]) if g["mode"] == "code" else str(g["value"]).casefold())
    if k == "unknown":
        return b == "UNKNOWN"
    if k == "exact":
        return b == (g["value"] if g.get("case") else g["value"].casefold())
    raise ValueError(k)


ABST = ("unknown", "not mentioned", "not stated", "does not say", "doesn" + chr(39) + "t say", "not provided",
        "no information", "cannot be determined", "can" + chr(39) + "t be determined", "not in the passage",
        "not given", "does not mention", "doesn" + chr(39) + "t mention", "not specified", "not included")


def word(v, low):
    return re.search(r"(?<![\w])" + re.escape(str(v).casefold()) + r"(?![\w])", low) is not None


def my_lenient(g, text):
    t = basic(text)
    low = t.casefold()
    k = g["kind"]
    if k == "number":
        ns = [Fraction(m.group(0).replace(",", "")) for m in
              re.finditer(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", low.replace("%", " "))]
        hedge = re.search(r"\b(or|either)\b", low) is not None and len(set(ns)) >= 2
        return bool(ns) and ns[-1] == Fraction(str(g["value"])) and not hedge
    if k == "kv":
        if "unknown" in low:
            return False
        return word(g["value"], low) and not any(word(d, low) for d in g.get("distractors", []))
    if k == "unknown":
        return any(p in low for p in ABST) and not any(word(v, low) for v in g.get("passage_values", []))
    if k == "exact":
        v = g["value"] if g.get("case") else g["value"].casefold()
        src = t if g.get("case") else low
        if re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", src) is None:
            return False
        if v.casefold() in ("yes", "no") and re.search(r"\b" + ("no" if v.casefold() == "yes" else "yes") + r"\b", low):
            return False
        return re.search(r"\b(or|either)\b", low) is None
    raise ValueError(k)


graded = {}
dis = defaultdict(int)
for s, g in gens.items():
    graded[s] = {}
    for r, x in zip(rows, g):
        fp = RG.grade(r["gold"], x["text"], bool(x["terminated"]))["correct"]
        fl = RG.grade_lenient(r["gold"], x["text"])
        mp = my_primary(r["gold"], x["text"], bool(x["terminated"]))
        ml = my_lenient(r["gold"], x["text"])
        if fp != mp:
            dis[(s, "primary")] += 1
            print("PRIMARY DISAGREE", s, r["pid"], repr(x["text"]), fp, mp)
        if fl != ml:
            dis[(s, "lenient")] += 1
            print("LENIENT DISAGREE", s, r["pid"], repr(x["text"]), fl, ml)
        graded[s][r["pid"]] = {"primary": mp, "lenient": ml, "frozen_primary": fp, "frozen_lenient": fl}
print("independent-vs-frozen grader disagreements (prompt level):", dict(dis) or 0)
CATS = ["unit_conv", "pct_ratio", "kv_recall", "unknown_twin", "format_follow"]


def items(s, gr):
    out = defaultdict(dict)
    for r in rows:
        out[r["category"]][r["item"]] = out[r["category"]].get(r["item"], True) and graded[s][r["pid"]][gr]
    return out


IT = {s: {gr: items(s, gr) for gr in ("primary", "lenient")} for s in gens}


def mcn(l, g):
    n = l + g
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(l, g) + 1)) / 2 ** n)


def holm(ps):
    o = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    m, run, out = len(o), 0.0, {}
    for i, (k, p) in enumerate(o):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def kmin(m):
    k = 1
    while not (m * 2.0 ** (1 - k) < 0.05):
        k += 1
    return min(k, 12)


def compare(arm, ref):
    res = {}
    for gr in ("primary", "lenient"):
        cat = {}
        for c in CATS:
            A, Rf = IT[arm][gr][c], IT[ref][gr][c]
            assert set(A) == set(Rf) and len(A) == 75
            lost = sum(Rf[i] and not A[i] for i in Rf)
            gained = sum(A[i] and not Rf[i] for i in Rf)
            cat[c] = dict(ref=sum(Rf.values()), arm=sum(A.values()), lost=lost, gained=gained, net=gained - lost,
                          p=mcn(lost, gained))
        adj = holm({c: v["p"] for c, v in cat.items()})
        for c in cat:
            v = cat[c]
            v["p_holm"] = adj[c]
            v["detected"] = v["lost"] > v["gained"] and adj[c] < 0.05
            v["over"] = v["net"] < -11
            v["flag"] = v["net"] < -5
            v["binding"] = v["ref"] >= kmin(len(CATS))
        pooled = sum(v["net"] for v in cat.values())
        margin = -0.04 * 75 * len(CATS)
        res[gr] = dict(cats=cat, pooled=pooled, lost=sum(v["lost"] for v in cat.values()),
                       gained=sum(v["gained"] for v in cat.values()), margin=margin, pooled_over=pooled < margin)
    det = any(v["detected"] for gr in res for v in res[gr]["cats"].values())
    over = any(res[gr]["pooled_over"] or any(v["over"] for v in res[gr]["cats"].values()) for gr in res)
    verdict = "LOSS DETECTED" if det else ("MARGIN EXCEEDED" if over else "RETAINED")
    k = sum(v["binding"] for v in res["primary"]["cats"].values())
    return verdict, k, res


ret = json.load(open(D + "/analysis/RET_RESULTS.json"))
allok = True
summary = {}
for name, arm, ref in [("t2s0_vs_off", "t2s0", "off"), ("x_bwf_vs_off", "x_bwf", "off"), ("x_bwpa_vs_off", "x_bwpa", "off"),
                       ("sparse_vs_supa", "off", "supa"), ("t2s0_vs_supa", "t2s0", "supa"),
                       ("x_bwf_vs_supa", "x_bwf", "supa"), ("x_bwpa_vs_supa", "x_bwpa", "supa")]:
    verdict, k, res = compare(arm, ref)
    summary[name] = {"verdict": verdict, "k": k, "res": res}
    print(f"\n== {name}: {verdict}, binding {k}/5 (min lost to fail {kmin(5)}), pooled primary {res['primary']['pooled']:+d} "
          f"(lost {res['primary']['lost']} gained {res['primary']['gained']}) lenient {res['lenient']['pooled']:+d} "
          f"(lost {res['lenient']['lost']} gained {res['lenient']['gained']}) (margin {res['primary']['margin']:g})")
    for gr in ("primary", "lenient"):
        for c, v in res[gr]["cats"].items():
            print(f"   {gr:8s} {c:14s} ref {v['ref']:2d} arm {v['arm']:2d} lost {v['lost']:2d} gained {v['gained']:2d} "
                  f"net {v['net']:+3d} p {v['p']:.4g} pHolm {v['p_holm']:.4g} det {v['detected']} over {v['over']} flag {v['flag']}")
    if name in ret["comparisons"]:
        rc = ret["comparisons"][name]
        same = rc["verdict"] == verdict and rc["binding"]["k"] == k
        for gr in ("primary", "lenient"):
            same &= rc["graders"][gr]["pooled"]["net"] == res[gr]["pooled"]
            same &= rc["graders"][gr]["pooled"]["margin"] == res[gr]["margin"]
            for c, v in res[gr]["cats"].items():
                rr = rc["graders"][gr]["categories"][c]
                same &= (rr["ref_correct"], rr["arm_correct"], rr["lost"], rr["gained"], rr["net"], rr["detected_loss"],
                         rr["margin_exceeded"], rr["flag_net_below_minus5"]) == (
                    v["ref"], v["arm"], v["lost"], v["gained"], v["net"], v["detected"], v["over"], v["flag"])
                same &= abs(rr["p"] - v["p"]) < 1e-12 and abs(rr["p_holm"] - v["p_holm"]) < 1e-12
        print("   MATCHES RET_RESULTS:", same)
        allok &= same
    else:
        print("   (not a RET_RESULTS comparison; report-only cumulative drift computed here)")
print("\nALL GATING/REPORTED COMPARISONS MATCH RET_RESULTS:", allok)
json.dump({"graded": graded}, open("graded.json", "w"))
json.dump({n: {"verdict": v["verdict"], "k": v["k"], "res": v["res"]} for n, v in summary.items()}, open("recompute_summary.json", "w"), indent=1)
