"""C1 extras: (a) Supa lenient-parse accuracy on state-CHANGING items; (b) paired CIs for exact-Δ
ridge Sparse-Supa at k1 (multi format) and one-frame, dev-selected layer and fixed L16/L20."""
import json, numpy as np, vh
from vh import env, fidelity
from c1_gen_lenient import lenient  # noqa (re-runs its summary print; harmless)
D = vh.ROOT / "runs" / "bank_pilot" / "supa"
rows = {r["item_id"]: r for r in vh.rows_all()}
for r in list(rows.values()):
    if r["k"] == 1:
        rows[r["cand_id"] + "-one"] = dict(r, item_id=r["cand_id"] + "-one")
out = {}
for l in (D / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines():
    g = json.loads(l); r = rows[g["item_id"]]; fr = g["item_id"].rsplit("-", 1)[1]
    if fr not in ("one", "k1") or r["states"][-1] == r["s0"]:
        continue
    s, _ = lenient(r["family"], g["text"])
    o = out.setdefault(f"{env.SHORT[r['family']]}|{fr}", [0, 0])
    o[0] += 1; o[1] += int(s is not None and env.dumps(s) == env.dumps(r["states"][-1]))
print("supa lenient state-exact on items whose true state changes:", {k: (v[0], round(v[1] / v[0], 3)) for k, v in out.items()})
res = {"supa_lenient_exact_on_changed": {k: v[1] / v[0] for k, v in out.items()}}
for fam in vh.FAMS:
    for frame in ("one", 1):
        sp = vh.sp3(fam, frame)
        tgt = lambda r, f=fam: fidelity.delta_of(f, r["s0"], r["states"][-1])
        nb = len(tgt(sp["train"][0]))
        e = {}
        for lay in (None, 16, 20):
            pa, pb = vh.probe("sparse", sp, tgt, nb, layer=lay), vh.probe("supa", sp, tgt, nb, layer=lay)
            xa, xb = vh.exact_items(pa["P"], pa["Y"]), vh.exact_items(pb["P"], pb["Y"])
            e[str(lay or "dev-selected")] = {"sparse": round(float(xa.mean()), 3), "supa": round(float(xb.mean()), 3),
                                             "layers": [pa["layer"], pb["layer"]], **vh.paired_boot(pa["rows"], xa, xb)}
        res[f"{fam}|{frame}"] = e
        print(fam, frame, json.dumps(e))
(vh.ROOT / "docs" / "verification" / "c1_extra.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
