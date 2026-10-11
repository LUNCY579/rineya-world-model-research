import json, sys, collections
from scipy.stats import binomtest, fisher_exact, beta
R = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/"
def load(p):
    rows = [json.loads(l) for l in open(R+p, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), ("dup", p)
    return {r["id"]: r for r in rows}
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k)
    return round(lo,4), round(hi,4)
def mcn(b, c):
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else 1.0
pairs = {"H1": ("BW_B3_HARD/h1_direct/plans.jsonl", "BW_TEST_B2/hard_direct/plans.jsonl"),
         "H2": ("BW_B3_HARD/h2_beam/plans.jsonl", "BW_TEST_B2/hard_beam/plans.jsonl")}
for name, (a, b) in pairs.items():
    A, B = load(a), load(b)
    print(name, "n", len(A), len(B), "same ids", set(A) == set(B))
    # val vs success field consistency
    for lab, D in (("new", A), ("b2", B)):
        mism = [i for i, r in D.items() if bool(r.get("val")) != bool(r.get("success"))]
        vnone = [i for i, r in D.items() if r.get("val") is None]
        print(f"  {lab}: VAL-success={sum(bool(r['val']) for r in D.values())} success-field={sum(bool(r['success']) for r in D.values())} val!=success:{len(mism)} val None:{len(vnone)}")
    ka = sum(bool(A[i]["val"]) for i in A); kb = sum(bool(B[i]["val"]) for i in A)
    b_ = sum(bool(A[i]["val"]) and not bool(B[i]["val"]) for i in A)
    c_ = sum(bool(B[i]["val"]) and not bool(A[i]["val"]) for i in A)
    both = sum(bool(A[i]["val"]) and bool(B[i]["val"]) for i in A)
    print(f"  new {ka}/110 CI {cp(ka,110)} ; b2 {kb}/110 CI {cp(kb,110)} ; new_only {b_} b2_only {c_} both {both} ; exact McNemar p={mcn(b_,c_)!r}")
    # mid-p
    from scipy.stats import binom
    n = b_ + c_; k = min(b_, c_)
    midp = 2*(binom.cdf(k, n, .5) - 0.5*binom.pmf(k, n, .5))
    print(f"  mid-p {midp:.5f}; one-sided exact {binom.cdf(k,n,.5):.5f}")
    # strata by block count
    st = collections.defaultdict(lambda: [0,0,0])
    for i in A:
        nb = A[i]["n_blocks"]; st[nb][0]+=1; st[nb][1]+=bool(A[i]["val"]); st[nb][2]+=bool(B[i]["val"])
    print("  by blocks (n, new, b2):", {k: tuple(v) for k, v in sorted(st.items())})
    inr = [i for i in A if A[i]["n_blocks"] <= 10]; outr = [i for i in A if A[i]["n_blocks"] > 10]
    for lab, S in (("6-10", inr), ("11-15", outr)):
        print(f"  {lab}: n={len(S)} new={sum(bool(A[i]['val']) for i in S)} b2={sum(bool(B[i]['val']) for i in S)}")
    fails = collections.Counter((A[i].get("fail"), bool(A[i]["val"])) for i in A)
    print("  new fail reasons:", dict(fails))
    failsb = collections.Counter((B[i].get("fail"), bool(B[i]["val"])) for i in B)
    print("  b2 fail reasons:", dict(failsb))
    if name == "H1":
        for lab, D in (("new", A), ("b2", B)):
            gt = [D[i]["gen_tokens"] for i in D]
            cap = sum(g >= 256 for g in gt)
            print(f"  {lab}: gen_tokens>=256 (cap hit): {cap}; max {max(gt)}; by blocks cap-hit:",
                  dict(collections.Counter(D[i]['n_blocks'] for i in D if D[i]['gen_tokens'] >= 256)))
        print("  keys:", list(A[next(iter(A))].keys()))
# Fisher unpaired vs published aggregates
print("\nFisher exact (two-sided), unpaired, n=110 each")
pub = {"DeepSeek-R1 ~59/110 (53.6%)": 59, "o1-preview 26/110 (23.6%)": 26, "o1-mini 10% (11/110)": 11}
for sysname, k in (("H2 Pa_v3 beam 88", 88), ("B2 beam 73", 73), ("H1 Pa_v3 direct 13", 13), ("B2 direct 5", 5)):
    for pn, pk in pub.items():
        tab = [[k, 110-k], [pk, 110-pk]]
        odds, p2 = fisher_exact(tab, alternative="two-sided")
        _, pg = fisher_exact(tab, alternative="greater")
        print(f"  {sysname} vs {pn}: diff={100*(k-pk)/110:+.1f}pp OR={odds:.3f} p2={p2:.3g} p_greater={pg:.3g}")
# R1 sensitivity: 53.6% of 110 = 58.96; check which integers round to 53.6%
for k in range(55, 63):
    print(k, round(100*k/110, 2))
print("R1 range sensitivity for H2 88:")
for pk in (58, 59, 60, 61):
    print(pk, fisher_exact([[88,22],[pk,110-pk]])[1])
