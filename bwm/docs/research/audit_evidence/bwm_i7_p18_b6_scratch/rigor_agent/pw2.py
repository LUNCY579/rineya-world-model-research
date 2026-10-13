exec(open("pw.py").read().split("for n1, n2 in")[0])
for n in (60, 100):
    for a, b in ((.10, .10), (.05, .15), (.05, .20), (.10, .25)):
        t = tri(n, a, b); print(f"retention n={n} p10={a} p01={b} P(block: c>b & p<.05)={sum(w for (bb, c), w in t.items() if c > bb and pval(bb, c) < .05):.3f}")
# P1 at n=150: false MET over null discordances
for pd in (.04, .06, .1, .16, .24):
    t = tri(150, pd/2, pd/2); print("P1 null disc", pd, "P(MET)", round(sum(w for (b, c), w in t.items() if b > c and pval(b, c) < .05 and (b - c) / 150 >= .05), 4))
