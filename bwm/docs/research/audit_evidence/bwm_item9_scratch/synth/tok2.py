import sys
sys.argv = ["x"]
exec(open("tok.py").read().split("MERGE =")[0])
mx = 0; mxi = 0
for i, plan in sorted(GT.items()):
    s, g = parse(CL / f"instances/logistics/generated_basic/instance-{i}.pddl")
    for a in plan:
        ns, ok = step(s, a)
        mx = max(mx, n(B.forward_target_delta(s, ns, ok)) + 1); s = ns
    # an illegal action delta
    mxi = max(mxi, n(B.forward_target_delta(s, s, False)) + 1)
print("max legal delta target tokens (+EOS) on all GT steps:", mx, "| illegal:", mxi)
