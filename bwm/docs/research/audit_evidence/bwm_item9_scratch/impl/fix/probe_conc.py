import sys
sys.path.insert(0, ".")
from valread import val_read, L
gt = L.planbench_gt()
# find an instance whose GT has two consecutive loads into the same vehicle at the same location
pick = None
for i, plan in sorted(gt.items()):
    for k in range(len(plan) - 1):
        x, y = plan[k], plan[k + 1]
        if x[0] == y[0] and x[0].startswith("load") and x[2] == y[2] and x[3] == y[3]:
            pick = (i, k); break
    if pick: break
i, k = pick
P = gt[i]; p = L.planbench_path(i)
print("instance", i, "k", k, P[k], P[k + 1], "len", len(P))
R = lambda a: "(" + " ".join(a) + ")"
def timed(times):
    return "".join(f"{t}: {R(a)}\n" for t, a in zip(times, P))
base = list(range(len(P)))
cases = {}
t = base[:]; t[k + 1] = t[k]; cases["conc_two_loads"] = timed(t)
t = base[:]; cases["dup_same_time"] = timed(base[:k + 1]) + f"{k}: {R(P[k])}\n" + "".join(f"{j + 1}: {R(P[j])}\n" for j in range(k + 1, len(P)))
# load + following action that needs the vehicle to stay (find a drive/fly after k+1)
j = next(j for j in range(k + 2, len(P)) if P[j][0] in ("drive-truck", "fly-airplane") and P[j][1] == P[k][2])
t = base[:]; t[j] = t[j - 1]; cases["conc_prev_with_move"] = timed(t)
# a move and the following unload at same time: unload's pre added by the move
t = base[:]; t[j + 1] = t[j]; cases["conc_move_then_unload"] = timed(t)
cases["all_same_time"] = timed([0] * len(P))
for name in ("time", "TIME", "Time", "times", "timer"):
    cases[f"kw_{name}_lead_num"] = f"{name} 3\n" + L.plan_to_pddl(P)
    cases[f"kw_{name}_between"] = R(P[0]) + f"\n{name}\n" + L.plan_to_pddl(P[1:])
    cases[f"kw_{name}_trail"] = L.plan_to_pddl(P) + f"{name}\n"
cases["dotfloat_ts"] = "".join(f".{j + 1}: {R(a)}\n" for j, a in enumerate(P))
cases["float_trailing_dot_ts"] = "".join(f"{j + 1}.: {R(a)}\n" for j, a in enumerate(P))
cases["exp_ts"] = "".join(f"{j + 1}e0: {R(a)}\n" for j, a in enumerate(P))
cases["plus_ts"] = "".join(f"+{j + 1}: {R(a)}\n" for j, a in enumerate(P))
cases["kw_and_between"] = R(P[0]) + "\nand\n" + L.plan_to_pddl(P[1:])
cases["kw_at_between"] = R(P[0]) + "\nat\n" + L.plan_to_pddl(P[1:])
cases["kw_at_lead_num"] = "at 3\n" + L.plan_to_pddl(P)
cases["kw_total_time_trail"] = L.plan_to_pddl(P) + "total-time 5\n"
for kname, txt in cases.items():
    kind, acts, valid, rc = val_read(p, txt)
    print(f"{kname:28s} {kind:5s} valid={valid!s:5s} rc={rc} n={None if acts is None else len(acts)}")
out = val_read(p, cases["conc_two_loads"], raw=True)
print(out[out.find("Plan Validation"):][:600])
