import sys
sys.path.insert(0, ".")
from valread import val_read, L
gt = L.planbench_gt(); P = gt[4]; p = L.planbench_path(4)
R = lambda a: "(" + " ".join(a) + ")"
for name, txt in {"exp_ts": "".join(f"{j + 1}e0: {R(a)}\n" for j, a in enumerate(P)),
                  "exp_one": "1e0: " + L.plan_to_pddl(P),
                  "num_name_inside": L.plan_to_pddl(P).replace("p1", "1p", 1),
                  "time_between_num": R(P[0]) + "\ntime 3\n" + L.plan_to_pddl(P[1:]),
                  "time_float": "time 2.5\n" + L.plan_to_pddl(P),
                  "ts_then_junk": "0: junk " + L.plan_to_pddl(P),
                  "dur_bad": R(P[0]) + " [x]\n" + L.plan_to_pddl(P[1:]),
                  "dur_unclosed_trail": L.plan_to_pddl(P) + "[1\n",
                  "two_colons": "0:: " + L.plan_to_pddl(P),
                  }.items():
    out = val_read(p, txt, raw=True)
    print("=====", name, val_read(p, txt)[0::2])
    print(out[:300].replace("\r", ""))
