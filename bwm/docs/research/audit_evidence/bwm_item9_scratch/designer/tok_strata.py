import re, json, statistics as st, sys
from pathlib import Path
src = Path("../lg_size.py").read_text()
head = src.split("def gt_plans")[0]
body = "def gt_plans" + src.split("def gt_plans")[1].split("def summarize")[0]
exec(compile(head + "\n" + body, "lgsz", "exec"))
gts = gt_plans(PB / "prompts/logistics/task_1_plan_generation.json")
for lo, hi in ((1, 3), (4, 4), (5, 7), (1, 4)):
    fr, pr, plen = [], [], []
    for iid, plan in sorted(gts.items()):
        objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{iid}.pddl")
        npk = sum(1 for f in init if f[0] == "obj")
        if not lo <= npk <= hi:
            continue
        plen.append(len(plan))
        states = [init]
        for a in plan:
            states.append(M.step(dom, states[-1], a)[0])
        for t, (s0, a) in enumerate(zip(states, plan)):
            ns = states[t + 1]
            fr.append(T(fwd_prompt(s0, a, False)) + T(B.forward_target_delta(s0, ns, True)) + 1)
            p = T(prop_prompt(s0, goal, False))
            pr.append(p + T(B.target_actions(plan[t:])) + 1)
    print(f"pk {lo}-{hi}: inst {len(plen)} planlen max {max(plen)} | F rows mean {st.mean(fr):.0f} max {max(fr)} | "
          f"Pa suffix rows n={len(pr)} mean {st.mean(pr):.0f} max {max(pr)} >512 {sum(x>512 for x in pr)} >640 {sum(x>640 for x in pr)}")
