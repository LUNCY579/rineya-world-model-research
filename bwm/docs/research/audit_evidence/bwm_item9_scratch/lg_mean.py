import statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import lg_size as L
B, M, T, dom = L.B, L.M, L.T, L.dom
gts = L.gt_plans(L.PB / "prompts/logistics/task_1_plan_generation.json")
for lo, hi in ((1, 4), (5, 7)):
    fr, pr, pr12 = [], [], []
    for iid, plan in sorted(gts.items()):
        objs, init, goal = M.parse_problem(L.LG / "generated_basic" / f"instance-{iid}.pddl")
        npk = sum(1 for f in init if f[0] == "obj")
        if not lo <= npk <= hi:
            continue
        states = [init]
        for a in plan:
            states.append(M.step(dom, states[-1], a)[0])
        for t, (s0, a) in enumerate(zip(states, plan)):
            ns = states[t + 1]
            fr.append(T(L.fwd_prompt(s0, a, False)) + T(B.forward_target_delta(s0, ns, True)) + 1)
            p = T(L.prop_prompt(s0, goal, False))
            pr.append(p + T(B.target_actions(plan[t:])) + 1)
            pr12.append(p + T(B.target_actions(plan[t:t + 12])) + 1)
    print(f"packages {lo}-{hi}: F rows n={len(fr)} mean {st.mean(fr):.0f} max {max(fr)} | proposer suffix rows n={len(pr)} "
          f"mean {st.mean(pr):.0f} max {max(pr)} >640 {sum(x > 640 for x in pr)} | 12-action-prefix rows mean {st.mean(pr12):.0f} max {max(pr12)}")
