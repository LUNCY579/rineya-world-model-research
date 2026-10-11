import statistics as st
from pathlib import Path
src = Path("../lg_size.py").read_text()
head = src.split("def gt_plans")[0]
body = "def gt_plans" + src.split("def gt_plans")[1].split("def summarize")[0]
exec(compile(head + "\n" + body, "lgsz", "exec"))
gts = gt_plans(PB / "prompts/logistics/task_1_plan_generation.json")
res = {}
for iid, plan in sorted(gts.items()):
    objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{iid}.pddl")
    npk = sum(1 for f in init if f[0] == "obj")
    s = init; un = [len(goal - s)]
    for a in plan:
        s = M.step(dom, s, a)[0]; un.append(len(goal - s))
    # gaps between strict decreases of unmet (steps), starting at step 0
    last = 0; gaps = []; first = None
    for t in range(1, len(un)):
        if un[t] < un[t-1]:
            gaps.append(t - last); last = t
            if first is None: first = t
    # also: any non-monotone (unmet increases)? 
    inc = sum(un[t] > un[t-1] for t in range(1, len(un)))
    res.setdefault(npk, []).append((first, max(gaps) if gaps else 0, inc, len(plan)))
for k in sorted(res):
    v = res[k]
    print(k, len(v), 'first-goal step med', st.median(x[0] for x in v), 'max', max(x[0] for x in v),
          '| max gap med', st.median(x[1] for x in v), 'max', max(x[1] for x in v),
          '| share maxgap>12', round(sum(x[1] > 12 for x in v)/len(v), 2), '>24', round(sum(x[1] > 24 for x in v)/len(v),2),
          '| plans with unmet increase', sum(x[2] > 0 for x in v))
