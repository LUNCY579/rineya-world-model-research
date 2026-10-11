import json, os, sys, subprocess
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_obf.py")).read().split("for x in d[\"instances\"]:")[0])
cnt = 0
for x in d["instances"]:
    q = x["query"]
    dom_txt = q.split("[DOMAIN]" + NL)[1].split("[QUERY PROBLEM]" + NL)[0]
    prob_txt = q.split("[QUERY PROBLEM]" + NL)[1].split("[PLAN]")[0]
    dp, pp, plp = [os.path.join(tmp, n) for n in ("d.pddl", "p.pddl", "plan.txt")]
    open(dp, "w").write(dom_txt); open(pp, "w").write(prob_txt)
    dom = M.parse_domain(dp); objs, init, goal = M.parse_problem(pp)
    plan = parse(x["extracted_llm_plan"])
    s = init; okk = True
    for a in plan:
        sc = dom.actions.get(a[0])
        if sc is None or len(a) - 1 != len(sc.params): okk = False; break
        b = dict(zip(sc.params, a[1:]))
        pre, add, dele = M._subst(sc.pre, b), M._subst(sc.add, b), M._subst(sc.dele, b)
        if not pre <= s: okk = False; break
        s = (s - dele) | add
    okk = okk and goal <= s
    if okk and not x["llm_correct"]:
        cnt += 1
        if cnt <= 2:
            print("instance", x["instance_id"]); print("RAW:", repr(x["llm_raw_response"][:600])); print("EXTRACTED:", repr(x["extracted_llm_plan"][:600]))
            open(plp, "w").write("".join("(" + " ".join(a) + ")" + NL for a in plan))
            print(subprocess.run([str(B.VAL_EXE), "-v", dp, pp, plp], capture_output=True, text=True).stdout[-1500:])
            print(prob_txt)
print("count", cnt)
