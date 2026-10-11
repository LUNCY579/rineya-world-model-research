"""Edge cases: raw plan texts through plan_from_pddl + execute vs val_run."""
import sys
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

gt = L.planbench_gt()
p = L.planbench_path(2)
objs, init, goal = L.parse_problem(p)
G = L.plan_to_pddl(gt[2])
cases = {
    "gt": G,
    "gt_upper": G.upper(),
    "gt_mixed_case_names": G.replace("p0", "P0"),
    "undeclared_obj_prefix": "(load-airplane p9 a0 l1-0)\n" + G,
    "undeclared_obj_after_goal": G + "(load-airplane p9 a0 l1-0)\n",
    "wrong_arity_after_goal": G + "(load-truck p0 t0)\n",
    "unknown_op_after_goal": G + "(pick-up p0)\n",
    "extra_arg_after_goal": G + "(fly-airplane a0 l0-0 l1-0 c0)\n",
    "timestamps": "\n".join(f"{i}: {ln} [1]" for i, ln in enumerate(G.strip().splitlines())) + "\n",
    "comment": "; hello (x y)\n" + G + "; cost = 3 (unit cost)\n",
    "empty": "",
    "selfloop_city_mismatch": "(drive-truck t1 l1-0 l1-0 c0)\n" + G,
    "fly_from_wrong": "(fly-airplane a0 l0-0 l0-0)\n" + G,
    "unclosed_last": G + "(fly-airplane a0 l0-0 l1-0\n",
    "duplicated_param_types": "(load-airplane a0 a0 l1-0)\n" + G,
}
for k, txt in cases.items():
    plan = L.plan_from_pddl(txt)
    ex = L.execute(init, goal, plan)
    v = L.val_run(p, txt)
    agree = ex["success"] == v["valid"]
    print(f"{k:28s} exec_success={ex['success']!s:5} val_valid={v['valid']!s:5} rc={v['returncode']} "
          f"{'AGREE' if agree else 'DISAGREE'} n_actions={len(plan)}")
