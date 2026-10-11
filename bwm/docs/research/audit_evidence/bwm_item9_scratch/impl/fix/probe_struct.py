import sys
sys.path.insert(0, ".")
from valread import val_read, L
gt = L.planbench_gt()
p = L.planbench_path(2)
G = L.plan_to_pddl(gt[2])
a, b, c = G.strip().split("\n")
print(val_read(p, G, raw=True)[:400])
cases = {
    "clean": G,
    "lead_prose": "Here is my plan:\n" + G,
    "lead_star": "*\n" + G,
    "between_word": f"{a} and\n{b}\n{c}\n",
    "trail_word": G + "done\n",
    "trail_word_then_junkgroup": G + "done\n(fly-airplane a0 l0-0 l0-0)\n",
    "stray_close_between": f"{a}\n)\n{b}\n{c}\n",
    "stray_close_trailing": G + ")\n",
    "nested_all": "(" + G + ")",
    "nested_first": f"({a})\n{b}\n{c}\n",
    "empty_group_lead": "()\n" + G,
    "empty_group_trail": G + "()\n",
    "unclosed_last": G + "(fly-airplane a0 l0-0 l1-0\n",
    "unclosed_mid": f"{a}\n(fly-airplane a1 l1-0 l0-0\n{c}\n",
    "numbered_dot": f"1. {a}\n2. {b}\n3. {c}\n",
    "numbered_paren": f"1) {a}\n2) {b}\n3) {c}\n",
    "step_colon": f"Step 1: {a}\nStep 2: {b}\nStep 3: {c}\n",
    "dash_list": f"- {a}\n- {b}\n- {c}\n",
    "star_list": f"* {a}\n* {b}\n* {c}\n",
    "ts_inc": f"0: {a}\n1: {b}\n2: {c}\n",
    "ts_inc_float": f"0.000: {a}\n0.5: {b}\n7.25: {c}\n",
    "ts_inc_dur": f"0: {a} [1]\n1: {b} [1]\n2: {c} [1]\n",
    "ts_inc_dur_float": f"0.000: {a} [1.000]\n1.000: {b} [1.000]\n2.000: {c} [1.000]\n",
    "dur_only": f"{a} [1]\n{b} [1]\n{c} [1]\n",
    "ts_equal_all": f"0: {a}\n0: {b}\n0: {c}\n",
    "ts_equal_12": f"0: {a}\n0: {b}\n1: {c}\n",
    "ts_dec": f"2: {a}\n1: {b}\n0: {c}\n",
    "ts_perm_sorted_valid": f"1: {b}\n0: {a}\n2: {c}\n",
    "ts_mixed_first": f"0: {a}\n{b}\n{c}\n",
    "ts_mixed_mid": f"{a}\n5: {b}\n{c}\n",
    "ts_mixed_last": f"{a}\n{b}\n0: {c}\n",
    "ts_mixed_last_big": f"{a}\n{b}\n9: {c}\n",
    "ts_neg": f"-1: {a}\n0: {b}\n1: {c}\n",
    "ts_nocolon": f"0 {a}\n1 {b}\n2 {c}\n",
    "time_kw": f"Time: 3\n{G}",
    "upper_ops": G.upper(),
    "keyword_group_trail": G + "(at p0 l0-0)\n",
    "keyword_arg": G.replace("p0 a1 l1-0", "p0 a1 at", 1),
    "keyword_op_lead": "(and p0)\n" + G,
    "underscore_name": G + "(_x)\n",
    "dot_in_name": G.replace("p0 a1", "p0.a1", 1),
    "comma_list_one_line": G.replace(")\n(", "), (").replace("\n", ""),
    "semicolon_inline": f"{a} ; first\n{b} ; second\n{c}\n",
    "hash_comment": f"# plan\n{G}",
    "json_list": '["' + '", "'.join([a, b, c]) + '"]',
    "md_fence": "```\n" + G + "```\n",
    "md_fence_pddl": "```pddl\n" + G + "```\n",
    "empty": "",
    "only_ws": "  \n\t\n",
    "time_ts_zero_dur": f"0: {a} [0]\n1: {b} [0]\n2: {c} [0]\n",
}
for k, t in cases.items():
    kind, acts, valid, rc = val_read(p, t)
    print(f"{k:28s} {kind:5s} valid={valid!s:5s} rc={rc} n={None if acts is None else len(acts)} "
          f"{'' if acts is None else [(tt, ' '.join(x)) for tt, x in acts]}"[:300])
