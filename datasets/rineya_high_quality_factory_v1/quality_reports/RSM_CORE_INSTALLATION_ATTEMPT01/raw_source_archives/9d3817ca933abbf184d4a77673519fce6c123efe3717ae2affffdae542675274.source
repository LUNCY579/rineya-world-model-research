"""Producer-side bounded original reusable-procedure tasks."""


def make_task(index, split, rng):
    dev = split == "dev_challenge"
    names = ["sprout", "weave", "threshold", "cascade"] if dev else ["bump", "fold", "gate"]
    bump, fold, gate = names[:3]
    amount = 1 + index % 4
    procedures = {
        bump: [dict(op="add", register="x", value=amount)],
        fold: [dict(op="scale", register="y", value=index % 3 - 1), dict(op="swap")],
        gate: [{"op": "if", "test": dict(register="x", cmp=["lt", "eq", "ge"][index % 3], value=index % 5 - 1),
                "then": [dict(op="call", name=bump)], "else": [dict(op="add", register="y", value=-amount)]}],
    }
    mode = index % 6
    main = [dict(op="call", name=name) for name in
            [(bump, bump), (gate,), (gate, fold, gate), (bump, gate), (fold, bump, fold), (gate, gate, bump)][mode]]
    if mode == 3:
        main[0] = {"op": "if", "test": dict(register="y", cmp="lt", value=0),
                   "then": [dict(op="call", name=fold)], "else": [dict(op="call", name=bump)]}
    if dev:
        procedures[names[3]] = [dict(op="call", name=gate), dict(op="call", name=fold)]
        main.append(dict(op="call", name=names[3]))
    task = dict(kind="skills", state=dict(x=rng.randint(-4, 5), y=rng.randint(-3, 5)), procedures=procedures, main=main)
    mechanism = ["repeated_procedure", "conditional_subgoal", "state_dependent_reuse", "conditional_composition",
                 "arithmetic_composition", "repeated_conditional"][mode]
    return task, mechanism, 4 if dev else 2 if mode == 0 else 3, "Finite supplied procedure execution; no learned hierarchy or skill acquisition claim."
