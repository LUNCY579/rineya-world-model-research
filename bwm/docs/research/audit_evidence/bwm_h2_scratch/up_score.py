import json, sys, collections
S = "C:/Users/clclc/AppData/Local/Temp/claude/bwm_h2_scratch/dl_upstream_hard/"
out = {}
for m in ("deepseek-r1-api", "o1-preview_chat", "o1-mini_chat"):
    d = json.load(open(S + m + ".json", encoding="utf-8"))
    ins = d["instances"]
    print("==", m, "prompt_type", d.get("prompt_type"), "n", len(ins), "inst keys", list(ins[0].keys()))
    steps = collections.Counter(str(i.get("steps")) for i in ins)
    print("  steps dist", sorted(steps.items(), key=lambda t: int(t[0])))
    s0 = {}; fin = {}
    for i in ins:
        sw = i["step_wise_dict"]
        ks = sorted(sw.keys(), key=int)
        s0[i["instance_id"]] = bool(sw["0"]["sound_verifier_json"]["validation_info"].get("is_valid_plan"))
        fin[i["instance_id"]] = bool(sw[ks[-1]]["sound_verifier_json"]["validation_info"].get("is_valid_plan"))
    print("  step-0 (direct) valid:", sum(s0.values()), "| last-step (loop) valid:", sum(fin.values()),
          "| act_correct field True:", sum(str(i.get("act_correct")) == "True" for i in ins))
    out[m] = {"step0": s0, "final": fin}
    if m == "deepseek-r1-api" or m == "o1-mini_chat":
        msg = ins[0].get("messages") or ins[0]["step_wise_dict"]["0"].get("sent_messages")
        txt = msg[0]["content"] if isinstance(msg[0], dict) else str(msg[0])
        print("  prompt has '[PLAN]' count:", txt.count("[PLAN]"), "| 'example' mentions:", txt.lower().count("example"), "| len", len(txt))
        print("  prompt tail:", repr(txt[-400:]))
json.dump(out, open("C:/Users/clclc/AppData/Local/Temp/claude/bwm_h2_scratch/upstream_flags.json", "w"))
