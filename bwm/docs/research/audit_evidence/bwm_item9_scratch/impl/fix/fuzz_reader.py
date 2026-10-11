import sys, random, re, time
from collections import Counter
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

STOP = ("Plan Validation details", "Plan executed", "Plan failed", "Goal not satisfied", "Plan Repair", "Plan valid",
        "Plan invalid", "Failed plans", "Successful plans")

def val_listing(out):
    """VAL -v: None for a bad plan / illegal symbol, 'other' if no listing (crash), else sorted [(time, action)]."""
    if "Bad plan description" in out or "Illegal symbol" in out:
        return None
    m = re.search(r"Plan size: (\d+)", out)
    if not m:
        return "other"
    sec = out[m.end():]
    sec = sec[:min([sec.find(s) for s in STOP if sec.find(s) >= 0] + [len(sec)])]
    acts, t = [], None
    for ln in sec.splitlines():
        ln = ln.strip()
        if re.fullmatch(r"-?[0-9.]+:", ln):
            t = float(ln[:-1])
        elif ln.startswith("("):
            acts.append((t, tuple(ln.strip("()").split())))
    assert len(acts) == int(m.group(1)) or int(m.group(1)) < len(acts), (m.group(1), acts)
    return sorted(acts)

SKIP = list(',"\'\{}|!#$%&@^`~') + ["\r", "\t", "\u00a0", "\u2192", "\x7f"]
JUNK = ["and", "then", "at", "Step", "time", "time 3", "time 2.5", "done.", "1.", "-", "*", ")", ":", "]", "[", "2", "1e0",
        "; note", "(", "()", "?", "=", ".5:", "1.:", "Here is the plan:", "```", "```pddl", "1)", "-1:", "1.5e", "1_", "x1-0", "#t", "#T1"]

def render(plan, rng):
    """A raw text for an action list, with random formatting and corruption. Returns (text, tags)."""
    tags = []
    lines = ["(" + " ".join(a) + ")" for a in plan]
    if rng.random() < 0.15:
        lines = [ln.upper() if rng.random() < 0.5 else ln for ln in lines]; tags.append("case")
    r = rng.random()
    n = len(lines)
    if r < 0.12:
        ts = list(range(n)); tags.append("ts_inc")
    elif r < 0.2:
        ts = [round(k * 0.5 + rng.random() * 0.4, 3) for k in range(n)]; tags.append("ts_float")
    elif r < 0.3 and n >= 2:
        ts = list(range(n)); k = rng.randrange(n - 1); ts[k + 1] = ts[k]; tags.append("ts_eq")
    elif r < 0.36:
        ts = list(range(n)); rng.shuffle(ts); tags.append("ts_perm")
    elif r < 0.42:
        ts = [rng.randrange(n + 2) if rng.random() < 0.5 else None for _ in range(n)]; tags.append("ts_mixed")
    elif r < 0.45:
        ts = [0] * n; tags.append("ts_all_eq")
    else:
        ts = [None] * n
    lines = [(f"{t}: " if t is not None else "") + ln for t, ln in zip(ts, lines)]
    if rng.random() < 0.12:
        d = rng.choice(["1", "1.000", "0"]); lines = [ln + f" [{d}]" for ln in lines]; tags.append("dur")
    if rng.random() < 0.08:
        lines = [f"{k + 1}{rng.choice(['.', ')', ':'])} {ln}" for k, ln in enumerate(lines)]; tags.append("numbered")
    sep = rng.choice(["\n", "\n", "\n", " ", ", ", "\r\n", "\n\n"])
    text = sep.join(lines) + rng.choice(["", "\n"])
    for _ in range(rng.choice([0, 0, 1, 2, 4])):
        pos = rng.randrange(len(text) + 1); text = text[:pos] + rng.choice(SKIP) + text[pos:]; tags.append("skip")
    if rng.random() < 0.2:
        cs = [m.start() for m in re.finditer(r"\)", text)]
        if cs:
            k = rng.choice(cs); text = text[:k] + ' ' + rng.choice(['c0', 'zzz', 'p0 t0', 'l0-0', 'a1']) + text[k:]; tags.append('extra_arg')
    if rng.random() < 0.15:
        ms = [m.start() for m in re.finditer(r"[ (][a-z]", text)]
        if ms:
            k = rng.choice(ms) + 1; text = text[:k] + rng.choice(['#', '#', '"', '@']) + text[k:]; tags.append('glue')
    if rng.random() < 0.25:
        parts = text.split("\n"); k = rng.randrange(len(parts) + 1)
        parts.insert(k, rng.choice(JUNK)); text = "\n".join(parts); tags.append("junk")
    if rng.random() < 0.06:
        text = rng.choice(["Here is my plan:\n", "Plan:\n", "# plan\n", "```\n"]) + text; tags.append("lead")
    if rng.random() < 0.06:
        text = text + rng.choice(["\nThis plan works.", "\n```", "\n; end", "\n)", "\n(", "\n2"]); tags.append("trail")
    r = rng.random()
    if r < 0.05:
        text = text.rstrip().rstrip(")"); tags.append("unclosed")
    elif r < 0.08 and ")" in text:
        k = rng.choice([m.start() for m in re.finditer(r"\)", text)]); text = text[:k] + text[k + 1:]; tags.append("drop_close")
    elif r < 0.10:
        text = "(" + text + ")"; tags.append("nest")
    return text, tags

def cases(rng, n):
    gt = L.planbench_gt()
    ids = sorted(gt)
    out = []
    while len(out) < n:
        iid = rng.choice(ids)
        objs, init, goal = L.parse_problem(L.planbench_path(iid))
        plan = list(gt[iid])
        r = rng.random()
        if r < 0.15:  # self-loops sprinkled in (concurrency fodder)
            s = init; new = []
            for a in plan:
                if rng.random() < 0.3:
                    new += [b for b in L.applicable_actions(s) if L.is_self_loop(b)][:1]
                new.append(a); s = L.step(s, a)[0]
            plan = new
        elif r < 0.3:  # one random typed action replaced
            k = rng.randrange(len(plan)); plan[k] = rng.choice(L.ground_actions(objs, init, self_loops=True))
        elif r < 0.4:
            plan = plan[:rng.randrange(len(plan))]
        if len(plan) > 12:
            plan = plan[:12] if rng.random() < 0.5 else plan
        text, tags = render(plan, rng)
        out.append((iid, init, goal, text, tags))
    return out

if __name__ == "__main__":
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    t0 = time.time(); st = Counter(); bad = []
    for iid, init, goal, text, tags in cases(rng, n):
        v = L.val_run(L.planbench_path(iid), text, verbose=True)
        vl = val_listing(v["stdout"])
        ours = L.read_val_plan(text)
        ex = L.execute_text(init, goal, text)
        ok_verdict = ex["success"] == v["valid"]
        if vl == "other":
            ok_read = True; st["val_other"] += 1
        else:
            ok_read = (vl is None and ours is None) or (vl is not None and ours is not None and vl == sorted(ours))
        st["verdict_ok" if ok_verdict else "VERDICT_DIFF"] += 1
        st["read_ok" if ok_read else "READ_DIFF"] += 1
        st["valid"] += v["valid"]; st["bad"] += vl is None
        if not (ok_verdict and ok_read):
            bad.append((iid, tags, ascii(text[:300]), "VAL", v["valid"], v["returncode"], vl if vl == "other" or vl is None else len(vl), "OURS", ex["success"], None if ours is None else len(ours)))
    print(dict(st), round(time.time() - t0, 1), "s")
    for b in bad[:25]:
        print(b)
