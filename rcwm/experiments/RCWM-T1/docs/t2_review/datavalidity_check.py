"""T2 review, data-validity lens (CPU only, read-only on project data).

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/datavalidity_check.py

1. Labels: every train/eval row re-derived by an independent oracle written here, plus env.step and
   the FDS oracle; prompt re-rendered from (frame, before, event) and compared byte-for-byte.
2. Strata re-derived independently; sub-structure inside strata (conditional wrap / boundary, P
   applicable-changing vs applicable-no-op, inapplicable reason, L statuses, degenerate params).
3. Disjointness train/eval: payload, operative key (FDS generate.operative_key), (before, answer).
4. Overlap with pilot TEST chain transitions (oracle-anchored) and the whole pilot/confirm banks.
5. Tokenization vs encode_pair: prompt/target boundary merge, lengths, supervised-token shares.
6. Stored Titan-Sparse pilot chains: P per-step accuracy by frame and step type (oracle-anchored).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rcwmt import env, runtime  # noqa: E402

sys.path.insert(0, str(env.FDS))
import generate as fdsgen  # type: ignore  # noqa: E402

CODE = {"ledger_replay": "L", "register_machine": "R", "prerequisite_plan": "P"}
tr = [json.loads(l) for l in (ROOT / "data" / "t2_train.jsonl").read_text(encoding="utf-8").splitlines()]
ev = [json.loads(l) for l in (ROOT / "data" / "t2_eval.jsonl").read_text(encoding="utf-8").splitlines()]
ALL = [("train", r) for r in tr] + [("eval", r) for r in ev]


def J(o):
    return json.dumps(o, separators=(",", ":"), sort_keys=True)


# ------------------------------------------------------------ independent oracle (written fresh)
def my_step(fam, b, e):
    if fam == "ledger_replay":
        out = dict(b)
        if e["status"] != "posted":
            return out
        a = e["account"]
        if e["kind"] == "add":
            out[a] += e["amount"]
        elif e["kind"] == "scale":
            out[a] *= e["amount"]
        else:
            out[a] -= e["amount"]
            out[e["other"]] += e["amount"]
        return out
    if fam == "prerequisite_plan":
        ok = all((b >> i) & 1 for i in range(8) if (e["pre_on"] >> i) & 1) and \
            not any((b >> i) & 1 for i in range(8) if (e["pre_off"] >> i) & 1)
        if not ok:
            return b
        s = b
        for i in range(8):
            if (e["delete"] >> i) & 1:
                s &= ~(1 << i)
        for i in range(8):
            if (e["add"] >> i) & 1:
                s |= 1 << i
        return s & 0xFF
    out = dict(b)
    r, o, v = e["register"], e["other"], e["value"]
    if e["kind"] == "swap":
        out[r], out[o] = b[o], b[r]
    elif e["kind"] == "add":
        out[r] = (b[r] + v) % 97
    elif e["kind"] == "mix":
        out[r] = (b[r] + 2 * b[o] + v) % 97
    else:
        out[r] = (b[r] + v) % 97 if b[o] > e["threshold"] else (b[r] - v) % 97
    return out


def my_stratum(fam, b, e):
    if fam == "ledger_replay":
        return f"{e['kind']}/" + ("posted" if e["status"] == "posted" else "void")
    if fam == "prerequisite_plan":
        app = (b & e["pre_on"]) == e["pre_on"] and (b & e["pre_off"]) == 0
        return "applicable" if app else "inapplicable"
    k = e["kind"]
    if k == "swap":
        return "swap"
    if k == "conditional":
        return "conditional/" + ("gt" if b[e["other"]] > e["threshold"] else "le")
    raw = b[e["register"]] + e["value"] + (2 * b[e["other"]] if k == "mix" else 0)
    return f"{k}/" + ("nowrap" if 0 <= raw <= 96 else "wrap")


print("== 1. labels and prompt rendering")
tok = runtime.tokenizer()
bad = Counter()
for split, r in ALL:
    fam, b, e = r["family"], r["before"], r["event"]
    mine = J(my_step(fam, b, e))
    if mine != r["target"]:
        bad["my_oracle"] += 1
    if J(env.step(fam, b, e)) != r["target"]:
        bad["env.step"] += 1
    if env.fds_oracles()[fam]({"before": b, "event": e}) != r["target"]:
        bad["fds"] += 1
    if my_stratum(fam, b, e) != r["stratum"]:
        bad["stratum"] += 1
    want_frame = "m1" if fam == "prerequisite_plan" else "one"
    if r["frame"] != want_frame:
        bad["frame"] += 1
    msgs = env.multi_step_messages(fam, b, [e]) if r["frame"] == "m1" else env.one_step_messages(fam, b, e)
    if tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) != r["prompt"]:
        bad["prompt_render"] += 1
    user = r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    u = json.loads(user)
    if (u["before"], u.get("event", (u.get("events") or [None])[0])) != (b, e):
        bad["prompt_payload"] += 1
    if not r["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        bad["gen_prompt_suffix"] += 1
print(f"   rows {len(ALL)}; mismatches: {dict(bad) or 'none'}")

print("== 2. sub-structure inside strata")
for split, rows in (("train", tr), ("eval", ev)):
    sub = Counter()
    for r in rows:
        fam, b, e, s = r["family"], r["before"], r["event"], r["stratum"]
        if fam == "register_machine":
            if e["kind"] == "conditional":
                gt = b[e["other"]] > e["threshold"]
                raw = b[e["register"]] + (e["value"] if gt else -e["value"])
                sub[(s, "wrap" if not 0 <= raw <= 96 else "nowrap")] += 1
                if b[e["other"]] == e["threshold"]:
                    sub[(s, "other==threshold")] += 1
                if abs(b[e["other"]] - e["threshold"]) <= 2:
                    sub[(s, "|other-thr|<=2")] += 1
            if e["kind"] in ("add", "mix", "conditional") and e["value"] % 97 == 0 and e["kind"] != "mix":
                sub[(s, "value==0 mod 97")] += 1
            if e["value"] < 0:
                sub[(s, "value<0")] += 1
            if r["target"] == J(b):
                sub[(s, "copy-equal")] += 1
        elif fam == "ledger_replay":
            sub[(s, "status=" + e["status"])] += 1
            if e["status"] == "posted" and e["kind"] == "scale" and e["amount"] in (0, 1):
                sub[(s, f"scale amount={e['amount']}")] += 1
            if e["status"] == "posted" and e["kind"] in ("add", "transfer") and e["amount"] == 0:
                sub[(s, "amount=0")] += 1
        else:
            if s == "applicable":
                after = env.step(fam, b, e)
                sub[(s, "changed" if after != b else "no-op (copy)")] += 1
            else:
                on_miss = (b & e["pre_on"]) != e["pre_on"]
                off_hit = (b & e["pre_off"]) != 0
                sub[(s, "pre_on missing" if on_miss and not off_hit else "pre_off set" if off_hit and not on_miss else "both")] += 1
                # would the action change the state if (wrongly) applied?
                sub[(s, "applied-anyway changes state" if ((b & ~e["delete"]) | e["add"]) != b else "applied-anyway == copy")] += 1
    print(f"   {split}:")
    for k in sorted(sub):
        print(f"      {k[0]:<18} {k[1]:<30} {sub[k]}")
vals = Counter(r["event"]["value"] for r in tr if r["family"] == "register_machine")
print("   R value range train:", min(vals), max(vals), "distinct", len(vals))
thr = [r["event"]["threshold"] for r in tr if r["family"] == "register_machine"]
print("   R threshold range train:", min(thr), max(thr))
amts = Counter(r["event"]["amount"] for r in tr if r["family"] == "ledger_replay")
print("   L amounts train: range", min(amts), max(amts), "most common", amts.most_common(8))
ticks = [r["event"]["tick"] for r in tr if r["family"] != "prerequisite_plan"]
print("   L/R tick train range", min(ticks), max(ticks))

print("== 3. train/eval disjointness")


def pkey(r):
    return J({"before": r["before"], "event": r["event"]})


def okey(r):
    return (r["family"], fdsgen.operative_key(r["family"], {"before": r["before"], "event": r["event"]}))


def okey_strict(fam, b, e):
    """Answer-equivalence finer than operative_key: inapplicable P and void L reduce to before only."""
    if fam == "prerequisite_plan" and not env.applicable(b, e):
        return (fam, J(b), "noop")
    return (fam, fdsgen.operative_key(fam, {"before": b, "event": e}))


def iokey(r):
    return (r["family"], J(r["before"]), r["target"])


for name, f in (("payload", pkey), ("operative", okey), ("answer-equiv", lambda r: okey_strict(r["family"], r["before"], r["event"])), ("(before,answer)", iokey)):
    a, b = Counter(f(r) for r in tr), Counter(f(r) for r in ev)
    inter = set(a) & set(b)
    dup_tr = sum(n - 1 for n in a.values() if n > 1)
    dup_ev = sum(n - 1 for n in b.values() if n > 1)
    by = Counter(next(r["family"][:1].upper() + "|" + r["stratum"] for r in ev if f(r) == k) for k in inter)
    print(f"   {name:<16} eval rows sharing key with train: {sum(b[k] for k in inter):4d}/{len(ev)}  "
          f"within-train dups {dup_tr}  within-eval dups {dup_ev}  {dict(by)}")

print("== 4. overlap with pilot TEST chain transitions (oracle-anchored) and banks")
bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
conf = json.loads((ROOT / "data" / "bank_confirm.json").read_text(encoding="utf-8"))


def bank_steps(bk, split=None):
    out = []
    for p in bk["parents"]:
        if split and p["split"] != split:
            continue
        cands = p["candidates"] + ([p["twin"]] if "twin" in p else [])
        for c in cands:
            st = env.rollout(p["family"], p["s0"], c["events"])
            for t, e in enumerate(c["events"]):
                out.append((p["family"], st[t], e, t))
    return out


test_steps = bank_steps(bank, "test")
t2_keys = {"payload": {pkey(r) for r in tr}, "operative": {okey(r) for r in tr},
           "answer-equiv": {okey_strict(r["family"], r["before"], r["event"]) for r in tr}}
t2e_keys = {"payload": {pkey(r) for r in ev}, "operative": {okey(r) for r in ev},
            "answer-equiv": {okey_strict(r["family"], r["before"], r["event"]) for r in ev}}
for lab, keys in (("T2 train", t2_keys), ("T2 eval", t2e_keys)):
    hits = defaultdict(Counter)
    for fam, b, e, t in test_steps:
        ks = {"payload": J({"before": b, "event": e}),
              "operative": (fam, fdsgen.operative_key(fam, {"before": b, "event": e})),
              "answer-equiv": okey_strict(fam, b, e)}
        hits[CODE[fam]]["n"] += 1
        for kn, kv in ks.items():
            if kv in keys[kn]:
                hits[CODE[fam]][kn] += 1
                hits[CODE[fam]][kn + "@" + my_stratum(fam, b, e)] += 1
    for f in "LPR":
        h = hits[f]
        extra = {k: v for k, v in h.items() if "@" in k}
        print(f"   {lab} vs pilot-TEST {f}: steps {h['n']}  payload {h['payload']}  operative {h['operative']}  "
              f"answer-equiv {h['answer-equiv']}  {extra}")
# P action-semantics overlap (library actions shared regardless of state)
act = lambda e: (e["add"], e["delete"], e["pre_on"], e["pre_off"])  # noqa: E731
tr_act = {act(r["event"]) for r in tr if r["family"] == "prerequisite_plan"}
te_act = {act(e) for fam, b, e, t in test_steps if fam == "prerequisite_plan"}
print(f"   P distinct action semantics: T2 train {len(tr_act)}, pilot TEST {len(te_act)}, shared {len(tr_act & te_act)}")
all_pay = {J({"before": b, "event": e}) for bk in (bank, conf) for fam, b, e, t in bank_steps(bk)}
print(f"   payload hits vs all pilot+confirm transitions: train {sum(pkey(r) in all_pay for r in tr)}, eval {sum(pkey(r) in all_pay for r in ev)}")
print("   twin key present: pilot", sum("twin" in p for p in bank["parents"]), "confirm", sum("twin" in p for p in conf["parents"]))

print("== 5. tokenization vs vendored encode_pair")
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, str(ROOT / "vendor_bwm"))
import importlib.util  # noqa: E402
src = (ROOT / "vendor_bwm" / "bwmn" / "lm.py").read_text(encoding="utf-8")
core_path = runtime.PATHS["sparse"]
print("   runtime tokenizer dir == trainer core dir:", str(runtime.TOKENIZER_DIR) == str(core_path),
      "| lm.py core path string present:", "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze" in src)
tt = AutoTokenizer.from_pretrained(str(core_path), local_files_only=True)
print("   eos/pad", tt.eos_token_id, tt.pad_token_id, "| eos token", tt.convert_ids_to_tokens([tt.eos_token_id]))
merge, lens, sup = 0, [], defaultdict(list)
special_in_target = 0
for split, r in ALL:
    p = tt(r["prompt"], add_special_tokens=False).input_ids
    t = tt(r["target"], add_special_tokens=False).input_ids
    joint = tt(r["prompt"] + r["target"], add_special_tokens=False).input_ids
    if joint != p + t:
        merge += 1
    if any(x >= 151643 for x in t):
        special_in_target += 1
    lens.append(len(p) + len(t) + 1)
    if split == "train":
        sup[(CODE[r["family"]], r["stratum"])].append(len(t) + 1)
print(f"   boundary merges (tok(p+t) != tok(p)+tok(t)): {merge}; special ids in target: {special_in_target}; "
      f"max len {max(lens)} (train_generic --max-len 512; train_sft default 320)")
tot = sum(sum(v) for v in sup.values())
fam_tok = Counter()
fam_rows = Counter()
for (f, s), v in sup.items():
    fam_tok[f] += sum(v)
    fam_rows[f] += len(v)
print("   train supervised tokens (target+EOS) by family: " + ", ".join(
    f"{f}: rows {fam_rows[f]/len(tr):.3f} tokens {fam_tok[f]/tot:.3f} (mean {fam_tok[f]/fam_rows[f]:.1f}/row)" for f in "LPR"))
for (f, s), v in sorted(sup.items()):
    print(f"      {f} {s:<18} rows {len(v):4d}  sup-token share {sum(v)/tot:.3f}")

print("== 6. stored Titan-Sparse pilot chains: P per-step accuracy by frame (oracle-anchored)")
cand = {c["cand_id"]: (p["family"], p["s0"], c["events"]) for p in bank["parents"] if p["split"] == "test" for c in p["candidates"]}
for mode in ("chain_one", "chain_m1"):
    acc = defaultdict(lambda: [0, 0])
    for l in (ROOT / "runs" / "bank_pilot" / "sparse" / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(l)
        fam, s0, evs = cand[g["cand_id"]]
        if fam != "prerequisite_plan":
            continue
        gold = env.rollout(fam, s0, evs)
        pred = [None if s is None else json.loads(s) for s in g["states"]]
        for t, e in enumerate(evs):
            if t + 1 >= len(pred) or pred[t] != gold[t]:
                continue
            app = env.applicable(gold[t], e)
            typ = ("app-changed" if gold[t + 1] != gold[t] else "app-noop") if app else "inapp"
            acc[typ][0] += 1
            acc[typ][1] += int(pred[t + 1] == gold[t + 1])
            if not app:
                wrong_applied = ((gold[t] & ~e["delete"]) | e["add"])
                acc["inapp: answered applied-anyway"][0] += 1
                acc["inapp: answered applied-anyway"][1] += int(pred[t + 1] == wrong_applied and wrong_applied != gold[t])
    print(f"   {mode}: " + ", ".join(f"{k} {c/n:.3f} (n={n})" for k, (n, c) in sorted(acc.items())))
