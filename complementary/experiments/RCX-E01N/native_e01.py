"""RCX-E01N native cells: LoRA students on frozen FP32 Titan-Sparse (CPU), rule-conditioned ledger data.

Arms (matched: 1,501 rows, 190 steps x 16, D-03 recipe, seed s; only the data regime differs):
  N-FX    fixed rule S+{posted}, statuses posted/void 75/25            (T2-like data from the RCX generator)
  N-VR    varied in-context rules, iid, canonical wording, real words  (bundle "ordinary" cell)
  N-CFSD  varied rules with counterfactual twins, nonce words, paraphrased wording (bundle "counterfactual/SD" cell)
Frame: Titan's verbatim ledger one-step frame with the status clause replaced by the supplied rule sentence
(native_probe.rcl_system); user turn = compact sorted JSON {"before","event"} with tick; target = compact sorted JSON
balances (Titan's trained answer format). Arith world (add/transfer).

Subcommands (from complementary/):
  python -B experiments/RCX-E01N/native_e01.py build   --out runs/RCX-E01N/cells
  python -B experiments/RCX-E01N/native_e01.py audit   --out runs/RCX-E01N/cells   (writes AUDIT.json; train requires it)
  python -B experiments/RCX-E01N/native_e01.py profile --out runs/RCX-E01N/cells --arm N-VR --steps 3
  python -B experiments/RCX-E01N/native_e01.py train   --out runs/RCX-E01N/cells --arm N-VR --seed 0
  python -B experiments/RCX-E01N/native_e01.py eval    --out runs/RCX-E01N/cells --arm N-VR --seed 0
CPU only; refuses model loads below 12 GB available RAM (D-39); rcx.guard between micro-batches.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(int(os.environ.get("RCX_THREADS", "4")))

import native_probe as NP  # noqa: E402
from rcx.guard import ThermalGuard  # noqa: E402
from rcx.native import lora_cpu as LC  # noqa: E402
from rcx.rcl import gen, score as S, templates as T, vocab as V, world as W  # noqa: E402

ARMS = {"N-FX": "FX", "N-VR": "VR", "N-CFSD": "CF_SD"}
N_ROWS, STEPS, BATCH = 1501, 190, 16
DATA_SEED = {"FX": 20261251, "VR": 20261252, "CF_SD": 20261253}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target_json(after):
    return json.dumps({k: after[k] for k in sorted(after)}, separators=(",", ":"))


def native_rows(regime, tok):
    gen.use_world("arith")
    recs = gen.make_train(regime, N_ROWS, DATA_SEED[regime])
    rows = []
    for r in recs:
        system = NP.rcl_system(T.render_rule(r["rule_ast"], r["tid"]))
        prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                          {"role": "user", "content": NP.user_json(r)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt, "target": target_json(r["gold"]["after"]), "regime": regime,
                     "parent_id": r["parent_id"], "variant": r["variant"], "decision": r["gold"]["decision"],
                     "semantic_hash": r["semantic_hash"], "skeleton_hash": r["skeleton_hash"],
                     "status": r["event"]["status"]})
    return rows


def cmd_build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LC.PATHS["sparse"]), local_files_only=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = {}
    for arm, regime in ARMS.items():
        rows = native_rows(regime, tok)
        p = out / f"train_{arm}.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        dec = sum(r["decision"] == "noop" for r in rows)
        ntok = [len(tok(r["prompt"] + r["target"], add_special_tokens=False).input_ids) for r in rows]
        man[arm] = {"path": p.name, "sha256": sha(p), "rows": len(rows), "noop": dec,
                    "parents": len({r["parent_id"] for r in rows}), "max_tokens": max(ntok),
                    "mean_tokens": sum(ntok) / len(ntok)}
        print(arm, man[arm], flush=True)
    (out / "TRAIN_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")


def load_rows(out, arm):
    return [json.loads(l) for l in open(Path(out) / f"train_{arm}.jsonl", encoding="utf-8")]


def cmd_profile(a):
    NP.need_ram()
    model, tok = LC.load_core("sparse")
    pm = LC.attach_lora(model, seed=0)
    rows = load_rows(a.out, a.arm)
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    t0 = time.time()
    log = LC.train_sft(pm, tok, rows, steps=a.steps, batch_size=BATCH, seed=0, log_every=1, guard=guard)
    dt = time.time() - t0
    rec = {"arm": a.arm, "steps": a.steps, "wall_s": dt, "s_per_step": dt / a.steps,
           "est_190_steps_h": dt / a.steps * STEPS / 3600, "train": log}
    print(json.dumps(rec, indent=1), flush=True)
    (Path(a.out) / f"PROFILE_{a.arm}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")


def cmd_train(a):
    out = Path(a.out) / f"{a.arm}_s{a.seed}"
    if (out / "adapter" / "adapter_model.safetensors").exists():
        print("adapter exists; refusing to retrain under the same id", flush=True)
        return
    # Provenance is fixed BEFORE the hours-long run: hashing the rows/code after training records whatever is on disk
    # at the end (the arith-pilot provenance slip in DECISIONS.md), not what was trained.
    rows_sha = sha(Path(a.out) / f"train_{a.arm}.jsonl")
    rows = load_rows(a.out, a.arm)
    code_sha = {"lora_cpu.py": sha(Path(LC.__file__)), "native_e01.py": sha(Path(__file__))}
    audit_sha = require_audit(a.out, a.arm, rows_sha)
    NP.need_ram()
    model, tok = LC.load_core("sparse")
    base_sha = sha(LC.PATHS["sparse"] / "model.safetensors")
    pm = LC.attach_lora(model, seed=a.seed)
    order = LC.row_order_for(len(rows), STEPS, BATCH, a.seed)
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    logf = open(Path(a.out) / f"train_{a.arm}_s{a.seed}.log", "a", encoding="utf-8", newline="\n")

    def P(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log = LC.train_sft(pm, tok, rows, steps=STEPS, batch_size=BATCH, seed=a.seed, row_order=order, logger=P,
                       guard=guard, ckpt_dir=str(out / "ckpt"), ckpt_every=10)
    meta = {"arm": a.arm, "regime": ARMS[a.arm], "seed": a.seed, "core": "titan-sparse", "base_sha256": base_sha,
            "rows_sha256": rows_sha, "audit_sha256": audit_sha, "recipe": "D-03 (r16/a32/do0.05; 190x16; "
            "lr 1e-4 WSD; AdamW 0.9/0.95) on CPU FP32, unmerged", "train": log, "code_sha256": code_sha}
    LC.save_adapter(pm, out / "adapter", meta)
    meta["adapter_sha256"] = sha(out / "adapter" / "adapter_model.safetensors")
    (out / "TRAIN_META.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    ck = out / "ckpt" / "state.pt"
    if ck.exists():
        ck.unlink()
    P(f"done {log['wall_s']}s adapter sha {meta['adapter_sha256'][:12]}")


def eval_battery(n_parents, surfaces):
    gen.use_world("arith")
    return [r for r in gen.make_battery("dev", n_parents, NP.BATTERY_SEED) if r["surface"] in surfaces]


def cmd_eval(a):
    out = Path(a.out)
    battery = eval_battery(a.n_parents, a.surfaces.split(","))
    tag = f"{a.arm}_s{a.seed}" if a.arm.startswith("N-") else a.arm
    res_path = out / f"eval_{tag}.json"
    if res_path.exists():
        print("eval exists", flush=True)
        return
    NP.need_ram()
    model, tok = LC.load_core("supa" if a.arm == "SUPA" else "sparse")
    if a.arm.startswith("N-"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(out / tag / "adapter"), is_trainable=False).eval()
    elif a.arm == "T2":
        model = NP.load_t2(model)

    def render(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts = [render(NP.rcl_system(T.render_rule(r["rule_ast"], r["tid"])), NP.user_json(r)) for r in battery]
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    t0 = time.time()
    outs = LC.generate(model, tok, prompts, guard=guard, log=lambda m: print(m, flush=True))
    res = [S.classify(r, NP.parse_answer(o, list(r["gold"]["after"].keys()))) for r, o in zip(battery, outs)]
    summ = S.aggregate(battery, res)
    # same bytes as native_probe's battery.jsonl, so battery_sha256 is comparable across the two scripts
    bsha = hashlib.sha256("".join(json.dumps(r) + "\n" for r in battery).encode()).hexdigest()
    rec = {"arm": a.arm, "seed": a.seed, "battery_seed": NP.BATTERY_SEED, "battery_sha256": bsha,
           "n": len(battery), "surfaces": a.surfaces, "wall_s": time.time() - t0,
           "composite": {s: S.composite(summ, s) for s in a.surfaces.split(",")},
           "composite_decision": {s: S.composite(summ, s, field="decision") for s in a.surfaces.split(",")},
           "cells": summ}
    with gzip.open(out / f"preds_{tag}.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (o, x) in enumerate(zip(outs, res)):
            f.write(json.dumps({"i": i, "out": o, **x}) + "\n")
    res_path.write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    print(f"[{tag}] composite {rec['composite']} decision {rec['composite_decision']} {rec['wall_s']:.0f}s",
          flush=True)


# ----------------------------------------------------------------------------- data-hygiene audit (AUDIT.json)

RCWM_T1 = ROOT.parent / "rcwm" / "experiments" / "RCWM-T1"  # read-only reference tree (never written)
FRAME_PRE, FRAME_POST = NP.rcl_system("\x00").split("\x00")
VERBATIM_CLAUSE = "If status is not posted, leave all balances unchanged."
STD_BAL, STD_AMT_ABS = (0, 98), 12  # gen._state / gen._event "std" ranges; M1_bal / M2_amt lie outside them
MATERIAL_SHARE_DIFF = 0.05  # |N-FX - T2| posted share above this is flagged (descriptive only)
HARD_FLAGS = ("frame", "gold", "readback_hash", "split_words", "probe_words", "excluded_words", "non_train_words",
              "rcwm_heldout_text", "heldout_template", "heldout_rule_factor", "heldout_magnitude")
_PROMPT_RE = re.compile(r"<\|im_start\|>system\n(.*)<\|im_end\|>\n<\|im_start\|>user\n(.*)<\|im_end\|>\n"
                        r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n", re.S)
_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def render_literal(system, user):
    """Qwen3 chat template (add_generation_prompt=True, enable_thinking=False) written out. The audit requires every
    stored prompt (rendered by the real tokenizer at build time) to equal it byte for byte."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def read_back(row):
    """Decode a native training row from its prompt TEXT alone (independent of the recorded metadata)."""
    m = _PROMPT_RE.fullmatch(row["prompt"])
    if not m:
        raise ValueError("not a system+user Qwen3 chat prompt")
    system, user = m.group(1), m.group(2)
    if not (system.startswith(FRAME_PRE) and system.endswith(FRAME_POST)):
        raise ValueError("system prompt is not the native ledger frame")
    sentence = system[len(FRAME_PRE):len(system) - len(FRAME_POST)]
    rule, tid = W.parse_rule(sentence)
    u = json.loads(user)
    if set(u) != {"before", "event"} or u["event"].get("tick") != 1:
        raise ValueError("user turn is not {before, event(tick=1)}")
    state, event = u["before"], {k: v for k, v in u["event"].items() if k != "tick"}
    ok, after = W.transition(rule, state, event)
    exact = (user == json.dumps(u, separators=(",", ":"), sort_keys=True)
             and row["prompt"] == render_literal(NP.rcl_system(sentence), user))
    return {"rule": rule, "tid": tid, "state": state, "event": event, "ok": ok, "after": after, "frame_exact": exact,
            "semantic_hash": W.semantic_hash(rule, state, event), "skeleton_hash": W.skeleton_hash(state, event)}


def rule_type(rule):
    if rule["op"] in ("AND", "OR"):
        return f"{rule['op']}({rule['args'][0]['op']},{rule['args'][1]['op']})"
    if rule["op"] in W.STATUS_OPS:
        return rule["op"] + ("{posted}" if rule["set"] == [gen.POSTED] else f"/size{len(rule['set'])}")
    return rule["op"]


def heldout_template_fragments(min_len=12):
    """Fixed text of the held-out (h*) rule templates that no TRAIN template contains."""
    pools = (T.SPOS, T.SNEG, T.AND_FRAMES, T.OR_FRAMES)
    train = [t for p in pools for k, t in p.items() if not k.startswith("h")]
    frags = {f for p in pools for k, t in p.items() if k.startswith("h") for f in re.split(r"\{[A-Z0-9]+\}", t)}
    return sorted(f for f in frags if len(f.strip()) >= min_len and not any(f in x for x in train))


def _ast_constants(path, names):
    """Module-level literal constants of a Python source file, read with ast (no import; read-only)."""
    import ast
    out = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            v = node.value
            od = isinstance(v, ast.Call) and getattr(v.func, "id", "") == "OrderedDict"
            val = ast.literal_eval(v.args[0] if od else v)
            out[node.targets[0].id] = dict(val) if od else val
    return out


def rcwm_heldout_texts(rcwm_root=RCWM_T1):
    """Strings only RCWM's T2b held-out surfaces produce (paraphrase 4 and field-name variant 4 of the prereq and
    register families; docs/t2b/{prereq,register}/DESIGN.md), built from RCWM's source constants the way RCWM's own
    audits build them (t2bp_common.heldout_fragments; t2br_make_data.surface_audit)."""
    s = Path(rcwm_root) / "scripts"
    bp = _ast_constants(s / "t2bp_common.py", {"PARAPHRASES", "STATE_SENTENCES", "FIELD_VARIANTS",
                                               "HELDOUT_SURFACE_IDX"})
    br = _ast_constants(s / "t2br_common.py", {"SD_PARAPHRASES", "SD_FIELDS", "HELDOUT_PARAPHRASE",
                                               "HELDOUT_NAME_IDX"})
    h = bp["HELDOUT_SURFACE_IDX"]
    para, state_s = bp["PARAPHRASES"][h], bp["STATE_SENTENCES"][h]
    texts = {"prereq paraphrase 4": para[:para.index("{")], "prereq state sentence 4": state_s[:state_s.index("{")],
             "prereq state sentence 4 tail": " bits wide"}
    for k, v in bp["FIELD_VARIANTS"][h].items():
        if v not in {x[k] for x in bp["FIELD_VARIANTS"][:h]}:
            texts[f"prereq field name 4 ({k})"] = json.dumps(v) + ":"
    for i, f in enumerate(re.split(r"\{[a-z_]+\}", br["SD_PARAPHRASES"][br["HELDOUT_PARAPHRASE"]])):
        if len(f.strip()) >= 30:
            texts[f"register paraphrase 4 fragment {i}"] = f
    for k, v in br["SD_FIELDS"].items():
        texts[f"register field name 4 ({k})"] = json.dumps(v[br["HELDOUT_NAME_IDX"]]) + ":"
    return texts


def t2_ledger_reference(rcwm_root=RCWM_T1):
    """Descriptive profile of T2's ledger training rows (RCWM data/t2_train.jsonl; read-only)."""
    p = Path(rcwm_root) / "data" / "t2_train.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    led = [r for r in rows if r["family"] == "ledger_replay"]
    st = Counter(r["event"]["status"] for r in led)
    bal = [v for r in led for v in r["before"].values()]
    ticks = [r["event"].get("tick") for r in led]
    return {"rows": len(led), "of_total_rows": len(rows), "frames": dict(Counter(r["frame"] for r in led)),
            "status_counts": dict(st), "posted_share": st["posted"] / len(led),
            "noop_target_share": sum(json.loads(r["target"]) == r["before"] for r in led) / len(led),
            "n_accounts": dict(Counter(len(r["before"]) for r in led)),
            "kinds": dict(Counter(r["event"]["kind"] for r in led)), "tick_range": [min(ticks), max(ticks)],
            "balance_range": [min(bal), max(bal)]}


def token_covariates(rows, tok, steps=STEPS, batch=BATCH, seeds=(0, 1, 2)):
    """Supervised (target + EOS) and processed (prompt + target + EOS, no padding) tokens exactly as
    lora_cpu.train_sft counts them (supervised_tokens / tokens_processed) under each seed's row order."""
    enc = [LC.encode_pair(tok, r["prompt"], r["target"], 10 ** 9) for r in rows]
    sup, proc = [e.n_target for e in enc], [len(e.input_ids) for e in enc]
    per_seed = {}
    for s in seeds:
        order = LC.row_order_for(len(rows), steps, batch, s)[:steps * batch]
        per_seed[str(s)] = {"supervised_tokens": sum(sup[i] for i in order),
                            "tokens_processed": sum(proc[i] for i in order)}
    return {"mean_supervised_per_row": sum(sup) / len(sup), "mean_processed_per_row": sum(proc) / len(proc),
            "max_processed_per_row": max(proc), "per_seed": per_seed}


def _balance(counter_by_key):
    return {k: {"n": sum(c.values()), "noop": c["noop"], "noop_share": round(c["noop"] / sum(c.values()), 4)}
            for k, c in sorted(counter_by_key.items())}


def audit_native(out_dir, tok=None, n_parents=40, surfaces=("in", "ho_tpl"), seeds=(0, 1, 2), rcwm_root=RCWM_T1,
                 write=True):
    """Data-hygiene / matching audit of the built native training rows (train_<arm>.jsonl) against the native eval
    battery gen.make_battery('dev', n_parents, BATTERY_SEED) restricted to `surfaces`. Every training row is decoded
    from its prompt TEXT (read_back) as well as from its recorded metadata.

    HARD checks (any failure -> ok=False; cmd_train refuses): manifest sha/rows; 1501 rows per arm; Titan's verbatim
    frame with only the status clause replaced, byte-exact; gold re-derived from the prompt; zero semantic_hash and
    zero skeleton_hash overlap with the eval battery (and with the full dev battery, all surfaces), recorded and
    read-back hashes; no DEV/SEALED pool word, no diagnostic probe word (near-miss/posted-like), no PREREG-3 word
    (vocab.EXCLUDED) anywhere in a training prompt; rule/status words only from the TRAIN pool or the posted/void
    anchors; no held-out template wording, held-out composition combo, size-3 set or M1/M2 magnitude in training;
    no RCWM T2b held-out paraphrase-4 / field-name-4 text; every (stratum, surface) cell of the battery has n_parents
    parents, each with an apply and a no-op variant; PREREG-3 words occur in the battery only in F4_nearmiss.
    DESCRIPTIVE: class balance per arm / rule type / template; account counts; token covariates (needs tok);
    N-FX vs T2 ledger rows; design notes. Writes out_dir/AUDIT.json (write=True) and returns the dict."""
    import inspect
    out = Path(out_dir)
    fail, notes = [], []
    man_p = out / "TRAIN_MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    if not man:
        fail.append("manifest_missing")
    # ---- reference frame (Titan's verbatim one-step ledger prompt) and RCWM held-out texts
    env_p = Path(rcwm_root) / "rcwmt" / "env.py"
    rcwm_texts, titan_accounts = None, None
    if env_p.exists() and (Path(rcwm_root) / "scripts").exists():
        envc = _ast_constants(env_p, {"SYSTEM_ONE", "ACCOUNTS"})
        titan_accounts = list(envc["ACCOUNTS"])
        if not envc["SYSTEM_ONE"]["ledger_replay"] == NP.VERBATIM_SYSTEM == NP.rcl_system(VERBATIM_CLAUSE):
            fail.append("frame_not_titan_verbatim")
        rcwm_texts = rcwm_heldout_texts(rcwm_root)
    else:
        fail.append("rcwm_reference_missing")
    # ---- battery
    gen.use_world("arith")
    full = gen.make_battery("dev", n_parents, NP.BATTERY_SEED)
    battery = [r for r in full if r["surface"] in surfaces]

    def hashes(recs):
        sem = {r["semantic_hash"] for r in recs} | {W.semantic_hash(r["rule_ast"], r["state"], r["event"])
                                                     for r in recs}
        skel = {r["skeleton_hash"] for r in recs} | {W.skeleton_hash(r["state"], r["event"]) for r in recs}
        return sem, skel

    ev_sem, ev_skel = hashes(battery)
    all_sem, all_skel = hashes(full)
    cells = defaultdict(lambda: defaultdict(set))
    for r in battery:
        cells[(r["group"], r["surface"])][r["parent_id"]].add(r["gold"]["decision"])
    strata = {}
    for s in gen.STRATA:
        for sf in surfaces:
            c = cells.get((s, sf), {})
            both = sum(d == {"apply", "noop"} for d in c.values())
            strata[f"{s}|{sf}"] = {"parents": len(c), "parents_with_apply_and_noop": both}
            if len(c) != n_parents or both != len(c):
                fail.append(f"battery_strata:{s}|{sf}")
    train_pool = set(V.REAL_TRAIN) | set(V.NONCE["train"])
    anchors = {gen.POSTED, gen.VOID}
    excl_bat, trainw_bat = Counter(), Counter()
    for r in battery:
        sa = W.status_atom(r["rule_ast"])
        words = {r["event"]["status"]} | set(sa["set"] if sa else ())
        excl_bat[r["group"]] += bool(words & V.EXCLUDED)
        trainw_bat[r["group"]] += bool(words & train_pool)
    excl_bat, trainw_bat = +excl_bat, +trainw_bat
    if set(excl_bat) - {"F4_nearmiss"}:
        fail.append("battery_excluded_outside_F4")
    if trainw_bat:
        fail.append("battery_train_words")
    bat_info = {"split": "dev", "seed": NP.BATTERY_SEED, "n_parents": n_parents, "surfaces": list(surfaces),
                "n": len(battery), "n_full_all_surfaces": len(full),
                "battery_sha256": hashlib.sha256("".join(json.dumps(r) + "\n" for r in battery).encode()).hexdigest(),
                "strata": strata, "n_accounts": {f"{sf}:{k}": v for (sf, k), v in
                                                 sorted(Counter((r["surface"], len(r["state"])) for r in battery).items())},
                "tids": dict(Counter(r["tid"] for r in battery)), "noop_share": round(
                    sum(r["gold"]["decision"] == "noop" for r in battery) / len(battery), 4),
                "prereg3_words_by_stratum": dict(excl_bat), "train_pool_words_by_stratum": dict(trainw_bat)}
    # ---- training arms
    dev_sealed = set(V.REAL_DEV) | set(V.NONCE["dev"]) | set(V.REAL_SEALED) | set(V.NONCE["sealed"])
    probes = set(V.NEARMISS_VOID) | set(V.POSTEDLIKE)
    excl_tok = {w for w in V.EXCLUDED if " " not in w}
    excl_phrase = [w for w in V.EXCLUDED if " " in w]
    tpl_frags = heldout_template_fragments()
    arms, rows_sha = {}, {}
    for arm, regime in ARMS.items():
        p = out / f"train_{arm}.jsonl"
        if not p.exists():
            fail.append(f"rows_missing:{arm}")
            continue
        rows_sha[arm] = sha(p)
        rows = load_rows(out, arm)
        if len(rows) != N_ROWS:
            fail.append(f"rows_count:{arm}")
        if man.get(arm, {}).get("sha256") != rows_sha[arm] or man.get(arm, {}).get("rows") != len(rows):
            fail.append(f"manifest_sha:{arm}")
        flags_n = Counter()
        sem, skel = set(), set()
        by_rt, by_tid, by_acc = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        statuses, decisions = Counter(), Counter()
        for r in rows:
            flags = set()
            text = r["prompt"]
            toks = set(_WORD_RE.findall(text))
            sem.add(r.get("semantic_hash"))
            skel.add(r.get("skeleton_hash"))
            if rcwm_texts and any(t in text for t in rcwm_texts.values()):
                flags.add("rcwm_heldout_text")
            if any(f in text for f in tpl_frags):
                flags.add("heldout_template")
            if toks & excl_tok or any(ph in text for ph in excl_phrase):
                flags.add("excluded_words")
            if toks & dev_sealed:
                flags.add("split_words")
            if toks & probes:
                flags.add("probe_words")
            try:
                rb = read_back(r)
            except (ValueError, KeyError, TypeError, AttributeError):
                flags_n.update(flags | {"frame"})
                continue
            sem.add(rb["semantic_hash"])
            skel.add(rb["skeleton_hash"])
            dec = "apply" if rb["ok"] else "noop"
            if not rb["frame_exact"]:
                flags.add("frame")
            if r["target"] != target_json(rb["after"]) or r.get("decision") != dec \
                    or r.get("status") != rb["event"]["status"]:
                flags.add("gold")
            if r.get("semantic_hash") != rb["semantic_hash"]:
                flags.add("readback_hash")
            sa = W.status_atom(rb["rule"])
            words = {rb["event"]["status"]} | set(sa["set"] if sa else ())
            if words & dev_sealed:
                flags.add("split_words")
            if words - train_pool - anchors:
                flags.add("non_train_words")
            if rb["tid"].split(":")[-1].startswith("h"):
                flags.add("heldout_template")
            if (rb["rule"]["op"] in ("AND", "OR")
                    and (rb["rule"]["op"], rb["rule"]["args"][1]["op"]) in gen.HELDOUT_COMBOS) \
                    or (sa and len(sa["set"]) >= 3):
                flags.add("heldout_rule_factor")  # V6_compose combos / V7_bigset
            if any(not STD_BAL[0] <= v <= STD_BAL[1] for v in rb["state"].values()) \
                    or abs(rb["event"]["amount"]) > STD_AMT_ABS:
                flags.add("heldout_magnitude")  # M1_bal / M2_amt
            flags_n.update(flags)
            by_rt[rule_type(rb["rule"])][dec] += 1
            by_tid[rb["tid"]][dec] += 1
            by_acc[len(rb["state"])][dec] += 1
            statuses[rb["event"]["status"]] += 1
            decisions[dec] += 1
        sem.discard(None)
        skel.discard(None)
        ov = {"semantic_eval_battery": len(sem & ev_sem), "skeleton_eval_battery": len(skel & ev_skel),
              "semantic_full_dev_battery": len(sem & all_sem), "skeleton_full_dev_battery": len(skel & all_skel)}
        if ov["semantic_eval_battery"] or ov["semantic_full_dev_battery"]:
            fail.append(f"semantic_overlap:{arm}")
        if ov["skeleton_eval_battery"] or ov["skeleton_full_dev_battery"]:
            fail.append(f"skeleton_overlap:{arm}")
        fail.extend(f"{f}:{arm}" for f in HARD_FLAGS if flags_n[f])
        n = len(rows)
        arms[arm] = {"regime": regime, "rows": n, "rows_sha256": rows_sha[arm],
                     "parents": len({r.get("parent_id") for r in rows}),
                     "variants": dict(Counter(r.get("variant") for r in rows)),
                     "hard_check_rows": {f: flags_n[f] for f in HARD_FLAGS}, "overlap": ov,
                     "noop_share": round(decisions["noop"] / n, 4), "decisions": dict(decisions),
                     "class_balance_by_rule_type": _balance(by_rt), "class_balance_by_template": _balance(by_tid),
                     "class_balance_by_n_accounts": _balance(by_acc),
                     "anchor_status_counts": {w: statuses[w] for w in sorted(anchors)},
                     "distinct_statuses": len(statuses)}
        if tok is not None:
            arms[arm]["tokens"] = token_covariates(rows, tok, seeds=seeds)
    # ---- matching (only the data regime may differ)
    lr = inspect.signature(LC.train_sft).parameters["lr"].default
    matched = {"rows": {a: v["rows"] for a, v in arms.items()}, "steps": STEPS, "batch": BATCH, "lr_peak": lr,
               "row_order": f"lora_cpu.row_order_for({N_ROWS}, {STEPS}, {BATCH}, seed): identical index sequence "
                            "for every arm at a given seed",
               "lora_init": "attach_lora(seed=seed) for every arm", "frame_pre": FRAME_PRE, "frame_post": FRAME_POST}
    if tok is not None and "N-FX" in arms:
        ref = arms["N-FX"]["tokens"]["per_seed"]
        matched["token_covariates_vs_N-FX"] = {
            a: {s: {k: {"diff": v["tokens"]["per_seed"][s][k] - ref[s][k],
                        "ratio": round(v["tokens"]["per_seed"][s][k] / ref[s][k], 4)} for k in ref[s]}
                for s in ref} for a, v in arms.items()}
    # ---- N-FX vs T2 ledger rows (descriptive)
    t2 = t2_ledger_reference(rcwm_root)
    fx_vs_t2 = None
    if t2 and "N-FX" in arms:
        fx = arms["N-FX"]
        fx_posted = fx["anchor_status_counts"]["posted"] / fx["rows"]
        diff = fx_posted - t2["posted_share"]
        fx_vs_t2 = {"n_fx_posted_share": round(fx_posted, 4), "t2_posted_share": round(t2["posted_share"], 4),
                    "diff": round(diff, 4), "material": abs(diff) > MATERIAL_SHARE_DIFF,
                    "threshold": MATERIAL_SHARE_DIFF, "n_fx_noop_target_share": fx["noop_share"],
                    "t2_noop_target_share": round(t2["noop_target_share"], 4),
                    "n_fx_ledger_rows": fx["rows"], "t2_ledger_rows": t2["rows"]}
    # ---- design notes (descriptive)
    eval_acc = sorted({len(r["state"]) for r in battery})
    for arm, v in arms.items():
        off = {k: c["n"] for k, c in v["class_balance_by_n_accounts"].items() if k not in eval_acc}
        if off:
            k = sum(off.values())
            notes.append(f"{arm}: {k}/{v['rows']} rows ({k / v['rows']:.1%}) have {sorted(off)}-account states; the "
                         f"eval battery ({','.join(surfaces)}) has only {eval_acc}-account states and Titan's frame "
                         f"uses accounts {titan_accounts}. Shorter targets also lower this arm's supervised tokens.")
    if t2:
        notes.append(f"T2 ledger rows: ticks {t2['tick_range']}, balances {t2['balance_range']}, kinds "
                     f"{sorted(t2['kinds'])}; native rows: tick 1 always, balances {list(STD_BAL)}, kinds "
                     f"{sorted(gen.WORLDS['arith'])}.")
    notes.append("user_json sorts keys, so the SD regime's state/event key-order factors are erased in the native "
                 "frame; N-CFSD's surface diversity is template wording (t0-t3) and account count only.")
    res = {"ok": not fail, "failures": fail, "out_dir": str(out), "battery": bat_info, "arms": arms,
           "matched": matched, "t2_ledger_reference": t2, "n_fx_vs_t2": fx_vs_t2, "design_notes": notes,
           "rows_sha256": rows_sha, "rcwm_heldout_texts": rcwm_texts, "heldout_template_fragments": tpl_frags,
           "tokenizer": None if tok is None else type(tok).__name__,
           "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(NP.__file__), Path(gen.__file__),
                                                    Path(W.__file__), Path(V.__file__), Path(T.__file__))}}
    if write:
        (out / "AUDIT.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    return res


def require_audit(out_dir, arm, rows_sha):
    """Refuse to train on rows that have not passed audit_native (returns the AUDIT.json sha)."""
    p = Path(out_dir) / "AUDIT.json"
    aud = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if not aud.get("ok") or aud.get("rows_sha256", {}).get(arm) != rows_sha:
        raise SystemExit(f"refusing to train {arm}: AUDIT.json is missing, failed, or audited other rows; run "
                         f"`native_e01.py audit --out {out_dir}` first")
    return sha(p)


def cmd_audit(a):
    tok = None
    if not a.no_tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(LC.PATHS["sparse"]), local_files_only=True)
    res = audit_native(a.out, tok=tok, n_parents=a.n_parents, surfaces=tuple(a.surfaces.split(",")))
    print(json.dumps({k: res[k] for k in ("ok", "failures", "n_fx_vs_t2", "design_notes")}, indent=1), flush=True)
    for arm, v in res["arms"].items():
        print(arm, "noop", v["noop_share"], "overlap", v["overlap"], "tokens", v.get("tokens", {}).get("per_seed"),
              flush=True)
    if not res["ok"]:
        raise SystemExit(f"AUDIT FAILED: {res['failures']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "audit", "profile", "train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="N-VR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-parents", dest="n_parents", type=int, default=40)
    ap.add_argument("--surfaces", default="in,ho_tpl")
    ap.add_argument("--no-tok", dest="no_tok", action="store_true", help="audit: skip token covariates")
    a = ap.parse_args()
    {"build": cmd_build, "audit": cmd_audit, "profile": cmd_profile, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
