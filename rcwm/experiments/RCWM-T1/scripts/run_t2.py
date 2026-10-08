"""RCWM-T2 (exploratory): rule-coverage LoRA on frozen Titan-Sparse.

  python scripts/run_t2.py train --tag t2_s0 --steps 190 --seed 0
  python scripts/run_t2.py eval  --tag t2_s0

train  vendored BWM-001 trainer (vendor_bwm/, D-15) under this session's GPU lease
eval   one GPU session, every stage adapter OFF (= Titan-Sparse, same process) and ON (D-19):
       (1) per-operation one-step accuracy on four sets: data/t2_eval.jsonl, its cross-frame
           re-render data/t2_eval_xframe.jsonl, the supplement data/t2_eval_supp.jsonl, and the
           oracle-anchored transitions of the pilot DEV+TEST chains (gold input state, chain frame);
       (2) chained rollouts (chain_one, chain_m1) on the pilot TEST plans;
       (3) off-task first-token log-probs on the 248 neutral prompts, plus 64-token greedy replies.
       Per-row texts are dumped for scripts/t2_analysis.py. Each stage writes its own file and is
       skipped on rerun if that file exists, so a crash loses at most one stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import env, generation, offtask, runtime  # noqa: E402
from make_t2_data import stratum as t2_stratum  # noqa: E402

CODE = {"ledger_replay": "L", "register_machine": "R", "prerequisite_plan": "P"}


PAIRING = ROOT / "runs" / "t2" / "PAIRING.json"     # D-22 switch: present + enabled -> train mode pairs train+eval


def _pairing():
    if not PAIRING.exists():
        return None
    cfg = json.loads(PAIRING.read_text(encoding="utf-8"))
    return cfg if cfg.get("enabled") else None


def _train_body(a):
    """The training itself, run inside a lease held by the caller."""
    sys.path.insert(0, str(ROOT / "vendor_bwm"))
    sys.path.insert(0, str(ROOT / "vendor_bwm" / "scripts"))
    import train_generic  # noqa: E402
    out = ROOT / "runs" / "t2" / a.tag
    sys.argv = ["train_generic.py", "--core", "titan-sparse", "--rows", str(ROOT / "data" / "t2_train.jsonl"),
                "--steps", str(a.steps), "--seed", str(a.seed), "--lr", str(a.lr), "--out", str(out)]
    train_generic.main()


def _free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(a):
    out = ROOT / "runs" / "t2" / a.tag
    if (out / "adapter").exists():
        raise SystemExit(f"{out}/adapter exists")
    cfg = _pairing()
    if cfg:
        return pair(a, cfg)
    with runtime.gpu_lease(label=f"rcwm-t1:t2train:{a.tag}", job=f"RCWM-T2 LoRA train {a.tag}"):
        _train_body(a)


def pair(a, cfg):
    """D-22 (agreed with the peer 2026-10-07): train, then evaluate, one seed in ONE lease turn.
    Cap: cfg['cap_h'] (4.5 h) from acquire. The eval starts only if cfg['eval_estimate_h'] fits the remaining
    budget, and stops before any stage that would start within cfg['stage_margin_h'] of the cap. Unfinished
    stages are resumed by the separate `run_t2.py eval` step that master_chain.sh runs next. A crash propagates
    (the lease is released by its context manager)."""
    out = ROOT / "runs" / "t2" / a.tag
    cap_s = 3600 * float(cfg.get("cap_h", 4.5))
    est_s = 3600 * float(cfg.get("eval_estimate_h", 2.5))
    margin_s = 3600 * float(cfg.get("stage_margin_h", 0.5))
    with runtime.gpu_lease(label=f"rcwm-t1:t2pair:{a.tag}", job=f"RCWM-T2 train+eval pair {a.tag} (D-22)"):
        t0 = time.time()
        _train_body(a)
        print(f"[t2] pair {a.tag}: train done in {time.time() - t0:.0f}s", flush=True)
        _free_gpu()
        if not (out / "adapter").exists():
            raise SystemExit(f"[t2] pair {a.tag}: no adapter after training")
        if (out / "eval.json").exists():
            return
        if time.time() - t0 + est_s > cap_s:
            print(f"[t2] pair {a.tag}: eval deferred to the next turn (budget: {time.time() - t0:.0f}s used of {cap_s:.0f}s)",
                  flush=True)
            return
        try:
            _eval_body(a, deadline=t0 + cap_s - margin_s)
        except _Deferred as e:
            print(f"[t2] pair {a.tag}: cap reached before stage '{e}'; eval resumes on the next turn", flush=True)
        print(f"[t2] pair {a.tag}: turn used {time.time() - t0:.0f}s of {cap_s:.0f}s", flush=True)


@torch.no_grad()
def gen_prompts(model, tok, prompts, max_new=160, budget=8000):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    enc = {i: tok(prompts[i], add_special_tokens=False)["input_ids"] for i in order}
    out, batch = [None] * len(prompts), []
    def flush(b):
        L = max(len(enc[i]) for i in b)
        ids = torch.full((len(b), L), generation.PAD, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, L - len(enc[i]):] = torch.tensor(enc[i]); att[r, L - len(enc[i]):] = 1
        g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=False, max_new_tokens=max_new,
                           eos_token_id=generation.EOS, pad_token_id=generation.PAD, temperature=None, top_p=None, top_k=None,
                           logits_processor=runtime.throttle_list())
        for r, i in enumerate(b):
            new = g[r, L:].tolist()
            cut = next((j for j, t in enumerate(new) if t in generation.EOS), len(new))
            out[i] = tok.decode(new[:cut], skip_special_tokens=True).strip()
        runtime.breathe()
    for i in order:
        if batch and (len(batch) + 1) * (len(enc[i]) + max_new) > budget:
            flush(batch); batch = []
        batch.append(i)
    if batch:
        flush(batch)
    return out


def per_op(rows, texts):
    res = {}
    for r, t in zip(rows, texts):
        st = env.parse_state(r["family"], t, lenient=True)
        k = f"{r['family'][:6]}|{r['stratum']}"
        n, c = res.get(k, (0, 0))
        res[k] = (n + 1, c + int(st is not None and env.dumps(st) == r["target"]))
    return {k: {"n": n, "acc": c / n} for k, (n, c) in sorted(res.items())}


def _load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pilot_items(bank, tok, splits=("dev", "test")):
    """Oracle-anchored chain transitions: gold input state, one event, in the chain's primary frame
    (one-step for L/R, single-event m1 for P). Unique per (split, payload)."""
    seen, items = set(), []
    for p in bank["parents"]:
        if p["split"] not in splits:
            continue
        fam = p["family"]
        m1 = fam == "prerequisite_plan"
        for c in p["candidates"]:
            st = env.rollout(fam, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                key = (p["split"], env.dumps({"before": st[t], "event": ev}))
                if key in seen:
                    continue
                seen.add(key)
                msgs = env.multi_step_messages(fam, st[t], [ev]) if m1 else env.one_step_messages(fam, st[t], ev)
                items.append({"prompt": generation.render(tok, msgs), "target": env.dumps(env.step(fam, st[t], ev)),
                              "family": fam, "stratum": t2_stratum(CODE[fam], st[t], ev), "frame": "m1" if m1 else "one",
                              "before": st[t], "event": ev, "split": p["split"], "step": t, "cand_id": c["cand_id"]})
    return items


def _dump(path, rows, base, t2):
    with path.open("w", encoding="utf-8") as f:
        for i, (r, b, t) in enumerate(zip(rows, base, t2)):
            f.write(json.dumps({"i": i, **{k: v for k, v in r.items() if k != "prompt"}, "base_text": b, "t2_text": t}) + "\n")


def _write_chains(path, chains, rr):
    with path.open("w", encoding="utf-8") as f:
        for c, r in zip(chains, rr):
            f.write(json.dumps({"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]],
                                "texts": r["texts"]}) + "\n")


def eval_sets(tok, bank):
    return {"t2_eval": _load(ROOT / "data" / "t2_eval.jsonl"), "xframe": _load(ROOT / "data" / "t2_eval_xframe.jsonl"),
            "supp": _load(ROOT / "data" / "t2_eval_supp.jsonl"), "pilot": pilot_items(bank, tok)}


def evaluate(a):
    out = ROOT / "runs" / "t2" / a.tag
    assert (out / "adapter").exists(), "train first"
    if (out / "eval.json").exists():
        raise SystemExit(f"{out}/eval.json exists")
    with runtime.gpu_lease(label=f"rcwm-t1:t2eval:{a.tag}", job=f"RCWM-T2 eval {a.tag}"):
        _eval_body(a)


class _Deferred(Exception):
    """Raised between eval stages when a paired turn reaches its cap (D-22)."""


def _eval_body(a, deadline=None):
    """The eval itself, run inside a lease held by the caller. With a deadline (paired turn), it stops
    BEFORE starting any stage once time.time() > deadline; finished stages stay on disk and the next
    `run_t2.py eval` reuses them."""
    from peft import PeftModel
    out = ROOT / "runs" / "t2" / a.tag
    ad = out / "adapter"

    def check(stage):
        if deadline is not None and time.time() > deadline:
            raise _Deferred(stage)
    bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
    chains = [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"]}
              for p in bank["parents"] if p["split"] == "test" for c in p["candidates"]]
    tok = runtime.tokenizer()
    sets = eval_sets(tok, bank)
    res = {"tag": a.tag, "n": {k: len(v) for k, v in sets.items()}}
    print(f"[t2] eval sets {res['n']}", flush=True)
    base, prov = runtime.load_arm("sparse")
    model = PeftModel.from_pretrained(base, str(ad), adapter_name="t2", is_trainable=False).eval()
    try:
        for name, rows in sets.items():                       # (1) per-op, OFF and ON
            f = out / f"gen_{name}.jsonl"
            if f.exists():
                d = _load(f)
                assert len(d) == len(rows), (f, len(d), len(rows))
                b, t = [r["base_text"] for r in d], [r["t2_text"] for r in d]
                print(f"[t2] {name}: reusing {f.name}", flush=True)
            else:
                check(name)
                t0 = time.time()
                pr = [r["prompt"] for r in rows]
                with model.disable_adapter():
                    b = gen_prompts(model, tok, pr)
                t = gen_prompts(model, tok, pr)
                _dump(f, rows, b, t)
                print(f"[t2] {name}: {len(rows)} rows x2 in {time.time() - t0:.0f}s", flush=True)
            res[f"per_op_{name}_base"], res[f"per_op_{name}_t2"] = per_op(rows, b), per_op(rows, t)
            for k in res[f"per_op_{name}_t2"]:
                print(f"   {name:<8} {k:<30} base {res[f'per_op_{name}_base'][k]['acc']:.3f} -> "
                      f"t2 {res[f'per_op_{name}_t2'][k]['acc']:.3f}  (n {res[f'per_op_{name}_t2'][k]['n']})", flush=True)
        res["per_op_base"], res["per_op_t2"] = res["per_op_t2_eval_base"], res["per_op_t2_eval_t2"]
        for mode in ("chain_one", "chain_m1"):                # (2) chains, OFF and ON
            for suffix, off in (("_base", True), ("", False)):
                f = out / f"gen_{mode}{suffix}.jsonl"
                if f.exists():
                    continue
                check(f"{mode}{suffix}")
                t1 = time.time()
                if off:
                    with model.disable_adapter():
                        rr = generation.chain(model, tok, chains, mode)
                else:
                    rr = generation.chain(model, tok, chains, mode)
                _write_chains(f, chains, rr)
                print(f"[t2] {mode}{suffix} {len(chains)} chains {time.time() - t1:.0f}s", flush=True)
        (out / "off").mkdir(exist_ok=True)                    # (3) off-task, OFF and ON
        if not (out / "off" / "offtask_meta.json").exists():
            check("offtask_off")
            with model.disable_adapter():
                offtask.capture_offtask(model, tok, out / "off")
        if not (out / "offtask_meta.json").exists():
            check("offtask_on")
            offtask.capture_offtask(model, tok, out)
        if not (out / "offtask_gen.jsonl").exists():
            check("offtask_gen")
            ot = offtask.load_prompts()
            op = [generation.render(tok, r["messages"]) for r in ot]
            with model.disable_adapter():
                ob = gen_prompts(model, tok, op, max_new=64)
            oo = gen_prompts(model, tok, op, max_new=64)
            with (out / "offtask_gen.jsonl").open("w", encoding="utf-8") as fh:
                for r, b, t in zip(ot, ob, oo):
                    fh.write(json.dumps({"id": r["id"], "reply_class": r["reply_class"], "base_text": b, "t2_text": t}) + "\n")
        print("[t2] off-task captured", flush=True)
    finally:
        runtime.unload(model)
    res["provenance"] = {"base_fingerprint": prov["param_fingerprint"], "adapter": str(ad),
                         "data_sha256": {k: hashlib.sha256((ROOT / "data" / f"{k}.jsonl").read_bytes()).hexdigest()
                                         for k in ("t2_train", "t2_eval", "t2_eval_xframe", "t2_eval_supp")}}
    (out / "eval.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("[t2] eval written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--tag", default="t2_s0")
    ap.add_argument("--steps", type=int, default=190)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    {"train": train, "eval": evaluate}[a.mode](a)


if __name__ == "__main__":
    main()
