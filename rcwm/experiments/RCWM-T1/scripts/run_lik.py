"""Likelihood ranking of candidate consequences: does the frozen model's OWN conditional
distribution prefer the true k-step consequence over heuristic and near-miss foils?

For every item (multi format k in {1,2,4}, and the trained one-step frame) of --splits, score
log p(answer + <|im_end|> | prompt) by teacher forcing, for:
  gold                 the true final state
  copy_s0              the start state
  first_only, last_only  one true transition only (k >= 2)
  ignore_pre (P)       preconditions ignored        ignore_status (L)  void events applied
  near_bit (P)         gold with one bit flipped (2 draws)
  near_val (L, R)      gold with one entry +-1 (2 draws)
Duplicates of gold are dropped. Answers are the canonical compact JSON the oracle emits (Titan's
trained output format). No fitting: ranking is parameter-free.

  python scripts/run_lik.py --bank bank_pilot --arm sparse --splits dev,test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, capture, env, fidelity, runtime  # noqa: E402

PAD = 151643
IM_END = 151645


def foils(fam, s0, events, gold_state, rng):
    out = {}
    h = fidelity.heuristic_states(fam, s0, events)
    out["copy_s0"] = s0
    if len(events) >= 2:
        out["first_only"] = h["first_only"]
        out["last_only"] = h["last_only"]
    if "ignore_pre" in h:
        out["ignore_pre"] = h["ignore_pre"]
    if "ignore_status" in h:
        out["ignore_status"] = h["ignore_status"]
    for d in range(2):
        if fam == "prerequisite_plan":
            out[f"near_bit{d}"] = gold_state ^ (1 << rng.randrange(env.NBITS))
        else:
            st = dict(gold_state)
            key = rng.choice(sorted(st))
            delta = rng.choice((-1, 1))
            st[key] = (st[key] + delta) % env.MOD if fam == "register_machine" else st[key] + delta
            out[f"near_val{d}"] = st
    gold_txt = env.dumps(gold_state)
    return {k: env.dumps(v) for k, v in out.items() if env.dumps(v) != gold_txt}


@torch.no_grad()
def score(model, tok, seqs, max_tokens=1500):
    """seqs: list of (prompt_ids, answer_ids). Returns summed log-prob of answer_ids (incl. EOS)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = np.zeros(len(seqs), dtype=np.float64)
    lm_head = model.get_output_embeddings()
    b = []
    def flush(b):
        L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in b)
        ids = torch.full((len(b), L), PAD, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            x = seqs[i][0] + seqs[i][1]
            ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        hid = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        # vectorised: one lm_head call over every answer position in the batch, then a segment sum
        rows_, cols_, tgt_, seg_ = [], [], [], []
        for r, i in enumerate(b):
            p, a = len(seqs[i][0]), len(seqs[i][1])
            rows_ += [r] * a; cols_ += list(range(p - 1, p - 1 + a)); tgt_ += list(seqs[i][1]); seg_ += [r] * a
        h = hid[torch.tensor(rows_, device=hid.device), torch.tensor(cols_, device=hid.device)]
        lp = torch.log_softmax(lm_head(h).float(), -1)
        tok_lp = lp[torch.arange(len(tgt_), device=lp.device), torch.tensor(tgt_, device=lp.device)]
        sums = torch.zeros(len(b), dtype=torch.float64, device=lp.device).index_add_(0, torch.tensor(seg_, device=lp.device), tok_lp.double())
        for r, i in enumerate(b):
            out[i] = float(sums[r])
        runtime.breathe()
    cur_max = 0
    for i in order:
        n = len(seqs[i][0]) + len(seqs[i][1])
        if b and max(cur_max, n) * (len(b) + 1) > max_tokens:
            flush(b); b, cur_max = [], 0
        b.append(i); cur_max = max(cur_max, n)
    if b:
        flush(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", default="dev,test")
    a = ap.parse_args()
    out_dir = ROOT / "runs" / a.bank / a.arm
    out_path = out_dir / "lik_rank.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists")
    bank = json.loads((ROOT / "data" / f"{a.bank}.json").read_text(encoding="utf-8"))
    items = [it for it in bankmod.items(bank) if it["split"] in set(a.splits.split(","))]
    tok = runtime.tokenizer()
    rng = random.Random(4242)
    seqs, meta = [], []
    for it in items:
        fam = it["family"]
        states = [json.loads(s) for s in it["gold_prefix_states"]]
        s0, gold = states[0], states[-1]
        evs = json.loads(it["messages"][1]["content"]).get("events") or [json.loads(it["messages"][1]["content"])["event"]]
        cands = {"gold": env.dumps(gold), **foils(fam, s0, evs, gold, rng)}
        p_ids = tok(capture.render(tok, it["messages"]), add_special_tokens=False)["input_ids"]
        for name, txt in cands.items():
            a_ids = tok(txt, add_special_tokens=False)["input_ids"] + [IM_END]
            seqs.append((p_ids, a_ids))
            meta.append((it["item_id"], name, txt))
    print(f"[{a.arm}] {len(items)} items, {len(seqs)} scored sequences", flush=True)
    with runtime.gpu_lease(label=f"rcwm-t1:lik:{a.arm}", job=f"RCWM-T1 likelihood ranking {a.arm}"):
        model, prov = runtime.load_arm(a.arm)
        t0 = time.time()
        lp = score(model, tok, seqs)
        runtime.unload(model)
    by = {}
    for (iid, name, txt), v in zip(meta, lp):
        by.setdefault(iid, {})[name] = v
    with out_path.open("w", encoding="utf-8") as f:
        for iid, d in by.items():
            f.write(json.dumps({"item_id": iid, "logp": d}) + "\n")
    print(f"[{a.arm}] likelihood ranking done in {time.time() - t0:.0f}s fingerprint {prov['param_fingerprint'][:12]}", flush=True)


if __name__ == "__main__":
    main()
