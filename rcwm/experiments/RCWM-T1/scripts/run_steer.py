"""Mode-vector steering test (mechanism experiment S).

Finding motivating it: on ledger/register, 97-99% of the Supa->Titan-Sparse answer-slot shift at
L16-28 is a constant per-family offset (scripts/mechanism_report.py). If the transition skill is
largely that offset, adding it to Supa should recover much of Titan's one-step accuracy, and
subtracting it from Titan should remove it.

Offsets v[fam, L] = mean over TRAIN one-frame items of (h_sparse - h_supa) at the answer slot,
from the captured states (no GPU needed to build them).

Conditions (greedy, one-frame TEST items, exact state, lenient + strict parse):
  base:<arm>                       no intervention
  add:<L>:<alpha>                  Supa + alpha*v[fam, L] at block-L output, answer slot onward
  rand:<L>                         Supa + random vector with ||v[fam, L]|| (control)
  sub:<L>                          Titan-Sparse - v[fam, L] (necessity)
"Answer slot onward": during prefill only the final prompt position is steered; every decoding
step's new position is steered.

  python scripts/run_steer.py --bank bank_pilot --layers 16,20,24 --alphas 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, data, env, generation, runtime  # noqa: E402


def offsets(bank_stem):
    S, T = data.Arm(bank_stem, "supa"), data.Arm(bank_stem, "sparse")
    out = {}
    for fam in env.FAMILIES:
        ids = [it["item_id"] for it in S.items if it["split"] == "train" and it["family"] == fam and it["item_id"].endswith("-one")]
        for L in S.layers:
            out[(fam, L)] = (T.X_last(ids, L).astype(np.float64) - S.X_last(ids, L).astype(np.float64)).mean(0)
    return out


@contextmanager
def steer(model, layer: int, vec_by_row: torch.Tensor | None):
    """Add vec_by_row[b] (shape [B, D]) to block `layer`'s output; prefill: last position only."""
    if vec_by_row is None:
        yield
        return
    block = model.model.layers[layer - 1]          # hidden_states[L] is the output of block L-1

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            h[:, -1, :] += vec_by_row.to(h.dtype)
        else:
            h[:, -1, :] += vec_by_row[: h.shape[0]].to(h.dtype)
        return out
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def gen_with(model, tok, items, layer, vecs, batch=16):
    """Greedy generation per family with a per-row steering vector (or none)."""
    res = {}
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        for s in range(0, len(fi), batch):
            b = fi[s: s + batch]
            prompts = [generation.render(tok, it["messages"]) for it in b]
            enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
            Lm = max(len(x) for x in enc)
            ids = torch.full((len(b), Lm), generation.PAD, dtype=torch.long)
            att = torch.zeros((len(b), Lm), dtype=torch.long)
            for r, x in enumerate(enc):
                ids[r, Lm - len(x):] = torch.tensor(x); att[r, Lm - len(x):] = 1
            v = None if vecs is None else torch.tensor(np.stack([vecs[fam]] * len(b)), dtype=torch.float32, device="cuda")
            with steer(model, layer, v):
                g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=False,
                                   max_new_tokens=generation.max_tokens(fam), eos_token_id=generation.EOS,
                                   pad_token_id=generation.PAD, temperature=None, top_p=None, top_k=None, logits_processor=runtime.throttle_list())
            for r, it in enumerate(b):
                new = g[r, Lm:].tolist()
                cut = next((j for j, t in enumerate(new) if t in generation.EOS), len(new))
                res[it["item_id"]] = tok.decode(new[:cut], skip_special_tokens=True).strip()
            runtime.breathe()
    return res


def score(items, texts):
    out = {}
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        changed = [it for it in fi if it["gold"] != it["gold_prefix_states"][0]]
        def acc(xs, lenient):
            ok = 0
            for it in xs:
                st = env.parse_state(fam, texts[it["item_id"]], lenient=lenient)
                ok += st is not None and env.dumps(st) == it["gold"]
            return ok / max(len(xs), 1)
        out[fam] = {"n": len(fi), "strict": acc(fi, False), "lenient": acc(fi, True),
                    "changed_n": len(changed), "changed_lenient": acc(changed, True)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--layers", default="16,20,24")
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out_dir = ROOT / "runs" / a.bank / "steer"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = json.loads((ROOT / "data" / f"{a.bank}.json").read_text(encoding="utf-8"))
    items = [it for it in bankmod.items(bank) if it["split"] == a.split and it["format"] == "one"]
    V = offsets(a.bank)
    layers = [int(x) for x in a.layers.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    tok = runtime.tokenizer()
    results = {}
    rng = np.random.default_rng(17)

    def run_cond(model, name, layer, vecs):
        t0 = time.time()
        texts = gen_with(model, tok, items, layer, vecs)
        results[name] = score(items, texts)
        (out_dir / f"gen_{name.replace(':', '_')}.json").write_text(json.dumps(texts) + "\n", encoding="utf-8")
        print(f"{name:<16} " + " | ".join(f"{f[:6]} strict {v['strict']:.3f} len {v['lenient']:.3f} chg {v['changed_lenient']:.3f}"
                                           for f, v in results[name].items()) + f" ({time.time() - t0:.0f}s)", flush=True)

    with runtime.gpu_lease(label="rcwm-t1:steer", job="RCWM-T1 mode-vector steering"):
        model, _ = runtime.load_arm("supa")
        run_cond(model, "base:supa", layers[0], None)
        for L in layers:
            for al in alphas:
                run_cond(model, f"add:{L}:{al}", L, {f: al * V[(f, L)] for f in env.FAMILIES})
            rv = {}
            for f in env.FAMILIES:
                r = rng.standard_normal(V[(f, L)].shape)
                rv[f] = r / np.linalg.norm(r) * np.linalg.norm(V[(f, L)])
            run_cond(model, f"rand:{L}", L, rv)
        runtime.unload(model)
        model, _ = runtime.load_arm("sparse")
        run_cond(model, "base:sparse", layers[0], None)
        for L in layers:
            run_cond(model, f"sub:{L}", L, {f: -V[(f, L)] for f in env.FAMILIES})
        runtime.unload(model)
    (out_dir / "steer_results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
