"""Capture hidden states (and optionally generative baselines) for one encoder arm.

  python scripts/run_arm.py --bank data/bank_pilot.json --arm sparse [--gen] [--parity]

One model per process; the shared-GPU lease is held for the whole job and released at exit.
Outputs are create-only under runs/<bank-stem>/<arm>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, capture, env, generation, offtask, runtime  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gen", action="store_true", help="generative baselines on the test split")
    ap.add_argument("--gen-splits", default="test")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--max-batch-tokens", type=int, default=2000)
    ap.add_argument("--out-stem", default=None, help="write under runs/<out-stem>/<arm> instead of runs/<bank-stem>/<arm>")
    ap.add_argument("--gen-only", action="store_true", help="skip capture (and parity, off-task); generation only")
    ap.add_argument("--no-offtask", action="store_true")
    a = ap.parse_args()

    bank_path = (ROOT / a.bank) if not a.bank.is_absolute() else a.bank
    out = ROOT / "runs" / (a.out_stem or bank_path.stem) / a.arm
    if (out / "DONE.json").exists():
        raise SystemExit(f"{out} already complete")
    out.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    items = bankmod.items(bank)
    tok = runtime.tokenizer()
    t0 = time.time()
    encoded = [capture.encode_item(tok, it) for it in items]
    ev_items = [i for i, it in enumerate(items) if it["format"] == "multi" and it["k"] == bank["horizon"]]
    ev_row = {i: r for r, i in enumerate(ev_items)}
    print(f"[{a.arm}] items {len(items)}, event rows {len(ev_items)}, encode {time.time() - t0:.1f}s, "
          f"tokens {sum(len(e['ids']) for e in encoded):,}", flush=True)
    index_path = out / "index.json"
    if index_path.exists() and json.loads(index_path.read_text(encoding="utf-8")).get("bank_sha256") != sha(bank_path):
        raise SystemExit(f"{index_path} was written for a different bank; refusing to desync captures (verification B5)")
    index_path.write_text(json.dumps({
        "bank": str(bank_path), "bank_sha256": sha(bank_path), "layers": list(capture.LAYERS),
        "items": [{"item_id": it["item_id"], "parent_id": it["parent_id"], "cand_id": it["cand_id"],
                   "family": it["family"], "split": it["split"], "k": it["k"], "format": it["format"],
                   "n_tokens": len(e["ids"])} for it, e in zip(items, encoded)],
        "event_rows": ev_items}) + "\n", encoding="utf-8")

    D, nL = 1024, len(capture.LAYERS)
    with runtime.gpu_lease(label=f"rcwm-t1:{a.arm}", job=f"RCWM-T1 capture {bank_path.stem} {a.arm}"):
        t1 = time.time()
        model, prov = runtime.load_arm(a.arm)
        prov["load_seconds"] = round(time.time() - t1, 1)
        prov["runtime"] = runtime.runtime_info()
        print(f"[{a.arm}] loaded in {prov['load_seconds']}s fingerprint {prov['param_fingerprint'][:12]}", flush=True)
        if a.parity and not a.gen_only and not (out / "parity.json").exists():
            t1 = time.time()
            par = capture.parity_check(model, [encoded[i] for i in ev_items], n=24)
            par["seconds"] = round(time.time() - t1, 1)
            (out / "parity.json").write_text(json.dumps(par, indent=2) + "\n", encoding="utf-8")
            print(f"[{a.arm}] parity {json.dumps(par['max_abs'])} rel {json.dumps(par['max_rel_to_rowmax'])}", flush=True)
        cap_marker = out / "CAPTURED.json"
        if a.gen_only:
            cap_s = None
            print(f"[{a.arm}] --gen-only: no capture", flush=True)
        elif cap_marker.exists() and json.loads(cap_marker.read_text(encoding="utf-8")).get("param_fingerprint", "")[:12] == prov["param_fingerprint"][:12]:
            cap_s = json.loads(cap_marker.read_text(encoding="utf-8"))["capture_seconds"]
            print(f"[{a.arm}] capture already complete for this fingerprint; skipping", flush=True)
        else:
            last = np.lib.format.open_memmap(out / "last.npy", mode="w+", dtype=np.float16, shape=(len(items), nL, D))
            evs = np.lib.format.open_memmap(out / "events.npy", mode="w+", dtype=np.float16,
                                            shape=(len(ev_items), bank["horizon"], nL, D))
            t1 = time.time()
            capture.capture(model, encoded, max_batch_tokens=a.max_batch_tokens, last_out=last,
                            events_out=evs, ev_row=ev_row)
            last.flush(); evs.flush()
            del last, evs
            cap_s = round(time.time() - t1, 1)
            cap_marker.write_text(json.dumps({"capture_seconds": cap_s, "param_fingerprint": prov["param_fingerprint"],
                                              "finished_unix": time.time()}) + "\n", encoding="utf-8")
            print(f"[{a.arm}] captured in {cap_s}s", flush=True)

        gen_s = None
        if a.gen:
            t1 = time.time()
            splits = set(a.gen_splits.split(","))
            gi = [i for i, it in enumerate(items) if it["split"] in splits]
            texts = generation.e2e(model, tok, [items[i] for i in gi])
            with (out / "gen_e2e.jsonl").open("x", encoding="utf-8") as f:
                for i, t in zip(gi, texts):
                    it = items[i]
                    pred = env.parse_state(it["family"], t, lenient=True)
                    f.write(json.dumps({"item_id": it["item_id"], "text": t,
                                        "parsed": pred is not None,
                                        "exact": pred is not None and env.dumps(pred) == it["gold"]}) + "\n")
            print(f"[{a.arm}] e2e generations {len(gi)} in {time.time() - t1:.0f}s", flush=True)
            chains = [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"]}
                      for p in bank["parents"] if p["split"] in splits
                      for c in p["candidates"] + ([p["twin"]] if "twin" in p else [])]
            for mode in ("chain_one", "chain_m1"):
                t2 = time.time()
                res = generation.chain(model, tok, chains, mode)
                with (out / f"gen_{mode}.jsonl").open("x", encoding="utf-8") as f:
                    for c, r in zip(chains, res):
                        f.write(json.dumps({"cand_id": c["cand_id"],
                                            "states": [None if s is None else env.dumps(s) for s in r["states"]],
                                            "texts": r["texts"]}) + "\n")
                print(f"[{a.arm}] {mode} {len(chains)} chains in {time.time() - t2:.0f}s", flush=True)
            gen_s = round(time.time() - t1, 1)
        if not (a.gen_only or a.no_offtask) and offtask.PROMPTS.exists() and not (out / "offtask_meta.json").exists():
            t1 = time.time()
            offtask.capture_offtask(model, tok, out)
            print(f"[{a.arm}] off-task capture {time.time() - t1:.0f}s", flush=True)
        runtime.unload(model)

    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    (out / "DONE.json").write_text(json.dumps({"capture_seconds": cap_s, "generation_seconds": gen_s,
                                               "finished_unix": time.time()}) + "\n", encoding="utf-8")
    print(f"[{a.arm}] done", flush=True)


if __name__ == "__main__":
    main()
