"""Per-item activation patching between Supa and Titan-Sparse on the same prompt (CPU, exploratory).

Question (after RESULTS_PILOT R9): the constant answer-slot offset carries format/mode, not the
transition computation. WHERE does the computation live? Here the residual stream of one model is
written into the other on the SAME prompt (same tokenizer, same left-padded batch, so positions
align), and both models then decode greedily on their own.

Notation: hidden_states index L = output of block L-1 (model.model.layers[L-1]); the patch replaces
that block's output tensor.

Conditions (each a full greedy generation of the items, scored exactly like run_steer.py):
  base:supa, base:sparse  no intervention (rerun here for paired comparison)
  T2S_last@L     Titan's block-L output at the LAST prompt position written into Supa's, prefill only
  T2S_all@L      the same at ALL real prompt positions (pads are left alone; they are masked as keys)
  S2T_all@L      reverse: Supa's prompt states written into Titan at all prompt positions
  T2S_all_mis@L  control: Titan's states for a DIFFERENT item of the same family (different parent,
                 different gold, different before-state, same token length if any exists, else the
                 nearest length, ties to the longer donor), right-aligned (see pick_mismatch).
                 Also scored against the DONOR's gold (donor_match).

Semantics.
- Prefill only: decode steps are never patched. The question is whether Titan's PROMPT ENCODING
  (its residual stream up to block L at the prompt positions) is enough for Supa's remaining blocks
  L..27 plus Supa's own decoding to produce the answer (T2S), or necessary for Titan (S2T).
- KV caches: blocks 0..L-1 (those producing hidden_states 1..L) computed their keys/values at the
  prompt positions from the RECEIVER's own residual stream, before the patch; those stay in the cache
  and every decode step attends to them. Only blocks L..27 see the patched prompt states. So decode
  steps still have the receiver's own lower-layer view of the prompt; T2S_all_mis therefore mixes
  the receiver item (lower KV) with the donor item (upper KV), which donor_match measures.
- At L = 28 only the last position would matter for the first token (no later block reads the
  other positions), which is why layers stop at 24.
- Blocks 0..7 are bit-identical across the two arms (R5; verified here per block by hash and logged
  in _meta.identical_blocks). For L <= 8 every patch is an exact no-op: T2S_*@8 == base:supa and
  S2T_all@8 == base:sparse by construction (the logged patch max|diff| is 0.0). L = 8 is still a
  useful cross-model mechanics check, but carries no information about the delta.

Implementation.
- One model in RAM at a time (host RAM is the binding constraint; ~2.4 GB per FP32 model). Donor
  states are captured per generation batch with generate(max_new_tokens=1) (the exact prefill code
  path of the receiver's generate, so identity patches are bit-exact) and held in RAM (~80 MB per
  layer per model for 144 items). Phases: [sparse] sanity (a) + capture -> [supa] sanity (b),
  base:supa, T2S_* (+ capture of Supa states) -> [sparse] base:sparse, S2T_all.
- Items: exactly runs/bank_pilot/steer_cpu/steer_cpu_results.json _meta.items, in that order (the
  selection code of run_steer_cpu.py is rerun and must reproduce it). --per-family N < 48 takes the
  first N of each family from that list. Batching, generation kwargs, decoding and scoring are those
  of run_steer_cpu.gen_with / run_steer.score / run_steer_cpu.correct.
- SANITY, asserted before any condition (results._sanity): (a) identity Titan->Titan all@L and
  (b) Supa->Supa all@L reproduce the unpatched texts exactly on the sanity items (first
  --sanity-per-family of each family, generated as their own batches) with patch max|diff| == 0;
  (c) in EVERY patched generate the hook patches exactly once, on the first (prefill) call with
  seq_len == prompt length, and every later call is a seq_len-1 decode step left untouched; the
  number of calls equals the number of generated tokens; (d) every batch has a real token at index
  -1 in every row, and 'last' masks exactly index -1.
- CPU only: CUDA hidden before torch is imported (empty string, never "-1"; DECISIONS I-21),
  is_available/device_count patched, below-normal priority, --threads torch threads (default 3).
  rcwmt.runtime/generation are imported (as in run_steer_cpu.py) but no CUDA-touching function of
  theirs is called.

  python scripts/run_patch_cpu.py --per-family 48 --layers 8,16,24 --out-name patch_cpu
  python scripts/run_patch_cpu.py --per-family 4 --layers 16 --out-name patch_cpu_smoke
"""
from __future__ import annotations

import os
import sys


def _argv_threads(default: int = 3) -> int:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return default


_THREADS = _argv_threads()
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # zero visible devices; NOT "-1" (DECISIONS I-21)
os.environ["RCWM_THREADS"] = str(_THREADS)       # run_steer_cpu reads this at import
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = str(_THREADS)

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

torch.cuda.is_available = lambda: False          # never query the driver
torch.cuda.device_count = lambda: 0

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import bank as bankmod, env, generation, runtime  # noqa: E402
from run_steer import score  # noqa: E402
from run_steer_cpu import correct, load_cpu  # noqa: E402

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
COND_KINDS = ("T2S_last", "T2S_all", "S2T_all", "T2S_all_mis")
DONOR = {"T2S_last": "sparse", "T2S_all": "sparse", "S2T_all": "supa", "T2S_all_mis": "sparse"}
RECEIVER = {"T2S_last": "supa", "T2S_all": "supa", "S2T_all": "sparse", "T2S_all_mis": "supa"}
MIN_FREE_GIB = 2.8                                # host RAM needed before loading one FP32 model


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------- items
def select_items(a, pool):
    """run_steer_cpu.py's selection (seed 17, 48/family); must equal the steering preview's list."""
    rng = np.random.default_rng(a.seed)
    sel = []
    for fam in env.FAMILIES:
        fi = [it for it in pool if it["family"] == fam]
        sel += [fi[i] for i in sorted(rng.choice(len(fi), size=min(48, len(fi)), replace=False))]
    ids = [it["item_id"] for it in sel]
    ref = Path(a.items_from)
    if ref.exists():
        ref_ids = json.loads(ref.read_text(encoding="utf-8"))["_meta"]["items"]
        assert ref_ids == ids, "rng selection does not reproduce the steering preview's item list"
        log(f"items: reproduced {ref.relative_to(ROOT)} _meta.items ({len(ids)} ids, same order)")
    else:
        log(f"WARNING: {ref} missing; using the rng selection without cross-check")
    byid = {it["item_id"]: it for it in pool}
    out = []
    for fam in env.FAMILIES:
        out += [byid[i] for i in ids if byid[i]["family"] == fam][: a.per_family]
    return out


def batches_of(items, size):
    """run_steer_cpu.gen_with's batching: per family in env.FAMILIES order, consecutive chunks."""
    out = []
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        out += [fi[s: s + size] for s in range(0, len(fi), size)]
    return out


def encode(tok, b):
    """Left-padded batch exactly as run_steer_cpu.gen_with builds it."""
    enc = [tok(generation.render(tok, it["messages"]), add_special_tokens=False)["input_ids"] for it in b]
    Lm = max(len(x) for x in enc)
    ids = torch.full((len(b), Lm), generation.PAD, dtype=torch.long)
    att = torch.zeros((len(b), Lm), dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, Lm - len(x):] = torch.tensor(x); att[r, Lm - len(x):] = 1
    assert bool(att[:, -1].all()), "sanity (d): a row has no real token at index -1"
    return ids, att, [len(x) for x in enc]


def pick_mismatch(items, pool, tok, seed):
    """Donor for each receiver item: same family, different parent, different gold and different
    before-state; token length equal if any candidate has it, else the nearest, ties broken toward
    the LONGER donor (then every receiver prompt position is patched; the donor's first tokens are
    dropped by right-alignment); final choice uniform with rng(seed) among the tied candidates."""
    rng = np.random.default_rng(seed)
    n_tok = {it["item_id"]: len(tok(generation.render(tok, it["messages"]), add_special_tokens=False)["input_ids"])
             for it in pool}
    out, stats = {}, []
    for it in items:
        cands = sorted((j for j in pool if j["family"] == it["family"] and j["parent_id"] != it["parent_id"]
                        and j["gold"] != it["gold"] and j["gold_prefix_states"][0] != it["gold_prefix_states"][0]),
                       key=lambda j: j["item_id"])
        li = n_tok[it["item_id"]]
        best = min((abs(n_tok[j["item_id"]] - li), n_tok[j["item_id"]] < li) for j in cands)
        tied = [j for j in cands if (abs(n_tok[j["item_id"]] - li), n_tok[j["item_id"]] < li) == best]
        j = tied[int(rng.integers(len(tied)))]
        out[it["item_id"]] = j
        stats.append(n_tok[j["item_id"]] - li)
    return out, stats


# ---------------------------------------------------------------------------------------- models
def load(arm):
    free = psutil.virtual_memory().available / 2 ** 30
    if free < MIN_FREE_GIB:
        raise SystemExit(f"only {free:.2f} GiB host RAM free (< {MIN_FREE_GIB}); not loading {arm}")
    t0 = time.time()
    m, fp = load_cpu(arm)
    log(f"loaded {arm} on CPU ({fp[:12]}, {time.time() - t0:.0f}s, {free:.1f} GiB free before load)")
    return m, fp


def unload(model):
    del model
    gc.collect()


def block_hashes(model):
    """sha1 over every parameter of each decoder block (full bytes, no copies)."""
    out = []
    for blk in model.model.layers:
        h = hashlib.sha1()
        for name, p in sorted(blk.named_parameters(), key=lambda kv: kv[0]):
            h.update(name.encode())
            h.update(np.ascontiguousarray(p.detach().numpy()))
        out.append(h.hexdigest()[:16])
    return out


# ------------------------------------------------------------------------------- capture + patch
@torch.no_grad()
def capture(model, tok, b, layers):
    """Block-L outputs for every L in `layers` during the prefill of generate(max_new_tokens=1):
    the same code path as the receiver's prefill, so identity patches are bit-exact."""
    ids, att, lens = encode(tok, b)
    got, handles = {}, []
    for L in layers:
        def hook(mod, inp, out, L=L):
            if L not in got:                      # first call = prefill
                h = out[0] if isinstance(out, tuple) else out
                assert h.shape[:2] == ids.shape
                got[L] = h.detach().clone()
            return out
        handles.append(model.model.layers[L - 1].register_forward_hook(hook))
    try:
        model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=1,
                       eos_token_id=generation.EOS, pad_token_id=generation.PAD,
                       temperature=None, top_p=None, top_k=None)
    finally:
        for hd in handles:
            hd.remove()
    assert set(got) == set(layers)
    return {"ids": ids, "att": att, "lens": lens, "H": got}


class Patcher:
    """Forward hook on block L-1: on the FIRST call (prefill, seq_len == prompt length) replace the
    output at mask positions by `patch`; every later call must be a seq_len-1 decode step and is
    left untouched. Counts calls for sanity (c)."""

    def __init__(self, patch, mask, n_prompt):
        self.patch, self.mask, self.n_prompt = patch, mask, n_prompt
        self.calls = self.prefill_patched = self.decode_calls = 0
        self.max_abs = self.rel = None

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.calls += 1
        if self.calls == 1:
            assert h.shape[1] == self.n_prompt and h.shape[0] == self.patch.shape[0], (h.shape, self.patch.shape)
            m = self.mask
            own, don = h[m], self.patch[m].to(h.dtype)
            d = don - own
            self.max_abs = float(d.abs().max()) if d.numel() else 0.0
            self.rel = float(d.norm(dim=-1).mean() / own.norm(dim=-1).mean()) if d.numel() else 0.0
            h = torch.where(m[..., None], self.patch.to(h.dtype), h)
            self.prefill_patched += 1
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        assert h.shape[1] == 1, f"sanity (c): non-decode call after prefill (seq_len {h.shape[1]})"
        self.decode_calls += 1
        return out


@torch.no_grad()
def generate(model, tok, b, layer=None, patch_fn=None):
    """run_steer_cpu.gen_with for one batch, optionally with a prefill-only patch at block `layer`.
    patch_fn(ids, att, lens) -> (patch [B, Lm, D], mask [B, Lm] bool)."""
    fam = b[0]["family"]
    assert all(it["family"] == fam for it in b)
    ids, att, lens = encode(tok, b)
    Lm = ids.shape[1]
    pt, handle, info = None, None, {}
    if patch_fn is not None:
        patch, mask = patch_fn(ids, att, lens)
        assert patch.shape[:2] == ids.shape and mask.shape == ids.shape
        assert not bool((mask & (att == 0)).any()), "patch mask touches a pad position"
        pt = Patcher(patch, mask, Lm)
        handle = model.model.layers[layer - 1].register_forward_hook(pt)
    try:
        g = model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                           max_new_tokens=generation.max_tokens(fam), eos_token_id=generation.EOS,
                           pad_token_id=generation.PAD, temperature=None, top_p=None, top_k=None)
    finally:
        if handle is not None:
            handle.remove()
    n_new = g.shape[1] - Lm
    if pt is not None:                               # sanity (c), every patched batch
        assert pt.prefill_patched == 1 and pt.calls == n_new and pt.decode_calls == n_new - 1, \
            (pt.prefill_patched, pt.calls, pt.decode_calls, n_new)
        info = {"calls": pt.calls, "prefill_patched": pt.prefill_patched, "decode_calls": pt.decode_calls,
                "n_new": n_new, "max_abs": pt.max_abs, "rel": pt.rel, "n_patched": int(pt.mask.sum())}
    texts = {}
    for r, it in enumerate(b):
        new = g[r, Lm:].tolist()
        cut = next((j for j, t in enumerate(new) if t in generation.EOS), len(new))
        texts[it["item_id"]] = tok.decode(new[:cut], skip_special_tokens=True).strip()
    return texts, info


def same_item_patch(cap, kind):
    """Donor captured on the identical batch: positions align row by row."""
    def fn(ids, att, lens):
        assert torch.equal(ids, cap["ids"]) and torch.equal(att, cap["att"]), "donor batch differs from receiver batch"
        if kind == "last":
            mask = torch.zeros_like(att, dtype=torch.bool)
            mask[:, -1] = True                       # sanity (d): left padding => last real token is -1
            assert bool(att[:, -1].all()) and int(mask.sum()) == att.shape[0]
        else:
            mask = att.bool()
        return cap["H"], mask
    return fn


def mismatch_patch(dcap, L):
    """Right-align donor j's real span into receiver row r: k = min(len_i, len_j) positions,
    receiver[-k:] <- donor[-k:]; receiver positions before that (len_j < len_i) keep their own
    states; donor tokens before that (len_j > len_i) are dropped."""
    def fn(ids, att, lens):
        B, Lm = ids.shape
        Hd, Ld, dl = dcap["H"][L], dcap["ids"].shape[1], dcap["lens"]
        assert Hd.shape[0] == B
        patch = torch.zeros((B, Lm, Hd.shape[-1]), dtype=Hd.dtype)
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        for r in range(B):
            k = min(lens[r], dl[r])
            patch[r, Lm - k:] = Hd[r, Ld - k:]
            mask[r, Lm - k:] = True
        return patch, mask
    return fn


# ------------------------------------------------------------------------------------- scoring
def parse_rate(items, texts):
    out = {}
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        out[fam] = sum(env.parse_state(fam, texts[it["item_id"]], lenient=True) is not None for it in fi) / max(len(fi), 1)
    return out


def donor_match(items, texts, donors):
    per = {}
    for it in items:
        st = env.parse_state(it["family"], texts[it["item_id"]], lenient=True)
        per[it["item_id"]] = bool(st is not None and env.dumps(st) == donors[it["item_id"]]["gold"])
    fam = {f: float(np.mean([per[it["item_id"]] for it in items if it["family"] == f])) for f in env.FAMILIES
           if any(it["family"] == f for it in items)}
    return fam, per


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--per-family", type=int, default=48)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mis-seed", type=int, default=17)
    ap.add_argument("--out-name", default="patch_cpu")
    ap.add_argument("--conditions", default=",".join(COND_KINDS),
                    help="comma list of " + ",".join(COND_KINDS) + " (baselines always run)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sanity-per-family", type=int, default=4)
    ap.add_argument("--items-from", default=str(ROOT / "runs" / "bank_pilot" / "steer_cpu" / "steer_cpu_results.json"))
    a = ap.parse_args()
    assert a.threads == _THREADS
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    layers = sorted(int(x) for x in a.layers.split(","))
    assert all(1 <= L <= 27 for L in layers)
    kinds = [k for k in a.conditions.split(",") if k]
    assert all(k in COND_KINDS for k in kinds), kinds

    out_dir = ROOT / "runs" / a.bank / a.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = json.loads((ROOT / "data" / f"{a.bank}.json").read_text(encoding="utf-8"))
    pool = [it for it in bankmod.items(bank) if it["split"] == "test" and it["format"] == "one"]
    items = select_items(a, pool)
    ids = [it["item_id"] for it in items]
    tok = runtime.tokenizer()
    B = batches_of(items, a.batch)
    sanity_items = [it for f in env.FAMILIES for it in [x for x in items if x["family"] == f][: a.sanity_per_family]]
    SB = batches_of(sanity_items, a.batch)

    res_path = out_dir / "patch_results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    assert results.get("_meta", {}).get("items", ids) == ids, "existing results are for a different subset"
    meta = results.setdefault("_meta", {})
    meta.update({"bank": a.bank, "items": ids, "threads": a.threads, "device": "cpu", "dtype": "float32",
                 "seed": a.seed, "batch": a.batch, "sanity_items": [it["item_id"] for it in sanity_items]})
    meta.setdefault("layers_run", [])
    meta["layers_run"] = sorted(set(meta["layers_run"]) | set(layers))
    sanity = results.setdefault("_sanity", {"T2T": {}, "S2S": {}})
    timing = results.setdefault("_timing", {})

    def save():
        res_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")

    donors, dl_stats = pick_mismatch(items, pool, tok, a.mis_seed)
    meta["mismatch"] = {
        "rule": "same family; different parent, gold and before-state; equal token length if available, "
                "else nearest (ties -> longer donor); right-aligned, k=min(len_i,len_j) positions",
        "mis_seed": a.mis_seed, "donor": {i: donors[i]["item_id"] for i in ids},
        "len_diff_counts": {str(d): dl_stats.count(d) for d in sorted(set(dl_stats))}}
    log(f"{len(items)} items in {len(B)} batches; mismatch donors: len(donor)-len(item) counts "
        f"{meta['mismatch']['len_diff_counts']}")

    conds = ["base:supa", "base:sparse"] + [f"{k}@{L}" for L in layers for k in kinds]
    pending = [c for c in conds if c not in results]
    log("done earlier: " + (", ".join(c for c in conds if c in results) or "none"))
    log("pending: " + (", ".join(pending) or "none"))

    def run_cond(name, model, layer=None, patch_for_batch=None, extra=None):
        if name in results:
            log(f"{name:<18} done earlier; skipping")
            return
        t0 = time.time()
        texts, infos = {}, []
        for bi, b in enumerate(B):
            t, info = generate(model, tok, b, layer, None if patch_for_batch is None else patch_for_batch(bi))
            texts.update(t)
            if info:
                infos.append(info)
        secs = time.time() - t0
        rec = {"score": score(items, texts), "correct": correct(items, texts), "parse": parse_rate(items, texts),
               "seconds": round(secs, 1), "n_items": len(items)}
        if infos:
            rec["patch"] = {"batches": len(infos), "prefill_patches": sum(i["prefill_patched"] for i in infos),
                            "decode_calls_unpatched": sum(i["decode_calls"] for i in infos),
                            "positions_patched": sum(i["n_patched"] for i in infos),
                            "max_abs_diff": max(i["max_abs"] for i in infos),
                            "mean_rel_diff": float(np.mean([i["rel"] for i in infos]))}
        if extra is not None:
            rec.update(extra(texts))
        results[name] = rec
        timing[name] = {"seconds": round(secs, 1), "n_items": len(items), "s_per_item": round(secs / len(items), 2)}
        (out_dir / f"gen_{name.replace(':', '_').replace('@', '_')}.json").write_text(json.dumps(texts) + "\n", encoding="utf-8")
        save()
        s = rec["score"]
        log(f"{name:<18} " + " | ".join(f"{f[:6]} len {v['lenient']:.3f} chg {v['changed_lenient']:.3f} parse {rec['parse'][f]:.2f}"
                                        for f, v in s.items())
            + (f" | patch max|d| {rec['patch']['max_abs_diff']:.3g} rel {rec['patch']['mean_rel_diff']:.3f}" if infos else "")
            + (f" | donor_match {rec['donor_match']}" if "donor_match" in rec else "") + f" ({secs:.0f}s)")

    def run_sanity(model, arm, key):
        """(a)/(b): identity patch arm->arm all@L on the sanity batches reproduces unpatched texts."""
        todo = [L for L in layers if not sanity[key].get(str(L), {}).get("pass")]
        if not todo:
            log(f"sanity {key}: passed earlier for layers {layers}; skipping")
            return
        t0 = time.time()
        ref = {}
        for b in SB:
            ref.update(generate(model, tok, b)[0])
        caps = [capture(model, tok, b, todo) for b in SB]
        for L in todo:
            got, infos = {}, []
            for b, cap in zip(SB, caps):
                fn = same_item_patch({"ids": cap["ids"], "att": cap["att"], "H": cap["H"][L]}, "all")
                t, info = generate(model, tok, b, L, fn)
                got.update(t); infos.append(info)
            diff = [i for i in ref if ref[i] != got[i]]
            max_abs = max(i["max_abs"] for i in infos)
            rec = {"pass": not diff and max_abs == 0.0, "n_items": len(ref), "texts_differ": diff,
                   "patch_max_abs_diff": max_abs,
                   "hook": {"prefill_patches": sum(i["prefill_patched"] for i in infos), "batches": len(infos),
                            "decode_calls_unpatched": sum(i["decode_calls"] for i in infos),
                            "generated_tokens": sum(i["n_new"] for i in infos),
                            "positions_patched": sum(i["n_patched"] for i in infos)},
                   "last_is_index_minus_1": True}
            sanity[key][str(L)] = rec
            save()
            log(f"sanity {key} all@{L}: {'PASS' if rec['pass'] else 'FAIL'} ({len(ref) - len(diff)}/{len(ref)} texts identical, "
                f"patch max|d| {max_abs}, hook: {rec['hook']})")
            assert rec["pass"], f"sanity {key} all@{L} failed: {diff[:5]}"
        # (d) 'last' mask on every batch of the run
        for b in B:
            _, att, _ = encode(tok, b)
            assert bool(att[:, -1].all())
        sanity[key]["seconds"] = round(time.time() - t0, 1)
        save()

    def need(*names):
        return [n for n in names if n in pending]

    t2s = [c for c in pending if c.startswith("T2S_")]
    s2t = [c for c in pending if c.startswith("S2T_")]
    lay = lambda cs: sorted({int(c.split("@")[1]) for c in cs})  # noqa: E731

    # ---- phase 1: Titan-Sparse -> sanity (a), capture Titan states for T2S_* ----
    tcap, mcap = {}, {}
    sanity_a_pending = any(not sanity["T2T"].get(str(L), {}).get("pass") for L in layers)
    if t2s or sanity_a_pending:
        model, fp = load("sparse")
        meta["fingerprint_sparse"] = fp
        meta["block_hash_sparse"] = block_hashes(model)
        run_sanity(model, "sparse", "T2T")
        t0 = time.time()
        same_layers = lay([c for c in t2s if not c.startswith("T2S_all_mis")])
        mis_layers = lay([c for c in t2s if c.startswith("T2S_all_mis")])
        if same_layers:
            tcap = {bi: capture(model, tok, b, same_layers) for bi, b in enumerate(B)}
        if mis_layers:
            mcap = {bi: capture(model, tok, [donors[it["item_id"]] for it in b], mis_layers) for bi, b in enumerate(B)}
        timing["capture_sparse"] = round(time.time() - t0, 1)
        log(f"captured Titan states: same-item layers {same_layers}, mismatch layers {mis_layers} ({time.time() - t0:.0f}s)")
        unload(model)

    # ---- phase 2: Supa -> sanity (b), base:supa, T2S_*, capture Supa states for S2T_all ----
    scap = {}
    sanity_b_pending = any(not sanity["S2S"].get(str(L), {}).get("pass") for L in layers)
    if t2s or s2t or sanity_b_pending or need("base:supa"):
        model, fp = load("supa")
        meta["fingerprint_supa"] = fp
        meta["block_hash_supa"] = block_hashes(model)
        if "block_hash_sparse" in meta:
            same = [i for i, (x, y) in enumerate(zip(meta["block_hash_supa"], meta["block_hash_sparse"])) if x == y]
            meta["identical_blocks"] = same
            noop = [L for L in layers if all(i in same for i in range(L))]
            log(f"blocks bit-identical across arms: {same}; patch is an exact no-op at L in {noop}")
        run_sanity(model, "supa", "S2S")
        run_cond("base:supa", model)
        for L in layers:
            if f"T2S_last@{L}" in pending:
                run_cond(f"T2S_last@{L}", model, L, lambda bi, L=L: same_item_patch(
                    {"ids": tcap[bi]["ids"], "att": tcap[bi]["att"], "H": tcap[bi]["H"][L]}, "last"))
            if f"T2S_all@{L}" in pending:
                run_cond(f"T2S_all@{L}", model, L, lambda bi, L=L: same_item_patch(
                    {"ids": tcap[bi]["ids"], "att": tcap[bi]["att"], "H": tcap[bi]["H"][L]}, "all"))
            if f"T2S_all_mis@{L}" in pending:
                def extra(texts):
                    fam, per = donor_match(items, texts, donors)
                    return {"donor_match": {f: round(v, 3) for f, v in fam.items()}, "donor_correct": per}
                run_cond(f"T2S_all_mis@{L}", model, L, lambda bi, L=L: mismatch_patch(mcap[bi], L), extra)
        tcap.clear(); mcap.clear(); gc.collect()
        if s2t:
            t0 = time.time()
            scap = {bi: capture(model, tok, b, lay(s2t)) for bi, b in enumerate(B)}
            timing["capture_supa"] = round(time.time() - t0, 1)
            log(f"captured Supa states: layers {lay(s2t)} ({time.time() - t0:.0f}s)")
        unload(model)

    # ---- phase 3: Titan-Sparse -> base:sparse, S2T_all ----
    if s2t or need("base:sparse"):
        model, fp = load("sparse")
        meta["fingerprint_sparse"] = fp
        run_cond("base:sparse", model)
        for L in layers:
            if f"S2T_all@{L}" in pending:
                run_cond(f"S2T_all@{L}", model, L, lambda bi, L=L: same_item_patch(
                    {"ids": scap[bi]["ids"], "att": scap[bi]["att"], "H": scap[bi]["H"][L]}, "all"))
        unload(model)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("[patch_cpu] finished")


if __name__ == "__main__":
    main()
