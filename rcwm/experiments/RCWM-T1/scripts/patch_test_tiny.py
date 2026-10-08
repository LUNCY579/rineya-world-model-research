"""End-to-end test of scripts/run_patch.py on CPU with TWO tiny random Qwen3 models (no real weights,
no GPU, no lease).

Stand-ins: "supa" = Qwen3 (4 blocks, hidden 64, 4 heads, 2 kv heads, head_dim 16, real vocab size
151936, tied head) drawn with seed 1001 and rounded to BF16-representable values; "sparse" = the same
embeddings, final norm and block 0, with blocks 1..3 redrawn in FP32 from seed 2002. This mirrors the
real pair (BF16 Supa file, FP32 Titan file, shared embeddings/head, bit-identical lower blocks), so
L = 1 is an exact no-op and L = 4 (last block) must hand the donor's first token to the receiver.
Weight files for the hybrid splices: supa saved as BF16 safetensors, sparse as FP32, like the cores.
Items: 6 real pilot TEST one-frame prompts per family (2 per parent, 3 parents), tokenized with the
real tokenizer (115-143 tokens, no truncation needed). The loader, the weight files, the item pool,
the hidden-size hint and the expected fingerprints are monkeypatched; everything else is the
production code path. Run flags: --layers 1,2,4 --dmis-layers 1,4 --hybrid-layers 2,4.

Checks
  0  item selection: --per-family 48 reproduces runs/bank_pilot/steer_cpu _meta.items; default = 640
  1  mismatch right-alignment (longer and shorter donor), the additive T2S_dmis difference patch and
     'last' masking on a synthetic staging
  2  Patcher/Capture reject a non-prefill first call and a multi-token later call; additive mode
  3  crash after 5 conditions: no patch_results.json (master_chain marker), progress kept
  4  resume: only the pending conditions run, Titan staging reused (no phase A), final file written,
     staged arrays deleted
  5  fresh uninterrupted run (--keep-states) == resumed run, condition by condition
  6  sanity: identity all/last (and additive zero-diff at dmis layers) exact (max|d| 0), zero@last
     changes outputs, hook counts; no-op layer L=1 (incl. T2S_dmis@1) reproduces the receiver baseline
     exactly; T2S_all/T2S_last@4 reproduce the donor's FIRST token (S2T_all@4 the reverse); T2S_mis@4
     the mismatched donor's first token; T2S_dmis@4 the first token of S(i) + T(j) - S(j) computed
     independently from the staged arrays; staged rows equal an unpadded forward's hidden_states[L];
     hybrids: splice sanity hash-equal and text-equal on both hosts, HlowT@4 == base:sparse and
     HupT@4 == base:supa texts exactly, HlowT@2 / HupT@2 first tokens == T2S_all@2 / S2T_all@2 per item;
     mis/dmis donor metrics and _derived present
  7  rerun of a finished directory loads nothing and leaves patch_results.json byte-identical;
     --plan loads nothing and writes nothing
  8  corrupted staged sanity states -> SanityError, nothing written but patch_sanity.json
  9  --device cuda wiring with stubbed runtime (lease once, load_arm sparse/supa/sparse, throttle and
     breathe once per generate) gives results identical to the cpu run; CUDA never initialised
 10  a wrong splice source (Supa's file served for Titan) trips the hash gate: SanityError, no hybrid
     result, no patch_results.json
 11  derived(): FR, labels, parent-cluster bootstrap CI, exact-length restriction and paired donor
     references on synthetic results

  python scripts/patch_test_tiny.py [--keep]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # NOT "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(2)

import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import run_patch as P  # noqa: E402
sys.argv = _argv
from rcwmt import env, runtime  # noqa: E402
import transformers  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models

N_LAYERS, HID = 4, 64
SEEDS = {"supa": 1001, "sparse": 2002}
LAYERS, DMIS_LAYERS, HYB_LAYERS = "1,2,4", "1,4", "2,4"
N_CONDS = 2 + 4 * 3 + 2 + 2 * 2
WFILES: dict[str, Path] = {}
CFG = dict(vocab_size=151936, hidden_size=HID, intermediate_size=128, num_hidden_layers=N_LAYERS,
           num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
           rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True)
LOADS: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    assert ok, f"{name}: {detail}"


@torch.no_grad()
def _draw_block(blk, g):
    for name, p in blk.named_parameters():
        if p.dim() == 2:                              # linear weights; norms stay at 1
            p.copy_(torch.randn(p.shape, generator=g) * 0.15)


@torch.no_grad()
def build(arm: str):
    cfg = Qwen3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(SEEDS["supa"])
    m = Qwen3ForCausalLM(cfg)
    g = torch.Generator().manual_seed(SEEDS["supa"])
    m.model.embed_tokens.weight.copy_(torch.randn(m.model.embed_tokens.weight.shape, generator=g))
    for blk in m.model.layers:
        _draw_block(blk, g)
    for p in m.parameters():                          # Supa is stored in BF16: make it BF16-exact
        p.copy_(p.to(torch.bfloat16).float())
    if arm == "sparse":                               # blocks 1..3 from a different seed, FP32
        g2 = torch.Generator().manual_seed(SEEDS["sparse"])
        for blk in m.model.layers[1:]:
            _draw_block(blk, g2)
    return m.eval()


def write_weight_files(d: Path):
    """The cores' single safetensors files: supa BF16, sparse FP32, no lm_head (tied)."""
    from safetensors.torch import save_file
    for arm, dt in (("supa", torch.bfloat16), ("sparse", torch.float32)):
        m = build(arm)
        WFILES[arm] = d / f"{arm}.safetensors"
        save_file({k: v.detach().to(dt).contiguous() for k, v in m.state_dict().items() if k.startswith("model.")},
                  str(WFILES[arm]))
        del m


def fake_load_cpu(arm):
    LOADS.append(arm)
    m = build(arm)
    return m, runtime.param_fingerprint(m)


def fake_pool(real_pool):
    out = []
    for fam in env.FAMILIES:
        fi = [it for it in real_pool if it["family"] == fam]
        parents = []
        for it in fi:
            if it["parent_id"] not in parents:
                parents.append(it["parent_id"])
        for pid in parents[:3]:
            out += [it for it in fi if it["parent_id"] == pid][:2]
    return out


def args_for(d, *extra, device="cpu"):
    return ["--device", device, "--layers", LAYERS, "--dmis-layers", DMIS_LAYERS, "--hybrid-layers", HYB_LAYERS,
            "--batch", "4", "--sanity-per-family", "3", "--max-new-tokens", "6", "--items-from", "",
            "--threads", "2", "--out-dir", str(d), *extra]


def load_res(d):
    return json.loads((Path(d) / "patch_results.json").read_text(encoding="utf-8"))


def gen_texts(d, name):
    return json.loads((Path(d) / f"gen_{name.replace(':', '_').replace('@', '_')}.json").read_text(encoding="utf-8"))


class Crash(Exception):
    pass


# ------------------------------------------------------------------------------------------ tests
def t0_selection(real_pool):
    print("0 item selection", flush=True)
    sel, note = P.select_items(real_pool, None, 17, "")
    check("default = all 640 TEST one-frame items in bank order",
          len(sel) == 640 and [i["item_id"] for i in sel] == [i["item_id"] for i in real_pool], f"{len(sel)}")
    if P.STEER_CPU_RESULTS.exists():
        sel48, note48 = P.select_items(real_pool, 48, 17, str(P.STEER_CPU_RESULTS))
        ref = json.loads(P.STEER_CPU_RESULTS.read_text(encoding="utf-8"))["_meta"]["items"]
        check("--per-family 48 reproduces steer_cpu _meta.items", [i["item_id"] for i in sel48] == ref and "equals" in note48, note48)
    else:
        check("steer_cpu results present for the cross-check", False, "missing")


def t1_alignment(tmp):
    print("1 mismatch alignment / last mask", flush=True)
    d = tmp / "align"
    d.mkdir()
    items = [{"item_id": "r", "family": env.FAMILIES[0]}, {"item_id": "long", "family": env.FAMILIES[0]},
             {"item_id": "short", "family": env.FAMILIES[0]}]
    enc = {"r": [11, 12, 13, 14, 15], "long": [21, 22, 23, 24, 25, 26, 27], "short": [31, 32, 33]}
    code = {"r": 1, "long": 2, "short": 3}
    lens = {k: len(v) for k, v in enc.items()}
    st = P.Staging(d, "synthetic")
    st.begin("x", items, lens, [items], [1], 2, 1)
    ids, att = P.left_pad([enc[i["item_id"]] for i in items])
    Lm = ids.shape[1]
    cap = torch.zeros((3, Lm, 2))
    for r, it in enumerate(items):
        n = lens[it["item_id"]]
        cap[r, Lm - n:, 0] = torch.tensor([code[it["item_id"]] * 100 + p for p in range(n)], dtype=torch.float32)
    st.write_batch(items, ids, {1: cap})
    st.finish(texts={}, first_token={}, fingerprint="x")
    st.open()
    rid, ratt = P.left_pad([enc["r"], [9] * 8])           # receiver row padded inside a longer batch
    Lr = rid.shape[1]
    for donor, k, first in (("long", 5, 2), ("short", 3, 0)):
        fn = P.mismatch_patch(st, [items[0], {"item_id": "pad", "family": env.FAMILIES[0]}],
                              {**enc, "pad": [9] * 8}, {"r": {"item_id": donor}, "pad": {"item_id": "r"}}, 1)
        patch, mask = fn(rid, ratt)
        got = patch[0, :, 0][mask[0]].tolist()
        want = [code[donor] * 100 + p for p in range(first, first + k)]
        check(f"donor '{donor}' right-aligned: receiver[-{k}:] <- donor[{first}:]",
              got == want and int(mask[0].sum()) == k and bool(mask[0, Lr - k:].all()), f"{got}")
    st2 = P.Staging(d, "synthetic_s")                  # a second core's staging for T2S_dmis
    st2.begin("y", items, lens, [items], [1], 2, 1)
    cap2 = torch.zeros((3, Lm, 2))
    for r, it in enumerate(items):
        n = lens[it["item_id"]]
        cap2[r, Lm - n:, 0] = torch.tensor([code[it["item_id"]] * 7 + 0.5 * p for p in range(n)], dtype=torch.float32)
    st2.write_batch(items, ids, {1: cap2})
    st2.finish(texts={}, first_token={}, fingerprint="y")
    st2.open()
    fn = P.diff_mismatch_patch(st, st2, [items[0], {"item_id": "pad", "family": env.FAMILIES[0]}],
                               {**enc, "pad": [9] * 8}, {"r": {"item_id": "long"}, "pad": {"item_id": "r"}}, 1)
    patch, mask = fn(rid, ratt)
    got = patch[0, :, 0][mask[0]].tolist()
    want = [(200 + p) - (14 + 0.5 * p) for p in range(2, 7)]
    check("T2S_dmis patch: additive, receiver[-5:] += T(long)[2:] - S(long)[2:]",
          getattr(fn, "additive", False) and got == want and int(mask[0].sum()) == 5 and bool(mask[0, Lr - 5:].all()), f"{got}")
    st2.delete()
    fn = P.same_item_patch(st, [items[0]], enc, 1, "last")
    pid, patt = P.left_pad([enc["r"]])
    patch, mask = fn(pid, patt)
    check("'last' masks only index -1 with the answer-slot row",
          int(mask.sum()) == 1 and bool(mask[0, -1]) and float(patch[0, -1, 0]) == 104.0)
    bad = pid.clone()
    bad[0, -2] += 1
    try:
        fn(bad, patt)
        ok = False
    except P.SanityError:
        ok = True
    check("donor/receiver token-id mismatch raises SanityError", ok)
    st.delete()


def t2_hook_guards():
    print("2 prefill detection guards", flush=True)
    B, Lm, D = 2, 5, 3
    pt = P.Patcher(torch.zeros(B, Lm, D), torch.ones(B, Lm, dtype=torch.bool), Lm)
    try:
        pt(None, None, torch.zeros(B, Lm - 2, D))
        ok = False
    except P.SanityError:
        ok = True
    check("Patcher: first call that is not the full prefill raises", ok)
    pt = P.Patcher(torch.zeros(B, Lm, D), torch.ones(B, Lm, dtype=torch.bool), Lm)
    pt(None, None, torch.ones(B, Lm, D))
    out = pt(None, None, torch.ones(B, 1, D))
    try:
        pt(None, None, torch.ones(B, 2, D))
        ok = False
    except P.SanityError:
        ok = True
    check("Patcher: decode step untouched; a later multi-token call raises",
          out is None and pt.decode_calls == 1 and ok and pt.max_abs == 1.0)
    m_last = torch.zeros(B, Lm, dtype=torch.bool)
    m_last[:, -1] = True
    pt = P.Patcher(torch.full((B, Lm, D), 2.0), m_last, Lm, additive=True)
    out = pt(None, None, torch.ones(B, Lm, D))
    check("Patcher additive: h + patch at the mask only, max|d| = |patch|",
          bool((out[:, -1] == 3.0).all()) and bool((out[:, :-1] == 1.0).all()) and pt.max_abs == 2.0)
    c = P.Capture(B, Lm)
    try:
        c(None, None, torch.zeros(B, 1, D))
        ok = False
    except P.SanityError:
        ok = True
    check("Capture: first call that is not the prefill raises", ok)


def run_main(argv):
    LOADS.clear()
    P.main(argv)
    return list(LOADS)


def t3_t4_crash_resume(tmp):
    print("3 crash after 5 conditions", flush=True)
    d = tmp / "resume"
    done = []

    def crash(name):
        done.append(name)
        if len(done) == 5:
            raise Crash(name)
    P._on_condition_done = crash
    try:
        run_main(args_for(d))
        crashed = False
    except Crash:
        crashed = True
    P._on_condition_done = None
    prog = json.loads((d / "patch_progress.json").read_text(encoding="utf-8"))
    conds = [k for k in prog if not k.startswith("_")]
    check("crash propagated; patch_results.json absent; progress holds 5 conditions",
          crashed and not (d / "patch_results.json").exists() and len(conds) == 5, f"{conds}")
    san = json.loads((d / "patch_sanity.json").read_text(encoding="utf-8"))
    check("both sanity records passed before the first result", san["sparse"]["pass"] and san["supa"]["pass"])
    check("first committed results are the baselines (after both sanities)", conds[:2] == ["base:sparse", "base:supa"], f"{conds}")

    print("4 resume", flush=True)
    ran = []
    P._on_condition_done = ran.append
    loads = run_main(args_for(d))
    P._on_condition_done = None
    res = load_res(d)
    allc = [k for k in res if not k.startswith("_")]
    check("resume ran exactly the pending conditions", sorted(ran) == sorted(set(allc) - set(conds)) and not set(ran) & set(conds),
          f"ran {ran}")
    check("Titan staging reused: phase A skipped (loads supa, sparse)", loads == ["supa", "sparse"], f"{loads}")
    check("final patch_results.json written, progress removed, staged arrays deleted",
          (d / "patch_results.json").exists() and not (d / "patch_progress.json").exists()
          and not list(d.glob("donor_*")) and not list(d.glob("sanity_*")), f"{sorted(p.name for p in d.iterdir())}")
    return d


def t5_t6_fresh(tmp, d_resume):
    print("5 fresh run == resumed run", flush=True)
    d = tmp / "fresh"
    loads = run_main(args_for(d, "--keep-states"))
    check("fresh run loads sparse, supa, sparse", loads == ["sparse", "supa", "sparse"], f"{loads}")
    rf, rr = load_res(d), load_res(d_resume)
    names = [k for k in rf if not k.startswith("_")]
    same = all(gen_texts(d, n) == gen_texts(d_resume, n) and rf[n]["correct"] == rr[n]["correct"]
               and rf[n]["first_token"] == rr[n]["first_token"] for n in names)
    check(f"{len(names)} conditions identical between fresh and resumed runs", same and len(names) == N_CONDS, f"{names}")

    print("6 sanity / identity / first-token checks", flush=True)
    san = rf["_sanity"]
    for arm in ("sparse", "supa"):
        s = san[arm]
        ok = s["pass"]
        for L, r in s["per_layer"].items():
            ws = ["identity_all", "identity_last"] + (["identity_add_zero_diff"] if L in DMIS_LAYERS.split(",") else [])
            ok &= all(w in r for w in ws)
            for w in ws:
                x = r[w]
                ok &= x["texts_identical"] == x["n"] and x["patch_max_abs_diff"] == 0.0
                ok &= x["prefill_patches"] == x["batches"] and x["decode_calls_unpatched"] == x["generated_tokens"] - x["batches"]
            ok &= r["zero_last_changed"] >= 1
        check(f"sanity {arm}: identity all/last (+additive zero-diff at dmis layers) exact, hook once per generate "
              f"in prefill, zero@last live", ok,
              f"zero changed {[r['zero_last_changed'] for r in s['per_layer'].values()]}")
    meta = rf["_meta"]
    check("no-op layers from parameter hashes == [1]", meta["noop_layers"] == [1], f"{meta['noop_layers']}")
    bs, bp = rf["base:sparse"]["first_token"], rf["base:supa"]["first_token"]
    ids = meta["items"]
    differ = sum(bs[i] != bp[i] for i in ids)
    check("baselines differ (non-vacuous first-token checks)", differ >= len(ids) // 2, f"{differ}/{len(ids)} first tokens differ")
    ts, tp = gen_texts(d, "base:sparse"), gen_texts(d, "base:supa")
    for n, base in (("T2S_last@1", tp), ("T2S_all@1", tp), ("S2T_all@1", ts), ("T2S_dmis@1", tp)):
        check(f"{n} is an exact no-op (== receiver baseline, max|d| 0)",
              gen_texts(d, n) == base and rf[n]["noop_exact"] and rf[n]["patch"]["max_abs_diff"] == 0.0)
    for n, want in (("T2S_all@4", bs), ("T2S_last@4", bs), ("S2T_all@4", bp)):
        got = rf[n]["first_token"]
        check(f"{n} reproduces the donor's FIRST token on every item", all(got[i] == want[i] for i in ids),
              f"ft_donor {rf[n]['ft_donor_all']}")
    don = {i: v["donor"] for i, v in meta["mismatch"]["per_item"].items()}
    got = rf["T2S_mis@4"]["first_token"]
    check("T2S_mis@4 reproduces the MISMATCHED donor's first token", all(got[i] == bs[don[i]] for i in ids))
    byid = {it["item_id"]: it for it in FAKE}
    check("mismatch donors: same family, different parent/gold/before-state",
          all(byid[j]["family"] == byid[i]["family"] and byid[j]["parent_id"] != byid[i]["parent_id"]
              and byid[j]["gold"] != byid[i]["gold"] for i, j in don.items()),
          f"len_diff counts {meta['mismatch']['len_diff_counts']}")
    check("T2S_all@2 changes Supa's output; T2S_mis@2 differs from T2S_all@2",
          gen_texts(d, "T2S_all@2") != tp and gen_texts(d, "T2S_mis@2") != gen_texts(d, "T2S_all@2"))
    okh = all(rf[n]["patch"]["prefill_patches"] == rf[n]["patch"]["batches"]
              and rf[n]["patch"]["decode_calls_unpatched"] == rf[n]["patch"]["generated_tokens"] - rf[n]["patch"]["batches"]
              for n in names if "@" in n and not n.startswith(("HlowT", "HupT")))
    check("every patched condition: one prefill patch per generate, decode never patched", okh)
    check("hybrid conditions carry no activation patch", all("patch" not in rf[n] for n in names if n.startswith(("HlowT", "HupT"))))
    # T2S_dmis@4: first token == argmax head(norm(S_i + (T_j - S_j))) at the answer slot, from the staged arrays
    m = build("supa")
    ia = json.loads((d / "donor_supa_index.json").read_text(encoding="utf-8"))
    ib = json.loads((d / "donor_sparse_index.json").read_text(encoding="utf-8"))
    aS, aT = np.load(d / "donor_supa_L4.npy"), np.load(d / "donor_sparse_L4.npy")
    exact = {i: v["exact_len"] for i, v in meta["mismatch"]["per_item"].items()}
    want = {}
    with torch.no_grad():
        for i in ids:
            j = don[i]
            s_i = aS[ia["offset"][i][0] + ia["offset"][i][1] - 1]
            dlt = aT[ib["offset"][j][0] + ib["offset"][j][1] - 1] - aS[ia["offset"][j][0] + ia["offset"][j][1] - 1]
            x = torch.from_numpy(s_i + dlt)[None, None]
            want[i] = int(m.lm_head(m.model.norm(x))[0, -1].argmax())
    del aS, aT
    got = rf["T2S_dmis@4"]["first_token"]
    check("T2S_dmis@4 first token == head(norm(S_i + T_j - S_j)) from the staged arrays, every item",
          all(got[i] == want[i] for i in ids), f"{sum(got[i] == want[i] for i in ids)}/{len(ids)}; exact-len {sum(exact.values())}")
    for n in ("T2S_mis@4", "T2S_dmis@4"):
        r = rf[n]
        check(f"{n}: donor_match_changed, echo and exact_len_subset recorded",
              all(k in r for k in ("donor_match", "donor_match_changed", "echo_donor_before_changed", "exact_len_subset",
                                   "format", "echo_before_changed")))
    # hybrids
    for host, other in (("supa", "sparse"), ("sparse", "supa")):
        sp = san[f"splice_{host}"]
        brief = {k: v for k, v in sp.items() if k != "differ"}
        check(f"splice sanity on {host}: blocks 1..3 from {other} hash-equal, sanity texts identical",
              sp["pass"] and sp["hash_equal_to_other_core"] and sp["spliced_blocks"] == [1, 2, 3]
              and sp["texts_identical"] == sp["n"], f"{brief}")
    check("HlowT@4 (Supa + all Titan blocks) == base:sparse texts exactly", gen_texts(d, "HlowT@4") == ts)
    check("HupT@4 (Titan + all Supa blocks) == base:supa texts exactly", gen_texts(d, "HupT@4") == tp)
    for n, cp in (("HlowT@2", "T2S_all@2"), ("HupT@2", "S2T_all@2")):
        a_, b_ = rf[n]["first_token"], rf[cp]["first_token"]
        check(f"{n} first token == {cp} first token on every item (same prompt computation)",
              all(a_[i] == b_[i] for i in ids) and rf[n]["ft_equals_counterpart"]["fraction"] == 1.0
              and rf["_derived"]["first_token_identity"][f"{n} vs {cp}"] == 1.0)
    check("HlowT@2 and HupT@2 differ from both baselines somewhere (non-vacuous)",
          all(gen_texts(d, n) != ts and gen_texts(d, n) != tp for n in ("HlowT@2", "HupT@2")))
    nb = [n for n in names if not n.startswith("base:")]
    check("_derived: FR record for every patched/hybrid condition and family",
          sorted(rf["_derived"]["fr"]) == sorted(nb) and all(len(v) == 3 for v in rf["_derived"]["fr"].values())
          and sorted(rf["_derived"]["mis_reference"]) == ["T2S_dmis@1", "T2S_dmis@4", "T2S_mis@1", "T2S_mis@2", "T2S_mis@4"])
    # staged rows == an unpadded forward's hidden_states[L] (L = output of block L-1), L = 1, 2
    idx = json.loads((d / "donor_supa_index.json").read_text(encoding="utf-8"))
    tok = runtime.tokenizer()
    from rcwmt import generation
    worst = 0.0
    for i in ids[::5]:
        x = tok(generation.render(tok, byid[i]["messages"]), add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            hs = m(input_ids=torch.tensor([x]), output_hidden_states=True, use_cache=False).hidden_states
        off, n = idx["offset"][i]
        for L in (1, 2):
            a = np.load(d / f"donor_supa_L{L}.npy", mmap_mode="r")
            worst = max(worst, float(np.abs(np.array(a[off: off + n]) - hs[L][0].numpy()).max()))
            del a
    check("staged rows == unpadded forward hidden_states[L] (L=1,2)", worst < 1e-4, f"max|d| {worst:.2e}")
    return d


def t7_rerun(d):
    print("7 rerun of a finished directory", flush=True)
    before = (d / "patch_results.json").read_bytes()
    loads = run_main(args_for(d, "--keep-states"))
    check("nothing loaded, patch_results.json unchanged", loads == [] and (d / "patch_results.json").read_bytes() == before)
    dp = d.parent / "plan_only"
    loads = run_main(args_for(dp, "--plan"))
    check("--plan loads nothing and creates no output directory", loads == [] and not dp.exists())


def t8_sanity_abort(tmp):
    print("8 corrupted sanity staging aborts", flush=True)
    d = tmp / "fail"
    orig = P.Staging.rows

    def bad_rows(self, item_id, L):
        r = orig(self, item_id, L)
        return r + 1e-3 if self.prefix.startswith("sanity_") else r
    P.Staging.rows = bad_rows
    try:
        run_main(args_for(d))
        raised = False
    except P.SanityError as e:
        raised, msg = True, str(e)
    finally:
        P.Staging.rows = orig
    files = sorted(p.name for p in d.iterdir())
    check("SanityError raised; no results, no progress, no gen files", raised and files == ["patch_sanity.json"],
          f"{files}; {msg if raised else ''}")


def t9_cuda_wiring(tmp, d_fresh):
    print("9 --device cuda wiring (stubbed runtime, CPU tensors)", flush=True)
    d = tmp / "cuda_wiring"
    calls = {"lease": 0, "load_arm": [], "throttle": 0, "breathe": 0, "unload": 0}

    @contextmanager
    def lease(label, job):
        assert label == "rcwm-t1:patch"
        calls["lease"] += 1
        yield

    def load_arm(arm):
        calls["load_arm"].append(arm)
        m = build(arm)
        return m, {"arm": arm, "dtype": "float32", "param_fingerprint": runtime.param_fingerprint(m)}

    def throttle_list():
        calls["throttle"] += 1
        return transformers.LogitsProcessorList([])

    def breathe():
        calls["breathe"] += 1

    saved = {k: getattr(runtime, k) for k in ("gpu_lease", "load_arm", "throttle_list", "breathe")}
    runtime.gpu_lease, runtime.load_arm, runtime.throttle_list, runtime.breathe = lease, load_arm, throttle_list, breathe
    P._TENSOR_DEVICE = "cpu"
    try:
        P.main(args_for(d, device="cuda"))
    finally:
        for k, v in saved.items():
            setattr(runtime, k, v)
        P._TENSOR_DEVICE = None
    r, rf = load_res(d), load_res(d_fresh)
    names = [k for k in rf if not k.startswith("_")]
    check("one lease for the whole job; load_arm sparse, supa, sparse",
          calls["lease"] == 1 and calls["load_arm"] == ["sparse", "supa", "sparse"], f"{calls}")
    check("throttle_list and breathe once per generate call", calls["throttle"] == calls["breathe"] > 0, f"{calls}")
    check("cuda-path results identical to the cpu run",
          all(gen_texts(d, n) == gen_texts(d_fresh, n) and r[n]["first_token"] == rf[n]["first_token"] for n in names))
    check("CUDA never initialised in this process", not torch.cuda.is_initialized())


def t10_splice_gate(tmp):
    print("10 wrong splice source trips the hash gate", flush=True)
    d = tmp / "splice_gate"
    saved = P.weight_file
    P.weight_file = lambda arm: WFILES["supa"]          # Supa's file served for Titan
    msg = ""
    try:
        run_main(["--device", "cpu", "--layers", "2", "--conditions", "T2S_all", "--dmis-layers", "", "--hybrid-layers", "2",
                  "--batch", "4", "--sanity-per-family", "3", "--max-new-tokens", "6", "--items-from", "",
                  "--threads", "2", "--out-dir", str(d)])
        raised = False
    except P.SanityError as e:
        raised, msg = True, str(e)
    finally:
        P.weight_file = saved
    prog = json.loads((d / "patch_progress.json").read_text(encoding="utf-8"))
    san = json.loads((d / "patch_sanity.json").read_text(encoding="utf-8"))
    check("SanityError from the hash gate; no hybrid result; no patch_results.json; splice record failed",
          raised and "hash" in msg and not (d / "patch_results.json").exists()
          and not any(k.startswith("H") for k in prog) and "T2S_all@2" in prog and san["splice_supa"]["pass"] is False,
          msg)


def t11_derived_unit():
    print("11 derived() on synthetic results", flush=True)
    fam = env.FAMILIES[0]
    items = [{"item_id": f"x{p}{k}", "family": fam, "parent_id": f"p{p}", "gold": "1", "gold_prefix_states": ["0", "1"]}
             for p in range(4) for k in range(4)]
    ids = [it["item_id"] for it in items]
    byid = {it["item_id"]: it for it in items}
    donors = {i: byid[f"x{(int(i[1]) + 1) % 4}{i[2]}"] for i in ids}
    mis_rec = {i: {"exact_len": i[2] != "3"} for i in ids}          # 4 non-exact pairs
    half = {i: int(i[2]) % 2 == 0 for i in ids}
    res = {"base:sparse": {"correct": {i: True for i in ids}, "first_token": {i: 1 for i in ids}},
           "base:supa": {"correct": {i: False for i in ids}, "first_token": {i: 2 for i in ids}},
           "T2S_all@2": {"correct": half, "first_token": {i: 1 for i in ids}},
           "T2S_last@2": {"correct": {i: i[1] == "0" for i in ids}, "first_token": {i: 1 for i in ids}},
           "HlowT@2": {"correct": {i: True for i in ids}, "first_token": {i: 1 for i in ids}},
           "HupT@2": {"correct": {i: False for i in ids}, "first_token": {i: 2 for i in ids}},
           "T2S_mis@2": {"correct": {i: i[2] == "3" for i in ids}, "first_token": {i: 1 for i in ids},
                         "donor_correct": {i: i[2] == "0" for i in ids}}}
    conds = list(res)
    dv = P.derived(items, res, conds, donors, mis_rec, draws=300)
    fr = dv["fr"]
    ok = (fr["T2S_all@2"][fam]["FR"] == 0.5 and fr["T2S_all@2"][fam]["label"] == "MID"
          and fr["HlowT@2"][fam]["label"] == "HI" and fr["HlowT@2"][fam]["FR_ci95"] == [1.0, 1.0]
          and fr["HupT@2"][fam]["label"] == "LO" and fr["T2S_mis@2"][fam]["n"] == 12 and fr["T2S_mis@2"][fam]["acc"] == 0.0)
    ci = fr["T2S_all@2"][fam]["FR_ci95"]
    ok &= ci[0] <= 0.5 <= ci[1]
    # one parent of four fully correct: FR 0.25; parent-level resampling gives an upper bound >= 0.5
    # (item-level resampling of 16 items would give about 0.47)
    cl = fr["T2S_last@2"][fam]
    ok &= cl["FR"] == 0.25 and cl["FR_ci95"][0] == 0.0 and cl["FR_ci95"][1] >= 0.5
    mr = dv["mis_reference"]["T2S_mis@2"][fam]
    ok &= (mr["n"] == 12 and mr["donor_match_changed"] == round(4 / 12, 4)
           and mr["paired_T2S_all_on_donors"] == round(8 / 12, 4) and mr["paired_base_sparse_on_donors"] == 1.0)
    ok &= dv["first_token_identity"] == {"HlowT@2 vs T2S_all@2": 1.0}
    check("FR / label / bootstrap CI / exact-length restriction / paired donor references", ok,
          f"{json.dumps(fr)} {json.dumps(dv['mis_reference'])}")


FAKE: list[dict] = []


def main():
    keep = "--keep" in sys.argv
    t0 = time.time()
    real_pool = P.load_pool("bank_pilot")
    FAKE.extend(fake_pool(real_pool))
    t0_selection(real_pool)
    del real_pool
    P.load_pool = lambda bank: list(FAKE)
    P.load_cpu = fake_load_cpu
    P.hidden_size_hint = lambda: HID
    P.weight_file = lambda arm: WFILES[arm]
    P.EXPECTED_FP = {arm: runtime.param_fingerprint(build(arm))[:12] for arm in ("supa", "sparse")}
    per_fam = {f[:6]: sum(it["family"] == f for it in FAKE) for f in env.FAMILIES}
    print(f"tiny models: fingerprints {P.EXPECTED_FP}; fake pool {len(FAKE)} items {per_fam}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="rcwm_patch_tiny_"))
    try:
        (tmp / "weights").mkdir()
        write_weight_files(tmp / "weights")
        t1_alignment(tmp)
        t2_hook_guards()
        d_resume = t3_t4_crash_resume(tmp)
        d_fresh = t5_t6_fresh(tmp, d_resume)
        t7_rerun(d_fresh)
        t8_sanity_abort(tmp)
        t9_cuda_wiring(tmp, d_fresh)
        t10_splice_gate(tmp)
        t11_derived_unit()
    finally:
        if keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    pm = psutil.Process().memory_info()
    peak = getattr(pm, "peak_wset", pm.rss) / 2 ** 20
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\nALL {n_ok}/{len(RESULTS)} CHECKS PASSED in {time.time() - t0:.0f}s; peak RSS {peak:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
