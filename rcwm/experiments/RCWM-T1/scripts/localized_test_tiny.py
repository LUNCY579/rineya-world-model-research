"""End-to-end test of scripts/run_localized.py and scripts/localized_analysis.py on CPU with TWO tiny random
Qwen3 models (no real weights, no GPU, no lease).

Stand-ins (28 blocks, as the real cores, so the arms' block ranges 8-15 / 16-27 and capture.LAYERS apply
unchanged; hidden 64, 4 heads, 2 kv heads, head_dim 16, real vocab 151936, tied head): "supa" drawn with seed
1001 and rounded to BF16-representable values; "sparse" = the same model plus an FP32 delta that is 25% dense
in blocks 8..27 and exactly zero in the embeddings, the final norm and blocks 0..7 (seed 2002), like D-21.
Weight files for the splices: supa BF16, sparse FP32 safetensors without a head (tied), like the cores. The
reference texts and recorded hashes come from a real run_patch.py run on the same tiny pair (HlowT@16,
HupT@16 built on the other host, bases), so the reproduction check compares two independent code paths.
Subsets: 24 off-task prompts, 48 chains (2 TEST parents per family), 6 one-frame items per family, 6 new
tokens. The loader, the weight files, the pools and the expected fingerprints are monkeypatched; everything
else is the production code path.

Checks
  0  file_group_hashes (header parse + chunked buffered reads, BF16 -> FP32 by bit shift) == run_patch.group_hashes
  1  run_patch reference run on the tiny pair (patch_results.json + gen files)
  2  crash after 5 stages: no DONE.json, finished stages recorded, the cut stage absent
  3  resume: exactly the pending stages run, one Supa load, DONE.json written, reproduction record reused
  4  builds: every group of every arm hash-equals the intended core (independent hashes of the built cores);
     loc8_15 / up16_27 / sparse differ from Supa in exactly blocks 8-15 / 16-27 / 8-27; spliced blocks as
     planned; fingerprints == those of models assembled independently from the two cores' state dicts
     (sparse == the natively built Titan stand-in); run_patch texts reproduced exactly for all four arms
  5  outputs: off-task states bit-identical to Supa exactly where blocks are Supa's (loc8_15/sparse
     layers <= 8, up16_27 <= 16; loc8_15 == sparse at <= 16) and different above; file formats
  6  fresh uninterrupted run == resumed run file by file; rerun of a finished directory loads nothing and
     leaves DONE.json byte-identical; --plan loads and writes nothing
  7  --device cuda wiring (stubbed runtime, CPU tensors): one lease 'rcwm-t1:localized', load_arm('supa')
     once, throttle/breathe per generate, outputs identical to the cpu run, CUDA never initialised
  8  gates: Titan's file replaced by Supa's -> SanityError before any load (with run_patch hashes: G1; without:
     G2); a corrupted splice -> SanityError, no output for that arm, no DONE.json; a doctored reference text ->
     WARNING recorded (MISMATCH, 11/12), run continues
  9  analysis (subprocess, no torch) on the outputs: KL 0 / top-1 1 / greedy 1 for supa-vs-supa, KL > 0 both
     directions for the other arms, an independent numpy KL equals the reported mean, ratios with CIs, state
     build check, per-parent k=4 regret == decisions.g_only, arm - sparse == decide.paired_bootstrap;
     denominator diagnostics (top-1 flips, greedy disagreements, prefix divergence) == independent counts,
     flags follow their rules; the D-23 verdict == the rule applied independently to the reported numbers
 10  analysis on synthetic chains: oracle chains -> regret 0, one-step and k=4 accuracy 1; copy chains ->
     regret == the copy-chain floor; end to end: loc8_15's log-probs := Supa's -> "localisation reduces
     interference", := sparse's -> "no reduction"; a stage file edited after the run -> analysis refuses
 11  D-23 gate (--wait-for): a file appearing later releases the wait; a missing file with --wait-max-h ->
     exit 4 before any load or write, both through main() and at script start (where torch is not imported)
 12  D-23 rule, prefix divergence and denominator flags as pure functions (subprocess, no torch): boundary
     cases (upper CI == 0.8, lower CI == 0.9, difference == 0.05, NaN CI, missing family)

  python scripts/localized_test_tiny.py [--keep]
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
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import run_patch as P  # noqa: E402
import run_localized as RL  # noqa: E402
sys.argv = _argv
from rcwmt import capture, data, decide, decisions, env, generation, offtask, runtime  # noqa: E402
import transformers  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models

N_LAYERS, HID = 28, 64
SEEDS = {"supa": 1001, "sparse": 2002}
CFG = dict(vocab_size=151936, hidden_size=HID, intermediate_size=128, num_hidden_layers=N_LAYERS,
           num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
           rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True)
N_PROMPTS, N_PARENTS, MAX_NEW = 24, 2, 6
WFILES: dict[str, Path] = {}
LOADS: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []
FAKE: list[dict] = []
KEEP: set[str] = set()
ARMS = RL.ARM_ORDER
EXPECTED_DIFF = {"supa": [], "loc8_15": [f"block{i}" for i in range(8, 16)],
                 "sparse": [f"block{i}" for i in range(8, 28)], "up16_27": [f"block{i}" for i in range(16, 28)]}
OUT_FILES =("offtask_logprobs.npy", "offtask_last.npy", "offtask_meta.json", "offtask_gen.jsonl", "gen_chain_one.jsonl")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    assert ok, f"{name}: {detail}"


class Crash(Exception):
    pass


# ------------------------------------------------------------------------------------- tiny models
@torch.no_grad()
def _draw_block(blk, g):
    for name, p in blk.named_parameters():
        if p.dim() == 2:                              # linear weights; norms stay at 1
            p.copy_(torch.randn(p.shape, generator=g) * 0.15)


@torch.no_grad()
def build(arm: str):
    """'supa' (BF16-exact) or 'sparse' (= supa + FP32 delta, 25% dense in blocks 8..27, zero elsewhere)."""
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
    if arm == "sparse":
        g2 = torch.Generator().manual_seed(SEEDS["sparse"])
        for blk in m.model.layers[8:]:
            for name, p in blk.named_parameters():
                if p.dim() == 2:
                    mask = torch.rand(p.shape, generator=g2) < 0.25
                    p.add_(torch.randn(p.shape, generator=g2) * 0.05 * mask)
    return m.eval()


@torch.no_grad()
def assemble(titan_blocks):
    """Independent arm construction: Supa's state dict with Titan's tensors for the given blocks."""
    m, t = build("supa"), build("sparse")
    sd_t = t.state_dict()
    for name, p in m.named_parameters():
        if name.startswith("model.layers.") and int(name.split(".")[2]) in titan_blocks:
            p.copy_(sd_t[name])
    del t
    return m


def write_weight_files(d: Path):
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


def rl_args(d, patch_dir, *extra, device="cpu"):
    return ["--device", device, "--out-dir", str(d), "--patch-dir", str(patch_dir), "--offtask-max-new", str(MAX_NEW),
            "--threads", "2", *extra]


def run_rl(argv):
    LOADS.clear()
    RL.main(argv)
    return list(LOADS)


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stage_names(d, arm):
    st = RL._read_json(Path(d) / arm / "STAGES.json") or {}
    return sorted(st.get("stages", {}))


# ------------------------------------------------------------------------------------------- tests
def t0_file_hashes():
    print("0 file-derived group hashes", flush=True)
    for arm in ("supa", "sparse"):
        F, G = RL.file_group_hashes(arm), P.group_hashes(build(arm))
        check(f"{arm}: file_group_hashes (buffered reads of {RL.HASH_CHUNK} elements, BF16 shift) == run_patch.group_hashes",
              F == G and len(F) == N_LAYERS + 2, f"{len(F)} groups")
    F = {a: RL.file_group_hashes(a) for a in ("supa", "sparse")}
    diff = RL.group_sort(g for g in F["supa"] if F["supa"][g] != F["sparse"][g])
    check("tiny cores differ in exactly blocks 8..27 (D-21 premise holds for the stand-ins)",
          diff == [f"block{i}" for i in range(8, 28)], f"{diff[:3]}..")


def t1_patch_reference(tmp):
    print("1 run_patch reference run on the tiny pair", flush=True)
    d = tmp / "patch"
    LOADS.clear()
    P.main(["--device", "cpu", "--layers", "16", "--conditions", "T2S_all", "--dmis-layers", "", "--hybrid-layers", "16",
            "--batch", "5", "--sanity-per-family", "2", "--max-new-tokens", str(MAX_NEW), "--items-from", "",
            "--threads", "2", "--out-dir", str(d)])
    res = rj(d / "patch_results.json")
    files = [f for f in RL.REF_FILES.values() if (d / f).exists()]
    check("run_patch wrote patch_results.json and the four reference files (bases, HlowT@16, HupT@16)",
          len(files) == 4 and "hash_sparse" in res["_meta"] and res["_meta"]["batch"] == 5, f"loads {LOADS}")
    ts, tp = rj(d / "gen_base_sparse.json"), rj(d / "gen_base_supa.json")
    check("tiny Supa and Titan stand-ins differ in their texts (non-vacuous reproduction checks)",
          sum(ts[i] != tp[i] for i in ts) >= len(ts) // 3, f"{sum(ts[i] != tp[i] for i in ts)}/{len(ts)} differ")
    return d


def t2_t3_crash_resume(tmp, patch_dir):
    print("2 crash after 5 stages", flush=True)
    d = tmp / "resume"
    done = []

    def crash(arm, s):
        done.append((arm, s))
        if len(done) == 5:
            raise Crash(f"{arm}:{s}")
    RL._on_stage_done = crash
    try:
        run_rl(rl_args(d, patch_dir))
        crashed = False
    except Crash:
        crashed = True
    finally:
        RL._on_stage_done = None
    check("crash propagated; no DONE.json", crashed and not (d / "DONE.json").exists(), f"{done}")
    check("supa: all three stages recorded; loc8_15: offtask + offtask_gen recorded, chain_one absent",
          stage_names(d, "supa") == sorted(RL.STAGES) and stage_names(d, "loc8_15") == ["offtask", "offtask_gen"]
          and not (d / "loc8_15" / "gen_chain_one.jsonl").exists() and not (d / "sparse").exists(),
          f"supa {stage_names(d, 'supa')}, loc8_15 {stage_names(d, 'loc8_15')}")
    check("torch.Tensor.cuda and the pacing hooks restored after the crash",
          "cuda" not in torch.Tensor.__dict__ and runtime.breathe.__module__ == "rcwmt.runtime"
          and generation.throttle_list.__module__ == "rcwmt.runtime")
    repro_before = rj(d / "loc8_15" / "ARM.json")["repro"]

    print("3 resume", flush=True)
    ran = []
    RL._on_stage_done = lambda arm, s: ran.append((arm, s))
    try:
        loads = run_rl(rl_args(d, patch_dir))
    finally:
        RL._on_stage_done = None
    want = [("loc8_15", "chain_one")] + [(a, s) for a in ("sparse", "up16_27") for s in RL.STAGES]
    check("resume ran exactly the pending stages, in order", ran == want, f"{ran}")
    check("one model load (supa) for the whole job", loads == ["supa"], f"{loads}")
    check("DONE.json written", (d / "DONE.json").exists())
    check("loc8_15 reproduction record reused on resume (not regenerated)", rj(d / "loc8_15" / "ARM.json")["repro"] == repro_before)
    return d


def t4_builds(d, patch_dir):
    print("4 builds, hashes, fingerprints, reproduction of run_patch", flush=True)
    H = {a: P.group_hashes(build(a)) for a in ("supa", "sparse")}
    exp_spliced = {"supa": [], "loc8_15": list(range(8, 16)), "sparse": list(range(16, 28)), "up16_27": list(range(8, 16))}
    for arm in ARMS:
        r = rj(d / arm / "ARM.json")
        tb = set(RL.titan_blocks(arm, N_LAYERS))
        want = {g: H["sparse" if (RL.block_index(g) is not None and RL.block_index(g) in tb) else "supa"][g] for g in H["supa"]}
        check(f"{arm}: every group hash-equals the intended core (independent hashes); differs from Supa in "
              f"{EXPECTED_DIFF[arm][0] + '..' + EXPECTED_DIFF[arm][-1] if EXPECTED_DIFF[arm] else 'nothing'} only",
              r["group_hashes"] == want and r["differs_from_supa"] == EXPECTED_DIFF[arm], f"spliced {r['spliced_in_this_build']}")
        check(f"{arm}: spliced blocks in this build == {exp_spliced[arm] and [exp_spliced[arm][0], exp_spliced[arm][-1]]}",
              r["spliced_in_this_build"] == exp_spliced[arm])
        fp_ind = runtime.param_fingerprint(assemble(tb))
        check(f"{arm}: fingerprint == the model assembled independently from the two cores' state dicts",
              r["fingerprint"] == fp_ind, f"{r['fingerprint'][:12]}")
        rp = r["repro"]
        check(f"{arm}: run_patch {RL.REF_FILES[arm]} reproduced exactly (first 4/family and whole first batches)",
              rp["status"] == "exact" and rp["gate_equal"] == rp["gate_n"] == 12
              and rp["batch_items_equal"] == rp["batch_items_compared"] == 15, f"{rp['gate_equal']}/{rp['gate_n']}, "
              f"batch {rp['batch_items_equal']}/{rp['batch_items_compared']}")
    tbs = {a: rj(d / a / "ARM.json")["titan_blocks"] for a in ARMS}
    check("recorded Titan block ranges: supa none, loc8_15 [8, 15], sparse [8, 27], up16_27 [16, 27]",
          tbs == {"supa": [], "loc8_15": [8, 15], "sparse": [8, 27], "up16_27": [16, 27]}, f"{tbs}")
    fps = {a: rj(d / a / "ARM.json")["fingerprint"] for a in ARMS}
    check("sparse arm fingerprint == the natively built Titan stand-in; supa == Supa; all four distinct",
          fps["sparse"] == runtime.param_fingerprint(build("sparse")) and fps["supa"] == runtime.param_fingerprint(build("supa"))
          and len(set(fps.values())) == 4)
    done = rj(d / "DONE.json")
    check("DONE.json: four arms, gates G1-G5, run_patch recorded hashes cross-checked",
          sorted(done["arms"]) == sorted(ARMS) and done["gates_passed"] == ["G1", "G2", "G3", "G4", "G5"]
          and str(done["run"]["source_hashes"]["equal_to_run_patch_recorded"]).endswith("patch_results.json")
          and done["run"]["source_hashes"]["loaded_supa_equals_file"] is True)


def t5_outputs(d):
    print("5 outputs: states and formats", flush=True)
    Hs = {a: np.load(d / a / "offtask_last.npy") for a in ARMS}
    layers = list(capture.LAYERS)
    eq = lambda a, b, L: bool(np.array_equal(Hs[a][:, layers.index(L)], Hs[b][:, layers.index(L)]))  # noqa: E731
    for arm, lo in (("loc8_15", 8), ("sparse", 8), ("up16_27", 16)):
        same = [L for L in layers if eq(arm, "supa", L)]
        check(f"{arm}: answer-slot states bit-identical to Supa exactly at layers <= {lo}",
              same == [L for L in layers if L <= lo], f"identical at {same}")
    same = [L for L in layers if eq("loc8_15", "sparse", L)]
    check("loc8_15 states bit-identical to sparse exactly at layers <= 16", same == [L for L in layers if L <= 16], f"{same}")
    ids = [r["id"] for r in offtask.load_prompts()]
    ok = True
    for arm in ARMS:
        m = rj(d / arm / "offtask_meta.json")
        g = [json.loads(l) for l in (d / arm / "offtask_gen.jsonl").read_text(encoding="utf-8").splitlines()]
        ch = [json.loads(l) for l in (d / arm / "gen_chain_one.jsonl").read_text(encoding="utf-8").splitlines()]
        lp = np.load(d / arm / "offtask_logprobs.npy")
        ok &= m["ids"] == ids and [r["id"] for r in g] == ids and all(isinstance(r["text"], str) for r in g)
        ok &= lp.shape == (N_PROMPTS, CFG["vocab_size"]) and lp.dtype == np.float16 and float(lp.min()) >= -60.0
        ok &= [r["cand_id"] for r in ch] == [c["cand_id"] for c in RL.load_chains(BANK)]
        ok &= all(len(r["states"]) == 5 and len(r["texts"]) >= 1 for r in ch)
    check(f"formats: {N_PROMPTS} prompts in order (meta, gen), fp16 log-probs >= -60, {len(KEEP)} chains {{cand_id, states, texts}}", ok)


def t6_fresh_rerun_plan(tmp, d_resume, patch_dir):
    print("6 fresh == resumed; rerun; --plan", flush=True)
    d = tmp / "fresh"
    gate = tmp / "gate_present.json"
    gate.write_text("{}\n", encoding="utf-8")
    loads = run_rl(rl_args(d, patch_dir, "--wait-for", str(gate), "--wait-for", "data/bank_pilot.json"))
    same = all((d / a / f).read_bytes() == (d_resume / a / f).read_bytes() for a in ARMS for f in OUT_FILES)
    check("fresh run (gate files present, one relative to RCWM-T1): one load; every output file byte-identical "
          "to the resumed run", loads == ["supa"] and same, f"{loads}")
    before = (d / "DONE.json").read_bytes()
    loads = run_rl(rl_args(d, patch_dir))
    check("rerun of a finished directory loads nothing; DONE.json unchanged", loads == [] and (d / "DONE.json").read_bytes() == before)
    dp = tmp / "plan_only"
    loads = run_rl(rl_args(dp, patch_dir, "--plan"))
    check("--plan loads nothing and creates no output directory", loads == [] and not dp.exists())
    return d


def t7_cuda_wiring(tmp, d_fresh, patch_dir):
    print("7 --device cuda wiring (stubbed runtime, CPU tensors)", flush=True)
    d = tmp / "cuda_wiring"
    calls = {"lease": [], "load_arm": [], "throttle": 0, "breathe": 0}

    @contextmanager
    def lease(label, job):
        calls["lease"].append(label)
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

    saved = {(m, k): getattr(m, k) for m, ks in ((runtime, ("gpu_lease", "load_arm", "throttle_list", "breathe")),
                                                  (generation, ("throttle_list", "breathe"))) for k in ks}
    runtime.gpu_lease, runtime.load_arm = lease, load_arm
    for m in (runtime, generation):
        m.throttle_list, m.breathe = throttle_list, breathe
    torch.Tensor.cuda = lambda self, *a, **k: self
    P._TENSOR_DEVICE = "cpu"
    try:
        RL.main(rl_args(d, patch_dir, device="cuda"))
    finally:
        for (m, k), v in saved.items():
            setattr(m, k, v)
        del torch.Tensor.cuda
        P._TENSOR_DEVICE = None
    n_off_batches = len(ARMS) * -(-N_PROMPTS // 8)
    check("one lease 'rcwm-t1:localized'; load_arm('supa') once",
          calls["lease"] == ["rcwm-t1:localized"] and calls["load_arm"] == ["supa"], f"{calls}")
    check("throttle on every generate, breathe after every generate and off-task forward batch",
          calls["throttle"] > 0 and calls["breathe"] == calls["throttle"] + n_off_batches, f"{calls}")
    same = all((d / a / f).read_bytes() == (d_fresh / a / f).read_bytes() for a in ARMS for f in OUT_FILES)
    check("cuda-path outputs byte-identical to the cpu run", same and (d / "DONE.json").exists())
    check("CUDA never initialised in this process", not torch.cuda.is_initialized())


def t8_gates(tmp, patch_dir):
    print("8 gates", flush=True)
    saved_wf = P.weight_file
    for label, pdir, gate in (("with run_patch hashes", patch_dir, "G1"), ("without run_patch hashes", tmp / "empty_patch", "G2")):
        pdir.mkdir(exist_ok=True)
        d = tmp / f"wrongfile_{gate}"
        P.weight_file = lambda arm: WFILES["supa"]          # Supa's file served for Titan
        msg = ""
        try:
            run_rl(rl_args(d, pdir))
            raised = False
        except P.SanityError as e:
            raised, msg = True, str(e)
        finally:
            P.weight_file = saved_wf
        check(f"Titan's file replaced by Supa's ({label}) -> {gate} SanityError before any model load, no arm output",
              raised and msg.startswith(gate) and LOADS == [] and not any((d / a).exists() for a in ARMS), msg[:120])
    # a corrupted splice (one element off after the copy) trips run_patch's hash gate
    d = tmp / "bad_splice"
    orig = P.splice_blocks

    def bad_splice(model, src_arm, blocks, dev):
        n = orig(model, src_arm, blocks, dev)
        if src_arm == "sparse":
            with torch.no_grad():
                model.model.layers[blocks[0]].mlp.down_proj.weight.view(-1)[0] += 1e-3
        return n
    P.splice_blocks = bad_splice
    msg = ""
    try:
        run_rl(rl_args(d, patch_dir))
        raised = False
    except P.SanityError as e:
        raised, msg = True, str(e)
    finally:
        P.splice_blocks = orig
    check("corrupted splice -> SanityError (G3); supa complete, loc8_15 has no record or output, no DONE.json",
          raised and "hash-equal" in msg and stage_names(d, "supa") == sorted(RL.STAGES)
          and not (d / "loc8_15" / "ARM.json").exists() and not any((d / "loc8_15" / f).exists() for f in OUT_FILES)
          and not (d / "DONE.json").exists(), msg[:120])
    # a doctored reference text: recorded as MISMATCH, not fatal
    pbad = tmp / "patch_doctored"
    shutil.copytree(patch_dir, pbad)
    ref = rj(pbad / "gen_base_supa.json")
    first = [it["item_id"] for it in FAKE if it["family"] == env.FAMILIES[1]][0]
    ref[first] = ref[first] + " (doctored)"
    (pbad / "gen_base_supa.json").write_text(json.dumps(ref) + "\n", encoding="utf-8")
    d = tmp / "doctored"
    run_rl(rl_args(d, pbad, "--arms", "supa"))
    rp = rj(d / "supa" / "ARM.json")["repro"]
    check("doctored reference: repro MISMATCH 11/12 with that item named; stages still run; no DONE.json",
          rp["status"] == "MISMATCH" and rp["gate_equal"] == 11 and rp["gate_differ"] == [first]
          and stage_names(d, "supa") == sorted(RL.STAGES) and not (d / "DONE.json").exists(), f"{rp['gate_differ']}")


def run_analysis(run_dir, out_md, patch_dir, tmp):
    cmd = [sys.executable, str(HERE / "localized_analysis.py"), "--run-dir", str(run_dir), "--out", str(out_md),
           "--patch-dir", str(patch_dir), "--stored-root", str(tmp / "no_stored"), "--allow-subset", "--boot", "400"]
    envv = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, env=envv, timeout=900)


def g_only_ref(d, arm, fam):
    rows_all = data.candidate_table(BANK)
    rows = [r for r in rows_all if r["split"] != "test" or r["cand_id"] in KEEP]
    ch = {json.loads(l)["cand_id"]: json.loads(l) for l in (d / arm / "gen_chain_one.jsonl").read_text(encoding="utf-8").splitlines()}
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return decisions.g_only(rows, fam, 4, {"chain_one": ch}, modes=("chain_one",))["chain_one"]


def same_nan(a, b):
    a = np.array([np.nan if x is None else x for x in a], float)
    b = np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b)))


def _lsm(a):
    a = a.astype(np.float64)
    m = a.max(1, keepdims=True)
    return a - (m + np.log(np.exp(a - m).sum(1, keepdims=True)))


def t9_analysis(tmp, d, patch_dir):
    print("9 analysis on the outputs (subprocess)", flush=True)
    out = tmp / "analysis" / "LOCALIZED_RESULTS.md"
    r = run_analysis(d, out, patch_dir, tmp)
    check("localized_analysis.py exits 0 and writes .md and .json", r.returncode == 0 and out.exists() and out.with_suffix(".json").exists(),
          (r.stdout + r.stderr)[-600:])
    res = rj(out.with_suffix(".json"))
    K = res["offtask"]["kl"]
    check("supa vs supa: KL 0 both directions (mean and median), top-1 agreement 1, greedy agreement 1",
          K["supa"]["arm_supa"]["mean"] == 0 and K["supa"]["arm_supa"]["median"] == 0 and K["supa"]["supa_arm"]["mean"] == 0
          and K["supa"]["top1_agree"] == 1 and res["offtask"]["greedy"]["supa"]["agree"] == 1)
    check("loc8_15, sparse, up16_27 vs supa: KL > 0 in both directions",
          all(K[a]["arm_supa"]["mean"] > 0 and K[a]["supa_arm"]["mean"] > 0 for a in ("loc8_15", "sparse", "up16_27")),
          ", ".join(f"{a} {K[a]['arm_supa']['mean']:.3g}/{K[a]['supa_arm']['mean']:.3g}" for a in ("loc8_15", "sparse", "up16_27")))
    A, S = np.load(d / "sparse" / "offtask_logprobs.npy"), np.load(d / "supa" / "offtask_logprobs.npy")
    a, s = _lsm(A), _lsm(S)
    kl = np.maximum((np.exp(a) * (a - s)).sum(1), 0.0)
    check("independent numpy KL(sparse||supa) mean == reported", abs(kl.mean() - K["sparse"]["arm_supa"]["mean"]) < 1e-9,
          f"{kl.mean():.6g} vs {K['sparse']['arm_supa']['mean']:.6g}")
    rat = res["offtask"]["ratios"]
    ok = all(np.isfinite(rat[k][arm]["ratio"]) and rat[k][arm]["ci95"][0] is not None
             for k in ("kl_arm_supa", "kl_supa_arm") for arm in ("loc8_15", "up16_27"))
    ok &= abs(rat["kl_arm_supa"]["loc8_15"]["ratio"] - K["loc8_15"]["arm_supa"]["mean"] / K["sparse"]["arm_supa"]["mean"]) < 1e-12
    check("drift ratios vs sparse with bootstrap CIs (KL both directions); ratio == mean KL(loc)/mean KL(sparse)", ok,
          f"KL ratio loc {rat['kl_arm_supa']['loc8_15']['ratio']:.3f} {rat['kl_arm_supa']['loc8_15']['ci95']}, "
          f"up {rat['kl_arm_supa']['up16_27']['ratio']:.3f}")
    bc = res["offtask"]["state_build_check"]
    check("analysis state build check: identical where blocks are Supa's (and loc8_15 == sparse at <= 16)",
          all(bc[k]["identical_where_expected"] for k in bc), f"{ {k: bc[k]['identical_layers'] for k in bc} }")
    ok, n = True, 0
    for fam in env.FAMILIES:
        f = env.SHORT[fam]
        pp = res["skill"]["per_parent_nregret"][f]
        for arm in ARMS:
            g = g_only_ref(d, arm, fam)
            ok &= pp["pids"] == g["pids"] and same_nan(pp[arm], g["nregret"])
            n += len(g["pids"])
        fa = res["skill"]["families"][f]["arms"]
        gs = np.asarray(g_only_ref(d, "sparse", fam)["nregret"], float)
        for arm in ("loc8_15", "up16_27"):
            pb = decide.paired_bootstrap(np.asarray(g_only_ref(d, arm, fam)["nregret"], float), gs)
            ok &= abs(fa[arm]["nregret_minus_sparse"]["mean_diff"] - pb["mean_diff"]) < 1e-12
    check("per-parent k=4 regret == decisions.g_only for every arm and family; arm - sparse == decide.paired_bootstrap",
          ok, f"{n} parent values")
    # denominator diagnostics and the graded greedy measure, recomputed independently
    lp = {a: np.load(d / a / "offtask_logprobs.npy") for a in ("supa", "sparse")}
    flips = int((lp["sparse"].argmax(1) != lp["supa"].argmax(1)).sum())
    gens = {a: [json.loads(l)["text"] for l in (d / a / "offtask_gen.jsonl").read_text(encoding="utf-8").splitlines()] for a in ARMS}

    def pdiv(x, y):
        if x == y:
            return 0.0
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        return 1.0 - k / max(len(x), len(y))
    ndis = sum(x != y for x, y in zip(gens["sparse"], gens["supa"]))
    pd_sp = float(np.mean([pdiv(x, y) for x, y in zip(gens["sparse"], gens["supa"])]))
    dn, G = res["offtask"]["denominator"], res["offtask"]["greedy"]
    klsp = np.sort(np.asarray(K["sparse"]["kl_arm_supa_per_prompt"]))[::-1]
    share = float(klsp[:int(np.ceil(0.1 * len(klsp)))].sum() / klsp.sum())
    check("denominator: top-1 flips, greedy disagreements, KL top-10% share == independent counts; prefix divergence "
          "0 for supa, == independent mean for sparse",
          dn["sparse_top1_flips"] == flips and dn["sparse_greedy_disagreements"] == ndis
          and abs(dn["sparse_kl_top10pct_share"]["kl_arm_supa"] - share) < 1e-12
          and G["supa"]["prefix_div_mean"] == 0 and abs(G["sparse"]["prefix_div_mean"] - pd_sp) < 1e-12,
          f"flips {flips}/{N_PROMPTS}, greedy differs {ndis}/{N_PROMPTS}, prefix div {pd_sp:.3f}, KL top-10% share {share:.2f}")
    want_g = ndis / N_PROMPTS > 0.9 or ndis < 10
    lv = {k: [x["level"] for x in rat[k]["flags"]] for k in ("kl_arm_supa", "kl_supa_arm", "greedy_dis", "greedy_prefix_div", "top1_dis")}
    check("flags follow their rules: greedy exact ratio UNINTERPRETABLE iff ceiling/floor, top-1 iff < 10 flips, "
          "prefix divergence iff < 10 disagreements, KL ratios never UNINTERPRETABLE (CAUTION only)",
          (not rat["greedy_dis"]["interpretable"]) == want_g and (not rat["top1_dis"]["interpretable"]) == (flips < 10)
          and (not rat["greedy_prefix_div"]["interpretable"]) == (ndis < 10)
          and rat["kl_arm_supa"]["interpretable"] and rat["kl_supa_arm"]["interpretable"]
          and ("CAUTION" in lv["kl_arm_supa"]) == (flips < 10 or share > 0.5), f"{lv}")
    # D-23 verdict, recomputed from the reported numbers
    V = res["d23_verdict"]
    lo, hi = rat["kl_arm_supa"]["loc8_15"]["ci95"]
    diffs = [res["skill"]["families"][f]["arms"]["loc8_15"]["nregret_minus_sparse"]["mean_diff"] for f in ("L", "P", "R")]
    reg_ok = all(x is not None and x <= 0.05 for x in diffs)
    want = "localisation reduces interference" if (hi < 0.8 and reg_ok) else ("no reduction" if lo > 0.9 else "inconclusive")
    lo2, hi2 = rat["kl_supa_arm"]["loc8_15"]["ci95"]
    want2 = "localisation reduces interference" if (hi2 < 0.8 and reg_ok) else ("no reduction" if lo2 > 0.9 else "inconclusive")
    md = out.read_text(encoding="utf-8")
    decisions_md = (RL.ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    check("D-23 verdict == the rule applied independently to the reported CI and regret differences (both KL "
          "directions); the rule in JSON and MD is D-23's text verbatim; verdict line in the MD",
          V["verdict"] == want and V["secondary"]["verdict"] == want2 and V["primary"]["kl_ratio_ci95"] == [lo, hi]
          and V["rule"] in decisions_md and V["rule"] in md and f"**Verdict (primary, KL(arm‖supa)): {want}.**" in md,
          f"{want} (KL ratio CI [{lo:.3f}, {hi:.3f}], regret diffs {[round(x, 3) for x in diffs]}); secondary {want2}")
    check("report has the D-23, key-ratio, regret, one-step and state-shift sections; KL labelled in nats",
          all(s in md for s in ("## Pre-stated reading (DECISIONS D-23)", "## Key ratios", "## Skill: k=4",
                                "## Skill: one-step", "### Answer-slot state shift", "KL in nats", "denominator flag")))


def _rewrite_stage_hash(d, arm, fname, stage="chain_one"):
    st = rj(d / arm / "STAGES.json")
    st["stages"][stage]["sha256"][fname] = RL.sha256_file(d / arm / fname)
    (d / arm / "STAGES.json").write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")


def t10_synthetic(tmp, d, patch_dir):
    print("10 analysis on synthetic chains", flush=True)
    ds = tmp / "synthetic"
    shutil.copytree(d, ds)
    cands = {c["cand_id"]: c for c in RL.load_chains(BANK)}
    for arm, kind in (("loc8_15", "oracle"), ("up16_27", "copy")):
        with (ds / arm / "gen_chain_one.jsonl").open("w", encoding="utf-8") as fh:
            for cid, c in cands.items():
                st = env.rollout(c["family"], c["s0"], c["events"])
                states = [env.dumps(x) for x in st] if kind == "oracle" else [env.dumps(c["s0"])] * (len(c["events"]) + 1)
                fh.write(json.dumps({"cand_id": cid, "states": states, "texts": states[1:]}) + "\n")
        _rewrite_stage_hash(ds, arm, "gen_chain_one.jsonl")
    out = tmp / "analysis_syn" / "R.md"
    r = run_analysis(ds, out, patch_dir, tmp)
    check("analysis on synthetic chains exits 0", r.returncode == 0, (r.stdout + r.stderr)[-600:])
    res = rj(out.with_suffix(".json"))
    ok = True
    for f, fr in res["skill"]["families"].items():
        o, c = fr["arms"]["loc8_15"], fr["arms"]["up16_27"]
        pp = [x for x in res["skill"]["per_parent_nregret"][f]["loc8_15"] if x is not None]
        ok &= all(abs(x) < 1e-9 for x in pp) and o["onestep_all"]["acc"] == 1.0 and o["chain_acc"]["4"]["acc"] == 1.0
        ok &= abs(c["nregret"]["mean"] - fr["copy_floor"]) < 1e-12
    check("oracle chains: regret 0 for every parent, one-step and k=4 accuracy 1; copy chains: regret == copy floor", ok)
    # end-to-end D-23 verdicts: loc8_15 off-task log-probs replaced by Supa's (no drift) or sparse's (full drift)
    for label, src, want in (("Supa's", "supa", "localisation reduces interference"), ("sparse's", "sparse", "no reduction")):
        dv = tmp / f"synthetic_lp_{src}"
        shutil.copytree(ds, dv)
        shutil.copyfile(dv / src / "offtask_logprobs.npy", dv / "loc8_15" / "offtask_logprobs.npy")
        _rewrite_stage_hash(dv, "loc8_15", "offtask_logprobs.npy", "offtask")
        outv = tmp / f"analysis_lp_{src}" / "R.md"
        r = run_analysis(dv, outv, patch_dir, tmp)
        V = rj(outv.with_suffix(".json"))["d23_verdict"] if r.returncode == 0 else {}
        check(f"end to end: loc8_15 log-probs := {label}, oracle loc8_15 chains -> D-23 verdict '{want}' (both KL directions)",
              r.returncode == 0 and V["verdict"] == want and V["secondary"]["verdict"] == want,
              (f"KL ratio {V['primary']['kl_ratio']:.3f} {V['primary']['kl_ratio_ci95']}" if V else (r.stdout + r.stderr)[-400:]))
        shutil.rmtree(dv, ignore_errors=True)
    # a stage file changed after the run: the analysis refuses
    with (ds / "sparse" / "offtask_gen.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    r = run_analysis(ds, tmp / "analysis_bad" / "R.md", patch_dir, tmp)
    check("a stage file edited after the run -> the analysis stops, nothing written",
          r.returncode != 0 and "sha256 differs" in (r.stdout + r.stderr) and not (tmp / "analysis_bad" / "R.md").exists())


def t11_gate(tmp, patch_dir):
    print("11 D-23 gate (--wait-for)", flush=True)
    import threading
    late = tmp / "gate_late.json"
    timer = threading.Timer(1.0, lambda: late.write_text("{}\n", encoding="utf-8"))
    t0 = time.time()
    timer.start()
    ok = RL.wait_for_files([str(late), "data/bank_pilot.json"], poll_s=0.1, max_h=0.01)
    el = time.time() - t0
    timer.join()
    check("a gate file appearing after 1 s releases the wait (second path relative to RCWM-T1, present)",
          ok and 0.9 <= el < 10, f"{el:.1f}s")
    missing = tmp / "never_written.json"
    d = tmp / "gate_main"
    LOADS.clear()
    try:
        RL.main(rl_args(d, patch_dir, "--wait-for", str(missing), "--wait-poll", "0.05", "--wait-max-h", str(0.4 / 3600)))
        code = None
    except SystemExit as e:
        code = e.code
    check("main(): gate file missing past --wait-max-h -> exit 4, nothing loaded, no output directory",
          code == 4 and LOADS == [] and not d.exists(), f"exit {code}, loads {LOADS}")
    # script start: the gate runs before torch is imported. torch is blocked in this child (sys.modules entry
    # None), so if the gate failed to stop the script it would die at `import torch` instead of loading anything.
    d2 = tmp / "gate_script"
    prog = ("import sys, runpy\nsys.modules['torch'] = None\n"
            f"sys.argv = [{str(HERE / 'run_localized.py')!r}, '--device', 'cpu', '--out-dir', {str(d2)!r}, "
            f"'--wait-for', {str(missing)!r}, '--wait-poll', '0.05', '--wait-max-h', '0.0001']\n"
            "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\nexcept SystemExit as e:\n"
            "    print('EXIT', e.code, 'torch_imported', sys.modules.get('torch') is not None)\n")
    envv = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=envv, timeout=120)
    check("script start: gate file missing -> exit 4 before torch is imported; nothing written",
          r.returncode == 0 and "EXIT 4 torch_imported False" in r.stdout and "gate: still missing" in r.stdout
          and not d2.exists(), (r.stdout + r.stderr)[-300:])


PURE = r'''
import json, math, sys
sys.path.insert(0, sys.argv[1])
import localized_analysis as LA
F = ["L", "P", "R"]
z = {f: 0.0 for f in F}
R_, N_, I_ = LA.REDUCES, LA.NO_REDUCTION, LA.INCONCLUSIVE
cases = [("reduces", [0.5, 0.7], z, R_), ("regret P +0.06", [0.5, 0.7], {**z, "P": 0.06}, I_),
         ("regret exactly +0.05", [0.5, 0.7], {**z, "R": 0.05}, R_), ("regret negative", [0.1, 0.79], {f: -0.3 for f in F}, R_),
         ("upper CI == 0.8", [0.5, 0.8], z, I_), ("no reduction", [0.95, 1.1], z, N_),
         ("no reduction despite regret", [0.91, 1.2], {f: 0.2 for f in F}, N_), ("lower CI == 0.9", [0.9, 1.0], z, I_),
         ("between", [0.7, 0.85], z, I_), ("between 2", [0.85, 0.95], z, I_),
         ("family missing", [0.5, 0.7], {"L": 0.0, "P": 0.0}, I_), ("family None", [0.5, 0.7], {**z, "L": None}, I_),
         ("family NaN", [0.5, 0.7], {**z, "L": float("nan")}, I_), ("CI NaN", [float("nan"), float("nan")], z, I_),
         ("CI None", [None, None], z, I_)]
rule = [[n, LA.d23_rule(c, d, F)["verdict"], w] for n, c, d, w in cases]
pd = [[a, b, LA.prefix_div(a, b), w] for a, b, w in (("abc", "abc", 0.0), ("", "", 0.0), ("abc", "abd", 1 / 3),
                                                    ("ab", "abcd", 0.5), ("x", "y", 1.0), ("", "a", 1.0))]
ts = [[LA.top_share([10] + [0] * 9), 1.0], [LA.top_share([1] * 20), 0.1], [LA.top_share([3, 1]), 0.75]]
def den(flips, ndis, n, sh):
    return {"sparse_top1_flips": flips, "sparse_greedy_disagreements": ndis, "sparse_greedy_disagree_rate": ndis / n,
            "sparse_kl_top10pct_share": {"kl_arm_supa": sh, "kl_supa_arm": sh}}
keys = ("kl_arm_supa", "greedy_dis", "greedy_prefix_div", "top1_dis")
fl = [[n, {k: sorted(x["level"] for x in LA.ratio_flags(k, d)) for k in keys}, w] for n, d, w in (
    ("few flips, ceiling", den(5, 240, 248, 0.3),
     {"kl_arm_supa": ["CAUTION"], "greedy_dis": ["UNINTERPRETABLE"], "greedy_prefix_div": [], "top1_dis": ["UNINTERPRETABLE"]}),
    ("concentrated KL", den(50, 100, 248, 0.6),
     {"kl_arm_supa": ["CAUTION"], "greedy_dis": [], "greedy_prefix_div": [], "top1_dis": []}),
    ("greedy floor", den(50, 5, 248, 0.2),
     {"kl_arm_supa": [], "greedy_dis": ["UNINTERPRETABLE"], "greedy_prefix_div": ["UNINTERPRETABLE"], "top1_dis": []}),
    ("all fine", den(10, 10, 248, 0.5), {"kl_arm_supa": [], "greedy_dis": [], "greedy_prefix_div": [], "top1_dis": []}))]
print(json.dumps({"rule": rule, "pd": pd, "ts": ts, "fl": fl, "torch": "torch" in sys.modules}))
'''


def t12_pure(tmp):
    print("12 D-23 rule, prefix divergence, denominator flags (pure functions; subprocess, no torch)", flush=True)
    p = tmp / "pure_check.py"
    p.write_text(PURE, encoding="utf-8")
    envv = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(p), str(HERE)], capture_output=True, text=True, env=envv, timeout=300)
    check("pure-function subprocess ran without torch", r.returncode == 0 and '"torch": false' in r.stdout, (r.stdout + r.stderr)[-400:])
    o = json.loads(r.stdout.strip().splitlines()[-1])
    bad = [x for x in o["rule"] if x[1] != x[2]]
    check(f"d23_rule: {len(o['rule'])} cases incl. upper CI == 0.8, lower CI == 0.9, difference == 0.05, missing / NaN "
          f"family, NaN CI", not bad, f"{bad}")
    bad = [x for x in o["pd"] if abs(x[2] - x[3]) > 1e-12] + [x for x in o["ts"] if abs(x[0] - x[1]) > 1e-12]
    check("prefix_div and top_share on hand-computed cases", not bad, f"{bad}")
    bad = [x for x in o["fl"] if x[1] != x[2]]
    check("ratio_flags: ceiling, floor, few flips, concentrated KL, and the thresholds themselves (10 / 0.5) not flagged",
          not bad, f"{bad}")


BANK: dict = {}


def main():
    keep = "--keep" in sys.argv
    t0 = time.time()
    real_pool = P.load_pool("bank_pilot")
    FAKE.extend(fake_pool(real_pool))
    del real_pool
    BANK.update(json.loads((RL.ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8")))
    for fam in env.FAMILIES:
        ps = [p for p in BANK["parents"] if p["split"] == "test" and p["family"] == fam][:N_PARENTS]
        KEEP.update(c["cand_id"] for p in ps for c in p["candidates"])
    orig_chains, orig_prompts = RL.load_chains, offtask.load_prompts
    RL.load_chains = lambda bank: [c for c in orig_chains(bank) if c["cand_id"] in KEEP]
    offtask.load_prompts = lambda: orig_prompts()[:N_PROMPTS]
    generation.max_tokens = lambda fam: MAX_NEW
    P.load_pool = lambda bank: list(FAKE)
    P.load_cpu = fake_load_cpu
    P.hidden_size_hint = lambda: HID
    P.weight_file = lambda arm: WFILES[arm]
    P.EXPECTED_FP = {arm: runtime.param_fingerprint(build(arm))[:12] for arm in ("supa", "sparse")}
    RL.HASH_CHUNK = 1 << 14                       # 16384-element reads: the embedding needs 594, the last one partial
    print(f"tiny models: fingerprints {P.EXPECTED_FP}; {len(FAKE)} one-frame items, {len(KEEP)} chains, "
          f"{N_PROMPTS} off-task prompts, {MAX_NEW} new tokens", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="rcwm_localized_tiny_"))
    try:
        (tmp / "weights").mkdir()
        write_weight_files(tmp / "weights")
        t0_file_hashes()
        patch_dir = t1_patch_reference(tmp)
        d_resume = t2_t3_crash_resume(tmp, patch_dir)
        t4_builds(d_resume, patch_dir)
        t5_outputs(d_resume)
        d_fresh = t6_fresh_rerun_plan(tmp, d_resume, patch_dir)
        t7_cuda_wiring(tmp, d_fresh, patch_dir)
        t8_gates(tmp, patch_dir)
        t9_analysis(tmp, d_fresh, patch_dir)
        t10_synthetic(tmp, d_fresh, patch_dir)
        t11_gate(tmp, patch_dir)
        t12_pure(tmp)
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
