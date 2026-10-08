"""CPU test of scripts/l1_screen.py (PREREG-L1 runner) with TWO tiny random Qwen3 stand-ins. No real model is loaded,
no GPU is touched, no lease is taken, nothing is written outside a temporary directory.

Stand-ins (28 blocks like the real cores, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary 151936,
tied head): "supa" drawn with seed 1001, rounded to BF16-representable values and STORED BF16; "sparse" = the same
model plus an FP32 delta 25% dense in blocks 8..27 and exactly zero in the embeddings, the final norm and blocks
0..7 (D-21), STORED FP32. Tokenizer and generation sidecars are byte copies of Supa 1.7's (read-only). The R14
record (per-group hashes) is replaced by hashes computed with run_patch.group_hashes - the code R12/R14 used - on
models assembled independently in torch, so every identity gate compares two independent code paths. The real
instruments (SELECT-VIEWS-001, the arithmetic panel, the long-context suite, the fresh and the bridge off-task
prompts, R14's stored log-probs) are read in place; runs use a row subset.

Checks
  1  pure functions: decision rule truth table and boundaries (C2 per family: REP-001 half-span tolerance AND no
     detected loss; C6 per OOD stratum: max(3, ceil(5% n)) tolerance AND no detected loss, a missing stratum or a
     broken pairing fails); the registered margins derive from REP-001; fragile flags; Clopper-Pearson reproduces
     Titan-Sparse's published bounds; exact McNemar; parity allowance; scorer on B28's own source answers -> 47 / 41
  2  build: --dry-hash --record == the independent R14-style hashes; the written file holds exactly Supa's
     widened tensors outside blocks 8..15 and Titan's FP32 tensors inside (tensor by tensor); sha == record ==
     receipt; sidecars byte copies; a rebuild re-verifies and writes nothing; gates B1, B2, B4, B5, B7 fire
  3  run (dense = the programme's DenseReference, vendored): G1 refuses a wrong file before any load; G2 refuses
     a loaded model that differs from its file; the registered order, observed stage by stage: supa and sparse on
     the pre-gate stages, the parity gate (not enforced on a subset), then sparse's OOD stage (still loaded), supa's
     OOD stage (reloaded) and L8-15; every post-gate stage records the gate it followed; a crash after 7 stages
     resumes exactly the pending stages and reproduces the uninterrupted run file by file; a resume in a changed
     environment is refused; a live run lock refuses a second process, a dead one is reclaimed
  4  forward-path cross-check: Qwen3ForCausalLM (transformers) reproduces the DenseReference texts exactly and its
     first-token log-probs within float16 storage error
  5  --device cuda wiring with CPU tensors (stubbed lease): one lease with the registered label; the host gate
     before the lease and before each of the 4 loads; breathe() and the host guard once per row; outputs identical to the cpu
     run; a run that waited while another process finished loads nothing; the lease's helpers spawn NO process
     while waiting on a held lock (subprocess patched to raise); HostGuard exits resumably on low RAM, a stale
     watchdog and a long RAM-caused pause
  6  freeze: run/analyze refuse without it; it refuses after outputs exist (in --out-dir and anywhere under
     runs/); it records the environment; --no-freeze refuses real weights; a frozen run refuses another
     environment; analyze refuses a run made before the freeze or with --no-freeze and accepts a frozen run; it
     detects a changed prereg file and changed registered constants
  7  parity gate and analyze: the gate's legs pass on the published anchor predictions and on R14's own
     log-probs and fail on perturbations; a failed, enforced gate writes PARITY_FAIL.json, never loads L8-15, lets
     no model read the OOD bank, and analyze issues NON-COMPARABLE with no L8-15 endpoint (and refuses OOD outputs
     next to PARITY_FAIL.json); verdict NON-COMPARABLE on a subset run with exactly the
     expected reason; numbers recomputed independently; create-only; tamper refusal; the HF path is a named reason;
     an overflowing long-context row is skipped (never truncated) and labelled report-only, not a C0 reason; C6 per
     stratum from the tiny run's OOD outputs, recomputed independently; two C6 fixtures (tiny-model run directories
     whose OOD outputs are replaced by full 600-row fixtures): L8-15 == Titan-Sparse passes C6, L8-15 losing 10 of 100
     rows in one stratum fails C6 naming that stratum; the report-only C6 binding flags (a stratum can fail only if
     Titan-Sparse is right on >= 6 of 100 rows), checked exhaustively against the rule
  8  inputs: the fresh prompts are 57 x 8, disjoint from the 248 R14 prompts and from the HSKILL-002 pools; the
     bridge is 228; the OOD bank is 6 x 100, pinned, in Titan's one-step frame, with zero recorded overlaps; the
     programme's official-shape check accepts Supa's real config and refuses the tiny one
  9  subprocesses (2): `plan` imports no torch; `run --wait-for <missing> --wait-max-h` exits 4 creating nothing

  python -B scripts/l1_test_tiny.py [--keep]
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # NOT "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(2)

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import l1_screen as L  # noqa: E402
import run_patch as RP  # noqa: E402
sys.argv = _argv
from rcwmt import runtime  # noqa: E402
from rcwmt import lease_vendored as LV  # noqa: E402
import transformers  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models

KEEP = "--keep" in sys.argv
PY = sys.executable
N_LAYERS, HID = 28, 64
SEEDS = {"supa": 1001, "sparse": 2002}
CFG = dict(vocab_size=151936, hidden_size=HID, intermediate_size=128, num_hidden_layers=N_LAYERS,
           num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=40960,
           rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True)
CONFIG_JSON = dict(CFG, model_type="qwen3", architectures=["Qwen3ForCausalLM"], torch_dtype="bfloat16",
                   hidden_act="silu", attention_bias=False, attention_dropout=0.0)
SUBSET = {"acq": 1, "ret": 1, "lc": 2, "drift": 3, "ood": 1}
REAL_SUPA = Path(L.PATHS["supa_dir"])
RESULTS: list[tuple[str, bool, str]] = []
LOADS: list[str] = []
STAGE_FILES = [f for s in L.STAGES for f in L.STAGE_FILES[s]]


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    assert ok, f"{name}: {detail}"


def raises(fn, needle: str = "", exc=L.SanityError) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return (needle in str(e)), str(e)[:200]
    return False, "no exception"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Crash(Exception):
    pass


# ---------------------------------------------------------------------------------------------- tiny models
@torch.no_grad()
def build(arm: str):
    cfg = Qwen3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(SEEDS["supa"])
    m = Qwen3ForCausalLM(cfg)
    g = torch.Generator().manual_seed(SEEDS["supa"])
    m.model.embed_tokens.weight.copy_(torch.randn(m.model.embed_tokens.weight.shape, generator=g))
    for blk in m.model.layers:
        for _, p in blk.named_parameters():
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.15)
    for p in m.parameters():
        p.copy_(p.to(torch.bfloat16).float())          # Supa is stored BF16: make it BF16-exact
    if arm == "sparse":
        g2 = torch.Generator().manual_seed(SEEDS["sparse"])
        for blk in m.model.layers[8:]:
            for _, p in blk.named_parameters():
                if p.dim() == 2:
                    mask = torch.rand(p.shape, generator=g2) < 0.25
                    p.add_(torch.randn(p.shape, generator=g2) * 0.05 * mask)
    return m.eval()


@torch.no_grad()
def assemble(titan_blocks):
    """Independent L8-15: Supa's model with Titan's tensors copied in for the given blocks."""
    m, t = build("supa"), build("sparse")
    sd_t = t.state_dict()
    for name, p in m.named_parameters():
        if name.startswith("model.layers.") and int(name.split(".")[2]) in titan_blocks:
            p.copy_(sd_t[name])
    del t
    return m


def make_core(d: Path, arm: str) -> Path:
    d.mkdir(parents=True)
    m = build(arm)
    dt = torch.bfloat16 if arm == "supa" else torch.float32
    save_file({k: v.detach().to(dt).contiguous() for k, v in m.state_dict().items() if k.startswith("model.")},
              str(d / "model.safetensors"), metadata={"format": "pt"})
    del m
    for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copyfile(REAL_SUPA / f, d / f)                     # read-only source
    (d / "config.json").write_text(json.dumps(CONFIG_JSON, indent=1) + "\n", encoding="utf-8")
    return d


def dense_hook(raw: dict):
    B = L.vendor_forward()["budget"]
    return B.Config(vocab_size=raw["vocab_size"], hidden_size=raw["hidden_size"],
                    intermediate_size=raw["intermediate_size"], num_attention_heads=raw["num_attention_heads"],
                    num_key_value_heads=raw["num_key_value_heads"], head_dim=raw["head_dim"],
                    encoder_layers=raw["num_hidden_layers"], decoder_layers=0, architecture="plain",
                    rope_theta=float(raw["rope_theta"]), rms_norm_eps=float(raw["rms_norm_eps"]),
                    max_position_embeddings=raw["max_position_embeddings"])


# ---------------------------------------------------------------------------------------------- harness
_orig_dense_load, _orig_hf_load = L.DenseBackend.load, L.HFBackend.load
CORRUPT: dict = {}


def _guarded(orig, tag):
    def load(self, weight_file, cfg_path):
        if not str(Path(weight_file).resolve()).startswith(str(TMP.resolve())):
            raise AssertionError(f"the test tried to load a weight file outside its temp dir: {weight_file}")
        LOADS.append(f"{tag}:{Path(weight_file).parent.name}")
        m = orig(self, weight_file, cfg_path)
        if CORRUPT.get("dir") == Path(weight_file).parent.name:
            with torch.no_grad():
                p = next(iter(m.parameters()))
                p.view(-1)[0] += 1e-3
        return m
    return load


L.DenseBackend.load = _guarded(_orig_dense_load, "dense")
L.HFBackend.load = _guarded(_orig_hf_load, "hf")


def run(out: Path, *extra, device="cpu", forward="dense", freeze=False) -> list[str]:
    LOADS.clear()
    argv = ["run", "--device", device, "--forward", forward, "--out-dir", str(out), "--threads", "2"]
    if not freeze:
        argv.append("--no-freeze")
    L.main(argv + list(extra))
    return list(LOADS)


def analyze(out: Path, *extra) -> dict:
    L.main(["analyze", "--out-dir", str(out), "--skip-tokenizer-check", *extra])
    return L.read_json(out / "L1_RESULTS.json")


def stage_hashes(out: Path, arms=L.ARMS) -> dict:
    return {f"{a}/{f}": sha(out / a / f) for a in arms for f in STAGE_FILES}


def fresh_dir(name: str, build_out: Path) -> Path:
    d = TMP / name
    d.mkdir()
    shutil.copytree(build_out / "arm-l8_15", d / "arm-l8_15")
    return d


def O6(lost: int, gained: int, n: int = 100) -> dict:
    """A C6 per-stratum record: the paired exact McNemar of M() plus the stratum's row count."""
    return {**M(lost, gained), "n": n}


def M(lost: int, gained: int) -> dict:
    """A paired exact McNemar record with the given discordant counts (plus 10 concordant rows)."""
    base = {**{f"l{i}": True for i in range(lost)}, **{f"g{i}": False for i in range(gained)},
            **{f"c{i}": True for i in range(10)}}
    arm = {**{f"l{i}": False for i in range(lost)}, **{f"g{i}": True for i in range(gained)},
           **{f"c{i}": True for i in range(10)}}
    return L.mcnemar_exact(base, arm)


# ---------------------------------------------------------------------------------------------- checks
def t1_pure():
    print("1 pure functions", flush=True)
    ok_fam = {"ledger_replay": M(20, 12), "prerequisite_plan": M(15, 10), "register_machine": M(5, 0)}
    ok_ood = {s: O6(0, 0) for s in L.OOD_STRATA}
    ok_ood.update(L_big=O6(5, 0), R_wrap=O6(10, 5))         # -5 at the tolerance: p .0625 and .30, no detected loss
    base = {"comparable": True, "comparable_reasons": [], "floors_cleared": 3, "acq_family": ok_fam,
            "arith_perm_l": 62, "retained_l": 39, "retained_t": 41, "drift_ci_upper": 0.75, "ood_stratum": ok_ood}
    D = lambda **kw: L.decide({**base, **kw})  # noqa: E731
    check("all conditions at their passing boundary -> PROMOTE (ledger -8, prerequisite -5, register -5 with 0 "
          "gains: p 0.0625 is no detected loss)", D()["verdict"] == L.PROMOTE, str(D()["conditions"]))
    fam = lambda f, m: {**ok_fam, f: m}  # noqa: E731
    ood = lambda s, m: {**ok_ood, s: m}  # noqa: E731
    cases = [({"comparable": False}, L.NON_COMPARABLE, "C0"),
             ({"comparable": False, "floors_cleared": 2}, L.NON_COMPARABLE, "C0"),
             ({"floors_cleared": 2}, L.DO_NOT_PROMOTE, "C1"),
             ({"acq_family": fam("ledger_replay", M(21, 12))}, L.DO_NOT_PROMOTE, "C2"),       # net -9 < -8
             ({"acq_family": fam("prerequisite_plan", M(16, 10))}, L.DO_NOT_PROMOTE, "C2"),   # net -6 < -5
             ({"acq_family": fam("register_machine", M(6, 0))}, L.DO_NOT_PROMOTE, "C2"),      # -6 in tol, p .031
             ({"acq_family": fam("ledger_replay", M(7, 0))}, L.DO_NOT_PROMOTE, "C2"),         # -7 in tol, detected
             ({"acq_family": fam("ledger_replay", M(40, 32))}, L.PROMOTE, None),              # -8, p .41
             ({"acq_family": {k: v for k, v in ok_fam.items() if k != "register_machine"}}, L.DO_NOT_PROMOTE, "C2"),
             ({"arith_perm_l": 61}, L.DO_NOT_PROMOTE, "C3"),
             ({"retained_l": 36, "retained_t": 36}, L.DO_NOT_PROMOTE, "C4"),   # below the absolute bar
             ({"retained_l": 38, "retained_t": 41}, L.DO_NOT_PROMOTE, "C4"),   # 3 below Titan-Sparse
             ({"retained_l": 37, "retained_t": 39}, L.PROMOTE, None),          # 2 below and >= 37
             ({"drift_ci_upper": 0.80}, L.DO_NOT_PROMOTE, "C5"),
             ({"drift_ci_upper": 0.7999}, L.PROMOTE, None),
             ({"drift_ci_upper": float("nan")}, L.DO_NOT_PROMOTE, "C5"),
             ({"drift_ci_upper": None}, L.DO_NOT_PROMOTE, "C5"),
             ({"ood_stratum": ood("P_appdel", O6(10, 0))}, L.DO_NOT_PROMOTE, "C6"),        # loses 10 of 100
             ({"ood_stratum": ood("L_void", O6(6, 0))}, L.DO_NOT_PROMOTE, "C6"),           # net -6 < -5
             ({"ood_stratum": ood("R_boundary", O6(11, 5))}, L.DO_NOT_PROMOTE, "C6"),      # net -6, p .21
             ({"ood_stratum": ood("P_inapp", O6(30, 25))}, L.PROMOTE, None),               # net -5, p .59
             ({"ood_stratum": ood("L_big", O6(4, 0, n=40))}, L.DO_NOT_PROMOTE, "C6"),      # n 40: tolerance -3
             ({"ood_stratum": ood("L_big", O6(3, 0, n=40))}, L.PROMOTE, None),             # n 40: -3, p .25
             ({"ood_stratum": ood("L_big", O6(7, 1, n=8))}, L.DO_NOT_PROMOTE, "C6"),       # -6 < -3 at n 8
             ({"ood_stratum": ood("R_wrap", {"error": "ROW ID MISMATCH", "n": 100})}, L.DO_NOT_PROMOTE, "C6"),
             ({"ood_stratum": {k: v for k, v in ok_ood.items() if k != "R_boundary"}}, L.DO_NOT_PROMOTE, "C6"),
             ({"ood_stratum": None}, L.DO_NOT_PROMOTE, "C6"),
             ({"comparable": False, "ood_stratum": ood("L_void", O6(10, 0))}, L.NON_COMPARABLE, "C0")]
    bad = []
    for kw, want, failed in cases:
        r = D(**kw)
        if r["verdict"] != want or (failed is not None and failed not in r["failed"]):
            bad.append((list(kw), r["verdict"], r["failed"]))
    check("decision rule boundaries (C0 dominates; C2 per family: tolerance and detected loss; C4 both legs; C5 "
          "strict, NaN/None fail; C6 per OOD stratum: -max(3, ceil(5% n)) and detected loss, missing stratum, broken "
          "pairing and absent input fail)", not bad, f"{len(cases)} cases" + (f"; bad {bad}" if bad else ""))
    r6 = D(ood_stratum=ood("P_appdel", O6(10, 0)))["c6_by_stratum"]["P_appdel"]
    check("C6 on 'L8-15 loses 10 of 100 in one stratum': net -10 < -5 and a detected loss (p = 2/1024)",
          r6["net"] == -10 and r6["tolerance"] == -5 and r6["detected_loss"] and not r6["pass"]
          and abs(r6["p"] - 2 / 1024) < 1e-12, str(r6))
    check("C6 tolerance max(3, ceil(5% of n)): n 100 -> 5, 61 -> 4, 60 -> 3, 20 -> 3, 1 -> 3; at n = 100 the detection "
          "clause is implied (5 lost / 0 gained is p .0625, 6 / 0 is p .031)",
          [L.ood_tol(n) for n in (100, 61, 60, 20, 1)] == [5, 4, 3, 3, 3] and O6(5, 0)["p"] > 0.05 > O6(6, 0)["p"]
          and L.OOD_N == 600 and L.OOD_N_PER_STRATUM == 100 and len(L.OOD_STRATA) == 6)
    bad_b = []
    for k in range(0, 12):                   # Titan-Sparse correct on k of a stratum's 100 rows; L8-15 loses <= k
        fails = any(not L.c6_stratum(O6(lost, gained), "x")["pass"] for lost in range(k + 1) for gained in range(31))
        if L.c6_binding(k, 100)["can_fail"] != fails:
            bad_b.append(k)
    check("C6 binding flag (report-only) == 'some outcome fails C6', exhaustively for Titan-Sparse correct on 0..11 of "
          "100 (lost <= correct, gained 0..30): a stratum can fail only from 6 correct; 4 at n = 40, 1 or 8",
          not bad_b and L.c6_min_lost_to_fail(100) == 6 and [L.c6_min_lost_to_fail(n) for n in (61, 60, 40, 8, 1)]
          == [5, 4, 4, 4, 4] and not L.c6_binding(1, 1)["can_fail"], f"mismatch at {bad_b}" if bad_b else "")
    check("the registered rule text is carried by every decision", D()["rule"] == L.DECISION_RULE)
    check("C2 / C4 margins derive from REP-001's three control draws: half-spans 8 / 5 / 6 and retained 2",
          L.ACQ_FAMILY_TOL == {"ledger_replay": 8, "prerequisite_plan": 5, "register_machine": 6}
          and L.RETAINED_MARGIN == 2 and [sum(L.REP001_CONTROL[f][i] for f in L.FAMILIES) for i in range(3)]
          == [234, 229, 228], str(L.ACQ_FAMILY_TOL))
    check("C4's -2 margin implies no detected loss (the smallest detectable net at p < .05 is -6: 6 lost, 0 gained)",
          M(6, 0)["p"] < 0.05 < M(5, 0)["p"] and M(4, 2)["p"] > 0.05)
    fl = L.fragile_flags({**base, "acq_canonical_l": {"ledger_replay": 43, "prerequisite_plan": 90,
                                                      "register_machine": 80}},
                         {f: L.min_count_clearing(f) for f in L.FAMILIES})
    check("fragile flags: ledger 43 vs 41 needed, C2 at its tolerance, C3 at 62, C4 at both bars, C6 L_big at its "
          "tolerance are flagged; C6 strata at net 0 are not",
          any(x.startswith("C1 ledger_replay") for x in fl) and any(x.startswith("C2 ledger_replay") for x in fl)
          and any(x.startswith("C3") for x in fl) and any(x.startswith("C4") for x in fl)
          and any(x.startswith("C6 L_big") for x in fl) and not any(x.startswith("C6 L_void") for x in fl)
          and [L.min_count_clearing(f) for f in L.FAMILIES] == [41, 17, 16], "; ".join(fl))
    pub = {"ledger_replay": (53, 0.3277), "prerequisite_plan": (96, 0.6658), "register_machine": (83, 0.5591)}
    got = {f: round(L.clopper_pearson(k, 128)[0], 4) for f, (k, _) in pub.items()}
    check("Clopper-Pearson lower bounds reproduce Titan-Sparse's published 0.3277 / 0.6658 / 0.5591",
          got == {f: v for f, (_, v) in pub.items()}, str(got))
    m = L.mcnemar_exact({1: True, 2: True, 3: False, 4: False}, {1: False, 2: True, 3: True, 4: True})
    check("exact McNemar: 1 lost, 2 gained, p = 1", (m["lost"], m["gained"], m["p"]) == (1, 2, 1.0), str(m))
    check("parity allowance max(3, floor(3% of n)): 384 -> 11, 256 -> 7, 150 -> 4, 24 -> 3",
          [L.allowed_diff(n) for n in (384, 256, 150, 24)] == [11, 7, 4, 3])
    ctx = L.ret_context()
    B = L.read_json(L.pinned("b28"))
    outs = [{"id": r["id"], "reply_class": next(x["reply_class"] for x in ctx["rows"] if x["id"] == r["id"]),
             "text": r["source"]["raw"], "terminated": r["source"]["terminated"],
             "new_tokens": r["source"].get("new_tokens", 0)} for r in B["rows"]]
    S = L.score_ret(ctx, outs)
    check("retention scorer on the frozen source's own answers: permissive 47, strict 41, retained 47, 0 lost",
          (S["arithmetic_permissive"], S["arithmetic_strict"], S["retained"], S["retained_strict"],
           S["by_bare_policy"]["permissive"]["lost_total"]) == (47, 41, 47, 41, 0),
          f"{S['arithmetic_permissive']}/{S['arithmetic_strict']}, retained {S['retained']}")


def t2_build(cores: dict):
    print("2 build", flush=True)
    rec_out = L.build_stream(cores["supa"] / "model.safetensors", cores["sparse"] / "model.safetensors", L._NullSink())
    check("B2 premise holds for the stand-ins; delta energy accounted only in blocks 8..27",
          rec_out["b2_titan_equals_supa_outside_blocks_8_27"] and 0 < rec_out["delta_energy_share_blocks_8_15"] < 1,
          f"share 8..15 {rec_out['delta_energy_share_blocks_8_15']:.3f}")
    L.main(["build", "--dry-hash", "--record"])
    exp = L.read_json(L.resolve("expected"))
    check("--dry-hash --record: group hashes == run_patch.group_hashes of an independently assembled L8-15",
          exp["group_hashes"] == R14["l8_15"] and exp["checks"]["B4"].startswith("per-group"),
          exp["out_weight_sha256"][:16])
    ok, msg = raises(lambda: L.main(["build", "--dry-hash", "--record"]), "create-only")
    check("the expectation record is create-only", ok, msg)
    out = TMP / "build"
    L.main(["build", "--out-dir", str(out)])
    arm = out / "arm-l8_15"
    got = load_file(str(arm / "model.safetensors"))
    s = load_file(str(cores["supa"] / "model.safetensors"))
    t = load_file(str(cores["sparse"] / "model.safetensors"))
    bad = [k for k in s if not torch.equal(got[k], (t[k] if L.block_of(k) in L.L8_15_BLOCKS else s[k].float()))]
    check("built tensors: Titan's FP32 in blocks 8..15, Supa widened elsewhere, all FP32, same names",
          not bad and set(got) == set(s) and all(v.dtype == torch.float32 for v in got.values()),
          f"{len(got)} tensors, {len(bad)} differ")
    rcpt = L.read_json(arm / "MATERIALIZATION.json")
    fsha = sha(arm / "model.safetensors")
    check("file sha256 == expectation record == receipt; receipt names Supa as source and the gates B1-B6",
          fsha == exp["out_weight_sha256"] == rcpt["out_weight_sha256"]
          and rcpt["source_weight_sha256"] == L.SHA["supa_weights"] and rcpt["record"] == "CHECKPOINT_MATERIALIZATION"
          and {"B1", "B3", "B4", "B5", "B6"} <= set(rcpt["gates"]), fsha[:16])
    check("sidecars are byte copies of Supa's (the programme's measured-arm convention)",
          all(sha(arm / f) == sha(cores["supa"] / f) for f in L.SIDECARS))
    alt = TMP / "save_file_twin.safetensors"
    save_file({k: got[k] for k in sorted(got)}, str(alt), metadata={"format": "pt"})
    print(f"  [INFO] safetensors.save_file of the same tensors is byte-identical: {sha(alt) == fsha}", flush=True)
    mt = (arm / "model.safetensors").stat().st_mtime_ns
    L.main(["build", "--out-dir", str(out)])
    check("B7: rebuilding over an existing arm re-verifies it and writes nothing",
          (arm / "model.safetensors").stat().st_mtime_ns == mt and not (out / "arm-l8_15.building").exists())
    bad_dir = TMP / "tampered"
    shutil.copytree(arm, bad_dir)
    with open(bad_dir / "model.safetensors", "r+b") as fh:
        fh.seek(-5, 2)
        b = fh.read(1)
        fh.seek(-5, 2)
        fh.write(bytes([b[0] ^ 1]))
    ok, msg = raises(lambda: L.verify_arm_dir(bad_dir), "differ")
    check("B7/G1: a tampered arm directory fails re-verification", ok, msg)
    shutil.rmtree(bad_dir)
    bad_sparse = TMP / "bad_sparse.safetensors"
    t2 = dict(t)
    k3 = next(k for k in t2 if L.block_of(k) == 3 and t2[k].dim() == 2)
    t2[k3] = t2[k3].clone()
    t2[k3].view(-1)[7] += 1e-3
    save_file(t2, str(bad_sparse), metadata={"format": "pt"})
    ok, msg = raises(lambda: L.build_stream(cores["supa"] / "model.safetensors", bad_sparse, L._NullSink()), "B2")
    check("B2: a donor that differs from Supa in block 3 is refused", ok, msg)
    keep = L.SHA["supa_weights"]
    L.SHA["supa_weights"] = "0" * 64
    ok, msg = raises(lambda: L._check_build_identity(rec_out), "B1")
    L.SHA["supa_weights"] = keep
    check("B1: an input whose hash is not the pin is refused", ok, msg)
    saved = R14["l8_15"]["block9"]
    R14["l8_15"]["block9"] = "0" * 16
    ok, msg = raises(lambda: L._check_build_identity(rec_out), "B4")
    R14["l8_15"]["block9"] = saved
    check("B4: a build whose group hashes differ from R14's loc8_15 record is refused", ok, msg)
    ok, msg = raises(lambda: L._check_build_identity({**rec_out, "out_weight_sha256": "f" * 64}), "B5")
    check("B5: a build that is not the frozen bytes is refused", ok, msg)
    return out


def t3_run(build_out: Path):
    print("3 run (dense, cpu)", flush=True)
    a = build_out                                           # arm-l8_15 lives here; runs share it
    keep = L.SHA["sparse_weights"]
    L.SHA["sparse_weights"] = "0" * 64
    ok, msg = raises(lambda: run(a), "G1")
    L.SHA["sparse_weights"] = keep
    check("G1: a weight file that is not its pin is refused before any load or output",
          ok and not LOADS and not (a / "RUN.json").exists(), msg)
    g2 = fresh_dir("run_g2", build_out)
    CORRUPT["dir"] = "core_supa"
    ok, msg = raises(lambda: run(g2), "G2")
    CORRUPT.clear()
    check("G2: a loaded model that differs from its file is refused; no stage output, no DONE.json",
          ok and not (g2 / "supa" / "acq.jsonl").exists() and not (g2 / "DONE.json").exists(), msg)
    shutil.rmtree(g2)
    seq = []
    L._on_stage_done = lambda arm, stage: seq.append(f"{arm}:{stage}")
    L._PARITY_HOOK = lambda g: (seq.append("GATE"), g)[1]
    try:
        loads = run(a)
    finally:
        L._on_stage_done, L._PARITY_HOOK = None, None
    gate = L.read_json(a / "PARITY_GATE.json")
    check("uninterrupted run: four loads (supa, sparse, supa again for its OOD stage, l8_15), DONE.json, every stage "
          "file present; the parity gate was evaluated before L8-15 and is not enforced on a subset",
          loads == ["dense:core_supa", "dense:core_sparse", "dense:core_supa", "dense:arm-l8_15"]
          and (a / "DONE.json").exists() and all((a / arm / f).exists() for arm in L.ARMS for f in STAGE_FILES)
          and gate["enforced"] is False and not (a / "PARITY_FAIL.json").exists()
          and L.read_json(a / "DONE.json")["parity_gate"]["enforced"] is False, str(loads))
    want = ([f"supa:{s}" for s in L.PRE_GATE_STAGES] + [f"sparse:{s}" for s in L.PRE_GATE_STAGES] + ["GATE"]
            + ["sparse:ood", "supa:ood"] + [f"l8_15:{s}" for s in L.STAGES])
    stg = {arm: L.read_json(a / arm / "STAGES.json")["stages"] for arm in L.ARMS}
    armrec = {arm: L.read_json(a / arm / "ARM.json") for arm in L.ARMS}
    check("registered order, observed: supa and sparse run the five pre-gate stages, then the parity gate, then sparse's "
          "OOD stage (kept loaded across the gate), supa's OOD stage (reloaded) and every L8-15 stage; no model read "
          "the OOD bank before the gate; every post-gate stage records the gate it followed, no pre-gate stage does",
          seq == want and L.PRE_GATE_STAGES + L.POST_GATE_STAGES == L.STAGES and L.POST_GATE_STAGES == ("ood",)
          and all(("parity_gate_before" in v) == (arm == "l8_15" or s == "ood") for arm in L.ARMS
                  for s, v in stg[arm].items())
          and [x["for_stages"] for x in armrec["supa"]["loads"]] == [list(L.PRE_GATE_STAGES), ["ood"]]
          and [x["for_stages"] for x in armrec["sparse"]["loads"]] == [list(L.PRE_GATE_STAGES)]
          and [x["for_stages"] for x in armrec["l8_15"]["loads"]] == [list(L.STAGES)],
          " -> ".join(seq))
    done = (a / "DONE.json").read_bytes()
    check("rerun of a finished directory loads nothing and leaves DONE.json byte-identical",
          run(a) == [] and (a / "DONE.json").read_bytes() == done)
    acq = L.read_jsonl(a / "l8_15" / "acq.jsonl")
    ret = L.read_jsonl(a / "l8_15" / "ret.jsonl")
    lc = L.read_jsonl(a / "l8_15" / "lc.jsonl")
    lp = np.load(a / "l8_15" / "drift_fresh.npy")
    ood = L.read_jsonl(a / "l8_15" / "ood.jsonl")
    bank = {r["id"]: r for r in L.load_ood_bank()}
    check("output formats: 3 acq / 3 ret / 2 lc / 6 ood rows (one per OOD stratum, the bank's first), ret prompts within "
          "512 tokens, OOD generations within the 96-token cap, drift (3, 151936) float16 log-probs that normalise",
          len(acq) == 3 and len(ret) == 3 and len(lc) == 2
          and all(1 <= r["input_tokens"] <= 512 for r in ret) and all(r["status"] == "ok" for r in lc)
          and [r["id"] for r in ood] == [f"ood-{s}-000" for s in L.OOD_STRATA]
          and all(r["stratum"] == bank[r["id"]]["stratum"] and 0 < r["input_tokens"] and r["generated_tokens"] <= 96
                  for r in ood)
          and lp.shape == (3, 151936) and lp.dtype == np.float16
          and np.allclose(np.exp(lp.astype(np.float64)).sum(1), 1.0, atol=2e-2), f"{lp.shape}")
    arm_rec = L.read_json(a / "l8_15" / "ARM.json")
    run_rec = L.read_json(a / "RUN.json")
    st = L.read_json(a / "l8_15" / "STAGES.json")
    check("ARM.json: G2 hashes == the frozen expectation's; FP32; TF32 off; DenseReference; PG named; RUN.json and "
          "every stage carry the runner sha, the invocation and the environment",
          arm_rec["group_hashes"] == L.read_json(L.resolve("expected"))["group_hashes"]
          and arm_rec["numerics"]["float32_matmul_precision"] == "highest" and arm_rec["record"] == "L1_ARM"
          and not arm_rec["numerics"]["cuda_matmul_allow_tf32"] and arm_rec["forward"] == "dense_reference@5b4c22d"
          and any(g.startswith("PG") for g in arm_rec["gates"]) and run_rec["record"] == "L1_RUN"
          and all(x["script_sha256"] == sha(HERE / "l1_screen.py") and x["no_freeze"] and x["env"] == L.env_record()
                  for x in run_rec["invocations"])
          and all(v["script_sha256"] == sha(HERE / "l1_screen.py") and "invocation" in v for v in st["stages"].values()))
    b = fresh_dir("run_resume", build_out)
    count = [0]

    def crash(arm, stage):
        count[0] += 1
        if count[0] == len(L.PRE_GATE_STAGES) + 2:        # supa's 5 pre-gate stages, then sparse acq and ret
            raise Crash(f"{arm}:{stage}")
    L._on_stage_done = crash
    try:
        run(b)
        crashed = False
    except Crash:
        crashed = True
    L._on_stage_done = None
    st = L.read_json(b / "sparse" / "STAGES.json")
    st_supa = L.read_json(b / "supa" / "STAGES.json")
    check("crash after 7 stages: no DONE.json; supa holds its 5 pre-gate stages (no OOD yet), sparse holds acq+ret "
          "only; the run lock is released",
          crashed and not (b / "DONE.json").exists() and sorted(st["stages"]) == ["acq", "ret"]
          and sorted(st_supa["stages"]) == sorted(L.PRE_GATE_STAGES) and not (b / "supa" / "ood.jsonl").exists()
          and not (b / ".run.lock").exists())
    rr = L.read_json(b / "RUN.json")
    keep_env = json.dumps(rr["invocations"][0]["env"])
    rr["invocations"][0]["env"]["transformers"] = "4.51.3"
    L.write_json(b / "RUN.json", rr)
    ok, msg = raises(lambda: run(b), "G0")
    rr["invocations"][0]["env"] = json.loads(keep_env)
    L.write_json(b / "RUN.json", rr)
    check("a resume in another environment is refused before any load (never stitched)", ok and not LOADS, msg)
    (b / ".run.lock").write_text(json.dumps({"pid": os.getpid(), "create_time": L._proc_create_time(os.getpid())}),
                                 encoding="utf-8")
    ok, msg = raises(lambda: run(b), "one waiter")
    check("a live run lock (another l1_screen process) refuses a second run on the same directory", ok and not LOADS, msg)
    (b / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
    loads = run(b)
    check("a dead owner's lock is reclaimed; the resume loads sparse (its pending stages, the gate, its OOD stage), "
          "supa (OOD only) and l8_15, and runs only the pending stages",
          loads == ["dense:core_sparse", "dense:core_supa", "dense:arm-l8_15"] and (b / "DONE.json").exists()
          and not (b / ".run.lock").exists(), str(loads))
    ha, hb = stage_hashes(a), stage_hashes(b)
    check("resumed run == uninterrupted run, file by file", ha == hb,
          f"{sum(ha[k] == hb[k] for k in ha)}/{len(ha)} equal")
    return a


def t4_hf(a: Path, build_out: Path):
    print("4 forward-path cross-check (Qwen3ForCausalLM)", flush=True)
    c = fresh_dir("run_hf", build_out)
    loads = run(c, forward="hf")
    check("hf run completes with four loads",
          loads == ["hf:core_supa", "hf:core_sparse", "hf:core_supa", "hf:arm-l8_15"], str(loads))
    same, n, diffs = 0, 0, []
    for arm in L.ARMS:
        for f, key in (("acq.jsonl", "prediction"), ("ret.jsonl", "text"), ("lc.jsonl", "prediction")):
            A = {r["id"]: r[key] for r in L.read_jsonl(a / arm / f)}
            C = {r["id"]: r[key] for r in L.read_jsonl(c / arm / f)}
            n += len(A)
            same += sum(A[i] == C[i] for i in A)
        for s in ("drift_fresh", "drift_bridge"):
            x = np.load(a / arm / f"{s}.npy").astype(np.float32)
            y = np.load(c / arm / f"{s}.npy").astype(np.float32)
            ulp = np.spacing(np.maximum(np.abs(x), np.abs(y)).astype(np.float16)).astype(np.float32)
            diffs.append((float(np.abs(x - y).max()), float((np.abs(x - y) / ulp).max())))
    check("DenseReference and Qwen3ForCausalLM produce identical greedy texts on every row of every arm",
          same == n, f"{same}/{n}")
    check("first-token log-probs agree within one float16 storage step everywhere",
          max(u for _, u in diffs) <= 1.0, f"max |d| {max(d for d, _ in diffs):.4f}, max {max(u for _, u in diffs):.2f} ulp")
    return c


class RecGuard:
    """Records the host-guard calls the runner makes (real=False: the token pause checks the temp-dir flag)."""
    real = False

    def __init__(self, events):
        self.events = events

    def pre_lease(self):
        self.events.append("pre_lease")

    def before_load(self):
        self.events.append("before_load")

    def during(self):
        self.events.append("during")


def t5_cuda_wiring(a: Path, build_out: Path):
    print("5 --device cuda wiring (CPU tensors, stubbed lease) and host safety", flush=True)
    d = fresh_dir("run_cuda", build_out)
    events, calls = [], {"breathe": 0}
    orig = (runtime.gpu_lease, runtime.breathe, L._GUARD_FACTORY, L.ROOT, L.TOKEN_PAUSE_S)
    lv_orig = {k: getattr(LV, k) for k in ("_heavy_python_pids", "_pid_alive", "_dedicated_gpu_bytes")}
    on_enter = []

    @contextmanager
    def lease(label, job):
        events.append(("lease", label, job))
        events.append(("lv_patched", LV._heavy_python_pids is L._psutil_python_rss and LV._pid_alive is L._psutil_pid_alive))
        for f in on_enter:
            f()
        yield

    def breathe():
        calls["breathe"] += 1
    runtime.gpu_lease, runtime.breathe = lease, breathe
    L._GUARD_FACTORY = lambda real: RecGuard(events)
    L.ROOT, L.TOKEN_PAUSE_S = TMP, 0.001                    # pause-flag lookups go to the temp dir
    L._TENSOR_DEVICE = "cpu"
    try:
        loads = run(d, device="cuda")
        # a run that waited for the lease while another process finished the directory loads nothing
        x = TMP / "run_raced"
        shutil.copytree(a, x, ignore=shutil.ignore_patterns("L1_RESULTS.*"))
        (x / "DONE.json").unlink()
        shutil.rmtree(x / "l8_15")
        on_enter.append(lambda: (shutil.copytree(a / "l8_15", x / "l8_15"), shutil.copyfile(a / "DONE.json", x / "DONE.json")))
        n_before = len(events)
        loads_raced = run(x, device="cuda")
        raced_events = events[n_before:]
        on_enter.clear()
    finally:
        runtime.gpu_lease, runtime.breathe, L._GUARD_FACTORY, L.ROOT, L.TOKEN_PAUSE_S = orig
        L._TENSOR_DEVICE = None
    per_arm = (3 * SUBSET["acq"] + 3 * SUBSET["ret"] + SUBSET["lc"] + 2 * SUBSET["drift"]   # 3 families, 3 classes
               + len(L.OOD_STRATA) * SUBSET["ood"])                                       # 6 OOD strata
    first = events[: events.index("before_load")]
    check("one lease with the registered label and job; the host gate (RAM, live watchdog) runs BEFORE the lease, "
          "the before-load gate before each of 4 loads; the lease helpers are the spawn-free ones while held",
          [e for e in events if isinstance(e, tuple) and e[0] == "lease"][0] == ("lease", L.LEASE_LABEL, L.LEASE_JOB)
          and first[0] == "pre_lease" and first[1][0] == "lease" and ("lv_patched", True) in events
          and events.count("before_load") == L.LOADS_PER_RUN == 4 and len(loads) == 4, str(first[:3]))
    check("runtime.breathe() and the host guard once per row", calls["breathe"] == 3 * per_arm
          and events.count("during") == 3 * per_arm, f"{calls['breathe']} == 3 x {per_arm}")
    check("after the run the lease module's own helpers are restored",
          all(getattr(LV, k) is v for k, v in lv_orig.items()))
    ha, hd = stage_hashes(a), stage_hashes(d)
    check("cuda-wired outputs == cpu outputs, file by file", ha == hd, f"{sum(ha[k] == hd[k] for k in ha)}/{len(ha)}")
    run_rec = L.read_json(d / "RUN.json")
    check("RUN.json records the lease label, FP32, TF32 off, the forward path and the registered constants",
          run_rec["lease_label"] == L.LEASE_LABEL and run_rec["dtype"] == "float32" and run_rec["tf32"] is False
          and run_rec["forward"] == "dense" and run_rec["registered"]["decision_rule"] == L.DECISION_RULE)
    check("pending is re-derived after the lease: a directory finished by another process while this one waited "
          "loads nothing", loads_raced == [] and "before_load" not in raced_events
          and any(isinstance(e, tuple) and e[0] == "lease" for e in raced_events), str(loads_raced))
    # the lease's helpers spawn no process while waiting on a held lock
    spawned = []

    def no_spawn(*args, **kw):
        spawned.append(args[0] if args else kw.get("args"))
        raise AssertionError("a process was spawned")
    sp = (subprocess.run, subprocess.Popen, LV.LOCK)
    fake = TMP / "fake-gpu.lock"
    fake.write_text(json.dumps({"pid": os.getpid(), "worker_pids": [], "created": time.time(), "label": "x"}),
                    encoding="utf-8")
    subprocess.run, subprocess.Popen, LV.LOCK = no_spawn, no_spawn, fake
    t0 = time.time()
    try:
        with L.noproc_lease():
            ok, msg = raises(lambda: LV.acquire("tiny-test", "tiny-test", timeout_seconds=1, poll_seconds=0.5), "",
                             LV.LeaseError)
    finally:
        subprocess.run, subprocess.Popen, LV.LOCK = sp
    check("waiting on a held lock (live holder pid) through the lease module: polls with psutil and the in-process "
          "PDH counter, zero processes spawned, the real lock never touched", ok and not spawned,
          f"{time.time() - t0:.1f}s; {msg[:60]}")
    # HostGuard with a fake clock
    root = TMP / "guard"
    (root / "runs").mkdir(parents=True)
    log = root / "runs" / "thermal_log.txt"
    log.write_text("x", encoding="utf-8")
    T = [time.time()]

    def tick(s, keep_alive=True):
        T[0] += s
        if keep_alive:
            os.utime(log, (T[0], T[0]))
    t0 = T[0]
    G = L.HostGuard(True, root=root, ram_fn=lambda: 2.0, clock=lambda: T[0], sleep=tick)
    ok1, m1 = raises(G.pre_lease, "host gate not met", L.ResumableExit)
    waited1 = T[0] - t0
    G = L.HostGuard(True, root=root, ram_fn=lambda: 20.0, clock=lambda: T[0], sleep=lambda s: tick(s, False))
    T[0] += 600                                              # the watchdog log is now 10 min old
    ok2, m2 = raises(G.before_load, "watchdog", L.ResumableExit)
    (root / "runs" / ".thermal_pause").write_text("1", encoding="utf-8")
    os.utime(log, (T[0], T[0]))
    G = L.HostGuard(True, root=root, ram_fn=lambda: 2.5, clock=lambda: T[0], sleep=tick)
    t3 = T[0]
    ok3, m3 = raises(G.during, "RAM-caused pause", L.ResumableExit)
    waited3 = T[0] - t3
    (root / "runs" / ".thermal_pause").unlink()
    G = L.HostGuard(True, root=root, ram_fn=lambda: 20.0, clock=lambda: T[0], sleep=tick)
    os.utime(log, (T[0], T[0]))
    G.pre_lease()
    G.before_load()
    G.during()
    check("HostGuard: low RAM before the lease exits 5 after the 2 h wait (nothing held); a silent watchdog stops a "
          "load; a RAM-caused pause holding the card exits after 15 min; a healthy host passes every gate",
          ok1 and ok2 and ok3 and L.PRE_LEASE_WAIT_S <= waited1 <= L.PRE_LEASE_WAIT_S + 120
          and L.RAM_PAUSE_EXIT_S <= waited3 <= L.RAM_PAUSE_EXIT_S + 10,
          f"waited {waited1:.0f}s / {waited3:.0f}s; {m1[:40]} | {m2[:40]} | {m3[:40]}")


def t6_freeze(a: Path, build_out: Path):
    print("6 freeze", flush=True)
    e = TMP / "run_nofreeze"
    ok, msg = raises(lambda: run(e, freeze=True), "freeze check failed")
    check("run refuses without a freeze record", ok and not (e / "RUN.json").exists(), msg)
    ok, msg = raises(lambda: L.main(["analyze", "--out-dir", str(a), "--allow-subset", "--skip-tokenizer-check"]),
                     "freeze check failed")
    check("analyze refuses without a freeze record", ok, msg)
    ok, msg = raises(lambda: L.main(["freeze", "--approved-by", "test", "--out-dir", str(a)]), "already exist")
    check("freeze refuses once L1 outputs exist in --out-dir", ok and not L.resolve("freeze").exists(), msg)
    f = TMP / "run_frozen"
    f.mkdir()
    fake_runs = TMP / "runs_root_with_output" / "somewhere"
    fake_runs.mkdir(parents=True)
    (fake_runs / "RUN.json").write_text(json.dumps({"record": "L1_RUN", "lease_label": L.LEASE_LABEL}), encoding="utf-8")
    L.PATHS["runs_root"] = TMP / "runs_root_with_output"
    ok, msg = raises(lambda: L.main(["freeze", "--approved-by", "test", "--out-dir", str(f)]), "already exist")
    L.PATHS["runs_root"] = TMP / "runs_root_empty"
    check("freeze refuses if any L1 output exists anywhere under runs/ (any --out-dir)", ok and not L.resolve("freeze").exists(), msg)
    L.main(["freeze", "--approved-by", "test", "--out-dir", str(f)])
    rec = L.read_json(L.resolve("freeze"))
    check("freeze writes the record (files incl. the queue hook and the reference-only vendor manifest, inputs, "
          "registered constants, predictions, the environment); verify == []",
          L.verify_freeze() == [] and rec["frozen_before_any_l1_output"] and rec["approved_by"] == "test"
          and rec["registered"]["drift_upper"] == L.DRIFT_UPPER and set(rec["predictions"]) == set(L.PREDICTIONS)
          and rec["environment"] == L.env_record() and {"queue_hook_draft", "vendor_refonly_manifest", "ood_bank",
                                                         "ood_bank_source", "ood_builder", "rcwmt_env"} <= set(rec["files"])
          and rec["registered"]["ood_bank_sha256"] == L.SHA["ood_bank"] == sha(L.resolve("ood_bank")))
    real = L.REAL_WEIGHT_SHA
    L.REAL_WEIGHT_SHA = frozenset(real | {L.SHA["supa_weights"]})
    g = fresh_dir("run_realguard", build_out)
    ok, msg = raises(lambda: run(g), "tiny-model test only")
    L.REAL_WEIGHT_SHA = real
    check("--no-freeze refuses a real core (here: the stand-in declared real) before any load", ok and not LOADS, msg)
    fr = L.read_json(L.resolve("freeze"))
    raw = L.resolve("freeze").read_bytes()
    fr["environment"] = {**fr["environment"], "transformers": "4.51.3"}
    L.resolve("freeze").write_bytes((json.dumps(fr, indent=1) + "\n").encode("utf-8"))
    ok, msg = raises(lambda: run(g, freeze=True), "G0")
    L.resolve("freeze").write_bytes(raw)
    check("a frozen run refuses an environment other than the frozen one", ok and not LOADS, msg)
    ok, msg = raises(lambda: L.main(["analyze", "--out-dir", str(a), "--allow-subset", "--skip-tokenizer-check"]),
                     "not the registered run")
    check("analyze refuses a run made before the freeze / with --no-freeze, even though the files now verify",
          ok and "--no-freeze" in msg, msg)
    loads = run(g, freeze=True)
    r = analyze(g, "--allow-subset")
    check("a run made after the freeze, with it, binds: analyze accepts it (NON-COMPARABLE only for the subset)",
          len(loads) == 4 and r["verdict"] == L.NON_COMPARABLE
          and r["decision"]["inputs"]["comparable_reasons"] == ["row subset (test run)"]
          and r["freeze_binding"].startswith("every invocation"), r["freeze_binding"])
    L.REGISTERED["drift_upper"] = 0.9
    bad = L.verify_freeze()
    L.REGISTERED["drift_upper"] = L.DRIFT_UPPER
    check("a changed registered constant is detected", "registered constants" in bad, str(bad))
    p = L.resolve("prereg")
    txt = p.read_text(encoding="utf-8")
    p.write_text(txt + "edit\n", encoding="utf-8")
    bad = L.verify_freeze()
    p.write_text(txt, encoding="utf-8")
    check("an edited preregistration file is detected; restoring it clears the check",
          "prereg" in bad and L.verify_freeze() == [], str(bad))
    check("a frozen run on a finished directory starts (freeze verified) and loads nothing", run(a, freeze=True) == [])
    L.resolve("freeze").unlink()


def t7_analyze(a: Path, c: Path, build_out: Path):
    print("7 parity gate and analyze", flush=True)
    views = L.load_views()
    vmap = {v["id"]: v for v in views}
    ctx = L.ret_context()
    for arm in ("supa", "sparse"):
        P = L.read_json(L.pinned(f"anchor_process_{arm}"))
        O = {d["id"]: {"prediction": d["prediction"], "terminated": d["terminated"],
                       "generated_tokens": d["generated_tokens"]} for d in P["predictions"]}
        leg = L.acq_leg(arm, L.score_acq(views, O), O, subset=False)
        O2 = dict(O)
        flip = [i for i in O2 if L.score_acq([vmap[i]], {i: O2[i]})["detail"][0]["canonical"]][:4]
        for i in flip:
            O2[i] = {**O2[i], "prediction": "x"}
        leg2 = L.acq_leg(arm, L.score_acq(views, O2), O2, subset=False)
        check(f"parity leg SELECT-VIEWS-001 ({arm}): the published predictions pass; 4 flipped rows pass the text "
              f"allowance but {'break the +-3 total' if len(flip) == 4 else 'cannot be flipped'}",
              leg["pass"] and leg["text"]["differ"] == 0 and (len(flip) < 4 or (not leg2["pass"] and leg2["text"]["pass"])),
              f"{leg['ours']['acq_total']}/{leg['published']['acq_total']}, flipped {len(flip)}")
    RT = L.read_json(L.pinned("anchor_ret_sparse"))
    samp = {s["id"]: s for s in RT["samples"]}
    rc = {r["id"]: r["reply_class"] for r in ctx["rows"]}
    O_ret = [{"id": r["id"], "reply_class": r["reply_class"], "text": samp[r["id"]]["arm"] if r["id"] in samp else "",
              "terminated": bool(samp[r["id"]]["arm_terminated"]) if r["id"] in samp else True, "new_tokens": 1}
             for r in ctx["rows"]]

    def E_ret(drop=0, cls_shift=0):
        by = {}
        for pol in ("permissive", "strict"):
            b = RT["by_bare_policy"][pol]
            lost = {i for c in b["per_class"].values() for i in c["lost_ids"]}
            cor = sorted(set(ctx["references"][pol]) - lost)[drop:]
            pc = {c: {"arm_correct": v["arm_correct"] + cls_shift, "terminated": v["terminated"]} for c, v in b["per_class"].items()}
            by[pol] = {"correct_ids": cor, "per_class": pc}
        return {"by_bare_policy": by, "arithmetic_permissive": 72, "arithmetic_strict": 63, "retained": 41}
    g_ok = L.ret_leg_sparse(E_ret(), O_ret, ctx, subset=False)
    g_rows = L.ret_leg_sparse(E_ret(drop=2), O_ret, ctx, subset=False)
    g_cls = L.ret_leg_sparse(E_ret(cls_shift=2), O_ret, ctx, subset=False)
    check("parity leg arithmetic panel (Titan-Sparse): the published lost ids, per-class counts and 24 samples pass; "
          "2 extra lost rows or a per-class shift of 2 fail", g_ok["pass"] and not g_rows["pass"] and not g_cls["pass"]
          and g_ok["samples"]["n"] == 24 and len(rc) == 150, str(g_ok["policies"]["permissive"]["differ_ids"]))
    R = L.r14_bridge()
    br = L.load_prompts("bridge")
    ids = [u["id"] for u in br]
    ref_supa = np.asarray(R["supa"][[R["row"][i] for i in ids]])
    leg_ok = L.bridge_leg("supa", ref_supa, ids)
    leg_bad = L.bridge_leg("sparse", ref_supa, ids)
    check("parity leg bridge log-probs: R14's own Supa arrays pass (KL 0, top-1 1.0); Supa's arrays judged as "
          "Titan-Sparse's fail", leg_ok["pass"] and leg_ok["kl_l1_r14_mean"] == 0.0 and leg_ok["top1_agreement"] == 1.0
          and not leg_bad["pass"], f"Supa-as-Sparse: KL {leg_bad['kl_l1_r14_mean']:.4f}, top-1 {leg_bad['top1_agreement']:.3f}")
    gate_full = L.parity_gate(a, subset=False)
    check("the tiny anchors judged as a full run fail the enforced gate on their statistics and bridge legs",
          gate_full["enforced"] and not gate_full["pass"] and "supa.bridge_logprobs" in gate_full["failed_legs"]
          and gate_full["legs"]["supa.select_views_001"]["stats_pass"] is False, ", ".join(gate_full["failed_legs"]))
    # a failed, enforced gate: L8-15 is never loaded
    p = fresh_dir("run_parity_fail", build_out)
    L._PARITY_HOOK = lambda g: {**g, "enforced": True}
    try:
        loads = run(p)
        loads2 = run(p)
    finally:
        L._PARITY_HOOK = None
    pf = L.read_json(p / "PARITY_FAIL.json")
    check("a failed, enforced parity gate: PARITY_FAIL.json (create-only), loads only the anchors, no L8-15 output, "
          "no model read the OOD bank (no anchor OOD output or stage), no DONE.json; a rerun loads nothing",
          loads == ["dense:core_supa", "dense:core_sparse"] and loads2 == []
          and not (p / "l8_15").exists() and not (p / "DONE.json").exists() and pf["failed_legs"]
          and not any((p / x / "ood.jsonl").exists() for x in ("supa", "sparse"))
          and all(sorted(L.read_json(p / x / "STAGES.json")["stages"]) == sorted(L.PRE_GATE_STAGES)
                  for x in ("supa", "sparse")) and "OOD bank" in pf["consequence"], str(loads))
    q = TMP / "run_parity_fail_ood"
    shutil.copytree(p, q)
    shutil.copyfile(p / "supa" / "acq.jsonl", q / "supa" / "ood.jsonl")
    ok, msg = raises(lambda: analyze(q, "--no-freeze", "--allow-subset"), "OOD outputs exist next to PARITY_FAIL")
    shutil.rmtree(q)
    check("analyze refuses OOD outputs next to PARITY_FAIL.json (a model would have read the bank without a passed "
          "gate)", ok, msg)
    rp = analyze(p, "--no-freeze", "--allow-subset")
    check("analyze after PARITY_FAIL: NON-COMPARABLE, no L8-15 endpoint computed, no OOD score for any arm, the "
          "registration ends",
          rp["verdict"] == L.NON_COMPARABLE and set(rp["summary"]) == {"supa", "sparse"} and "never loaded" in rp["l8_15"]
          and "contrasts" not in rp and "drift" not in rp
          and not any(k.startswith("ood") for x in rp["summary"].values() for k in x))
    ok, msg = raises(lambda: L.main(["analyze", "--out-dir", str(a), "--no-freeze", "--skip-tokenizer-check"]),
                     "subset")
    check("analyze refuses a subset run without --allow-subset", ok, msg)
    r = analyze(a, "--no-freeze", "--allow-subset")
    reasons = r["decision"]["inputs"]["comparable_reasons"]
    check("verdict NON-COMPARABLE on a subset run; the only reason is the subset (the gate is recorded, re-derived "
          "and equal, but not enforced on a subset)",
          r["verdict"] == L.NON_COMPARABLE and reasons == ["row subset (test run)"]
          and r["parity_gate"]["enforced"] is False, "; ".join(reasons))
    check("scorer parity on the published artifacts passes inside analyze", all(x["pass"] for x in r["scorer_parity"]),
          f"{len(r['scorer_parity'])} checks")
    check("C2 is evaluated per family from the paired McNemar records (L8-15 vs Titan-Sparse)",
          set(r["decision"]["c2_by_family"]) == set(L.FAMILIES)
          and all(r["decision"]["c2_by_family"][f]["net"] == r["contrasts"]["acquisition_l8_15_vs_sparse"][f]["net"]
                  for f in L.FAMILIES))
    indep = {}
    for arm in L.ARMS:
        k = 0
        for row in L.read_jsonl(a / arm / "acq.jsonl"):
            try:
                k += json.dumps(json.loads(row["prediction"]), sort_keys=True, separators=(",", ":")) == vmap[row["id"]]["reference"]
            except (ValueError, TypeError):
                pass
        indep[arm] = k
    check("acquisition recomputed independently == reported, all arms",
          all(r["summary"][arm]["acquisition"] == indep[arm] for arm in L.ARMS), str(indep))

    def kl(P, Q):
        P, Q = P.astype(np.float64), Q.astype(np.float64)
        P = P - np.log(np.exp(P - P.max(1, keepdims=True)).sum(1, keepdims=True)) - P.max(1, keepdims=True)
        Q = Q - np.log(np.exp(Q - Q.max(1, keepdims=True)).sum(1, keepdims=True)) - Q.max(1, keepdims=True)
        return np.maximum((np.exp(P) * (P - Q)).sum(1), 0.0)
    lps = {arm: np.load(a / arm / "drift_fresh.npy") for arm in L.ARMS}
    want = kl(lps["l8_15"], lps["supa"]).mean() / kl(lps["sparse"], lps["supa"]).mean()
    got = r["drift"]["drift_fresh"]["ratio_l8_15_over_sparse"]["kl_arm_supa"]["ratio"]
    up = r["drift"]["drift_fresh"]["ratio_l8_15_over_sparse"]["kl_arm_supa"]["ci95"][1]
    check("C5 input: drift ratio recomputed independently == reported; the decision uses its CI upper bound",
          abs(got - want) < 1e-9 and r["decision"]["inputs"]["drift_ci_upper"] == up, f"{got:.4f} vs {want:.4f}")
    check("supa's own drift is exactly 0", r["summary"]["supa"]["drift_fresh_kl_arm_supa"] == 0.0)
    E = L.ood_env()
    bank = {x["id"]: x for x in L.load_ood_bank()}
    indep6 = {}
    for arm in L.ARMS:
        rows = L.read_jsonl(a / arm / "ood.jsonl")
        indep6[arm] = {x["id"]: E.parse_state(x["family"], x["prediction"], lenient=True) == json.loads(bank[x["id"]]["gold"])
                       for x in rows}
    c6 = r["decision"]["c6_by_stratum"]
    check("C6 on the tiny run: computed per stratum (6 strata, n = 1 each) from the paired OOD outputs; correctness "
          "recomputed independently with rcwmt.env.parse_state == reported, all arms; the report-only binding flags "
          "mark all six strata non-binding (a 1-row stratum cannot fail C6)",
          set(c6) == set(L.OOD_STRATA) and all(v["n"] == 1 for v in c6.values())
          and r["c6_binding_report_only"]["non_binding"] == list(L.OOD_STRATA)
          and all(r["summary"][arm]["ood_correct"] == sum(indep6[arm].values()) for arm in L.ARMS)
          and all(r["decision"]["inputs"]["ood_stratum"][s] == r["contrasts"]["ood_l8_15_vs_sparse"][s]
                  for s in L.OOD_STRATA),
          str({k: (v["lost"], v["gained"]) for k, v in c6.items()}))
    check("L1_RESULTS.md written with the registered rule verbatim",
          L.DECISION_RULE in (a / "L1_RESULTS.md").read_text(encoding="utf-8"))
    # C6 fixtures: a tiny-model run directory whose OOD outputs are replaced by full 600-row fixtures (stage and DONE
    # hashes updated, so analyze accepts the directory as an untampered run)
    fx_eq = c6_fixture(a, "c6_equal", lose_stratum=None)
    v = fx_eq["decision"]["c6_by_stratum"]
    S = fx_eq["summary"]
    check("C6 fixture 'L8-15 == Titan-Sparse' (both exact on all 600; Supa copies the input): C6 passes, every stratum "
          "n 100, net 0; Supa's report-only counts are the copy floor (100 on L_void / P_inapp, 0 elsewhere); every "
          "stratum is binding (Titan-Sparse 100 >= 6)",
          fx_eq["decision"]["conditions"]["C6"] is True and "C6" not in fx_eq["decision"]["failed"]
          and all(v[s]["n"] == 100 and v[s]["net"] == 0 and v[s]["pass"] for s in L.OOD_STRATA)
          and S["sparse"]["ood_correct"] == S["l8_15"]["ood_correct"] == 600
          and fx_eq["c6_binding_report_only"]["non_binding"] == []
          and all(x["can_fail"] and x["min_lost_to_fail"] == 6 for x in fx_eq["c6_binding_report_only"]["strata"].values())
          and {s: S["supa"]["ood_per_stratum"][s]["correct"] for s in L.OOD_STRATA}
          == {"L_big": 0, "L_void": 100, "R_wrap": 0, "R_boundary": 0, "P_inapp": 100, "P_appdel": 0},
          str({s: v[s]["net"] for s in L.OOD_STRATA}))
    fx_loss = c6_fixture(a, "c6_loss", lose_stratum="R_wrap")
    v = fx_loss["decision"]["c6_by_stratum"]
    md = (TMP / "c6_loss" / "L1_RESULTS.md").read_text(encoding="utf-8")
    check("C6 fixture 'L8-15 loses 10 of 100 in R_wrap': C6 fails on R_wrap only (lost 10, gained 0, net -10 < -5, "
          "detected loss p = 2/1024), the other five strata pass; L1_RESULTS.md names it",
          fx_loss["decision"]["conditions"]["C6"] is False and "C6" in fx_loss["decision"]["failed"]
          and (v["R_wrap"]["lost"], v["R_wrap"]["gained"], v["R_wrap"]["net"]) == (10, 0, -10)
          and v["R_wrap"]["detected_loss"] and not v["R_wrap"]["pass"]
          and all(v[s]["pass"] for s in L.OOD_STRATA if s != "R_wrap")
          and "C6 R_wrap (n 100): net -10" in md and "OOD R_wrap" in md and "C6 binding (report-only" in md,
          str(v["R_wrap"]))
    ok, msg = raises(lambda: analyze(a, "--no-freeze", "--allow-subset"), "create-only")
    check("analysis is create-only", ok, msg)
    t = TMP / "run_tamper"
    shutil.copytree(a, t)
    for f in ("L1_RESULTS.json", "L1_RESULTS.md"):
        (t / f).unlink()
    with open(t / "sparse" / "ret.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    ok, msg = raises(lambda: analyze(t, "--no-freeze", "--allow-subset"), "changed after the run")
    check("analysis refuses a stage file edited after the run", ok, msg)
    shutil.rmtree(t)
    rc_ = analyze(c, "--no-freeze", "--allow-subset")
    check("the HF forward path is a named comparability failure",
          any("DenseReference" in x for x in rc_["decision"]["inputs"]["comparable_reasons"]))
    o = fresh_dir("run_overflow", build_out)
    keep = (L.LC_MAX_CONTEXT, L.ACQ_MAX_NEW, dict(L._SUBSET))
    L.LC_MAX_CONTEXT, L.ACQ_MAX_NEW = 1170, 4                # row 1 fits (1145 + 24), row 2 (1152 + 24) does not
    L._SUBSET.update(drift=1)
    try:
        run(o)
    finally:
        L.LC_MAX_CONTEXT, L.ACQ_MAX_NEW = keep[0], keep[1]
        L._SUBSET.clear()
        L._SUBSET.update(keep[2])
    lc = L.read_jsonl(o / "supa" / "lc.jsonl")
    ro = analyze(o, "--no-freeze", "--allow-subset")
    check("a long-context row that does not fit is recorded as skipped (never truncated); it labels the report-only "
          "long-context numbers non-comparable and is NOT a C0 reason",
          [x["status"] for x in lc] == ["ok", "skipped_context_overflow"] and ro["long_context_comparable"] is False
          and ro["decision"]["inputs"]["comparable_reasons"] == ["row subset (test run)"],
          str([x["input_tokens"] for x in lc]))


def c6_fixture(a: Path, name: str, lose_stratum: str | None, lose_n: int = 10) -> dict:
    """Copy the tiny run `a`, replace every arm's ood.jsonl by a full 600-row fixture (Titan-Sparse exact on every row,
    L8-15 identical except `lose_n` rows of `lose_stratum` answered '{}', Supa copying the input state), update the
    STAGES / DONE hashes so analyze treats the directory as an untampered run, and analyze it."""
    d = TMP / name
    shutil.copytree(a, d, ignore=shutil.ignore_patterns("L1_RESULTS.*"))
    E = L.ood_env()
    bank = L.load_ood_bank()
    lose = {x["id"] for x in [b for b in bank if b["stratum"] == lose_stratum][:lose_n]} if lose_stratum else set()

    def rec(row, text):
        return {"id": row["id"], "stratum": row["stratum"], "family": row["family"], "input_tokens": 1,
                "generated_tokens": 1, "terminated": True, "prediction": text}
    outs = {"supa": [rec(b, E.dumps(b["before"])) for b in bank], "sparse": [rec(b, b["gold"]) for b in bank],
            "l8_15": [rec(b, "{}" if b["id"] in lose else b["gold"]) for b in bank]}
    done = L.read_json(d / "DONE.json")
    for arm, recs in outs.items():
        L._write_jsonl(d / arm / "ood.jsonl", recs)
        h = sha(d / arm / "ood.jsonl")
        st = L.read_json(d / arm / "STAGES.json")
        st["stages"]["ood"]["sha256"]["ood.jsonl"] = h
        L.write_json(d / arm / "STAGES.json", st)
        done["arms"][arm]["files_sha256"]["ood.jsonl"] = h
    L.write_json(d / "DONE.json", done)
    return analyze(d, "--no-freeze", "--allow-subset")


def t8_inputs():
    print("8 inputs", flush=True)
    bank = L.load_ood_bank()
    src = L.read_json(L.resolve("ood_bank_source"))
    E = L.ood_env()
    check("OOD bank: 600 = 6 strata x 100, pinned sha == SOURCE record, built with no model call, every recorded "
          "overlap 0 (payload and operative content, 10 sources incl. the historical set and SELECT-VIEWS-001), every "
          "row in Titan's one-step frame with oracle gold",
          len(bank) == 600 and sha(L.resolve("ood_bank")) == L.SHA["ood_bank"] == src["file_sha256"]
          and src["model_calls"] == 0 and len(src["overlap_counts"]) == 10
          and all(v == {"payload": 0, "operative": 0} for v in src["overlap_counts"].values())
          and all(b["messages"] == E.one_step_messages(b["family"], b["before"], b["event"])
                  and b["gold"] == E.dumps(E.step(b["family"], b["before"], b["event"])) for b in bank),
          f"{src['exclusion_union']}")
    fr = L.load_prompts("fresh")
    used = L.load_prompts("used")
    br = L.load_prompts("bridge")
    pools = L.hskill_pools()
    sl = {r["id"]: r for r in L.read_jsonl(L.pinned("slice"))}
    tf_used = {sl[u["id"]]["metadata"].get("template_family") for u in used}
    per = {}
    for r in fr:
        per[r["reply_class"]] = per.get(r["reply_class"], 0) + 1
    check("fresh prompts: 456 = 57 classes x 8; disjoint in id, template family and text from the 248; no id or user "
          "text of the HSKILL-002 pools; bridge = 228 of the 248, none in the pools",
          len(fr) == 456 == L.FRESH_N and len(per) == 57 and set(per.values()) == {8}
          and not ({r["id"] for r in fr} & {u["id"] for u in used})
          and not ({r["template_family"] for r in fr} & tf_used)
          and not ({L.canonical(r["messages"]) for r in fr} & {L.canonical(u["messages"]) for u in used})
          and not any(L.in_pools(r, pools) for r in fr)
          and len(br) == 228 == L.BRIDGE_N and not any(L.in_pools(u, pools) for u in br))
    src = L.vendor_forward()["source"]
    c = src.source_config(L.read_json(REAL_SUPA / "config.json"))
    ok, msg = raises(lambda: src.source_config(CONFIG_JSON), "", ValueError)
    check("the programme's official-shape check accepts Supa 1.7's config (28 x 1024, plain) and refuses the tiny one",
          c.encoder_layers == 28 and c.hidden_size == 1024 and c.architecture == "plain" and ok, msg)


def t9_subprocesses():
    print("9 subprocesses", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([PY, "-B", str(HERE / "l1_screen.py"), "plan"], capture_output=True, text=True, env=env,
                       timeout=300, encoding="utf-8")
    plan = json.loads(p.stdout[p.stdout.index("{"):])
    check("`plan` imports no torch and prices one lease of three arms, the 600 OOD rows included",
          p.returncode == 0 and plan["torch_imported"] is False and plan["one_lease"] == L.LEASE_LABEL
          and plan["rows_per_arm"]["drift"] == 684 and plan["rows_per_arm"]["ood"] == 600
          and plan["generations_per_arm"] == 1390 and plan["streamed_loads"] == 4,
          f"expected {plan['gpu_hours_3_arms']['expected']} h")
    out = TMP / "wait_out"
    p = subprocess.run([PY, "-B", str(HERE / "l1_screen.py"), "run", "--device", "cpu", "--wait-for",
                        str(TMP / "never_appears"), "--wait-max-h", "0.0001", "--wait-poll", "0.1", "--out-dir", str(out)],
                       capture_output=True, text=True, env=env, timeout=300, encoding="utf-8")
    check("`run --wait-for <missing> --wait-max-h` exits 4 at script start, creating nothing",
          p.returncode == 4 and not out.exists(), p.stdout.strip()[-120:])
    flag = TMP / "appears_later"
    threading.Timer(0.4, lambda: flag.write_text("x")).start()
    t0 = time.time()
    check("in-process gate: a file appearing later releases the wait",
          L.wait_for_files([str(flag)], poll_s=0.1, max_h=0.01) and time.time() - t0 < 30)


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    global TMP, R14
    t_start = time.time()
    TMP = Path(tempfile.mkdtemp(prefix="l1_tiny_"))
    print(f"temp dir {TMP}", flush=True)
    real_out_default_exists = L.OUT_DEFAULT.exists()
    real_lock = Path(LV.LOCK)
    saved = (dict(L.PATHS), dict(L.SHA), L.r14_hashes, L._DENSE_CONFIG_HOOK, L._SUBSET)
    try:
        cores = {arm: make_core(TMP / f"core_{arm}", arm) for arm in ("supa", "sparse")}
        R14 = {"supa": RP.group_hashes(build("supa")), "sparse": RP.group_hashes(build("sparse")),
               "l8_15": RP.group_hashes(assemble(range(8, 16)))}
        L.r14_hashes = lambda: {**R14, "loc8_15_fingerprint": None, "loc8_15_titan_blocks": [8, 15]}
        (TMP / "runs_root_empty").mkdir()
        L.PATHS.update(supa_dir=cores["supa"], sparse_dir=cores["sparse"], expected=TMP / "L8_15_EXPECTED.json",
                       freeze=TMP / "PREREG_L1_FREEZE.json", prereg=TMP / "PREREG_L1_DRAFT.md",
                       runs_root=TMP / "runs_root_empty")
        (TMP / "PREREG_L1_DRAFT.md").write_text("tiny-test stand-in for the preregistration\n", encoding="utf-8")
        L.SHA.update(supa_weights=sha(cores["supa"] / "model.safetensors"),
                     sparse_weights=sha(cores["sparse"] / "model.safetensors"))
        L._DENSE_CONFIG_HOOK = dense_hook
        L._SUBSET = dict(SUBSET)
        t1_pure()
        build_out = t2_build(cores)
        a = t3_run(build_out)
        c = t4_hf(a, build_out)
        t5_cuda_wiring(a, build_out)
        t6_freeze(a, build_out)
        t7_analyze(a, c, build_out)
        t8_inputs()
        t9_subprocesses()
        check("the test never created the real output directory runs/l1_screen",
              L.OUT_DEFAULT.exists() == real_out_default_exists)
        check("the lease module points at the real lock path again (the test used a temp-dir lock only)",
              Path(LV.LOCK) == real_lock)
        check("CUDA was never initialised", not torch.cuda.is_initialized())
    finally:
        L.PATHS.clear()
        L.PATHS.update(saved[0])
        L.SHA.clear()
        L.SHA.update(saved[1])
        L.r14_hashes, L._DENSE_CONFIG_HOOK, L._SUBSET = saved[2], saved[3], saved[4]
        if not KEEP:
            shutil.rmtree(TMP, ignore_errors=True)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\nl1_test_tiny: {n_ok}/{len(RESULTS)} checks pass in {time.time() - t_start:.0f}s"
          + (f"; kept {TMP}" if KEEP else ""), flush=True)
    return 0 if n_ok == len(RESULTS) else 1


TMP: Path
R14: dict

if __name__ == "__main__":
    sys.exit(main())
