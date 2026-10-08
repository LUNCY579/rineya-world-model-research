"""Block-localised Titan deltas: off-task drift against the transition skill (exploratory, R12 follow-up).

Question. RESULTS_PILOT R12: Supa + Titan-Sparse's delta restricted to blocks 8-15 (= run_patch HlowT@16)
recovers the one-step skill in every family, while Supa + the delta in blocks 16-27 only (= HupT@16)
recovers none of it, although blocks 16-27 hold 59% of the delta energy. Does the dispensable upper part
carry off-task drift (the representational side of forgetting)? If it does, a block-localised delta keeps
the skill with less interference. Analysis: scripts/localized_analysis.py (CPU, no model).

Arms. Titan-Sparse = Supa + a delta that is exactly zero in the embeddings, the final norm and blocks 0-7
(D-21). Every arm is "Supa with Titan's blocks in a range":
  supa      none                      (load fingerprint a9952c55f5a1 asserted)
  loc8_15   Titan's blocks 8..15      (= HlowT@16: Supa + Titan's blocks 0..15, 0..7 being identical)
  sparse    Titan's blocks 8..27      (= Titan-Sparse, blocks 0..7 being identical; built fingerprint
                                       818be9c6b6a3 asserted)
  up16_27   Titan's blocks 16..27     (= HupT@16, there built on a Titan host; here on Supa, same weights)
ONE real model in memory: Supa is loaded once (runtime.load_arm, FP32, TF32 off) and every arm is built IN
PLACE with run_patch's splice utilities (set_block_sources / splice_blocks: per-tensor safetensors reads; a
Titan FP32 tensor is copied as is, a Supa BF16 tensor upcast exactly as the FP32 load does), in the order
supa -> loc8_15 -> sparse -> up16_27 (8 + 12 + 8 = 28 block copies). Configs and generation configs of the
two cores are identical apart from torch_dtype, so a hybrid on a Supa host equals one on a Titan host.

Per arm, into runs/<bank>/<out-name>/<arm>/ (each stage resumable on its own):
  offtask      rcwmt.offtask.capture_offtask: the 248 neutral prompts; first-token log-probs (fp16, clamped
               at -60) and answer-slot states at capture.LAYERS -> offtask_logprobs.npy, offtask_last.npy,
               offtask_meta.json (written last by capture_offtask; deleted before the stage starts)
  offtask_gen  64-token greedy replies to the same prompts through run_t2.gen_prompts on
               generation.render(messages): run_t2's offtask_gen (same rendering, batching, EOS cut, strip)
               -> offtask_gen.jsonl {id, reply_class, text}
  chain_one    rcwmt.generation.chain(mode="chain_one") on the 640 pilot TEST chains (run_t2's list: every
               candidate of every TEST parent, bank order, no twins) -> gen_chain_one.jsonl, written by
               run_t2._write_chains ({cand_id, states, texts}, as run_t2 / run_arm)
JSONL outputs are written to a .tmp file and renamed. STAGES.json records each finished stage with the arm's
fingerprint and the sha256 of its files; a stage counts as done only if that record, the fingerprint in
ARM.json and the file hashes all agree. DONE.json is written once, at the end, when all four arms have all
three stages; a rerun of a finished directory loads nothing.

BUILD GATES (fatal: SanityError, exit 3, no stage of the arm runs). G1 and G2 run before the lease.
  G1 source hashes: per-group sha1 of both cores read from their safetensors files (file_group_hashes:
     byte-for-byte the digest run_patch.group_hashes takes on a loaded model; plain buffered reads of
     <= 64 MiB, no mmap).
     When run_patch's patch_results.json (or patch_progress.json) exists, both must equal its recorded
     _meta.hash_supa / hash_sparse (hashes of genuine loads of both cores in the R12 run). After the load,
     the loaded Supa's group_hashes must equal the file-derived Supa hashes.
  G2 D-21 premise: the cores differ in exactly blocks 8..27 (embeddings, final norm, blocks 0..7 equal).
  G3 splice: every spliced block hash-equals the source core's block (run_patch.set_block_sources).
  G4 composition: after each build, EVERY parameter group of the model hash-equals the core it must hold
     (Titan for blocks in the arm's range, Supa otherwise), so loc8_15 / up16_27 / sparse differ from Supa
     in exactly blocks 8-15 / 16-27 / 8-27.
  G5 fingerprints: runtime.param_fingerprint of the supa arm a9952c55f5a1 and of the built sparse arm
     818be9c6b6a3 (run_patch.EXPECTED_FP; the latter equals a genuine Titan-Sparse load).
  On resume an arm must rebuild to the fingerprint recorded in its ARM.json.
REPRODUCTION CHECK (not fatal: WARNING and recorded in ARM.json "repro"). If <patch-dir>/patch_results.json
(or patch_progress.json) and the arm's reference file exist (supa gen_base_supa.json, sparse
gen_base_sparse.json, loc8_15 gen_HlowT_16.json, up16_27 gen_HupT_16.json), the FIRST batch of each family
of that run is regenerated exactly as run_patch did (its _meta items, batch size and max_new_tokens;
run_patch.batches_of / run_batch, so the left-padded batch composition is identical) and the texts of the
first --repro-per-family (4) items per family must equal run_patch's texts exactly; the other items of those
batches are recorded too. Done once per arm and fingerprint (reused on resume).

Cost (--plan prints it; real pilot): per arm 248 off-task replies + <= 4 x 640 chain steps + 48 reproduction
generations. Upper bound 4 x 2856 = 11,424 greedy generations = 1.18 h at 2.7 generations/s. Expected
(conservative) ~11,200: Titan's stored chains attempt 2,445 of 2,560 steps (used for sparse and loc8_15);
Supa's stored chains attempt only 1,134, but they predate the 160-token cap and the D-14 batch budget (60 of
their texts are truncated), so supa and up16_27 are taken at the full 2,560. Plus ~5 min of loading, hashing
and off-task forward passes: plan for ~1.25 h of GPU time (not D-23's ~40 min). Disk ~0.35 GB.

SCHEDULING GATE (D-23: start only after the T2 s0 pair has finished): --wait-for PATH (repeatable) blocks
until every PATH exists, polling every --wait-poll seconds (60), for at most --wait-max-h hours (0 = no
limit; on expiry exit 4 without loading or writing anything). Started as a script the wait runs before torch
is imported; the lease is requested only after it.

CPU path (--device cpu): for the tiny test only (scripts/localized_test_tiny.py). The production
instruments call .cuda() and pace the GPU with runtime.breathe / throttle_list; on CPU, cpu_shims() makes
.cuda() return the tensor itself and the pacing hooks no-ops for the duration of the run, so the same code
runs without touching the driver. CUDA is hidden before torch is imported (empty string, never "-1";
DECISIONS I-21).

  python scripts/run_localized.py --device cuda --out-name localized --wait-for runs/t2/t2_s0/eval.json
                                                                   (one lease, rcwm-t1:localized, after T2 s0)
  python scripts/run_localized.py --device cuda --plan             (prints the plan, loads nothing)
"""
from __future__ import annotations

import os
import sys


def _argv_value(flag: str, default: str | None = None) -> str | None:
    for i, x in enumerate(sys.argv):
        if x == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return default


def _argv_values(flag: str) -> list[str]:
    out = []
    for i, x in enumerate(sys.argv):
        if x == flag and i + 1 < len(sys.argv):
            out.append(sys.argv[i + 1])
        elif x.startswith(flag + "="):
            out.append(x.split("=", 1)[1])
    return out


def wait_for_files(paths, poll_s: float = 60.0, max_h: float = 0.0, report_every_s: float = 1800.0) -> bool:
    """D-23 scheduling gate: block until every path exists (relative paths are taken from the RCWM-T1
    directory). Standard library only: started as a script, this runs BEFORE torch is imported, so a waiting
    process holds a few MB of RAM and never touches the driver. Returns False (caller exits 4, nothing loaded
    or written) if max_h > 0 hours pass first; max_h = 0 waits indefinitely."""
    import time as _t
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    ps = [p if p.is_absolute() else root / p for p in (_P(x) for x in paths)]
    t0 = last = _t.time()
    missing = [p for p in ps if not p.exists()]
    if missing:
        print(f"[{_t.strftime('%H:%M:%S')}] gate: waiting for {[str(p) for p in missing]} (poll {poll_s:g}s, "
              f"max {'unlimited' if max_h <= 0 else f'{max_h:g} h'})", flush=True)
    while missing:
        if max_h > 0 and _t.time() - t0 >= max_h * 3600:
            print(f"[{_t.strftime('%H:%M:%S')}] gate: still missing {[str(p) for p in missing]} after {max_h:g} h; "
                  f"exiting without loading anything", flush=True)
            return False
        _t.sleep(poll_s)
        missing = [p for p in ps if not p.exists()]
        if missing and _t.time() - last >= report_every_s:
            last = _t.time()
            print(f"[{_t.strftime('%H:%M:%S')}] gate: still waiting ({(_t.time() - t0) / 3600:.2f} h)", flush=True)
    if ps:
        print(f"[{_t.strftime('%H:%M:%S')}] gate: present: {[str(p) for p in ps]} (waited {(_t.time() - t0) / 60:.1f} min)",
              flush=True)
    return True


# D-23 gate before anything heavy is imported (script start only; main() re-checks, instantly if present).
if __name__ == "__main__" and "--plan" not in sys.argv and _argv_values("--wait-for"):
    if not wait_for_files(_argv_values("--wait-for"), float(_argv_value("--wait-poll", "60")),
                          float(_argv_value("--wait-max-h", "0"))):
        sys.exit(4)

# CPU mode: hide CUDA before torch is imported. Empty string, NEVER "-1" (DECISIONS I-21).
_CPU = _argv_value("--device") == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
if _CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = str(_argv_value("--threads", "2"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

if _CPU:                                           # never query the driver (I-21)
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_patch as P  # noqa: E402  (decides CPU mode from the same sys.argv / environment)
import run_t2  # noqa: E402
from rcwmt import capture, env, generation, offtask, runtime  # noqa: E402

N_LAYERS = 28
ARM_ORDER = ("supa", "loc8_15", "sparse", "up16_27")       # build order: 8 + 12 + 8 block copies
# [lo, hi). sparse = Titan's blocks 8..27: blocks 0..7 (and the embeddings / final norm) are bit-identical in
# the two cores (gate G2), so this IS Titan-Sparse (G5 asserts its fingerprint 818be9c6b6a3).
TITAN_RANGE = {"supa": (0, 0), "loc8_15": (8, 16), "sparse": (8, N_LAYERS), "up16_27": (16, N_LAYERS)}
DELTA_BLOCKS = tuple(range(8, N_LAYERS))                   # D-21: the cores differ in exactly these blocks
REF_FILES = {"supa": "gen_base_supa.json", "sparse": "gen_base_sparse.json",
             "loc8_15": "gen_HlowT_16.json", "up16_27": "gen_HupT_16.json"}
STAGES = ("offtask", "offtask_gen", "chain_one")
STAGE_FILES = {"offtask": ("offtask_meta.json", "offtask_logprobs.npy", "offtask_last.npy"),
               "offtask_gen": ("offtask_gen.jsonl",), "chain_one": ("gen_chain_one.jsonl",)}
LEASE_LABEL = "rcwm-t1:localized"
LEASE_JOB = "RCWM-T1 block-localised delta arms (supa, loc8_15, sparse, up16_27): off-task drift vs skill"
PATCH_DIR = ROOT / "runs" / "bank_pilot" / "patch"
HASH_CHUNK = 1 << 24                               # elements per hashed read (<= 64 MiB of FP32)
RUN_KEYS = ("bank", "bank_sha256", "n_chains", "chain_ids_sha256", "n_prompts", "prompt_ids_sha256",
            "offtask_max_new", "chain_max_tokens")   # must not change between invocations on one directory

_on_stage_done = None                              # test hook: callable(arm, stage) after each stage


def log(msg: str) -> None:
    P.log(msg)


def _write_json(path: Path, obj) -> None:
    P._write_json(path, obj)


def _read_json(path: Path):
    return P._read_json(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def block_index(group: str) -> int | None:
    return int(group[5:]) if group.startswith("block") else None


def group_sort(groups) -> list[str]:
    return sorted(groups, key=lambda g: (block_index(g) is None, block_index(g) or 0, g))


def titan_blocks(arm: str, n: int) -> list[int]:
    lo, hi = TITAN_RANGE[arm]
    return list(range(lo, min(hi, n)))


# ------------------------------------------------------------------------------------------ inputs
def load_chains(bank: dict) -> list[dict]:
    """run_t2._eval_body's chain list: every candidate of every pilot TEST parent, bank order, no twins
    (test hook: replaced by a subset)."""
    return [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"]}
            for p in bank["parents"] if p["split"] == "test" for c in p["candidates"]]


def patch_meta(patch_dir: Path):
    """run_patch's _meta (items, batch, max_new_tokens, recorded group hashes), or None."""
    for name in ("patch_results.json", "patch_progress.json"):
        r = _read_json(patch_dir / name)
        if r and "_meta" in r:
            return {**r["_meta"], "_file": str(patch_dir / name)}
    return None


# ------------------------------------------------------------------------------------- hashing
_ST_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


def _fp32_bytes(buf: bytes, dtype: str) -> bytes:
    """Little-endian stored values -> the FP32 bytes an FP32 load holds (BF16 -> FP32 is a 16-bit shift)."""
    if dtype == "F32":
        return buf
    if dtype == "BF16":
        return (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).tobytes()
    return np.frombuffer(buf, dtype="<f2").astype("<f4").tobytes()


def file_group_hashes(arm: str) -> dict:
    """sha1 per parameter group (block i, embeddings, final norm) of a core's stored tensors upcast to FP32:
    byte-for-byte the digest run_patch.group_hashes computes on the loaded model (same names in sorted order,
    the same FP32 bytes; sha1 is streamed, so chunking does not change it). The safetensors header is parsed
    and the data read with plain buffered reads of <= HASH_CHUNK elements (no mmap, no torch), so host RAM
    holds one chunk at a time. Keys outside 'model.' (an untied head) are not parameters of the tied model."""
    path = P.weight_file(arm)
    with open(path, "rb") as fh:
        n_hdr = int.from_bytes(fh.read(8), "little")
        hdr = {k: v for k, v in json.loads(fh.read(n_hdr)).items() if k != "__metadata__"}
        base = 8 + n_hdr
        groups: dict[str, list[str]] = {}
        for k in hdr:
            if k.startswith("model."):
                groups.setdefault(P._group_key(k), []).append(k)
        out = {}
        for key in group_sort(groups):
            h = hashlib.sha1()
            for name in sorted(groups[key]):
                info = hdr[name]
                dt, (a, b) = info["dtype"], info["data_offsets"]
                if dt not in _ST_BYTES:
                    raise P.SanityError(f"{path}: {name} has dtype {dt}; only {sorted(_ST_BYTES)} are hashed")
                size = _ST_BYTES[dt]
                n_el = (b - a) // size
                if n_el * size != b - a or n_el != int(np.prod(info["shape"], dtype=np.int64)):
                    raise P.SanityError(f"{path}: {name} byte range does not match its shape")
                h.update(name.encode())
                fh.seek(base + a)
                for s in range(0, n_el, HASH_CHUNK):
                    k = min(HASH_CHUNK, n_el - s)
                    buf = fh.read(k * size)
                    if len(buf) != k * size:
                        raise P.SanityError(f"{path}: short read in {name}")
                    h.update(_fp32_bytes(buf, dt))
            out[key] = h.hexdigest()[:16]
    return out


def source_hashes(pmeta) -> tuple[dict, dict]:
    """G1 (file side) and G2, before the lease. Returns ({core: group hashes}, record)."""
    t0 = time.time()
    F = {arm: file_group_hashes(arm) for arm in ("supa", "sparse")}
    rec = {"file_hash_seconds": round(time.time() - t0, 1), "groups": group_sort(F["supa"])}
    if set(F["supa"]) != set(F["sparse"]):
        raise P.SanityError("the two cores' files hold different parameter groups")
    if pmeta and "hash_supa" in pmeta and "hash_sparse" in pmeta:
        for arm in ("supa", "sparse"):
            want = pmeta[f"hash_{arm}"]
            bad = group_sort(g for g in set(F[arm]) | set(want) if F[arm].get(g) != want.get(g))
            if bad:
                raise P.SanityError(f"G1: {arm}'s file-derived group hashes differ from run_patch's recorded hashes "
                                    f"({pmeta['_file']}) in {bad[:6]}")
        rec["equal_to_run_patch_recorded"] = pmeta["_file"]
    else:
        rec["equal_to_run_patch_recorded"] = "not available (no run_patch _meta)"
    diff = group_sort(g for g in F["supa"] if F["supa"][g] != F["sparse"][g])
    want = [f"block{i}" for i in DELTA_BLOCKS]
    if diff != want:
        raise P.SanityError(f"G2: the cores differ in {diff}, not in exactly blocks {DELTA_BLOCKS[0]}..{DELTA_BLOCKS[-1]} "
                            f"(D-21 premise)")
    rec["cores_differ_in"] = diff
    return F, rec


# ---------------------------------------------------------------------------------------- backend
class Backend(P.Backend):
    """run_patch's Backend (load = runtime.load_arm + fingerprint check on cuda; throttle + breathe in
    run_batch) with this job's lease label."""

    def lease(self):
        return runtime.gpu_lease(label=LEASE_LABEL, job=LEASE_JOB) if self.cuda else nullcontext()


@contextmanager
def cpu_shims(active: bool):
    """--device cpu: the production instruments move tensors with .cuda() and pace the GPU with
    runtime.breathe / runtime.throttle_list (torch.cuda.synchronize, nvidia-smi). For the duration of the
    run, .cuda() returns the tensor itself and the pacing hooks are no-ops. Inactive on --device cuda."""
    if not active:
        yield
        return
    from transformers import LogitsProcessorList
    had_cuda = "cuda" in torch.Tensor.__dict__
    saved_cuda = torch.Tensor.__dict__.get("cuda")
    saved = {(m, k): getattr(m, k) for m in (runtime, generation) for k in ("breathe", "throttle_list")}
    torch.Tensor.cuda = lambda self, *a, **k: self
    for m in (runtime, generation):
        m.breathe = lambda: None
        m.throttle_list = lambda: LogitsProcessorList([])
    try:
        yield
    finally:
        if had_cuda:
            torch.Tensor.cuda = saved_cuda
        else:
            del torch.Tensor.cuda
        for (m, k), v in saved.items():
            setattr(m, k, v)


# ------------------------------------------------------------------------------------------ build
def build_arm(model, arm: str, cur: list, hashes: dict, be) -> dict:
    """Splice the arm in place (G3), then G4 (composition) and G5 (fingerprints)."""
    n = model.config.num_hidden_layers
    tb = set(titan_blocks(arm, n))
    want = ["sparse" if i in tb else "supa" for i in range(n)]
    t0 = time.time()
    spliced = P.set_block_sources(model, cur, want, hashes, be.dev)          # G3 (hard gate inside)
    G = P.group_hashes(model)
    expect = {g: hashes[want[block_index(g)] if block_index(g) is not None else "supa"][g] for g in hashes["supa"]}
    bad = group_sort(g for g in set(G) | set(expect) if G.get(g) != expect.get(g))
    if bad:
        raise P.SanityError(f"G4 {arm}: groups {bad[:6]} do not hash-equal the core they must hold")
    diff_supa = group_sort(g for g in G if G[g] != hashes["supa"][g])
    want_diff = [f"block{i}" for i in sorted(tb) if i in DELTA_BLOCKS]
    if diff_supa != want_diff:
        raise P.SanityError(f"G4 {arm}: differs from Supa in {diff_supa}, expected {want_diff}")
    fp = runtime.param_fingerprint(model)
    exp_fp = P.EXPECTED_FP.get(arm)
    if exp_fp is not None and not fp.startswith(exp_fp):
        raise P.SanityError(f"G5 {arm}: fingerprint {fp[:12]} != expected {exp_fp}")
    rec = {"arm": arm, "host": "supa", "titan_blocks": [min(tb), max(tb)] if tb else [],
           "spliced_in_this_build": spliced, "build_seconds": round(time.time() - t0, 1),
           "differs_from_supa": diff_supa,
           "differs_from_sparse": group_sort(g for g in G if G[g] != hashes["sparse"][g]),
           "group_hashes": G, "fingerprint": fp, "fingerprint_expected": exp_fp,
           "gates": ["G3 spliced blocks hash-equal the source core", "G4 every group hash-equals its intended core",
                     "G5 fingerprint" + (f" == {exp_fp}" if exp_fp else " recorded")]}
    log(f"built {arm}: spliced {spliced or 'nothing'}; differs from Supa in "
        f"{(diff_supa[0] + '..' + diff_supa[-1]) if diff_supa else 'nothing'}; fingerprint {fp[:12]}"
        + (f" (== expected {exp_fp})" if exp_fp else ""))
    return rec


def repro_check(be, model, arm: str, tok, patch_dir: Path, pmeta, per_family: int, fp: str) -> dict:
    """Regenerate the first run_patch batch of each family and compare with run_patch's texts (module doc)."""
    ref_path = patch_dir / REF_FILES[arm]
    if pmeta is None or not ref_path.exists():
        return {"status": "skipped", "reason": f"no run_patch _meta or no {ref_path.name} in {patch_dir}", "fingerprint": fp}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    pool = {it["item_id"]: it for it in P.load_pool(pmeta.get("bank", "bank_pilot"))}
    missing = [i for i in pmeta["items"] if i not in pool]
    if missing:
        return {"status": "skipped", "reason": f"{len(missing)} run_patch items not in the pool", "fingerprint": fp}
    items = [pool[i] for i in pmeta["items"]]
    mx = pmeta["max_new_tokens"]
    batches = P.batches_of(items, int(pmeta["batch"]))
    t0 = time.time()
    got, gate = {}, []
    for fam in env.FAMILIES:
        b = next((b for b in batches if b[0]["family"] == fam), None)
        if b is None:
            continue
        enc = {it["item_id"]: tok(generation.render(tok, it["messages"]), add_special_tokens=False)["input_ids"] for it in b}
        o = P.run_batch(be, model, b, enc, int(mx[fam] if isinstance(mx, dict) else mx))
        got.update(o["texts"])
        gate += [it["item_id"] for it in items if it["family"] == fam][:per_family]
    differ = [i for i in got if got[i] != ref.get(i)]
    gdiff = [i for i in gate if got.get(i) != ref.get(i)]
    rec = {"status": "exact" if not gdiff else "MISMATCH", "reference": str(ref_path),
           "run_patch_meta": pmeta["_file"], "batch": int(pmeta["batch"]), "max_new_tokens": mx,
           "first_per_family": per_family, "gate_n": len(gate), "gate_equal": len(gate) - len(gdiff),
           "gate_differ": gdiff, "batch_items_compared": len(got), "batch_items_equal": len(got) - len(differ),
           "batch_differ": differ, "differ_texts": {i: {"got": got[i], "ref": ref.get(i)} for i in differ[:12]},
           "seconds": round(time.time() - t0, 1), "fingerprint": fp}
    msg = (f"repro {arm} vs {ref_path.name}: first {per_family}/family {rec['gate_equal']}/{rec['gate_n']} equal, "
           f"whole first batches {rec['batch_items_equal']}/{rec['batch_items_compared']}")
    log(msg if not gdiff else "WARNING " + msg + f" (differ: {gdiff}); recorded, continuing")
    return rec


# ----------------------------------------------------------------------------------------- stages
def stage_done(d: Path, stage: str) -> bool:
    st, arm_rec = _read_json(d / "STAGES.json"), _read_json(d / "ARM.json")
    if not (st and arm_rec and st.get("fingerprint") == arm_rec.get("fingerprint") and stage in st.get("stages", {})):
        return False
    files = st["stages"][stage].get("sha256", {})
    return all((d / f).exists() and files.get(f) == sha256_file(d / f) for f in STAGE_FILES[stage])


def run_stage(stage: str, model, tok, d: Path, chains: list[dict], a) -> dict:
    t0 = time.time()
    if stage == "offtask":
        (d / "offtask_meta.json").unlink(missing_ok=True)         # capture_offtask writes it last
        meta = offtask.capture_offtask(model, tok, d)
        info = {"n_prompts": meta["n"], "layers": meta["layers"], "prompts_sha256": meta["prompts_sha256"]}
    elif stage == "offtask_gen":
        ot = offtask.load_prompts()
        texts = run_t2.gen_prompts(model, tok, [generation.render(tok, r["messages"]) for r in ot], max_new=a.offtask_max_new)
        tmp = d / "offtask_gen.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for r, t in zip(ot, texts):
                fh.write(json.dumps({"id": r["id"], "reply_class": r["reply_class"], "text": t}) + "\n")
        os.replace(tmp, d / "offtask_gen.jsonl")
        info = {"n_prompts": len(ot), "max_new_tokens": a.offtask_max_new, "generations": len(ot)}
    elif stage == "chain_one":
        rr = generation.chain(model, tok, chains, "chain_one")
        tmp = d / "gen_chain_one.jsonl.tmp"
        run_t2._write_chains(tmp, chains, rr)
        os.replace(tmp, d / "gen_chain_one.jsonl")
        steps = sum(len(r["texts"]) for r in rr)
        info = {"n_chains": len(chains), "generations": steps,
                "complete_chains": sum(r["states"][-1] is not None for r in rr)}
    else:
        raise ValueError(stage)
    secs = time.time() - t0
    info.update(seconds=round(secs, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                sha256={f: sha256_file(d / f) for f in STAGE_FILES[stage]})
    if info.get("generations"):
        info["generations_per_s"] = round(info["generations"] / max(secs, 1e-9), 3)
    return info


# -------------------------------------------------------------------------------------------- plan
def plan(a, chains, prompts, pmeta) -> dict:
    steps_max = sum(len(c["events"]) for c in chains)
    stored = {}
    for arm in ("supa", "sparse"):
        p = ROOT / "runs" / a.bank / arm / "gen_chain_one.jsonl"
        if p.exists():
            stored[arm] = sum(len(json.loads(l)["texts"]) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    # Conservative: Titan's stored step count for the arms that carry blocks 8-15 (its L/R answers are <= 34
    # characters, unaffected by the token cap); the full count for supa and up16_27, because Supa's stored
    # chains predate the 160-token cap and the D-14 batch budget (truncated texts end chains early).
    exp_steps = {"supa": steps_max, "loc8_15": stored.get("sparse", steps_max),
                 "sparse": stored.get("sparse", steps_max), "up16_27": steps_max}
    repro = 0                                        # first run_patch batch of each family, per arm
    if pmeta is not None:
        pool = {it["item_id"]: it for it in P.load_pool(pmeta.get("bank", a.bank))}
        if all(i in pool for i in pmeta["items"]):
            firsts = {}
            for b in P.batches_of([pool[i] for i in pmeta["items"]], int(pmeta["batch"])):
                firsts.setdefault(b[0]["family"], len(b))
            repro = sum(firsts.values())
    gens_max = sum(len(prompts) + steps_max + repro for arm in ARM_ORDER)
    gens_exp = sum(len(prompts) + exp_steps[arm] + repro for arm in ARM_ORDER)
    gens_low = gens_exp - (exp_steps["supa"] - stored.get("supa", exp_steps["supa"]))
    overhead_h = 5 / 60
    tb = {arm: ([titan_blocks(arm, N_LAYERS)[0], titan_blocks(arm, N_LAYERS)[-1]] if titan_blocks(arm, N_LAYERS) else [])
          for arm in ARM_ORDER}
    out = {"arms": list(ARM_ORDER), "titan_blocks": tb,
           "n_prompts": len(prompts), "n_chains": len(chains), "chain_steps_max_per_arm": steps_max,
           "stored_chain_steps": stored, "expected_chain_steps": exp_steps,
           "expected_note": "supa and up16_27 at the full step count: Supa's stored chains predate the 160-token cap "
                            "and the D-14 batch budget; generations_low uses Supa's stored count instead",
           "repro_generations_per_arm": repro, "offtask_gen_per_arm": len(prompts),
           "offtask_forward_batches_per_arm": -(-len(prompts) // 8),
           "generations_upper_bound": gens_max, "generations_expected": gens_exp, "generations_low": gens_low,
           "rate_gens_per_s": a.rate,
           "hours_upper_bound": round(gens_max / a.rate / 3600, 2), "hours_expected": round(gens_exp / a.rate / 3600, 2),
           "hours_low": round(gens_low / a.rate / 3600, 2), "overhead_hours": round(overhead_h, 2),
           "plan_for_hours_incl_overhead": round(gens_max / a.rate / 3600 + overhead_h, 2),
           "overhead_note": "~5 min: one Supa load, file hashing of both cores (~3.6 GB read), 4 full group-hash "
                            "passes, 28 block copies, 4 x 31 off-task forward batches",
           "gate": {"wait_for": a.wait_for or [], "note": "D-23: launch with --wait-for runs/t2/t2_s0/eval.json"},
           "disk_GiB_estimate": round(4 * (len(prompts) * (151936 + len(capture.LAYERS) * 1024) * 2) / 2 ** 30 + 0.01, 3)}
    print(json.dumps(out, indent=1), flush=True)
    return out


# -------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--out-name", default="localized")
    ap.add_argument("--out-dir", default=None, help="override runs/<bank>/<out-name>")
    ap.add_argument("--arms", default=",".join(ARM_ORDER), help="subset of " + ",".join(ARM_ORDER) + " (built in that order)")
    ap.add_argument("--patch-dir", default=str(PATCH_DIR), help="run_patch output with the reference texts and hashes")
    ap.add_argument("--repro-per-family", type=int, default=4)
    ap.add_argument("--offtask-max-new", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2, help="cpu only")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit (loads no model, writes nothing)")
    ap.add_argument("--rate", type=float, default=2.7, help="--plan: greedy generations per second")
    ap.add_argument("--wait-for", action="append", default=None, metavar="PATH",
                    help="D-23 gate (repeatable): start only when PATH exists, e.g. runs/t2/t2_s0/eval.json")
    ap.add_argument("--wait-poll", type=float, default=60.0, help="gate poll interval, seconds")
    ap.add_argument("--wait-max-h", type=float, default=0.0, help="give up (exit 4, nothing loaded) after this many hours; 0 = never")
    return ap.parse_args(argv)


def finalize(out_dir: Path) -> bool:
    status = {arm: {s: stage_done(out_dir / arm, s) for s in STAGES} for arm in ARM_ORDER}
    if not all(all(v.values()) for v in status.values()):
        log("not complete yet: " + ", ".join(f"{arm}:{s}" for arm, v in status.items() for s, ok in v.items() if not ok))
        return False
    arms = {}
    for arm in ARM_ORDER:
        r, st = _read_json(out_dir / arm / "ARM.json"), _read_json(out_dir / arm / "STAGES.json")
        rp = r.get("repro", {})
        arms[arm] = {"fingerprint": r["fingerprint"], "titan_blocks": r["titan_blocks"], "differs_from_supa": r["differs_from_supa"],
                     "repro": {k: rp.get(k) for k in ("status", "reference", "gate_equal", "gate_n", "batch_items_equal",
                                                      "batch_items_compared")},
                     "stages": {s: {k: v for k, v in st["stages"][s].items() if k != "sha256"} for s in STAGES},
                     "files_sha256": {f: h for s in STAGES for f, h in st["stages"][s]["sha256"].items()}}
    run = _read_json(out_dir / "RUN.json") or {}
    _write_json(out_dir / "DONE.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": arms,
                                        "run": {k: run.get(k) for k in RUN_KEYS + ("source_hashes",)},
                                        "gates_passed": ["G1", "G2", "G3", "G4", "G5"]})
    log(f"wrote {out_dir / 'DONE.json'}")
    return True


def main(argv=None):
    a = parse_args(argv)
    if a.wait_for and not a.plan and not wait_for_files(a.wait_for, a.wait_poll, a.wait_max_h):
        raise SystemExit(4)                                          # D-23 gate not met: nothing loaded or written
    cuda = a.device == "cuda"
    if cuda:
        if P._TENSOR_DEVICE is None:
            assert not _CPU and os.environ.get("CUDA_VISIBLE_DEVICES") != "", "--device cuda in a CUDA-hidden process"
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
        torch.set_num_threads(a.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    arms = [x for x in ARM_ORDER if x in a.arms.split(",")]
    assert arms and set(a.arms.split(",")) <= set(ARM_ORDER), a.arms
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "runs" / a.bank / a.out_name
    patch_dir = Path(a.patch_dir)
    bank_path = ROOT / "data" / f"{a.bank}.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    chains = load_chains(bank)
    prompts = offtask.load_prompts()
    pmeta = patch_meta(patch_dir)
    if a.plan:
        plan(a, chains, prompts, pmeta)
        return
    if (out_dir / "DONE.json").exists():
        log(f"[localized] {out_dir / 'DONE.json'} exists: nothing pending")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    run = {"bank": a.bank, "bank_sha256": sha256_file(bank_path), "n_chains": len(chains),
           "chain_ids_sha256": sha256_text("\n".join(c["cand_id"] for c in chains)), "n_prompts": len(prompts),
           "prompt_ids_sha256": sha256_text("\n".join(r["id"] for r in prompts)), "offtask_max_new": a.offtask_max_new,
           "chain_max_tokens": {f: generation.max_tokens(f) for f in env.FAMILIES}}
    old = _read_json(out_dir / "RUN.json")
    if old:
        bad = [k for k in RUN_KEYS if old.get(k) != run[k]]
        if bad:
            raise SystemExit(f"{out_dir} was started with different {bad}; use another --out-name")
    run = {**(old or {}), **run, "arms": list(ARM_ORDER), "arm_titan_blocks": {x: TITAN_RANGE[x] for x in ARM_ORDER},
           "lease_label": LEASE_LABEL, "device": a.device, "dtype": "float32", "torch": torch.__version__,
           "transformers": __import__("transformers").__version__, "patch_dir": str(patch_dir),
           "repro_per_family": a.repro_per_family, "chain_ids": [c["cand_id"] for c in chains]}
    run.setdefault("invocations", []).append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": arms,
                                              "argv": list(argv) if argv is not None else sys.argv[1:]})
    _write_json(out_dir / "RUN.json", run)

    pending = {arm: [s for s in STAGES if not stage_done(out_dir / arm, s)] for arm in arms}
    log(f"{len(chains)} chains, {len(prompts)} off-task prompts; arms {arms}")
    log("pending: " + (", ".join(f"{arm}:{'/'.join(v)}" for arm, v in pending.items() if v) or "none"))
    if not any(pending.values()):
        finalize(out_dir)
        return

    F, src_rec = source_hashes(pmeta)                                # G1 (files) + G2, before the lease
    log(f"source hashes from the files ({src_rec['file_hash_seconds']}s): cores differ in exactly "
        f"{src_rec['cores_differ_in'][0]}..{src_rec['cores_differ_in'][-1]}; run_patch recorded hashes: "
        f"{'equal' if not str(src_rec['equal_to_run_patch_recorded']).startswith('not') else 'not available'}")
    tok = runtime.tokenizer()
    P._TOK[0] = tok
    be = Backend(a.device)
    with be.lease():                                                 # ONE lease for the whole job (cuda)
        model, prov = be.load("supa")                                # fingerprint a9952c55f5a1 asserted
        try:
            with cpu_shims(not cuda):
                n = model.config.num_hidden_layers
                if n != N_LAYERS:
                    raise SystemExit(f"num_hidden_layers {n} != {N_LAYERS}")
                H = P.group_hashes(model)
                bad = group_sort(g for g in set(H) | set(F["supa"]) if H.get(g) != F["supa"].get(g))
                if bad:
                    raise P.SanityError(f"G1: the loaded Supa's group hashes differ from its file's in {bad[:6]}")
                src_rec["loaded_supa_equals_file"] = True
                run["source_hashes"] = {**src_rec, "supa": F["supa"], "sparse": F["sparse"],
                                        "load_fingerprint_supa": prov["param_fingerprint"]}
                _write_json(out_dir / "RUN.json", run)
                hashes = {"supa": F["supa"], "sparse": F["sparse"]}
                cur = ["supa"] * n
                for arm in arms:
                    if not pending[arm]:
                        continue
                    d = out_dir / arm
                    d.mkdir(exist_ok=True)
                    rec = build_arm(model, arm, cur, hashes, be)
                    old_arm = _read_json(d / "ARM.json")
                    if old_arm and (old_arm.get("fingerprint") != rec["fingerprint"]
                                    or old_arm.get("group_hashes") != rec["group_hashes"]):
                        raise P.SanityError(f"{arm}: rebuilt weights differ from the recorded build "
                                            f"({old_arm.get('fingerprint', '')[:12]} -> {rec['fingerprint'][:12]})")
                    old_st = _read_json(d / "STAGES.json")
                    if old_st and old_st.get("fingerprint") != rec["fingerprint"]:
                        raise P.SanityError(f"{arm}: STAGES.json was written by another build")
                    rp = (old_arm or {}).get("repro")
                    if rp and rp.get("fingerprint") == rec["fingerprint"]:
                        rec["repro"] = rp
                        log(f"repro {arm}: reusing the recorded check ({rp.get('status')})")
                    else:
                        rec["repro"] = repro_check(be, model, arm, tok, patch_dir, pmeta, a.repro_per_family, rec["fingerprint"])
                    rec["recorded"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    _write_json(d / "ARM.json", rec)
                    st = old_st or {"arm": arm, "fingerprint": rec["fingerprint"], "stages": {}}
                    for s in pending[arm]:
                        log(f"{arm}: {s} ...")
                        info = run_stage(s, model, tok, d, chains, a)
                        st["stages"][s] = info
                        _write_json(d / "STAGES.json", st)
                        rate = f", {info['generations_per_s']} gen/s" if info.get("generations_per_s") else ""
                        log(f"{arm}: {s} done in {info['seconds']:.0f}s"
                            + (f" ({info['generations']} generations{rate})" if info.get("generations") else "")
                            + (f", {info['complete_chains']}/{info['n_chains']} chains complete" if "complete_chains" in info else ""))
                        if _on_stage_done is not None:
                            _on_stage_done(arm, s)
        finally:
            del model
            be.release()
    if finalize(out_dir):
        log("[localized] finished")


if __name__ == "__main__":
    try:
        main()
    except P.SanityError as e:
        log(f"SANITY FAILED, aborting (no stage of the failing arm was run): {e}")
        sys.exit(3)
