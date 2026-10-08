"""Per-item activation patching and weight-spliced hybrids between Supa and Titan-Sparse (exploratory).

Question (after RESULTS_PILOT R9: the constant answer-slot offset carries format/mode, not the
transition computation): WHERE does the computation live? The residual stream of one model at a
block boundary is written into the other model on the same prompt, in the same left-padded batch
(same tokenizer, so positions align), and the receiver then decodes greedily on its own. Two
weight-spliced hybrids apply one side of the delta at EVERY position (prompt and decode).

Layer index L = hidden_states[L] = output of block L-1 (model.model.layers[L-1]); the patch
replaces that block's output tensor. 1 <= L <= num_hidden_layers. (L = 28 is the last block's
output BEFORE the final norm, unlike HF hidden_states[28], which is after it.) Titan = Supa + delta
on blocks 8..27; embeddings, blocks 0..7 and the final norm are bit-identical (noop_layers 1..8).

Conditions (each a full greedy generation over the items, scored with run_steer.score plus per-item
lenient correctness, as in run_steer_cpu):
  base:sparse, base:supa  no intervention
  T2S_last@L   Supa receives Titan's block-L output at the LAST prompt position (prefill only)
  T2S_all@L    Supa receives Titan's block-L outputs at ALL real prompt positions (prefill only;
               left pads keep the receiver's own values, they are masked as keys)
  S2T_all@L    Titan receives Supa's block-L outputs at all real prompt positions (prefill only)
  T2S_mis@L    control: Supa receives Titan's all-position states of a DIFFERENT item (see below);
               also scored against the DONOR's gold (donor_match, donor_match_changed)
  -- additions after design review (default --dmis-layers 24, --hybrid-layers 16,24) --
  T2S_dmis@L   difference swap: at every real prompt position of item i, h <- h_own + (h_T(j) -
               h_S(j)), j = the T2S_mis donor, right-aligned; additive, prefill only. The matched
               version h_S(i) + (h_T(i) - h_S(i)) is T2S_all@L; only the item-specific remainder of
               the delta-induced shift is swapped (R5: the per-family constant is 70-99.6% of it).
  HlowT@L      hybrid weights: Supa with Titan's blocks 0..L-1 (in effect 8..L-1), at all positions
  HupT@L       hybrid weights: Titan with Supa's blocks 0..L-1 (= Supa + Titan's blocks L..27)
               Hybrids are built in place on the loaded model (one model in memory) from per-tensor
               safetensors reads of the other core (Titan FP32; Supa BF16, upcast exactly), and
               every spliced block is checked against that core's parameter hashes.
Decode steps are never patched. KV cache consequence: blocks 0..L-1 (those producing
hidden_states 1..L) computed and cached their keys/values at the prompt positions from the
RECEIVER's own stream before the patch; only blocks L..N-1 see the patched stream. Every decode
step therefore attends to the receiver's own lower-layer KV and the patched upper-layer KV, and the
generated tokens themselves are processed by the receiver alone (all of its blocks).

Mismatched donor (T2S_mis, T2S_dmis). For receiver item i the donor j is drawn from the run's items:
same family, different parent, different gold, different before-state; same prompt token count if
any candidate has it, else the nearest count, ties to the LONGER donor; final pick uniform with
rng(--mis-seed) among the tied candidates (sorted by id). Alignment is right-aligned on real tokens:
k = min(n_i, n_j); receiver positions [-k:] <- donor real positions [-k:]. A shorter donor leaves the
receiver's first n_i - k real positions unpatched; a longer donor's first n_j - k tokens are dropped.
The answer slot and the chat-template tail always align. Per item: donor id, n_j - n_i, exact flag
(_meta.mismatch). On the pilot 634/640 pairs have equal length; the other 6 (ledger, donor one token
longer) shift the shared template prefix by one position, so the receiver's position 0 (attention
sink) gets the donor's position-1 state: mis/dmis aggregates are ALSO reported on the exact-length
pairs only (exact_len_subset, _derived), and those are the ones to read.

Staging layout (runs/<bank>/<out>/). Donor states are per-item, right-aligned REAL-token spans,
independent of padding:
  donor_<arm>_L{L}.npy   float32 [T, D], T = sum of prompt token counts over the run's items; item
                         i occupies rows offset[i] : offset[i] + n_i in prompt order (row n_i - 1 is
                         the answer slot)
  donor_<arm>_ids.npy    int32 [T], the token ids of those rows (receiver/donor id check)
  donor_<arm>_index.json items, {item: [offset, n]}, layers, D, batches, fingerprint, block hashes,
                         baseline texts and first tokens, complete flag
States are captured inside the baseline generate() of the same left-padded batches (batch
composition is part of the index and must match), so same-item patches reproduce bit-for-bit the
tensor the receiver's own batch computes when donor == receiver. Phases (one real model in memory
at a time): A Titan: sanity, base:sparse + staging -> B Supa: sanity, base:supa + staging, T2S_*
(incl. T2S_dmis) from the staged states, then splice sanity and HlowT@L (L ascending) -> C Titan:
sanity, S2T_*, then splice sanity and HupT@L. Staged arrays are deleted at the end unless
--keep-states. base:sparse is committed only after Supa's sanity has passed, from the texts stored
in Titan's staging index, so no result exists before both sanity checks passed.

Prefill detection (transformers 5.14.1, checked in source). generate() makes ONE prefill forward
over the full left-padded prompt (GenerationMixin._prefill; generation_config.prefill_chunk_size is
asserted None, no compile with the default DynamicCache), then one forward per new token with
seq_len 1. Qwen3DecoderLayer returns a Tensor in 5.x (a tuple is handled for 4.x). The hooks do not
guess from h.shape[1] > 1: they act on their FIRST call and raise unless that call has
seq_len == prompt length and every later call has seq_len == 1; calls == generated tokens is
asserted per batch, and no hook may remain registered afterwards.

STARTUP SANITY (each model load, before that model produces any result; failure aborts with exit 3
and writes no result): on the first --sanity-per-family items of each family (own batches), the
unpatched generation is staged through the same on-disk format, then for every L: identity all@L
and identity last@L with the model's OWN staged states must reproduce the unpatched texts exactly
with patch max|diff| == 0; a zero vector at the last position must change at least one text (hook
is live); every patched generate patches exactly once in prefill and never in decode (counts); every
same-item patch asserts donor token ids == receiver row ids; for L in --dmis-layers the additive
path with a zero difference (own - own) must also reproduce the texts exactly. Recorded in
patch_sanity.json (with the reference texts). SPLICE SANITY (before the first hybrid of a phase):
every differing block is spliced from the other core and must hash-equal that core's block (hard
gate, exit 3); the spliced model must then reproduce the other core's sanity texts (recorded; with
equal hashes a mismatch can only be cross-load nondeterminism and is logged as a WARNING).

Outputs: patch_results.json (written only when every requested condition is done; master_chain.sh
uses its existence as the completion marker), patch_progress.json while running (resume skips
conditions already there), patch_sanity.json, gen_<cond>.json texts. One line per condition is
printed like run_steer.py. patch_results.json["_derived"] holds FR with paired parent-cluster
bootstrap CIs and the HI/MID/LO labels of the table below, the T2S_mis/T2S_dmis paired references,
and the hybrid/patch first-token identities. Design note: docs/patching/DESIGN.md.

WHAT EACH CONDITION CAN AND CANNOT SHOW. L and R answers start with '{"' in BOTH arms, so every
content token of an L/R answer is produced at a decode step (generated index 3-19 on Titan's correct
outputs), which a prefill patch never reaches. P is the exception: Titan's first token is the
leading digit (Supa wraps the answer in JSON).
  T2S_all@L   Supa's upper blocks see Titan's block-L PROMPT stream; Supa's lower blocks still build
              the lower-layer prompt KV and process every generated token. HIGH = Titan's lower-delta
              encoding of the prompt suffices for Supa's upper blocks (a generic mode/rule unlock is
              not excluded: T2S_dmis). LOW does NOT show that the lower delta is unneeded: the delta
              may act at decode positions or need Titan's upper blocks (HlowT / HupT separate these).
  S2T_all@L   removes only the lower delta's effect on Titan's block-L PROMPT stream; Titan's delta
              blocks 8..L-1 still process every decode token over Titan's own lower prompt KV. HIGH
              is predicted by computation above L AND by lower-delta work at decode / lower KV
              (leak): uninformative alone (compare HupT@L; S2T_all - HupT = leak share). LOW =
              Titan's upper blocks need the lower delta's block-L prompt stream although its lower
              decode path was intact (parse collapse instead = stream incompatibility).
  T2S_last@L  one patched position. For L/R a LOWER BOUND on what the answer slot holds (decode
              steps copy payload digits from Supa's own stream), so T2S_last << T2S_all does not
              localise item information to earlier positions. Least confounded on P, whose whole
              Titan answer is a few digit tokens from generated index 0 (it first differs from the
              before-state at index 0-2).
  T2S_mis@L   leak calibration, NOT a mode-vs-computation control (mode-only and item-specific
              computation both predict "follows the donor"). donor_match_changed (donors whose state
              changes; on P most donors do not, and echoing their before-state would count as a
              match) ~ the paired T2S_all@L accuracy on the same donors = the patched upper stream
              overrides the receiver's own lower KV and decode path, so T2S readings are trustworthy;
              a high receiver echo rate (echo_before_changed) = leak. Receiver accuracy is capped at
              A_S under every hypothesis: uninformative.
  T2S_dmis@L  read only where T2S_all@L is HIGH (otherwise there is no gain to attribute).
  HlowT@L     lower delta at EVERY position (prompt and decode, lower KV included); no leak.
  HupT@L      upper delta at every position on Supa's lower stack.
  ft_donor    first-token agreement with the donor (hybrids: with the spliced-in core). Uninformative
              for L/R (both arms start '{"'); for P it mixes the mode switch ('{"' -> digit, which the
              constant offset alone produces, R5) with the first digit. Read accuracy and format.

DECISION TABLE (per family, changed items, exact-length pairs for mis/dmis; A_T = base:sparse,
A_S = base:supa; FR = (acc - A_S) / (A_T - A_S); HI: FR >= 0.7, LO: FR <= 0.2, else MID; fixed
before the run; undefined when A_T - A_S < 0.10)
  T2S_all@L  HlowT@L  HupT@L   reading at cut L
  HI         HI       LO       the transition is computed by delta blocks 8..L-1 into the PROMPT
                               stream and Supa's upper blocks read it out; then T2S_dmis@L:
                                 HI (~T2S_all) -> a generic family-level mode/rule signal unlocks
                                   Supa's own computation (check per operation, e.g. R swap)
                                 LO with T2S_all's format kept (format classes like T2S_all) ->
                                   item-specific computation below L
                                 donor_match_changed > 0 -> the shift itself carries j's answer
                                 parse/format collapse -> off-manifold, uninterpretable
  LO         HI       LO       the lower delta is necessary and sufficient but works on decode
                               tokens or through the lower-layer prompt KV, not through the block-L
                               prompt stream
  LO         LO       HI       computation in delta blocks >= L (S2T_all - HupT = the leak share)
  LO         LO       LO       both halves needed (serial / distributed), or stitching incompatible:
                               check parse/format (collapse = incompatibility)
  HI         HI       HI       the two halves are redundant
  HI         LO       any      inconsistent with a clean cut: Titan's lower blocks at decode
                               positions conflict with Supa's upper blocks (incompatibility)
  any MID                      partial: the computation straddles L
  S2T_all@L LO                 Titan's upper blocks need the lower delta's block-L prompt stream
  parse/format moves, accuracy does not        format/mode effect only (R9)
  Across L: where HlowT(L) rises from A_S (L <= 8) to A_T (L = 28) = where the computation
  completes; where HupT(L) falls = which blocks are necessary. Checks: per item, first token of
  HlowT@L == T2S_all@L and HupT@L == S2T_all@L (identical prompt computation; < 1 can only be
  cross-load nondeterminism). At L <= 8 T2S_last / T2S_all / S2T_all / T2S_dmis are exact no-ops
  (noop_exact must be true): a full-scale check of the cross-model staging, not evidence about the
  delta; T2S_mis@8 is a leak calibration with only blocks 0..7 of the receiver's own KV.
Caveats: exploratory pilot TEST items (TEST was seen in development); one greedy decode; patches are
prefill-only; items cluster by parent (8 per parent, 26-28 parents per family): read the bootstrap
CIs; break results down by operation post hoc (R8).

Cost (default run, from --plan): 640 items x 19 conditions (14 spec + T2S_dmis@24 + HlowT/HupT@16,24)
+ 3 loads x 132 sanity generations + 2 x 12 splice-sanity generations = 12580 greedy generations,
~1.3 h at 2.7 generations/s (~0.8 h at 4.4/s; up to ~2 h if patched outputs run to the 160-token
cap). The additions cost +3260 generations (+35% over the 9320 of the spec conditions). Staging:
80,848 prompt tokens x 1024 x 4 B x 3 layers = 0.93 GiB per arm, 1.85 GiB peak on disk, deleted at
the end; the hybrids need no disk and no extra VRAM. Mismatch donors: 634/640 exact length, 6 one
token longer.

  python scripts/run_patch.py --device cuda --out-name patch            (master_chain.sh, leased)
  python scripts/run_patch.py --device cpu --per-family 48 --out-name patch_cpu
  python scripts/run_patch.py --device cuda --plan                      (prints plan, loads nothing)
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


# CPU mode: hide CUDA before torch is imported. Empty string, NEVER "-1" (DECISIONS I-21).
_CPU = _argv_value("--device") == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
if _CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _T = _argv_value("--threads", "2")
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = str(_T)

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import nullcontext  # noqa: E402
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
from rcwmt import bank as bankmod, env, generation, runtime  # noqa: E402
from run_steer import score  # noqa: E402

KINDS = ("T2S_last", "T2S_all", "S2T_all", "T2S_mis")          # spec conditions, at every --layers L
DMIS, HYB = "T2S_dmis", ("HlowT", "HupT")                       # additions (--dmis-layers, --hybrid-layers)
DONOR = {"T2S_last": "sparse", "T2S_all": "sparse", "S2T_all": "supa", "T2S_mis": "sparse", DMIS: "sparse",
         "HlowT": "sparse", "HupT": "supa"}                     # hybrids: the core whose blocks < L are spliced in
RECEIVER = {"T2S_last": "supa", "T2S_all": "supa", "S2T_all": "sparse", "T2S_mis": "supa", DMIS: "supa",
            "HlowT": "supa", "HupT": "sparse"}                  # hybrids: the loaded (host) model
STAGED = {"T2S_last": ("sparse",), "T2S_all": ("sparse",), "S2T_all": ("supa",), "T2S_mis": ("sparse",),
          DMIS: ("sparse", "supa"), "HlowT": (), "HupT": ()}   # staged states each kind reads
COUNTERPART = {"HlowT": "T2S_all", "HupT": "S2T_all"}           # same prompt computation -> same first token
FR_HI, FR_LO, FR_MIN_GAP = 0.7, 0.2, 0.10                       # decision-table thresholds (docstring)
EXPECTED_FP = {"supa": "a9952c55f5a1", "sparse": "818be9c6b6a3"}   # as run_steer_cpu.FINGERPRINTS
LEASE_LABEL = "rcwm-t1:patch"
LEASE_JOB = "RCWM-T1 per-item activation patching Supa<->Titan-Sparse (R9 follow-up)"
STEER_CPU_RESULTS = ROOT / "runs" / "bank_pilot" / "steer_cpu" / "steer_cpu_results.json"
MIN_FREE_RAM_GIB_CPU = 2.8                         # host RAM before loading one FP32 model on CPU
DISK_MARGIN_GIB = 2.0
PREFILL_NOTE = ("transformers {tv}: generate() = one prefill forward over the full left-padded prompt "
                "(_prefill; prefill_chunk_size asserted None) then one seq_len-1 forward per new token; "
                "hooks act on their FIRST call, assert seq_len == prompt length there and == 1 afterwards, "
                "calls == generated tokens, no hook left registered")

_TENSOR_DEVICE: str | None = None                  # test hook: run the cuda wiring with CPU tensors
_on_condition_done = None                          # test hook: callable(name) after each commit


class SanityError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------ items
def load_pool(bank_stem: str) -> list[dict]:
    """All TEST one-frame items in bank order (test hook: replaced by a fake pool)."""
    bank = json.loads((ROOT / "data" / f"{bank_stem}.json").read_text(encoding="utf-8"))
    return [it for it in bankmod.items(bank) if it["split"] == "test" and it["format"] == "one"]


def select_items(pool, per_family, seed, items_from):
    """Default: every pool item in bank order. --per-family N: run_steer_cpu.py's rule (rng(seed),
    per family, sorted choice); asserted equal to steer_cpu _meta.items when that file exists and
    used the same N and seed."""
    if per_family is None:
        return list(pool), "all TEST one-frame items in bank order"
    rng = np.random.default_rng(seed)
    sel = []
    for fam in env.FAMILIES:
        fi = [it for it in pool if it["family"] == fam]
        sel += [fi[i] for i in sorted(rng.choice(len(fi), size=min(per_family, len(fi)), replace=False))]
    ids = [it["item_id"] for it in sel]
    note = f"run_steer_cpu rule: rng({seed}).choice, {per_family} per family"
    ref = Path(items_from) if items_from else None
    if ref is not None and ref.exists():
        m = json.loads(ref.read_text(encoding="utf-8")).get("_meta", {})
        ref_ids = m.get("items", [])
        byid = {it["item_id"]: it for it in pool}
        cnt = Counter(byid[i]["family"] for i in ref_ids if i in byid)
        want = {f: min(per_family, sum(it["family"] == f for it in pool)) for f in env.FAMILIES}
        if all(i in byid for i in ref_ids) and m.get("seed") == seed and all(cnt[f] == want[f] for f in env.FAMILIES):
            assert ref_ids == ids, f"selection does not reproduce {ref} _meta.items"
            note += f"; equals {ref.name} _meta.items ({len(ids)} ids, same order)"
        else:
            note += f"; {ref.name} used a different N/seed, not compared"
    return sel, note


def batches_of(items, size):
    """run_steer.gen_with's batching: per family in env.FAMILIES order, consecutive chunks."""
    out = []
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        out += [fi[s: s + size] for s in range(0, len(fi), size)]
    return out


def left_pad(seqs):
    Lm = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), Lm), generation.PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), Lm), dtype=torch.long)
    for r, x in enumerate(seqs):
        ids[r, Lm - len(x):] = torch.tensor(x)
        att[r, Lm - len(x):] = 1
    assert bool(att[:, -1].all()), "a row has no real token at index -1"
    return ids, att


def pick_mismatch(items, enc, seed):
    rng = np.random.default_rng(seed)
    n = {it["item_id"]: len(enc[it["item_id"]]) for it in items}
    donors, rec = {}, {}
    for it in items:
        i = it["item_id"]
        cands = sorted((j for j in items if j["family"] == it["family"] and j["parent_id"] != it["parent_id"]
                        and j["gold"] != it["gold"] and j["gold_prefix_states"][0] != it["gold_prefix_states"][0]),
                       key=lambda j: j["item_id"])
        if not cands:
            raise SystemExit(f"no mismatch donor for {i}")
        key = lambda j: (abs(n[j["item_id"]] - n[i]), n[j["item_id"]] < n[i])  # noqa: E731
        best = min(key(j) for j in cands)
        tied = [j for j in cands if key(j) == best]
        j = tied[int(rng.integers(len(tied)))]
        donors[i] = j
        rec[i] = {"donor": j["item_id"], "len_diff": n[j["item_id"]] - n[i], "exact_len": n[j["item_id"]] == n[i]}
    return donors, rec


# ---------------------------------------------------------------------------------------- staging
class Staging:
    """Per-item real-token spans of block outputs, FP32 on disk (layout in the module docstring)."""

    def __init__(self, out_dir: Path, prefix: str):
        self.dir, self.prefix = out_dir, prefix
        self.idx, self.mm, self.ids, self.written = None, {}, None, set()

    @property
    def index_path(self) -> Path:
        return self.dir / f"{self.prefix}_index.json"

    def arr_path(self, L: int) -> Path:
        return self.dir / f"{self.prefix}_L{L}.npy"

    @property
    def ids_path(self) -> Path:
        return self.dir / f"{self.prefix}_ids.npy"

    def check(self, items, lens, batches, layers, fp_prefix, max_new) -> tuple[bool, str]:
        idx = _read_json(self.index_path)
        if not idx:
            return False, "no index"
        if not idx.get("complete"):
            return False, "incomplete"
        if idx["items"] != [it["item_id"] for it in items] or idx["len"] != [lens[it["item_id"]] for it in items]:
            return False, "different items"
        if idx["batches"] != [[it["item_id"] for it in b] for b in batches]:
            return False, "different batches"
        if not set(layers) <= set(idx["layers"]):
            return False, f"layers {idx['layers']} lack {sorted(set(layers) - set(idx['layers']))}"
        if not str(idx.get("fingerprint", "")).startswith(fp_prefix):
            return False, "fingerprint"
        if idx.get("max_new_tokens") != max_new:
            return False, "max_new_tokens"
        T = sum(idx["len"])
        for L in idx["layers"]:
            if not self.arr_path(L).exists():
                return False, f"missing L{L}"
            a = np.load(self.arr_path(L), mmap_mode="r")
            shape = a.shape
            del a
            if shape != (T, idx["D"]):
                return False, f"shape L{L}"
        return (self.ids_path.exists(), "ok" if self.ids_path.exists() else "missing ids")

    def begin(self, arm, items, lens, batches, layers, D, max_new):
        self.delete()
        order = [it["item_id"] for it in items]
        offs, o = {}, 0
        for i in order:
            offs[i] = [o, lens[i]]
            o += lens[i]
        self.idx = {"arm": arm, "items": order, "len": [lens[i] for i in order], "offset": offs,
                    "layers": sorted(layers), "D": int(D), "dtype": "float32",
                    "batches": [[it["item_id"] for it in b] for b in batches], "max_new_tokens": max_new,
                    "complete": False, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "layout": "rows offset[i]:offset[i]+n_i = item i's real prompt tokens in order; "
                              "row n_i-1 = answer slot; layer L = output of block L-1"}
        _write_json(self.index_path, self.idx)
        self.mm = {L: np.lib.format.open_memmap(self.arr_path(L), mode="w+", dtype=np.float32, shape=(o, int(D)))
                   for L in self.idx["layers"]}
        self.ids = np.lib.format.open_memmap(self.ids_path, mode="w+", dtype=np.int32, shape=(o,))
        self.written = set()

    def write_batch(self, b, ids: torch.Tensor, cap: dict):
        Lm = ids.shape[1]
        for r, it in enumerate(b):
            off, n = self.idx["offset"][it["item_id"]]
            self.ids[off: off + n] = ids[r, Lm - n:].numpy().astype(np.int32)
            for L in self.idx["layers"]:
                self.mm[L][off: off + n] = cap[L][r, Lm - n:].numpy()
            self.written.add(it["item_id"])
        for m in list(self.mm.values()) + [self.ids]:  # bound dirty mapped pages (host RAM)
            m.flush()

    def finish(self, **extra):
        assert self.written == set(self.idx["items"]), "staging: not every item written"
        for m in list(self.mm.values()) + [self.ids]:
            m.flush()
        self.close()
        self.idx.update(extra, complete=True, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        _write_json(self.index_path, self.idx)

    def open(self):
        self.close()
        self.idx = _read_json(self.index_path)
        assert self.idx and self.idx.get("complete"), f"{self.prefix}: staging incomplete"
        self.mm = {L: np.load(self.arr_path(L), mmap_mode="r") for L in self.idx["layers"]}
        self.ids = np.load(self.ids_path, mmap_mode="r")
        return self

    def rows(self, item_id: str, L: int) -> np.ndarray:
        off, n = self.idx["offset"][item_id]
        return np.array(self.mm[L][off: off + n], dtype=np.float32, copy=True)

    def token_ids(self, item_id: str) -> np.ndarray:
        off, n = self.idx["offset"][item_id]
        return np.array(self.ids[off: off + n], dtype=np.int64, copy=True)

    def close(self):
        self.mm, self.ids = {}, None
        gc.collect()                                 # release Windows file mappings

    def delete(self):
        self.close()
        for p in list(self.dir.glob(f"{self.prefix}_L*.npy")) + [self.ids_path, self.index_path]:
            if p.exists():
                p.unlink()

    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob(f"{self.prefix}_*") if p.is_file())


# ----------------------------------------------------------------------------------------- models
def load_cpu(arm: str):
    """run_steer_cpu.load_cpu (copied, not imported: importing it hides CUDA at import time)."""
    free = psutil.virtual_memory().available / 2 ** 30
    if free < MIN_FREE_RAM_GIB_CPU:
        raise SystemExit(f"only {free:.2f} GiB host RAM free (< {MIN_FREE_RAM_GIB_CPU}); not loading {arm}")
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(runtime.PATHS[arm]), dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, device_map={"": "cpu"}, attn_implementation="sdpa").eval()
    return m, runtime.param_fingerprint(m)


def hidden_size_hint() -> int:
    return int(json.loads((runtime.PATHS["sparse"] / "config.json").read_text(encoding="utf-8"))["hidden_size"])


class Backend:
    """Everything that differs between --device cuda and --device cpu."""

    def __init__(self, device: str):
        self.device = device
        self.cuda = device == "cuda"
        self.dev = torch.device(_TENSOR_DEVICE or device)
        self.loads = []

    def lease(self):
        return runtime.gpu_lease(label=LEASE_LABEL, job=LEASE_JOB) if self.cuda else nullcontext()

    def load(self, arm: str):
        t0 = time.time()
        if self.cuda:
            m, prov = runtime.load_arm(arm)
            fp = prov["param_fingerprint"]
        else:
            m, fp = load_cpu(arm)
            prov = {"arm": arm, "dtype": "float32", "device": "cpu", "param_fingerprint": fp,
                    "source": str(runtime.PATHS.get(arm, ""))}
        if not fp.startswith(EXPECTED_FP[arm]):
            raise SystemExit(f"{arm}: fingerprint {fp[:12]} != expected {EXPECTED_FP[arm]}")
        assert all(p.dtype == torch.float32 for p in m.parameters())
        self.loads.append(arm)
        log(f"loaded {arm} on {self.device} ({fp[:12]}, {time.time() - t0:.0f}s)")
        return m, prov

    def release(self):
        gc.collect()
        if self.cuda and _TENSOR_DEVICE is None:
            torch.cuda.empty_cache()

    def gen_kwargs(self) -> dict:
        return {"logits_processor": runtime.throttle_list()} if self.cuda else {}

    def after_batch(self):
        if self.cuda:
            runtime.breathe()                     # duty cycle + thermal pause flag (I-10..I-15)


def _group_key(name: str) -> str:
    return "block" + name.split(".")[2] if name.startswith("model.layers.") else name.rsplit(".", 1)[0]


@torch.no_grad()
def group_hashes(model, only=None) -> dict:
    """sha1 over the full bytes of each parameter group (embed, block i, norm), chunked so no more
    than 64 MiB is copied to host at a time. `only`: restrict to these group keys."""
    groups: dict[str, list] = {}
    for name, p in model.named_parameters():
        key = _group_key(name)
        if only is None or key in only:
            groups.setdefault(key, []).append((name, p))
    out = {}
    for key, ps in groups.items():
        h = hashlib.sha1()
        for name, p in sorted(ps, key=lambda kv: kv[0]):
            h.update(name.encode())
            flat = p.detach().reshape(-1)
            for s in range(0, flat.numel(), 1 << 24):
                h.update(flat[s: s + (1 << 24)].float().cpu().numpy().tobytes())
        out[key] = h.hexdigest()[:16]
    return out


def noop_layers(ha: dict, hb: dict, n_layers: int) -> list[int]:
    """L such that embeddings and blocks 0..L-1 are bit-identical: block-L outputs coincide."""
    if ha.get("model.embed_tokens") != hb.get("model.embed_tokens"):
        return []
    out = []
    for L in range(1, n_layers + 1):
        if ha.get(f"block{L - 1}") == hb.get(f"block{L - 1}"):
            out.append(L)
        else:
            break
    return out


# ------------------------------------------------------------------------------ weight splicing
def weight_file(arm: str) -> Path:
    """The core's single safetensors file (test hook)."""
    return runtime._safetensor_path(runtime.PATHS[arm])


@torch.no_grad()
def splice_blocks(model, src_arm: str, blocks, dev) -> int:
    """Overwrite `blocks` of the loaded model in place with src_arm's stored tensors, one tensor at a
    time through safetensors (as runtime._apply_delta): an FP32 source is copied as is, a BF16 source
    upcast, which is exactly what the FP32 load of that core does. Returns the elements copied."""
    from safetensors import safe_open
    params = dict(model.named_parameters())
    n = 0
    with safe_open(str(weight_file(src_arm)), framework="pt", device=str(dev)) as f:
        keys = set(f.keys())
        for i in blocks:
            pre = f"model.layers.{i}."
            names = sorted(k for k in params if k.startswith(pre))
            if not names or names != sorted(k for k in keys if k.startswith(pre)):
                raise SanityError(f"splice {src_arm} block {i}: stored tensor names differ from the model's")
            for k in names:
                t = f.get_tensor(k)
                if tuple(t.shape) != tuple(params[k].shape):
                    raise SanityError(f"splice {src_arm} {k}: shape {tuple(t.shape)} != {tuple(params[k].shape)}")
                params[k].data.copy_(t.to(torch.float32))
                n += t.numel()
                del t
    return n


def set_block_sources(model, cur: list, want: list, hashes: dict, dev) -> list[int]:
    """Make block i hold core want[i]'s weights; cur[i] tracks what it holds now (updated in place).
    Blocks that are bit-identical across the cores are only relabelled. Every spliced block must
    hash-equal the source core's block (hard gate). Returns the spliced block indices."""
    todo: dict[str, list[int]] = {}
    for i, (c, w) in enumerate(zip(cur, want)):
        if c == w:
            continue
        if hashes[c][f"block{i}"] == hashes[w][f"block{i}"]:
            cur[i] = w
        else:
            todo.setdefault(w, []).append(i)
    done = []
    for arm, blocks in todo.items():
        splice_blocks(model, arm, blocks, dev)
        got = group_hashes(model, only={f"block{i}" for i in blocks})
        bad = [i for i in blocks if got[f"block{i}"] != hashes[arm][f"block{i}"]]
        if bad:
            raise SanityError(f"spliced blocks {bad} do not hash-equal {arm}'s blocks")
        for i in blocks:
            cur[i] = arm
        done += blocks
    return sorted(done)


# ------------------------------------------------------------------------------- hooks + generate
class Capture:
    """Forward hook: keep the FIRST call's (prefill) block output on CPU; later calls must be
    seq_len-1 decode steps."""

    def __init__(self, B: int, n_prompt: int):
        self.B, self.n_prompt, self.calls, self.h = B, n_prompt, 0, None

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.calls += 1
        if self.calls == 1:
            if tuple(h.shape[:2]) != (self.B, self.n_prompt):
                raise SanityError(f"capture: first call shape {tuple(h.shape)} is not the prefill ({self.B}, {self.n_prompt})")
            self.h = h.detach().to("cpu", torch.float32, copy=True)
        elif h.shape[1] != 1:
            raise SanityError(f"capture: call {self.calls} has seq_len {h.shape[1]} (expected a decode step)")
        return None


class Patcher:
    """Forward hook: on the FIRST call (prefill, seq_len == prompt length) replace the block output
    at mask positions by `patch` (additive=True: add `patch` there); every later call must be a
    seq_len-1 decode step, left untouched. max_abs / rel describe the change actually applied."""

    def __init__(self, patch: torch.Tensor, mask: torch.Tensor, n_prompt: int, additive: bool = False):
        self.patch, self.mask, self.n_prompt, self.additive = patch, mask, n_prompt, additive
        self.calls = self.prefill_patched = self.decode_calls = 0
        self.max_abs = self.rel = 0.0

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.calls += 1
        if self.calls == 1:
            if tuple(h.shape) != tuple(self.patch.shape) or h.shape[1] != self.n_prompt:
                raise SanityError(f"patch: first call shape {tuple(h.shape)} is not the prefill {tuple(self.patch.shape)}")
            m = self.mask
            new = h + self.patch.to(h.dtype) if self.additive else self.patch.to(h.dtype)
            if bool(m.any()):
                own = h[m]
                d = new[m] - own
                self.max_abs = float(d.abs().max())
                self.rel = float(d.norm(dim=-1).mean() / own.norm(dim=-1).mean().clamp_min(1e-12))
            h = torch.where(m[..., None], new, h)
            self.prefill_patched += 1
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        if h.shape[1] != 1:
            raise SanityError(f"patch: call {self.calls} has seq_len {h.shape[1]} (a non-decode call after prefill)")
        self.decode_calls += 1
        return None


@torch.no_grad()
def run_batch(be: Backend, model, b, enc, max_new, layer=None, patch_fn=None, capture_layers=()):
    """run_steer.gen_with for one batch (same padding, kwargs, decoding), optionally with a
    prefill-only patch at block `layer` and/or prefill capture at `capture_layers`."""
    fam = b[0]["family"]
    assert all(it["family"] == fam for it in b)
    ids, att = left_pad([enc[it["item_id"]] for it in b])
    B, Lm = ids.shape
    if getattr(model.generation_config, "prefill_chunk_size", None) is not None:
        raise SanityError("chunked prefill is on; the prefill-only hooks assume one prefill call")
    blocks = model.model.layers
    touched = sorted(set(capture_layers) | ({layer} if layer else set()))
    n_hooks0 = {L: len(blocks[L - 1]._forward_hooks) for L in touched}
    handles, caps, pt = [], {}, None
    try:
        for L in capture_layers:
            caps[L] = Capture(B, Lm)
            handles.append(blocks[L - 1].register_forward_hook(caps[L]))
        if patch_fn is not None:
            patch, mask = patch_fn(ids, att)
            assert patch.shape[:2] == ids.shape and mask.shape == ids.shape
            if bool((mask & (att == 0)).any()):
                raise SanityError("patch mask touches a pad position")
            pt = Patcher(patch.to(be.dev), mask.to(be.dev), Lm, additive=bool(getattr(patch_fn, "additive", False)))
            handles.append(blocks[layer - 1].register_forward_hook(pt))
        g = model.generate(input_ids=ids.to(be.dev), attention_mask=att.to(be.dev), do_sample=False,
                           max_new_tokens=max_new, eos_token_id=generation.EOS, pad_token_id=generation.PAD,
                           temperature=None, top_p=None, top_k=None, **be.gen_kwargs())
    finally:
        for hd in handles:
            hd.remove()
    for L in touched:
        if len(blocks[L - 1]._forward_hooks) != n_hooks0[L]:
            raise SanityError(f"a forward hook is still registered on block {L - 1}")
    g = g.cpu()
    n_new = g.shape[1] - Lm
    for L, c in caps.items():
        if c.calls != n_new or c.h is None:
            raise SanityError(f"capture L{L}: {c.calls} calls for {n_new} generated tokens")
    info = None
    if pt is not None:
        if not (pt.prefill_patched == 1 and pt.calls == n_new and pt.decode_calls == n_new - 1):
            raise SanityError(f"hook count: prefill {pt.prefill_patched}, calls {pt.calls}, decode {pt.decode_calls}, "
                              f"tokens {n_new}")
        info = {"calls": pt.calls, "prefill_patched": pt.prefill_patched, "decode_calls": pt.decode_calls,
                "n_new": n_new, "max_abs": pt.max_abs, "rel": pt.rel, "n_patched": int(pt.mask.sum())}
    texts, first = {}, {}
    for r, it in enumerate(b):
        new = g[r, Lm:].tolist()
        first[it["item_id"]] = new[0] if new else None
        cut = next((j for j, t in enumerate(new) if t in generation.EOS), len(new))
        texts[it["item_id"]] = _decode(new[:cut])
    be.after_batch()
    return {"texts": texts, "first": first, "ids": ids, "att": att, "cap": {L: c.h for L, c in caps.items()},
            "info": info, "n_new": n_new}


_TOK = [None]                                      # the tokenizer, set in main()


def _decode(tokens):
    return _TOK[0].decode(tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------------- patch builders
def _check_rows(st: Staging, b, enc, ids, att):
    """Donor/receiver input ids identical per row (same-item patches) and left padding as expected."""
    Lm = ids.shape[1]
    for r, it in enumerate(b):
        n = len(enc[it["item_id"]])
        if not (bool(att[r, Lm - n:].all()) and not bool(att[r, : Lm - n].any())):
            raise SanityError(f"row {r}: unexpected padding")
        if not np.array_equal(st.token_ids(it["item_id"]), ids[r, Lm - n:].numpy()):
            raise SanityError(f"{it['item_id']}: donor token ids differ from the receiver row")


def same_item_patch(st: Staging, b, enc, L, which):
    def fn(ids, att):
        _check_rows(st, b, enc, ids, att)
        B, Lm = ids.shape
        patch = torch.zeros((B, Lm, st.idx["D"]), dtype=torch.float32)
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        for r, it in enumerate(b):
            rows = torch.from_numpy(st.rows(it["item_id"], L))
            n = rows.shape[0]
            if which == "all":
                patch[r, Lm - n:] = rows
                mask[r, Lm - n:] = True
            else:                                    # left padding: the answer slot is index -1
                patch[r, Lm - 1] = rows[-1]
                mask[r, Lm - 1] = True
        if which == "last":
            assert int(mask.sum()) == B and bool(mask[:, -1].all())
        return patch, mask
    return fn


def zero_last_patch(D):
    def fn(ids, att):
        B, Lm = ids.shape
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        mask[:, -1] = True
        return torch.zeros((B, Lm, D), dtype=torch.float32), mask
    return fn


def mismatch_patch(st: Staging, b, enc, donors, L):
    """Right-aligned: k = min(n_i, n_j); receiver[-k:] <- donor real span[-k:]."""
    def fn(ids, att):
        B, Lm = ids.shape
        patch = torch.zeros((B, Lm, st.idx["D"]), dtype=torch.float32)
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        for r, it in enumerate(b):
            j = donors[it["item_id"]]["item_id"]
            if not np.array_equal(st.token_ids(j), np.asarray(enc[j])):
                raise SanityError(f"staged ids for donor {j} differ from its tokenization")
            rows = torch.from_numpy(st.rows(j, L))
            k = min(len(enc[it["item_id"]]), rows.shape[0])
            patch[r, Lm - k:] = rows[rows.shape[0] - k:]
            mask[r, Lm - k:] = True
        return patch, mask
    return fn


def diff_mismatch_patch(stT: Staging, stS: Staging, b, enc, donors, L):
    """T2S_dmis, additive and right-aligned like mismatch_patch: receiver[-k:] += T(j)[-k:] - S(j)[-k:]."""
    def fn(ids, att):
        B, Lm = ids.shape
        patch = torch.zeros((B, Lm, stT.idx["D"]), dtype=torch.float32)
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        for r, it in enumerate(b):
            j = donors[it["item_id"]]["item_id"]
            for st in (stT, stS):
                if not np.array_equal(st.token_ids(j), np.asarray(enc[j])):
                    raise SanityError(f"{st.prefix}: staged ids for donor {j} differ from its tokenization")
            d = torch.from_numpy(stT.rows(j, L) - stS.rows(j, L))
            k = min(len(enc[it["item_id"]]), d.shape[0])
            patch[r, Lm - k:] = d[d.shape[0] - k:]
            mask[r, Lm - k:] = True
        return patch, mask
    fn.additive = True
    return fn


def zero_diff_patch(st: Staging, b, enc, L):
    """Sanity of the additive path: own(i) - own(i) == 0 added at every real position -> identity."""
    def fn(ids, att):
        _check_rows(st, b, enc, ids, att)
        B, Lm = ids.shape
        patch = torch.zeros((B, Lm, st.idx["D"]), dtype=torch.float32)
        mask = torch.zeros((B, Lm), dtype=torch.bool)
        for r, it in enumerate(b):
            rows = st.rows(it["item_id"], L)
            patch[r, Lm - rows.shape[0]:] = torch.from_numpy(rows - rows)
            mask[r, Lm - rows.shape[0]:] = True
        return patch, mask
    fn.additive = True
    return fn


# ----------------------------------------------------------------------------------------- scoring
def correct(items, texts):
    return {it["item_id"]: bool((st := env.parse_state(it["family"], texts[it["item_id"]], lenient=True)) is not None
                                and env.dumps(st) == it["gold"]) for it in items}


def parse_rate(items, texts):
    return {f: float(np.mean([env.parse_state(f, texts[it["item_id"]], lenient=True) is not None
                              for it in items if it["family"] == f]))
            for f in env.FAMILIES if any(it["family"] == f for it in items)}


def fam_mean(items, per):
    return {f: round(float(np.mean([per[it["item_id"]] for it in items if it["family"] == f])), 4)
            for f in env.FAMILIES if any(it["family"] == f for it in items)}


def fam_mean_opt(items, per):
    """fam_mean over the items present in `per` (families without any -> None)."""
    out = {}
    for f in env.FAMILIES:
        v = [per[it["item_id"]] for it in items if it["family"] == f and it["item_id"] in per]
        if any(it["family"] == f for it in items):
            out[f] = round(float(np.mean(v)), 4) if v else None
    return out


def changed(it) -> bool:
    return it["gold"] != it["gold_prefix_states"][0]


def text_metrics(items, texts):
    """Output format classes (bare = strict parse, wrapped = lenient-only parse, unparsed) and the
    echo rate: on items whose state changes, the output equals the item's own before-state."""
    fmt, echo = {}, {}
    for it in items:
        t, f = texts[it["item_id"]], it["family"]
        lenient = env.parse_state(f, t, lenient=True)
        fmt[it["item_id"]] = ("bare" if env.parse_state(f, t, lenient=False) is not None
                              else "wrapped" if lenient is not None else "unparsed")
        if changed(it):
            echo[it["item_id"]] = lenient is not None and env.dumps(lenient) == it["gold_prefix_states"][0]
    classes = {f: {c: round(float(np.mean([fmt[it["item_id"]] == c for it in items if it["family"] == f])), 4)
                   for c in ("bare", "wrapped", "unparsed")}
               for f in env.FAMILIES if any(it["family"] == f for it in items)}
    return {"format": classes, "echo_before_changed": fam_mean_opt(items, echo)}


def donor_metrics(items, texts, donors, exact):
    """T2S_mis / T2S_dmis: agreement with the DONOR item j. donor_match = output state == gold(j);
    donor_match_changed = the same on items whose donor state changes (gold(j) != before(j)), since
    echoing an unchanged donor's before-state would also 'match'; echo_donor_before_changed = output
    == before(j) there. exact_len_subset: the same plus receiver accuracy on exact-length pairs."""
    dm, dmc, ed = {}, {}, {}
    for it in items:
        i, j = it["item_id"], donors[it["item_id"]]
        s = env.parse_state(it["family"], texts[i], lenient=True)
        out = env.dumps(s) if s is not None else None
        dm[i] = out == j["gold"]
        if changed(j):
            dmc[i] = out == j["gold"]
            ed[i] = out == j["gold_prefix_states"][0]
    ex = [it for it in items if exact[it["item_id"]]]
    cr = correct(items, texts)
    rc = {it["item_id"]: cr[it["item_id"]] for it in items if changed(it)}
    return {"donor_match": fam_mean(items, dm), "donor_correct": dm,
            "donor_match_changed": fam_mean_opt(items, dmc), "echo_donor_before_changed": fam_mean_opt(items, ed),
            "exact_len_subset": {"excluded": [it["item_id"] for it in items if not exact[it["item_id"]]],
                                 "changed_lenient": fam_mean_opt(ex, rc),
                                 "donor_match_changed": fam_mean_opt(ex, dmc)}}


def _label(fr):
    if fr is None or not np.isfinite(fr):
        return "undefined"
    return "HI" if fr >= FR_HI else "LO" if fr <= FR_LO else "MID"


def derived(items, results, conds, donors, mis_rec, seed=17, draws=2000) -> dict:
    """Decision-table quantities from the per-item results (no model): per family, on changed items
    (exact-length pairs for T2S_mis/T2S_dmis), acc and FR = (acc - A_S)/(A_T - A_S) with a paired
    parent-cluster bootstrap 95% CI (parents resampled with replacement, the same draws for every
    condition), the HI/MID/LO label; T2S_mis/T2S_dmis donor_match_changed against the paired
    references (T2S_all@L and base:sparse accuracy on the same donors); first-token identities."""
    out = {"rule": f"FR = (acc - A_S)/(A_T - A_S) on changed items; HI FR >= {FR_HI}, LO FR <= {FR_LO}, else MID; "
                   f"undefined if A_T - A_S < {FR_MIN_GAP}; 95% CI: paired parent-cluster bootstrap, {draws} draws, "
                   f"seed {seed}; T2S_mis/T2S_dmis on exact-length pairs only",
           "fr": {}, "mis_reference": {}, "first_token_identity": {}}
    if "base:sparse" not in results or "base:supa" not in results:
        return out
    rng = np.random.default_rng(seed)
    exact = {i: r["exact_len"] for i, r in mis_rec.items()}
    for fam in env.FAMILIES:
        ch = [it for it in items if it["family"] == fam and changed(it)]
        if not ch:
            continue
        parents = sorted({it["parent_id"] for it in ch})
        pix = np.array([parents.index(it["parent_id"]) for it in ch])
        Wd = rng.multinomial(len(parents), np.full(len(parents), 1.0 / len(parents)), size=draws)[:, pix].astype(float)

        def vec(name, subset):
            return np.array([float(results[name]["correct"][it["item_id"]]) for it in ch]) * subset

        for c in conds:
            if c not in results or c.startswith("base:"):
                continue
            kind = c.split("@")[0]
            sub = np.array([1.0 if (kind not in ("T2S_mis", DMIS) or exact[it["item_id"]]) else 0.0 for it in ch])
            W = Wd * sub
            n_w = W.sum(1)
            aT, aS, x = (W @ vec("base:sparse", sub)), (W @ vec("base:supa", sub)), (W @ vec(c, sub))
            with np.errstate(invalid="ignore", divide="ignore"):
                aT, aS, x = aT / n_w, aS / n_w, x / n_w
                fr_b = np.where(aT - aS >= FR_MIN_GAP, (x - aS) / (aT - aS), np.nan)
            m = sub > 0
            if not m.any():
                out["fr"].setdefault(c, {})[fam] = {"n": 0, "acc": None, "A_T": None, "A_S": None, "FR": None,
                                                    "FR_ci95": None, "label": "undefined"}
                continue
            pT, pS, px = (vec("base:sparse", sub)[m].mean(), vec("base:supa", sub)[m].mean(), vec(c, sub)[m].mean())
            fr = (px - pS) / (pT - pS) if pT - pS >= FR_MIN_GAP else None
            ci = [round(float(v), 3) for v in np.nanpercentile(fr_b, [2.5, 97.5])] if np.isfinite(fr_b).any() else None
            out["fr"].setdefault(c, {})[fam] = {"n": int(m.sum()), "acc": round(float(px), 4), "A_T": round(float(pT), 4),
                                                "A_S": round(float(pS), 4), "FR": None if fr is None else round(float(fr), 3),
                                                "FR_ci95": ci, "label": _label(fr)}
    for c in conds:
        kind = c.split("@")[0]
        if c in results and kind in ("T2S_mis", DMIS):
            L = c.split("@")[1]
            ref = results.get(f"T2S_all@{L}")
            rec = {}
            for fam in env.FAMILIES:
                rs = [it for it in items if it["family"] == fam and exact[it["item_id"]] and changed(donors[it["item_id"]])]
                if not rs:
                    continue
                dc = results[c]["donor_correct"]
                rec[fam] = {"n": len(rs), "donor_match_changed": round(float(np.mean([dc[it["item_id"]] for it in rs])), 4),
                            "paired_base_sparse_on_donors": round(float(np.mean(
                                [results["base:sparse"]["correct"][donors[it["item_id"]]["item_id"]] for it in rs])), 4),
                            "paired_T2S_all_on_donors": None if ref is None else round(float(np.mean(
                                [ref["correct"][donors[it["item_id"]]["item_id"]] for it in rs])), 4)}
            out["mis_reference"][c] = rec
        if c in results and kind in COUNTERPART:
            cp = f"{COUNTERPART[kind]}@{c.split('@')[1]}"
            if cp in results:
                a, b = results[c]["first_token"], results[cp]["first_token"]
                out["first_token_identity"][f"{c} vs {cp}"] = round(float(np.mean([a[i] == b[i] for i in a])), 4)
    return out


# -------------------------------------------------------------------------------------------- plan
def plan_counts(a, items, layers, kinds, dmis_layers, hyb_layers) -> dict:
    """Generations and generate() calls of a fresh run (loads: A Titan, B Supa, C Titan if needed)."""
    s_items = [it for f in env.FAMILIES for it in [x for x in items if x["family"] == f][: a.sanity_per_family]]
    n_s, sb = len(s_items), len(batches_of(s_items, a.batch))
    per_load = 1 + 3 * len(layers) + len(dmis_layers)                  # ref + all/last/zero per L + zero-diff per dmis L
    loads = 2 + (1 if ("S2T_all" in kinds or hyb_layers) else 0)
    splice_phases = 2 if hyb_layers else 0                              # HlowT in B, HupT in C
    n_conds = 2 + len(kinds) * len(layers) + len(dmis_layers) + 2 * len(hyb_layers)
    nb = len(batches_of(items, a.batch))
    return {"n_conditions": n_conds, "model_loads": loads, "sanity_generations_per_load": n_s * per_load,
            "splice_sanity_generations": n_s * splice_phases,
            "generations": len(items) * n_conds + n_s * per_load * loads + n_s * splice_phases,
            "generate_calls": nb * n_conds + sb * per_load * loads + sb * splice_phases}


def plan(a, items, enc, layers, kinds, conds, mis_rec, dmis_layers, hyb_layers):
    T = sum(len(enc[it["item_id"]]) for it in items)
    D = hidden_size_hint()
    per_arm = T * D * 4 * len(layers)
    full = plan_counts(a, items, layers, kinds, dmis_layers, hyb_layers)
    spec = plan_counts(a, items, layers, kinds, [], [])
    lens = [len(enc[it["item_id"]]) for it in items]
    gens = full["generations"]
    print(json.dumps({
        "items": len(items), "items_per_family": {f: sum(it["family"] == f for it in items) for f in env.FAMILIES},
        "prompt_tokens": T, "prompt_len_min_max": [min(lens), max(lens)], "hidden": D, "layers": layers,
        "dmis_layers": dmis_layers, "hybrid_layers": hyb_layers, "conditions": conds, "batch": a.batch,
        **full, "generations_spec_only": spec["generations"],
        "additions_share_of_spec": round(gens / spec["generations"] - 1, 4),
        "staging_bytes_per_arm": per_arm, "staging_GiB_total": round(2 * per_arm / 2 ** 30, 3),
        "projected_hours_at_rate": round(gens / a.rate / 3600, 2), "rate_gens_per_s": a.rate,
        "mismatch_len_diff_counts": dict(sorted(Counter(r["len_diff"] for r in mis_rec.values()).items())),
        "mismatch_exact_len_fraction": round(float(np.mean([r["exact_len"] for r in mis_rec.values()])), 4)}, indent=1))


# -------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--conditions", default=",".join(KINDS), help="subset of " + ",".join(KINDS) + " (baselines always run)")
    ap.add_argument("--dmis-layers", default="24", help="T2S_dmis layers (subset of --layers; '' = none)")
    ap.add_argument("--hybrid-layers", default="16,24", help="HlowT/HupT cut layers ('' = none)")
    ap.add_argument("--per-family", type=int, default=None, help="random subset per family (run_steer_cpu rule); default all")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--mis-seed", type=int, default=17)
    ap.add_argument("--out-name", default="patch")
    ap.add_argument("--out-dir", default=None, help="override runs/<bank>/<out-name>")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sanity-per-family", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default generation.max_tokens(family) = 160")
    ap.add_argument("--items-from", default=str(STEER_CPU_RESULTS))
    ap.add_argument("--keep-states", action="store_true")
    ap.add_argument("--threads", type=int, default=2, help="cpu only")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit (loads no model)")
    ap.add_argument("--rate", type=float, default=2.7, help="--plan: greedy generations per second")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cuda = a.device == "cuda"
    if cuda:
        if _TENSOR_DEVICE is None:
            assert not _CPU and os.environ.get("CUDA_VISIBLE_DEVICES") != "", "--device cuda in a CUDA-hidden process"
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
        torch.set_num_threads(a.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    layers = sorted({int(x) for x in a.layers.split(",") if x})
    dmis_layers = sorted({int(x) for x in a.dmis_layers.split(",") if x})
    hyb_layers = sorted({int(x) for x in a.hybrid_layers.split(",") if x})
    kinds = [k for k in KINDS if k in a.conditions.split(",")]
    assert layers and all(L >= 1 for L in layers) and set(a.conditions.split(",")) <= set(KINDS), (layers, a.conditions)
    assert set(dmis_layers) <= set(layers), "--dmis-layers must be staged layers (subset of --layers)"
    assert all(L >= 1 for L in hyb_layers), hyb_layers
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "runs" / a.bank / a.out_name

    pool = load_pool(a.bank)
    items, sel_note = select_items(pool, a.per_family, a.seed, a.items_from)
    tok = runtime.tokenizer()
    _TOK[0] = tok
    enc = {it["item_id"]: tok(generation.render(tok, it["messages"]), add_special_tokens=False)["input_ids"]
           for it in items}
    lens = {i: len(x) for i, x in enc.items()}
    ids_all = [it["item_id"] for it in items]
    B = batches_of(items, a.batch)
    sanity_items = [it for f in env.FAMILIES for it in [x for x in items if x["family"] == f][: a.sanity_per_family]]
    SB = batches_of(sanity_items, a.batch)
    donors, mis_rec = pick_mismatch(items, enc, a.mis_seed)
    conds = (["base:sparse", "base:supa"] + [f"{k}@{L}" for L in layers for k in kinds]
             + [f"{DMIS}@{L}" for L in dmis_layers] + [f"{k}@{L}" for k in HYB for L in hyb_layers])
    max_new_of = (lambda fam: a.max_new_tokens) if a.max_new_tokens else generation.max_tokens
    max_new_key = a.max_new_tokens or {f: generation.max_tokens(f) for f in env.FAMILIES}
    if a.plan:
        plan(a, items, enc, layers, kinds, conds, mis_rec, dmis_layers, hyb_layers)
        return
    out_dir.mkdir(parents=True, exist_ok=True)       # after --plan, which writes nothing

    p_final, p_prog, p_san = out_dir / "patch_results.json", out_dir / "patch_progress.json", out_dir / "patch_sanity.json"
    results = _read_json(p_prog) or _read_json(p_final) or {}
    meta = results.setdefault("_meta", {})
    for k, v in (("items", ids_all), ("batch", a.batch), ("max_new_tokens", max_new_key)):
        if k in meta and meta[k] != v:
            raise SystemExit(f"existing results in {out_dir} have a different {k}; use another --out-name")
    meta.update({"bank": a.bank, "items": ids_all, "item_selection": sel_note, "batch": a.batch,
                 "max_new_tokens": max_new_key, "device": a.device, "dtype": "float32", "seed": a.seed,
                 "sanity_items": [it["item_id"] for it in sanity_items],
                 "prefill_detection": PREFILL_NOTE.format(tv=__import__("transformers").__version__),
                 "torch": torch.__version__, "layer_index": "L = output of block L-1 (hidden_states[L])",
                 "staging": "per-item right-aligned real-token spans, FP32 (module docstring)",
                 "mismatch": {"rule": "same family; different parent, gold and before-state; equal prompt token count "
                                      "if available else nearest (ties -> longer donor); rng(mis_seed) among ties; "
                                      "right-aligned, k = min(n_i, n_j) positions",
                              "mis_seed": a.mis_seed, "per_item": mis_rec,
                              "len_diff_counts": dict(sorted(Counter(str(r["len_diff"]) for r in mis_rec.values()).items())),
                              "exact_len_fraction": round(float(np.mean([r["exact_len"] for r in mis_rec.values()])), 4)}})
    meta["layers_run"] = sorted(set(meta.get("layers_run", [])) | set(layers))
    meta["additions"] = {
        "dmis_layers": dmis_layers, "hybrid_layers": hyb_layers,
        DMIS: "h <- h_own + (T(j) - S(j)) at every real prompt position, j = the T2S_mis donor, right-aligned, "
              "prefill only",
        "HlowT": "Supa with Titan's blocks 0..L-1 at every position (in-place per-tensor splice, hash-checked)",
        "HupT": "Titan with Supa's blocks 0..L-1 at every position (in-place per-tensor splice, hash-checked)",
        "decision_thresholds": {"HI": FR_HI, "LO": FR_LO, "min_gap": FR_MIN_GAP}}
    timing = results.setdefault("_timing", {})
    sanity = _read_json(p_san) or {}
    splice_passed: set[str] = set()                  # hosts whose splice sanity passed in THIS process
    pending = [c for c in conds if c not in results]
    log(f"{len(items)} items ({sel_note}), {len(B)} batches, {sum(lens.values())} prompt tokens; layers {layers}")
    log(f"mismatch donors: len(donor)-len(item) counts {meta['mismatch']['len_diff_counts']}")
    log("done earlier: " + (", ".join(c for c in conds if c in results) or "none"))
    log("pending: " + (", ".join(pending) or "none"))

    stA, stB = Staging(out_dir, "donor_sparse"), Staging(out_dir, "donor_supa")
    st_of = {"sparse": stA, "supa": stB}

    def st_ok(arm):
        ok, why = st_of[arm].check(items, lens, B, layers, EXPECTED_FP[arm], max_new_key)
        return ok, why

    def sanity_ok(arm):
        r = sanity.get(arm, {})
        return bool(r.get("pass")) and str(r.get("fingerprint", "")).startswith(EXPECTED_FP[arm]) \
            and set(layers) <= set(r.get("layers", [])) and r.get("items") == meta["sanity_items"] \
            and set(dmis_layers) <= set(r.get("dmis_layers", [])) and "ref_texts" in r

    def finalize():
        if all(c in results for c in conds):
            results["_sanity"] = sanity
            results["_derived"] = dv = derived(items, results, conds, donors, mis_rec)
            for c, per in dv["fr"].items():
                log(f"decision {c:<14} " + " | ".join(
                    f"{f[:6]} n {v['n']} acc {'-' if v['acc'] is None else format(v['acc'], '.3f')} "
                    f"FR {'-' if v['FR'] is None else v['FR']} {v['FR_ci95'] or ''} {v['label']}"
                    for f, v in per.items()))
            for k, v in dv["first_token_identity"].items():
                log(f"first-token identity {k}: {v:.4f}" + ("" if v == 1.0 else "  WARNING: expected 1.0"))
            meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json(p_final, results)
            if p_prog.exists():
                p_prog.unlink()
            if not a.keep_states:
                for st in (stA, stB):
                    st.delete()
            log(f"wrote {p_final}" + ("" if a.keep_states else "; staged states deleted"))

    if not pending:
        if not p_final.exists() or p_prog.exists():
            finalize()
        log("[patch] nothing pending")
        return

    base_first: dict[str, dict] = {}
    base_texts: dict[str, dict] = {}

    def save_progress():
        _write_json(p_prog, results)

    def gen_path(name):
        return out_dir / f"gen_{name.replace(':', '_').replace('@', '_')}.json"

    def commit(name, texts, first, seconds, extra=None):
        if not (sanity_ok("sparse") and sanity_ok("supa")):
            raise SanityError(f"refusing to write {name}: sanity has not passed for both models")
        kind = name.split("@")[0]
        if kind in HYB and RECEIVER[kind] not in splice_passed:
            raise SanityError(f"refusing to write {name}: splice sanity has not passed on {RECEIVER[kind]}")
        rec = {"score": score(items, texts), "correct": correct(items, texts), "parse": parse_rate(items, texts),
               **text_metrics(items, texts), "first_token": first, "seconds": round(seconds, 1), "n_items": len(items)}
        rec.update(extra or {})
        results[name] = rec
        timing[name] = {"seconds": round(seconds, 1), "s_per_item": round(seconds / max(len(items), 1), 3)}
        gen_path(name).write_text(json.dumps(texts) + "\n", encoding="utf-8")
        save_progress()
        s = rec["score"]
        line = f"{name:<16} " + " | ".join(f"{f[:6]} strict {v['strict']:.3f} len {v['lenient']:.3f} chg {v['changed_lenient']:.3f}"
                                           for f, v in s.items())
        line += " | parse " + "/".join(f"{v:.2f}" for v in rec["parse"].values())
        line += " | bare " + "/".join(f"{v['bare']:.2f}" for v in rec["format"].values())
        if "ft_donor" in rec and "prerequisite_plan" in rec["ft_donor"]:
            line += f" | ft donor P {rec['ft_donor']['prerequisite_plan']:.3f}"
        if "donor_match_changed" in rec:
            line += " | donor_match_chg " + "/".join("-" if v is None else f"{v:.3f}" for v in rec["donor_match_changed"].values())
        if "ft_equals_counterpart" in rec:
            line += f" | ft == {rec['ft_equals_counterpart']['condition']} {rec['ft_equals_counterpart']['fraction']:.3f}"
        if rec.get("noop_expected"):
            line += f" | noop_exact {rec['noop_exact']}"
        print(line + f" ({seconds:.0f}s)", flush=True)
        if _on_condition_done is not None:
            _on_condition_done(name)

    def baselines_from_results():
        for arm in ("sparse", "supa"):
            n = f"base:{arm}"
            if n in results and arm not in base_first:
                base_first[arm] = results[n]["first_token"]
                base_texts[arm] = json.loads(gen_path(n).read_text(encoding="utf-8"))

    def try_commit_bases():
        for arm in ("sparse", "supa"):
            n = f"base:{arm}"
            if n in results or not (sanity_ok("sparse") and sanity_ok("supa")):
                continue
            ok, _ = st_ok(arm)
            if ok:
                idx = _read_json(st_of[arm].index_path)
                commit(n, idx["texts"], idx["first_token"], idx.get("seconds", 0.0),
                       {"staged_from": st_of[arm].prefix, "fingerprint": idx["fingerprint"]})
        baselines_from_results()

    def stage(model, arm, prov):
        st = st_of[arm]
        t0 = time.time()
        st.begin(arm, items, lens, B, layers, model.config.hidden_size, max_new_key)
        texts, first = {}, {}
        for b in B:
            o = run_batch(be, model, b, enc, max_new_of(b[0]["family"]), capture_layers=layers)
            st.write_batch(b, o["ids"], o["cap"])
            texts.update(o["texts"])
            first.update(o["first"])
        secs = time.time() - t0
        st.finish(texts=texts, first_token=first, fingerprint=prov["param_fingerprint"], seconds=round(secs, 1),
                  block_hashes=meta.get(f"hash_{arm}"))
        n = f"base:{arm}"
        if n in results:                             # restaged after an earlier commit: determinism record
            old = json.loads(gen_path(n).read_text(encoding="utf-8"))
            meta.setdefault("restage_text_mismatch", {})[arm] = sum(old[i] != texts[i] for i in ids_all)
        log(f"staged {arm}: base texts + layers {layers}, {st.nbytes() / 2 ** 20:.0f} MiB ({secs:.0f}s)")

    def run_sanity(model, arm, prov):
        t0 = time.time()
        st = Staging(out_dir, f"sanity_{arm}")
        rec = {"pass": False, "fingerprint": prov["param_fingerprint"], "layers": layers, "dmis_layers": dmis_layers,
               "items": meta["sanity_items"], "device": a.device, "per_layer": {}}
        try:
            st.begin(arm, sanity_items, lens, SB, layers, model.config.hidden_size, max_new_key)
            ref = {}
            for b in SB:
                o = run_batch(be, model, b, enc, max_new_of(b[0]["family"]), capture_layers=layers)
                st.write_batch(b, o["ids"], o["cap"])
                ref.update(o["texts"])
            st.finish(texts=ref, fingerprint=prov["param_fingerprint"])
            st.open()
            for L in layers:
                r = {}
                checks = [("all", lambda b: same_item_patch(st, b, enc, L, "all")),
                          ("last", lambda b: same_item_patch(st, b, enc, L, "last"))]
                if L in dmis_layers:                 # additive path (T2S_dmis) with a zero difference
                    checks.append(("add_zero_diff", lambda b: zero_diff_patch(st, b, enc, L)))
                for which, mk in checks:
                    got, infos = {}, []
                    for b in SB:
                        o = run_batch(be, model, b, enc, max_new_of(b[0]["family"]), L, mk(b))
                        got.update(o["texts"])
                        infos.append(o["info"])
                    diff = [i for i in ref if got[i] != ref[i]]
                    mx = max(x["max_abs"] for x in infos)
                    r[f"identity_{which}"] = {
                        "texts_identical": len(ref) - len(diff), "n": len(ref), "differ": diff, "patch_max_abs_diff": mx,
                        "prefill_patches": sum(x["prefill_patched"] for x in infos), "batches": len(infos),
                        "decode_calls_unpatched": sum(x["decode_calls"] for x in infos),
                        "generated_tokens": sum(x["n_new"] for x in infos),
                        "positions_patched": sum(x["n_patched"] for x in infos)}
                    if diff or mx != 0.0:
                        rec["per_layer"][str(L)] = r
                        raise SanityError(f"{arm} identity {which}@{L}: {len(diff)} texts differ, patch max|d| {mx}")
                got = {}
                for b in SB:
                    got.update(run_batch(be, model, b, enc, max_new_of(b[0]["family"]), L, zero_last_patch(model.config.hidden_size))["texts"])
                n_changed = sum(got[i] != ref[i] for i in ref)
                r["zero_last_changed"] = n_changed
                rec["per_layer"][str(L)] = r
                if n_changed == 0:
                    raise SanityError(f"{arm} zero-vector@{L} changed no output: the hook is not live")
                log(f"sanity {arm} L{L}: identity {'/'.join(w for w, _ in checks)} exact on {len(ref)}/{len(ref)}, max|d| 0, "
                    f"hook prefill {r['identity_all']['prefill_patches']}/{r['identity_all']['batches']} batches, "
                    f"decode unpatched {r['identity_all']['decode_calls_unpatched']}; zero@last changed {n_changed}/{len(ref)}")
            rec["ref_texts"] = ref                   # reference for the splice sanity of the other core
            rec["pass"] = True
        finally:
            rec["seconds"] = round(time.time() - t0, 1)
            sanity[arm] = rec
            _write_json(p_san, sanity)
            st.delete()
        return rec

    def run_condition(model, name):
        kind, L = name.split("@")
        L = int(L)
        don, rcv = DONOR[kind], RECEIVER[kind]
        opened = [st_of[arm].open() for arm in STAGED[kind]]
        st = st_of[don]
        try:
            t0 = time.time()
            texts, first, infos = {}, {}, []
            for b in B:
                if kind == "T2S_mis":
                    fn = mismatch_patch(st, b, enc, donors, L)
                elif kind == DMIS:
                    fn = diff_mismatch_patch(stA, stB, b, enc, donors, L)
                else:
                    fn = same_item_patch(st, b, enc, L, "last" if kind == "T2S_last" else "all")
                o = run_batch(be, model, b, enc, max_new_of(b[0]["family"]), L, fn)
                texts.update(o["texts"])
                first.update(o["first"])
                infos.append(o["info"])
            secs = time.time() - t0
        finally:
            for s_ in opened:
                s_.close()
        dsrc = {i: (donors[i]["item_id"] if kind == "T2S_mis" else i) for i in ids_all}
        ftd = {i: first[i] == base_first[don][dsrc[i]] for i in ids_all}
        ftr = {i: first[i] == base_first[rcv][i] for i in ids_all}
        same = {i: texts[i] == base_texts[rcv][i] for i in ids_all}
        extra = {"donor": don, "receiver": rcv, "layer": L,
                 "ft_donor": fam_mean(items, ftd), "ft_receiver": fam_mean(items, ftr),
                 "ft_donor_all": round(float(np.mean(list(ftd.values()))), 4),
                 "ft_receiver_all": round(float(np.mean(list(ftr.values()))), 4),
                 "text_equals_receiver_base": fam_mean(items, same),
                 "patch": {"batches": len(infos), "prefill_patches": sum(x["prefill_patched"] for x in infos),
                           "decode_calls_unpatched": sum(x["decode_calls"] for x in infos),
                           "generated_tokens": sum(x["n_new"] for x in infos),
                           "positions_patched": sum(x["n_patched"] for x in infos),
                           "max_abs_diff": max(x["max_abs"] for x in infos),
                           "mean_rel_diff": round(float(np.mean([x["rel"] for x in infos])), 6)}}
        if kind in ("T2S_mis", DMIS):
            extra.update(donor_metrics(items, texts, donors, {i: r["exact_len"] for i, r in mis_rec.items()}))
        if kind == DMIS:                             # first token vs Titan's own first token of the DONOR item
            extra["ft_vs_donor_item_sparse"] = fam_mean(items, {i: first[i] == base_first["sparse"][donors[i]["item_id"]]
                                                                for i in ids_all})
        if kind != "T2S_mis" and L in meta.get("noop_layers", []):
            extra.update(noop_expected=True, noop_exact=all(same.values()))
            if not all(same.values()) or extra["patch"]["max_abs_diff"] != 0.0:
                log(f"WARNING {name}: expected an exact no-op (identical blocks below L) but "
                    f"{sum(not v for v in same.values())} texts differ, patch max|d| {extra['patch']['max_abs_diff']}")
        commit(name, texts, first, secs, extra)

    def hashes():
        if "hash_sparse" not in meta or "hash_supa" not in meta:
            raise SystemExit("parameter hashes of both cores are needed for the hybrids")
        return {"sparse": meta["hash_sparse"], "supa": meta["hash_supa"]}

    def splice_sanity(model, host, cur):
        """Before the first hybrid on this host: splice EVERY differing block from the other core (hash
        gate inside set_block_sources) and regenerate the sanity items: the result must be the other
        core, whose sanity reference texts were recorded on the same batches."""
        other = "supa" if host == "sparse" else "sparse"
        t0 = time.time()
        rec = {"pass": False, "host": host, "from": other, "host_fingerprint": meta.get(f"fingerprint_{host}")}
        try:
            if not sanity_ok(other):
                raise SanityError(f"splice sanity on {host}: no passed {other} sanity record with reference texts")
            ref = sanity[other]["ref_texts"]
            rec["spliced_blocks"] = set_block_sources(model, cur, [other] * len(cur), hashes(), be.dev)
            rec["hash_equal_to_other_core"] = True
            got = {}
            for b in SB:
                got.update(run_batch(be, model, b, enc, max_new_of(b[0]["family"]))["texts"])
            diff = [i for i in ref if got[i] != ref[i]]
            rec.update(texts_identical=len(ref) - len(diff), n=len(ref), differ=diff)
            if diff:
                log(f"WARNING splice sanity on {host}: blocks hash-equal {other}'s but {len(diff)}/{len(ref)} texts "
                    f"differ from {other}'s sanity texts (cross-load nondeterminism)")
            rec["pass"] = True
            splice_passed.add(host)
            log(f"splice sanity on {host}: {len(rec['spliced_blocks'])} blocks from {other} hash-equal; "
                f"{len(ref) - len(diff)}/{len(ref)} texts equal {other}'s sanity texts")
        finally:
            rec["seconds"] = round(time.time() - t0, 1)
            sanity[f"splice_{host}"] = rec
            _write_json(p_san, sanity)

    def run_hybrid(model, name, cur):
        kind, L = name.split("@")
        L = int(L)
        host, src = RECEIVER[kind], DONOR[kind]
        t_s = time.time()
        spliced = set_block_sources(model, cur, [src if i < L else host for i in range(len(cur))], hashes(), be.dev)
        t0 = time.time()
        texts, first = {}, {}
        for b in B:
            o = run_batch(be, model, b, enc, max_new_of(b[0]["family"]))
            texts.update(o["texts"])
            first.update(o["first"])
        secs = time.time() - t0
        hs = hashes()
        differing = [i for i in range(min(L, len(cur))) if hs[src][f"block{i}"] != hs[host][f"block{i}"]]
        ftd = {i: first[i] == base_first[src][i] for i in ids_all}
        ftr = {i: first[i] == base_first[host][i] for i in ids_all}
        same = {i: texts[i] == base_texts[host][i] for i in ids_all}
        extra = {"donor": src, "receiver": host, "layer": L, "host": host, "blocks_below_L_from": src,
                 "differing_blocks_from_src": differing, "spliced_in_this_step": spliced,
                 "splice_seconds": round(t0 - t_s, 1),
                 "ft_donor": fam_mean(items, ftd), "ft_receiver": fam_mean(items, ftr),
                 "ft_donor_all": round(float(np.mean(list(ftd.values()))), 4),
                 "ft_receiver_all": round(float(np.mean(list(ftr.values()))), 4),
                 "text_equals_receiver_base": fam_mean(items, same),
                 "text_equals_donor_base": fam_mean(items, {i: texts[i] == base_texts[src][i] for i in ids_all})}
        cp = f"{COUNTERPART[kind]}@{L}"
        if cp in results:
            eq = [first[i] == results[cp]["first_token"][i] for i in ids_all]
            extra["ft_equals_counterpart"] = {"condition": cp, "fraction": round(float(np.mean(eq)), 4)}
            if not all(eq):
                log(f"WARNING {name}: first token differs from {cp} on {len(eq) - sum(eq)} items "
                    f"(same prompt computation expected; cross-load nondeterminism?)")
        if not differing:
            extra.update(noop_expected=True, noop_exact=all(same.values()))
        commit(name, texts, first, secs, extra)

    def run_hybrids(model, host, todo):
        """Splice sanity, then the pending hybrids on this host in ascending L (incremental splices)."""
        if not todo:
            return
        cur = [host] * model.config.num_hidden_layers
        splice_sanity(model, host, cur)
        for c in sorted(todo, key=lambda c: int(c.split("@")[1])):
            run_hybrid(model, c, cur)

    def model_phase(arm, body):
        model, prov = be.load(arm)
        try:
            n_layers = model.config.num_hidden_layers
            if max(layers + hyb_layers) > n_layers:
                raise SystemExit(f"layer {max(layers + hyb_layers)} > num_hidden_layers {n_layers}")
            hs = group_hashes(model)
            if meta.get(f"hash_{arm}") not in (None, hs):
                raise SystemExit(f"{arm}: parameter hashes changed between loads")
            meta[f"hash_{arm}"] = hs
            meta[f"fingerprint_{arm}"] = prov["param_fingerprint"]
            other = "supa" if arm == "sparse" else "sparse"
            if f"hash_{other}" not in meta:          # resumed run: the other arm's hashes from its staging index
                h_o = (_read_json(st_of[other].index_path) or {}).get("block_hashes")
                if h_o:
                    meta[f"hash_{other}"] = h_o
            if "hash_sparse" in meta and "hash_supa" in meta:
                meta["noop_layers"] = noop_layers(meta["hash_sparse"], meta["hash_supa"], n_layers)
                meta["identical_groups"] = sorted(k for k in hs if meta["hash_sparse"].get(k) == meta["hash_supa"].get(k))
            body(model, prov)
        finally:
            del model
            be.release()

    def body_A(model, prov):
        run_sanity(model, "sparse", prov)
        if not st_ok("sparse")[0] and ("base:sparse" in pending or t2s_p):
            stage(model, "sparse", prov)

    def body_B(model, prov):
        run_sanity(model, "supa", prov)
        log(f"identical parameter groups across arms: {meta.get('identical_groups')}; "
            f"exact no-op layers {meta.get('noop_layers')}")
        try_commit_bases()
        if not st_ok("supa")[0] and ("base:supa" not in results or s2t_p or dmis_p):
            stage(model, "supa", prov)
        try_commit_bases()
        if t2s_p and not st_ok("sparse")[0]:
            raise SystemExit("Titan staging missing for T2S conditions")
        for c in conds:                              # patches first: the hybrids below rewrite the weights
            if c in t2s_p and c not in results:
                run_condition(model, c)
        run_hybrids(model, "supa", [c for c in hyb_B if c not in results])

    def body_C(model, prov):
        run_sanity(model, "sparse", prov)
        try_commit_bases()
        if s2t_p and not st_ok("supa")[0]:
            raise SystemExit("Supa staging missing for S2T conditions")
        for c in conds:
            if c in s2t_p and c not in results:
                run_condition(model, c)
        run_hybrids(model, "sparse", [c for c in hyb_C if c not in results])

    # ---- plan the phases (after committing baselines whose staging and sanity already exist) ----
    try_commit_bases()
    pending = [c for c in conds if c not in results]
    kind_of = {c: c.split("@")[0] for c in pending if "@" in c}
    t2s_p = [c for c, k in kind_of.items() if k not in HYB and RECEIVER[k] == "supa"]     # phase B patches
    s2t_p = [c for c, k in kind_of.items() if k not in HYB and RECEIVER[k] == "sparse"]   # phase C patches
    dmis_p = [c for c, k in kind_of.items() if k == DMIS]                                 # need both stagings
    hyb_B = [c for c, k in kind_of.items() if k in HYB and RECEIVER[k] == "supa"]
    hyb_C = [c for c, k in kind_of.items() if k in HYB and RECEIVER[k] == "sparse"]
    okA, whyA = st_ok("sparse")
    okB, whyB = st_ok("supa")
    need_stA = ("base:sparse" in pending or bool(t2s_p)) and not okA
    need_stB = ("base:supa" in pending or bool(s2t_p) or bool(dmis_p)) and not okB
    phaseA = bool(pending) and (need_stA or not sanity_ok("sparse"))
    phaseB = bool(pending) and (need_stB or bool(t2s_p) or bool(hyb_B) or not sanity_ok("supa") or "base:supa" in pending)
    phaseC = bool(s2t_p) or bool(hyb_C)
    log(f"staging sparse: {'reuse' if okA else whyA}; supa: {'reuse' if okB else whyB}; "
        f"phases A={phaseA} B={phaseB} C={phaseC}")
    D0 = hidden_size_hint()
    T = sum(lens.values())
    need = (int(need_stA) + int(need_stB)) * T * D0 * 4 * len(layers) + 2 * len(sanity_items) * 200 * D0 * 4 * len(layers)
    free = shutil.disk_usage(out_dir).free
    log(f"disk: staging needs {need / 2 ** 30:.2f} GiB (+{DISK_MARGIN_GIB} margin), free {free / 2 ** 30:.1f} GiB")
    if free < need + DISK_MARGIN_GIB * 2 ** 30:
        raise SystemExit("not enough free disk for the staged states")

    be = Backend(a.device)
    if phaseA or phaseB or phaseC:
        with be.lease():                             # ONE lease for the whole job (cuda); no-op on cpu
            if phaseA:
                model_phase("sparse", body_A)
            if phaseB:
                model_phase("supa", body_B)
            if phaseC:
                model_phase("sparse", body_C)
    finalize()
    if not all(c in results for c in conds):
        missing = [c for c in conds if c not in results]
        raise SystemExit(f"conditions still missing after the run: {missing}")
    log("[patch] finished")


if __name__ == "__main__":
    try:
        main()
    except SanityError as e:
        log(f"SANITY FAILED, aborting with no result written: {e}")
        sys.exit(3)
