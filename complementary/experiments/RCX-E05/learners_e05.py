"""RCX-E05 learned arms: SF, UVFA, UVFA+aux, per-goal PGV / FT-UVFA / SF-FT in a BatchedMLP, renaming and stabilizer
augmentation, a resumable trainer with safe points and early stopping, the library psi-hat cache, inference, and the M2
unit sizing (PREREG_E05_DRAFT.md §5.1-5.5, §10.2; DESIGN.md rev 3 §5). CPU only: CUDA_VISIBLE_DEVICES = '' before torch
is imported (if it is not yet) and torch.cuda patched to report no device, as in rcx/bw/models.py; <= 4 torch threads
(set_threads; default 4, or E05_TORCH_THREADS).

Inputs (§5.1). State x_s (n*n + 3n = 70 at n = 7): on-matrix (x[i*n + j] = 1 iff block i is on block j), ontable (n),
holding (n), clear (n). Goal/policy code z = (w / 3, c) (F + 1 = 57), float32. Every network is depth x [Linear,
LayerNorm, ReLU] at width 256 plus a linear head (n = 7 parameters: SF 180,537, UVFA 166,145, UVFA+aux 180,794, PGV
151,553 / 43,009 / 13,313 at width 256 / 128 / 64; tested). Network kinds (outputs):
  sf    psi-hat(x_s, z): F endpoint-fact logits (sigmoid) + 1 length unit (softplus T-hat); loss BCE (mean over facts)
        + Huber(T-hat/10, T/10)
  uvfa  V(x_s, z) -> V*; Huber (delta 1)
  aux   UVFA+aux: out[0] = V (used at test), out[1:] = the sf head; loss Huber(V) + 1.0 * sf loss
  pgv   per-goal V(x_s), no goal input; Huber
FT-UVFA_400 = a uvfa network fine-tuned with its code input fixed to the parent's (w'/3, c'); SF-FT_400 = an sf network
likewise (psi targets).

BatchedMLP. B independent networks stacked: per-model weights [B, in, out] through torch.baddbmm, per-model LayerNorm
affine parameters, one AdamW over the stacked tensors (elementwise, so = B independent optimizers; the loss is the SUM
of per-model means, so each model's gradient is its own). A zero-shot arm is a BatchedMLP with B = 1. Equivalence with B
separately trained nn.Module networks is tested (50 steps, < 1e-5).

Trainer. Data order: per model a numpy PCG64 stream; mode "epoch" (a fresh permutation per epoch, last batch short;
zero-shot arms) or "stream" (permutations concatenated, every batch full; per-goal arms). Augmentation per example from
a second per-model stream: "rename" (sigma uniform over all n! renamings, applied jointly to x_s, the code's w part and
the psi target's fact part; V and T unchanged; zero-shot arms), "stab" (sigma uniform over Stab(w'), exact and
label-preserving; per-goal arms). Validation rows are never augmented. Early stopping: per model and per horizon h, the
weights with the lowest validation loss among the checks at steps <= h (every `check_every` steps, plus step 0 when
step0 = True: FT's zero-shot candidate); strict improvement only, so ties go to the earlier step. Plain snapshots at
`snap_steps` (the 20-epoch checkpoint of the 40-epoch run; it equals an independent 20-epoch run, tested).

Safe points (M2). Trainer.run(safe_point, clock, min_interval, ckpt) calls safe_point(trainer) after a step that ends an
epoch and after any step once >= min_interval (60) seconds have passed since the previous call (min_interval = 0: after
every step); never after the final step (the caller's batch-end check follows). `trainer.epoch_end` tells which. True =
yield: the checkpoint is written atomically (every model's weights, AdamW state, early-stopping and plain snapshots,
data-order and augmentation stream states, queues, step, validation history, fingerprint) and run returns "yielded"; a
new Trainer built with the same arguments + load() resumes the run exactly (tested <= 1e-6; bit-identical in practice
at fixed threads). load() refuses a checkpoint whose fingerprint differs. LibraryPsi.build has the same protocol.
Unit sizing: one uninterruptible unit = one optimizer step of a BatchedMLP batch (6 x params x batch x B FLOP); at the
planning rate (19 GFLOP/s before G-P5, else the measured rate) none may exceed 30 s (check_units, max_models), so
checks are <= 60 + 30 = 90 s apart, and <= 120 s at half the rate (check_gap; tested with injected costs).

Library psi-hat cache and inference. LibraryPsi: psi-hat(s, z) for every state and library code, float16 [|L|, N, F+1],
built one code at a time in chunks of 4,096 states (memory or an .npy memmap with a progress file; resumable). Evaluator:
cached single-state forward passes. sf_scores (SF-GPI L + {w'}, SF-noGPI {w'}, SF-L L, SF-GPI-5 {w'} + nearest_codes)
and value_scores give soft.scored_policy move scores; sf_values = w-bar.psi-hat. Every value depends only on (model,
state, code), never on the order of cells or a resume (fixed forward composition).

Low-level API used by run_e05's Adapters: state_features, label_rows, shuffle_psi, zero_shot_trainer, pergoal_task,
pergoal_trainer, pergoal_seeds, stab_ids, Evaluator, LibraryPsi, sf_scores, value_scores, sf_values, nearest_codes,
goal_code, BatchedMLP, MLP, new_model, arm_dims, arm_loss, n_facts, set_threads, torch.
Higher-level entry points (tested; usable instead of hand-assembled trainers):
  train_zero_shot(arm, space, data, *, seed, epochs, lr, keep, out_dir, safe_point) -> {epoch: model path} | None
      arm SF | UVFA | UVFA+aux | SF-shuffled; data rows s/states, g/goal, psi, v and W, c per row or per goal id;
      trains max(keep) epochs (= the snapshots of the `epochs`-epoch run: constant lr); None = yielded.
  fit_pergoal(kind, space, jobs, cfg, *, out_dir, safe_point) -> {job key: model path} | None
      kind PGV | FT | SF-FT; jobs = [dict(key, states, v (psi for SF-FT), stab (permutation rows), seed, w, c, init
      (FT/SF-FT: replicate model path))]; cfg = width (PGV), steps, keep (= the early-stopping horizon; min(steps, keep)
      steps are run, identical to the keep-step checkpoint of the longer run), lr, batch, eval_every, val_n or
      val_frac, step0, optional wd; BatchedMLP batches of B = min(64, max_models) models, safe_point("batch") after
      each; per job: PGV init seed = job seed, order derive(seed, "order"), stabilizer derive(seed, "stab").
  load_model(path) (this module's files and run_e05's {"kind", "dims", "state"} files), predict_psi(model, space,
  states, W, c) -> [k, m, F + 1], predict_value(model, space, states, W, c) -> [k, m] (pgv ignores the code),
  time_step(model, width, batch, B, steps, n) (G-P5: real Trainer steps on synthetic rows).
  Their safe_point takes a kind ("step" | "epoch" | "batch") and is called after every step (min_interval 0); the
  callee decides whether a lane check is due.

Seeds (App. A; tags not fixed by the prereg are marked *): zero-shot init torch.manual_seed(replicate seed), data order
derive(seed, "order")*, renaming stream derive(20261243, "rename:<seed>")*, SF-shuffled stream 20261243:shuffle (goals
in id order, Python random.shuffle of each goal's row positions); pergoal_seeds: PGV init, order and stabilizer streams
derive(20261249, "pgv-init|pgv-order|pgv-stab:<key>:<k>")*, FT order and stabilizer derive(20261249,
"ft-order|ft-stab:<key>:<replicate>")*.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

if "torch" not in sys.modules:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def cuda_off() -> None:
    """Report no CUDA device (RCWM I-21 recipe, as rcx/bw/models.py)."""
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0


cuda_off()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.bw import soft as SO  # noqa: E402

SEEDS = {"train_labels": 20261243, "selection": 20261245, "replicates": (20261246, 20261247, 20261248),
         "pergoal": 20261249}
WIDTH, DEPTH, HUBER_DELTA, T_SCALE, LAMBDA_AUX, WD, ZS_BATCH = 256, 3, 1.0, 10.0, 1.0, 1e-4, 256
MAX_THREADS, B_MAX = 4, 64
UNIT_MAX_S, CHECK_MIN_S, PLAN_GFLOPS = 30.0, 60.0, 19.0
KINDS = ("sf", "uvfa", "aux", "pgv")
ARM_KIND = {"SF": "sf", "SF-shuffled": "sf", "UVFA": "uvfa", "UVFA+aux": "aux"}
PG_KIND = {"PGV": "pgv", "FT": "uvfa", "SF-FT": "sf"}
MODEL_KIND = {"SF": ("sf", 256), "UVFA": ("uvfa", 256), "UVFA+aux": ("aux", 256), "PGV64": ("pgv", 64),
              "PGV128": ("pgv", 128), "PGV256": ("pgv", 256)}
CONFIGS = {  # §5.2-5.3 (fixed now); run_e05 assembles the per-goal configs itself
    "zero-shot": dict(width=256, lr=1e-3, wd=1e-4, batch=256, epochs=40, snap_epochs=(20, 40)),
    "uvfa-ext": dict(width=256, lr=3e-4, wd=1e-4, batch=256, epochs=80, snap_epochs=(80,)),
    "pgv400": dict(widths=(64, 256), lr=1e-3, wd=1e-4, batch=64, steps=2000, horizons=(500, 2000), check_every=100,
                   val_frac=0.1),
    "pgv400-ext": dict(width=128, lr=3e-4, wd=1e-4, batch=64, steps=4000, horizons=(4000,), check_every=100,
                       val_frac=0.1),
    "pgv1600": dict(width=256, lr=1e-3, wd=1e-4, batch=128, steps=2000, horizons=(2000,), check_every=100,
                    val_frac=0.1),
    "pgv6400": dict(width=256, lr=1e-3, wd=1e-4, batch=256, steps=2500, horizons=(2500,), check_every=100,
                    val_frac=0.1),
    "r-ext": dict(width=128, lr=3e-4, wd=1e-4, batch={400: 64, 1600: 128, 6400: 256}, steps=4000, horizons=(4000,),
                  check_every=200, val_frac=0.1),
    "ft400": dict(lr=3e-4, wd=1e-4, batch=64, steps=400, horizons=(100, 400), check_every=25, step0=True,
                  val_frac=0.1),
    "ft400-ext": dict(lr=1e-4, wd=1e-4, batch=64, steps=1000, horizons=(1000,), check_every=25, step0=True,
                      val_frac=0.1),
}
CACHE_BYTES = 256 * 2 ** 20
FORMAT = 2
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def derive(seed: int, tag: str) -> int:
    """App. A: int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big") (= families_e05.derive)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def default_threads() -> int:
    return int(os.environ.get("E05_TORCH_THREADS", MAX_THREADS))


def set_threads(threads: int) -> None:
    if not 1 <= int(threads) <= MAX_THREADS:
        raise ValueError(f"torch threads must be 1..{MAX_THREADS}")
    if torch.get_num_threads() != int(threads):
        torch.set_num_threads(int(threads))


def _canon(obj):
    """JSON round trip (tuples -> lists, sorted keys): the comparable form of a fingerprint."""
    return json.loads(json.dumps(obj, sort_keys=True))


def _sha_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- inputs and renaming

def state_dim(n: int) -> int:
    return n * n + 3 * n


def n_facts(n: int) -> int:
    return n * (n + 1)


def state_features(space) -> np.ndarray:
    """x_s for every state: float32 [N, n*n + 3n] (on-matrix, ontable, holding, clear)."""
    S, n = space.states.astype(np.int64), space.n
    on = (S[:, :, None] == np.arange(n)[None, None, :]).reshape(len(S), n * n)
    held = S == n + 1
    covered = np.stack([(S == i).any(axis=1) for i in range(n)], axis=1)
    return np.concatenate([on, S == n, held, ~held & ~covered], axis=1).astype(np.float32)


def features(space) -> np.ndarray:
    """state_features cached on the space object (read-only use)."""
    X = getattr(space, "_e05_X", None)
    if X is None:
        X = space._e05_X = state_features(space)
    return X


def goal_codes(W, c) -> np.ndarray:
    """z = (w / 3, c) per row: float32 [m, F + 1] (W [m, F] or [F]; c scalar or [m])."""
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    c = np.broadcast_to(np.asarray(c, dtype=np.float64), (len(W),))
    return np.concatenate([W / 3.0, c[:, None]], axis=1).astype(np.float32)


def goal_code(w, c: float) -> np.ndarray:
    """z = (w / 3, c), float32 [F + 1]."""
    return goal_codes(w, c)[0]


@lru_cache(maxsize=None)
def perms(n: int) -> np.ndarray:
    """All renamings sigma (block i -> sigma[i]) in itertools order (= families_e05.perms)."""
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


@lru_cache(maxsize=None)
def _perm_index(n: int) -> dict:
    return {tuple(int(v) for v in p): i for i, p in enumerate(perms(n))}


def perm_ids(sigmas, n: int) -> np.ndarray:
    """Indices into perms(n) of permutation rows (e.g. families_e05.stabilizer(w, n))."""
    idx = _perm_index(n)
    return np.array([idx[tuple(int(v) for v in s)] for s in np.atleast_2d(sigmas)], dtype=np.int64)


@lru_cache(maxsize=None)
def gathers(n: int) -> tuple:
    """(Gs [n!, D], Gf [n!, F]): x renamed by perms(n)[r] is x[Gs[r]]; a fact vector v renamed is v[Gf[r]]."""
    P, D, Fn = perms(n), state_dim(n), n_facts(n)
    i, j = np.divmod(np.arange(n * n), n)
    Gs, Gf = np.empty((len(P), D), dtype=np.int64), np.empty((len(P), Fn), dtype=np.int64)
    for r, sig in enumerate(P):
        Gs[r, np.concatenate([sig[i] * n + sig[j], n * n + sig, n * n + n + sig, n * n + 2 * n + sig])] = np.arange(D)
        Gf[r, SO.fact_perm(n, sig)] = np.arange(Fn)
    return Gs, Gf


def stab_ids(w, n: int) -> np.ndarray:
    """Indices r into perms(n) with sigma_r(w) = w (rows = families_e05.stabilizer(w, n))."""
    w = np.asarray(w)
    return np.flatnonzero((w[gathers(n)[1]] == w).all(axis=1))


# ----------------------------------------------------------------------------- networks

class MLP(nn.Module):
    """depth x [Linear, LayerNorm, ReLU] at `width`, then a linear head."""

    def __init__(self, d_in: int, d_out: int, width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        self.dims = (int(d_in), int(d_out), int(width), int(depth))
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.ReLU()]
            d = width
        self.trunk, self.head = nn.Sequential(*layers), nn.Linear(width, d_out)

    def forward(self, x):
        return self.head(self.trunk(x))


def arm_dims(kind: str, n: int) -> tuple:
    if kind not in KINDS:
        raise ValueError(kind)
    D, Fn = state_dim(n), n_facts(n)
    return (D if kind == "pgv" else D + Fn + 1), {"sf": Fn + 1, "uvfa": 1, "aux": Fn + 2, "pgv": 1}[kind]


def new_model(kind: str, n: int, seed: int, width: int = WIDTH, depth: int = DEPTH) -> MLP:
    """Fresh network, initialized under torch.manual_seed(seed) (the global RNG state is restored)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return MLP(*arm_dims(kind, n), width, depth)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def weights_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, t in model.state_dict().items():
        t = t.detach().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class BatchedMLP(nn.Module):
    """B independent MLPs: x [B, M, d_in] -> [B, M, d_out] (see module doc)."""

    def __init__(self, B: int, d_in: int, d_out: int, width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        self.B, self.dims = B, (d_in, d_out, width, depth)
        ins = [d_in] + [width] * (depth - 1)
        P = lambda *s: nn.Parameter(torch.zeros(*s))  # noqa: E731
        self.W = nn.ParameterList([P(B, a, width) for a in ins])
        self.b = nn.ParameterList([P(B, 1, width) for _ in ins])
        self.g = nn.ParameterList([nn.Parameter(torch.ones(B, 1, width)) for _ in ins])
        self.beta = nn.ParameterList([P(B, 1, width) for _ in ins])
        self.hW, self.hb = P(B, width, d_out), P(B, 1, d_out)

    def forward(self, x):
        width = self.dims[2]
        for W, b, g, be in zip(self.W, self.b, self.g, self.beta):
            x = torch.relu(F.layer_norm(torch.baddbmm(b, x, W), (width,)) * g + be)
        return torch.baddbmm(self.hb, x, self.hW)

    @classmethod
    def from_models(cls, models) -> "BatchedMLP":
        bm = cls(len(models), *models[0].dims)
        with torch.no_grad():
            for k, m in enumerate(models):
                if m.dims != bm.dims:
                    raise ValueError("models differ in shape")
                lins = [x for x in m.trunk if isinstance(x, nn.Linear)]
                lns = [x for x in m.trunk if isinstance(x, nn.LayerNorm)]
                for i, (lin, ln) in enumerate(zip(lins, lns)):
                    bm.W[i][k].copy_(lin.weight.T)
                    bm.b[i][k, 0].copy_(lin.bias)
                    bm.g[i][k, 0].copy_(ln.weight)
                    bm.beta[i][k, 0].copy_(ln.bias)
                bm.hW[k].copy_(m.head.weight.T)
                bm.hb[k, 0].copy_(m.head.bias)
        return bm

    def to_models(self, state: dict | None = None) -> list:
        """The B networks as MLPs (from the current parameters or a {name: tensor [B, ...]} snapshot)."""
        st = dict(self.named_parameters()) if state is None else state
        out = []
        with torch.no_grad():
            for k in range(self.B):
                m = MLP(*self.dims)
                lins = [x for x in m.trunk if isinstance(x, nn.Linear)]
                lns = [x for x in m.trunk if isinstance(x, nn.LayerNorm)]
                for i, (lin, ln) in enumerate(zip(lins, lns)):
                    lin.weight.copy_(st[f"W.{i}"][k].T)
                    lin.bias.copy_(st[f"b.{i}"][k, 0])
                    ln.weight.copy_(st[f"g.{i}"][k, 0])
                    ln.bias.copy_(st[f"beta.{i}"][k, 0])
                m.head.weight.copy_(st["hW"][k].T)
                m.head.bias.copy_(st["hb"][k, 0])
                out.append(m.eval())
        return out


# ----------------------------------------------------------------------------- losses and outputs

def _sf_loss(o, psi):
    Fn = psi.shape[-1] - 1
    bce = F.binary_cross_entropy_with_logits(o[..., :Fn], psi[..., :Fn], reduction="none").mean(dim=(-2, -1))
    t = F.huber_loss(F.softplus(o[..., Fn]) / T_SCALE, psi[..., Fn] / T_SCALE, delta=HUBER_DELTA, reduction="none")
    return bce + t.mean(dim=-1)


def arm_loss(kind: str, out, y: dict, lam: float = LAMBDA_AUX):
    """Per-model mean loss [B] for outputs out [B, M, d_out] and targets y {'v': [B, M], 'psi': [B, M, F + 1]}."""
    if kind in ("uvfa", "pgv"):
        return F.huber_loss(out[..., 0], y["v"], delta=HUBER_DELTA, reduction="none").mean(dim=-1)
    if kind == "sf":
        return _sf_loss(out, y["psi"])
    if kind == "aux":
        v = F.huber_loss(out[..., 0], y["v"], delta=HUBER_DELTA, reduction="none").mean(dim=-1)
        return v + lam * _sf_loss(out[..., 1:], y["psi"])
    raise ValueError(kind)


def outputs(kind: str, out) -> dict:
    """Network outputs -> {'v': value, 'psi': (endpoint-fact probabilities, T-hat)} as float64 numpy."""
    res = {}
    if kind in ("uvfa", "pgv", "aux"):
        res["v"] = out[..., 0].double().numpy()
    if kind in ("sf", "aux"):
        o = out[..., 1:] if kind == "aux" else out
        res["psi"] = torch.cat([torch.sigmoid(o[..., :-1]), F.softplus(o[..., -1:])], dim=-1).double().numpy()
    return res


# ----------------------------------------------------------------------------- data

@dataclass
class Task:
    """Training data of one model: state indices into the feature table, targets {'v', 'psi'} (float32), per-row goal
    ids into the trainer's code table (zero-shot) or a fixed code (FT, SF-FT; None for pgv), augmentation."""
    s: np.ndarray
    y: dict
    s_val: np.ndarray | None = None
    y_val: dict | None = None
    g: np.ndarray | None = None
    g_val: np.ndarray | None = None
    code: np.ndarray | None = None
    aug: str = "none"
    stab: np.ndarray | None = None


def label_rows(space, phi, goals, label_idx, tables=None) -> dict:
    """Exact zero-shot labels for goals [(w, c)]: rows s, g (goal id), psi = (phi(end), T) and v = V* (asserted =
    w-bar.psi), float64; plus W [n_goals, F] and c [n_goals] per goal id."""
    s, g, ps, v = [], [], [], []
    for i, ((w, c), idx) in enumerate(zip(goals, label_idx)):
        tab = SO.sf_table(space, SO.stop_values(phi, w), c) if tables is None else tables[i]
        idx = np.asarray(idx, dtype=np.int64)
        p = SO.psi(phi, tab, idx)
        if not np.array_equal(SO.wbar_dot(p, w, c), tab.V[idx]):
            raise AssertionError("V* != w-bar.psi")
        s.append(idx)
        g.append(np.full(len(idx), i, dtype=np.int64))
        ps.append(p)
        v.append(tab.V[idx])
    return dict(s=np.concatenate(s), g=np.concatenate(g), psi=np.concatenate(ps), v=np.concatenate(v),
                W=np.stack([np.asarray(w, dtype=np.float64) for w, _ in goals]),
                c=np.array([float(c) for _, c in goals]))


def shuffle_psi(rows: dict, goals, seed: int | None = None) -> dict:
    """SF-shuffled targets: psi rows permuted across states within each goal (stream 20261243:shuffle, goals in id
    order); V is recomputed as w-bar.psi of the permuted rows; states and goal ids are unchanged."""
    rng = random.Random(derive(SEEDS["train_labels"], "shuffle") if seed is None else seed)
    out = {k: np.array(v, copy=True) for k, v in rows.items()}
    for i, (w, c) in enumerate(goals):
        pos = np.flatnonzero(rows["g"] == i)
        perm = list(pos)
        rng.shuffle(perm)
        out["psi"][pos] = rows["psi"][perm]
        out["v"][pos] = SO.wbar_dot(np.asarray(rows["psi"][perm], dtype=np.float64), w, c).astype(out["v"].dtype)
    return out


def zero_shot_task(rows: dict, kind: str, aug: str = "rename") -> Task:
    keys = {"sf": ("psi",), "uvfa": ("v",), "aux": ("v", "psi")}[kind]
    return Task(s=np.asarray(rows["s"], dtype=np.int64), y={k: np.asarray(rows[k], dtype=np.float32) for k in keys},
                g=np.asarray(rows["g"], dtype=np.int64), aug=aug)


def rows_task(states, targets: dict, nv: int, code=None, stab=None) -> Task:
    """Per-goal rows in draw order: the first nv are the early-stopping split, the rest are trained on; targets
    {'v'} and/or {'psi'}; stabilizer augmentation if `stab` (ids into perms(n)) is given."""
    s = np.asarray(states, dtype=np.int64)
    if not 0 < nv < len(s):
        raise ValueError(f"validation split {nv} of {len(s)} rows")
    y = {k: np.asarray(v, dtype=np.float32) for k, v in targets.items()}
    return Task(s=s[nv:], y={k: v[nv:] for k, v in y.items()}, s_val=s[:nv], y_val={k: v[:nv] for k, v in y.items()},
                code=None if code is None else np.asarray(code, dtype=np.float32),
                aug="none" if stab is None else "stab", stab=None if stab is None else np.asarray(stab, dtype=np.int64))


def pergoal_task(tab: SO.SFTable, idx, kind: str = "pgv", phi=None, val_frac: float = 0.1, code=None,
                 stab=None) -> Task:
    """Per-goal labels of `tab` at the label indices idx (draw order): the first round(val_frac * k) are the
    early-stopping split; targets V* (and psi for sf/aux, needs phi); stabilizer augmentation if `stab` is given."""
    idx = np.asarray(idx, dtype=np.int64)
    y = {}
    if kind in ("uvfa", "pgv", "aux"):
        y["v"] = tab.V[idx]
    if kind in ("sf", "aux"):
        y["psi"] = SO.psi(phi, tab, idx)
    return rows_task(idx, y, int(round(len(idx) * val_frac)), code, stab)


def n_val(n_rows: int, cfg: dict) -> int:
    """Size of the early-stopping split: cfg val_n, else round(val_frac * k)."""
    return int(cfg["val_n"]) if cfg.get("val_n") is not None else int(round(float(cfg.get("val_frac", 0.1)) * n_rows))


class Sampler:
    """Per-model batch indices: mode 'epoch' (a permutation per epoch, short last batch) or 'stream'."""

    def __init__(self, n_rows: int, batch: int, seed: int, mode: str = "stream"):
        if mode not in ("epoch", "stream"):
            raise ValueError(mode)
        self.n, self.batch, self.mode = n_rows, batch, mode
        self.gen, self.queue = np.random.Generator(np.random.PCG64(seed)), []

    def next(self) -> tuple:
        """(indices, epoch ended?)."""
        if self.mode == "epoch":
            if not self.queue:
                self.queue = self.gen.permutation(self.n).tolist()
            idx, self.queue = self.queue[:self.batch], self.queue[self.batch:]
            return np.asarray(idx, dtype=np.int64), not self.queue
        while len(self.queue) < self.batch:
            self.queue += self.gen.permutation(self.n).tolist()
        idx, self.queue = self.queue[:self.batch], self.queue[self.batch:]
        return np.asarray(idx, dtype=np.int64), False

    def state(self) -> dict:
        return dict(gen=self.gen.bit_generator.state, queue=[int(v) for v in self.queue])

    def load(self, st: dict) -> None:
        self.gen.bit_generator.state = st["gen"]
        self.queue = [int(v) for v in st["queue"]]


# ----------------------------------------------------------------------------- trainer

class Trainer:
    """Resumable AdamW training of B networks of one kind (see module doc)."""

    def __init__(self, kind: str, models, tasks, X, n: int, *, batch: int, lr: float, wd: float, order_seeds,
                 aug_seeds, codes=None, steps: int | None = None, epochs: int | None = None, check_every: int = 0,
                 horizons=(), step0: bool = False, snap_steps=(), threads: int | None = None,
                 fingerprint: dict | None = None, lam: float = LAMBDA_AUX):
        models, tasks = list(models), list(tasks)
        if kind not in KINDS or len(models) != len(tasks) or not tasks or (steps is None) == (epochs is None):
            raise ValueError("bad trainer arguments")
        set_threads(default_threads() if threads is None else threads)
        self.kind, self.tasks, self.X, self.n, self.codes, self.lam = kind, tasks, X, n, codes, lam
        self.B, self.batch = len(tasks), int(batch)
        self.mode = "epoch" if epochs is not None else "stream"
        sizes = {len(t.s) for t in tasks}
        if self.mode == "epoch":
            if len(sizes) != 1:
                raise ValueError("epoch mode needs equal training sizes")
            self.per_epoch = math.ceil(sizes.pop() / self.batch)
            self.total = int(epochs) * self.per_epoch
        else:
            self.per_epoch, self.total = None, int(steps)
        self.check_every, self.horizons, self.step0 = int(check_every), tuple(sorted(int(h) for h in horizons)), step0
        if (self.check_every or step0) and not self.horizons:
            raise ValueError("early stopping needs horizons")
        if self.horizons and len({len(t.s_val) for t in tasks}) != 1:
            raise ValueError("early stopping needs equal validation sizes")
        self.snap_steps = set(int(s) for s in snap_steps)
        self.fingerprint = _canon(dict(fingerprint or {}))
        self.Gs, self.Gf = gathers(n)
        self.net = BatchedMLP.from_models(models)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        self.samplers = [Sampler(len(t.s), self.batch, int(s), self.mode) for t, s in zip(tasks, order_seeds)]
        self.augs = [np.random.Generator(np.random.PCG64(int(s))) for s in aug_seeds]
        self.step, self.val_hist, self.gaps, self.last_loss, self.snaps, self.epoch_end = 0, [], [], None, {}, False
        params = dict(self.net.named_parameters())
        self.es = {h: dict(loss=np.full(self.B, np.inf), step=np.full(self.B, -1, dtype=np.int64),
                           snap={k: p.detach().clone() for k, p in params.items()}) for h in self.horizons}
        if step0:
            self._check(0)

    # -- data
    def rows(self, m: int, idx, val: bool = False) -> tuple:
        """(inputs float32 [len, d_in], targets dict, renaming ids or None) of model m's rows idx (augmented in
        training, never in validation)."""
        t = self.tasks[m]
        s = (t.s_val if val else t.s)[idx]
        y = {k: v[idx] for k, v in (t.y_val if val else t.y).items()}
        r = None
        if t.aug not in ("none", "rename", "stab"):
            raise ValueError(t.aug)
        if not val and t.aug == "rename":
            r = self.augs[m].integers(len(self.Gs), size=len(idx))
        elif not val and t.aug == "stab":
            r = t.stab[self.augs[m].integers(len(t.stab), size=len(idx))]
        x = self.X[s] if r is None else self.X[s[:, None], self.Gs[r]]
        if self.kind != "pgv":
            if t.code is None:
                z = self.codes[(t.g_val if val else t.g)[idx]]
            else:
                z = np.broadcast_to(t.code, (len(idx), len(t.code)))
            if r is not None:
                z = np.concatenate([np.take_along_axis(z[:, :-1], self.Gf[r], axis=1), z[:, -1:]], axis=1)
            x = np.concatenate([x, z], axis=1)
        if r is not None and "psi" in y:
            p = y["psi"]
            y["psi"] = np.concatenate([np.take_along_axis(p[:, :-1], self.Gf[r], axis=1), p[:, -1:]], axis=1)
        return np.ascontiguousarray(x, dtype=np.float32), y, r

    @staticmethod
    def _stack(parts):
        x = torch.from_numpy(np.stack([p[0] for p in parts]))
        y = {k: torch.from_numpy(np.ascontiguousarray(np.stack([p[1][k] for p in parts]), dtype=np.float32))
             for k in parts[0][1]}
        return x, y

    # -- steps and checks
    def _one_step(self) -> bool:
        draws = [smp.next() for smp in self.samplers]
        x, y = self._stack([self.rows(m, d[0]) for m, d in enumerate(draws)])
        self.net.train()
        loss = arm_loss(self.kind, self.net(x), y, self.lam)
        self.opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        self.opt.step()
        self.step += 1
        self.last_loss = loss.detach().double().numpy()
        if self.check_every and self.step % self.check_every == 0:
            self._check(self.step)
        if self.step in self.snap_steps:
            self.snaps[self.step] = {k: p.detach().clone() for k, p in self.net.named_parameters()}
        self.epoch_end = bool(draws[0][1])
        return self.epoch_end

    @torch.no_grad()
    def val_losses(self) -> np.ndarray:
        x, y = self._stack([self.rows(m, np.arange(len(t.s_val)), val=True) for m, t in enumerate(self.tasks)])
        self.net.eval()
        return arm_loss(self.kind, self.net(x), y, self.lam).double().numpy()

    @torch.no_grad()
    def _check(self, t: int) -> None:
        L = self.val_losses()
        self.val_hist.append([int(t), [float(v) for v in L]])
        params = dict(self.net.named_parameters())
        for h, e in self.es.items():
            if t > h:
                continue
            for m in np.flatnonzero(L < e["loss"]):
                e["loss"][m], e["step"][m] = L[m], t
                for k, p in params.items():
                    e["snap"][k][m] = p[m]

    def run(self, safe_point=None, clock=time.monotonic, min_interval: float = CHECK_MIN_S, ckpt=None) -> str:
        """Train to the end ('done') or until safe_point(trainer) returns True at a check ('yielded'; the checkpoint is
        written to `ckpt` first). Checks: after a step that ends an epoch, and after a step once >= min_interval
        seconds have passed since the previous check (0: every step); never after the final step."""
        last = clock()
        while self.step < self.total:
            end = self._one_step()
            if self.step >= self.total:
                break
            now = clock()
            if end or now - last >= min_interval:
                self.gaps.append(now - last)
                last = now
                if safe_point is not None and safe_point(self):
                    if ckpt is not None:
                        self.save(ckpt)
                    return "yielded"
        return "done"

    # -- checkpoints
    def state_dict(self) -> dict:
        meta = dict(format=FORMAT, kind=self.kind, step=self.step, samplers=[s.state() for s in self.samplers],
                    augs=[g.bit_generator.state for g in self.augs], val_hist=self.val_hist,
                    fingerprint=self.fingerprint, es_steps={str(h): e["step"].tolist() for h, e in self.es.items()})
        return dict(meta=json.dumps(meta), net=self.net.state_dict(), opt=self.opt.state_dict(),
                    es={str(h): dict(loss=torch.from_numpy(e["loss"].copy()), snap=e["snap"])
                        for h, e in self.es.items()},
                    snaps={str(t): d for t, d in self.snaps.items()})

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> "Trainer":
        d = torch.load(path, weights_only=True, map_location="cpu")
        meta = json.loads(d["meta"])
        if meta["fingerprint"] != self.fingerprint:
            raise ValueError("checkpoint fingerprint differs (seed, config, label set, code or threads)")
        if meta["kind"] != self.kind or len(meta["samplers"]) != self.B or meta.get("format") != FORMAT:
            raise ValueError("checkpoint does not match this trainer")
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        for s, st in zip(self.samplers, meta["samplers"]):
            s.load(st)
        for g, st in zip(self.augs, meta["augs"]):
            g.bit_generator.state = st
        self.step = int(meta["step"])
        self.es = {h: dict(loss=d["es"][str(h)]["loss"].numpy().copy(),
                           step=np.asarray(meta["es_steps"][str(h)], dtype=np.int64),
                           snap=d["es"][str(h)]["snap"]) for h in self.horizons}
        self.snaps = {int(t): v for t, v in d["snaps"].items()}
        self.val_hist = [list(v) for v in meta["val_hist"]]
        return self

    # -- results
    def current(self) -> list:
        return self.net.to_models()

    def best(self, h: int) -> tuple:
        """(per-model MLPs at their best check <= h, best steps, best validation losses)."""
        e = self.es[int(h)]
        return self.net.to_models(e["snap"]), e["step"].copy(), e["loss"].copy()

    def snapshot(self, t: int) -> list:
        return self.net.to_models(self.snaps[int(t)])

    def unit_flops(self) -> float:
        return step_flops(n_params(self.net) // self.B, self.batch, self.B)


def zero_shot_trainer(kind: str, X, n: int, codes, rows: dict, seed: int, cfg: dict | None = None,
                      threads: int = MAX_THREADS, fingerprint: dict | None = None) -> Trainer:
    """SF / UVFA / UVFA+aux at one replicate seed (cfg over CONFIGS['zero-shot']: width, lr, wd, batch, epochs,
    snap_epochs): epochs, joint renaming augmentation, snapshots at snap_epochs (SF, UVFA and UVFA+aux at the same seed
    share init stream, data order and renamings)."""
    cfg = dict(CONFIGS["zero-shot"], **(cfg or {}))
    task = zero_shot_task(rows, kind)
    per_epoch = math.ceil(len(task.s) / cfg["batch"])
    return Trainer(kind, [new_model(kind, n, seed, cfg["width"])], [task], X, n, codes=codes, batch=cfg["batch"],
                   lr=cfg["lr"], wd=cfg["wd"], epochs=cfg["epochs"], order_seeds=[derive(seed, "order")],
                   aug_seeds=[derive(SEEDS["train_labels"], f"rename:{seed}")],
                   snap_steps=[int(e) * per_epoch for e in cfg["snap_epochs"]], threads=threads,
                   fingerprint=fingerprint)


def pergoal_seeds(arm: str, key: str, k: int | None = None, replicate: int | None = None) -> tuple:
    """(init, order, augmentation) seeds of one per-goal fit (init None for FT, which starts from the UVFA)."""
    base = SEEDS["pergoal"]
    if arm == "pgv":
        tag = f"{key}:{k}"
        return derive(base, f"pgv-init:{tag}"), derive(base, f"pgv-order:{tag}"), derive(base, f"pgv-stab:{tag}")
    if arm == "ft":
        tag = f"{key}:{replicate}"
        return None, derive(base, f"ft-order:{tag}"), derive(base, f"ft-stab:{tag}")
    raise ValueError(arm)


def pergoal_trainer(kind: str, tasks, X, n: int, cfg: dict, *, models=None, init_seeds=None, order_seeds, aug_seeds,
                    width: int | None = None, threads: int = MAX_THREADS, fingerprint: dict | None = None,
                    batch: int | None = None) -> Trainer:
    """PGV (kind 'pgv', fresh models from init_seeds) or FT / SF-FT (kind 'uvfa' / 'sf', `models` = copies of the
    replicate, tasks with fixed codes) as one BatchedMLP; cfg: lr, wd, batch, steps, check_every, horizons, step0."""
    if models is None:
        models = [new_model(kind, n, s, width or cfg["width"]) for s in init_seeds]
    return Trainer(kind, models, tasks, X, n, batch=batch or cfg["batch"], lr=cfg["lr"], wd=cfg.get("wd", WD),
                   steps=cfg["steps"], order_seeds=order_seeds, aug_seeds=aug_seeds,
                   check_every=cfg["check_every"], horizons=cfg["horizons"], step0=cfg.get("step0", False),
                   threads=threads, fingerprint=fingerprint)


# ----------------------------------------------------------------------------- unit sizing (M2)

def step_flops(params: int, batch: int, B: int = 1) -> float:
    """FLOPs of one optimizer step (6 x parameters x samples)."""
    return 6.0 * params * batch * B


def unit_seconds(flops: float, gflops: float = PLAN_GFLOPS) -> float:
    return flops / (gflops * 1e9)


def max_models(params: int, batch: int, gflops: float = PLAN_GFLOPS, limit: float = UNIT_MAX_S, cap: int = B_MAX) -> int:
    """Largest B <= cap whose optimizer step takes <= limit seconds at `gflops` (0 if even B = 1 does not fit)."""
    return max(0, min(cap, int(limit * gflops * 1e9 // step_flops(params, batch))))


def check_units(units: dict, limit: float = UNIT_MAX_S) -> dict:
    """units {name: seconds}; ValueError naming every unit > limit (an uninterruptible unit must be <= 30 s)."""
    bad = {k: v for k, v in units.items() if not v <= limit}
    if bad:
        raise ValueError(f"uninterruptible units > {limit} s: {bad}")
    return units


def check_gap(unit_s: float, min_interval: float = CHECK_MIN_S) -> float:
    """Worst spacing of consecutive safe-point checks: a check is due >= min_interval s after the previous one and runs
    at the end of the unit in progress."""
    return min_interval + unit_s


def plan_units(n: int = 7, gflops: float = PLAN_GFLOPS, B: int = B_MAX) -> dict:
    """Seconds per uninterruptible optimizer step of every planned training configuration (§5.2-5.3 sizes; per-goal
    fits as BatchedMLP batches of B models; the library psi-hat build's unit, one code over the n = 7 states in chunks,
    is counted per chunk)."""
    out = {}
    for kind in ("sf", "uvfa", "aux"):
        out[f"step:{kind}:B1:b{ZS_BATCH}"] = unit_seconds(step_flops(n_params(MLP(*arm_dims(kind, n))), ZS_BATCH), gflops)
    for w, b in ((64, 64), (256, 64), (128, 64), (128, 128), (128, 256), (256, 128), (256, 256)):
        out[f"step:pgv{w}:B{B}:b{b}"] = unit_seconds(step_flops(n_params(MLP(*arm_dims("pgv", n), w)), b, B), gflops)
    for kind in ("uvfa", "sf"):  # FT-UVFA_400, SF-FT_400
        out[f"step:ft-{kind}:B{B}:b64"] = unit_seconds(step_flops(n_params(MLP(*arm_dims(kind, n))), 64, B), gflops)
    out["library:code"] = unit_seconds(2.0 * n_params(MLP(*arm_dims("sf", n))) * 65_990, gflops)
    return out


# ----------------------------------------------------------------------------- inference: Evaluator, library psi-hat

class Evaluator:
    """Cached single-row inference of one network (one state per forward pass: fixed composition)."""

    def __init__(self, model: MLP, kind: str, X, code=None):
        self.model, self.kind, self.X = model.eval(), kind, X
        self.code = None if code is None else np.asarray(code, dtype=np.float32)
        if (kind == "pgv") != (self.code is None):
            raise ValueError("pgv takes no code; every other kind needs one")
        self._c, self.forwards = {}, 0

    @torch.no_grad()
    def out(self, s: int) -> dict:
        r = self._c.get(s)
        if r is None:
            x = self.X[s] if self.code is None else np.concatenate([self.X[s], self.code])
            o = self.model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)[None]))[0]
            r = self._c[s] = outputs(self.kind, o)
            self.forwards += 1
        return r

    def value(self, idx) -> np.ndarray:
        return np.array([self.out(int(s))["v"] for s in np.atleast_1d(idx)], dtype=np.float64)

    def psi(self, idx) -> np.ndarray:
        return np.stack([self.out(int(s))["psi"] for s in np.atleast_1d(idx)])


class LibraryPsi:
    """psi-hat(s, z) for every state and every library code z: float16 [|L|, N, F + 1], in memory or as an .npy memmap
    at `path` with a progress file `<path>.json` (built one code per unit; build() resumes after a yield)."""

    def __init__(self, model: MLP, kind: str, X, codes, path=None, chunk: int = 4096):
        if kind not in ("sf", "aux"):
            raise ValueError("library psi needs an sf or aux network")
        self.model, self.kind, self.X, self.chunk = model.eval(), kind, X, chunk
        self.codes = np.asarray(codes, dtype=np.float32)
        shape = (len(self.codes), len(X), self.codes.shape[1])
        self.path, self.done = (None if path is None else Path(path)), 0
        if self.path is None:
            self.arr = np.zeros(shape, dtype=np.float16)
        else:
            prog = self._prog()
            if self.path.exists() and prog.exists():
                self.done = int(json.loads(prog.read_text(encoding="utf-8"))["done"])
                self.arr = np.lib.format.open_memmap(self.path, mode="r+")
                if self.arr.shape != shape or self.arr.dtype != np.float16:
                    raise ValueError("library cache file does not match")
            else:
                self.arr = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float16, shape=shape)

    def _prog(self) -> Path:
        return self.path.with_name(self.path.name + ".json")

    @torch.no_grad()
    def build(self, safe_point=None, clock=time.monotonic, min_interval: float = CHECK_MIN_S) -> str:
        last = clock()
        for z in range(self.done, len(self.codes)):
            for a in range(0, len(self.X), self.chunk):
                xs = self.X[a:a + self.chunk]
                x = np.concatenate([xs, np.broadcast_to(self.codes[z], (len(xs), self.codes.shape[1]))], axis=1)
                self.arr[z, a:a + len(xs)] = outputs(self.kind, self.model(torch.from_numpy(x)))["psi"]
            self.done = z + 1
            if self.path is not None:
                self.arr.flush()
                tmp = self._prog().with_name(self._prog().name + ".tmp")
                tmp.write_text(json.dumps({"done": self.done, "shape": list(self.arr.shape)}), encoding="utf-8")
                os.replace(tmp, self._prog())
            now = clock()
            if now - last >= min_interval:
                last = now
                if safe_point is not None and self.done < len(self.codes) and safe_point(self):
                    return "yielded"
        return "done"

    def rows(self, idx) -> np.ndarray:
        """float64 [len(idx), |L|, F + 1]."""
        if self.done < len(self.codes):
            raise RuntimeError("library cache not built")
        return np.asarray(self.arr[:, np.atleast_1d(idx)], dtype=np.float64).transpose(1, 0, 2)

    def close(self) -> None:
        """Drop the memmap (Windows keeps the file open while it lives)."""
        if self.path is not None and self.arr is not None:
            self.arr.flush()
        self.arr = None


def nearest_codes(lib_codes, code, k: int = 4) -> np.ndarray:
    """SF-GPI-5: indices of the k library codes with the highest cosine to `code` (ties to the lower index)."""
    L = np.asarray(lib_codes, dtype=np.float64)
    z = np.asarray(code, dtype=np.float64)
    cos = (L @ z) / (np.linalg.norm(L, axis=1) * np.linalg.norm(z))
    return np.argsort(-cos, kind="stable")[:k]


def sf_scores(w, c: float, own: Evaluator | None = None, lib: LibraryPsi | None = None, lib_idx=None):
    """Move scores of an SF arm: max over the codes in C of w-bar'.[(0, 1) + psi-hat(s', z)] (soft.gpi_scores);
    C = the library codes (all, or lib_idx) plus w' itself if `own` is given."""
    if own is None and lib is None:
        raise ValueError("empty code set")

    def fn(s, acts, dsts):
        parts = []
        if lib is not None:
            L = lib.rows(dsts)
            parts.append(L if lib_idx is None else L[:, np.asarray(lib_idx)])
        if own is not None:
            parts.append(own.psi(dsts)[:, None, :])
        return SO.gpi_scores(np.concatenate(parts, axis=1), w, c)

    return fn


def value_scores(ev: Evaluator, c: float):
    """Move scores of a value arm (UVFA, UVFA+aux, PGV, FT): -c' + V-hat(s')."""
    return lambda s, acts, dsts: ev.value(dsts) - c


def sf_values(ev: Evaluator, idx, w, c: float) -> np.ndarray:
    """V-hat = w-bar.psi-hat(s, w) (G-P4's SF value)."""
    return SO.wbar_dot(ev.psi(idx), w, c)


# ----------------------------------------------------------------------------- model files, multi-code inference

class Loaded:
    """A saved network: .model (MLP, eval mode), .kind, .n, .meta; a unique .uid keys the prediction cache."""
    _uids = itertools.count()

    def __init__(self, model: MLP, meta: dict, path=None):
        self.model, self.meta, self.path = model.eval(), dict(meta), None if path is None else str(path)
        self.kind, self.n, self.uid = meta["kind"], int(meta["n"]), next(Loaded._uids)


def save_model(path, model: MLP, meta: dict) -> str:
    path = Path(path)
    d_in, d_out, width, depth = model.dims
    meta = dict(meta, format=FORMAT, d_in=d_in, d_out=d_out, width=width, depth=depth)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(meta=json.dumps(meta, sort_keys=True), state=model.state_dict()), tmp)
    os.replace(tmp, path)
    return str(path)


def _n_of(kind: str, d_in: int) -> int:
    for n in range(1, 16):
        if arm_dims(kind, n)[0] == d_in:
            return n
    raise ValueError(f"no n with input width {d_in} for {kind}")


def load_model(path) -> Loaded:
    """A model file of save_model ({'meta', 'state'}) or of run_e05's Adapters.save ({'kind', 'dims', 'state'})."""
    d = torch.load(path, weights_only=True, map_location="cpu")
    if "meta" in d:
        meta = json.loads(d["meta"])
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: unknown model file format")
    else:
        d_in, d_out, width, depth = (int(v) for v in d["dims"])
        meta = dict(kind=d["kind"], n=_n_of(d["kind"], d_in), d_in=d_in, d_out=d_out, width=width, depth=depth)
    m = MLP(meta["d_in"], meta["d_out"], meta["width"], meta["depth"])
    m.load_state_dict(d["state"])
    set_threads(default_threads())
    return Loaded(m, meta, path)


_PRED: OrderedDict = OrderedDict()
_PRED_BYTES = [0]


def clear_cache() -> None:
    _PRED.clear()
    _PRED_BYTES[0] = 0


def _cached(key, fn) -> np.ndarray:
    r = _PRED.get(key)
    if r is not None:
        _PRED.move_to_end(key)
        return r
    r = _PRED[key] = fn()
    _PRED_BYTES[0] += r.nbytes
    while _PRED_BYTES[0] > CACHE_BYTES and len(_PRED) > 1:
        _PRED_BYTES[0] -= _PRED.popitem(last=False)[1].nbytes
    return r


def _as_loaded(model) -> Loaded:
    return model if isinstance(model, Loaded) else load_model(model)


@torch.no_grad()
def _forward(lm: Loaded, x: np.ndarray, what: str) -> np.ndarray:
    if x.shape[1] != lm.meta["d_in"]:
        raise ValueError("input width does not match the model (wrong n?)")
    out = lm.model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
    return outputs(lm.kind, out)[what].astype(np.float32)


def predict_psi(model, space, states, W, c) -> np.ndarray:
    """psi-hat(s, z) float64 [len(states), m, F + 1] for the m codes z = (W[i]/3, c[i]) (sf or aux networks); one
    forward per state over its m codes (fixed composition), cached per (model, code set, state)."""
    lm = _as_loaded(model)
    if lm.kind not in ("sf", "aux"):
        raise ValueError(f"{lm.kind} network has no psi head")
    X, codes = features(space), goal_codes(W, c)
    key = hashlib.sha1(codes.tobytes()).hexdigest()

    def one(s):
        return _forward(lm, np.concatenate([np.broadcast_to(X[s], (len(codes), X.shape[1])), codes], axis=1), "psi")
    return np.stack([_cached((lm.uid, "psi", key, int(s)), lambda s=int(s): one(s))
                     for s in np.atleast_1d(states)]).astype(np.float64)


def predict_value(model, space, states, W=None, c=None) -> np.ndarray:
    """V-hat(s, z) float64 [len(states), m] (uvfa, aux; FT models take the parent's code); a pgv network ignores the
    code and its value is repeated over the m codes (m = 1 without W)."""
    lm = _as_loaded(model)
    X = features(space)
    if lm.kind == "pgv":
        m = 1 if W is None else len(np.atleast_2d(W))
        v = [_cached((lm.uid, "v", None, int(s)), lambda s=int(s): _forward(lm, X[s][None], "v"))
             for s in np.atleast_1d(states)]
        return np.repeat(np.stack(v).astype(np.float64), m, axis=1)
    if lm.kind not in ("uvfa", "aux"):
        raise ValueError(f"{lm.kind} network has no value head")
    codes = goal_codes(W, c)
    key = hashlib.sha1(codes.tobytes()).hexdigest()

    def one(s):
        return _forward(lm, np.concatenate([np.broadcast_to(X[s], (len(codes), X.shape[1])), codes], axis=1), "v")
    return np.stack([_cached((lm.uid, "v", key, int(s)), lambda s=int(s): one(s))
                     for s in np.atleast_1d(states)]).astype(np.float64)


# ----------------------------------------------------------------------------- higher-level entry points

def _kind_callback(safe_point):
    """Adapt a safe_point(kind) callback to Trainer.run's safe_point(trainer)."""
    if safe_point is None:
        return None
    return lambda tr: safe_point("epoch" if tr.epoch_end else "step")


def train_zero_shot(arm: str, space, data: dict, *, seed: int, epochs: int, lr: float, keep, out_dir,
                    safe_point=None, threads: int | None = None, width: int = WIDTH, batch: int = ZS_BATCH,
                    wd: float = WD, gflops: float = PLAN_GFLOPS):
    """One zero-shot model (SF, UVFA, UVFA+aux or SF-shuffled) at one seed; {epoch: model path} for the kept epochs,
    or None if it yielded at a safe point (call again with the same arguments to resume)."""
    if arm not in ARM_KIND:
        raise ValueError(arm)
    kind, n, out = ARM_KIND[arm], space.n, Path(out_dir)
    keep = sorted({int(e) for e in np.atleast_1d(keep)})
    if not keep or keep[0] < 1 or keep[-1] > int(epochs):
        raise ValueError("kept epochs must lie in 1..epochs")
    threads = default_threads() if threads is None else int(threads)
    S = np.asarray(data["states"] if "states" in data else data["s"], dtype=np.int64)
    gid = np.asarray(data["goal"] if "goal" in data else data["g"], dtype=np.int64)
    W, C = np.atleast_2d(np.asarray(data["W"], dtype=np.float64)), np.atleast_1d(np.asarray(data["c"], np.float64))
    n_goals = int(gid.max()) + 1
    if len(W) == len(S):  # per-row W, c
        first = np.array([np.flatnonzero(gid == i)[0] for i in range(n_goals)])
        if not np.array_equal(goal_codes(W, C), goal_codes(W[first], C[first])[gid]):
            raise ValueError("W or c differ within a goal id")
        W, C = W[first], C[first]
    elif len(W) != n_goals:
        raise ValueError("W, c must be given per row or per goal id")
    codes = goal_codes(W, C)
    rows = dict(s=S, g=gid, psi=np.asarray(data["psi"], dtype=np.float64), v=np.asarray(data["v"], dtype=np.float64))
    if arm == "SF-shuffled":
        rows = shuffle_psi(rows, [(W[i], float(C[i])) for i in range(n_goals)])
    fp = _canon(dict(arm=arm, kind=kind, seed=int(seed), epochs=int(epochs), keep=keep, lr=float(lr), wd=float(wd),
                     batch=int(batch), width=int(width), n=n, threads=threads, code=CODE_SHA,
                     labels=_sha_arrays(S, gid, W, C, np.asarray(data["psi"]), np.asarray(data["v"]))))
    paths = {e: out / f"e{e}.pt" for e in keep}
    done = out / "done.json"
    if done.exists():
        if json.loads(done.read_text(encoding="utf-8"))["fingerprint"] != fp:
            raise ValueError(f"{out}: holds a different zero-shot run")
        if all(p.exists() for p in paths.values()):
            return {e: str(p) for e, p in paths.items()}
    check_units({f"step:{kind}": unit_seconds(step_flops(n_params(MLP(*arm_dims(kind, n), width)), batch), gflops)})
    tr = zero_shot_trainer(kind, features(space), n, codes, rows, seed,
                           dict(width=width, lr=lr, wd=wd, batch=batch, epochs=keep[-1], snap_epochs=keep),
                           threads=threads, fingerprint=fp)
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        tr.load(ckpt)
    if tr.run(_kind_callback(safe_point), min_interval=0.0, ckpt=ckpt) == "yielded":
        return None
    for e in keep:
        save_model(paths[e], tr.snapshot(e * tr.per_epoch)[0],
                   dict(kind=kind, n=n, arm=arm, seed=int(seed), epoch=e, steps=e * tr.per_epoch, lr=float(lr)))
    _write_json(done, dict(fingerprint=fp, models={str(e): p.name for e, p in paths.items()}))
    ckpt.unlink(missing_ok=True)
    return {e: str(p) for e, p in paths.items()}


def _fname(key: str) -> str:
    return hashlib.sha256(str(key).encode()).hexdigest()[:20]


def fit_pergoal(kind: str, space, jobs, cfg: dict, *, out_dir, safe_point=None, threads: int | None = None,
                B: int | None = None, gflops: float = PLAN_GFLOPS):
    """Per-goal fits (PGV from scratch, FT-UVFA / SF-FT from a replicate) as BatchedMLP batches of <= B models;
    {job key: early-stopped model path}, or None if it yielded (call again with the same arguments to resume)."""
    if kind not in PG_KIND:
        raise ValueError(kind)
    n, X, out, jobs = space.n, features(space), Path(out_dir), list(jobs)
    keys = [str(j["key"]) for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate job keys")
    threads = default_threads() if threads is None else int(threads)
    horizon = int(cfg.get("keep", cfg["steps"]))
    steps = min(int(cfg["steps"]), horizon)
    every = int(cfg.get("eval_every", cfg.get("check_every", 0)))
    if every <= 0 or horizon <= 0:
        raise ValueError("per-goal fits need early stopping (eval_every > 0)")
    lr, wd, batch, step0 = float(cfg["lr"]), float(cfg.get("wd", WD)), int(cfg["batch"]), bool(cfg.get("step0", False))
    inits = {}
    if kind == "PGV":
        net_kind, width = "pgv", int(cfg["width"])
    else:
        for j in jobs:
            p = str(j["init"])
            if p not in inits:
                inits[p] = (load_model(p), hashlib.sha256(Path(p).read_bytes()).hexdigest())
        net_kind, width = PG_KIND[kind], {lm.meta["width"] for lm, _ in inits.values()}
        if {lm.kind for lm, _ in inits.values()} != {net_kind} or len(width) != 1:
            raise ValueError(f"{kind} needs {net_kind} replicate models of one width")
        width = width.pop()
    params = n_params(MLP(*arm_dims(net_kind, n), width))
    B = int(B) if B else max_models(params, batch, gflops)
    if B < 1:
        raise ValueError("even one model's optimizer step exceeds the 30-s unit")
    check_units({f"step:{kind}:B{B}:b{batch}": unit_seconds(step_flops(params, batch, B), gflops)})
    targets = {"pgv": ("v",), "uvfa": ("v",), "sf": ("psi",)}[net_kind]
    paths = {k: out / "models" / f"{_fname(k)}.pt" for k in keys}
    for bi in range(0, len(jobs), B):
        chunk, tag = jobs[bi:bi + B], f"batch{bi // B:04d}"
        fp = _canon(dict(kind=kind, net_kind=net_kind, n=n, width=width, B=B, threads=threads, code=CODE_SHA,
                         steps=steps, horizon=horizon, every=every, step0=step0, lr=lr, wd=wd, batch=batch,
                         cfg={k: v for k, v in cfg.items()},
                         jobs=[dict(key=str(j["key"]), seed=int(j["seed"]),
                                    labels=_sha_arrays(*[np.asarray(j[t], dtype=np.float32) for t in targets],
                                                       np.asarray(j["states"], dtype=np.int64)),
                                    stab=None if j.get("stab") is None else _sha_arrays(np.asarray(j["stab"])),
                                    code=None if net_kind == "pgv" else _sha_arrays(goal_code(j["w"], j["c"])),
                                    init=None if kind == "PGV" else inits[str(j["init"])][1]) for j in chunk]))
        marker = out / f"{tag}.done.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8"))["fingerprint"] != fp:
                raise ValueError(f"{out}: holds a different per-goal fit")
            if all(paths[str(j["key"])].exists() for j in chunk):
                continue
        tasks, models = [], []
        for j in chunk:
            tasks.append(rows_task(j["states"], {t: j[t] for t in targets}, n_val(len(j["states"]), cfg),
                                   code=None if net_kind == "pgv" else goal_code(j["w"], j["c"]),
                                   stab=None if j.get("stab") is None else perm_ids(j["stab"], n)))
            models.append(new_model("pgv", n, int(j["seed"]), width) if kind == "PGV"
                          else copy.deepcopy(inits[str(j["init"])][0].model))
        tr = Trainer(net_kind, models, tasks, X, n, batch=batch, lr=lr, wd=wd, steps=steps,
                     order_seeds=[derive(int(j["seed"]), "order") for j in chunk],
                     aug_seeds=[derive(int(j["seed"]), "stab") for j in chunk], check_every=every,
                     horizons=(horizon,), step0=step0, threads=threads, fingerprint=fp)
        ckpt = out / f"{tag}.ckpt"
        if ckpt.exists():
            tr.load(ckpt)
        if tr.run(_kind_callback(safe_point), min_interval=0.0, ckpt=ckpt) == "yielded":
            return None
        best, bsteps, blosses = tr.best(horizon)
        for j, m, st, ls in zip(chunk, best, bsteps, blosses):
            save_model(paths[str(j["key"])], m, dict(kind=net_kind, n=n, arm=kind, key=str(j["key"]),
                                                     best_step=int(st), val_loss=float(ls), steps=steps,
                                                     n_stab=None if j.get("stab") is None else len(j["stab"])))
        _write_json(marker, dict(fingerprint=fp, best_step=[int(v) for v in bsteps],
                                 val_loss=[float(v) for v in blosses], val_hist=tr.val_hist))
        ckpt.unlink(missing_ok=True)
        if bi + B < len(jobs) and safe_point is not None and safe_point("batch"):
            return None
    return {k: str(paths[k]) for k in keys}


def time_step(model: str, width: int | None = None, batch: int = ZS_BATCH, B: int = 1, steps: int = 5, n: int = 7,
              threads: int | None = None) -> float:
    """G-P5: seconds per optimizer step of B stacked `model` networks at `batch` (synthetic rows, stabilizer-style
    augmentation over all renamings, through Trainer._one_step after one warm-up step)."""
    kind, w0 = MODEL_KIND[model]
    width, D, Fn, nrow = int(width or w0), state_dim(n), n_facts(n), 400
    rng = np.random.default_rng(0)
    X = (rng.random((512, D)) < 0.3).astype(np.float32)
    keys = {"sf": ("psi",), "uvfa": ("v",), "aux": ("v", "psi"), "pgv": ("v",)}[kind]
    tasks = []
    for _ in range(B):
        y = {"v": rng.standard_normal(nrow), "psi": np.concatenate([rng.random((nrow, Fn)) < 0.2,
                                                                     rng.integers(0, 20, (nrow, 1))], axis=1)}
        tasks.append(Task(s=rng.integers(0, len(X), nrow), y={k: y[k].astype(np.float32) for k in keys},
                          code=None if kind == "pgv" else rng.random(Fn + 1).astype(np.float32), aug="stab",
                          stab=np.arange(len(perms(n)))))
    tr = Trainer(kind, [new_model(kind, n, m, width) for m in range(B)], tasks, X, n, batch=batch, lr=1e-3, wd=WD,
                 steps=steps + 1, order_seeds=range(B), aug_seeds=range(B), threads=threads)
    tr._one_step()
    t0 = time.perf_counter()
    for _ in range(steps):
        tr._one_step()
    return (time.perf_counter() - t0) / steps
