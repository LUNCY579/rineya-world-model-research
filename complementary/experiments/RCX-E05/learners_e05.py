"""RCX-E05 learned arms: SF, UVFA, UVFA+aux, per-goal PGV and FT in a BatchedMLP, renaming and stabilizer augmentation,
a resumable trainer with safe points and early stopping, the library psi-hat cache, and the M2 unit sizing
(PREREG_E05_DRAFT.md §5.1-5.5, §10.2; DESIGN.md rev 3 §5). CPU only: CUDA_VISIBLE_DEVICES = '' before torch is
imported (if it is not yet) and torch.cuda patched to report no device, as in rcx/bw/models.py; <= 4 torch threads.

Inputs (§5.1). State x_s (n*n + 3n = 70 at n = 7): on-matrix (x[i*n + j] = 1 iff block i is on block j), ontable (n),
holding (n), clear (n). Goal/policy code z = (w / 3, c) (F + 1 = 57). Every network is depth x [Linear, LayerNorm,
ReLU] at width 256 plus a linear head (n = 7 parameters: SF 180,537, UVFA 166,145, UVFA+aux 180,794, PGV 151,553 at
width 256 and 13,313 at width 64; tested). Arm kinds (outputs):
  sf    psi-hat(x_s, z): F endpoint-fact logits (sigmoid) + 1 length unit (softplus T-hat); loss BCE (mean over facts)
        + Huber(T-hat/10, T/10)
  uvfa  V(x_s, z) -> V*; Huber (delta 1)            aux  UVFA+aux: out[0] = V (used at test), out[1:] = the sf head;
  pgv   per-goal V(x_s), no goal input; Huber            loss Huber(V) + 1.0 * sf loss
FT-UVFA_400 = a uvfa network with its code input fixed to the parent's (w'/3, c'); SF-FT_400 = an sf network likewise.

BatchedMLP. B independent networks stacked: per-model weights [B, in, out] through torch.baddbmm, per-model LayerNorm
affine parameters, one AdamW over the stacked tensors (elementwise, so = B independent optimizers; the loss is the SUM
of per-model means, so each model's gradient is its own). A zero-shot arm is a BatchedMLP with B = 1.

Trainer. Data orders: per model a numpy PCG64 stream; mode "epoch" (a fresh permutation per epoch, last batch short;
zero-shot arms) or "stream" (permutations concatenated, every batch full; per-goal arms). Augmentation per example from
a second per-model stream: "rename" (sigma uniform over all n! renamings, applied jointly to x_s, the code's w part and
the psi target's fact part; V and T unchanged; zero-shot arms), "stab" (sigma uniform over Stab(w'); exact and
label-preserving; per-goal arms). Early stopping: per model and per grid horizon h, the weights with the lowest
validation loss among the checks at steps <= h (checks every `check_every` steps, plus step 0 when step0=True: FT's
zero-shot candidate), strict improvement only, so ties go to the earlier step. Plain snapshots at `snap_steps` (the
20-epoch checkpoint of the 40-epoch run). Safe points (M2): after an optimizer step once >= 60 s have passed since the
previous check, and at every epoch end; `safe_point(trainer)` returning True writes the checkpoint (every model's
weights, AdamW state, early-stopping snapshots, data-order and augmentation stream states, queues, step) atomically and
returns "yielded"; a new Trainer built with the same arguments + load() resumes the run exactly (tested <= 1e-6; in
practice bit-identical at fixed threads). load() refuses a checkpoint whose fingerprint differs.

Unit sizing (M2). An uninterruptible unit is one optimizer step of a BatchedMLP batch (6 * params * batch * B FLOP) or
one evaluation cell; at the planning rate (19 GFLOP/s before G-P5) no unit may exceed 30 s, so consecutive checks are
<= 60 + 30 = 90 s apart, and <= 120 s at half the rate. `check_units` rejects any planned unit > 30 s; `max_models`
sizes B.

Inference. `Evaluator`: cached single-row forward passes (fixed composition, so a value never depends on which other
states were evaluated, the order of cells, or a resume). `LibraryPsi`: psi-hat(s, z) for every state and library code,
float16 [|L|, N, F + 1], built one code at a time (in memory or an .npy memmap with a progress file; resumable).
Move scores: SF arms by soft.gpi_scores over the code set C (SF-GPI: L + {w'}; SF-noGPI: {w'}; SF-L: L; SF-GPI-5:
{w'} + the 4 library codes nearest to z' by cosine, ties to the lower index); value arms -c' + V-hat(s').

Seeds (App. A; tags not fixed by the prereg are marked *): zero-shot init = torch.manual_seed(replicate seed), data order
derive(seed, "order")*, renaming stream derive(20261243, "rename:<seed>")*, SF-shuffled 20261243:shuffle; PGV init,
order and stabilizer streams derive(20261249, "pgv-init|pgv-order|pgv-stab:<key>:<k>")*; FT order and stabilizer
derive(20261249, "ft-order|ft-stab:<key>:<replicate>")*.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
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
WIDTH, DEPTH, HUBER_DELTA, T_SCALE, LAMBDA_AUX = 256, 3, 1.0, 10.0, 1.0
MAX_THREADS = 4
UNIT_MAX_S, CHECK_MIN_S, PLAN_GFLOPS = 30.0, 60.0, 19.0
KINDS = ("sf", "uvfa", "aux", "pgv")
CONFIGS = {  # §5.2-5.3 (fixed now); keys marked * in learners' docstring summary are literal readings
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


def derive(seed: int, tag: str) -> int:
    """App. A: int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big") (= families_e05.derive)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def set_threads(threads: int) -> None:
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"torch threads must be 1..{MAX_THREADS}")
    torch.set_num_threads(threads)


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


def goal_code(w, c: float) -> np.ndarray:
    """z = (w / 3, c), float32 [F + 1]."""
    return np.concatenate([np.asarray(w, dtype=np.float64) / 3.0, [float(c)]]).astype(np.float32)


@lru_cache(maxsize=None)
def perms(n: int) -> np.ndarray:
    """All renamings sigma (block i -> sigma[i]) in itertools order (= families_e05.perms)."""
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


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
        self.dims = (d_in, d_out, width, depth)
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
        torch.manual_seed(seed)
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
                for l, (lin, ln) in enumerate(zip(lins, lns)):
                    bm.W[l][k].copy_(lin.weight.T)
                    bm.b[l][k, 0].copy_(lin.bias)
                    bm.g[l][k, 0].copy_(ln.weight)
                    bm.beta[l][k, 0].copy_(ln.bias)
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
                for l, (lin, ln) in enumerate(zip(lins, lns)):
                    lin.weight.copy_(st[f"W.{l}"][k].T)
                    lin.bias.copy_(st[f"b.{l}"][k, 0])
                    ln.weight.copy_(st[f"g.{l}"][k, 0])
                    ln.bias.copy_(st[f"beta.{l}"][k, 0])
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
    """Training data of one model: state indices into the feature table, targets {'v', 'psi'} (float32), per-row
    goal ids into the trainer's code table (zero-shot) or a fixed code (FT, SF-FT; None for pgv), augmentation."""
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
    """Exact zero-shot labels: rows (s, g) with psi = (phi(end), T) and V = w-bar.psi (asserted) for goals [(w, c)]."""
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
    return dict(s=np.concatenate(s), g=np.concatenate(g), psi=np.concatenate(ps).astype(np.float32),
                v=np.concatenate(v).astype(np.float32))


def shuffle_psi(rows: dict, goals, seed: int | None = None) -> dict:
    """SF-shuffled targets: psi rows permuted across states within each goal (stream 20261243:shuffle); V is
    recomputed as w-bar.psi of the permuted rows, states and goal ids are unchanged."""
    rng = random.Random(derive(SEEDS["train_labels"], "shuffle") if seed is None else seed)
    out = {k: np.array(v, copy=True) for k, v in rows.items()}
    for i, (w, c) in enumerate(goals):
        pos = np.flatnonzero(rows["g"] == i)
        perm = list(pos)
        rng.shuffle(perm)
        out["psi"][pos] = rows["psi"][perm]
        out["v"][pos] = SO.wbar_dot(rows["psi"][perm].astype(np.float64), w, c).astype(np.float32)
    return out


def zero_shot_task(rows: dict, kind: str, aug: str = "rename") -> Task:
    keys = {"sf": ("psi",), "uvfa": ("v",), "aux": ("v", "psi")}[kind]
    return Task(s=rows["s"], y={k: rows[k] for k in keys}, g=rows["g"], aug=aug)


def pergoal_task(tab: SO.SFTable, idx, kind: str = "pgv", phi=None, val_frac: float = 0.1, code=None,
                 stab=None) -> Task:
    """Per-goal labels (draw order of `idx`): the first round(val_frac * k) are the early-stopping split, the rest
    are trained on; targets V* (and psi for sf/aux, needs phi); stabilizer augmentation if `stab` is given."""
    idx = np.asarray(idx, dtype=np.int64)
    nv = int(round(len(idx) * val_frac))

    def ys(ii):
        y = {}
        if kind in ("uvfa", "pgv", "aux"):
            y["v"] = tab.V[ii].astype(np.float32)
        if kind in ("sf", "aux"):
            y["psi"] = SO.psi(phi, tab, ii).astype(np.float32)
        return y

    return Task(s=idx[nv:], y=ys(idx[nv:]), s_val=idx[:nv], y_val=ys(idx[:nv]),
                code=None if code is None else np.asarray(code, dtype=np.float32),
                aug="none" if stab is None else "stab", stab=None if stab is None else np.asarray(stab))


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
                 horizons=(), step0: bool = False, snap_steps=(), threads: int = 1, fingerprint: dict | None = None,
                 lam: float = LAMBDA_AUX):
        if kind not in KINDS or len(models) != len(tasks) or (steps is None) == (epochs is None):
            raise ValueError("bad trainer arguments")
        set_threads(threads)
        self.kind, self.tasks, self.X, self.n, self.codes, self.lam = kind, list(tasks), X, n, codes, lam
        self.B, self.batch = len(tasks), batch
        self.mode = "epoch" if epochs is not None else "stream"
        sizes = {len(t.s) for t in tasks}
        if self.mode == "epoch":
            if len(sizes) != 1:
                raise ValueError("epoch mode needs equal training sizes")
            self.total = epochs * math.ceil(sizes.pop() / batch)
        else:
            self.total = int(steps)
        self.check_every, self.horizons, self.step0 = int(check_every), tuple(sorted(horizons)), step0
        if (self.check_every or step0) and not self.horizons:
            raise ValueError("early stopping needs horizons")
        self.snap_steps = set(int(s) for s in snap_steps)
        self.fingerprint = dict(fingerprint or {})
        self.Gs, self.Gf = gathers(n)
        self.net = BatchedMLP.from_models(list(models))
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        self.samplers = [Sampler(len(t.s), batch, s, self.mode) for t, s in zip(tasks, order_seeds)]
        self.augs = [np.random.Generator(np.random.PCG64(s)) for s in aug_seeds]
        self.step, self.val_hist, self.gaps, self.last_loss = 0, [], [], None
        params = dict(self.net.named_parameters())
        self.es = {h: dict(loss=np.full(self.B, np.inf), step=np.full(self.B, -1, dtype=np.int64),
                           snap={k: p.detach().clone() for k, p in params.items()}) for h in self.horizons}
        self.snaps = {}
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

    def _stack(self, parts):
        x = torch.from_numpy(np.stack([p[0] for p in parts]))
        y = {k: torch.from_numpy(np.stack([p[1][k] for p in parts])) for k in parts[0][1]}
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
        return draws[0][1]

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
        """Train to the end ('done') or until safe_point(trainer) asks to yield at a check ('yielded'; the
        checkpoint is written to `ckpt` first). Checks: epoch ends, and steps after >= min_interval seconds."""
        last = clock()
        while self.step < self.total:
            end = self._one_step()
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
        return dict(version=1, kind=self.kind, step=self.step, net=self.net.state_dict(), opt=self.opt.state_dict(),
                    samplers=[s.state() for s in self.samplers], augs=[g.bit_generator.state for g in self.augs],
                    es={str(h): dict(loss=torch.from_numpy(e["loss"].copy()), step=torch.from_numpy(e["step"].copy()),
                                     snap=e["snap"]) for h, e in self.es.items()},
                    snaps={str(t): d for t, d in self.snaps.items()}, val_hist=self.val_hist,
                    fingerprint=self.fingerprint)

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path) -> "Trainer":
        d = torch.load(path, weights_only=True)
        if json.dumps(d["fingerprint"], sort_keys=True) != json.dumps(self.fingerprint, sort_keys=True):
            raise ValueError("checkpoint fingerprint differs (seed, config, label set or code digest)")
        if d["kind"] != self.kind or len(d["samplers"]) != self.B:
            raise ValueError("checkpoint does not match this trainer")
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        for s, st in zip(self.samplers, d["samplers"]):
            s.load(st)
        for g, st in zip(self.augs, d["augs"]):
            g.bit_generator.state = st
        self.step = int(d["step"])
        self.es = {h: dict(loss=d["es"][str(h)]["loss"].numpy().copy(), step=d["es"][str(h)]["step"].numpy().copy(),
                           snap=d["es"][str(h)]["snap"]) for h in self.horizons}
        self.snaps = {int(t): v for t, v in d["snaps"].items()}
        self.val_hist = [list(v) for v in d["val_hist"]]
        return self

    # -- results
    def current(self) -> list:
        return self.net.to_models()

    def best(self, h: int) -> tuple:
        """(per-model MLPs at their best check <= h, best steps, best validation losses)."""
        e = self.es[h]
        return self.net.to_models(e["snap"]), e["step"].copy(), e["loss"].copy()

    def snapshot(self, t: int) -> list:
        return self.net.to_models(self.snaps[t])

    def unit_flops(self) -> float:
        return step_flops(n_params(self.net) // self.B, self.batch, self.B)


def zero_shot_trainer(kind: str, X, n: int, codes, rows: dict, seed: int, cfg: dict | None = None,
                      threads: int = MAX_THREADS, fingerprint: dict | None = None) -> Trainer:
    """SF / UVFA / UVFA+aux at one replicate seed: epochs, joint renaming augmentation, snapshots at snap_epochs
    (SF, UVFA and UVFA+aux at the same seed share init stream, data order and renamings)."""
    cfg = dict(CONFIGS["zero-shot"], **(cfg or {}))
    task = zero_shot_task(rows, kind)
    per_epoch = math.ceil(len(task.s) / cfg["batch"])
    return Trainer(kind, [new_model(kind, n, seed, cfg["width"])], [task], X, n, codes=codes, batch=cfg["batch"],
                   lr=cfg["lr"], wd=cfg["wd"], epochs=cfg["epochs"], order_seeds=[derive(seed, "order")],
                   aug_seeds=[derive(SEEDS["train_labels"], f"rename:{seed}")],
                   snap_steps=[e * per_epoch for e in cfg["snap_epochs"]], threads=threads, fingerprint=fingerprint)


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
    replicate, tasks with fixed codes) as one BatchedMLP; early stopping per cfg (horizons, check_every, step0)."""
    if models is None:
        models = [new_model(kind, n, s, width or cfg["width"]) for s in init_seeds]
    return Trainer(kind, models, tasks, X, n, batch=batch or cfg["batch"], lr=cfg["lr"], wd=cfg["wd"],
                   steps=cfg["steps"], order_seeds=order_seeds, aug_seeds=aug_seeds,
                   check_every=cfg["check_every"], horizons=cfg["horizons"], step0=cfg.get("step0", False),
                   threads=threads, fingerprint=fingerprint)


# ----------------------------------------------------------------------------- unit sizing (M2)

def step_flops(params: int, batch: int, B: int = 1) -> float:
    """FLOPs of one optimizer step (6 x parameters x samples)."""
    return 6.0 * params * batch * B


def unit_seconds(flops: float, gflops: float = PLAN_GFLOPS) -> float:
    return flops / (gflops * 1e9)


def max_models(params: int, batch: int, gflops: float = PLAN_GFLOPS, limit: float = UNIT_MAX_S, cap: int = 64) -> int:
    """Largest B <= cap whose optimizer step takes <= limit seconds at `gflops`."""
    return max(1, min(cap, int(limit * gflops * 1e9 // step_flops(params, batch))))


def check_units(units: dict, limit: float = UNIT_MAX_S) -> dict:
    """units {name: seconds}; ValueError naming every unit > limit (an uninterruptible unit must be <= 30 s)."""
    bad = {k: v for k, v in units.items() if v > limit}
    if bad:
        raise ValueError(f"uninterruptible units > {limit} s: {bad}")
    return units


def check_gap(unit_s: float, min_interval: float = CHECK_MIN_S) -> float:
    """Worst spacing of consecutive safe-point checks: a check is due >= min_interval s after the previous one and
    runs at the end of the unit in progress."""
    return min_interval + unit_s


def plan_units(n: int = 7, gflops: float = PLAN_GFLOPS, B: int = 64, eval_cells: dict | None = None) -> dict:
    """Seconds per uninterruptible unit of every planned training step (§5.2-5.3 sizes, B models per BatchedMLP
    batch) plus injected evaluation-cell costs (measured at G-P5)."""
    out = {}
    for kind in ("sf", "uvfa", "aux"):
        out[f"step:{kind}"] = unit_seconds(step_flops(n_params(MLP(*arm_dims(kind, n))), 256), gflops)
    for name, (w, b) in {"pgv400-w64": (64, 64), "pgv400-w256": (256, 64), "pgv-w128-b64": (128, 64),
                         "pgv-w128-b128": (128, 128), "pgv-w128-b256": (128, 256), "pgv1600": (256, 128),
                         "pgv6400": (256, 256)}.items():
        out[f"step:{name}:B{B}"] = unit_seconds(step_flops(n_params(MLP(*arm_dims("pgv", n), w)), b, B), gflops)
    out[f"step:ft400:B{B}"] = unit_seconds(step_flops(n_params(MLP(*arm_dims("uvfa", n))), 64, B), gflops)
    out.update(eval_cells or {})
    return out


# ----------------------------------------------------------------------------- inference

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


def nearest_codes(lib_codes, code, k: int = 4) -> np.ndarray:
    """Indices of the k library codes with the highest cosine to `code` (ties to the lower index)."""
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
