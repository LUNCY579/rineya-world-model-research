"""Value models for RCX-E07/E08 (pure torch, CPU): RelGNN (R-GCN style) and FlatMLP, a seeded trainer, LearnedH.

CPU-only: if torch is not yet imported, CUDA_VISIBLE_DEVICES is set to '' first (never '-1'), and torch.cuda is patched
to report no device (RCWM I-21 recipe). `fit` sets torch threads (default 2).

RelGNN: node encoder -> K rounds of typed message passing over the encode.graph edges plus their inverses (2 * n_rel
relation types):  h <- LayerNorm(h + ReLU(W_self h + sum_r AGG_{u in N_r(v)} W_r h_u)), AGG = sum or mean (per
relation, over in-neighbours). Weights per round (or shared across rounds: share_rounds=True, so K can be changed at
test time). Readout: 'node_sum' (default; an MLP head scores every node and the scores are summed — additive in the
number of blocks, the natural form for a cost-to-go that grows with n), 'sum' or 'mean' (pool, then MLP head).
Size-agnostic: no parameter depends on n. Loss: Huber on raw h*.
FlatMLP: MLP on encode.flat vectors (n_max fixed, 361 inputs for n_max = 12).
"""
from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import asdict, dataclass

if "torch" not in sys.modules:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

from . import bwm_import as BI  # noqa: E402
from . import encode as EN  # noqa: E402


# ----------------------------------------------------------------------------- batching

def collate_graphs(graphs: list[dict], n_rel: int) -> dict:
    """Concatenate encode.graph dicts into one batch; adds inverse edges (type + n_rel)."""
    xs, srcs, dsts, rels, gidx, nb = [], [], [], [], [], []
    off = 0
    for g, gr in enumerate(graphs):
        n_nodes = gr["x"].shape[0]
        xs.append(gr["x"])
        e = gr["edges"]
        srcs += [e[:, 0] + off, e[:, 1] + off]
        dsts += [e[:, 1] + off, e[:, 0] + off]
        rels += [e[:, 2], e[:, 2] + n_rel]
        gidx.append(np.full(n_nodes, g, dtype=np.int64))
        nb.append(gr["n_blocks"])
        off += n_nodes
    return {"x": torch.from_numpy(np.concatenate(xs)), "src": torch.from_numpy(np.concatenate(srcs)),
            "dst": torch.from_numpy(np.concatenate(dsts)), "rel": torch.from_numpy(np.concatenate(rels)),
            "graph": torch.from_numpy(np.concatenate(gidx)), "n_graphs": len(graphs),
            "n_blocks": torch.tensor(nb, dtype=torch.float32)}


def collate_flat(vecs: list) -> torch.Tensor:
    return torch.from_numpy(np.stack(vecs).astype(np.float32))


# ----------------------------------------------------------------------------- models

@dataclass
class RelGNNConfig:
    node_in: int = len(EN.NODE_FEATS)
    n_rel: int = len(EN.EDGE_TYPES)
    hidden: int = 48
    rounds: int = 4
    agg: str = "sum"            # sum | mean
    readout: str = "node_sum"   # node_sum | sum | mean
    head_hidden: int = 64
    share_rounds: bool = False


class RelGNN(nn.Module):
    kind = "gnn"

    def __init__(self, cfg: RelGNNConfig = RelGNNConfig()):
        super().__init__()
        if cfg.agg not in ("sum", "mean") or cfg.readout not in ("node_sum", "sum", "mean"):
            raise ValueError("bad agg/readout")
        self.cfg = cfg
        H, R = cfg.hidden, 2 * cfg.n_rel
        L = 1 if cfg.share_rounds else cfg.rounds
        self.inp = nn.Linear(cfg.node_in, H)
        self.w_rel = nn.ParameterList([nn.Parameter(torch.randn(R, H, H) / H ** 0.5) for _ in range(L)])
        self.w_self = nn.ModuleList([nn.Linear(H, H) for _ in range(L)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.head = nn.Sequential(nn.Linear(H, cfg.head_hidden), nn.ReLU(), nn.Linear(cfg.head_hidden, 1))

    def forward(self, b: dict, rounds: int | None = None) -> torch.Tensor:
        cfg = self.cfg
        R = 2 * cfg.n_rel
        h = F.relu(self.inp(b["x"]))
        N = h.shape[0]
        src, dst, rel = b["src"], b["dst"], b["rel"]
        if cfg.agg == "mean":
            key = dst * R + rel
            deg = torch.zeros(N * R).index_add_(0, key, torch.ones_like(key, dtype=torch.float32))
            inv_deg = (1.0 / deg[key]).unsqueeze(1)
        for k in range(rounds or cfg.rounds):
            j = 0 if cfg.share_rounds else k
            hw = torch.einsum("nh,rhk->rnk", h, self.w_rel[j])
            msg = hw[rel, src]
            if cfg.agg == "mean":
                msg = msg * inv_deg
            m = torch.zeros_like(h).index_add_(0, dst, msg)
            h = self.norm[j](h + F.relu(self.w_self[j](h) + m))
        G = b["n_graphs"]
        if cfg.readout == "node_sum":
            return torch.zeros(G).index_add_(0, b["graph"], self.head(h).squeeze(1))
        pooled = torch.zeros(G, h.shape[1]).index_add_(0, b["graph"], h)
        if cfg.readout == "mean":
            cnt = torch.zeros(G).index_add_(0, b["graph"], torch.ones(N))
            pooled = pooled / cnt.unsqueeze(1)
        return self.head(pooled).squeeze(1)

    def collate(self, graphs):
        return collate_graphs(graphs, self.cfg.n_rel)


@dataclass
class FlatMLPConfig:
    n_max: int = EN.N_MAX
    hidden: tuple = (256, 256)


class FlatMLP(nn.Module):
    kind = "flat"

    def __init__(self, cfg: FlatMLPConfig = FlatMLPConfig()):
        super().__init__()
        self.cfg = cfg
        dims = [EN.flat_dim(cfg.n_max), *cfg.hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def collate(self, vecs):
        return collate_flat(vecs)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def encode_for(model, s, goal, objs):
    if model.kind == "gnn":
        return EN.graph(s, goal, objs, derived=model.cfg.n_rel > len(EN.EDGE_TYPES))
    return EN.flat(s, goal, objs, n_max=model.cfg.n_max)


# ----------------------------------------------------------------------------- training

def fit(model, examples: list, targets, *, steps: int | None = None, time_budget_s: float | None = None,
        batch_size: int = 64, lr: float = 2e-3, weight_decay: float = 0.0, huber_delta: float = 1.0,
        seed: int = 20261276, threads: int = 2, log_every: int = 0, log=print) -> list:
    """Seeded minibatch AdamW on Huber loss. Stops at `steps` or `time_budget_s` (whichever first; at least one must
    be given; a time budget makes the step count machine-dependent). Returns [(step, batch loss)]."""
    if steps is None and time_budget_s is None:
        raise ValueError("give steps and/or time_budget_s")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    y_all = torch.tensor(np.asarray(targets, dtype=np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    hist, step, t0, order = [], 0, time.perf_counter(), []
    while True:
        if steps is not None and step >= steps:
            break
        if time_budget_s is not None and time.perf_counter() - t0 >= time_budget_s:
            break
        if len(order) < batch_size:
            perm = list(range(len(examples)))
            rng.shuffle(perm)
            order += perm
        idx, order = order[:batch_size], order[batch_size:]
        pred = model(model.collate([examples[i] for i in idx]))
        loss = F.huber_loss(pred, y_all[idx], delta=huber_delta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        hist.append((step, float(loss)))
        if log_every and step % log_every == 0:
            log(f"step {step} loss {float(loss):.4f} ({time.perf_counter() - t0:.1f}s)")
    model.eval()
    return hist


@torch.no_grad()
def predict(model, examples: list, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(model.collate(examples[i:i + batch_size])).numpy() for i in range(0, len(examples), batch_size)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class LearnedH:
    """Batched learned heuristic for search: h.batch(states, goal) -> list[float]; objs = B.blocks_of(state)."""

    def __init__(self, model):
        self.model = model
        self.B, _ = BI.load()

    def batch(self, states, goal) -> list[float]:
        if not states:
            return []
        ex = [encode_for(self.model, s, goal, self.B.blocks_of(s)) for s in states]
        return [float(v) for v in predict(self.model, ex)]

    def __call__(self, s, goal) -> float:
        return self.batch([s], goal)[0]


def config_dict(model) -> dict:
    return {"kind": model.kind, **asdict(model.cfg), "n_params": n_params(model)}
