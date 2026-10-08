"""Load a captured arm and assemble per-family design matrices and targets.

Consequence feature vector φ (decision features) per family:
  P: final bits (8, binary). (No cost term: a generated final state cannot reveal which
     actions applied, so a cost feature would bias latent-vs-generative comparisons.)
  L: sign_neg (4, bin) + changed (4, bin) + clip(final/100, -3, 3) (4, cont)
  R: hi>48 (3, bin) + parity (3, bin) + changed (3, bin)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import env

ROOT = Path(__file__).resolve().parents[1]

PHI_SPEC = {
    "prerequisite_plan": {"bin": 8, "cont": 0},
    "ledger_replay": {"bin": 8, "cont": 4},
    "register_machine": {"bin": 9, "cont": 0},
}


def phi_from_features(fam: str, f: dict) -> np.ndarray:
    if fam == "prerequisite_plan":
        return np.array(f["final"], dtype=np.float32)
    if fam == "ledger_replay":
        return np.array(f["sign_neg"] + f["changed"] + [float(np.clip(v / 100.0, -3, 3)) for v in f["final"]],
                        dtype=np.float32)
    if fam == "register_machine":
        return np.array(f["hi"] + f["parity"] + f["changed"], dtype=np.float32)
    raise ValueError(fam)


def phi_from_state(fam: str, s0, events: list[dict], state_k) -> np.ndarray | None:
    """φ for a PREDICTED final state (generative arms); same definition as phi_from_features."""
    if state_k is None:
        return None
    if fam == "prerequisite_plan":
        return np.array(env.bits(state_k), dtype=np.float32)
    if fam == "ledger_replay":
        fin = [state_k[a] for a in env.ACCOUNTS]
        return np.array([int(v < 0) for v in fin] + [int(state_k[a] != s0[a]) for a in env.ACCOUNTS]
                        + [float(np.clip(v / 100.0, -3, 3)) for v in fin], dtype=np.float32)
    if fam == "register_machine":
        fin = [state_k[r] for r in env.REGISTERS]
        return np.array([int(v > 48) for v in fin] + [v % 2 for v in fin]
                        + [int(state_k[r] != s0[r]) for r in env.REGISTERS], dtype=np.float32)
    raise ValueError(fam)


class Arm:
    """Memory-mapped view of one captured arm."""

    def __init__(self, bank_stem: str, arm: str):
        self.dir = ROOT / "runs" / bank_stem / arm
        idx = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.layers = idx["layers"]
        self.items = idx["items"]
        self.event_rows = idx["event_rows"]
        self.last = np.load(self.dir / "last.npy", mmap_mode="r")
        self.events = np.load(self.dir / "events.npy", mmap_mode="r")
        self.pos = {it["item_id"]: i for i, it in enumerate(self.items)}
        self.ev_pos = {self.items[i]["item_id"]: r for r, i in enumerate(self.event_rows)}

    def layer_index(self, layer: int) -> int:
        return self.layers.index(layer)

    def X_last(self, item_ids: list[str], layer: int) -> np.ndarray:
        li = self.layer_index(layer)
        rows = np.array([self.pos[i] for i in item_ids])
        order = np.argsort(rows)
        out = np.empty((len(rows), self.last.shape[-1]), dtype=np.float32)
        out[order] = self.last[rows[order], li, :].astype(np.float32)
        return out

    def X_event(self, item_ids: list[str], j: int, layer: int) -> np.ndarray:
        """State at event boundary j (1-based) from the k=H item of each candidate."""
        li = self.layer_index(layer)
        rows = np.array([self.ev_pos[i] for i in item_ids])
        order = np.argsort(rows)
        out = np.empty((len(rows), self.events.shape[-1]), dtype=np.float32)
        out[order] = self.events[rows[order], j - 1, li, :].astype(np.float32)
        return out


def load_bank(bank_stem: str) -> dict:
    return json.loads((ROOT / "data" / f"{bank_stem}.json").read_text(encoding="utf-8"))


def candidate_table(bank: dict) -> list[dict]:
    """One row per (candidate, k) with gold φ, gold prefix states and metadata."""
    rows = []
    for p in bank["parents"]:
        fam = p["family"]
        for c in p["candidates"]:
            st = env.rollout(fam, p["s0"], c["events"])
            for k in (1, 2, 4):
                f = env.features(fam, st[: k + 1], c["events"][:k])
                rows.append({"parent_id": p["parent_id"], "cand_id": c["cand_id"], "family": fam,
                             "split": p["split"], "k": k, "item_id": f"{c['cand_id']}-k{k}",
                             "phi": phi_from_features(fam, f), "features": f,
                             "gold": env.dumps(st[k]), "s0": p["s0"], "events": c["events"][:k],
                             "states": st[: k + 1]})
    return rows


class ArmUnion:
    """One encoder arm captured on several banks (e.g. pilot train/dev + sealed confirm test).
    Item ids are globally unique (parent ids are content hashes), so lookups dispatch by id."""

    def __init__(self, parts: list[Arm]):
        self.parts = parts
        self.layers = parts[0].layers
        assert all(p.layers == self.layers for p in parts)
        self.items = [it for p in parts for it in p.items]
        self._owner = {}
        for p in parts:
            for it in p.items:
                assert it["item_id"] not in self._owner, it["item_id"]
                self._owner[it["item_id"]] = p
        self.dir = parts[0].dir

    def layer_index(self, layer: int) -> int:
        return self.layers.index(layer)

    def _split_call(self, item_ids, fn):
        out = None
        groups = {}
        for n, i in enumerate(item_ids):
            groups.setdefault(id(self._owner[i]), (self._owner[i], []))[1].append(n)
        for part, idx in groups.values():
            X = fn(part, [item_ids[n] for n in idx])
            if out is None:
                out = np.empty((len(item_ids), X.shape[1]), dtype=X.dtype)
            out[idx] = X
        return out

    def X_last(self, item_ids, layer):
        return self._split_call(item_ids, lambda p, ids: p.X_last(ids, layer))

    def X_event(self, item_ids, j, layer):
        return self._split_call(item_ids, lambda p, ids: p.X_event(ids, j, layer))


def candidate_table_multi(bank_stems: list[str]) -> list[dict]:
    rows = []
    for stem in bank_stems:
        rows += candidate_table(load_bank(stem))
    return rows
