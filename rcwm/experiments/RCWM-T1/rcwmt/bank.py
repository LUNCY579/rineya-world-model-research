"""Build the RCWM-T1 bank: parents (start state) x K precommitted candidate plans of horizon H.

Every candidate's gold trajectory comes from env.step, cross-checked transition by transition
against the FDS oracles. Every one-step sub-transition payload is checked against the exclusion
set harvested from all historical transition records (generate.harvest_exclusions), so no
transition Titan was trained or scored on reappears. Splits are assigned per parent by hash before
any model sees anything; all candidates of a parent share its split.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

from . import env

SPLIT_SALT = "RCWM-T1-parent-split-v1"


def split_for(parent_id: str) -> str:
    h = int(hashlib.sha256(f"{SPLIT_SALT}|{parent_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("dev" if h < 85 else "test")


def load_exclusions() -> set[str]:
    sys.path.insert(0, str(env.FDS))
    from generate import harvest_exclusions  # type: ignore
    return harvest_exclusions()


def _order_twin(fam: str, s0, events: list[dict], rng: random.Random, tries: int = 24):
    """Same start state and event multiset, different order, different final state (or None)."""
    final = env.rollout(fam, s0, events)[-1]
    for _ in range(tries):
        perm = list(range(len(events)))
        rng.shuffle(perm)
        if perm == sorted(perm):
            continue
        evs = [dict(events[i]) for i in perm]
        if fam != "prerequisite_plan":
            for t, e in enumerate(evs):
                e["tick"] = t + 1
        if env.rollout(fam, s0, evs)[-1] != final:
            return evs, perm
    return None


def build(n_parents: int, k_cand: int, horizon: int, seed: int, exclusions: set[str] | None,
          n_actions: int = 6, min_distinct_p: int = 0, order_twins: bool = False,
          split_override: str | None = None, exclude_parent_keys: set[str] | None = None) -> dict:
    """min_distinct_p: reject prerequisite parents whose K candidates reach fewer distinct final
    states (amendment S2). order_twins: per parent, one extra candidate that permutes candidate 0's
    events and changes its final state; twins are fidelity-only, never part of a decision set."""
    m = env.load_marginals()
    rng = random.Random(seed)
    parents, collisions, seen = [], 0, set()
    rejected_low_distinct = 0
    for fam in env.FAMILIES:
        made = 0
        while made < n_parents:
            s0 = env.gen_start(fam, rng, m)
            lib = env.gen_action_library(rng, m, n_actions) if fam == "prerequisite_plan" else None
            cands, bad = [], False
            for c in range(k_cand):
                if lib is not None:
                    events = [dict(rng.choice(lib)) for _ in range(horizon)]
                else:
                    events = [env.gen_event(fam, rng, m, tick=t + 1) for t in range(horizon)]
                states = env.rollout(fam, s0, events)
                for t, ev in enumerate(events):
                    env.cross_check(fam, states[t], ev)
                    if exclusions is not None and env.dumps({"before": states[t], "event": ev}) in exclusions:
                        bad = True
                cands.append({"events": events})
            if bad:
                collisions += 1
                continue
            if fam == "prerequisite_plan" and min_distinct_p:
                finals = {env.rollout(fam, s0, c["events"])[-1] for c in cands}
                if len(finals) < min_distinct_p:
                    rejected_low_distinct += 1
                    continue
            twin = None
            if order_twins:
                tw = _order_twin(fam, s0, cands[0]["events"], rng)
                if tw is None:
                    continue
                twin = {"source": 0, "events": tw[0], "perm": tw[1]}
                states = env.rollout(fam, s0, twin["events"])
                for t, ev in enumerate(twin["events"]):
                    env.cross_check(fam, states[t], ev)
                    if exclusions is not None and env.dumps({"before": states[t], "event": ev}) in exclusions:
                        bad = True
                if bad:
                    collisions += 1
                    continue
            key = env.dumps({"f": fam, "s0": s0, "c": [c["events"] for c in cands]})
            if key in seen or (exclude_parent_keys and key in exclude_parent_keys):
                continue
            seen.add(key)
            pid = f"T1{env.SHORT[fam]}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            for ci, c in enumerate(cands):
                c["cand_id"] = f"{pid}-c{ci}"
            p = {"parent_id": pid, "family": fam, "split": split_override or split_for(pid), "s0": s0,
                 "library": lib, "candidates": cands, "parent_key_sha256": hashlib.sha256(key.encode()).hexdigest()}
            if twin is not None:
                twin["cand_id"] = f"{pid}-t0"
                p["twin"] = twin
            parents.append(p)
            made += 1
    return {"schema": 1, "bank": "RCWM-T1", "seed": seed, "n_parents_per_family": n_parents,
            "min_distinct_p": min_distinct_p, "order_twins": order_twins, "split_override": split_override,
            "parents_rejected_low_distinct_p": rejected_low_distinct,
            "k_candidates": k_cand, "horizon": horizon, "n_actions": n_actions,
            "recorded_marginals_sha256": env.RECORDED_SHA,
            "exclusion_set_size": None if exclusions is None else len(exclusions),
            "parents_rejected_for_collision": collisions, "parents": parents}


def items(bank: dict, ks=(1, 2, 4)) -> list[dict]:
    """Flatten to prompt items: one per (candidate, k) in the multi-step format, plus the
    verbatim one-step format for the first event (Titan's trained frame) as k=0 control rows."""
    out = []
    for p in bank["parents"]:
        fam = p["family"]
        for c in p["candidates"]:
            evs = c["events"]
            states = env.rollout(fam, p["s0"], evs)
            for k in ks:
                sub = evs[:k]
                st = states[: k + 1]
                msgs = env.multi_step_messages(fam, p["s0"], sub)
                out.append({"item_id": f"{c['cand_id']}-k{k}", "parent_id": p["parent_id"],
                            "cand_id": c["cand_id"], "family": fam, "split": p["split"], "k": k,
                            "format": "multi", "messages": msgs,
                            "gold": env.dumps(st[-1]), "gold_prefix_states": [env.dumps(s) for s in st],
                            "features": env.features(fam, st, sub)})
            msgs1 = env.one_step_messages(fam, p["s0"], evs[0])
            out.append({"item_id": f"{c['cand_id']}-one", "parent_id": p["parent_id"],
                        "cand_id": c["cand_id"], "family": fam, "split": p["split"], "k": 1,
                        "format": "one", "messages": msgs1, "gold": env.dumps(states[1]),
                        "gold_prefix_states": [env.dumps(states[0]), env.dumps(states[1])],
                        "features": env.features(fam, states[:2], evs[:1])})
        if "twin" in p:
            tw = p["twin"]
            H = len(tw["events"])
            st = env.rollout(fam, p["s0"], tw["events"])
            out.append({"item_id": f"{tw['cand_id']}-k{H}", "parent_id": p["parent_id"], "cand_id": tw["cand_id"],
                        "family": fam, "split": p["split"], "k": H, "format": "multi", "twin": True,
                        "messages": env.multi_step_messages(fam, p["s0"], tw["events"]),
                        "gold": env.dumps(st[-1]), "gold_prefix_states": [env.dumps(x) for x in st],
                        "features": env.features(fam, st, tw["events"])})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", type=int, default=800)
    ap.add_argument("--k-cand", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-exclusions", action="store_true")
    ap.add_argument("--min-distinct-p", type=int, default=0)
    ap.add_argument("--order-twins", action="store_true")
    ap.add_argument("--split-override", default=None)
    ap.add_argument("--exclude-bank", type=Path, default=None, help="dedupe parents against this bank")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    t0 = time.time()
    excl = None if a.no_exclusions else load_exclusions()
    print(f"exclusions: {None if excl is None else len(excl)} ({time.time() - t0:.1f}s)", flush=True)
    prior = None
    if a.exclude_bank:
        pb = json.loads(a.exclude_bank.read_text(encoding="utf-8"))
        prior = {env.dumps({"f": p["family"], "s0": p["s0"], "c": [c["events"] for c in p["candidates"]]}) for p in pb["parents"]}
        if excl is not None:            # no transition of a prior bank may reappear either
            for p in pb["parents"]:
                for c in p["candidates"]:
                    st = env.rollout(p["family"], p["s0"], c["events"])
                    for t, ev in enumerate(c["events"]):
                        excl.add(env.dumps({"before": st[t], "event": ev}))
    bank = build(a.parents, a.k_cand, a.horizon, a.seed, excl, min_distinct_p=a.min_distinct_p,
                 order_twins=a.order_twins, split_override=a.split_override, exclude_parent_keys=prior)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(bank) + "\n", encoding="utf-8")
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    from collections import Counter
    sc = Counter((p["family"], p["split"]) for p in bank["parents"])
    print(f"wrote {a.out} sha256={sha}")
    print(f"collision-rejected parents: {bank['parents_rejected_for_collision']}")
    for k, v in sorted(sc.items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
