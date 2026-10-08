"""PREREG-2 sealed confirmation bank (DECISIONS D-24). Built WITHOUT any model call.

  python scripts/build_confirm2_bank.py            # writes data/bank_confirm2.json (create-only)
  python scripts/build_confirm2_bank.py --verify   # re-checks an existing bank, writes nothing

The bank is produced by the FROZEN PREREG-1 builder `rcwmt.bank.build()` (sha b08b0229...), with a new
seed and the same structural settings as data/bank_confirm.json except that no order twins are made
(PREREG-2 uses chains only; twins were a PREREG-1 fidelity secondary):

  150 parents per family (ledger, prerequisite, register), K = 8 precommitted plans, H = 4,
  every parent split = test, prerequisite parents need >= 4 distinct final states,
  seed 20261020 (not used by any earlier RCWM-T1 builder, analysis or bootstrap).

Freshness. No (before, event) payload of any candidate may equal a payload of:
  * the historical transition records (FDS generate.harvest_exclusions, the PREREG-1 source),
  * data/bank_pilot.json, data/bank_confirm.json (incl. order twins), data/bank_s4000.json,
  * data/t2_train.jsonl, t2_eval.jsonl, t2_eval_xframe.jsonl, t2_eval_supp.jsonl;
and no parent may equal a parent of the three earlier banks. The builder enforces payload-level
freshness exactly as PREREG-1 did. Afterwards this script re-verifies it independently from the
written file, and reports operative-content overlaps (the payload minus fields that cannot change
the next state: ledger tick and the `other` of add/scale, register tick and unused fields,
prerequisite name and cost). Content overlaps are descriptive; PREREG-2 registers a sensitivity
analysis that drops parents with any content overlap.

Nothing here imports torch, loads a tokenizer or a model, or reads any model output.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rcwmt import bank as bankmod, env  # noqa: E402

OUT = ROOT / "data" / "bank_confirm2.json"
SEED = 20261020
N_PARENTS = 150
K_CAND = 8
HORIZON = 4
MIN_DISTINCT_P = 4
FROZEN_BANK_PY_SHA = "b08b0229c566f9f5b50c7c0c5738f7fff90b768a827e7e32c9880ff69352592f"
PRIOR_BANKS = ("bank_pilot.json", "bank_confirm.json", "bank_s4000.json")
T2_FILES = ("t2_train.jsonl", "t2_eval.jsonl", "t2_eval_xframe.jsonl", "t2_eval_supp.jsonl")
EXPECTED_SHA = {"bank_pilot.json": "7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a",
                "bank_confirm.json": "8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16",
                "bank_s4000.json": "af1336c261375524a4a6eaf92cc48dc76b1159df093ce2705d279688f22bb4ce",
                "t2_train.jsonl": "bf5a2276", "t2_eval.jsonl": "78b5c022", "t2_eval_xframe.jsonl": "ddc0d798",
                "t2_eval_supp.jsonl": "2c1efbc2"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def payload(before, ev) -> str:
    return env.dumps({"before": before, "event": ev})


def content_key(fam: str, before, ev) -> str:
    """Operative content of a transition: only fields that can change the next state."""
    if fam == "ledger_replay":
        op = {"kind": ev["kind"], "account": ev["account"], "amount": ev["amount"], "status": ev["status"]}
        if ev["kind"] == "transfer":
            op["other"] = ev["other"]
        return env.dumps({"f": "L", "before": before, "op": op})
    if fam == "register_machine":
        k = ev["kind"]
        op = {"kind": k, "register": ev["register"]}
        if k in ("swap", "mix", "conditional"):
            op["other"] = ev["other"]
        if k in ("add", "mix", "conditional"):
            op["value"] = ev["value"]
        if k == "conditional":
            op["threshold"] = ev["threshold"]
        return env.dumps({"f": "R", "before": before, "op": op})
    return env.dumps({"f": "P", "before": before,
                      "op": {f: ev[f] for f in ("pre_on", "pre_off", "delete", "add")}})


def bank_transitions(b: dict):
    """(family, before, event) for every step of every candidate (and order twin) of a bank."""
    for p in b["parents"]:
        fam = p["family"]
        for c in p["candidates"] + ([p["twin"]] if "twin" in p else []):
            st = env.rollout(fam, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                yield fam, st[t], ev


def parent_key(p: dict) -> str:
    return env.dumps({"f": p["family"], "s0": p["s0"], "c": [c["events"] for c in p["candidates"]]})


def load_sources() -> tuple[dict, dict, set, dict]:
    """{source: payload set}, {source: content-key set}, prior parent keys, {source: sha256}."""
    pay, con, shas = {}, {}, {}
    t0 = time.time()
    hist = bankmod.load_exclusions()
    pay["historical"] = set(hist)
    con["historical"] = set()
    for s in hist:                                   # historical payloads: content keys where parseable
        try:
            d = json.loads(s)
            ev = d["event"]
            fam = ("prerequisite_plan" if "pre_on" in ev else
                   "ledger_replay" if "account" in ev else "register_machine" if "register" in ev else None)
            if fam is not None:
                con["historical"].add(content_key(fam, d["before"], ev))
        except (ValueError, KeyError, TypeError):
            pass
    print(f"historical payloads: {len(hist)} ({time.time() - t0:.1f}s)", flush=True)
    prior = set()
    for name in PRIOR_BANKS:
        path = ROOT / "data" / name
        shas[name] = sha256(path)
        assert shas[name] == EXPECTED_SHA[name], f"{name}: sha256 {shas[name]} != expected"
        b = json.loads(path.read_text(encoding="utf-8"))
        pay[name], con[name] = set(), set()
        for fam, before, ev in bank_transitions(b):
            pay[name].add(payload(before, ev))
            con[name].add(content_key(fam, before, ev))
        prior |= {parent_key(p) for p in b["parents"]}
        del b
        print(f"{name}: {len(pay[name])} payloads", flush=True)
    for name in T2_FILES:
        path = ROOT / "data" / name
        shas[name] = sha256(path)
        assert shas[name].startswith(EXPECTED_SHA[name]), f"{name}: sha256 {shas[name][:8]} != {EXPECTED_SHA[name]}"
        pay[name], con[name] = set(), set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pay[name].add(payload(r["before"], r["event"]))
                con[name].add(content_key(r["family"], r["before"], r["event"]))
        print(f"{name}: {len(pay[name])} payloads", flush=True)
    return pay, con, prior, shas


def freshness_report(bank: dict, pay: dict, con: dict, prior: set) -> dict:
    fams = env.FAMILIES
    n_tr = Counter()
    uniq = defaultdict(set)
    pay_hits = {s: Counter() for s in pay}
    con_hits = {s: Counter() for s in con}
    con_parents = {s: defaultdict(set) for s in con}
    any_con_parents = defaultdict(set)
    for p in bank["parents"]:
        fam = p["family"]
        for c in p["candidates"]:
            st = env.rollout(fam, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                pl, ck = payload(st[t], ev), content_key(fam, st[t], ev)
                n_tr[fam] += 1
                uniq[fam].add(pl)
                for s in pay:
                    if pl in pay[s]:
                        pay_hits[s][fam] += 1
                for s in con:
                    if ck in con[s]:
                        con_hits[s][fam] += 1
                        con_parents[s][fam].add(p["parent_id"])
                        any_con_parents[fam].add(p["parent_id"])
    parent_dupes = sum(parent_key(p) in prior for p in bank["parents"])
    rep = {"transitions": dict(n_tr), "unique_payloads": {f: len(uniq[f]) for f in fams},
           "payload_collisions": {s: {f: pay_hits[s][f] for f in fams} for s in pay},
           "content_overlap_transitions": {s: {f: con_hits[s][f] for f in fams} for s in con},
           "content_overlap_parents": {s: {f: len(con_parents[s][f]) for f in fams} for s in con},
           "content_overlap_parents_any_source": {f: len(any_con_parents[f]) for f in fams},
           "content_overlap_parent_ids": {f: sorted(any_con_parents[f]) for f in fams},
           "parents_equal_to_prior_bank_parents": parent_dupes,
           "source_sizes_payloads": {s: len(v) for s, v in pay.items()}}
    return rep


def structure_report(bank: dict) -> dict:
    """Bank properties computed from the oracle only (no model output exists or is read)."""
    out = {}
    for fam in env.FAMILIES:
        ps = [p for p in bank["parents"] if p["family"] == fam]
        copy = {k: [] for k in (1, 2, 4)}
        finals = []
        ops = Counter()
        void_chains = 0
        for p in ps:
            fs = set()
            for c in p["candidates"]:
                st = env.rollout(fam, p["s0"], c["events"])
                for k in (1, 2, 4):
                    copy[k].append(env.dumps(st[k]) == env.dumps(st[0]))
                fs.add(env.dumps(st[-1]))
                for t, ev in enumerate(c["events"]):
                    if fam == "ledger_replay":
                        ops[f"{ev['kind']}/{'posted' if ev['status'] == 'posted' else 'void'}"] += 1
                    elif fam == "register_machine":
                        raw = None
                        if ev["kind"] in ("add", "mix"):
                            raw = st[t][ev["register"]] + (ev["value"] if ev["kind"] == "add" else 2 * st[t][ev["other"]] + ev["value"])
                        ops[ev["kind"] + ("" if raw is None else ("/wrap" if not 0 <= raw < env.MOD else "/nowrap"))] += 1
                    else:
                        ops["applicable" if env.applicable(st[t], ev) else "inapplicable"] += 1
                if fam == "ledger_replay" and any(e["status"] != "posted" for e in c["events"]):
                    void_chains += 1
            finals.append(len(fs))
        out[fam] = {"parents": len(ps), "candidates": sum(len(p["candidates"]) for p in ps),
                    "copy_chain_exact_floor": {f"k{k}": sum(v) / len(v) for k, v in copy.items()},
                    "distinct_finals_per_parent": {"min": min(finals), "mean": sum(finals) / len(finals)},
                    "step_ops": dict(sorted(ops.items()))}
        if fam == "ledger_replay":
            out[fam]["chains_with_any_void_event"] = void_chains
    return out


def _peak_gib():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 3)
    except Exception:  # noqa: BLE001
        return None


def verify(bank: dict) -> list[str]:
    errs = []
    for fam in env.FAMILIES:
        n = sum(p["family"] == fam for p in bank["parents"])
        if n != N_PARENTS:
            errs.append(f"{fam}: {n} parents")
    ids = [p["parent_id"] for p in bank["parents"]]
    if len(set(ids)) != len(ids):
        errs.append("duplicate parent ids")
    for p in bank["parents"]:
        if p["split"] != "test":
            errs.append(f"{p['parent_id']}: split {p['split']}")
        if len(p["candidates"]) != K_CAND or any(len(c["events"]) != HORIZON for c in p["candidates"]):
            errs.append(f"{p['parent_id']}: shape")
        if "twin" in p:
            errs.append(f"{p['parent_id']}: has an order twin")
        if [c["cand_id"] for c in p["candidates"]] != [f"{p['parent_id']}-c{i}" for i in range(K_CAND)]:
            errs.append(f"{p['parent_id']}: cand ids")
        if p["family"] == "prerequisite_plan":
            fs = {env.dumps(env.rollout(p["family"], p["s0"], c["events"])[-1]) for c in p["candidates"]}
            if len(fs) < MIN_DISTINCT_P:
                errs.append(f"{p['parent_id']}: {len(fs)} distinct finals")
        for c in p["candidates"]:
            st = env.rollout(p["family"], p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                env.cross_check(p["family"], st[t], ev)          # FDS oracle, independent implementation
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-check an existing data/bank_confirm2.json; write nothing")
    a = ap.parse_args()
    if not a.verify and OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT} (create-only)")
    if a.verify and not OUT.exists():
        raise SystemExit(f"{OUT} does not exist")
    assert sha256(ROOT / "rcwmt" / "bank.py") == FROZEN_BANK_PY_SHA, "rcwmt/bank.py differs from the PREREG-1 freeze"
    t0 = time.time()
    pay, con, prior, shas = load_sources()
    excl = set().union(*pay.values())
    print(f"exclusion union: {len(excl)} payloads; prior parents: {len(prior)}", flush=True)
    if not a.verify:
        bank = bankmod.build(N_PARENTS, K_CAND, HORIZON, SEED, excl, min_distinct_p=MIN_DISTINCT_P,
                             order_twins=False, split_override="test", exclude_parent_keys=prior)
        bank["purpose"] = ("PREREG-2 sealed confirmation bank (DECISIONS D-24): T2 seed-0 adapter ON vs OFF, chained "
                           "self-rollout decisions; built without any model call")
        bank["builder"] = {"script": "scripts/build_confirm2_bank.py", "script_sha256": sha256(Path(__file__)),
                           "rcwmt_bank_py_sha256": FROZEN_BANK_PY_SHA}
        bank["exclusion_sources"] = {s: {"payloads": len(v), **({"sha256": shas[s]} if s in shas else {})} for s, v in pay.items()}
        bank["exclusion_union_payloads"] = len(excl)
        with OUT.open("x", encoding="utf-8") as f:
            f.write(json.dumps(bank) + "\n")
        print(f"wrote {OUT}", flush=True)
    bank = json.loads(OUT.read_text(encoding="utf-8"))           # every check below reads the written file
    errs = verify(bank)
    fr = freshness_report(bank, pay, con, prior)
    st = structure_report(bank)
    bad = {s: v for s, v in fr["payload_collisions"].items() if any(v.values())}
    if bad or fr["parents_equal_to_prior_bank_parents"]:
        errs.append(f"payload collisions {bad}; parent duplicates {fr['parents_equal_to_prior_bank_parents']}")
    sha = sha256(OUT)
    summary = {"bank": str(OUT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha, "seed": bank["seed"],
               "n_parents_per_family": bank["n_parents_per_family"], "k_candidates": bank["k_candidates"],
               "horizon": bank["horizon"], "min_distinct_p": bank["min_distinct_p"], "order_twins": bank["order_twins"],
               "split_override": bank["split_override"], "parents_rejected_for_collision": bank["parents_rejected_for_collision"],
               "parents_rejected_low_distinct_p": bank["parents_rejected_low_distinct_p"],
               "exclusion_set_size_in_bank": bank["exclusion_set_size"], "exclusion_union_now": len(excl),
               "freshness": {k: v for k, v in fr.items() if k != "content_overlap_parent_ids"},
               "content_overlap_parent_ids": fr["content_overlap_parent_ids"], "structure": st,
               "input_sha256": shas, "verify_errors": errs, "torch_imported": "torch" in sys.modules,
               "seconds": round(time.time() - t0, 1), "peak_rss_gib": _peak_gib()}
    print(json.dumps({k: v for k, v in summary.items() if k != "content_overlap_parent_ids"}, indent=1))
    print(f"sha256 {sha}")
    assert "torch" not in sys.modules, "torch must not be imported"
    if errs:
        raise SystemExit(f"VERIFY FAILED: {errs[:10]}")
    print("verify: OK (payload-fresh against every source; oracle cross-checked; structure as specified)")
    log = ROOT / "docs" / "prereg2" / ("bank_build_log.json" if not a.verify else "bank_verify_log.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {log}")


if __name__ == "__main__":
    main()
