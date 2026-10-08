"""PREREG-L1 amendment A-OOD (DECISIONS D-28(a)): the out-of-distribution / rule-extrapolation bank behind C6.

Built WITHOUT any model call, tokenizer or torch import. Create-only. Seed fixed below.

  python -B scripts/l1_make_ood.py            # writes docs/l1/ood_bank.jsonl + docs/l1/ood_bank.SOURCE.json
  python -B scripts/l1_make_ood.py --verify   # re-checks the written bank from the file (incl. a fresh harvest of
                                              # the historical set); writes nothing

Rows: one-step transitions in Titan's trained one-step frame (rcwmt.env.one_step_messages(fam, before, event)), gold by
rcwmt.env.step and checked against the validated FDS oracle on every row (rcwmt.env.cross_check). Six strata, 100 each:

  L_big       ledger, at least one balance |b| in [100, 2000] (each account big with prob 1/2, at least one; a big
              balance is sign x uniform{100..2000}, the others from gen_start's +-99 recorded pool, i.e. outside the
              range every RCWM bank used); event from env.gen_event's recorded marginals, status as drawn; rows whose
              event leaves the state unchanged are rejected (so every void event and add 0 / scale 1 / transfer 0)
  L_void      ledger, start from env.gen_start, event from env.gen_event with status 'void', and applying the same
              event as posted WOULD change the state (gold: unchanged)
  R_wrap      register add (50) or mix (50) whose raw sum (r + v, or r + 2 o + v) lies outside [0, 96]: wrap or underflow
  R_boundary  register conditional whose `other` value is threshold-1, threshold or threshold+1 (offset drawn uniformly
              over those that stay in [0, 96], `other` set to threshold + offset), value != 0 mod 97 (so the branch
              decides the answer)
  P_inapp     prerequisite action (env.gen_action_library) that is inapplicable (a pre_on bit unset or a pre_off bit
              set) where applying it anyway WOULD change the state (gold: unchanged, the FDS oracle convention)
  P_appdel    applicable prerequisite action whose delete clears a set bit that add does not restore

Freshness. No row's (before, event) payload, and no row's operative content (FDS generate.operative_key: the fields that
determine the answer; tick / name / cost dropped), equals one of
  * the historical set: every transition payload in the records tree, harvested exactly as rcwmt.bank.load_exclusions()
    (FDS generate.harvest_exclusions) does, EXCEPT that files named HSKILL_CONFIRM_V2* (the operator-reserved
    HSKILL-CONFIRM-002, UNSPENT_AND_UNREAD) are skipped by name: never opened, never hashed;
  * every transition (candidates and order twins) of data/bank_pilot.json, bank_confirm.json, bank_confirm2.json,
    bank_s4000.json;
  * every row of data/t2_train.jsonl, t2_eval.jsonl, t2_eval_xframe.jsonl, t2_eval_supp.jsonl;
  * SELECT-VIEWS-001 (records tree, read-only; its user turn is exactly {before, event}).
Payloads and operative contents are also unique within the bank.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # nothing here touches the driver; empty string, never "-1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "l1"
OUT = DOCS / "ood_bank.jsonl"
SOURCE = DOCS / "ood_bank.SOURCE.json"
sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402

RECORDS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
SELECT_VIEWS = RECORDS / "select-views-001" / "SELECT_VIEWS.jsonl"
RESERVED_PREFIX = "HSKILL_CONFIRM_V2"              # HSKILL-CONFIRM-002 and its own records: never opened or hashed

# ------------------------------------------------------------------------------------------------ fixed design
SEED = 20261028                                    # not used by any earlier RCWM-T1 builder, analysis or bootstrap
N_PER_STRATUM = 100
STRATA = ("L_big", "L_void", "R_wrap", "R_boundary", "P_inapp", "P_appdel")
FAMILY = {"L_big": "ledger_replay", "L_void": "ledger_replay", "R_wrap": "register_machine",
          "R_boundary": "register_machine", "P_inapp": "prerequisite_plan", "P_appdel": "prerequisite_plan"}
BIG_LO, BIG_HI = 100, 2000
R_WRAP_QUOTA = {"add": 50, "mix": 50}
MAX_DRAWS = 5_000_000

EXPECTED_SHA = {   # the exclusion sources (data/ is read-only here)
    "bank_pilot.json": "7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a",
    "bank_confirm.json": "8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16",
    "bank_confirm2.json": "fafb13e573a20eb5045ef204a7cdbf8da4393d21604245e9df26995f5be58aea",
    "bank_s4000.json": "af1336c261375524a4a6eaf92cc48dc76b1159df093ce2705d279688f22bb4ce",
    "t2_train.jsonl": "bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744",
    "t2_eval.jsonl": "78b5c02255e4ee7c53d4b91d66c0ac2cae4427e927f47832e297e5ffde54eb59",
    "t2_eval_xframe.jsonl": "ddc0d7982ba163f78475af9f5f803dc76ba167ea56d6cc7999fd220d172302de",
    "t2_eval_supp.jsonl": "2c1efbc207a4c56f2ed135a4987dc4aa118353b35292e243c74b282459ff9086",
    "SELECT_VIEWS.jsonl": "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3",
}
BANKS = ("bank_pilot.json", "bank_confirm.json", "bank_confirm2.json", "bank_s4000.json")
T2_FILES = ("t2_train.jsonl", "t2_eval.jsonl", "t2_eval_xframe.jsonl", "t2_eval_supp.jsonl")
HISTORICAL_RECORDED = 6464        # load_exclusions() size recorded by bank_confirm2 / the T2 builder (2026-10-08)
CODE_SHA = {"rcwmt/env.py": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2",
            "fds/oracles.py": "6d22a5e40a69fb8f5c79230d2c2dc7fa51af605de70d3c0fc7cd7d69fe4c959c",
            "fds/generate.py": "e7e0288af3b5b5c3930446295094734abc55ed22761a22521fa49b346d42d77e"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(lines: list, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    lines.append(line)
    print(line, flush=True)


_G: list = []


def _fds():
    """FDS generate.py (read-only import; -B / dont_write_bytecode: no __pycache__ is written there)."""
    if not _G:
        if str(env.FDS) not in sys.path:
            sys.path.insert(0, str(env.FDS))
        import generate as G  # type: ignore
        if Path(G.__file__).resolve() != (env.FDS / "generate.py").resolve():
            raise SystemExit(f"'generate' imported from {G.__file__}, not from the FDS instrument")
        _G.append(G)
    return _G[0]


def check_code() -> dict:
    got = {"rcwmt/env.py": sha256_file(ROOT / "rcwmt" / "env.py"), "fds/oracles.py": sha256_file(env.FDS / "oracles.py"),
           "fds/generate.py": sha256_file(env.FDS / "generate.py")}
    if got != CODE_SHA:
        raise SystemExit(f"code identity changed: {got}")
    return got


def payload(before, ev) -> str:
    return env.dumps({"before": before, "event": ev})


def opkey(fam: str, before, ev) -> str:
    """FDS generate.operative_key: the answer-determining fields only."""
    return env.dumps([fam, _fds().operative_key(fam, {"before": before, "event": ev})])


# ------------------------------------------------------------------------------------------------ exclusion sources
def harvest_historical() -> tuple[set, list]:
    """FDS generate.harvest_exclusions, line for line, except (1) files named HSKILL_CONFIRM_V2* are skipped by name
    before any open, and (2) .jsonl lines are processed as they are parsed (same result, lower peak RAM)."""
    G = _fds()
    seen: set[str] = set()
    skipped = []

    def visit(doc):
        stack = [doc]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "before" in cur and "event" in cur:
                    seen.add(G.dumps({"before": cur["before"], "event": cur["event"]}))
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str) and '"before"' in cur:
                for o in G._objects_in(cur):
                    seen.add(G.dumps({"before": o["before"], "event": o["event"]}))

    for path in list(RECORDS.rglob("*.jsonl")) + list(RECORDS.rglob("*.json")):
        if path.name.startswith(RESERVED_PREFIX):
            skipped.append(str(path.relative_to(RECORDS)).replace("\\", "/"))
            continue
        try:
            if path.stat().st_size > 60_000_000:
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "before" not in raw:
            continue
        if path.suffix == ".jsonl":
            for line in raw.splitlines():
                if line.strip():
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    visit(doc)
        else:
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                continue
            visit(doc)
        del raw
    return seen, sorted(skipped)


def _family_of(u: dict) -> str | None:
    b = u.get("before")
    if isinstance(b, int) and not isinstance(b, bool):
        return "prerequisite_plan"
    if isinstance(b, dict):
        return "ledger_replay" if "A" in b else ("register_machine" if "X" in b else None)
    return None


def op_keys_of_payloads(payloads) -> set:
    out = set()
    for s in payloads:
        try:
            u = json.loads(s)
            fam = _family_of(u)
            if fam is not None:
                out.add(opkey(fam, u["before"], u["event"]))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return out


def bank_transitions(b: dict):
    for p in b["parents"]:
        fam = p["family"]
        for c in p["candidates"] + ([p["twin"]] if p.get("twin") else []):
            st = env.rollout(fam, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                yield fam, st[t], ev


def load_sources(lines: list, historical: bool = True) -> dict:
    """{source: {"payloads": set, "ops": set, "sha256": str | None, ...}} for every exclusion source."""
    src = {}
    if historical:
        t0 = time.time()
        hist, skipped = harvest_historical()
        src["historical"] = {"payloads": hist, "ops": op_keys_of_payloads(hist), "sha256": None,
                             "skipped_by_name_never_opened": skipped}
        log(lines, f"historical set: {len(hist)} payloads ({time.time() - t0:.0f}s; recorded load_exclusions size "
                   f"{HISTORICAL_RECORDED}); skipped by name, never opened: {skipped}")
    for name in BANKS:
        p = ROOT / "data" / name
        sha = sha256_file(p)
        if sha != EXPECTED_SHA[name]:
            raise SystemExit(f"{name}: sha256 {sha} != pinned {EXPECTED_SHA[name]}")
        b = json.loads(p.read_text(encoding="utf-8"))
        pay, ops = set(), set()
        for fam, before, ev in bank_transitions(b):
            pay.add(payload(before, ev))
            ops.add(opkey(fam, before, ev))
        del b
        src[name] = {"payloads": pay, "ops": ops, "sha256": sha}
        log(lines, f"{name}: {len(pay)} payloads, {len(ops)} operative contents")
    for name in T2_FILES:
        p = ROOT / "data" / name
        sha = sha256_file(p)
        if sha != EXPECTED_SHA[name]:
            raise SystemExit(f"{name}: sha256 {sha} != pinned {EXPECTED_SHA[name]}")
        pay, ops = set(), set()
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pay.add(payload(r["before"], r["event"]))
                ops.add(opkey(r["family"], r["before"], r["event"]))
        src[name] = {"payloads": pay, "ops": ops, "sha256": sha}
        log(lines, f"{name}: {len(pay)} payloads")
    sha = sha256_file(SELECT_VIEWS)
    if sha != EXPECTED_SHA["SELECT_VIEWS.jsonl"]:
        raise SystemExit(f"SELECT-VIEWS-001 sha256 {sha} != pinned")
    pay, ops = set(), set()
    for line in SELECT_VIEWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            u = json.loads(r["prompt"][1]["content"])
            pay.add(payload(u["before"], u["event"]))
            ops.add(opkey(r["family"], u["before"], u["event"]))
    src["SELECT-VIEWS-001"] = {"payloads": pay, "ops": ops, "sha256": sha,
                               "mapping": "user turn == {before, event} (prompt[1].content)"}
    log(lines, f"SELECT-VIEWS-001: {len(pay)} payloads")
    return src


# ------------------------------------------------------------------------------------------------ strata
def raw_register_sum(before: dict, ev: dict) -> int:
    r = before[ev["register"]]
    return r + ev["value"] if ev["kind"] == "add" else r + 2 * before[ev["other"]] + ev["value"]


def stratum_problem(stratum: str, before, ev) -> str | None:
    """None if (before, event) belongs to the stratum, else the reason it does not."""
    fam = FAMILY[stratum]
    after = env.step(fam, before, ev)
    if stratum == "L_big":
        if not any(BIG_LO <= abs(v) <= BIG_HI for v in before.values()):
            return "no balance with |b| in [100, 2000]"
        if any(abs(v) > BIG_HI for v in before.values()):
            return "a balance beyond 2000"
        return None if after != before else "event leaves the state unchanged"
    if stratum == "L_void":
        if ev["status"] != "void":
            return "status is not void"
        return None if env.step(fam, before, {**ev, "status": "posted"}) != before else "posted version is a no-op"
    if stratum == "R_wrap":
        if ev["kind"] not in ("add", "mix"):
            return "not add / mix"
        return None if not 0 <= raw_register_sum(before, ev) <= env.MOD - 1 else "raw sum inside [0, 96]"
    if stratum == "R_boundary":
        if ev["kind"] != "conditional":
            return "not conditional"
        if before[ev["other"]] - ev["threshold"] not in (-1, 0, 1):
            return "other not within 1 of threshold"
        return None if ev["value"] % env.MOD != 0 else "value == 0 mod 97"
    if stratum == "P_inapp":
        if env.applicable(before, ev):
            return "applicable"
        return None if ((before & ~ev["delete"]) | ev["add"]) != before else "applying it would not change the state"
    if stratum == "P_appdel":
        if not env.applicable(before, ev):
            return "inapplicable"
        return None if before & ev["delete"] & ~ev["add"] else "delete clears no set bit (or add restores it)"
    raise ValueError(stratum)


def meta_of(stratum: str, before, ev) -> dict:
    if stratum == "L_big":
        big = sorted(a for a, v in before.items() if abs(v) >= BIG_LO)
        touched = {ev["account"]} | ({ev["other"]} if ev["kind"] == "transfer" else set())
        return {"big_accounts": big, "event_touches_big": bool(touched & set(big)), "kind": ev["kind"]}
    if stratum == "L_void":
        return {"kind": ev["kind"]}
    if stratum == "R_wrap":
        raw = raw_register_sum(before, ev)
        return {"kind": ev["kind"], "raw": raw, "direction": "overflow" if raw > env.MOD - 1 else "underflow"}
    if stratum == "R_boundary":
        d = before[ev["other"]] - ev["threshold"]
        return {"offset": d, "branch": "add" if before[ev["other"]] > ev["threshold"] else "subtract"}
    if stratum == "P_inapp":
        on_unmet = (before & ev["pre_on"]) != ev["pre_on"]
        off_set = (before & ev["pre_off"]) != 0
        return {"why": "both" if on_unmet and off_set else ("pre_on_unmet" if on_unmet else "pre_off_set")}
    if stratum == "P_appdel":
        return {"bits_cleared": bin(before & ev["delete"] & ~ev["add"]).count("1")}
    raise ValueError(stratum)


def draw(stratum: str, rng: random.Random, m: env.Marginals, small_pool: list) -> tuple:
    fam = FAMILY[stratum]
    if stratum == "L_big":
        while True:
            big = [a for a in env.ACCOUNTS if rng.random() < 0.5]
            if big:
                break
        before = {a: (rng.choice((-1, 1)) * rng.randint(BIG_LO, BIG_HI) if a in big else rng.choice(small_pool))
                  for a in env.ACCOUNTS}
        return before, env.gen_event(fam, rng, m, tick=rng.randint(1, 99))
    if fam == "prerequisite_plan":
        return env.gen_start(fam, rng, m), env.gen_action_library(rng, m, 1)[0]
    before = env.gen_start(fam, rng, m)
    ev = env.gen_event(fam, rng, m, tick=rng.randint(1, 99))
    if stratum == "R_boundary" and ev["kind"] == "conditional":
        d = rng.choice([x for x in (-1, 0, 1) if 0 <= ev["threshold"] + x <= env.MOD - 1])
        before = dict(before)
        before[ev["other"]] = ev["threshold"] + d
    return before, ev


def build(lines: list) -> tuple[list, dict]:
    code = check_code()
    log(lines, f"code identity: {code}")
    src = load_sources(lines)
    excl_pay = set().union(*(v["payloads"] for v in src.values()))
    excl_ops = set().union(*(v["ops"] for v in src.values()))
    log(lines, f"exclusion union: {len(excl_pay)} payloads, {len(excl_ops)} operative contents")
    m = env.load_marginals()
    small_pool = [b for b in m.ledger_balances if -99 <= b <= 99]       # gen_start's pool
    rng = random.Random(SEED)
    rows, seen_pay, seen_ops = [], set(), set()
    rejected = {s: Counter() for s in STRATA}
    draws = {}
    for stratum in STRATA:
        fam = FAMILY[stratum]
        quota = dict(R_WRAP_QUOTA) if stratum == "R_wrap" else None
        made, n = 0, 0
        while made < N_PER_STRATUM:
            n += 1
            if n > MAX_DRAWS:
                raise SystemExit(f"{stratum}: could not fill after {MAX_DRAWS} draws")
            before, ev = draw(stratum, rng, m, small_pool)
            why = stratum_problem(stratum, before, ev)
            if why is not None:
                rejected[stratum][why] += 1
                continue
            if quota is not None:
                if quota.get(ev["kind"], 0) <= 0:
                    rejected[stratum]["kind quota full"] += 1
                    continue
            pl, ok = payload(before, ev), opkey(fam, before, ev)
            if pl in excl_pay:
                rejected[stratum]["payload in an exclusion source"] += 1
                continue
            if ok in excl_ops:
                rejected[stratum]["operative content in an exclusion source"] += 1
                continue
            if pl in seen_pay or ok in seen_ops:
                rejected[stratum]["duplicate within the bank"] += 1
                continue
            env.cross_check(fam, before, ev)                              # FDS oracle == env.step
            if quota is not None:
                quota[ev["kind"]] -= 1
            seen_pay.add(pl)
            seen_ops.add(ok)
            after = env.step(fam, before, ev)
            rows.append({"id": f"ood-{stratum}-{made:03d}", "stratum": stratum, "family": fam, "before": before,
                         "event": ev, "messages": env.one_step_messages(fam, before, ev), "gold": env.dumps(after),
                         "meta": meta_of(stratum, before, ev)})
            made += 1
        draws[stratum] = n
        log(lines, f"{stratum}: {made} rows from {n} draws; rejected {dict(rejected[stratum])}")
    overlaps = overlap_counts(rows, src)
    if any(v["payload"] or v["operative"] for v in overlaps.values()):
        raise SystemExit(f"overlap with an exclusion source: {overlaps}")
    log(lines, f"overlaps (payload / operative content) with every source: all 0 ({len(overlaps)} sources)")
    return rows, {"src": src, "rejected": rejected, "draws": draws, "overlaps": overlaps, "code": code,
                  "excl_union": {"payloads": len(excl_pay), "operative": len(excl_ops)}}


def overlap_counts(rows: list, src: dict) -> dict:
    pay = {payload(r["before"], r["event"]) for r in rows}
    ops = {opkey(r["family"], r["before"], r["event"]) for r in rows}
    return {name: {"payload": len(pay & v["payloads"]), "operative": len(ops & v["ops"])} for name, v in src.items()}


def describe(rows: list) -> dict:
    out = {}
    for s in STRATA:
        rs = [r for r in rows if r["stratum"] == s]
        d = {}
        for k in sorted({k for r in rs for k in r["meta"] if k not in ("raw", "big_accounts")}):
            d[k] = dict(sorted(Counter(str(r["meta"][k]) for r in rs).items()))
        if s == "L_big":
            d["n_big_accounts"] = dict(sorted(Counter(len(r["meta"]["big_accounts"]) for r in rs).items()))
            vals = [v for r in rs for v in r["before"].values()]
            d["balance_range"] = [min(vals), max(vals)]
        if s == "R_wrap":
            raws = [r["meta"]["raw"] for r in rs]
            d["raw_range"] = [min(raws), max(raws)]
        out[s] = d
    return out


# ------------------------------------------------------------------------------------------------ verify
def verify(bank_path: Path = OUT, source_path: Path = SOURCE, historical: bool = True, quiet: bool = False) -> list:
    """Independent re-check of a written bank from its file. Returns [(check, ok, detail)]."""
    lines: list = []
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    src_rec = json.loads(source_path.read_text(encoding="utf-8"))
    fsha = sha256_file(bank_path)
    chk("bank file sha256 == its SOURCE record", fsha == src_rec["file_sha256"], fsha[:16])
    rows = [json.loads(l) for l in bank_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    per = Counter(r["stratum"] for r in rows)
    chk(f"{len(STRATA)} strata x {N_PER_STRATUM} = {len(STRATA) * N_PER_STRATUM} rows, ids unique, in stratum order",
        len(rows) == len(STRATA) * N_PER_STRATUM and per == Counter({s: N_PER_STRATUM for s in STRATA})
        and len({r["id"] for r in rows}) == len(rows) and [r["stratum"] for r in rows] == sorted(
            (r["stratum"] for r in rows), key=STRATA.index), dict(per))
    bad_frame, bad_gold, bad_oracle, bad_stratum, bad_meta = [], [], [], [], []
    for r in rows:
        fam = r["family"]
        if fam != FAMILY[r["stratum"]] or r["messages"] != env.one_step_messages(fam, r["before"], r["event"]):
            bad_frame.append(r["id"])
        if r["gold"] != env.dumps(env.step(fam, r["before"], r["event"])):
            bad_gold.append(r["id"])
        try:
            env.cross_check(fam, r["before"], r["event"])
        except AssertionError:
            bad_oracle.append(r["id"])
        if stratum_problem(r["stratum"], r["before"], r["event"]) is not None:
            bad_stratum.append(r["id"])
        if r["meta"] != meta_of(r["stratum"], r["before"], r["event"]):
            bad_meta.append(r["id"])
    chk("every row is in Titan's trained one-step frame (messages == rcwmt.env.one_step_messages)", not bad_frame,
        f"{len(bad_frame)} differ")
    chk("every gold == rcwmt.env.step, and the validated FDS oracle agrees on every row (env.cross_check)",
        not bad_gold and not bad_oracle, f"{len(bad_gold)} gold / {len(bad_oracle)} oracle mismatches")
    chk("every row satisfies its stratum's definition; recorded meta re-derives", not bad_stratum and not bad_meta,
        f"{len(bad_stratum)} / {len(bad_meta)}")
    pay = [payload(r["before"], r["event"]) for r in rows]
    ops = [opkey(r["family"], r["before"], r["event"]) for r in rows]
    chk("payloads and operative contents unique within the bank", len(set(pay)) == len(pay) == len(set(ops)))
    chk("R_wrap holds 50 add + 50 mix", Counter(r["event"]["kind"] for r in rows if r["stratum"] == "R_wrap")
        == Counter(R_WRAP_QUOTA))
    src = load_sources(lines, historical=historical)
    ov = overlap_counts(rows, src)
    chk("zero payload and zero operative-content overlap with every exclusion source"
        + ("" if historical else " (historical set not re-harvested here)"),
        all(v["payload"] == 0 and v["operative"] == 0 for v in ov.values()),
        "; ".join(f"{k} {v['payload']}/{v['operative']}" for k, v in ov.items()))
    sizes_ok = all(src_rec["exclusion_sources"][k]["payloads"] == len(v["payloads"]) for k, v in src.items())
    chk("exclusion-source sizes equal the build record's", sizes_ok,
        ", ".join(f"{k} {len(v['payloads'])}" for k, v in src.items()))
    if historical:
        chk("historical harvest skipped exactly the reserved HSKILL_CONFIRM_V2* files by name (never opened)",
            src["historical"]["skipped_by_name_never_opened"]
            == src_rec["exclusion_sources"]["historical"]["skipped_by_name_never_opened"]
            and all(Path(p).name.startswith(RESERVED_PREFIX) for p in src["historical"]["skipped_by_name_never_opened"]),
            str(src["historical"]["skipped_by_name_never_opened"]))
    chk("SOURCE record: no model call, seed and builder identity as fixed here",
        src_rec["model_calls"] == 0 and src_rec["seed"] == SEED and src_rec["builder_sha256"] == sha256_file(Path(__file__))
        and src_rec["n_per_stratum"] == N_PER_STRATUM and src_rec["strata"] == list(STRATA))
    return res


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="re-check the written bank; write nothing")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.verify:
        res = verify()
        print(f"ood bank verify: {sum(ok for _, ok, _ in res)}/{len(res)} checks pass", flush=True)
        return 0 if all(ok for _, ok, _ in res) else 1
    if OUT.exists() or SOURCE.exists():
        raise SystemExit(f"refusing to overwrite {OUT} / {SOURCE} (create-only)")
    lines: list = []
    t0 = time.time()
    log(lines, f"PREREG-L1 A-OOD bank: seed {SEED}, {N_PER_STRATUM} per stratum, strata {list(STRATA)}; no model call")
    rows, info = build(lines)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    fsha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log(lines, f"bank: {len(rows)} rows, sha256 {fsha}; {time.time() - t0:.0f}s")
    src = info["src"]
    rec = {
        "record": "L1_OOD_BANK", "schema": 1, "amendment": "A-OOD (PREREG-L1 criterion C6; DECISIONS D-28(a))",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_calls": 0,
        "builder": "scripts/l1_make_ood.py", "builder_sha256": sha256_file(Path(__file__)),
        "code_sha256": info["code"], "recorded_marginals_sha256": env.RECORDED_SHA,
        "seed": SEED, "n_per_stratum": N_PER_STRATUM, "strata": list(STRATA), "family_of_stratum": FAMILY,
        "definitions": {
            "L_big": "ledger; at least one balance |b| in [100, 2000] (each account big w.p. 1/2, >= 1; big = sign x "
                     "uniform{100..2000}; others from gen_start's +-99 recorded pool); event env.gen_event (recorded "
                     "marginals), status as drawn; rows whose event leaves the state unchanged rejected",
            "L_void": "ledger; env.gen_start + env.gen_event with status void, where the same event posted would "
                      "change the state; gold = unchanged",
            "R_wrap": "register add (50) or mix (50), raw sum outside [0, 96] (wrap or underflow); env.gen_start + "
                      "env.gen_event",
            "R_boundary": "register conditional; other := threshold + d, d uniform over the offsets in {-1, 0, +1} "
                          "that keep it in [0, 96]; value != 0 mod 97; env.gen_start + env.gen_event",
            "P_inapp": "prerequisite action (env.gen_action_library) inapplicable (pre_on unmet or pre_off set) where "
                       "applying it anyway would change the state; gold = unchanged (FDS oracle convention)",
            "P_appdel": "applicable prerequisite action whose delete clears a set bit not restored by add"},
        "frame": "Titan's trained one-step frame: rcwmt.env.one_step_messages(family, before, event) (SYSTEM_ONE)",
        "gold": "rcwmt.env.dumps(rcwmt.env.step(family, before, event)); rcwmt.env.cross_check (validated FDS oracle) "
                "passed on every row",
        "scoring": "PREREG-L1 C6: correct iff rcwmt.env.parse_state(family, prediction, lenient=True) == json.loads(gold)",
        "exclusion": ("no (before, event) payload and no operative content (FDS generate.operative_key) of any row "
                      "equals one of any source below; payloads and operative contents unique within the bank"),
        "exclusion_sources": {k: {"payloads": len(v["payloads"]), "operative_contents": len(v["ops"]),
                                  **({"sha256": v["sha256"]} if v.get("sha256") else {}),
                                  **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"],
                                      "harvest": ("rcwmt.bank.load_exclusions() == FDS generate.harvest_exclusions(), "
                                                  "re-implemented line for line with one change: files named "
                                                  "HSKILL_CONFIRM_V2* are skipped by name before any open (the "
                                                  "operator-reserved HSKILL-CONFIRM-002 is never opened or hashed)"),
                                      "recorded_load_exclusions_size_2026_10_08": HISTORICAL_RECORDED}
                                     if k == "historical" else {}),
                                  **({"mapping": v["mapping"]} if "mapping" in v else {})}
                              for k, v in src.items()},
        "exclusion_union": info["excl_union"],
        "overlap_counts": info["overlaps"],
        "draws": info["draws"], "rejected": {s: dict(c) for s, c in info["rejected"].items()},
        "describe": describe(rows),
        "provenance": ("Fresh. Built 2026-10-08 after the 8-15 cut was chosen, from no data that chose it (the RCWM "
                       "pilot bank and R14's 248 prompts are excluded or unrelated), and read by no model before the "
                       "PREREG-L1 freeze. Ledger balances in L_big lie outside the +-99 range of every RCWM bank."),
        "use": ("PREREG-L1 endpoint C6 (selection) and nothing else; paired L8-15 vs Titan-Sparse per stratum, Supa "
                "report-only. Spent for confirmation by that use."),
        "file": str(OUT), "file_sha256": fsha, "rows": len(rows), "build_log": lines}
    tmp = Path(str(OUT) + ".tmp")
    with open(tmp, "xb") as fh:
        fh.write(text.encode("utf-8"))
    os.replace(tmp, OUT)
    tmp = Path(str(SOURCE) + ".tmp")
    with open(tmp, "xb") as fh:
        fh.write((json.dumps(rec, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, SOURCE)
    print(json.dumps({"file_sha256": fsha, "rows": len(rows), "per_stratum": dict(Counter(r["stratum"] for r in rows)),
                      "overlap_counts": info["overlaps"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
