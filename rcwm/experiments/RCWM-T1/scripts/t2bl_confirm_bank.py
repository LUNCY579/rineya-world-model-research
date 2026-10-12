"""PREREG-5 sealed instruments: the one-step map bank (part a) and the planning draw (part b), minted AT THE FREEZE from a
commit-reveal seed rule (as PREREG-LB: BWM commits to sha256(salt) in its co-sign and reveals the salt only for the
freeze, so nobody can compute either set before the freeze). Torch-free and model-free.

  seed(tag) = int(sha256('<tag>|<sha256 of PREREG_5_DRAFT.md>|<freshness-union digest>|<BWM salt>')[:12], 16)
      tags: 'PREREG-5 map bank v1', 'PREREG-5 planning draw v1'
  map bank       t2bl_make_data.build_bank(seed_map, t2bl_common.CONFIRM_STRATA, union, 'p5', 'p5-map', split='confirm'):
                 28 strata on 100 bases (2,701 rows), the dev bank's derivations rendered with the CONFIRM half of every
                 split held-out partition (words, names, key orders, paraphrase 5), plus the report-only H_order_top /
                 H_order_inner and PREREG-3's eight unseen-status strata instead of the dev bank's pooled P3_words
  planning draw  lbeam_problems.make_set(generator v2 primary spec: 36 per optimal depth 2-6, V 8, K 8, D_CLOSE 8,
                 seed_plan, union + every map-bank payload): N = 180, in range and closed under the vocabulary
  union          t2bl_make_data.build_union(skip_prefix=None): PREREG-LB's union (name-skipping historical harvest,
                 every pinned source, PREREG-3's bank, every data/**/*.json[l] incl. T2b-ledger's training set and dev
                 bank), the binding transitions of PREREG-LB's spent dev sets and confirmation draw, every {before,
                 event} payload under docs/ except docs/t2b/ledger/. HSKILL_CONFIRM_V2* skipped by name; never
                 rcwmt.bank.load_exclusions().
Both sets are written create-only as data/t2bl_p5_map.jsonl and data/t2bl_p5_plan.jsonl with .SOURCE.json records (spec,
seed rule inputs, rejections, overlap counts; no content), sealed by sha256 in PREREG_5_FREEZE.json, and never printed or
inspected before the registered run. Both regenerate byte for byte from (seed, recorded rejections) without the union.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t2bl_common as C  # noqa: E402
import t2bl_make_data as MD  # noqa: E402

TAG_MAP = "PREREG-5 map bank v1"
TAG_PLAN = "PREREG-5 planning draw v1"
SEED_FORMULA = "int(sha256('<tag>|<draft sha256>|<union digest>|<BWM salt>')[:12], 16)"
MAP_PREFIX, MAP_TAG = "p5", "p5-map"
PLAN_SET_NAME = "p5"
N_PER_DEPTH, DEPTHS = 36, (2, 3, 4, 5, 6)
MAP_REL, MAP_SOURCE_REL = "data/t2bl_p5_map.jsonl", "data/t2bl_p5_map.SOURCE.json"
PLAN_REL, PLAN_SOURCE_REL = "data/t2bl_p5_plan.jsonl", "data/t2bl_p5_plan.SOURCE.json"
SPENT_SEEDS = (C.SEED_TRAIN, C.SEED_DEV, C.ORDER_SEED, C.TEST_SEED, *C.SPENT_KIT_SEEDS,   # 20261161-20261166
               20261141, 20261142, 20261143)


class BankError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise BankError(msg)


def seed_from_rule(tag: str, draft_sha: str, digest: str, salt: str) -> int:
    return int(hashlib.sha256(f"{tag}|{draft_sha}|{digest}|{salt}".encode("utf-8")).hexdigest()[:12], 16)


def union_digest(excl: dict) -> str:
    h = hashlib.sha256()
    for s in sorted(excl["pay"]):
        h.update(s.encode("utf-8") + b"\n")
    h.update(b"\x00")
    for s in sorted(excl["ops"]):
        h.update(s.encode("utf-8") + b"\n")
    return h.hexdigest()


def plan_spec(seed: int):
    import lbeam_problems as P
    return P.SetSpec(name=PLAN_SET_NAME, seed=seed, n_in_per_depth=N_PER_DEPTH, n_oor_per_depth=0, depths=DEPTHS,
                     V=P.V_DEFAULT, K=P.K_FIXED, d_close=P.D_CLOSE)


def map_items(rows: list) -> tuple:
    import lbeam_problems as P
    L1 = P._L1()
    return ({L1.payload(r["before"], r["event"]) for r in rows},
            {L1.opkey(C.LEDGER, r["before"], r["event"]) for r in rows})


def mint(draft_sha: str, excl: dict, salt: str, map_n_bases: int = C.N_BASES, plan_n_per_depth: int = N_PER_DEPTH) -> dict:
    """Both sets (in memory) and their SOURCE records. Nothing about a row or an instance is printed."""
    import lbeam_problems as P
    dig = union_digest(excl)
    s_map, s_plan = seed_from_rule(TAG_MAP, draft_sha, dig, salt), seed_from_rule(TAG_PLAN, draft_sha, dig, salt)
    need(s_map not in SPENT_SEEDS and s_plan not in SPENT_SEEDS and s_map != s_plan, "the seed rule gave a spent seed")
    rows, mrec = MD.build_bank(s_map, C.CONFIRM_STRATA, excl, MAP_PREFIX, MAP_TAG, n_bases=map_n_bases, split="confirm")
    ov_map = MD.overlap([(r["before"], r["event"]) for r in rows], excl)
    need(not ov_map["payload_overlap"] and not ov_map["operative_overlap"], f"the map bank overlaps the union {ov_map}")
    regen, _ = MD.build_bank(s_map, C.CONFIRM_STRATA, MD.empty_union(), MAP_PREFIX, MAP_TAG,
                             replay=mrec["fresh_rejections"], n_bases=map_n_bases, split="confirm")
    map_text = C.dumps_rows(rows)
    need(C.dumps_rows(regen) == map_text, "the map bank does not regenerate from (seed, recorded rejections)")
    import t2bl_audit as AU
    au = AU.audit_bank(rows, C.CONFIRM_STRATA, "confirm")
    need(not au["problems"], f"the map bank does not pass the stratum audit ({len(au['problems'])} problems; not shown)")
    mpay, mops = map_items(rows)
    excl2 = {"pay": excl["pay"] | mpay, "ops": excl["ops"] | mops, "historical": excl.get("historical"),
             "sources": {"union": f"the map bank's union ({len(excl['pay'])} payloads; sources in {Path(MAP_SOURCE_REL).name})",
                         "map_bank": {"payloads": len(mpay), "operative": len(mops)}},
             "rule": str(excl.get("rule") or "") + " + every map-bank payload and operative content"}
    spec = plan_spec(s_plan)
    if plan_n_per_depth != N_PER_DEPTH:
        spec = P.SetSpec(**{**spec.__dict__, "n_in_per_depth": plan_n_per_depth})
    insts, prec = P.make_set(spec, excl2, log=None)
    ov_plan = P.overlap_counts(insts, excl2)
    need(not ov_plan["payload_overlap"] and not ov_plan["operative_overlap"], f"the planning draw overlaps {ov_plan}")
    ver = P.verify_instances(insts)
    need(ver["ok"], f"the planning draw does not verify ({len(ver.get('problems') or [])} problems; not shown)")
    regen_p, _ = P.make_set(spec, replay_rejections=prec["fresh_rejections"])
    plan_text = P.dumps_set(insts)
    need(P.set_sha(regen_p) == prec["file_sha256"] == C.sha256_text(plan_text), "the planning draw does not regenerate")
    per_depth = {d: sum(1 for i in insts if i["opt_depth"] == d) for d in DEPTHS}
    need(per_depth == {d: plan_n_per_depth for d in DEPTHS}, "the planning draw is not balanced over d*")
    seed_rule = {"formula": SEED_FORMULA, "draft_sha256": draft_sha, "union_digest": dig, "bwm_salt": salt,
                 "bwm_commit": hashlib.sha256(str(salt).encode("utf-8")).hexdigest()}
    union_rec = {"rule": excl.get("rule"), "n_payloads": len(excl["pay"]), "n_operative": len(excl["ops"]),
                 "sources": excl.get("sources"), "errors": MD.union_errors(excl)}
    msrc = {"record": "PREREG5_MAP_BANK", "file": Path(MAP_REL).name, "file_sha256": C.sha256_text(map_text),
            "seed_rule": {**seed_rule, "tag": TAG_MAP, "seed": s_map}, **{k: v for k, v in mrec.items() if k != "seed"},
            "strata": {s: sum(r["stratum"] == s for r in rows) for s in C.CONFIRM_STRATA}, "overlap": ov_map,
            "regenerates": True, "stratum_audit_problems": 0, "union": union_rec,
            "code_sha256": C.code_hashes(MD.BUILD_CODE + ("scripts/t2bl_confirm_bank.py",)),
            "use": "PREREG-5 part (a): sealed at the freeze; read only by t2bl_confirm_run.py and t2bl_confirm_analysis.py",
            "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    psrc = {**prec, "record": "PREREG5_PLAN_DRAW", "seed_rule": {**seed_rule, "tag": TAG_PLAN, "seed": s_plan},
            "overlap": ov_plan, "verify_ok": True, "regenerated_equal": True, "per_depth": per_depth,
            "exclusion": "the map bank's union + every map-bank payload and operative content",
            "use": "PREREG-5 part (b): sealed at the freeze; read only by t2bl_confirm_run.py and t2bl_confirm_analysis.py",
            "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return {"map_rows": rows, "map_text": map_text, "map_source": msrc, "plan_insts": insts, "plan_text": plan_text,
            "plan_source": psrc, "seeds": {"map": s_map, "plan": s_plan}}


def write_sealed(root: Path, M: dict) -> dict:
    root = Path(root)
    for rel in (MAP_REL, MAP_SOURCE_REL, PLAN_REL, PLAN_SOURCE_REL):
        need(not (root / rel).exists(), f"{rel} exists (create-only)")
    C.write_create(root / MAP_REL, M["map_text"])
    C.write_create(root / MAP_SOURCE_REL, json.dumps(M["map_source"], indent=1) + "\n")
    C.write_create(root / PLAN_REL, M["plan_text"])
    C.write_create(root / PLAN_SOURCE_REL, json.dumps(M["plan_source"], indent=1) + "\n")
    return {"map": {"file_sha256": M["map_source"]["file_sha256"], "n": len(M["map_rows"])},
            "plan": {"file_sha256": M["plan_source"]["file_sha256"], "n": len(M["plan_insts"])}}


def verify_sealed(root: Path, fz: dict | None = None) -> list:
    """From the files alone: hashes equal the SOURCE (and the freeze record), both regenerate from (seed, recorded
    rejections), the seed rule recomputes, the salt matches its commitment, the map rows pass the stratum audit, the
    draw verifies. No union is rebuilt (the sealed files are now part of data/)."""
    import lbeam_problems as P
    import t2bl_audit as AU
    root = Path(root)
    errs = []
    ms, ps = C.read_json(root / MAP_SOURCE_REL), C.read_json(root / PLAN_SOURCE_REL)
    if ms is None or ps is None:
        return ["a sealed SOURCE record is missing"]
    if C.sha256(root / MAP_REL) != ms["file_sha256"] or C.sha256(root / PLAN_REL) != ps["file_sha256"]:
        errs.append("a sealed file differs from its SOURCE record")
    if fz is not None and (fz.get("map_bank", {}).get("file_sha256") != ms["file_sha256"]
                           or fz.get("plan_draw", {}).get("file_sha256") != ps["file_sha256"]):
        errs.append("a sealed file differs from the freeze record")
    for src, tag in ((ms, TAG_MAP), (ps, TAG_PLAN)):
        sr = src["seed_rule"]
        if sr["seed"] != seed_from_rule(tag, sr["draft_sha256"], sr["union_digest"], sr["bwm_salt"]):
            errs.append(f"{tag}: the seed does not follow the recorded rule")
        if hashlib.sha256(str(sr["bwm_salt"]).encode("utf-8")).hexdigest() != sr["bwm_commit"]:
            errs.append(f"{tag}: BWM's salt does not match its commitment")
    rows = C.load_jsonl(root / MAP_REL)
    regen, _ = MD.build_bank(ms["seed_rule"]["seed"], C.CONFIRM_STRATA, MD.empty_union(), MAP_PREFIX, MAP_TAG,
                             replay=ms["fresh_rejections"], n_bases=ms["n_bases"], split="confirm")
    if C.dumps_rows(regen) != C.dumps_rows(rows):
        errs.append("the map bank does not regenerate from its seed and recorded rejections")
    if AU.audit_bank(rows, C.CONFIRM_STRATA, "confirm")["problems"]:
        errs.append("the map bank fails the stratum audit")
    insts = P.load_set(root / PLAN_REL)
    spec = P.spec_from_record(ps)
    regen_p, _ = P.make_set(spec, replay_rejections=ps["fresh_rejections"])
    if P.set_sha(regen_p) != ps["file_sha256"] or P.set_sha(insts) != ps["file_sha256"]:
        errs.append("the planning draw does not regenerate from its seed and recorded rejections")
    if not P.verify_instances(insts)["ok"]:
        errs.append("the planning draw does not verify")
    return errs
