"""Item 7, addendum A (research/PREREG_ITEM7_ADDENDUM_A_DRAFT.md): report-only metrics for item 7's runs. DRAFT.

  python -B scripts/item7_addendum_a.py hash [--without-screen-freeze]
        A.0: runs/BW_SEEDS/ADDENDUM_A_INPUTS.json = sha256, line count and size of every plans.jsonl, summary.json and
        fcache.jsonl under runs/BW_SEEDS/{R0,R1,R2}_{beam,direct}/ (merged and per shard) plus the six
        runs/FREEZE_I7_*.json. Nothing is parsed. Written once. It refuses unless the B4 screen is frozen
        (runs/FREEZE_B4S_a_beam.json), because this file closes the screen's late-freeze window; with
        --without-screen-freeze it records that the screen was not frozen (the screen is then void).
  python -B scripts/item7_addendum_a.py analyze
        A.1-A.4 for R0, R1, R2 -> runs/BW_SEEDS/ADDENDUM_A_REPORT.{json,md} (written once). Every input is re-hashed
        against A.0's table first; any difference, or a new file, refuses.
  python -B scripts/item7_addendum_a.py faudit-screen
        A.1 on the screen's arm (set b4s_a_beam, prereg B.6) -> runs/BW_B4_SCREEN/FAUDIT_b4s_a_beam.json, checked
        against the screen's own hash table runs/BW_B4_SCREEN/ANALYSIS_INPUTS.json (written once).

CPU only. No model and no torch: bwmn/blocks.py (exact simulator, parser, prompts), scripts/freplay.py (imported
read-only, sha-pinned: FREPLAY's definitions by reference), bwmn/flops.py (pure; sha recorded) and, for A.4 only, the
Titan-Sparse tokenizer.json through the `tokenizers` library (no transformers; TOKENIZERS_PARALLELISM=false).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402

SYSTEMS = ("R0", "R1", "R2")
CONFIGS = ("beam", "direct")
BEAM_SHARDS = 4                                     # = bw_seeds.BEAM_SHARDS
SEEDS_DIR = "runs/BW_SEEDS"
RUN_DONE_REL = "runs/BW_SEEDS/.run_done"
SEEDS_REPORT_REL = "runs/BW_SEEDS/SEEDS_REPORT.json"
INPUTS_REL = "runs/BW_SEEDS/ADDENDUM_A_INPUTS.json"  # = b4_screen.ADDENDUM_A_INPUTS_REL
REPORT_REL = "runs/BW_SEEDS/ADDENDUM_A_REPORT"
SCREEN_FREEZE_REL = "runs/FREEZE_B4S_a_beam.json"
SCREEN_INPUTS_REL = "runs/BW_B4_SCREEN/ANALYSIS_INPUTS.json"
SCREEN_ARM_REL = "runs/BW_B4_SCREEN/a_beam"
FAUDIT_SCREEN_REL = "runs/BW_B4_SCREEN/FAUDIT_b4s_a_beam.json"
FREEZE_I7 = tuple(f"runs/FREEZE_I7_{s}_{c}.json" for s in SYSTEMS for c in CONFIGS)
SEEDS_FRESH_REL = "runs/BW_val_large/instances_seeds_fresh.json"
FREPLAY_REL = "scripts/freplay.py"
FLOPS_REL = "bwmn/flops.py"
PINS = {  # checked before use (tests replace this table); scripts/freplay.py = P-15's registered script
    "PREREG_ITEM7_SEEDS.md": "956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d",
    "scripts/bw_seeds.py": "b783a1916bce655cf93e3d55bedcb276dae3dab2bbdc214e00ff633c828f3b08",
    SEEDS_FRESH_REL: "6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc",
    "research/PREREG_FREPLAY.md": "994d74c0b721724a49cd24774b5ecf0fc12027d058718575f276b100b3e975f8",
    "bwmn/blocks.py": "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83",
    "bwmn/blocks_planner.py": "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960",  # semantics only
}
FREPLAY_SHA256 = "866ca5b696b465d0de77d0882b38323df50f1bb3bcc216ff68aaadbb0beb0a3b"   # read-only import, pinned
TITAN_DIR = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                 "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
TITAN_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
F_MAX_NEW = 40                                      # LMForwardBW's delta-format generation cap (EOS included)
ROUND_CAP_MIN = 12
TOKEN_CAPS = (2000, 4000, 6000, 8000, 10000, 12000, 16000)
CLAIMS = {s: f"item 7 {s} S-beam: 'every plan S-beam verified in imagination was valid'" for s in SYSTEMS}
CLAIMS["b4s_a_beam"] = "B4 screen arm a: verified-plan statements"


class PinError(RuntimeError):
    """An input differs from its pin or from A.0's table."""


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(root, p) -> str:
    return str(Path(p).relative_to(root)).replace("\\", "/")


def check_pins(root=ROOT) -> dict:
    out = {}
    for r, want in PINS.items():
        p = Path(root) / r
        if not p.exists() or sha256_file(p) != want:
            raise PinError(f"{r}: {sha256_file(p) if p.exists() else 'missing'} != pinned {want}")
        out[r] = want
    return out


# ----------------------------------------------------------------------------- A.0: hash before any parse

def expected_files() -> list:
    out = []
    for s in SYSTEMS:
        d = f"{SEEDS_DIR}/{s}_direct"
        out += [f"{d}/plans.jsonl", f"{d}/summary.json"]
        d = f"{SEEDS_DIR}/{s}_beam"
        out += [f"{d}/plans.jsonl", f"{d}/summary.json"]
        for i in range(BEAM_SHARDS):
            out += [f"{d}/shard_{i}of{BEAM_SHARDS}/{f}" for f in ("plans.jsonl", "summary.json", "fcache.jsonl")]
    return out + list(FREEZE_I7)


def discovered_files(root) -> list:
    """Every plans.jsonl / summary.json / fcache.jsonl under the six run directories (merged and per shard)."""
    root, out = Path(root), []
    for s in SYSTEMS:
        for c in CONFIGS:
            d = root / SEEDS_DIR / f"{s}_{c}"
            if d.exists():
                out += [rel(root, p) for p in d.rglob("*")
                        if p.is_file() and p.name in ("plans.jsonl", "summary.json", "fcache.jsonl")]
    return sorted(out)


def file_entry(p) -> dict:
    data = Path(p).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "lines": len(data.splitlines()), "bytes": len(data)}


def stage_hash(a, root=ROOT) -> int:
    root = Path(root)
    out = root / INPUTS_REL
    if out.exists():
        sys.exit(f"{out} exists (written once)")
    screen = (root / SCREEN_FREEZE_REL).exists()
    if not screen and not getattr(a, "without_screen_freeze", False):
        sys.exit(f"{SCREEN_FREEZE_REL} missing: this step closes the B4 screen's late-freeze window "
                 "(b4_screen.py freeze refuses once ADDENDUM_A_INPUTS.json exists). Freeze the screen first (the late "
                 "freeze `--allow-after-r0` is mandatory if item 7 already ran) or pass --without-screen-freeze, which "
                 "records that the screen was never frozen and is therefore void")
    if not (root / RUN_DONE_REL).exists():
        sys.exit(f"{RUN_DONE_REL} missing: item 7's runs are not complete")
    exp = expected_files()
    missing = [r for r in exp if not (root / r).exists()]
    if missing:
        sys.exit(f"item 7 outputs incomplete: {missing[:5]}")
    files = sorted(set(exp) | set(discovered_files(root)))
    rec = {"what": "PREREG_ITEM7_ADDENDUM_A A.0: sha256 of every item-7 output file, written before any is parsed "
                   "(PREREG_FREPLAY section 9's sha table for the new sets)",
           "files": {r: file_entry(root / r) for r in files},
           "screen_freeze_present": screen,
           "screen_freeze_sha256": sha256_file(root / SCREEN_FREEZE_REL) if screen else None,
           "context": {SEEDS_REPORT_REL: sha256_file(root / SEEDS_REPORT_REL)
                       if (root / SEEDS_REPORT_REL).exists() else None},
           "script": "scripts/item7_addendum_a.py", "script_sha256": sha256_file(Path(__file__)),
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: {len(files)} files hashed (screen frozen: {screen})")
    return 0


def verify_inputs(root=ROOT) -> dict:
    """Every file of A.0's table re-hashed (refuse on any difference) and no new item-7 output file since."""
    root = Path(root)
    p = root / INPUTS_REL
    if not p.exists():
        raise PinError(f"{INPUTS_REL} missing: run the A.0 hash step first")
    rec = json.loads(p.read_text())
    bad = [r for r, e in rec["files"].items() if not (root / r).exists() or sha256_file(root / r) != e["sha256"]]
    new = sorted(set(discovered_files(root)) - set(rec["files"]))
    if bad or new:
        raise PinError(f"item-7 outputs changed since A.0: changed {bad[:5]}, new {new[:5]}")
    return rec


# ----------------------------------------------------------------------------- inputs

def load_instances(root=ROOT) -> dict:
    """seeds_fresh exactly as blocks_eval.py's _json_set builds it (id 'seeds_fresh/<index>')."""
    p = Path(root) / SEEDS_FRESH_REL
    if SEEDS_FRESH_REL in PINS and sha256_file(p) != PINS[SEEDS_FRESH_REL]:
        raise PinError(f"{SEEDS_FRESH_REL} differs from its pin")
    return {f"seeds_fresh/{i}": {"id": f"seeds_fresh/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
                                 "goal": frozenset(map(tuple, r["goal"]))}
            for i, r in enumerate(json.loads(p.read_text()))}


def rows_of(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def ok(r) -> bool:
    return bool(r.get("val", r["success"]))


def id_key(i: str):
    a, _, b = i.rpartition("/")
    return (a, int(b)) if b.isdigit() else (a, 0, b)


def load_freplay():
    """scripts/freplay.py as a private module (read-only; no bytecode), sha-pinned (FREPLAY definitions unchanged)."""
    p = ROOT / FREPLAY_REL
    got = sha256_file(p)
    if got != FREPLAY_SHA256:
        raise PinError(f"{FREPLAY_REL} {got} != pinned {FREPLAY_SHA256}")
    spec = importlib.util.spec_from_file_location("_i7a_freplay", p)
    mod = importlib.util.module_from_spec(spec)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dwb
    if mod.PREREG_SHA != PINS.get("research/PREREG_FREPLAY.md", mod.PREREG_SHA):
        raise PinError("freplay.py's prereg pin differs from PREREG_FREPLAY.md's")
    return mod


# ----------------------------------------------------------------------------- A.1: per-system F audit (FREPLAY)

def f_audit(FR, root, name: str, run_rel: str, insts: dict) -> dict:
    """FREPLAY sections 2-7 on one beam run's caches (units, classes, strata, per-instance forests, coverage, the two
    readings), with FREPLAY's own functions; only the instance loader is new (seeds_fresh)."""
    root = Path(root)
    files = [f"{run_rel}/shard_{i}of{BEAM_SHARDS}/fcache.jsonl" for i in range(BEAM_SHARDS)]
    caches = {f: FR.read_cache(str(root / f)) for f in files}
    memo = FR.Memo()
    units, conflicts, occ = FR.build_units(files, caches, memo)
    st = FR.strata(units)
    shard_of = {}
    for i in range(BEAM_SHARDS):
        for r in rows_of(root / run_rel / f"shard_{i}of{BEAM_SHARDS}" / "plans.jsonl"):
            if r["id"] in shard_of:
                raise PinError(f"{r['id']} in two shards")
            shard_of[r["id"]] = files[i]
    merged = rows_of(root / run_rel / "plans.jsonl")
    if set(shard_of) != {r["id"] for r in merged}:
        raise PinError(f"{run_rel}: shard ids != merged ids")
    idx = {}
    for f in files:
        bs = {}
        for (s, a) in caches[f][0]:
            bs.setdefault(s, []).append(a)
        idx[f] = {s: sorted(v) for s, v in bs.items()}
    rows = []
    for r in sorted(merged, key=lambda x: id_key(x["id"])):
        it, f = insts[r["id"]], shard_of[r["id"]]
        cache = caches[f][0]
        states, keys = FR.forest(it["init"], idx[f], cache, memo)
        cc = Counter(x["cls"] for _, x in keys)
        harm = sorted(({"key_id": FR.key_id(kk), "class": x["cls"], "subtype": x["sub"], "key": FR.key_json(kk),
                        "text": cache[kk]} for kk, x in keys if x["cls"] in FR.HARMFUL), key=lambda y: y["key_id"])
        cov = FR.coverage({"init": it["init"], "goal": it["goal"], "plan": [tuple(x.split()) for x in r["plan"]]},
                          cache, memo)
        rows.append({"id": r["id"], "n_blocks": len(it["objs"]), "success": bool(r["success"]),
                     "verified": bool(r["verified"]), "fail": r["fail"], "cache": f, "forest_states": len(states),
                     "forest_keys": len(keys), "forest_classes": {c: cc[c] for c in FR.CLASSES},
                     "flagged": bool(harm), "support_exposed": cc["false_legal_rejected_by_support_check"] > 0,
                     "harmful_keys": harm, "coverage": cov})
    harmful_units = sorted(({"key_id": FR.key_id(kk), "class": x["cls"], "subtype": x["sub"], "key": FR.key_json(kk),
                             "text": text, "n_blocks": len(B.blocks_of(kk[0]))}
                            for kk, (x, text, _) in units.items() if x["cls"] in FR.HARMFUL), key=lambda y: y["key_id"])
    r2 = FR.reading_r2(name, st["total"])
    r2["claim"] = CLAIMS.get(name.replace("i7_", "").replace("_beam", ""), CLAIMS.get(name))
    t = st["total"]
    return {"files": {f: caches[f][1] for f in files}, "unique_keys": len(units), "conflicting_keys": len(conflicts),
            "conflicts": conflicts, "occurrence_level": occ, "strata": st, "harmful_keys": harmful_units,
            "instances": rows,
            "checks": {"plans_not_covered": [x["id"] for x in rows if not x["coverage"]["covered"]],
                       "verified_plans_not_goal_under_F": [x["id"] for x in rows if x["verified"]
                                                           and x["coverage"]["goal_at_end"] is not True]},
            "FREPLAY_R1": FR.reading_r1(rows), "FREPLAY_R2": r2,
            "wording": ("verified by the learned F plus a domain-agnostic delta-consistency check (a predicted deletion "
                        "must be a fact of the state)") if r2["wording_change_triggered"] else None,
            "additional_report_only": {"flag_accuracy_schema_valid": t["secondary"]["flag_accuracy"],
                                       "delta_exact": t["secondary"]["delta_exact"],
                                       "harmful_by_class": {c: t["classes"][c] for c in FR.HARMFUL}}}


def seed_spread(audits: dict) -> dict:
    """Report-only: the spread of the per-system rates across the three F seeds (no test, no reading)."""
    def p(a, path):
        x = a
        for k in path:
            x = x[k]
        return x["p"]
    paths = {"harmful": ("strata", "total", "rates", "harmful"),
             "false_illegal": ("strata", "total", "rates", "false_illegal"),
             "support_share_R2": ("strata", "total", "rates", "support_share_R2"),
             "flag_accuracy": ("additional_report_only", "flag_accuracy_schema_valid"),
             "delta_exact": ("additional_report_only", "delta_exact")}
    out = {}
    for k, path in paths.items():
        v = {s: p(a, path) for s, a in audits.items()}
        xs = [x for x in v.values() if x is not None]
        out[k] = {"by_system": v, "min": min(xs) if xs else None, "max": max(xs) if xs else None}
    return out


# ----------------------------------------------------------------------------- A.2: failure taxonomy

def unmet_after(it, plan_strs):
    ex = B.execute(it["init"], it["goal"], [tuple(x.split()) for x in plan_strs])
    return len(it["goal"] - ex["final"]) if ex["valid"] else None


def beam_class(r) -> str:
    if ok(r):
        return "solved"
    if r["verified"]:
        return "verified_but_invalid"
    return r.get("fail") or "other"


def direct_class(r) -> str:
    if ok(r):
        return "solved"
    if r.get("fail") == "unparseable":
        return "unparseable"
    return "invalid_action" if not r.get("valid", True) else "goal_not_reached"


def beam_taxonomy(rows: list, insts: dict) -> dict:
    cls = Counter(beam_class(r) for r in rows)
    dist, stalled = Counter(), []
    for r in rows:
        if ok(r) or r["verified"]:
            continue
        u = unmet_after(insts[r["id"]], r["plan"])
        dist["invalid_prefix" if u is None else str(u)] += 1
        if u == 1:
            stalled.append(r["id"])
    return {"classes": {c: cls[c] for c in ("solved", "verified_but_invalid", "dead_end", "budget", "other")},
            "unverified_successes": sorted((r["id"] for r in rows if ok(r) and not r["verified"]), key=id_key),
            "val_vs_executor_disagreements": sum(bool(r.get("val", r["success"])) != bool(r["success"]) for r in rows),
            "unverified_failures_unmet_goal_facts": {"note": "executor's true state after the submitted prefix (the "
                                                             "best node's); an invalid prefix is counted separately",
                                                     "distribution": dict(sorted(dist.items())),
                                                     "stalled_at_1": len(stalled), "ids_stalled_at_1": stalled}}


def direct_taxonomy(rows: list, m_cap: int) -> dict:
    cls = Counter(direct_class(r) for r in rows)
    return {"classes": {c: cls[c] for c in ("solved", "unparseable", "invalid_action", "goal_not_reached")},
            "at_cap": {"cap_tokens": m_cap, "n": sum(r["gen_tokens"] >= m_cap for r in rows),
                       "by_class": dict(Counter(direct_class(r) for r in rows if r["gen_tokens"] >= m_cap))},
            "val_vs_executor_disagreements": sum(bool(r.get("val", r["success"])) != bool(r["success"]) for r in rows)}


def crosstab(a: dict, b: dict, fa, fb) -> dict:
    """{class_a: {class_b: n}} over the shared ids."""
    out = {}
    for i in sorted(set(a) & set(b), key=id_key):
        ca, cb = fa(a[i]), fb(b[i])
        out.setdefault(ca, Counter())[cb] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


# ----------------------------------------------------------------------------- A.3: round-capped anytime curves

def anytime(rows: list) -> dict:
    """S_r = VAL successes with rounds <= r (exact for a round cap r by D-48c/D-50: the first r rounds of plan_beam do
    not depend on any later cap; E24 unchanged); the token curve is approximate (a token cap would cut a round)."""
    top = max([ROUND_CAP_MIN] + [r["rounds"] for r in rows])
    return {"rounds_exact": {str(t): sum(ok(r) and r["rounds"] <= t for r in rows) for t in range(1, top + 1)},
            "tokens_approx": {str(T): sum(ok(r) and r["gen_tokens"] <= T for r in rows) for T in TOKEN_CAPS}}


def curve_diff(a: dict, b: dict) -> dict:
    return {k: a[k] - b.get(k, b[max(b, key=int)]) for k in a}


# ----------------------------------------------------------------------------- A.4: FLOPs and token ledger

class RustTokenCounter:
    """Token counts with the Titan-Sparse tokenizer.json through the `tokenizers` library only (sequential)."""

    def __init__(self, model_dir=TITAN_DIR):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        p = Path(model_dir) / "tokenizer.json"
        if sha256_file(p) != TITAN_TOKENIZER_SHA256:
            raise PinError(f"{p} differs from its pin")
        from tokenizers import Tokenizer
        self.tok, self.sha256 = Tokenizer.from_file(str(p)), TITAN_TOKENIZER_SHA256

    def count(self, texts: list) -> list:
        return [len(e.ids) for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]


def load_token_counter():
    return RustTokenCounter()


def flops_module():
    from bwmn import flops as FL
    return FL


def flop_params_for(root, adapter_rel: str) -> dict:
    """flop_params(Titan-Sparse config, extra_nonemb = the adapter's LoRA parameters from its safetensors header)."""
    FL = flops_module()
    return FL.flop_params(TITAN_DIR / "config.json",
                          FL.safetensors_params(Path(root) / adapter_rel / "adapter_model.safetensors"))


def beam_ledger(root, run_rel: str, args: dict, insts: dict, counter, fpF: dict, fpP: dict) -> dict:
    """F: exact from the caches (per shard: every unique key was computed once in that shard; prompt =
    B.forward_prompt_delta tokens, generated = the cache text's tokens + EOS, capped at 40). Proposer: ESTIMATE
    (expansions are logged only as a per-shard mean): per instance x_i = min(E + W - 1, 1 + W (rounds - 1)), an upper
    bound on its expansions (a round adds at most W; plan_beam stops at the first round end with >= E), scaled
    so the run total equals the logged total; P = the initial state's proposer prompt; the instance's gen_tokens spread
    evenly over its expansions; shared_prefix_breakdown with the convexity bound on within-sample pairs (cap M)."""
    FL = flops_module()
    root = Path(root)
    f_total, f_keys_shard, prompt_tok, gen_tok, union = 0, 0, 0, 0, set()
    logged_x, logged_q, wall = 0.0, 0, 0.0
    for i in range(BEAM_SHARDS):
        d = root / run_rel / f"shard_{i}of{BEAM_SHARDS}"
        keys = {}
        for line in (d / "fcache.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue                            # a torn last line (LMForwardBW skips it too)
            keys[(frozenset(map(tuple, r["s"])), tuple(r["a"]))] = r["text"]
        ks = list(keys)
        P = counter.count([B.forward_prompt_delta(s, a) for s, a in ks])
        G = [min(g + 1, F_MAX_NEW) for g in counter.count([keys[k] for k in ks])]
        f_total += sum(FL.flops_breakdown(fpF, p, g)["total"] for p, g in zip(P, G))
        f_keys_shard += len(ks)
        prompt_tok += sum(P)
        gen_tok += sum(G)
        union |= set(ks)
        s = json.loads((d / "summary.json").read_text())["summary"]
        logged_x += s["expansions_per_instance"] * s["n"]
        logged_q += s.get("f_model_queries") or 0
        wall += s.get("wall_s") or 0.0
    rows = sorted(rows_of(root / run_rel / "plans.jsonl"), key=lambda r: id_key(r["id"]))
    n = len(rows)
    E, W, M = int(args["E"]), int(args["W"]), int(args["M"])
    x = [min(E + W - 1, 1 + W * (r["rounds"] - 1)) for r in rows]
    scale = logged_x / sum(x) if sum(x) else 0.0
    P0 = counter.count([B.proposer_prompt(insts[r["id"]]["init"], insts[r["id"]]["goal"]) for r in rows])
    per = {}
    p_total = 0.0
    for r, xi, p0 in zip(rows, x, P0):
        xs = xi * scale
        g = r["gen_tokens"] / xs if xs else 0.0
        fl = xs * FL.shared_prefix_breakdown(fpP, p0, [g], sample_pairs=FL.beam_proposer_pairs_bound(g, M))["total"]
        p_total += fl
        per[r["id"]] = {"proposer_flops_estimate": fl, "gen_tokens": r["gen_tokens"], "rounds": r["rounds"],
                        "expansions_bound": xi, "proposer_prompt_tokens": p0}
    solved = sum(map(ok, rows))
    return {"logged_exact": {"n": n, "gen_tokens_total": sum(r["gen_tokens"] for r in rows),
                             "rounds_total": sum(r["rounds"] for r in rows), "f_model_queries": logged_q,
                             "expansions_total": round(logged_x, 2), "expansions_per_instance": round(logged_x / n, 3)
                             if n else None, "wall_s": round(wall, 1)},
            "from_caches_exact": {"f_unique_keys_sum_over_shards": f_keys_shard,
                                  "f_unique_keys_union_FR02": len(union), "f_prompt_tokens": prompt_tok,
                                  "f_generated_tokens": gen_tok},
            "flops": {"f_exact_given_tokenizer": f_total, "proposer_ESTIMATE": p_total, "total": f_total + p_total,
                      "per_instance": (f_total + p_total) / n if n else None,
                      "per_solved_instance": (f_total + p_total) / solved if solved else None},
            "tokens": {"proposer_gen_per_instance": sum(r["gen_tokens"] for r in rows) / n if n else None,
                       "f_prompt_plus_gen_per_instance": (prompt_tok + gen_tok) / n if n else None},
            "expansion_scale_to_logged_total": scale, "per_instance": per}


def direct_ledger(root, run_rel: str, insts: dict, counter, fpP: dict) -> dict:
    FL = flops_module()
    rows = sorted(rows_of(Path(root) / run_rel / "plans.jsonl"), key=lambda r: id_key(r["id"]))
    P0 = counter.count([B.proposer_prompt(insts[r["id"]]["init"], insts[r["id"]]["goal"]) for r in rows])
    total = sum(FL.flops_breakdown(fpP, p, r["gen_tokens"])["total"] for r, p in zip(rows, P0))
    n, solved = len(rows), sum(map(ok, rows))
    return {"n": n, "gen_tokens_total": sum(r["gen_tokens"] for r in rows),
            "flops": {"total": total, "per_instance": total / n if n else None,
                      "per_solved_instance": total / solved if solved else None}}


# ----------------------------------------------------------------------------- analyze

def stage_analyze(a, root=ROOT) -> int:
    root = Path(root)
    out_json, out_md = root / (REPORT_REL + ".json"), root / (REPORT_REL + ".md")
    if out_json.exists() or out_md.exists():
        sys.exit(f"{out_json} exists (written once)")
    inputs = verify_inputs(root)                 # A.0 first: nothing is parsed before this check
    pins = check_pins(root)
    FR = load_freplay()
    insts = load_instances(root)
    frz = {f: json.loads((root / f).read_text()) for f in FREEZE_I7}
    runs = {}
    for s in SYSTEMS:
        for c in CONFIGS:
            rows = rows_of(root / SEEDS_DIR / f"{s}_{c}" / "plans.jsonl")
            if {r["id"] for r in rows} != set(insts) or len(rows) != len(insts):
                raise PinError(f"{s}_{c}: rows do not cover seeds_fresh exactly")
            runs[(s, c)] = {r["id"]: r for r in rows}
    rep = {"what": "PREREG_ITEM7_ADDENDUM_A section A: report-only metrics (no test; no reading changes item 7)",
           "A1_f_audit": {}, "A2_taxonomy": {"beam": {}, "direct": {}, "crosstabs": {}}, "A3_anytime": {},
           "A4_ledger": {}}
    audits = {}
    for s in SYSTEMS:
        audits[s] = f_audit(FR, root, f"i7_{s}_beam", f"{SEEDS_DIR}/{s}_beam", insts)
    rep["A1_f_audit"] = {"sets": {f"i7_{s}_beam": audits[s] for s in SYSTEMS}, "seed_spread_report_only":
                         seed_spread(audits),
                         "FREPLAY_R2_any_triggered": any(audits[s]["FREPLAY_R2"]["wording_change_triggered"]
                                                         for s in SYSTEMS)}
    for s in SYSTEMS:
        rep["A2_taxonomy"]["beam"][s] = beam_taxonomy(list(runs[(s, "beam")].values()), insts)
        rep["A2_taxonomy"]["direct"][s] = direct_taxonomy(list(runs[(s, "direct")].values()),
                                                          int(frz[f"runs/FREEZE_I7_{s}_direct.json"]["args"]["M"]))
        rep["A2_taxonomy"]["crosstabs"][f"{s}: beam x direct"] = crosstab(runs[(s, "beam")], runs[(s, "direct")],
                                                                          beam_class, direct_class)
    for s in ("R1", "R2"):
        rep["A2_taxonomy"]["crosstabs"][f"beam: R0 x {s}"] = crosstab(runs[("R0", "beam")], runs[(s, "beam")],
                                                                      beam_class, beam_class)
        rep["A2_taxonomy"]["crosstabs"][f"direct: R0 x {s}"] = crosstab(runs[("R0", "direct")], runs[(s, "direct")],
                                                                        direct_class, direct_class)
    curves = {s: anytime(list(runs[(s, "beam")].values())) for s in SYSTEMS}
    rep["A3_anytime"] = {"curves": curves,
                         "differences": {f"{s}-R0": {k: curve_diff(curves[s][k], curves["R0"][k])
                                                     for k in ("rounds_exact", "tokens_approx")} for s in ("R1", "R2")}}
    counter = load_token_counter()
    led = {}
    for s in SYSTEMS:
        fa = frz[f"runs/FREEZE_I7_{s}_beam.json"]["args"]
        fd = frz[f"runs/FREEZE_I7_{s}_direct.json"]["args"]
        bl = beam_ledger(root, f"{SEEDS_DIR}/{s}_beam", fa, insts, counter, flop_params_for(root, fa["fwd_adapter"]),
                         flop_params_for(root, fa["pa"]))
        dl = direct_ledger(root, f"{SEEDS_DIR}/{s}_direct", insts, counter, flop_params_for(root, fd["pa"]))
        led[s] = {"beam": bl, "direct": dl,
                  "beam_over_direct_flops_per_instance": (bl["flops"]["per_instance"] / dl["flops"]["per_instance"])
                  if dl["flops"]["per_instance"] else None}
    rep["A4_ledger"] = {"systems": led, "labels": {"f": "exact given the tokenizer", "beam proposer": "ESTIMATE",
                                                   "direct": "exact given the tokenizer"},
                        "no_claim": "no iso-FLOP claim and no comparison with API models follows from these numbers"}
    rep["meta"] = {"inputs_sha256": sha256_file(root / INPUTS_REL), "inputs_files": len(inputs["files"]),
                   "screen_freeze_present_at_A0": inputs.get("screen_freeze_present"), "pins": pins,
                   "freplay_sha256": FREPLAY_SHA256, "flops_sha256": sha256_file(ROOT / FLOPS_REL),
                   "tokenizer_sha256": getattr(counter, "sha256", None), "script_sha256": sha256_file(Path(__file__)),
                   "torch_loaded": "torch" in sys.modules, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    out_json.write_text(json.dumps(rep, indent=1, sort_keys=True))
    out_md.write_text(render_md(rep))
    print(f"wrote {out_json}")
    return 0


def render_md(rep: dict) -> str:
    L = ["# Item 7 addendum A: report-only metrics", "", "Nothing here is a test; no number changes item 7's outcome.", "",
         "| system | beam solved | verified-but-invalid | dead_end | budget | stalled at 1 | FREPLAY-R2 | R1 [L, U] |",
         "|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        t = rep["A2_taxonomy"]["beam"][s]
        a = rep["A1_f_audit"]["sets"][f"i7_{s}_beam"]
        L.append(f"| {s} | {t['classes']['solved']} | {t['classes']['verified_but_invalid']} | "
                 f"{t['classes']['dead_end']} | {t['classes']['budget']} | "
                 f"{t['unverified_failures_unmet_goal_facts']['stalled_at_1']} | {a['FREPLAY_R2']['R2']['p']} | "
                 f"[{a['FREPLAY_R1']['exactF_success_lower_L']}, {a['FREPLAY_R1']['exactF_success_upper_U']}] |")
    L += ["", "| system | beam FLOPs/instance (F exact + proposer ESTIMATE) | direct FLOPs/instance | ratio |",
          "|---|---|---|---|"]
    for s in SYSTEMS:
        x = rep["A4_ledger"]["systems"][s]
        L.append(f"| {s} | {x['beam']['flops']['per_instance']:.4g} | {x['direct']['flops']['per_instance']:.4g} | "
                 f"{x['beam_over_direct_flops_per_instance']:.3g} |")
    return "\n".join(L) + "\n"


def stage_faudit_screen(a, root=ROOT) -> int:
    """B.6: the screen arm's F audit under A.1's definitions, on files the screen hashed before parsing them."""
    root = Path(root)
    out = root / FAUDIT_SCREEN_REL
    if out.exists():
        sys.exit(f"{out} exists (written once)")
    p = root / SCREEN_INPUTS_REL
    if not p.exists():
        sys.exit(f"{SCREEN_INPUTS_REL} missing: the screen's analysis hashes its inputs first")
    table = json.loads(p.read_text())
    need = [f"{SCREEN_ARM_REL}/plans.jsonl"] + [f"{SCREEN_ARM_REL}/shard_{i}of{BEAM_SHARDS}/{f}"
                                                 for i in range(BEAM_SHARDS) for f in ("plans.jsonl", "fcache.jsonl")]
    bad = [r for r in need if r not in table or sha256_file(root / r) != table[r]]
    if bad:
        raise PinError(f"screen arm files differ from {SCREEN_INPUTS_REL}: {bad[:5]}")
    check_pins(root)
    rep = f_audit(load_freplay(), root, "b4s_a_beam", SCREEN_ARM_REL, load_instances(root))
    rep["meta"] = {"screen_inputs_sha256": sha256_file(p), "freplay_sha256": FREPLAY_SHA256,
                   "script_sha256": sha256_file(Path(__file__)), "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["hash", "analyze", "faudit-screen"])
    ap.add_argument("--without-screen-freeze", action="store_true")
    a = ap.parse_args(argv)
    return {"hash": stage_hash, "analyze": stage_analyze, "faudit-screen": stage_faudit_screen}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
