"""BWM's Blocksworld exclusion recipe for RCX-E07/E08 fresh instances (BWM reply to CX Q-02).

Exactly the recipe of BWM `scripts/bw_b3_prep.py::fresh`, plus `B.test_signatures()` (named in BWM's reply; it is
what BWM's own training-data generator `blocks_data.py` excludes):
  * renaming signatures (B.signature, n <= 7 brute force): B.test_signatures() (PlanBench blocksworld generated_basic,
    generated_basic_3, unsolvable generated_basic = PB-600 and friends) and B.hard_signatures() (hard-110, n <= 7);
  * invariants VL.invariant = (n, sorted init tower heights, sorted goal tower heights), any n, over: hard-110 PDDL,
    runs/BW_val_large/instances_{val_large,mirror_hard,b3_fresh}.json and bw_distill.problems(240).
Instance files are read only to compute signatures/invariants; no instance is copied into RCX. EXCLUSIONS.json stores
the signature hashes, the invariants (as JSON lists), counts and the sha256 of every source file read.

is_excluded(objs, init, goal) applies the recipe as VL.make does: signature check for n <= 7, invariant check for all n.
Coverage note: invariants exist only for n = 6..15 and signatures only for n <= 7 (PlanBench basic sets are 3-5 blocks,
hard-110 has 2 instances with <= 7 blocks), so n = 8..15 instances are screened by the coarse invariant alone, exactly
as in BWM's own generators.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import bwm_import as BI

REPO = Path(__file__).resolve().parents[2]
EXCLUSIONS_PATH = REPO / "experiments" / "RCX-E07" / "EXCLUSIONS.json"
TEST_SIG_DIRS = ("blocksworld/generated_basic", "blocksworld/generated_basic_3", "unsolvable_blocksworld/generated_basic")
HARD_DIR = "blocksworld_hard/generated"
VAL_LARGE_FILES = ("instances_val_large.json", "instances_mirror_hard.json", "instances_b3_fresh.json")
DISTILL_N = 240


def _inv_json(inv) -> list:
    return [inv[0], list(inv[1]), list(inv[2])]


def _inv_key(x) -> tuple:
    return (int(x[0]), tuple(x[1]), tuple(x[2]))


def _b3_fresh_regeneration_check(B, VL, hard, hard_sigs, inv_pre_b3, src_sha, rel) -> dict:
    """Validate the ast-extracted bw_distill.problems end to end, without running bw_distill or writing anything.

    bw_b3_prep.py::fresh wrote instances_b3_fresh.json = json.dumps(VL.make(Random(FRESH_SEED), sorted hard sizes,
    B.hard_signatures(), invariants of hard + val_large + mirror + problems(240))). Recomputing it in memory and
    comparing sha256 with the file checks the extracted problems() (its invariants steer VL.make's rejections). The
    regenerated instances are hashed and discarded immediately (never stored, never used).
    """
    import ast
    import random

    prep = BI.BWM_ROOT / "scripts" / "bw_b3_prep.py"
    text = prep.read_bytes()
    src_sha[rel(prep)] = BI.sha256_bytes(text)
    seed = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["FRESH_SEED"]:
            seed = ast.literal_eval(node.value)
    sizes = sorted(len(h[0]) for h in hard)
    regen_sha = BI.sha256_bytes(json.dumps(VL.make(random.Random(seed), sizes, hard_sigs, inv_pre_b3)).encode())
    file_sha = src_sha["runs/BW_val_large/instances_b3_fresh.json"]
    return {"fresh_seed": seed, "regenerated_sha256_matches_file": regen_sha == file_sha}


def build_exclusions(path: Path | None = EXCLUSIONS_PATH) -> dict:
    """Compute the exclusion sets from the pinned BWM code and BWM's instance files; optionally save JSON (LF)."""
    B, VL = BI.load()
    problems, distill_seed = BI.distill_problems()
    bench = B.BENCH
    src_sha, counts, by_n = {}, {}, {}

    def rel(p: Path) -> str:
        return p.resolve().relative_to(BI.BWM_ROOT.resolve()).as_posix()

    for sub in TEST_SIG_DIRS + (HARD_DIR,):
        files = sorted((bench / sub).glob("*.pddl"))
        counts[f"pddl_files:{sub}"] = len(files)
        for p in files:
            src_sha[rel(p)] = BI.file_sha(p)
    test_sigs = B.test_signatures()
    hard_sigs = B.hard_signatures()
    sig_ns = {}  # block counts of the instances behind each signature source (as B.*_signatures filter them)
    for sub in TEST_SIG_DIRS:
        c = Counter()
        for p in (bench / sub).glob("*.pddl"):
            try:
                c[len(B.parse_problem(p)[0])] += 1
            except Exception:
                c["parse_error"] += 1
        sig_ns[sub] = dict(sorted(c.items(), key=str))

    inv_by_source = {}
    hard = [B.parse_problem(p) for p in sorted((bench / HARD_DIR).glob("*.pddl"))]
    inv_by_source["hard110"] = [VL.invariant(*h) for h in hard]
    for fn in VAL_LARGE_FILES:
        p = BI.BWM_ROOT / "runs" / "BW_val_large" / fn
        raw = p.read_bytes()
        src_sha[rel(p)] = BI.sha256_bytes(raw)
        rows = json.loads(raw.decode("utf-8"))
        inv_by_source[fn] = [VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"])))
                             for r in rows]
    dp = problems(DISTILL_N)
    if [it["id"] for it in dp] != [f"distill/{i}" for i in range(DISTILL_N)]:
        raise RuntimeError("bw_distill.problems returned unexpected ids")
    inv_by_source[f"bw_distill.problems({DISTILL_N})"] = [VL.invariant(it["objs"], it["init"], it["goal"]) for it in dp]
    for k, v in inv_by_source.items():
        counts[f"instances:{k}"] = len(v)
        counts[f"unique_invariants:{k}"] = len(set(v))
        by_n[k] = dict(sorted(Counter(x[0] for x in v).items()))
    inv_all = set().union(*map(set, inv_by_source.values()))
    hard_small = [h for h in hard if len(h[0]) <= 7]
    sig_ns[HARD_DIR + " (n <= 7)"] = dict(sorted(Counter(len(h[0]) for h in hard_small).items()))
    if {B.signature(*h) for h in hard_small} != hard_sigs:
        raise RuntimeError("hard_signatures() does not match the hard instances with <= 7 blocks")
    inv_pre_b3 = set().union(*(set(v) for k, v in inv_by_source.items() if k != "instances_b3_fresh.json"))
    b3_check = _b3_fresh_regeneration_check(B, VL, hard, hard_sigs, inv_pre_b3, src_sha, rel)
    counts.update(test_signatures=len(test_sigs), hard_signatures=len(hard_sigs),
                  signatures_union=len(test_sigs | hard_sigs), invariants_union=len(inv_all))
    for rel_code in BI.PINS:
        src_sha[rel_code] = BI.PINS[rel_code]
    out = {
        "what": "RCX-E07/E08 exclusions = BWM scripts/bw_b3_prep.py::fresh recipe + B.test_signatures()",
        "rule": "exclude if (n <= 7 and B.signature(objs, init, goal) in signatures) or VL.invariant(objs, init, goal) "
                "in invariants",
        "bwm_import": BI.provenance(),
        "bw_distill_seed": distill_seed,
        "bw_distill_problems_check": b3_check,
        "counts": counts,
        "signature_block_counts_by_source": sig_ns,
        "test_signature_block_counts": sorted({k for v in (sig_ns[s] for s in TEST_SIG_DIRS) for k in v
                                               if isinstance(k, int) and k <= 7}),
        "invariant_block_counts_by_source": by_n,
        "test_signatures": sorted(test_sigs),
        "hard_signatures": sorted(hard_sigs),
        "invariants": sorted(_inv_json(x) for x in inv_all),
        "source_sha256": dict(sorted(src_sha.items())),
    }
    if path is not None:
        Path(path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        load_exclusions.cache_clear()
    return out


def verify_sources(ex: dict) -> list[str]:
    """Return the BWM source files whose current sha256 differs from the one recorded in `ex` (empty = unchanged)."""
    bad = []
    for r, want in ex["source_sha256"].items():
        p = BI.BWM_ROOT / r
        if not p.exists() or BI.file_sha(p) != want:
            bad.append(r)
    return bad


@lru_cache(maxsize=4)
def load_exclusions(path: str = str(EXCLUSIONS_PATH), verify: bool = True) -> dict:
    """Load EXCLUSIONS.json as {'signatures': set, 'invariants': set, 'raw': dict}; fail closed if sources changed."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["bwm_import"]["pins"] != BI.PINS:
        raise BI.PinError("EXCLUSIONS.json was built with different BWM code pins; rebuild it")
    if verify:
        bad = verify_sources(raw)
        if bad:
            raise BI.PinError(f"BWM exclusion sources changed since EXCLUSIONS.json was built: {bad[:5]}")
    return {"signatures": frozenset(raw["test_signatures"]) | frozenset(raw["hard_signatures"]),
            "invariants": frozenset(_inv_key(x) for x in raw["invariants"]), "raw": raw}


def exclusion_reason(objs, init, goal, ex: dict | None = None, sig: str | None = None) -> str | None:
    """'invariant' / 'signature' if the instance falls under BWM's exclusion recipe, else None.

    The cheap invariant test runs first (the recipe is an OR, so the order changes only the reported reason). `sig`
    may pass a precomputed B.signature(objs, init, goal) (n <= 7 brute force is ~0.1 s at n = 7).
    """
    B, VL = BI.load()
    ex = ex or load_exclusions()
    if VL.invariant(list(objs), init, goal) in ex["invariants"]:
        return "invariant"
    if len(objs) <= 7:
        if sig is None:
            sig = B.signature(list(objs), init, goal)
        if sig in ex["signatures"]:
            return "signature"
    return None


def is_excluded(objs, init, goal, ex: dict | None = None) -> bool:
    return exclusion_reason(objs, init, goal, ex) is not None


def row_exclusion_reason(objs, state, goal, ex: dict | None = None) -> str | None:
    """The recipe applied to a dataset ROW (state, goal) viewed as an instance (state as init), so that no training row
    equals a reserved instance's root with its h* label. Same decision as exclusion_reason, computed faster:
      * invariant test for every n;
      * signature test only for handempty states with n in test_signature_block_counts (3-5): every reserved instance
        has a handempty init (a signature fixes the fact multiset, so a holding state cannot match), and the hard-110
        signatures (n = 6, 7) are subsumed by the invariant test (renaming-equivalent instances share invariants, and
        every hard instance's invariant is in the set).
    """
    B, VL = BI.load()
    ex = ex or load_exclusions()
    if VL.invariant(list(objs), state, goal) in ex["invariants"]:
        return "invariant"
    if len(objs) in ex["raw"]["test_signature_block_counts"] and ("handempty",) in state:
        if B.signature(list(objs), state, goal) in ex["signatures"]:
            return "signature"
    return None


if __name__ == "__main__":
    import time

    t0 = time.time()
    res = build_exclusions()
    print(json.dumps(res["counts"], indent=1))
    print("bw_distill.problems check:", res["bw_distill_problems_check"])
    print(f"built in {time.time() - t0:.1f}s ->", EXCLUSIONS_PATH)
    print("bytecode check:", BI.check_no_bytecode())
