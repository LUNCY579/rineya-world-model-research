"""Mystery Blocksworld data: smoke sample (N instances per regime) + self-checks. CPU only; tokenizer only (no model).

Writes --out (default runs/_mystery_smoke/):
  forbidden_words.json                 every word in the vendored PlanBench assets (+ format / Blocksworld words; cache)
  CONFIG.json                          everything the data depends on (seed, budget, tokenizer / vocabulary / test-set /
                                       code hashes); --resume refuses to continue under a different CONFIG
  instances_{A,B}.jsonl                generated instances (domain JSON, rendered rules, init/goal/plan, schema hash,
                                       slot seed, rejection counts; Regime A also its Blocksworld-space twin + renaming)
  proposer_{A,B}.jsonl                 proposer rows (every optimal-plan suffix not leaking a test signature + one noisy
                                       plan), actions/states targets, n_tokens
  forward_{A,B}.jsonl                  goal-blind transition rows (legal and illegal), n_tokens
  EVAL_ONLY_mystery_prompt_examples.jsonl   3 rendered PlanBench Mystery prompts (format reference; test-only)
  SUMMARY.json                         self-check results, rejection counts, plan-length histograms, token lengths
Generation streams slot by slot (instance i has its own RNG, M.instance_seed): a slot's rows are written and flushed
before its instance record, which commits the slot. --resume truncates uncommitted tails and continues; existing
outputs are never overwritten silently (--resume or --overwrite is required). The token budget (--max-tokens, prompt +
target + EOS under the Titan tokenizer, both proposer target forms and forward rows) is enforced at generation: a
candidate with an overlength row is redrawn. Train with max_len >= --max-tokens so lm.encode_pair drops nothing.
Self-checks (assertions): BFS solvability/optimality, vocabulary disjointness (+ substring and single-token filters),
isomorphism exclusion (Blocksworld and Mystery detected; no Regime-B domain in the Blocksworld family or containing a
>= 3 x 3 Blocksworld fragment), generic executor == bwmn/blocks.py on Blocksworld (and on Mystery through the
recovered renaming), executor == VAL on Mystery plans and on our generated domains (write_domain), Regime-A rendering
with Mystery's names reproduces the PlanBench Mystery prompt byte-for-byte, and - re-read from disk - no emitted
Regime-A proposer row is a renaming of a test problem, every row fits the budget, no duplicate rows, 30-50%
single-letter-object instances, targets round-trip. PlanBench Mystery eval set: 603 instances = 262 distinct problems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # shared machine: keep the tokenizer single-threaded
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import mystery as M  # noqa: E402

TITAN = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
             "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
REGIMES = ("A", "B")
N_FORWARD = 2
OUTPUT_FILES = tuple(f"{k}_{r}.jsonl" for r in REGIMES for k in ("instances", "proposer", "forward")) + \
    ("CONFIG.json", "SUMMARY.json", "EVAL_ONLY_mystery_prompt_examples.jsonl")
MYSTERY_EVAL_INSTANCES, MYSTERY_EVAL_DISTINCT = 603, 262


def user_words(prompt: str) -> set:
    user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    return M.extract_words(user)


def tok_stats(lens) -> dict:
    lens = sorted(lens)
    return {"n": len(lens), "median": statistics.median(lens), "p95": lens[int(0.95 * (len(lens) - 1))],
            "max": lens[-1]}


def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    assert len(tok.encode("<|im_start|>", add_special_tokens=False).ids) == 1
    return tok


def token_fns(tok):
    """(count, single_token): token count of a text; whether a name is ONE token bare or after a space."""
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    return count, (lambda w: count(w) == 1 or count(" " + w) == 1)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inst_id(reg: str, i: int) -> str:
    return f"{reg}{i:04d}"


# ----------------------------------------------------------------------------- output safety: no overwrite, resume

def prepare_out(out: Path, resume: bool, overwrite: bool) -> list[str]:
    """Refuse to touch existing outputs unless --resume (continue them) or --overwrite (delete them first)."""
    if resume and overwrite:
        raise SystemExit("--resume and --overwrite are exclusive")
    found = [f for f in OUTPUT_FILES if (out / f).exists()]
    if found and not (resume or overwrite):
        raise SystemExit(f"{out} already holds outputs ({', '.join(found)}); pass --resume to continue them or "
                         f"--overwrite to replace them")
    if overwrite:
        for f in found:
            (out / f).unlink()
    out.mkdir(parents=True, exist_ok=True)
    return found


def check_config(out: Path, cfg: dict, resume: bool) -> str:
    p = out / "CONFIG.json"
    if p.exists():
        assert resume, p  # prepare_out deleted it otherwise
        old = json.loads(p.read_text())
        diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
        if diff:
            raise SystemExit(f"--resume: {out} was generated under a different configuration ({diff}); "
                             f"use a new --out or --overwrite")
        return "resumed"
    if resume and any((out / f"{k}_{r}.jsonl").exists() for r in REGIMES for k in ("instances", "proposer",
                                                                                     "forward")):
        raise SystemExit(f"--resume: {out} has data files but no CONFIG.json; cannot verify they match")
    p.write_text(json.dumps(cfg, indent=1))
    return "fresh"


def _truncate(path: Path, keep, collect: bool = False) -> tuple[list, int]:
    """Keep the longest prefix of complete JSON lines with keep(obj) true and cut the file there (streaming). Returns
    (the kept objects if collect else [], number of removed lines)."""
    if not path.exists():
        return [], 0
    kept, end, removed, tail = [], 0, 0, False
    with open(path, "rb") as fh:
        pos = 0
        for line in fh:
            pos += len(line)
            try:
                obj = json.loads(line) if line.endswith(b"\n") else None
            except ValueError:
                obj = None
            if not tail and obj is not None and keep(obj):
                end = pos
                if collect:
                    kept.append(obj)
            else:
                tail = True
                removed += 1
    if removed:
        with open(path, "r+b") as fh:
            fh.truncate(end)
    return kept, removed


def load_committed(out: Path, reg: str) -> tuple[list, int]:
    """Committed slots = the contiguous prefix of complete instance records A0000, A0001, ...; rows of a slot are
    written before its record, so rows of uncommitted slots can only form the tail of the rows files."""
    n = [0]

    def nxt(r):
        ok = r.get("id") == inst_id(reg, n[0])
        n[0] += ok
        return ok

    recs, removed = _truncate(out / f"instances_{reg}.jsonl", nxt, collect=True)
    ids = {r["id"] for r in recs}
    for k in ("proposer", "forward"):
        removed += _truncate(out / f"{k}_{reg}.jsonl", lambda r: r.get("id") in ids)[1]
    return recs, removed


# ----------------------------------------------------------------------------- per-instance checks (before writing)

def check_instance(reg: str, it: dict, rows: dict, ctx) -> None:
    dom, objs = it["domain"], it["objs"]
    # BFS solvability / optimality (independent search) + executor success
    p2 = M.bfs_plan(dom, it["init"], it["goal"], objs, max_states=400000)
    assert p2 is not None and len(p2) == len(it["plan"]), reg
    assert M.execute(dom, it["init"], it["goal"], it["plan"])["success"], reg
    if reg == "B":
        assert 2 <= len(it["plan"]) <= 12 and 3 <= len(objs) <= 6
        assert M.domain_rejection(dom) is None and not M.is_blocksworld_family(dom)
        assert not M.contains_blocksworld_fragment(dom)
    else:
        assert M.is_blocksworld(dom) and it["sig"] not in ctx.test_sigs
        R = it["renaming"]  # renamed BW trajectory == generic-executor trajectory
        s_bw, s = it["bw"]["init"], it["init"]
        for x_bw, x in zip(it["bw"]["plan"], it["plan"]):
            s_bw, _ = B.step(s_bw, x_bw)
            s, ok = M.step(dom, s, x)
            assert ok and R.facts(s_bw) == s
    # vocabulary: names unique; every word name alphabetic, >= 2 letters, never in PlanBench, no banned substring, not
    # a single token; prompt words are names or format words. Single-letter objects (letter slots) are a..o.
    names = M.instance_names(it)
    assert len(set(names)) == len(names)
    words = list(dom.preds) + list(dom.actions) + ([] if it["letter_objects"] else list(objs))
    assert all(n.isalpha() and len(n) >= 2 for n in words)
    assert not set(words) & ctx.forb, (reg, set(words) & ctx.forb)
    assert not [w for w in words if any(b in w for b in ctx.banned)]
    assert not [w for w in words if ctx.single_token(w) or w in M.PDDL_WORDS]
    if it["letter_objects"]:
        assert all(o in M.LETTERS for o in objs) and not set(objs) & set(words)
    pw = user_words(M.proposer_prompt(dom, objs, it["init"], it["goal"]))
    assert pw <= set(names) | M.FORMAT_WORDS | {"x", "y"}, pw - set(names) - M.FORMAT_WORDS
    assert not pw & (set(ctx.deceptive) | M.BW_WORDS)
    # rows: suffix-signature guard, distinct rows (noisy plan != optimal plan), token budget
    prop, fwd = rows["proposer"], rows["forward"]
    assert len({(r["prompt"], r["target_actions"]) for r in prop}) == len(prop)
    assert prop[-1]["t"] == -1 and prop[-1]["target_actions"] != M.target_actions(it["plan"])
    assert len({r["prompt"] for r in fwd}) == len(fwd)
    if reg == "A":
        assert all(r["sig"] not in ctx.test_sigs for r in prop)
    assert all(max(r["n_tokens"].values()) <= ctx.max_tokens for r in prop)
    assert all(r["n_tokens"] <= ctx.max_tokens for r in fwd)


# ----------------------------------------------------------------------------- streaming generation

def generate_regime(out: Path, reg: str, n: int, ctx, log=print) -> dict:
    """Slots len(committed)..n-1, each written (rows, flush, record, flush) as soon as it is generated and checked."""
    t = time.time()
    done, removed = load_committed(out, reg)
    if len(done) > n:
        raise SystemExit(f"{out}: regime {reg} already has {len(done)} > --n {n} instances")
    seen = {r["sig"] for r in done if "sig" in r}
    fh = {k: open(out / f"{k}_{reg}.jsonl", "a", encoding="utf-8", newline="\n")
          for k in ("proposer", "forward", "instances")}
    try:
        for i in range(len(done), n):
            inst, rows, rej = M.make_instance(reg, ctx.seed, i, ctx.forb, ctx.test_sigs, seen,
                                              single_token=ctx.single_token, count_tokens=ctx.count,
                                              max_tokens=ctx.max_tokens, max_states=ctx.max_states,
                                              n_forward=N_FORWARD)
            check_instance(reg, inst, rows, ctx)
            iid = inst_id(reg, i)
            for k in ("proposer", "forward"):
                for x in rows[k]:
                    fh[k].write(json.dumps(dict(x, id=iid)) + "\n")
                fh[k].flush()
            fh["instances"].write(json.dumps(dict(M.instance_record(inst, iid), rejections=dict(rej))) + "\n")
            fh["instances"].flush()
            if "sig" in inst:
                seen.add(inst["sig"])
            if (i + 1) % 50 == 0:
                log(f"  {reg}: {i + 1}/{n} ({time.time() - t:.0f}s)")
    finally:
        for f in fh.values():
            f.close()
    return {"resumed_from": len(done), "removed_uncommitted_lines": removed, "generated_now": n - len(done),
            "seconds": round(time.time() - t, 1)}


# ----------------------------------------------------------------------------- verification of what is on disk

def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def val_check_generated(inst: dict, extra_plans, work: Path) -> int:
    """write_domain / write_problem round-trip, VAL accepts the optimal plan, VAL == executor on perturbed plans."""
    dom = inst["domain"]
    dtxt = M.write_domain(dom)
    work.mkdir(parents=True, exist_ok=True)
    dp, pp = work / "domain.pddl", work / "problem.pddl"
    dp.write_text(dtxt)
    pp.write_text(M.write_problem(dom, inst["objs"], inst["init"], inst["goal"]))
    assert M.parse_domain(dp).same_as(dom)
    assert M.parse_problem(pp) == (sorted(inst["objs"]), inst["init"], inst["goal"])
    plan = list(inst["plan"])
    assert B.val_check(pp, plan, dp)
    plans = [plan[:-1], [plan[1], plan[0]] + plan[2:] if len(plan) > 1 else plan] + list(extra_plans)
    for q in plans:
        if q:
            assert B.val_check(pp, q, dp) == M.execute(dom, inst["init"], inst["goal"], q)["success"], q
    return 1 + sum(1 for q in plans if q)


def verify_regime(out: Path, reg: str, n: int, ctx) -> dict:
    """Re-read instances / rows from disk and assert the data contracts on exactly what will be trained on."""
    recs = list(_read_jsonl(out / f"instances_{reg}.jsonl"))
    assert [r["id"] for r in recs] == [inst_id(reg, i) for i in range(n)], "instances file is not slots 0..n-1"
    insts = {r["id"]: M.instance_from_record(r) for r in recs}
    rinv = {k: v["renaming"].inverse() for k, v in insts.items() if reg == "A"}
    lens, hashes, nobj, nobj_letter, rej = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in recs:
        it = insts[r["id"]]
        lens[len(it["plan"])] += 1
        hashes[r["schema_hash"]] += 1
        nobj[len(it["objs"])] += 1
        nobj_letter[len(it["objs"])] += it["letter_objects"]
        rej.update(r["rejections"])
        assert it["letter_objects"] == M.letter_objects(int(r["id"][1:]))
        assert it["letter_objects"] == all(len(o) == 1 for o in it["objs"])
    share = sum(nobj_letter.values()) / n
    if n >= 2:
        assert 0.3 <= share <= 0.5, share
    toks = {k: [] for k in ("proposer_prompt", "target_actions", "target_states", "proposer_prompt+states_target+eos",
                            "forward_prompt", "forward_target")}
    keys, by_inst, outcomes, n_sig = set(), {}, Counter(), 0
    last = -1
    for row in _read_jsonl(out / f"proposer_{reg}.jsonl"):
        it, iid = insts[row["id"]], row["id"]
        assert int(iid[1:]) >= last, "rows out of slot order"
        last = int(iid[1:])
        dom, objs = it["domain"], set(it["objs"])
        # token budget, recounted
        p, ta, ts = ctx.count(row["prompt"]), ctx.count(row["target_actions"]), ctx.count(row["target_states"])
        assert row["n_tokens"] == {"actions": p + ta + 1, "states": p + ts + 1}
        assert max(row["n_tokens"].values()) <= ctx.max_tokens
        for k, v in zip(toks, (p, ta, ts, p + ts + 1)):
            toks[k].append(v)
        # no duplicate rows anywhere
        key = sha256((row["prompt"] + "\0" + row["target_actions"]).encode())
        assert key not in keys, iid
        keys.add(key)
        outcomes[row["outcome"]] += 1
        by_inst.setdefault(iid, []).append(row)
        # targets round-trip to the executor's trajectory from the prompt's state
        u = M.prompt_user(row["prompt"])
        s, goal = M.state_from_list(u["state"], dom, objs), M.state_from_list(u["goal"], dom, objs)
        assert s is not None and goal == it["goal"] and u["outcome"] == row["outcome"]
        acts, _ = M.parse_plan(row["target_actions"], dom, objs, "actions")
        acts2, sts = M.parse_plan(row["target_states"], dom, objs, "states")
        assert acts == acts2
        if row["t"] >= 0:
            assert acts == list(it["plan"][row["t"]:])
        for x, ps in zip(acts, sts):
            s, _ = M.step(dom, s, x)
            assert s == ps
        assert (row["outcome"] == "success") == M.execute(dom, M.state_from_list(u["state"], dom, objs), goal,
                                                          acts)["success"]
        # suffix-signature guard, recomputed from the emitted prompt: map (state, goal) back to Blocksworld space
        if reg == "A":
            R, bw = rinv[iid], it["bw"]
            sig = B.signature(bw["objs"], R.facts(M.state_from_list(u["state"], dom, objs)), R.facts(goal))
            assert sig == row["sig"] and sig not in ctx.test_sigs, iid
            n_sig += 1
    assert set(by_inst) == set(insts)
    for iid, rr in by_inst.items():
        assert [r["t"] for r in rr][-1] == -1 and sum(r["t"] == -1 for r in rr) == 1
        assert rr[-1]["target_actions"] != M.target_actions(insts[iid]["plan"])  # noisy row resampled
        assert len(rr) - 1 + insts[iid]["suffix_rows_dropped"] == len(insts[iid]["plan"])
    legal, fkeys = Counter(), set()
    for row in _read_jsonl(out / f"forward_{reg}.jsonl"):
        it = insts[row["id"]]
        dom, objs = it["domain"], set(it["objs"])
        p, tt = ctx.count(row["prompt"]), ctx.count(row["target"])
        assert row["n_tokens"] == p + tt + 1 <= ctx.max_tokens
        toks["forward_prompt"].append(p)
        toks["forward_target"].append(tt)
        assert row["prompt"] not in fkeys
        fkeys.add(row["prompt"])
        u = M.prompt_user(row["prompt"])
        s, a = M.state_from_list(u["state"], dom, objs), M.parse_action(u["action"], dom, objs)
        assert M.parse_forward(row["target"], dom, objs) == M.step(dom, s, a) and row["legal"] == M.step(dom, s, a)[1]
        legal[str(row["legal"])] += 1
    n_val = 0
    if B.VAL_EXE.exists():
        for r in recs[:ctx.val_checks]:
            noisy = M.parse_plan(by_inst[r["id"]][-1]["target_actions"], insts[r["id"]]["domain"],
                                 set(insts[r["id"]]["objs"]))[0]
            n_val += val_check_generated(insts[r["id"]], [noisy], out / "_val_tmp")
        shutil.rmtree(out / "_val_tmp", ignore_errors=True)
    return {"rejections": dict(sorted(rej.items())), "plan_len_hist": dict(sorted(lens.items())),
            "n_objects": dict(sorted(nobj.items())), "n_objects_single_letter": dict(sorted(nobj_letter.items())),
            "single_letter_object_share": round(share, 4), "distinct_schema_hashes": len(hashes),
            "proposer_rows": len(keys), "forward_rows": len(fkeys), "proposer_outcomes": dict(outcomes),
            "forward_legal": dict(legal), "suffix_rows_dropped_test_signature": sum(r["suffix_rows_dropped"]
                                                                                    for r in recs),
            "rows_signature_rechecked": n_sig, "val_generated_domain_plans": n_val,
            "tokens_titan": {k: tok_stats(v) for k, v in toks.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="instances per regime (total; --resume continues up to it)")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--max-states", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="token budget per row (prompt + target + EOS, Titan tokenizer), enforced at generation")
    ap.add_argument("--val-checks", type=int, default=10)
    ap.add_argument("--tokenizer", default=str(TITAN))
    ap.add_argument("--out", default="runs/_mystery_smoke")
    ap.add_argument("--regimes", default="AB")
    ap.add_argument("--resume", action="store_true", help="continue the outputs in --out (same CONFIG required)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing outputs in --out first")
    a = ap.parse_args(argv)
    regimes = [r for r in REGIMES if r in a.regimes]
    assert regimes and set(a.regimes) <= set(REGIMES), a.regimes
    out = ROOT / a.out
    replaced = prepare_out(out, a.resume, a.overwrite)
    rng = random.Random(a.seed)  # self-checks only; data slots use M.instance_seed
    T0 = time.time()
    summary = {"args": vars(a), "checks": {}}
    chk = summary["checks"]

    # ------------------------------------------------------------------ tokenizer (required: budget + name filter)
    tok = load_tokenizer(a.tokenizer)
    count, single_token = token_fns(tok)
    assert single_token("spoon") and single_token("hound") and not single_token("droshdoiv")

    # ------------------------------------------------------------------ vocabulary
    t = time.time()
    forb = M.forbidden_words(cache=out / "forbidden_words.json")
    myst, bw = M.mystery_domain(), M.bw_domain()
    deceptive = sorted(set(myst.preds) | set(myst.actions))
    assert set(deceptive) <= forb and set(bw.preds) | set(bw.actions) <= forb | {"pick-up", "put-down"}
    banned = M.banned_substrings()
    assert set(deceptive) <= set(banned) and {"hand", "hold", "table", "stack", "clear", "pick"} <= set(banned)
    ng = M.NameGen(random.Random(0), forb, single_token=single_token)
    assert not ng.ok("zhand") and not ng.ok("spainor") and not ng.ok("spoon") and ng.ok("droshdoiv")
    summary["vocab"] = {"n_forbidden": len(forb), "n_source_files": len(M.vocab_sources()),
                        "deceptive_vocab": deceptive, "banned_substrings": list(banned),
                        "single_token_filter": "reject names that are one Titan token bare or after a space",
                        "randomized_vocab_files": [str(p) for p in M.randomized_vocab_files()],
                        "seconds": round(time.time() - t, 1)}
    print("forbidden words", len(forb), "from", summary["vocab"]["n_source_files"], "files;",
          "randomized-vocab files:", len(summary["vocab"]["randomized_vocab_files"]), flush=True)

    # ------------------------------------------------------------------ isomorphism machinery
    fam = M.bw_family_hashes()
    assert M.bw_hash() in fam and M.is_blocksworld(bw)
    assert M.is_blocksworld(myst) and M.domain_rejection(myst) == "blocksworld_family"
    R_m2b, R_b2m = M.mystery_to_bw(), M.find_isomorphism(bw, myst)
    assert R_b2m is not None and R_m2b.domain(myst).same_as(bw) and R_b2m.domain(bw).same_as(myst)
    # positive controls: renamed + flipped + param-permuted Blocksworld; projection (drop handempty)
    ctrl = M.Renaming({p: f"q{i}" for i, p in enumerate(sorted(bw.preds))},
                      {x: f"r{i}" for i, x in enumerate(sorted(bw.actions))}, {}, frozenset({"on"}),
                      {"stack": (1, 0), "unstack": (1, 0)}).domain(bw)
    proj = M._project(bw, set(bw.preds) - {"handempty"}, sorted(bw.actions))
    assert M.is_blocksworld(ctrl) and M.domain_rejection(ctrl) == "blocksworld_family"
    assert M.is_blocksworld_family(proj) and not M.is_blocksworld(proj)
    # containment controls: a 3-predicate x 3-action Blocksworld fragment, renamed and flipped, plus an extra
    # predicate (also added to one action) and an extra action, is not in the family but contains the fragment
    frag = M._project(bw, {"clear", "holding", "on"}, ("put-down", "stack", "unstack"))
    fr = M.Renaming({"clear": "qa", "holding": "qb", "on": "qc"}, {"put-down": "ra", "stack": "rb", "unstack": "rc"},
                    {}, frozenset({"on"}), {"stack": (1, 0)}).domain(frag)
    sc = fr.actions["ra"]
    acts = dict(fr.actions, ra=M.Schema(sc.params, tuple(sorted(sc.pre + (("qd", "X"),))), sc.add, sc.dele),
                rd=M.Schema(("X",), (("qd", "X"),), (("qa", "X"),), (("qd", "X"),)))
    cont = M.Domain("ctrl-contains", dict(fr.preds, qd=1), acts)
    assert not M.is_blocksworld_family(cont) and M.contains_blocksworld_fragment(cont)
    assert M.domain_rejection(cont) == "blocksworld_fragment" and M.contains_blocksworld_fragment(myst)
    chk["isomorphism"] = {"bw_family_size": len(fam), "blocksworld_detected": True, "mystery_detected": True,
                          "renamed_flipped_bw_detected": True, "projection_detected": True,
                          "bw_fragments_3x3": sum(len(v) for v in M.bw_fragment_hashes().values()),
                          "contained_fragment_detected": True,
                          "mystery_to_bw": {"pred": R_m2b.pred, "act": R_m2b.act}}
    print("isomorphism: BW family", len(fam), "hashes; BW / Mystery / renamed BW / projection / contained 3x3",
          "fragment detected", flush=True)

    # ------------------------------------------------------------------ executor vs blocks.py
    t = time.time()
    n_cmp = 0
    for _ in range(300):
        objs = sorted(rng.sample(B.NAMES, rng.randint(2, 5)))
        s = B.towers_to_state(B.random_towers(objs, rng))
        states = [s]
        for _ in range(10):
            s = B.step(s, rng.choice([x for x in B.ground_actions(objs) if B.applicable(s, x)]))[0]
            states.append(s)
        allf = [("handempty",)] + [(p, x) for p in ("clear", "ontable", "holding") for x in objs] + \
               [("on", x, y) for x in objs for y in objs if x != y]
        states.append(frozenset(f for f in allf if rng.random() < 0.4))  # arbitrary (unreachable) states too
        acts = B.ground_actions(objs) + [(op, x, x) for op in ("stack", "unstack") for x in objs]
        for st in states:
            for act in acts:
                ref = B.step(st, act)
                assert M.applicable(bw, st, act) == B.applicable(st, act) and M.step(bw, st, act) == ref, (st, act)
                ns, ok = M.step(myst, R_b2m.facts(st), R_b2m.action(act))
                assert (ns, ok) == (R_b2m.facts(ref[0]), ref[1])
                n_cmp += 1
    n_bfs = 0
    while n_bfs < 60:
        g = B.generate(rng, rng.choice([3, 4, 5]))
        if g is None:
            continue
        objs, init, goal = g
        p1, p2 = B.bfs_plan(init, goal, objs), M.bfs_plan(bw, init, goal, objs)
        assert len(p1) == len(p2) and M.execute(bw, init, goal, p1)["success"] and B.execute(init, goal, p2)["success"]
        n_bfs += 1
    chk["executor_vs_blocks"] = {"transitions_compared": n_cmp, "bfs_lengths_compared": n_bfs, "agree": True,
                                 "also_through_mystery_renaming": True, "seconds": round(time.time() - t, 1)}
    print("executor == blocks.py on", n_cmp, "transitions (+ Mystery renaming) and", n_bfs, "BFS plans", flush=True)

    # ------------------------------------------------------------------ PlanBench Mystery: renderer + VAL
    minst = M.mystery_instances()
    for it in minst[:20]:
        bw_init, bw_goal = R_m2b.facts(it["init"]), R_m2b.facts(it["goal"])
        assert R_b2m.facts(bw_init) == it["init"]
        again = M.proposer_prompt(R_b2m.domain(bw, "regime-a"), it["objs"], R_b2m.facts(bw_init),
                                  R_b2m.facts(bw_goal))
        assert again == M.mystery_prompt(it)
    chk["format_identity"] = {"mystery_instances": len(minst), "regimeA_pipeline_reproduces_mystery_prompt": 20}
    n_val = 0
    if B.VAL_EXE.exists():
        for it in minst[:a.val_checks]:
            plan = M.bfs_plan(myst, it["init"], it["goal"], it["objs"])
            assert plan and M.execute(myst, it["init"], it["goal"], plan)["success"]
            assert M.val_check_mystery(it["path"], plan)
            bad = plan[:-1] if len(plan) > 1 else plan + plan
            assert M.val_check_mystery(it["path"], bad) == M.execute(myst, it["init"], it["goal"], bad)["success"]
            swapped = [plan[1], plan[0]] + plan[2:] if len(plan) > 1 else plan
            assert M.val_check_mystery(it["path"], swapped) == \
                M.execute(myst, it["init"], it["goal"], swapped)["success"]
            n_val += 1
        # write_domain reproduces the Mystery domain for VAL (same verdicts as the vendored PDDL)
        wd = out / "_val_tmp"
        wd.mkdir(exist_ok=True)
        (wd / "mystery_written.pddl").write_text(M.write_domain(myst))
        assert M.parse_domain(wd / "mystery_written.pddl").same_as(myst)
        for it in minst[:min(3, a.val_checks)]:
            plan = M.bfs_plan(myst, it["init"], it["goal"], it["objs"])
            assert B.val_check(Path(it["path"]), plan, wd / "mystery_written.pddl")
            assert not B.val_check(Path(it["path"]), plan[:-1], wd / "mystery_written.pddl")
        shutil.rmtree(wd, ignore_errors=True)
    chk["executor_vs_VAL_mystery"] = {"instances": n_val, "plans_per_instance": 3, "agree": n_val > 0}
    print("executor == VAL on", n_val, "Mystery instances x 3 plans", flush=True)

    # ------------------------------------------------------------------ PlanBench Mystery eval set (by signature)
    ev = M.mystery_eval_set()
    assert len(ev) == MYSTERY_EVAL_INSTANCES and len({it["sig"] for it in ev}) == MYSTERY_EVAL_DISTINCT
    by_nb = {}
    for it in ev:
        by_nb.setdefault(it["n_blocks"], []).append(it["sig"])
    rep = M.eval_report(ev, [1] * len(ev), n_boot=0)
    assert rep["all"]["per_instance"]["acc"] == 1.0 and rep["all"]["dedup_signature"]["n_distinct"] == 262
    summary["mystery_eval_set"] = {"instances": len(ev), "distinct_signatures": len({it["sig"] for it in ev}),
                                   "by_blocks": {nb: {"instances": len(v), "distinct": len(set(v))}
                                                 for nb, v in sorted(by_nb.items())},
                                   "report": "M.eval_report: accuracy per instance and de-duplicated by signature, "
                                             "overall and by block count, cluster-bootstrap CIs"}
    print("Mystery eval set:", len(ev), "instances =", MYSTERY_EVAL_DISTINCT, "distinct problems;",
          summary["mystery_eval_set"]["by_blocks"], flush=True)

    # ------------------------------------------------------------------ test signatures
    t = time.time()
    test_sigs = M.test_signatures()
    hard = M.hard_test_signatures()
    n_hard = sum(len(M.parse_problem(p)[0]) <= 7 for p in M.HARD_DIR.glob("*.pddl"))
    assert hard and len(hard) == n_hard and hard <= test_sigs and B.test_signatures() <= test_sigs
    assert {it["sig"] for it in ev} <= test_sigs
    summary["test_signatures_excluded"] = len(test_sigs)
    summary["test_signatures"] = {"blocksworld_basic_basic3_unsolvable": len(B.test_signatures()),
                                  "blocksworld_hard_le7_blocks": len(hard),
                                  "mystery": len(M.mystery_test_signatures()), "union": len(test_sigs)}
    print("test signatures (BW + Hard<=7 + Mystery):", len(test_sigs), f"({time.time() - t:.0f}s)", flush=True)

    # ------------------------------------------------------------------ config (resume must match)
    code = {f: sha256((ROOT / f).read_bytes()) for f in ("bwmn/mystery.py", "bwmn/blocks.py",
                                                         "scripts/mystery_data.py")}
    cfg = {"seed": a.seed, "max_states": a.max_states, "max_tokens": a.max_tokens, "n_forward": N_FORWARD,
           "tokenizer_sha256": sha256((Path(a.tokenizer) / "tokenizer.json").read_bytes()),
           "forbidden_sha256": sha256("\n".join(sorted(forb)).encode()),
           "test_signatures_sha256": sha256("\n".join(sorted(test_sigs)).encode()), "code_sha256": code}
    summary["output"] = {"mode": check_config(out, cfg, a.resume), "replaced_files": replaced}
    ctx = SimpleNamespace(seed=a.seed, forb=forb, test_sigs=test_sigs, deceptive=deceptive, banned=banned,
                          single_token=single_token, count=count, max_tokens=a.max_tokens, max_states=a.max_states,
                          val_checks=a.val_checks)

    # ------------------------------------------------------------------ generation (streaming, per-slot seeds)
    for reg in regimes:
        summary[f"regime_{reg}"] = {"generation": generate_regime(out, reg, a.n, ctx)}
        print(f"generated {reg}: {summary[f'regime_{reg}']['generation']}", flush=True)
    chk["bfs_solvable_and_optimal"] = True
    chk["no_vocab_collisions"] = {"checked_instances": len(regimes) * a.n, "collisions": 0}
    chk["regime_B_blocksworld_family"] = 0
    chk["regime_B_contains_blocksworld_fragment"] = 0
    print("per-instance checks passed (BFS optimality, vocabulary, isomorphism, guard, budget)", flush=True)

    # ------------------------------------------------------------------ re-read from disk: data contracts
    for reg in regimes:
        summary[f"regime_{reg}"].update(verify_regime(out, reg, a.n, ctx))
        s = summary[f"regime_{reg}"]
        print(f"verified {reg} on disk: {s['proposer_rows']} proposer / {s['forward_rows']} forward rows, "
              f"suffix rows dropped {s['suffix_rows_dropped_test_signature']}, single-letter share "
              f"{s['single_letter_object_share']}, VAL plans {s['val_generated_domain_plans']}", flush=True)
    chk["targets_roundtrip"] = True
    chk["suffix_signature_guard"] = {reg: {"rows_rechecked_from_prompt": summary[f"regime_{reg}"]
                                           ["rows_signature_rechecked"], "in_test_set": 0} for reg in regimes}
    chk["token_budget"] = {"max_tokens": a.max_tokens, "rows_over_budget": 0,
                           "instances_redrawn": {reg: summary[f"regime_{reg}"]["rejections"].get("token_budget", 0)
                                                 for reg in regimes}}
    chk["no_duplicate_rows"] = True
    chk["executor_vs_VAL_generated_domains"] = {reg: summary[f"regime_{reg}"]["val_generated_domain_plans"]
                                                for reg in regimes}
    with open(out / "EVAL_ONLY_mystery_prompt_examples.jsonl", "w", encoding="utf-8") as fh:
        for it in minst[:3]:
            fh.write(json.dumps({"id": it["id"], "prompt": M.mystery_prompt(it)}) + "\n")

    # ------------------------------------------------------------------ token lengths (Titan tokenizer, CPU)
    ev_lens = [count(M.mystery_prompt(it)) for it in minst]
    assert max(ev_lens) < a.max_tokens
    summary["tokens_titan"] = {reg: summary[f"regime_{reg}"].pop("tokens_titan") for reg in regimes}
    summary["tokens_titan"]["mystery_eval"] = {"proposer_prompt": tok_stats(ev_lens)}
    for k, v in summary["tokens_titan"].items():
        print("tokens", k, {kk: (vv["median"], vv["max"]) for kk, vv in v.items()})

    summary["seconds_total"] = round(time.time() - T0, 1)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print("ALL CHECKS PASSED", f"({summary['seconds_total']}s) ->", out)
    return summary


if __name__ == "__main__":
    main()
