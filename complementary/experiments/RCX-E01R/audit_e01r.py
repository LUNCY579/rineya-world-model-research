"""RCX-E01R training-row audit (PREREG_E01R section 4, "Training-row audit"; covariates of section 3; review item 5).

Mirrors the hard checks of experiments/RCX-E01N/native_e01.audit_native for the E01R treatment rows (R1, R2, R3) and
reports per-arm covariates. Pure Python: native_e01, native_probe and lora_cpu import torch, so their pure helpers are
executed from their own source (_borrow); nothing loads a model, and a tokenizer only with the CLI's --tok. P3 is never
read here (R1's P3 check is audit_tr_pool.py).

    python -B experiments/RCX-E01R/audit_e01r.py --out runs/RCX-E01R/cells --arms E-R2,E-R3 --tok      (branch B4)
    (B3 only: --arms E-R1,E-R3 [--pool experiments/RCX-E01R/pools/TR_v1_audited.json]; E-R1 refuses unless
     runs/RCX-E01R/dev2/BRANCH_STEP_A.json (--branch) comes from the frozen decide_branch.py and says B3)
Reads <out>/train_<arm>.jsonl and <out>/TRAIN_MANIFEST_E01R.json (native_e01r.py build; arm names as there, R1/R2/R3
accepted as aliases), the R1 pool with its <tag>_AUDIT.json (audit_tr_pool.py), and the control rows
runs/RCX-E01N/cells/train_N-CFSD.jsonl (sha256 checked against that directory's TRAIN_MANIFEST.json). Writes
<out>/AUDIT_E01R.json {"ok", "arm_ok": {arm: bool}, "rows_sha256": {arm: sha256}, ...}, which native_e01r.require_audit
(and require_audit_e01r here) checks before training and the mint; arm_ok[arm] is true iff no failure names that arm
and no failure is arm-independent (so an arm dropped before the mint, PREREG section 2, does not block the other).
AUDIT_E01R.json is CREATE-ONLY: the audit refuses if it exists (a failed record is kept; re-auditing before the mint
means removing it, a logged deviation) and refuses outright once <out>/SEALED_COMMIT.json exists (the mint binds the
audit's sha256; train and the SEALED opening check it).

E01R reader (review item 5). world._SINGLE is compiled from SPOS/SNEG at import, so templates registered later
(recipes.register_pool: 'tr:<id>') are invisible to world.parse_rule and to native_e01.read_back. make_reader() compiles
the single-atom templates of templates.SPOS/SNEG at call time (after register_pool) with world._tpl_regex and
world._LIST_RE, adds world's composition and solo frames, and requires exactly one reading (template and AST). The
held-out fragments come from the original t*/h* templates only, so a registered wording can neither add nor hide one. As
in audit_native, a template id whose last component starts with h reads as held out, which is why pool ids starting
with h are refused (review item 14).

HARD checks per treatment arm (any failure -> ok False; the control is audited the same way, but its failures are
reported separately and do not gate):
  rows                exactly n_rows (1,501) rows; in the CLI, the manifest's sha256 and row count
  frame               Titan's frame byte-exact (native_e01.render_literal), the user turn canonical with event keys
                      exactly world.EVENT_KEYS, and the rule sentence read back uniquely and re-rendered byte-exact
  gold                the target re-derived from the prompt by rcx.rcl.world: native arms target_json(after); trace
                      arms (R3) exactly two lines with line 2 == target_json(after); the decision and status fields
  trace_line1         trace arms: line 1 == trace_target.check_line(rule, state, event), parsing back to the gold
                      decision and coherent
  readback_hash       the recorded semantic_hash equals the read-back one
  split_words, probe_words, excluded_words, non_train_words, rcwm_heldout_text, heldout_template, heldout_rule_factor,
  heldout_magnitude   as audit_native; split_words also catches DEV/SEALED words in the rule sentence in any case
  pool_wording        a single-atom template outside the original t*/h*: from a test pool (p2:/p3: ids), or failing
                      audit_tr_pool's visible checks (status word or stem, RCX near-duplicate, P2_v2 near-duplicate
                      under the stricter rule Jaccard >= 0.35 or a shared 4-word sequence, P2 v1 under the 0.5/5 rule)
  overlap             zero semantic and zero skeleton overlap with every battery given; by default E01N DEV (seed
                      20261240, all surfaces), DEV2 (seed 20261261; its p2 copies share the in hashes, and the CLI adds
                      the built battery.jsonl) and the opened E01N SEALED (seed 88527345, in + ho_tpl)
  control_match       'prompts' (R3): prompts byte-identical to the control's, in order, and line 2 the control's
                      target; 'wording' (R1): every row has the control's user turn and target, and every row not
                      read as a pool wording the control's prompt
  over_max_len        (with a tokenizer) a row longer than max_len 512 tokens (lora_cpu.train_sft would refuse it)
DESCRIPTIVE per arm and the control: rows by rule type (S+, S-, composition), parents, no-op share, class balance by
rule type, wording source (single:t*, single:tr, frame:t*, ...), template and account count, pooled rows per cell,
target bytes, and (with a tokenizer) supervised/processed tokens per training seed as lora_cpu.train_sft counts them.
"""
import argparse
import ast as pyast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_tr_pool as ATP  # noqa: E402
import trace_target as TT  # noqa: E402
from rcx.rcl import gen, templates as T, vocab as V, wording_audit as WA, world as W  # noqa: E402

E01N = ROOT / "experiments" / "RCX-E01N"
LORA_CPU = ROOT / "rcx" / "native" / "lora_cpu.py"
RCWM_T1 = ROOT.parent / "rcwm" / "experiments" / "RCWM-T1"  # read-only reference tree (never written)
POOLS = HERE / "pools"
P2_FILES = (POOLS / "P2_DEV_v2.json", POOLS / "P2_DEV.json")  # (P2_v2, P2 v1)
DEV2_DIR = ROOT / "runs" / "RCX-E01R" / "dev2"
CONTROL_DIR = ROOT / "runs" / "RCX-E01N" / "cells"
N_ROWS, STEPS, BATCH, MAX_LEN = 1501, 190, 16, 512
E01N_SEALED_SEED = 88527345  # opened (CX D-30); 2,240 records over in + ho_tpl
ARM_SPEC = {"E-R1": ("native", "wording"), "E-R2": ("native", None), "E-R3": ("trace", "prompts")}  # = native_e01r
R1_POOL = POOLS / "TR_v1_audited.json"
BRANCH_PATH = DEV2_DIR / "BRANCH_STEP_A.json"
DECIDE_SHA256 = "6835b0402120a2847423d8af08814cc89f311292b110365487b39d61636080d6"  # decide_branch.py, CX D-37
HARD_FLAGS = ("frame", "gold", "trace_line1", "readback_hash", "split_words", "probe_words", "excluded_words",
              "non_train_words", "rcwm_heldout_text", "heldout_template", "heldout_rule_factor", "heldout_magnitude",
              "pool_wording", "over_max_len")
STD_BAL, STD_AMT_ABS = (0, 98), 12  # = native_e01 (gen._state / gen._event "std" ranges)
VERBATIM_CLAUSE = "If status is not posted, leave all balances unchanged."
ORIG_ID = re.compile(r"[th][0-9]+")
SOLO_FRAME = "An event takes effect only if {C}; otherwise nothing changes."  # = world._SOLO / templates.render_rule
_PROMPT_RE = re.compile(r"<\|im_start\|>system\n(.*)<\|im_end\|>\n<\|im_start\|>user\n(.*)<\|im_end\|>\n"
                        r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n", re.S)  # = native_e01._PROMPT_RE
_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _borrow(path, names, **glb):
    """The named top-level functions of `path`, executed from its own source with the given globals (as
    trace_target._borrow); the scripts themselves import torch and are never imported."""
    tree = pyast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    defs = [n for n in tree.body if isinstance(n, pyast.FunctionDef) and n.name in names]
    if sorted(d.name for d in defs) != sorted(names):
        raise ImportError(f"{Path(path).name}: expected exactly one top-level def of each of {names}")
    ns = {"json": json, "re": re, "Path": Path, "__name__": Path(path).stem, **glb}
    exec(compile(pyast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return [ns[n] for n in names]


Encoded = namedtuple("Encoded", "input_ids labels n_target")  # lora_cpu.Encoded's fields
rcl_system, user_json = _borrow(E01N / "native_probe.py", ("rcl_system", "user_json"))
render_literal, _ast_constants, _balance = _borrow(E01N / "native_e01.py", ("render_literal", "_ast_constants",
                                                                              "_balance"))
(rule_type,) = _borrow(E01N / "native_e01.py", ("rule_type",), W=W, gen=gen)
(rcwm_heldout_texts,) = _borrow(E01N / "native_e01.py", ("rcwm_heldout_texts",), _ast_constants=_ast_constants,
                                RCWM_T1=RCWM_T1)
encode_pair, row_order_for = _borrow(LORA_CPU, ("encode_pair", "row_order_for"), Encoded=Encoded, random=random)
NP_CONST = _ast_constants(E01N / "native_probe.py", {"VERBATIM_SYSTEM", "BATTERY_SEED"})
DEV2_CONST = _ast_constants(HERE / "run_e01r.py", {"DEV2_SEED", "N_PARENTS", "STRATA"})
FRAME_PRE, FRAME_POST = rcl_system("\x00").split("\x00")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- reader

def make_reader(spos=None, sneg=None):
    """Rule-sentence reader over the single-atom templates in templates.SPOS/SNEG NOW (after any register_pool) or the
    given dicts, plus world's composition and solo frames. read(sentence) -> (ast, tid, rendered) with tid as in
    world.parse_rule ('S+:t0', 'S-:tr:r012', 'AND:t1', 'SOLO') and `rendered` the reading re-rendered from its
    template. Raises ValueError unless exactly one (template, AST) reading exists."""
    singles = [(op, tid, tpl, W._tpl_regex(tpl, {"L": W._LIST_RE}))
               for op, pool in (("S+", T.SPOS if spos is None else spos), ("S-", T.SNEG if sneg is None else sneg))
               for tid, tpl in sorted(pool.items())]
    frames = {"AND": T.AND_FRAMES, "OR": T.OR_FRAMES}

    def read(sentence):
        found = []
        for op, tid, tpl, rx in singles:
            m = rx.match(sentence)
            if m:
                try:
                    ast = {"op": op, "set": W._split_list(m.group("L"))}
                except ValueError:
                    continue
                found.append((ast, f"{op}:{tid}", tpl.format(L=T.render_list(ast["set"]))))
        for op, fid, rx in W._FRAMES:
            m = rx.match(sentence)
            if m:
                try:
                    ast = {"op": op, "args": [W._parse_clause(m.group("C1")), W._parse_clause(m.group("C2"))]}
                except ValueError:
                    continue
                found.append((ast, f"{op}:{fid}", frames[op][fid].format(C1=T.clause(ast["args"][0]),
                                                                        C2=T.clause(ast["args"][1]))))
        m = W._SOLO.match(sentence)
        if m:
            try:
                ast = W._parse_clause(m.group("C"))
                found.append((ast, "SOLO", SOLO_FRAME.format(C=T.clause(ast))))
            except ValueError:
                pass
        if len({(json.dumps(a, sort_keys=True), tid) for a, tid, _ in found}) != 1:
            raise ValueError(f"rule sentence not uniquely readable ({len(found)} readings)")
        return found[0]

    read.templates = {f"{op}:{tid}": tpl for op, tid, tpl, _ in singles}
    return read


def read_back(row, reader):
    """Decode a training row from its prompt TEXT alone (native_e01.read_back with the E01R reader)."""
    m = _PROMPT_RE.fullmatch(row["prompt"])
    if not m:
        raise ValueError("not a system+user Qwen3 chat prompt")
    system, user = m.group(1), m.group(2)
    if len(system) < len(FRAME_PRE) + len(FRAME_POST) or not (system.startswith(FRAME_PRE)
                                                              and system.endswith(FRAME_POST)):
        raise ValueError("system prompt is not the native ledger frame")
    sentence = system[len(FRAME_PRE):len(system) - len(FRAME_POST)]
    rule, tid, rendered = reader(sentence)
    u = json.loads(user)
    if set(u) != {"before", "event"} or u["event"].get("tick") != 1:
        raise ValueError("user turn is not {before, event(tick=1)}")
    state, event = u["before"], {k: v for k, v in u["event"].items() if k != "tick"}
    if sorted(event) != sorted(W.EVENT_KEYS) or not all(isinstance(v, int) for v in state.values()):
        raise ValueError("bad event keys or state values")
    ok, after = W.transition(rule, state, event)
    exact = (user == json.dumps(u, separators=(",", ":"), sort_keys=True) and rendered == sentence
             and row["prompt"] == render_literal(rcl_system(sentence), user))
    return {"rule": rule, "tid": tid, "sentence": sentence, "state": state, "event": event, "ok": ok, "after": after,
            "frame_exact": exact, "semantic_hash": W.semantic_hash(rule, state, event),
            "skeleton_hash": W.skeleton_hash(state, event)}


def wording_source(tid):
    """'solo', or '<single|frame>:<family>' with family t* / h* (original templates) or a registered prefix (tr, p2)."""
    if tid == "SOLO":
        return "solo"
    op, raw = tid.split(":", 1)
    fam = raw[0] + "*" if ORIG_ID.fullmatch(raw) else (raw.split(":")[0] if ":" in raw else "other")
    return f"{'single' if op in W.STATUS_OPS else 'frame'}:{fam}"


def heldout_fragments(min_len=12):
    """native_e01.heldout_template_fragments over the ORIGINAL t*/h* templates only: fixed text of the held-out rule
    templates and frames that no t* template contains."""
    pools = [{k: v for k, v in p.items() if ORIG_ID.fullmatch(k)} for p in (T.SPOS, T.SNEG, T.AND_FRAMES,
                                                                              T.OR_FRAMES)]
    train = [t for p in pools for k, t in p.items() if k.startswith("t")]
    frags = {f for p in pools for k, t in p.items() if k.startswith("h") for f in re.split(r"\{[A-Z0-9]+\}", t)}
    return sorted(f for f in frags if len(f.strip()) >= min_len and not any(f in x for x in train))


# ----------------------------------------------------------------------------- references

def battery_specs():
    """Hard-overlap batteries: name -> (split, n_parents, seed, strata, surfaces)."""
    return {"E01N_DEV": ("dev", 40, NP_CONST["BATTERY_SEED"], gen.STRATA, gen.SURFACES),
            "DEV2": ("dev", DEV2_CONST["N_PARENTS"], DEV2_CONST["DEV2_SEED"], tuple(DEV2_CONST["STRATA"]), ("in",)),
            "E01N_SEALED": ("sealed", 40, E01N_SEALED_SEED, gen.STRATA, ("in", "ho_tpl"))}


def build_batteries(specs=None):
    gen.use_world("arith")
    return {k: gen.make_battery(sp, n, seed, strata=strata, surfaces=sf)
            for k, (sp, n, seed, strata, sf) in (battery_specs() if specs is None else specs).items()}


def record_hashes(recs):
    sem = {r["semantic_hash"] for r in recs} | {W.semantic_hash(r["rule_ast"], r["state"], r["event"]) for r in recs}
    skel = {r["skeleton_hash"] for r in recs} | {W.skeleton_hash(r["state"], r["event"]) for r in recs}
    return sem, skel


def context(p2_files=P2_FILES, rcwm_texts=None, rcwm_root=RCWM_T1):
    """Word sets, fragments, P2 texts and RCWM held-out texts shared by every arm; returns (ctx, failures)."""
    fail = []
    dev_sealed = set(V.REAL_DEV) | set(V.NONCE["dev"]) | set(V.REAL_SEALED) | set(V.NONCE["sealed"])
    p2 = [[t["text"] for t in json.loads(Path(p).read_text(encoding="utf-8"))["templates"]] for p in p2_files]
    if rcwm_texts is None:
        env_p = Path(rcwm_root) / "rcwmt" / "env.py"
        if env_p.exists() and (Path(rcwm_root) / "scripts").exists():
            if not _ast_constants(env_p, {"SYSTEM_ONE"})["SYSTEM_ONE"]["ledger_replay"] == NP_CONST["VERBATIM_SYSTEM"] \
                    == rcl_system(VERBATIM_CLAUSE):
                fail.append("frame_not_titan_verbatim")
            rcwm_texts = rcwm_heldout_texts(rcwm_root)
        else:
            fail.append("rcwm_reference_missing")
            rcwm_texts = {}
    ctx = {"dev_sealed": dev_sealed, "dev_sealed_lower": {w.lower() for w in dev_sealed},
           "probes": set(V.NEARMISS_VOID) | set(V.POSTEDLIKE),
           "excl_tok": {w for w in V.EXCLUDED if " " not in w}, "excl_phrase": [w for w in V.EXCLUDED if " " in w],
           "train_pool": set(V.REAL_TRAIN) | set(V.NONCE["train"]), "anchors": {gen.POSTED, gen.VOID},
           "frags": heldout_fragments(), "rcwm_texts": rcwm_texts, "p2_v2": p2[0],
           "p2_v1": p2[1] if len(p2) > 1 else []}
    return ctx, fail


def pool_wording_reason(tid, text, ctx):
    """None for original t*/h* templates and frames; else the reason a registered single-atom template fails."""
    src = wording_source(tid)
    if not src.startswith("single:") or src in ("single:t*", "single:h*"):
        return None
    if src in ("single:p2", "single:p3"):
        return "test_pool"
    return ATP.visible_drops([{"id": tid, "text": text}], ctx["p2_v2"], ctx["p2_v1"]).get(tid)


# ----------------------------------------------------------------------------- audit

def token_covariates(rows, tok, steps=STEPS, batch=BATCH, seeds=(0, 1, 2), max_len=MAX_LEN):
    """native_e01.token_covariates (lora_cpu.encode_pair counts, row_order_for orders) plus rows over max_len."""
    enc = [encode_pair(tok, r["prompt"], r["target"], 10 ** 9) for r in rows]
    sup, proc = [e.n_target for e in enc], [len(e.input_ids) for e in enc]
    per_seed = {}
    for s in seeds:
        order = row_order_for(len(rows), steps, batch, s)[:steps * batch]
        per_seed[str(s)] = {"supervised_tokens": sum(sup[i] for i in order),
                            "tokens_processed": sum(proc[i] for i in order)}
    return {"mean_supervised_per_row": sum(sup) / len(sup), "mean_processed_per_row": sum(proc) / len(proc),
            "max_processed_per_row": max(proc), "rows_over_max_len": sum(p > max_len for p in proc),
            "max_len": max_len, "per_seed": per_seed}


def _check_row(r, reader, ctx, target):
    """-> (flags, read-back or None)."""
    flags = set()
    text = r.get("prompt") if isinstance(r.get("prompt"), str) else ""
    toks = set(_WORD_RE.findall(text))
    if ctx["rcwm_texts"] and any(t in text for t in ctx["rcwm_texts"].values()):
        flags.add("rcwm_heldout_text")
    if any(f in text for f in ctx["frags"]):
        flags.add("heldout_template")
    if toks & ctx["excl_tok"] or any(ph in text for ph in ctx["excl_phrase"]):
        flags.add("excluded_words")
    if toks & ctx["dev_sealed"]:
        flags.add("split_words")
    if toks & ctx["probes"]:
        flags.add("probe_words")
    try:
        rb = read_back(r, reader)
        dec = "apply" if rb["ok"] else "noop"
        native = TT.target_json(rb["after"])
        tgt = r.get("target")
        if target == "trace":
            line1, sep, line2 = tgt.partition("\n") if isinstance(tgt, str) else ("", "", None)
            if not sep or line2 != native:
                flags.add("gold")
            p = TT.parse_check(line1)
            if line1 != TT.check_line(rb["rule"], rb["state"], rb["event"]) or p is None or p["decision"] != dec \
                    or not TT._coherent(p):
                flags.add("trace_line1")
        elif tgt != native:
            flags.add("gold")
    except (ValueError, KeyError, TypeError, AttributeError):
        return flags | {"frame"}, None
    if not rb["frame_exact"]:
        flags.add("frame")
    if r.get("decision") != dec or r.get("status") != rb["event"]["status"]:
        flags.add("gold")
    if r.get("semantic_hash") != rb["semantic_hash"]:
        flags.add("readback_hash")
    sa = W.status_atom(rb["rule"])
    words = {rb["event"]["status"]} | set(sa["set"] if sa else ())
    if words & ctx["dev_sealed"] or set(WA.words(rb["sentence"])) & ctx["dev_sealed_lower"]:
        flags.add("split_words")
    if words - ctx["train_pool"] - ctx["anchors"]:
        flags.add("non_train_words")
    if rb["tid"].split(":")[-1].lower().startswith("h"):
        flags.add("heldout_template")
    rule = rb["rule"]
    if (rule["op"] in ("AND", "OR") and (rule["op"], rule["args"][1]["op"]) in gen.HELDOUT_COMBOS) \
            or (sa and len(sa["set"]) >= 3):
        flags.add("heldout_rule_factor")
    if any(not STD_BAL[0] <= v <= STD_BAL[1] for v in rb["state"].values()) \
            or abs(rb["event"]["amount"]) > STD_AMT_ABS:
        flags.add("heldout_magnitude")
    return flags, rb


def _user_part(prompt):
    m = _PROMPT_RE.fullmatch(prompt) if isinstance(prompt, str) else None
    return m.group(2) if m else None


def control_mismatches(rows, control, match, tids):
    """Rows violating the arm's relation to the control (None if no relation): 'prompts' (R3) or 'wording' (R1)."""
    if not match or control is None:
        return None
    bad = abs(len(rows) - len(control))
    for r, c, tid in zip(rows, control, tids):
        if match == "prompts":
            ok = r.get("prompt") == c["prompt"] and isinstance(r.get("target"), str) \
                 and r["target"].partition("\n")[2] == c["target"]
        elif match == "wording":
            pooled = tid is not None and wording_source(tid) not in ("single:t*", "frame:t*", "solo")
            ok = r.get("target") == c["target"] and _user_part(r.get("prompt")) == _user_part(c["prompt"]) \
                and (pooled or r.get("prompt") == c["prompt"])
        else:
            raise ValueError(f"unknown control relation {match!r}")
        bad += not ok
    return bad


def audit_arm(rows, reader, ctx, target="native", hashes=None, control=None, match=None, tok=None, seeds=(0, 1, 2),
              max_len=MAX_LEN, pool_cells=None):
    """Hard-check counts and covariates of one arm's rows. hashes: {battery: (semantic set, skeleton set)}."""
    if target not in ("native", "trace"):
        raise ValueError(f"target {target!r}")
    flags_n, sem, skel, tids = Counter(), set(), set(), []
    by_rt, by_src, by_tid, by_acc, by_kind = (defaultdict(Counter) for _ in range(5))
    statuses, decisions, cells, pool_bad = Counter(), Counter(), Counter(), {}
    for r in rows:
        sem.add(r.get("semantic_hash"))
        skel.add(r.get("skeleton_hash"))
        flags, rb = _check_row(r, reader, ctx, target)
        tids.append(rb["tid"] if rb else None)
        if rb is None:
            flags_n.update(flags)
            continue
        tid = rb["tid"]
        if tid in reader.templates and tid not in pool_bad:
            pool_bad[tid] = pool_wording_reason(tid, reader.templates[tid], ctx)
        if pool_bad.get(tid):
            flags.add("pool_wording")
        flags_n.update(flags)
        sem.add(rb["semantic_hash"])
        skel.add(rb["skeleton_hash"])
        dec = "apply" if rb["ok"] else "noop"
        op = rb["rule"]["op"]
        by_kind[op if op in W.STATUS_OPS else "composition" if op in ("AND", "OR") else "other"][dec] += 1
        by_rt[rule_type(rb["rule"])][dec] += 1
        by_src[wording_source(tid)][dec] += 1
        by_tid[tid][dec] += 1
        by_acc[len(rb["state"])][dec] += 1
        statuses[rb["event"]["status"]] += 1
        decisions[dec] += 1
        raw = tid.split(":", 1)[1] if ":" in tid else tid
        if pool_cells and raw in pool_cells:
            cells[pool_cells[raw]] += 1
    sem.discard(None)
    skel.discard(None)
    n = len(rows)
    tb = [len(r["target"].encode("utf-8")) for r in rows if isinstance(r.get("target"), str)]
    out = {"target": target, "control_relation": match, "rows": n, "parents": len({r.get("parent_id") for r in rows}),
           "variants": dict(Counter(r.get("variant") for r in rows)),
           "hard_check_rows": {f: flags_n[f] for f in HARD_FLAGS},
           "overlap": {b: {"semantic": len(sem & s), "skeleton": len(skel & k)}
                       for b, (s, k) in (hashes or {}).items()},
           "control_mismatch_rows": control_mismatches(rows, control, match, tids),
           "noop_share": round(decisions["noop"] / max(1, n), 4), "decisions": dict(decisions),
           "rows_by_rule_type": {k: sum(c.values()) for k, c in sorted(by_kind.items())},
           "class_balance_by_rule_type": _balance(by_rt), "class_balance_by_rule_kind": _balance(by_kind),
           "class_balance_by_wording_source": _balance(by_src), "class_balance_by_template": _balance(by_tid),
           "class_balance_by_n_accounts": _balance(by_acc), "pooled_rows_by_cell": dict(sorted(cells.items())),
           "pool_templates_flagged": {k: v for k, v in sorted(pool_bad.items()) if v},
           "anchor_status_counts": {w: statuses[w] for w in sorted(ctx["anchors"])},
           "distinct_statuses": len(statuses),
           "target_bytes": {"mean": round(sum(tb) / max(1, len(tb)), 2), "max": max(tb, default=0)}}
    if tok is not None:
        out["tokens"] = token_covariates(rows, tok, seeds=seeds, max_len=max_len)
        out["hard_check_rows"]["over_max_len"] = out["tokens"]["rows_over_max_len"]
    return out


def arm_failures(name, a, n_rows):
    fail = [f"rows_count:{name}"] if a["rows"] != n_rows else []
    fail += [f"{f}:{name}" for f in HARD_FLAGS if a["hard_check_rows"][f]]
    fail += [f"{k}_overlap:{b}:{name}" for b, v in a["overlap"].items() for k in ("semantic", "skeleton") if v[k]]
    if a["control_mismatch_rows"]:
        fail.append(f"control_match:{name}")
    return fail


def audit_e01r(arms, control=None, tok=None, n_rows=N_ROWS, batteries=None, reader=None, pool_cells=None,
               p2_files=P2_FILES, rcwm_texts=None, rcwm_root=RCWM_T1, seeds=(0, 1, 2), max_len=MAX_LEN):
    """AUDIT_E01R-style dict for treatment arms {name: {"rows": [...], "target": "native"|"trace", "match": None|
    "prompts"|"wording"}} and the control rows (N-CFSD; audited alike, failures reported apart, never gating).
    batteries: {name: records} (default build_batteries()); reader: default make_reader() now, i.e. after
    register_pool; rcwm_texts: default read from the RCWM tree (with the Titan frame check)."""
    reader = make_reader() if reader is None else reader
    ctx, fail = context(p2_files, rcwm_texts, rcwm_root)
    batteries = build_batteries() if batteries is None else batteries
    hashes = {b: record_hashes(recs) for b, recs in batteries.items()}
    kw = {"hashes": hashes, "tok": tok, "seeds": seeds, "max_len": max_len, "pool_cells": pool_cells}
    out = {}
    for name, spec in arms.items():
        out[name] = audit_arm(spec["rows"], reader, ctx, spec.get("target", "native"), control=control,
                              match=spec.get("match"), **kw)
        fail += arm_failures(name, out[name], n_rows)
    ctl = None if control is None else audit_arm(control, reader, ctx, "native", **kw)
    notes = []
    if ctl:
        for name, a in out.items():
            if a["parents"] != ctl["parents"]:
                notes.append(f"{name}: {a['parents']} parents vs the control's {ctl['parents']}; rows by rule type "
                             f"{a['rows_by_rule_type']} vs {ctl['rows_by_rule_type']} (declared covariate)")
    matched = {"rows": {k: v["rows"] for k, v in out.items()}, "n_rows_required": n_rows, "steps": STEPS,
               "batch": BATCH, "max_len": max_len}
    if tok is not None and ctl:
        ref = ctl["tokens"]["per_seed"]
        matched["token_covariates_vs_control"] = {
            a: {s: {k: {"diff": v["tokens"]["per_seed"][s][k] - ref[s][k],
                        "ratio": round(v["tokens"]["per_seed"][s][k] / max(1, ref[s][k]), 4)} for k in ref[s]}
                for s in ref} for a, v in out.items()}
    return {"ok": not fail, "failures": fail,
            "control_failures": [] if ctl is None else arm_failures("control", ctl, n_rows), "arms": out,
            "control": ctl, "matched": matched, "design_notes": notes,
            "batteries": {b: {"n": len(r), "sha256": sha_bytes("".join(json.dumps(x) + "\n" for x in r).encode())}
                          for b, r in batteries.items()},
            "reader": {"single_templates": len(reader.templates),
                       "registered": sorted(t for t in reader.templates if wording_source(t) not in
                                            ("single:t*", "single:h*"))},
            "heldout_template_fragments": ctx["frags"], "rcwm_heldout_texts": ctx["rcwm_texts"],
            "tokenizer": None if tok is None else type(tok).__name__,
            "code_sha256": {p.name: sha_bytes(p.read_bytes()) for p in (
                Path(__file__), HERE / "trace_target.py", HERE / "audit_tr_pool.py", E01N / "native_e01.py",
                E01N / "native_probe.py", LORA_CPU, Path(gen.__file__), Path(W.__file__), Path(V.__file__),
                Path(T.__file__), Path(WA.__file__))}}


# ----------------------------------------------------------------------------- CLI

def load_rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def require_audit_e01r(out_dir, arm, rows_sha):
    """Refuse to train on rows that have not passed audit_e01r: ok, or arm_ok[arm], for these rows (returns the
    AUDIT_E01R.json sha)."""
    p = Path(out_dir) / "AUDIT_E01R.json"
    aud = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    passed = aud.get("ok") is True or (aud.get("arm_ok") or {}).get(arm) is True
    if not passed or aud.get("rows_sha256", {}).get(arm) != rows_sha:
        raise SystemExit(f"refusing to train {arm}: AUDIT_E01R.json is missing, failed for this arm, or audited "
                         "other rows")
    return sha_bytes(p.read_bytes())


def arm_ok(failures, names):
    """{arm: True iff no failure names the arm (suffix ':<arm>') and no failure is arm-independent}."""
    own = {a: [f for f in failures if f.endswith(f":{a}")] for a in names}
    shared = [f for f in failures if not any(f.endswith(f":{a}") for a in names)]
    return {a: not shared and not own[a] for a in names}


def require_r1_branch(path):
    """E-R1 (the R1 pool) is audited only under branch B3: BRANCH_STEP_A.json from the frozen decide_branch.py."""
    p = Path(path)
    q = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if q.get("code_sha256") != DECIDE_SHA256 or q.get("branch") != "B3":
        raise SystemExit(f"E-R1 is audited only under branch B3; {p.name} "
                         f"{'is missing' if not q else 'says ' + str(q.get('branch'))}"
                         f"{'' if q.get('code_sha256') in (None, DECIDE_SHA256) else ' (not the frozen rule)'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated subset of E-R1,E-R2,E-R3 (or R1,R2,R3)")
    ap.add_argument("--pool", default=str(R1_POOL), help="E-R1: the audited training pool (audit_tr_pool.py output)")
    ap.add_argument("--control-dir", default=str(CONTROL_DIR))
    ap.add_argument("--tok", action="store_true", help="load the Titan tokenizer for token covariates (no weights)")
    ap.add_argument("--branch", default=str(BRANCH_PATH), help="BRANCH_STEP_A.json (E-R1 needs branch B3)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    audit_p = out / "AUDIT_E01R.json"
    if (out / "SEALED_COMMIT.json").exists():
        raise SystemExit("the SEALED commitment exists: the row audit it binds is never rewritten after the mint")
    if audit_p.exists():
        raise SystemExit(f"{audit_p.name} exists; it is create-only (a failed audit is removed only before the mint, "
                         "as a logged deviation)")
    names = [x if x.startswith("E-") else f"E-{x}" for x in (y.strip() for y in a.arms.split(",")) if x]
    if not names or set(names) - set(ARM_SPEC) or len(set(names)) != len(names):
        raise SystemExit(f"--arms must be a subset of {sorted(ARM_SPEC)}")
    if "E-R1" in names:
        require_r1_branch(a.branch)
    man_p = out / "TRAIN_MANIFEST_E01R.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    fail, pool_cells, pool_info = [] if man else ["manifest_missing"], None, None
    if "E-R1" in names:
        pool_p = Path(a.pool)
        if pool_p.name.endswith(".secret") or not pool_p.name.endswith("_audited.json"):
            raise SystemExit("E-R1 needs the audited training pool <tag>_audited.json (sealed files are refused)")
        aud_p = pool_p.with_name(pool_p.name.replace("_audited.json", "_AUDIT.json"))
        pool_aud = json.loads(aud_p.read_text(encoding="utf-8"))
        pool_sha = sha_bytes(pool_p.read_bytes())
        if pool_aud["outputs"]["audited"]["sha256"] != pool_sha or not pool_aud["meets_minimum"]:
            fail.append("pool_audit")
        if (man.get("r1_pool") or {}).get("sha256", pool_sha) != pool_sha:
            fail.append("pool_manifest")
        pool = json.loads(pool_p.read_text(encoding="utf-8"))["templates"]
        import recipes
        recipes.register_pool(pool, "tr")
        pool_cells = {f"tr:{t['id']}": f"{t['pol']}/{t['form']}" for t in pool}
        pool_info = {"file": pool_p.name, "sha256": pool_sha, "audit": aud_p.name, "n": len(pool)}
    entries = man.get("arms", man)  # native_e01r.build_cells nests the per-arm entries under "arms"
    arms, rows_sha = {}, {}
    for name in names:
        p = out / f"train_{name}.jsonl"
        rows_sha[name] = sha_bytes(p.read_bytes())
        rows = load_rows(p)
        if entries.get(name, {}).get("sha256") != rows_sha[name] or entries.get(name, {}).get("rows") != len(rows):
            fail.append(f"manifest_sha:{name}")
        arms[name] = {"rows": rows, "target": ARM_SPEC[name][0], "match": ARM_SPEC[name][1]}
    cdir = Path(a.control_dir)
    ctl_p = cdir / "train_N-CFSD.jsonl"
    ctl_sha = sha_bytes(ctl_p.read_bytes())
    if json.loads((cdir / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))["N-CFSD"]["sha256"] != ctl_sha:
        fail.append("control_sha")
    batteries = build_batteries()
    bp, mp = DEV2_DIR / "battery.jsonl", DEV2_DIR / "BATTERY_META.json"
    dev2_file = None
    if bp.exists() and mp.exists():
        data = bp.read_bytes()
        dev2_file = sha_bytes(data) == json.loads(mp.read_text(encoding="utf-8"))["battery_sha256"]
        if dev2_file:
            batteries["DEV2"] = batteries["DEV2"] + [json.loads(x) for x in data.decode().splitlines()]
        else:
            fail.append("dev2_battery_sha")
    tok = None
    if a.tok:
        from transformers import AutoTokenizer
        from rcx.native import lora_cpu as LC
        tok = AutoTokenizer.from_pretrained(str(LC.PATHS["sparse"]), local_files_only=True)
    res = audit_e01r(arms, control=load_rows(ctl_p), tok=tok, batteries=batteries, pool_cells=pool_cells)
    res["failures"] = fail + res["failures"]
    res["ok"] = not res["failures"]
    res["arm_ok"] = arm_ok(res["failures"], names)
    res.update({"out_dir": str(out), "rows_sha256": rows_sha, "control_rows_sha256": ctl_sha, "pool": pool_info,
                "dev2_battery_file_used": dev2_file})
    with open(audit_p, "x", encoding="utf-8", newline="\n") as f:  # create-only
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("ok", "arm_ok", "failures", "control_failures", "design_notes")}, indent=1))
    for name, v in res["arms"].items():
        print(name, "rows", v["rows"], "noop", v["noop_share"], "by type", v["rows_by_rule_type"], "sources",
              {k: c["n"] for k, c in v["class_balance_by_wording_source"].items()}, "tokens",
              v.get("tokens", {}).get("per_seed"), flush=True)
    if not res["ok"]:
        raise SystemExit(f"AUDIT FAILED: {res['failures']}")
    return res


if __name__ == "__main__":
    main()
