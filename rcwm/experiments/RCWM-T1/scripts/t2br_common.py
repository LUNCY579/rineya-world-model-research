"""T2b-register screen (RCWM roadmap item 3; joint roadmap item 5b with BWM-001): shared definitions. Torch-free.

Question. Do intermediates in the answer ("trace": compare, then compute, then answer) teach frozen Titan-Sparse the
register rules that bare-answer rows did not? T2's evidence: mix 0.022 -> 0.022, conditional branch choice at chance
(0.51) while the in-branch arithmetic is right, OFF outputs the unreduced sum. Decision point 3 asks whether any gain
is supervision or only extra decode compute; the token-matched "filler" arm answers that.

Environment semantics (rcwmt/env.py, frozen; quoted here, never changed). State {"X","Y","Z"}: integers modulo 97.
One instruction per step (env.step):
  swap         reg[r], reg[o] = reg[o], reg[r]
  add          reg[r] = (reg[r] + v) % 97
  mix          reg[r] = (reg[r] + 2 * reg[o] + v) % 97
  conditional  reg[r] = (reg[r] + (v if reg[o] > threshold else -v)) % 97
The conditional predicate is STRICT: reg[o] == threshold subtracts (the frame text: "adds value if other is greater
than threshold, otherwise subtracts value"). Python's % is non-negative, so an underflow wraps up (3 - 5 -> 95).
'wrap' (as scripts/make_t2_data.stratum): the raw value of the updated register (r + v; r + 2 o + v; r +/- v on the
chosen branch) lies outside [0, 96].

Joint trace format (BWM D-52 + the binding amendments 1-5; reference implementation, read-only:
Rineya_BWM_001_Research_Bundle/native_v1/bwmn/traces.py). One JSON object, keys in LITERAL order cmp -> calc -> after,
serialized with an explicit order (never sort_keys, never a sorting canonicaliser):
  cmp    the branch predicate's INPUTS and its boolean. conditional: {"x": reg[other], "thr": threshold,
         "ok": reg[other] > threshold} (env's strict '>'; the joint proposal's '>=' is replaced by the env semantics).
         swap / add / mix have no branch predicate: null (as BWM's 'finish').
  calc   the chosen branch's arithmetic: {"raw": the unreduced value of the updated register, "mod": raw % 97}
         (add: r + v; mix: r + 2 o + v; conditional: r + v if ok else r - v). swap: null (no arithmetic).
  after  the family's EXISTING answer: the register state JSON exactly as env.dumps writes it ({"X":..,"Y":..,"Z":..}).
Arms on the SAME rows and the SAME prompt (Titan's verbatim one-step frame, env.one_step_messages):
  plain  the existing answer only (= after)
  trace  {"cmp":..,"calc":..,"after":..}
  filler {"fill":" . . .","after":..}: the fill string is a run of the one-token unit " ." chosen so that the WHOLE
         filler target has exactly as many Titan tokens as that row's trace target (checked per row).
Parsing. parse_trace rejects any other key order, extra or missing keys and duplicate keys; parse_filler likewise
(["fill","after"]). The decision parser reads ONLY 'after': after_text() re-serializes it (sorted, compact; BWM's
convention) for rcwmt.env.parse_state(lenient=True), the parser of every existing chain/regret pipeline. A malformed
trace or filler answer has no 'after' (= a wrong answer). cmp_ok / calc_ok are scored separately (mechanism).

Surface diversity (decision point 1: if PREREG-3 says TRIGGER). OFF by default. `t2br_make_data.py --surface-diversity
on` re-renders the SAME training rows with rule-text paraphrases (SD_PARAPHRASES) and field-name variants (SD_FIELDS);
golds and targets are unchanged. No code change is needed to switch it on.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402

FAM = "register_machine"
MOD = env.MOD
REGS = env.REGISTERS
ARMS = ("plain", "trace", "filler")
SIDES = ("off", "plain", "trace", "filler")          # off = adapter disabled (Titan-Sparse); scored with the plain parser
SIDE_ARM = {"off": "plain", "plain": "plain", "trace": "trace", "filler": "filler"}
FILL_UNIT = " ."
H = 4                                                # chain horizon
K = 8                                                # candidates per chain parent (the pilot / PREREG-2 decision setup)

DATA = ROOT / "data"
DOCS = ROOT / "docs" / "t2b" / "register"
TRAIN_STEM = "t2br_train"
DEV = DATA / "t2br_dev.jsonl"
CHAINS = DATA / "t2br_chains.jsonl"
DEV_SOURCE = DATA / "t2br_dev.SOURCE.json"
TOKENIZER_JSON = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                      r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")   # == rcwmt.runtime.TOKENIZER_DIR


def train_path(arm: str, variant: str = "base") -> Path:
    return DATA / (f"{TRAIN_STEM}_{arm}.jsonl" if variant == "base" else f"{TRAIN_STEM}_{variant}_{arm}.jsonl")


def train_source(variant: str = "base") -> Path:
    return DATA / (f"{TRAIN_STEM}.SOURCE.json" if variant == "base" else f"{TRAIN_STEM}_{variant}.SOURCE.json")


# ---------------------------------------------------------------------------------------------- value ranges
IN_V = (-8, 11)                  # recorded CANDIDATE_VIEWS register 'value' range (asserted against env.load_marginals)
OOR_V_ABS = (12, 40)             # out-of-range values: |v| in 12..40, both signs
THR_IN = (0, 96)                 # recorded threshold range (= the register range)
OOR_THR = ((-40, -1), (97, 136))  # out-of-range thresholds, outside the register range: ok always true / always false

# ---------------------------------------------------------------------------------------------- dev screen strata
# name -> (group, role, description). group 'in' = in-distribution value/threshold ranges, 'oor' = out of range.
DEV_STRATA = OrderedDict([
    ("mix", ("in", "noninferiority", "mix, in-range value, 50 wrap / 50 no-wrap")),
    ("wrap", ("in", "reported", "add whose raw sum leaves [0, 96] (40 over / 40 under), in-range value")),
    ("cond_xm1", ("in", "noninferiority", "conditional with other = threshold - 1 (ok false: subtract); triplet member")),
    ("cond_x0", ("in", "noninferiority", "conditional with other = threshold (ok false under the STRICT '>'); triplet")),
    ("cond_xp1", ("in", "noninferiority", "conditional with other = threshold + 1 (ok true: add); triplet member")),
    ("swap", ("in", "control", "swap of two unequal registers (untouched skill: not in training)")),
    ("add_nowrap", ("in", "control", "add, in-range value != 0, no wrap (untouched skill: not in training)")),
    ("mix_oor", ("oor", "primary", "mix with |value| in 12..40 (outside the trained -8..11), 50 wrap / 50 no-wrap")),
    ("wrap_oor", ("oor", "reported", "add with |value| in 12..40 whose raw sum leaves [0, 96] (40 over / 40 under)")),
    ("cond_oor_val", ("oor", "primary", "conditional pairs at the boundary (other = threshold, ok false / threshold - 1, "
                                        "ok true), |value| in 12..40")),
    ("cond_oor_thr", ("oor", "primary", "conditional pairs with thresholds outside the register range: -40..-1 (ok "
                                        "true) / 97..136 (ok false); value, registers in range")),
])
DEV_N = {"mix": 100, "wrap": 80, "cond_xm1": 60, "cond_x0": 60, "cond_xp1": 60, "swap": 80, "add_nowrap": 80,
         "mix_oor": 100, "wrap_oor": 80, "cond_oor_val": 100, "cond_oor_thr": 100}
# outcome groups of the pre-stated rule
PRIMARY = OrderedDict([("conditional", ("cond_oor_val", "cond_oor_thr")), ("mix", ("mix_oor",))])
NONINF = OrderedDict([("conditional_in", ("cond_xm1", "cond_x0", "cond_xp1")), ("mix_in", ("mix",))])
CONTROLS = ("swap", "add_nowrap")
CHAIN_STRATA = OrderedDict([("CH_IN", 32), ("CH_X", 16)])   # parents x K=8 candidates x H=4 steps

# ---------------------------------------------------------------------------------------------- pre-stated go rule
GAIN_CONDITIONAL = 0.25          # trace - plain on the primary 'conditional' outcome
GAIN_MIX = 0.30                  # trace - plain on the primary 'mix' outcome
FILLER_STOP = 0.10               # decision point 3: trace - filler < 0.10 on BOTH primaries -> stop
CONTROL_LOSS = 0.05              # controls: trace ON - OFF >= -0.05 on swap and add_nowrap
NONINF_MARGIN = 0.05             # in-distribution non-inferiority: trace - plain >= -0.05 (conditional_in, mix_in)
GO_RULE_TEXT = (
    "PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank; 95% cluster-"
    "bootstrap CIs reported, not gating). Outcome = exact 'after' state (decision parser reads only 'after'; a "
    "malformed trace/filler answer is wrong). PRIMARY outcomes are the out-of-range strata (joint amendment 5): "
    "'conditional' = cond_oor_val + cond_oor_thr pooled (branch-balanced by construction); 'mix' = mix_oor. "
    "GO iff ALL of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below "
    "trace: trace - filler > 0 on both primaries; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap "
    "(untouched skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced "
    "mean of cond_xm1/cond_x0/cond_xp1) and on mix_in. DECISION POINT 3 (checked first): if trace - filler < 0.10 on "
    "BOTH primaries, STOP and report 'extra decode compute, not supervision' (sub-reason 'neither format helped' when "
    "filler - plain < 0.10 on both primaries as well). Otherwise NO-GO, naming every failed condition. The screen is "
    "seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.")

# ---------------------------------------------------------------------------------------------- semantics


def branch_ok(before: dict, ev: dict) -> bool:
    """The env's conditional predicate (strict)."""
    return before[ev["other"]] > ev["threshold"]


def raw_value(before: dict, ev: dict):
    k, r = ev["kind"], before[ev["register"]]
    if k == "add":
        return r + ev["value"]
    if k == "mix":
        return r + 2 * before[ev["other"]] + ev["value"]
    if k == "conditional":
        return r + (ev["value"] if branch_ok(before, ev) else -ev["value"])
    return None


def wraps(before: dict, ev: dict) -> bool:
    raw = raw_value(before, ev)
    return raw is not None and not 0 <= raw < MOD


def op_class(before: dict, ev: dict) -> str:
    """scripts/make_t2_data.stratum('R', ...) (that module imports torch; replicated as t2_analysis.t2_stratum does)."""
    k = ev["kind"]
    if k == "swap":
        return "swap"
    if k == "conditional":
        return "conditional/" + ("gt" if before[ev["other"]] > ev["threshold"] else "le")
    raw = before[ev["register"]] + (ev["value"] if k == "add" else 2 * before[ev["other"]] + ev["value"])
    return f"{k}/" + ("wrap" if not 0 <= raw < MOD else "nowrap")


def parts(before: dict, ev: dict):
    """(cmp, calc, after) of the joint trace for one register step; dicts in their literal key order."""
    after = env.step(FAM, before, ev)
    k = ev["kind"]
    cmp = {"x": before[ev["other"]], "thr": ev["threshold"], "ok": branch_ok(before, ev)} if k == "conditional" else None
    if k == "swap":
        calc = None
    else:
        raw = raw_value(before, ev)
        calc = {"raw": raw, "mod": raw % MOD}
        if after[ev["register"]] != calc["mod"]:
            raise AssertionError(f"calc disagrees with env.step: {before} {ev}")
    return cmp, calc, after


def _dump(obj) -> str:
    """Compact JSON in the object's OWN key order (never sorted)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def plain_text(after: dict) -> str:
    return env.dumps(after)


def trace_text(cmp, calc, after) -> str:
    return '{"cmp":' + _dump(cmp) + ',"calc":' + _dump(calc) + ',"after":' + env.dumps(after) + "}"


def filler_text(k: int, after: dict) -> str:
    return '{"fill":"' + FILL_UNIT * k + '","after":' + env.dumps(after) + "}"


def matched_filler(trace: str, after: dict, count) -> tuple[str, int]:
    """The filler target whose token count equals the trace target's EXACTLY (amendment 4); count(text) -> int."""
    n = count(trace)
    for k in range(1, n + 1):
        t = filler_text(k, after)
        c = count(t)
        if c == n:
            return t, k
        if c > n:
            break
    raise ValueError(f"no filler length matches the trace's {n} tokens: {trace}")


def targets(before: dict, ev: dict, count) -> dict:
    cmp, calc, after = parts(before, ev)
    tr = trace_text(cmp, calc, after)
    fi, k = matched_filler(tr, after, count)
    return {"plain": plain_text(after), "trace": tr, "filler": fi, "fill_units": k, "cmp": cmp, "calc": calc,
            "after": env.dumps(after)}


# ---------------------------------------------------------------------------------------------- parsing and scoring


def _loads_ordered(text: str):
    """json.loads that keeps key order and REJECTS duplicate keys."""
    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate key")
        return dict(pairs)
    return json.loads(text, object_pairs_hook=hook)


def parse_trace(text):
    """(cmp, calc, after) or None. Keys must be exactly cmp, calc, after, in that order (no extra, no duplicates)."""
    if not isinstance(text, str):
        return None
    try:
        d = _loads_ordered(text.strip())
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(d, dict) or list(d) != ["cmp", "calc", "after"]:
        return None
    return d["cmp"], d["calc"], d["after"]


def parse_filler(text):
    """(fill, after) or None. Keys must be exactly fill, after, in that order."""
    if not isinstance(text, str):
        return None
    try:
        d = _loads_ordered(text.strip())
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(d, dict) or list(d) != ["fill", "after"]:
        return None
    return d["fill"], d["after"]


def after_text(text, arm: str):
    """The 'after' part for the family's existing parser (plain: the text itself). BWM traces.after_text convention."""
    if arm == "plain":
        return text
    if arm == "trace":
        p = parse_trace(text)
        return None if p is None else json.dumps(p[2], separators=(",", ":"), sort_keys=True)
    if arm == "filler":
        p = parse_filler(text)
        return None if p is None else json.dumps(p[1], separators=(",", ":"), sort_keys=True)
    raise ValueError(arm)


def parse_after(text, arm: str):
    """Decision parser: the state in 'after' via env.parse_state(lenient=True), or None (malformed = wrong)."""
    a = after_text(text, arm)
    if a is None:
        return None
    return env.parse_state(FAM, a, lenient=True)


def same_json(a, b) -> bool:
    """Equal as ORDERED JSON (key order and types matter: true != 1)."""
    try:
        return _dump(a) == _dump(b)
    except (TypeError, ValueError):
        return False


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def score_text(row: dict, text, arm: str) -> dict:
    """Per-row score. after_ok is the outcome; the trace fields are mechanism diagnostics."""
    st = parse_after(text, arm)
    res = {"parsed": st is not None, "after_ok": st is not None and env.dumps(st) == row["gold_after"]}
    if arm == "trace":
        p = parse_trace(text)
        res["format_ok"] = p is not None
        gc, gk = row["gold_cmp"], row["gold_calc"]
        if p is None:
            res.update(cmp_ok=False, ok_ok=False, calc_ok=False, after_follows_calc=False, calc_follows_cmp=False)
        else:
            c, k, _ = p
            res["cmp_ok"] = same_json(c, gc)
            res["ok_ok"] = gc is not None and isinstance(c, dict) and isinstance(c.get("ok"), bool) and c["ok"] == gc["ok"]
            res["calc_ok"] = same_json(k, gk)
            ev, b = row["event"], row["before"]
            fol = False
            if gk is None:
                fol = k is None and st is not None
            elif isinstance(k, dict) and _int(k.get("mod")) and st is not None:
                exp = dict(b)
                exp[ev["register"]] = k["mod"]
                fol = env.dumps(st) == env.dumps(exp)
            res["after_follows_calc"] = fol
            cf = False
            if ev["kind"] == "conditional" and isinstance(c, dict) and isinstance(c.get("ok"), bool) \
                    and isinstance(k, dict) and _int(k.get("raw")):
                cf = k["raw"] == b[ev["register"]] + (ev["value"] if c["ok"] else -ev["value"])
            res["calc_follows_cmp"] = cf
    elif arm == "filler":
        p = parse_filler(text)
        res["format_ok"] = p is not None
        res["fill_exact"] = p is not None and p[0] == FILL_UNIT * row["fill_units"]
    return res


# ---------------------------------------------------------------------------------------------- surface heuristics


def heuristic_after(name: str, before: dict, ev: dict):
    """A deterministic shortcut's answer for one step, or None where the heuristic does not apply."""
    k, r = ev["kind"], ev["register"]
    o, v, t = ev.get("other"), ev.get("value", 0), ev.get("threshold")
    b = dict(before)
    if name == "copy":
        return b
    if name == "unreduced":                          # the correct operation (correct branch) without the modulo
        if k == "swap":
            return env.step(FAM, before, ev)
        b[r] = raw_value(before, ev)
        return b
    if k == "conditional":
        dec = {"always_add": True, "always_sub": False, "wrong_register": before[r] > t, "non_strict": before[o] >= t,
               "x_hi": before[o] > 48, "thr_lo": t < 48, "v_pos": v > 0}.get(name)
        if dec is None:
            return None
        b[r] = (before[r] + (v if dec else -v)) % MOD
        return b
    if k == "mix":
        if name == "mix_1x":
            b[r] = (before[r] + before[o] + v) % MOD
            return b
        if name == "mix_no_v":
            b[r] = (before[r] + 2 * before[o]) % MOD
            return b
    return None


GATED_HEURISTICS = ("copy", "unreduced", "always_add", "always_sub")
DIAG_HEURISTICS = ("wrong_register", "non_strict", "x_hi", "thr_lo", "v_pos", "mix_1x", "mix_no_v")
AUDIT_MAX = 0.55

# ---------------------------------------------------------------------------------------------- prompts


def render_prompt(messages: list) -> str:
    """Qwen3 chat template with add_generation_prompt=True, enable_thinking=False (rcwmt.capture.render), written out
    so the data builder needs no torch. Verified equal to every register prompt of data/t2_train.jsonl (rendered by
    the real template) at build time, and to tok.apply_chat_template on every row in the tiny test."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def canonical_messages(before: dict, ev: dict) -> list:
    return env.one_step_messages(FAM, before, ev)


# Surface diversity (OFF by default). Paraphrases keep the exact semantics (strict '>', modulo 97, simultaneous swap,
# mix = 2 x other + value) and name the event fields through placeholders, so a field-name variant renames the field
# in the prompt JSON AND in the rule text together. SD_PARAPHRASES[0] with the canonical names == env.SYSTEM_ONE.
SD_PARAPHRASES = (
    "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap exchanges {register} and "
    "{other} simultaneously. add adds {value} to {register}. mix adds (2 times {other} plus {value}) to {register}. "
    "conditional adds {value} if {other} is greater than {threshold}, otherwise subtracts {value}. Reduce updated "
    "numeric registers modulo 97; preserve other registers. Return only the complete resulting registers as a JSON "
    "object.",
    "You are given registers holding integers modulo 97 and one instruction. swap: the registers named by {register} "
    "and {other} exchange values at the same time. add: the {register} register increases by {value}. mix: the "
    "{register} register increases by twice the {other} register plus {value}. conditional: if the {other} register is "
    "strictly greater than {threshold}, the {register} register increases by {value}; otherwise it decreases by "
    "{value}. Reduce every updated register modulo 97 and leave the rest unchanged. Output only the complete resulting "
    "registers as a JSON object.",
    "Execute the single instruction on the registers (integers modulo 97). swap - exchange the {register} and {other} "
    "registers simultaneously; add - add {value} to the {register} register; mix - add 2 x the {other} register + "
    "{value} to the {register} register; conditional - when the {other} register is greater than {threshold}, add "
    "{value} to the {register} register, else subtract {value} from it. Each updated register is reduced modulo 97; "
    "untouched registers keep their values. Reply with only the complete resulting registers as a JSON object.",
    "Registers are integers modulo 97. Carry out the one instruction given. For swap, the {register} and {other} "
    "registers trade values simultaneously. For add, {value} is added to the {register} register. For mix, the "
    "{register} register gains two times the {other} register plus {value}. For conditional, the {register} register "
    "gains {value} if the {other} register is greater than {threshold} and loses {value} otherwise. Take updated "
    "registers modulo 97 and keep the others as they are. Return just the complete resulting registers as a JSON "
    "object.",
)
SD_FIELDS = OrderedDict([
    ("register", ("register", "target", "dst")),
    ("other", ("other", "source", "src")),
    ("value", ("value", "amount", "delta")),
    ("threshold", ("threshold", "limit", "cutoff")),
    ("kind", ("kind", "op")),
    ("before", ("before", "registers", "state")),
    ("event", ("event", "instruction")),
])


def draw_sd(rng: random.Random) -> dict:
    return {"paraphrase": rng.randrange(len(SD_PARAPHRASES)), **{f: rng.choice(v) for f, v in SD_FIELDS.items()}}


def sd_messages(before: dict, ev: dict, sd: dict | None) -> list:
    if sd is None:
        return canonical_messages(before, ev)
    names = {f: sd[f] for f in SD_FIELDS}
    system = SD_PARAPHRASES[sd["paraphrase"]].format(**{f: names[f] for f in ("register", "other", "value", "threshold")})
    ev2 = {names.get(k, k): v for k, v in ev.items()}
    user = env.dumps({names["before"]: before, names["event"]: ev2})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _check_sd_canonical() -> None:
    canon = {f: v[0] for f, v in SD_FIELDS.items()}
    if SD_PARAPHRASES[0].format(**canon) != env.SYSTEM_ONE[FAM]:
        raise AssertionError("SD_PARAPHRASES[0] with canonical names must equal env.SYSTEM_ONE[register_machine]")
    for v in SD_FIELDS.values():
        if len(set(v)) != len(v):
            raise AssertionError("duplicate field-name variant")


_check_sd_canonical()

# ---------------------------------------------------------------------------------------------- io


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path) -> list:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_dev(path: Path = DEV) -> list:
    return load_jsonl(path)


def load_chain_bank(path: Path = CHAINS) -> list:
    """Chain parents in the rcwmt bank format (parent_id, family, split, s0, candidates[cand_id, events])."""
    return load_jsonl(path)


def flat_chains(parents: list) -> list:
    return [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"], "stratum": p["stratum"],
             "parent_id": p["parent_id"]} for p in parents for c in p["candidates"]]


_TOK = []


def token_counter():
    """Titan's tokenizer through the `tokenizers` library (no torch): text -> number of ids, no special tokens. The tiny
    test checks it against transformers' AutoTokenizer (what the trainer uses) on every target."""
    if not _TOK:
        from tokenizers import Tokenizer
        _TOK.append(Tokenizer.from_file(str(TOKENIZER_JSON)))
    t = _TOK[0]
    return lambda s: len(t.encode(s, add_special_tokens=False).ids)
