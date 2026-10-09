"""RCWM-T2b prerequisite representation screen (RCWM roadmap item 2; joint roadmap 5a): shared, torch-free definitions.

Question. T2's rule-coverage LoRA taught the prerequisite family a COPY shortcut (false copy 0.039 -> 0.624; applicable
state-changing 0.587 -> 0.353), and Titan's own prerequisite "rule" is integer addition (R10/R11). Does a de-shortcut
training set (every audited heuristic at chance on the training rows, minimal pairs), and/or a different TARGET
REPRESENTATION, let the same T2 recipe (LoRA r16, 190 steps, vendored BWM trainer, frozen FP32 Titan-Sparse) learn the
precondition rule and the bitwise update, including out of range (10/12-bit masks)?

Arms: the SAME training rows rendered four ways (only D-int, S-trace and B-bits are trained by default):
  dint    D-int     the data control. Prompt = T2's prerequisite frame byte for byte (env.multi_step_messages with one
                    event); target = the bare integer state, e.g.  37
  strace  S-trace   joint trace format (BWM D-52 / RCWM, binding amendments 1-4). Target, keys in this literal order:
                    {"cmp":{"on_miss":[2],"off_hit":[],"viol":1,"ok":false},"calc":37,"after":37}
                    on_miss = ascending bit positions (0 = least significant) of pre_on bits that are clear in the state;
                    off_hit = bit positions of pre_off bits that are set; viol = len(on_miss) + len(off_hit)
                    (= popcount((pre_on & ~before) | (pre_off & before)), pre_on and pre_off are disjoint); ok = viol == 0;
                    calc = (before & ~delete) | add if ok else before (the chosen branch's result); after = the integer
                    state (P's existing answer format, unchanged).
  bbits   B-bits    state and the four masks written as w-digit bit strings, most significant bit first, in the prompt;
                    target  {"applicable":true,"after":"00110101"}  (flag first, then the bit-string state).
  ffill   F-filler  OFF unless the GPU budget allows (t2bp_run.py --with-filler). Target {"fill":" . . .","after":37}:
                    the fill string is the one-token unit " ." repeated so the WHOLE target has exactly the trace
                    target's token count under Titan's tokenizer (checked per row).
Every arm's system prompt keeps the rule sentences of T2's frame byte for byte; only the format sentence ("Return only
...") differs, and B-bits adds the bit-string convention to the state sentence. The decision pipelines read only the
state: D-int is parsed exactly like rcwmt.env.parse_state(..., lenient=True) (generalised to w bits); S-trace, B-bits
and F-filler targets are parsed STRICTLY (literal key order, no extra or duplicate keys, typed values, in-range state;
a malformed answer is a wrong answer). cmp_ok / calc_ok (S-trace) and flag_ok (B-bits) are scored separately.

Surface diversity (decision point 1: if PREREG-3 says TRIGGER, items 2-3 must diversify surface forms). OFF by default;
switched on WITHOUT code changes by `t2bp_make_data.py --surface-diversity on` (writes data/t2bp_train_sd_<arm>.jsonl)
and `t2bp_run.py --data-variant sd`. ON: each training row gets a rule-text paraphrase (PARAPHRASES[0..3], 0 =
canonical) and a field-name variant (FIELD_VARIANTS[0..3], 0 = canonical) drawn from a seeded RNG; PARAPHRASES[4] and
FIELD_VARIANTS[4] are reserved for held-out evaluation (stage E3 of a `--data-variant sd` run: the dev one-step rows in
surface 4, OFF and ON, descriptive) and never used in training. Targets do not change.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402

FAM = "prerequisite_plan"
DATA = ROOT / "data"
DOCS = ROOT / "docs" / "t2b" / "prereq"
OUT_ROOT = ROOT / "runs" / "t2bp"
FILLER_TAG = "s0_ffill"               # the R5 decode-compute control, run on its own only if S-trace wins the screen
TOKENIZER_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                     r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")          # == rcwmt.runtime.TOKENIZER_DIR
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
SEED = 20261111                       # not used by any earlier RCWM-T1 builder, analysis or bootstrap
ARMS = ("dint", "strace", "bbits", "ffill")
SCREEN_ARMS = ("dint", "strace", "bbits")
CONTROL_ARMS = ("ffill",)
ARM_LABEL = {"dint": "D-int", "strace": "S-trace", "bbits": "B-bits", "ffill": "F-filler"}
FILL_UNIT = " ."
WIDTH_ID = 8
OOD_WIDTHS = (10, 12)
CMP_KEYS = ("on_miss", "off_hit", "viol", "ok")
TRACE_KEYS = ("cmp", "calc", "after")
BBITS_KEYS = ("applicable", "after")
FILL_KEYS = ("fill", "after")

TRAIN_FILES = {a: DATA / f"t2bp_train_{a}.jsonl" for a in ARMS}
TRAIN_FILES_SD = {a: DATA / f"t2bp_train_sd_{a}.jsonl" for a in ARMS}
TRAIN_SOURCE = DATA / "t2bp_train.SOURCE.json"
TRAIN_SOURCE_SD = DATA / "t2bp_train_sd.SOURCE.json"
DEV = DATA / "t2bp_dev.jsonl"
DEV_SOURCE = DATA / "t2bp_dev.SOURCE.json"

# ---------------------------------------------------------------------------------------------- the design
# Row types (prerequisite one-step rows; b = before, a = add, d = delete; env generator: a single-bit add, d disjoint
# from a). Each type fixes which of the audited heuristics is right (copy, apply-regardless, int b+a, OR b|a):
#   A1 applicable, a not in b, d clears nothing set   [0,1,1,1]  gold = b|a = b+a (the carry-free, delete-free case)
#   A2 applicable, a not in b, d clears a set bit     [0,1,0,0]  gold = (b&~d)|a
#   A3 applicable, a already in b, d clears nothing   [1,1,0,1]  gold = b (the carry case Titan gets wrong, R10)
#   A4 applicable, a already in b, d clears a set bit [0,1,0,0]  gold = b&~d: wrong under b+a AND (b&~d)+a ("int-add-wrong")
#   Ib inapplicable, a not in b                        [1,0,0,0]  gold = b
#   Ia inapplicable, a in b, d clears a set bit        [1,0,0,1]  gold = b
#   Za applicable, add = 0 (delete-only action), d clears a set bit   [0,1,0,0]  gold = b&~d
#   Zi inapplicable, add = 0, d clears a set bit                      [1,0,1,1]  gold = b
# Derivation. Integer b+a is right only on A1 and Zi (a = 0 inapplicable: b + 0 = b) among these types; apply-regardless
# is right on A1, A2, A3, A4, Za. So (#apply - #int_add) = #A2 + #A3 + #A4 + #Za - #Zi, and holding both at 0.50 (+-0.02)
# requires #Zi - #Za = #A2 + #A3 + #A4 (+- 60 rows): every delete or carry row needs an extra inapplicable add=0 row. The
# price is an "add 0 -> mostly copy" cue on the training rows (none in the dev bank; docs/t2b/prereq/DESIGN.md 3.1).
# Likewise OR - int_add = #A3 + #Ia, so carry-without-delete rows (A3) and Ia rows are capped at ~60 together (50 here).
# Revision 2 (review, before any output): A4 (the int-add-wrong skill) 65 -> 150 rows, so the precondition-aware integer
# composites are separated from the bitwise rule on 165 rows (A3 + A4) instead of 80; Ia 20 -> 35 so the "add bit already
# set -> apply" cue stays near its revision-1 level (applicable share among add-bit-set rows 0.825; was 0.80).
TRAIN_UNPAIRED = {"A1": 330, "A2": 15, "A3": 15, "A4": 115, "Ib": 295, "Zi": 180}
TRAIN_PAIRS = {"A1": 160, "A2": 25, "A4": 35, "Za": 55}         # minimal pairs: applicable member type -> count
PAIR_INAPP = {"A1": "Ib", "A2": "Ib", "A4": "Ia", "Za": "Zi"}    # the inapplicable member's type
N_TRAIN = sum(TRAIN_UNPAIRED.values()) + 2 * sum(TRAIN_PAIRS.values())     # 1,500
N_A4_TRAIN = TRAIN_UNPAIRED["A4"] + TRAIN_PAIRS["A4"]                     # 150 (quoted by the STOP wording)
AUDITED = ("copy", "apply_regardless", "int_add", "or_add")
AUDIT_ALIASES = {"apply_ignoring_pre": "apply_regardless"}     # "(before&~del)|add ignoring preconditions" == apply-regardless
DESCRIPTIVE_HEURISTICS = ("int_add_del", "delete_only")
AUDIT_LO, AUDIT_HI = 0.48, 0.52
QUOTA_MIN = {"minimal_pairs": 250, "int_add_wrong_strict": 150, "applicable_delete": 200, "int_add_wrong_r11": 200,
             "inapp_pre_on": 300, "inapp_pre_off": 300}

# dev screen bank (no model call; fresh against every used payload set and the training rows). Revision 2 appends
# D_inapp_addset (inapplicable rows whose add bit is already set and whose delete clears a set bit: the precondition
# mirror of D_int_add_wrong); the 950 revision-1 rows are unchanged and come first.
DEV_STRATA = ("D_app_add", "D_app_unchanged", "D_app_delete", "D_int_add_wrong", "D_inapp_pre_on", "D_inapp_pre_off",
              "D_ood10", "D_ood12", "D_inapp_addset")
DEV_N = {"D_app_add": 100, "D_app_unchanged": 100, "D_app_delete": 100, "D_int_add_wrong": 150, "D_inapp_pre_on": 100,
         "D_inapp_pre_off": 100, "D_ood10": 150, "D_ood12": 150, "D_inapp_addset": 100}
DEV_TYPE = {"D_app_add": "A1", "D_app_unchanged": "A3", "D_app_delete": "A2", "D_int_add_wrong": "A4"}
OOD_SUBTYPES = ("A1", "A2", "A3", "A4", "inapp_pre_on", "inapp_pre_off")    # 25 each per OOD stratum
ADDSET_SUBTYPES = ("Ia/pre_on", "Ia/pre_off")                              # 50 each in D_inapp_addset
OOD_WIDTH = {"D_ood10": 10, "D_ood12": 12}
ID_STRATA = ("D_app_add", "D_app_unchanged", "D_app_delete", "D_int_add_wrong", "D_inapp_pre_on", "D_inapp_pre_off",
             "D_inapp_addset")
FC_STRATA = ("D_app_add", "D_app_delete", "D_int_add_wrong")          # 8-bit rows whose gold differs from the input
INAPP_STRATA = ("D_inapp_pre_on", "D_inapp_pre_off", "D_inapp_addset")  # applying would change the state on every row
CHAIN_PARENTS, CHAIN_K, CHAIN_H, CHAIN_ACTIONS, CHAIN_MIN_DISTINCT = 30, 8, 4, 6, 4

# frozen inputs: their sha256 are recorded by `t2bp_analysis.py --freeze-rule` in docs/t2b/prereq/SCREEN_RULE.json and
# checked by the runner (before the lease) and by the analysis (before any output is read)
FROZEN_CODE = ("scripts/t2bp_common.py", "scripts/t2bp_make_data.py", "scripts/t2bp_run.py", "scripts/t2bp_analysis.py",
               "rcwmt/env.py", "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/capture.py", "rcwmt/data.py",
               "rcwmt/decide.py", "rcwmt/decisions.py", "rcwmt/fidelity.py",
               "vendor_bwm/scripts/train_generic.py", "vendor_bwm/bwmn/lm.py")
FROZEN_DATA = ("data/t2bp_dev.jsonl", "data/t2bp_dev.SOURCE.json", "data/t2bp_train_dint.jsonl",
               "data/t2bp_train_strace.jsonl", "data/t2bp_train_bbits.jsonl", "data/t2bp_train_ffill.jsonl",
               "data/t2bp_train.SOURCE.json")
SCREEN_RULE_JSON = "SCREEN_RULE.json"
SCREEN_RULE_MD = "SCREEN_RULE.md"

# ---------------------------------------------------------------------------------------------- surface forms
FIELD_VARIANTS = (
    {"before": "before", "events": "events", "add": "add", "delete": "delete", "pre_on": "pre_on", "pre_off": "pre_off"},
    {"before": "state", "events": "actions", "add": "set", "delete": "clear", "pre_on": "require_on", "pre_off": "require_off"},
    {"before": "current", "events": "steps", "add": "add_bits", "delete": "del_bits", "pre_on": "need_on", "pre_off": "need_off"},
    {"before": "flags", "events": "ops", "add": "turn_on", "delete": "turn_off", "pre_on": "if_on", "pre_off": "if_off"},
    {"before": "mask", "events": "moves", "add": "plus", "delete": "minus", "pre_on": "must_on", "pre_off": "must_off"},  # held out
)
PARAPHRASES = (   # the rule sentences; {pre_on} etc. are the field names of the row's variant
    "Apply each action to the state, in order. An action is applicable only if all its {pre_on} bits are set and all "
    "its {pre_off} bits are clear in the current state; an inapplicable action leaves the state unchanged. For an "
    "applicable action, apply {delete} bits first, then {add} bits.",
    "Process the actions one after another. An action may be used only when every bit of its {pre_on} mask is 1 and "
    "every bit of its {pre_off} mask is 0 in the current state; otherwise the action does nothing. When an action is "
    "used, first clear its {delete} bits, then set its {add} bits.",
    "Go through the actions in the given order. An action takes effect only if the current state has all of its "
    "{pre_on} bits on and none of its {pre_off} bits on; an action that does not take effect keeps the state the same. "
    "An action that takes effect removes its {delete} bits and then adds its {add} bits.",
    "Handle the actions in sequence. First check the preconditions: all {pre_on} bits must be set and all {pre_off} "
    "bits must be unset in the current state. If the check fails, leave the state as it is. If it passes, turn off the "
    "{delete} bits, then turn on the {add} bits.",
    "Apply the actions in order. Each action fires only when its {pre_on} bits are all present and its {pre_off} bits "
    "are all absent in the current state; an action that does not fire changes nothing. A firing action drops its "
    "{delete} bits before setting its {add} bits.",                                                     # held out
)
TRAIN_SURFACE_IDX = (0, 1, 2, 3)
HELDOUT_SURFACE_IDX = 4
HELDOUT_SURFACE = {"para": HELDOUT_SURFACE_IDX, "names": HELDOUT_SURFACE_IDX}   # stage E3 of a surface-diversity run
TAIL = "Names and costs do not change the state."


def _article(w: int) -> str:
    return "an" if str(w).startswith("8") or str(w).startswith("11") or w == 18 else "a"


def names_of(surface) -> dict:
    return FIELD_VARIANTS[0 if not surface else int(surface.get("names", 0))]


def system_text(arm: str, w: int, surface=None) -> str:
    nm = names_of(surface)
    para = PARAPHRASES[0 if not surface else int(surface.get("para", 0))].format(**nm)
    if arm == "bbits":
        state = (f"The state is {_article(w)} {w}-bit flag mask, written as a string of {w} binary digits with the most "
                 f"significant bit first; {nm['pre_on']}, {nm['pre_off']}, {nm['delete']} and {nm['add']} are written "
                 "the same way.")
    else:
        state = f"The state is {_article(w)} {w}-bit flag mask."
    if arm == "dint":
        ret = "Return only the final resulting state as a JSON integer."
    elif arm == "strace":
        ret = (f"Return only a JSON object with the keys cmp, calc and after, in this order: cmp holds on_miss (the "
               f"positions of {nm['pre_on']} bits that are clear, bit 0 being the least significant), off_hit (the "
               f"positions of {nm['pre_off']} bits that are set), viol (how many positions are listed) and ok (true if "
               "the action is applicable); calc is the state the action produces; after is the final resulting state "
               "as a JSON integer.")
    elif arm == "bbits":
        ret = (f"Return only a JSON object with the keys applicable (true or false) and after (the final resulting state "
               f"as a string of {w} binary digits), in this order.")
    elif arm == "ffill":
        ret = ("Return only a JSON object with the keys fill (a run of dots) and after (the final resulting state as a "
               "JSON integer), in this order.")
    else:
        raise ValueError(arm)
    return " ".join((state, para, ret, TAIL))


def bitstr(x: int, w: int) -> str:
    if not 0 <= x < 2 ** w:
        raise ValueError(f"{x} does not fit {w} bits")
    return format(x, f"0{w}b")


def user_text(arm: str, before: int, event: dict, w: int, surface=None) -> str:
    nm = names_of(surface)
    ev = dict(event)
    b = before
    if arm == "bbits":
        b = bitstr(before, w)
        for k in ("add", "delete", "pre_on", "pre_off"):
            ev[k] = bitstr(event[k], w)
    ev = {nm.get(k, k): v for k, v in ev.items()}
    return env.dumps({nm["before"]: b, nm["events"]: [ev]})


def messages(arm: str, before: int, event: dict, w: int, surface=None) -> list:
    return [{"role": "system", "content": system_text(arm, w, surface)},
            {"role": "user", "content": user_text(arm, before, event, w, surface)}]


def render_chat(msgs: list) -> str:
    """Qwen3 chat template with add_generation_prompt=True, enable_thinking=False (== capture.render; the data builder
    checks it byte for byte against T2's stored prompts, the tiny test against tok.apply_chat_template)."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------------------------- gold and heuristics
def applicable(before: int, ev: dict) -> bool:
    return env.applicable(before, ev)


def gold(before: int, ev: dict) -> int:
    return env.step(FAM, before, ev)


def bit_positions(x: int) -> list:
    return [i for i in range(max(x.bit_length(), 1)) if (x >> i) & 1]


def gold_cmp(before: int, ev: dict) -> dict:
    on_miss = bit_positions(ev["pre_on"] & ~before)
    off_hit = bit_positions(ev["pre_off"] & before)
    viol = len(on_miss) + len(off_hit)
    return {"on_miss": on_miss, "off_hit": off_hit, "viol": viol, "ok": viol == 0}


def heuristics(before: int, ev: dict) -> dict:
    b, a, d = before, ev["add"], ev["delete"]
    return {"copy": b, "apply_regardless": (b & ~d) | a, "apply_ignoring_pre": (b & ~d) | a, "int_add": b + a,
            "or_add": b | a, "int_add_del": (b & ~d) + a, "delete_only": b & ~d}


def _apply(before: int, ev: dict) -> int:
    return (before & ~ev["delete"]) | ev["add"]


def _pre_on_ok(before: int, ev: dict) -> bool:
    return ev["pre_on"] & ~before == 0


def _pre_off_ok(before: int, ev: dict) -> bool:
    return ev["pre_off"] & before == 0


# Shortcut policies (descriptive): their accuracy on the training rows (AUDIT.json), their floors on every dev stratum
# (t2bp_dev.SOURCE.json) and their verdicts in the analysis dry run. "pa_" = precondition-aware: inapplicable -> copy,
# applicable -> the named arithmetic (Titan's own reading, R11, is pa_int_add).
SHORTCUT_POLICIES = {
    "pa_int_add": lambda b, e: b if not applicable(b, e) else b + e["add"],
    "pa_int_add_del": lambda b, e: b if not applicable(b, e) else (b & ~e["delete"]) + e["add"],
    "pa_or_add": lambda b, e: b if not applicable(b, e) else b | e["add"],
    "pa_delete_only": lambda b, e: b if not applicable(b, e) else b & ~e["delete"],
    "pre_on_only": lambda b, e: _apply(b, e) if _pre_on_ok(b, e) else b,          # ignores pre_off
    "pre_off_only": lambda b, e: _apply(b, e) if _pre_off_ok(b, e) else b,        # ignores pre_on
    "precount_cue": lambda b, e: b if bin(e["pre_on"] | e["pre_off"]).count("1") >= 2 else _apply(b, e),
    "add0_cue": lambda b, e: b if e["add"] == 0 else _apply(b, e),
    "addset_apply": lambda b, e: _apply(b, e) if (e["add"] and e["add"] & b == e["add"]) else gold(b, e),
}


def shortcut_states(before: int, ev: dict) -> dict:
    return {k: f(before, ev) for k, f in SHORTCUT_POLICIES.items()}


def app_delete(before: int, ev: dict) -> bool:
    """The delete clears a set bit that the add does not restore (L1 P_appdel / T2 app-del, when applicable)."""
    return (ev["delete"] & before & ~ev["add"]) != 0


def violation(before: int, ev: dict) -> str | None:
    miss, hit = ev["pre_on"] & ~before, ev["pre_off"] & before
    if not miss and not hit:
        return None
    return "pre_on" if miss and not hit else ("pre_off" if hit and not miss else "both")


def rtype(before: int, ev: dict) -> str:
    a, d = ev["add"], ev["delete"]
    app = applicable(before, ev)
    d_hit = (d & before) != 0
    if a == 0:
        if not d_hit:
            return "Z_noop"
        return "Za" if app else "Zi"
    if a & before == a:
        a_in = True
    elif a & before == 0:
        a_in = False
    else:
        return "other"
    if app:
        return {(False, False): "A1", (False, True): "A2", (True, False): "A3", (True, True): "A4"}[(a_in, d_hit)]
    if not a_in:
        return "Ib"
    return "Ia" if d_hit else "I_noop"


def payload(before: int, ev: dict) -> str:
    return env.dumps({"before": before, "event": ev})


def opkey(before: int, ev: dict) -> str:
    """FDS generate.operative_key for prerequisite_plan, prefixed with the family (as l1_make_ood.opkey)."""
    return env.dumps([FAM, env.dumps({"before": before, **{x: ev[x] for x in ("add", "delete", "pre_on", "pre_off")}})])


def structural_key(before: int, ev: dict, w: int) -> str:
    """Bit-permutation-invariant signature (descriptive only): the multiset of per-bit (b, add, delete, pre_on, pre_off)."""
    cols = sorted(tuple((x >> i) & 1 for x in (before, ev["add"], ev["delete"], ev["pre_on"], ev["pre_off"])) for i in range(w))
    return json.dumps([w, cols])


# ---------------------------------------------------------------------------------------------- targets
def _dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def trace_text(before: int, ev: dict) -> str:
    c = gold_cmp(before, ev)
    calc = gold(before, ev)
    cmp_txt = _dump({k: c[k] for k in CMP_KEYS})
    return '{"cmp":' + cmp_txt + ',"calc":' + _dump(calc) + ',"after":' + _dump(gold(before, ev)) + "}"


def filler_text(after: int, k: int) -> str:
    return '{"fill":"' + FILL_UNIT * k + '","after":' + _dump(after) + "}"


def fill_k(before: int, ev: dict, count) -> int:
    """The unit count k such that count(filler) == count(trace) exactly (count: text -> number of tokens)."""
    n = count(trace_text(before, ev))
    after = gold(before, ev)
    k = max(1, n - count(filler_text(after, 1)) + 1)
    for _ in range(64):
        c = count(filler_text(after, k))
        if c == n:
            return k
        k += 1 if c < n else -1
        if k < 1:
            break
    raise ValueError(f"no filler length matches the trace token count {n} for {before} {ev}")


def target_text(arm: str, before: int, ev: dict, w: int, count=None) -> str:
    after = gold(before, ev)
    if arm == "dint":
        return _dump(after)
    if arm == "strace":
        return trace_text(before, ev)
    if arm == "bbits":
        return '{"applicable":' + _dump(applicable(before, ev)) + ',"after":"' + bitstr(after, w) + '"}'
    if arm == "ffill":
        if count is None:
            raise ValueError("the filler target needs the tokenizer's token count")
        return filler_text(after, fill_k(before, ev, count))
    raise ValueError(arm)


def render_prediction(arm: str, state: int, before: int, ev: dict, w: int, count=None) -> str:
    """A well-formed answer in the arm's format whose state is `state` (for heuristic floors; mechanism fields follow
    the heuristic: ok/flag = state differs from before, calc = state)."""
    if arm == "dint":
        return _dump(state)
    if arm == "strace":
        c = gold_cmp(before, ev)
        return '{"cmp":' + _dump({k: c[k] for k in CMP_KEYS}) + ',"calc":' + _dump(state) + ',"after":' + _dump(state) + "}"
    if arm == "bbits":
        return '{"applicable":' + _dump(state != before) + ',"after":"' + bitstr(state, w) + '"}'
    if arm == "ffill":
        return filler_text(state, 3)
    raise ValueError(arm)


# ---------------------------------------------------------------------------------------------- parsers and scoring
def _int_ok(v, w: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** w


def parse_dint(text: str, w: int):
    """rcwmt.env.parse_state(prerequisite_plan, text, lenient=True), line for line, with the range [0, 2**w)."""
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        t2 = env._EQ_RHS.sub(r'"\1": \2', t)
        try:
            v = json.loads(t2)
        except (json.JSONDecodeError, ValueError):
            v = env._last_after_value(t2)
            if v is None:
                return None
    if isinstance(v, dict) and "after" in v:
        v = v["after"]
    return v if _int_ok(v, w) else None


class _Dup(ValueError):
    pass


def _pairs_no_dup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise _Dup(k)
        d[k] = v
    return d


def loads_strict(text: str):
    """json.loads that keeps key order and rejects duplicate keys; None if not valid JSON."""
    try:
        return json.loads(text.strip(), object_pairs_hook=_pairs_no_dup)
    except (json.JSONDecodeError, ValueError, _Dup):
        return None


def parse(arm: str, text: str, w: int) -> dict:
    """{'after': int | None, 'well_formed': bool, + mechanism fields}. The decision pipelines read only 'after'."""
    if arm == "dint":
        a = parse_dint(text, w)
        return {"after": a, "well_formed": a is not None}
    d = loads_strict(text)
    keys = {"strace": TRACE_KEYS, "bbits": BBITS_KEYS, "ffill": FILL_KEYS}[arm]
    if not isinstance(d, dict) or tuple(d) != keys:
        return {"after": None, "well_formed": False}
    if arm == "bbits":
        s, fl = d["after"], d["applicable"]
        ok = isinstance(fl, bool) and isinstance(s, str) and len(s) == w and set(s) <= {"0", "1"}
        return {"after": int(s, 2) if ok else None, "well_formed": ok, "flag": fl if isinstance(fl, bool) else None}
    a = d["after"] if _int_ok(d["after"], w) else None
    if arm == "ffill":
        ok = a is not None and isinstance(d["fill"], str)
        return {"after": a if ok else None, "well_formed": ok}
    return {"after": a, "well_formed": a is not None, "cmp": d["cmp"], "calc": d["calc"]}


def parse_after(arm: str, text: str, w: int):
    return parse(arm, text, w)["after"]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def format_ok(arm: str, text: str) -> bool:
    """The answer is in the arm's FORMAT, whatever its value: D-int - the lenient parser finds an integer; S-trace /
    F-filler - the literal keys and an integer 'after'; B-bits - the literal keys, a boolean flag and a non-empty binary
    string. An out-of-range state (e.g. Titan's integer reading 200 + 64 = 264 in an 8-bit task) is a wrong answer in
    the right format, not a format failure (the instrument-validity gate WF counts this, not 'well_formed')."""
    if arm == "dint":
        t = text.strip()
        if t.endswith("."):
            t = t[:-1]
        try:
            v = json.loads(t)
        except (json.JSONDecodeError, ValueError):
            t2 = env._EQ_RHS.sub(r'"\1": \2', t)
            try:
                v = json.loads(t2)
            except (json.JSONDecodeError, ValueError):
                v = env._last_after_value(t2)
        if isinstance(v, dict) and "after" in v:
            v = v["after"]
        return _is_int(v)
    d = loads_strict(text)
    keys = {"strace": TRACE_KEYS, "bbits": BBITS_KEYS, "ffill": FILL_KEYS}[arm]
    if not isinstance(d, dict) or tuple(d) != keys:
        return False
    if arm == "bbits":
        s = d["after"]
        return isinstance(d["applicable"], bool) and isinstance(s, str) and len(s) > 0 and set(s) <= {"0", "1"}
    return _is_int(d["after"]) and (arm != "ffill" or isinstance(d["fill"], str))


def cmp_ok(parsed_cmp, before: int, ev: dict) -> bool:
    g = gold_cmp(before, ev)
    return isinstance(parsed_cmp, dict) and tuple(parsed_cmp) == CMP_KEYS and parsed_cmp == g and \
        all(isinstance(parsed_cmp[k], list) for k in ("on_miss", "off_hit")) and \
        isinstance(parsed_cmp["viol"], int) and not isinstance(parsed_cmp["viol"], bool) and isinstance(parsed_cmp["ok"], bool)


def calc_ok(parsed_calc, before: int, ev: dict) -> bool:
    return isinstance(parsed_calc, int) and not isinstance(parsed_calc, bool) and parsed_calc == gold(before, ev)


def score_row(arm: str, text: str, row: dict) -> dict:
    """Per-row scores: correct (after == gold), copy (after == before), well_formed, + mechanism fields."""
    w, b, ev = row["width"], row["before"], row["event"]
    p = parse(arm, text, w)
    a = p["after"]
    out = {"correct": a is not None and a == row["gold"], "copy": a is not None and a == b, "well_formed": p["well_formed"],
           "format_ok": format_ok(arm, text), "after": a}
    h = heuristics(b, ev)
    out["agrees"] = {k: a is not None and a == v for k, v in h.items()}
    if arm == "strace":
        out["cmp_ok"] = p["well_formed"] and cmp_ok(p.get("cmp"), b, ev)
        out["calc_ok"] = p["well_formed"] and calc_ok(p.get("calc"), b, ev)
    if arm == "bbits":
        out["flag_ok"] = p["well_formed"] and p.get("flag") == applicable(b, ev)
    return out


# ---------------------------------------------------------------------------------------------- audit (training rows)
def audit_rows(rows: list) -> dict:
    """Recompute every training-set property from the canonical rows. ok=False -> the builder refuses to write and
    t2bp_audit.py exits non-zero."""
    n = len(rows)
    errs = []
    acc = Counter()
    for r in rows:
        b, ev, w = r["before"], r["event"], r["width"]
        if w != WIDTH_ID:
            errs.append(f"{r['id']}: training rows are {WIDTH_ID}-bit only")
        g = gold(b, ev)
        if g != r["gold"]:
            errs.append(f"{r['id']}: gold {r['gold']} != env.step {g}")
        if rtype(b, ev) != r["type"]:
            errs.append(f"{r['id']}: type {r['type']} != {rtype(b, ev)}")
        if r["applicable"] != applicable(b, ev):
            errs.append(f"{r['id']}: applicable flag wrong")
        h = heuristics(b, ev)
        if h["apply_ignoring_pre"] != h["apply_regardless"]:
            errs.append(f"{r['id']}: alias heuristics differ")
        for k, v in h.items():
            acc[k] += int(v == g)
    scores = {k: acc[k] / n for k in list(AUDITED) + list(AUDIT_ALIASES) + list(DESCRIPTIVE_HEURISTICS)}
    balance = {k: AUDIT_LO <= scores[k] <= AUDIT_HI for k in AUDITED}
    for alias, base in AUDIT_ALIASES.items():
        balance[alias] = scores[alias] == scores[base]
    types = Counter(r["type"] for r in rows)
    want_types = Counter()
    for t, c in TRAIN_UNPAIRED.items():
        want_types[t] += c
    for t, c in TRAIN_PAIRS.items():
        want_types[t] += c
        want_types[PAIR_INAPP[t]] += c
    # minimal pairs
    by_pair = {}
    for r in rows:
        if r.get("pair_id"):
            by_pair.setdefault(r["pair_id"], []).append(r)
    pair_bad = []
    for pid, mem in sorted(by_pair.items()):
        if len(mem) != 2:
            pair_bad.append(f"{pid}: {len(mem)} members")
            continue
        x, y = sorted(mem, key=lambda r: not r["applicable"])
        ex, ey = x["event"], y["event"]
        diff_keys = [k for k in set(ex) | set(ey) if ex.get(k) != ey.get(k)]
        one_bit = len(diff_keys) == 1 and diff_keys[0] in ("pre_on", "pre_off") and \
            bin(ex[diff_keys[0]] ^ ey[diff_keys[0]]).count("1") == 1
        if not (x["before"] == y["before"] and one_bit and x["applicable"] and not y["applicable"]
                and x["gold"] != y["gold"] and y["flip"] == diff_keys[0] == violation(y["before"], ey)):
            pair_bad.append(pid)
    viol = Counter(violation(r["before"], r["event"]) for r in rows if not r["applicable"])
    quotas = {"minimal_pairs": len(by_pair) - len(pair_bad),
              "int_add_wrong_strict": types["A4"],
              "applicable_delete": sum(1 for r in rows if r["applicable"] and app_delete(r["before"], r["event"])),
              "int_add_wrong_r11": sum(1 for r in rows if r["applicable"] and r["gold"] != r["before"] + r["event"]["add"]),
              "inapp_pre_on": viol["pre_on"], "inapp_pre_off": viol["pre_off"]}
    quota_ok = {k: quotas[k] >= QUOTA_MIN[k] for k in QUOTA_MIN}
    uniq_pay = len({payload(r["before"], r["event"]) for r in rows}) == n
    uniq_ops = len({opkey(r["before"], r["event"]) for r in rows}) == n
    ok = (not errs and all(balance.values()) and dict(types) == dict(+want_types) and not pair_bad and all(quota_ok.values())
          and uniq_pay and uniq_ops and n == N_TRAIN and viol["both"] == 0)
    sc_acc = Counter()
    for r in rows:
        for k, v in shortcut_states(r["before"], r["event"]).items():
            sc_acc[k] += int(v == r["gold"])
    sep = {"pa_int_add": sum(1 for r in rows if r["type"] in ("A2", "A3", "A4", "Za")),
           "pa_int_add_del": sum(1 for r in rows if r["type"] in ("A3", "A4")),
           "pa_or_add": sum(1 for r in rows if r["type"] in ("A2", "A4", "Za"))}
    return {"ok": ok, "n": n, "heuristic_accuracy": scores, "balance_gate": balance,
            "gate": f"each of {list(AUDITED)} in [{AUDIT_LO}, {AUDIT_HI}]; {list(AUDIT_ALIASES)} identical to apply_regardless",
            "types": dict(sorted(types.items())), "types_expected": dict(sorted(want_types.items())), "quotas": quotas,
            "quota_min": QUOTA_MIN, "quota_ok": quota_ok, "minimal_pair_errors": pair_bad[:10],
            "payloads_unique": uniq_pay, "operative_contents_unique": uniq_ops, "violations": dict(viol),
            "row_errors": errs[:10], "cue_ceilings": cue_ceilings(rows),
            "shortcut_accuracy_descriptive": {k: round(v / n, 4) for k, v in sc_acc.items()},
            "rows_separating_precondition_aware_composites_from_the_rule": sep,
            "minimal_pair_twin_artefacts": twin_artefacts(rows)}


def twin_artefacts(rows: list) -> dict:
    """Descriptive: precondition bit counts that occur ONLY on minimal-pair twins (the twin adds one precondition bit, so
    a count above the recorded marginals' maximum marks an inapplicable row on the training rows; none in the dev bank)."""
    pc = lambda x: bin(x).count("1")   # noqa: E731
    base = [r for r in rows if r.get("pair_role") != "inapp"]
    twins = [r for r in rows if r.get("pair_role") == "inapp"]
    out = {}
    for k in ("pre_on", "pre_off"):
        seen = {pc(r["event"][k]) for r in base}
        only = Counter(pc(r["event"][k]) for r in twins if pc(r["event"][k]) not in seen)
        out[f"{k}_popcounts_only_on_twins"] = {str(c): v for c, v in sorted(only.items())}
    out["twin_rows_with_such_a_count"] = sum(
        1 for r in twins if any(pc(r["event"][k]) not in {pc(x["event"][k]) for x in base} for k in ("pre_on", "pre_off")))
    return out


def cue_ceilings(rows: list) -> dict:
    """Descriptive: how well cheap surface features predict applicability (majority lookup per feature value), and the
    best two-branch answer rule 'feature -> copy, else apply'."""
    feats = {
        "add==0": lambda r: r["event"]["add"] == 0,
        "add bit already set": lambda r: r["event"]["add"] != 0 and (r["event"]["add"] & r["before"]) == r["event"]["add"],
        "delete hits a set bit": lambda r: (r["event"]["delete"] & r["before"]) != 0,
        "pre_on==0": lambda r: r["event"]["pre_on"] == 0,
        "pre_off==0": lambda r: r["event"]["pre_off"] == 0,
        "popcount(pre_on)": lambda r: bin(r["event"]["pre_on"]).count("1"),
        "popcount(pre_off)": lambda r: bin(r["event"]["pre_off"]).count("1"),
        "popcount(pre_on|pre_off)": lambda r: bin(r["event"]["pre_on"] | r["event"]["pre_off"]).count("1"),
        "popcount(before)": lambda r: bin(r["before"]).count("1"),
    }
    out = {}
    for name, f in feats.items():
        tab = {}
        for r in rows:
            tab.setdefault(f(r), Counter())[r["applicable"]] += 1
        out[name] = round(sum(max(c.values()) for c in tab.values()) / len(rows), 4)
    rule = sum(int((r["gold"] == r["before"]) if r["event"]["add"] == 0 else (r["gold"] == heuristics(r["before"], r["event"])["apply_regardless"]))
               for r in rows) / len(rows)
    nz = [r for r in rows if r["event"]["add"] != 0]
    out["answer rule 'add==0 -> copy, else apply-regardless'"] = round(rule, 4)
    out["apply-regardless on add!=0 rows"] = round(sum(r["gold"] == heuristics(r["before"], r["event"])["apply_regardless"] for r in nz) / len(nz), 4)
    out["copy on add!=0 rows"] = round(sum(r["gold"] == r["before"] for r in nz) / len(nz), 4)
    aset = [r for r in nz if (r["event"]["add"] & r["before"]) == r["event"]["add"]]
    out["applicable share among add-bit-already-set rows"] = round(sum(r["applicable"] for r in aset) / len(aset), 4) if aset else None
    return out


# ---------------------------------------------------------------------------------------------- files
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_inputs(root: Path = ROOT) -> dict:
    """{repo-relative path: sha256} of every frozen input (FROZEN_CODE + FROZEN_DATA)."""
    return {rel: sha256(root / rel) for rel in FROZEN_CODE + FROZEN_DATA}


def read_rule_record(path: Path | None = None):
    try:
        return json.loads(Path(path or DOCS / SCREEN_RULE_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def frozen_problems(rec, files: dict) -> list:
    """files: {repo-relative name: path on disk}; each file's sha256 must equal the frozen screen-rule record's."""
    if not isinstance(rec, dict) or rec.get("record") != "T2BP_SCREEN_RULE" or not isinstance(rec.get("inputs"), dict):
        return ["docs/t2b/prereq/SCREEN_RULE.json is missing or not a screen-rule record (t2bp_analysis.py --freeze-rule)"]
    bad = []
    for rel, p in files.items():
        want = rec["inputs"].get(rel)
        if want is None:
            bad.append(f"{rel}: not recorded in the frozen screen rule")
        elif not Path(p).is_file() or sha256(Path(p)) != want:
            bad.append(f"{rel}: differs from the frozen screen rule (sha256)")
    return bad


def run_outputs(out_root: Path | None = None) -> list:
    """Every file under runs/t2bp (any run output); empty before the first GPU turn."""
    r = Path(out_root or OUT_ROOT)
    return sorted(str(p.relative_to(r)).replace("\\", "/") for p in r.rglob("*") if p.is_file()) if r.is_dir() else []


def write_doc(name: str, text: str, docs: Path | None = None) -> Path:
    """Kit records under docs/t2b/prereq (AUDIT, DRYRUN, SCORER_SELFTEST, TINY_TEST, GEN_PLAN). Before any runs/t2bp
    output exists a rerun refreshes them (they carry the code hashes they were made with); once an output exists they
    are never overwritten: the new record is written next to the old one with a UTC timestamp suffix."""
    d = Path(docs or DOCS)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if p.exists() and run_outputs():
        p = d / f"{p.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{p.suffix}"
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_dev(path: Path = DEV) -> tuple[list, list]:
    """(one-step rows, chain parents) of the dev screen bank."""
    rows, parents = [], []
    for r in load_jsonl(path):
        (rows if r["kind"] == "onestep" else parents).append(r)
    return rows, parents


def dev_chains(parents: list) -> list:
    return [{"cand_id": c["cand_id"], "parent_id": p["parent_id"], "s0": p["s0"], "events": c["events"], "width": WIDTH_ID}
            for p in parents for c in p["candidates"]]


def chain_gold_states(ch: dict) -> list:
    return env.rollout(FAM, ch["s0"], ch["events"])


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]
