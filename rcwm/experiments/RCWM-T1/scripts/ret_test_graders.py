"""CPU tests of the retention graders, the generator, the freshness harvest, minting, the freeze and the runner's --plan.
No torch, no model, no GPU, no lease. One extra process (the PYTHONHASHSEED independence check). Threads 2.

  python -B scripts/ret_test_graders.py [--no-record]
Writes docs/retention/GRADER_TEST.json (unless --no-record; refused once the kit is frozen).

  1  primary grader on adversarial answer strings (every kind; wrappers, prefixes, units, hedges, ranges, explanations,
     number words, scientific notation, bad comma groups, fractions (lowest terms only, never for a whole number),
     meridiem rules, case rules, duplicate JSON keys, type mismatches, unterminated answers); the lenient grader (the
     go rule's second grader) on its own adversarial set (verbose right answers, hedges / alternatives, distractors,
     abstention phrases, case, JSON); malformed golds rejected
  2  generator determinism: same seed -> same bytes (twice in-process, once in a child process with another
     PYTHONHASHSEED); a new seed -> zero shared user turns and zero shared passage sentences
  3  INDEPENDENT GOLD RE-DERIVATION: a solver that reads only the prompt text (its own regexes, unit table, datetime
     calendar, sorting, latest-value record semantics, absence checks for the unanswerable twins) recomputes every
     gold of four full dev-plan draws (both pre-registered dev seeds, and seed 7 with surface diversity off / on)
  4  tense: every record question template (every theme x field name) and every generated question about a corrected
     record is in the present tense (no did / was / were / had; a present-tense verb)
  5  strata: every cell's subtype mix is the fixed round-robin mix, identical across seeds and settings
  6  gold self-test (canonical answers 100% correct, negatives 0%) on three seeds; passages 150-300 Qwen3 tokens
  7  surface diversity: OFF -> one system prompt; ON -> several, golds still self-test, recorded in the seal
  8  freshness harvest on a fake corpus: an exact user turn (other case/whitespace) and an embedded gated sentence are
     found; a file named HSKILL_CONFIRM_V2* holding a user turn is skipped by name and never stat()ed or opened
  9  mint / verify in a test root: create-only; seed reuse refused; per-setting dev seeds; confirm refused without the
     dev selection of its setting, with a surface-diversity mismatch or with either dev seed; the confirm draw holds
     exactly the selected cells; verify regenerates the bytes and reports the interpreter; a tampered draw is refused
 10  freeze in a copied root: binds every kit file and records the tokenizer pins and the interpreter; refused when
     DESIGN.md lacks the rules verbatim or the tokenizer differs from the pins; a registered dev mint needs the fixed
     dev seed; the dev-draw preview (both settings, one harvest pass) is reproducible and mintable
 11  ret_run.py --plan in-process: torch not imported; dev 2,025 prompts, confirm 675; GPU-h per turn and per arm
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # never "-1" (DECISIONS I-21)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import builtins  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from fractions import Fraction as F  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import ret_common as RC  # noqa: E402
import ret_gen as G  # noqa: E402
import ret_grade as RG  # noqa: E402

RESULTS: list = []
COUNTS = {"grader_cases": 0}


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)


NUM = {"kind": "number", "value": "350", "units": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"]}
DEC = {"kind": "number", "value": "2.5", "units": []}
BIG = {"kind": "number", "value": "1234", "units": []}
PCT = {"kind": "number", "value": "25", "units": ["%", "per cent", "percent"]}
CLK = {"kind": "clock", "minute": 13 * 60 + 15}
MID = {"kind": "clock", "minute": 0}
WDY = {"kind": "weekday", "value": "thursday"}
DAT = {"kind": "date", "month": 4, "day": 5, "year": 2027}
NLS = {"kind": "num_list", "value": [-3, 7, 9]}
WLS = {"kind": "word_list", "value": ["apple", "banana", "cherry"], "vocab": ["apple", "banana", "cherry"]}
RAT = {"kind": "ratio", "a": 2, "b": 3}
KVC = {"kind": "kv", "mode": "code", "value": "4721", "distractors": ["1111", "2222"]}
KVW = {"kind": "kv", "mode": "word", "value": "Lisbon", "distractors": ["Oslo", "Lima"]}
UNK = {"kind": "unknown", "passage_values": ["4721", "1111"]}
UPP = {"kind": "exact", "value": "PINE", "case": True}
YES = {"kind": "exact", "value": "yes", "case": False}
LET = {"kind": "exact", "value": "e", "case": True}
HYP = {"kind": "exact", "value": "P-I-N-E", "case": True}
JSN = {"kind": "json", "value": {"item": "pine", "count": 3}}

CASES = [   # (gold, text, expected primary); terminated unless the text is wrapped in a tuple (text, False)
    (NUM, "350", True), (NUM, "350.", True), (NUM, "  350 \n", True), (NUM, "350 cm", True), (NUM, "350cm", True),
    (NUM, "350 centimeters", True), (NUM, "350 CM", True), (NUM, "**350**", True), (NUM, "`350`", True),
    (NUM, "The answer is 350.", True), (NUM, "Answer: 350", True), (NUM, "350.0", True), (NUM, "+350", True),
    (NUM, "350/1", False), (NUM, "700/2", False), (NUM, "**350**.", True), (NUM, "**350.**", True),
    (NUM, "Final answer: 350", True),
    (NUM, "350 mm", False), (NUM, "350 m", False), (NUM, "3.5e2", False), (NUM, "350 or 351", False),
    (NUM, "about 350", False), (NUM, "~350", False), (NUM, "350!", False), (NUM, "350 (3.5 x 100)", False),
    (NUM, "three hundred fifty", False), (NUM, "35O", False), (NUM, "3 50", False), (NUM, "350,", False),
    (NUM, "350..", False), (NUM, "-350", False), (NUM, "", False), (NUM, "Answer:", False), (NUM, "351", False),
    (NUM, "350\n\nExplanation: 3.5 x 100.", False), (NUM, "It's 350", False), (NUM, "350-360", False),
    (NUM, ("350", False), False), (NUM, "350 cm.", True), (NUM, "350 centimetres long", False),
    (DEC, "2.5", True), (DEC, "2.50", True), (DEC, "5/2", True), (DEC, "10/4", False), (DEC, "5 / 2", True),
    (DEC, ".25e1", False), (DEC, "2,5", False), (DEC, "2½", False), (DEC, "2.49", False), (DEC, "2 1/2", False),
    (BIG, "1,234", True), (BIG, "1234", True), (BIG, "1,23,4", False), (BIG, "12,34", False), (BIG, "1 234", False),
    (BIG, "2468/2", False),
    (PCT, "25%", True), (PCT, "25 %", True), (PCT, "25 percent", True), (PCT, "25 per cent", True), (PCT, "0.25", False),
    (PCT, "25 percentage points", False), (PCT, "50/2", False),
    (CLK, "13:15", True), (CLK, "1:15 PM", True), (CLK, "1:15 pm", True), (CLK, "1:15pm", True), (CLK, "1:15 p.m.", True),
    (CLK, "At 13:15", True), (CLK, "**13:15**", True), (CLK, "13:15.", True), (CLK, "1:15", False), (CLK, "13:15 pm", False),
    (CLK, "13.15", False), (CLK, "1315", False), (CLK, "13:15 or 13:20", False), (CLK, "quarter past one", False),
    (CLK, "13:15 hours", False), (CLK, "It ends at 13:15", False),
    (MID, "00:00", True), (MID, "0:00", True), (MID, "midnight", True), (MID, "12:00 am", True), (MID, "24:00", False),
    (MID, "12:00", False),
    (WDY, "Thursday", True), (WDY, "thursday.", True), (WDY, "On Thursday", True), (WDY, "**Thursday**", True),
    (WDY, "THURSDAY", True), (WDY, "Thu", False), (WDY, "Thursday or Friday", False), (WDY, "It's Thursday", False),
    (WDY, "Thursday, 5 days later", False), (WDY, "Friday", False),
    (DAT, "April 5", True), (DAT, "April 5th", True), (DAT, "5 April", True), (DAT, "the 5th of April", True),
    (DAT, "April 5, 2027", True), (DAT, "april 5", True), (DAT, "5 April 2027", True), (DAT, "Apr 5", False),
    (DAT, "April 5, 2026", False), (DAT, "4/5", False), (DAT, "April 6", False), (DAT, "April 5 or 6", False),
    (DAT, "2027-04-05", False),
    (NLS, "-3, 7, 9", True), (NLS, "-3,7,9", True), (NLS, "-3, 7 and 9", True), (NLS, "-3, 7, and 9", True),
    (NLS, "−3, 7, 9", True), (NLS, "-3 7 9", False), (NLS, "9, 7, -3", False), (NLS, "-3, 7", False),
    (NLS, "-3, 7, 9, 9", False), (NLS, "[-3, 7, 9]", False), (NLS, "-3, 7.0, 9", False), (NLS, "-3, 7, 9.", True),
    (NLS, "-3, 14/2, 9", False),
    (WLS, "apple, banana, cherry", True), (WLS, "Apple, Banana, Cherry", True), (WLS, "apple, banana and cherry", True),
    (WLS, "banana, apple, cherry", False), (WLS, "apple banana cherry", False), (WLS, "apple, banana, cherry, date", False),
    (WLS, "1. apple 2. banana 3. cherry", False),
    (RAT, "2:3", True), (RAT, "2 : 3", True), (RAT, "2 to 3", True), (RAT, "The answer is 2:3.", True),
    (RAT, "4:6", False), (RAT, "3:2", False), (RAT, "2/3", False), (RAT, "2:3:1", False),
    (KVC, "4721", True), (KVC, "4721.", True), (KVC, "**4721**", True), (KVC, "Answer: 4721", True),
    (KVC, "4,721", False), (KVC, "The code is 4721", False), (KVC, "4721 or 1111", False), (KVC, "1111", False),
    (KVC, "47210", False), (KVC, "UNKNOWN", False),
    (KVW, "Lisbon", True), (KVW, "lisbon", True), (KVW, "LISBON.", True), (KVW, "the Lisbon depot", False),
    (KVW, "Lisbon depot", False), (KVW, "Oslo", False),
    (UNK, "UNKNOWN", True), (UNK, "UNKNOWN.", True), (UNK, "**UNKNOWN**", True), (UNK, "Answer: UNKNOWN", True),
    (UNK, "`UNKNOWN`", True), (UNK, " UNKNOWN \n", True), (UNK, "\"UNKNOWN\"", True), (UNK, "unknown", False),
    (UNK, "Unknown", False), (UNK, "UNKNOWN.\nThe passage does not say.", False), (UNK, "UNKNOWN or 4721", False),
    (UNK, "UNK", False), (UNK, "I don't know", False), (UNK, "N/A", False), (UNK, "4721", False),
    (UNK, ("UNKNOWN", False), False),
    (UPP, "PINE", True), (UPP, "PINE.", True), (UPP, "**PINE**", True), (UPP, "\"PINE\"", True), (UPP, "pine", False),
    (UPP, "Pine", False), (UPP, "PINE PINE", False), (UPP, "PINES", False), (UPP, "P I N E", False),
    (YES, "yes", True), (YES, "Yes.", True), (YES, "YES", True), (YES, "yes, it is", False), (YES, "yes or no", False),
    (YES, "no", False), (YES, "Yes!", False), (YES, "y", False),
    (LET, "e", True), (LET, "e.", True), (LET, "E", False), (LET, "e is the first letter", False),
    (HYP, "P-I-N-E", True), (HYP, "PINE", False), (HYP, "P - I - N - E", False), (HYP, "p-i-n-e", False),
    (JSN, '{"item": "pine", "count": 3}', True), (JSN, '{"count":3,"item":"pine"}', True),
    (JSN, '```json\n{"item": "pine", "count": 3}\n```', True), (JSN, '`{"item": "pine", "count": 3}`', True),
    (JSN, "{'item': 'pine', 'count': 3}", False), (JSN, '{"item": "pine", "count": "3"}', False),
    (JSN, '{"item": "pine", "count": 3, "x": 1}', False), (JSN, '{"item": "pine", "count": 3} Done', False),
    (JSN, '{"item": "pine", "item": "pine", "count": 3}', False), (JSN, '{"item": "Pine", "count": 3}', False),
    (JSN, '{"item": "pine", "count": 3.0}', False), (JSN, '{"item": "pine", "count": true}', False),
    (JSN, '["pine", 3]', False),
]
LENIENT = [   # (gold, text, expected lenient): the go rule's second grader (format-neutral, alternatives wrong)
    (NUM, "The answer is 350 cm because 3.5 x 100 = 350.", True), (NUM, "350 or 351", False), (NUM, "About 350", True),
    (NUM, "It is 350 centimeters long.", True), (NUM, "3.5 m is 350 cm", True), (NUM, "351", False),
    (NUM, "350 cm, or 3.5 m", False), (NUM, "either 350 or 360", False), (NUM, "Approximately 350.", True),
    (NUM, "", False), (NUM, "three hundred fifty", False), (NUM, "350 (350 cm)", True),
    (NUM, "I computed 3.5 x 100 = 360, so 350", True), (NUM, "350 is wrong; the answer is 360", False),
    (PCT, "25%", True), (PCT, "It is 25 percent.", True), (PCT, "0.25", False),
    (DEC, "2.5 hours", True), (DEC, "5/2", False),
    (CLK, "It ends at 1:15 PM.", True), (CLK, "It ends at 1:15.", False), (CLK, "13:15", True),
    (CLK, "The shift ends at 13:15, or 1:15 pm", True), (CLK, "13:15 or 13:20", False), (CLK, "from 11:00 to 13:15", True),
    (MID, "It ends at midnight (00:00).", True), (MID, "12:00 am", True),
    (WDY, "10 days after Monday is Thursday", True), (WDY, "Thu", True), (WDY, "Thursday or Friday", False),
    (WDY, "It is Thursday, not Friday", False), (WDY, "THURSDAY", True), (WDY, "The day is Friday.", False),
    (WDY, "Monday + 10 days = Thursday", True),
    (DAT, "It falls on Apr 5.", True), (DAT, "5 April 2027", True), (DAT, "April 5 or April 6", False),
    (DAT, "May 5", False), (DAT, "The date will be the 5th of April.", True),
    (NLS, "[-3, 7, 9]", True), (NLS, "-3, 9, 7", False), (NLS, "Sorted: -3, 7, 9.", True),
    (NLS, "-3, 7, 9 (3 numbers)", False),
    (WLS, "1. apple 2. banana 3. cherry", True), (WLS, "Banana, Apple, Cherry", False),
    (WLS, "apple, banana, cherry, apple", False), (WLS, "Here they are: apple, banana, cherry", True),
    (RAT, "simplest form: 2 to 3", True), (RAT, "2:3 or 4:6", False), (RAT, "4:6 simplifies to 2:3", True),
    (RAT, "3:2", False),
    (KVC, "The code is 4721.", True), (KVC, "4721 or 1111", False), (KVC, "UNKNOWN (4721?)", False),
    (KVC, "It was 1111, now 4721", False), (KVC, "47210", False), (KVC, "4721", True),
    (KVW, "the Lisbon depot", True), (KVW, "Lisbon, not Oslo", False), (KVW, "LISBON", True), (KVW, "Lisboa", False),
    (UNK, "Unknown - the passage does not say.", True), (UNK, "It is not mentioned; maybe 4721", False),
    (UNK, "UNKNOWN", True), (UNK, "The passage doesn't say.", True), (UNK, "I cannot be sure", False),
    (UNK, "Not stated, but 1111 is listed", False),
    (UPP, "Sure: PINE", True), (UPP, "pine", False), (UPP, "PINE or PINES", False), (UPP, "PINES", False),
    (YES, "Yes, Ama has more.", True), (YES, "yes or no", False), (YES, "No.", False), (YES, "Yes - no doubt", False),
    (LET, "The first letter is e.", True), (LET, "E", False),
    (HYP, "P-I-N-E", True), (HYP, "PINE", False),
    (JSN, 'Here: {"item": "pine", "count": 3}', True),
    (JSN, '{"item": "pine", "count": 3} or {"item": "pine", "count": 4}', False),
    (JSN, '{"item": "pine", "count": "3"}', False),
]
BAD_GOLDS = [{"kind": "number", "value": "3e2"}, {"kind": "number", "value": "5", "units": ["CM"]},
             {"kind": "clock", "minute": 1440}, {"kind": "weekday", "value": "Thursday"}, {"kind": "nope"},
             {"kind": "word_list", "value": ["Apple"]}, {"kind": "kv", "mode": "x", "value": "1"},
             {"kind": "json", "value": [1]}, {"kind": "ratio", "a": "2", "b": 3}]


def t_graders():
    bad = []
    for g, t, want in CASES:
        text, term = (t if isinstance(t, tuple) else (t, True))
        got = RG.grade(g, text, term)["correct"]
        COUNTS["grader_cases"] += 1
        if got != want:
            bad.append((g["kind"], text, want, got))
    check(f"primary grader: {len(CASES)} adversarial cases", not bad, bad[:6])
    badl = []
    for g, t, want in LENIENT:
        got = RG.grade_lenient(g, t)
        COUNTS["grader_cases"] += 1
        if got != want:
            badl.append((g["kind"], t, want, got))
    check(f"lenient grader (decision-bearing co-condition): {len(LENIENT)} adversarial cases", not badl, badl[:6])
    rej = 0
    for g in BAD_GOLDS:
        try:
            RG.validate_gold(g)
        except RG.GoldError:
            rej += 1
    check(f"malformed golds rejected {rej}/{len(BAD_GOLDS)}", rej == len(BAD_GOLDS))
    COUNTS.update(primary_cases=len(CASES), lenient_cases=len(LENIENT), malformed_golds=len(BAD_GOLDS))


# ------------------------------------------------------------------------------------------------ independent solver
UNIT = {"meters": ("len", F(1)), "centimeters": ("len", F(1, 100)), "millimeters": ("len", F(1, 1000)),
        "kilometers": ("len", F(1000)), "kilograms": ("mass", F(1000)), "grams": ("mass", F(1)),
        "liters": ("vol", F(1000)), "milliliters": ("vol", F(1)), "hours": ("time", F(3600)),
        "minutes": ("time", F(60)), "seconds": ("time", F(1)), "days": ("time", F(86400)), "weeks": ("time", F(604800)),
        "feet": ("imp", F(12)), "inches": ("imp", F(1)), "yards": ("imp", F(36)), "pounds": ("wt", F(16)),
        "ounces": ("wt", F(1))}
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
       "December"]
ORDS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
NAME = r"[A-Z][a-z]+ [A-Z][a-z]+"
FN = r"[a-z]+(?: [a-z]+)?"
# record / correction sentences of the six passage themes, written independently of ret_gen.THEMES
REC = [
    rf"(?P<who>{NAME}) has locker (?P<n>\d+), and its {FN} is (?P<v>\d{{4}})\.",
    rf"(?P<who>{NAME}) drives van (?P<n>\d+) and starts every route at the (?P<v>[A-Z][a-z]+) {FN}\.",
    rf"(?P<who>{NAME}) rents {FN} (?P<n>\d+) and is growing (?P<v>[a-z]+) this season\.",
    rf"(?P<who>{NAME}) borrowed {FN} (?P<n>\d+), which is painted (?P<v>[a-z]+)\.",
    rf"(?P<who>{NAME}) logged sample tray (?P<n>\d+) in {FN} (?P<v>\d{{3}})\.",
    rf"(?P<who>{NAME}) booked {FN} (?P<n>\d+) for (?P<v>[a-z]+) practice\.",
]
CORR = [
    rf"Later, the {FN} of locker (?P<n>\d+), used by (?P<who>{NAME}), was changed to (?P<v>\d{{4}})\.",
    rf"Later, (?P<who>{NAME}) and van (?P<n>\d+) were moved to the (?P<v>[A-Z][a-z]+) {FN}\.",
    rf"Later, (?P<who>{NAME}) switched {FN} (?P<n>\d+) to growing (?P<v>[a-z]+)\.",
    rf"Later, {FN} (?P<n>\d+), borrowed by (?P<who>{NAME}), was repainted (?P<v>[a-z]+)\.",
    rf"Later, sample tray (?P<n>\d+) from (?P<who>{NAME}) was moved to {FN} (?P<v>\d{{3}})\.",
    rf"Later, (?P<who>{NAME}) changed the {FN} (?P<n>\d+) booking to (?P<v>[a-z]+) practice\.",
]
PAST = re.compile(r"\b(did|was|were|had)\b")
PRESENT = re.compile(r"\b(is|are|does|do|holds|opens)\b")


def present_tense(q: str) -> bool:
    return PAST.search(q) is None and PRESENT.search(q) is not None


def _m(rx, s):
    m = re.search(rx, s)
    if not m:
        raise ValueError(f"no match {rx!r} in {s[:120]!r}")
    return m


def solve_arith(u):
    sents = re.split(r"(?<=[.?])\s+", u)
    m = re.fullmatch(r"([A-Z][a-z]+) has (\d+) ([a-z]+)\.", sents[0])
    who, cur, obj = m.group(1), F(int(m.group(2))), m.group(3)
    for s in sents[1:]:
        if s.startswith("How many"):
            if not re.fullmatch(rf"How many {obj} does {who} have now\?", s):
                raise ValueError(s)
            return ("number", cur)
        if m := re.fullmatch(rf"{who} (?:buys|finds) (\d+) more {obj}\.", s):
            cur += int(m.group(1))
        elif m := re.fullmatch(rf"[A-Z][a-z]+ gives {who} (\d+) more {obj}\.", s):
            cur += int(m.group(1))
        elif m := re.fullmatch(rf"{who} (?:gives (\d+) {obj} to [A-Z][a-z]+|uses (\d+) of the {obj}|sells (\d+) {obj})\.", s):
            cur -= int(next(g for g in m.groups() if g))
        elif s == f"Then {who} doubles the number of {obj}.":
            cur *= 2
        elif s == f"Then {who} triples the number of {obj}.":
            cur *= 3
        elif s == f"Then the number of {obj} {who} has becomes four times as large.":
            cur *= 4
        elif m := re.fullmatch(rf"{who} splits the {obj} into (\d+) equal groups and keeps only one group\.", s):
            cur /= int(m.group(1))
        else:
            raise ValueError(s)
    raise ValueError("no question")


def solve(r):
    u = RC.user_text(r["messages"])
    c, st = r["category"], r["stratum"]
    if c == "arith_word":
        return solve_arith(u)
    if c == "unit_conv":
        m = _m(r" (\d+(?:\.\d+)?) (" + "|".join(UNIT) + r")\b", u)
        x, src = F(m.group(1)), m.group(2)
        dst = _m(r" in (" + "|".join(UNIT) + r")\?", u).group(1)
        if UNIT[src][0] != UNIT[dst][0]:
            raise ValueError("dimension")
        return ("number", x * UNIT[src][1] / UNIT[dst][1])
    if c == "date_time":
        if st in ("clock_add", "clock_midnight"):
            m = _m(r"starts at (\d\d):(\d\d) and lasts (?:(\d+) hours?)?(?: and )?(?:(\d+) minutes)?\. At what time", u)
            t = datetime.datetime(2027, 1, 1, int(m.group(1)), int(m.group(2))) + datetime.timedelta(
                hours=int(m.group(3) or 0), minutes=int(m.group(4) or 0))
            return ("clock", t.hour * 60 + t.minute)
        if st == "duration_between":
            m = _m(r"runs from (\d\d):(\d\d) to (\d\d):(\d\d) on the same day", u)
            a = datetime.datetime(2027, 1, 1, int(m.group(1)), int(m.group(2)))
            b = datetime.datetime(2027, 1, 1, int(m.group(3)), int(m.group(4)))
            return ("number", F(int((b - a).total_seconds() // 60)))
        if st in ("weekday_add", "weekday_back"):
            m = _m(r"is on a ([A-Z][a-z]+)\. What day of the week is it (\d+) days (later|earlier)\?", u)
            d0 = datetime.date(2027, 1, 4) + datetime.timedelta(days=WD.index(m.group(1)))   # 2027-01-04 is a Monday
            n = int(m.group(2)) * (1 if m.group(3) == "later" else -1)
            return ("weekday", WD[(d0 + datetime.timedelta(days=n)).weekday()].lower())
        if st == "date_add":
            m = _m(r"In 2027, .* is on ([A-Z][a-z]+) (\d+)\. What date is (\d+) days later\?", u)
            d = datetime.date(2027, MON.index(m.group(1)) + 1, int(m.group(2))) + datetime.timedelta(days=int(m.group(3)))
            if d.year != 2027:
                raise ValueError("year")
            return ("date", (d.month, d.day))
    if c == "count_sort":
        if st == "count_word":
            m = _m(r'list of words: ([a-z, ]+)\. How many times does the word "([a-z]+)" appear', u)
            return ("number", F(m.group(1).split(", ").count(m.group(2))))
        if st == "count_gt":
            m = _m(r"whiteboard: ([-\d, ]+)\. How many of them are greater than (-?\d+)\?", u)
            return ("number", F(sum(int(x) > int(m.group(2)) for x in m.group(1).split(", "))))
        if st.startswith("sort_num"):
            m = _m(r"whiteboard: ([-\d, ]+)\. Sort them from (smallest to largest|largest to smallest)\.", u)
            return ("num_list", sorted((int(x) for x in m.group(1).split(", ")), reverse=m.group(2).startswith("largest")))
        if st == "sort_words":
            m = _m(r"list of words: ([a-z, ]+)\. Sort the words in alphabetical order\.", u)
            return ("word_list", sorted(m.group(1).split(", ")))
    if c == "pct_ratio":
        if st in ("pct_of", "pct_of_hard"):
            m = _m(r"total of (\d+) coins\. What is ([\d.]+)% of that total", u)
            return ("number", F(m.group(1)) * F(m.group(2)) / 100)
        if st == "ratio_scale":
            m = _m(r", (\d+) [a-z]+ cost (\d+) coins\. At the same price for each, how many coins do (\d+) [a-z]+ cost\?", u)
            return ("number", F(int(m.group(2)), int(m.group(1))) * int(m.group(3)))
        if st == "pct_change":
            m = _m(r"costs (\d+) coins\. The price goes (up|down) by (\d+)%\. What is the new price", u)
            return ("number", F(m.group(1)) * (100 + (1 if m.group(2) == "up" else -1) * int(m.group(3))) / 100)
        if st == "what_pct":
            m = _m(r"answered (\d+) of (\d+) quiz questions correctly\. What percentage", u)
            return ("number", F(100 * int(m.group(1)), int(m.group(2))))
        if st == "ratio_simplify":
            m = _m(r"paint in the ratio (\d+):(\d+)\. Write this ratio in simplest form\.", u)
            a, b = int(m.group(1)), int(m.group(2))
            return ("ratio", (a // math.gcd(a, b), b // math.gcd(a, b)))
        if st == "successive":
            m = _m(r"costs (\d+) coins\. Its price rises by (\d+)%, and then the new price falls by (\d+)%\.", u)
            return ("number", F(m.group(1)) * (100 + int(m.group(2))) / 100 * (100 - int(m.group(3))) / 100)
        if st == "ratio_split":
            m = _m(rf"({NAME}) and {NAME} share (\d+) coins in the ratio (\d+):(\d+)\. How many coins does "
                   rf"({NAME}) get\?", u)
            if m.group(1) != m.group(5):
                raise ValueError("who")
            return ("number", F(int(m.group(2)) * int(m.group(3)), int(m.group(3)) + int(m.group(4))))
    if c == "format_follow":
        if st == "upper_echo":
            return ("exact", _m(r'only the word "([a-z]+)" written in capital letters', u).group(1).upper())
        if st == "yes_no":
            m = _m(rf"({NAME}) has (\d+) ([a-z]+) and ({NAME}) has (\d+) \3\. Does \1 have more \3 than \4\?", u)
            return ("exact", "yes" if int(m.group(2)) > int(m.group(5)) else "no")
        if st in ("nth_word", "last_word"):
            seq = _m(r'sequence of words: "([a-z ]+)"', u).group(1).split()
            if st == "last_word":
                return ("exact", seq[-1])
            return ("exact", seq[ORDS[_m(r"only the ([a-z]+) word of that sequence", u).group(1)] - 1])
        if st == "first_letter":
            return ("exact", _m(r'first letter of the word "([a-z]+)", as a single lowercase letter', u).group(1)[0])
        if st == "reverse":
            return ("exact", _m(r'Write the word "([a-z]+)" backwards in lowercase', u).group(1)[::-1])
        if st == "hyphen_spell":
            return ("exact", "-".join(_m(r'Spell the word "([a-z]+)" in capital letters', u).group(1).upper()))
        if st == "json":
            m = _m(r'key "item" set to the string "([a-z]+)" and the key "count" set to the number (\d+)\.', u)
            return ("json", {"item": m.group(1), "count": int(m.group(2))})
    if c in ("kv_recall", "unknown_twin"):
        return solve_passage(r, u)
    raise ValueError(f"unhandled {c}/{st}")


def solve_passage(r, u):
    paras = u.split("\n\n")
    passage = paras[0]
    q = next(s for s in re.split(r"(?<=[.?])\s+", paras[-1]) if s.endswith("?"))
    recs = {}
    for sent in re.split(r"(?<=\.)\s+", passage):
        m = next((mm for rx in REC if (mm := re.fullmatch(rx, sent))), None)
        if m:
            if m["who"] in recs:
                raise ValueError("duplicate record name")
            recs[m["who"]] = {"n": m["n"], "v": m["v"], "corrected": False}
            continue
        m = next((mm for rx in CORR if (mm := re.fullmatch(rx, sent))), None)
        if m:
            rec = recs[m["who"]]
            if rec["n"] != m["n"]:
                raise ValueError("correction names another record number")
            rec["v"], rec["corrected"] = m["v"], True             # latest value
            continue
        if re.search(r"\d", sent):
            raise ValueError(f"unparsed passage sentence {sent!r}")
    names = re.findall(NAME, q)
    asked = [n for n in names if n in recs]
    info = {"question": q, "corrected": False}
    if not asked:                                                # absent person or near-miss name: unanswerable
        if not names:
            raise ValueError(f"no name in {q!r}")
        info["absent_name"] = names[0]
        return ("unknown", None), info
    who = asked[0]
    info["corrected"] = recs[who]["corrected"]
    m = re.fullmatch(rf"What is {who}'s ([a-z' ]+)\?", q)
    if m and m.group(1).casefold() not in passage.casefold():   # an attribute the passage never mentions
        info["absent_attribute"] = m.group(1)
        return ("unknown", None), info
    return ("kv", recs[who]["v"]), info


def gold_value(g):
    k = g["kind"]
    return {"number": lambda: ("number", F(g["value"])), "clock": lambda: ("clock", g["minute"]),
            "weekday": lambda: ("weekday", g["value"]), "date": lambda: ("date", (g["month"], g["day"])),
            "num_list": lambda: ("num_list", list(g["value"])), "word_list": lambda: ("word_list", list(g["value"])),
            "ratio": lambda: ("ratio", (g["a"], g["b"])), "exact": lambda: ("exact", g["value"]),
            "json": lambda: ("json", g["value"]), "kv": lambda: ("kv", str(g["value"])),
            "unknown": lambda: ("unknown", None)}[k]()


def rederive(rows) -> dict:
    bad, fails, tense_bad, n_corr, n = [], [], [], 0, 0
    for r in rows:
        try:
            got = solve(r)
        except Exception as e:  # noqa: BLE001
            fails.append((r["pid"], repr(e)[:120]))
            continue
        if isinstance(got[0], tuple):
            got, info = got
            if info["corrected"]:
                n_corr += 1
                if not present_tense(info["question"]):
                    tense_bad.append((r["pid"], info["question"]))
        n += 1
        if got != gold_value(r["gold"]):
            bad.append((r["pid"], got, gold_value(r["gold"])))
    return {"checked": n, "mismatches": len(bad), "solver_failures": len(fails), "corrected_questions": n_corr,
            "corrected_not_present_tense": len(tense_bad), "examples": (bad + fails + tense_bad)[:4]}


def t_generator():
    plan = G.dev_plan()
    a, _ = G.generate(RC.DEV_SEED, "dev", plan)
    b, _ = G.generate(RC.DEV_SEED, "dev", plan)
    sa, sb = (RC.sha256_bytes(G.serialize(x).encode()) for x in (a, b))
    check("same seed -> same bytes (in-process)", sa == sb, sa[:16])
    code = ("import sys; sys.path.insert(0, r'%s'); import ret_gen as G, ret_common as RC; "
            "r,_ = G.generate(RC.DEV_SEED, 'dev', G.dev_plan()); print(RC.sha256_bytes(G.serialize(r).encode()))"
            % str(ROOT / "scripts"))
    env = {**os.environ, "PYTHONHASHSEED": "12345", "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=300)
    check("same seed -> same sha256 in a child process with PYTHONHASHSEED=12345", out.stdout.strip() == sa,
          out.stdout.strip()[:16] + " " + out.stderr[-200:])
    COUNTS["dev_seed_sha256_in_memory"] = sa
    c, _ = G.generate(RC.DEV_SEED + 1, "dev", plan)
    ua = {RC.norm_text(RC.user_text(r["messages"])) for r in a}
    uc = {RC.norm_text(RC.user_text(r["messages"])) for r in c}
    pa, pc = G.passage_sentences(a, G.GATE_WORDS), G.passage_sentences(c, G.GATE_WORDS)
    ia, ic = set(G.item_unique_sentences(a, G.GATE_WORDS)), set(G.item_unique_sentences(c, G.GATE_WORDS))
    oa, oc = {(r["category"], r["op"]) for r in a}, {(r["category"], r["op"]) for r in c}
    COUNTS["new_seed_overlap"] = {"user_turns": len(ua & uc), "passage_sentences": len(pa & pc),
                                  "item_unique_sentences_ge8_report": len(ia & ic), "operative_report": len(oa & oc),
                                  "n_prompts": len(a)}
    coll = Counter()
    for s in range(9001, 9011):                      # report-only: how often would the F1 gate refuse a new draw?
        for r in G.generate(s, "dev", plan)[0]:
            if RC.norm_text(RC.user_text(r["messages"])) in ua:
                coll[r["category"]] += 1
    COUNTS["user_turn_collisions_10_new_seeds_vs_dev"] = dict(coll)
    check("new seeds vs the dev draw: at most 1 shared user turn over 10 full new draws (the mint gate refuses any; "
          "report-only rate)", sum(coll.values()) <= 1, dict(coll))
    check("dev plan: 1,800 items / 2,025 prompts; 75 items per cell", len({r["item"] for r in a}) == 1800 and len(a) == 2025)
    # 3-5 independent gold re-derivation, tense, strata
    draws = {"dev_sd0": a, "dev_sd1": G.generate(RC.DEV_SEED_SD, "dev", plan, diversity=True)[0],
             "seed7_sd0": G.generate(7, "dev", plan)[0], "seed7_sd1": G.generate(7, "dev", plan, diversity=True)[0]}
    ov = G.overlap(draws["dev_sd0"], draws["dev_sd1"])
    check("the two pre-registered dev draws share no user turn and no passage sentence (both mintable)",
          ov["user_turn_overlap"] == 0 and ov["passage_sentence_overlap"] == 0, ov)
    COUNTS["dev_sd0_vs_dev_sd1_overlap"] = ov
    rd = {k: rederive(v) for k, v in draws.items()}
    COUNTS["independent_rederivation"] = rd
    check("independent solver re-derives every gold of four draws (2,025 prompts each; both dev seeds, seed 7 off/on)",
          all(v["checked"] == 2025 and v["mismatches"] == 0 and v["solver_failures"] == 0 for v in rd.values()),
          {k: (v["checked"], v["mismatches"], v["solver_failures"], v["examples"][:1]) for k, v in rd.items()})
    check("every generated question about a corrected record is in the present tense",
          all(v["corrected_not_present_tense"] == 0 and v["corrected_questions"] > 0 for v in rd.values()),
          {k: (v["corrected_questions"], v["corrected_not_present_tense"]) for k, v in rd.items()})
    tmpl_bad = [(t, q) for t, th in G.THEMES.items() for q in th["q"] for fn in th["fn"]
                if not present_tense(q.format(who="Ama Okafor", fn=fn))]
    check("every record question template (6 themes x 2 templates x 3 field names) is in the present tense",
          not tmpl_bad, tmpl_bad[:3])
    want = {}
    for cat, lv in G.STRATA.items():
        for lev, st in lv.items():
            for i, s in enumerate(st):
                want[f"{cat}|L{lev}|{s}"] = RC.N_PER_CELL // len(st) + (1 if i < RC.N_PER_CELL % len(st) else 0)
    strata = {k: G.describe(v)["strata"] for k, v in draws.items()}
    check("strata: every cell's subtype mix is the fixed round-robin mix in every draw (seed- and setting-independent)",
          all(s == want for s in strata.values()), [k for k, s in strata.items() if s != want])
    COUNTS["strata_per_cell"] = want
    ok_self, ok_tok, mm = True, True, []
    for seed in (RC.DEV_SEED, 7, 20261199):
        rows = a if seed == RC.DEV_SEED else (draws["seed7_sd0"] if seed == 7 else G.generate(seed, "dev", plan)[0])
        st = G.gold_selftest(rows)
        ok_self &= st["ok"]
        ta = G.token_audit(rows)
        ok_tok &= ta["ok"]
        mm.append((ta["passage_tokens"]["min"], ta["passage_tokens"]["max"]))
        COUNTS.setdefault("selftest", []).append({k: st[k] for k in ("positives", "positives_correct", "negatives",
                                                                     "negatives_rejected")})
    check("gold self-test 100% / 0% on three seeds", ok_self, COUNTS["selftest"])
    check("passages within 150-300 Qwen3 tokens (pinned tokenizer) on three seeds", ok_tok, mm)
    COUNTS["passage_token_range_3_seeds"] = mm
    d_off, d_on = draws["seed7_sd0"], draws["seed7_sd1"]
    sys_off = {r["messages"][0]["content"] for r in d_off}
    sys_on = {r["messages"][0]["content"] for r in d_on}
    fn_on = {w for r in d_on if r["category"] == "kv_recall" for th in G.THEMES.values() for w in th["fn"][1:]
             if w in RC.user_text(r["messages"])}
    ta_on = G.token_audit(d_on)
    check("surface diversity OFF -> one system prompt; ON -> several, field-name variants used, golds self-test, "
          "passages 150-300 tokens",
          len(sys_off) == 1 and len(sys_on) >= 3 and len(fn_on) >= 6 and G.gold_selftest(d_on)["ok"] and ta_on["ok"],
          (len(sys_off), len(sys_on), sorted(fn_on), ta_on["passage_tokens"]["min"], ta_on["passage_tokens"]["max"]))
    check("surface diversity is part of the cell seed (ON != OFF bytes)", G.serialize(d_off) != G.serialize(d_on))


def t_freshness():
    tmp = Path(tempfile.mkdtemp(prefix="ret_fresh_"))
    try:
        rows, _ = G.generate(5, "dev", {"kv_recall": [1], "arith_word": [1]}, n=5)
        u0 = RC.user_text(rows[0]["messages"])
        kv = next(r for r in rows if r["category"] == "kv_recall")
        gated = sorted(G.item_unique_sentences([kv], G.GATE_WORDS))[0]
        corp = tmp / "corpus"
        corp.mkdir()
        (corp / "a.json").write_text(json.dumps({"messages": [{"role": "user", "content": "  " + u0.upper() + "  "}]}),
                                     encoding="utf-8")
        (corp / "b.jsonl").write_text(json.dumps({"text": "Intro words here. " + gated.capitalize() + " More text."}) + "\n",
                                      encoding="utf-8")
        res_name = corp / "HSKILL_CONFIRM_V2_fake.jsonl"
        res_name.write_text(json.dumps({"content": RC.user_text(rows[1]["messages"])}) + "\n", encoding="utf-8")
        touched = []
        real_open, real_stat, real_rt = builtins.open, Path.stat, Path.read_text

        def spy_open(f, *a, **k):
            touched.append(("open", str(f)))
            return real_open(f, *a, **k)

        def spy_stat(self, *a, **k):
            touched.append(("stat", str(self)))
            return real_stat(self, *a, **k)

        def spy_rt(self, *a, **k):
            touched.append(("read_text", str(self)))
            return real_rt(self, *a, **k)

        builtins.open, Path.stat, Path.read_text = spy_open, spy_stat, spy_rt
        try:
            res = G.freshness_scan(rows, sources=(("fake", corp, False),), verbose=False)
        finally:
            builtins.open, Path.stat, Path.read_text = real_open, real_stat, real_rt
        reserved_touched = [t for t in touched if "HSKILL_CONFIRM_V2" in t[1]]
        check("harvest finds an exact user turn (other case / whitespace): F1 = 1", res["F1_user_turn_hits"] == 1,
              res["F1_user_turn_hits"])
        check("harvest finds an embedded gated sentence: F2 >= 1", res["F2_gate_sentence_hits"] >= 1, res["F2_gate_sentence_hits"])
        check("HSKILL_CONFIRM_V2* skipped by name, never stat()ed or opened",
              res["sources"]["fake"]["skipped_by_name_never_opened"] == ["HSKILL_CONFIRM_V2_fake.jsonl"]
              and not reserved_touched and res["skipped_by_name_all_reserved"], reserved_touched)
        check("a gated hit fails the scan", res["ok"] is False)
        multi = G.freshness_scan_multi({"x": rows, "y": G.generate(6, "dev", {"arith_word": [1]}, n=5)[0]},
                                       sources=(("fake", corp, False),), verbose=False)
        check("one harvest pass judges several draws separately (preview)", multi["x"]["F1_user_turn_hits"] == 1
              and multi["y"]["F1_user_turn_hits"] == 0 and multi["y"]["ok"] is True)
        check("the BWM bundle is a freshness source", any(n == "bwm_bundle" for n, _, _ in G.SCAN_SOURCES))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _refused(name, fn):
    try:
        fn()
        check(name, False)
    except RC.RetStop as e:
        check(name, True, str(e)[:80])


def t_mint():
    tmp = Path(tempfile.mkdtemp(prefix="ret_mint_"))
    try:
        (tmp / "docs" / "retention" / "draws").mkdir(parents=True)
        s1 = G.mint("dev", seed=101, scan=False, root=tmp, require_freeze=False, n=6)
        p1 = tmp / s1["file"]
        check("mint writes draw + seal; sha matches; the seal records the interpreter",
              p1.is_file() and RC.sha256_file(p1) == s1["file_sha256"]
              and s1["python"]["version"] == platform.python_version())
        v = G.verify(p1)
        check("verify regenerates the draw byte for byte and reports the interpreter",
              v["ok"] and v["regenerates_identically"] and v["python_same"], v)
        ssd = G.mint("dev", diversity=True, scan=False, root=tmp, require_freeze=False, n=6)
        check("the surface-diversity dev draw uses its own pre-registered seed (20261125)",
              ssd["seed"] == RC.DEV_SEED_SD == 20261125 and ssd["surface_diversity"] is True
              and ssd["seed_source"] == "fixed DEV_SEED_SD", (ssd["seed"], ssd["seed_source"]))
        _refused("seed reuse refused", lambda: G.mint("dev", seed=101, scan=False, root=tmp, require_freeze=False, n=6))
        _refused("registered mint refused without a kit freeze", lambda: G.mint("dev", seed=102, scan=False, root=tmp))
        _refused("confirm refused without DEV_SELECTION.json",
                 lambda: G.mint("confirm", seed=103, scan=False, root=tmp, require_freeze=False, n=6))
        sel = {"selected": {"arith_word": 2, "unknown_twin": 3, "format_follow": 1}, "excluded": [], "surface_diversity": False}
        (tmp / RC.selection_rel(False)).write_text(json.dumps(sel), encoding="utf-8")
        _refused("a surface-diversity confirm refused without DEV_SELECTION_SD.json (the OFF selection does not govern it)",
                 lambda: G.mint("confirm", seed=104, diversity=True, scan=False, root=tmp, require_freeze=False, n=6))
        (tmp / RC.selection_rel(True)).write_text(json.dumps(sel), encoding="utf-8")       # says surface_diversity False
        _refused("confirm refused when the selection's recorded setting differs",
                 lambda: G.mint("confirm", seed=105, diversity=True, scan=False, root=tmp, require_freeze=False, n=6))
        for s in (RC.DEV_SEED, RC.DEV_SEED_SD):
            _refused(f"confirm refused with dev seed {s}",
                     lambda: G.mint("confirm", seed=s, scan=False, root=tmp, require_freeze=False, n=6))
        s2 = G.mint("confirm", scan=False, root=tmp, require_freeze=False, n=6)
        rows2, _ = G.load_draw(tmp / s2["file"])
        check("confirm draw: fresh OS-entropy seed, exactly the selected cells, cross-draw overlap 0",
              s2["seed_source"].startswith("OS entropy") and
              {(r["category"], r["level"]) for r in rows2} == {("arith_word", 2), ("unknown_twin", 3), ("format_follow", 1)}
              and s2["cross_draw"]["F1_user_turn_overlap"] == 0 and s2["selection_sha256"] is not None, s2["seed"])
        clash = G.generate(202, "dev", G.dev_plan(), False, 6)[0]          # an earlier draw holding seed 202's prompts
        (tmp / "docs/retention/draws/dev_999.jsonl").write_text(G.serialize(clash), encoding="utf-8")
        (tmp / "docs/retention/draws/dev_999.SEAL.json").write_text(json.dumps(
            {"seed": 999, "file": "docs/retention/draws/dev_999.jsonl"}), encoding="utf-8")
        _refused("the cross-draw gate refuses a draw that shares user turns with an earlier draw",
                 lambda: G.mint("dev", seed=202, scan=False, root=tmp, require_freeze=False, n=6))
        p2 = tmp / s2["file"]
        txt = p2.read_text(encoding="utf-8")
        p2.write_text(txt.replace("UNKNOWN", "UNKNOWN!", 1), encoding="utf-8")
        _refused("a tampered draw is refused", lambda: G.load_draw(p2))
        COUNTS["mint_test_seconds"] = s1["mint_seconds"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_freeze():
    import ret_analysis as RA                          # torch-free; the freeze checks its rules against DESIGN.md
    tmp = Path(tempfile.mkdtemp(prefix="ret_freeze_"))
    try:
        for rel in RC.FROZEN_CODE + RC.FROZEN_DOCS:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, tmp / rel)
        (tmp / "docs" / "retention" / "draws").mkdir(parents=True, exist_ok=True)
        cur = {k: v for k, v in RC.code_sha(tmp).items() if v}
        for rel in RC.TEST_RECORDS:
            (tmp / rel).write_text(json.dumps({"ok": True, "code_sha256": cur}), encoding="utf-8")
        design_bytes = (tmp / "docs/retention/DESIGN.md").read_bytes()
        design = design_bytes.decode("utf-8")
        check("DESIGN.md states RULE_DEV and RULE_CONFIRM verbatim", RA.RULE_DEV in design and RA.RULE_CONFIRM in design)
        real_pins = dict(RC.TOKENIZER_PINS)
        RC.TOKENIZER_PINS["merges.txt"] = "0" * 64
        try:
            _refused("freeze refused when the tokenizer on disk differs from the pins",
                     lambda: G.freeze("test approval 2026-10-09", root=tmp))
        finally:
            RC.TOKENIZER_PINS.clear()
            RC.TOKENIZER_PINS.update(real_pins)
        (tmp / "docs/retention/DESIGN.md").write_bytes(design.replace(RA.RULE_CONFIRM[:200], "edited ").encode("utf-8"))
        _refused("freeze refused when DESIGN.md does not state the rules verbatim",
                 lambda: G.freeze("test approval 2026-10-09", root=tmp))
        (tmp / "docs/retention/DESIGN.md").write_bytes(design_bytes)            # byte-identical (no newline translation)
        rec = G.freeze("test approval 2026-10-09", root=tmp)
        ok, errs, _ = RC.verify_freeze(tmp)
        check("freeze binds every kit file + test record, the tokenizer pins, the chat template and the interpreter",
              ok and set(RC.FROZEN_CODE + RC.FROZEN_DOCS + RC.TEST_RECORDS) <= set(rec["files"])
              and rec["tokenizer_pins"] == RC.TOKENIZER_PINS and rec["chat_template_sha256"] == RC.CHAT_TEMPLATE_SHA
              and rec["rules_verbatim_in_design"] and rec["python"]["version"] == platform.python_version(), errs)
        _refused("a second freeze is refused (create-only)", lambda: G.freeze("test approval 2026-10-09", root=tmp))
        _refused("a registered dev mint needs the pre-registered dev seed",
                 lambda: G.mint("dev", seed=555, scan=False, root=tmp, n=3))
        empty = tmp / "empty_corpus"
        empty.mkdir()
        recs = G.preview(sources=(("empty", empty, False),), scan=True, record=False, n=6)
        same = all(r["bytes_sha256"] == RC.sha256_bytes(G.serialize(G.generate(RC.DEV_SEEDS[sd], "dev", G.dev_plan(),
                                                                                       sd, 6)[0]).encode("utf-8"))
                   for sd, r in recs.items())
        check("preview: both pre-registered dev draws, reproducible bytes, zero overlap between them, ok",
              set(recs) == {False, True} and same and all(r["ok"] for r in recs.values())
              and recs[True]["draw_id"] == "dev_20261125", {sd: r["overlap_with_other_dev_draw"] for sd, r in recs.items()})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_plan():
    import argparse
    import ret_run as R
    ns = argparse.Namespace(mode=None, draw=None, record=False)
    rc = R.cmd_plan(ns, {}, {})
    dev = R.plan_for(R.Cfg(draw=Path("x"), mode="dev"), R.design_prompt_counts()["dev"])
    con = R.plan_for(R.Cfg(draw=Path("x"), mode="confirm"), R.design_prompt_counts()["confirm_all_categories"])
    check("--plan imports no torch; dev 2,025 prompts, confirm 675; one turn each",
          rc == 0 and "torch" not in sys.modules and dev["n_prompts"] == 2025 and con["n_prompts"] == 675
          and len(dev["turns_planning"]) == 1 and len(con["turns_planning"]) == 1)
    COUNTS["plan"] = {"dev_gpu_h": [dev["gpu_h_planning"], dev["gpu_h_optimistic"]],
                      "confirm_gpu_h_3_arms": [con["gpu_h_planning"], con["gpu_h_optimistic"]],
                      "per_arm_h": [con["per_arm_h_planning"], con["per_arm_h_optimistic"]], "lp_disk_gb": con["lp_disk_gb"]}
    short = R.simulate_turns(R.Cfg(draw=Path("x"), mode="confirm"), R.counts(675, R.Cfg(draw=Path("x"), mode="confirm")),
                             R.RATE_PLAN, cap_h=0.9)
    check("D-22 rule splits the confirm job when the cap is short", len(short) >= 2, short)


def main() -> int:
    t0 = time.time()
    for fn in (t_graders, t_generator, t_freshness, t_mint, t_freeze, t_plan):
        print(f"[{fn.__name__}]", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(e))
    ok = all(r["ok"] for r in RESULTS)
    print(f"{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass ({time.time() - t0:.0f}s)")
    print(json.dumps({k: COUNTS[k] for k in ("primary_cases", "lenient_cases", "malformed_golds", "new_seed_overlap")
                      if k in COUNTS}))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                      for k, v in COUNTS.get("independent_rederivation", {}).items()}))
    if "--no-record" not in sys.argv:
        RC.need(not RC.FREEZE.exists(), "the kit is frozen: GRADER_TEST.json is a frozen record")
        rec = {"ok": ok, "n_checks": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "counts": COUNTS,
               "checks": RESULTS, "recorded_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "code_sha256": {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_common.py", "scripts/ret_grade.py",
                                                                      "scripts/ret_gen.py", "scripts/ret_run.py",
                                                                      "scripts/ret_analysis.py",
                                                                      "scripts/ret_test_graders.py",
                                                                      "docs/retention/DESIGN.md")}}
        (RC.DOCS / "GRADER_TEST.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
