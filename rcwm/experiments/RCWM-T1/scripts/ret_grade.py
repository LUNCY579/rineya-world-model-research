"""Retention-suite graders. Pure functions: no torch, no model, no filesystem, no clock.

PRIMARY grader (decision-bearing): a typed WHOLE-ANSWER grammar modelled on the programme's typed retention contract
(docs/l1/vendor_5b4c22d/experiments/architecture_benchmark_001/TYPED_SCORING_CONTRACT.md and retention.py, read-only):
the whole generated answer is parsed as the item's declared kind and compared exactly; a value scraped out of prose,
a second answer, a hedge, a range or an explanation is wrong. Normalisation, in this order (docs/retention/DESIGN.md
section 4 states the same list):

  N0  EOS termination within the 48-token cap is required (a truncated answer is wrong: "did not terminate").
  N1  Unicode NFKC; typographic quotes, the Unicode minus and en/em dashes become ASCII ' " -; NBSP becomes a space.
  N2  Outer whitespace stripped, inner runs of whitespace (incl. newlines) collapsed to one space.
  N3  Case folded, EXCEPT for case-sensitive kinds (unknown: the literal UNKNOWN; exact items that demand a case; json).
  N4  At most one closing sentence period removed; then at most ONE wrapping pair removed from the whole answer
      (** **, __ __, * *, ` `, " ", ' '); then, only if N4 removed no period yet, one closing period again.
      json only: one surrounding ``` / ```json code fence is removed first.
  N5  At most one prefix from the fixed list 'final answer:', 'answer:', 'the answer is', 'it is', 'it will be'
      (matched case-insensitively); clock answers may instead begin with 'at', weekday/date answers with 'on'.
  N6  The remainder is parsed as the declared kind and must be consumed entirely. Numbers: ASCII digits, optional
      sign, exact rationals (integers, finite decimals, one a/b fraction), commas only in valid groups of three, no
      scientific notation, no number words. A declared unit alias may follow a number; any other suffix is wrong.

Kinds: number (value + allowed unit aliases), clock (minute of day; 24-hour unless am/pm is written), weekday (full
English name), date (full month name + day, either order, optional ordinal suffix, optional 'of', optional year equal
to the item's), num_list / word_list (comma-separated, optional 'and' before the last element; exact order and
multiplicity), ratio ('a:b' or 'a to b', exactly the requested form), kv (exactly the value: code digits or a word),
unknown (exactly the literal UNKNOWN, case-sensitive), exact (one declared string, case-sensitive iff declared),
json (one JSON object equal to the declared object; duplicate keys are wrong).

LENIENT grader (report-only, never decision-bearing): value extraction from free text (last number / time / weekday /
date / ratio, in-vocabulary list order, kv value present with no other passage value, abstention phrases). It exists
to tell format drift from skill loss in the report.
"""
from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction

PREFIXES = ("final answer:", "answer:", "the answer is", "it is", "it will be")
EXTRA_PREFIX = {"clock": ("at",), "weekday": ("on",), "date": ("on",)}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")
MERIDIEM_AM = ("am", "a.m.", "a.m")
MERIDIEM_PM = ("pm", "p.m.", "p.m")
KINDS = ("number", "clock", "weekday", "date", "num_list", "word_list", "ratio", "kv", "unknown", "exact", "json")
CASE_SENSITIVE_KINDS = ("unknown", "json")
WRAPPERS = (("**", "**"), ("__", "__"), ("*", "*"), ("`", "`"), ('"', '"'), ("'", "'"))
ABSTAIN_PHRASES = ("unknown", "not mentioned", "not stated", "does not say", "doesn't say", "not provided",
                   "no information", "cannot be determined", "can't be determined", "not in the passage", "not given",
                   "does not mention", "doesn't mention", "not specified", "not included")

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "−": "-", "–": "-",
                         "—": "-", " ": " "})
_SIGN = r"[+-]?"
_INTBODY = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_INT_RE = re.compile(f"^{_SIGN}{_INTBODY}$")
_DEC_RE = re.compile(f"^{_SIGN}(?:{_INTBODY}\\.\\d+|\\.\\d+)$")
_LEAD_NUM_RE = re.compile(r"^[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?")
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_BARE_HOUR_RE = re.compile(r"^(\d{1,2})$")
_LIST_SEP = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")
_WORD_RE = re.compile(r"^[a-z][a-z'-]*$")
_RATIO_RE = re.compile(r"^([+-]?\d+)\s*(?::|to)\s*([+-]?\d+)$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_ORD = r"(?:st|nd|rd|th)?"
_MONTH_ALT = "|".join(MONTHS)
_DATE_MD = re.compile(rf"^({_MONTH_ALT})\s+(\d{{1,2}}){_ORD}(?:,?\s+(\d{{4}}))?$")
_DATE_DM = re.compile(rf"^(?:the\s+)?(\d{{1,2}}){_ORD}\s+(?:of\s+)?({_MONTH_ALT})(?:,?\s+(\d{{4}}))?$")


class ParseError(ValueError):
    """The answer does not parse as the declared kind: an ordinary wrong answer."""


class GoldError(ValueError):
    """A malformed gold specification: invalidates the item (the generator self-test catches it)."""


# ------------------------------------------------------------------------------------------------ primitives
def parse_rational(token: str) -> Fraction:
    t = token.strip()
    if not t:
        raise ParseError("empty number")
    if "/" in t:
        parts = t.split("/")
        if len(parts) != 2:
            raise ParseError(f"not a single fraction: {token!r}")
        num, den = parts[0].strip(), parts[1].strip()
        if not _INT_RE.match(num) or not _INT_RE.match(den):
            raise ParseError(f"fraction parts must be integers: {token!r}")
        d = int(den.replace(",", ""))
        if d <= 0:
            raise ParseError(f"non-positive denominator: {token!r}")
        return Fraction(int(num.replace(",", "")), d)
    if _INT_RE.match(t) or _DEC_RE.match(t):
        return Fraction(t.replace(",", ""))
    raise ParseError(f"not an exact rational: {token!r}")


def _split_number(text: str) -> tuple[Fraction, str]:
    m = _LEAD_NUM_RE.match(text)
    if not m:
        raise ParseError(f"no leading number in {text!r}")
    return parse_rational(m.group(0)), text[m.end():].strip()


def _basic(text: str) -> str:
    """N1 + N2."""
    t = unicodedata.normalize("NFKC", str(text or "")).translate(_QUOTES)
    return " ".join(t.split())


def _strip_period(t: str) -> tuple[str, bool]:
    return (t[:-1].rstrip(), True) if t.endswith(".") else (t, False)


def _unwrap(t: str) -> str:
    for a, b in WRAPPERS:                     # '**' / '__' are tried before '*' / '_'-free singles
        if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
            inner = t[len(a): len(t) - len(b)].strip()
            if inner:
                return inner
    return t


def _strip_prefix(t: str, kind: str) -> str:
    cands = sorted(PREFIXES + EXTRA_PREFIX.get(kind, ()), key=len, reverse=True)
    low = t.casefold()
    for p in cands:
        if low == p:
            raise ParseError("prefix with no answer")
        if low.startswith(p + " "):
            return t[len(p) + 1:].strip()
        if p.endswith(":") and low.startswith(p):
            return t[len(p):].strip()
    return t


def prepare(text: str, kind: str, case_sensitive: bool) -> str:
    """N1-N5: the answer body that N6 must consume entirely."""
    t = _basic(text)
    if not case_sensitive:
        t = t.casefold()
    if kind == "json":
        m = _FENCE_RE.match(t)
        if m:
            t = m.group(1).strip()
    t, stripped = _strip_period(t)
    u = _unwrap(t)
    if u != t and not stripped:
        u, _ = _strip_period(u)
    t = u.strip()
    t = _strip_prefix(t, kind)
    if not t:
        raise ParseError("empty answer")
    return t


def clock_minutes(text: str) -> int:
    """Programme rule: no meridiem = 24-hour clock; am/pm = 12-hour clock; 'noon' / 'midnight'."""
    if text == "midnight":
        return 0
    if text == "noon":
        return 12 * 60
    meridiem = None
    for suf in MERIDIEM_AM + MERIDIEM_PM:
        if text.endswith(" " + suf) or (text.endswith(suf) and text[: -len(suf)][-1:].isdigit()):
            meridiem = "am" if suf in MERIDIEM_AM else "pm"
            text = text[: -len(suf)].strip()
            break
    m = _CLOCK_RE.match(text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    elif meridiem and _BARE_HOUR_RE.match(text):
        h, mi = int(text), 0
    else:
        raise ParseError(f"not a clock time: {text!r}")
    if not 0 <= mi <= 59:
        raise ParseError("minute out of range")
    if meridiem is None:
        if not 0 <= h <= 23:
            raise ParseError("24-hour clock out of range")
        return h * 60 + mi
    if not 1 <= h <= 12:
        raise ParseError("12-hour clock out of range")
    h = (0 if h == 12 else h) if meridiem == "am" else (12 if h == 12 else h + 12)
    return h * 60 + mi


def _no_dup_pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ParseError("duplicate JSON key")
    return dict(pairs)


# ------------------------------------------------------------------------------------------------ gold validation
def validate_gold(g: dict) -> None:
    k = g.get("kind")
    if k not in KINDS:
        raise GoldError(f"unknown kind {k!r}")
    try:
        if k == "number":
            parse_rational(str(g["value"]))
            if not all(isinstance(u, str) and u == u.casefold() and u.strip() == u for u in g.get("units", [])):
                raise GoldError("units must be casefolded strings")
        elif k == "clock":
            if not (isinstance(g["minute"], int) and 0 <= g["minute"] < 1440):
                raise GoldError("minute of day out of range")
        elif k == "weekday":
            if g["value"] not in WEEKDAYS:
                raise GoldError("weekday")
        elif k == "date":
            if not (1 <= g["month"] <= 12 and 1 <= g["day"] <= 31):
                raise GoldError("date")
        elif k == "num_list":
            if not (g["value"] and all(isinstance(x, int) for x in g["value"])):
                raise GoldError("num_list")
        elif k == "word_list":
            if not (g["value"] and all(_WORD_RE.match(x) for x in g["value"])):
                raise GoldError("word_list")
        elif k == "ratio":
            if not (isinstance(g["a"], int) and isinstance(g["b"], int)):
                raise GoldError("ratio")
        elif k == "kv":
            if g["mode"] not in ("code", "word") or not str(g["value"]).strip():
                raise GoldError("kv")
        elif k == "exact":
            if not isinstance(g["value"], str) or not g["value"].strip():
                raise GoldError("exact")
        elif k == "json":
            if not isinstance(g["value"], dict):
                raise GoldError("json")
    except (KeyError, TypeError, ParseError) as e:
        raise GoldError(f"malformed gold {g}: {e}") from e


# ------------------------------------------------------------------------------------------------ primary grader
def parse_answer(text: str, g: dict):
    """Parse the whole answer as the gold's kind (raises ParseError). Returns a comparable value."""
    k = g["kind"]
    cs = k in CASE_SENSITIVE_KINDS or (k == "exact" and bool(g.get("case")))
    body = prepare(text, k, cs)
    if k == "number":
        v, rest = _split_number(body)
        if rest and rest not in g.get("units", []):
            raise ParseError(f"unit {rest!r} is not declared for this item")
        return v
    if k == "clock":
        return clock_minutes(body)
    if k == "weekday":
        if body not in WEEKDAYS:
            raise ParseError(f"not a weekday name: {body!r}")
        return body
    if k == "date":
        m = _DATE_MD.match(body)
        if m:
            mon, day, yr = MONTHS.index(m.group(1)) + 1, int(m.group(2)), m.group(3)
        else:
            m = _DATE_DM.match(body)
            if not m:
                raise ParseError(f"not a date: {body!r}")
            mon, day, yr = MONTHS.index(m.group(2)) + 1, int(m.group(1)), m.group(3)
        if yr is not None and int(yr) != g.get("year"):
            raise ParseError("year differs")
        return (mon, day)
    if k == "num_list":
        parts = _LIST_SEP.split(body)
        out = []
        for p in parts:
            v = parse_rational(p)
            if v.denominator != 1 or "." in p or "/" in p:
                raise ParseError(f"not an integer: {p!r}")
            out.append(int(v))
        return out
    if k == "word_list":
        parts = _LIST_SEP.split(body)
        if not all(_WORD_RE.match(p) for p in parts):
            raise ParseError("not a word list")
        return parts
    if k == "ratio":
        m = _RATIO_RE.match(body)
        if not m:
            raise ParseError(f"not a ratio: {body!r}")
        return (int(m.group(1)), int(m.group(2)))
    if k == "kv":
        return body
    if k == "unknown":
        return body
    if k == "exact":
        return body
    if k == "json":
        try:
            obj = json.loads(body, object_pairs_hook=_no_dup_pairs)
        except json.JSONDecodeError as e:
            raise ParseError(f"not JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ParseError("not a JSON object")
        return obj
    raise GoldError(f"unhandled kind {k!r}")


def target_of(g: dict):
    k = g["kind"]
    if k == "number":
        return parse_rational(str(g["value"]))
    if k == "clock":
        return g["minute"]
    if k == "weekday":
        return g["value"]
    if k == "date":
        return (g["month"], g["day"])
    if k in ("num_list",):
        return list(g["value"])
    if k == "word_list":
        return [w.casefold() for w in g["value"]]
    if k == "ratio":
        return (g["a"], g["b"])
    if k == "kv":
        return str(g["value"]) if g["mode"] == "code" else str(g["value"]).casefold()
    if k == "unknown":
        return "UNKNOWN"
    if k == "exact":
        return g["value"] if g.get("case") else g["value"].casefold()
    if k == "json":
        return g["value"]
    raise GoldError(k)


def grade(g: dict, text: str, terminated: bool = True) -> dict:
    """Primary (decision-bearing) grade of one answer: {'correct', 'reason', 'parsed'}."""
    validate_gold(g)
    rec = {"correct": False, "reason": None, "parsed": None}
    if not terminated:
        rec["reason"] = "did not terminate"
        return rec
    try:
        got = parse_answer(text, g)
    except ParseError as e:
        rec["reason"] = str(e)[:120]
        return rec
    want = target_of(g)
    if g["kind"] == "number":
        ok = got == want
    elif g["kind"] == "json":
        ok = got == want and all(type(got[k]) is type(want[k]) for k in want)
    else:
        ok = got == want
    rec["parsed"] = str(got)[:120]
    rec["correct"] = bool(ok)
    if not ok:
        rec["reason"] = "value mismatch"
    return rec


# ------------------------------------------------------------------------------------------------ lenient (report-only)
_NUM_ANY = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_TIME_ANY = re.compile(r"(\d{1,2}):(\d{2})(?:\s*(a\.?m\.?|p\.?m\.?))?")
_RATIO_ANY = re.compile(r"(\d+)\s*(?::|to)\s*(\d+)")
_ABBR_DAYS = {d[:3]: d for d in WEEKDAYS}
_ABBR_MON = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}


def _numbers(t: str) -> list:
    out = []
    for m in _NUM_ANY.finditer(t):
        try:
            out.append(Fraction(m.group(0).replace(",", "")))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def grade_lenient(g: dict, text: str) -> bool:
    """Report-only value extraction (see the module doc). Never used by any decision."""
    validate_gold(g)
    t = _basic(text)
    low = t.casefold()
    k = g["kind"]
    if k == "number":
        ns = _numbers(low.replace("%", " "))
        return bool(ns) and ns[-1] == parse_rational(str(g["value"]))
    if k == "clock":
        ms = list(_TIME_ANY.finditer(low))
        if not ms:
            return False
        m = ms[-1]
        mer = (m.group(3) or "").replace(".", "")
        try:
            v = clock_minutes(f"{int(m.group(1))}:{m.group(2)}" + (f" {mer}" if mer else ""))
        except ParseError:
            return False
        return v == g["minute"]
    if k == "weekday":
        found = [w for w in re.findall(r"[a-z]+", low) if w in WEEKDAYS or w in _ABBR_DAYS]
        return bool(found) and _ABBR_DAYS.get(found[-1], found[-1]) == g["value"]
    if k == "date":
        hits = []
        for m in re.finditer(r"([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b|(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)", low):
            name, day = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            mon = MONTHS.index(name) + 1 if name in MONTHS else _ABBR_MON.get(name)
            if mon:
                hits.append((mon, int(day)))
        return bool(hits) and hits[-1] == (g["month"], g["day"])
    if k == "num_list":
        return [int(x) for x in _numbers(low) if x.denominator == 1] == list(g["value"])
    if k == "word_list":
        vocab = {w.casefold() for w in g.get("vocab", g["value"])}
        seq = [w for w in re.findall(r"[a-z][a-z'-]*", low) if w in vocab]
        return seq == [w.casefold() for w in g["value"]]
    if k == "ratio":
        ms = list(_RATIO_ANY.finditer(low))
        return bool(ms) and (int(ms[-1].group(1)), int(ms[-1].group(2))) == (g["a"], g["b"])
    if k == "kv":
        def has(v):
            return re.search(r"(?<![\w])" + re.escape(str(v).casefold()) + r"(?![\w])", low) is not None
        if "unknown" in low:
            return False
        return has(g["value"]) and not any(has(d) for d in g.get("distractors", []))
    if k == "unknown":
        if not any(p in low for p in ABSTAIN_PHRASES):
            return False
        return not any(re.search(r"(?<![\w])" + re.escape(str(v).casefold()) + r"(?![\w])", low)
                       for v in g.get("passage_values", []))
    if k == "exact":
        v = g["value"] if g.get("case") else g["value"].casefold()
        src = t if g.get("case") else low
        return re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", src) is not None
    if k == "json":
        m = re.search(r"\{.*?\}", t, re.S)
        if not m:
            return False
        try:
            return json.loads(m.group(0)) == g["value"]
        except json.JSONDecodeError:
            return False
    raise GoldError(k)


# ------------------------------------------------------------------------------------------------ canonical answers
def canonical_answer(g: dict) -> str:
    """The answer string the primary grader must accept (generator self-test, tiny tests)."""
    k = g["kind"]
    if k == "number":
        return str(g["value"])
    if k == "clock":
        return f"{g['minute'] // 60:02d}:{g['minute'] % 60:02d}"
    if k == "weekday":
        return g["value"].capitalize()
    if k == "date":
        return f"{MONTHS[g['month'] - 1].capitalize()} {g['day']}"
    if k == "num_list":
        return ", ".join(str(x) for x in g["value"])
    if k == "word_list":
        return ", ".join(g["value"])
    if k == "ratio":
        return f"{g['a']}:{g['b']}"
    if k == "kv":
        return str(g["value"])
    if k == "unknown":
        return "UNKNOWN"
    if k == "exact":
        return g["value"]
    if k == "json":
        return json.dumps(g["value"])
    raise GoldError(k)
