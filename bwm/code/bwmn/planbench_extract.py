"""PlanBench Blocksworld Task 1 (plan generation): instances, colour <-> letter mapping, free text -> PDDL plan.

Pure functions over the vendored assets (bench/planbench): importing this module sets no environment variables,
imports no torch and writes nothing. Used by scripts/planbench_qwen.py (local comparators) and its tests.

Scorers (free-text answer -> PDDL plan, then local VAL):
- extract_plan (primary): rule-based, no LM. Colour names -> PDDL letters via a per-instance mapping inferred by
  aligning the query's initial/goal description with the parsed PDDL init/goal (infer_mapping). Lines are split into
  clauses; a clause contributes actions only if it begins (after list markers such as "1." / "1)" / "(a)" and filler
  words such as "first", "then", "next", "I will") with an action verb: unstack | stack | pick up | put down |
  put/place/set/lower X on Y (stack) or on the table / down (put-down) | take/remove/lift/grab X (from Y) (pick-up /
  unstack). "move" is a composite, not a domain action, and is not extracted. "it"/"the block" resolve to the held
  block (after a grasp) or the last colour named in the line. Markdown table rows are read as lines (step-number
  cells dropped; a cell that is a complete action stands alone, other columns are commentary). If the answer contains
  plan headings ("Plan:", "Actions in the plan:", "### Final Answer", "[PLAN]"), only the region after the last plan
  heading that contains actions is used (up to the next non-plan heading such as "Verification:"), so step-by-step
  narration followed by a recap is not double counted; otherwise every action line is used in order. A line that
  carries actions is a plan heading only if the text before its colon is a bare label ("Plan:", "Final answer:",
  "Therefore, the final plan is:"), never a sentence ("... to complete the plan: stack the red block on ...").
  A heading inside a numbered list whose numbering continues across it (2. ... / Next: / 3. ...), or one that says it
  continues the plan ("Finally, to complete the plan:"), neither starts nor ends a plan region.
  Lines indented under an action-bearing line are commentary on that step and contribute no actions, except under a
  group header (an action-bearing line ending in ':' whose indented list items carry >= 2 other actions; then the
  header's own summary actions are dropped). Underspecified steps (missing argument) are dropped and logged in
  extract_info.issues (as PlanBench's own line extractor does); colours not in the instance are kept as-is (VAL
  rejects them).
- text_to_plan_blocksworld (second scorer, variant "pb_raw"): a faithful port of PlanBench's own reference parser
  (LLMs-Planning, MIT), applied to the same answer text; "pb_md" applies it after markdown-only normalisation
  (pb_markdown_text). Reproduces the published extracted_llm_plan of every rule-extracted model (see tests).

Instances: load_instances (PlanBench query + PDDL problem + mapping), stratified_subset (seeded, strata = block count x
optimal plan length).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import blocks as B

RES = B.BENCH / "results"
SETS = {"generated_basic_3": "blocksworld_3", "generated_basic": "blocksworld"}  # instance dir -> results subdir
SET_ALIAS = {"all": ("generated_basic_3", "generated_basic"), "bw3": ("generated_basic_3",), "bw": ("generated_basic",)}
TASK_FILE = {"zero_shot": "task_1_plan_generation_zero_shot.json", "one_shot": "task_1_plan_generation.json"}

# PlanBench configs/blocksworld.yaml encoded_objects (used only to break ties between automorphic alignments and as a
# last-resort fallback; the mapping is inferred per instance).
COLORS = ("red", "blue", "orange", "yellow", "white", "magenta", "black", "cyan", "green", "violet", "silver", "gold")
CANON = dict(zip(COLORS, "abcdefghijkl"))

# ----------------------------------------------------------------------------- colour <-> letter alignment

_F_ON = re.compile(r"the (\w+) block is on top of the (\w+) block")
_F_TABLE = re.compile(r"the (\w+) block is on the table")
_F_CLEAR = re.compile(r"the (\w+) block is clear")
_F_HOLD = re.compile(r"the hand is currently holding (?:the )?(\w+) block")


def nl_facts(text: str) -> frozenset:
    t = text.lower()
    f = {("on", a, b) for a, b in _F_ON.findall(t)} | {("ontable", a) for a in _F_TABLE.findall(t)}
    f |= {("clear", a) for a in _F_CLEAR.findall(t)} | {("holding", a) for a in _F_HOLD.findall(t)}
    if "the hand is empty" in t:
        f.add(("handempty",))
    return frozenset(f)


def query_facts(query: str) -> tuple[frozenset, frozenset]:
    """Initial and goal facts (colour names) of the LAST [STATEMENT] in a PlanBench query (one-shot prompts have two)."""
    st = query.rsplit("[STATEMENT]", 1)[-1].lower()
    m = re.search(r"as initial conditions i have that,?(.*?)my goal is to have that(.*?)(?:\n|what is the plan|\[plan|$)",
                  st, re.S)
    if not m:
        raise ValueError("cannot locate initial conditions / goal in query")
    return nl_facts(m.group(1)), nl_facts(m.group(2))


def _alignments(cinit, cgoal, objs, init, goal, cap: int = 64) -> list[dict]:
    colors = sorted({x for f in cinit | cgoal for x in f[1:]})
    if len(colors) != len(objs):
        return []
    facts = [(f, init) for f in cinit] + [(f, goal) for f in cgoal]
    sols: list[dict] = []

    def ok(m):
        for f, target in facts:
            if all(x in m for x in f[1:]) and (f[0],) + tuple(m[x] for x in f[1:]) not in target:
                return False
        return True

    def rec(i, m, used):
        if len(sols) >= cap:
            return
        if i == len(colors):
            mi = frozenset((f[0],) + tuple(m[x] for x in f[1:]) for f in cinit)
            mg = frozenset((f[0],) + tuple(m[x] for x in f[1:]) for f in cgoal)
            if mi == init and mg == goal:
                sols.append(dict(m))
            return
        for o in objs:
            if o in used:
                continue
            m[colors[i]] = o
            if ok(m):
                rec(i + 1, m, used | {o})
            del m[colors[i]]

    rec(0, {}, frozenset())
    return sols


def _plan_lines(x) -> list[str]:
    if isinstance(x, list):
        x = "\n".join(map(str, x))
    return [ln.strip().strip("()").strip().lower() for ln in (x or "").splitlines() if ln.strip().strip("()").strip()]


def mapping_from_translation(record: dict) -> dict | None:
    """colour -> letter pairs from a published record's colour-named 'raw_translation' aligned with its lettered
    'extracted_llm_plan' (only for results whose extraction went through PlanBench's LLM translator)."""
    rt = record.get("raw_translation")
    if not rt:
        return None
    col = [re.sub(r"[_-]block\b", "", ln).split() for ln in _plan_lines(rt.split("[PDDL PLAN]")[-1])
           if ln.split() and ln.split()[0] in ("pick-up", "put-down", "stack", "unstack")]
    let = [ln.split() for ln in _plan_lines(record.get("extracted_llm_plan"))]
    if not col or len(col) != len(let):
        return None
    m: dict = {}
    for a, b in zip(col, let):
        if a[0] != b[0] or len(a) != len(b):
            return None
        for c, l in zip(a[1:], b[1:]):
            if m.setdefault(c, l) != l:
                return None
    return m


def infer_mapping(query: str, objs, init, goal, record: dict | None = None) -> tuple[dict, dict]:
    """Colour -> PDDL letter for one instance. Primary: every bijection that maps the query's initial and goal facts
    exactly onto the PDDL init/goal; automorphic ties (interchangeable blocks; VAL verdicts are invariant under them)
    are broken toward PlanBench's encoded_objects. Fallbacks: raw_translation alignment, then encoded_objects."""
    try:
        cinit, cgoal = query_facts(query)
        sols = _alignments(cinit, cgoal, list(objs), init, goal)
    except ValueError:
        sols = []
    if sols:
        canon = [s for s in sols if all(CANON.get(c) == l for c, l in s.items())]
        pick = canon[0] if canon else sols[0]
        return pick, {"source": "aligned", "n_consistent": len(sols), "canonical": bool(canon)}
    tm = mapping_from_translation(record or {})
    if tm:
        return tm, {"source": "raw_translation", "n_consistent": 0, "canonical": all(CANON.get(c) == l for c, l in
                                                                                       tm.items())}
    m = {c: l for c, l in CANON.items() if l in set(objs)}
    return m, {"source": "canonical_fallback", "n_consistent": 0, "canonical": True}


# ----------------------------------------------------------------------------- free text -> PDDL plan

_COLOR_ALT = "|".join(COLORS)
# Base (imperative / "I will ...") forms only: participles such as ", placing the red block on the blue block" describe
# a result in prose and never start an action.
_VERB = (r"(?P<unstack>\bun-?stack\b)"
         r"|(?P<stack>\bstack\b)"
         r"|(?P<pickup>\bpick(?:[\s-]*up\b|\b)|\bpickup\b)"
         r"|(?P<putdown>\bput[\s-]*down\b|\bputdown\b)"
         r"|(?P<place>\b(?:put|place|set|lower)\b)"
         r"|(?P<take>\b(?:take|remove|lift|grab)\b)")
_VERB_RE = re.compile(_VERB)
_TOK_RE = re.compile(
    _VERB + rf"|(?P<color>\b(?:{_COLOR_ALT})\b)(?:[\s-]+blocks?\b)?"
    r"|(?P<pron>\bit\b|\b(?:the|that|this) block\b)|(?P<table>\btable\b)|(?P<down>\bdown\b)"
    r"|(?P<cut>\b(?:since|because|so|as|which|while|now|leaving|making|until|where|whereupon|resulting|allowing|thereby"
    r"|thus|ensuring|in order|to (?:make|clear|free|allow|expose|get|reach|achieve|access|enable|prepare))\b)")
# Clause-initial filler. Numeric list markers ("2.", "2)", "(2)") sit outside the word-boundary group: a ')' or '.'
# followed by a space is not a word boundary, so "plan: 1) unstack ..." used to keep its "1)" and lose the action.
_FILLER = re.compile(
    r"^(?:(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|now|finally|lastly|last|after(?:wards?| that| this)?"
    r"|subsequently|and|so|also|we|i|will|would|can|could|should|must|need to|have to|to|let's|let us|please|simply"
    r"|just|step\s*\d+|action\s*\d+)\b|\(?\d{1,3}[.)])[\s,:.)]*")
_LIST_MARK = re.compile(r"^(?:[-+•·]\s+|\*\s+|\(?(\d{1,3})[.)]\s*|\(?[a-z][.)]\s+)")
_STEP_MARK = re.compile(r"^(?:step|action|move)\s*(\d+)\s*[:.)\-]?\s*")
_CLAUSE_SPLIT = re.compile(r"[.;!?](?:\s+|$)|,\s*|\s*(?:→|->|=>|—|–| - )\s*|:\s*")
_PLAN_KW = re.compile(r"\b(?:plan|plans|solution|answer|sequence|actions|steps|moves)\b|\[plan\]")
# What may stand before the colon of a plan heading that carries its own actions on the same line: a bare label,
# optionally after a discourse marker ("Therefore, the final plan is: ..."); never a purpose clause or a sentence
# ("Finally, to complete the plan: stack ...", "Steps 9-10: Finally, place the blue block on ...").
_PLAN_LABEL = re.compile(
    r"(?:(?:therefore|thus|so|hence|finally|in summary|in conclusion|to summarize|to sum up|in short|overall)\s*,?\s+)?"
    r"(?:(?:here is|here's|below is|this is|the following is)\s+)?"
    r"(?:(?:the|my|our|a|an|final|complete|full|whole|entire|optimal|proposed|resulting|revised|corrected|correct"
    r"|minimal|shortest|valid|overall|step-by-step|detailed|concise|simple|new|updated)\s+)*"
    r"(?:plans?|solution|answer|action sequence|list of actions|sequence(?:\s+of\s+(?:actions|steps|moves))?|actions"
    r"|steps|moves)(?:\s+(?:is|would be|will be|summary|sequence|in order|in pddl))?|\[plan\]")
_NEG_KW = re.compile(r"\b(?:verif\w*|check\w*|explanation|explain\w*|reasoning|analysis|analy[sz]\w*|initial|final state"
                     r"|goal state|current state|state after|restrictions?|constraints?|rules?|preconditions?"
                     r"|postconditions?|effects?|notes?)\b|\[plan end\]")
_STOP_KW = re.compile(r"\b(?:verif\w*|check\w*|validat\w*|explanation|explain\w*|final state|resulting state|why|notes?"
                      r"|alternatives?|summary of states?)\b|\[plan end\]")
# A heading that continues the plan already under way ("Finally, to complete the plan:") is not a new plan section.
_CONT_KW = re.compile(r"\b(?:(?:complet|finish|continu)(?:e|es|ed|ing)?\s+(?:the|this|our|my)\s+plan|rest of the plan"
                      r"|remaining (?:steps|actions|moves) of the plan)\b")
_TABLE_SEP = re.compile(r":?-{2,}:?|")
_TABLE_NUM = re.compile(r"(?:step\s*)?#?\d{1,3}[.)]?|")
RELEASE = ("putdown", "stack", "place")  # verbs whose first argument is the held block


def answer_text(text: str, thinking: bool = False) -> tuple[str, str]:
    """Final-answer part of a completion: text after the last </think>. An unclosed <think> (or, in thinking mode, no
    </think> at all) means the model never produced an answer: nothing is parsed from its reasoning."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], "after_think"
    if "<think>" in text or thinking:
        return "", "unclosed_think"
    return text, "full"


def _normalize(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub("[‐‑‒]", "-", t).replace("’", "'").replace(" ", " ")
    return t.replace("\t", "    ").lower()


_FN_VERB = r"un-?stack|stack|pick[\s-]?up|put[\s-]?down"


def _clean(s: str) -> str:
    s = re.sub(r"(?<=\w)_(?=\w)", "-", s)
    s = re.sub(r"[*_`|]+", " ", s)
    # PDDL / call notation -> words: "(unstack blue orange)", "Unstack(blue, orange)" -> "unstack blue orange"
    s = re.sub(rf"\(\s*({_FN_VERB})\s+([^()]*)\)", r" \1 \2 ", s)
    s = re.sub(rf"\b({_FN_VERB})\s*\(([^()]*)\)", lambda m: f" {m.group(1)} {m.group(2).replace(',', ' ')} ", s)
    # remaining parentheticals are commentary ("(now holding the blue block)") and never supply arguments
    s = re.sub(r"\([^()]*\)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_filler(cl: str) -> str:
    cl = cl.strip(" \"'([{")
    prev = None
    while cl and prev != cl:
        prev, cl = cl, _FILLER.sub("", cl, count=1).strip(" \"'([{")
    return cl


def _table_row(raw: str) -> str | None:
    """Markdown table row -> plain line: None if raw is not a table row, "" for separator rows. Leading step-number
    cells are dropped. A cell that is a complete action ("Unstack blue from orange") stands alone and the other columns
    (resulting state, remarks) are dropped; otherwise the cells are read left to right ("Put down | blue | table")."""
    s = raw.strip()
    if not (s.startswith("|") and s.count("|") >= 2):
        return None
    cells = [c.strip() for c in s.strip("|").split("|")]
    if all(_TABLE_SEP.fullmatch(c) for c in cells):
        return ""
    while cells and _TABLE_NUM.fullmatch(_clean(cells[0])):
        cells.pop(0)
    indent = raw[: len(raw) - len(raw.lstrip(" "))]
    for c in cells:
        f = _strip_filler(_clean(c))
        if _VERB_RE.match(f) and re.search(rf"\b(?:{_COLOR_ALT})\b", f):
            return indent + c
    return indent + " ".join(cells)


def _parse_clause(cl: str, st: dict, issues: list) -> list[tuple]:
    """Actions of one verb-initial clause. st: {'last_color', 'held'} (line-level context)."""
    acts, cur = [], None

    def pron(c):
        """'it' / 'the block', resolved where it occurs: first argument of a release verb -> the held block;
        otherwise the last colour named before it on this line (not the action's own first argument)."""
        kind, args = c[0], c[1]
        if not args and kind in RELEASE and st["held"]:
            return st["held"]
        if st["last_color"] and (not args or st["last_color"] != args[0]):
            return st["last_color"]
        return None

    def close(c):
        if c is None:
            return
        kind, res, table, down = c
        if None in res:
            issues.append(f"unresolved pronoun: {cl[:60]!r}")
            return
        a = None
        if kind in ("unstack", "stack") and len(res) >= 2:
            a = (kind, res[0], res[1])
        elif kind == "pickup" and res:
            a = ("pick-up", res[0])
        elif kind == "putdown" and res:
            a = ("put-down", res[0])
        elif kind == "place" and len(res) >= 2:
            a = ("stack", res[0], res[1])
        elif kind == "place" and len(res) == 1 and (table or down):
            a = ("put-down", res[0])
        elif kind == "take" and len(res) >= 2:
            a = ("unstack", res[0], res[1])
        elif kind == "take" and len(res) == 1:
            a = ("pick-up", res[0])
        if a is None:
            issues.append(f"underspecified {kind}: {cl[:60]!r}")
            return
        acts.append(a)
        st["held"] = a[1] if a[0] in ("pick-up", "unstack") else None

    for m in _TOK_RE.finditer(cl):
        g = m.lastgroup
        if g in ("unstack", "stack", "pickup", "putdown", "place", "take"):
            close(cur)
            cur = [g, [], False, False]
        elif g == "cut":
            break
        elif cur is None:
            continue
        elif cur[2] and cur[0] == "place":
            pass  # destination already the table: "place X on the table, away from Y" is a put-down
        elif g == "color":
            cur[1].append(m.group("color"))
        elif g == "pron":
            cur[1].append(pron(cur))
        elif g == "table":
            cur[2] = bool(cur[1]) or cur[0] != "place"
        elif g == "down":
            cur[3] = True
        if g == "color":
            st["last_color"] = m.group("color")
    close(cur)
    return acts


def _line_info(raw: str, st: dict, issues: list, table: bool = False) -> dict:
    indent = len(raw) - len(raw.lstrip(" "))
    s = raw.strip()
    open_end = bool(s) and not re.search(r"[.!?:;)\]]$", s.rstrip(" *_`"))  # sentence runs on to the next line
    md = len(s) - len(s.lstrip("#"))
    s = s.lstrip("#").strip().lstrip(">").strip()
    bold_line = bool(re.fullmatch(r"\*\*[^*]+\*\*:?", s))
    lm = _LIST_MARK.match(s)
    marked = bool(lm) or table
    s = _LIST_MARK.sub("", s, count=1)
    s = _clean(s)
    lm2 = _LIST_MARK.match(s)
    s = _LIST_MARK.sub("", s, count=1)
    sm = _STEP_MARK.match(s)
    stepped = bool(sm)
    s = _STEP_MARK.sub("", s, count=1)
    num = next((int(m.group(1)) for m in (lm, lm2, sm) if m and m.group(1)), None)
    st["last_color"] = None
    acts = []
    for cl in _CLAUSE_SPLIT.split(s):
        cl = _strip_filler(cl)
        if cl and _VERB_RE.match(cl):
            acts += _parse_clause(cl, st, issues)
        else:
            for m in re.finditer(rf"\b({_COLOR_ALT})\b", cl):
                st["last_color"] = m.group(1)
    # "1. **Stack blue on yellow**: Stack the blue block onto the yellow block." restates within one line; the same
    # action twice in a row is never applicable twice, so an immediate repeat inside a line is a restatement.
    n0 = len(acts)
    acts = [x for k, x in enumerate(acts) if k == 0 or acts[k - 1] != x]
    st["line_repeats"] += n0 - len(acts)
    text = s.strip()
    heading = (not acts) and bool(text) and (md > 0 or bold_line or text.endswith(":") or text in ("[plan]",
                                                                                              "[plan end]"))
    head_txt = text.split(":")[0] if ":" in text else text
    if acts:
        # same-line actions: only a bare label before the colon makes this a plan heading ("Plan: unstack ..."), never
        # a sentence that mentions the plan ("Finally, to complete the plan: stack the red block on the blue block")
        plan_head = ":" in s and bool(_PLAN_LABEL.fullmatch(head_txt.strip(" ,.-"))) and not _NEG_KW.search(head_txt)
    else:
        plan_head = heading and bool(_PLAN_KW.search(head_txt)) and not _NEG_KW.search(head_txt) and not re.match(
            r"step\s*\d", head_txt)
    cont = heading and bool(_CONT_KW.search(text))
    if cont:  # transparent: neither starts nor ends a plan region
        heading = plan_head = False
    if text == "[plan end]":
        heading, plan_head = True, False
    return {"indent": indent, "acts": acts, "heading": heading, "plan_head": bool(plan_head), "marked": marked,
            "stepped": stepped, "md": md, "text": text, "num": num, "cont": cont, "open_end": open_end}


def _continued_headings(lines: list[dict]) -> int:
    """A heading between two numbered action lines whose numbers continue (2. ... / "Next, to complete the plan:" /
    3. ...) is an interjection inside one list: it neither starts nor ends a plan region. Returns how many."""
    numbered = [(i, L["num"]) for i, L in enumerate(lines) if L["acts"] and L["num"] is not None]
    n = 0
    for h, L in enumerate(lines):
        if not L["heading"] or L["acts"] or L["stepped"]:
            continue
        before = [k for i, k in numbered if i < h]
        after = [k for i, k in numbered if i > h]
        if before and after and after[0] == before[-1] + 1:
            L["heading"] = L["plan_head"] = False
            n += 1
    return n


def _select_actions(lines: list[dict], start: int, stop: int) -> tuple[list[tuple], dict]:
    """Actions of lines[start:stop] with the nesting rules: a line indented under an action-bearing line is commentary
    on that step (its actions are dropped; counted as a restatement if identical to the parent's), unless the parent
    is a group header ("2. Then stack orange on blue:" over indented list items carrying >= 2 other actions), whose own
    summary actions are dropped instead. An unmarked indented line right after a parent whose sentence runs on (no
    closing punctuation: "1. Unstack blue from orange and" / "   put it down.") is the same step, wrapped."""
    idx = [i for i in range(start, stop) if lines[i]["text"]]
    parent: dict = {}
    wrapped = set()
    stack: list[int] = []
    for k, i in enumerate(idx):
        while stack and lines[stack[-1]]["indent"] >= lines[i]["indent"]:
            stack.pop()
        parent[i] = stack[-1] if stack else None
        L, prev = lines[i], (idx[k - 1] if k else None)
        if prev is not None and lines[prev]["open_end"] and not (L["marked"] or L["stepped"] or L["heading"]) and (
                (prev == parent[i]) or (prev in wrapped and lines[prev]["indent"] == L["indent"])):
            wrapped.add(i)
            parent[i] = parent[prev]  # read as part of the line it continues
        stack.append(i)
    group = set()
    for p in idx:
        P = lines[p]
        if not P["acts"] or not P["text"].endswith(":"):
            continue
        kids = [i for i in idx if parent[i] == p and lines[i]["acts"]]
        kid_acts = [a for i in kids for a in lines[i]["acts"]]
        if kids and all(lines[i]["marked"] or lines[i]["stepped"] for i in kids) and len(kid_acts) >= 2 \
                and kid_acts != P["acts"]:
            group.add(p)
    acts: list[tuple] = []
    info = {"dropped_restatements": 0, "dropped_commentary_lines": 0, "dropped_commentary_actions": 0,
            "group_headers": len(group), "wrapped_lines": len(wrapped)}
    for i in idx:
        L = lines[i]
        if not L["acts"] or i in group:
            continue
        anc, under = parent[i], None
        while anc is not None:
            if lines[anc]["acts"] and anc not in group:
                under = anc
                break
            anc = parent[anc]
        if under is None:
            acts += L["acts"]
        elif L["acts"] == lines[under]["acts"]:
            info["dropped_restatements"] += 1
        else:
            info["dropped_commentary_lines"] += 1
            info["dropped_commentary_actions"] += len(L["acts"])
    return acts, info


def extract_plan(text: str, cmap: dict, thinking: bool = False) -> tuple[list[tuple], dict]:
    """Free-text answer -> list of PDDL action tuples over the instance's letters (see module docstring)."""
    ans, src = answer_text(text or "", thinking)
    issues: list[str] = []
    # single-letter abbreviations the answer itself defines, e.g. "the orange block (O)" ... "Unstack O from Y"
    alias: dict = {}
    for m in re.finditer(rf"\b((?i:{_COLOR_ALT}))\b(?:\s+(?i:block))?\s*\(([A-Z])\)", ans):
        alias.setdefault(m.group(2), set()).add(m.group(1).lower())
    alias = {k: next(iter(v)) for k, v in alias.items() if len(v) == 1}
    if alias:
        ans = re.sub(r"\b([A-Z])\b", lambda m: alias.get(m.group(1), m.group(1)), ans)
    st = {"last_color": None, "held": None, "line_repeats": 0}
    lines = []
    n_table = 0
    for raw in _normalize(ans).split("\n"):
        if raw.strip().startswith("```"):
            continue
        row = _table_row(raw)
        if row == "":
            continue  # table separator row
        n_table += row is not None
        lines.append(_line_info(raw if row is None else row, st, issues, table=row is not None))
    n_continued = _continued_headings(lines) + sum(L["cont"] for L in lines)
    act_idx = [i for i, L in enumerate(lines) if L["acts"]]
    region, start, stop = "all", 0, len(lines)
    for h in reversed([i for i, L in enumerate(lines) if L["plan_head"]]):
        end = len(lines)
        seen = False
        for j in range(h + 1, len(lines)):
            L = lines[j]
            if L["acts"]:
                seen = True
            elif seen and L["heading"] and not L["plan_head"] and not L["marked"] and not L["stepped"] \
                    and L["indent"] <= lines[h]["indent"]:
                end = j
                break
            elif seen and L["plan_head"]:
                end = j
                break
        if lines[h]["acts"] or any(lines[j]["acts"] for j in range(h + 1, end)):
            region, start, stop = f"after_plan_heading:{lines[h]['text'][:50]}", h, end
            break
    if region == "all" and act_idx:  # no plan heading: stop at a verification/explanation heading after the actions
        for j in range(act_idx[0] + 1, len(lines)):
            L = lines[j]
            if L["heading"] and not L["marked"] and not L["stepped"] and _STOP_KW.search(L["text"]):
                region, stop = f"all_until:{L['text'][:50]}", j
                break
    acts, nest = _select_actions(lines, start, stop)
    unknown = sorted({x for a in acts for x in a[1:] if x not in cmap})
    plan = [(a[0],) + tuple(cmap.get(x, x) for x in a[1:]) for a in acts]
    info = {"answer_source": src, "region": region, "n_action_lines": sum(start <= i < stop for i in act_idx),
            **nest, "line_repeats_collapsed": st["line_repeats"], "table_rows": n_table,
            "continued_headings": n_continued, "aliases": alias, "unknown_colors": unknown, "issues": issues[:20]}
    return plan, info


# ----------------------------------------------------------------------------- PlanBench's own parser (2nd scorer)
# Port of get_ordered_objects / text_to_plan_blocksworld from LLMs-Planning plan-bench/utils/text_to_pddl.py at commit
# fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284 (https://github.com/karthikv792/LLMs-Planning; MIT License, Copyright (c)
# 2024 Valmeekam Karthik; bench/planbench/LICENSE_LLMs-Planning_MIT.txt). Changes: the plan is returned as action
# tuples instead of being written to a plan file; tarski's problem.actions is replaced by the domain file's action
# names and arities in file order (the order tarski lists them in); numpy.where -> list.index. Everything else is kept
# as upstream, quirks included: at most one action per line; when a line contains several action words the first in
# DOMAIN order wins (pick-up, put-down, stack, unstack), not the first in the line; an action word must be a whole
# whitespace-separated token ("**unstack" is not); objects must be written "<colour> block", are ordered by first
# occurrence and counted once; a line whose object count differs from the action's arity is skipped; the '[COST]'
# check never fires because the text is lowercased first. Upstream applies it to the whole response (API models
# expose no reasoning); here it is applied to answer_text().

PB_ENCODED_OBJECTS = {"a": "red block", "b": "blue block", "c": "orange block", "d": "yellow block",
                      "e": "white block", "f": "magenta block", "g": "black block", "h": "cyan block",
                      "i": "green block", "j": "violet block", "k": "silver block", "l": "gold block"}
PB_ACTIONS = {"pick-up": "pick up the {}", "put-down": "put down the {}", "stack": "stack the {} on top of the {}",
              "unstack": "unstack the {} from on top of the {}"}  # configs/blocksworld.yaml (== blocksworld_3.yaml)


def domain_actions(domain_path: Path = B.DOMAIN_PDDL) -> dict:
    """Action name -> number of parameters, in domain-file order (tarski's problem.actions order)."""
    txt = Path(domain_path).read_text().lower()
    return {n: len(p.split()) for n, p in re.findall(r"\(:action\s+(\S+)\s+:parameters\s*\(([^)]*)\)", txt)}


def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))

    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]


def text_to_plan_blocksworld(text: str, action_set: dict | None = None, data: dict | None = None
                             ) -> tuple[list[tuple], str]:
    """PlanBench's Blocksworld text -> PDDL plan (see the port note above). action_set: name -> arity (default: the
    vendored domain); data: {'encoded_objects', 'actions'} (default: configs/blocksworld.yaml). Returns (plan tuples,
    upstream's colour-named readable plan)."""
    action_set = action_set if action_set is not None else domain_actions()
    data = data if data is not None else {"encoded_objects": PB_ENCODED_OBJECTS, "actions": PB_ACTIONS}
    # ----------- GET DICTIONARIES ----------- #
    LD = data['encoded_objects']  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary
    AD = {}  # Action Dictionary
    for k, v in data['actions'].items():
        word = v.split(' ')[0]
        if word in k:
            AD[k] = k.replace("-", " ")
        else:
            AD[k] = word

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = [i.lower() for i in list(action_set.keys())]
    text_actions = [AD[x] for x in raw_actions]

    text = text.lower().strip()
    for raw_action, text_action in zip(raw_actions, text_actions):
        text = text.replace(text_action, raw_action)
    object_names = [x.lower() for x in LD.values()]

    # ----------- GET PLAN FROM TEXT ----------- #
    plan = []
    readable_plan = ""
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if '[COST]' in line:
            break
        # Extracting actions
        action_list = [action in line.split() for action in raw_actions]
        if sum(action_list) == 0:
            continue
        action = raw_actions[action_list.index(True)]
        # Extracting Objects
        n_objs = actions_params_dict[action]
        objs = get_ordered_objects(object_names, line)
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace(' block', '') for obj in objs]
        objs = [BD[x] for x in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        plan.append((action,) + tuple(objs[:n_objs + 1]))
        readable_plan += f"{readable_action}\n"
    return plan, readable_plan


_PB_INLINE_ITEM = re.compile(r"(?<=\S)\s+(?=\(?\d{1,2}[.)]\s+\S)")


def pb_markdown_text(text: str) -> str:
    """Markdown-only normalisation for the "pb_md" variant (the parser itself is unchanged): code-fence lines dropped;
    table rows -> one line (cells joined by spaces, separator rows dropped); heading '#'s, '*' and '`' removed and '_'
    removed where it delimits emphasis (not inside words); a list marker glued to its text ("1)unstack") gets a space;
    inline numbered lists ("1) ... 2) ...", "1. ... 2. ...") are split one item per line, since the parser reads at
    most one action per line. The published Gemini-1.5 extractions equal pb_raw after '*' removal (see tests)."""
    out = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = raw.strip()
        if s.startswith("```"):
            continue
        if s.startswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(_TABLE_SEP.fullmatch(c) for c in cells):
                continue
            s = " ".join(c for c in cells if c)
        s = re.sub(r"^#+\s*", "", s)
        s = re.sub(r"[*`]+", "", s)
        s = re.sub(r"(?<!\w)_+|_+(?!\w)", "", s)
        s = re.sub(r"^(\(?\d{1,3}[.)])(?=[^\W\d_])", r"\1 ", s.strip())
        out += _PB_INLINE_ITEM.split(s)
    return "\n".join(out)


def planbench_plans(text: str, thinking: bool = False) -> dict:
    """Second-scorer variants on the answer part of a completion: {'pb_raw': plan, 'pb_md': plan} (letters)."""
    ans, _ = answer_text(text or "", thinking)
    return {"pb_raw": text_to_plan_blocksworld(ans)[0], "pb_md": text_to_plan_blocksworld(pb_markdown_text(ans))[0]}


# ----------------------------------------------------------------------------- instances

def load_instances(sets=("generated_basic_3", "generated_basic"), prompt_type: str = "zero_shot",
                   query_source: str = "o1-preview_chat") -> list[dict]:
    out = []
    for inst_dir in sets:
        f = RES / SETS[inst_dir] / query_source / TASK_FILE[prompt_type]
        recs = json.loads(f.read_text(encoding="utf-8"))["instances"]
        for r in sorted(recs, key=lambda r: int(r["instance_id"])):
            p = B.BENCH / "blocksworld" / inst_dir / f"instance-{r['instance_id']}.pddl"
            if not p.exists():
                continue
            objs, init, goal = B.parse_problem(p)
            cmap, minfo = infer_mapping(r["query"], objs, init, goal, r)
            out.append({"id": f"{inst_dir}/instance-{r['instance_id']}", "set": inst_dir,
                        "instance_id": int(r["instance_id"]), "query": r["query"], "path": p, "objs": objs,
                        "init": init, "goal": goal, "cmap": cmap, "map_info": minfo, "record": r})
    return out


def optimal_length(inst: dict) -> int:
    """Optimal plan length by exact BFS (matches PlanBench's ground_truth_plan length on all 600 Task-1 instances)."""
    p = B.bfs_plan(inst["init"], inst["goal"], list(inst["objs"]))
    if p is None:
        raise ValueError(f"{inst.get('id')}: no plan found by BFS")
    return len(p)


def stratum(inst: dict) -> str:
    return f"{len(inst['objs'])}blocks_len{optimal_length(inst)}"


def stratified_subset(insts: list[dict], n: int, seed: int = 0, key=stratum) -> tuple[list[dict], dict]:
    """Seeded stratified sample of n instances; strata = key(inst) (default: block count x optimal plan length).
    Allocation is proportional to stratum size by largest remainders (equal remainders ordered by a seeded draw), so
    every stratum holding >= len(insts)/n instances is represented; within a stratum instances are drawn with
    random.Random(seed).sample. Deterministic for (insts, n, seed); returned in input order with {stratum: (size,
    drawn)}."""
    if not 0 < n <= len(insts):
        raise ValueError(f"subset size {n} not in [1, {len(insts)}]")
    strata: dict = {}
    for i in insts:
        strata.setdefault(key(i), []).append(i)
    rng = random.Random(seed)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(insts) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    tie = {k: rng.random() for k in keys}
    for k in sorted(keys, key=lambda k: (alloc[k] - quota[k], tie[k]))[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen = set()
    for k in keys:
        chosen |= {i["id"] for i in rng.sample(strata[k], alloc[k])}
    assert len(chosen) == n, (len(chosen), n)
    return [i for i in insts if i["id"] in chosen], {k: (len(strata[k]), alloc[k]) for k in keys}
