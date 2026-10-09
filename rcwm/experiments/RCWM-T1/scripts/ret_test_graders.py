"""CPU tests of the retention graders, the generator, the freshness harvest, minting and the runner's --plan. No torch,
no model, no GPU, no lease. One extra process (the PYTHONHASHSEED independence check). Threads 2.

  python -B scripts/ret_test_graders.py [--no-record]
Writes docs/retention/GRADER_TEST.json (unless --no-record; refused once the kit is frozen).

  1  primary grader on adversarial answer strings (every kind; wrappers, prefixes, units, hedges, ranges, explanations,
     number words, scientific notation, bad comma groups, fractions, meridiem rules, case rules, duplicate JSON keys,
     type mismatches, unterminated answers) + the lenient grader on the cases where it must differ
  2  gold validation rejects malformed golds
  3  generator determinism: same seed -> same bytes (twice in-process, once in a child process with another
     PYTHONHASHSEED); a new seed -> zero shared user turns and zero shared passage sentences
  4  gold self-test (canonical answers 100% correct, negatives 0%) on three seeds; passages 150-300 Qwen3 tokens
  5  surface diversity: OFF -> one system prompt; ON -> several, golds still self-test, recorded in the seal
  6  freshness harvest on a fake corpus: an exact user turn (other case/whitespace) and an embedded gated sentence are
     found; a file named HSKILL_CONFIRM_V2* holding a user turn is skipped by name and never stat()ed or opened
  7  mint / verify in a test root: create-only; seed reuse refused; confirm refused without a dev selection, with a
     surface-diversity mismatch or with the dev seed; the confirm draw holds exactly the selected cells; verify
     regenerates the bytes; a tampered draw is refused by load_draw
  8  ret_run.py --plan in-process: torch not imported; dev 2,025 prompts, confirm 675; GPU-h per turn and per arm
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # never "-1" (DECISIONS I-21)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import builtins  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
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
    (NUM, "350/1", True), (NUM, "**350**.", True), (NUM, "**350.**", True), (NUM, "Final answer: 350", True),
    (NUM, "350 mm", False), (NUM, "350 m", False), (NUM, "3.5e2", False), (NUM, "350 or 351", False),
    (NUM, "about 350", False), (NUM, "~350", False), (NUM, "350!", False), (NUM, "350 (3.5 x 100)", False),
    (NUM, "three hundred fifty", False), (NUM, "35O", False), (NUM, "3 50", False), (NUM, "350,", False),
    (NUM, "350..", False), (NUM, "-350", False), (NUM, "", False), (NUM, "Answer:", False), (NUM, "351", False),
    (NUM, "350\n\nExplanation: 3.5 x 100.", False), (NUM, "It's 350", False), (NUM, "350-360", False),
    (NUM, ("350", False), False), (NUM, "350 cm.", True), (NUM, "350 centimetres long", False),
    (DEC, "2.5", True), (DEC, "2.50", True), (DEC, "5/2", True), (DEC, ".25e1", False), (DEC, "2,5", False),
    (DEC, "2½", False), (DEC, "2.49", False), (DEC, "2 1/2", False),
    (BIG, "1,234", True), (BIG, "1234", True), (BIG, "1,23,4", False), (BIG, "12,34", False), (BIG, "1 234", False),
    (PCT, "25%", True), (PCT, "25 %", True), (PCT, "25 percent", True), (PCT, "25 per cent", True), (PCT, "0.25", False),
    (PCT, "25 percentage points", False),
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
LENIENT = [   # (gold, text, expected lenient): where value extraction must accept or still reject
    (NUM, "The answer is 350 cm because 3.5 x 100 = 350.", True), (NUM, "350 or 351", False), (NUM, "About 350", True),
    (CLK, "It ends at 1:15 PM.", True), (CLK, "It ends at 1:15.", False), (WDY, "10 days after Monday is Thursday", True),
    (WDY, "Thu", True), (DAT, "It falls on Apr 5.", True), (NLS, "[-3, 7, 9]", True), (NLS, "-3, 9, 7", False),
    (WLS, "1. apple 2. banana 3. cherry", True), (RAT, "simplest form: 2 to 3", True), (KVC, "The code is 4721.", True),
    (KVC, "4721 or 1111", False), (KVC, "UNKNOWN (4721?)", False), (UNK, "Unknown - the passage does not say.", True),
    (UNK, "It is not mentioned; maybe 4721", False), (UPP, "Sure: PINE", True), (JSN, 'Here: {"item": "pine", "count": 3}', True),
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
    check(f"lenient (report-only) grader: {len(LENIENT)} cases", not badl, badl[:6])
    rej = 0
    for g in BAD_GOLDS:
        try:
            RG.validate_gold(g)
        except RG.GoldError:
            rej += 1
    check(f"malformed golds rejected {rej}/{len(BAD_GOLDS)}", rej == len(BAD_GOLDS))
    COUNTS["malformed_golds"] = len(BAD_GOLDS)


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
    check("new seed -> disjoint user turns and passage sentences", not (ua & uc) and not (pa & pc),
          (len(ua & uc), len(pa & pc)))
    COUNTS["new_seed_overlap"] = {"user_turns": len(ua & uc), "passage_sentences": len(pa & pc),
                                  "item_unique_sentences_ge8_report": len(ia & ic), "operative_report": len(oa & oc),
                                  "n_prompts": len(a)}
    check("dev plan: 1,800 items / 2,025 prompts; 75 items per cell", len({r["item"] for r in a}) == 1800 and len(a) == 2025)
    ok_self, ok_tok, mm = True, True, []
    for seed in (RC.DEV_SEED, 7, 20261199):
        rows, _ = G.generate(seed, "dev", plan)
        st = G.gold_selftest(rows)
        ok_self &= st["ok"]
        ta = G.token_audit(rows)
        ok_tok &= ta["ok"]
        mm.append((ta["passage_tokens"]["min"], ta["passage_tokens"]["max"]))
        COUNTS.setdefault("selftest", []).append({k: st[k] for k in ("positives", "positives_correct", "negatives",
                                                                     "negatives_rejected")})
    check("gold self-test 100% / 0% on three seeds", ok_self, COUNTS["selftest"])
    check("passages within 150-300 Qwen3 tokens on three seeds", ok_tok, mm)
    COUNTS["passage_token_range_3_seeds"] = mm
    d_off, _ = G.generate(11, "dev", plan, diversity=False)
    d_on, _ = G.generate(11, "dev", plan, diversity=True)
    sys_off = {r["messages"][0]["content"] for r in d_off}
    sys_on = {r["messages"][0]["content"] for r in d_on}
    fn_on = {w for r in d_on if r["category"] == "kv_recall" for th in G.THEMES.values() for w in th["fn"][1:]
             if w in RC.user_text(r["messages"])}
    ta_on = G.token_audit(d_on)
    check("surface diversity OFF -> one system prompt; ON -> several, field-name variants used, golds self-test, "
          "passages 150-300 tokens",
          len(sys_off) == 1 and len(sys_on) >= 3 and len(fn_on) >= 6 and G.gold_selftest(d_on)["ok"] and ta_on["ok"],
          (len(sys_off), len(sys_on), sorted(fn_on), ta_on["passage_tokens"]["min"], ta_on["passage_tokens"]["max"]))
    check("surface diversity is part of the cell seed (ON != OFF bytes)",
          G.serialize(d_off) != G.serialize(d_on))


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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_mint():
    tmp = Path(tempfile.mkdtemp(prefix="ret_mint_"))
    try:
        (tmp / "docs" / "retention" / "draws").mkdir(parents=True)
        s1 = G.mint("dev", seed=101, scan=False, root=tmp, require_freeze=False, n=6)
        p1 = tmp / s1["file"]
        check("mint writes draw + seal; sha matches", p1.is_file() and RC.sha256_file(p1) == s1["file_sha256"])
        v = G.verify(p1)
        check("verify regenerates the draw byte for byte", v["ok"] and v["regenerates_identically"], v)
        for name, fn in (("seed reuse refused", lambda: G.mint("dev", seed=101, scan=False, root=tmp, require_freeze=False, n=6)),
                         ("registered mint refused without a kit freeze", lambda: G.mint("dev", seed=102, scan=False, root=tmp)),
                         ("confirm refused without DEV_SELECTION.json",
                          lambda: G.mint("confirm", seed=103, scan=False, root=tmp, require_freeze=False, n=6))):
            try:
                fn()
                check(name, False)
            except RC.RetStop:
                check(name, True)
        sel = {"selected": {"arith_word": 2, "unknown_twin": 3, "format_follow": 1}, "excluded": [], "surface_diversity": False}
        (tmp / "docs" / "retention" / "DEV_SELECTION.json").write_text(json.dumps(sel), encoding="utf-8")
        for name, fn in (("confirm refused with a surface-diversity mismatch",
                          lambda: G.mint("confirm", seed=104, diversity=True, scan=False, root=tmp, require_freeze=False, n=6)),
                         ("confirm refused with the dev seed",
                          lambda: G.mint("confirm", seed=RC.DEV_SEED, scan=False, root=tmp, require_freeze=False, n=6))):
            try:
                fn()
                check(name, False)
            except RC.RetStop:
                check(name, True)
        s2 = G.mint("confirm", scan=False, root=tmp, require_freeze=False, n=6)
        rows2, _ = G.load_draw(tmp / s2["file"])
        check("confirm draw: fresh OS-entropy seed, exactly the selected cells, cross-draw overlap 0",
              s2["seed_source"].startswith("OS entropy") and
              {(r["category"], r["level"]) for r in rows2} == {("arith_word", 2), ("unknown_twin", 3), ("format_follow", 1)}
              and s2["cross_draw"]["F1_user_turn_overlap"] == 0 and s2["selection_sha256"] is not None, s2["seed"])
        p2 = tmp / s2["file"]
        txt = p2.read_text(encoding="utf-8")
        p2.write_text(txt.replace("UNKNOWN", "UNKNOWN!", 1), encoding="utf-8")
        try:
            G.load_draw(p2)
            check("a tampered draw is refused", False)
        except RC.RetStop:
            check("a tampered draw is refused", True)
        COUNTS["mint_test_seconds"] = s1["mint_seconds"]
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
    for fn in (t_graders, t_generator, t_freshness, t_mint, t_plan):
        print(f"[{fn.__name__}]", flush=True)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(e))
    ok = all(r["ok"] for r in RESULTS)
    print(f"{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass ({time.time() - t0:.0f}s)")
    if "--no-record" not in sys.argv:
        RC.need(not RC.FREEZE.exists(), "the kit is frozen: GRADER_TEST.json is a frozen record")
        rec = {"ok": ok, "n_checks": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "counts": COUNTS,
               "checks": RESULTS, "recorded_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "code_sha256": {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_common.py", "scripts/ret_grade.py",
                                                                      "scripts/ret_gen.py", "scripts/ret_run.py",
                                                                      "scripts/ret_test_graders.py")}}
        (RC.DOCS / "GRADER_TEST.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
