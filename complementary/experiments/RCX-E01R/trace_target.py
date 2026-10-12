"""RCX-E01R recipe R3 (DESIGN.md section 4): a predicate-first trace target for the native E01N frame.

The native answer is the compact sorted JSON of the balances after one event (native_e01.target_json). R3 supervises two
lines instead. Line 1 writes out the rule's checks on this event and the resulting decision; line 2 is the unchanged
native answer. The predicate is thus computed explicitly before the state (the E01 `cmp -> after` trace hypothesis;
RCWM R23). Line 1 is derived only from the rule AST, the event and (for BGE) the state, never from the rule wording.
Every truth value comes from rcx.rcl.world (eval_atom, listed, eval_rule), and trace_target refuses any record whose
line 1 does not parse back to the gold decision.

Line 1 grammar (fixed format: single spaces, nothing else on the line; atoms in AST order, the connective after them):
  line1 := "check " atom " -> " dec                           single-atom rule (S+ or S-, or a lone other atom)
         | "check " atom (" " atom)+ " " conn " -> " dec      AND / OR rule
  atom  := "status=" JSTR " listed=" ("yes"|"no") " " ("only-listed"|"all-but-listed") ":" truth   S+ | S-
         | "kind=" KIND " in=" KIND ("|" KIND)* ":" truth                                         K+
         | "amount=" INT " at-least=" INT ":" truth  |  "amount=" INT " below=" INT ":" truth       AGE | ALT
         | "balance=" INT " at-least=" INT ":" truth          BGE (balance of the event's account before the event)
  truth := "true" | "false"     conn := "AND" | "OR"     dec := "apply" | "noop"
  JSTR = json.dumps(status) (canonical escaping only); KIND = [a-z]+; INT = 0 | -?[1-9][0-9]*
`listed` is membership in that atom's own status set. only-listed (S+) is true iff listed, all-but-listed (S-) is true
iff not listed. The connective is followed by the decision. The atoms start with distinct keys, the status is a
delimited JSON string, and the tail is " [conn ]-> dec", so a line has exactly one parse.

Examples (arith world, DEV battery seed 20261240):
  check status="vervil" listed=no only-listed:false -> noop                                  V1_pos: S+, unlisted
  check status="keron" listed=yes all-but-listed:false -> noop                               V2_neg: S-, listed
  check status="minted" listed=no only-listed:false amount=-2 below=2:true AND -> noop      V6: AND(S+, ALT)
  check status="routed" listed=no all-but-listed:true amount=1 at-least=6:false OR -> apply  V6: OR(S-, AGE)
Full target = line1 + "\\n" + target_json(after), e.g.
  'check status="vervil" listed=no only-listed:false -> noop\\n{"A":33,"B":51,"C":42,"D":13}'

Token budget (token_budget_note() reads rcx/native/lora_cpu.py as source; nothing is imported or modified):
- lora_cpu.generate(model, tok, prompts, max_new=64, max_batch_tokens=3000, guard=None, log=None) is greedy.
  It calls model.generate(max_new_tokens=max_new, eos_token_id=EOS_IDS=[151645, 151643], do_sample=False).
- A row stops only at <|im_end|> / <|endoftext|> or after max_new new tokens. Nothing stops at the end of the JSON or
  at a newline (no stop strings, no stopping criteria), so the "\\n" inside the R3 target is harmless.
- The E01N/E01R evals use the default max_new=64, which fits the native answer: at most 68 bytes on the DEV battery,
  mean 25-29 bytes in the train regimes.
- R3 targets are longer: line 1 alone has a mean of 57-68 bytes. Full targets are at most 154 bytes on the DEV battery
  (all 13 strata x 6 surfaces; the longest is an OR composition over 6 word-named accounts) and at most 128 bytes in
  the FX/VR/CF_SD train sets (means 87-97).
- Token counts were not measured: this step may not load a tokenizer. Qwen's byte-level BPE gives fewer tokens than
  bytes (digits are single tokens), but compositions on many-account or large-value surfaces can plausibly pass 64.
  A truncated output loses line 2 and scores as a format error.
- generate accepts max_new as a keyword, so the R3 eval needs no change to lora_cpu.py:
  LC.generate(model, tok, prompts, guard=..., log=..., **generate_kwargs_for_trace(...)).
  - With no arguments, that call gives the static bound TRACE_MAX_NEW, suitable for a prereg.
  - With the battery's targets and the tokenizer that the eval loads anyway, it gives the exact need.
  - The UTF-8 byte length of a target is a guaranteed upper bound on its byte-level-BPE token count.
    TRACE_MAX_NEW = 160 is that bound: the DEV battery maximum, 154, + 1 for EOS, rounded up. It is an empirical
    maximum over the generator's draws, not a proof for unseen vocabulary. tests/test_e01r_trace.py re-checks it.
  - Recommended: the R3 build step loads the tokenizer anyway, so it should record
    generate_kwargs_for_trace(train + eval targets, tok) and preregister that tighter value.
- EOS: train_sft/encode_pair already append EOS after the target (eos=True). R3 rows trained through train_sft
  therefore learn to stop after line 2. A model that does not stop runs to max_new, and line 2 then fails to parse,
  exactly as for the native answer.
- Cost: model.generate returns when every row of a batch has emitted EOS. Wall time therefore follows the generated
  length (about 2x the native answer), not max_new. Two other effects:
  - the batch size min(32, 3000 // (prompt_len + max_new)) shrinks, e.g. 14 -> 9 rows at a 150-token prompt with
    max_new 160. That means more sequential decode passes; a tokenizer-measured max_new keeps the shrink small.
    Raising max_batch_tokens instead would raise peak RAM, so it is not done here;
  - train_sft's max_len=512 is far away: native rows are at most 171 tokens (runs/RCX-E01N/cells/TRAIN_MANIFEST.json).
- Scoring is unchanged: score.classify(rec, parse_trace_answer(out, keys)[0]). The balances come only from line 2 and
  are parsed by native_probe.parse_answer itself.
  - native_probe.parse_answer and native_e01.target_json are executed from those files' own source (_borrow). Those
    scripts import torch at module top, which this module must not.
"""
import ast as pyast
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.rcl import score as S, world as W  # noqa: E402

E01N = ROOT / "experiments" / "RCX-E01N"
LORA_CPU = ROOT / "rcx" / "native" / "lora_cpu.py"
TRACE_MAX_NEW = 160  # UTF-8 byte bound (+1 EOS) over the DEV generator; see the module docstring
TAG = {"S+": "only-listed", "S-": "all-but-listed"}
CONNS = ("AND", "OR")


def _borrow(path, *names):
    """The named top-level functions of `path`, executed from its own source (they may use only json and each other).
    This reuses native_probe / native_e01 code verbatim without importing those scripts, which import torch."""
    tree = pyast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    defs = [n for n in tree.body if isinstance(n, pyast.FunctionDef) and n.name in names]
    if sorted(d.name for d in defs) != sorted(names):
        raise ImportError(f"{Path(path).name}: expected exactly one top-level def of each of {names}")
    ns = {"json": json, "__name__": Path(path).stem}
    exec(compile(pyast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return [ns[n] for n in names]


parse_answer, _no_dup_keys = _borrow(E01N / "native_probe.py", "parse_answer", "_no_dup_keys")
(target_json,) = _borrow(E01N / "native_e01.py", "target_json")


# ----------------------------------------------------------------------------- rendering

def _tf(b):
    return "true" if b else "false"


def _atom_text(a, state, event):
    op, truth = a["op"], _tf(W.eval_atom(a, state, event))
    if op in W.STATUS_OPS:
        return f'status={json.dumps(event["status"])} listed={"yes" if W.listed(a, event) else "no"} {TAG[op]}:{truth}'
    if op == "K+":
        return f'kind={event["kind"]} in={"|".join(a["kinds"])}:{truth}'
    if op in ("AGE", "ALT"):
        return f'amount={event["amount"]} {"at-least" if op == "AGE" else "below"}={a["t"]}:{truth}'
    if op == "BGE":
        return f'balance={state[event["account"]]} at-least={a["t"]}:{truth}'
    raise ValueError(f"unknown atom {op}")


def check_line(rule, state, event):
    """Line 1 for (rule AST, state, event); see the module grammar."""
    comp = rule["op"] in CONNS
    args = rule["args"] if comp else [rule]
    if comp and len(args) < 2:
        raise ValueError("a connective needs at least two arguments")
    parts = [_atom_text(a, state, event) for a in args] + ([rule["op"]] if comp else [])
    return "check " + " ".join(parts) + " -> " + ("apply" if W.eval_rule(rule, state, event) else "noop")


def trace_target(rec):
    """Deterministic two-line R3 target: check_line + "\\n" + native_e01.target_json(after). Fails closed unless line 1
    parses back to the gold decision and the world's after equals the gold after."""
    rule, state, event, gold = rec["rule_ast"], rec["state"], rec["event"], rec["gold"]
    ok, after = W.transition(rule, state, event)
    line1 = check_line(rule, state, event)
    p = parse_check(line1)
    if p is None or p["decision"] != gold["decision"] or ok != gold["ok"] or after != gold["after"] \
            or not _coherent(p):
        raise ValueError(f"line 1 does not round-trip to the gold decision: {line1!r}")
    return line1 + "\n" + target_json(after)


# ----------------------------------------------------------------------------- parsing

_INT = r"(?:0|-?[1-9][0-9]*)"
_ATOM_RE = re.compile(
    r'status=(?P<s>"(?:[^"\\]|\\.)*") listed=(?P<l>yes|no) (?P<p>only-listed|all-but-listed):(?P<sv>true|false)'
    r"|kind=(?P<k>[a-z]+) in=(?P<ks>[a-z]+(?:\|[a-z]+)*):(?P<kv>true|false)"
    rf"|amount=(?P<a>{_INT}) (?P<ar>at-least|below)=(?P<at>{_INT}):(?P<av>true|false)"
    rf"|balance=(?P<b>{_INT}) at-least=(?P<bt>{_INT}):(?P<bv>true|false)")
_LINE_RE = re.compile(r"check (?P<body>.+?)(?: (?P<conn>AND|OR))? -> (?P<dec>apply|noop)")


def _atom_dict(m):
    g = m.groupdict()
    if g["s"] is not None:
        try:
            s = json.loads(g["s"])
        except ValueError:
            return None
        if not isinstance(s, str) or json.dumps(s) != g["s"]:
            return None
        return {"op": "S+" if g["p"] == "only-listed" else "S-", "status": s, "listed": g["l"] == "yes",
                "truth": g["sv"] == "true"}
    if g["k"] is not None:
        return {"op": "K+", "kind": g["k"], "kinds": g["ks"].split("|"), "truth": g["kv"] == "true"}
    if g["a"] is not None:
        return {"op": "AGE" if g["ar"] == "at-least" else "ALT", "amount": int(g["a"]), "t": int(g["at"]),
                "truth": g["av"] == "true"}
    return {"op": "BGE", "balance": int(g["b"]), "t": int(g["bt"]), "truth": g["bv"] == "true"}


def parse_check(line):
    """Strict line-1 parser: dict(atoms=[...], conn=None|'AND'|'OR', decision='apply'|'noop') or None."""
    m = _LINE_RE.fullmatch(line)
    if m is None:
        return None
    body, pos, atoms = m.group("body"), 0, []
    while True:
        a = _ATOM_RE.match(body, pos)
        d = None if a is None else _atom_dict(a)
        if d is None:
            return None
        atoms.append(d)
        pos = a.end()
        if pos == len(body):
            break
        if body[pos] != " ":
            return None
        pos += 1
    if (m.group("conn") is None) != (len(atoms) == 1):
        return None
    return {"atoms": atoms, "conn": m.group("conn"), "decision": m.group("dec")}


def _atom_holds(a):
    """Truth of a parsed atom from its own stated facts, via world.eval_atom on a one-account synthetic event."""
    ast = {"op": a["op"], "set": [a["status"]] if a.get("listed") else [], "kinds": a.get("kinds"), "t": a.get("t")}
    ev = {"account": "_", "other": "_", "status": a.get("status"), "kind": a.get("kind"), "amount": a.get("amount")}
    return W.eval_atom(ast, {"_": a.get("balance")}, ev)


def _coherent(p):
    """Stated truths follow from the stated facts, and the decision from the truths and the connective."""
    truths = [a["truth"] for a in p["atoms"]]
    if any(_atom_holds(a) != t for a, t in zip(p["atoms"], truths)):
        return False
    ok = all(truths) if p["conn"] == "AND" else any(truths)  # single atom: any == the atom
    return ("apply" if ok else "noop") == p["decision"]


def _split(text):
    head, sep, rest = text.strip().partition("\n")
    return head.rstrip(), (rest if sep else None)


def parse_trace_answer(text, keys):
    """-> (after, decision, well_formed).
    - after: the balances from line 2 only, i.e. everything after the first newline of the stripped output, parsed by
      native_probe.parse_answer (same leniency and strictness: duplicate keys, extra keys, non-int values and trailing
      text are rejected). It is returned even when line 1 is malformed. A one-line output (e.g. the bare native
      answer) has no line 2, so after is None.
    - decision: as stated on line 1, or None if line 1 is malformed (it is not checked against the balances here).
    - well_formed: line 1 parses and after is not None."""
    head, rest = _split(text)
    p = parse_check(head)
    after = parse_answer(rest, keys) if rest is not None else None
    return after, (p["decision"] if p else None), bool(p is not None and after is not None)


def predicate_consistency(rec, text):
    """Descriptive line-1 metrics for one output (scoring stays on the line-2 balances via score.classify):
      stated_decision_ok              1 if line 1 states the gold decision, else 0 (a malformed line 1 counts 0)
      balances_match_stated_decision  1/0: the balances' intent (score.classify: unchanged = noop) equals the stated
                                      decision; None if line 1 or line 2 is unusable
      line1_exact                     line 1 is byte-identical to the gold line 1
      line1_coherent                  1/0: the stated truths follow from the stated facts, and the decision from them
                                      and the connective, right or wrong; None if malformed"""
    keys = list(rec["gold"]["after"])
    head, _ = _split(text)
    p = parse_check(head)
    after, dec, wf = parse_trace_answer(text, keys)
    intent = S.classify(rec, after)["intent"]
    return {"well_formed": int(wf), "stated_decision": dec,
            "stated_decision_ok": int(dec is not None and dec == rec["gold"]["decision"]),
            "balances_match_stated_decision": None if dec is None or after is None else int(intent == dec),
            "line1_exact": int(head == check_line(rec["rule_ast"], rec["state"], rec["event"])),
            "line1_coherent": None if p is None else int(_coherent(p))}


# ----------------------------------------------------------------------------- token budget

def token_budget_note(path=LORA_CPU):
    """How rcx/native/lora_cpu.generate bounds and stops decoding, read from its source with ast (no import, no edit).
    Returns dict(accepts_max_new, max_new_default, max_batch_tokens_default, eos_ids, model_generate_kwargs,
    stops_only_at_eos)."""
    tree = pyast.parse(Path(path).read_text(encoding="utf-8"))
    fn = [n for n in tree.body if isinstance(n, pyast.FunctionDef) and n.name == "generate"]
    if len(fn) != 1:
        raise RuntimeError(f"{path}: no unique top-level generate()")
    a = fn[0].args
    pos = a.posonlyargs + a.args
    defaults = {x.arg: pyast.literal_eval(d) for x, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)}
    defaults.update({x.arg: pyast.literal_eval(d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
    eos = [pyast.literal_eval(n.value) for n in tree.body if isinstance(n, pyast.Assign)
           and any(isinstance(t, pyast.Name) and t.id == "EOS_IDS" for t in n.targets)]
    calls = [c for c in pyast.walk(fn[0]) if isinstance(c, pyast.Call) and isinstance(c.func, pyast.Attribute)
             and c.func.attr == "generate"]
    kws = {k.arg: pyast.unparse(k.value) for c in calls for k in c.keywords}
    stop_keys = {"stopping_criteria", "stop_strings", "logits_processor", "max_length", "min_new_tokens"}
    return {"accepts_max_new": "max_new" in [x.arg for x in pos + a.kwonlyargs],
            "max_new_default": defaults.get("max_new"), "max_batch_tokens_default": defaults.get("max_batch_tokens"),
            "eos_ids": eos[0] if len(eos) == 1 else None, "model_generate_kwargs": kws,
            "stops_only_at_eos": kws.get("eos_token_id") == "EOS_IDS" and kws.get("max_new_tokens") == "max_new"
            and not (stop_keys & set(kws))}


def generate_kwargs_for_trace(targets=None, tok=None):
    """Keyword arguments for lora_cpu.generate when decoding R3 traces: {"max_new": n}, never below generate's default.
    - targets=None: the static, prereg-able TRACE_MAX_NEW.
    - Otherwise: the longest target + 1 (EOS), measured in tokens of `tok` if given (tok(t, add_special_tokens=False)
      ["input_ids"]), else in UTF-8 bytes (an upper bound on byte-level BPE tokens).
    Raises if lora_cpu.generate no longer accepts max_new."""
    note = token_budget_note()
    if not note["accepts_max_new"]:
        raise RuntimeError("rcx/native/lora_cpu.generate has no max_new parameter: the R3 eval cannot raise the "
                           "64-token budget through it")
    if targets is None:
        need = TRACE_MAX_NEW
    else:
        need = 1 + max(len(tok(t, add_special_tokens=False)["input_ids"]) if tok is not None
                       else len(t.encode("utf-8")) for t in targets)
    return {"max_new": max(need, note["max_new_default"] or 0)}
