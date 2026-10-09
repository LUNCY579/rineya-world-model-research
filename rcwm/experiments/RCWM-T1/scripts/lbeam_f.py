"""LedgerF: the RCWM-T1 ledger forward model F for BWM's plan_beam (joint roadmap item 3; BWM DECISIONS D-53).

Interface (BWM): predict(keys) with keys = list of (state, action), both hashable; ONE batched generation request per
call for all keys not yet memoised; returns per key (next_state, legal) or None (unsupported / unparseable: the
candidate stops). Memoised on (state, action). Exposes .model_queries (number of NEW model calls, i.e. distinct keys
sent to the model) and .generated_tokens.

  state   lbeam_problems canonical tuple (("A", a), ("B", b), ("C", c), ("D", d))
  action  lbeam_problems event tuple (kind, account, other, amount, status, tick)

Arms
  T2             Titan-Sparse (FP32, fp 818be9c6b6a3) + the PREREG-2 T2 seed-0 LoRA (runs/t2/t2_s0/adapter, sha 92bc7c74...),
                 UNMERGED PEFT (never dense-merged, D-28); the active adapter is set and checked from PEFT's LoRA layer
                 state before every generation
  OFF            the same PeftModel inside model.disable_adapter() (= Titan-Sparse), same process
  ORACLE         rcwmt.env.step, no model (tests, the upper bound); legal True for every key
  IGNORE_STATUS  diagnostic floor: env.step with every status treated as 'posted' (PREREG-2's F_ignore_status, the
                 rule-ablated true simulator); never a comparator for a claim

Frame: LedgerF ALWAYS renders the canonical frame (PREREG-3 = STATUS: TRIGGER; T2 is reliable only inside its trained
surface, so the primary never renders anything else). It has no surface option; before every generation call it
asserts that each prompt is byte-identical to canonical_messages(state, action) = env.one_step_messages with Titan's
verbatim system text and the compact SORTED-key JSON user turn, accounts A-D, statuses posted / void.

Frame and decoding (identical to PREREG-2's ledger chain_one step, rcwmt.generation.chain mode 'chain_one'):
  messages = rcwmt.env.one_step_messages('ledger_replay', before, event) (Titan's verbatim one-step system text, compact
  sorted-key JSON user turn); generation = rcwmt.generation.generate_texts (greedy, do_sample False, EOS [151645, 151643],
  thinking disabled via the chat template, left padding, the registered batching rule of PREREG-1 Amendment 1:
  <= 32 sequences and batch x (prompt + 160) <= 8000 tokens per model.generate call, duty-cycle pacing), max_new_tokens
  = rcwmt.generation.max_tokens('ledger_replay') = 160; parsing = rcwmt.env.parse_state('ledger_replay', text,
  lenient=True). ONE generate_texts call per predict() list; generate_texts itself splits a long list into memory-safe
  model.generate sub-batches (counted in .n_model_generate_calls).

Legality. The ledger family has no illegal events: rcwmt.env.step applies every event to every state (status 'posted':
add / scale / transfer; any other status: no-op, which is the rule the frame states). So every PARSED answer is
returned with legal=True. A parse failure, or an answer whose account set differs from the input state's, is None.

Accounting
  .model_queries      distinct keys newly computed by this object (model arms: sent to the model; ORACLE /
                      IGNORE_STATUS: the evaluations a model would have needed, for budget comparisons)
  .generated_tokens   new tokens generated before EOS, summed over rows (exact, read from the generate() outputs)
  .requested          distinct keys requested since reset_accounting() (cache-independent budget measure; attribute
                      to instances with lbeam_problems.owner_of_action)
  .n_predict, .n_predict_new (predict calls with at least one new key), .n_generate_fn_calls,
  .n_model_generate_calls, .decode_steps, .new_keys (ordered), .texts (raw answers)
Optional persistent cache (cache=path): JSONL of raw answers, appended and fsynced after every batch, reloaded at
construction; the first line records class + arm + frame + identity and a mismatch refuses to load (arms, the
canonical frame and audit surfaces never mix).

BudgetF(inner, owner_of_action, cap) enforces hard per-instance budgets (cap distinct keys per instance; new keys beyond
it return None), e.g. cap = each instance's fixed-candidate call count for a strictly matched-call comparison.

RangeGuardF(inner, applies_to=lbeam_problems.in_range_actions(insts)) is the registered input-range guard for the
in-range primary (MUST-FIX of the 2026-10-09 review; DECISIONS D-34(c)): a key whose INPUT is outside T2's verified
range (any |balance| > 98, an amount T2 was not trained on, a status other than posted / void; lbeam_problems.
in_verified_range) returns None (unsupported: the candidate stops) WITHOUT reaching the inner F, identically for every
arm (T2 / OFF / ORACLE / IGNORE_STATUS / SIM); it reads only F's own imagined input, so it is model-free and allowed by
BWM's None semantics. Refused keys are not model calls; .refused / .oor_key_share() log the out-of-range exposure per
arm. Wrap order for matched budgets: RangeGuardF(BudgetF(LedgerF(...))) (refused keys are never charged). On a
generator-v2 primary set with the registered plan_beam max_depth the guard is INERT for an exact F (every state an
exact search can reach is a closure state, |b| <= 98; tested: 0 refusals under ORACLE / IGNORE_STATUS); it only bites
when a model F imagines a wrong state beyond the bound.

SurfaceAuditF (D-35 surface-form audit; a SEPARATE class, REPORT-ONLY, never the primary; not a LedgerF): renders the
SAME canonical (state, action) in a held-out surface and maps the answer back to canonical form; the planner, proposer,
hooks and problems are unchanged, so every audit arm is paired with the primary by instance. Renderings reuse PREREG-3's
frame helpers (scripts/p3_common.messages / user_json / parse_ledger, imported read-only), i.e. the strata PREREG-3
measured T2 on:
  keys_top_only     'event' first, inner keys sorted                 (PREREG-3 F_keyorder, even variant)
  keys_reversed     'event' first, event and balance keys reverse-sorted (a fixed never-sorted order; D-36 held-out)
  names_letters     accounts A-D rendered as E-H                      (PREREG-3 A_renamed, letters)
  names_words       accounts A-D rendered as bank / card / cash / fees (PREREG-3 A_renamed, words; sorted, so the
                    sorted-key rendering keeps every position)
  status_cancelled  status 'void' rendered as 'cancelled'             (PREREG-3 U_cancelled: an unseen not-applied word)
  status_pending    status 'void' rendered as 'pending'               (PREREG-3 U_pending)
  heldout_combined  keys_reversed + names_words + status_cancelled    (D-36's combined held-out rendering)
The gold never changes: the frame's rule treats every status other than 'posted' as a no-op (env.step), so a rule-level
F plans identically under every surface and a trigger-level F does not. PREREG-3 already measured T2 s0 as trigger-level
on held-out status words, key order and word names, so these arms are descriptive (claim scope), never comparators.

SimulatedF (diagnostic, model-free; for sizing GPU turns only, never a comparator): a deterministic noisy F, keyed by
sha256 on (seed, state, action): a void step is applied as posted w.p. p_void_applied, a posted step's answer has one
account off by 1..9 w.p. p_wrong, the answer is unparseable (None) w.p. p_none. Presets SIM_T2 (p_wrong .09: PREREG-2
four-step exact 0.698 = .914 per step; void 1.00), SIM_T2_P3 (p_wrong .16: PREREG-3 T_posted|at s0 0.84), SIM_OFF
(void applied .90, p_wrong .55: PREREG-2 step level OFF void 0.04-0.13, posted 0.30-0.57).

Model loading is injected: LedgerF(arm, model=..., tok=...) or LedgerF(arm, loader=SharedLoader(fn)) (fn() -> (model,
tok), called lazily once). The GPU runner uses load_registered_t2() INSIDE rcwmt.runtime.gpu_lease(...); this module
never takes the lease and never loads a model by itself. Torch is imported only when a model arm generates.

Batch composition. generate_texts sorts a list by prompt length and left-pads; in FP32 an answer can in principle depend
on its batch-mates at rounding level (PREREG-2's anchor reproduced 100% identical states with identical composition).
With the memo (and cache) F is defined as the FIRST answer computed for a key; batch invariance should be measured at
the GPU anchor before a registered run (open question in docs/ledger_beam/INTERFACE.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True

from contextlib import nullcontext  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rcwmt import env  # noqa: E402  (torch-free)
import lbeam_problems as P  # noqa: E402  (torch-free)

LEDGER = "ledger_replay"
ARMS = ("T2", "OFF", "ORACLE", "IGNORE_STATUS")
MODEL_ARMS = ("T2", "OFF")
T2_ADAPTER_NAME = "t2"                   # PREREG-2's adapter name (confirm2_gen.run_gpu)
CACHE_SCHEMA = "lbeam-f-cache-v3"        # v3: the header records the class and the frame (canonical / audit surface)
CANONICAL = "canonical"

# ---------------------------------------------------------------------------------------------- surfaces (D-35 audit)
WORD_NAMES4 = ("bank", "card", "cash", "fees")      # 4 of p3_common.WORD_NAMES, sorted
SURFACES = {                                        # SurfaceAuditF only; LedgerF never renders these
    "keys_top_only": {"key_order": "top_only"},
    "keys_reversed": {"key_order": "reversed"},
    "names_letters": {"names": ("E", "F", "G", "H")},
    "names_words": {"names": WORD_NAMES4},
    "status_cancelled": {"void_as": "cancelled"},
    "status_pending": {"void_as": "pending"},
    "heldout_combined": {"key_order": "reversed", "names": WORD_NAMES4, "void_as": "cancelled"},
}
_P3C: list = []


def _p3c():
    """scripts/p3_common (PREREG-3's frame helpers; torch-free, no environment side effects), imported read-only."""
    if not _P3C:
        import p3_common
        _P3C.append(p3_common)
    return _P3C[0]


def surface_payload(state: tuple, action: tuple, surface: str) -> tuple:
    """(before dict, event dict, key order or None) of (state, action) rendered in audit surface `surface`."""
    sf = SURFACES[surface]
    before, ev = P.state_to_dict(state), P.action_to_event(action)
    if sf.get("void_as") and ev["status"] == "void":
        ev = {**ev, "status": sf["void_as"]}
    if sf.get("names"):
        mp = dict(zip(P.ACCOUNTS, sf["names"]))
        before = {mp[a]: v for a, v in before.items()}
        ev = {**ev, "account": mp[ev["account"]], "other": mp[ev["other"]]}
    ko = None
    if sf.get("key_order") == "top_only":
        ko = {"top": ["event", "before"], "event": sorted(ev), "before": sorted(before)}
    elif sf.get("key_order") == "reversed":
        ko = {"top": ["event", "before"], "event": sorted(ev, reverse=True), "before": sorted(before, reverse=True)}
    return before, ev, ko


def messages(state: tuple, action: tuple) -> list:
    """Titan's exact one-step ledger frame for (state, action): env.one_step_messages (verbatim system text, compact
    sorted-key JSON user turn). The ONLY rendering LedgerF uses."""
    return env.one_step_messages(LEDGER, P.state_to_dict(state), P.action_to_event(action))


canonical_messages = messages


def is_canonical_frame(msgs, state: tuple, action: tuple) -> bool:
    """msgs is exactly the canonical frame of (state, action): two turns, Titan's verbatim one-step system text, and the
    user turn env.dumps({"before", "event"}) (compact, keys sorted at every level), accounts A-D in order, the six
    event fields, status posted / void."""
    if not (P.is_state(state) and P.is_action(action) and action[4] in ("posted", "void")):
        return False
    if not (isinstance(msgs, list) and len(msgs) == 2 and msgs[0] == {"role": "system", "content": env.SYSTEM_ONE[LEDGER]}
            and isinstance(msgs[1], dict) and msgs[1].get("role") == "user"):
        return False
    u = msgs[1].get("content")
    try:
        d = json.loads(u)
    except (TypeError, ValueError):
        return False
    return (u == env.dumps({"before": P.state_to_dict(state), "event": P.action_to_event(action)})
            and list(d) == ["before", "event"] and list(d["before"]) == list(P.ACCOUNTS)
            and list(d["event"]) == sorted(P.ACTION_FIELDS))


def audit_messages(state: tuple, action: tuple, surface: str) -> list:
    """The same payload rendered in a held-out D-35 audit surface via PREREG-3's frame helpers (SurfaceAuditF only)."""
    before, ev, ko = surface_payload(state, action, surface)
    return _p3c().messages(before, ev, ko)


def parse_answer(text, state: tuple):
    """env.parse_state(ledger, text, lenient=True) (PREREG-2's parser); None if unparseable or if the answer's account
    set differs from the input state's. Returns the canonical next state or None."""
    if text is None:
        return None
    d = env.parse_state(LEDGER, text, lenient=True)
    if d is None or set(d) != {a for a, _ in state}:
        return None
    return P.state_from_dict(d)


def audit_parse(text, state: tuple, surface: str):
    """Audit surfaces: renamed surfaces use PREREG-3's parse_ledger with the rendered names (the same lenient steps),
    mapped back to A-D; key-order and status surfaces use parse_answer."""
    if text is None:
        return None
    names = SURFACES[surface].get("names")
    if not names:
        return parse_answer(text, state)
    d = _p3c().parse_ledger(text, list(names))
    if d is None or set(d) != set(names):
        return None
    inv = dict(zip(names, P.ACCOUNTS))
    return P.state_from_dict({inv[k]: v for k, v in d.items()})


def peft_side(model) -> str:
    """'off' if PEFT's LoRA layers are disabled, else the active adapter name; 'plain' for a model without LoRA."""
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            return "off" if m.disable_adapters else str(m.active_adapters[0])
    return "plain"


def any_merged(model) -> bool:
    from peft.tuners.lora import LoraLayer
    return any(getattr(m, "merged", False) for m in model.modules() if isinstance(m, LoraLayer))


class _CountingModel:
    """Wraps the model handed to the generation function: counts model.generate calls, decode steps and the new tokens
    each row generated before its first EOS (exactly the tokens generate_texts decodes)."""

    def __init__(self, model, eos):
        self._m = model
        self._eos = set(eos)
        self.calls = self.new_tokens = self.decode_steps = self.rows = 0
        self.prompts_per_call: list = []

    def generate(self, *args, **kw):
        g = self._m.generate(*args, **kw)
        ids = kw.get("input_ids", args[0] if args else None)
        seq = g.sequences if hasattr(g, "sequences") else g
        L = ids.shape[1]
        self.calls += 1
        self.prompts_per_call.append(int(ids.shape[0]))
        self.decode_steps += int(seq.shape[1] - L)
        for r in range(seq.shape[0]):
            n = 0
            for t in seq[r, L:].tolist():
                if t in self._eos:
                    break
                n += 1
            self.new_tokens += n
            self.rows += 1
        return g

    def __getattr__(self, name):
        return getattr(self._m, name)


class SharedLoader:
    """fn() -> (model, tok), called at most once and shared by every LedgerF that holds this loader (T2 and OFF read the
    same PeftModel in one process)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self._res = None

    def __call__(self):
        if self._res is None:
            self.calls += 1
            res = self.fn()
            self._res = (res[0], res[1])
        return self._res


class _ModelPathF:
    """Shared machinery of LedgerF and SurfaceAuditF (memo, accounting, the batched model path, the cache). Not used
    directly: subclasses fix the frame (_render / _parse)."""
    frame = None

    def __init__(self, arm: str, model=None, tok=None, loader=None, adapter_name: str | None = None, generate_fn=None,
                 max_new_tokens: int | None = None, cache=None, identity: dict | None = None):
        if type(self) is _ModelPathF:
            raise TypeError("use LedgerF (canonical frame) or SurfaceAuditF (report-only audit)")
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} not in {ARMS}")
        if arm in MODEL_ARMS and model is None and loader is None:
            raise ValueError(f"arm {arm} needs model+tok or a loader")
        self.arm = arm
        self.name = f"ledger-{arm.lower()}" + ("" if self.frame == CANONICAL else f"@{self.frame}")
        self._model, self._tok, self._loader = model, tok, loader
        self.adapter_name = adapter_name
        self.generate_fn = generate_fn
        self.max_new_tokens = max_new_tokens
        self.identity = dict(identity or {})
        self.memo: dict = {}
        self.texts: dict = {}
        self.model_queries = 0
        self.generated_tokens = 0
        self.decode_steps = 0
        self.n_predict = 0
        self.n_predict_new = 0
        self.n_generate_fn_calls = 0
        self.n_model_generate_calls = 0
        self.prompts_per_model_generate: list = []
        self.new_keys: list = []
        self.requested: set = set()
        self.cache = None if cache is None else Path(cache)
        self.cache_loaded = 0
        if self.cache is not None and self.cache.exists():
            self._load_cache()

    # ---------------------------------------------------------------- accounting
    def reset_accounting(self) -> None:
        """Start a new budget window (the memo is kept: F is a frozen function, reuse changes no result)."""
        self.requested = set()

    def calls_by_owner(self, owner_of_action: dict) -> dict:
        out: dict = {}
        for s, a in self.requested:
            k = owner_of_action.get(a)
            out[k] = out.get(k, 0) + 1
        return out

    def stats(self) -> dict:
        return {"class": type(self).__name__, "arm": self.arm, "frame": self.frame, "model_queries": self.model_queries,
                "generated_tokens": self.generated_tokens,
                "requested_unique": len(self.requested), "n_predict": self.n_predict, "n_predict_new": self.n_predict_new,
                "n_generate_fn_calls": self.n_generate_fn_calls, "n_model_generate_calls": self.n_model_generate_calls,
                "decode_steps": self.decode_steps, "memo": len(self.memo),
                "unsupported_in_memo": sum(v is None for v in self.memo.values()), "cache_loaded": self.cache_loaded}

    # ---------------------------------------------------------------- the interface
    def predict(self, keys):
        keys = list(keys)
        self.n_predict += 1
        for k in keys:
            if not (isinstance(k, tuple) and len(k) == 2 and P.is_state(k[0]) and P.is_action(k[1])):
                raise ValueError(f"not a canonical (state, action) key: {k!r}")
        self.requested.update(keys)
        need = list(dict.fromkeys(k for k in keys if k not in self.memo))
        if need:
            self.n_predict_new += 1
            for k, r in zip(need, self._compute(need)):
                self.memo[k] = r
            self.model_queries += len(need)
            self.new_keys.extend(need)
        return [self.memo[k] for k in keys]

    def _compute(self, need: list) -> list:
        """Results for the distinct, not yet memoised keys (one generation call for the model arms)."""
        if self.arm == "ORACLE":
            return [(P.step_state(s, a), True) for s, a in need]
        if self.arm == "IGNORE_STATUS":
            return [(P.step_ignore_status(s, a), True) for s, a in need]
        texts = self._generate(need)
        out = []
        for k, t in zip(need, texts):
            self.texts[k] = t
            ns = self._parse(t, k[0])
            out.append(None if ns is None else (ns, True))
        if self.cache is not None:
            self._append_cache(need)
        return out

    def _render(self, state, action) -> list:
        raise NotImplementedError

    def _parse(self, text, state):
        raise NotImplementedError

    def _check_prompts(self, need: list, msgs: list) -> None:
        pass

    # ---------------------------------------------------------------- model path
    def _ensure_model(self):
        if self._model is None:
            self._model, self._tok = self._loader()
        return self._model, self._tok

    def _generate(self, need: list) -> list:
        model, tok = self._ensure_model()
        from rcwmt import generation          # imports torch (model arms only)
        gen = self.generate_fn or generation.generate_texts
        mnt = self.max_new_tokens or generation.max_tokens(LEDGER)
        msgs = [self._render(s, a) for s, a in need]
        self._check_prompts(need, msgs)
        if any_merged(model):
            raise RuntimeError("a LoRA layer is merged: the T2 adapter must stay unmerged (D-28)")
        proxy = _CountingModel(model, generation.EOS)
        if self.arm == "OFF":
            ctx = model.disable_adapter()
        else:
            if self.adapter_name:
                model.set_adapter(self.adapter_name)
            ctx = nullcontext()
        with ctx:
            side = peft_side(model)
            if self.arm == "OFF" and side != "off":
                raise RuntimeError(f"OFF arm but PEFT side is {side!r}")
            if self.arm == "T2" and (side in ("off", "plain") or (self.adapter_name and side != self.adapter_name)):
                raise RuntimeError(f"T2 arm but PEFT side is {side!r} (adapter {self.adapter_name!r})")
            texts = gen(proxy, tok, msgs, mnt)
        if len(texts) != len(need):
            raise RuntimeError(f"generation returned {len(texts)} texts for {len(need)} prompts")
        self.n_generate_fn_calls += 1
        self.n_model_generate_calls += proxy.calls
        self.prompts_per_model_generate.extend(proxy.prompts_per_call)
        self.generated_tokens += proxy.new_tokens
        self.decode_steps += proxy.decode_steps
        return list(texts)

    # ---------------------------------------------------------------- persistent cache
    def _header(self) -> dict:
        return {"schema": CACHE_SCHEMA, "class": type(self).__name__, "arm": self.arm, "frame": self.frame,
                "identity": self.identity}

    def _load_cache(self) -> None:
        lines = self.cache.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        head = json.loads(lines[0])
        if head != self._header():
            raise RuntimeError(f"cache {self.cache} belongs to {head}, not {self._header()}")
        for line in lines[1:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # torn last line after a crash
            k = (P._tt(r["s"]), P._tt(r["a"]))
            self.texts[k] = r["text"]
            ns = self._parse(r["text"], k[0])
            self.memo[k] = None if ns is None else (ns, True)
            self.cache_loaded += 1

    def _append_cache(self, need: list) -> None:
        new = not self.cache.exists() or self.cache.stat().st_size == 0
        with open(self.cache, "a", encoding="utf-8") as fh:
            if new:
                fh.write(json.dumps(self._header()) + "\n")
            for k in need:
                fh.write(json.dumps({"s": k[0], "a": k[1], "text": self.texts[k]}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


class LedgerF(_ModelPathF):
    """THE forward model F of the joint beam test. Canonical frame only (no surface option): every prompt is
    canonical_messages(state, action), asserted byte for byte before each generation call."""
    frame = CANONICAL

    def __init__(self, arm: str, model=None, tok=None, loader=None, adapter_name: str | None = None, generate_fn=None,
                 max_new_tokens: int | None = None, cache=None, identity: dict | None = None):
        super().__init__(arm, model=model, tok=tok, loader=loader, adapter_name=adapter_name, generate_fn=generate_fn,
                         max_new_tokens=max_new_tokens, cache=cache, identity=identity)

    def _render(self, state, action) -> list:
        return messages(state, action)

    def _parse(self, text, state):
        return parse_answer(text, state)

    def _check_prompts(self, need: list, msgs: list) -> None:
        for (s, a), m in zip(need, msgs):
            if not is_canonical_frame(m, s, a):
                raise RuntimeError(f"LedgerF prompt is not the canonical frame for {(s, a)!r}")


class SurfaceAuditF(_ModelPathF):
    """D-35 surface-form audit arm (REPORT-ONLY; never the primary, never a comparator; not a LedgerF). A model arm
    (T2 / OFF) reading the same canonical (state, action) rendered in a held-out surface (SURFACES); answers are mapped
    back to canonical A-D. Use under the same RangeGuardF as the primary, with its own cache file."""

    def __init__(self, arm: str, surface: str, model=None, tok=None, loader=None, adapter_name: str | None = None,
                 generate_fn=None, max_new_tokens: int | None = None, cache=None, identity: dict | None = None):
        if surface not in SURFACES:
            raise ValueError(f"audit surface {surface!r} not in {tuple(SURFACES)} (the canonical frame is LedgerF's)")
        if arm not in MODEL_ARMS:
            raise ValueError("a surface only changes what a MODEL reads; ORACLE / IGNORE_STATUS are surface-free")
        self.frame = surface
        super().__init__(arm, model=model, tok=tok, loader=loader, adapter_name=adapter_name, generate_fn=generate_fn,
                         max_new_tokens=max_new_tokens, cache=cache, identity=identity)

    @property
    def surface(self) -> str:
        return self.frame

    def _render(self, state, action) -> list:
        return audit_messages(state, action, self.frame)

    def _parse(self, text, state):
        return audit_parse(text, state, self.frame)


# ---------------------------------------------------------------------------------------------- hard per-instance budget
class BudgetF:
    """Hard matched-call budgets: wraps an F and charges every distinct (state, action) key to the instance that owns its
    action (lbeam_problems.owner_of_action). Once an instance has been charged `cap` distinct keys (cap: int, or dict
    id -> int, e.g. that instance's fixed-candidate call count), any NEW key of that instance returns None (unsupported:
    the candidate stops); keys it was already charged for stay free. Keys are charged in list order (deterministic for
    plan_beam). The inner F still receives ONE predict call per list (the admitted keys)."""

    def __init__(self, inner, owner_of_action: dict, cap):
        self.inner, self.owner, self.cap = inner, owner_of_action, cap
        self.charged: dict = {}
        self.refused = 0

    @property
    def model_queries(self):
        return self.inner.model_queries

    @property
    def generated_tokens(self):
        return getattr(self.inner, "generated_tokens", 0)

    def _cap(self, iid) -> int:
        return self.cap.get(iid, 0) if isinstance(self.cap, dict) else int(self.cap)

    def calls_by_owner(self) -> dict:
        return {k: len(v) for k, v in self.charged.items()}

    def predict(self, keys):
        keys = list(keys)
        admit = []
        for k in keys:
            iid = self.owner.get(k[1])
            got = self.charged.setdefault(iid, set())
            if k in got:
                admit.append(True)
            elif len(got) < self._cap(iid):
                got.add(k)
                admit.append(True)
            else:
                admit.append(False)
                self.refused += 1
        inner_keys = [k for k, a in zip(keys, admit) if a]
        res = iter(self.inner.predict(inner_keys) if inner_keys else [])
        return [next(res) if a else None for a in admit]


# ---------------------------------------------------------------------------------------------- input-range guard
def _check_key(k) -> None:
    if not (isinstance(k, tuple) and len(k) == 2 and P.is_state(k[0]) and P.is_action(k[1])):
        raise ValueError(f"not a canonical (state, action) key: {k!r}")


class RangeGuardF:
    """The registered input-range guard of the in-range primary (D-34(c); review MUST-FIX 2026-10-09). A key (state,
    action) whose action is in `applies_to` (None: every key) and whose INPUT is outside T2's verified range
    (lbeam_problems.in_verified_range: any |balance| > bound, an untrained amount, a status other than posted / void)
    returns None (unsupported: BWM's candidate stops) and never reaches the inner F. Identical for every arm; reads only
    F's own imagined input (model-free). The admitted keys of a list go to the inner F in ONE predict call.

    Exposure log (distinct keys): .seen (requested through the guard), .refused (out of range), oor_key_share(),
    exposure(owner_of_action, range_of_id) per range stratum, refused_by_owner(owner_of_action)."""

    def __init__(self, inner, applies_to=None, bound: int = P.IN_RANGE_MAX_ABS):
        self.inner = inner
        self.applies_to = None if applies_to is None else frozenset(applies_to)
        self.bound = bound
        self.seen: set = set()
        self.refused: set = set()
        self.n_predict = 0

    @property
    def model_queries(self):
        return self.inner.model_queries

    @property
    def generated_tokens(self):
        return getattr(self.inner, "generated_tokens", 0)

    @property
    def name(self):
        return f"{getattr(self.inner, 'name', 'f')}+guard{self.bound}"

    def admits(self, k) -> bool:
        return (self.applies_to is not None and k[1] not in self.applies_to) or P.in_verified_range(k[0], k[1], self.bound)

    def predict(self, keys):
        keys = list(keys)
        for k in keys:
            _check_key(k)
        self.n_predict += 1
        adm = [self.admits(k) for k in keys]
        self.seen.update(keys)
        self.refused.update(k for k, a in zip(keys, adm) if not a)
        inner_keys = [k for k, a in zip(keys, adm) if a]
        res = iter(self.inner.predict(inner_keys) if inner_keys else [])
        return [next(res) if a else None for a in adm]

    def calls_by_owner(self, *a):
        return self.inner.calls_by_owner(*a)

    def refused_by_owner(self, owner_of_action: dict) -> dict:
        out: dict = {}
        for s, a in self.refused:
            k = owner_of_action.get(a)
            out[k] = out.get(k, 0) + 1
        return out

    def oor_key_share(self) -> float | None:
        guarded = [k for k in self.seen if self.applies_to is None or k[1] in self.applies_to]
        return round(len(self.refused) / len(guarded), 4) if guarded else None

    def exposure(self, owner_of_action: dict, range_of_id: dict) -> dict:
        """Per range stratum: distinct keys requested, out-of-verified-range keys among them (refused if guarded)."""
        out: dict = {}
        for k in self.seen:
            r = range_of_id.get(owner_of_action.get(k[1]))
            d = out.setdefault(r, {"keys": 0, "outside_verified_range": 0, "refused": 0})
            d["keys"] += 1
            d["outside_verified_range"] += not P.in_verified_range(k[0], k[1], self.bound)
            d["refused"] += k in self.refused
        for d in out.values():
            d["share_outside"] = round(d["outside_verified_range"] / d["keys"], 4) if d["keys"] else None
        return out

    def stats(self) -> dict:
        return {"guard_bound": self.bound, "guarded_scope": "all" if self.applies_to is None else len(self.applies_to),
                "seen_unique": len(self.seen), "refused_unique": len(self.refused), "oor_key_share": self.oor_key_share(),
                "inner": self.inner.stats() if hasattr(self.inner, "stats") else None}


def guarded(f, insts, bound: int = P.IN_RANGE_MAX_ABS) -> RangeGuardF:
    """The registered wrapper for a problem set: RangeGuardF(f, applies_to=in_range_actions(insts)) (the 'in' instances
    are guarded; the report-only 'oor' instances are not)."""
    return RangeGuardF(f, applies_to=P.in_range_actions(insts), bound=bound)


# ---------------------------------------------------------------------------------------------- simulated noisy F
SIM_PRESETS = {
    "SIM_T2": {"p_void_applied": 0.0, "p_wrong": 0.09, "p_none": 0.0},       # PREREG-2: k4 exact 0.698 -> .914 / step
    "SIM_T2_P3": {"p_void_applied": 0.0, "p_wrong": 0.16, "p_none": 0.0},    # PREREG-3: T_posted|at s0 0.84
    "SIM_OFF": {"p_void_applied": 0.90, "p_wrong": 0.55, "p_none": 0.0},     # PREREG-2 OFF: void .04-.13, posted .30-.57
}


class SimulatedF(LedgerF):
    """Diagnostic, model-free noisy F for sizing GPU turns (never a comparator for a claim). Deterministic per key:
    sha256(seed | state | action | tag) uniforms decide whether the answer is None (p_none), whether a void step is
    applied as posted (p_void_applied), and whether a posted step's answer has one account off by +-1..9 (p_wrong).
    Accounting (model_queries, requested, calls_by_owner) is LedgerF's."""

    def __init__(self, preset: str | None = None, p_void_applied: float = 0.0, p_wrong: float = 0.0,
                 p_none: float = 0.0, seed: int = 0):
        super().__init__("ORACLE")
        prm = dict(SIM_PRESETS[preset]) if preset else {"p_void_applied": p_void_applied, "p_wrong": p_wrong,
                                                        "p_none": p_none}
        self.params = {**prm, "seed": seed}
        self.arm = "SIM"
        self.name = f"ledger-sim-{(preset or 'custom').lower()}"

    def _u(self, s, a, tag) -> float:
        h = hashlib.sha256(f"{self.params['seed']}|{s}|{a}|{tag}".encode()).hexdigest()
        return int(h[:13], 16) / 16 ** 13

    def _compute(self, need: list) -> list:
        p = self.params
        out = []
        for s, a in need:
            if self._u(s, a, "none") < p["p_none"]:
                out.append(None)
                continue
            eff = a
            if a[4] != "posted" and self._u(s, a, "void") < p["p_void_applied"]:
                eff = a[:4] + ("posted",) + a[5:]
            ns = P.step_state(s, eff)
            if eff[4] == "posted" and self._u(s, a, "wrong") < p["p_wrong"]:
                v = list(P.vec(ns))
                v[int(self._u(s, a, "acct") * 4)] += (1 + int(self._u(s, a, "mag") * 9)) * (1 if self._u(s, a, "sign") < .5 else -1)
                ns = P.from_vec(tuple(v))
            out.append((ns, True))
        return out


# ---------------------------------------------------------------------------------------------- registered GPU loader
def load_registered_t2(adapter_name: str = T2_ADAPTER_NAME):
    """Titan-Sparse (FP32) + the PREREG-2 T2 seed-0 adapter, unmerged, exactly as scripts/confirm2_gen.run_gpu loads it
    (hashes and fingerprint checked). GPU only: call it INSIDE rcwmt.runtime.gpu_lease(...) from the GPU runner.
    Returns (model, tok, provenance)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("load_registered_t2 is GPU-only (CUDA_VISIBLE_DEVICES is empty): run it from the GPU runner "
                           "inside rcwmt.runtime.gpu_lease(...)")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("load_registered_t2 is GPU-only: no CUDA device")
    import confirm2_gen as G2               # torch-free at import; registered pins
    import peft
    from peft import PeftModel
    from rcwmt import runtime
    ad = G2.ADAPTER_DIR
    a_sha, c_sha = G2.sha256(ad / "adapter_model.safetensors"), G2.sha256(ad / "adapter_config.json")
    if a_sha != G2.ADAPTER_SHA or c_sha != G2.ADAPTER_CONFIG_SHA:
        raise RuntimeError(f"T2 s0 adapter hash {a_sha[:12]}/{c_sha[:12]} != the registered pins")
    base, p = runtime.load_arm("sparse")
    if not p["param_fingerprint"].startswith(G2.FP["sparse"]):
        raise RuntimeError(f"Titan-Sparse fingerprint {p['param_fingerprint'][:12]} != {G2.FP['sparse']}")
    model = PeftModel.from_pretrained(base, str(ad), adapter_name=adapter_name, is_trainable=False).eval()
    if any_merged(model):
        raise RuntimeError("merged LoRA layer after load (D-28)")
    tok = runtime.tokenizer()
    prov = {"sparse_fingerprint": p["param_fingerprint"], "adapter_sha256": a_sha, "adapter_config_sha256": c_sha,
            "adapter_name": adapter_name, "peft": peft.__version__, "runtime": runtime.runtime_info(),
            "registered_runtime": G2.REGISTERED_RUNTIME}
    return model, tok, prov
