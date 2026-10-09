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
  .n_predict, .n_generate_fn_calls, .n_model_generate_calls, .decode_steps, .new_keys (ordered), .texts (raw answers)
Optional persistent cache (cache=path): JSONL of raw answers, appended and fsynced after every batch, reloaded at
construction; the first line records arm + identity and a mismatch refuses to load (arms never mix).

Model loading is injected: LedgerF(arm, model=..., tok=...) or LedgerF(arm, loader=SharedLoader(fn)) (fn() -> (model,
tok), called lazily once). The GPU runner uses load_registered_t2() INSIDE rcwmt.runtime.gpu_lease(...); this module
never takes the lease and never loads a model by itself. Torch is imported only when a model arm generates.
"""
from __future__ import annotations

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
CACHE_SCHEMA = "lbeam-f-cache-v1"


def messages(state: tuple, action: tuple) -> list:
    """Titan's exact one-step ledger frame for (state, action)."""
    return env.one_step_messages(LEDGER, P.state_to_dict(state), P.action_to_event(action))


def parse_answer(text, state: tuple):
    """env.parse_state(ledger, text, lenient=True) (PREREG-2's parser); None if unparseable or if the answer's account
    set differs from the input state's. Returns the canonical next state or None."""
    if text is None:
        return None
    d = env.parse_state(LEDGER, text, lenient=True)
    if d is None or set(d) != {a for a, _ in state}:
        return None
    return P.state_from_dict(d)


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


class LedgerF:
    def __init__(self, arm: str, model=None, tok=None, loader=None, adapter_name: str | None = None, generate_fn=None,
                 max_new_tokens: int | None = None, cache=None, identity: dict | None = None):
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} not in {ARMS}")
        if arm in MODEL_ARMS and model is None and loader is None:
            raise ValueError(f"arm {arm} needs model+tok or a loader")
        self.arm = arm
        self.name = f"ledger-{arm.lower()}"
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
        return {"arm": self.arm, "model_queries": self.model_queries, "generated_tokens": self.generated_tokens,
                "requested_unique": len(self.requested), "n_predict": self.n_predict,
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
            if self.arm == "ORACLE":
                for k in need:
                    self.memo[k] = (P.step_state(k[0], k[1]), True)
            elif self.arm == "IGNORE_STATUS":
                for k in need:
                    self.memo[k] = (P.step_ignore_status(k[0], k[1]), True)
            else:
                texts = self._generate(need)
                for k, t in zip(need, texts):
                    self.texts[k] = t
                    ns = parse_answer(t, k[0])
                    self.memo[k] = None if ns is None else (ns, True)
                if self.cache is not None:
                    self._append_cache(need)
            self.model_queries += len(need)
            self.new_keys.extend(need)
        return [self.memo[k] for k in keys]

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
        msgs = [messages(s, a) for s, a in need]
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
        return {"schema": CACHE_SCHEMA, "arm": self.arm, "identity": self.identity}

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
            ns = parse_answer(r["text"], k[0])
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
