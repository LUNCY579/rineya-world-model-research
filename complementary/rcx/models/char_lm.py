"""Character-level decoder-only transformer for seq2seq-as-LM tasks, tuned for small CPU runs.

Sequence format: BOS + input chars + SEP + target chars + EOS. The loss covers target chars and EOS only.
Training right-pads to the batch maximum; greedy decoding left-pads and uses a KV cache with per-row
position ids, and is checked to match naive full-recompute decoding exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3
N_SPECIAL = 4
CHARS = "\n" + "".join(map(chr, range(32, 127)))
VOCAB_SIZE = N_SPECIAL + len(CHARS)
IGNORE = -100
SPECIAL_CHAR = "�"
_STOI = {c: i + N_SPECIAL for i, c in enumerate(CHARS)}
_BOOKKEEPING = ("log_every", "eval_every", "guard_every", "checkpoint_path", "checkpoint_every")


def encode(text: str) -> list[int]:
    """Encode text as char ids (no specials added); raise ValueError on chars outside the vocab."""
    try:
        return [_STOI[c] for c in text]
    except KeyError as e:
        raise ValueError(f"character {e.args[0]!r} is not in the char_lm vocab") from None


def decode(ids) -> str:
    """Decode ids to text. Special ids (PAD/BOS/SEP/EOS) become U+FFFD; out-of-range ids raise ValueError."""
    out = []
    for i in ids:
        i = int(i)
        if not 0 <= i < VOCAB_SIZE:
            raise ValueError(f"id {i} is out of range")
        out.append(CHARS[i - N_SPECIAL] if i >= N_SPECIAL else SPECIAL_CHAR)
    return "".join(out)


@dataclass
class CharLMConfig:
    """Architecture hyperparameters. max_len bounds the number of positions fed to the model."""
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_len: int = 256
    rope_base: float = 10000.0
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self):
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer")


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class _Block(nn.Module):
    def __init__(self, c: CharLMConfig):
        super().__init__()
        self.h, self.hd, self.p = c.n_heads, c.d_model // c.n_heads, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.fc1 = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.fc2 = nn.Linear(c.d_ff, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, mask=None, kv=None, start=0):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if kv is not None:
            kc, vc = kv
            kc[:, :, start:start + T] = k
            vc[:, :, start:start + T] = v
            k, v = kc[:, :, :start + T], vc[:, :, :start + T]
        p = self.p if self.training else 0.0
        if mask is None:
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class CharLM(nn.Module):
    """Pre-LN decoder-only transformer with RoPE, SDPA causal attention, GELU MLP and tied embeddings."""

    def __init__(self, config: CharLMConfig):
        super().__init__()
        self.config = c = config
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        hd = c.d_model // c.n_heads
        inv = 1.0 / c.rope_base ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
        ang = torch.outer(torch.arange(c.max_len, dtype=torch.float64), inv)
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            for w in (b.proj.weight, b.fc2.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * c.n_layers))

    def forward(self, idx, pos=None, mask=None, cache=None, start=0):
        """Return logits (B, T, V).

        Training / naive path: only `idx`; causal attention, positions 0..T-1 (right padding is harmless).
        Cached path: `pos` (B, T) position ids, `mask` (B, 1, T, start+T) bool (True = attend) and `cache`
        from new_cache(); keys/values for the T new tokens are written at cache slots start..start+T-1.
        """
        T = idx.shape[1]
        if pos is None:
            if cache is not None or mask is not None:
                raise ValueError("cached / masked calls need explicit position ids")
            if T > self.config.max_len:
                raise ValueError(f"sequence length {T} exceeds max_len {self.config.max_len}")
            cos, sin = self.cos[:T], self.sin[:T]
        else:
            if mask is None:
                raise ValueError("calls with position ids need an attention mask")
            cos, sin = self.cos[pos].unsqueeze(1), self.sin[pos].unsqueeze(1)
        x = self.drop(self.emb(idx))
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, mask, None if cache is None else cache[i], start)
        return F.linear(self.ln_f(x), self.emb.weight)

    def new_cache(self, batch: int, length: int):
        """Allocate a zeroed per-layer (k, v) cache of shape (batch, n_heads, length, head_dim)."""
        c, w = self.config, self.emb.weight
        shape = (batch, c.n_heads, length, c.d_model // c.n_heads)
        return [tuple(torch.zeros(shape, dtype=w.dtype, device=w.device) for _ in range(2))
                for _ in range(c.n_layers)]


def param_count(model: nn.Module) -> int:
    """Number of unique trainable parameters (tied embeddings counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _atomic_save(obj, path):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save(model: CharLM, path) -> None:
    """Atomically save config and weights to `path`."""
    _atomic_save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load(path) -> CharLM:
    """Load a model written by save(); returned in eval mode on CPU."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = CharLM(CharLMConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ----------------------------------------------------------------------------------------------- batching

def _encode_record(rec):
    n_in = len(rec["input"])
    return torch.tensor([BOS] + encode(rec["input"]) + [SEP] + encode(rec["target"]) + [EOS]), n_in


def _collate(items):
    T = max(len(s) for s, _ in items) - 1
    x = torch.full((len(items), T), PAD, dtype=torch.long)
    y = torch.full((len(items), T), IGNORE, dtype=torch.long)
    tokens = target_tokens = 0
    for i, (s, n_in) in enumerate(items):
        L = len(s)
        x[i, :L - 1] = s[:-1]
        y[i, n_in + 1:L - 1] = s[n_in + 2:]
        tokens += L - 1
        target_tokens += L - n_in - 2
    return x, y, tokens, target_tokens


def make_batch(records):
    """Right-padded batch (x, y, tokens, target_tokens); y is IGNORE except at target chars and EOS."""
    return _collate([_encode_record(r) for r in records])


def lm_loss(logits, y):
    """Mean cross-entropy over supervised positions (y != IGNORE)."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


class _Sampler:
    """Seeded index stream: a fresh shuffle each epoch, optionally length-bucketed batches."""

    def __init__(self, lengths, batch_size, seed, bucket):
        self.lengths, self.bs, self.bucket = lengths, batch_size, bucket
        self.rng = random.Random(seed)
        self.perm, self.pos, self.epoch, self.queue = [], 0, 0, []

    def _take(self, k):
        out = []
        while len(out) < k:
            if self.pos >= len(self.perm):
                self.perm = list(range(len(self.lengths)))
                self.rng.shuffle(self.perm)
                self.pos, self.epoch = 0, self.epoch + 1
            m = min(k - len(out), len(self.perm) - self.pos)
            out += self.perm[self.pos:self.pos + m]
            self.pos += m
        return out

    def next(self):
        if self.bucket <= 0:
            return self._take(self.bs)
        if not self.queue:
            pool = sorted(self._take(self.bs * self.bucket), key=self.lengths.__getitem__)
            self.queue = [pool[i:i + self.bs] for i in range(0, len(pool), self.bs)]
            self.rng.shuffle(self.queue)
        return self.queue.pop()

    def state(self):
        return {"rng": self.rng.getstate(), "perm": self.perm, "pos": self.pos, "epoch": self.epoch,
                "queue": self.queue}

    def restore(self, s):
        v, st, g = s["rng"]
        self.rng.setstate((v, tuple(st), g))
        self.perm, self.pos, self.epoch, self.queue = s["perm"], s["pos"], s["epoch"], s["queue"]


# ----------------------------------------------------------------------------------------------- training

@dataclass
class TrainConfig:
    """Optimisation and bookkeeping settings for train().

    bucket > 0 sorts pools of bucket * batch_size examples by length before batching (less padding).
    Intervals are in steps: log_every=0 disables logging and guard_every=0 never calls the guard;
    eval_every=0 / checkpoint_every=0 keep only the final eval (if eval_fn) / checkpoint (if checkpoint_path).
    An existing checkpoint_path is resumed automatically; keep checkpoint_every a multiple of log_every so
    resumed logs average the same windows.
    """
    model: CharLMConfig = field(default_factory=CharLMConfig)
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_lr_frac: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    bucket: int = 0
    log_every: int = 50
    eval_every: int = 0
    guard_every: int = 10
    checkpoint_path: str | None = None
    checkpoint_every: int = 0


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr_frac * cfg.lr at cfg.steps."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _fingerprint(records):
    h = hashlib.sha1()
    for r in records:
        h.update(r["input"].encode())
        h.update(b"\x00")
        h.update(r["target"].encode())
        h.update(b"\x01")
    return f"{len(records)}:{h.hexdigest()}"


def _log_lines(path, last_step):
    """JSONL lines of an existing log whose step <= last_step (malformed lines dropped)."""
    if not os.path.exists(path):
        return []
    keep = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line)["step"] <= last_step:
                    keep.append(line.rstrip("\n") + "\n")
            except (ValueError, KeyError, TypeError):
                continue
    return keep


def _optimizer(model, cfg):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))


def train(records, cfg: TrainConfig, seed: int, log_path=None, guard=None, eval_fn=None):
    """Train a fresh CharLM on records ({'input', 'target'} dicts); return (model in eval mode, history).

    Deterministic given seed (CPU). guard.check() is called every cfg.guard_every steps and may block; its
    wait is excluded from tokens/s. eval_fn(model, step) -> JSON-able dict runs in eval mode every
    cfg.eval_every steps and at the end. log_path receives JSONL lines mirroring history: a fresh run truncates
    it, a resumed run keeps only the records up to the checkpoint step and appends from there. history holds
    'log', 'evals', 'tokens' (non-pad positions fed), 'target_tokens' (supervised), 'padded_tokens',
    'steps', 'epochs', 'elapsed_s', 'compute_s', 'guard_wait_s', 'params', 'resumed_from'.
    """
    if not records:
        raise ValueError("no records")
    items = [_encode_record(r) for r in records]
    longest = max(len(s) for s, _ in items) - 1
    if longest > cfg.model.max_len:
        raise ValueError(f"longest record needs {longest} positions > max_len {cfg.model.max_len}")
    torch.manual_seed(seed)
    model = CharLM(cfg.model)
    opt = _optimizer(model, cfg)
    sampler = _Sampler([len(s) for s, _ in items], cfg.batch_size, seed, cfg.bucket)
    hist = {"log": [], "evals": [], "tokens": 0, "target_tokens": 0, "padded_tokens": 0, "steps": 0,
            "epochs": 0, "elapsed_s": 0.0, "compute_s": 0.0, "guard_wait_s": 0.0,
            "params": param_count(model), "resumed_from": None}
    cfg_d = {k: v for k, v in asdict(cfg).items() if k not in _BOOKKEEPING}
    meta = json.dumps({"seed": seed, "data": _fingerprint(records), "cfg": cfg_d}, sort_keys=True)
    start = 0
    ckpt = cfg.checkpoint_path
    if ckpt and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        if ck["meta"] != meta:
            raise ValueError(f"checkpoint {ckpt} does not match this run (seed, data or config differ)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"])
        state = json.loads(ck["state"])
        sampler.restore(state["sampler"])
        hist, start = state["history"], ck["step"]
        hist["resumed_from"] = start

    def save_ckpt(step):
        state = json.dumps({"sampler": sampler.state(), "history": hist}, default=str)
        _atomic_save({"meta": meta, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                      "torch_rng": torch.get_rng_state(), "state": state}, ckpt)

    log_f = None
    if log_path:
        keep = _log_lines(log_path, start) if start else []
        with open(log_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(keep)
        log_f = open(log_path, "a", encoding="utf-8", newline="\n")

    def emit(rec):
        if log_f:
            log_f.write(json.dumps(rec, default=str) + "\n")
            log_f.flush()

    t_run, elapsed0 = time.perf_counter(), hist["elapsed_s"]
    win = {"loss": 0.0, "n": 0, "tok": 0, "t": 0.0}
    model.train()
    try:
        for step in range(start, cfg.steps):
            if guard is not None and cfg.guard_every > 0 and step % cfg.guard_every == 0:
                t = time.perf_counter()
                guard.check()
                hist["guard_wait_s"] += time.perf_counter() - t
            t0 = time.perf_counter()
            x, y, tok, sup = _collate([items[i] for i in sampler.next()])
            lr = lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            loss = lm_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) if cfg.grad_clip > 0 else None
            opt.step()
            dt = time.perf_counter() - t0
            done = step + 1
            hist["tokens"] += tok
            hist["target_tokens"] += sup
            hist["padded_tokens"] += x.numel()
            hist["compute_s"] += dt
            hist["steps"], hist["epochs"] = done, sampler.epoch
            hist["elapsed_s"] = elapsed0 + time.perf_counter() - t_run
            win["loss"] += loss.item()
            win["n"] += 1
            win["tok"] += tok
            win["t"] += dt
            if cfg.log_every > 0 and (done % cfg.log_every == 0 or done == cfg.steps):
                rec = {"step": done, "loss": win["loss"] / win["n"], "lr": lr,
                       "grad_norm": None if gn is None else float(gn), "tok_s": win["tok"] / win["t"],
                       "step_s": win["t"] / win["n"], "elapsed_s": hist["elapsed_s"], "tokens": hist["tokens"],
                       "target_tokens": hist["target_tokens"], "guard_wait_s": hist["guard_wait_s"]}
                hist["log"].append(rec)
                emit(rec)
                win = {"loss": 0.0, "n": 0, "tok": 0, "t": 0.0}
            if eval_fn is not None and ((cfg.eval_every > 0 and done % cfg.eval_every == 0) or done == cfg.steps):
                rng = torch.get_rng_state()
                model.eval()
                res = eval_fn(model, done)
                model.train()
                torch.set_rng_state(rng)
                hist["evals"].append({"step": done, **(res or {})})
                emit({"step": done, "eval": res})
            if ckpt and ((cfg.checkpoint_every > 0 and done % cfg.checkpoint_every == 0) or done == cfg.steps):
                save_ckpt(done)
    finally:
        if log_f:
            log_f.close()
    return model.eval(), hist


# ----------------------------------------------------------------------------------------------- decoding

def _prompt(text):
    return [BOS] + encode(text) + [SEP]


def _limit(model, n_prompt, max_new):
    if n_prompt > model.config.max_len:
        raise ValueError(f"prompt needs {n_prompt} positions > max_len {model.config.max_len}")
    return max(0, min(max_new, model.config.max_len - n_prompt + 1))


def _greedy_batch(model, prompts, max_new):
    B, lens = len(prompts), [len(p) for p in prompts]
    P = max(lens)
    limits = [_limit(model, n, max_new) for n in lens]
    steps = max(limits)
    outs, fin, active = [[] for _ in prompts], [False] * B, [n > 0 for n in limits]
    if steps == 0:
        return list(zip(outs, fin))
    dev = model.emb.weight.device
    ids = torch.full((B, P), PAD, dtype=torch.long)
    valid = torch.zeros(B, P + steps - 1, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p)
        valid[i, P - len(p):P] = True
    pos = (valid[:, :P].long().cumsum(1) - 1).clamp_(min=0)
    causal = torch.ones(P, P, dtype=torch.bool).tril_()
    mask = (causal & valid[:, None, :P]) | torch.eye(P, dtype=torch.bool)
    cache = model.new_cache(B, P + steps - 1)
    logits = model(ids.to(dev), pos.to(dev), mask.unsqueeze(1).to(dev), cache, 0)[:, -1]
    lens_t = torch.tensor(lens)
    for t in range(steps):
        nxt = logits.argmax(-1)
        for i, tok in enumerate(nxt.tolist()):
            if not active[i]:
                continue
            if tok == EOS:
                active[i], fin[i] = False, True
            else:
                outs[i].append(tok)
                active[i] = len(outs[i]) < limits[i]
        if not any(active):
            break
        valid[:, P + t] = True
        p = (lens_t + t).clamp_(max=model.config.max_len - 1).unsqueeze(1)
        m = valid[:, None, None, :P + t + 1]
        logits = model(nxt.unsqueeze(1), p.to(dev), m.to(dev), cache, P + t)[:, -1]
    return list(zip(outs, fin))


@torch.inference_mode()
def generate(model: CharLM, inputs: list[str], max_new: int, batch_size: int = 64, return_done: bool = False):
    """Batched greedy decoding with a KV cache; returns the decoded target string per input.

    Prompts are BOS + input + SEP, left-padded within length-sorted batches; position ids count only real
    tokens, so results equal generate_naive(). max_new counts new tokens including EOS, so use
    max_target_len + 1 to let the model finish. With return_done, also returns whether each row emitted EOS.
    """
    was_training = model.training
    model.eval()
    try:
        prompts = [_prompt(s) for s in inputs]
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        texts, done = [""] * len(prompts), [False] * len(prompts)
        for j in range(0, len(order), batch_size):
            idx = order[j:j + batch_size]
            for i, (ids, fin) in zip(idx, _greedy_batch(model, [prompts[i] for i in idx], max_new)):
                texts[i], done[i] = decode(ids), fin
    finally:
        model.train(was_training)
    return (texts, done) if return_done else texts


@torch.inference_mode()
def generate_naive(model: CharLM, inputs: list[str], max_new: int, return_done: bool = False):
    """Reference greedy decoding: one sequence at a time, full recompute per token, no cache or padding."""
    was_training = model.training
    model.eval()
    dev = model.emb.weight.device
    texts, done = [], []
    try:
        for s in inputs:
            seq = _prompt(s)
            out, fin = [], False
            for _ in range(_limit(model, len(seq), max_new)):
                tok = int(model(torch.tensor([seq], device=dev))[0, -1].argmax())
                if tok == EOS:
                    fin = True
                    break
                out.append(tok)
                seq.append(tok)
            texts.append(decode(out))
            done.append(fin)
    finally:
        model.train(was_training)
    return (texts, done) if return_done else texts
