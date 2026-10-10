"""Symbol-token decoder-only transformer for RCL (RCX-E01 mechanism study): identifiers are single tokens with
FROZEN RANDOM embeddings, so matching an unseen status word against the RULE line needs identity-based binding.

Tokenization (lossless, ''.join(tokenize(t)) == t): a maximal run of [A-Za-z_-] that contains at least one
letter or '_' is a word; every other character is its own token (so the '-' of '-6' is a minus sign and digits
are single tokens). A word is a BASE token if it occurs in the rcx.rcl.templates pools / frames / clauses or is
a fixed key / literal / kind (EXTRA_BASE); otherwise it must be in the SYMBOL table (all rcx.rcl.vocab pool
words plus the account names), else encode() raises ValueError (fail closed).

Vocabulary layout: [PAD, BOS, SEP, EOS] + CHARS + BASE_WORDS (sorted) + SYMBOLS (sorted). The first
N_TRAINABLE rows of the embedding matrix are trained; the last N_SYMBOLS rows are a frozen buffer drawn from
N(0, symbol_std^2) (default 0.02 = the trainable init std) with a generator seeded by SymLMConfig.symbol_seed.
The buffer is never seen by the optimizer (hence no weight decay, no update) and is saved/restored bit-exactly
with the weights. Input embedding and output head are tied over the full vocabulary, so a symbol can be emitted
by copying.

Architecture, sequence format (BOS + input + SEP + target + EOS, loss on target + EOS), trainer and KV-cache
decoding are rcx.models.lm_core, i.e. exactly char_lm's. Token counts and max_len are in symbol-tokenizer tokens.

Training notes (membership toy 'X,Y:Z' -> true/false, 2 layers, d64, trained on NONCE['train'], tested on
unseen NONCE['dev']; see tests/test_sym_lm.py TestLearnability):
- Binding sits behind a loss plateau set by the number of samples seen (~50K to leave chance at 5 tokens, about
  the same at batch 32 and 256) that grows steeply with sequence length (11-token formats at batch 32 were still
  at chance after 48K samples). Trainable symbol rows plateau the same way, so this is an optimisation property
  of the relational circuit, not of the frozen table. lr 1e-2 is unstable; 3e-3 works.
- With the fixed frozen table the model also memorises codes for the specific training vectors: fresh pairs of
  training symbols reach 96-98% while unseen symbols peak at 0.86-0.94 and then erode (to 0.81-0.83) as training
  continues. symbol_resample=True removes this (unseen == seen: 0.95 at 180K samples, 0.98 at 230K).
- Emitting a symbol through the tied head is slow at symbol_std 0.02 (logit = h . e_s, |e_s| ~ 0.02 sqrt(d)):
  copying the query symbol reached 69% at 600 steps (batch 32), vs 97% at 300 steps with symbol_std 0.5.
  Symbol scale did not speed up the matching itself (inputs are layer-normed).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

import torch

from rcx.rcl import templates as _T
from rcx.rcl import vocab as _V

from . import lm_core as core
from .lm_core import BOS, EOS, IGNORE, N_SPECIAL, PAD, SEP, SPECIAL_CHAR, lm_loss, param_count  # noqa: F401

WORD_RE = r"[A-Za-z_-]*[A-Za-z_][A-Za-z_-]*"
_TOKEN_RE = re.compile(WORD_RE + r"|[\s\S]")
_WORD_FULL = re.compile(WORD_RE)

# Words that are not in the template strings but are fixed parts of the RCL formats (SPEC_RCL 3-4).
EXTRA_BASE = ("RULE", "STATE", "EVENT",
              "account", "amount", "kind", "other", "status",
              "after", "cmp", "calc", "fill", "listed", "ok",
              "true", "false", "null",
              "add", "transfer", "scale", "set", "swap")
ACCOUNT_NAMES = tuple("ABCDEFGH") + tuple("PQRSTU") + ("north", "south", "east", "west", "pine", "oak")
# Single-character tokens: newline + printable ASCII except letters and '_' (those always belong to words).
CHARS = "\n" + "".join(c for c in map(chr, range(32, 127)) if not re.fullmatch(r"[A-Za-z_]", c))
DIGITS = "0123456789"


def template_strings():
    """Every rule-template string of rcx.rcl.templates with its {slots} intact (pools, frames, clauses, solo)."""
    out = []
    for pool in (_T.SPOS, _T.SNEG, _T.AND_FRAMES, _T.OR_FRAMES, _T.CLAUSE):
        out += [pool[k] for k in sorted(pool)]
    out.append(_T.render_rule({"op": "K+", "kinds": ["add"]}, "t0"))  # single non-status atom frame
    return out


def _words(text):
    return [m.group(0) for m in _TOKEN_RE.finditer(text) if _WORD_FULL.fullmatch(m.group(0))]


def _base_words():
    words = {w for s in template_strings() for w in _words(re.sub(r"\{[A-Z0-9]+\}", " ", s))}
    return tuple(sorted(words | set(EXTRA_BASE)))


def _symbol_words():
    pools = (_V.REAL_TRAIN + _V.REAL_DEV + _V.REAL_SEALED + tuple(w for sp in sorted(_V.NONCE) for w in _V.NONCE[sp])
             + (_V.ANCHOR_APPLY, _V.ANCHOR_NOOP) + _V.NEARMISS_VOID + _V.POSTEDLIKE + ACCOUNT_NAMES)
    return tuple(sorted(set(pools)))


class SymTokenizer:
    """Deterministic lossless tokenizer over the closed RCL vocabulary (see module docstring).

    Attributes: tokens (id -> piece; specials are None), n_trainable, n_symbols, vocab_size, symbol_offset,
    base_words, symbols, fingerprint (sha256 of the token table; stored in checkpoints).
    """

    def __init__(self):
        self.base_words = _base_words()
        self.symbols = _symbol_words()
        clash = set(self.base_words) & set(self.symbols)
        if clash:
            raise RuntimeError(f"words are both BASE and SYMBOL: {sorted(clash)}")
        bad = [w for w in self.base_words + self.symbols if not _WORD_FULL.fullmatch(w)]
        if bad:
            raise RuntimeError(f"table words that do not tokenize as one word: {bad}")
        self.tokens = (None,) * N_SPECIAL + tuple(CHARS) + self.base_words + self.symbols
        self.symbol_offset = N_SPECIAL + len(CHARS) + len(self.base_words)
        self.n_trainable = self.symbol_offset
        self.n_symbols = len(self.symbols)
        self.vocab_size = len(self.tokens)
        self._ids = {t: i for i, t in enumerate(self.tokens) if t is not None}
        if len(self._ids) != self.vocab_size - N_SPECIAL:
            raise RuntimeError("duplicate tokens in the vocabulary")
        h = hashlib.sha256("\x00".join(t for t in self.tokens if t is not None).encode())
        h.update(f"|{N_SPECIAL}|{self.symbol_offset}|{self.vocab_size}".encode())
        self.fingerprint = h.hexdigest()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Partition text into pieces (words and single characters); ''.join(pieces) == text."""
        return [m.group(0) for m in _TOKEN_RE.finditer(text)]

    def classify(self, piece: str) -> str:
        """'base' | 'symbol' | 'digit' | 'char' for an in-vocabulary piece; ValueError otherwise."""
        i = self._ids.get(piece)
        if i is None:
            raise ValueError(f"{piece!r} is not in the sym_lm vocab")
        if i >= self.symbol_offset:
            return "symbol"
        if i >= N_SPECIAL + len(CHARS):
            return "base"
        return "digit" if piece in DIGITS else "char"

    def encode(self, text: str) -> list[int]:
        """Encode text as token ids (no specials added); raise ValueError on unknown words or characters."""
        out = []
        for piece in self.tokenize(text):
            i = self._ids.get(piece)
            if i is None:
                what = "word" if _WORD_FULL.fullmatch(piece) else "character"
                raise ValueError(f"{what} {piece!r} is not in the sym_lm vocab")
            out.append(i)
        return out

    def decode(self, ids) -> str:
        """Concatenate token pieces. Special ids (PAD/BOS/SEP/EOS) become U+FFFD; out-of-range ids raise ValueError."""
        out = []
        for i in ids:
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"id {i} is out of range")
            out.append(self.tokens[i] if i >= N_SPECIAL else SPECIAL_CHAR)
        return "".join(out)

    def is_symbol(self, i: int) -> bool:
        return self.symbol_offset <= int(i) < self.vocab_size

    def symbol_mask(self) -> torch.Tensor:
        """Bool (vocab_size,) mask of symbol ids."""
        m = torch.zeros(self.vocab_size, dtype=torch.bool)
        m[self.symbol_offset:] = True
        return m


TOKENIZER = SymTokenizer()
encode, decode, tokenize, is_symbol = TOKENIZER.encode, TOKENIZER.decode, TOKENIZER.tokenize, TOKENIZER.is_symbol
VOCAB_SIZE, N_SYMBOLS, N_TRAINABLE = TOKENIZER.vocab_size, TOKENIZER.n_symbols, TOKENIZER.n_trainable
SYMBOL_OFFSET = TOKENIZER.symbol_offset


# ----------------------------------------------------------------------------------------------- model

@dataclass
class SymLMConfig:
    """Architecture hyperparameters (max_len in tokens). symbol_seed seeds the frozen symbol embeddings, whose
    std defaults to the trainable init std (lm_core.INIT_STD = 0.02).

    symbol_resample (binding-only control, default off): in train mode every forward pass uses a FRESH random
    symbol table (drawn from the global torch RNG, so runs stay deterministic and resumable); eval mode and
    decoding use the canonical frozen table. Nothing symbol-specific (including 'posted' triggers) can then be
    learned. vocab_size / n_symbols record the tokenizer table sizes and must match it.
    """
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_len: int = 256
    symbol_seed: int = 0
    symbol_std: float = core.INIT_STD
    symbol_resample: bool = False
    rope_base: float = 10000.0
    vocab_size: int = VOCAB_SIZE
    n_symbols: int = N_SYMBOLS

    def __post_init__(self):
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer")
        if (self.vocab_size, self.n_symbols) != (VOCAB_SIZE, N_SYMBOLS):
            raise ValueError(f"config vocab ({self.vocab_size}, {self.n_symbols} symbols) does not match the "
                             f"tokenizer ({VOCAB_SIZE}, {N_SYMBOLS} symbols)")


def symbol_table(config: SymLMConfig) -> torch.Tensor:
    """The frozen (n_symbols, d_model) symbol embeddings for a config: N(0, symbol_std^2), seeded by symbol_seed
    with a private generator (the global torch RNG is not consumed)."""
    g = torch.Generator().manual_seed(config.symbol_seed)
    return torch.randn(config.n_symbols, config.d_model, generator=g) * config.symbol_std


class SymLM(core.DecoderLM):
    """lm_core.DecoderLM whose embedding matrix is cat(trainable emb [N_TRAINABLE rows], frozen sym_emb buffer)."""

    def __init__(self, config: SymLMConfig):
        super().__init__(config, n_rows=config.vocab_size - config.n_symbols)
        self.register_buffer("sym_emb", symbol_table(config))  # persistent, requires_grad=False, not a Parameter

    def embedding_matrix(self):
        sym = self.sym_emb
        if self.training and self.config.symbol_resample:
            sym = torch.randn(sym.shape, dtype=sym.dtype, device=sym.device) * self.config.symbol_std
        return torch.cat((self.emb.weight, sym), 0)


def frozen_param_count(model: SymLM) -> int:
    return model.sym_emb.numel()


def save(model: SymLM, path) -> None:
    """Atomically save config, weights (including the frozen symbol table) and the vocab fingerprint."""
    core.atomic_save({"config": asdict(model.config), "state_dict": model.state_dict(),
                      "vocab": TOKENIZER.fingerprint}, path)


def load(path) -> SymLM:
    """Load a model written by save(); returned in eval mode on CPU. The symbol table comes from the file."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("vocab") != TOKENIZER.fingerprint:
        raise ValueError(f"{path} was saved with a different sym_lm vocabulary")
    model = SymLM(SymLMConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ----------------------------------------------------------------------------------------------- training / decoding

@dataclass
class TrainConfig(core.TrainConfig):
    """lm_core.TrainConfig with a SymLMConfig model (same fields and semantics as char_lm.TrainConfig)."""
    model: SymLMConfig = field(default_factory=SymLMConfig)


def make_batch(records):
    """Right-padded batch (x, y, tokens, target_tokens); y is IGNORE except at target tokens and EOS."""
    return core.make_batch(TOKENIZER, records)


def train(records, cfg: TrainConfig, seed: int, log_path=None, guard=None, eval_fn=None):
    """Train a fresh SymLM; same contract as char_lm.train (see lm_core.train). Returns (model, history)."""
    return core.train(SymLM, TOKENIZER, records, cfg, seed, log_path=log_path, guard=guard, eval_fn=eval_fn)


def generate(model: SymLM, inputs: list[str], max_new: int, batch_size: int = 64, return_done: bool = False):
    """Batched KV-cache greedy decoding (== generate_naive). max_new counts new tokens including EOS."""
    return core.generate(model, TOKENIZER, inputs, max_new, batch_size=batch_size, return_done=return_done)


def generate_naive(model: SymLM, inputs: list[str], max_new: int, return_done: bool = False):
    """Reference greedy decoding: one sequence at a time, full recompute per token."""
    return core.generate_naive(model, TOKENIZER, inputs, max_new, return_done=return_done)
