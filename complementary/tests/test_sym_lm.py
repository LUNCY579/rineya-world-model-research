"""Tests for rcx.models.sym_lm (symbol-token learner) and the shared rcx.models.lm_core.

Run: python -B -m unittest tests.test_sym_lm -v   (CPU only; RCX_THREADS threads, default 4; about a minute,
most of it the learnability test)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import random
import shutil
import tempfile
import time
import unittest

import torch

torch.set_num_threads(int(os.environ.get("RCX_THREADS", "4")))

from rcx.models import char_lm, lm_core
from rcx.models import sym_lm as S
from rcx.rcl import gen
from rcx.rcl import templates as T
from rcx.rcl import vocab as V
from rcx.rcl import world as W

TOK = S.TOKENIZER
TRAIN_NONCE, DEV_NONCE = list(V.NONCE["train"]), list(V.NONCE["dev"])
SEP_CHARS = [c for c in S.CHARS if c not in "\n-"]  # '-' would glue onto a neighbouring word


def _member_records(rng, words, n, fmt="{x},{y}:{z}"):
    """List of 2 symbols + query symbol -> 'true' iff the query is in the list (balanced)."""
    out = []
    for _ in range(n):
        x, y = rng.sample(words, 2)
        if rng.random() < 0.5:
            z, t = rng.choice((x, y)), "true"
        else:
            z = rng.choice(words)
            while z in (x, y):
                z = rng.choice(words)
            t = "false"
        out.append({"input": fmt.format(x=x, y=y, z=z), "target": t})
    return out


def _rcl_like_records(rng, n):
    """Mix of short and RULE/EVENT-style membership records (varied lengths, symbols in input and target)."""
    long_fmt = 'RULE: Only events with status {x} or {y} take effect.\nEVENT: {{"status":"{z}"}}'
    out = []
    for r in _member_records(rng, TRAIN_NONCE, n):
        if rng.random() < 0.5:
            x, y, z = r["input"].replace(":", ",").split(",")
            r = {"input": long_fmt.format(x=x, y=y, z=z), "target": '{"status":"' + z + '","listed":' + r["target"]
                 + "}"}
        out.append(r)
    return out


def _config(**kw):
    base = dict(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.1, max_len=64, symbol_seed=0)
    base.update(kw)
    return S.SymLMConfig(**base)


def _random_model(seed, dtype=torch.float32, max_len=40):
    torch.manual_seed(seed)
    model = S.SymLM(_config(n_heads=4, max_len=max_len, symbol_seed=seed))
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)
        model.sym_emb.normal_(0.0, 0.5, generator=g)
        model.emb.weight[S.EOS] *= 3.0
    return model.to(dtype).train()


def _random_texts(rng, n, lo, hi):
    """Random encodable texts: words (base or symbol) never adjacent, so the pieces are the exact tokenization."""
    words = list(TOK.base_words) + list(TOK.symbols)
    out = []
    for _ in range(n):
        pieces, k = [], rng.randint(lo, hi)
        while len(pieces) < k:
            if pieces and S._WORD_FULL.fullmatch(pieces[-1]) or rng.random() < 0.4:
                pieces.append(rng.choice(SEP_CHARS + ["\n"]))
            else:
                pieces.append(rng.choice(words))
        out.append("".join(pieces[:k]))
    return out


class _IdCodec:
    """Codec whose decode returns the raw id tuple (exact id comparison of decoders)."""
    encode = staticmethod(S.encode)

    @staticmethod
    def decode(ids):
        return tuple(int(i) for i in ids)


# ----------------------------------------------------------------------------------------------- tokenizer

class TestTokenizer(unittest.TestCase):
    def check_text(self, text):
        pieces = TOK.tokenize(text)
        self.assertEqual("".join(pieces), text)
        ids = TOK.encode(text)
        self.assertEqual(len(ids), len(pieces))
        self.assertEqual(TOK.decode(ids), text)
        self.assertTrue(all(S.N_SPECIAL <= i < S.VOCAB_SIZE for i in ids))
        return ids

    def check_record(self, r):
        ids = self.check_text(r["input"])
        st = r["event"]["status"]
        self.assertEqual(TOK.classify(st), "symbol", st)
        self.assertIn(TOK._ids[st], ids)
        for name in list(r["state"]) + [r["event"]["account"], r["event"]["other"]]:
            self.assertEqual(TOK.classify(name), "symbol", name)
        for fmt, t in r["targets"].items():
            self.check_text(t)
        for w in TOK.tokenize(r["input"].split("\n")[0]):
            if S._WORD_FULL.fullmatch(w) and TOK.classify(w) == "symbol":
                self.assertTrue(w in V.NONCE["train"] + V.NONCE["dev"] or w in V.REAL_TRAIN + V.REAL_DEV
                                or w in (V.ANCHOR_APPLY, V.ANCHOR_NOOP), w)

    def test_lossless_on_rcl_data(self):
        """2,000 records per regime and the dev battery, in both worlds, inputs and ans/trace/fill targets."""
        prev = gen.KINDS
        n_tok = n_chr = 0
        try:
            for world in ("arith", "copy"):
                gen.use_world(world)
                train = []
                for i, regime in enumerate(gen.REGIMES):
                    recs = gen.make_train(regime, 2000, 100 + i)
                    self.assertEqual(len(recs), 2000)
                    train += recs
                units = gen.fill_units(train)
                battery = gen.make_battery("dev", 5, 1)
                for r in train + battery:
                    self.check_record(gen.with_fill(r, units))
                    n_tok += len(TOK.encode(r["input"]))
                    n_chr += len(r["input"])
                kinds = {r["event"]["kind"] for r in train + battery}
                self.assertEqual(kinds, set(gen.WORLDS[world]))
                self.assertEqual({r["surface"] for r in battery}, set(gen.SURFACES))
        finally:
            gen.KINDS = prev
        self.assertLess(n_tok / n_chr, 0.75)

    def test_classification_spot_checks(self):
        self.assertEqual(TOK.tokenize('"amount":-6'), ['"', "amount", '"', ":", "-", "6"])
        self.assertEqual([TOK.classify(p) for p in TOK.tokenize('"amount":-6')],
                         ["char", "base", "char", "char", "char", "digit"])
        for w in ("un-posted", "posted_not", "void_", "nonposted", "Void", "VOID", "voided", "posted", "void",
                  "A", "H", "P", "U", "north", "oak") + V.REAL_SEALED[:2] + V.NONCE["sealed"][:2]:
            self.assertEqual(TOK.tokenize(w), [w])
            self.assertEqual(TOK.classify(w), "symbol", w)
            self.assertTrue(S.is_symbol(S.encode(w)[0]))
        self.assertNotEqual(S.encode("Void"), S.encode("void"))
        self.assertEqual(TOK.tokenize('"status":"un-posted"}'), ['"', "status", '"', ":", '"', "un-posted", '"', "}"])
        for w in ("Only", "RULE", "STATE", "EVENT", "status", "listed", "true", "null", "set", "swap", "scale",
                  "Balances", "either", "below", "balance", "cmp", "calc", "fill", "ok"):
            self.assertEqual(TOK.classify(w), "base", w)
            self.assertFalse(S.is_symbol(S.encode(w)[0]))
        self.assertEqual(TOK.tokenize("12\n -"), ["1", "2", "\n", " ", "-"])
        self.assertEqual([TOK.classify(c) for c in "{}[];.,"], ["char"] * 7)
        self.assertEqual(TOK.tokenize("ab-cd_ ef--3"), ["ab-cd_", " ", "ef--", "3"])
        self.assertEqual(TOK.tokenize("--3"), ["-", "-", "3"])

    def test_tables(self):
        words = set()
        for s in S.template_strings():
            for p in TOK.tokenize(s.replace("{", " ").replace("}", " ")):
                if S._WORD_FULL.fullmatch(p) and p not in ("L", "B", "K", "T", "C", "C1", "C2"):
                    words.add(p)
        self.assertEqual(set(TOK.base_words), words | set(S.EXTRA_BASE))
        pools = (set(V.REAL_TRAIN + V.REAL_DEV + V.REAL_SEALED) | {w for v in V.NONCE.values() for w in v}
                 | {V.ANCHOR_APPLY, V.ANCHOR_NOOP} | set(V.NEARMISS_VOID) | set(V.POSTEDLIKE))
        self.assertEqual(set(TOK.symbols), pools | set(S.ACCOUNT_NAMES))
        self.assertEqual(list(TOK.symbols), sorted(TOK.symbols))
        self.assertEqual(list(TOK.base_words), sorted(TOK.base_words))
        self.assertFalse(set(TOK.symbols) & set(TOK.base_words))
        names = set(gen.LETTERS) | set(gen.LETTERS2) | set(gen.WORDS) | set(W.CANON_NAMES) | {"E", "F"}
        self.assertTrue(names <= set(S.ACCOUNT_NAMES))
        self.assertEqual(S.VOCAB_SIZE, S.N_TRAINABLE + S.N_SYMBOLS)
        self.assertEqual(S.SYMBOL_OFFSET, S.N_TRAINABLE)
        self.assertTrue(S.is_symbol(S.SYMBOL_OFFSET) and S.is_symbol(S.VOCAB_SIZE - 1))
        self.assertFalse(S.is_symbol(S.SYMBOL_OFFSET - 1) or S.is_symbol(S.VOCAB_SIZE) or S.is_symbol(S.EOS))
        self.assertEqual(int(TOK.symbol_mask().sum()), S.N_SYMBOLS)
        again = S.SymTokenizer()
        self.assertEqual((again.tokens, again.fingerprint), (TOK.tokens, TOK.fingerprint))

    def test_unknown_raise(self):
        for bad in ("tavo", "mirek", "pending", "dax", "xelth", "Posted", "x", "_", "-posted", "posted-",
                    "café", "a\tb", "\r", "\x00", "status:tavo"):
            with self.assertRaises(ValueError, msg=bad):
                S.encode(bad)
        with self.assertRaises(ValueError):
            S.decode([S.VOCAB_SIZE])
        with self.assertRaises(ValueError):
            S.decode([-1])
        self.assertEqual(S.decode([S.SEP] + S.encode("posted")), S.SPECIAL_CHAR + "posted")


# ----------------------------------------------------------------------------------------------- model

class TestFrozenSymbols(unittest.TestCase):
    def test_symbol_table(self):
        cfg = _config(d_model=64, symbol_seed=3)
        torch.manual_seed(0)
        a = S.SymLM(cfg)
        torch.manual_seed(1)
        b = S.SymLM(cfg)
        c = S.SymLM(_config(d_model=64, symbol_seed=4))
        self.assertEqual(tuple(a.sym_emb.shape), (S.N_SYMBOLS, 64))
        self.assertTrue(torch.equal(a.sym_emb, b.sym_emb))
        self.assertTrue(torch.equal(a.sym_emb, S.symbol_table(cfg)))
        self.assertFalse(torch.equal(a.sym_emb, c.sym_emb))
        self.assertAlmostEqual(a.sym_emb.std().item(), lm_core.INIT_STD, delta=0.001)
        self.assertAlmostEqual(a.emb.weight.std().item(), lm_core.INIT_STD, delta=0.002)
        self.assertFalse(a.sym_emb.requires_grad)
        self.assertNotIn("sym_emb", dict(a.named_parameters()))
        self.assertIn("sym_emb", a.state_dict())
        self.assertEqual(tuple(a.embedding_matrix().shape), (S.VOCAB_SIZE, 64))
        self.assertEqual(S.param_count(a), sum(p.numel() for p in a.parameters()))
        self.assertEqual(S.frozen_param_count(a), S.N_SYMBOLS * 64)
        opt = lm_core.optimizer(a, S.TrainConfig(model=cfg))
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        self.assertNotIn(id(a.sym_emb), in_opt)
        self.assertEqual(in_opt, {id(p) for p in a.parameters()})
        decay = {id(p) for p in opt.param_groups[0]["params"]}
        self.assertIn(id(a.emb.weight), decay)
        with self.assertRaises(ValueError):
            S.SymLMConfig(vocab_size=S.VOCAB_SIZE + 1)

    def test_symbols_are_used_and_frozen_through_training(self):
        recs = _rcl_like_records(random.Random(5), 200)
        cfg = S.TrainConfig(model=_config(symbol_seed=11), steps=8, batch_size=16, lr=1e-2, warmup=2,
                            weight_decay=0.5, log_every=0)
        torch.manual_seed(0)
        init = S.SymLM(cfg.model)
        model, _ = S.train(recs, cfg, seed=0)
        self.assertTrue(torch.equal(model.sym_emb, S.symbol_table(cfg.model)))
        self.assertTrue(torch.equal(model.sym_emb, init.sym_emb))
        self.assertFalse(torch.equal(model.emb.weight, init.emb.weight))
        x = S.make_batch(recs[:4])[0]
        with torch.no_grad():
            before = model(x)
            model.sym_emb.add_(0.01)
            self.assertFalse(torch.equal(before, model(x)))
        loss = S.lm_loss(model.train()(x), S.make_batch(recs[:4])[1])
        loss.backward()
        self.assertIsNone(model.sym_emb.grad)
        self.assertIsNotNone(model.emb.weight.grad)

    def test_resample_mode(self):
        """symbol_resample: fresh tables in train mode only; training stays deterministic and resumable."""
        mcfg = _config(dropout=0.0, symbol_resample=True)
        torch.manual_seed(0)
        m = S.SymLM(mcfg)
        x = S.make_batch(_rcl_like_records(random.Random(1), 4))[0]
        with torch.no_grad():
            m.train()
            a, b = m(x), m(x)
            m.eval()
            c, d = m(x), m(x)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(c, d))
        self.assertTrue(torch.equal(m.sym_emb, S.symbol_table(mcfg)))
        tmp = tempfile.mkdtemp(prefix="sym_lm_rs_")
        try:
            recs = _rcl_like_records(random.Random(2), 100)
            cfg = S.TrainConfig(model=mcfg, steps=12, batch_size=8, lr=3e-3, warmup=2, log_every=3, guard_every=3,
                                checkpoint_path=os.path.join(tmp, "ck.pt"), checkpoint_every=6)
            ref, _ = S.train(recs, S.TrainConfig(**{**cfg.__dict__, "checkpoint_path": None}), seed=3)
            with self.assertRaises(KeyboardInterrupt):
                S.train(recs, cfg, seed=3, guard=_Guard(fail_at=4))
            res, h = S.train(recs, cfg, seed=3)
            self.assertEqual(h["resumed_from"], 6)
            sd = res.state_dict()
            self.assertTrue(all(torch.equal(v, sd[k]) for k, v in ref.state_dict().items()))
            self.assertTrue(torch.equal(ref.sym_emb, S.symbol_table(mcfg)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- decoding

class TestCachedDecoding(unittest.TestCase):
    def test_cache_matches_naive_greedy(self):
        """Batched left-padded KV-cache decoding == per-sequence full-recompute decoding (ids and texts)."""
        rng = random.Random(0)
        n_eos = n_trunc = n_sym = 0
        for dtype in (torch.float32, torch.float64):
            for seed in range(3):
                model = _random_model(seed, dtype)
                inputs = _random_texts(rng, 23, 0, 34)
                fast = lm_core.generate(model, _IdCodec, inputs, 12, batch_size=7, return_done=True)
                slow = lm_core.generate_naive(model, _IdCodec, inputs, 12, return_done=True)
                self.assertEqual(fast, slow, f"dtype={dtype} seed={seed}")
                texts = S.generate(model, inputs, 12, batch_size=5, return_done=True)
                self.assertEqual(texts, S.generate_naive(model, inputs, 12, return_done=True))
                self.assertEqual(texts[0], [S.decode(ids) for ids in fast[0]])
                self.assertTrue(model.training)
                n_eos += sum(fast[1])
                n_trunc += sum(len(t) < 12 and not d for t, d in zip(*fast))
                n_sym += sum(S.is_symbol(i) for ids in fast[0] for i in ids)
        self.assertGreater(n_eos, 0, "EOS stopping path not exercised")
        self.assertGreater(n_trunc, 0, "max_len truncation path not exercised")
        self.assertGreater(n_sym, 0, "no symbol was ever emitted")

    def test_edge_cases(self):
        model = _random_model(1)
        self.assertEqual(S.generate(model, ["posted", ""], 0), ["", ""])
        self.assertEqual(S.generate(model, [], 5), [])
        with self.assertRaises(ValueError):
            S.generate(model, ["A " * 19 + "A"], 5)   # 39 tokens + BOS + SEP > max_len 40
        with self.assertRaises(ValueError):
            S.generate(model, ["tavo"], 5)
        x = ["A " * 18 + "A"]                          # 37 tokens + 2 = 39 positions
        self.assertEqual(S.generate(model, x, 5, return_done=True)[0], S.generate_naive(model, x, 5))


# ----------------------------------------------------------------------------------------------- loss masking

class TestLossMasking(unittest.TestCase):
    def setUp(self):
        prev = gen.KINDS
        try:
            gen.use_world("copy")
            r = gen.make_train("CF_SD", 3, 4)
        finally:
            gen.KINDS = prev
        self.recs = ([{"input": x["input"], "target": x["targets"]["trace"]} for x in r[:2]]
                     + [{"input": "posted", "target": ""}, {"input": "", "target": "A"}])

    def test_batch_layout(self):
        x, y, tok, sup = S.make_batch(self.recs)
        enc = [(S.encode(r["input"]), S.encode(r["target"])) for r in self.recs]
        self.assertEqual(x.shape, (4, max(len(a) + len(b) + 2 for a, b in enc)))
        self.assertEqual(sup, sum(len(b) + 1 for _, b in enc))
        self.assertEqual(tok, sum(len(a) + len(b) + 2 for a, b in enc))
        for i, (a, b) in enumerate(enc):
            n = len(a) + len(b) + 2
            self.assertEqual(x[i, :n].tolist(), [S.BOS] + a + [S.SEP] + b)
            self.assertTrue((x[i, n:] == S.PAD).all())
            sup_pos = (y[i] != S.IGNORE).nonzero().flatten().tolist()
            self.assertEqual(sup_pos, list(range(len(a) + 1, n)))
            self.assertEqual(y[i, sup_pos].tolist(), b + [S.EOS])
        self.assertTrue(any(S.is_symbol(t) for t in y[0].tolist() if t != S.IGNORE))

    def test_input_positions_get_no_loss(self):
        x, y, _, _ = S.make_batch(self.recs)
        model = _random_model(2, max_len=x.shape[1]).eval()
        logits = model(x).detach().requires_grad_(True)
        loss = S.lm_loss(logits, y)
        loss.backward()
        unsup = y == S.IGNORE
        self.assertEqual(logits.grad[unsup].abs().max().item(), 0.0)
        self.assertTrue((logits.grad[~unsup].abs().sum(-1) > 0).all())
        with torch.no_grad():
            noisy = logits + 50 * torch.randn_like(logits) * unsup.unsqueeze(-1)
            self.assertEqual(S.lm_loss(noisy, y).item(), loss.item())
            torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits[~unsup], y[~unsup]))


# ----------------------------------------------------------------------------------------------- trainer

class _Guard:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def check(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise KeyboardInterrupt("simulated crash")


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sym_lm_test_")
        self.recs = _rcl_like_records(random.Random(3), 300)
        self.cfg = S.TrainConfig(model=_config(symbol_seed=5), steps=30, batch_size=16, lr=3e-3, warmup=5,
                                 bucket=4, log_every=5, guard_every=5)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deterministic_resume_guard_and_log(self):
        a, ha = S.train(self.recs, self.cfg, seed=7)
        a2, _ = S.train(self.recs, self.cfg, seed=7)
        b0, _ = S.train(self.recs, self.cfg, seed=8)
        sd_a, sd_a2 = a.state_dict(), a2.state_dict()
        self.assertTrue(all(torch.equal(sd_a[k], sd_a2[k]) for k in sd_a))
        self.assertFalse(torch.equal(sd_a["emb.weight"], b0.state_dict()["emb.weight"]))
        self.assertTrue(torch.equal(sd_a["sym_emb"], b0.state_dict()["sym_emb"]))  # table depends on symbol_seed

        ck = os.path.join(self.tmp, "ck.pt")
        log = os.path.join(self.tmp, "log.jsonl")
        cfg = S.TrainConfig(**{**self.cfg.__dict__, "checkpoint_path": ck, "checkpoint_every": 10})
        with self.assertRaises(KeyboardInterrupt):
            S.train(self.recs, cfg, seed=7, log_path=log, guard=_Guard(fail_at=4))
        self.assertTrue(os.path.exists(ck))
        guard = _Guard()
        c, hc = S.train(self.recs, cfg, seed=7, log_path=log, guard=guard)
        self.assertEqual(hc["resumed_from"], 10)
        self.assertEqual(guard.calls, 4)
        sd_c = c.state_dict()
        self.assertTrue(all(torch.equal(sd_a[k], sd_c[k]) for k in sd_a))
        self.assertEqual([r["loss"] for r in ha["log"]], [r["loss"] for r in hc["log"]])
        for key in ("tokens", "target_tokens", "padded_tokens", "steps"):
            self.assertEqual(ha[key], hc[key], key)
        with open(log, encoding="utf-8") as f:
            self.assertEqual([json.loads(line) for line in f], json.loads(json.dumps(hc["log"])))
        with self.assertRaises(ValueError):
            S.train(self.recs[:-1], cfg, seed=7)
        other = S.TrainConfig(**{**cfg.__dict__, "model": _config(symbol_seed=6)})
        with self.assertRaises(ValueError):
            S.train(self.recs, other, seed=7)
        too_short = S.TrainConfig(**{**self.cfg.__dict__, "model": _config(max_len=20)})
        with self.assertRaises(ValueError):
            S.train(self.recs, too_short, seed=7)

    def test_token_counts_eval_and_save_load(self):
        cfg = S.TrainConfig(**{**self.cfg.__dict__, "steps": 3, "bucket": 0, "batch_size": 300, "log_every": 1})
        calls = []
        eval_fn = lambda m, step: calls.append((step, m.training)) or {"n": len(calls)}
        model, hist = S.train(self.recs, cfg, seed=1, eval_fn=eval_fn)
        self.assertEqual(calls, [(3, False)])
        self.assertEqual(hist["evals"], [{"step": 3, "n": 1}])
        enc = [(len(S.encode(r["input"])), len(S.encode(r["target"]))) for r in self.recs]
        self.assertEqual((hist["target_tokens"], hist["tokens"]),
                         (3 * sum(b + 1 for _, b in enc), 3 * sum(a + b + 2 for a, b in enc)))
        self.assertEqual(hist["params"], S.param_count(model))
        path = os.path.join(self.tmp, "m.pt")
        with torch.no_grad():
            model.sym_emb.mul_(1.5)          # load must restore the saved table, not regenerate it from the seed
        S.save(model, path)
        again = S.load(path)
        self.assertTrue(torch.equal(again.sym_emb, model.sym_emb))
        self.assertFalse(torch.equal(again.sym_emb, S.symbol_table(model.config)))
        self.assertEqual(again.config, model.config)
        x = S.make_batch(self.recs[:4])[0]
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), again(x)))
        ck = torch.load(path, weights_only=True)
        ck["vocab"] = "0" * 64
        torch.save(ck, path)
        with self.assertRaises(ValueError):
            S.load(path)


class TestCoreMatchesCharLM(unittest.TestCase):
    """lm_core with char_lm's codec and an equal-config DecoderLM reproduces char_lm bit-exactly."""

    def test_model_train_and_generate_equivalence(self):
        rng = random.Random(3)
        recs = [{"input": s, "target": s[::-1]} for s in
                ("".join(rng.choice("abcdef0123") for _ in range(rng.randint(3, 7))) for _ in range(200))]
        cfg = char_lm.TrainConfig(model=char_lm.CharLMConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64,
                                                             dropout=0.1, max_len=32),
                                  steps=12, batch_size=16, lr=3e-3, warmup=3, bucket=2, log_every=3)
        torch.manual_seed(4)
        ref = char_lm.CharLM(cfg.model)
        torch.manual_seed(4)
        core = lm_core.DecoderLM(cfg.model)
        self.assertEqual(ref.state_dict().keys(), core.state_dict().keys())
        self.assertTrue(all(torch.equal(v, core.state_dict()[k]) for k, v in ref.state_dict().items()))

        a, ha = char_lm.train(recs, cfg, seed=2)
        b, hb = lm_core.train(lm_core.DecoderLM, char_lm, recs, cfg, seed=2)
        sd_b = b.state_dict()
        self.assertTrue(all(torch.equal(v, sd_b[k]) for k, v in a.state_dict().items()))
        for key in ("tokens", "target_tokens", "padded_tokens", "steps", "epochs", "params"):
            self.assertEqual(ha[key], hb[key], key)
        for ra, rb in zip(ha["log"], hb["log"]):
            self.assertEqual((ra["loss"], ra["lr"], ra["grad_norm"]), (rb["loss"], rb["lr"], rb["grad_norm"]))
        inputs = [r["input"] for r in recs[:30]]
        self.assertEqual(char_lm.generate(a, inputs, 9, batch_size=8, return_done=True),
                         lm_core.generate(b, char_lm, inputs, 9, batch_size=8, return_done=True))
        self.assertEqual(char_lm.generate_naive(a, inputs[:5], 9), lm_core.generate_naive(b, char_lm, inputs[:5], 9))
        self.assertEqual(char_lm.make_batch(recs[:5])[1].tolist(), lm_core.make_batch(char_lm, recs[:5])[1].tolist())


# ----------------------------------------------------------------------------------------------- learnability

class TestLearnability(unittest.TestCase):
    """'X,Y:Z' -> true iff Z in {X, Y}; trained on NONCE['train'], tested on unseen NONCE['dev'] symbols.

    Symbol rows are frozen random vectors, so accuracy on unseen dev symbols far above chance means identity-based
    binding. Batch 256 / lr 3e-3: binding needs ~50K samples to leave the plateau (see the sym_lm docstring).
    """

    def run_toy(self, steps, **model_kw):
        self.assertFalse(set(TRAIN_NONCE) & set(DEV_NONCE))
        train = _member_records(random.Random(1), TRAIN_NONCE, 40000)
        dev = _member_records(random.Random(2), DEV_NONCE, 400)
        seen = _member_records(random.Random(3), TRAIN_NONCE, 400)   # fresh pairs of training symbols
        mk = dict(d_model=64, n_layers=2, n_heads=4, d_ff=256, max_len=16)
        mk.update(model_kw)
        cfg = S.TrainConfig(model=S.SymLMConfig(**mk), steps=steps, batch_size=256, lr=3e-3, warmup=50,
                            weight_decay=0.01, log_every=100)
        t0 = time.perf_counter()
        model, hist = S.train(train, cfg, seed=0)
        outs, done = S.generate(model, [r["input"] for r in dev], 2, batch_size=400, return_done=True)
        elapsed = time.perf_counter() - t0
        acc = sum(o == r["target"] for o, r in zip(outs, dev)) / len(dev)
        seen_acc = sum(o == r["target"] for o, r in
                       zip(S.generate(model, [r["input"] for r in seen], 2, batch_size=400), seen)) / len(seen)
        print(f"\n  membership {model_kw or 'fixed table'}: unseen={acc:.3f} seen={seen_acc:.3f} "
              f"time={elapsed:.1f}s threads={torch.get_num_threads()} "
              f"loss={[round(r['loss'], 3) for r in hist['log']]}")
        self.assertTrue(all(done))
        self.assertTrue(torch.equal(model.sym_emb, S.symbol_table(cfg.model)))
        return acc, elapsed

    def test_membership_binding_resampled_symbols(self):
        """Binding-only learner (fresh symbol table each step): > 90% on unseen symbols within 60 s
        (0.983 unseen == 0.983 seen in 45 s at 2 threads when written)."""
        acc, elapsed = self.run_toy(900, d_ff=128, symbol_resample=True)
        self.assertGreater(acc, 0.90)
        self.assertLess(elapsed, 60.0)

    def test_membership_binding_fixed_table(self):
        """Default fixed frozen table: binds unseen symbols well above chance (it also memorises codes for the
        training vectors, so unseen accuracy peaks around 0.86-0.94 and erodes with longer training; 0.877 unseen
        vs 0.905 seen in 24 s at 2 threads when written)."""
        acc, elapsed = self.run_toy(400)
        self.assertGreater(acc, 0.75)
        self.assertLess(elapsed, 40.0)


if __name__ == "__main__":
    unittest.main()
