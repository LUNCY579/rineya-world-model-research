"""Tests for rcx.models.char_lm: tokenizer, cached-vs-naive decoding, loss masking, trainer, learnability.

Run: python -B -m unittest tests.test_char_lm -v   (CPU only, 4 threads, well under a minute)
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

torch.set_num_threads(4)

from rcx.models import char_lm as lm

ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"


def _random_model(seed, dtype=torch.float32, max_len=40):
    torch.manual_seed(seed)
    model = lm.CharLM(lm.CharLMConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.1, max_len=max_len))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)
        model.emb.weight[lm.EOS] *= 3.0
    return model.to(dtype).train()


def _random_texts(rng, n, lo, hi, alphabet=lm.CHARS):
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))) for _ in range(n)]


def _reverse_records(rng, n, exclude=()):
    out, exclude = [], set(exclude)
    while len(out) < n:
        s = "".join(rng.choice(ALNUM) for _ in range(rng.randint(3, 7)))
        if s not in exclude:
            out.append({"input": s, "target": s[::-1]})
    return out


class TestTokenizer(unittest.TestCase):
    def test_vocab_and_roundtrip(self):
        self.assertEqual(lm.VOCAB_SIZE, 100)
        self.assertEqual((lm.PAD, lm.BOS, lm.SEP, lm.EOS), (0, 1, 2, 3))
        text = "RULE: x\nSTATE: {\"A\":-3,\"B\":7}~"
        ids = lm.encode(text)
        self.assertTrue(all(i >= lm.N_SPECIAL for i in ids))
        self.assertEqual(lm.decode(ids), text)
        self.assertEqual(lm.encode("\n "), [4, 5])

    def test_unknown_chars_raise(self):
        for bad in ("café", "a\tb", "\r", "\x00"):
            with self.assertRaises(ValueError):
                lm.encode(bad)
        with self.assertRaises(ValueError):
            lm.decode([lm.VOCAB_SIZE])
        self.assertEqual(lm.decode([lm.SEP, 4 + 1 + ord("A") - 32]), lm.SPECIAL_CHAR + "A")


class TestCachedDecoding(unittest.TestCase):
    def test_cache_matches_naive_greedy(self):
        """Batched left-padded KV-cache decoding == per-sequence full-recompute decoding (fp32 and fp64)."""
        rng = random.Random(0)
        n_eos = n_trunc = 0
        for dtype in (torch.float32, torch.float64):
            for seed in range(3):
                model = _random_model(seed, dtype)
                inputs = _random_texts(rng, 23, 0, 34)
                fast = lm.generate(model, inputs, 12, batch_size=7, return_done=True)
                slow = lm.generate_naive(model, inputs, 12, return_done=True)
                self.assertEqual(fast, slow, f"dtype={dtype} seed={seed}")
                self.assertTrue(model.training)
                n_eos += sum(fast[1])
                n_trunc += sum(len(t) < 12 and not d for t, d in zip(*fast))
        self.assertGreater(n_eos, 0, "EOS stopping path not exercised")
        self.assertGreater(n_trunc, 0, "max_len truncation path not exercised")

    def test_cached_logits_match_full_forward(self):
        model = _random_model(5, torch.float64).eval()
        prompts = [lm._prompt("hello world"), lm._prompt("ab")]
        P = max(map(len, prompts))
        ids = torch.full((2, P), lm.PAD)
        valid = torch.zeros(2, P + 1, dtype=torch.bool)
        for i, p in enumerate(prompts):
            ids[i, P - len(p):] = torch.tensor(p)
            valid[i, P - len(p):P] = True
        pos = (valid[:, :P].long().cumsum(1) - 1).clamp(min=0)
        mask = (torch.ones(P, P, dtype=torch.bool).tril() & valid[:, None, :P]) | torch.eye(P, dtype=torch.bool)
        cache = model.new_cache(2, P + 1)
        with torch.no_grad():
            pre = model(ids, pos, mask.unsqueeze(1), cache, 0)[:, -1]
            nxt = torch.tensor([[40], [41]])
            valid[:, P] = True
            step = model(nxt, torch.tensor([[len(p)] for p in prompts]), valid[:, None, None, :], cache, P)[:, -1]
            for i, p in enumerate(prompts):
                full = model(torch.tensor([p + [int(nxt[i])]]))[0]
                torch.testing.assert_close(pre[i], full[-2], rtol=0, atol=1e-10)
                torch.testing.assert_close(step[i], full[-1], rtol=0, atol=1e-10)

    def test_edge_cases(self):
        model = _random_model(1)
        self.assertEqual(lm.generate(model, ["abc", ""], 0), ["", ""])
        self.assertEqual(lm.generate(model, [], 5), [])
        with self.assertRaises(ValueError):
            lm.generate(model, ["x" * 39], 5)
        self.assertEqual(lm.generate(model, ["x" * 38], 5, return_done=True)[0], lm.generate_naive(model, ["x" * 38], 5))

    def test_max_len_cutoff_is_exact(self):
        """Without EOS, an n-position prompt yields exactly min(max_new, max_len - n + 1) tokens in both decoders.

        Both decoders share _limit, so cache-vs-naive equality alone cannot catch an off-by-one in the cut-off.
        """
        model = _random_model(4, torch.float64, max_len=20)
        with torch.no_grad():
            model.emb.weight[lm.EOS] = 0.0
        inputs = ["a" * k for k in range(19)]
        for max_new in (5, 100):
            expected = [min(max_new, 20 - (k + 2) + 1) for k in range(19)]
            for texts, done in (lm.generate(model, inputs, max_new, batch_size=6, return_done=True),
                                lm.generate_naive(model, inputs, max_new, return_done=True)):
                self.assertFalse(any(done))
                self.assertEqual([len(t) for t in texts], expected)


class TestLossMasking(unittest.TestCase):
    def test_batch_layout(self):
        recs = [{"input": "abc12", "target": "21cba"}, {"input": "x", "target": "\nyz"}, {"input": "", "target": ""}]
        x, y, tok, sup = lm.make_batch(recs)
        self.assertEqual(x.shape, (3, 12))
        self.assertEqual(sup, sum(len(r["target"]) + 1 for r in recs))
        self.assertEqual(tok, sum(len(r["input"]) + len(r["target"]) + 2 for r in recs))
        for i, r in enumerate(recs):
            n_in, n = len(r["input"]), len(r["input"]) + len(r["target"]) + 2
            self.assertEqual(x[i, :n].tolist(), [lm.BOS] + lm.encode(r["input"]) + [lm.SEP] + lm.encode(r["target"]))
            self.assertTrue((x[i, n:] == lm.PAD).all())
            sup_pos = (y[i] != lm.IGNORE).nonzero().flatten().tolist()
            self.assertEqual(sup_pos, list(range(n_in + 1, n)))
            self.assertEqual(y[i, sup_pos].tolist(), lm.encode(r["target"]) + [lm.EOS])

    def test_input_positions_get_no_loss(self):
        recs = [{"input": "abc12", "target": "21cba"}, {"input": "hello", "target": "o"}]
        x, y, _, _ = lm.make_batch(recs)
        model = _random_model(2).eval()
        logits = model(x).detach().requires_grad_(True)
        loss = lm.lm_loss(logits, y)
        loss.backward()
        unsup = y == lm.IGNORE
        self.assertEqual(logits.grad[unsup].abs().max().item(), 0.0)
        self.assertTrue((logits.grad[~unsup].abs().sum(-1) > 0).all())
        with torch.no_grad():
            noisy = logits + 50 * torch.randn_like(logits) * unsup.unsqueeze(-1)
            self.assertEqual(lm.lm_loss(noisy, y).item(), loss.item())
            manual = torch.nn.functional.cross_entropy(logits[~unsup], y[~unsup])
            torch.testing.assert_close(loss, manual)


class _Guard:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def check(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise KeyboardInterrupt("simulated crash")


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="char_lm_test_")
        self.recs = _reverse_records(random.Random(3), 300)
        self.cfg = lm.TrainConfig(model=lm.CharLMConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.1,
                                                        max_len=32),
                                  steps=30, batch_size=16, lr=3e-3, warmup=5, bucket=4, log_every=5, guard_every=5)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deterministic_resume_guard_and_log(self):
        a, ha = lm.train(self.recs, self.cfg, seed=7)
        a2, _ = lm.train(self.recs, self.cfg, seed=7)
        b0, _ = lm.train(self.recs, self.cfg, seed=8)
        sd_a, sd_a2 = a.state_dict(), a2.state_dict()
        self.assertTrue(all(torch.equal(sd_a[k], sd_a2[k]) for k in sd_a))
        self.assertFalse(torch.equal(sd_a["emb.weight"], b0.state_dict()["emb.weight"]))

        ck = os.path.join(self.tmp, "ck.pt")
        log = os.path.join(self.tmp, "log.jsonl")
        cfg = lm.TrainConfig(**{**self.cfg.__dict__, "checkpoint_path": ck, "checkpoint_every": 10})
        crash = _Guard(fail_at=4)
        with self.assertRaises(KeyboardInterrupt):
            lm.train(self.recs, cfg, seed=7, log_path=log, guard=crash)
        self.assertTrue(os.path.exists(ck))
        guard = _Guard()
        c, hc = lm.train(self.recs, cfg, seed=7, log_path=log, guard=guard)
        self.assertEqual(hc["resumed_from"], 10)
        self.assertEqual(guard.calls, 4)
        sd_c = c.state_dict()
        self.assertTrue(all(torch.equal(sd_a[k], sd_c[k]) for k in sd_a))
        self.assertEqual([r["loss"] for r in ha["log"]], [r["loss"] for r in hc["log"]])
        for key in ("tokens", "target_tokens", "padded_tokens", "steps"):
            self.assertEqual(ha[key], hc[key], key)
        def read_log():
            with open(log, encoding="utf-8") as f:
                return [json.loads(line) for line in f]

        self.assertEqual(read_log(), json.loads(json.dumps(hc["log"])))
        with self.assertRaises(ValueError):
            lm.train(self.recs[:-1], cfg, seed=7)
        os.remove(ck)
        lm.train(self.recs, cfg, seed=7, log_path=log)
        self.assertEqual([r["step"] for r in read_log()], [5, 10, 15, 20, 25, 30])

    def test_token_counts_and_save_load(self):
        cfg = lm.TrainConfig(**{**self.cfg.__dict__, "steps": 3, "bucket": 0, "batch_size": 300, "log_every": 1})
        calls = []
        eval_fn = lambda m, step: calls.append((step, m.training)) or {"n": len(calls)}
        model, hist = lm.train(self.recs, cfg, seed=1, eval_fn=eval_fn)
        self.assertEqual(calls, [(3, False)])
        self.assertEqual(hist["evals"], [{"step": 3, "n": 1}])
        per_epoch_sup = sum(len(r["target"]) + 1 for r in self.recs)
        per_epoch_tok = sum(len(r["input"]) + len(r["target"]) + 2 for r in self.recs)
        self.assertEqual((hist["target_tokens"], hist["tokens"]), (3 * per_epoch_sup, 3 * per_epoch_tok))
        self.assertEqual(hist["params"], lm.param_count(model))
        path = os.path.join(self.tmp, "m.pt")
        lm.save(model, path)
        again = lm.load(path)
        x = lm.make_batch(self.recs[:4])[0]
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), again(x)))


class TestLearnability(unittest.TestCase):
    def test_reverse_task_learned(self):
        """Reverse 3-7 char strings: >95% exact match on held-out strings within 60 s of CPU."""
        test = _reverse_records(random.Random(99), 300)
        train = _reverse_records(random.Random(1), 20000, exclude={r["input"] for r in test})
        cfg = lm.TrainConfig(model=lm.CharLMConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, max_len=32),
                             steps=600, batch_size=64, lr=3e-3, warmup=50, weight_decay=0.01, log_every=100)
        t0 = time.perf_counter()
        model, hist = lm.train(train, cfg, seed=0)
        outs, done = lm.generate(model, [r["input"] for r in test], 9, batch_size=128, return_done=True)
        elapsed = time.perf_counter() - t0
        acc = sum(o == r["target"] for o, r in zip(outs, test)) / len(test)
        print(f"\n  reverse task: exact={acc:.3f} loss={hist['log'][-1]['loss']:.4f} time={elapsed:.1f}s "
              f"train_tok/s={hist['tokens'] / hist['compute_s']:.0f}")
        self.assertGreater(acc, 0.95)
        self.assertLess(elapsed, 60.0)
        self.assertTrue(all(done))
        sub = [r["input"] for r in test[:40]]
        self.assertEqual(lm.generate(model, sub, 9, batch_size=16), lm.generate_naive(model, sub, 9))


if __name__ == "__main__":
    unittest.main()
