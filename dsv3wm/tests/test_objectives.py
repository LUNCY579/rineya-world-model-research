import unittest
import torch
from torch import nn
from types import SimpleNamespace
from dsv3wm.objectives import SequentialMTP, token_objective, masked_chunked_ce, multi_horizon_loss
from dsv3wm.config import Config


class CausalDecoder(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids).cumsum(1))


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.emb = nn.Embedding(23, 8)
        self.head = nn.Linear(8, 23)
        self.dec = CausalDecoder(self.emb)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.labels = self.ids.clone()
        self.labels[:, :2] = -100
        self.att = torch.ones_like(self.ids)

    def test_chunked_value_and_gradient_parity(self):
        h = torch.randn(1, 6, 8, requires_grad=True)
        y = self.labels[:, 1:]
        loss, n = masked_chunked_ce(self.head, h, y, 2)
        ref = nn.functional.cross_entropy(self.head(h[y != -100]), y[y != -100])
        self.assertEqual(n, 5)
        torch.testing.assert_close(loss, ref)
        torch.testing.assert_close(torch.autograd.grad(loss, h, retain_graph=True)[0], torch.autograd.grad(ref, h)[0])

    def test_disabled_auxiliary_is_identical(self):
        loss, _, h = token_objective(self.dec, self.head, self.emb, self.ids, self.att, self.labels)
        ref = nn.functional.cross_entropy(self.head(h[:, :-1][self.labels[:, 1:] != -100]), self.labels[:, 1:][self.labels[:, 1:] != -100])
        torch.testing.assert_close(loss, ref)

    def test_four_token_auxiliary_has_no_target_leakage(self):
        mtp = SequentialMTP(8, 3, 4)
        for k, rep in mtp.representations(self.dec(self.ids, self.att).last_hidden_state, self.ids, self.emb):
            changed = self.ids.clone()
            changed[:, k + 1:] = 22
            changed_rep = dict(mtp.representations(self.dec(changed, self.att).last_hidden_state, changed, self.emb))[k]
            torch.testing.assert_close(rep[:, 0], changed_rep[:, 0])
        loss, metrics, _ = token_objective(self.dec, self.head, self.emb, self.ids, self.att, self.labels, mtp)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['aux_tokens'], 4 + 3 + 2)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in mtp.parameters()))

    def test_padding_never_supervised(self):
        labels = self.labels.clone()
        labels[:, -2:] = -100
        _, metrics, _ = token_objective(self.dec, self.head, self.emb, self.ids, self.att, labels, SequentialMTP(8, 3, 4))
        self.assertEqual(metrics['base_tokens'], 3)
        self.assertEqual(metrics['aux_tokens'], 3)

    def test_horizon_values_and_gradients(self):
        values = {h: torch.tensor(float(h), requires_grad=True) for h in (1, 2, 4)}
        legal = torch.tensor(2.0, requires_grad=True)
        total = multi_horizon_loss(values, .25, .75, legal, .1)
        self.assertAlmostEqual(total.item(), 4.7, places=5)
        grads = torch.autograd.grad(total, [*values.values(), legal])
        for got, expected in zip(grads, [1, .25, .75, .1]):
            self.assertAlmostEqual(got.item(), expected, places=5)
        with self.assertRaises(ValueError):
            multi_horizon_loss({2: legal})

    def test_config_fails_closed(self):
        for args in ({'arm':'bad'}, {'steps':0}, {'mtp_tokens':3}, {'device':'bf16'}, {'lr':float('nan')}, {'shuffled_aux':'false'}):
            with self.assertRaises(ValueError):
                Config(**args)


if __name__ == '__main__':
    unittest.main()
