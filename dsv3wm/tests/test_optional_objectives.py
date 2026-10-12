import unittest
from types import SimpleNamespace
import torch
from torch import nn
from dsv3wm.objectives import SequentialMTP,token_objective
from dsv3wm.config import Config


class Decoder(nn.Module):
    def __init__(self,embedding):super().__init__();self.embedding=embedding
    def forward(self,input_ids,attention_mask,use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(1))


class OptionalObjectiveTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(13)
        embedding=nn.Embedding(31,8);head=nn.Linear(8,31)
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9]])
        labels=ids.clone();labels[:,:2]=-100
        return Decoder(embedding),head,embedding,ids,torch.ones_like(ids),labels
    def test_shared_four_token_module_finite_gradients_and_capacity(self):
        module=SequentialMTP(8,3,4,shared=True)
        self.assertEqual(sum(p.numel() for p in module.parameters()),sum(p.numel() for p in SequentialMTP(8,3,2).parameters()))
        loss,metrics,_=token_objective(*self.fixture(),mtp=module)
        loss.backward()
        self.assertEqual(metrics['aux_tokens'],6+5+4)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
    def test_shuffled_target_control_changes_loss_only_not_causal_hidden(self):
        args=self.fixture();module=SequentialMTP(8,3,2)
        real,stats,h=token_objective(*args,mtp=module)
        torch.manual_seed(22)
        shuffled,negative,other=token_objective(*args,mtp=module,shuffled=True)
        torch.testing.assert_close(h,other)
        torch.testing.assert_close(stats['base_loss'],negative['base_loss'])
        self.assertEqual(stats['aux_tokens'],negative['aux_tokens'])
        self.assertNotAlmostEqual(real.item(),shuffled.item(),places=4)
        shuffled.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
    def test_boolean_token_horizon_is_not_integer_config(self):
        with self.assertRaises(ValueError):Config(mtp_tokens=True)


if __name__=='__main__':unittest.main()
