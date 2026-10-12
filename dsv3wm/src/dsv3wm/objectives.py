"""Causal sequential auxiliary modules and separate world-state supervision.

These lightweight positionwise bottlenecks replace V3's full transformer MTP
blocks. Depth k consumes h_i^(k-1) and token i+k, then predicts token i+k+1.
No auxiliary module attends across positions, and no inference uses them.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class AuxiliaryStep(nn.Module):
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.norm_h = nn.RMSNorm(hidden_size)
        self.norm_e = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, hidden, embedding):
        return hidden + self.up(F.silu(self.down(torch.cat((self.norm_h(hidden), self.norm_e(embedding)), -1))))


class SequentialMTP(nn.Module):
    def __init__(self, hidden_size, rank=16, tokens=2, shared=False):
        super().__init__()
        if tokens not in (1, 2, 4):
            raise ValueError("total predicted tokens must be 1/2/4")
        self.depth = tokens - 1
        self.shared = shared
        self.modules_by_depth = nn.ModuleList([
            AuxiliaryStep(hidden_size, rank) for _ in range(1 if shared and self.depth else self.depth)
        ])

    def representations(self, hidden, ids, embedding):
        current = hidden
        for k in range(1, self.depth + 1):
            module = self.modules_by_depth[0 if self.shared else k - 1]
            current = module(current[:, :-1], embedding(ids[:, k:]))
            yield k, current


def masked_chunked_ce(head, hidden, targets, chunk=8):
    mask = targets != -100
    hs, ts = hidden[mask], targets[mask]
    if not ts.numel():
        return hidden.sum() * 0, 0
    loss = hidden.new_zeros((), dtype=torch.float32)
    for offset in range(0, ts.numel(), chunk):
        def part(h, t):
            return F.cross_entropy(head(h).float(), t, reduction="sum")
        loss = loss + checkpoint(part, hs[offset:offset + chunk], ts[offset:offset + chunk], use_reentrant=False)
    return loss / ts.numel(), int(ts.numel())


def token_objective(decoder, head, embedding, ids, attention, labels, mtp=None, weight=0.3, chunk=8, shuffled=False):
    hidden = decoder(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    base, n = masked_chunked_ce(head, hidden[:, :-1], labels[:, 1:], chunk)
    aux_losses, aux_tokens = [], 0
    if mtp is not None:
        for k, rep in mtp.representations(hidden, ids, embedding):
            targets = labels[:, k + 1:].clone()
            valid = torch.ones_like(targets, dtype=torch.bool)
            for j in range(1, k + 2):
                valid &= labels[:, j:labels.shape[1] - (k + 1) + j] != -100
            targets[~valid] = -100
            if shuffled:
                # Deterministic random control from the experiment's seeded torch RNG.
                mask = targets != -100
                values = targets[mask]
                targets[mask] = values[torch.randperm(values.numel(), device=values.device)]
            loss, count = masked_chunked_ce(head, rep[:, :-1], targets, chunk)
            aux_losses.append(loss)
            aux_tokens += count
    total = base + weight * torch.stack(aux_losses).mean() if aux_losses else base
    return total, {"base_loss": base.detach(), "aux_losses": [x.detach() for x in aux_losses],
                   "base_tokens": n, "aux_tokens": aux_tokens}, hidden


def multi_horizon_loss(losses, lambda2=0.5, lambda4=0.5, legality_loss=None, beta=0.1):
    if 1 not in losses or set(losses) - {1, 2, 4}:
        raise ValueError("horizon 1 required; only 1/2/4 supported")
    result = losses[1]
    for horizon, weight in ((2, lambda2), (4, lambda4)):
        if horizon in losses:
            result = result + weight * losses[horizon]
    if legality_loss is not None:
        result = result + beta * legality_loss
    return result
