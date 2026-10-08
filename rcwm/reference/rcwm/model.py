"""A small causal action-token Transformer with a learned latent bottleneck.

The supervised output is measured features. It is deliberately NOT a claim of
self-supervised outcome discovery, calibrated uncertainty or native Rineya support.
"""
from __future__ import annotations
import torch
from torch import nn
from .masks import causal_mask


class FeatureTrajectoryTransformer(nn.Module):
    def __init__(self, state_dim=4, actions=5, features=5, width=48,
                 heads=4, layers=2, latent=24, max_horizon=16):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or max_horizon < 1:
            raise ValueError("valid width/head division and positive horizon required")
        self.max_horizon = max_horizon
        self.state = nn.Linear(state_dim,width)
        self.action = nn.Embedding(actions,width)
        self.position = nn.Embedding(max_horizon+1,width)
        self.type_embedding = nn.Embedding(2,width)
        layer = nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*2,
                                           dropout=0.0,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(width),
                                              enable_nested_tensor=False)
        self.latent = nn.Sequential(nn.Linear(width,latent),nn.Tanh())
        self.decode = nn.Linear(latent,features)

    def forward(self, observed_initial: torch.Tensor, candidate_actions: torch.Tensor):
        if observed_initial.ndim != 2 or candidate_actions.ndim != 2:
            raise ValueError("expected state [batch,state] and actions [batch,horizon]")
        if len(observed_initial) != len(candidate_actions):
            raise ValueError("batch sizes differ")
        h = candidate_actions.shape[1]
        if h < 1 or h > self.max_horizon:
            raise ValueError("horizon outside configured range")
        if candidate_actions.dtype != torch.long:
            raise ValueError("actions must have torch.long dtype")
        if not torch.isfinite(observed_initial).all():
            raise ValueError("state must be finite")
        if (candidate_actions < 0).any() or (candidate_actions >= self.action.num_embeddings).any():
            raise ValueError("action ID out of range")
        x = torch.cat([self.state(observed_initial).unsqueeze(1),
                       self.action(candidate_actions)],dim=1)
        positions = torch.arange(h+1,device=x.device)
        types = torch.cat([torch.zeros(1,device=x.device,dtype=torch.long),
                           torch.ones(h,device=x.device,dtype=torch.long)])
        x = x + self.position(positions)[None] + self.type_embedding(types)[None]
        encoded = self.encoder(x,mask=causal_mask(h+1,device=x.device))[:,1:]
        z = self.latent(encoded)
        return {"features":self.decode(z),"latent":z}
