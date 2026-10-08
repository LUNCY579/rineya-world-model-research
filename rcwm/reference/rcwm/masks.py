"""Masks for torch.nn.Transformer: True means BLOCKED, not allowed.

DAG utilities validate availability; they are not a production scheduler.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


def causal_mask(length: int, *, device=None) -> torch.Tensor:
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def prefix_plan_mask(prefix: int, actions: int, *, device=None) -> torch.Tensor:
    """History sees history only; plan slots see history and all proposed actions.

    Valid only for fully precommitted plan metadata, NOT actual future outcomes.
    """
    if prefix < 1 or actions < 1:
        raise ValueError("positive prefix and action counts required")
    n = prefix + actions
    blocked = torch.zeros(n, n, dtype=torch.bool, device=device)
    blocked[:prefix, prefix:] = True
    return blocked


def validate_dag(parents: list[list[int]]) -> None:
    """Require a topologically sorted DAG with unique, earlier parent indices."""
    for i, dependencies in enumerate(parents):
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("duplicate dependency")
        if any(not isinstance(j, int) or isinstance(j, bool) or j < 0 or j >= i
               for j in dependencies):
            raise ValueError("parents must precede child in topological order")


@dataclass(frozen=True)
class ObservedEvent:
    available_at: float
    observed: bool = True


def available_observation_mask(query_times, events: list[ObservedEvent]) -> torch.Tensor:
    """Query-key visibility by observation availability; True blocks a key."""
    import math
    if not query_times or not events:
        raise ValueError("nonempty times and events required")
    if any(not math.isfinite(float(t)) for t in query_times):
        raise ValueError("finite query times required")
    if any(not math.isfinite(e.available_at) for e in events):
        raise ValueError("finite availability times required")
    return torch.tensor([[not e.observed or e.available_at > t for e in events]
                         for t in query_times], dtype=torch.bool)
