"""Original deterministic toy environment. This is NOT a generalization benchmark.

Actions: wait, take_key, open_door, collect, shortcut.
Feature order: key_gained, door_opened, object_gained, damage, action_cost.
Fixed observed windows do not bootstrap; success is an event paid at most once.
"""
from __future__ import annotations
import numpy as np
ACTIONS = ("wait", "take_key", "open_door", "collect", "shortcut")
FEATURES = ("key_gained", "door_opened", "object_gained", "damage", "action_cost")


def transition(state, action: int):
    s = np.asarray(state, dtype=float).copy()
    if s.shape != (4,) or not np.isin(s,[0,1]).all():
        raise ValueError("state must be binary [key,door,object,damage]")
    if not isinstance(action,(int,np.integer)) or not 0 <= action < len(ACTIONS):
        raise ValueError("invalid action")
    before = s.copy()
    if action == 1: s[0] = 1
    elif action == 2 and s[0]: s[1] = 1
    elif action == 3 and s[1]: s[2] = 1
    elif action == 4 and not s[2]: s[2], s[3] = 1, 1
    features = np.r_[s-before, 0.1 if action == 0 else 1.0].astype(np.float32)
    return s.astype(np.float32), features


def rollout(initial, actions):
    if len(actions) == 0:
        raise ValueError("nonempty action plan required")
    state = np.asarray(initial,dtype=np.float32).copy()
    outputs, states = [], []
    for a in actions:
        state, f = transition(state,int(a))
        outputs.append(f); states.append(state)
    return np.stack(outputs), np.stack(states)


def make_dataset(n: int, horizon: int, seed: int = 0):
    if n < 1 or horizon < 1:
        raise ValueError("positive sizes required")
    rng = np.random.default_rng(seed)
    states = np.zeros((n,4),dtype=np.float32)
    actions = rng.integers(0,len(ACTIONS),(n,horizon),dtype=np.int64)
    for i in range(n):
        for a in rng.integers(0,len(ACTIONS),int(rng.integers(0,4))):
            states[i],_ = transition(states[i],int(a))
    targets = np.stack([rollout(s,a)[0] for s,a in zip(states,actions)])
    return states, actions, targets
