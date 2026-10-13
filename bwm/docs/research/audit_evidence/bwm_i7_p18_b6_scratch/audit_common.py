"""Shared helpers for the read-only Blocksworld contamination audit (Method A: reuse repo helpers).

Nothing here writes under ROOT. Run with `python -B` so no __pycache__ is created inside the repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B  # noqa: E402  (repo domain helpers: signature, parse_problem, state_from_json)
import blocks_val_large as VL  # noqa: E402  (repo invariant helpers: towers_of, goal_towers, invariant)


# ----------------------------------------------------------------------------- validity + exact part-keys

def valid_state(s: frozenset, objs) -> bool:
    """Complete, physically valid Blocksworld state over exactly `objs`."""
    objs = set(objs)
    pos = {}
    above = {}
    hold = []
    hand_empty = False
    clear = set()
    for f in s:
        if f == ("handempty",):
            hand_empty = True
        elif f[0] == "holding" and len(f) == 2:
            hold.append(f[1])
            if f[1] in pos:
                return False
            pos[f[1]] = "held"
        elif f[0] == "ontable" and len(f) == 2:
            if f[1] in pos:
                return False
            pos[f[1]] = "table"
        elif f[0] == "on" and len(f) == 3:
            x, y = f[1], f[2]
            if x in pos or y in above or x == y:
                return False
            pos[x] = y
            above[y] = x
        elif f[0] == "clear" and len(f) == 2:
            clear.add(f[1])
        else:
            return False
    if set(pos) != objs or (clear | set(above)) - objs:
        return False
    if len(hold) > 1 or hand_empty == bool(hold):
        return False
    if any(pos.get(y) == "held" for y in above):  # nothing can sit on a held block
        return False
    # grounded: every chain reaches the table or the hand (no cycles)
    for x in objs:
        seen = set()
        while pos[x] not in ("table", "held"):
            if x in seen:
                return False
            seen.add(x)
            x = pos[x]
    exp_clear = {x for x in objs if x not in above and pos[x] != "held"}
    return clear == exp_clear


def state_key(s: frozenset, objs):
    """Exact renaming-invariant key of a VALID state: (n, sorted table-tower heights, held count).
    Uses the repo helper VL.towers_of (which counts a held block as a height-1 tower) plus the held flag."""
    if not valid_state(s, objs):
        return None
    held = sum(1 for f in s if f[0] == "holding")
    return (len(objs), tuple(VL.towers_of(s, objs)), held)


def goal_key(goal: frozenset, objs):
    """Exact renaming-invariant key of an 'on'-only goal whose on-relation is a linear forest: (n, sorted chain
    lengths) via the repo helper VL.goal_towers. None if the goal has other facts or is not a linear forest."""
    if any(f[0] != "on" or len(f) != 3 for f in goal):
        return None
    lo, hi = set(), set()
    for f in goal:
        if f[1] in lo or f[2] in hi or f[1] == f[2] or f[1] not in objs or f[2] not in objs:
            return None
        lo.add(f[1])
        hi.add(f[2])
    gt = VL.goal_towers(goal, objs)
    if sum(gt) != len(objs):  # cycle
        return None
    return (len(objs), tuple(gt))


# ----------------------------------------------------------------------------- explicit isomorphism

def _struct(objs, init, goal):
    U = {o: set() for o in objs}
    E = {k: {} for k in ("i_on", "i_und", "g_on", "g_und")}
    nul = set()
    for pre, fs in (("i", init), ("g", goal)):
        for f in fs:
            if len(f) == 1:
                nul.add((pre,) + tuple(f))
            elif len(f) == 2:
                if f[1] not in U:
                    return None
                U[f[1]].add(pre + "_" + f[0])
            elif len(f) == 3 and f[0] == "on":
                x, y = f[1], f[2]
                if x not in U or y not in U or x in E[pre + "_on"] or y in E[pre + "_und"]:
                    return None  # not functional: outside the scope of this routine
                E[pre + "_on"][x] = y
                E[pre + "_und"][y] = x
            else:
                return None
    return U, E, nul


def rename(fs, m):
    return frozenset(tuple([f[0]] + [m[x] for x in f[1:]]) for f in fs)


def find_iso(A, Bp):
    """Explicit bijection m: objs_A -> objs_B with m(init_A) == init_B and m(goal_A) == goal_B, or None.
    Complete for problems whose init/goal 'on' relations are functional (every valid state / linear-forest goal):
    a component of the labelled graph is fixed once one vertex's image is fixed, every root image is tried, and
    isomorphic components are interchangeable (exchange argument). Sound: any returned map is re-verified on the
    full fact sets."""
    oa, ia, ga = A
    ob, ib, gb = Bp
    if len(oa) != len(ob) or len(ia) != len(ib) or len(ga) != len(gb):
        return None
    sa, sb = _struct(oa, ia, ga), _struct(ob, ib, gb)
    if sa is None or sb is None:
        raise ValueError("non-functional structure")
    Ua, Ea, Na = sa
    Ub, Eb, Nb = sb
    if Na != Nb:
        return None
    m, used = {}, set()
    for r in oa:
        if r in m:
            continue
        ok = False
        for y in ob:
            if y in used or Ua[r] != Ub[y]:
                continue
            lm, lu, q, bad = {r: y}, {y}, [(r, y)], False
            while q and not bad:
                x, z = q.pop()
                if Ua[x] != Ub[z]:
                    bad = True
                    break
                for L in Ea:
                    xa, zb = Ea[L].get(x), Eb[L].get(z)
                    if (xa is None) != (zb is None):
                        bad = True
                        break
                    if xa is None:
                        continue
                    if xa in lm or xa in m:
                        if lm.get(xa, m.get(xa)) != zb:
                            bad = True
                            break
                    elif zb in lu or zb in used:
                        bad = True
                        break
                    else:
                        lm[xa] = zb
                        lu.add(zb)
                        q.append((xa, zb))
            if not bad:
                m.update(lm)
                used |= lu
                ok = True
                break
        if not ok:
            return None
    if rename(ia, m) != ib or rename(ga, m) != gb:
        raise AssertionError("find_iso returned a non-isomorphism")
    return m


# ----------------------------------------------------------------------------- row parsing

def user_json(prompt: str) -> dict:
    return json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])


def objs_of(state, goal):
    out = set(B.blocks_of(state))
    for f in goal:
        out.update(f[1:])
    return sorted(out)
