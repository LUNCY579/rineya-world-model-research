"""S-roll (PREREG-1 sec 6, registered-but-unimplemented secondary; EXPLORATORY, pilot bank only).

Question: does Titan-Sparse's chained self-rollout (chain_one / chain_m1) beat a GENERIC one-step
learner, trained on one-step transitions and chained the same way (its own discrete prediction fed
back, 4 times)? If not, the LM contributes nothing beyond "having a one-step simulator".

Pipeline (deterministic given seeds; CPU only, 2 threads, never touches the GPU):
  1. One-step transitions (state_t, event_t) -> state_{t+1} along the GOLD trajectories of every
     candidate plan. TRAIN parents -> fitting (exact duplicates removed), DEV parents -> early
     stopping + hyper-parameter selection only, TEST parents -> never fitted on. Every transition is
     cross-checked against the validated FDS oracle (env.cross_check) and every chained gold state
     against data.candidate_table's k=1/2/4 gold. Titan's OWN 384 training rows (RRP-015 control
     microfit-train.jsonl, sha-pinned; Titan-Sparse is a sparsified delta of that control) are parsed
     and verified against env.step for the TITANROWS arm.
  2. Learner: MLP on decide.symbolic(fam, state, [event], kmax=1), standardised as heads.Standardizer.
     Outputs always decode to a VALID discrete state (P: 8 sigmoid bits; R: 97-way classes of the
     value or of the mod-97 residue, or a rounded regression mod 97; L: rounded integer delta
     regression, gated delta regression or classification over TRAIN deltas). Output head (and, for
     R at FULL, a one-hot-augmented input), width, depth and LR are chosen on DEV in two stages
     (see select()) under one of two criteria, each a separate, complete arm family:
       crit=exact  (the S-roll spec): DEV one-step exact-state accuracy, seed 17, resample 0.
       crit=regret (decision-relevant; the PREREG-1 S selection rule): DEV k=4 normalised regret of
                   the DEV rollout under the SELECTION goals (seed 1002, never the test goals 1001);
                   FULL: seed 17; N128: mean over resamples 0-2 (seed 17). For n > 1000 this
                   criterion also gets a longer schedule (<= 8000 steps, patience 20), because
                   regression heads were still improving at the 4000-step cap.
     Checkpointing is on DEV one-step exact accuracy under both criteria (MATCHED128_NOSTOP keeps the
     final weights instead, as a sensitivity arm).
  3. Budgets: FULL (all TRAIN), TITAN-MATCHED = MATCHED128 (128 per family sampled from TRAIN: equal
     row count, in-distribution; 5 resamples x 5 seeds), TITANROWS (Titan's own 128 rows per family,
     5 seeds), a learning curve (512, 2000; resample i paired with seed i), NOOPFREE128 (P only: pool
     without inapplicable actions; Titan's P rows were all applicable) and MATCHED128_NOSTOP.
  4. Rollout from s0, final state at k=1,2,4 -> data.phi_from_state -> regret with
     decisions.g_only itself (same TEST parents, goals seed 1001, regret_table, fresh rng 17 per
     method, parent-mean fill). Per-parent regret is the nanmean over models (prereg1 S rule).
  5. Titan chain_one / chain_m1 / e2e regrets through the same g_only call (one mode per call, i.e.
     fresh rng 17, as prereg1.Split.regret does), Titan with the registered last-parsed fill
     (prereg1._gen_phi), parse rates, oracle and random references, paired bootstrap of
     (Titan chain_one - S-roll) with decide.paired_bootstrap (all parents; without TEST parents that
     share a transition with TRAIN; without parents whose candidates all reach one true phi),
     exact-state accuracy, one-step exact accuracy per operation stratum (S-roll vs Titan).

  python scripts/s_roll.py                       # everything: both criteria, L, P, R, then the report
                                                 # (about 2 h of wall time on 2 threads)
  python scripts/s_roll.py --crit exact --family L --stage select   # one stage (each < 25 min)
  python scripts/s_roll.py --crit regret --family R --stage fit
  python scripts/s_roll.py --report-only         # merge per-family outputs -> s_roll.{json,log} and
                                                 #   docs/s_roll/S_ROLL_PILOT.md
  python scripts/s_roll.py --smoke               # tiny schedule; writes *_smoke outputs only
"""
from __future__ import annotations

import os

# --- hard rules: no GPU, 2 threads (DECISIONS I-21: CUDA_VISIBLE_DEVICES must be "" not "-1") ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["RCWM_THREADS"] = "2"          # rcwmt.heads reads this at import
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(2)

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcwmt import data, decide, decisions, env, prereg1  # noqa: E402

torch.set_num_threads(2)
assert not torch.cuda.is_available()
try:                                       # shared laptop: CPU work runs below normal priority
    import psutil  # noqa: E402
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:                          # priority is a courtesy, never a reason to fail
    pass
# decisions.g_only takes nanmean over parents whose candidates all failed to parse (Titan P) and over
# parents excluded by a zero regret denominator; both are intended, so silence the numpy warning.
warnings.filterwarnings("ignore", message="Mean of empty slice")

BANK = "bank_pilot"
FAMS = env.FAMILIES
SHORT = env.SHORT
SEEDS = (17, 18, 19, 20, 21)
KS = (1, 2, 4)
H = 4
MATCHED_N = 128
CURVE_NS = (512, 2000)
N_CURVE_MODELS = 3             # curve arm i = (resample i, seed SEEDS[i]); optional arm, kept cheap
N_RESAMPLES = 5
RESAMPLE_SEED0 = 9000          # resample r uses np.random.default_rng(RESAMPLE_SEED0 + r)
NOOPFREE_SEED0 = 9500
SEL_GOAL_SEED = prereg1.SEL_GOAL_SEED      # 1002: DEV selection goals (prereg1), never the test goals
N_SEL_RESAMPLES = 3            # crit=regret at N128: DEV regret averaged over resamples 0-2, seed 17
CRITS = ("exact", "regret")
CRIT_TAG = {"exact": "", "regret": "_regsel"}
HEADS = {"ledger_replay": ("cls_delta", "reg_delta", "gated_delta"),
         "prerequisite_plan": ("bits",),
         "register_machine": ("cls97", "cls_res", "reg_abs", "reg_delta", "gated_delta")}
# Titan-Sparse = sparsified (theta_ctrl - theta_supa) of the RRP-015 control, which was trained on
# exactly these 384 rows (128/family, Forge v2 TRAIN positives; DATA_CONTRACT.json in the same dir).
TITAN_ROWS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/rrp015-20260917/"
                  r"control-prepared/microfit-train.jsonl")
TITAN_ROWS_SHA = "0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1"
OUT = ROOT / "runs" / BANK / "analysis"
DOC = ROOT / "docs" / "s_roll"


# ============================================================================ logging

class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# ============================================================================ states <-> arrays

def st_arr(fam, s) -> np.ndarray:
    if fam == "ledger_replay":
        return np.array([s[a] for a in env.ACCOUNTS], np.int64)
    if fam == "register_machine":
        return np.array([s[r] for r in env.REGISTERS], np.int64)
    return np.array(env.bits(s), np.int64)


def arr_st(fam, a):
    if fam == "ledger_replay":
        return {k: int(v) for k, v in zip(env.ACCOUNTS, a)}
    if fam == "register_machine":
        return {k: int(v) for k, v in zip(env.REGISTERS, a)}
    return int(sum(int(b) << i for i, b in enumerate(a)))


def enc(fam, s, e, encoding="sym") -> np.ndarray:
    """'sym': decide.symbolic(fam, s, [e], kmax=1) exactly.  'sym1h' (R only): the same vector plus a
    97-way one-hot of each register value (an equivalent exact encoding that exposes values as tokens)."""
    x = decide.symbolic(fam, s, [e], kmax=1)
    if encoding == "sym1h":
        oh = np.zeros(len(env.REGISTERS) * env.MOD, np.float32)
        for i, r in enumerate(env.REGISTERS):
            oh[i * env.MOD + s[r]] = 1.0
        x = np.concatenate([x, oh])
    return x


# ============================================================================ operation strata

def strata(fam, s, e) -> list[str]:
    """Operation strata of one transition, computed on its INPUT state (fine + coarse labels)."""
    if fam == "ledger_replay":
        st = "posted" if e["status"] == "posted" else "void"
        out = [f"{e['kind']}|{st}"]
        if st == "void":
            out.append("void|any")
        return out
    if fam == "register_machine":
        k, r, o, v = e["kind"], s[e["register"]], s[e["other"]], e["value"]
        if k == "swap":
            return ["swap"]
        if k == "add":
            return ["add", "add|" + ("nowrap" if 0 <= r + v < env.MOD else "wrap")]
        if k == "mix":
            return ["mix", "mix|" + ("nowrap" if 0 <= r + 2 * o + v < env.MOD else "wrap")]
        return ["conditional", "conditional|" + ("gt" if o > e["threshold"] else "le")]
    return ["applicable" if env.applicable(s, e) else "inapplicable"]


def is_noop_row(fam, e, s) -> bool:
    """No-op rows: void ledger events, inapplicable actions. (Titan's own rows: 17/128 L rows are void,
    0/128 P rows are inapplicable, so only the P version of this pool mimics Titan's data.)"""
    if fam == "ledger_replay":
        return e["status"] != "posted"
    if fam == "prerequisite_plan":
        return not env.applicable(s, e)
    return False


# ============================================================================ data

def load_inputs(log):
    bank = data.load_bank(BANK)
    rows = data.candidate_table(bank)
    gold = {(r["cand_id"], r["k"]): r for r in rows}
    T = {f: {"train": [], "dev": [], "test": []} for f in FAMS}
    cands = {f: [] for f in FAMS}             # TEST candidates for rollouts
    dcands = {f: [] for f in FAMS}            # DEV candidates (crit=regret selection only)
    n_check = 0
    for p in bank["parents"]:
        fam = p["family"]
        for c in p["candidates"]:
            states = [p["s0"]]
            for t, e in enumerate(c["events"]):
                env.cross_check(fam, states[-1], e)          # FDS oracle, independent implementation
                n_check += 1
                s2 = env.step(fam, states[-1], e)
                T[fam][p["split"]].append({"parent_id": p["parent_id"], "cand_id": c["cand_id"], "t": t,
                                           "s": states[-1], "e": e, "s2": s2})
                states.append(s2)
            assert len(states) == H + 1
            for k in KS:
                r = gold[(c["cand_id"], k)]
                assert env.dumps(states[k]) == r["gold"], (c["cand_id"], k)
                ph = data.phi_from_state(fam, p["s0"], c["events"][:k], states[k])
                assert np.array_equal(ph, r["phi"]), (c["cand_id"], k)
            rec = {"cand_id": c["cand_id"], "parent_id": p["parent_id"], "s0": p["s0"],
                   "events": c["events"], "gold": states}
            if p["split"] == "test":
                cands[fam].append(rec)
            elif p["split"] == "dev":
                dcands[fam].append(rec)
    log(f"verified {n_check} gold one-step transitions against the FDS oracle; chained gold == bank "
        f"k=1/2/4 gold and phi for every candidate")
    return bank, rows, T, cands, dcands


def load_titan_rows(log):
    """Titan's own 384 one-step training rows -> {fam: [transition]}; every label re-derived."""
    assert sha(TITAN_ROWS) == TITAN_ROWS_SHA, "Titan training-row file changed"
    out = {f: [] for f in FAMS}
    for line in TITAN_ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        fam = r["family"]
        assert r["prompt"][0]["content"] == env.SYSTEM_ONE[fam]
        u = json.loads(r["prompt"][1]["content"])
        s, e = u["before"], u["event"]
        env.cross_check(fam, s, e)
        s2 = env.step(fam, s, e)
        assert env.dumps(s2) == r["reference"], r["id"]
        out[fam].append({"parent_id": r["parent_id"], "cand_id": r["id"], "t": 0, "s": s, "e": e, "s2": s2})
    assert all(len(out[f]) == MATCHED_N for f in FAMS)
    log(f"Titan's training rows: {sum(len(v) for v in out.values())} rows ({TITAN_ROWS_SHA[:12]}...), every "
        f"reference == env.step and the FDS oracle; system prompts == env.SYSTEM_ONE")
    return out


def arrays(fam, trs, encoding="sym"):
    X = np.stack([enc(fam, r["s"], r["e"], encoding) for r in trs]).astype(np.float32)
    S = np.stack([st_arr(fam, r["s"]) for r in trs])
    S2 = np.stack([st_arr(fam, r["s2"]) for r in trs])
    return X, S, S2


def dedupe(fam, trs):
    seen, out = set(), []
    for r in trs:
        key = enc(fam, r["s"], r["e"]).tobytes() + st_arr(fam, r["s2"]).tobytes()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def degenerate_pids(rows, fam, k):
    """TEST parents whose candidates all reach the SAME true phi at k (indistinguishable under every
    goal; regret_table's float32 denominator can still pass den > 1e-9 for some of them)."""
    by = {}
    for r in rows:
        if r["family"] == fam and r["k"] == k and r["split"] == "test":
            by.setdefault(r["parent_id"], []).append(np.asarray(r["phi"]))
    return sorted(p for p, v in by.items() if all(np.array_equal(v[0], x) for x in v[1:]))


# ============================================================================ learner

class _Net(nn.Module):
    def __init__(self, d_in, d_out, hidden, depth):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OneStep:
    """Generic one-step learner f(state, event) -> next discrete state."""

    def __init__(self, fam, head, hidden, depth, lr, seed, wd=1e-4, batch=128, max_steps=6000,
                 eval_every=100, patience=20, encoding="sym", keep="best"):
        self.fam, self.head, self.hidden, self.depth, self.lr, self.seed = fam, head, hidden, depth, lr, seed
        self.encoding, self.keep = encoding, keep
        self.wd, self.batch, self.max_steps, self.eval_every, self.patience = wd, batch, max_steps, eval_every, patience

    # ---- output heads.  m = number of state slots (L: 4 balances, R: 3 registers, P: 8 bits).
    #   P bits        8 sigmoid logits (BCE) -> threshold 0.5
    #   R cls97       97-way softmax per register over the NEXT value (CE)
    #   R cls_res     97-way softmax per register over the residue (next - current) mod 97 (CE)
    #   R reg_abs     regression of the next value ((v-48)/28, MSE) -> round, mod 97
    #   R reg_delta   regression of the wrapped delta ((next-cur+48) mod 97 - 48)/28 (MSE) -> round, add, mod 97
    #   L reg_delta   regression of the integer delta (next-cur)/10 (Huber, delta 1) -> round, add
    #   L cls_delta   softmax per account over every delta value observed in TRAIN (CE; unseen = unlearnable)
    #   L/R gated_delta  per slot a change logit (BCE) + the reg_delta regression trained on changed slots
    #                 only; decode: unchanged slots copied exactly, changed slots = rounded regression
    def _m(self):
        return {"ledger_replay": 4, "register_machine": 3, "prerequisite_plan": 8}[self.fam]

    def _delta(self, S, S2):
        if self.fam == "register_machine":
            return (((S2 - S + 48) % env.MOD) - 48) / 28.0
        return (S2 - S) / 10.0

    def _targets(self, S, S2, fit=False):
        h, fam = self.head, self.fam
        if h == "bits":
            return torch.tensor(S2, dtype=torch.float32)
        if h == "cls97":
            return torch.tensor(S2, dtype=torch.long)
        if h == "cls_res":
            return torch.tensor((S2 - S) % env.MOD, dtype=torch.long)
        if h == "reg_abs":
            return torch.tensor((S2 - 48) / 28.0, dtype=torch.float32)
        if h == "reg_delta":
            return torch.tensor(self._delta(S, S2), dtype=torch.float32)
        if h == "gated_delta":
            return torch.tensor(np.concatenate([(S2 != S).astype(np.float64), self._delta(S, S2)], 1), dtype=torch.float32)
        if h == "cls_delta":
            d = S2 - S
            if fit:
                self.vocab = np.unique(d)
            idx = np.searchsorted(self.vocab, d)
            idx = np.clip(idx, 0, len(self.vocab) - 1)
            idx = np.where(self.vocab[idx] == d, idx, -100)        # unseen delta: unlearnable
            return torch.tensor(idx, dtype=torch.long)
        raise ValueError(h)

    def _d_out(self):
        h, m = self.head, self._m()
        return {"bits": 8, "cls97": 3 * env.MOD, "cls_res": 3 * env.MOD, "reg_abs": 3, "reg_delta": m,
                "gated_delta": 2 * m}.get(h) or m * len(self.vocab)

    def _reg_loss(self, out, y):
        return F.huber_loss(out, y, delta=1.0) if self.fam == "ledger_replay" else F.mse_loss(out, y)

    def _loss(self, out, y):
        h, m = self.head, self._m()
        if h == "bits":
            return F.binary_cross_entropy_with_logits(out, y)
        if h in ("cls97", "cls_res"):
            return F.cross_entropy(out.reshape(-1, env.MOD), y.reshape(-1))
        if h == "cls_delta":
            yy = y.reshape(-1)
            if (yy >= 0).sum() == 0:
                return out.sum() * 0.0
            return F.cross_entropy(out.reshape(-1, len(self.vocab)), yy, ignore_index=-100)
        if h == "gated_delta":
            ch = y[:, :m]
            bce = F.binary_cross_entropy_with_logits(out[:, :m], ch)
            if self.fam == "ledger_replay":
                r = F.huber_loss(out[:, m:], y[:, m:], delta=1.0, reduction="none")
            else:
                r = (out[:, m:] - y[:, m:]) ** 2
            return bce + (r * ch).sum() / ch.sum().clamp(min=1.0)
        return self._reg_loss(out, y)

    def _decode(self, out: np.ndarray, S: np.ndarray) -> np.ndarray:
        h, fam, m = self.head, self.fam, self._m()
        R_ = fam == "register_machine"
        unit = 28.0 if R_ else 10.0
        if h == "bits":
            return (out > 0).astype(np.int64)
        if h == "cls97":
            return out.reshape(len(out), 3, env.MOD).argmax(-1).astype(np.int64)
        if h == "cls_res":
            return np.mod(S + out.reshape(len(out), 3, env.MOD).argmax(-1), env.MOD)
        if h == "reg_abs":
            return np.mod(np.rint(out * 28.0 + 48).astype(np.int64), env.MOD)
        if h in ("reg_delta", "gated_delta"):
            d = np.rint(out[:, -m:].astype(np.float64) * unit).astype(np.int64)
            if h == "gated_delta":
                d = np.where(out[:, :m] > 0, d, 0)
            return np.mod(S + d, env.MOD) if R_ else S + d
        if h == "cls_delta":
            return S + self.vocab[out.reshape(len(out), m, len(self.vocab)).argmax(-1)]
        raise ValueError(h)

    def _x(self, X):
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)

    def fit(self, X, S, S2, Xdv, Sdv, S2dv):
        self.mu, self.sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-3   # heads.Standardizer
        ytr = self._targets(S, S2, fit=True)
        ydv = self._targets(Sdv, S2dv)
        xtr, xdv = self._x(X), self._x(Xdv)
        torch.manual_seed(self.seed)
        net = _Net(X.shape[1], self._d_out(), self.hidden, self.depth)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, self.max_steps) / self.max_steps)))
        g = torch.Generator().manual_seed(self.seed)
        n, b = len(X), min(self.batch, len(X))
        perm, ptr = torch.randperm(n, generator=g), 0
        best_acc, best_loss, best_state, best_step, bad, step = -1.0, float("inf"), None, 0, 0, 0
        for step in range(1, self.max_steps + 1):
            if ptr + b > n:
                perm, ptr = torch.randperm(n, generator=g), 0
            idx = perm[ptr: ptr + b]
            ptr += b
            net.train()
            opt.zero_grad()
            self._loss(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
            sched.step()
            if step % self.eval_every == 0 or step == self.max_steps:
                net.eval()
                with torch.no_grad():
                    o = net(xdv)
                    dl = float(self._loss(o, ydv))
                acc = float((self._decode(o.numpy(), Sdv) == S2dv).all(1).mean())
                if self.keep == "last":                     # sensitivity: no early stopping, final weights
                    best_acc, best_loss, best_step = acc, dl, step
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                elif acc > best_acc + 1e-12 or (acc >= best_acc - 1e-12 and dl < best_loss - 1e-6):
                    best_acc, best_loss, best_step, bad = acc, dl, step, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        net.load_state_dict(best_state)
        net.eval()
        self.net, self.dev_acc, self.dev_loss, self.best_step, self.steps_run = net, best_acc, best_loss, best_step, step
        return self

    def predict(self, X, S) -> np.ndarray:
        with torch.no_grad():
            return self._decode(self.net(self._x(X)).numpy(), S)

    def step_fn(self):
        fam = self.fam

        def f(states, events):
            X = np.stack([enc(fam, s, e, self.encoding) for s, e in zip(states, events)]).astype(np.float32)
            S = np.stack([st_arr(fam, s) for s in states])
            return [arr_st(fam, a) for a in self.predict(X, S)]
        return f


def oracle_step_fn(fam):
    return lambda states, events: [env.step(fam, s, e) for s, e in zip(states, events)]


# ============================================================================ rollout + regret

def rollout(cands, step_fn):
    cur = [c["s0"] for c in cands]
    hist = [[c["s0"]] for c in cands]
    for t in range(H):
        cur = step_fn(cur, [c["events"][t] for c in cands])
        for h_, s in zip(hist, cur):
            h_.append(s)
    return {c["cand_id"]: {"states": [env.dumps(s) for s in h_]} for c, h_ in zip(cands, hist)}


def chain_regrets(rows, fam, chains, mode="sroll"):
    """decisions.g_only on the given chains, one mode per call (fresh rng 17) -> {k: (pids, nregret)}."""
    out = {}
    for k in KS:
        r = decisions.g_only(rows, fam, k, {mode: chains}, modes=(mode,))[mode]
        out[k] = (r["pids"], np.asarray(r["nregret"], np.float64))
    return out


_DEV_SPLIT = {}


def dev_regret(rows, fam, chains, k=4):
    """DEV normalised regret at k under the SELECTION goals (seed 1002) -- prereg1's selection rule
    (Split + _gen_phi + parent-mean fill + rng 17). Used only by crit=regret selection."""
    if fam not in _DEV_SPLIT:
        dv = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "dev"]
        G = decide.goals(len(dv[0]["phi"]), prereg1.N_TEST_GOALS, SEL_GOAL_SEED)
        _DEV_SPLIT[fam] = (dv, G, prereg1.Split(dv, G, k))
    dv, G, sp = _DEV_SPLIT[fam]
    P, ok = prereg1._gen_phi(fam, dv, {"m": chains}, "m")
    assert ok.all()
    return float(np.nanmean(sp.regret(prereg1._fill_parent_mean(sp.cube(P @ G.T)))))


def titan_lastparsed(rows, fam, k, gen, mode):
    """Titan with the registered last-parsed fill (PREREG-1 sensitivity): prereg1._gen_phi(fill=
    'last_parsed') then parent-mean for what is still missing, test goals 1001, rng 17."""
    te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
    G = decide.goals(len(te[0]["phi"]), prereg1.N_TEST_GOALS, prereg1.TEST_GOAL_SEED)
    sp = prereg1.Split(te, G, k)
    P, ok = prereg1._gen_phi(fam, te, gen, mode, fill="last_parsed")
    return sp.pids, np.asarray(sp.regret(prereg1._fill_parent_mean(sp.cube(P @ G.T))), np.float64)


def exact_at_k(cands, chains):
    """Exact-state accuracy of a chain dict at each k over TEST candidates (unparseable = wrong)."""
    res = {}
    for k in KS:
        hit = []
        for c in cands:
            g = chains.get(c["cand_id"])
            s = None if g is None else g["states"][k]
            hit.append(s is not None and json.loads(s) == c["gold"][k])
        res[k] = float(np.mean(hit))
    return res


def parse_rate(cands, chains):
    return {k: float(np.mean([chains.get(c["cand_id"]) is not None and chains[c["cand_id"]]["states"][k] is not None
                              for c in cands])) for k in KS}


def onestep_stats(fam, chains, cands, rule):
    """Per-stratum one-step exact accuracy of a CHAIN (Titan's or S-roll's).
    rule='input_correct' (the RESULTS_PILOT R8 rule): steps whose input state equals the gold state;
         output judged against the gold next state; strata on the gold input. (Reproduces R8 exactly
         for R; for L/P this gives 1-5 more steps per stratum than R8, all of them wrong outputs.)
    rule='own_input' (no oracle conditioning on the input): every step with a parsed input; output
         judged against env.step(own input, event); strata on own input.
    rule='step1': first step only (input = gold s0 for every candidate; an unselected sample).
    An unparseable output counts as wrong under every rule (n_unparsed reports how many). S-roll never
    emits an unparseable state, so this only matters for Titan."""
    acc = {}
    for c in cands:
        g = chains.get(c["cand_id"])
        if g is None:
            continue
        ps = [None if s is None else json.loads(s) for s in g["states"]]
        for t in range(H):
            if rule == "step1" and t > 0:
                break
            e = c["events"][t]
            if rule in ("input_correct", "step1"):
                if ps[t] is None or ps[t] != c["gold"][t]:
                    continue
                inp, want = c["gold"][t], c["gold"][t + 1]
            else:
                if ps[t] is None:
                    break
                inp = ps[t]
                want = env.step(fam, inp, e)
            ok = ps[t + 1] is not None and ps[t + 1] == want
            for lab in strata(fam, inp, e) + ["ALL"]:
                a = acc.setdefault(lab, [0, 0, 0])
                a[0] += 1
                a[1] += int(ok)
                a[2] += int(ps[t + 1] is None)
    return {lab: {"n": v[0], "acc": v[1] / v[0], "n_unparsed": v[2]} for lab, v in sorted(acc.items())}


def teacher_forced_stats(fam, model, trs):
    X, S, S2 = arrays(fam, trs, model.encoding)
    ok = (model.predict(X, S) == S2).all(1)
    acc = {}
    for r, o in zip(trs, ok):
        for lab in strata(fam, r["s"], r["e"]) + ["ALL"]:
            a = acc.setdefault(lab, [0, 0])
            a[0] += 1
            a[1] += int(o)
    return {lab: {"n": v[0], "acc": v[1] / v[0]} for lab, v in sorted(acc.items())}


def mean_stats(list_of_stats):
    """Average accuracy over models (per stratum); n is per model (identical for teacher-forced)."""
    labs = sorted({lab for s in list_of_stats for lab in s})
    out = {}
    for lab in labs:
        xs = [s[lab]["acc"] for s in list_of_stats if lab in s]
        ns = [s[lab]["n"] for s in list_of_stats if lab in s]
        out[lab] = {"n_mean": float(np.mean(ns)), "acc_mean": float(np.mean(xs)), "acc_sd": float(np.std(xs)),
                    "n_models": len(xs)}
    return out


def titan_error_anatomy(fam, gen, cands):
    """What Titan's WRONG first steps look like (chain_one, gold s0 input)."""
    c = collections.Counter()
    for cd in cands:
        g = gen["chain_one"].get(cd["cand_id"])
        st = None if g is None else g["states"][1]
        t = None if st is None else json.loads(st)
        s, e, true = cd["s0"], cd["events"][0], cd["gold"][1]
        if fam == "prerequisite_plan":
            ign = (s & ~e["delete"]) | e["add"]          # the step if the precondition is ignored
            if env.applicable(s, e):
                c["applicable_n"] += 1
                if t != true:
                    c["applicable_wrong"] += 1
                    c["applicable_wrong_unparsed"] += int(t is None)
                    c["applicable_wrong_eq_s_plus_add"] += int(t == s + e["add"])
                    c["applicable_wrong_eq_s_or_add"] += int(t == (s | e["add"]))
            else:
                c["inapplicable_n"] += 1
                c["inapplicable_titan_eq_ignore_precondition"] += int(t == ign)
                c["inapplicable_titan_eq_s_plus_add"] += int(t == s + e["add"])
                if ign == true:
                    c["inapplicable_where_ignoring_gives_gold_n"] += 1
                    c["inapplicable_where_ignoring_gives_gold_titan_right"] += int(t == true)
                    c["inapplicable_where_ignoring_gives_gold_titan_eq_s_plus_add"] += int(t == s + e["add"])
        elif fam == "ledger_replay" and e["status"] != "posted":
            c["void_n"] += 1
            c["void_titan_right"] += int(t == true)
            c["void_titan_eq_applied_as_posted"] += int(t == env.step(fam, s, {**e, "status": "posted"}))
    return dict(sorted(c.items()))


# ============================================================================ selection + fitting

STAGE_A = {"hidden": 256, "depth": 2, "lr": 3e-3}                  # head (x encoding) screen
STAGE_B = ({"hidden": 256, "depth": 2, "lr": 1e-3}, {"hidden": 512, "depth": 2, "lr": 3e-3},
           {"hidden": 256, "depth": 3, "lr": 3e-3})                # size / LR for the chosen head
SMOKE_CFG = {"hidden": 64, "depth": 1, "lr": 3e-3}


def train_kw(n, smoke, crit="exact"):
    """Optimisation schedule by training-set size (AdamW wd 1e-4, cosine decay to 5% over max_steps,
    DEV exact accuracy every eval_every steps, best checkpoint kept, stop after `patience` evaluations
    without improvement). crit=regret, n > 1000: twice the step cap and patience 20."""
    if smoke:
        return dict(batch=min(128, n), max_steps=200, eval_every=50, patience=3)
    if n <= 256:
        return dict(batch=n, max_steps=2000, eval_every=25, patience=15)
    if n <= 1000:
        return dict(batch=256, max_steps=3000, eval_every=50, patience=15)
    if crit == "regret":
        return dict(batch=256, max_steps=8000, eval_every=100, patience=20)
    return dict(batch=256, max_steps=4000, eval_every=100, patience=12)


def make(fam, cfg, seed, n, smoke, crit="exact", keep="best"):
    return OneStep(fam, cfg["head"], cfg["hidden"], cfg["depth"], cfg["lr"], seed,
                   encoding=cfg.get("encoding", "sym"), keep=keep, **train_kw(n, smoke, crit))


def sample(pool, n, seed):
    idx = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def fit_one(fam, cfg, seed, tr, dv, smoke, crit="exact", keep="best"):
    e = cfg.get("encoding", "sym")
    return make(fam, cfg, seed, len(tr), smoke, crit, keep).fit(*arrays(fam, tr, e), *arrays(fam, dv, e))


def select(fam, sets, dv, smoke, log, tag, with_encodings=False, crit="exact", rows=None, dcands=None):
    """Two-stage DEV selection: stage A screens every output head (and, for R at FULL, the
    one-hot-augmented input encoding for the two classification heads) at a default size; stage B
    tries 3 size/LR variants of the stage-A winner. `sets` = [(train transitions, seed)] the criterion
    is averaged over. crit=exact: DEV one-step exact-state accuracy (ties -> lower DEV loss).
    crit=regret: DEV k=4 regret of the DEV rollout under the selection goals (ties -> DEV exact)."""
    res = []

    def run(cfg, stage):
        t0 = time.time()
        ex, lo, rg, bs, sr = [], [], [], [], []
        for trs, seed in sets:
            m = fit_one(fam, cfg, seed, trs, dv, smoke, crit)
            ex.append(m.dev_acc)
            lo.append(m.dev_loss)
            bs.append(m.best_step)
            sr.append(m.steps_run)
            if crit == "regret":
                rg.append(dev_regret(rows, fam, rollout(dcands, m.step_fn())))
        rec = {**cfg, "stage": stage, "dev_exact": float(np.mean(ex)), "dev_loss": float(np.mean(lo)),
               "best_step": bs if len(bs) > 1 else bs[0], "steps_run": sr if len(sr) > 1 else sr[0],
               "sec": round(time.time() - t0, 1)}
        if crit == "regret":
            rec["dev_regret_k4"] = float(np.mean(rg))
            rec["dev_regret_k4_per_model"] = rg
        res.append(rec)
        log(f"  select[{crit}] {SHORT[fam]} {tag:>5} {stage} {cfg} dev_exact={rec['dev_exact']:.4f}"
            + (f" dev_regret4={rec['dev_regret_k4']:.4f}" if crit == "regret" else "")
            + f" best_step={bs}/{sr} ({time.time() - t0:.1f}s)")

    base = SMOKE_CFG if smoke else STAGE_A
    for h in HEADS[fam]:
        run({"head": h, "encoding": "sym", **base}, "A")
    if with_encodings and fam == "register_machine":
        for h in ("cls97", "cls_res"):
            run({"head": h, "encoding": "sym1h", **base}, "A")
    if crit == "exact":
        key = lambda r: (r["dev_exact"], -r["dev_loss"])
    else:
        key = lambda r: (-r["dev_regret_k4"], r["dev_exact"])
    a = max(res, key=key)
    if not smoke:
        for sz in STAGE_B:
            run({"head": a["head"], "encoding": a["encoding"], **sz}, "B")
    best = max(res, key=key)
    chosen = {k: best[k] for k in ("head", "encoding", "hidden", "depth", "lr")}
    log(f"  select[{crit}] {SHORT[fam]} {tag:>5} CHOSEN {chosen} dev_exact={best['dev_exact']:.4f}"
        + (f" dev_regret4={best['dev_regret_k4']:.4f}" if crit == "regret" else ""))
    return chosen, res


# ============================================================================ per-family run

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize_pp(pp: np.ndarray) -> dict:
    b = decide.bootstrap_mean(pp.copy())
    return {"mean": b["mean"], "ci95": b["ci95"], "n": b["n"]}


def load_gen():
    gen = {}
    for mode in ("chain_one", "chain_m1"):
        p = ROOT / "runs" / BANK / "sparse" / f"gen_{mode}.jsonl"
        gen[mode] = {r["cand_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    p = ROOT / "runs" / BANK / "sparse" / "gen_e2e.jsonl"
    gen["e2e"] = {r["item_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    return gen


def paths(f, crit, sfx):
    t = CRIT_TAG[crit]
    return {"select": OUT / f"s_roll_select{t}_{f}{sfx}.json", "select_log": OUT / f"s_roll_select{t}_{f}{sfx}.log",
            "out": OUT / f"s_roll{t}_{f}{sfx}.json", "log": OUT / f"s_roll{t}_{f}{sfx}.log",
            "npz": OUT / f"s_roll_per_model{t}_{f}{sfx}.npz"}


def run_family(fam, smoke, log, inputs, stage="all", sfx="", crit="exact"):
    bank, rows, T, cands, dcands, gen, dry, dry_pp, trows = inputs
    f = SHORT[fam]
    P_ = paths(f, crit, sfx)
    t_start = time.time()
    out = {"family": fam, "crit": crit, "data": {}, "selection": {}, "regret": {}, "contrasts": {},
           "exact_state": {}, "onestep": {}, "curve": {}, "sanity": {}, "per_parent": {}, "fits": {},
           "titan": {}}
    npz = {}
    tr_raw, dv, te = T[fam]["train"], T[fam]["dev"], T[fam]["test"]
    tr = dedupe(fam, tr_raw)
    te_pids = {r["parent_id"] for r in te}
    assert not te_pids & {r["parent_id"] for r in tr_raw + dv}
    trkeys = {enc(fam, r["s"], r["e"]).tobytes() for r in tr}
    te_hit = [enc(fam, r["s"], r["e"]).tobytes() in trkeys for r in te]
    te_in_tr = float(np.mean(te_hit))
    overlap_pids = sorted({r["parent_id"] for r, h in zip(te, te_hit) if h})
    tr_states = {env.dumps(r["s"]) for r in tr}
    tr_s0 = {env.dumps(r["s"]) for r in tr_raw if r["t"] == 0}
    noop_pool = [r for r in tr if not is_noop_row(fam, r["e"], r["s"])]
    strat_tr, strat_ti = {}, {}
    for r in tr:
        for lab in strata(fam, r["s"], r["e"]):
            strat_tr[lab] = strat_tr.get(lab, 0) + 1
    for r in trows[fam]:
        for lab in strata(fam, r["s"], r["e"]):
            strat_ti[lab] = strat_ti.get(lab, 0) + 1
    degen = {k: degenerate_pids(rows, fam, k) for k in KS}
    vocab_ceiling = {}
    if fam == "ledger_replay":         # cls_delta can only emit per-account deltas seen in TRAIN
        voc = set(np.unique(np.concatenate([st_arr(fam, r["s2"]) - st_arr(fam, r["s"]) for r in tr])).tolist())
        cnt = {}
        for r in te:
            ok = all(int(d) in voc for d in st_arr(fam, r["s2"]) - st_arr(fam, r["s"]))
            for lab in strata(fam, r["s"], r["e"]) + ["ALL"]:
                a = cnt.setdefault(lab, [0, 0])
                a[0] += 1
                a[1] += int(ok)
        vocab_ceiling = {"vocab_size": len(voc),
                         "test_frac_all_deltas_in_vocab": {lab: v[1] / v[0] for lab, v in sorted(cnt.items())}}
    out["data"] = {"cls_delta_vocab_ceiling": vocab_ceiling,"n_train_raw": len(tr_raw), "n_train_dedup": len(tr), "n_dev": len(dv), "n_test": len(te),
                   "n_test_cands": len(cands[fam]), "n_test_parents": len(te_pids),
                   "test_inputs_seen_in_train_frac": te_in_tr, "test_overlap_parents": overlap_pids,
                   "test_input_state_seen_as_train_input_frac": float(np.mean([env.dumps(r["s"]) in tr_states for r in te])),
                   "test_s0_seen_as_train_s0_frac": float(np.mean([env.dumps(c["s0"]) in tr_s0 for c in cands[fam]])),
                   "n_train_noopfree_pool": len(noop_pool), "train_strata": dict(sorted(strat_tr.items())),
                   "titan_rows_n": len(trows[fam]), "titan_rows_dedup_n": len(dedupe(fam, trows[fam])),
                   "titan_rows_strata": dict(sorted(strat_ti.items())),
                   "titan_rows_test_inputs_overlap_frac": float(np.mean(
                       [enc(fam, r["s"], r["e"]).tobytes() in {enc(fam, q["s"], q["e"]).tobytes() for q in trows[fam]}
                        for r in te])),
                   "degenerate_test_parents": {k: v for k, v in degen.items()}}
    log(f"[{f}] transitions train raw {len(tr_raw)} dedup {len(tr)} | dev {len(dv)} | test {len(te)} "
        f"({len(te_pids)} parents) | TEST inputs seen in TRAIN {te_in_tr:.3f} ({len(overlap_pids)} parents) | "
        f"Titan rows {len(trows[fam])} {strat_ti} | degenerate TEST parents k1/2/4 "
        f"{[len(degen[k]) for k in KS]}")

    # ---------------- sanity: oracle one-step model through the identical rollout+phi+regret path
    orc = rollout(cands[fam], oracle_step_fn(fam))
    orc_reg = chain_regrets(rows, fam, orc, mode="oracle_roll")
    orc_ex = exact_at_k(cands[fam], orc)
    copy = {c["cand_id"]: {"states": [env.dumps(c["s0"])] * (H + 1)} for c in cands[fam]}
    rnd_reg = chain_regrets(rows, fam, copy, mode="copy_s0")
    out["sanity"] = {"oracle_rollout_regret": {k: float(np.nanmean(orc_reg[k][1])) for k in KS},
                     "oracle_rollout_regret_max": {k: float(np.nanmax(orc_reg[k][1])) for k in KS},
                     "oracle_rollout_exact": orc_ex}
    assert all(abs(out["sanity"]["oracle_rollout_regret_max"][k]) < 1e-9 for k in KS), out["sanity"]
    assert all(orc_ex[k] == 1.0 for k in KS), orc_ex
    log(f"[{f}] SANITY oracle one-step rollout: regret {out['sanity']['oracle_rollout_regret']} exact {orc_ex}")

    # ---------------- Titan (same g_only path, one mode per call = fresh rng 17)
    titan, titan_lp = {}, {}
    for mode in ("chain_one", "chain_m1", "e2e"):
        titan[mode] = {}
        for k in KS:
            r = decisions.g_only(rows, fam, k, gen, modes=(mode,))[mode]
            titan[mode][k] = (r["pids"], np.asarray(r["nregret"], np.float64))
    chk = {}
    for mode in ("chain_one", "chain_m1", "e2e"):
        for k in KS:
            ref = dry[f"{fam}|k{k}"]["means"][f"Gtitan_{mode}"]
            got = float(np.nanmean(titan[mode][k][1]))
            chk[f"{mode}|k{k}"] = {"ours": got, "PREREG_1_DRYRUN": ref, "absdiff": abs(got - ref)}
    for mode in ("chain_one", "chain_m1"):
        titan_lp[mode] = {}
        for k in KS:
            pids_lp, v = titan_lastparsed(rows, fam, k, gen, mode)
            assert pids_lp == titan[mode][k][0]
            titan_lp[mode][k] = v
            ref = np.asarray(dry_pp[f"{fam}|k{k}|Gtitan_{mode}_fill_lastparsed"], np.float64)
            d = float(np.nanmax(np.abs(v - ref))) if np.isfinite(v - ref).any() else 0.0
            assert np.array_equal(np.isnan(v), np.isnan(ref)) and d < 1e-6, (mode, k, d)
            chk[f"{mode}_lastparsed|k{k}"] = {"ours": float(np.nanmean(v)), "PREREG_1_DRYRUN_PER_PARENT": float(np.nanmean(ref)),
                                              "absdiff": d}
    out["sanity"]["titan_vs_dryrun"] = chk
    worst = max(v["absdiff"] for v in chk.values())
    log(f"[{f}] SANITY Titan regrets (parent-mean and last-parsed fill) vs PREREG_1_DRYRUN: max |diff| = {worst:.2e}; "
        f"chain_one k4 {chk['chain_one|k4']['ours']:.4f}, last-parsed {chk['chain_one_lastparsed|k4']['ours']:.4f}")
    assert worst < 1e-6, chk
    out["titan"] = {"parse_rate": {m: parse_rate(cands[fam], gen[m]) for m in ("chain_one", "chain_m1")},
                    "error_anatomy_step1": titan_error_anatomy(fam, gen, cands[fam])}
    log(f"[{f}] Titan parse rate chain_one k1/2/4 {out['titan']['parse_rate']['chain_one']} | anatomy "
        f"{out['titan']['error_anatomy_step1']}")

    # ---------------- selection on DEV; cached so selection and fitting can run as separate processes
    sel_path = P_["select"]
    if stage in ("select", "all"):
        t0 = time.time()
        n128 = [sample(tr, MATCHED_N, RESAMPLE_SEED0 + r) for r in range(N_SEL_RESAMPLES)]
        if crit == "exact":
            sets_full, sets_128 = [(tr, SEEDS[0])], [(n128[0], SEEDS[0])]
        else:
            sets_full, sets_128 = [(tr, SEEDS[0])], [(x, SEEDS[0]) for x in n128]
        kw = dict(crit=crit, rows=rows, dcands=dcands[fam])
        sel = {"FULL": select(fam, sets_full, dv, smoke, log, "FULL", with_encodings=True, **kw),
               "N128": select(fam, sets_128, dv, smoke, log, "N128", **kw)}
        sel_json = {b: {"chosen": c, "grid": g_} for b, (c, g_) in sel.items()}
        sel_path.write_text(json.dumps({"selection": sel_json, "crit": crit, "meta": code_meta(),
                                        "wall_sec": round(time.time() - t0, 1)},
                                       indent=1, default=float) + "\n", encoding="utf-8")
        log(f"[{f}] selection written to {sel_path.name} ({time.time() - t0:.0f}s)")
        if stage == "select":
            return None, None
    loaded = json.loads(sel_path.read_text(encoding="utf-8"))
    assert loaded.get("crit", "exact") == crit
    sel_json = loaded["selection"]
    sel = {b: (v["chosen"], v["grid"]) for b, v in sel_json.items()}
    out["selection"] = {**sel_json, "curve_rule": "N512 / N2000 reuse the FULL-selected configuration",
                        "n128_rule": "MATCHED128, TITANROWS, NOOPFREE128 and MATCHED128_NOSTOP use the "
                                     "N128-selected configuration",
                        "selection_file": sel_path.name, "selection_file_sha256": sha(sel_path),
                        "selection_wall_sec": loaded.get("wall_sec")}

    # ---------------- fits
    arms = {}

    def run_arm(name, cfg, train_sets, keep="best"):
        res = []
        for tag, trs, seed in train_sets:
            t0 = time.time()
            m = fit_one(fam, cfg, seed, trs, dv, smoke, crit, keep)
            res.append((tag, m, len(trs)))
            log(f"  fit[{crit}] {f} {name:<17} {tag:<11} n={len(trs):5d} dev_exact={m.dev_acc:.4f} "
                f"step={m.best_step}/{m.steps_run} ({time.time() - t0:.1f}s)")
        arms[name] = res
        bsteps = [m_.best_step for _, m_, _ in res]
        out["fits"][name] = {"config": cfg, "keep": keep, "schedule": train_kw(len(train_sets[0][1]), smoke, crit),
                             "best_step_min_median_max": [int(np.min(bsteps)), float(np.median(bsteps)), int(np.max(bsteps))],
                             "models": [{"tag": t_, "n_train": n_, "dev_exact": m_.dev_acc,
                                         "best_step": m_.best_step, "steps_run": m_.steps_run}
                                        for t_, m_, n_ in res]}

    cfg_full, cfg_128 = sel["FULL"][0], sel["N128"][0]
    run_arm("FULL", cfg_full, [(f"seed{s}", tr, s) for s in SEEDS])
    run_arm("MATCHED128", cfg_128,
            [(f"r{r}_seed{s}", sample(tr, MATCHED_N, RESAMPLE_SEED0 + r), s)
             for r in range(N_RESAMPLES) for s in SEEDS])
    run_arm("TITANROWS", cfg_128, [(f"seed{s}", trows[fam], s) for s in SEEDS])
    if crit == "exact" and fam == "prerequisite_plan":
        run_arm("NOOPFREE128", cfg_128,
                [(f"r{r}_seed{s}", sample(noop_pool, MATCHED_N, NOOPFREE_SEED0 + r), s)
                 for r in range(N_RESAMPLES) for s in SEEDS])
    if crit == "exact":
        run_arm("MATCHED128_NOSTOP", cfg_128,
                [(f"r{r}_seed{SEEDS[0]}", sample(tr, MATCHED_N, RESAMPLE_SEED0 + r), SEEDS[0])
                 for r in range(N_RESAMPLES)], keep="last")
    for n in CURVE_NS:
        if n < len(tr):
            run_arm(f"N{n}", cfg_full,
                    [(f"r{i}_seed{s}", sample(tr, n, RESAMPLE_SEED0 + i), s)
                     for i, s in enumerate(SEEDS[:N_CURVE_MODELS])])

    # ---------------- rollouts, regrets, accuracies
    reg, ex, os_ = {}, {}, {}
    pids = titan["chain_one"][4][0]
    pp_store = {}
    for name, models in arms.items():
        per_k = {k: [] for k in KS}
        exs, tf, ic, own, s1 = [], [], [], [], []
        for tag, m, ntr in models:
            ch = rollout(cands[fam], m.step_fn())
            rg = chain_regrets(rows, fam, ch)
            for k in KS:
                assert rg[k][0] == titan["chain_one"][k][0]
                per_k[k].append(rg[k][1])
            exs.append(exact_at_k(cands[fam], ch))
            tf.append(teacher_forced_stats(fam, m, te))
            ic.append(onestep_stats(fam, ch, cands[fam], "input_correct"))
            own.append(onestep_stats(fam, ch, cands[fam], "own_input"))
            s1.append(onestep_stats(fam, ch, cands[fam], "step1"))
            if name == "FULL":
                npz[f"{f}|FULL|{tag}|chains"] = np.array([json.dumps(ch[c["cand_id"]]["states"]) for c in cands[fam]])
        reg[name] = {}
        for k in KS:
            stack = np.stack(per_k[k])
            npz[f"{f}|{name}|k{k}|per_model_nregret"] = stack
            with np.errstate(all="ignore"):
                pp = np.nanmean(stack, 0)
            pm = [float(np.nanmean(x)) for x in stack]
            reg[name][k] = {**summarize_pp(pp), "per_model_mean": pm, "per_model_sd": float(np.std(pm)),
                            "n_models": len(stack), "n_train": [x[2] for x in models]}
            if name in ("MATCHED128", "NOOPFREE128"):
                reg[name][k]["per_resample_mean"] = np.array(pm).reshape(N_RESAMPLES, len(SEEDS)).mean(1).tolist()
            pp_store[(name, k)] = pp
            out["per_parent"].setdefault(f"S-roll {name}", {})[f"k{k}"] = pp.tolist()
        ex[name] = {k: {"mean": float(np.mean([e[k] for e in exs])), "sd": float(np.std([e[k] for e in exs]))}
                    for k in KS}
        os_[name] = {"teacher_forced_test": mean_stats(tf), "chain_input_correct": mean_stats(ic),
                     "chain_own_input": mean_stats(own), "chain_step1": mean_stats(s1),
                     "dev_exact_mean": float(np.mean([m.dev_acc for _, m, _ in models]))}
        log(f"[{f}] S-roll[{crit}] {name:<17} regret k1/2/4 = " + " / ".join(f"{reg[name][k]['mean']:.3f}" for k in KS)
            + " | exact k1/2/4 = " + " / ".join(f"{ex[name][k]['mean']:.3f}" for k in KS)
            + f" | TEST one-step exact {os_[name]['teacher_forced_test']['ALL']['acc_mean']:.3f}")
    for mode in ("chain_one", "chain_m1", "e2e"):
        reg[f"Titan {mode}"] = {k: summarize_pp(titan[mode][k][1]) for k in KS}
        for k in KS:
            out["per_parent"].setdefault(f"Titan {mode}", {})[f"k{k}"] = titan[mode][k][1].tolist()
    for mode in ("chain_one", "chain_m1"):
        reg[f"Titan {mode} [last-parsed fill]"] = {k: summarize_pp(titan_lp[mode][k]) for k in KS}
        for k in KS:
            out["per_parent"].setdefault(f"Titan {mode} [last-parsed fill]", {})[f"k{k}"] = titan_lp[mode][k].tolist()
    reg["oracle"] = {k: summarize_pp(orc_reg[k][1]) for k in KS}
    reg["random (all candidates tied)"] = {k: summarize_pp(rnd_reg[k][1]) for k in KS}
    for k in KS:
        out["per_parent"].setdefault("random (all candidates tied)", {})[f"k{k}"] = rnd_reg[k][1].tolist()
    out["per_parent"]["pids"] = list(pids)
    reg["ref: S (k-step symbolic, equal label budget; PREREG_1_DRYRUN)"] = {
        k: {"mean": dry[f"{fam}|k{k}"]["means"]["S"]} for k in KS}
    if "F_touched" in dry[f"{fam}|k4"]["means"]:
        reg["ref: F_touched (PREREG_1_DRYRUN)"] = {k: {"mean": dry[f"{fam}|k{k}"]["means"]["F_touched"]} for k in KS}
    out["regret"] = reg

    for mode in ("chain_one", "chain_m1"):
        ex[f"Titan {mode}"] = {k: {"mean": v} for k, v in exact_at_k(cands[fam], gen[mode]).items()}
        os_[f"Titan {mode}"] = {"chain_input_correct": onestep_stats(fam, gen[mode], cands[fam], "input_correct"),
                                "chain_own_input": onestep_stats(fam, gen[mode], cands[fam], "own_input"),
                                "chain_step1": onestep_stats(fam, gen[mode], cands[fam], "step1")}
    out["exact_state"] = ex
    out["onestep"] = os_

    con = {}
    keep_ov = np.array([p not in set(overlap_pids) for p in pids])
    for name in arms:
        for k in KS:
            for mode in ("chain_one", "chain_m1"):
                con[f"Titan {mode} - S-roll {name}|k{k}"] = decide.paired_bootstrap(titan[mode][k][1], pp_store[(name, k)])
            con[f"Titan chain_one [last-parsed fill] - S-roll {name}|k{k}"] = decide.paired_bootstrap(
                titan_lp["chain_one"][k], pp_store[(name, k)])
            a, b = titan["chain_one"][k][1], pp_store[(name, k)]
            if overlap_pids:
                c_ = decide.paired_bootstrap(a[keep_ov], b[keep_ov])
                con[f"Titan chain_one - S-roll {name}|k{k}|excl_overlap"] = {**c_, "n_parents_dropped": len(overlap_pids)}
            if degen[k]:
                kd = np.array([p not in set(degen[k]) for p in pids])
                c_ = decide.paired_bootstrap(a[kd], b[kd])
                con[f"Titan chain_one - S-roll {name}|k{k}|excl_degenerate"] = {
                    **c_, "n_degenerate": len(degen[k]),
                    "n_degenerate_kept_by_frozen_rule": int(sum(not np.isnan(a[pids.index(p)]) for p in degen[k]))}
    out["contrasts"] = con
    for k in KS:
        c1 = con[f"Titan chain_one - S-roll FULL|k{k}"]
        c2 = con[f"Titan chain_one - S-roll MATCHED128|k{k}"]
        c3 = con[f"Titan chain_one - S-roll TITANROWS|k{k}"]
        log(f"[{f}] k{k} Titan chain_one {reg['Titan chain_one'][k]['mean']:.3f} | S-roll[{crit}] FULL "
            f"{reg['FULL'][k]['mean']:.3f} diff {c1['mean_diff']:+.3f} [{c1['ci95'][0]:+.3f}, {c1['ci95'][1]:+.3f}]"
            f" | MATCHED {reg['MATCHED128'][k]['mean']:.3f} diff {c2['mean_diff']:+.3f} "
            f"[{c2['ci95'][0]:+.3f}, {c2['ci95'][1]:+.3f}] | TITANROWS {reg['TITANROWS'][k]['mean']:.3f} diff "
            f"{c3['mean_diff']:+.3f} [{c3['ci95'][0]:+.3f}, {c3['ci95'][1]:+.3f}]")
    out["curve"] = {name: {"n_train_mean": float(np.mean(reg[name][4]["n_train"])), "regret_k4": reg[name][4]["mean"],
                           "test_onestep_exact": os_[name]["teacher_forced_test"]["ALL"]["acc_mean"],
                           "exact_k4": ex[name][4]["mean"]}
                    for name in arms if name in ("FULL", "MATCHED128") or name.startswith("N") and name[1:].isdigit()}
    out["wall_sec"] = round(time.time() - t_start, 1)
    log(f"[{f}] family done in {out['wall_sec']:.0f}s")
    return out, npz


# ============================================================================ main

def code_meta():
    return {"code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                            [Path(__file__).resolve()] + sorted((ROOT / "rcwmt").glob("*.py"))},
            "gen_sha256": {m: sha(ROOT / "runs" / BANK / "sparse" / f"gen_{m}.jsonl") for m in ("chain_one", "chain_m1", "e2e")},
            "bank_sha256": sha(ROOT / "data" / f"{BANK}.json"),
            "titan_rows_sha256": TITAN_ROWS_SHA,
            "torch": torch.__version__, "numpy": np.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", help="L, P, R or all (default: all three, then the report)")
    ap.add_argument("--crit", default="both", choices=("both",) + CRITS,
                    help="selection criterion: exact (spec), regret (decision-relevant) or both")
    ap.add_argument("--report-only", action="store_true", help="merge existing per-family outputs and render")
    ap.add_argument("--stage", default="all", choices=("all", "select", "fit"),
                    help="select: DEV selection only (cached); fit: fits/evaluation using the cached selection")
    ap.add_argument("--smoke", action="store_true", help="tiny schedule; writes *_smoke outputs only")
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    short2fam = {v: k for k, v in SHORT.items()}
    fams = [] if a.report_only else (list(FAMS) if a.family == "all" else [short2fam[x] for x in a.family.split(",")])
    crits = CRITS if a.crit == "both" else (a.crit,)
    inputs = None
    for crit in crits:
        for fam in fams:
            P_ = paths(SHORT[fam], crit, sfx)
            log = Log(P_["select_log"] if a.stage == "select" else P_["log"])
            log(f"S-roll family {SHORT[fam]} crit={crit} stage={a.stage}; torch {torch.__version__} "
                f"threads={torch.get_num_threads()} cuda_visible={os.environ['CUDA_VISIBLE_DEVICES']!r} "
                f"cuda_available={torch.cuda.is_available()} smoke={a.smoke}")
            if inputs is None:
                bank, rows, T, cands, dcands = load_inputs(log)
                trows = load_titan_rows(log)
                dry = json.loads((OUT / "PREREG_1_DRYRUN.json").read_text(encoding="utf-8"))["per_family"]
                dry_pp = dict(np.load(OUT / "PREREG_1_DRYRUN_PER_PARENT.npz", allow_pickle=True))
                inputs = (bank, rows, T, cands, dcands, load_gen(), dry, dry_pp, trows)
            res, npz = run_family(fam, a.smoke, log, inputs, a.stage, sfx, crit)
            if res is None:
                log.f.close()
                continue
            res["meta"] = {**code_meta(), "smoke": a.smoke}
            P_["out"].write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
            np.savez_compressed(P_["npz"], **npz)
            log(f"wrote {P_['out'].name} / {P_['npz'].name}")
            log.f.close()
    full_run = a.family == "all" and a.stage == "all" and a.crit == "both"
    if full_run and not a.smoke:
        # depth-extension robustness check (DEV-only depth search on the crit=exact FULL configuration);
        # a separate process so its torch state cannot interact with this one
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "s_roll_deep.py")], check=True,
                       cwd=str(ROOT))
    if a.report_only or full_run:
        report(sfx)


def report(sfx):
    log = Log(OUT / f"s_roll{sfx}.log")
    meta_now = code_meta()
    R = {"fam": {}, "cross": {}}
    for fam in FAMS:
        f = SHORT[fam]
        parts = {}
        for crit in CRITS:
            p = paths(f, crit, sfx)["out"]
            if p.exists():
                parts[crit] = json.loads(p.read_text(encoding="utf-8"))
        if not parts:
            continue
        R["fam"][fam] = parts
        # regret-selected minus spec (exact-selected) S-roll, per arm and k (same TEST parents)
        if len(parts) == 2:
            cx = {}
            ppx, ppr = parts["exact"]["per_parent"], parts["regret"]["per_parent"]
            for name in ppr:
                if name.startswith("S-roll") and name in ppx:
                    for k in KS:
                        cx[f"{name} [regret-sel] - {name} [exact-sel]|k{k}"] = decide.paired_bootstrap(
                            np.array(ppr[name][f"k{k}"], np.float64), np.array(ppx[name][f"k{k}"], np.float64))
            R["cross"][fam] = cx
    stale = {SHORT[f]: {c: [k for k, v in p_["meta"]["code_sha256"].items() if meta_now["code_sha256"].get(k) != v]
                        for c, p_ in parts.items()} for f, parts in R["fam"].items()}
    R["meta"] = {**meta_now, "bank": BANK, "seeds": SEEDS, "matched_n": MATCHED_N, "n_resamples": N_RESAMPLES,
                 "curve_ns": CURVE_NS, "n_curve_models": N_CURVE_MODELS, "resample_seed0": RESAMPLE_SEED0,
                 "noopfree_seed0": NOOPFREE_SEED0, "sel_goal_seed": SEL_GOAL_SEED, "n_sel_resamples": N_SEL_RESAMPLES,
                 "stage_a": STAGE_A, "stage_b": STAGE_B, "heads": HEADS, "titan_rows": str(TITAN_ROWS),
                 "families_present": [SHORT[f] for f in R["fam"]], "code_changed_since_family_run": stale,
                 "family_wall_sec": {SHORT[f]: {c: p_["wall_sec"] for c, p_ in parts.items()} for f, parts in R["fam"].items()}}
    for fam in R["fam"]:
        f = SHORT[fam]
        for crit in CRITS:
            P_ = paths(f, crit, sfx)
            for lp in (P_["select_log"], P_["log"]):
                if lp.exists():
                    log.f.write(f"===== {lp.name} =====\n" + lp.read_text(encoding="utf-8"))
    log(f"report: families {R['meta']['families_present']}; code changed since family run: {stale}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import s_roll_report  # noqa: E402  (formatting only)
    R["meta"]["report_code_sha256"] = sha(Path(s_roll_report.__file__))
    (OUT / f"s_roll{sfx}.json").write_text(json.dumps(R, indent=1, default=float) + "\n", encoding="utf-8")
    (DOC / f"S_ROLL_PILOT{sfx}.md").write_text(s_roll_report.render(json.loads(json.dumps(R, default=float))),
                                               encoding="utf-8")
    log(f"wrote {OUT / f's_roll{sfx}.json'} and {DOC / f'S_ROLL_PILOT{sfx}.md'}")


if __name__ == "__main__":
    main()
