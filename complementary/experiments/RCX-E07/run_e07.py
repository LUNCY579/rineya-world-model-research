"""RCX-E07/E08 runner: relational value learning for Blocksworld (RGM-001 x RVM-001; CPU mechanism tier).

A learned heuristic here is an external specialist (tier B); nothing in this run supports a Rineya-core claim.

Usage (from complementary/):
  python -B experiments/RCX-E07/run_e07.py --out runs/RCX-E07/smoke --phase all --smoke --threads 2
  python -B experiments/RCX-E07/run_e07.py --out runs/RCX-E07/main --phase all
Phases: data, train, e07, e08, report ('all' = in order). Each phase is skipped when its outputs exist (resume by
re-running the same command); the run's config is frozen in <out>/config.json at first use and a mismatch aborts.
Determinism: fixed seeds throughout; float results are reproducible for the same --threads value.
See experiments/RCX-E07/README.md for the design, outputs and the decisions taken where the design was ambiguous.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.cuda.is_available = lambda: False  # CPU-only recipe (RCWM I-21)
torch.cuda.device_count = lambda: 0

from rcx import stats as ST  # noqa: E402
from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import data as D  # noqa: E402
from rcx.bw import encode as EN  # noqa: E402
from rcx.bw import excl as EX  # noqa: E402
from rcx.bw import heuristics as H  # noqa: E402
from rcx.bw import models as M  # noqa: E402
from rcx.bw import search as S  # noqa: E402
from rcx.bw import space as SP  # noqa: E402
from rcx.guard import ThermalGuard  # noqa: E402

PHASES = ("data", "train", "e07", "e08", "report")
ARCHS = ("gnn", "mlp")
HANDCRAFTED = ("unmet_goal_facts", "misplaced_blocks", "misplaced_lb", "h_add", "h0")
E08_HEURS = ("gnn", "mlp", "unmet_goal_facts", "misplaced_blocks", "h_add", "h0")
CODE_FILES = ["experiments/RCX-E07/run_e07.py", "rcx/stats.py", "rcx/guard.py"] + \
    [f"rcx/bw/{m}.py" for m in ("__init__", "bwm_import", "excl", "space", "data", "encode", "models", "heuristics",
                                "search")]

FULL = dict(
    train_sizes=[4, 5, 6, 7], train_per_size=40, train_all_states_max_n=6, train_states_large=3000, goal_frac=0.5,
    val_sizes=[4, 5, 6, 7], val_per_size=10, val_states=300,
    id_sizes=[5, 6], ood_sizes=[8, 10, 12], test_per_size=15, k_states=60,
    discovery_draws=3000, min_class_count=5, val_class_frac=0.2, test_class_frac=0.25,
    space_max_n=8, astar_budget=300_000,
    seeds=[20261276, 20261277, 20261278], epochs=12, batch_size=64, lr=2e-3, time_cap_s=3600.0,
    e08_sizes=[8, 10, 12, 14, 15], e08_per_size=20, budgets=[500, 2000], h0_budgets=[500], opt_budget=300_000,
    tie_tol=1e-5, bootstrap_resamples=5000,
)
SMOKE = dict(FULL, train_sizes=[4, 5, 6], train_per_size=2, train_all_states_max_n=5, train_states_large=300,
             val_sizes=[4, 5, 6], val_per_size=1, val_states=50, id_sizes=[5], test_per_size=2, k_states=6,
             discovery_draws=600, astar_budget=60_000, seeds=[20261276], epochs=2, time_cap_s=25.0,
             e08_sizes=[8, 10], e08_per_size=2, budgets=[100, 300], h0_budgets=[100], opt_budget=20_000,
             bootstrap_resamples=200)
OVERRIDABLE = ("train_per_size", "val_per_size", "test_per_size", "k_states", "epochs", "time_cap_s", "astar_budget",
               "opt_budget", "e08_per_size", "budgets")


# ----------------------------------------------------------------------------- small utilities

def fs_json(fs):
    return sorted(map(list, fs))


def fs_from(x):
    return frozenset(map(tuple, x))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_jsonl_gz(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, separators=(",", ":")) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Ctx:
    def __init__(self, out: Path, cfg: dict, smoke: bool):
        self.out, self.cfg, self.smoke = out, cfg, smoke
        self.log_path = out / "log.txt"
        self.guard = ThermalGuard(log=self.log)
        self.last_guard = time.monotonic()
        self.guard_wait = 0.0

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def courtesy(self, force: bool = False, every_s: float = 30.0) -> float:
        """ThermalGuard check (reads the RCWM watchdog log; no subprocess) at most every `every_s` unless forced."""
        if not force and time.monotonic() - self.last_guard < every_s:
            return 0.0
        waited = self.guard.check()
        self.last_guard = time.monotonic()
        self.guard_wait += waited
        return waited


def size_of_key(gk: str) -> int:
    if not gk.startswith("chains:"):
        raise ValueError(f"unexpected goal key {gk!r} (generator goals are on-pair chains)")
    return int(gk.split(":")[1])


def load_instances(out):
    raw = json.loads((out / "data" / "instances.json").read_text(encoding="utf-8"))
    for it in raw:
        it["init"], it["goal"] = fs_from(it["init"]), fs_from(it["goal"])
    return {it["id"]: it for it in raw}


def successors_bwm(B, objs, s):
    return [(a, B.step(s, a)[0]) for a in B.ground_actions(objs) if B.applicable(s, a)]


# ----------------------------------------------------------------------------- phase: data

def allocate_classes(ctx, B, VL, ex):
    """Goal-class split per size: discovery draws (split seed) -> classes seen >= min_class_count -> seeded shuffle ->
    test (ID sizes only), val, train. A goal class (renaming class of the goal) lives in exactly one split."""
    cfg = ctx.cfg
    rng = random.Random(D.SEEDS["split"])
    alloc, disc = {}, {}
    for n in sorted(set(cfg["train_sizes"]) | set(cfg["val_sizes"]) | set(cfg["id_sizes"])):
        cnt = Counter()
        for _ in range(cfg["discovery_draws"]):
            g = B.generate(rng, n, full_goal_frac=cfg["goal_frac"])
            if g is not None and VL.invariant(*g) not in ex["invariants"]:
                cnt[D.goal_key(g[0], g[2])] += 1
        classes = sorted(k for k, v in cnt.items() if v >= cfg["min_class_count"])
        rng.shuffle(classes)
        C = len(classes)
        n_test = math.ceil(cfg["test_class_frac"] * C) if n in cfg["id_sizes"] else 0
        n_val = max(1, int(cfg["val_class_frac"] * C + 0.5)) if n in cfg["val_sizes"] else 0
        n_train = C - n_test - n_val if n in cfg["train_sizes"] else 0
        if n in cfg["train_sizes"] and n_train < 1:
            raise SystemExit(f"n={n}: {C} goal classes cannot feed train/val/test")
        alloc[n] = {"test": classes[:n_test], "val": classes[n_test:n_test + n_val],
                    "train": classes[n_test + n_val:n_test + n_val + n_train]}
        disc[n] = dict(sorted(cnt.items()))
    return alloc, disc


def label_test_instance(ctx, it, ex, stats):
    cfg, n, objs, goal = ctx.cfg, it["n"], it["objs"], it["goal"]
    rng = random.Random(f"{D.SEEDS['state_sampling']}:{it['id']}")
    start = SP.from_facts(it["init"], objs)
    cand = [start]
    for _ in range(cfg["k_states"]):
        pos = start
        for _ in range(rng.randint(0, 2 * n)):
            pos = rng.choice(SP.successors(pos))[1]
        cand.append(pos)
    states, seen = [], set()
    for p in cand:
        if p not in seen:
            seen.add(p)
            states.append(p)
    stats["test_states_sampled"] += len(cand)
    stats["test_states_deduped"] += len(states)
    gl = SP.GoalLabels(objs, goal) if n <= cfg["space_max_n"] else None
    rows = []
    for p in states:
        s = SP.to_facts(p, objs)
        why = EX.row_exclusion_reason(objs, s, goal, ex)
        if why:
            stats[f"test_rows_excluded_{why}"] += 1
            continue
        t0 = time.perf_counter()
        if gl is not None:
            i = gl.sid(s)
            h, opt, exp, how = gl.h_id(i), gl.opt_actions_id(i), None, "space"
        else:
            h, opt, exp = SP.astar_optimal(objs, s, goal, cfg["astar_budget"])
            how = "astar"
        dt = time.perf_counter() - t0
        miss = None if h is not None else "astar_budget"
        stats["test_rows_missing"] += miss is not None
        rows.append({"i": it["id"], "n": n, "s": fs_json(s), "h": h, "opt": None if opt is None else
                     [list(a) for a in opt], "miss": miss, "how": how, "exp": exp, "t": round(dt, 5)})
        ctx.courtesy()
    return rows


def phase_data(ctx):
    cfg, out = ctx.cfg, ctx.out
    dd = out / "data"
    if (dd / "meta.json").exists():
        ctx.log("data: outputs exist, skipping")
        return
    dd.mkdir(parents=True, exist_ok=True)
    B, VL = BI.load()
    ex = EX.load_exclusions()
    t0 = time.perf_counter()
    alloc, disc = allocate_classes(ctx, B, VL, ex)
    cls = {sp: {k for n in alloc for k in alloc[n][sp]} for sp in ("train", "val", "test")}
    ctx.log("data: goal classes per size " + json.dumps({n: {k: len(v) for k, v in a.items()} for n, a in
                                                         alloc.items()}))
    seen = set()
    gen = {}
    sizes = [n for n in cfg["train_sizes"] for _ in range(min(cfg["train_per_size"], len(alloc[n]["train"])))]
    train, gen["train"] = D.fresh_instances(D.SEEDS["train_instances"], sizes, full_goal_frac=cfg["goal_frac"],
                                            seen=seen, ex=ex, prefix="train", unique="goal",
                                            accept=cls["train"].__contains__, strict=False)
    sizes = [n for n in cfg["val_sizes"] for _ in range(min(cfg["val_per_size"], len(alloc[n]["val"])))]
    val, gen["val"] = D.fresh_instances(D.SEEDS["val_instances"], sizes, full_goal_frac=cfg["goal_frac"], seen=seen,
                                        ex=ex, prefix="val", unique="goal", accept=cls["val"].__contains__,
                                        strict=False)
    sizes = [n for n in cfg["id_sizes"] + cfg["ood_sizes"] for _ in range(cfg["test_per_size"])]
    test, gen["test"] = D.fresh_instances(
        D.SEEDS["test_ood_instances"], sizes, full_goal_frac=cfg["goal_frac"], seen=seen, ex=ex, prefix="test",
        unique="instance", accept=lambda gk: size_of_key(gk) not in cfg["id_sizes"] or gk in cls["test"],
        strict=False)
    g_tr, g_va, g_te = ({it["goal_key"] for it in x} for x in (train, val, test))
    assert not (g_tr & g_va) and not (g_te & (g_tr | g_va)), "goal classes overlap across splits"
    ctx.log(f"data: instances train {len(train)} val {len(val)} test {len(test)}; generation {gen}")
    stats = {"rows": {}, "build": {}}
    rows = {"train": [], "val": [], "test": []}
    label_t = defaultdict(float)
    for split, inst in (("train", train), ("val", val)):
        for n in sorted({it["n"] for it in inst}):
            part = [it for it in inst if it["n"] == n]
            if split == "val":
                k = cfg["val_states"]
            else:
                k = None if n <= cfg["train_all_states_max_n"] else cfg["train_states_large"]
            t1 = time.perf_counter()
            r, st = D.build_dataset(part, k, with_opt=False, ex=ex)
            label_t[f"{split}:{n}"] = time.perf_counter() - t1
            stats["build"][f"{split}:{n}"] = st
            rows[split] += [{"i": x["inst"], "n": n, "s": fs_json(x["state"]), "h": x["h"]} for x in r]
            ctx.courtesy()
    tstats = Counter()
    for it in test:
        t1 = time.perf_counter()
        rows["test"] += label_test_instance(ctx, it, ex, tstats)
        label_t[f"test:{it['n']}"] += time.perf_counter() - t1
    stats["test_labelling"] = dict(tstats)
    for split, rr in rows.items():
        stats["rows"][split] = dict(sorted(Counter(r["n"] for r in rr).items()))
        write_jsonl_gz(dd / f"{split}.jsonl.gz", rr)
    all_inst = [dict(it, split=sp) for sp, inst in (("train", train), ("val", val), ("test", test)) for it in inst]
    write_json(dd / "instances.json", [{"id": it["id"], "split": it["split"], "n": it["n"], "objs": it["objs"],
                                        "init": fs_json(it["init"]), "goal": fs_json(it["goal"]),
                                        "goal_key": it["goal_key"]} for it in all_inst])
    per_state = defaultdict(list)
    for r in rows["test"]:
        per_state[r["n"]].append(r["t"])
    meta = {"alloc": {str(n): a for n, a in alloc.items()}, "discovery": {str(n): d for n, d in disc.items()},
            "generation": gen, "instances": {sp: dict(sorted(Counter(it["n"] for it in x).items()))
                                             for sp, x in (("train", train), ("val", val), ("test", test))},
            "stats": stats, "label_seconds": dict(label_t),
            "test_label_seconds_per_state": {str(n): float(np.mean(v)) for n, v in sorted(per_state.items())},
            "test_label_seconds_max": {str(n): float(np.max(v)) for n, v in sorted(per_state.items())},
            "exclusions_sha256": sha_file(EX.EXCLUSIONS_PATH), "wall_s": time.perf_counter() - t0}
    write_json(dd / "meta.json", meta)
    ctx.log(f"data: rows {stats['rows']} test labelling {dict(tstats)} in {meta['wall_s']:.1f}s")


# ----------------------------------------------------------------------------- phase: train

def encode_rows(arch, rows, inst):
    out = []
    for r in rows:
        it = inst[r["i"]]
        s = fs_from(r["s"])
        out.append(EN.graph(s, it["goal"], it["objs"]) if arch == "gnn" else EN.flat(s, it["goal"], it["objs"]))
    return out


def new_model(arch):
    return M.RelGNN(M.RelGNNConfig()) if arch == "gnn" else M.FlatMLP(M.FlatMLPConfig())


def train_one(ctx, arch, seed, X, y, Xv, yv):
    cfg = ctx.cfg
    torch.manual_seed(seed)
    model = new_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    rng = random.Random(seed)
    yt = torch.tensor(y, dtype=torch.float32)
    best_mae, best_state, best_ep = math.inf, None, -1
    hist, step, cap_hit = [], 0, False
    t0, gw0 = time.perf_counter(), ctx.guard_wait
    for ep in range(cfg["epochs"]):
        perm = list(range(len(X)))
        rng.shuffle(perm)
        model.train()
        losses, te, gw_ep = [], time.perf_counter(), ctx.guard_wait
        for b0 in range(0, len(perm), cfg["batch_size"]):
            idx = perm[b0:b0 + cfg["batch_size"]]
            loss = F.huber_loss(model(model.collate([X[i] for i in idx])), yt[idx], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss))
            ctx.courtesy()
            if time.perf_counter() - t0 - (ctx.guard_wait - gw0) > cfg["time_cap_s"]:
                cap_hit = True
                break
        train_s = time.perf_counter() - te - (ctx.guard_wait - gw_ep)  # guard sleep excluded
        pv = M.predict(model, Xv)
        mae, rmse = float(np.mean(np.abs(pv - yv))), float(np.sqrt(np.mean((pv - yv) ** 2)))
        hist.append({"epoch": ep, "steps": step, "train_loss": float(np.mean(losses)), "val_mae": mae,
                     "val_rmse": rmse, "train_s": round(train_s, 3), "sec_per_step": train_s / max(1, len(losses))})
        ctx.log(f"train {arch} s{seed} ep{ep} steps {step} loss {np.mean(losses):.4f} val_mae {mae:.4f}")
        if mae < best_mae:
            best_mae, best_state, best_ep = mae, copy.deepcopy(model.state_dict()), ep
        ctx.courtesy(force=True)
        if cap_hit:
            ctx.log(f"train {arch} s{seed}: TIME CAP {cfg['time_cap_s']}s hit at step {step} (budget mismatch!)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.perf_counter() - t0
    return model, {"arch": arch, "seed": seed, "config": M.config_dict(model), "n_params": M.n_params(model),
                   "epochs_run": len(hist), "steps": step, "best_epoch": best_ep, "best_val_mae": best_mae,
                   "cap_hit": cap_hit, "wall_s": wall, "guard_wait_s": ctx.guard_wait - gw0,
                   "sec_per_step": float(np.mean([h["sec_per_step"] for h in hist])), "history": hist}


def phase_train(ctx):
    cfg, out = ctx.cfg, ctx.out
    md = out / "models"
    md.mkdir(exist_ok=True)
    todo = [(a, s) for a in ARCHS for s in cfg["seeds"] if not (md / f"{a}_s{s}.json").exists()]
    if not todo:
        ctx.log("train: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    inst = load_instances(out)
    tr, va = read_jsonl_gz(out / "data" / "train.jsonl.gz"), read_jsonl_gz(out / "data" / "val.jsonl.gz")
    y = np.array([r["h"] for r in tr], dtype=np.float32)
    yv = np.array([r["h"] for r in va], dtype=np.float32)
    for arch in ARCHS:
        seeds = [s for a, s in todo if a == arch]
        if not seeds:
            continue
        t1 = time.perf_counter()
        X, Xv = encode_rows(arch, tr, inst), encode_rows(arch, va, inst)
        enc_s = time.perf_counter() - t1
        ctx.log(f"train: encoded {len(X)} train / {len(Xv)} val rows for {arch} in {enc_s:.1f}s")
        for seed in seeds:
            model, info = train_one(ctx, arch, seed, X, y, Xv, yv)
            info["encode_s"] = enc_s
            info["train_rows"], info["val_rows"] = len(X), len(Xv)
            torch.save(model.state_dict(), md / f"{arch}_s{seed}.pt")
            write_json(md / f"{arch}_s{seed}.json", info)
            ctx.log(f"train: {arch} s{seed} params {info['n_params']} best ep {info['best_epoch']} "
                    f"val_mae {info['best_val_mae']:.4f} in {info['wall_s']:.1f}s")


def load_model(out, arch, seed):
    model = new_model(arch)
    model.load_state_dict(torch.load(out / "models" / f"{arch}_s{seed}.pt", weights_only=True))
    model.eval()
    return model


# ----------------------------------------------------------------------------- phase: e07

def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(spearmanr(a, b)[0])


def group_sizes(cfg):
    groups = {str(n): [n] for n in cfg["id_sizes"] + cfg["ood_sizes"]}
    groups["ID"], groups["OOD"] = list(cfg["id_sizes"]), list(cfg["ood_sizes"])
    return groups


def metric_block(rows, model, learned):
    lab = [r for r in rows if r["h"] is not None]
    pol = [r["pol"][model] for r in lab if r["pol"][model] is not None]
    ties = [r["tie"][model] for r in lab if r["pol"][model] is not None]
    out = {"n_states": len(lab), "n_policy_states": len(pol),
           "spearman": spearman([r["pred"][model] for r in lab], [r["h"] for r in lab]),
           "policy_acc": float(np.mean(pol)) if pol else None, "tie_rate": float(np.mean(ties)) if ties else None}
    if learned:
        err = np.array([r["pred"][model] - r["h"] for r in lab], float)
        out.update(mae=float(np.mean(np.abs(err))) if len(err) else None,
                   rmse=float(np.sqrt(np.mean(err ** 2))) if len(err) else None,
                   exact=float(np.mean([round(r["pred"][model]) == r["h"] for r in lab])) if lab else None)
    return out


def phase_e07(ctx):
    cfg, out = ctx.cfg, ctx.out
    if (out / "e07_metrics.json").exists():
        ctx.log("e07: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    t0 = time.perf_counter()
    B, _ = BI.load()
    inst = load_instances(out)
    test = read_jsonl_gz(out / "data" / "test.jsonl.gz")
    flat, rows = [], []  # flat: (inst_id, state) to score; rows: per test state with indices into flat
    for r in test:
        it = inst[r["i"]]
        s = fs_from(r["s"])
        succ = successors_bwm(B, it["objs"], s)
        base = len(flat)
        flat.append((r["i"], s))
        flat += [(r["i"], ns) for _, ns in succ]
        rows.append({"i": r["i"], "n": r["n"], "h": r["h"], "miss": r["miss"], "idx": base,
                     "succ_idx": list(range(base + 1, base + 1 + len(succ))), "succ_act": [a for a, _ in succ],
                     "opt": None if r["opt"] is None else {tuple(a) for a in r["opt"]}})
    ctx.log(f"e07: {len(rows)} test states, {len(flat)} states to score")
    scores, timing = {}, {}
    for arch in ARCHS:
        t1 = time.perf_counter()
        enc = [EN.graph(s, inst[i]["goal"], inst[i]["objs"]) if arch == "gnn" else
               EN.flat(s, inst[i]["goal"], inst[i]["objs"]) for i, s in flat]
        timing[f"encode_{arch}"] = time.perf_counter() - t1
        for seed in cfg["seeds"]:
            t1 = time.perf_counter()
            scores[f"{arch}_s{seed}"] = M.predict(load_model(out, arch, seed), enc).astype(float)
            timing[f"predict_{arch}_s{seed}"] = time.perf_counter() - t1
            ctx.courtesy()
    for name in HANDCRAFTED:
        fn = H.BASELINES[name]
        t1 = time.perf_counter()
        scores[name] = np.array([fn(s, inst[i]["goal"]) for i, s in flat], float)
        timing[f"score_{name}"] = time.perf_counter() - t1
        ctx.courtesy()
    models = list(scores)
    tol = cfg["tie_tol"]
    for r in rows:
        r["pred"], r["pol"], r["tie"] = {}, {}, {}
        for m in models:
            v = scores[m]
            r["pred"][m] = float(v[r["idx"]])
            if r["h"] is None or r["h"] == 0 or r["opt"] is None or not r["succ_idx"]:
                r["pol"][m], r["tie"][m] = None, None
                continue
            sv = v[r["succ_idx"]]
            lo = float(np.min(sv))
            mins = [j for j, x in enumerate(sv) if x <= lo + tol * max(1.0, abs(lo))]
            r["pol"][m] = int(all(r["succ_act"][j] in r["opt"] for j in mins))
            r["tie"][m] = len(mins) > 1
    groups = group_sizes(cfg)
    learned = {a: [f"{a}_s{s}" for s in cfg["seeds"]] for a in ARCHS}
    metrics = {"per_model": {}, "learned_mean": {}, "comparisons": {}, "missing_labels": {}, "timing": timing}
    for g, ns in groups.items():
        sel = [r for r in rows if r["n"] in ns]
        metrics["missing_labels"][g] = sum(r["miss"] is not None for r in sel)
        for m in models:
            metrics["per_model"].setdefault(m, {})[g] = metric_block(sel, m, m not in HANDCRAFTED)
        for a, ms in learned.items():
            blk = {}
            for key in ("mae", "rmse", "spearman", "policy_acc", "tie_rate", "exact"):
                vals = [metrics["per_model"][m][g][key] for m in ms if metrics["per_model"][m][g][key] is not None]
                blk[key] = {"mean": float(np.mean(vals)) if vals else None,
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "per_seed": vals}
            metrics["learned_mean"].setdefault(a, {})[g] = blk

    def arm(m, sel):
        d = defaultdict(list)
        for r in sel:
            if r["pol"][m] is not None:
                d[r["i"]].append(r["pol"][m])
        return d

    for g in [str(n) for n in cfg["ood_sizes"]] + ["OOD", "ID"]:
        sel = [r for r in rows if r["n"] in groups[g]]
        parents = sorted({r["i"] for r in sel if r["pol"][models[0]] is not None})
        best_hc = max(HANDCRAFTED, key=lambda m: metrics["per_model"][m][g]["policy_acc"] or -1)
        comp = {"best_handcrafted": best_hc, "n_instances": len(parents),
                "note": "context only (not in the design's comparison list)" if g == "ID" else ""}
        gnn = [arm(m, sel) for m in learned["gnn"]]
        for label, base in (("gnn_minus_mlp", [arm(m, sel) for m in learned["mlp"]]),
                            ("gnn_minus_best_handcrafted", [arm(best_hc, sel)])):
            try:
                comp[label] = ST.seed_parent_bootstrap(base, gnn, resamples=cfg["bootstrap_resamples"])
            except ValueError as e:
                comp[label] = {"error": str(e)}
        metrics["comparisons"][g] = comp
    write_jsonl_gz(out / "e07_scores.jsonl.gz", [{k: r[k] for k in ("i", "n", "h", "miss", "pred", "pol", "tie")}
                                                 for r in rows])
    metrics["wall_s"] = time.perf_counter() - t0
    metrics["n_test_states"] = len(rows)
    metrics["n_scored_states"] = len(flat)
    write_json(out / "e07_metrics.json", metrics)
    ctx.log(f"e07: done in {metrics['wall_s']:.1f}s")


# ----------------------------------------------------------------------------- phase: e08

def e08_instances(ctx, inst07):
    cfg, out = ctx.cfg, ctx.out
    path = out / "e08_instances.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for it in raw:
            it["init"], it["goal"] = fs_from(it["init"]), fs_from(it["goal"])
        return raw
    seen = {D.instance_key(it["objs"], it["init"], it["goal"]) for it in inst07.values()
            if it["split"] == "test" and it["n"] > 7}
    sizes = [n for n in cfg["e08_sizes"] for _ in range(cfg["e08_per_size"])]
    inst, st = D.fresh_instances(D.SEEDS["e08_instances"], sizes, full_goal_frac=1.0, seen=seen, prefix="e08")
    ctx.log(f"e08: {len(inst)} instances {st}")
    for it in inst:
        t1 = time.perf_counter()
        h, exp = SP.astar_hstar(it["objs"], it["init"], it["goal"], cfg["opt_budget"])
        it.update(opt=h, opt_expansions=exp, opt_s=time.perf_counter() - t1)
        ctx.courtesy()
    write_json(path, [dict(it, init=fs_json(it["init"]), goal=fs_json(it["goal"]), gen_stats=st) for it in inst])
    return inst


def phase_e08(ctx):
    cfg, out = ctx.cfg, ctx.out
    if (out / "e08_metrics.json").exists():
        ctx.log("e08: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    t0 = time.perf_counter()
    B, _ = BI.load()
    inst = e08_instances(ctx, load_instances(out))
    s0 = cfg["seeds"][0]
    heur = {"gnn": M.LearnedH(load_model(out, "gnn", s0)), "mlp": M.LearnedH(load_model(out, "mlp", s0))}
    heur.update({k: H.BASELINES[k] for k in E08_HEURS if k not in heur})
    n_max_mlp = M.FlatMLPConfig().n_max
    runs_path = out / "e08_runs.jsonl"
    done = {}
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["inst"], r["heur"], r["budget"])] = r
    with open(runs_path, "a", encoding="utf-8", newline="\n") as fh:
        for it in inst:
            for name in E08_HEURS:
                budgets = sorted(cfg["h0_budgets"] if name == "h0" else cfg["budgets"], reverse=True)
                for b in budgets:
                    key = (it["id"], name, b)
                    if key in done:
                        continue
                    base = {"inst": it["id"], "n": it["n"], "heur": name, "budget": b, "opt": it["opt"]}
                    if name == "mlp" and it["n"] > n_max_mlp:
                        rec = dict(base, na=f"n={it['n']} > FlatMLP n_max={n_max_mlp}")
                    else:
                        big = [done.get((it["id"], name, bb)) for bb in budgets if bb > b]
                        big = [x for x in big if x and x.get("found") and x["expansions"] <= b]
                        if big:  # GBFS is deterministic: a run that succeeded within b expansions is the b-run
                            rec = dict(big[0], budget=b, derived_from=big[0]["budget"])
                        else:
                            r = S.gbfs(it["objs"], it["init"], it["goal"], heur[name], max_expansions=b)
                            rec = dict(base, **{k: r[k] for k in ("success", "found", "plan_len", "expansions",
                                                                  "generated", "h_calls", "reason", "seconds")},
                                       plan=[" ".join(a) for a in r["plan"]])
                        rec["subopt"] = (rec["plan_len"] / it["opt"]) if rec.get("success") and it["opt"] else None
                    done[key] = rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    ctx.courtesy()
    metrics = e08_metrics(cfg, inst, done)
    metrics["wall_s"] = time.perf_counter() - t0
    metrics["opt_coverage"] = {str(n): f"{sum(1 for it in inst if it['n'] == n and it['opt'] is not None)}/"
                                       f"{sum(1 for it in inst if it['n'] == n)}" for n in cfg["e08_sizes"]}
    metrics["opt_seconds"] = {str(n): float(np.mean([it["opt_s"] for it in inst if it["n"] == n]))
                              for n in cfg["e08_sizes"]}
    metrics["opt_sec_per_expansion"] = (sum(it["opt_s"] for it in inst) /
                                        max(1, sum(it["opt_expansions"] for it in inst)))
    write_json(out / "e08_metrics.json", metrics)
    ctx.log(f"e08: done in {metrics['wall_s']:.1f}s")


def e08_metrics(cfg, inst, done):
    sizes = [str(n) for n in cfg["e08_sizes"]] + ["all"]
    res = {"per": {}, "mcnemar": {}}
    by_n = {it["id"]: it["n"] for it in inst}
    for name in E08_HEURS:
        for b in (cfg["h0_budgets"] if name == "h0" else cfg["budgets"]):
            for sz in sizes:
                recs = [done[(i, name, b)] for i in by_n if sz == "all" or str(by_n[i]) == sz]
                recs = [r for r in recs if "na" not in r]
                if not recs:
                    res["per"][f"{name}|{b}|{sz}"] = {"n": 0, "na": True}
                    continue
                k = sum(bool(r["success"]) for r in recs)
                succ = [r for r in recs if r["success"]]
                sub = [r["subopt"] for r in succ if r["subopt"] is not None]
                res["per"][f"{name}|{b}|{sz}"] = {
                    "n": len(recs), "success": k, "rate": k / len(recs), "wilson": ST.wilson(k, len(recs)),
                    "plan_len_mean": float(np.mean([r["plan_len"] for r in succ])) if succ else None,
                    "expansions_median": float(np.median([r["expansions"] for r in recs])),
                    "generated_mean": float(np.mean([r["generated"] for r in recs])),
                    "h_calls_mean": float(np.mean([r["h_calls"] for r in recs])),
                    "seconds_mean": float(np.mean([r["seconds"] for r in recs])),
                    "seconds_total": float(np.sum([r["seconds"] for r in recs if "derived_from" not in r])),
                    "subopt_mean": float(np.mean(sub)) if sub else None, "subopt_n": len(sub),
                    "invalid_found_plans": sum(1 for r in recs if r["found"] and not r["success"])}
    for b in cfg["budgets"]:
        for base in ("unmet_goal_facts", "misplaced_blocks"):
            for sz in sizes:
                ids = [i for i in by_n if sz == "all" or str(by_n[i]) == sz]
                bb = sum(1 for i in ids if done[(i, base, b)]["success"] and not done[(i, "gnn", b)]["success"])
                cc = sum(1 for i in ids if done[(i, "gnn", b)]["success"] and not done[(i, base, b)]["success"])
                res["mcnemar"][f"gnn_vs_{base}|{b}|{sz}"] = {"baseline_only": bb, "gnn_only": cc, "n": len(ids),
                                                            "p_exact": ST.mcnemar_exact(bb, cc)}
    return res


# ----------------------------------------------------------------------------- phase: report

def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def n_partitions(n: int) -> int:
    p = [1] + [0] * n
    for k in range(1, n + 1):
        for m in range(k, n + 1):
            p[m] += p[m - k]
    return p[n]


def estimate_full(ctx, meta, trains, m07, m08):
    """Extrapolate the FULL design's CPU time from this (smoke) run's unit costs. Indicative only."""
    full = FULL
    est = {}
    space_states = {n: SP.n_states_formula(n) for n in range(3, 9)}
    t_row = {}
    for key, secs in meta["label_seconds"].items():
        split, n = key.split(":")
        rows = meta["stats"]["rows"].get(split, {}).get(n) or meta["stats"]["rows"].get(split, {}).get(int(n))
        if split in ("train", "val") and rows:
            t_row[int(n)] = secs / rows
    per_row = max(t_row.values()) if t_row else 2e-4
    full_alloc = {}  # the full design's class allocation: every partition of n with a part >= 2 is a goal class
    for n in sorted(set(full["train_sizes"]) | set(full["val_sizes"])):
        C = n_partitions(n) - 1
        nt = math.ceil(full["test_class_frac"] * C) if n in full["id_sizes"] else 0
        nv = max(1, int(full["val_class_frac"] * C + 0.5))
        full_alloc[n] = {"test": nt, "val": nv, "train": C - nt - nv}
    est["full_class_allocation"] = full_alloc
    full_train_rows = sum(min(full["train_per_size"], full_alloc[n]["train"]) *
                          (space_states[n] if n <= full["train_all_states_max_n"] else full["train_states_large"])
                          for n in full["train_sizes"])
    full_val_rows = sum(min(full["val_per_size"], full_alloc[n]["val"]) * min(full["val_states"], space_states[n])
                        for n in full["val_sizes"])
    est["data_train_val_labels_s"] = (full_train_rows + full_val_rows) * per_row + 10.0
    tl = {int(k): v for k, v in meta["test_label_seconds_per_state"].items()}
    tmax = {int(k): v for k, v in meta["test_label_seconds_max"].items()}
    per_inst_states = 0.9 * (full["k_states"] + 1)
    est["data_test_labels_s"] = sum(full["test_per_size"] * per_inst_states * tl.get(n, max(tl.values()))
                                    for n in full["id_sizes"] + full["ood_sizes"]) + 3.0
    est["data_test_labels_note"] = (f"per-state A* seconds measured on few smoke states: {tl}; max {tmax}; the n=12 "
                                    f"cost is heavy-tailed (budget {full['astar_budget']} expansions caps it)")
    sps = {a: np.mean([t["sec_per_step"] for t in trains if t["arch"] == a]) for a in ARCHS}
    steps_full = full["epochs"] * math.ceil(full_train_rows / full["batch_size"])
    est["train_s"] = sum(sps[a] * steps_full * len(full["seeds"]) for a in ARCHS)
    est["train_steps_per_model"] = steps_full
    est["train_rows_full_upper"] = full_train_rows  # before row-level exclusions
    est["val_rows_full_upper"] = full_val_rows
    n_states_e07 = (len(full["id_sizes"]) + len(full["ood_sizes"])) * full["test_per_size"] * per_inst_states
    n_scored_full = n_states_e07 * m07["n_scored_states"] / max(1, m07["n_test_states"])
    tm, ns = m07["timing"], max(1, m07["n_scored_states"])
    per_state = (sum(v for k, v in tm.items() if k.startswith(("encode_", "score_"))) / ns +
                 sum(v for k, v in tm.items() if k.startswith("predict_")) / ns * len(full["seeds"]) /
                 len(ctx.cfg["seeds"]))
    est["e07_s"] = per_state * n_scored_full
    per_exp = {}
    for name in E08_HEURS:
        for key, v in m08["per"].items():
            hn, b, sz = key.split("|")
            if hn == name and sz != "all" and v.get("n"):
                per_exp.setdefault(name, []).append((int(sz), v["seconds_mean"] / max(1.0, v["expansions_median"])))
    upper = 0.0
    for name in E08_HEURS:
        if name not in per_exp:
            continue
        n_ref, t_ref = max(per_exp[name])
        for n in full["e08_sizes"]:
            if name == "mlp" and n > 12:
                continue
            budgets = full["h0_budgets"] if name == "h0" else full["budgets"]
            upper += full["e08_per_size"] * sum(budgets) * t_ref * (n / n_ref) ** 2
    est["e08_search_upper_s"] = upper
    est["e08_opt_astar_upper_s"] = (len(full["e08_sizes"]) * full["e08_per_size"] * full["opt_budget"] *
                                    m08["opt_sec_per_expansion"])
    est["total_expected_s"] = (est["data_train_val_labels_s"] + est["data_test_labels_s"] + est["train_s"] +
                               est["e07_s"] + 0.5 * est["e08_search_upper_s"] + 0.3 * est["e08_opt_astar_upper_s"])
    est["assumptions"] = ("expected: half of the E08 GBFS runs exhaust their budget, 30% of optimal-length A* runs "
                          "hit the budget; upper: every search exhausts its budget and test labelling costs 3x the "
                          "smoke per-state A* time; expansion cost scales with n^2 (ground actions)")
    est["total_upper_s"] = (est["data_train_val_labels_s"] + 3 * est["data_test_labels_s"] + est["train_s"] +
                            est["e07_s"] + est["e08_search_upper_s"] + est["e08_opt_astar_upper_s"])
    return est


def phase_report(ctx, timings):
    cfg, out = ctx.cfg, ctx.out
    meta = json.loads((out / "data" / "meta.json").read_text(encoding="utf-8"))
    trains = [json.loads((out / "models" / f"{a}_s{s}.json").read_text(encoding="utf-8"))
              for a in ARCHS for s in cfg["seeds"]]
    m07 = json.loads((out / "e07_metrics.json").read_text(encoding="utf-8"))
    m08 = json.loads((out / "e08_metrics.json").read_text(encoding="utf-8"))
    groups = list(group_sizes(cfg))
    L = [f"# RCX-E07/E08 report ({'SMOKE — plumbing check only, numbers are not results' if ctx.smoke else 'full'})",
         "", "CPU mechanism tier. A learned heuristic is an external specialist (tier B); no Rineya-core claim.",
         "Descriptive unless a frozen preregistration names the comparison.", ""]
    L += ["## Data", "", "| split | instances by n | rows by n |", "|---|---|---|"]
    for sp in ("train", "val", "test"):
        L.append(f"| {sp} | {meta['instances'][sp]} | {meta['stats']['rows'][sp]} |")
    L += ["", f"Goal-class allocation (classes per size): " + "; ".join(
        f"n={n}: train {len(a['train'])}, val {len(a['val'])}, test {len(a['test'])}" for n, a in meta["alloc"].items()),
        f"Instance generation filters: `{json.dumps(meta['generation'])}`",
        f"Row-level exclusions (train/val builds): `{json.dumps(meta['stats']['build'])}`",
        f"Test labelling: `{json.dumps(meta['stats']['test_labelling'])}`", ""]
    params = {a: next(t["n_params"] for t in trains if t["arch"] == a) for a in ARCHS}
    L += ["## Training", "", "| model | seed | params | epochs | steps | best epoch | best val MAE | cap hit | wall s |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in trains:
        L.append(f"| {t['arch']} | {t['seed']} | {t['n_params']} | {t['epochs_run']} | {t['steps']} | "
                 f"{t['best_epoch']} | {t['best_val_mae']:.4f} | {t['cap_hit']} | {t['wall_s']:.1f} |")
    L += ["", f"Parameters: RelGNN {params['gnn']}, FlatMLP {params['mlp']}.", ""]
    hdr = "| model | " + " | ".join(groups) + " |"
    sep = "|---|" + "---|" * len(groups)
    for key, title, learned_only in (("policy_acc", "Policy accuracy (argmin successor optimal; ties strict)", False),
                                     ("spearman", "Spearman rank correlation with h*", False),
                                     ("tie_rate", "Tie rate among argmin successors", False),
                                     ("mae", "MAE vs h*", True), ("rmse", "RMSE vs h*", True),
                                     ("exact", "Calibrated exact (round(pred) == h*)", True)):
        L += [f"## E07 — {title}", "", hdr, sep]
        for a in ARCHS:
            cells = []
            for g in groups:
                blk = m07["learned_mean"][a][g][key]
                cells.append(fmt(blk["mean"]) + (f" ± {blk['sd']:.3f}" if blk["sd"] is not None else ""))
            L.append(f"| {a.upper()} (mean ± sd over {len(cfg['seeds'])} seeds) | " + " | ".join(cells) + " |")
        if not learned_only:
            for m in HANDCRAFTED:
                L.append(f"| {m} | " + " | ".join(fmt(m07["per_model"][m][g][key]) for g in groups) + " |")
        L.append("")
    L += [f"States per group (labelled / policy-evaluable): " + ", ".join(
        f"{g}: {m07['per_model']['h0'][g]['n_states']}/{m07['per_model']['h0'][g]['n_policy_states']}"
        for g in groups), f"Missing labels (A* budget): {m07['missing_labels']}", ""]
    L += ["## E07 — paired policy-accuracy differences (instance-clustered bootstrap, seeds averaged within instance)",
          "", "| scope | comparison | baseline | mean base | mean GNN | diff | 95% CI | instances |",
          "|---|---|---|---|---|---|---|---|"]
    for g, comp in m07["comparisons"].items():
        for label in ("gnn_minus_mlp", "gnn_minus_best_handcrafted"):
            c = comp[label]
            base = "MLP" if label == "gnn_minus_mlp" else comp["best_handcrafted"]
            if "error" in c:
                L.append(f"| {g} | {label} | {base} | – | – | – | {c['error']} | {comp['n_instances']} |")
            else:
                L.append(f"| {g}{' (context)' if g == 'ID' else ''} | {label} | {base} | {c['mean_a']:.3f} | "
                         f"{c['mean_b']:.3f} | {c['diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
                         f"{c['n_parents']} |")
    L += ["", "Instance-macro means (each instance weighted equally) differ from the state-pooled table above.", ""]
    sizes = [str(n) for n in cfg["e08_sizes"]] + ["all"]
    for b in cfg["budgets"]:
        L += [f"## E08 — GBFS success, max_expansions = {b}", "",
              "| heuristic | " + " | ".join(sizes) + " | rate [Wilson 95%] | median exp | mean plan len | "
              "mean subopt | mean s |", "|---|" + "---|" * (len(sizes) + 5)]
        for name in E08_HEURS:
            bb = b if name != "h0" else (b if b in cfg["h0_budgets"] else None)
            if bb is None:
                continue
            cells = []
            for sz in sizes:
                v = m08["per"][f"{name}|{b}|{sz}"]
                cells.append("n/a" if v.get("na") else f"{v['success']}/{v['n']}")
            a = m08["per"][f"{name}|{b}|all"]
            if a.get("na"):
                continue
            L.append(f"| {name} | " + " | ".join(cells) + f" | {a['rate']:.3f} [{a['wilson'][0]:.3f}, "
                     f"{a['wilson'][1]:.3f}] | {a['expansions_median']:.0f} | {fmt(a['plan_len_mean'], 1)} | "
                     f"{fmt(a['subopt_mean'], 3)} (n={a['subopt_n']}) | {a['seconds_mean']:.3f} |")
        L.append("")
    L += ["## E08 — paired success, exact McNemar (GNN seed-1 vs baseline)", "",
          "| comparison | budget | size | baseline only | GNN only | n | p |", "|---|---|---|---|---|---|---|"]
    for key, v in m08["mcnemar"].items():
        comp, b, sz = key.split("|")
        L.append(f"| {comp} | {b} | {sz} | {v['baseline_only']} | {v['gnn_only']} | {v['n']} | {v['p_exact']:.4f} |")
    L += ["", f"Optimal-length coverage (A* within {cfg['opt_budget']} expansions): {m08['opt_coverage']}; "
          f"mean A* seconds {json.dumps({k: round(v, 3) for k, v in m08['opt_seconds'].items()})}.",
          "FlatMLP cannot encode n > 12 (n_max = 12): its 14- and 15-block cells are n/a by construction.", ""]
    shas = {f: sha_file(ROOT / f) for f in CODE_FILES}
    prov = {"code_sha256": shas, "exclusions_json_sha256": sha_file(EX.EXCLUSIONS_PATH), "bwm_pins": BI.PINS,
            "seeds": {**D.SEEDS, "model_seeds": cfg["seeds"]}, "config": cfg, "smoke": ctx.smoke,
            "threads": ctx.threads, "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "wall_s": timings,
            "guard_wait_s": ctx.guard_wait, "bwm_bytecode_check": BI.check_no_bytecode()}
    L += ["## Provenance", "", "```json", json.dumps(prov, indent=1), "```", ""]
    est = None
    if ctx.smoke:
        est = estimate_full(ctx, meta, trains, m07, m08)
        L += ["## Estimated full-run CPU time (extrapolated from this smoke run; indicative)", "", "```json",
              json.dumps(est, indent=1), "```", ""]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    write_json(out / "metrics.json", {"e07": m07, "e08": m08, "train": trains, "data": meta, "provenance": prov,
                                      "estimate_full": est})
    ctx.log("report: wrote REPORT.md and metrics.json")
    return est


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir, e.g. runs/RCX-E07/main")
    ap.add_argument("--phase", choices=PHASES + ("all",), default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny configuration for a < 2 min plumbing check")
    ap.add_argument("--seeds", default=None, help="comma-separated model seeds (default 20261276,20261277,20261278)")
    ap.add_argument("--threads", type=int, default=4)
    for k in OVERRIDABLE:
        ap.add_argument("--" + k.replace("_", "-"), default=None)
    a = ap.parse_args()
    cfg = copy.deepcopy(SMOKE if a.smoke else FULL)
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    for k in OVERRIDABLE:
        v = getattr(a, k)
        if v is not None:
            cfg[k] = [int(x) for x in v.split(",")] if k == "budgets" else type(FULL[k])(float(v) if "." in v else
                                                                                          int(v))
    for s in cfg["seeds"]:
        if not D.SEED_RANGE[0] <= s <= D.SEED_RANGE[1]:
            raise SystemExit(f"model seed {s} outside {D.SEED_RANGE}")
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    if str(out.resolve()).startswith(str(BI.BUNDLE_ROOT.resolve())):
        raise SystemExit("refusing to write inside the BWM bundle")
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "config.json"
    frozen = {"cfg": cfg, "smoke": a.smoke}
    if cpath.exists():
        old = json.loads(cpath.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(frozen)):
            raise SystemExit(f"{cpath} differs from this invocation's config; use a new --out")
    else:
        write_json(cpath, frozen)
    ctx = Ctx(out, cfg, a.smoke)
    ctx.threads = a.threads
    torch.set_num_threads(a.threads)
    ctx.log(f"run_e07 start phase={a.phase} smoke={a.smoke} threads={a.threads} out={out}")
    tpath = out / "timings.json"
    timings = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    ctx.courtesy(force=True)
    for ph in (PHASES if a.phase == "all" else (a.phase,)):
        t0 = time.perf_counter()
        if ph == "report":
            est = phase_report(ctx, timings)
            if est:
                ctx.log(f"estimate (full design): expected ~{est['total_expected_s'] / 3600:.2f} h, upper "
                        f"~{est['total_upper_s'] / 3600:.2f} h CPU")
        else:
            {"data": phase_data, "train": phase_train, "e07": phase_e07, "e08": phase_e08}[ph](ctx)
        dt = time.perf_counter() - t0
        if dt > 1.0 or ph not in timings:
            timings[ph] = round(dt, 2)
        write_json(tpath, timings)
    diff = BI.check_no_bytecode()
    if diff["new_pycache_dirs"] or diff["ours"]:
        ctx.log(f"WARNING BWM bytecode check: {diff}")
    ctx.log(f"run_e07 end; guard wait {ctx.guard_wait:.1f}s; timings {timings}")


if __name__ == "__main__":
    main()
