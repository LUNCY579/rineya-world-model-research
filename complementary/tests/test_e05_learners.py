"""Tests for RCX-E05 learned arms (experiments/RCX-E05/learners_e05.py; PREREG_E05_DRAFT.md §2.1 test table, §5,
§10.2 M2): BatchedMLP = independent models, per-model early stopping, FT's step-0 candidate, exact label-preserving
augmentations, checkpoint = shorter run, mid-batch / mid-epoch resume = uninterrupted run, SF-shuffled, safe-point unit
sizing, inference with a fixed composition, the CUDA-off patch.

Tiny networks (width 8-16) on n = 4 / 5 spaces, literal seeds 0-9, one torch thread; never reported as results.
Run: python -B -m unittest tests.test_e05_learners -v   (CPU, one thread; a few seconds)
"""
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["E05_TORCH_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", "RCX-E05", fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FAM = _load("families_e05_ut", "families_e05.py")
LE = _load("learners_e05_ut", "learners_e05.py")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

N = 4


def setup(n=N):
    sp = SP.get_space(n)
    return sp, SO.get_phi(sp), LE.features(sp)


def goals(n, k, seed, fam="TRAIN"):
    rng = random.Random(seed)
    return [(np.asarray(g["w"], dtype=np.float64), g["c"]) for g in (FAM.draw_goal(fam, rng, n) for _ in range(k))]


def label_data(sp, phi, gs, per, seed):
    rng = np.random.default_rng(seed)
    return LE.label_rows(sp, phi, gs, [rng.choice(len(sp), per, replace=False) for _ in gs])


def same_weights(a, b, tol=0.0):
    da, db = a.state_dict(), b.state_dict()
    return da.keys() == db.keys() and all(float((da[k] - db[k]).abs().max()) <= tol for k in da)


def pg_jobs(sp, phi, gs, k, seed0=0, with_psi=False):
    rng = np.random.default_rng(seed0)
    out = []
    for i, (w, c) in enumerate(gs):
        tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
        idx = rng.choice(len(sp), k, replace=False)
        j = dict(key=f"lin|free|c-2|{i}:{seed0}", states=idx, v=tab.V[idx], stab=FAM.stabilizer(w.astype(np.int64), sp.n),
                 seed=seed0 + i, w=w, c=c)
        if with_psi:
            j["psi"] = SO.psi(phi, tab, idx)
        out.append(j)
    return out


class Counter:
    """safe_point callback: records kinds; returns True on the calls listed in `stop_at` (1-based)."""

    def __init__(self, stop_at=()):
        self.kinds, self.stop_at = [], set(stop_at)

    def __call__(self, kind):
        self.kinds.append(kind)
        return len(self.kinds) in self.stop_at


class TestInputs(unittest.TestCase):
    def test_state_features(self):
        sp, phi, X = setup()
        n, fi = sp.n, SO.fact_index(sp.n)
        self.assertEqual(X.shape, (len(sp), LE.state_dim(n)))
        for i in range(n):
            self.assertTrue(np.array_equal(X[:, i * n + i], np.zeros(len(sp))))
            self.assertTrue(np.array_equal(X[:, n * n + i], phi[:, fi[("ontable", i)]]))
            self.assertTrue(np.array_equal(X[:, n * n + n + i], (sp.states[:, i] == n + 1).astype(np.float32)))
            self.assertTrue(np.array_equal(X[:, n * n + 2 * n + i], phi[:, fi[("clear", i)]]))
            for j in range(n):
                if i != j:
                    self.assertTrue(np.array_equal(X[:, i * n + j], phi[:, fi[("on", i, j)]]))
        self.assertIs(LE.features(sp), X)

    def test_param_counts_n7(self):
        """§5.1: SF 180,537; UVFA 166,145; UVFA+aux 180,794; PGV 151,553 (w256), 43,009 (w128), 13,313 (w64)."""
        cnt = {k: LE.n_params(LE.MLP(*LE.arm_dims(k, 7))) for k in ("sf", "uvfa", "aux", "pgv")}
        self.assertEqual(cnt, dict(sf=180_537, uvfa=166_145, aux=180_794, pgv=151_553))
        self.assertEqual([LE.n_params(LE.MLP(*LE.arm_dims("pgv", 7), w)) for w in (128, 64)], [43_009, 13_313])
        self.assertEqual((LE.state_dim(7), LE.n_facts(7)), (70, 56))

    def test_renaming_gathers_and_stabilizer(self):
        sp, phi, X = setup()
        n, rng = sp.n, random.Random(0)
        Gs, Gf = LE.gathers(n)
        for r in rng.sample(range(len(LE.perms(n))), 6):
            sig = LE.perms(n)[r]
            sp_perm = SO.state_perm(sp, sig)
            self.assertTrue(np.array_equal(X[sp_perm], X[:, Gs[r]]))
            self.assertTrue(np.array_equal(phi[sp_perm], phi[:, Gf[r]]))
        self.assertTrue(np.array_equal(LE.perms(5), FAM.perms(5)))
        for w, _ in goals(n, 8, 1) + goals(5, 8, 2, "H-NEG"):
            m = int(round((np.sqrt(1 + 4 * len(w)) - 1) / 2))
            ids = LE.stab_ids(w, m)
            self.assertTrue(np.array_equal(ids, LE.perm_ids(FAM.stabilizer(w, m), m)))
            for r in ids:
                self.assertTrue(np.array_equal(SO.rename_weight(w, LE.perms(m)[r]), w))


class TestAugmentation(unittest.TestCase):
    """Augmentations preserve labels exactly."""

    def test_rename_preserves_labels(self):
        sp, phi, X = setup()
        n, D = sp.n, LE.state_dim(sp.n)
        gs = goals(n, 3, 3)
        rows = label_data(sp, phi, gs, 20, 0)
        codes = LE.goal_codes(np.stack([w for w, _ in gs]), [c for _, c in gs])
        tr = LE.zero_shot_trainer("aux", X, n, codes, rows, 0, dict(epochs=1, lr=1e-3, width=8, batch=16), threads=1)
        idx = np.arange(len(rows["s"]))
        x, y, r = tr.rows(0, idx)
        self.assertGreater(len(set(r.tolist())), 5)
        for i in idx:
            sig, (w, c), s = LE.perms(n)[r[i]], gs[rows["g"][i]], rows["s"][i]
            sp_perm, ws = SO.state_perm(sp, sig), SO.rename_weight(w, sig)
            self.assertTrue(np.array_equal(x[i, :D], X[sp_perm[s]]))
            self.assertTrue(np.array_equal(x[i, D:], LE.goal_code(ws, c)))
            self.assertEqual(y["v"][i], rows["v"][i])
            psi_r = y["psi"][i].astype(np.float64)
            self.assertEqual(SO.wbar_dot(psi_r, ws, c), rows["v"][i])  # exact
            tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
            tab_s = SO.sf_table(sp, SO.stop_values(phi, ws), c)
            self.assertEqual(tab_s.V[sp_perm[s]], rows["v"][i])  # V*_{sigma w}(sigma s) = V*_w(s)
            self.assertTrue(np.array_equal(psi_r[:-1], phi[sp_perm[tab.end[s]]]))  # renamed endpoint

    def test_stab_preserves_labels(self):
        sp, phi, X = setup()
        n, D = sp.n, LE.state_dim(sp.n)
        w = np.zeros(LE.n_facts(n))
        w[SO.fact_index(n)[("on", 0, 1)]] = 2  # mentions 2 blocks: |Stab| = 2! = 2
        tab = SO.sf_table(sp, SO.stop_values(phi, w), 0.25)
        ids = LE.perm_ids(FAM.stabilizer(w, n), n)
        self.assertEqual(len(ids), 2)
        idx = np.arange(len(sp))
        for kind, code in (("pgv", None), ("uvfa", LE.goal_code(w, 0.25))):
            task = LE.pergoal_task(tab, idx, kind, phi, 0.16, code, ids)  # run_e05's call: first round(0.16 k) = 20
            ref = LE.rows_task(idx, {"v": tab.V[idx]}, 20, code=code, stab=ids)
            self.assertTrue(np.array_equal(task.s_val, idx[:20]) and np.array_equal(task.s, idx[20:]))
            self.assertTrue(all(np.array_equal(task.y[k], ref.y[k]) and np.array_equal(task.y_val[k], ref.y_val[k])
                                for k in ("v",)) and task.aug == "stab" and set(task.y) == {"v"})
            tr = LE.Trainer(kind, [LE.new_model(kind, n, 0, 8)], [task], X, n, batch=8, lr=1e-3, wd=1e-4, steps=1,
                            order_seeds=[0], aug_seeds=[1], threads=1)
            rows = np.arange(len(task.s))
            x, y, r = tr.rows(0, rows)
            self.assertEqual(set(r.tolist()), set(ids.tolist()))
            for i in rows:
                s, sp_perm = task.s[i], SO.state_perm(sp, LE.perms(n)[r[i]])
                self.assertTrue(np.array_equal(x[i, :D], X[sp_perm[s]]))
                self.assertEqual(y["v"][i], task.y["v"][i])
                self.assertEqual(tab.V[sp_perm[s]], tab.V[s])
                if code is not None:
                    self.assertTrue(np.array_equal(x[i, D:], code))
            xv, _, rv = tr.rows(0, np.arange(20), val=True)  # validation rows are never augmented
            self.assertIsNone(rv)
            self.assertTrue(np.array_equal(xv[:, :D], X[task.s_val]))


def ref_train(kind, model, task, X, batch, lr, wd, steps, order_seed):
    """One plain nn.Module network trained with its own AdamW on the trainer's batch order (no augmentation)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    smp = LE.Sampler(len(task.s), batch, order_seed, "stream")
    for _ in range(steps):
        idx = smp.next()[0]
        x = X[task.s[idx]]
        if task.code is not None:
            x = np.concatenate([x, np.broadcast_to(task.code, (len(idx), len(task.code)))], axis=1)
        o = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        if kind == "pgv":
            loss = F.huber_loss(o[:, 0], torch.from_numpy(task.y["v"][idx]), delta=1.0)
        else:
            p = torch.from_numpy(task.y["psi"][idx])
            Fn = p.shape[1] - 1
            loss = (F.binary_cross_entropy_with_logits(o[:, :Fn], p[:, :Fn])
                    + F.huber_loss(F.softplus(o[:, Fn]) / 10, p[:, Fn] / 10, delta=1.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


class TestBatchedMLP(unittest.TestCase):
    def test_equals_independent_models(self):
        """B stacked models = B separately trained networks (same init and data order, 50 steps, < 1e-5)."""
        sp, phi, X = setup()
        n = sp.n
        LE.set_threads(1)
        for kind in ("pgv", "sf"):
            jobs = pg_jobs(sp, phi, goals(n, 3, 4), 40, with_psi=True)
            tasks = [LE.rows_task(j["states"], {"v": j["v"]} if kind == "pgv" else {"psi": j["psi"]}, 4,
                                     code=None if kind == "pgv" else LE.goal_code(j["w"], j["c"])) for j in jobs]
            models = [LE.new_model(kind, n, 10 + m, 16) for m in range(3)]
            bm = LE.BatchedMLP.from_models(models)
            xb = torch.from_numpy(np.stack([np.ascontiguousarray(
                np.concatenate([X[t.s[:5]]] + ([] if t.code is None else [np.broadcast_to(t.code, (5, len(t.code)))]),
                               axis=1), dtype=np.float32) for t in tasks]))
            with torch.no_grad():
                out = bm(xb)
                for m in range(3):
                    self.assertLess(float((out[m] - models[m](xb[m])).abs().max()), 1e-6)
            tr = LE.Trainer(kind, models, tasks, X, n, batch=8, lr=3e-3, wd=1e-4, steps=50, order_seeds=[5, 6, 7],
                            aug_seeds=[0, 1, 2], threads=1)
            self.assertEqual(tr.run(), "done")
            got = tr.current()
            for m in range(3):
                ref = ref_train(kind, LE.new_model(kind, n, 10 + m, 16), tasks[m], X, 8, 3e-3, 1e-4, 50, 5 + m)
                self.assertFalse(same_weights(ref, LE.new_model(kind, n, 10 + m, 16), 1e-3))  # it trained
                self.assertTrue(same_weights(got[m], ref, 1e-5), (kind, m))

    def test_per_model_early_stopping(self):
        sp, phi, X = setup()
        n = sp.n
        jobs = pg_jobs(sp, phi, goals(n, 3, 5), 40)
        tasks = [LE.rows_task(j["states"], {"v": j["v"]}, 8) for j in jobs]

        def make(steps):
            return LE.Trainer("pgv", [LE.new_model("pgv", n, m, 16) for m in range(3)], tasks, X, n, batch=8, lr=1e-2,
                              wd=1e-4, steps=steps, order_seeds=[0, 1, 2], aug_seeds=[3, 4, 5], check_every=5,
                              horizons=(20, 40), threads=1)
        tr = make(40)
        tr.run()
        self.assertEqual([t for t, _ in tr.val_hist], list(range(5, 41, 5)))
        for h in (20, 40):
            models, steps, losses = tr.best(h)
            for m in range(3):
                cand = [(L[m], t) for t, L in tr.val_hist if t <= h]
                best = min(cand)  # lowest loss, then the earliest step
                self.assertEqual((float(losses[m]), int(steps[m])), (best[0], best[1]))
                short = make(int(steps[m]))
                short.run()
                self.assertTrue(same_weights(models[m], short.current()[m]))
        self.assertTrue((tr.best(20)[1] <= 20).all() and (tr.best(40)[2] <= tr.best(20)[2]).all())


class TestFT(unittest.TestCase):
    def test_step0_candidate(self):
        """An FT run whose validation loss never improves returns the zero-shot UVFA weights exactly; without the step-0
        candidate it would not."""
        sp, phi, X = setup()
        n = sp.n
        LE.set_threads(1)
        (w, c), = goals(n, 1, 6)
        zs = LE.new_model("uvfa", n, 3, 16)
        rng = np.random.default_rng(0)
        states = rng.choice(len(sp), 50, replace=False)
        code = LE.goal_code(w, c)
        xv = np.ascontiguousarray(np.concatenate([X[states[:10]], np.broadcast_to(code, (10, len(code)))], axis=1),
                                  dtype=np.float32)
        with torch.no_grad():
            pred = LE.BatchedMLP.from_models([zs])(torch.from_numpy(xv[None]))[0, :, 0].numpy()
        v = np.concatenate([pred, 5.0 * rng.standard_normal(40).astype(np.float32)])
        job = dict(key="k", states=states, v=v, stab=FAM.stabilizer(w.astype(np.int64), n), seed=1, w=w, c=c)
        cfg = dict(steps=60, keep=60, lr=1e-2, batch=8, eval_every=5, val_n=10, step0=True)
        with tempfile.TemporaryDirectory() as d:
            init = LE.save_model(Path(d) / "uvfa.pt", zs, dict(kind="uvfa", n=n, arm="UVFA"))
            job["init"] = init
            res = LE.fit_pergoal("FT", sp, [job], cfg, out_dir=Path(d) / "ft", threads=1)
            ft = LE.load_model(res["k"])
            self.assertTrue(same_weights(ft.model, zs))
            self.assertEqual((ft.meta["best_step"], ft.meta["val_loss"], ft.kind), (0, 0.0, "uvfa"))
            mark = json.loads((Path(d) / "ft" / "batch0000.done.json").read_text(encoding="utf-8"))
            self.assertEqual(mark["val_hist"][0], [0, [0.0]])
            self.assertTrue(all(L[0] > 0.0 for _, L in mark["val_hist"][1:]))
            res2 = LE.fit_pergoal("FT", sp, [job], dict(cfg, step0=False), out_dir=Path(d) / "ft2", threads=1)
            ft2 = LE.load_model(res2["k"])
            self.assertGreater(ft2.meta["best_step"], 0)
            self.assertFalse(same_weights(ft2.model, zs))
            pv = LE.predict_value(ft, sp, states[:10], w[None], np.array([c]))[:, 0]
            self.assertLess(float(np.abs(pv - pred).max()), 1e-6)


    def test_step0_low_level(self):
        """The same through run_e05's calls: pergoal_trainer('uvfa', [task], ..., models=[copy of the UVFA]) with
        step0 and the stabilizer augmentation; best(h) is the zero-shot model at step 0 for every horizon."""
        sp, phi, X = setup()
        n = sp.n
        LE.set_threads(1)
        (w, c), = goals(n, 1, 19)
        zs = LE.new_model("uvfa", n, 8, 16)
        tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
        idx = np.random.default_rng(1).choice(len(sp), 50, replace=False)
        task = LE.pergoal_task(tab, idx, "uvfa", phi, 0.2, LE.goal_code(w, c), LE.stab_ids(w, n))
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate(
            [X[task.s_val], np.broadcast_to(task.code, (len(task.s_val), len(task.code)))], axis=1), dtype=np.float32))
        with torch.no_grad():
            task.y_val["v"] = LE.BatchedMLP.from_models([zs])(x[None])[0, :, 0].numpy()
        task.y["v"] = task.y["v"] + 3.0
        cfg = dict(lr=1e-2, wd=1e-4, batch=8, steps=30, check_every=5, horizons=(10, 30), step0=True)
        tr = LE.pergoal_trainer("uvfa", [task], X, n, cfg, models=[LE.copy.deepcopy(zs)], order_seeds=[1], aug_seeds=[2],
                                threads=1)
        self.assertEqual(tr.val_hist[0], [0, [0.0]])
        tr.run()
        for h in (10, 30):
            models, steps, losses = tr.best(h)
            self.assertEqual((int(steps[0]), float(losses[0])), (0, 0.0))
            self.assertTrue(same_weights(models[0], zs))
        self.assertFalse(same_weights(tr.current()[0], zs, 1e-4))


class TestCheckpoints(unittest.TestCase):
    def test_epoch_snapshot_equals_shorter_run(self):
        """The 2-epoch snapshot of a 4-epoch run = an independent 2-epoch run (SF and UVFA+aux)."""
        sp, phi, _ = setup()
        data = label_data(sp, phi, goals(sp.n, 4, 7), 24, 1)  # 96 rows, batch 32: 3 steps per epoch
        with tempfile.TemporaryDirectory() as d:
            for arm in ("SF", "UVFA+aux"):
                kw = dict(seed=2, lr=3e-3, width=16, batch=32, threads=1)
                long = LE.train_zero_shot(arm, sp, data, epochs=4, keep=(2, 4), out_dir=Path(d) / arm / "a", **kw)
                short = LE.train_zero_shot(arm, sp, data, epochs=2, keep=(2,), out_dir=Path(d) / arm / "b", **kw)
                self.assertEqual(set(long), {2, 4})
                a, b, e4 = LE.load_model(long[2]), LE.load_model(short[2]), LE.load_model(long[4])
                self.assertTrue(same_weights(a.model, b.model))
                self.assertFalse(same_weights(a.model, e4.model, 1e-4))
                self.assertEqual((a.meta["epoch"], a.meta["steps"], a.kind), (2, 6, LE.ARM_KIND[arm]))
                mtime = os.path.getmtime(long[4])  # a completed run is not recomputed
                self.assertEqual(LE.train_zero_shot(arm, sp, data, epochs=4, keep=(2, 4), out_dir=Path(d) / arm / "a",
                                                    **kw), long)
                self.assertEqual(os.path.getmtime(long[4]), mtime)
                with self.assertRaises(ValueError):
                    LE.train_zero_shot(arm, sp, data, epochs=4, keep=(2, 4), out_dir=Path(d) / arm / "a",
                                       **dict(kw, lr=1e-3))
            with self.assertRaises(ValueError):
                LE.train_zero_shot("SF", sp, data, seed=0, epochs=2, lr=1e-3, keep=(3,), out_dir=Path(d) / "x")
            per_row = dict(states=data["s"], goal=data["g"], psi=data["psi"], v=data["v"], W=data["W"][data["g"]],
                           c=data["c"][data["g"]])  # W, c per row (the other accepted layout): the same run
            kw = dict(seed=2, lr=3e-3, width=16, batch=32, threads=1)
            pr = LE.train_zero_shot("SF", sp, per_row, epochs=2, keep=(2,), out_dir=Path(d) / "pr", **kw)
            self.assertTrue(same_weights(LE.load_model(pr[2]).model, LE.load_model(
                LE.train_zero_shot("SF", sp, data, epochs=2, keep=(2,), out_dir=Path(d) / "SF" / "b", **kw)[2]).model))

    def test_mid_epoch_resume_zero_shot(self):
        sp, phi, _ = setup()
        data = label_data(sp, phi, goals(sp.n, 4, 8), 24, 2)
        kw = dict(seed=3, epochs=3, lr=3e-3, keep=(3,), width=16, batch=32, threads=1)
        with tempfile.TemporaryDirectory() as d:
            rec = Counter()
            ref = LE.train_zero_shot("UVFA", sp, data, out_dir=Path(d) / "ref", safe_point=rec, **kw)
            self.assertEqual(rec.kinds, ["step", "step", "epoch", "step", "step", "epoch", "step", "step"])
            sp4 = Counter(stop_at=(4,))  # after step 4: mid-epoch
            self.assertIsNone(LE.train_zero_shot("UVFA", sp, data, out_dir=Path(d) / "y", safe_point=sp4, **kw))
            self.assertTrue((Path(d) / "y" / "ckpt.pt").exists())
            got = LE.train_zero_shot("UVFA", sp, data, out_dir=Path(d) / "y", safe_point=Counter(), **kw)
            self.assertFalse((Path(d) / "y" / "ckpt.pt").exists())
            self.assertTrue(same_weights(LE.load_model(got[3]).model, LE.load_model(ref[3]).model, 1e-6))

    def test_mid_batch_resume_pergoal(self):
        """PGV with stabilizer augmentation, 5 jobs in BatchedMLP batches of 2: a yield mid-batch, then one at a batch
        end, then completion = the uninterrupted run (<= 1e-6); a changed config refuses the directory."""
        sp, phi, _ = setup()
        jobs = pg_jobs(sp, phi, goals(sp.n, 5, 9), 40, seed0=3)
        cfg = dict(width=16, steps=30, keep=30, lr=1e-2, batch=8, eval_every=5, val_n=8)
        with tempfile.TemporaryDirectory() as d:
            ref = LE.fit_pergoal("PGV", sp, jobs, cfg, out_dir=Path(d) / "ref", B=2, threads=1)
            rec = Counter(stop_at=(40,))  # batch 0: 29 step calls + 1 batch call; call 40 is mid-batch 1
            self.assertIsNone(LE.fit_pergoal("PGV", sp, jobs, cfg, out_dir=Path(d) / "y", safe_point=rec, B=2,
                                             threads=1))
            self.assertEqual((rec.kinds[29], rec.kinds[39]), ("batch", "step"))
            self.assertTrue((Path(d) / "y" / "batch0001.ckpt").exists())
            self.assertTrue((Path(d) / "y" / "batch0000.done.json").exists())
            at_batch_end = lambda kind: kind == "batch"  # noqa: E731
            self.assertIsNone(LE.fit_pergoal("PGV", sp, jobs, cfg, out_dir=Path(d) / "y", safe_point=at_batch_end, B=2,
                                             threads=1))
            self.assertFalse((Path(d) / "y" / "batch0001.ckpt").exists())
            got = LE.fit_pergoal("PGV", sp, jobs, cfg, out_dir=Path(d) / "y", safe_point=Counter(), B=2, threads=1)
            self.assertEqual(set(got), {j["key"] for j in jobs})
            for j in jobs:
                a, b = LE.load_model(got[j["key"]]), LE.load_model(ref[j["key"]])
                self.assertTrue(same_weights(a.model, b.model, 1e-6))
                self.assertEqual(a.meta["best_step"], b.meta["best_step"])
                self.assertEqual(a.meta["n_stab"], len(j["stab"]))
            with self.assertRaises(ValueError):
                LE.fit_pergoal("PGV", sp, jobs, dict(cfg, lr=3e-3), out_dir=Path(d) / "y", B=2, threads=1)

    def test_ft_and_sfft_resume(self):
        """FT (uvfa replicate, fixed code, step 0) and SF-FT (sf replicate, psi targets) resume exactly mid-batch."""
        sp, phi, _ = setup()
        n = sp.n
        gs = goals(n, 3, 10)
        with tempfile.TemporaryDirectory() as d:
            for kind, net in (("FT", "uvfa"), ("SF-FT", "sf")):
                init = LE.save_model(Path(d) / f"{net}.pt", LE.new_model(net, n, 4, 16), dict(kind=net, n=n, arm=net))
                jobs = [dict(j, init=init) for j in pg_jobs(sp, phi, gs, 40, seed0=7, with_psi=True)]
                cfg = dict(steps=20, keep=20, lr=3e-3, batch=8, eval_every=5, val_n=8, step0=True)
                ref = LE.fit_pergoal(kind, sp, jobs, cfg, out_dir=Path(d) / kind / "ref", B=2, threads=1)
                self.assertIsNone(LE.fit_pergoal(kind, sp, jobs, cfg, out_dir=Path(d) / kind / "y",
                                                 safe_point=Counter(stop_at=(7,)), B=2, threads=1))
                got = LE.fit_pergoal(kind, sp, jobs, cfg, out_dir=Path(d) / kind / "y", B=2, threads=1)
                for j in jobs:
                    a, b = LE.load_model(got[j["key"]]), LE.load_model(ref[j["key"]])
                    self.assertTrue(same_weights(a.model, b.model, 1e-6))
                    self.assertEqual(a.kind, net)
            with self.assertRaises(ValueError):  # FT needs uvfa replicates
                LE.fit_pergoal("FT", sp, [dict(jobs[0], init=init)], cfg, out_dir=Path(d) / "bad", threads=1)


class TestShuffle(unittest.TestCase):
    def test_sf_shuffled_permutes_within_goal_only(self):
        sp, phi, _ = setup()
        gs = goals(sp.n, 4, 11)
        rows = label_data(sp, phi, gs, 30, 3)
        sh = LE.shuffle_psi(rows, gs)
        self.assertTrue(np.array_equal(sh["s"], rows["s"]) and np.array_equal(sh["g"], rows["g"]))
        moved = 0
        for i, (w, c) in enumerate(gs):
            pos = np.flatnonzero(rows["g"] == i)
            a = sorted(map(tuple, rows["psi"][pos].tolist()))
            b = sorted(map(tuple, sh["psi"][pos].tolist()))
            self.assertEqual(a, b)  # a permutation of this goal's own rows
            moved += int((sh["psi"][pos] != rows["psi"][pos]).any(axis=1).sum())
            self.assertTrue(np.array_equal(sh["v"][pos], SO.wbar_dot(sh["psi"][pos], w, c)))
        self.assertGreater(moved, 30)
        again = LE.shuffle_psi(rows, gs)
        self.assertTrue(np.array_equal(again["psi"], sh["psi"]))  # stream 20261243:shuffle
        self.assertTrue(np.array_equal(LE.shuffle_psi(rows, gs, LE.derive(20261243, "shuffle"))["psi"], sh["psi"]))
        self.assertFalse(np.array_equal(LE.shuffle_psi(rows, gs, 0)["psi"], sh["psi"]))
        with tempfile.TemporaryDirectory() as d:
            kw = dict(seed=0, epochs=1, lr=1e-3, keep=(1,), width=8, batch=32, threads=1)
            a = LE.load_model(LE.train_zero_shot("SF-shuffled", sp, rows, out_dir=Path(d) / "a", **kw)[1])
            b = LE.load_model(LE.train_zero_shot("SF", sp, rows, out_dir=Path(d) / "b", **kw)[1])
            self.assertFalse(same_weights(a.model, b.model, 1e-7))


class TestUnits(unittest.TestCase):
    """M2: no uninterruptible unit > 30 s; checks <= 90 s apart at the planning rate, <= 120 s at half of it."""

    def test_plan_and_limits(self):
        plan = LE.plan_units(7, LE.PLAN_GFLOPS)
        self.assertEqual(LE.check_units(plan), plan)
        worst = max(plan.values())
        self.assertLessEqual(LE.check_gap(worst), 90.0)
        self.assertLessEqual(LE.check_gap(max(LE.plan_units(7, LE.PLAN_GFLOPS / 2).values())), 120.0)
        self.assertAlmostEqual(plan["step:pgv256:B64:b256"], 6 * 151_553 * 256 * 64 / 19e9)
        with self.assertRaises(ValueError):
            LE.check_units({"big": 30.5})
        for g in (0.05, 1.0, 19.0):
            B = LE.max_models(151_553, 256, g)
            self.assertLessEqual(LE.unit_seconds(LE.step_flops(151_553, 256, B), g), 30.0)
            if B < LE.B_MAX:
                self.assertGreater(LE.unit_seconds(LE.step_flops(151_553, 256, B + 1), g), 30.0)
        self.assertEqual(LE.max_models(151_553, 256, 1e-6), 0)
        sp, phi, _ = setup()
        jobs = pg_jobs(sp, phi, goals(sp.n, 2, 12), 40)
        cfg = dict(width=16, steps=10, keep=10, lr=1e-3, batch=8, eval_every=5, val_n=8)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):  # even B = 1 exceeds 30 s at this rate
                LE.fit_pergoal("PGV", sp, jobs, cfg, out_dir=d, gflops=1e-9, threads=1)
            with self.assertRaises(ValueError):
                LE.train_zero_shot("SF", sp, label_data(sp, phi, goals(sp.n, 2, 1), 16, 0), seed=0, epochs=1, lr=1e-3,
                                   keep=(1,), out_dir=d, width=8, batch=16, threads=1, gflops=1e-12)

    def test_check_spacing_with_injected_costs(self):
        """With injected per-step costs (a fake clock), Trainer.run's checks (>= 60 s apart, plus epoch ends) are
        <= 60 s + one unit apart: <= 90 s at the planning rate and <= 120 s at half of it."""
        sp, phi, X = setup()
        job = pg_jobs(sp, phi, goals(sp.n, 1, 13), 40)[0]
        worst = max(LE.plan_units(7, LE.PLAN_GFLOPS).values())
        for unit_s, bound in ((worst, 90.0), (2 * worst, 120.0), (30.0, 90.0), (60.0, 120.0)):
            calls = [0]

            def clock(unit_s=unit_s):  # one call at the start, then one after every step
                calls[0] += 1
                return (calls[0] - 1) * unit_s
            seen = []
            steps = int(400 / unit_s) + 2
            tr = LE.Trainer("pgv", [LE.new_model("pgv", sp.n, 0, 4)], [LE.rows_task(job["states"], {"v": job["v"]}, 8)],
                            X, sp.n, batch=4, lr=1e-3, wd=1e-4, steps=steps, order_seeds=[0], aug_seeds=[0], threads=1)
            self.assertEqual(tr.run(lambda t: seen.append(t.step) and False, clock=clock), "done")
            self.assertEqual(len(seen), len(tr.gaps))
            self.assertGreater(len(seen), 3)
            self.assertLessEqual(max(tr.gaps), min(LE.check_gap(unit_s), bound) + 1e-9)
            self.assertGreaterEqual(min(tr.gaps), LE.CHECK_MIN_S)

    def test_run_protocol_and_resume_low_level(self):
        """The protocol run_e05 uses: zero_shot_trainer(kind, X, n, codes, rows, seed, cfg) and Trainer.run(safe_point
        (trainer), clock, min_interval=60, ckpt): calls at epoch ends and >= 60 s apart, never after the final step; a
        yield writes the checkpoint and a new trainer + load() finishes identically to the uninterrupted run."""
        sp, phi, X = setup()
        n = sp.n
        gs = goals(n, 4, 16)
        rows = label_data(sp, phi, gs, 24, 5)  # 96 rows, batch 32: 3 steps per epoch, 4 epochs = 12 steps
        codes = LE.goal_codes(rows["W"], rows["c"])
        cfg = dict(width=16, batch=32, epochs=4, lr=3e-3, snap_epochs=(2, 4))

        def make():
            return LE.zero_shot_trainer("sf", X, n, codes, rows, 4, cfg, threads=1, fingerprint={"run": 1})

        def clock_of(unit):
            calls = [0]

            def clock():
                calls[0] += 1
                return (calls[0] - 1) * unit
            return clock
        ref = make()
        steps = []
        self.assertEqual(ref.run(lambda t: steps.append((t.step, t.epoch_end)) and False, clock=clock_of(35.0)), "done")
        self.assertEqual(steps, [(2, False), (3, True), (5, False), (6, True), (8, False), (9, True), (11, False)])
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d) / "c.pt"
            tr = make()
            self.assertEqual(tr.run(lambda t: t.step == 5, clock=clock_of(35.0), ckpt=ck), "yielded")  # mid-epoch
            self.assertEqual(tr.step, 5)
            again = make().load(ck)
            self.assertEqual(again.step, 5)
            self.assertEqual(again.run(clock=clock_of(35.0)), "done")
            for a, b in zip(again.current() + again.snapshot(6) + again.snapshot(12),
                            ref.current() + ref.snapshot(6) + ref.snapshot(12)):
                self.assertTrue(same_weights(a, b, 1e-6))
            other = LE.zero_shot_trainer("sf", X, n, codes, rows, 4, cfg, threads=1, fingerprint={"run": 2})
            with self.assertRaises(ValueError):
                other.load(ck)


class TestInference(unittest.TestCase):
    def test_predict_fixed_composition(self):
        sp, phi, X = setup()
        n = sp.n
        LE.set_threads(1)
        gs = goals(n, 3, 14)
        W, C = np.stack([w for w, _ in gs]), np.array([c for _, c in gs])
        codes = LE.goal_codes(W, C)
        with tempfile.TemporaryDirectory() as d:
            for kind in ("sf", "uvfa", "aux", "pgv"):
                net = LE.new_model(kind, n, 5, 16)
                lm = LE.load_model(LE.save_model(Path(d) / f"{kind}.pt", net, dict(kind=kind, n=n)))
                self.assertTrue(same_weights(lm.model, net))
                S = np.array([3, 17, 40, 3])
                with torch.no_grad():
                    for s in S[:3]:
                        x = X[s][None] if kind == "pgv" else np.concatenate([np.broadcast_to(X[s], (3, X.shape[1])),
                                                                             codes], axis=1)
                        direct = LE.outputs(kind, net(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))))
                        if kind in ("sf", "aux"):
                            got = LE.predict_psi(lm, sp, [s], W, C)[0]
                            self.assertLess(float(np.abs(got - direct["psi"]).max()), 1e-6)
                        if kind != "sf":
                            got = LE.predict_value(lm, sp, [s], W, C)[0]
                            self.assertLess(float(np.abs(got - direct["v"]).max()), 1e-6)
                fn = LE.predict_psi if kind in ("sf", "aux") else LE.predict_value
                full = fn(lm, sp, S, W, C)
                self.assertEqual(full.shape[:2], (4, 3))
                LE.clear_cache()
                rev = fn(lm, sp, S[::-1], W, C)[::-1]  # other order, fresh cache: bit-identical
                self.assertTrue(np.array_equal(full, rev))
                self.assertTrue(np.array_equal(fn(lm, sp, S[1:2], W, C)[0], full[1]))
                if kind in ("sf", "aux"):
                    self.assertTrue(((full[..., :-1] >= 0) & (full[..., :-1] <= 1)).all() and (full[..., -1] >= 0).all())
                if kind == "pgv":
                    self.assertTrue(np.array_equal(full, np.repeat(LE.predict_value(lm, sp, S), 3, axis=1)))
                if kind == "uvfa":
                    with self.assertRaises(ValueError):
                        LE.predict_psi(lm, sp, S, W, C)
                if kind == "sf":
                    with self.assertRaises(ValueError):
                        LE.predict_value(lm, sp, S, W, C)
            with self.assertRaises(ValueError):  # wrong n
                LE.predict_value(LE.load_model(Path(d) / "pgv.pt"), SP.get_space(5), [0])

    def test_evaluator_library_and_scores(self):
        """Evaluator (single-state forwards), LibraryPsi (memory = memmap with a yield and resume), sf_scores over the
        code sets of SF-GPI / SF-noGPI / SF-L / SF-GPI-5, value_scores and sf_values; run_e05's model-file format."""
        import gc
        sp, phi, X = setup()
        n = sp.n
        LE.set_threads(1)
        lib = goals(n, 5, 17)
        w, c = goals(n, 1, 18)[0]
        lib_codes = LE.goal_codes(np.stack([v for v, _ in lib]), [cc for _, cc in lib])
        sf, uv = LE.new_model("sf", n, 6, 16), LE.new_model("uvfa", n, 7, 16)
        own = LE.Evaluator(sf, "sf", X, LE.goal_code(w, c))
        S = np.array([0, 9, 33])
        with torch.no_grad():
            for s in S:
                x = np.concatenate([X[s], LE.goal_code(w, c)])[None].astype(np.float32)
                self.assertTrue(np.array_equal(own.psi([s])[0], LE.outputs("sf", sf(torch.from_numpy(x)))["psi"][0]))
        self.assertEqual(own.forwards, 3)
        own.psi(S)
        self.assertEqual(own.forwards, 3)  # cached
        mem = LE.LibraryPsi(sf, "sf", X, lib_codes, chunk=50)
        self.assertEqual(mem.build(min_interval=0.0), "done")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lib.npy"
            lp = LE.LibraryPsi(sf, "sf", X, lib_codes, path=path, chunk=50)
            self.assertEqual(lp.build(safe_point=lambda b: b.done == 2, min_interval=0.0), "yielded")
            lp.close()
            del lp
            gc.collect()
            lp = LE.LibraryPsi(sf, "sf", X, lib_codes, path=path, chunk=50)
            self.assertEqual(lp.done, 2)
            self.assertEqual(lp.build(min_interval=0.0), "done")
            self.assertTrue(np.array_equal(lp.rows(np.arange(len(sp))), mem.rows(np.arange(len(sp)))))
            self.assertEqual(json.loads((Path(d) / "lib.npy.json").read_text(encoding="utf-8"))["done"], 5)
            lp.close()
            del lp
            gc.collect()
        ref = mem.rows(S)
        self.assertEqual(ref.shape, (3, 5, LE.n_facts(n) + 1))
        self.assertLess(float(np.abs(ref[:, 1] - LE.Evaluator(sf, "sf", X, lib_codes[1]).psi(S)).max()), 2e-2)  # f16
        near = LE.nearest_codes(lib_codes, LE.goal_code(w, c), 4)
        for C, kw in ((5, dict(own=own, lib=mem)), (0, dict(own=own)), (5, dict(lib=mem)),
                      (4, dict(own=own, lib=mem, lib_idx=near))):
            got = LE.sf_scores(w, c, **kw)(0, None, S)
            parts = ([ref if "lib_idx" not in kw else ref[:, near]] if "lib" in kw else []) + (
                [own.psi(S)[:, None]] if "own" in kw else [])
            self.assertTrue(np.array_equal(got, SO.gpi_scores(np.concatenate(parts, axis=1), w, c)))
            self.assertEqual(sum(p.shape[1] for p in parts), C + ("own" in kw))
        with self.assertRaises(ValueError):
            LE.sf_scores(w, c)
        self.assertTrue(np.array_equal(LE.sf_values(own, S, w, c), SO.wbar_dot(own.psi(S), w, c)))
        ev = LE.Evaluator(uv, "uvfa", X, LE.goal_code(w, c))
        self.assertTrue(np.array_equal(LE.value_scores(ev, c)(0, None, S), ev.value(S) - c))
        with self.assertRaises(ValueError):
            LE.Evaluator(uv, "pgv", X, LE.goal_code(w, c))
        with tempfile.TemporaryDirectory() as d:  # run_e05's Adapters.save format {"kind", "dims", "state"}
            p = Path(d) / "uv.pt"
            torch.save(dict(kind="uvfa", dims=list(uv.dims), state=uv.state_dict()), p)
            lm = LE.load_model(p)
            self.assertEqual((lm.kind, lm.n), ("uvfa", n))
            self.assertLess(float(np.abs(LE.predict_value(lm, sp, S, w[None], [c])[:, 0] - ev.value(S)).max()), 1e-6)

    def test_label_rows_codes_nearest_and_timing(self):
        sp, phi, _ = setup()
        gs = goals(sp.n, 2, 15)
        rows = label_data(sp, phi, gs, 10, 4)
        for i, (w, c) in enumerate(gs):
            pos = rows["g"] == i
            self.assertTrue(np.array_equal(SO.wbar_dot(rows["psi"][pos], w, c), rows["v"][pos]))
            self.assertTrue(np.array_equal(rows["W"][i], w) and rows["c"][i] == c)
        self.assertEqual(rows["W"].shape, (2, LE.n_facts(sp.n)))
        self.assertEqual(LE.pergoal_seeds("pgv", "k", 400), tuple(LE.derive(20261249, f"pgv-{t}:k:400")
                                                                  for t in ("init", "order", "stab")))
        self.assertEqual(LE.pergoal_seeds("ft", "k", replicate=20261246),
                         (None, LE.derive(20261249, "ft-order:k:20261246"), LE.derive(20261249, "ft-stab:k:20261246")))
        z = LE.goal_code(np.array([3.0, 0.0, -3.0]), 0.25)
        self.assertTrue(np.array_equal(z, np.array([1.0, 0.0, -1.0, 0.25], dtype=np.float32)))
        L = np.array([[1, 0, 0, 0.25], [0, 1, 0, 0.25], [1, 0, 0, 0.25], [2, 0, 0, 0.5]], dtype=np.float64)
        self.assertEqual(LE.nearest_codes(L, np.array([1, 0, 0, 0.25]), 3).tolist(), [0, 2, 3])
        for model, B, batch in (("PGV64", 2, 8), ("SF", 1, 16), ("UVFA+aux", 2, 8)):
            self.assertGreater(LE.time_step(model, width=8, batch=batch, B=B, steps=1, n=sp.n, threads=1), 0.0)
        with self.assertRaises(ValueError):
            LE.set_threads(5)


class TestCudaOff(unittest.TestCase):
    def test_cuda_off(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertFalse(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 0)
        LE.cuda_off()
        self.assertFalse(torch.cuda.is_available())
        m = LE.new_model("sf", N, 0, 8)
        self.assertTrue(all(p.device.type == "cpu" for p in m.parameters()))
        self.assertEqual(LE.default_threads(), 1)


if __name__ == "__main__":
    unittest.main()
