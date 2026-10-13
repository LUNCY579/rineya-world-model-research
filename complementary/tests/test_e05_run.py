"""RCX-E05 runner (experiments/RCX-E05/run_e05.py; PREREG_E05_DRAFT.md §2, §8, §9, §10). Custody (digest formula and
scope, committed-sha refusal on a temporary git repository, create-only and keyed append-only outputs, ATTEMPTS and
PROVENANCE), §8 sizing against the prereg's power tables, the lane driver (a refused start writes nothing; yield and
resume), the VOID gate, and one end-to-end pass at n = 5 (pilot-a with a yield -> pilot-b -> pilot-report -> draw-test
-> oracle-test -> train -> pgv -> ft -> eval -> extras) with fake learners (no torch: exact oracle scores, a deliberately
weak selection UVFA that triggers the one UVFA extension) whose evals/*.json are read back through analyze_e05's
loaders. Temporary RINEYA_LANE_DIR / TEMP / runs / repo directories, an injected frozen clock, RAM, guard and process
scan; never reported as results.
Run: python -B -m unittest tests.test_e05_run -v   (CPU, one thread)
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rcx import lane as LN  # noqa: E402
from rcx.bw import soft as SO  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RU = _load("run_e05", ROOT / "experiments" / "RCX-E05" / "run_e05.py")
FA, LA = RU.FA, RU.LA
NOW = dt.datetime(2026, 10, 20, 12, 0)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Sandbox(unittest.TestCase):
    """Temporary lane dir, TEMP, runs dir and an injected lane environment (day mode, no window ahead)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.lane, self.temp, self.runs = base / "lane", base / "temp", base / "runs"
        for d in (self.lane, self.temp, self.runs):
            d.mkdir()
        self._env = {k: os.environ.get(k) for k in ("RINEYA_LANE_DIR", "TEMP")}
        os.environ["RINEYA_LANE_DIR"], os.environ["TEMP"] = str(self.lane), str(self.temp)
        win = base / "quiet_windows.json"
        win.write_text(json.dumps({"end_margin_min": 5, "windows": [["2026-10-12T23:00", "2026-10-13T07:00"]]}),
                       encoding="utf-8")
        paths = LA.Paths(windows=win, extra_windows=base / "extra_windows.json", train_log=base / "train.log",
                         post_log=base / "post.log", events=self.runs / "LANE_EVENTS.jsonl",
                         thermal_flag=base / ".thermal_pause")
        self.lenv = LA.Env(paths=paths, now=lambda: NOW, ram=lambda: 32.0, guard=lambda resuming: ([], None),
                           strays=lambda: [], pid=os.getpid())

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, capture_output=True, text=True, check=True)


class TestCustody(unittest.TestCase):
    def test_digest_formula_and_scope(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for rel in RU.NEW_T3 + tuple(RU.PINNED) + ("experiments/RCX-E05/lane_e05.py",):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(rel.encode())
            for stage, files in (("t2", RU.NEW_T2), ("t3", RU.NEW_T3)):
                rels = sorted(files + tuple(RU.PINNED))
                want = sha("".join(f"{r}\t{sha(r.encode())}\n" for r in rels).encode())
                cd = RU.code_digest(stage, root)
                self.assertEqual(cd["digest"], want)
                self.assertEqual(set(cd["files"]), set(rels))
                self.assertEqual(set(cd["pins_bad"]), set(RU.PINNED))  # temp content never matches the pins
            d3 = RU.code_digest("t3", root)["digest"]
            (root / "experiments/RCX-E05/lane_e05.py").write_bytes(b"changed")  # operational: outside the digest
            self.assertEqual(RU.code_digest("t3", root)["digest"], d3)
            d2 = RU.code_digest("t2", root)["digest"]
            (root / "experiments/RCX-E05/analyze_e05.py").write_bytes(b"changed")  # T3 only
            self.assertEqual(RU.code_digest("t2", root)["digest"], d2)
            self.assertNotEqual(RU.code_digest("t3", root)["digest"], d3)
            (root / "experiments/RCX-E05/run_e05.py").unlink()
            self.assertIsNone(RU.code_digest("t2", root)["digest"])

    def test_real_tree_pins_hold(self):
        for stage in ("t2", "t3"):
            cd = RU.code_digest(stage)
            self.assertEqual((cd["missing"], cd["pins_bad"]), ([], {}), stage)
            self.assertRegex(cd["digest"], r"^[0-9a-f]{64}$")
        self.assertNotIn("experiments/RCX-E05/lane_e05.py", RU.code_digest("t3")["files"])

    def test_committed_sha_refusal_uses_head_not_working_copy(self):
        a, b = sha(b"committed"), sha(b"working copy only")
        with tempfile.TemporaryDirectory() as t:
            git(t, "init", "-q")
            (Path(t) / "DECISIONS.md").write_text(f"# D\nmanifest {a}\n", encoding="utf-8")
            git(t, "add", "DECISIONS.md")
            git(t, "commit", "-q", "-m", "x")
            with open(Path(t) / "DECISIONS.md", "a", encoding="utf-8") as f:
                f.write(f"uncommitted {b}\n")
            text = RU.committed_decisions(Path(t))
            self.assertIn(a, text)
            self.assertNotIn(b, text)
            RU.require_committed(a, "x", text)
            for bad in (b, a[:63], a.upper(), "", None):
                with self.assertRaises(SystemExit):
                    RU.require_committed(bad, "x", text)
        with tempfile.TemporaryDirectory() as t:  # not a repository: refuse
            with self.assertRaises(SystemExit):
                RU.committed_decisions(Path(t))

    def test_create_only_and_keyed_append_only(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x" / "R.json"
            s = RU.write_create_only(p, {"a": 1})
            self.assertEqual(s, sha(p.read_bytes()))
            with self.assertRaises(FileExistsError):
                RU.write_create_only(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual([q.name for q in p.parent.iterdir()], ["R.json"])  # no temp file left
            lp = Path(t) / "log.jsonl"
            lg = RU.KeyedLog(lp)
            lg.add("k1", {"v": 1}, "D")
            lg.add_many([("k2", {"v": 2}), ("k3", {"v": 3})], "D")
            with self.assertRaises(RuntimeError):
                lg.add("k2", {"v": 9})
            with self.assertRaises(RuntimeError):
                lg.add_many([("k4", {}), ("k4", {})])
            with open(lp, "a", encoding="utf-8") as f:
                f.write('{"key": "torn"')  # a crash mid-line
            lg2 = RU.KeyedLog(lp)
            self.assertEqual(set(lg2.rows), {"k1", "k2", "k3"})
            self.assertEqual(lg2.digests(), {"D"})
            lg2.add("k5", {"v": 5})
            self.assertEqual(set(RU.KeyedLog(lp).rows), {"k1", "k2", "k3", "k5"})
            with open(lp, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": "k1"}) + "\n")
            with self.assertRaises(SystemExit):
                RU.KeyedLog(lp)

    def test_require_digest_and_prereg(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for rel in RU.NEW_T3 + tuple(RU.PINNED):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, root / rel)
            d2 = RU.code_digest("t2", root)["digest"]
            pr = root / "PREREG_E05.md"
            pr.write_text("frozen", encoding="utf-8")
            text = [""]
            ctx = RU.Ctx(runs=root / "runs", root=root, decisions=lambda: text[0], prereg=pr,
                         exclusions=lambda s: None, adapters=object())
            with self.assertRaises(SystemExit):
                RU.require_digest(ctx, "t2", d2)  # not committed yet
            text[0] = f"T2 {d2}"
            self.assertEqual(RU.require_digest(ctx, "t2", d2)["digest"], d2)
            (root / "experiments/RCX-E05/run_e05.py").write_text("# edited\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                RU.require_digest(ctx, "t2", d2)  # committed but the code changed
            ps = sha(pr.read_bytes())
            with self.assertRaises(SystemExit):
                RU.require_prereg(ctx, ps)
            text[0] += f" prereg {ps}"
            self.assertEqual(RU.require_prereg(ctx, ps), ps)
            pr.write_text("frozen + edit", encoding="utf-8")
            with self.assertRaises(SystemExit):
                RU.require_prereg(ctx, ps)
            shutil.copyfile(ROOT / "rcx/stats.py", root / "rcx/bw/space.py")  # a pinned file replaced
            self.assertIn("rcx/bw/space.py", RU.code_digest("t2", root)["pins_bad"])


class TestSizing(unittest.TestCase):
    def test_n_id_lookup(self):
        for s, n in ((0.05, 96), (0.138, 96), (0.15, 114), (0.16, 129), (0.17, 146), (0.174, 152), (0.3, 152)):
            self.assertEqual(RU.n_id_for(s), n, s)

    def test_power_tables(self):
        tol = 0.0051  # the prereg prints two decimals
        p1m = {0.15: (0.97, 0.98), 0.20: (0.89, 0.95), 0.25: (0.71, 0.84)}
        for s, (a, b) in p1m.items():
            self.assertAlmostEqual(RU.power_p1(s, 96, 56, 56), a, delta=tol)
            self.assertAlmostEqual(RU.power_p1(s, 152, 72, 72), b, delta=tol)
        p1id = {96: (0.97, 0.85, 0.58, 0.39), 120: (0.99, 0.92, 0.69, 0.48), 152: (0.99, 0.95, 0.80, 0.59)}
        low = {96: ((0.50, 0.39, 0.22, 0.14), (0.84, 0.64, 0.39, 0.25)),
               152: ((0.50, 0.50, 0.35, 0.22), (0.89, 0.79, 0.59, 0.39))}
        for n, row in p1id.items():
            for s, v in zip((0.10, 0.15, 0.20, 0.25), row):
                self.assertAlmostEqual(RU.power_p1(s, n, macro=False), v, delta=tol, msg=(n, s))
        for n, (r3, r4) in low.items():
            for s, v3, v4 in zip((0.10, 0.15, 0.20, 0.25), r3, r4):
                self.assertAlmostEqual(RU.power_p1(s, n, macro=False, effect=0.03), v3, delta=tol, msg=(n, s))
                self.assertAlmostEqual(RU.power_p1(s, n, macro=False, effect=0.04), v4, delta=tol, msg=(n, s))
        p2 = {0.15: (1.00, 1.00, 0.80, 0.90), 0.20: (0.94, 0.98, 0.55, 0.68), 0.25: (0.80, 0.90, 0.39, 0.49),
              0.30: (0.64, 0.77, 0.29, 0.37)}
        for s, (a, b, c, d) in p2.items():
            self.assertAlmostEqual(RU.power_p2(s, 96, 56, 56), a, delta=tol)
            self.assertAlmostEqual(RU.power_p2(s, 152, 72, 72), b, delta=tol)
            self.assertAlmostEqual(RU.power_p2(s, 96, 56, 56, d2=0.02), c, delta=tol)
            self.assertAlmostEqual(RU.power_p2(s, 152, 72, 72, d2=0.02), d, delta=tol)

    def test_size_plan_rules(self):
        pl = RU.size_plan(0.10, 0.10, 0.10, 0.10, p2_dropped=False)
        self.assertEqual((pl["n_id"], pl["n_k"], pl["n_neg"], pl["caps"]), (96, 56, 56, []))
        pl = RU.size_plan(0.10, 0.30, 0.10, 0.10, p2_dropped=False)  # P1-M power < 0.80 at 56 -> 72
        self.assertEqual(pl["n_k"], 72)
        pl = RU.size_plan(0.10, 0.10, 0.10, 0.35, p2_dropped=False)  # P2 power < 0.80 at 56 -> 72
        self.assertEqual(pl["n_k"], 72)
        self.assertEqual(RU.size_plan(0.10, 0.10, 0.10, 0.35, p2_dropped=True)["n_k"], 56)  # sigma-2 skipped
        pl = RU.size_plan(0.20, 0.10, 0.1, 0.1, p2_dropped=False, max_n_id=120)
        self.assertEqual((pl["n_id"], pl["caps"]), (120, ["unseen support classes"]))
        pl = RU.size_plan(0.20, 0.30, 0.1, 0.1, p2_dropped=False, fits=lambda n_id, n_k: False)
        self.assertEqual((pl["n_id"], pl["n_k"], pl["fits"]), (96, 56, False))
        self.assertIn("cpu", pl["caps"])
        pl = RU.size_plan(0.20, 0.10, 0.1, 0.1, p2_dropped=False, fits=lambda n_id, n_k: n_id <= 130)
        self.assertEqual(pl["n_id"], 130)

    def test_cuts_in_prereg_order(self):
        cuts, pr = RU.choose_cuts(lambda c: {"total": 200.0 - 20 * len(c)})
        self.assertEqual(cuts, ["budget-curve", "SF-FT_400", "H-KN"])
        self.assertEqual(pr["total"], 140.0)
        self.assertEqual(RU.choose_cuts(lambda c: {"total": 150.0})[0], [])
        rates = {"step_s": {k: 0.01 for k in (LA.step_name(m, B, b) for m, B, b in LA.STEP_UNITS)}, "vi_s": 0.3,
                 "cell_s": {"ENUM-MIS±@H-NEG": {"mean": 2.0}}}
        cfg = dict(zs_steps={"SF": 4000, "UVFA": 4000, "UVFA+aux": 4000}, pgv=dict(width=256, steps=500), ft_steps=100,
                   pg400="PGV", rext=False)
        prev = RU.project_confirmation(rates, 96, 56, 56, cfg)
        self.assertGreater(prev["total"], prev["protected"])
        for i in range(1, len(RU.CUT_ORDER) + 1):
            pr = RU.project_confirmation(rates, 96, 56, 56, cfg, RU.CUT_ORDER[:i])
            self.assertLessEqual(pr["total"], prev["total"] + 1e-9)
            prev = pr
        self.assertEqual(prev["total"], prev["protected"])  # items 1-2 are the only non-protected items


class Unit3(Sandbox):
    """The lane driver on three keyed toy units."""

    def ctx(self):
        return RU.Ctx(runs=self.runs, env=self.lenv, decisions=lambda: "", exclusions=lambda s: None,
                      adapters=object())

    def plan_for(self, log, marker_after=None):
        def unit(i):
            def run(sl):
                log.add(f"u{i}", {"i": i})
                if i == marker_after:
                    LN.waiting_path("bwm").write_text("{}", encoding="utf-8")
                return not sl.safe_point("batch")
            return run
        return lambda: [RU.Unit(f"u{i}", 1.0, unit(i)) for i in range(3) if f"u{i}" not in log]

    def test_refused_start_writes_nothing(self):
        LN.waiting_path("rcx").write_text("{}", encoding="utf-8")
        log = RU.KeyedLog(self.runs / "toy.jsonl")
        before = sorted(p.name for p in self.runs.rglob("*"))
        self.assertEqual(RU.drive(self.ctx(), "toy", self.plan_for(log), attempt="t", cd=None), RU.EXIT_REFUSED)
        self.assertEqual(sorted(p.name for p in self.runs.rglob("*")), before)
        self.assertIsNone(LN.read_lock())

    def test_yield_then_resume(self):
        ctx, log = self.ctx(), RU.KeyedLog(self.runs / "toy.jsonl")
        self.assertEqual(RU.drive(ctx, "toy", self.plan_for(log, marker_after=0), attempt="t", cd=None),
                         RU.EXIT_REQUEUE)
        self.assertIsNone(LN.read_lock())  # released
        self.assertEqual(set(log.rows), {"u0"})
        ev = [json.loads(x) for x in (self.runs / "LANE_EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in ev], ["acquire", "exit"])
        self.assertIn("waiting marker", ev[-1]["reason"])
        self.assertEqual(ev[0]["party"], "rcx-e05")
        LN.waiting_path("bwm").unlink()
        self.assertEqual(RU.drive(ctx, "toy", self.plan_for(log), attempt="t", cd=None), 0)
        self.assertEqual(set(RU.KeyedLog(self.runs / "toy.jsonl").rows), {"u0", "u1", "u2"})
        att = RU.attempt_events(ctx)
        self.assertEqual([(e["event"], e.get("complete")) for e in att],
                         [("start", None), ("end", False), ("start", None), ("end", True)])
        prov = [json.loads(x) for x in (self.runs / "PROVENANCE.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(prov), 2)
        self.assertEqual(set(prov[0]["reference_shas"]), {"experiments/RCX-E05/lane_e05.py", "rcx/lane.py",
                                                          "rcx/guard.py"})
        self.assertFalse(list(self.lane.glob("rineya-cpu-lane.waiting.rcx-e05")))  # never a marker of its own


class TestGates(Sandbox):
    def ctx(self, text):
        return RU.Ctx(runs=self.runs, env=self.lenv, decisions=lambda: text[0], exclusions=lambda s: None,
                      adapters=object())

    def args(self, k, m, d):
        return RU.build_parser().parse_args(["oracle-test", "--attempt", str(k), "--manifest-sha", m,
                                             "--code-digest", d])

    def test_attempts_learned_gate_and_custody_void(self):
        d3 = RU.code_digest("t3")["digest"]
        text = [f"T3 {d3}"]
        ctx = self.ctx(text)
        RU.check_attempt_allowed(ctx, 1)
        for k in (0, 4):
            with self.assertRaises(SystemExit):
                RU.check_attempt_allowed(ctx, k)
        with self.assertRaises(SystemExit):
            RU.check_attempt_allowed(ctx, 2)  # attempt 1 did not end VOID
        man = ctx.adir(1) / "TEST_MANIFEST.json"
        m = RU.write_create_only(man, {"attempt": 1, "n": 5, "parents": [], "code_digest_t3": d3})
        with self.assertRaises(SystemExit):
            RU.oracle_test(ctx, self.args(1, m, d3))  # manifest sha not committed: refusal
        self.assertFalse((ctx.adir(1) / "ORACLE_TEST_REPORT.json").exists())
        text[0] += f" manifest {m}"
        for rep in (None, {"status": "VOID", "manifest_sha256": m, "code_digest": d3},
                    {"status": "PASS", "manifest_sha256": "0" * 64, "code_digest": d3}):
            p = ctx.adir(1) / "ORACLE_TEST_REPORT.json"
            p.unlink(missing_ok=True)
            if rep:
                p.write_text(json.dumps(rep), encoding="utf-8")
            with self.assertRaises(SystemExit):
                RU.require_oracle_pass(ctx, 1, m, d3)
        (ctx.adir(1) / "ORACLE_TEST_REPORT.json").unlink()
        RU.log_attempt(ctx, "first-learned-test", attempt=1, phase="pgv")
        with self.assertRaises(SystemExit):
            RU.oracle_test(ctx, self.args(1, m, d3))  # no VOID after a learned-arm TEST computation
        # attempt 2: the committed manifest is edited before oracle-test -> VOID, recorded without any computation
        RU.write_create_only(ctx.adir(1) / "ORACLE_TEST_REPORT.json", {"status": "VOID"})
        man2 = ctx.adir(2) / "TEST_MANIFEST.json"
        m2 = RU.write_create_only(man2, {"attempt": 2, "n": 5, "parents": [], "code_digest_t3": d3})
        text[0] += f" manifest2 {m2}"
        man2.write_text(man2.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertEqual(RU.oracle_test(ctx, self.args(2, m2, d3)), 0)
        r = json.loads((ctx.adir(2) / "ORACLE_TEST_REPORT.json").read_text(encoding="utf-8"))
        self.assertEqual((r["status"], r["checks"]["manifest"]["status"]), ("VOID", "VOID"))
        self.assertEqual(RU.attempt_events(ctx, 2)[-1]["void"], ["manifest"])
        with self.assertRaises(SystemExit):
            RU.oracle_test(ctx, self.args(2, m2, d3))  # create-only
        # attempt 3: a committed T3 digest that the code no longer matches -> VOID
        fake = sha(b"an older T3 digest")
        m3 = RU.write_create_only(ctx.adir(3) / "TEST_MANIFEST.json", {"attempt": 3, "n": 5, "parents": [],
                                                                       "code_digest_t3": fake})
        text[0] += f" {m3} {fake}"
        self.assertEqual(RU.oracle_test(ctx, self.args(3, m3, fake)), 0)
        r = json.loads((ctx.adir(3) / "ORACLE_TEST_REPORT.json").read_text(encoding="utf-8"))
        self.assertEqual((r["status"], r["checks"]["code_digest"]["status"]), ("VOID", "VOID"))
        self.assertIsNone(LN.read_lock())


class TestRealAdapters(unittest.TestCase):
    """The adapter contract against the real enum_e05 / learners_e05 at n = 4 with tiny models (torch CPU, 1 thread):
    zero-shot SF and UVFA, PGV and FT per-goal fits, every learned arm's closed loop, the G-P4 values, ENUM-MIS± and
    the G-P5 step timer."""

    def test_contract(self):
        os.environ["E05_TORCH_THREADS"] = "1"
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        n, fi = 4, SO.fact_index(4)
        goals = []
        for items in ({("on", 0, 1): 2, ("ontable", 2): 1}, {("on", 2, 3): 1, ("on", 1, 0): 3, ("ontable", 3): 1}):
            w = np.zeros(n * (n + 1))
            for f, x in items.items():
                w[fi[f]] = x
            goals.append(dict(w=w, c=0.25, rtype="lin", dyn="free", G=(), R=0.0, K=len(items), n_neg=0))
        ps = [FA.make_parent("DEV-ID", n, g) for g in goals]
        d = RU.Data(None, {"n": n, "parents": [p.to_json() for p in ps]}, ".")
        rng = random.Random(0)
        for p in ps:
            pool = FA.start_pool(d.space, d.table(p))
            p.starts = [int(x) for x in rng.sample(list(pool), 2)]
        idxs = [rng.sample(range(len(d.space)), 24) for _ in ps]
        data = RU.train_rows(d, [(p.weights(), p.c) for p in ps], idxs)
        lib = SimpleNamespace(W=np.stack([p.weights() for p in ps]), c=np.array([p.c for p in ps]))
        ad = RU.Adapters()
        never = lambda kind="step": False  # noqa: E731
        with tempfile.TemporaryDirectory() as t:
            sf = ad.train_zero_shot("SF", d, data, seed=0, cfg=dict(epochs=2, lr=1e-3, keep=(1, 2)), out_dir=Path(t) / "sf",
                                    safe_point=never)
            uv = ad.train_zero_shot("UVFA", d, data, seed=0, cfg=dict(epochs=1, lr=1e-3, keep=(1,)),
                                    out_dir=Path(t) / "uv", safe_point=never)
            self.assertEqual(sorted(sf), [1, 2])
            self.assertEqual((ad.kind(sf[2]), ad.kind(uv[1])), ("sf", "uvfa"))
            jobs = [dict(key=p.key, states=np.asarray(ix[:12]), v=d.table(p).V[np.asarray(ix[:12])],
                         stab=FA.stabilizer(np.asarray(p.w), n), seed=7 + i, w=p.weights(), c=p.c)
                    for i, (p, ix) in enumerate(zip(ps, idxs))]
            pgv = ad.fit_pergoal("PGV", d, jobs, dict(width=64, steps=4, keep=4, lr=1e-3, batch=8, eval_every=2,
                                                      val_n=4, wd=1e-4), out_dir=Path(t) / "pgv", safe_point=never)
            ft = ad.fit_pergoal("FT", d, [dict(j, init=uv[1]) for j in jobs],
                                dict(steps=2, keep=2, lr=3e-4, batch=8, eval_every=1, val_n=4, step0=True),
                                out_dir=Path(t) / "ft", safe_point=never)
            self.assertEqual(set(pgv), set(ft), {p.key for p in ps})
            for p in ps:
                for arm, path in (("SF-GPI", sf[2]), ("SF-noGPI", sf[2]), ("SF-L", sf[2]), ("UVFA", uv[1]),
                                  ("PGV", pgv[p.key]), ("FT", ft[p.key])):
                    pol = RU.learned_policy(arm, path, d, p, False, ad, lib=lib)
                    for s0 in p.starts:
                        r = SO.evaluate(d.space, d.table(p), s0, pol)
                        self.assertTrue(0.0 <= r["nreg"] <= 1.0, (arm, r))
                S = np.asarray(p.starts)
                self.assertEqual(RU.model_values(ad, sf[2], d, p, S).shape, (2,))
                self.assertEqual(RU.model_values(ad, uv[1], d, p, S).shape, (2,))
                pol, diag = ad.enum_policy(d, p, "ENUM-MIS±")
                SO.evaluate(d.space, d.table(p), p.starts[0], pol)
                self.assertIn("all_tied", diag)
            self.assertGreater(ad.time_step("PGV64", n, 64, 8, 2), 0.0)


# ----------------------------------------------------------------------------- end to end at n = 5 (fake learners)

class FakeAdapters(RU.Adapters):
    """No torch. 'Models' are JSON files; SF networks return exact psi tables of their code (free dynamics), value
    networks exact V* of their code, PGV networks the parent's V*; the selection UVFA at the base config is weak (it
    returns w.phi, the copy-state value), so G-P4(b) fails and the one UVFA extension runs. ENUM-H*± and GNN-ENUM± run
    as ENUM-MIS± (speed; they have their own tests)."""

    def __init__(self):
        self.calls, self.marker_once, self._tabs = Counter(), set(), {}

    def enum_policy(self, d, p, arm):
        return super().enum_policy(d, p, "ENUM-MIS±" if arm in ("ENUM-H*±", "GNN-ENUM±") else arm)

    @staticmethod
    def _write(path, **meta):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta), encoding="utf-8")
        return str(path)

    def train_zero_shot(self, model, d, data, *, seed, cfg, out_dir, safe_point):
        self.calls[f"zs:{model}"] += 1
        assert {"states", "goal", "W", "c", "psi", "v"} <= set(data) and len(data["W"]) == len(data["states"])
        if model in self.marker_once:  # a peer asks for the lane during training: the safe point must yield
            self.marker_once.discard(model)
            LN.waiting_path("bwm").write_text("{}", encoding="utf-8")
            if safe_point("epoch"):
                return None
        kind = {"SF": "sf", "SF-shuffled": "sf", "UVFA": "uvfa", "UVFA+aux": "aux"}[model]
        weak = model == "UVFA" and seed == RU.SEEDS["selection"] and cfg["epochs"] == RU.ZS_BASE["epochs"]
        return {int(e): self._write(Path(out_dir) / f"e{e}.json", kind=kind, weak=weak, seed=seed)
                for e in cfg["keep"]}

    def fit_pergoal(self, kind, d, jobs, cfg, *, out_dir, safe_point):
        self.calls[f"pg:{kind}"] += 1
        net = {"PGV": "pgv", "FT": "uvfa", "SF-FT": "sf"}[kind]
        for j in jobs:
            assert len(j["states"]) == len(j["v"]) and j["stab"].shape[1] == d.n
            assert ("init" in j) == (kind != "PGV") and (("psi" in j) == (kind == "SF-FT"))
        return {j["key"]: self._write(Path(out_dir) / f"{RU.fname(j['key'])}.json", kind=net, key=j["key"],
                                      weak=False) for j in jobs}

    def _meta(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def kind(self, path):
        return self._meta(path)["kind"]

    def _tab(self, d, w, c):
        k = (d.n, tuple(np.asarray(w, dtype=np.float64).tolist()), float(c))
        if k not in self._tabs:
            self._tabs[k] = SO.sf_table(d.space, SO.stop_values(d.phi, w), c)
        return self._tabs[k]

    def psi(self, path, d, states, W, c):
        S = np.asarray(states, dtype=np.int64)
        return np.stack([SO.psi(d.phi, self._tab(d, w, cc), S) for w, cc in zip(np.atleast_2d(W),
                                                                                  np.atleast_1d(c))], axis=1)

    def value(self, path, d, states, W=None, c=None):
        meta, S = self._meta(path), np.asarray(states, dtype=np.int64)
        if meta["kind"] == "pgv":
            return d.table(d.by_key[meta["key"]]).V[S][:, None]
        W, c = np.atleast_2d(W), np.atleast_1d(c)
        if meta["weak"]:
            return np.stack([SO.stop_values(d.phi, w)[S] for w in W], axis=1)
        return np.stack([self._tab(d, w, cc).V[S] for w, cc in zip(W, c)], axis=1)

    def time_step(self, model, n, width, batch, B):
        return 1e-4 * B * batch / 64


SMALL = dict(train=12, dev_id_s=4, dev_id_u=4, dev_ood=2, bank=16, starts=4, hid_s_max=2, subset_per_family=1,
             subset_per_stratum=1, test_n_id=4, test_n_k=2, test_n_neg=2,
             **{"H-COST": 1, "H-SPAN": 1, "H-KN": 1, "H-CONJ": 1, "H-DYN": 1})


class TestEndToEnd(Sandbox):
    def setUp(self):
        super().setUp()
        self.root = Path(self.tmp.name) / "repo"
        for rel in RU.NEW_T3 + tuple(RU.PINNED):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, self.root / rel)
        sets = {"runs/RCX-E07/main/data/instances.json": [{"split": "test", "objs": list("abcde")}],
                "runs/RCX-E07/e07b/data/testb_instances.json": [{"objs": list("abcdefghij")}],
                "runs/RCX-E11/main/a/instances.json": [{"objs": list("abcdefghij")}],
                "runs/RCX-E11/main/b/data/instances.json": [{"split": "exec", "objs": list("abcdefghij")},
                                                            {"split": "train", "objs": list("abcdefg")}]}
        for rel, obj in sets.items():
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(json.dumps(obj), encoding="utf-8")
        self.text = [""]
        self.ad = FakeAdapters()
        self.prereg = Path(self.tmp.name) / "PREREG_E05.md"
        arms = RU.ZS_TEST + RU.TEST_REF_ARMS + ("PGV", "FT", "ENUM-H*±", "GNN-ENUM±")
        bench = dict(step_s={LA.step_name(m, B, b): 1e-6 for m, B, b in LA.STEP_UNITS}, vi_s=1e-3,
                     cell_s={a: dict(mean=1e-4, max=1e-4, n=1) for a in arms})  # fast: no §10.3 cut applies
        self.ctx = RU.Ctx(runs=self.runs, root=self.root, n=5, env=self.lenv, decisions=lambda: self.text[0],
                          exclusions=lambda space: None, adapters=self.ad, prereg=self.prereg, sizes=SMALL,
                          q_sizes=dict(q1=2, q2_goals=2, q2_states=3, q2_n=4, q3=20, q5=2), bench_override=bench)

    def run_cmd(self, *argv):
        return RU.run_command(self.ctx, RU.build_parser().parse_args(list(argv)))

    def test_pipeline(self):
        d2, d3 = RU.code_digest("t2", self.root)["digest"], RU.code_digest("t3", self.root)["digest"]
        with self.assertRaises(SystemExit):
            self.run_cmd("pilot-a", "--expect-code-digest", d2)  # T2 digest not committed
        self.text[0] = f"E05 pilot code digest {d2}\n"
        # pilot-a: a peer marker appears during the first SF training -> yield (75), then resume to completion
        self.ad.marker_once.add("SF")
        self.assertEqual(self.run_cmd("pilot-a", "--expect-code-digest", d2), RU.EXIT_REQUEUE)
        self.assertIsNone(LN.read_lock())
        LN.waiting_path("bwm").unlink()
        self.assertEqual(self.run_cmd("pilot-a", "--expect-code-digest", d2), 0)
        with self.assertRaises(SystemExit):
            self.run_cmd("pilot-report", "--expect-code-digest", d2)  # pilot-b pending
        self.assertEqual(self.run_cmd("pilot-b", "--expect-code-digest", d2), 0)
        self.assertEqual(self.run_cmd("pilot-report", "--expect-code-digest", d2), 0)
        rep = json.loads((self.ctx.pdir / "PILOT_REPORT.json").read_text(encoding="utf-8"))
        g1 = rep["G_P1"]
        self.assertTrue(all(v["ok"] for v in g1["checks"].values()), g1["checks"])
        self.assertTrue(g1["Q7"]["ok"], g1["Q7"])
        g4 = rep["G_P4"]["at400"]
        self.assertTrue(g4["a"]["ok"], g4["a"])
        self.assertEqual((g4["b"]["cfg"], g4["b"]["extension"]["ok"]), ("ext", True))  # the one UVFA extension
        self.assertEqual(rep["selected"]["UVFA"][2:], ["ext", 80])
        self.assertFalse(rep["p1_comparator_invalid"])
        det = rep["determinations"]
        self.assertEqual((det["pg400"], det["p2_dropped"], det["r_ext"]), ("FT-UVFA_400", False, False))
        self.assertEqual(det["kcurve_arms"], {"400": "PGV_400", "1600": "PGV_1600", "6400": "PGV_6400"})
        self.assertEqual(rep["sizing"]["n_id"], 96)  # exact fakes: zero DEV variance
        self.assertEqual((rep["G_P5"]["cuts"], det["cut_items"]), ([], []))
        self.assertTrue(rep["G_P5"]["units"]["ok"])
        self.assertEqual(rep["dev_nreg"]["SF-GPI"]["DEV-ID"], 0.0)
        self.assertEqual(rep["dev_nreg"]["ORACLE"]["DEV-ID"], 0.0)
        self.assertEqual(set(rep["fill"]), {"digests", "n_labels", "library", "G_P1", "selection_G_P4", "G_P3",
                                            "G_P3b", "dev_nreg", "sigma", "sizes", "powers", "G_P5", "exclusions",
                                            "reference_shas", "T3"})
        man_p = json.loads((self.ctx.pdir / "PILOT_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(man_p["code_digest_t2"], d2)
        with self.assertRaises(SystemExit):
            self.run_cmd("pilot-report", "--expect-code-digest", d2)  # create-only
        # T3 freeze and the TEST draw
        self.prereg.write_text("PREREG_E05 (test)", encoding="utf-8")
        ps = sha(self.prereg.read_bytes())
        with self.assertRaises(SystemExit):
            self.run_cmd("draw-test", "--attempt", "1", "--expect-prereg-sha", ps, "--expect-code-digest", d3)
        self.text[0] += f"prereg {ps}\nT3 {d3}\n"
        self.assertEqual(self.run_cmd("draw-test", "--attempt", "1", "--expect-prereg-sha", ps,
                                      "--expect-code-digest", d3), 0)
        adir = self.ctx.adir(1)
        m = sha((adir / "TEST_MANIFEST.json").read_bytes())
        man = json.loads((adir / "TEST_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(Counter(p["family"] for p in man["parents"]),
                         Counter({"H-ID": 4, "H-K": 2, "H-NEG": 2, "H-COST": 1, "H-SPAN": 1, "H-KN": 1, "H-CONJ": 1,
                                  "H-DYN": 1}))
        self.assertEqual(len(man["subset"]), 4)
        tst = ["--attempt", "1", "--manifest-sha", m, "--code-digest", d3]
        with self.assertRaises(SystemExit):
            self.run_cmd("oracle-test", *tst)  # manifest sha not yet committed
        self.text[0] += f"TEST_MANIFEST {m}\n"
        with self.assertRaises(SystemExit):
            self.run_cmd("pgv", *tst)  # no ORACLE_TEST_REPORT PASS yet
        self.assertEqual(self.run_cmd("oracle-test", *tst), 0)
        orep = json.loads((adir / "ORACLE_TEST_REPORT.json").read_text(encoding="utf-8"))
        self.assertEqual(orep["status"], "PASS", {k: v for k, v in orep["checks"].items() if v["status"] != "PASS"})
        self.assertEqual(set(orep["checks"]), {"manifest", "code_digest", "Q3", "Q4", "Q7", "Q8:splits",
                                               "Q8:exclusions", "Q8:n7", "Q8:subset"})
        self.assertFalse(RU.first_learned(self.ctx, 1))
        self.assertEqual(self.run_cmd("train", "--expect-prereg-sha", ps, "--expect-code-digest", d3), 0)
        with self.assertRaises(SystemExit):
            self.run_cmd("eval", *tst)  # pgv and ft first
        self.assertEqual(self.run_cmd("pgv", *tst), 0)
        self.assertTrue(RU.first_learned(self.ctx, 1))
        with self.assertRaises(SystemExit):
            self.run_cmd("oracle-test", *tst)  # VOID can no longer be declared
        self.assertEqual(self.run_cmd("ft", *tst), 0)
        self.assertEqual(self.run_cmd("eval", *tst), 0)
        self.assertEqual(self.run_cmd("extras", *tst), 0)
        self.assertEqual(self.run_cmd("eval", *tst), 0)  # complete: nothing pending, nothing rewritten
        evs = [e["event"] for e in RU.attempt_events(self.ctx, 1)]
        self.assertEqual(evs.count("first-learned-test"), 1)
        self.assertLess(evs.index("oracle-test"), evs.index("first-learned-test"))
        self.assertIsNone(LN.read_lock())
        self.assertEqual(self.ad.calls["zs:SF-shuffled"], 1)
        # the eval files satisfy analyze_e05's input contract and cover the protected core
        AN = _load("analyze_e05", ROOT / "experiments" / "RCX-E05" / "analyze_e05.py")
        design = AN.make_design(man)
        det = AN.pilot_determinations(rep)
        cells, records = AN.load_evals(adir, design, m, d3, det)
        self.assertEqual(AN.absent_groups(cells, AN.core_groups(design, det)), {})
        for arm in ("SF-GPI", "UVFA", "UVFA+aux", "FT-UVFA_400"):
            self.assertEqual(sorted(cells[arm]), list(RU.SEEDS["replicates"]), arm)
        for arm in ("PGV_400", "PGV_1600", "PGV_6400", "ENUM-MIS±", "ORACLE", "SF-FT_400", "SF-GPI [16x400]",
                    "ENUM-MIS (rev 1)"):
            self.assertIn(arm, cells)
        prim = {k for k, v in design["parents"].items() if v["family"] in FA.PRIMARY}
        for rep_ in RU.SEEDS["replicates"]:
            for k in prim:
                self.assertEqual(cells["SF-GPI"][rep_][k]["nreg"], [0.0] * 4)  # exact psi: GPI = ORACLE
        self.assertEqual(set(cells["PGV_1600"][None]), set(man["subset"]))
        n_files = len(list((adir / "evals").glob("*.json")))
        self.assertEqual(n_files, len(records))
        for lg in ("cells", "banks", "fits", "oracle_cells"):
            RU.KeyedLog(adir / f"{lg}.jsonl")  # no duplicate keys after every phase and the rerun


if __name__ == "__main__":
    unittest.main()
