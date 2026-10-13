"""RCX-E05 lane gate (experiments/RCX-E05/lane_e05.py; PREREG_E05_DRAFT.md §10.2, §2.1 test table). Everything runs
on temporary RINEYA_LANE_DIR / TEMP directories with an injected clock, RAM, guard, process scan and pid liveness."""
import datetime as dt
import importlib.util
import json
import os
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcx import lane as LN  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("lane_e05", ROOT / "experiments" / "RCX-E05" / "lane_e05.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lane_e05"] = mod
    spec.loader.exec_module(mod)
    return mod


LE = _load()
T = dt.datetime


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += dt.timedelta(seconds=s)


REFUSAL = "{} lane: not acquiring (inside the window ending {}, but a 170-min block would end too late); waiting (no marker)"


class LaneBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.lane, self.temp, self.runs = base / "lane", base / "temp", base / "runs"
        for d in (self.lane, self.temp, self.runs):
            d.mkdir()
        self._env = {k: os.environ.get(k) for k in ("RINEYA_LANE_DIR", "TEMP")}
        os.environ["RINEYA_LANE_DIR"], os.environ["TEMP"] = str(self.lane), str(self.temp)
        self.paths = LE.Paths(windows=self.runs / "quiet_windows.json", extra_windows=self.runs / "extra_windows.json",
                              train_log=self.runs / "train_chain.log", post_log=self.runs / "post_train_chain.log",
                              events=self.runs / "LANE_EVENTS.jsonl", thermal_flag=self.runs / ".thermal_pause")
        self.write_windows([["2026-10-12T23:00", "2026-10-13T07:00"], ["2026-10-13T23:00", "2026-10-14T07:00"]])
        self.paths.train_log.write_text("16:24:22 train chain v2 start\n16:24:22 lane: not acquiring (outside the "
                                        "quiet windows; next opens 10-12 23:00); waiting (no marker)\n", encoding="utf-8")
        self.paths.post_log.write_text("16:24:23 post-train chain v2 start\n", encoding="utf-8")
        self.clock = Clock(T(2026, 10, 12, 14, 0))
        self.ram, self.tz, self.strays, self.alive = [12.0], [[]], [[]], set()
        self.me = os.getpid()
        self.env = LE.Env(paths=self.paths, now=self.clock, ram=lambda: self.ram[0],
                          pid_alive=lambda p: int(p) == self.me or int(p) in self.alive,
                          guard=lambda resuming: (list(self.tz[0]) if not resuming else
                                                  (["guard: hot (resuming)"] if self.tz_resume_hot else []), None),
                          strays=lambda: list(self.strays[0]), pid=self.me)
        self.tz_resume_hot = False

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def write_windows(self, ws, path=None, margin=5):
        (path or self.paths.windows).write_text(json.dumps({"end_margin_min": margin, "windows": ws}),
                                                encoding="utf-8")

    def gpu(self, pid=None, workers=()):
        (self.temp / "rineya-gpu-0.lock").write_text(json.dumps({"pid": pid, "worker_pids": list(workers),
                                                                 "label": "bwm:bundle-x"}), encoding="utf-8")

    def files(self):
        return sorted(str(p.relative_to(self.tmp.name)) for p in Path(self.tmp.name).rglob("*") if p.is_file())

    def check(self, L=5.0):
        return LE.start_check(self.env, L)


class TestStartCondition(LaneBase):
    def test_day_mode_ok_and_party(self):
        d = self.check()
        self.assertTrue(d.ok, d.reasons)
        self.assertEqual(d.mode, "day")
        self.assertEqual(d.deadline, T(2026, 10, 12, 22, 50))
        s = LE.Session.open(self.env, "E05 pilot-a", 5.0)
        lock = LN.read_lock()
        self.assertEqual((lock["holder"], lock["pid"]), ("rcx-e05", self.me))
        self.assertAlmostEqual(lock["expected_end"] - lock["started"], 15 * 60, places=3)
        self.assertTrue(s.release("done"))
        self.assertIsNone(LN.read_lock())
        evs = [json.loads(x) for x in self.paths.events.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in evs], ["acquire", "exit"])

    def test_refuses_any_waiting_marker(self):
        for party in ("rcx", "bwm", "dsv3wm", "rcx-e05"):
            m = LN.waiting_path(party)
            m.write_text("{}", encoding="utf-8")
            d = self.check()
            self.assertFalse(d.ok)
            self.assertTrue(any("waiting marker" in r for r in d.reasons))
            m.unlink()

    def test_refuses_held_lane_and_reclaims_stale(self):
        LN.lock_path().write_text(json.dumps({"holder": "bwm", "job": "x", "pid": self.me}), encoding="utf-8")
        self.assertFalse(self.check().ok)
        LN.lock_path().write_text("{not json", encoding="utf-8")  # unparseable lock: held
        self.assertFalse(self.check().ok)
        dead = 999_999
        while LN.pid_alive(dead):
            dead += 1
        LN.lock_path().write_text(json.dumps({"holder": "bwm", "job": "x", "pid": dead}), encoding="utf-8")
        d = self.check()
        self.assertTrue(d.ok, d.reasons)
        self.assertEqual(d.lane, "stale")
        self.assertTrue(LN.lock_path().exists())  # the check itself writes nothing
        s = LE.Session.open(self.env, "job", 5.0)
        self.assertEqual(LN.read_lock()["holder"], "rcx-e05")
        self.assertTrue(list(self.lane.glob("rineya-cpu-lane.json.stale-*.bak")))
        s.release()

    def test_ram_floor(self):
        for ram, ok in ((None, False), (3.9, False), (9.99, False), (10.0, True), (32.0, True)):
            self.ram[0] = ram
            self.assertEqual(self.check().ok, ok, ram)

    def test_windows_files(self):
        self.paths.windows.write_text("{broken", encoding="utf-8")
        self.assertFalse(self.check().ok)
        self.paths.windows.unlink()
        self.assertFalse(self.check().ok)  # the required file is missing
        self.write_windows([["2026-10-12T23:00", "2026-10-13T07:00"]])
        self.assertTrue(self.check().ok)  # absent extras: no extra windows
        self.paths.extra_windows.write_text("[]", encoding="utf-8")
        self.assertFalse(self.check().ok)  # present but unparseable extras
        self.write_windows([["2026-10-12T14:00", "2026-10-12T16:00"]], self.paths.extra_windows)
        d = self.check()
        self.assertEqual(d.mode, "night-tail")  # an extra window covers now
        self.assertFalse(d.ok)  # and E01R's log has no refusal for it
        self.write_windows([["2026-10-12T14:20", "2026-10-12T16:00"]], self.paths.extra_windows)
        d = self.check(L=10.0)  # an extra window opening at 14:20: day mode ends by 14:10
        self.assertTrue(d.ok, d.reasons)
        self.assertEqual((d.mode, d.deadline), ("day", T(2026, 10, 12, 14, 10)))
        self.assertFalse(self.check(L=10.01).ok)

    def test_gpu_lease(self):
        self.gpu(pid=4242)
        self.assertTrue(self.check().ok)  # dead holder pid, no workers: no live holder
        self.alive.add(4242)
        self.assertFalse(self.check().ok)
        self.alive.clear()
        self.gpu(pid=4242, workers=[1, 2, 4343])
        self.assertTrue(self.check().ok)
        self.alive.add(4343)
        self.assertFalse(self.check().ok)  # a live worker pid is a live holder
        (self.temp / "rineya-gpu-0.lock").write_text("{partial", encoding="utf-8")
        self.assertFalse(self.check().ok)  # unparseable -> refuse
        (self.temp / "rineya-gpu-0.lock").write_text(json.dumps({"label": "x"}), encoding="utf-8")
        self.assertFalse(self.check().ok)  # no pid -> unparseable

    def test_day_mode_pre_window_rule(self):
        self.clock.t = T(2026, 10, 12, 22, 35)
        self.assertTrue(self.check(L=15.0).ok)  # ends 22:50 = 23:00 - 10 min
        self.assertFalse(self.check(L=15.01).ok)
        self.clock.t = T(2026, 10, 12, 22, 45)
        self.assertFalse(self.check(L=5.01).ok)
        self.assertTrue(self.check(L=5.0).ok)
        self.assertAlmostEqual(LE.slice_budget_min(self.env), 5.0)
        self.clock.t = T(2026, 10, 15, 9, 0)  # all windows expired: no time limit beyond 15 min
        self.assertTrue(self.check(L=15.0).ok)
        self.assertFalse(self.check(L=15.5).ok)
        self.assertEqual(LE.slice_budget_min(self.env), 15.0)

    def test_guard_flag_strays_length(self):
        self.tz[0] = ["guard: tz01=97.5 >= 97"]
        self.assertFalse(self.check().ok)
        self.tz[0] = []
        self.paths.thermal_flag.write_text("hot", encoding="utf-8")
        self.assertFalse(self.check().ok)
        self.paths.thermal_flag.unlink()
        self.strays[0] = [(123, "python -m unittest tests.test_x")]
        self.assertFalse(self.check().ok)
        self.strays[0] = []

        def boom():
            raise RuntimeError("no psutil")
        self.env.strays = boom
        self.assertFalse(self.check().ok)  # unknown process table: fail closed
        self.env.strays = lambda: []
        self.assertFalse(self.check(L=15.01).ok)
        self.assertFalse(self.check(L=0.0).ok)
        self.assertTrue(self.check(L=15.0).ok)

    def test_guard_hysteresis_after_guard_yield(self):
        s = LE.Session.open(self.env, "job", 5.0)
        self.tz[0] = ["guard: tz01=97.2 >= 97"]
        self.clock.advance(61)
        self.assertIn("guard", s.check())
        s.release(s.reason)
        self.tz[0], self.tz_resume_hot = [], True  # 95 C: below 97, above 93
        self.assertFalse(self.check().ok)
        self.tz_resume_hot = False  # back at <= 93 C
        self.assertTrue(self.check().ok)

    def test_projected_min(self):
        self.assertAlmostEqual(LE.projected_min(gflop=19.0 * 600), 12.0)
        self.assertAlmostEqual(LE.projected_min(seconds=60), 1.2)


class TestNightTail(LaneBase):
    def setUp(self):
        super().setUp()
        self.clock.t = T(2026, 10, 13, 5, 0)

    def test_refused_while_e01r_may_take_a_block(self):
        d = self.check()
        self.assertEqual(d.mode, "night-tail")
        self.assertFalse(d.ok)  # newest lane line is "outside the windows"
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write("23:00:05 lane: acquired for 'RCX E01R train E-R2_s0'\n")
        self.assertFalse(self.check().ok)

    def test_refusal_line_for_this_window(self):
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write(REFUSAL.format("04:13:00", "10-13 07:00") + "\n")
        d = self.check()
        self.assertTrue(d.ok, d.reasons)
        self.assertEqual(d.deadline, T(2026, 10, 13, 6, 55))
        self.assertTrue(self.check(L=15.0).ok)  # 05:15 <= 06:55
        self.clock.t = T(2026, 10, 13, 6, 45)
        self.assertTrue(self.check(L=10.0).ok)  # ends 06:55 exactly
        self.assertFalse(self.check(L=10.01).ok)
        with open(self.paths.train_log, "a", encoding="utf-8") as f:  # E01R log noise after the refusal
            f.write("06:46:00 released\n")
        self.assertTrue(self.check(L=5.0).ok)

    def test_refusal_for_another_window_or_unreadable_log(self):
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write(REFUSAL.format("04:13:00", "10-12 07:00") + "\n")  # last night's window
        self.assertFalse(self.check().ok)
        self.paths.train_log.unlink()
        self.assertFalse(self.check().ok)

    def test_post_train_log_is_active_once_started(self):
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write(REFUSAL.format("04:13:00", "10-13 07:00") + "\n")
        self.assertTrue(self.check().ok)
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write("04:20:00 TRAIN CHAIN DONE\n")
        self.assertFalse(self.check().ok)  # post-train chain active, no lane line yet
        with open(self.paths.post_log, "a", encoding="utf-8") as f:
            f.write("04:21:00 lane: acquired for 'RCX E01R smoke x6'\n")
        self.assertFalse(self.check().ok)
        with open(self.paths.post_log, "a", encoding="utf-8") as f:
            f.write(REFUSAL.replace("170", "75").format("05:00:00", "10-13 07:00") + "\n")
        self.assertTrue(self.check().ok)
        self.paths.post_log.unlink()
        self.assertFalse(self.check().ok)

    def test_release_file_and_e01r_holding(self):
        rel = self.temp / "rineya-quiet-release.rcx.json"
        rel.write_text("{}", encoding="utf-8")
        self.assertTrue(self.check().ok)
        LN.lock_path().write_text(json.dumps({"holder": "rcx", "job": "E01R", "pid": self.me}), encoding="utf-8")
        self.assertFalse(self.check().ok)
        LN.lock_path().unlink()
        self.gpu(pid=self.me)
        self.assertFalse(self.check().ok)  # night-tail also needs no live GPU holder

    def test_overlapping_windows_all_must_allow(self):
        self.write_windows([["2026-10-13T04:00", "2026-10-13T06:00"]], self.paths.extra_windows)
        with open(self.paths.train_log, "a", encoding="utf-8") as f:
            f.write(REFUSAL.format("04:13:00", "10-13 07:00") + "\n")
        self.assertFalse(self.check().ok)  # no refusal for the extra window ending 06:00
        (self.temp / "rineya-quiet-release.rcx.json").write_text("{}", encoding="utf-8")
        d = self.check(L=15.0)
        self.assertTrue(d.ok, d.reasons)
        self.assertEqual(d.deadline, T(2026, 10, 13, 5, 55))


class TestNeverWrites(LaneBase):
    def test_refusals_write_nothing(self):
        before = self.files()
        for setup in (lambda: LN.waiting_path("bwm").write_text("{}", encoding="utf-8"),
                      lambda: self.ram.__setitem__(0, 2.0),
                      lambda: self.gpu(pid=self.me)):
            setup()
            snap = self.files()
            with self.assertRaises(LE.Refused):
                LE.Session.open(self.env, "job", 5.0)
            self.assertEqual(self.files(), snap)
        self.assertFalse(self.paths.events.exists())
        self.assertNotIn("rineya-cpu-lane.json", [Path(f).name for f in self.files()])
        del before

    def test_session_never_writes_markers_gpu_or_release(self):
        s = LE.Session.open(self.env, "job", 5.0)
        for _ in range(30):
            self.clock.advance(61)
            if s.check():
                break
        s.release(s.reason or "done")
        names = {Path(f).name for f in self.files()}
        self.assertFalse({n for n in names if n.startswith("rineya-cpu-lane.waiting.")})
        self.assertFalse({n for n in names if n.startswith("rineya-gpu-0.")})
        self.assertNotIn("rineya-quiet-release.rcx.json", names)
        self.assertNotIn("rineya-cpu-lane.json", names)


class TestSafePoints(LaneBase):
    def open(self):
        return LE.Session.open(self.env, "job", 5.0)

    def test_check_spacing_and_force(self):
        s = self.open()
        self.clock.advance(30)
        self.assertIsNone(s.check())
        self.assertEqual(s.checks, 0)  # < 60 s: no check
        self.assertIsNone(s.check(force=True))  # epoch/batch end
        self.assertEqual(s.checks, 1)
        self.clock.advance(59)
        s.check()
        self.assertEqual(s.checks, 1)
        self.clock.advance(1)
        s.check()
        self.assertEqual(s.checks, 2)
        s.release()

    def assert_yields(self, mutate, what, mode_time=None):
        if mode_time:
            self.clock.t = mode_time
        s = self.open()
        self.clock.advance(61)
        self.assertIsNone(s.check(), "no reason before the change")
        mutate()
        self.clock.advance(61)
        r = s.check()
        self.assertIsNotNone(r)
        self.assertIn(what, r)
        s.release(r)
        self.assertIsNone(LN.read_lock())

    def test_yield_on_marker(self):
        self.assert_yields(lambda: LN.waiting_path("rcx").write_text("{}", encoding="utf-8"), "waiting marker")

    def test_yield_on_new_gpu_holder(self):
        self.assert_yields(lambda: self.gpu(pid=999, workers=[self.me]), "GPU lease")

    def test_yield_on_unreadable_gpu_lease(self):
        self.assert_yields(lambda: (self.temp / "rineya-gpu-0.lock").write_text("{", encoding="utf-8"), "GPU lease")

    def test_yield_on_guard_and_ram(self):
        self.assert_yields(lambda: self.tz[0].append("tz01=97.1 >= 97"), "guard")
        self.tz[0] = []
        self.assert_yields(lambda: self.paths.thermal_flag.write_text("x", encoding="utf-8"), "thermal_pause")
        self.paths.thermal_flag.unlink()
        self.assert_yields(lambda: self.ram.__setitem__(0, 5.9), "free RAM")
        self.ram[0] = 12.0
        self.assert_yields(lambda: self.ram.__setitem__(0, None), "free RAM")

    def test_ram_between_6_and_10_does_not_yield(self):
        s = self.open()
        self.ram[0] = 6.0
        self.clock.advance(61)
        self.assertIsNone(s.check())
        s.release()

    def test_yield_at_15_min(self):
        s = self.open()
        for _ in range(14):
            self.clock.advance(60)
            self.assertIsNone(s.check())
        self.clock.advance(60)
        self.assertIn("15 min elapsed", s.check())
        s.release()

    def test_yield_at_pre_window_mark(self):
        self.clock.t = T(2026, 10, 12, 22, 40)
        s = LE.Session.open(self.env, "job", 9.0)
        self.clock.advance(9 * 60)
        self.assertIsNone(s.check())  # 22:49
        self.clock.advance(61)
        self.assertIn("next window opens", s.check())  # 22:50:01
        s.release()

    def test_yield_on_window_added_since_start(self):
        def add():
            self.write_windows([["2026-10-12T14:01", "2026-10-12T15:00"]], self.paths.extra_windows)
        self.assert_yields(add, "inside a window")

    def test_yield_on_unreadable_windows(self):
        self.assert_yields(lambda: self.paths.windows.write_text("{", encoding="utf-8"), "windows")

    def test_night_tail_end_margin(self):
        self.clock.t = T(2026, 10, 13, 6, 40)
        (self.temp / "rineya-quiet-release.rcx.json").write_text("{}", encoding="utf-8")
        s = LE.Session.open(self.env, "job", 15.0)
        self.assertEqual(s.mode, "night-tail")
        self.gpu(pid=self.me)  # a GPU holder is not a night-tail yield reason (literal §10.2)
        self.clock.advance(14 * 60 + 59)
        self.assertIsNone(s.check())  # 06:54:59
        self.clock.advance(1)
        self.assertIn("W.end - margin", s.check(force=True))  # 06:55
        s.release()

    def test_yield_when_lock_lost(self):
        self.assert_yields(lambda: LN.lock_path().unlink(), "no longer ours")

    def test_yield_mid_batch_checkpoint_resumes_exactly(self):
        """Protocol test: a job checkpointing at a mid-batch yield and resuming in a new session reproduces the
        uninterrupted run (the learners' own exactness is tests/test_e05_learners.py)."""
        def run(state, sess, n_steps, ckpt):
            rng = random.Random()
            rng.setstate(state["rng"])
            while state["step"] < n_steps:
                state["acc"] = (state["acc"] * 31 + rng.randrange(1000)) % 1_000_003
                state["step"] += 1
                self.clock.advance(9)
                if sess is not None and sess.check(force=state["step"] % 10 == 0):
                    state["rng"] = rng.getstate()
                    ckpt.write_bytes(pickle.dumps(state))
                    sess.release(sess.reason)
                    return False
            state["rng"] = rng.getstate()
            return True

        n = 120  # 83 resumed steps x 9 s stay below the 15-min slice
        ref = {"step": 0, "acc": 1, "rng": random.Random(5).getstate()}
        run(ref, None, n, None)
        ckpt = self.runs / "ckpt.pkl"
        state = {"step": 0, "acc": 1, "rng": random.Random(5).getstate()}
        s1 = self.open()
        mark = []
        orig = s1.yield_reasons

        def reasons(now):
            if state["step"] >= 37 and not mark:  # a peer marker appears mid-batch
                mark.append(1)
                LN.waiting_path("bwm").write_text("{}", encoding="utf-8")
            return orig(now)
        s1.yield_reasons = reasons
        self.assertFalse(run(state, s1, n, ckpt))
        self.assertIsNone(LN.read_lock())
        LN.waiting_path("bwm").unlink()
        resumed = pickle.loads(ckpt.read_bytes())
        self.assertNotEqual(resumed["step"] % 10, 0)  # the yield happened mid-batch (a 60-s check, not a batch end)
        s2 = self.open()
        self.assertTrue(run(resumed, s2, n, ckpt))
        s2.release()
        self.assertEqual((resumed["step"], resumed["acc"]), (ref["step"], ref["acc"]))


class TestUnitPlan(unittest.TestCase):
    def test_gap_bounds(self):
        self.assertEqual(LE.check_gaps([30] * 4), [60.0, 60.0])
        self.assertEqual(LE.check_gaps([29] * 6), [87.0, 87.0])
        self.assertEqual(LE.check_gaps([10, 10], boundaries=[0]), [10.0])
        self.assertLessEqual(LE.max_check_gap([30.0] * 50), 90.0)
        self.assertLessEqual(LE.max_check_gap([30.0] * 50, scale=2.0), 120.0)

    def test_every_planned_unit_injected(self):
        names = set(LE.plan_costs())
        self.assertEqual(len(names), len(LE.STEP_UNITS) + len(LE.EVAL_ARMS))
        rng = random.Random(3)
        for _ in range(20):  # injected step and cell costs for every (model size, B, batch) and evaluation unit
            costs = {k: rng.uniform(0.01, 30.0) for k in names}
            rep = LE.validate_units(costs)
            self.assertTrue(rep["ok"], rep)
            self.assertLessEqual(rep["max_gap_s"], 90.0)
            self.assertLessEqual(rep["max_gap_slow_s"], 120.0)
        costs = {k: 30.0 for k in names}
        self.assertTrue(LE.validate_units(costs)["ok"])
        costs["cell:ENUM-MIS±"] = 30.5  # a unit projected > 30 s is rejected
        rep = LE.validate_units(costs)
        self.assertFalse(rep["ok"])
        self.assertEqual(set(rep["rejected"]), {"cell:ENUM-MIS±"})

    def test_plan_costs_from_rates(self):
        c = LE.plan_costs(rate_gflops=19.0)
        self.assertAlmostEqual(c[LE.step_name("PGV256", 64, 256)], 6 * 151_553 * 256 * 64 / 19e9)
        self.assertTrue(LE.validate_units(c)["ok"])  # the pessimistic planning rate fits
        c2 = LE.plan_costs(step_s={LE.step_name("SF", 1, 256): 0.5}, cell_s={"cell:SF-GPI": 3.0})
        self.assertEqual((c2[LE.step_name("SF", 1, 256)], c2["cell:SF-GPI"]), (0.5, 3.0))

    def test_reference_shas(self):
        shas = LE.reference_shas()
        self.assertEqual(set(shas), {"experiments/RCX-E05/lane_e05.py", "rcx/lane.py", "rcx/guard.py"})
        self.assertTrue(all(isinstance(v, str) and len(v) == 64 for v in shas.values()))

    def test_no_outcome_imports(self):
        src = (ROOT / "experiments" / "RCX-E05" / "lane_e05.py").read_text(encoding="utf-8")
        for mod in ("soft", "families_e05", "enum_e05", "learners_e05", "run_e05", "analyze_e05"):
            self.assertNotRegex(src, rf"(?m)^\s*(from|import)\s+\S*{mod}")
        self.assertNotIn("LN.acquire(", src)  # never the marker-writing acquire


if __name__ == "__main__":
    unittest.main()
