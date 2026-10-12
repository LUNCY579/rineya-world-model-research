import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcx import lane as L  # noqa: E402


class TestLane(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RINEYA_LANE_DIR"] = self.tmp.name
        self.me = os.getpid()

    def tearDown(self):
        os.environ.pop("RINEYA_LANE_DIR", None)
        self.tmp.cleanup()

    def test_acquire_release_free_lane(self):
        rec = L.acquire("job", self.me, 1, need_ram=1.0, ram_fn=lambda: 99.0, poll_s=0)
        self.assertEqual(rec["holder"], "rcx")
        self.assertEqual(L.read_lock()["pid"], self.me)
        self.assertFalse(L.release(self.me + 1))
        self.assertTrue(L.release(self.me))
        self.assertIsNone(L.read_lock())

    def test_waits_on_peer_and_writes_marker(self):
        L.lock_path().write_text(json.dumps({"holder": "bwm", "job": "4B", "pid": self.me}), encoding="utf-8")
        with self.assertRaises(TimeoutError):
            L.acquire("job", self.me, 1, poll_s=0, max_wait_s=0, log=lambda m: None)
        self.assertTrue(L.waiting_path("rcx").exists())

    def test_low_ram_free_lane_waits_without_marker(self):
        with self.assertRaises(TimeoutError):
            L.acquire("job", self.me, 1, need_ram=12.5, ram_fn=lambda: 9.0, poll_s=0, max_wait_s=0, log=lambda m: None)
        self.assertFalse(L.waiting_path("rcx").exists())
        self.assertIsNone(L.read_lock())

    def test_stale_lock_is_reclaimed(self):
        dead = 999999
        while L.pid_alive(dead):
            dead += 1
        L.lock_path().write_text(json.dumps({"holder": "bwm", "job": "x", "pid": dead}), encoding="utf-8")
        rec = L.acquire("job", self.me, 1, poll_s=0, log=lambda m: None)
        self.assertEqual(rec["holder"], "rcx")
        self.assertTrue(list(Path(self.tmp.name).glob("rineya-cpu-lane.json.stale-*.bak")))

    def test_window_rule(self):
        import datetime as dt
        ws = [(dt.datetime(2026, 10, 12, 23, 0), dt.datetime(2026, 10, 13, 7, 0))]
        self.assertTrue(L.window_ok(ws, 170, dt.datetime(2026, 10, 12, 23, 0))[0])
        self.assertTrue(L.window_ok(ws, 170, dt.datetime(2026, 10, 13, 4, 5))[0])     # ends 06:55 exactly
        self.assertFalse(L.window_ok(ws, 170, dt.datetime(2026, 10, 13, 4, 6))[0])    # would end after 06:55
        self.assertFalse(L.window_ok(ws, 10, dt.datetime(2026, 10, 12, 22, 59))[0])   # before the window
        self.assertFalse(L.window_ok(ws, 10, dt.datetime(2026, 10, 13, 7, 0))[0])     # after it
        self.assertIn("never", L.window_ok(ws, 10, dt.datetime(2026, 10, 14, 0, 0))[1])

    def test_acquire_outside_window_waits_without_marker(self):
        import datetime as dt
        ws = [(dt.datetime(2026, 10, 12, 23, 0), dt.datetime(2026, 10, 13, 7, 0))]
        L.lock_path().write_text(json.dumps({"holder": "bwm", "job": "x", "pid": self.me}), encoding="utf-8")
        with self.assertRaises(TimeoutError):
            L.acquire("job", self.me, 170, poll_s=0, max_wait_s=0, log=lambda m: None, windows=ws,
                      now_fn=lambda: dt.datetime(2026, 10, 12, 15, 0))
        self.assertFalse(L.waiting_path("rcx").exists())   # no marker outside a window, even if a peer holds it
        L.lock_path().unlink()
        rec = L.acquire("job", self.me, 170, poll_s=0, log=lambda m: None, windows=ws,
                        now_fn=lambda: dt.datetime(2026, 10, 12, 23, 30))
        self.assertEqual(rec["holder"], "rcx")

    def test_load_windows(self):
        p = Path(self.tmp.name) / "w.json"
        p.write_text(json.dumps({"end_margin_min": 5, "windows": [["2026-10-12T23:00", "2026-10-13T07:00"]]}),
                     encoding="utf-8")
        ws, m = L.load_windows(p)
        self.assertEqual((len(ws), m, ws[0][0].hour, ws[0][1].hour), (1, 5.0, 23, 7))

    def test_marker_removed_on_acquire(self):
        L.waiting_path("rcx").write_text("{}", encoding="utf-8")
        L.acquire("job", self.me, 1, poll_s=0, log=lambda m: None)
        self.assertFalse(L.waiting_path("rcx").exists())


if __name__ == "__main__":
    unittest.main()
