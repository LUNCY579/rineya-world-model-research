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

    def test_marker_removed_on_acquire(self):
        L.waiting_path("rcx").write_text("{}", encoding="utf-8")
        L.acquire("job", self.me, 1, poll_s=0, log=lambda m: None)
        self.assertFalse(L.waiting_path("rcx").exists())


if __name__ == "__main__":
    unittest.main()
