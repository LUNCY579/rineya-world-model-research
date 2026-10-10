"""Unit tests for rcx.guard (fake log and flag in a temp dir; all waits are sub-second)."""
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
from rcx import guard as G  # noqa: E402


def log_line(tz, ram=8.0, when=None, suffix=''):
    return f'{(when or datetime.now()):%H:%M:%S} t=70 util=50 power=40.0 ac=True ram={ram} cpu=30 tz01={tz}{suffix}\n'


class GuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / 'thermal_log.txt'
        self.flag = Path(self._tmp.name) / '.thermal_pause'
        self.msgs = []

    def write(self, text, mode='a'):
        with open(self.log_path, mode, newline='\n', encoding='utf-8') as handle:
            handle.write(text)

    def later(self, delay, fn):
        timer = threading.Timer(delay, fn)
        timer.start()
        self.addCleanup(timer.cancel)

    def guard(self, **kw):
        kw = dict(dict(poll_s=0.02, ram_fn=lambda: None, max_pause_s=10.0), **kw)
        return G.ThermalGuard(log=self.msgs.append, log_path=self.log_path, flag_path=self.flag, **kw)

    def joined(self):
        return '\n'.join(self.msgs)

    def test_cool_does_not_block(self):
        self.write(log_line(90.1))
        g = self.guard()
        t0 = time.monotonic()
        self.assertEqual(g.check(), 0.0)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual((g.paused_s, g.pauses, self.msgs), (0.0, 0, []))

    def test_hot_blocks_until_log_cools_with_hysteresis(self):
        self.write(log_line(98.1))
        self.later(0.25, lambda: self.write(log_line(95.1)))
        self.later(0.50, lambda: self.write(log_line(92.1)))
        g = self.guard()
        slept = g.check()
        self.assertGreaterEqual(slept, 0.45)
        self.assertLess(slept, 5.0)
        self.assertAlmostEqual(g.paused_s, slept)
        self.assertEqual(g.pauses, 1)
        self.assertIn('PAUSE: tz01=98.1>=97', self.joined())
        self.assertIn('RESUME after', self.joined())
        self.assertIn('total paused', self.joined())

    def test_flag_blocks_until_removed(self):
        self.write(log_line(85.1))
        self.flag.write_text('tz01 98.1', encoding='utf-8')
        self.later(0.3, self.flag.unlink)
        g = self.guard()
        self.assertGreaterEqual(g.check(), 0.25)
        self.assertIn('pause flag (tz01 98.1)', self.joined())

    def scripted_sleep(self, *actions):
        """Fake sleep running actions[i] on the i-th call; records the calls and fails fast if they run out."""
        sleeps = []

        def sleep(s):
            if len(sleeps) >= len(actions):
                raise AssertionError('guard still paused after the scripted actions')
            actions[len(sleeps)]()
            sleeps.append(s)
        return sleep, sleeps

    def test_flag_pause_resumes_when_flag_clears_despite_warm_tz(self):
        self.write(log_line(95.1))
        self.flag.write_text('77', encoding='utf-8')
        sleep, sleeps = self.scripted_sleep(lambda: None, self.flag.unlink)
        self.guard(sleep=sleep).check()
        self.assertEqual(len(sleeps), 2)
        self.assertNotIn('guard error', self.joined())

    def test_ram_pause_resumes_without_tz_hysteresis(self):
        self.write(log_line(95.1))
        values = iter([2.5, 4.2])
        sleep, sleeps = self.scripted_sleep(lambda: None)
        self.guard(ram_fn=lambda: next(values), sleep=sleep).check()
        self.assertEqual(len(sleeps), 1)
        self.assertNotIn('guard error', self.joined())

    def test_tz_reaching_pause_during_flag_pause_latches_hysteresis(self):
        self.write(log_line(95.1))
        self.flag.write_text('77', encoding='utf-8')
        sleep, sleeps = self.scripted_sleep(lambda: self.write(log_line(97.1)),
                                            lambda: (self.flag.unlink(), self.write(log_line(95.1))),
                                            lambda: self.write(log_line(92.1)))
        self.guard(sleep=sleep).check()
        self.assertEqual(len(sleeps), 3)
        self.assertNotIn('guard error', self.joined())

    def test_flag_ignored_when_not_honored(self):
        self.write(log_line(85.1))
        self.flag.write_text('battery', encoding='utf-8')
        self.assertEqual(self.guard(honor_flag=False).check(), 0.0)

    def test_stale_log_throttles_and_warns_once(self):
        self.write(log_line(99.1, when=datetime.now() - timedelta(seconds=600)))
        self.flag.write_text('tz01 98.1', encoding='utf-8')
        sleeps = []
        g = self.guard(sleep=sleeps.append)
        self.assertEqual(g.check(), 5.0)
        self.assertEqual(g.check(), 5.0)
        self.assertEqual(sleeps, [5.0, 5.0])
        self.assertEqual((g.throttled_s, g.paused_s, g.pauses), (10.0, 10.0, 0))
        warnings = [m for m in self.msgs if 'WARNING log stale' in m]
        self.assertEqual(len(warnings), 1)
        self.assertIn('pause flag present but untrusted', warnings[0])
        self.flag.unlink()
        self.write(log_line(90.1))
        self.assertEqual(g.check(), 0.0)
        self.assertIn('log fresh again', self.msgs[-1])

    def test_stale_log_real_sleep_is_bounded(self):
        self.write(log_line(99.1, when=datetime.now() - timedelta(seconds=600)))
        g = self.guard(throttle_s=0.05)
        t0 = time.monotonic()
        g.check()
        self.assertTrue(0.04 <= time.monotonic() - t0 < 1.0)

    def test_unreadable_or_garbage_log_throttles(self):
        sleeps = []
        g = self.guard(sleep=sleeps.append)
        g.check()
        self.write('12:00:00 power request HELD (AC + GPU lease held)\nnot a line\n')
        g.check()
        self.assertEqual(sleeps, [5.0, 5.0])
        self.assertIn('log unreadable', self.joined())

    def test_ram_floor_pauses_with_hysteresis(self):
        self.write(log_line(85.1))
        values = iter([2.5, 3.5, 3.9, 4.2])
        sleeps = []
        g = self.guard(ram_fn=lambda: next(values), sleep=sleeps.append)
        g.check()
        self.assertEqual(len(sleeps), 3)
        self.assertIn('ram=2.5GB<3', self.joined())

    def test_log_ram_used_when_live_ram_unknown(self):
        self.write(log_line(85.1, ram=2.0))
        g = self.guard(sleep=lambda s: self.write(log_line(85.1, ram=5.0)))
        g.check()
        self.assertIn('ram=2.0GB<3', self.joined())
        self.assertEqual(g.pauses, 1)

    def test_max_pause_caps_blocking(self):
        self.write(log_line(98.1))
        g = self.guard(max_pause_s=0.1)
        slept = g.check()
        self.assertTrue(0.1 <= slept < 2.0)
        self.assertIn('max_pause_s=0.1 reached', self.joined())

    def test_internal_error_never_raises(self):
        self.write(log_line(85.1))
        sleeps = []

        def boom():
            raise RuntimeError('sensor')
        g = self.guard(ram_fn=boom, sleep=sleeps.append)
        self.assertEqual(g.check(), 5.0)
        self.assertIn('guard error', self.joined())

    def test_no_subprocess_spawned(self):
        self.write(log_line(98.1))
        self.later(0.1, lambda: self.write(log_line(90.1)))
        with mock.patch('subprocess.Popen', side_effect=AssertionError('spawn')), \
                mock.patch('os.system', side_effect=AssertionError('spawn')):
            g = G.ThermalGuard(log=self.msgs.append, log_path=self.log_path, flag_path=self.flag, poll_s=0.02,
                               max_pause_s=10.0)
            g.check()
        self.assertNotIn('guard error', self.joined())

    def test_latest_reading_reads_only_tail(self):
        old = ''.join(log_line(80.1, when=datetime.now() - timedelta(seconds=30)) for _ in range(80))
        self.write(old + log_line(96.1, ram=5.5, suffix=' PAUSE') + '12:00:00 AC power: True\n')
        r = G.latest_reading(self.log_path)
        self.assertEqual((r['t'], r['tz01'], r['ram']), (70.0, 96.1, 5.5))
        self.assertLess(r['age_s'], 5)
        self.assertEqual(set(r), {'t', 'tz01', 'ram', 'ts', 'age_s'})
        self.write('12:00:01 power request HELD (AC + GPU lease held): idle sleep and display-off blocked\n' * 40)
        self.assertIn('tz01=96.1', self.log_path.read_text(encoding='utf-8'))
        self.assertIsNone(G.latest_reading(self.log_path))
        self.assertEqual(G.latest_reading(self.log_path, tail_bytes=8192)['tz01'], 96.1)

    def test_unterminated_last_line_ignored(self):
        self.write(log_line(98.1) + log_line(90.1).rstrip('\n')[:-2])
        self.assertEqual(G.latest_reading(self.log_path)['tz01'], 98.1)

    def test_day_old_log_is_stale(self):
        self.write(log_line(90.1))
        day_ago = time.time() - 86400
        os.utime(self.log_path, (day_ago, day_ago))
        self.assertGreater(G.latest_reading(self.log_path)['age_s'], 86000)
        sleeps = []
        self.guard(sleep=sleeps.append).check()
        self.assertEqual(sleeps, [5.0])

    def test_tod_age_handles_midnight_and_skew(self):
        now = datetime(2026, 10, 10, 0, 0, 10)
        self.assertEqual(G.tod_age_s('23:59:50', now), 20)
        self.assertEqual(G.tod_age_s('00:00:05', now), 5)
        self.assertEqual(G.tod_age_s('00:01:00', now), 0.0)
        self.assertEqual(G.tod_age_s('12:00:00', now), 43210)
        self.assertEqual(G.tod_age_s('22:31:40', datetime(2026, 10, 9, 22, 34, 40)), 180)
        with self.assertRaises(ValueError):
            G.tod_age_s('25:00:00', now)

    def test_parse_line(self):
        r = G.parse_line('22:31:40 t=70 util=77 power=43.01 ac=True ram=5.5 cpu=37 tz01=95.1')
        self.assertEqual(r, dict(ts='22:31:40', t=70.0, tz01=95.1, ram=5.5))
        self.assertIsNone(G.parse_line('14:18:12 t=86 util=57 power=59.97 PAUSE'))
        self.assertIsNone(G.parse_line('22:02:36 AC power: True'))

    @unittest.skipUnless(sys.platform == 'win32', 'GlobalMemoryStatusEx is Windows-only')
    def test_available_ram_gb(self):
        ram = G.available_ram_gb()
        self.assertIsInstance(ram, float)
        self.assertTrue(0.0 < ram < 4096.0)

    @unittest.skipUnless(G.DEFAULT_LOG.exists(), 'shared watchdog log not present')
    def test_real_watchdog_log_parses(self):
        r = G.latest_reading()
        self.assertIsNotNone(r)
        self.assertIsInstance(r['tz01'], float)


if __name__ == '__main__':
    unittest.main()
