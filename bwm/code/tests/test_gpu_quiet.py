"""D-84 quiet windows in scripts/gpu_run.py: no NEW BWM GPU job starts inside an agreed GPU-idle window; after a window,
no GPU load beside a peer's straddling CPU-lane block, and RCWM's guard-relaunched chain gets its owed turn.

Fake clocks and temp paths throughout: nothing here touches the real lock, turn, lane or waiting files, or the card.
Run: python -B -m pytest -q tests/test_gpu_quiet.py   (GPU_RUN_UNDER_TEST=<path> tests a candidate before install)
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_PID = 999_001   # stands for a live peer process (gpu_lease._pid_alive is patched)


@pytest.fixture()
def gr(tmp_path, monkeypatch):
    src = Path(os.environ.get("GPU_RUN_UNDER_TEST", ROOT / "scripts" / "gpu_run.py"))  # a candidate before install
    loader = importlib.machinery.SourceFileLoader("gpu_run_under_test", str(src))
    spec = importlib.util.spec_from_loader("gpu_run_under_test", loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "WAITING", tmp_path / "waiting.bwm.json")
    monkeypatch.setattr(m, "PEER_WAITING", tmp_path / "waiting.rcwm.json")
    monkeypatch.setattr(m, "TURN", tmp_path / "turn.json")
    monkeypatch.setattr(m, "QUIET", tmp_path / "gpu_quiet_windows.json")
    monkeypatch.setattr(m, "QUIET_RELEASE", tmp_path / "release.json")
    monkeypatch.setattr(m, "LANE", tmp_path / "cpu-lane.json")
    monkeypatch.setattr(m, "YIELD", tmp_path / ".yield.json")
    monkeypatch.setattr(m.gpu_lease, "LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(m.gpu_lease, "_pid_alive", lambda pid: pid == LIVE_PID)
    return m


CFG = {"windows": [{"start": "2026-10-12T23:00", "end": "2026-10-13T07:00"},
                   {"start": "2026-10-13T23:00", "end": "2026-10-14T07:00"}]}


def t(gr, s):
    return gr._local_unix(s)


class Clock:
    def __init__(self, now, hook=None):
        self.now, self.sleeps, self.hook = now, [], hook

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
        if self.hook:
            self.hook(self)


# ---------------------------------------------------------------------------------------------- quiet_window_end


def test_inside_window_returns_its_end(gr):
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:30"), CFG) == t(gr, "2026-10-13T07:00")
    assert gr.quiet_window_end(t(gr, "2026-10-13T23:30"), CFG) == t(gr, "2026-10-14T07:00")


def test_start_inclusive_end_exclusive(gr):
    assert gr.quiet_window_end(t(gr, "2026-10-12T23:00"), CFG) == t(gr, "2026-10-13T07:00")
    assert gr.quiet_window_end(t(gr, "2026-10-13T07:00"), CFG) is None
    assert gr.quiet_window_end(t(gr, "2026-10-12T23:00") - 1, CFG) is None


def test_outside_and_after_the_list_expires(gr):
    assert gr.quiet_window_end(t(gr, "2026-10-13T12:00"), CFG) is None
    assert gr.quiet_window_end(t(gr, "2026-10-15T02:00"), CFG) is None  # the list is dated: it expires by itself


@pytest.mark.parametrize("bad", [None, ["not", "a", "dict"], {"windows": None}, {"windows": 5}, {"windows": True},
                                 {"windows": "2026-10-12T23:00"}, {"windows": {"start": "x"}}, {}])
def test_fail_open_on_malformed_lists_never_raises(gr, bad):
    now = t(gr, "2026-10-13T02:30")
    if bad is None:
        assert gr.quiet_window_end(now) is None                   # no file
    else:
        assert gr.quiet_window_end(now, bad) is None


def test_bad_entries_skipped_good_ones_apply(gr):
    bad = {"windows": [{"start": "23:00"}, {"start": "2026-10-12 23:00", "end": "x"}, 7,
                       {"start": "2026-10-12T23:00", "end": "2026-10-13T07:00"}]}
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:30"), bad) == t(gr, "2026-10-13T07:00")


def test_unreadable_file_fails_open(gr):
    gr.QUIET.write_text("{not json")
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:30")) is None


def test_file_with_bom_is_read(gr):
    gr.QUIET.write_bytes(b"\xef\xbb\xbf" + json.dumps(CFG).encode())  # Windows PowerShell 5.1 writes one
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:30")) == t(gr, "2026-10-13T07:00")


def test_release_marker_ends_all_windows(gr):
    gr.QUIET_RELEASE.write_text("{}")
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:30"), CFG) is None


def test_last_window_end(gr):
    assert gr.last_window_end(t(gr, "2026-10-12T12:00"), CFG) is None
    assert gr.last_window_end(t(gr, "2026-10-13T02:00"), CFG) is None
    assert gr.last_window_end(t(gr, "2026-10-13T07:00"), CFG) == t(gr, "2026-10-13T07:00")
    assert gr.last_window_end(t(gr, "2026-10-14T09:00"), CFG) == t(gr, "2026-10-14T07:00")


# ------------------------------------------------------------------------------------------- report_quiet_list


def test_report_quiet_list_says_what_it_read(gr, capsys):
    now = t(gr, "2026-10-12T12:00")
    gr.report_quiet_list(now)
    assert "no windows file" in capsys.readouterr().out
    gr.QUIET.write_text("{not json")
    gr.report_quiet_list(now)
    assert "UNREADABLE" in capsys.readouterr().out
    gr.QUIET.write_text(json.dumps({"windows": CFG["windows"] + [{"start": "2026-10-14T23:00:00", "end": "x"}]}))
    gr.report_quiet_list(now)
    out = capsys.readouterr().out
    assert "2 window(s), 1 skipped" in out and "2026-10-12 23:00" in out and "RELEASED" not in out
    gr.QUIET_RELEASE.write_text("{}")
    gr.report_quiet_list(now)
    assert "RELEASED by RCX" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ honour_quiet_windows


def test_no_wait_outside_a_window(gr):
    c = Clock(t(gr, "2026-10-13T12:00"))
    assert gr.honour_quiet_windows("x", clock=c, sleep=c.sleep, cfg=CFG) is False
    assert c.sleeps == [] and not gr.WAITING.exists()


def test_waits_until_the_end_in_short_polls_and_keeps_the_waiting_marker(gr):
    c = Clock(t(gr, "2026-10-13T06:50"))
    assert gr.honour_quiet_windows("bundleJ", clock=c, sleep=c.sleep, cfg=CFG) is True
    assert t(gr, "2026-10-13T07:00") <= c.now < t(gr, "2026-10-13T07:01")
    assert max(c.sleeps) <= 60.0 and min(c.sleeps) >= 1.0
    w = json.loads(gr.WAITING.read_text())
    assert w["label"] == "bwm:quiet-window" and isinstance(w["pid"], int)


def test_release_marker_mid_wait_stops_early_and_says_so(gr, capsys):
    def hook(c):
        if len(c.sleeps) == 3:
            gr.QUIET_RELEASE.write_text("{}")

    c = Clock(t(gr, "2026-10-13T01:00"), hook)
    assert gr.honour_quiet_windows("x", clock=c, sleep=c.sleep, cfg=CFG) is True
    assert len(c.sleeps) == 3 and c.now < t(gr, "2026-10-13T01:05")
    assert "ended EARLY" in capsys.readouterr().out


def test_normal_end_is_not_reported_as_early(gr, capsys):
    c = Clock(t(gr, "2026-10-13T06:58"))
    gr.honour_quiet_windows("x", clock=c, sleep=c.sleep, cfg=CFG)
    assert "EARLY" not in capsys.readouterr().out


def test_overlong_window_is_a_typo_and_skipped(gr):
    typo = {"windows": [{"start": "2026-10-12T23:00", "end": "2027-10-13T07:00"}]}
    assert gr.quiet_window_end(t(gr, "2026-10-13T02:00"), typo) is None and gr._windows(typo) == []


def test_list_is_reread_while_waiting(gr):
    gr.QUIET.write_text(json.dumps(CFG))

    def hook(c):
        if len(c.sleeps) == 2:
            gr.QUIET.write_text(json.dumps({"windows": []}))  # the lead cancels the windows

    c = Clock(t(gr, "2026-10-13T01:00"), hook)
    assert gr.honour_quiet_windows("x", clock=c, sleep=c.sleep) is True
    assert len(c.sleeps) == 2


def test_waiting_marker_write_failure_does_not_fail_the_job(gr, monkeypatch, tmp_path):
    d = tmp_path / "a_directory"
    d.mkdir()
    monkeypatch.setattr(gr, "WAITING", d)          # write_text on a directory raises an OSError
    c = Clock(t(gr, "2026-10-13T06:59"))
    assert gr.honour_quiet_windows("x", clock=c, sleep=c.sleep, cfg=CFG) is True


# ------------------------------------------------------------------------------------------------ honour_yield


def _yield_file(gr):
    gr.YIELD.write_text(json.dumps({"label_prefix": "rcwm-t1:", "max_wait_s": 21600, "give_up_if_not_seen_s": 5400}))


def test_yield_is_not_used_up_inside_a_window(gr):
    """Post-reboot yield written at 22:15 (I-16/I-17 procedure): RCWM's guard keeps it off the card all night, so
    without D-84 freezing the yield would give up at 23:45 and BWM could take a second turn in a row at 07:00."""
    gr.QUIET.write_text(json.dumps(CFG))
    _yield_file(gr)

    def hook(c):
        if c.now < t(gr, "2026-10-13T06:59"):
            assert gr.YIELD.exists()

    c = Clock(t(gr, "2026-10-12T22:15"), hook)
    gr.honour_yield(clock=c, sleep=c.sleep)
    assert not gr.YIELD.exists()
    assert c.now >= t(gr, "2026-10-13T08:30")       # 07:00 + 5400 s: the give-up clock starts when RCWM can run


@pytest.mark.parametrize("content", [b"\xef\xbb\xbf{\"label_prefix\": \"rcwm-t1:\", \"give_up_if_not_seen_s\": 60}",
                                     b"{\"label_prefix\": \"rcwm-t1:\", \"max_wait_s\": null, \"give_up_if_not_seen_s\": 60}"])
def test_yield_file_with_bom_or_nulls_still_works(gr, content):
    gr.YIELD.write_bytes(content)
    c = Clock(t(gr, "2026-10-12T12:00"))
    gr.honour_yield(clock=c, sleep=c.sleep)
    assert not gr.YIELD.exists() and t(gr, "2026-10-12T12:01") <= c.now < t(gr, "2026-10-12T12:02")


@pytest.mark.parametrize("content", [b"{not json", b"{}", b"[1]", b"{\"label\": null, \"label_prefix\": null}"])
def test_unusable_yield_file_is_ignored_not_fatal(gr, content):
    gr.YIELD.write_bytes(content)
    c = Clock(t(gr, "2026-10-12T12:00"))
    gr.honour_yield(clock=c, sleep=c.sleep)
    assert c.sleeps == []


def test_yield_without_windows_gives_up_as_before(gr):
    _yield_file(gr)
    c = Clock(t(gr, "2026-10-12T22:15"))
    gr.honour_yield(clock=c, sleep=c.sleep)
    assert t(gr, "2026-10-12T23:45") <= c.now < t(gr, "2026-10-12T23:46")


def test_yield_ends_when_the_peer_has_had_its_turn_after_the_window(gr):
    gr.QUIET.write_text(json.dumps(CFG))
    _yield_file(gr)

    def hook(c):
        if t(gr, "2026-10-13T07:01") <= c.now < t(gr, "2026-10-13T07:30"):
            gr.gpu_lease.LOCK.write_text(json.dumps({"label": "rcwm-t1:lik:supa"}))
        elif c.now >= t(gr, "2026-10-13T07:30") and gr.gpu_lease.LOCK.exists():
            gr.gpu_lease.LOCK.unlink()

    c = Clock(t(gr, "2026-10-13T02:00"), hook)
    gr.honour_yield(clock=c, sleep=c.sleep)
    assert t(gr, "2026-10-13T07:30") <= c.now < t(gr, "2026-10-13T07:31") and not gr.YIELD.exists()


# ------------------------------------------------------------------------------------------- post_window_grace


def test_grace_waits_for_rcwm_relaunch_when_bwm_went_last(gr):
    gr.TURN.write_text(json.dumps({"last": "bwm"}))
    gr.PEER_WAITING.write_text(json.dumps({"pid": 12345, "label": "rcwm-t1:killed"}))   # dead pid (guard killed it)

    def hook(c):
        if c.now >= t(gr, "2026-10-13T07:02"):
            gr.PEER_WAITING.write_text(json.dumps({"pid": LIVE_PID, "label": "rcwm-t1:pvr"}))

    c = Clock(t(gr, "2026-10-13T07:00"), hook)
    assert gr.post_window_grace(clock=c, sleep=c.sleep, cfg=CFG) is True
    assert t(gr, "2026-10-13T07:02") <= c.now < t(gr, "2026-10-13T07:03")


def test_grace_is_bounded(gr):
    gr.TURN.write_text(json.dumps({"last": "bwm"}))
    c = Clock(t(gr, "2026-10-13T07:00"))
    assert gr.post_window_grace(clock=c, sleep=c.sleep, cfg=CFG) is True
    assert t(gr, "2026-10-13T07:15") <= c.now < t(gr, "2026-10-13T07:16")


@pytest.mark.parametrize("state", ["turn_rcwm", "rcwm_lock", "daytime", "before_any_window", "mid_window"])
def test_no_grace_when_not_owed(gr, state):
    gr.TURN.write_text(json.dumps({"last": "rcwm" if state == "turn_rcwm" else "bwm"}))
    if state == "rcwm_lock":
        gr.gpu_lease.LOCK.write_text(json.dumps({"label": "rcwm-t1:pvr"}))
    now = {"daytime": "2026-10-13T12:00", "before_any_window": "2026-10-12T20:00",
           "mid_window": "2026-10-13T02:00"}.get(state, "2026-10-13T07:00")
    c = Clock(t(gr, now))
    assert gr.post_window_grace(clock=c, sleep=c.sleep, cfg=CFG) is False and c.sleeps == []


# -------------------------------------------------------------------------------------- hold_for_peer_lane_after_window


def _lane(gr, holder="rcx", pid=LIVE_PID, started="2026-10-13T05:00"):
    gr.LANE.write_text(json.dumps({"holder": holder, "job": "RCX E01R train a3", "pid": pid,
                                   "started": t(gr, started), "expected_end": t(gr, started) + 12000}))


def test_lane_hold_until_the_straddling_block_releases(gr):
    _lane(gr)

    def hook(c):
        if c.now >= t(gr, "2026-10-13T08:20"):
            gr.LANE.unlink(missing_ok=True)

    gr.WAITING.write_text(json.dumps({"pid": os.getpid(), "label": "bwm:quiet-window"}))   # from the window wait
    marker_during = []

    def hook2(c):
        marker_during.append(gr.WAITING.exists())
        hook(c)

    c = Clock(t(gr, "2026-10-13T07:00"), hook2)
    assert gr.hold_for_peer_lane_after_window(clock=c, sleep=c.sleep, cfg=CFG) is True
    assert t(gr, "2026-10-13T08:20") <= c.now < t(gr, "2026-10-13T08:22")
    # BWM's own courtesy: it is not advertised as waiting meanwhile (RCWM's yield to it is capped at 2700 s)
    assert marker_during and not any(marker_during)


def test_lane_hold_leaves_a_foreign_waiting_marker_alone(gr):
    _lane(gr)
    gr.WAITING.write_text(json.dumps({"pid": 4242, "label": "bwm:other-launcher"}))
    c = Clock(t(gr, "2026-10-13T07:00"), lambda c: gr.LANE.unlink(missing_ok=True))
    assert gr.hold_for_peer_lane_after_window(clock=c, sleep=c.sleep, cfg=CFG) is True
    assert gr.WAITING.exists()


def test_pid_reuse_is_not_a_live_holder(gr, monkeypatch):
    import time
    monkeypatch.setattr(gr.gpu_lease, "_pid_alive", lambda pid: True)
    assert gr._pid_alive_since(os.getpid(), time.time() + 100) is True      # our process predates that lock
    assert gr._pid_alive_since(os.getpid(), 0.0) is False                   # created long after the lock: reused pid


def test_lane_hold_is_bounded(gr):
    _lane(gr)
    c = Clock(t(gr, "2026-10-13T07:00"))
    assert gr.hold_for_peer_lane_after_window(clock=c, sleep=c.sleep, cfg=CFG) is True
    assert t(gr, "2026-10-13T11:00") <= c.now < t(gr, "2026-10-13T11:02")


@pytest.mark.parametrize("case", ["started_after_window", "holder_bwm", "holder_other_party", "released", "dead_pid",
                                  "no_lane", "daytime_later", "inside_window", "no_started_field"])
def test_no_lane_hold(gr, case):
    if case == "started_after_window":
        _lane(gr, started="2026-10-13T07:05")
    elif case == "holder_bwm":
        _lane(gr, holder="bwm")
    elif case == "holder_other_party":
        _lane(gr, holder="dsv3wm")
    elif case == "released":
        _lane(gr)
        gr.QUIET_RELEASE.write_text("{}")
    elif case == "dead_pid":
        _lane(gr, pid=4321)
    elif case == "no_started_field":
        gr.LANE.write_text(json.dumps({"holder": "rcx", "pid": LIVE_PID}))
    elif case != "no_lane":
        _lane(gr)
    now = {"daytime_later": "2026-10-13T11:30", "inside_window": "2026-10-13T03:00",
           "started_after_window": "2026-10-13T07:10"}.get(case, "2026-10-13T07:00")
    c = Clock(t(gr, now))
    assert gr.hold_for_peer_lane_after_window(clock=c, sleep=c.sleep, cfg=CFG) is False and c.sleeps == []


# ----------------------------------------------------------------------------------------- acquire_outside_quiet


def test_acquire_without_window_is_the_old_path(gr, monkeypatch):
    monkeypatch.setattr(gr, "wait_for_turn", lambda label: None)
    c = Clock(t(gr, "2026-10-13T12:00"))
    calls = []
    tok = gr.acquire_outside_quiet("j", "bwm:j", "job", acquire=lambda *a, **k: calls.append(k) or "T1",
                                   release=lambda tok: pytest.fail("released"), clock=c, sleep=c.sleep)
    assert tok == "T1" and calls == [{"timeout_seconds": 6 * 3600}] and c.sleeps == []


def test_courtesy_gap_kept_after_our_own_turn(gr, monkeypatch):
    monkeypatch.setattr(gr, "wait_for_turn", lambda label: None)
    gr.TURN.write_text(json.dumps({"last": "bwm"}))
    c = Clock(t(gr, "2026-10-13T12:00"))
    gr.acquire_outside_quiet("j", "bwm:j", "job", acquire=lambda *a, **k: "T", release=None, clock=c, sleep=c.sleep)
    assert c.sleeps == [30]


def test_card_freed_inside_a_window_is_released_unused_then_reacquired_after_it(gr, monkeypatch):
    """Waiting for a peer's 4.5 h lease from 21:00 runs past 23:00: the job must not start at 01:30."""
    gr.QUIET.write_text(json.dumps(CFG))
    gr.TURN.write_text(json.dumps({"last": "rcwm"}))
    monkeypatch.setattr(gr, "wait_for_turn", lambda label: None)
    got, released = [], []

    def acquire(label, job, timeout_seconds):
        if not got:
            c.now = t(gr, "2026-10-13T01:30")      # the peer's lease ended inside the window
        got.append(c.now)
        return f"T{len(got)}"

    c = Clock(t(gr, "2026-10-12T21:00"))
    tok = gr.acquire_outside_quiet("bundleJ", "bwm:bundle-bundleJ", "job", acquire=acquire, release=released.append,
                                   clock=c, sleep=c.sleep)
    assert released == ["T1"]
    assert tok == "T2" and got[1] >= t(gr, "2026-10-13T07:00")
    assert not gr.TURN.read_text().count('"bwm"')   # releasing unused never writes turn.json


def test_morning_after_bwm_went_last_rcwm_goes_first(gr, monkeypatch):
    """Review finding (major): with turn.last == bwm and RCWM's waiter killed by its guard, BWM must not take the card at
    07:00:30 before RCWM's relaunched chain registers."""
    gr.QUIET.write_text(json.dumps(CFG))
    gr.TURN.write_text(json.dumps({"last": "bwm"}))
    gr.PEER_WAITING.write_text(json.dumps({"pid": 12345, "label": "rcwm-t1:killed"}))
    monkeypatch.setattr(gr, "wait_for_turn", lambda label: None)

    def hook(c):
        if c.now >= t(gr, "2026-10-13T07:01"):
            gr.gpu_lease.LOCK.write_text(json.dumps({"label": "rcwm-t1:pvr"}))   # RCWM claims at 07:01

    c = Clock(t(gr, "2026-10-13T06:40"), hook)
    got = []
    gr.acquire_outside_quiet("bundleJ", "bwm:bundle-bundleJ", "job",
                             acquire=lambda *a, **k: got.append(c.now) or "T", release=None, clock=c, sleep=c.sleep)
    assert got and got[0] >= t(gr, "2026-10-13T07:01")   # (the real acquire then queues behind RCWM's lease)


def test_post_acquire_check_error_fails_open_without_leaking_the_token(gr, monkeypatch):
    monkeypatch.setattr(gr, "wait_for_turn", lambda label: None)
    calls = {"n": 0}
    real = gr.quiet_window_end

    def flaky(now, cfg=None, release=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("boom")
        return real(now, cfg, release)

    monkeypatch.setattr(gr, "quiet_window_end", flaky)
    c = Clock(t(gr, "2026-10-13T12:00"))
    tok = gr.acquire_outside_quiet("j", "bwm:j", "job", acquire=lambda *a, **k: "T", release=None, clock=c,
                                   sleep=c.sleep)
    assert tok == "T"


# ---------------------------------------------------------------------------------------------------- main wiring


@pytest.mark.parametrize("child_rc", [0, 75, 3])
def test_main_wiring_passes_rc_records_worker_pid_bundle_fields_and_turn(gr, monkeypatch, child_rc):
    def fake_acquire(label, lease_label, job):
        assert lease_label == "bwm:bundle-bundleZ"
        gr.gpu_lease.LOCK.write_text(json.dumps({"token": "TOK", "label": lease_label, "worker_pids": []}))
        gr.WAITING.write_text("{}")
        return "TOK"

    released = []
    monkeypatch.setattr(gr, "acquire_outside_quiet", fake_acquire)
    monkeypatch.setattr(gr.gpu_lease, "release", released.append)
    seen = {}

    class P:
        pid = 4242

        def __init__(self, cmd, cwd, env):
            seen["cmd"], seen["env"] = cmd, env

        def wait(self):
            seen["lock"] = json.loads(gr.gpu_lease.LOCK.read_text())
            return child_rc

    monkeypatch.setattr(gr.subprocess, "Popen", P)
    monkeypatch.setattr(sys, "argv", ["gpu_run.py", "bundleZ", "--", "scripts/bundle.py", "runs/bundles/z.json"])
    with pytest.raises(SystemExit) as e:
        gr.main()
    assert e.value.code == child_rc            # orchestrate.py requeues on 75 and marks done only on 0
    assert released == ["TOK"]
    assert seen["lock"]["worker_pids"] == [4242] and seen["lock"]["bundle_budget_s"] == 16200
    assert "BWM_LEASE_ACQUIRED" in seen["env"] and seen["cmd"][1:] == ["scripts/bundle.py", "runs/bundles/z.json"]
    assert not gr.WAITING.exists()
    turn = json.loads(gr.TURN.read_text())
    assert turn["last"] == "bwm" and turn["label"] == "bwm:bundleZ" and "bundle_start" in turn


def test_main_survives_a_waiting_marker_that_cannot_be_removed(gr, monkeypatch, tmp_path):
    d = tmp_path / "marker_dir"
    d.mkdir()
    (d / "x").write_text("x")                    # a non-empty directory: unlink raises an OSError, not FileNotFound
    monkeypatch.setattr(gr, "WAITING", d)
    monkeypatch.setattr(gr, "acquire_outside_quiet", lambda *a: "TOK")
    monkeypatch.setattr(gr.gpu_lease, "release", lambda tok: None)

    class P:
        pid = 1

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(gr.subprocess, "Popen", P)
    monkeypatch.setattr(sys, "argv", ["gpu_run.py", "x", "--", "scripts/y.py"])
    with pytest.raises(SystemExit) as e:
        gr.main()
    assert e.value.code == 0


def test_main_releases_the_card_if_the_child_cannot_start(gr, monkeypatch):
    monkeypatch.setattr(gr, "acquire_outside_quiet", lambda *a: "TOK")
    released = []
    monkeypatch.setattr(gr.gpu_lease, "release", released.append)

    def boom(*a, **k):
        raise OSError("cannot start")

    monkeypatch.setattr(gr.subprocess, "Popen", boom)
    monkeypatch.setattr(sys, "argv", ["gpu_run.py", "x", "--", "scripts/y.py"])
    with pytest.raises(OSError):
        gr.main()
    assert released == ["TOK"] and not gr.TURN.exists()


# ------------------------------------------------------------------------------------------- installed windows file


def test_installed_windows_file_is_well_formed_and_parsed_as_written(gr):
    p = ROOT / "runs" / "gpu_quiet_windows.json"
    if not p.exists():
        pytest.skip("no windows file installed")
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "write the list with a byte copy, not PowerShell text cmdlets"
    cfg = json.loads(raw.decode("utf-8"))
    assert len(gr._windows(cfg)) == len(cfg["windows"]) > 0, "an entry would be skipped"
    for w in cfg["windows"]:
        a, b = t(gr, w["start"]), t(gr, w["end"])
        assert 0 < b - a <= 9 * 3600, w
        assert gr.quiet_window_end(a, cfg) == b and gr.quiet_window_end(b, cfg) is None
