#!/usr/bin/env python3
"""External watchdog for each worker invocation, including stalled subprocesses."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.pipelines.common import atomic_json
config=json.loads((ROOT/'configs/production.json').read_text())
process=None
try:
    process=subprocess.Popen([sys.executable,'-m','goal_conditioned_decision_data.pipelines.worker',*sys.argv[1:]],cwd=ROOT.parent,start_new_session=True)
    code=process.wait(timeout=config['cycle_timeout_seconds']+160)
    # Clean up any abandoned command descendants even after a normal exit.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    raise SystemExit(code)
except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    # The worker may exit while a descendant remains. Kill the whole group
    # regardless of whether the parent acknowledged TERM.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()
    atomic_json(ROOT/'state/PAUSED.json',{'reason':'worker watchdog timeout; queue and immutable shards retained','utc_epoch':time.time()})
    atomic_json(ROOT/'state/heartbeat.json',{'status':'paused','reason':'watchdog_timeout','utc_epoch':time.time()})
    raise SystemExit(1)
