"""Read-only PDH/RAM observations of a named native child; no owner-file writes.

Diagnostic observer started after the native run. It does not alter the sampler,
guard, process state or predictive computation. Stops when that child exits.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    import psutil
    import win32pdh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--creation-time", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=20.)
    args = parser.parse_args()
    if args.interval < 5 or args.out.exists():
        raise ValueError("interval>=5 and a fresh exclusive output path required")
    worker = psutil.Process(args.pid)
    if abs(worker.create_time() - args.creation_time) > 1:
        raise ValueError("native child identity mismatch")
    query = win32pdh.OpenQuery()
    paths = win32pdh.ExpandCounterPath(r"\Thermal Zone Information(*)\Temperature")
    counters = [win32pdh.AddCounter(query, path) for path in paths]
    win32pdh.CollectQueryData(query)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        try:
            while worker.is_running():
                win32pdh.CollectQueryData(query)
                zones = [{"counter": path, "celsius": round(win32pdh.GetFormattedCounterValue(
                         counter, win32pdh.PDH_FMT_DOUBLE)[1] - 273.15, 1)}
                         for path, counter in zip(paths, counters)]
                memory = psutil.virtual_memory()
                row = {"t": time.time(), "pid": args.pid, "creation_time": args.creation_time,
                       "zones": zones, "available_ram_gib": memory.available / 2**30,
                       "worker_rss_bytes": worker.memory_info().rss,
                       "worker_cpu_seconds": sum(worker.cpu_times()[:2]), "process_state_changed": False,
                       "source_sha256": source_sha, "scope": "after-launch diagnostic observation; no guard-file or model changes"}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if any(zone["celsius"] >= 98 for zone in zones) or memory.available < 4 * 2**30:
                    print(json.dumps({"resource_alert": True, **row}), flush=True)
                try:
                    worker.wait(timeout=args.interval)
                    break
                except psutil.TimeoutExpired:
                    pass
        except psutil.NoSuchProcess:
            pass
        finally:
            win32pdh.CloseQuery(query)
    print(json.dumps({"out": str(args.out), "native_child_exited": True}), flush=True)


if __name__ == "__main__":
    main()
