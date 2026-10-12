"""Run one local research command and record its observed host-memory cost."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def stop_owned_processes(process, observed):
    """Cancel only this command's recorded subprocess identities, then reap it."""
    import psutil
    targets = dict(observed)
    if process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            for item in [parent, *parent.children(recursive=True)]:
                targets[(item.pid, item.create_time())] = item
        except psutil.Error:
            # Discovery failure must not discard previously observed identities.
            pass
    live = []
    for item in reversed(list(targets.values())):
        try:
            if item.is_running():  # psutil compares creation time, avoiding PID reuse.
                item.terminate()
                live.append(item)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            # Continue cleanup of the rest of our known tree and the Popen root.
            continue
    if process.poll() is None:
        process.terminate()
    _, remaining = psutil.wait_procs(live, timeout=5)
    for item in remaining:
        if item.is_running():
            item.kill()
    psutil.wait_procs(remaining, timeout=5)
    process.wait(timeout=5)


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('a research script and arguments are required')
    if args.out.exists():
        raise FileExistsError('resource record exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *command], 'cwd': str(Path.cwd()),
              'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sample_interval_s': 0.5, 'peak_sampled_process_tree_rss_bytes': 0,
              'per_process_peak_wset_bytes': {}, 'peak_gpu_bytes': None,
              'scope': 'local subprocess tree host RAM; GPU memory is not measured by this wrapper'}
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    started = time.perf_counter()
    process, observed = None, {}
    try:
        process = subprocess.Popen([sys.executable, *command], shell=False)
        record['root_pid'] = process.pid
        while True:
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            rss = 0
            for item in processes:
                try:
                    observed[(item.pid, item.create_time())] = item
                    memory = item.memory_info()
                    rss += memory.rss
                    peak = getattr(memory, 'peak_wset', None)
                    if peak is not None:
                        key = str(item.pid)
                        record['per_process_peak_wset_bytes'][key] = max(peak, record['per_process_peak_wset_bytes'].get(key, 0))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            record['peak_sampled_process_tree_rss_bytes'] = max(rss, record['peak_sampled_process_tree_rss_bytes'])
            if process.poll() is not None:
                break
            time.sleep(0.5)
        record.update(status='complete' if process.returncode == 0 else 'failed', exit_code=process.returncode)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        if process is not None:
            try:
                stop_owned_processes(process, observed)
                record.update(exit_code=process.returncode, child_cleanup='terminated/reaped owned subprocess identities')
            except BaseException as cleanup_error:
                record['child_cleanup_error'] = repr(cleanup_error)
        raise
    finally:
        record['wall_duration_s'] = time.perf_counter() - started
        args.out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'resource_record': str(args.out), 'exit_code': process.returncode,
                      'peak_sampled_process_tree_rss_bytes': record['peak_sampled_process_tree_rss_bytes']}), flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
