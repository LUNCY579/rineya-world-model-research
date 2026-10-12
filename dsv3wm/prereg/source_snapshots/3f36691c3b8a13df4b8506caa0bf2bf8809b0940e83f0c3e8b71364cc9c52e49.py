"""Run one local research command and record its observed host-memory cost."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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
    process = subprocess.Popen([sys.executable, *command], shell=False)
    record['root_pid'] = process.pid
    try:
        while True:
            try:
                parent = psutil.Process(process.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            rss = 0
            for item in processes:
                try:
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
    finally:
        record['wall_duration_s'] = time.perf_counter() - started
        args.out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'resource_record': str(args.out), 'exit_code': process.returncode,
                      'peak_sampled_process_tree_rss_bytes': record['peak_sampled_process_tree_rss_bytes']}), flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
