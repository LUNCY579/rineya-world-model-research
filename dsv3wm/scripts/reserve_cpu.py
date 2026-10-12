"""Hold this session's CPU lease for an explicitly coordinated peer native job."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime
    ap=argparse.ArgumentParser()
    ap.add_argument('--marker',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--job',default='rdw-E01-reservation')
    ap.add_argument('--expected-seconds',type=int,default=600)
    a=ap.parse_args()
    marker,out=Path(a.marker),Path(a.out)
    if marker.exists() or out.exists():
        raise ValueError('reservation/marker already exists')
    if a.expected_seconds<=0:raise ValueError('positive reservation duration required')
    with runtime(Config(device='cpu'),a.job):
        import os,tempfile
        lane=Path(os.environ.get('TEMP') or tempfile.gettempdir())/'rineya-cpu-lane.json'
        record=json.loads(lane.read_text(encoding='utf-8'))
        record['expected_end']=record['started']+a.expected_seconds
        # Update only the lease still bearing our token. Duration is metadata;
        # never expire a live peer model merely because its estimate elapsed.
        on_disk=json.loads(lane.read_text(encoding='utf-8'))
        if on_disk.get('token')!=record['token']:raise RuntimeError('reservation ownership changed')
        lane.write_text(json.dumps({**on_disk,'expected_end':record['expected_end']}),encoding='utf-8')
        record.update(status='reserved',completion_marker=str(marker),started_utc_unix=time.time())
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps({'status':'reserved','lane':str(lane),'token':record['token'],'owner_pid':record['pid']}),flush=True)
        while not marker.exists():
            time.sleep(2)
        completion=json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(completion,dict):
            raise ValueError('peer completion marker must be JSON object')
        # Marker is peer evidence of subprocess exit, never executed as instructions.
        record.update(status='peer-completed',completed_unix=time.time(),peer_completion=completion)
        out.write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n')
    print('Peer native slot released',flush=True)


if __name__=='__main__':
    main()
