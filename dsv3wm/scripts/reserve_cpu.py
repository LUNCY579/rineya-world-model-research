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
    a=ap.parse_args()
    marker,out=Path(a.marker),Path(a.out)
    if marker.exists() or out.exists():
        raise ValueError('reservation/marker already exists')
    with runtime(Config(device='cpu'),'rdw-E01-reservation'):
        import os,tempfile
        lane=Path(os.environ.get('TEMP') or tempfile.gettempdir())/'rineya-cpu-lane.json'
        record=json.loads(lane.read_text(encoding='utf-8'))
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
