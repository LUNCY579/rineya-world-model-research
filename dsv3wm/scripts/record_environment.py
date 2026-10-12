"""Read-only resource/runtime snapshot; no CUDA context or weights loaded."""
import argparse,importlib.metadata,json,platform,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out)
if out.exists():raise ValueError('environment snapshot exists')
memory=psutil.virtual_memory();disk=shutil.disk_usage(out.parent.resolve())
value={'at_utc':datetime.now(timezone.utc).isoformat(),'client_timezone':'America/Chicago',
       'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),
       'cpu_logical':psutil.cpu_count(),'cpu_physical':psutil.cpu_count(logical=False),
       'ram_total_bytes':memory.total,'ram_available_bytes':memory.available,'disk_free_bytes':disk.free,
       'libraries':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','safetensors','tokenizers','psutil')},
       'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu','--format=csv'],text=True),
       'native_uncached_A0_observed_working_set_bytes':3736174592,
       'observed_working_set_scope':'one in-flight CIM sample from this session at 2026-10-12 03:19 UTC; not peak RSS',
       'gpu_probe':'failed lease before CUDA/model load; no reclaim performed'}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n');print(str(out))
