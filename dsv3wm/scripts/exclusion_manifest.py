"""Hash private local artifacts; sealed-label identity comes from collection manifest."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
if out.exists():raise ValueError('exclusion manifest exists')
paths=[p for directory in (ROOT/'artifacts',ROOT/'datasets') for p in directory.rglob('*') if p.is_file()]
items=[]
for p in sorted(paths):
    if p.suffix=='.json' and p.name=='manifest.json':continue
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    items.append({'path':p.relative_to(REPO).as_posix(),'bytes':p.stat().st_size,'sha256':digest.hexdigest(),
                  'reason':'private adapter/auxiliary weights or regenerable fresh training/development inputs; excluded from Git'})
manifest=json.loads((ROOT/'data_manifests/pilot-v1.json').read_text(encoding='utf-8'))
items.append({**manifest['sealed_confirmation'],'reason':'sealed external confirmation; original generation hash reused WITHOUT reading labels'})
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'files':items},indent=2),encoding='utf-8',newline='\n')
print(str(out))
