"""Recover the pre-clarification trainer bytes and verify against its frozen hash."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/train_recovery.py';s=p.read_text(encoding='utf-8')
s=s.replace('a.config,a.train,a.qualification,INITIAL_F','a.config,a.train,INITIAL_F')
s=s.replace("qualification.get('freeze_sha256')!=json.loads(Path(a.freeze).read_text())['qualification_freeze_sha256']",
            "qualification.get('freeze_sha256')!=freeze_sha")
s=s.replace('optimizer=torch.optim.AdamW(parameters,lr=cfg.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)',
            'optimizer=torch.optim.AdamW(parameters,lr=cfg.lr)')
s=s.replace("            result['optimizer']={'kind':'AdamW','lr':cfg.lr,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.01,'scheduler':None,'grad_clip':1.0}\n",'')
b=s.encode();f=json.loads((ROOT/'prereg/RECOVERY_FREEZE_V1.json').read_text())
expected=f['files'][str(p.resolve()).replace('\\','/')]
actual=hashlib.sha256(b).hexdigest()
if actual!=expected:raise ValueError(f'v1 reconstruction mismatch: {actual} != {expected}')
out=ROOT/'prereg/source_snapshots'/(expected+'.py')
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('xb') as stream:stream.write(b)
print(json.dumps({'archived_source':str(out),'sha256':actual}))
