"""Existing registered BWM F behavior on constructed permitted smoke fixtures."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.native import runtime,load,sha256
    from dsv3wm.config import Config
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    if sha256(Path(a.adapter)/'adapter_model.safetensors')!='c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466':
        raise ValueError('registered F checkpoint mismatch')
    out=Path(a.out)
    if out.exists():raise ValueError('registered behavior probe exists')
    cfg=Config(device='cpu');result={'status':'started','command':sys.argv,
        'scope':'constructed original-format smoke cases; not historical sealed benchmark or nonoverlap claim'}
    started=time.perf_counter()
    try:
        with runtime(cfg,'registered-forward-parity'):
            from bwmn import rw2
            cases=[(3,'halve',32),(32,'add1',32),(4,'double',16),(-4,'double',32),(0,'add1',24),(2,'finish',16)]
            fixtures=[{'x':x,'action':action,'limit':limit,'expected':list(rw2.exact_step(x,action,limit)),
                       'prompt':rw2.forward_prompt(x,action,limit)} for x,action,limit in cases]
            frozen=out.with_suffix('.fixtures.json')
            frozen.write_text(json.dumps(fixtures,indent=2),encoding='utf-8',newline='\n')
            result['fixtures_sha256']=sha256(frozen)
            model,tok,lm=load(cfg,a.adapter);lm.DUTY_RATIO=0.0
            predictions=lm.sample_crn(model,tok,[f['prompt'] for f in fixtures],[None]*len(fixtures),max_new_tokens=24)
            rows=[]
            for fixture,prediction in zip(fixtures,predictions):
                parsed=rw2.parse_forward(prediction['text'])
                rows.append({'input':{k:fixture[k] for k in ('x','action','limit')},'generated':prediction,
                             'correct':parsed is not None and list(parsed)==fixture['expected']})
            result.update(status='passed' if all(r['correct'] for r in rows) else 'failed',n=len(rows),
                          correct=sum(r['correct'] for r in rows),rows=rows,
                          adapter_sha256=sha256(Path(a.adapter)/'adapter_model.safetensors'),
                          parent_sha256=sha256(lm.MODELS['titan-sparse']/'model.safetensors'),script_sha256=sha256(__file__))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['duration_s']=time.perf_counter()-started
        out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
