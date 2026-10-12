"""Explicit cached development instrument with raw traces and loss-only audits."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime,load,sha256,verify_freeze,response_prompt
    from dsv3wm.cached_native import CachedPredictor,chat_prompt
    from dsv3wm.data import load_records,conditioning,target
    from dsv3wm.evaluation import score_dynamics
    from dsv3wm.planning import learned_beam_search,verify_plan,PlanningBudget
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--adapter',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--prompt-style',choices=['plain','chat'],default='plain')
    ap.add_argument('--freeze',required=True)
    ap.add_argument('--probe',required=True)
    ap.add_argument('--planning',action='store_true')
    a=ap.parse_args()
    cfg=Config.read(a.config)
    freeze_sha=verify_freeze(a.freeze)
    frozen=json.loads(Path(a.freeze).read_text(encoding='utf-8'))
    required={'dsv3wm/src/dsv3wm/cached_native.py','dsv3wm/scripts/cached_pilot.py','dsv3wm/scripts/cache_probe.py',
              'dsv3wm/prereg/PILOT_AMENDMENT_2_CACHE.md','bwm/code/bwmn/lm.py'}
    members={key.replace('\\','/') for key in frozen['files']}
    if frozen.get('instrument')!='native-standard-kv-v3' or not required<=members:
        raise ValueError('complete standard-KV v3 source/instrument freeze required')
    probe=json.loads(Path(a.probe).read_text(encoding='utf-8'))
    if probe.get('status')!='passed' or probe.get('script_sha256')!=sha256(Path(__file__).parent/'cache_probe.py'):
        raise ValueError('qualified native cache probe with unchanged source required')
    rows=[r for r in load_records(a.data,expected_split='dev') if r['domain']=='register']
    out=Path(a.out)
    if out.exists():
        raise ValueError('cached record exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'started','config':cfg.dict(),'config_sha256':sha256(a.config),'data_sha256':sha256(a.data),
            'adapter_sha256':sha256(Path(a.adapter)/'adapter_model.safetensors'),'freeze_sha256':freeze_sha,'probe_sha256':sha256(a.probe),
            'command':sys.argv,'prompt_style':a.prompt_style,'cache_policy':'standard BWM KV / FP32 / no compression',
            'script_sha256':sha256(__file__),'oracle_feedback':False}
    started=time.perf_counter()
    try:
        with runtime(cfg,'cached-'+cfg.arm):
            model,tok,lm=load(cfg,a.adapter)
            predictor=CachedPredictor(model,tok,cfg,max_decoder_token_positions=120000 if a.planning else None,prompt_style=a.prompt_style)
            if a.planning:
                from dsv3wm.domains import _semantics
                tasks=[];seen=set()
                for row in rows:
                    goal={'kind':'state','state':row['labels']['states'][3]}
                    distance=_semantics('register').opt_moves(row['s0']['x'],goal['state']['x'],row['rules']['limit'])
                    if distance is not None and 3<=distance<=4 and row['parent_world_id'] not in seen:
                        seen.add(row['parent_world_id'])
                        tasks.append({'conditioning':{**conditioning(row,1),'actions':[]},'goal':goal,
                                      'parent_world_id':row['parent_world_id'],'optimal_moves':distance})
                    if len(tasks)==1:break
                payload=json.dumps(tasks,sort_keys=True)
                import hashlib
                result['planning_inputs_sha256']=hashlib.sha256(payload.encode()).hexdigest()
                (out.with_suffix('.inputs.json')).write_text(payload,encoding='utf-8',newline='\n')
                budget=PlanningBudget(max_model_calls=20,max_depth=4,beam_width=1,max_candidates=16)
                outcomes=[]
                for task in tasks:
                    choice=learned_beam_search(task['conditioning'],task['goal'],predictor,budget)['choice']
                    scored=verify_plan(task['conditioning'],task['goal'],choice) if choice is not None else {'success':False,'all_legal':False}
                    outcomes.append({'parent_world_id':task['parent_world_id'],'success':scored['success'],
                                     'valid':scored['all_legal'],'choice':choice,'optimal_moves':task['optimal_moves']})
                result.update(n=len(outcomes),successes=sum(x['success'] for x in outcomes),outcomes=outcomes,budget=budget.__dict__)
            else:
                import torch
                ces=[]
                model.eval()
                for row in rows[:2]:
                    mode='full' if cfg.target_mode=='mixed' else cfg.target_mode
                    prompt=(chat_prompt if a.prompt_style=='chat' else response_prompt)(conditioning(row,1),mode)
                    label=json.dumps(target(row,1,mode),sort_keys=True,separators=(',',':'))
                    encoded=lm.encode_pair(tok,prompt,label,cfg.max_length)
                    if encoded is None:raise ValueError('overlength')
                    ids,lab,att=(x.to(next(model.parameters()).device) for x in lm.collate([encoded],tok.pad_token_id))
                    with torch.no_grad():ce,_=lm.chunked_ce(model,ids,att,lab,cfg.ce_chunk)
                    ces.append(float(ce))
                result.update(score_dynamics(predictor,rows[:2]))
                result['teacher_forced_one_step_ce']=sum(ces)/len(ces)
            result.update(status='pilot-evaluated',forward_model_calls=predictor.calls,input_tokens=predictor.input_tokens,
                          output_tokens=predictor.output_tokens,autoregressive_decoder_forwards=predictor.decoder_forwards,
                          decoder_token_positions=predictor.decoder_token_positions,kv_cache_peak_bytes_estimate=predictor.kv_cache_peak_bytes_estimate,
                          kv_cache_estimate_scope='resident K/V tensors only, excludes allocation transients/weights',budget_exhausted=predictor.budget_exhausted)
            traces=out.with_suffix('.traces.json')
            traces.write_text(json.dumps(predictor.traces,indent=2),encoding='utf-8',newline='\n')
            result['traces_sha256']=sha256(traces)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['duration_s']=time.perf_counter()-started
        out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps({'out':str(out),'status':result['status'],'duration_s':result['duration_s']}),flush=True)


if __name__=='__main__':main()
