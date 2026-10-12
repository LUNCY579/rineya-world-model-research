"""Native development planning feasibility screen; exact scoring AFTER choice."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.data import load_records
    from dsv3wm.native import runtime, load, Predictor, sha256, verify_freeze
    from dsv3wm.planning import PlanningBudget, learned_beam_search, verify_plan
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--freeze',default=str(Path(__file__).resolve().parents[1]/'prereg/EVALUATION_FREEZE_V2.json'))
    a=ap.parse_args()
    cfg=Config.read(a.config)
    freeze_sha=verify_freeze(a.freeze)
    rows=[r for r in load_records(a.data,expected_split='dev') if r['domain']=='register']
    # Generation/scoring instruments establish hardness before native outputs.
    from dsv3wm.domains import _semantics
    tasks=[]
    parents=set()
    for row in rows:
        goal={'kind':'state','state':row['labels']['states'][3]}
        distance=_semantics('register').opt_moves(row['s0']['x'],goal['state']['x'],row['rules']['limit'])
        if distance is not None and 3<=distance<=4 and row['parent_world_id'] not in parents:
            parents.add(row['parent_world_id'])
            tasks.append({'parent_world_id':row['parent_world_id'],'domain':row['domain'],'rules':row['rules'],
                          's0':row['s0'],'goal':goal,'optimal_moves':distance})
        if len(tasks)==2:
            break
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise ValueError('planning record already exists')
    # Immutable input snapshot written before learned model execution.
    inputs=out.with_suffix('.inputs.json')
    if inputs.exists():
        raise ValueError('planning input snapshot already exists')
    inputs.write_text(json.dumps(tasks,indent=2),encoding='utf-8',newline='\n')
    budget=PlanningBudget(max_model_calls=20,max_depth=4,beam_width=1,max_candidates=16)
    result={'status':'started','config':cfg.dict(),'command':sys.argv,'freeze_sha256':freeze_sha,
            'task_sha256':sha256(inputs),'script_sha256':sha256(__file__),'budget':budget.__dict__,'max_decoder_token_positions_per_episode':120000,
            'scope':'fresh register development, optimum3-4 moves, at most2 independent parents; not P2 confirmation',
            'outcomes':[]}
    started=time.perf_counter()
    try:
        if not tasks:
            result['status']='deferred-no-eligible-dev-planning-parents'
        else:
            with runtime(cfg,'planning-'+cfg.arm):
                model,tok,_=load(cfg,a.adapter)
                for task in tasks:
                    predictor=Predictor(model,tok,cfg,max_decoder_token_positions=120000)
                    conditioning={key:task[key] for key in ('domain','rules','s0')}
                    conditioning['actions']=[]
                    ranked=learned_beam_search(conditioning,task['goal'],predictor,budget)
                    choice=ranked['choice']
                    # NO verification result re-enters search or selection.
                    scored=verify_plan(conditioning,task['goal'],choice) if choice is not None else {'success':False,'all_legal':False}
                    result['outcomes'].append({'parent_world_id':task['parent_world_id'],'optimal_moves':task['optimal_moves'],
                        'success':scored['success'],'valid':scored['all_legal'],'choice':choice,'model_calls':predictor.calls,
                        'output_tokens':predictor.output_tokens,'input_tokens':predictor.input_tokens,
                        'decoder_token_positions':predictor.decoder_token_positions,'budget_exhausted':predictor.budget_exhausted,
                        'malformed_predictions':ranked['malformed_predictions'],'oracle_feedback':False})
                result['status']='pilot-evaluated'
        result['n']=len(result['outcomes'])
        result['successes']=sum(r['success'] for r in result['outcomes'])
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result['duration_s']=time.perf_counter()-started
        out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
