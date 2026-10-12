"""Recovery development readout with inherited F; never opens confirmation data."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.recovery import RecoveryConfig,INITIAL_F,load_warm,LegacyPredictor,batched_rollout
    from dsv3wm.native import runtime,load,sha256,verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.data import load_records,conditioning
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--adapter',default=str(INITIAL_F));ap.add_argument('--out',required=True)
    ap.add_argument('--freeze',required=True)
    ap.add_argument('--baseline-only',action='store_true');a=ap.parse_args()
    cfg=RecoveryConfig.read(a.config);freeze_sha=verify_freeze(a.freeze)
    verify_bound_inputs(a.freeze,[a.config,a.data,Path(a.adapter)/'adapter_model.safetensors',Path(a.adapter)/'adapter_config.json'])
    rows=load_records(a.data,expected_split='dev');out=Path(a.out)
    if out.exists():raise ValueError('recovery result exists')
    if any(r['domain']!='register' for r in rows) or len({r['parent_world_id'] for r in rows})!=len(rows):
        raise ValueError('unique register development parents required')
    result={'status':'started','config':cfg.dict(),'command':sys.argv,'data_sha256':sha256(a.data),'freeze_sha256':freeze_sha,
            'adapter_sha256':sha256(Path(a.adapter)/'adapter_model.safetensors'),'oracle_feedback':False,
            'instrument':'legacy registered-F interface / standard KV / batched greedy', 'per_parent':[]}
    started=time.perf_counter()
    try:
        with runtime(cfg,'recovery-'+('B0' if a.baseline_only else cfg.arm)):
            model,tok,lm=load(cfg);model=load_warm(model,a.adapter)
            predictor=LegacyPredictor(model,tok,cfg)
            rollouts=batched_rollout(predictor,rows,4)
            direct={}
            for h in ((1,) if a.baseline_only else (1,2,4)):
                direct[h]=predictor.predict_batch([conditioning(r,h) for r in rows])
            summary={}
            for i,row in enumerate(rows):
                cohort='id' if row['provenance']['cohort']=='recovery_id_teacher_known' else 'ood'
                entry={'parent_world_id':row['parent_world_id'],'cohort':cohort,'horizons':{}}
                for h in (1,2,4):
                    r=rollouts[i][h-1] if len(rollouts[i])>=h else {'malformed':True}
                    d=direct[h][i] if h in direct else None
                    entry['horizons'][str(h)]={'rollout_correct':not r.get('malformed') and r.get('state')==row['labels']['states'][h-1],
                         'rollout_malformed':bool(r.get('malformed')),
                         'direct_correct':None if d is None else not d.get('malformed') and d.get('state')==row['labels']['states'][h-1],
                         'direct_malformed':None if d is None else bool(d.get('malformed')),
                         'legality_correct':not r.get('malformed') and r.get('legal')==row['labels']['legal'][h-1]}
                result['per_parent'].append(entry)
            for cohort in ('id','ood','all'):
                selected=[e for e in result['per_parent'] if cohort=='all' or e['cohort']==cohort]
                summary[cohort]={}
                for h in ('1','2','4'):
                    metrics={'n':len(selected)}
                    for field in ('rollout_correct','rollout_malformed','direct_correct','direct_malformed','legality_correct'):
                        vals=[e['horizons'][h][field] for e in selected]
                        metrics[field]=None if all(v is None for v in vals) else sum(bool(v) for v in vals)
                    summary[cohort][h]=metrics
            traces=out.with_suffix('.traces.json');traces.write_text(json.dumps(predictor.traces,indent=2),encoding='utf-8',newline='\n')
            result.update(status='development-evaluated',summary=summary,model_queries=predictor.calls,input_tokens=predictor.input_tokens,
                          output_tokens=predictor.output_tokens,processed_positions_upper_bound=predictor.decoder_token_positions,
                          traces_sha256=sha256(traces),baseline_qualified=summary['id']['1']['rollout_correct']>=10 and summary['id']['1']['rollout_malformed']<=1)
    except BaseException as exc:result.update(status='failed',error=repr(exc));raise
    finally:
        result['duration_s']=time.perf_counter()-started;out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result['summary'] if 'summary' in result else result),flush=True)


if __name__=='__main__':main()
