"""Bounded warm-start recovery training; original registered files are read only."""
import argparse,json,sys,time,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.recovery import RecoveryConfig,INITIAL_F,INITIAL_F_SHA,load_warm,legacy_prompt
    from dsv3wm.native import runtime,load,sha256,verify_freeze,boundary
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.data import load_records,conditioning
    from dsv3wm.recovery_data import legacy_target
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--freeze',required=True);ap.add_argument('--qualification',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();cfg=RecoveryConfig.read(a.config);freeze_sha=verify_freeze(a.freeze)
    verify_bound_inputs(a.freeze,[a.config,a.train,INITIAL_F/'adapter_model.safetensors',INITIAL_F/'adapter_config.json'])
    qualification=json.loads(Path(a.qualification).read_text(encoding='utf-8'))
    if (qualification.get('status')!='development-evaluated' or qualification.get('baseline_qualified') is not True
        or qualification.get('adapter_sha256')!=INITIAL_F_SHA or qualification.get('freeze_sha256')!=freeze_sha):
        raise ValueError('useful registered baseline qualification required before recovery training')
    if cfg.arm in {'A3','A4'}:raise ValueError('H3 training requires later independently verified corpus qualification')
    if cfg.legality_weight!=0.0:raise ValueError('recovery pilot legality is token-supervised; explicit head is a later ablation')
    rows=load_records(a.train,expected_split='train')
    if any(r['domain']!='register' for r in rows):raise ValueError('registered register recovery only')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result={'status':'started','config':cfg.dict(),'command':sys.argv,'config_sha256':sha256(a.config),'data_sha256':sha256(a.train),
            'initial_adapter_sha256':INITIAL_F_SHA,'freeze_sha256':freeze_sha,'qualification_sha256':sha256(a.qualification),
            'loss_curve':[],'row_order':[],'counts':{'input_tokens':0,'supervised_tokens':0,'aux_tokens':0,'forwards':0,'transitions':0}}
    started=time.perf_counter()
    try:
        with runtime(cfg,'recovery-train-'+cfg.arm):
            import torch
            from dsv3wm.objectives import SequentialMTP,token_objective,multi_horizon_loss
            model,tok,lm=load(cfg);model=load_warm(model,trainable=True)
            peft=model.peft_config['default']
            if peft.r!=16 or peft.lora_alpha!=32 or peft.lora_dropout!=0.05:raise ValueError('registered adapter config drift')
            model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
            decoder,head=lm._decoder_and_head(model);embedding=model.get_input_embeddings();device=next(model.parameters()).device
            # Auxiliary initialization does not advance the common dropout stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9000+cfg.seed)
                mtp=SequentialMTP(model.config.hidden_size,cfg.mtp_rank,cfg.mtp_tokens,cfg.share_aux).to(device) if cfg.arm=='A1' else None
            parameters=[p for p in model.parameters() if p.requires_grad]+(list(mtp.parameters()) if mtp else [])
            optimizer=torch.optim.AdamW(parameters,lr=cfg.lr)
            rng=random.Random(177+cfg.seed);order=[rng.randrange(len(rows)) for _ in range(cfg.steps*cfg.batch_size)]
            result['row_order']=order;model.train();training_started=time.perf_counter()
            for step in range(cfg.steps):
                boundary(cfg);lm.thermal_guard();optimizer.zero_grad(set_to_none=True)
                total_value=0.0
                for position,index in enumerate(order[step*cfg.batch_size:(step+1)*cfg.batch_size]):
                    row=rows[index];losses={}
                    for h in cfg.horizons:
                        # h1 has matching dropout across A0/A1/A2 at a common step.
                        torch.manual_seed(100000+cfg.seed*10000+step*100+position*10+h)
                        prompt=legacy_prompt(conditioning(row,h));label=json.dumps(legacy_target(row,h),sort_keys=True,separators=(',',':'))
                        encoded=lm.encode_pair(tok,prompt,label,cfg.max_length)
                        if encoded is None:raise ValueError('overlength recovery training record')
                        ids,lab,att=(x.to(device) for x in lm.collate([encoded],tok.pad_token_id))
                        loss,stats,_=token_objective(decoder,head,embedding,ids,att,lab,mtp,cfg.mtp_weight,cfg.ce_chunk,cfg.shuffled_aux)
                        losses[h]=loss
                        result['counts']['input_tokens']+=ids.numel();result['counts']['supervised_tokens']+=stats['base_tokens']
                        result['counts']['aux_tokens']+=stats['aux_tokens'];result['counts']['forwards']+=1;result['counts']['transitions']+=1
                    loss=multi_horizon_loss(losses,cfg.lambda2,cfg.lambda4)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite recovery loss')
                    (loss/cfg.batch_size).backward();total_value+=float(loss.detach())/cfg.batch_size
                torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True);optimizer.step()
                entry={'step':step+1,'loss':total_value,'elapsed_s':time.perf_counter()-training_started}
                result['loss_curve'].append(entry);print(json.dumps(entry),flush=True)
                if (step+1)%16==0:
                    # Immutable per-step recovery snapshots, not resume claims.
                    model.save_pretrained(out/f'checkpoint_step{step+1}')
                if cfg.device=='cuda':lm.gpu_yield()
            model.save_pretrained(out/'adapter');torch.save(mtp.state_dict() if mtp else {},out/'auxiliary.pt')
            result.update(status='pilot-trained',optimizer_steps=cfg.steps,trainable_parameters=sum(p.numel() for p in parameters),
                          lora_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          training_duration_s=time.perf_counter()-training_started,
                          adapter_sha256=sha256(out/'adapter/adapter_model.safetensors'),core_sha256=sha256(lm.MODELS[cfg.core]/'model.safetensors'),
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device=='cuda' else None,
                          peak_rss_bytes=getattr(__import__('psutil').Process().memory_info(),'peak_wset',None),
                          parameter_token_proxy=6*596049920*result['counts']['input_tokens'],
                          proxy_scope='total-parent parameter-token estimate; excludes checkpoint/auxiliary work, not measured FLOPs')
    except BaseException as exc:result.update(status='failed',error=repr(exc));raise
    finally:
        result['duration_s']=time.perf_counter()-started
        (out/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')


if __name__=='__main__':main()
