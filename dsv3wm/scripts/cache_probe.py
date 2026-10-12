"""Native standard-KV causal/logit/greedy parity and latency on a fresh fixture."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime,load,sha256
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--adapter',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('cache probe exists')
    cfg=Config(device='cpu')
    result={'status':'started','command':sys.argv,'scope':'standard existing KV cache, native A0, constructed original-format case'}
    started=time.perf_counter()
    try:
        with runtime(cfg,'cache-parity'):
            import torch
            from transformers import DynamicCache
            from bwmn import rw2
            model,tok,lm=load(cfg,a.adapter);model.eval();lm.DUTY_RATIO=0.0
            prompt=rw2.forward_prompt(4,'double',16)
            ids=torch.tensor([tok(prompt,add_special_tokens=False).input_ids],dtype=torch.long)
            cache=DynamicCache();maximum=0.0;argmax=True
            with torch.no_grad():
                cached=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True,logits_to_keep=1)
                full=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
                for step in range(8):
                    maximum=max(maximum,float((cached.logits-full.logits).abs().max()))
                    torch.testing.assert_close(cached.logits,full.logits,rtol=1e-4,atol=1e-4)
                    argmax=argmax and bool(torch.equal(cached.logits.argmax(-1),full.logits.argmax(-1)))
                    next_token=full.logits[:,-1,:].argmax(-1)[:,None]
                    ids=torch.cat([ids,next_token],1)
                    cached=model(input_ids=next_token,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    full=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
                assert argmax
                encoded=tok(prompt,return_tensors='pt',add_special_tokens=False)
                t=time.perf_counter()
                sequence=model.generate(**encoded,max_new_tokens=8,do_sample=False,use_cache=False,logits_to_keep=1,pad_token_id=tok.pad_token_id,eos_token_id=lm.EOS_IDS)
                uncached_s=time.perf_counter()-t
                raw=sequence[0,encoded.input_ids.shape[1]:]
                text=tok.decode(raw,skip_special_tokens=True)
                cached_hf=model.generate(**encoded,max_new_tokens=8,do_sample=False,use_cache=True,logits_to_keep=1,pad_token_id=tok.pad_token_id,eos_token_id=lm.EOS_IDS)
                assert torch.equal(sequence,cached_hf)
                t=time.perf_counter();cached_generation=lm.sample_crn(model,tok,[prompt],[None],max_new_tokens=8)[0]
                cached_s=time.perf_counter()-t
                assert cached_generation['text']==text
                assert cached_generation['n_tokens']==int(raw.numel())
                assert cached_generation['hit_eos']==bool(int(raw[-1]) in lm.EOS_IDS)
            result.update(status='passed',max_logit_abs_difference=maximum,all_argmax_equal=argmax,greedy_text_equal=True,
                          hf_greedy_raw_token_ids_equal=True,bwm_greedy_text_length_eos_equal=True,
                          cached_s=cached_s,uncached_s=uncached_s,greedy_tokens=int(raw.numel()),
                          parent_sha256=sha256(lm.MODELS['titan-sparse']/'model.safetensors'),
                          adapter_sha256=sha256(Path(a.adapter)/'adapter_model.safetensors'),script_sha256=sha256(__file__))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['duration_s']=time.perf_counter()-started
        out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
