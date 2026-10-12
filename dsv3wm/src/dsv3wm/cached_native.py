"""Explicit standard-KV instrument, reusing BWM's validated greedy decoder.

No quantization, compression or attention conversion. Keep the original uncached
instrument intact. Each trace records conditioning and actual decoded text.
"""
from .native import response_prompt, parse_prediction, boundary


def chat_prompt(conditioning, mode):
    import json
    # This is the existing Titan/Qwen non-thinking chat format, not a solver call.
    from .domains import _semantics
    system=('Predict the resulting state after the ordered actions using the supplied rules. '
            'Return only a JSON object with state (or delta), legal, terminated and failed. '
            'The flags must be JSON booleans. Illegal actions leave the state unchanged.')
    return _semantics('register').render_chat(system,json.dumps({**conditioning,'output_mode':mode},separators=(',',':')))


class CachedPredictor:
    def __init__(self,model,tokenizer,config,max_decoder_token_positions=None,prompt_style='plain'):
        if prompt_style not in ('plain','chat'):
            raise ValueError('explicit plain/chat style required')
        self.model,self.tokenizer,self.config=model,tokenizer,config
        self.prompt_style=prompt_style
        self.calls=self.input_tokens=self.output_tokens=self.decoder_token_positions=self.decoder_forwards=0
        self.max_decoder_token_positions=max_decoder_token_positions
        self.budget_exhausted=False
        self.kv_cache_peak_bytes_estimate=0
        self.traces=[]

    def __call__(self,conditioning):
        from bwmn import lm
        boundary(self.config)
        mode='full' if self.config.target_mode=='mixed' else self.config.target_mode
        prompt=(chat_prompt if self.prompt_style=='chat' else response_prompt)(conditioning,mode)
        n=len(self.tokenizer(prompt,add_special_tokens=False).input_ids)
        if n>self.config.max_length:
            return {'malformed':True,'error':'overlength'}
        limit=self.config.generation_tokens
        if self.max_decoder_token_positions is not None:
            # The custom BWM decoder may perform one unused final forward at the
            # token cap; n+g is a conservative exact upper bound for positions.
            limit=min(limit,self.max_decoder_token_positions-self.decoder_token_positions-n)
            if limit<=0:
                self.budget_exhausted=True
                return {'malformed':True,'error':'decoder-token-budget-exhausted'}
        self.model.eval()
        # CPU lane already caps native work to four threads. Retain thermal checks;
        # don't apply the GPU's proportional idle interval to CPU-only work.
        if self.config.device=='cpu':
            lm.DUTY_RATIO=0.0
        generated=lm.sample_crn(self.model,self.tokenizer,[prompt],[None],max_new_tokens=limit)[0]
        g=int(generated['n_tokens'])
        extra=int(not generated['hit_eos'])
        self.calls+=1
        self.input_tokens+=n
        self.output_tokens+=g
        self.decoder_forwards+=g+extra
        self.decoder_token_positions+=n+g-1+extra
        cache_tokens=n+g-1+extra
        c=self.model.config
        self.kv_cache_peak_bytes_estimate=max(self.kv_cache_peak_bytes_estimate,
            cache_tokens*c.num_hidden_layers*2*c.num_key_value_heads*c.head_dim*4)
        parsed=parse_prediction(generated['text'].strip(),conditioning,mode)
        self.traces.append({'conditioning':conditioning,'prompt_style':self.prompt_style,
                            'generated':generated,'parsed':parsed})
        return parsed
