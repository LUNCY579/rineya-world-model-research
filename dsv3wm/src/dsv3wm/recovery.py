"""Registered-F-compatible causal interfaces for the stronger development pilot."""
import json
from pathlib import Path
from dataclasses import dataclass
from .config import Config
from . import REPO
from .native import sha256,boundary

INITIAL_F=REPO.parent/'Rineya_BWM_001_Research_Bundle/native_v1/runs/F_titan_main/adapter'
INITIAL_F_SHA='c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466'


@dataclass(frozen=True)
class RecoveryConfig(Config):
    steps:int=32
    rank:int=16
    lr:float=0.00002
    mtp_weight:float=0.1
    legality_weight:float=0.0
    generation_tokens:int=32
    max_length:int=512
    def __post_init__(self):
        super().__post_init__()
        if self.rank!=16 or self.target_mode!='full':
            raise ValueError('recovery binds registered rank16/full-state interface')


def legacy_prompt(inputs):
    from .domains import _semantics
    rw=_semantics('register')
    if (set(inputs)!={'domain','rules','s0','actions'} or inputs['domain']!='register'
        or set(inputs['s0'])!={'x'} or type(inputs['s0']['x']) is not int
        or type(inputs['rules'].get('limit')) is not int or not inputs['actions']):
        raise ValueError('register initial-state/action conditioning required')
    actions=inputs['actions'];x=inputs['s0']['x'];limit=inputs['rules']['limit']
    if len(actions)==1:
        if isinstance(actions[0],str) and actions[0] in rw.ACTIONS:
            return rw.forward_prompt(x,actions[0],limit)
        system=rw.SYS_FORWARD+' Unknown or malformed actions are illegal and leave x unchanged.'
        return rw.render_chat(system,rw.canonical({'action':actions[0],'limit':limit,'x':x}))
    system=(rw.SYS_FORWARD+' Apply the supplied ordered actions in sequence without receiving intermediate states. '
            'Illegal actions leave x unchanged. After finish, the state is absorbing and later actions are ignored. '
            'Return only JSON with x, legal and terminated. legal refers to the last supplied action and is false '
            'for an ignored action; terminated stays true after finish. Unknown or malformed actions are illegal no-ops.')
    return rw.render_chat(system,rw.canonical({'actions':actions,'limit':limit,'x':x}))


def legacy_parse(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    try:
        value=json.loads(text.strip(),object_pairs_hook=unique)
        if (not isinstance(value,dict) or set(value)!={'x','legal','terminated'} or type(value['x']) is not int
            or type(value['legal']) is not bool or type(value['terminated']) is not bool):
            raise ValueError('legacy complete JSON schema required')
        return {'state':{'x':value['x']},'legal':value['legal'],'terminated':value['terminated'],
                'failed':not value['legal']}
    except (ValueError,TypeError,AttributeError):
        return {'malformed':True,'text':text}


def load_warm(model,adapter=INITIAL_F,trainable=False):
    from peft import PeftModel
    if Path(adapter)==INITIAL_F and sha256(Path(adapter)/'adapter_model.safetensors')!=INITIAL_F_SHA:
        raise ValueError('initial registered F identity mismatch')
    return PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable)


class LegacyPredictor:
    """Learned-only batches; exact rules never execute in this inference object."""
    def __init__(self,model,tokenizer,config):
        self.model,self.tokenizer,self.config=model,tokenizer,config
        self.calls=self.decoder_token_positions=self.input_tokens=self.output_tokens=self.decoder_forwards=0
        self.traces=[]
    def predict_batch(self,inputs):
        from bwmn import lm
        boundary(self.config)
        prompts=[legacy_prompt(item) for item in inputs]
        lengths=[len(self.tokenizer(p,add_special_tokens=False).input_ids) for p in prompts]
        if any(n>self.config.max_length for n in lengths):raise ValueError('overlength recovery input')
        self.model.eval()
        if self.config.device=='cpu':lm.DUTY_RATIO=0.0
        generated=lm.sample_crn(self.model,self.tokenizer,prompts,[None]*len(prompts),
                                 max_new_tokens=self.config.generation_tokens)
        outputs=[]
        max_generated=max((g['n_tokens'] for g in generated),default=0)
        max_prompt=max(lengths,default=0)
        batch_capped=any(not g['hit_eos'] for g in generated)
        # Broad batches can pad finished rows; this is a conservative processed
        # position upper bound, not the unpadded per-query cache estimate.
        self.decoder_token_positions+=len(inputs)*(max_prompt+max_generated-1+int(batch_capped))
        self.decoder_forwards+=max_generated+int(batch_capped)
        for item,prompt,n,g in zip(inputs,prompts,lengths,generated):
            result=legacy_parse(g['text']);outputs.append(result)
            self.calls+=1;self.input_tokens+=n;self.output_tokens+=g['n_tokens']
            self.traces.append({'conditioning':item,'generated':g,'parsed':result})
        return outputs
    def __call__(self,inputs):return self.predict_batch([inputs])[0]


def batched_rollout(predictor,records,max_horizon=4):
    import copy
    states=[copy.deepcopy(r['s0']) for r in records]
    outputs=[[] for _ in records];dead=[False]*len(records);terminated=[False]*len(records)
    for step in range(max_horizon):
        active=[i for i in range(len(records)) if not dead[i] and not terminated[i]]
        inputs=[{'domain':'register','rules':records[i]['rules'],'s0':states[i],'actions':[records[i]['actions'][step]]} for i in active]
        results=predictor.predict_batch(inputs) if active else []
        for i,result in zip(active,results):
            outputs[i].append(result)
            if result.get('malformed'):dead[i]=True
            else:states[i]=copy.deepcopy(result['state']);terminated[i]=result['terminated']
        for i in range(len(records)):
            if i not in active and terminated[i]:
                outputs[i].append({'state':copy.deepcopy(states[i]),'legal':False,'terminated':True,'skipped':True})
    return outputs
