"""Read-only reuse of the validated BWM native loader and unmerged LoRA.

Callers acquire the shared GPU lease BEFORE any CUDA availability query/load.
CPU mode is set before importing BWM; its CPU-driver guard is preserved.
"""
import os
import sys
import hashlib
from pathlib import Path
from contextlib import contextmanager
from . import REPO

EXPECTED_CORE_SHA = 'a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def runtime(config, job):
    os.environ['BWM_DEVICE'] = config.device
    sys.path.insert(0, str(REPO / 'bwm' / 'code'))
    if config.device == 'cuda':
        from bwmn.gpu_lease import lease
        with lease('dsv3wm:' + job, job, timeout_seconds=0):
            yield
    else:
        yield


def load(config, adapter=None):
    from bwmn import lm
    import torch
    torch.set_num_threads(config.threads)
    lm.set_determinism(config.seed)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=True)
    core_path = lm.MODELS[config.core]
    actual = sha256(core_path / 'model.safetensors')
    if actual != EXPECTED_CORE_SHA:
        raise RuntimeError('parent checkpoint changed; audit and preregistration required')
    model, tokenizer = lm.load_core(config.core)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('FP32 parent required')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer, lm


def response_prompt(conditioning, target_mode):
    import json
    # A single serializer serves all mechanisms. No state labels/goal in conditioning.
    payload = {**conditioning, 'output_mode': target_mode}
    return ('Predict the resulting state after the ordered actions. Illegal actions leave the state unchanged. '
            'Return a JSON object with state (or delta), legal, terminated, failed.\n'
            + json.dumps(payload, sort_keys=False, separators=(',', ':')) + '\nAnswer:')


class Predictor:
    """Generation-only learned component. No simulator, repair oracle, or teacher states."""
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.calls = 0
        self.input_tokens = self.output_tokens = 0

    def __call__(self, conditioning):
        import json
        import torch
        mode = 'full' if self.config.target_mode == 'mixed' else self.config.target_mode
        prompt = response_prompt(conditioning, mode)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        if inputs.input_ids.shape[1] > self.config.max_length:
            return {'malformed': True, 'error': 'overlength'}
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        self.calls += 1
        self.input_tokens += int(inputs['input_ids'].numel())
        self.model.eval()
        with torch.no_grad():
            # Cache disabled until an independent native cache-parity gate is met.
            seq = self.model.generate(**inputs, max_new_tokens=self.config.generation_tokens,
                                      do_sample=False, use_cache=False, pad_token_id=self.tokenizer.pad_token_id)
        new = seq[0, inputs['input_ids'].shape[1]:]
        self.output_tokens += int(new.numel())
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError('object required')
            if mode == 'delta':
                from .domains import apply_delta
                value['state'] = apply_delta(conditioning['domain'], conditioning['s0'], value['delta'])
            if 'state' not in value or type(value.get('legal')) is not bool:
                raise ValueError('missing state/boolean legality')
            return value
        except (ValueError, KeyError, TypeError):
            return {'malformed': True, 'text': text}
