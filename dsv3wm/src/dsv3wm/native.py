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
EXPECTED_SIDECARS = {
    'config.json':'25ace0268b891881bf1d5a95d18611a74ce1896922107d10adc526257a1962e8',
    'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(path):
    import json
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    for name, expected in record['files'].items():
        actual = sha256(REPO / name)
        if actual != expected:
            raise RuntimeError(f'freeze drift: {name}; append an amendment before proceeding')
    return sha256(path)


@contextmanager
def runtime(config, job):
    os.environ['BWM_DEVICE'] = config.device
    sys.path.insert(0, str(REPO / 'bwm' / 'code'))
    if config.device == 'cuda':
        from bwmn.gpu_lease import lease
        with lease('dsv3wm:' + job, job, timeout_seconds=0):
            yield
    else:
        import importlib.util
        path = REPO / 'bwm' / 'code' / 'scripts' / 'planbench_qwen.py'
        spec = importlib.util.spec_from_file_location('_dsv3wm_cpu_lane', path)
        lane = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lane)
        # Reuse protocol functions verbatim, with this programme's own identity.
        lane.LANE_PARTY = 'dsv3wm'
        lane.LANE_PRIORITY = ('rcx', 'bwm')
        record = lane.lane_acquire(job, expected_s=600, wait=False)
        try:
            yield
        finally:
            lane.lane_release(record)


def boundary(config):
    """Yield at work boundaries when an earlier programme requests the CPU lane."""
    if config.device == 'cpu':
        import tempfile
        directory = Path(os.environ.get('RINEYA_LANE_DIR') or os.environ.get('TEMP') or tempfile.gettempdir())
        if any((directory / ('rineya-cpu-lane.waiting.' + party)).exists() for party in ('rcx','bwm')):
            raise RuntimeError('CPU lane priority waiter; yield and preserve failed/partial run')
        import psutil
        if psutil.virtual_memory().available < 4 * 2**30:
            raise RuntimeError('native CPU boundary free RAM below 4 GiB')


def load(config, adapter=None):
    from bwmn import lm
    import torch
    if config.device == 'cpu' and lm.host_ram_free_gb()[0] < 12:
        raise RuntimeError('native CPU load requires at least 12 GiB free host RAM')
    torch.set_num_threads(config.threads)
    lm.set_determinism(config.seed)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=True)
    core_path = lm.MODELS[config.core]
    actual = sha256(core_path / 'model.safetensors')
    if actual != EXPECTED_CORE_SHA:
        raise RuntimeError('parent checkpoint changed; audit and preregistration required')
    for name, expected in EXPECTED_SIDECARS.items():
        if sha256(core_path / name) != expected:
            raise RuntimeError(f'parent sidecar changed: {name}')
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
    def __init__(self, model, tokenizer, config, max_decoder_token_positions=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.calls = 0
        self.input_tokens = self.output_tokens = 0
        self.decoder_token_positions = 0
        self.max_decoder_token_positions = max_decoder_token_positions
        self.budget_exhausted = False

    def __call__(self, conditioning):
        import json
        import torch
        boundary(self.config)
        mode = 'full' if self.config.target_mode == 'mixed' else self.config.target_mode
        prompt = response_prompt(conditioning, mode)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        if inputs.input_ids.shape[1] > self.config.max_length:
            return {'malformed': True, 'error': 'overlength'}
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        n = int(inputs['input_ids'].numel())
        generation_limit = self.config.generation_tokens
        if self.max_decoder_token_positions is not None:
            remaining = self.max_decoder_token_positions - self.decoder_token_positions
            # No cache: decoder token positions for g generated tokens = n*g + g*(g-1)/2.
            while generation_limit > 0 and n * generation_limit + generation_limit * (generation_limit - 1) // 2 > remaining:
                generation_limit -= 1
            if generation_limit == 0:
                self.budget_exhausted = True
                return {'malformed':True, 'error':'decoder-token-budget-exhausted'}
        self.calls += 1
        self.input_tokens += n
        self.model.eval()
        with torch.no_grad():
            # Cache disabled until an independent native cache-parity gate is met.
            seq = self.model.generate(**inputs, max_new_tokens=generation_limit,
                                      do_sample=False, use_cache=False, pad_token_id=self.tokenizer.pad_token_id)
        new = seq[0, inputs['input_ids'].shape[1]:]
        self.output_tokens += int(new.numel())
        g = int(new.numel())
        self.decoder_token_positions += n * g + g * (g - 1) // 2
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return parse_prediction(text, conditioning, mode)


def parse_prediction(text, conditioning, mode='full'):
    import json
    from .planning import validate_state
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant')
    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        state_key = 'delta' if mode == 'delta' else 'state'
        if not isinstance(value, dict) or set(value) != {state_key,'legal','terminated','failed'}:
            raise ValueError('exact response schema required')
        if any(type(value[k]) is not bool for k in ('legal','terminated','failed')):
            raise ValueError('boolean flags required')
        if mode == 'delta':
            from .domains import apply_delta
            value['state'] = apply_delta(conditioning['domain'], conditioning['s0'], value['delta'])
        if not validate_state(conditioning['domain'], value['state'], conditioning['rules']):
            raise ValueError('invalid state representation')
        return value
    except (ValueError, KeyError, TypeError, OverflowError):
        return {'malformed': True, 'text': text}
