"""Prospective CPU11 admission; inherited precision/model/adapter load semantics.

The historical native loader remains unchanged. Callers still acquire its CPU
lane and use its four-GiB work-boundary check. This profile changes admission
only, and requires exactly four CPU threads.
"""
import os
from .native import EXPECTED_CORE_SHA, EXPECTED_SIDECARS, sha256, runtime, boundary

ADMISSION_PROFILE = {
    "name": "dsv3wm-cpu11-v1",
    "device": "cpu",
    "threads": 4,
    "load_free_host_ram_floor_gib": 11,
    "admission_memory_kinds": ["physical", "commit"],
    "historical_load_free_host_ram_floor_gib": 12,
    "unchanged_boundary_free_host_ram_floor_gib": 4,
    "runtime": "dsv3wm.native.runtime",
    "boundary": "dsv3wm.native.boundary",
    "sampler_cpu_batch_tokens_parameter": 8000,
    "sampler_batch_geometry": "unchanged inherited heuristic; not a hard padded-token cap",
    "sampler_ram_floor_gib": 8,
    "prediction_change": False,
}
_LAST_ADMISSION_CHECK = None


def validate_environment():
    if os.environ.get('BWM_CPU_BATCH_TOKENS') not in (None, '8000'):
        raise ValueError('CPU11 requires unchanged BWM_CPU_BATCH_TOKENS unset or8000')
    raw_floor = os.environ.get('BWM_RAM_FLOOR_GB')
    try:
        if raw_floor is not None and float(raw_floor) != 8.:
            raise ValueError('CPU11 requires unchanged sampler RAM floor8 GiB')
    except (ValueError, TypeError):
        raise ValueError('CPU11 requires BWM_RAM_FLOOR_GB unset or equivalent to8 GiB') from None


def admission_profile():
    validate_environment()
    return {**ADMISSION_PROFILE, 'environment': {name: os.environ.get(name) for name in
            ('BWM_CPU_BATCH_TOKENS', 'BWM_RAM_FLOOR_GB')}}


def last_admission_check():
    return None if _LAST_ADMISSION_CHECK is None else dict(_LAST_ADMISSION_CHECK)


def load(config, adapter=None):
    global _LAST_ADMISSION_CHECK
    _LAST_ADMISSION_CHECK = None
    if config.device != 'cpu':
        raise ValueError('CPU11 admission permits CPU only')
    if type(config.threads) is not int or config.threads != 4:
        raise ValueError('CPU11 admission requires exactly four CPU threads')
    validate_environment()
    from bwmn import lm
    import torch
    physical, commit = lm.host_ram_free_gb()
    passed = physical >= 11 and commit >= 11
    _LAST_ADMISSION_CHECK = {
        'physical_available_gib': physical,
        'commit_available_gib': commit,
        'required_physical_gib': 11,
        'required_commit_gib': 11,
        'physical_qualified': physical >= 11,
        'commit_qualified': commit >= 11,
        'passed': passed,
        'status': 'admitted' if passed else 'refused',
        'measurement_stage': 'after BWM/torch imports; before core hash/load',
    }
    if not passed:
        raise RuntimeError('native CPU11 load requires at least 11 GiB physical and commit availability')
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
