"""Pure input/lineage binding; never loads models or confirmation labels."""
import json
from pathlib import Path
from . import REPO
from .native import sha256


def verify_bound_inputs(freeze_path,paths):
    frozen=json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    identities={(REPO/name).resolve():digest for name,digest in frozen['files'].items()}
    for item in paths:
        path=Path(item).resolve()
        if path not in identities:
            raise ValueError(f'input absent from freeze: {path}')
        if sha256(path)!=identities[path]:
            raise ValueError(f'input hash differs from freeze: {path}')


def validate_training_lineage(directory,dataset,arm):
    directory,dataset=Path(directory).resolve(),Path(dataset).resolve()
    record=json.loads((directory/'run.json').read_text(encoding='utf-8'))
    if record['status']!='pilot-trained' or record['config']['arm']!=arm:
        raise ValueError('trained arm identity mismatch')
    config=(REPO/record['command'][record['command'].index('--config')+1]).resolve()
    if sha256(config)!=record['config_sha256'] or sha256(dataset/'train_register.jsonl')!=record['data_sha256']:
        raise ValueError('recorded config/data lineage mismatch')
    from .config import Config
    if Config.read(config).dict()!=record['config']:
        raise ValueError('expanded config mismatch')
    model=directory/'adapter/adapter_model.safetensors'
    if sha256(model)!=record['adapter_sha256']:
        raise ValueError('recorded adapter weight lineage mismatch')
    adapter_config=directory/'adapter/adapter_config.json'
    peft=json.loads(adapter_config.read_text(encoding='utf-8'))
    if (peft['r']!=record['config']['rank'] or peft['lora_alpha']!=2*peft['r']
        or peft['lora_dropout']!=0.0 or set(peft['target_modules'])!={'q_proj','v_proj'}):
        raise ValueError('adapter config differs from operational training recipe')
    from .native import EXPECTED_CORE_SHA
    if record['core_sha256']!=EXPECTED_CORE_SHA:
        raise ValueError('recorded parent checkpoint mismatch')
    return config,adapter_config
