"""Owned byte-freeze utility. No task/gold parsing or solver imports."""
import hashlib, json, os, platform, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PUBLIC = [
    'schemas/RSM_OPTION_RESUMPTION_V1.md',
    'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md',
    'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json',
    'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md',
    'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md',
    'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md',
    'schemas/canonical/v1.schema.json', 'schemas/SEMANTICS.md',
    'schemas/EXTENDED_SEMANTICS.md',
]
PREP = 'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z/'
CORRECTED = 'reviews/adversarial/RSM-option-resumption-public-interface-correction-20261010/attempt01/'
DEPENDENCIES = [PREP + x for x in ['frame_solver_r1.py', 'recursive_simulator_r1.py',
    'hand_fixtures_r1.json','test_preparation_r1.py','RUN_CONFIG_R1.json','INPUT_FIRST_PROTOCOL.md',
    'PREEXEC_R1.json','FINAL_RECEIPT.json','FINAL_COUNT_CLARIFICATION.json','FINAL_CUSTODY_R2.json']]
DEPENDENCIES += [CORRECTED + x for x in ['control_probes_r2.py','FINAL_REVIEW.json','FINAL_CUSTODY.json']]
INPUT = ['reviews/input_only/RSM-option-resumption-pilot-v1.jsonl',
         'reviews/input_only/RSM-option-resumption-pilot-v1.INPUT_CUSTODY.json']
EXPECTED = {
 INPUT[0]:'d99074c6cd18d1bd9709a09f833bfe9fea50c0c3981ece532c3951aac220f6b1',
 INPUT[1]:'231250e9819e72608a9a34bc32b3efea22f2a7943b5e4a0eacf8560015457e76',
 PREP+'frame_solver_r1.py':'407268916c35b10e36b18d7a7d3b7a37c880306237dafe3b12bd1f4942eeaf24',
 PREP+'recursive_simulator_r1.py':'032723d5db52e3a76a9583c02332aa7cf6fc22759bba0739eddc265e37f3db5a',
 CORRECTED+'control_probes_r2.py':'75d4282375afbec8ed552bfa5f36da2d1b907f7ddd5a810b36b534def1c997a7',
}

def binding(path, archive=True):
    p = Path(path)
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if name in EXPECTED and h != EXPECTED[name]:
        raise RuntimeError('authorized binding mismatch: '+name)
    result = {'path':name,'size_bytes':len(raw),'sha256':h}
    if archive:
        dest = OUT/'source_archives'/(h+'.source')
        if not dest.exists():
            with dest.open('xb') as f: f.write(raw)
        if dest.read_bytes() != raw: raise RuntimeError('archive mismatch')
        result['archive'] = str(dest.relative_to(ROOT))
    return result

if __name__ == '__main__':
    mode = sys.argv[1]
    paths = PUBLIC + DEPENDENCIES + INPUT + [str(Path(__file__))]
    if mode == 'preexec':
        paths += [str(p) for p in sorted(OUT.glob('*.py')) if p.name != Path(__file__).name]
        paths += [str(p) for p in sorted(OUT.glob('RUN_CONFIG*.json'))]
    records = [binding(ROOT/p if not Path(p).is_absolute() else p) for p in paths]
    runtime = [binding(Path(sys.executable).resolve(),archive=True)]
    runtime += [binding(Path(x.__file__),archive=True) for x in [json,hashlib,platform]]
    receipt = {'actor':'/root/adversarial_auditor','phase':mode,'source_bindings':records,
               'runtime_bindings':runtime,'python':sys.version,'platform':platform.platform(),
               'no_solver_imports_by_this_freezer':True,'no_candidate_gold_access':True,
               'scope':'raw allowed public/input/own dependency archives before execution',
               'disclosure':'same-model/public-guided, prior own handmade preparation and grammar reuse; prior backup opaque-byte exposure only'}
    target = OUT/('ACCESS_FREEZE.json' if mode=='access' else 'PREEXECUTION_FREEZE.json')
    with target.open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps({'path':str(target.relative_to(ROOT)), 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'bindings':len(records)+len(runtime)}))
