"""Metadata/byte freezer only; no owned/new review/helper/checker imports or tests.
Private listed producer/reference source payloads are opaque bytes: no code or
fixture semantics are parsed, displayed, imported or executed by this utility.
"""
import hashlib,importlib.util,json,os,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
INPUT=ROOT/'reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01'
PREP=ROOT/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'
CORRECTED=ROOT/'reviews/adversarial/RSM-option-resumption-public-interface-correction-20261010/attempt01'
PUBLIC=[ROOT/p for p in ['schemas/RSM_OPTION_RESUMPTION_V1.md','research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md',
  'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json','schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md',
  'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md','schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md',
  'schemas/canonical/v1.schema.json','schemas/SEMANTICS.md','schemas/EXTENDED_SEMANTICS.md']]
DATA=[ROOT/p for p in ['data/RSM-001/candidate_renderings/option-resumption-pilot-v1.jsonl',
  'data/RSM-001/semantic_parents/option-resumption-pilot-v1.jsonl','data/RSM-001/manifests/option-resumption-pilot-v1.family_controls.json',
  'reviews/input_only/RSM-legacy-v0001.jsonl','reviews/input_only/RSM-legacy-v0001.manifest.json',
  'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl','reviews/input_only/RSM-option-resumption-pilot-v1.INPUT_CUSTODY.json']]
EXPECTED={
 'data/RSM-001/candidate_renderings/option-resumption-pilot-v1.jsonl':('c20eebd1eba211bd069ae9424d2d5a8094f657432fbe25ffffdcc5d307756bef',6032762),
 'reviews/input_only/RSM-legacy-v0001.jsonl':('ba62d973531521bfc57a5b52ee630a61f3384d9e8dfd41becea6a1eb63ca86c1',71432),
 'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl':('d99074c6cd18d1bd9709a09f833bfe9fea50c0c3981ece532c3951aac220f6b1',4512472),
 str((INPUT/'frame_predictions.jsonl').relative_to(ROOT)):('d67500ab37d1835d21511fe8cf6a9934cf8dd78cd019abddc5d13846567d466b',518392),
 str((INPUT/'recursive_predictions.jsonl').relative_to(ROOT)):('231679b1cdc1b31785612a9bf4baab5131ff32d5c2841903835015afd19059c6',519472),
}
# Only actual manifest source-binding paths are accepted as opaque private source
# custody. No root/private formal/expert/receipt metadata aliases are copied.
OPAQUE_ALLOWED={
 'factory/common.py',
 'generators/skills/option_resumption_pilot_v1_attempt01/PRODUCTION_CONFIG_ATTEMPT01.json',
 'generators/skills/option_resumption_pilot_v1_attempt01/calibration.py',
 'generators/skills/option_resumption_pilot_v1_attempt01/generator.py',
 'generators/skills/option_resumption_pilot_v1_attempt01/normalization.py',
 'generators/skills/option_resumption_pilot_v1_attempt01/runner.py',
 'generators/skills/prospective_option_resumption_attempt01/hand_fixtures_attempt01.json',
 'generators/skills/prospective_option_resumption_attempt01/reference_attempt01.py',
}
def binding(path,category):
    p=Path(path).resolve();b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
    name=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if name in EXPECTED:assert (h,len(b))==EXPECTED[name],name+' authorized binding'
    assert len(b)<25000000,'raw archive cap'
    dest=OUT/'source_archives'/(h+'.source')
    if not dest.exists():
        with dest.open('xb') as f:f.write(b)
    assert dest.read_bytes()==b,'archive roundtrip'
    return {'path':name,'size_bytes':len(b),'sha256':h,'category':category,'archive':str(dest.relative_to(ROOT))}
def main():
    config=json.loads((OUT/'RUN_CONFIG.json').read_text())
    manifest=json.loads((ROOT/config['manifest_path']).read_text())
    semantic_paths=PUBLIC+DATA+list(sorted(INPUT.rglob('*')))
    semantic_paths=[p for p in semantic_paths if p.is_file()]
    semantic_paths += [PREP/p for p in ['frame_solver_r1.py','recursive_simulator_r1.py','hand_fixtures_r1.json','test_preparation_r1.py','FINAL_RECEIPT.json','FINAL_COUNT_CLARIFICATION.json','FINAL_CUSTODY_R2.json']]
    semantic_paths += [CORRECTED/p for p in ['control_probes_r2.py','test_scoped_correction_r1.py','hand_fixtures.json','FINAL_REVIEW.json','FINAL_CUSTODY.json']]
    semantic_paths += [p for p in sorted(OUT.iterdir()) if p.is_file()]
    categories={p.resolve():'authorized public/data/own immutable evidence or new preparation' for p in semantic_paths}
    for name,h in manifest['source_bindings'].items():
        p=ROOT/name
        assert name in OPAQUE_ALLOWED or p in PUBLIC or p in DATA,'ungranted extra source binding'
        assert not p.is_symlink() and p.resolve().is_relative_to(ROOT),'literal source binding path'
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==h,'actual opaque/public source SHA'
        categories[p.resolve()]='OPAQUE PRIVATE BYTE CUSTODY ONLY' if name in OPAQUE_ALLOWED else 'public operative source binding'
    operative=[binding(p,c) for p,c in sorted(categories.items(),key=lambda x:str(x[0]))]
    runtime_paths={Path(sys.executable).resolve():'resolved actualPython'}
    modules=[]
    # All actually loaded stdlib files at freeze time, plus prospective direct
    # imports and their package sources, before any new review/helper imports.
    for name,module in sorted(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and Path(path).exists() and '/opt/anaconda3/' in str(Path(path).resolve()):runtime_paths[Path(path).resolve()]='actually loaded stdlib '+name
    requested=['copy','datetime','itertools','math','re','collections','pathlib','time','_datetime','_hashlib','_json','json.decoder','json.encoder','json.scanner','importlib.util']
    for name in requested:
        spec=importlib.util.find_spec(name)
        origin=spec.origin if spec else None;modules.append({'module':name,'origin':origin})
        if origin not in {None,'built-in','frozen'}:runtime_paths[Path(origin).resolve()]='prospective declared stdlib '+name
    runtime=[binding(p,c) for p,c in sorted(runtime_paths.items(),key=lambda x:str(x[0]))]
    # Make all actual runtime files operative so future execution checks their
    # before/after hashes together with data/own predictors/raw metadata sources.
    operative += runtime
    receipt={'actual_author_execution':'/root/adversarial_auditor','stage':'CANONICAL_SOURCE_PREPARATION_ONLY_NOT_EXECUTED',
      'new_comparison_source':binding(OUT/'canonical_review_r1.py','operative prepared source'),
      'operative_bindings':operative,'loaded_and_declared_stdlib_modules':modules,'python':sys.version,'resolved_python':str(Path(sys.executable).resolve()),
      'platform':platform.platform(),'six_public_bindings':[binding(p,'public') for p in PUBLIC[:6]],
      'actual_candidate':binding(ROOT/config['candidate_path'],'authorized gold raw archive'),
      'actual_parents':binding(ROOT/config['parent_path'],'authorized parents'),
      'actual_public_control_manifest':binding(ROOT/config['manifest_path'],'public untrusted metadata'),
      'actual_target_free_legacy':binding(ROOT/config['legacy_input_path'],'targetfree legacy'),
      'private_producer_reference_bytes_only':True,'no_four_original_c35_metadata_payloads_copied':True,
      'opaque_private_source_paths':sorted(OPAQUE_ALLOWED),'new_helpers_imported':False,'comparison_executed':False,'new_tests_executed':False,
      'root_execution_grant_pending':True,'no_dataset_verdict_or_publication_approval':True,
      'disclosure':'same model/commonpublic design, own frozen solver/grammar/calibration/input-first reuse; actual gold/metadata layout read only after inputfirstfreeze. No other private qualityvotes read.'}
    p=OUT/'SOURCE_FREEZE.json'
    with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    # Final preparation custody excludes itself but binds all current raw archives
    # and source/config/fixture-plan files. New reports later use fresh names.
    files=[{'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob('*')) if p.is_file()]
    final={'actual_author_execution':'/root/adversarial_auditor','phase':'PREPARATION_FINAL_ONLY','source_freeze_sha256':hashlib.sha256((OUT/'SOURCE_FREEZE.json').read_bytes()).hexdigest(),
      'comparison_source_sha256':receipt['new_comparison_source']['sha256'],'files':files,'all_archives_below25MB':True,'comparison_imports_tests_execution':False,'stop_writes':True}
    p=OUT/'PREPARATION_FINAL_CUSTODY.json'
    with p.open('x') as f:json.dump(final,f,indent=2);f.write('\n')
    print(json.dumps({'source_sha256':final['comparison_source_sha256'],'source_freeze_sha256':final['source_freeze_sha256'],'custody_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'operative_bindings':len(operative),'runtime_files':len(runtime),'preparation_files':len(files),'comparison_executed':False,'new_tests_executed':False,'stop_writes':True}))
if __name__=='__main__':main()
