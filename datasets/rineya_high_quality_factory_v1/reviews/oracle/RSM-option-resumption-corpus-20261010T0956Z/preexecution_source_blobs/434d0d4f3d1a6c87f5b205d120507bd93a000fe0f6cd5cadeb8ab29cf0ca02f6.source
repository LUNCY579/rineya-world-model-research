"""Frozen deterministic hand-cohort constructor; no production source/input.

Uses own prior public-only feasibility programs by exact hash, then official
independent solver to derive original histories/checkpoints.  These old graphs
are deliberately exposed developer fixtures, not proposed producer families.
"""
from copy import deepcopy
import importlib.util
import json
from hashlib import sha256

FIXTURE_PATH='reviews/oracle/RSM-option-resumption-prospective-20261010T083826Z/fixtures_attempt01.py'
FIXTURE_SHA='cdc1d44d370e54cd879534b4379ce5df40d04f6fe78cf20d4d5fc9acdad6378f'

def make_contexts(checker,base,ordinal=0):
    path=checker.ROOT/FIXTURE_PATH
    if sha256(path.read_bytes()).hexdigest()!=FIXTURE_SHA: raise ValueError('frozen hand fixture hash')
    spec=importlib.util.spec_from_file_location('rsm_own_fixture_contexts',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks,_,_=module.contexts(base,checker.solve)
    cp=tasks['X']['checkpoint']
    if ordinal%3==0: cp['state']['x']+=1
    elif ordinal%3==1: cp['frames'][-1]['next_index']+=1
    else: cp['frames'][-1]['bindings']['b']+=1
    return tasks

def build(checker):
    path=checker.ROOT/FIXTURE_PATH
    if sha256(path.read_bytes()).hexdigest()!=FIXTURE_SHA: raise ValueError('frozen hand fixture hash')
    spec=importlib.util.spec_from_file_location('rsm_own_fixtures',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    public=(checker.ROOT/'schemas/RSM_OPTION_RESUMPTION_V1.md').read_text()
    source_bytes={p:(checker.ROOT/p).read_bytes() for p in checker.PUBLIC}
    manifest_path='reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/hand-control-manifest.json'
    rows=[]; coordinates={}; by_family={}
    for ordinal,family in enumerate(module.hand_programs()):
        idx=family['source_family_index']; key='hand-origin-'+str(idx)
        tasks,_,_=module.contexts(family['task'],checker.solve)
        cp=tasks['X']['checkpoint']
        if ordinal%3==0: cp['state']['x']+=1
        elif ordinal%3==1: cp['frames'][-1]['next_index']+=1
        else: cp['frames'][-1]['bindings']['b']+=1
        by_family[key]=tasks
        for context in checker.CONTEXTS:
            for style in checker.STYLES:
                task=deepcopy(tasks[context]); presentation=task if style=='forward' else module.reverse_objects(task)
                sid='arbitrary-'+str(len(rows)+101)
                row={'sample_id':sid,'semantic_parent_id':'hand-parent-'+str(idx)+'-'+context,
                     'semantic_family_id':'hand-semantic-'+str(idx),'track_id':'RSM-001',
                     'source_family_key':key,'source_family_index':idx,'controlled_context':context,'renderer':style,
                     'skill_tags':['finite_supplied_policy','hand_development_fixture'],'difficulty':3,'structural_difficulty':3,
                     'generator_version':'hand-fixture-1','oracle_version':'own-task427','generation_seed':idx,
                     'source_provenance':{'curriculum_difficulty_explanation':'Saved register-derived arguments remain in nested caller/selected-branch frames after guard reversal; validating committed history and the pending continuation requires these dependencies.'},
                     'property_witnesses':{},'family_control_claims':[], 'semantic_task':task,
                     'rendered_input':public+'\nTask JSON:\n'+json.dumps(presentation,ensure_ascii=False,separators=(',',':')),
                     'verified_target':checker.solve(task),'target_format':'json','split':family['split'],
                     'review_status':'PENDING','review_receipt_ids':[]}
                rows.append(row); coordinates[key,context,style]=row
    entries=[]
    for row in rows:
        family=row['source_family_key']; style=row['renderer']; context=row['controlled_context']
        for j,(name,contexts) in enumerate(checker.RELATIONS.items()):
            pair=[coordinates[family,c,style] for c in contexts] if contexts else [coordinates[family,context,s] for s in checker.STYLES]
            ids=[v['sample_id'] for v in pair]
            # Shared literal ID across distinct properties is intentionally legal.
            cid=' shared public evidence '+family+' '+style if contexts else 'view '+family+' '+context
            claim={'name':name,'scope':'corpus_variant','evidence_manifest':manifest_path,'control_id':cid,'pair_sample_ids':ids}
            if j%3==1: claim['property']=claim.pop('name')
            if j%3==2: claim['property']=name
            if j==0: claim.update(source_family_key=family,source_family_index=row['source_family_index'],controlled_context=context,renderer=style,diagnostics={'untrusted':'checked independently'})
            row['family_control_claims'].append(claim)
            lookup=(cid,name,tuple(ids))
            if not any((e['control_id'],checker.identifier(e),tuple(e['pair_sample_ids']))==lookup for e in entries):
                entries.append({'property':name,'scope':'corpus_variant','control_id':cid,'pair_sample_ids':ids,
                   'source_family_key':family,'source_family_index':row['source_family_index'],
                   'controlled_contexts':[v['controlled_context'] for v in pair],'renderers':[v['renderer'] for v in pair],
                   'candidate_sha256':None,'diagnostic_annotation':{'optional':'open entry diagnostics'}})
    for row in rows: row['record_hash']=checker.record_hash(row)
    candidate=b''.join((checker.canonical(row)+'\n').encode() for row in rows)
    h=sha256(candidate).hexdigest(); byid={r['sample_id']:r for r in rows}
    for e in entries:
        e['candidate_sha256']=h; e['pair_record_hashes']=[byid[v]['record_hash'] for v in e['pair_sample_ids']]
    manifest={'metadata_version':1,'batch':'option-resumption-pilot-v1','candidate_sha256':h,
              'source_bindings':deepcopy(checker.PUBLIC),'control_pairs':entries,'open_top_diagnostic':'not proof'}
    return {'rows':rows,'manifest':manifest,'manifest_path':manifest_path,'candidate_bytes':candidate,
            'manifest_sources':{manifest_path:checker.canonical(manifest).encode()},'source_bytes':source_bytes,
            'legacy_input_bytes':(checker.ROOT/'reviews/input_only/RSM-legacy-v0001.jsonl').read_bytes()}

def recustody(checker,bundle):
    """Rehash deliberate typed negative mutations; no stale-hash shortcut."""
    for row in bundle['rows']: row['record_hash']=checker.record_hash(row)
    raw=b''.join((checker.canonical(row)+'\n').encode() for row in bundle['rows']); bundle['candidate_bytes']=raw
    h=sha256(raw).hexdigest(); byid={r['sample_id']:r for r in bundle['rows']}
    bundle['manifest']['candidate_sha256']=h
    for e in bundle['manifest']['control_pairs']:
        e['candidate_sha256']=h
        if 'pair_record_hashes' in e and all(v in byid for v in e['pair_sample_ids']): e['pair_record_hashes']=[byid[v]['record_hash'] for v in e['pair_sample_ids']]
    bundle['manifest_sources']={bundle['manifest_path']:checker.canonical(bundle['manifest']).encode()}
    return bundle

def rerender(checker,row):
    def reverse(v):
        if type(v) is dict: return {k:reverse(vv) for k,vv in reversed(list(v.items()))}
        if type(v) is list: return [reverse(x) for x in v]
        return v
    t=row['semantic_task']; p=t if row['renderer']=='forward' else reverse(t)
    row['rendered_input']=(checker.ROOT/'schemas/RSM_OPTION_RESUMPTION_V1.md').read_text()+'\nTask JSON:\n'+json.dumps(p,ensure_ascii=False,separators=(',',':'))
