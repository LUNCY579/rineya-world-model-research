"""Root custody/view creation for a disclosed same-task renderer revision."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from factory.catalog import catalog
from factory.common import ROOT, canonical
from factory.lineage import source_key
from factory.pipeline import schema_validate

OLD = 'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl'
NEW = 'data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl'
HASHES = {OLD: '57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23',
          NEW: 'b306f67b2208406027d26d95094c649913e590291857764a9bf17edb9bcd923c'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r): return source_key(r), r['controlled_context'], r['renderer']
def main():
    view = ROOT/'reviews/input_only/planning-gap-v2-renderer-r1.jsonl'
    freeze = ROOT/'reviews/input_only/planning-gap-v2-renderer-r1.freeze.json'
    assert not view.exists() and not freeze.exists()
    for p,h in HASHES.items(): assert sha(ROOT/p) == h
    old = [json.loads(l) for l in (ROOT/OLD).read_bytes().splitlines()]
    rows = [json.loads(l) for l in (ROOT/NEW).read_bytes().splitlines()]
    original = {key(r):r for r in old}
    assert len(original) == len(rows) == len(old) == 408
    assert {key(r) for r in rows} == set(original)
    assert len({r['sample_id'] for r in rows}) == 408
    assert not {r['sample_id'] for r in rows}.intersection(r['sample_id'] for r in old)
    public=[]; joins=[]; changed=[]
    for r in rows:
        schema_validate(r); o=original[key(r)]
        for name in ('semantic_task','verified_target','split','controlled_context','primary_stratum','difficulty','structural_difficulty','skill_tags','target_format'):
            assert canonical(r[name]) == canonical(o[name]), (key(r),name)
        assert r['review_status']=='PENDING' and r['review_receipt_ids']==[]
        public.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'source_family_key':source_key(r),'split':r['split'],'semantic_task':r['semantic_task'],'rendered_input':r['rendered_input']})
        joins.append({'source_family_key':source_key(r),'controlled_context':r['controlled_context'],'renderer':r['renderer'],'original_sample_id':o['sample_id'],'original_record_hash':o['record_hash'],'revision_sample_id':r['sample_id'],'revision_record_hash':r['record_hash'],'exact_typed_semantic_task_and_target_join':True,'visible_input_changed':r['rendered_input']!=o['rendered_input']})
        if r['rendered_input']!=o['rendered_input']: changed.append(r['sample_id'])
    assert len(changed)==22
    assert Counter(r['split'] for r in rows)==Counter(train=340,dev_challenge=68)
    families={source_key(r) for r in rows}; assert len(families)==48
    active,report=catalog(); assert len(active)==3340 and report['active_source_families']==648
    assert not families.intersection(source_key(r) for r in active)
    data=''.join(canonical(r)+'\n' for r in public).encode()
    with view.open('xb') as f:f.write(data)
    for p,h in HASHES.items(): assert sha(ROOT/p)==h
    payload={'root_execution':'/root','at_utc':datetime.now(timezone.utc).isoformat(),'source_script_sha256':sha(Path(__file__)), 'candidate_bindings':HASHES,'input_view_sha256':hashlib.sha256(data).hexdigest(),'input_view_bytes':len(data),'input_fields':sorted(public[0]),'rows':408,'reused_semantic_tasks':204,'reused_source_families':48,'new_semantic_tasks':0,'new_source_families':0,'visible_inputs_changed':22,'all408_exact_typed_semantics_and_targets_preserved':True,'split_counts':dict(Counter(r['split'] for r in rows)),'active_approved_records':3340,'active_source_families':648,'canonical_bytes_unchanged':True,'prior_canonical_exposure_disclosed':True,'unseen_blind_review_claim':False,'own_parent_prediction_reuse_permitted_only_after_exact_input_semantic_joins':True,'review_requirement':'Each actual reviewer must bind fresh input bytes and freeze own all408 input-derived/reused predictions and first judgments before fresh canonical candidate/control-manifest access; disclose original canonical exposure and correlated same-task reuse. No producer private/formal private/other verdict reads.','scope':'Custody/input-only reviewer view; target equality checked by root but no targets exported. No approval or student export.','exact_custody_joins':joins,'changed_sample_ids':changed}
    with freeze.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps({'rows':408,'input_view_sha256':payload['input_view_sha256'],'freeze_sha256':sha(freeze),'changed_visible_inputs':22,'new_semantic_tasks':0}))

if __name__=='__main__':main()
