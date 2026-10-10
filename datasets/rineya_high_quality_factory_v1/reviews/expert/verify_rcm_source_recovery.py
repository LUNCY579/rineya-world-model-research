"""Read-only RCM lineage/current-generation verification.

No producer expectation/source body or other reviewer verdict is inspected.
The current generator executes its normal solver dependency for reproduction;
semantic checks separately use the expert-owned public-prompt interpreter.
All generator filesystem writes and candidate-file reads are disabled.
"""
import collections
import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from independent_expansion import BASE,OUT,canonical,sha
from independent_review import causal,same

sys.path.insert(0,str(BASE));sys.dont_write_bytecode=True

def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    candidate_path=BASE/'data/RCM-001/candidate_renderings/expansion-v1.jsonl'
    receipt_path=OUT/'expansion-v1.jsonl'
    release=BASE/'data/RCM-001/releases/v0002'
    manifest_path=release/'MANIFEST.json';canonical_path=release/'canonical.jsonl'
    candidates=[json.loads(l) for l in candidate_path.read_text().splitlines()]
    receipts={r['sample_id']:r for r in [json.loads(l) for l in receipt_path.read_text().splitlines()] if r['sample_id'].startswith('RCM-001-')}
    released=[json.loads(l) for l in canonical_path.read_text().splitlines()]
    manifest=json.loads(manifest_path.read_text())
    assert len(candidates)==len(receipts)==len(released)==1200
    lookup={r['sample_id']:r for r in candidates};failures=[];counts=collections.Counter()
    for c in candidates:
        assert sha({k:v for k,v in c.items() if k!='record_hash'})==c['record_hash']
        receipt=receipts[c['sample_id']]
        assert receipt['record_hash']==c['record_hash'] and receipt['decision']=='ACCEPT'
        assert receipt['reviewer_execution_id']=='/root/expert_reviewer'
        task=json.loads(c['rendered_input'].split('Task JSON:',1)[1]);assert same(task,c['semantic_task'])
        result=causal(task);expected={'after':result['after']} if c['target_format']=='answer_only' else result
        assert same(expected,c['verified_target'])
        counts['candidate_hash_exact']+=1;counts['original_expert_exact_accept']+=1;counts['fresh_expert_semantics_match']+=1
    for r in released:
        c=lookup[r['sample_id']];assert sha({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash']
        assert r['reviewed_candidate_hash']==c['record_hash']
        assert receipts[r['sample_id']]['receipt_id'] in r['review_receipt_ids']
        assert len(r['review_receipt_ids'])==3 and r['review_status']=='ACCEPTED'
        restore=copy.deepcopy(r);restore.pop('reviewed_candidate_hash');restore['review_status']=c['review_status'];restore['review_receipt_ids']=c['review_receipt_ids'];restore['record_hash']=c['record_hash']
        assert same(restore,c)
        counts['release_hash_exact']+=1;counts['release_original_candidate_lineage_exact']+=1;counts['release_three_receipt_IDs_and_own_membership']+=1
    assert file_sha(canonical_path)==manifest['files']['canonical.jsonl']['sha256']
    # Hash bytes only: source body/producer expectation contents are not read by
    # the model. This is current-byte provenance, not historical source recovery.
    current_hashes={p:file_sha(BASE/p) for p in manifest['reproducibility_code_sha256'] if (BASE/p).exists()}
    differences={p:{'historical_sha256':v,'current_sha256':current_hashes.get(p)} for p,v in manifest['reproducibility_code_sha256'].items() if current_hashes.get(p)!=v}
    before_sources=current_hashes.copy();before_candidate_sha=file_sha(candidate_path);before_release_sha=file_sha(canonical_path)
    # Current generate() includes immutable-output checks. A nonexistent
    # destination prevents any reuse of candidate bytes; writer calls are
    # captured in memory, and an audit hook rejects every real mutation.
    active={'generation':True};write_requests=[];candidate_reads=[];blocked_mutations=[]
    mutation_events={'os.remove','os.rename','os.mkdir','os.rmdir','os.symlink','os.link','os.chmod','os.chown','os.truncate'}
    def audit(event,args):
        if not active['generation']:return
        if event in mutation_events:
            blocked_mutations.append({'event':event,'args':str(args)});raise RuntimeError('Generation filesystem mutation forbidden: '+event)
        if event=='open':
            path,mode,flags=args
            if isinstance(path,(str,bytes,os.PathLike)):
                text=os.fsdecode(path)
                if 'candidate_renderings' in text and text.endswith('.jsonl'):
                    candidate_reads.append(text);raise RuntimeError('Reproduction cannot read stored candidates')
            if (isinstance(mode,str) and any(x in mode for x in ('w','a','x','+'))) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                blocked_mutations.append({'event':'open_write','path':str(path)});raise RuntimeError('Generation write forbidden')
    sys.addaudithook(audit)
    common=importlib.import_module('factory.common')
    def capture_jsonl(path,records):write_requests.append({'kind':'jsonl','path':str(path),'records':copy.deepcopy(records)})
    def capture_json(path,value):write_requests.append({'kind':'json','path':str(path),'value':copy.deepcopy(value)})
    common.write_jsonl=capture_jsonl;common.write_json=capture_json
    generator=importlib.import_module('generators.causal.generate')
    generator.ROOT=BASE/'reviews/expert/__never_created_rcm_reproduction_destination__'
    assert not generator.ROOT.exists()
    generated=generator.generate(count=150,batch='expansion-v1',offset=30)
    active['generation']=False
    assert len(generated)==1200
    generated_by_id={r['sample_id']:r for r in generated}
    mismatches=[sid for sid,c in lookup.items() if sid not in generated_by_id or not same(c,generated_by_id[sid])]
    assert not mismatches,mismatches[:10]
    assert all(file_sha(BASE/p)==v for p,v in before_sources.items())
    assert file_sha(candidate_path)==before_candidate_sha and file_sha(canonical_path)==before_release_sha
    assert not generator.ROOT.exists()
    report={
      'review_type':'CURRENT_PROVENANCE_VERIFICATION','reviewer_execution_id':'/root/expert_reviewer','actual_execution':'/opt/anaconda3/bin/python3 -B reviews/expert/verify_rcm_source_recovery.py',
      'current_reproduction_result':'PASS','original_expert_candidate_decisions':'1200 ACCEPT, unchanged','counts':dict(counts),
      'source_candidate_file':str(candidate_path.relative_to(BASE)),'source_candidate_file_sha256':before_candidate_sha,
      'original_expert_receipts_file':str(receipt_path.relative_to(BASE)),'original_expert_receipts_file_sha256':file_sha(receipt_path),
      'v0002_canonical_sha256':before_release_sha,'v0002_manifest_sha256':file_sha(manifest_path),'v0002_manifest_canonical_hash_match':True,
      'original_candidate_to_v0002_lineage_rows':1200,'current_generator_reproduced_rows':len(generated),'reproduction_parameters':{'count':150,'batch':'expansion-v1','offset':30},
      'full_record_reproduction_mismatches':mismatches,'captured_writer_calls':[{'kind':x['kind'],'path':x['path'],'record_count':len(x['records']) if 'records' in x else None} for x in write_requests],
      'real_generation_mutations':blocked_mutations,'generation_reads_of_existing_candidate_jsonl':candidate_reads,
      'filesystem_write_policy':'Python bytecode disabled; write_json/write_jsonl replaced by memory captures; real write opens and mutation audit events rejected. Output ROOT points to a never-created path to avoid reading stored candidate bytes.',
      'current_source_sha256':current_hashes,'historical_manifest_current_byte_mismatches':differences,
      'causal_generator_matches_historical_manifest':current_hashes.get('generators/causal/generate.py')==manifest['reproducibility_code_sha256']['generators/causal/generate.py'],
      'source_and_original_candidate_and_release_bytes_unchanged_after_execution':True,
      'semantic_independence':'All original1200 visible tasks/targets rechecked with expert-owned interpreter. Current generator ran its normal unmodified factory.reference.solve dependency only for reproduction; reference source/expectations were not inspected or used as independent semantic evidence.',
      'other_reviewer_scope':'No other reviewer verdict files read. Canonical release contains3 receipt IDs per row and the matching original expert ID; the other2 receipt contents/decisions were not independently inspected.',
      'historical_limitation':'The old broad factory MANIFEST names exact historical source hashes that differ from current bytes. Reproducing the same causal candidates with current source does not recover those historical bytes or establish a byte-identical historical release build. This execution does not repeat the backup auditor dedicated Git/history recovery search and cannot prove global unavailability.',
      'advice':'A new immutable full RCM supersession can accurately bind the active1200 reviewed candidates to currently available stable source bytes and a freshly verified reproduction. It must identify itself as a current-source supersession of the same candidates, retain v0002 untouched, carry unchanged exact source-candidate/receipt lineage, and explicitly preserve the historical exact-source limitation. Do not claim restoration of the missing historical source bytes or fresh independent examples/reviews.',
      'publication_scope':'Verification/advice only; no v0003 release, pipeline or status files edited and no release approval issued.',
      'own_script_sha256':file_sha(Path(__file__))
    }
    output=OUT/'rcm-source-recovery-verification.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('counts','current_reproduction_result','current_generator_reproduced_rows','full_record_reproduction_mismatches','causal_generator_matches_historical_manifest','historical_manifest_current_byte_mismatches','captured_writer_calls','real_generation_mutations','generation_reads_of_existing_candidate_jsonl')},indent=2))

if __name__=='__main__':main()
