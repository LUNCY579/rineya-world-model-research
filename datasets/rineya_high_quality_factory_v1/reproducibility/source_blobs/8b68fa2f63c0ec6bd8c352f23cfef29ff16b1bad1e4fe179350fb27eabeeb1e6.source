"""New frozen toy byte-custody supplement; no production/review data reads."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def need(value,message):
    if not value:raise ValueError(message)

def run():
    own=Path(__file__);raw=own.read_bytes();h=sha(raw)
    need((BASE/'reproducibility/source_blobs'/(h+'.source')).read_bytes()==raw,'new custody source must be frozen before execution')
    hand_path=HERE/'hand_probe.py'
    need(sha(hand_path.read_bytes())=='a2a5ebb08478d0275ee6e14f6df2f099a0770f72f7e63e6b99770d34736ff577','frozen hand source mismatch')
    spec=importlib.util.spec_from_file_location('own_frozen_hand',hand_path);hand=importlib.util.module_from_spec(spec);spec.loader.exec_module(hand)
    rows,entries,_=hand.fixture()
    manifest_path=HERE/'hand_manifest.json';candidate_path=HERE/'hand_rows.jsonl'
    relative=str(manifest_path.relative_to(BASE))
    for sid,row in rows.items():
        row['family_control_claims']=[dict(hand.make_claim(rows,sid,name),evidence_manifest=relative) for name in hand.NAMES]
    candidate_raw=b''.join((enc(row)+'\n').encode() for _,row in sorted(rows.items()))
    candidate_hash=sha(candidate_raw)
    for entry in entries:entry['exact_candidate_sha256']=candidate_hash
    manifest={'supplied_same_table_v2_pairs':entries,'scope':'Illustrative existing public collection, not a new mandatory closed envelope.'}
    manifest_raw=(enc(manifest)+'\n').encode();manifest_hash=sha(manifest_raw)
    need(not candidate_path.exists() and not manifest_path.exists(),'all toy outputs must be new and immutable')
    candidate_path.write_bytes(candidate_raw);manifest_path.write_bytes(manifest_raw)
    def verify(candidate,manifest_bytes,expected_manifest_hash):
        need(sha(manifest_bytes)==expected_manifest_hash,'actual manifest byte hash mismatch')
        parsed=json.loads(manifest_bytes);parsed_rows={}
        line_hashes={}
        for line in candidate.splitlines():
            row=json.loads(line);sid=row['sample_id'];need(sid not in parsed_rows,'duplicate actual row ID')
            parsed_rows[sid]=row;line_hashes[sid]=sha(line)
        candidate_sha=sha(candidate);proofs=[]
        for sid,row in parsed_rows.items():
            for claim in row['family_control_claims']:
                need(claim['evidence_manifest']==relative,'toy path does not resolve to bound actual manifest')
                proof=hand.join(parsed_rows,sid,claim,parsed['supplied_same_table_v2_pairs'],candidate_sha,True)
                proof['resolved_actual_record_line_sha256']=[line_hashes[x] for x in proof['actual_pair']]
                proofs.append(proof)
        return proofs,line_hashes
    proofs,line_hashes=verify(candidate_raw,manifest_raw,manifest_hash)
    need(len(proofs)==48,'toy claims unexpectedly missing')
    negatives=[]
    for label,candidate,metadata,expected in (
        ('candidate byte corruption',candidate_raw.replace(b'marker',b'MARKER',1),manifest_raw,manifest_hash),
        ('manifest byte corruption',candidate_raw,manifest_raw+b' ',manifest_hash),
        ('wrong manifest custody binding',candidate_raw,manifest_raw,'0'*64)):
        try:verify(candidate,metadata,expected)
        except ValueError as exc:negatives.append({'label':label,'error':str(exc)})
        else:raise AssertionError('custody corruption silently accepted '+label)
    bindings={str(own.relative_to(BASE)):h,str(hand_path.relative_to(BASE)):sha(hand_path.read_bytes()),
        str(candidate_path.relative_to(BASE)):candidate_hash,str(manifest_path.relative_to(BASE)):manifest_hash}
    receipt={'actual_execution_id':'/root/formal_oracle','role':'Nonbinding public-only toy byte-custody demonstration; no data approval',
        'at_utc':datetime.now(timezone.utc).isoformat(),'source_and_output_sha256':bindings,
        'positive_actual_byte_bound_claim_joins':len(proofs),'resolved_actual_row_line_hashes':line_hashes,
        'pair_proofs':proofs,'preserved_expected_custody_failures':negatives,
        'no_mandatory_producer_pair_record_hashes':all('pair_record_hashes' not in x for x in entries),
        'scope':'Actual hashes describe only newly executed24-row/3-world handmade artifacts. Raw line hashes are measured custody evidence, not asserted canonical-record hashes or new required manifest fields. No real candidate/gold/checker/factory/reviewer data read. Full3456-core/canonical/profile/schedule/style gate untested. This supplements the explicit placeholder fixture-hash initial mechanism tests without editing their frozen source/history.'}
    out=HERE/'custody_execution.json';need(not out.exists(),'receipt immutable');out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(enc({'receipt_path':str(out.relative_to(BASE)),'receipt_sha256':sha(out.read_bytes()),
        'actual_toy_candidate_sha256':candidate_hash,'actual_toy_manifest_sha256':manifest_hash,
        'positive_joins':len(proofs),'expected_rejected_corruptions':len(negatives)}))

if __name__=='__main__':run()
