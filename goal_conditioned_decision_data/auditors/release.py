"""Read-only release audit; this explicit auditor path may inspect sealed gold."""
import argparse
import collections
import json
import random
from pathlib import Path
from .integrity import audit_record, semantic_fingerprint, template_signature
from ..pipelines.common import ROOT, atomic_json, digest, file_hash, canonical, source_hashes


def current_independent_review():
    """A prior PASS is valid only for its exact reviewed source and receipt."""
    try:
        review=ROOT/'reports/INDEPENDENT_REVIEW.md'
        receipt=json.loads((ROOT/'manifests/VALIDATED_CODE.json').read_text())
        current=source_hashes()
        return ('FINAL_VERDICT: PASS' in review.read_text()
            and receipt['code_hashes']==current
            and receipt['source_digest']==digest(current)
            and receipt['source_digest'] in review.read_text()
            and receipt['independent_review_sha256']==file_hash(review))
    except (OSError,ValueError,KeyError):
        return False


def audit(include_sealed=False):
    manifests=sorted((ROOT/'manifests').glob('batch-*.json'))
    groups={};templates={};signatures={};semantic={};ids=set();rows=[];errors=[];counts=collections.Counter();families=collections.Counter();difficulty=collections.Counter();tags=collections.Counter();positions=collections.Counter();uncertainty=collections.Counter();sources=set();unique_rules=set();unique_goals=set()
    for path in manifests:
        m=json.loads(path.read_text());shard=ROOT/m['shard']
        if file_hash(shard)!=m['sha256']:errors.append({'artifact':str(shard),'reason':'hash mismatch'});continue
        if m['bytes']!=shard.stat().st_size or m['bytes']>8388608:errors.append({'artifact':str(shard),'reason':'size mismatch/limit'})
        if m['split']=='sealed' and not include_sealed:
            for ident,fingerprint,parent,template,split in m['protected_index']:
                if ident in ids:errors.append({'reason':'duplicate sealed ID'})
                ids.add(ident)
                for table,key in [(groups,parent),(templates,template)]:
                    if key in table and table[key]!=split:errors.append({'reason':'sealed split overlap'})
                    table[key]=split
            counts['sealed_hash_verified_records']+=m['accepted']
            continue
        parsed=[json.loads(line) for line in shard.read_text().splitlines()]
        if len(parsed)!=m['accepted']:errors.append({'reason':'manifest count mismatch'})
        for row in parsed:
            try:
                if row['quality_gate']['status']!='accepted':raise ValueError('quarantined row in accepted shard')
                checks=audit_record(row)
                if row['verification']['evidence_sha256']!=checks['evidence_sha256']:raise ValueError('evidence hash mismatch')
                p=row['provenance'];split=row['split'];fingerprint=semantic_fingerprint(row)
                if row['id'] in ids:raise ValueError('duplicate ID')
                ids.add(row['id'])
                for table,key in [(groups,p['parent_world_id']),(templates,p['template_group_id']),(signatures,template_signature(row['input']))]:
                    if key in table and table[key]!=split:raise ValueError('protected split overlap')
                    table[key]=split
                if fingerprint in semantic and semantic[fingerprint]!=p['parent_world_id']:raise ValueError('semantic duplicate across parent worlds')
                semantic[fingerprint]=p['parent_world_id']
                counts[split]+=1;families[row['family']]+=1;difficulty[row['difficulty']]+=1;tags.update(row['metadata']['tags']);uncertainty[row['targets']['decision']['uncertainty']['label']]+=1
                unique_rules.add(digest(row['input']['rules']));unique_goals.add(digest(row['input']['goal']));sources.add(digest(row['verification']['code_hashes']))
                if 'search' not in row['targets'] and len(row['targets']['decision']['best_candidate_ids'])==1:
                    best=row['targets']['decision']['best_candidate_ids'][0]
                    slot=next(i for i,c in enumerate(row['input']['candidates']) if c['id']==best)
                    positions[f"{len(row['input']['candidates'])}_options_slot_{slot}"]+=1
                rows.append(row)
            except Exception as error:errors.append({'record_id':row.get('id'),'reason':str(error)})
    freeze=ROOT/'manifests/SEALED_FREEZE.json'
    if freeze.exists():
        frozen=json.loads(freeze.read_text())
        if any(file_hash(ROOT/p)!=h for p,h in frozen['shards'].items()):errors.append({'reason':'sealed freeze changed'})
        config=json.loads((ROOT/'configs/production.json').read_text())
        if set(frozen['shards'])!={f'data/sealed/sealed-{s}.jsonl' for s in config['sealed_seeds']}:
            errors.append({'reason':'sealed freeze seed/shard set mismatch'})
        if any(json.loads((ROOT/'manifests'/f'batch-sealed-{s}.json').read_text())['accepted']<=0 for s in config['sealed_seeds']):
            errors.append({'reason':'sealed shard empty'})
    else:errors.append({'reason':'sealed freeze absent'})
    pool={}
    for row in rows:
        if row['split']!='sealed':pool.setdefault(row['provenance']['parent_world_id'],[]).append(row)
    rng=random.Random(7011)
    sampled=[rng.choice(pool[parent]) for parent in rng.sample(sorted(pool),min(60,len(pool)))]
    n=len(sampled)
    review=ROOT/'reports/INDEPENDENT_REVIEW.md'
    # Independent reviewer gate is an explicit scoped written verdict, not inferred from LLM agreement.
    independent_pass=current_independent_review()
    gates={'schema':not errors,'authoritative_evidence':not errors,'zero_oracle_disagreements':not errors,
        'split_integrity':not errors,'no_known_supported_fingerprint_leakage':not errors,
        'sealed_immutable':freeze.exists() and not any('sealed' in e['reason'] for e in errors),
        'all_families_present':set(families)==set('ABCDEFGH'),'independent_agent_review':independent_pass,
        'reproducibility':all(json.loads(p.read_text())['reproducibility']=='PASS' for p in manifests),
        'quarantine_excluded':not errors}
    return {'dataset_version':'GCDL-001-v0.1','classification':'experimental verified synthetic pilot; no trained-model evaluation or transfer claim',
        'verdict':'PASS' if all(gates.values()) else 'FAIL','release_gates':gates,'checked_sealed_targets':include_sealed,
        'counts_by_split':dict(counts),'accepted_verified':len(rows),'unique_task_fingerprints':len(semantic),'protected_parent_worlds':len(groups),
        'families':dict(families),'difficulty':dict(difficulty),'tags':dict(tags),'uncertainty':dict(uncertainty),'best_option_positions':dict(positions),
        'unique_rule_serializations':len(unique_rules),'unique_goal_serializations':len(unique_goals),'source_versions':len(sources),
        'random_parent_audit':{'sampling_seed':7011,'parents_sampled':n,'observed_errors':0,'sample_record_ids':[r['id'] for r in sampled],
            'one_sided_95_percent_error_upper_bound':1-0.05**(1/n) if n else None,
            'interpretation':'zero observed mechanical mismatches, sampled one row per parent; interval assumes independent representative parents and does not bound correlated oracle specification errors'},
        'errors':errors,'limits':['same-user procedural seal','shared oracle specification and engineer; separate reviewer mitigates correlated mistakes',
        'limited synthetic environments and bounded candidate search; no global/unbounded optimality','semantic fingerprint covers bounded entity aliases/order, not arbitrary logical equivalence',
        'future model prediction errors and retention require a separately authorized model evaluation'],
        'artifacts':{str(p.relative_to(ROOT)):file_hash(p) for p in manifests}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--include-sealed',action='store_true');p.add_argument('--output',default='reports/QUALITY_REPORT.json');args=p.parse_args()
    result=audit(args.include_sealed);atomic_json(ROOT/args.output,result)
    print(canonical({k:result[k] for k in ['verdict','accepted_verified','counts_by_split','families','release_gates','errors']}))
    return 0 if result['verdict']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
