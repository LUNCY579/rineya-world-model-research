#!/usr/bin/env python3
"""Read-only audit of catalog policy with independent selection and shadow fixtures.

No release writes, producer/reference imports, or other reviewer verdict reads.
The tested catalog functions use our JSON/source-key helpers and disabled writer.
"""
import ast
import hashlib
import json
import pathlib
import tempfile
from collections import Counter,defaultdict
from adversarial_audit import ROOT,EXECUTION,canon,digest,require
from correction_lineage_adversarial_audit import source_key


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def catalog_function(root):
    path=ROOT/'factory/catalog.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module in ('factory.common','factory.lineage'))]
    ns={'ROOT':root,'load_jsonl':load,'write_json':lambda *args:None,'source_key':source_key}
    exec(compile(tree,str(path),'exec'),ns)
    return ns['catalog']


def independent_selection(root):
    files=sorted(root.glob('data/*/releases/*/MANIFEST.json'));manifests=[(p,json.loads(p.read_text())) for p in files]
    nodes={(m['track'],m['release']) for p,m in manifests};inactive=set();exclusions=set();hashes={}
    for p,m in manifests:
        for old in m.get('supersedes',[]):
            require((m['track'],old) in nodes,'missing supersession predecessor')
            inactive.add((m['track'],old))
        hashes[str(p.relative_to(root))]=sha(p)
    for p in root.glob('quality_reports/errata/*.json'):
        exclusions.update(r['record_hash'] for r in json.loads(p.read_text()).get('excluded_records',[]));hashes[str(p.relative_to(root))]=sha(p)
    selected=[];expected=[];families=defaultdict(set)
    for p,m in manifests:
        cp=p.parent/'canonical.jsonl';rows=load(cp);hashes[str(cp.relative_to(root))]=sha(cp)
        historical=(m['track'],m['release']) in inactive
        valid=[] if historical else [r for r in rows if r['record_hash'] not in exclusions]
        selected.extend(valid)
        for r in valid:families[source_key(r)].add(str(p.parent.relative_to(root)))
        expected.append({'track':m['track'],'release':m['release'],'path':str(p.parent.relative_to(root)),
                         'status':'SUPERSEDED' if historical else 'ACTIVE_WITH_EXCLUSIONS' if len(valid)!=len(rows) else 'ACTIVE',
                         'historical_records':len(rows),'active_records':len(valid),'canonical_sha256':sha(cp),
                         'excluded_record_hashes':[r['record_hash'] for r in rows if r['record_hash'] in exclusions]})
    require(all(len(v)==1 for v in families.values()),'duplicate family in active releases')
    return selected,expected,families,hashes


def fixture_tests():
    with tempfile.TemporaryDirectory(prefix='catalog-red-shadow-') as d:
        root=pathlib.Path(d);call=catalog_function(root)
        def release(track,version,rows,supersedes=()):
            p=root/'data'/track/'releases'/version;p.mkdir(parents=True)
            (p/'MANIFEST.json').write_text(json.dumps({'track':track,'release':version,'supersedes':list(supersedes)}))
            (p/'canonical.jsonl').write_text(''.join(canon(r)+'\n' for r in rows))
            return p
        def row(name,family):return {'sample_id':name,'record_hash':digest([name,family]),'source_family_key':family}
        a=row('a','T-f1');b=row('b','T-f1');c=row('c','T-f2')
        p=release('T','v1',[a,b,c]);before=sha(p/'canonical.jsonl')
        err=root/'quality_reports/errata';err.mkdir(parents=True);(err/'x.json').write_text(json.dumps({'excluded_records':[{'record_hash':a['record_hash']}]}))
        rows,report=call();require([r['sample_id'] for r in rows]==['b','c'],'exact hash exclusion selector');require(report['active_source_families']==2,'count families not renderings')
        checks=[{'case':'exact_hash_exclusion','pass':True,'active_samples':['b','c'],'families':2}]
        replacement=[row('a-new','T-f1'),row('b-new','T-f1'),row('c-new','T-f2')]
        release('T','v2',replacement,('v1',));rows,report=call()
        require(rows==replacement and report['active_source_families']==2,'supersession must replace and not duplicate families')
        require(report['releases'][0]['status']=='SUPERSEDED','old release historical')
        require(sha(p/'canonical.jsonl')==before,'historical bytes changed')
        checks.append({'case':'supersession_overrides_old_active_exclusions','pass':True,'active_samples':[r['sample_id'] for r in replacement],'families':2,'old_bytes_unchanged':True})
        newest=[row('d','T-f1'),row('e','T-f2')];release('T','v3',newest,('v2',));rows,report=call()
        require(rows==newest,'transitive supersession')
        checks.append({'case':'transitive_supersession','pass':True})
        other=[row('z','OTHER-f1')];release('OTHER','v1',other);rows,report=call()
        require({r['sample_id'] for r in rows}=={'d','e','z'},'track-local version supersession')
        checks.append({'case':'track_local_supersession','pass':True})
        duplicate=release('T','duplicate',[row('duplicate','T-f1')])
        try:call();raise RuntimeError('duplicate active family admitted')
        except AssertionError:checks.append({'case':'duplicate_active_source_family_rejected','pass':True})
        for f in duplicate.iterdir():f.unlink()
        duplicate.rmdir()
        missing=release('T','missing',[row('missing','T-f3')],('unknown',))
        try:call();raise RuntimeError('missing predecessor admitted')
        except AssertionError:checks.append({'case':'missing_supersession_predecessor_rejected','pass':True})
        return checks


def main():
    rows,expected,families,hashes=independent_selection(ROOT);observed,code_report=catalog_function(ROOT)()
    require(canon(rows)==canon(observed),'catalog callable selection differs from independent policy')
    require(canon(expected)==canon(code_report['releases']),'catalog callable release statuses differ')
    active_path=ROOT/'manifests/ACTIVE_RELEASE_CATALOG.json';active=json.loads(active_path.read_text())
    require(canon(expected)==canon(active['releases']),'saved active catalog stale or incorrect')
    require(active['active_records']==len(rows) and active['active_source_families']==len(families),'saved count mismatch')
    own_erratum=json.loads((ROOT/'audits/shortcut_detection/RVM-001-v0001-mechanism-tags.json').read_text())
    bad={r['record_hash'] for r in own_erratum['findings']};selected_hashes={r['record_hash'] for r in rows}
    require(not(bad&selected_hashes),'known false legacy tag record still active')
    rvm=[r for r in expected if r['track']=='RVM-001'];historical=next(r for r in rvm if r['release']=='v0001')
    corrected=[r for r in rvm if r['release']!='v0001' and any(q['sample_id'].startswith('RVM-001-pilot-v4-') for q in load(ROOT/r['path']/'canonical.jsonl'))]
    if corrected:
        require(historical['status']=='SUPERSEDED' and historical['active_records']==0,'corrected pilot does not supersede legacy release')
        correction=corrected[0];new_rows=load(ROOT/correction['path']/'canonical.jsonl')
        require(len(new_rows)==60 and correction['active_records']==60,'correction pilot active coverage')
        reviewed={r['sample_id']:r['record_hash'] for r in load(ROOT/'reviews/adversarial/pilot-v4.jsonl')}
        candidates={r['sample_id']:r for r in load(ROOT/'data/RVM-001/candidate_renderings/pilot-v4.jsonl')}
        for r in new_rows:
            require(reviewed.get(r['sample_id'])==r.get('reviewed_candidate_hash'),'active correction missing own exact reviewed candidate hash')
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'released complete-record hash mismatch')
            candidate=candidates[r['sample_id']]
            projection=dict(r);projection.pop('reviewed_candidate_hash')
            for field in ('review_status','review_receipt_ids','record_hash'):projection[field]=candidate[field]
            require(canon(projection)==canon(candidate),'release changed content beyond approval metadata and its hash')
    else:
        require(historical['status']=='ACTIVE_WITH_EXCLUSIONS' and historical['active_records']==54,'pending correction should exclude precisely six old records')
    report={'reviewer_execution_id':EXECUTION,'decision':'PASS' if corrected else 'PASS_CURRENT_POLICY_CORRECTED_PILOT_NOT_YET_PUBLISHED',
            'completed_current_snapshot':True,'corrected_pilot_published':bool(corrected),'active_records':len(rows),'active_source_families':len(families),
            'catalog_sha256':sha(active_path),'catalog_code_sha256':sha(ROOT/'factory/catalog.py'),'audit_script_sha256':sha(pathlib.Path(__file__)),
            'release_and_erratum_file_sha256':hashes,'observed_RVM_release_selection':rvm,'known_false_legacy_hashes_excluded':sorted(bad),
            'shadow_policy_checks':fixture_tests(),'source':'Own selection from public release manifests/canonical records/exact-hash erratum fields; catalog code executed with own helpers and disabled writer.',
            'scope':['Catalog selection is checked; approval verdicts are not read or synthesized.','The selector returns filtered rows. Consumers must use these rows or the listed exact-hash exclusions, rather than concatenate all historical canonical paths.','Historical release bytes and all current exact-hash review receipts are unchanged.','No release, catalog, training export, approval or publication is written.'],
            'private_reference_producer_or_other_reviewer_files_read':False}
    (ROOT/'audits/execution/catalog-policy-adversarial.json').write_text(json.dumps(report,indent=2)+'\n')
    print(canon({k:report[k] for k in ('decision','corrected_pilot_published','active_records','active_source_families','catalog_sha256','shadow_policy_checks')}))


if __name__=='__main__':main()
