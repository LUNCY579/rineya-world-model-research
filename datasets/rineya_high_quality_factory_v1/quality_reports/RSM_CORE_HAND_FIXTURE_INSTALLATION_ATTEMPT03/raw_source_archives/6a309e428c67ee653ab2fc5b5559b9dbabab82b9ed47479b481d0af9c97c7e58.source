"""Authoritative active release selection, without rewriting historical bytes."""
import hashlib
import json
from factory.common import ROOT,load_jsonl,write_json
from factory.lineage import source_key


def catalog():
    releases=[];inactive=set()
    for path in sorted(ROOT.glob('data/*/releases/*/MANIFEST.json')):
        m=json.loads(path.read_text());m['path']=str(path.parent.relative_to(ROOT));releases.append(m)
        inactive.update((m['track'],old) for old in m.get('supersedes',[]))
    known={(m['track'],m['release']) for m in releases}
    assert inactive<=known,'Superseded release missing'
    exclusions={}
    for path in ROOT.glob('quality_reports/errata/*.json'):
        e=json.loads(path.read_text())
        for row in e.get('excluded_records',[]):exclusions[row['record_hash']]=str(path.relative_to(ROOT))
    selected=[];entries=[];seen_families={}
    for m in releases:
        historical=(m['track'],m['release']) in inactive
        path=ROOT/m['path'];rows=load_jsonl(path/'canonical.jsonl')
        valid=[r for r in rows if r['record_hash'] not in exclusions] if not historical else []
        for r in valid:
            key=source_key(r)
            assert key not in seen_families or seen_families[key]==m['path'],'Duplicate active source family across releases'
            seen_families[key]=m['path']
        selected.extend(valid)
        entries.append({'track':m['track'],'release':m['release'],'path':m['path'],'status':'SUPERSEDED' if historical else 'ACTIVE_WITH_EXCLUSIONS' if len(valid)<len(rows) else 'ACTIVE',
                        'canonical_sha256':hashlib.sha256((path/'canonical.jsonl').read_bytes()).hexdigest(),'historical_records':len(rows),'active_records':len(valid),
                        'excluded_record_hashes':[r['record_hash'] for r in rows if r['record_hash'] in exclusions]})
    report={'scope':'Use ACTIVE catalog releases only. Historical/superseded releases and exact-hash errata exclusions are retained for audit. Source families across revisions are counted once.',
            'active_records':len(selected),'active_source_families':len(seen_families),'releases':entries,'errata':sorted(set(exclusions.values()))}
    return selected,report


def write_catalog():
    rows,report=catalog();write_json(ROOT/'manifests/ACTIVE_RELEASE_CATALOG.json',report);return rows,report


if __name__=='__main__':
    _,r=write_catalog();print({'active_records':r['active_records'],'active_source_families':r['active_source_families']})
