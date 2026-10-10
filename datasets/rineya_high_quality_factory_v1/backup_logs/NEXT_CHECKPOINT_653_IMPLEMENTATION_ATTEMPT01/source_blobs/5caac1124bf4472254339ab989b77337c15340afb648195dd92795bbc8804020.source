"""Complete source/representation inventory engine. No copy or Git mutation."""
from __future__ import annotations
import json, os
from pathlib import Path
import next653_common as c

FIELDS={'pending_observed','required_closure','byte_evolution','raw_relocations','stable_root_hold','producer_generation_status','producer_writes_stopped','integration_predicates'}

def integration_gate(receipt,predicates):
    if type(predicates) is not list or len(predicates)<2:raise ValueError('Actual successful integration result and stable-source predicates required')
    successful_exit=False;zero_failures=False;unchanged=False;seen=set()
    for row in predicates:
        if type(row) is not dict or set(row)!={'field','equals'} or type(row['field']) is not list or not row['field'] or any(type(k) is not str or not k for k in row['field']):raise ValueError('Closed literal integration field predicate required')
        key=tuple(row['field'])
        if key in seen:raise ValueError('Duplicate integration predicate')
        seen.add(key);value=receipt
        for part in key:
            if type(value) is not dict or part not in value:raise ValueError('Actual integration result field absent')
            value=value[part]
        if type(value) is not type(row['equals']) or value!=row['equals']:raise ValueError('Actual integration result/stable source condition fails')
        successful_exit |= type(value) is int and value==0 and key[-1] in ('exit_code','actual_exit_code','returncode')
        zero_failures |= type(value) is int and value==0 and key[-1] in ('failures','failed','tests_failed','failure_count')
        unchanged |= value is True and key[-1] in ('bindings_unchanged','sources_unchanged','all_bindings_unchanged')
    if not successful_exit or not zero_failures or not unchanged:raise ValueError('Actual successful exit, zero failures and unchanged-bindings evidence required')
    return True

def normalize_record(record):
    if type(record) is not dict:raise ValueError('Lossless record object required')
    r=dict(record)
    for dest,alias in [('raw_bytes','raw_length'),('gzip_path','compressed_path'),('gzip_bytes','compressed_length'),('gzip_sha256','compressed_sha256')]:
        if dest in r and alias in r and (type(r[dest]) is not type(r[alias]) or r[dest]!=r[alias]):raise ValueError('Lossless aliases conflict')
        if dest not in r and alias in r:r[dest]=r[alias]
    for role in ('raw','gzip'):
        value=r.get(role+'_path')
        if type(value) is not str:raise ValueError('Lossless declared path type differs')
        if Path(value).is_absolute():
            p=Path(value)
            if str(p)!=value or not p.is_relative_to(c.FACTORY):raise ValueError('Absolute lossless path outside factory/alias')
            value=p.relative_to(c.FACTORY).as_posix()
        r[role+'_path']=c.name(value);c.number(r.get(role+'_bytes'));c.sha(r.get(role+'_sha256'))
    return r

def validate_evolution(prior,entries,allowed):
    oldfiles,oldraw=c.all_entries(prior);old=oldfiles
    if type(allowed) is not list:raise ValueError('Exact evolution list required')
    changes={p:(c.projection(e),c.projection(entries[p])) for p,e in old.items() if p in entries and c.projection(e)!=c.projection(entries[p])}
    approved={}
    for row in allowed:
        if type(row) is not dict or set(row)!={'prior','current','preserved_parent'} or row['preserved_parent']!=c.PARENT:raise ValueError('Closed parent-bound byte evolution required')
        a,b=row['prior'],row['current'];c.entry(a);c.entry(b)
        if a['path']!=b['path'] or a['path'] not in oldfiles or a['path'] in approved or '/releases/' in '/'+a['path']:raise ValueError('Immutable/raw/duplicate/path-switched evolution')
        if a['path'] in c.SecretSession().expected:raise ValueError('Four exact metadata bytes cannot evolve')
        approved[a['path']]=(c.projection(a),c.projection(b))
    if changes!=approved:raise ValueError('Every current byte evolution must equal the fresh root-bound configuration')
    return [{'path':p,'historical':a,'current':b,'historical_bytes_preserved_by_parent':c.PARENT} for p,(a,b) in sorted(changes.items())]

def pending_cut(paths,observed,previous):
    if type(observed) is not list:raise ValueError('Observed pending rows required')
    deferred={}
    for row in observed:
        if type(row) is not dict or set(row)!={'path','size','sha256','actor_execution','status_at_cut'} or row['status_at_cut']!='RUNNING' or type(row['actor_execution']) is not str or not row['actor_execution'].startswith('/root/'):
            raise ValueError('Only actual observed owned RUNNING rows may defer')
        c.entry(row);p=row['path']
        if p in deferred or p in previous or p not in paths:raise ValueError('Prior/duplicate/absent path cannot defer')
        c.equal(row,c.hash_file(c.safe(c.FACTORY,p)));deferred[p]=row
    return deferred

def historical_rule(manifest,record,raw_entry,entries,compare=None,relocations=None):
    compare=c.compare_gzip if compare is None else compare
    relocations={} if relocations is None else relocations
    cfg=c.fixed_json('history');old=cfg['old_raw'];rep=cfg['old_representation'];cur=cfg['current_pack_record']
    def pack_record(rec,role):return {'path':rec[role+'_path'],'size':rec[role]['bytes'],'sha256':rec[role]['sha256']}
    expected_raw=pack_record(cur,'raw')
    chain=relocations.get(old['path'])
    actual_current=c.projection(raw_entry)
    if actual_current!=expected_raw:
        if chain is None or c.projection(chain['prior_raw'])!=expected_raw or c.projection(chain['current_raw'])!=actual_current:raise ValueError('Fixed279/149 history may advance only through the exact separately validated149 relocation')
    if (manifest!=rep['manifest']['path'] or c.projection(entries.get(manifest,{}))!=c.projection(rep['manifest'])
            or record['raw_path']!=old['path']
            or (record['raw_bytes'],record['raw_sha256'])!=(old['size'],old['sha256'])
            or record['gzip_path']!=rep['gzip']['path'] or (record['gzip_bytes'],record['gzip_sha256'])!=(rep['gzip']['size'],rep['gzip']['sha256'])
            or c.projection(entries.get(rep['gzip']['path'],{}))!=c.projection(rep['gzip'])):raise ValueError('Unbound stale manifest rejected')
    for role in ('gzip','manifest'):
        r=pack_record(cur,role)
        if r['path'] not in entries or c.projection(entries[r['path']])!=r:raise ValueError('Exact current149 package changed/missing')
    archives=[]
    for rec in cfg['archive_pack_records']:
        r=pack_record(rec,'raw')
        if r['path'] not in entries or c.projection(entries[r['path']])!=r or (r['size'],r['sha256'])!=(old['size'],old['sha256']):raise ValueError('Historical279 archive changed/missing')
        for role in ('gzip','manifest'):
            p=pack_record(rec,role)
            if p['path'] not in entries or c.projection(entries[p['path']])!=p:raise ValueError('Historical279 package changed/missing')
        compare(c.safe(c.FACTORY,r['path']),c.safe(c.FACTORY,rep['gzip']['path']));archives.append(r)
    return {'old_manifest':c.projection(rep['manifest']),'old_mutable_raw':old,'bound_parent_mutable149':expected_raw,
            'current_mutable_raw':actual_current,'exact_later149_relocation':chain,
            'archived_old_raws_actually_byte_compared':archives,'historical_mapping_is_not_current_raw_mapping':True}

def validate_relocations(prior,entries,rows):
    _,oldraw=c.all_entries(prior);mapping={}
    if type(rows) is not list:raise ValueError('Exact raw relocation list required')
    for r in rows:
        if type(r) is not dict or set(r)!={'prior_raw','current_raw','archived_old_raws'}:raise ValueError('Closed raw relocation rule required')
        a,b=r['prior_raw'],r['current_raw'];c.entry(a);c.entry(b);p=a['path']
        if p in mapping or p not in oldraw or c.projection(oldraw[p])!=c.projection(a) or b['path']!=p or c.projection(entries.get(p,{}))!=c.projection(b):raise ValueError('Relocation old/current binding differs')
        arcs=r['archived_old_raws']
        if type(arcs) is not list or not arcs or len({x.get('path') for x in arcs})!=len(arcs):raise ValueError('Exact archived old raw bindings required')
        for arc in arcs:
            c.entry(arc)
            if arc['path']==p or arc['size']!=a['size'] or arc['sha256']!=a['sha256'] or c.projection(entries.get(arc['path'],{}))!=c.projection(arc):raise ValueError('Archived old raw changed/aliased/missing')
            for rep in oldraw[p]['lossless_representations']:
                c.compare_gzip(c.safe(c.FACTORY,arc['path']),c.safe(c.FACTORY,rep['gzip']['path']))
        mapping[p]=r
    changed={p for p,row in oldraw.items() if p not in entries or c.projection(entries[p])!=c.projection(row)}
    if changed!=set(mapping):raise ValueError('Only exact declared raw relocations may evolve')
    return mapping

def stale_relocation(manifest,record,raw_entry,entries,prior,mapping):
    p=record['raw_path'];r=mapping.get(p)
    if r is None:raise ValueError('Unbound stale mapping')
    _,oldraw=c.all_entries(prior)
    matching=[x for x in oldraw[p]['lossless_representations'] if x['manifest']['path']==manifest and x['gzip']['path']==record['gzip_path']]
    if len(matching)!=1:raise ValueError('Stale manifest must be an exact inherited representation')
    rep=matching[0];old=r['prior_raw']
    if ((record['raw_bytes'],record['raw_sha256'])!=(old['size'],old['sha256']) or c.projection(entries[manifest])!=c.projection(rep['manifest'])
            or (record['gzip_bytes'],record['gzip_sha256'])!=(rep['gzip']['size'],rep['gzip']['sha256']) or c.projection(entries[record['gzip_path']])!=c.projection(rep['gzip'])
            or c.projection(raw_entry)!=c.projection(r['current_raw'])):raise ValueError('Relocation stale mapping is not exact prior bytes')
    return {'manifest':manifest,'bound_relocation':r,'historical_representation_retained':True}

def assemble_representations(entries,prior,relocations):
    raw={p:r for p,r in entries.items() if r['size']>c.CAP};representations={};history=[]
    _,previous_raw=c.all_entries(prior)
    inherited_manifests={rep['manifest']['path'] for r in previous_raw.values() for rep in r['lossless_representations']}
    for p,row in sorted(entries.items()):
        if not (p.endswith('compression_manifest.json') or p.endswith('.lossless.json') or p in inherited_manifests):continue
        if row['size']>c.CAP:raise ValueError('Lossless manifest must be capped tracked bytes')
        value=json.loads(c.safe(c.FACTORY,p).read_bytes())
        if type(value) is not dict:raise ValueError('Lossless manifest object required')
        records=value.get('artifacts',value.get('items',[value]))
        if type(records) is not list or not records:raise ValueError('Lossless manifest records required')
        for original in records:
            rec=normalize_record(original);rawname=rec['raw_path'];gzname=rec['gzip_path']
            if rawname not in entries:raise ValueError('Lossless raw absent from exact cut')
            current=entries[rawname]
            if (current['size'],current['sha256'])!=(rec['raw_bytes'],rec['raw_sha256']):
                old=c.fixed_json('history')['old_representation']['manifest']['path']
                history.append(historical_rule(p,rec,current,entries,relocations=relocations) if p==old else stale_relocation(p,rec,current,entries,prior,relocations));continue
            if gzname not in entries or (entries[gzname]['size'],entries[gzname]['sha256'])!=(rec['gzip_bytes'],rec['gzip_sha256']) or entries[gzname]['size']>c.CAP:raise ValueError('Exact lossless gzip binding differs')
            if '/releases/' in '/'+p or '/releases/' in '/'+gzname:raise ValueError('Package cannot be inside immutable release')
            if rawname in raw:
                c.compare_gzip(c.safe(c.FACTORY,rawname),c.safe(c.FACTORY,gzname))
                rep={'gzip':c.projection(entries[gzname]),'manifest':c.projection(row),'exact_raw_roundtrip_verified':True,'uncompressed_secret_scan':'PASS'}
                if rep in representations.setdefault(rawname,[]):raise ValueError('Duplicate manifest raw representation')
                representations[rawname].append(rep)
    old_manifest=c.fixed_json('history')['old_representation']['manifest']['path']
    if len([h for h in history if h.get('old_manifest',{}).get('path')==old_manifest])!=1:raise ValueError('Exact historical279/current149 distinction must remain verified')
    result=[]
    for p,row in sorted(raw.items()):
        reps=representations.get(p)
        if not reps:raise ValueError('Unrepresented oversized raw; no silent omission: '+p)
        result.append({**row,'reason':'raw_exceeds_25MiB_preserved_local','lossless_representations':reps})
    return result,history

def execute(cfg):
    if (set(cfg['fields'])!=FIELDS or cfg['fields']['stable_root_hold'] is not True or cfg['fields']['producer_writes_stopped'] is not True
            or cfg['fields']['producer_generation_status'] not in ('NOT_AUTHORIZED','PROVISIONAL_UNAPPROVED','COMPLETED_UNAPPROVED')):
        raise ValueError('Stable integration/producer closure grant and exact fields required')
    integration_gate(c.bound_json(cfg['inputs']['integration_execution']),cfg['fields']['integration_predicates'])
    producer=c.bound_json(cfg['inputs']['producer_freeze'])
    if type(producer) is not dict or not producer:raise ValueError('Actual nonempty producer source/config freeze required')
    prior=c.fixed_json('prior');oldfiles,oldraw=c.all_entries(prior)
    before=c.custody(c.PARENT);session=c.SecretSession()
    # One observed name cut. Names born afterward remain local, not invented deferrals.
    paths={p.relative_to(c.FACTORY).as_posix():p for p in c.FACTORY.rglob('*') if p.is_file()}
    generated={p for p in paths if '__pycache__' in Path(p).parts or Path(p).suffix=='.pyc' or Path(p).name=='.DS_Store'}
    deferred=pending_cut(paths,cfg['fields']['pending_observed'],set(oldfiles)|set(oldraw))
    entries={}
    for p in sorted(set(paths)-generated-set(deferred)):
        c.safe(c.FACTORY,p);entries[p]={'path':p,**session.file(paths[p],p,'source_before')}
    c.preserve_prior(prior,entries,deferred)
    for row in cfg['fields']['required_closure']:
        c.validate_binding(row)
        if row['path'] not in entries or c.projection(entries[row['path']])!=row:raise ValueError('Operative code/config/reference closure absent from exact cut')
    for row in cfg['inputs'].values():
        if row['path'] not in entries or c.projection(entries[row['path']])!=row:raise ValueError('Phase evidence omitted from exact cut')
    evolution=validate_evolution(prior,entries,cfg['fields']['byte_evolution'])
    relocations=validate_relocations(prior,entries,cfg['fields']['raw_relocations'])
    # Every inherited recovery artifact remains immutable even when raw aliases evolve.
    for r in oldraw.values():
        for rep in r['lossless_representations']:
            for role in ('gzip','manifest'):
                e=rep[role]
                if e['path'] not in entries or c.projection(entries[e['path']])!=c.projection(e):raise ValueError('Inherited recovery bytes changed/missing')
    raw,history=assemble_representations(entries,prior,relocations)
    spec={'schema':'next-checkpoint-653-exact-inventory-v1','checkpoint_id':cfg['checkpoint_id'],'expected_parent_sha':c.PARENT,
          'files':[entries[p] for p in sorted(entries) if entries[p]['size']<=c.CAP],'raw_size_exclusions':raw,
          'deferred_observed_pending_paths':list(deferred.values()),'excluded_generated_paths':sorted(generated),
          'legitimate_current_byte_evolution':evolution,'historical_raw_mappings':history,'raw_relocations':list(relocations.values()),
          'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],'grants_data_approval':False,
          'self_inventory_boundary':'SPEC and execution/stage/proofs/configs born after this path cut remain local until a future cut'}
    vector=c.scope(spec);c.capacity(spec,before['disk_free_bytes'])
    session.complete('source_before')
    c.verify_sources(spec,session,'source_after')
    if c.current_oversized_names()!={r['path'] for r in raw}:raise ValueError('Oversized names changed at cut')
    after=c.custody(c.PARENT);c.capacity(spec,after['disk_free_bytes']);c.recheck_inputs(cfg)
    report={'status':'PASS_EXACT_INVENTORY_ONLY','author_execution':'/root/backup_manager','actual_runner_execution':cfg['runner_execution'],
            'at_utc':c.now(),'exact_scope_before_report':vector,'custody_before':before,'custody_after':after,
            'metadata_reports':{'source_before':session.complete('source_before'),'source_after':session.complete('source_after')},
            'historical_raw_mappings':history,'deferred_actual_observed_pending_at_cut':list(deferred.values()),
            'all_raw_actual_local_gzip_comparisons':'PASS','blockers':[],'secret_scan_status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS',
            'allowed_exact_match_detections':4,'unbound_secret_detections':0,'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],
            'legitimate_byte_evolution':evolution,'grants_data_approval':False,'same_model_disclosure':'Independent execution; inherited same model'}
    reportpath=c.safe_output(cfg['outputs']['report']);reportrow=c.write_exclusive(reportpath,report)
    reportproof=session.file(reportpath,reportrow['path'],'source_after')
    spec['files'].append({'path':reportrow['path'],**reportproof});spec['files'].sort(key=lambda r:r['path'])
    spec['inventory_report']=reportrow;spec['exact_scope']=c.scope(spec);c.capacity(spec,c.shutil.disk_usage(c.FACTORY).free)
    # The report is immutable and included, while the SPEC cannot inventory itself.
    specrow=c.write_exclusive(c.safe_output(cfg['outputs']['spec']),spec)
    receipt={'status':'PASS_EXACT_INVENTORY_ONLY','spec':specrow,'report':reportrow,'exact_scope':spec['exact_scope'],
             'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],'author_execution':'/root/backup_manager',
             'actual_runner_execution':cfg['runner_execution'],'copy_stage_commit_push':False,'grants_data_approval':False}
    receiptrow=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    return {**receipt,'receipt':receiptrow}
