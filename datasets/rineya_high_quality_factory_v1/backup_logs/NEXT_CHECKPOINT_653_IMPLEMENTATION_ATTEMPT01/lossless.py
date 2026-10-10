"""Separate NEW actual gzip downloads and byte comparisons, derived from full scope."""
from __future__ import annotations
import next653_common as c
from next653_recovery import validate_full_report, validate_stage_dependencies

def representation_plan(spec):
    files,raw=c.all_entries(spec);c.scope(spec);pairs={};references=[]
    for p,row in sorted(raw.items()):
        for rep in row['lossless_representations']:
            packed=files[rep['gzip']['path']];manifest=files[rep['manifest']['path']]
            references.append({'raw_path':p,'gzip_path':packed['path'],'manifest_path':manifest['path'],'manifest_git_blob':manifest['git_blob'],'manifest_sha256':manifest['sha256']})
            key=(p,packed['path'])
            pair={'raw':row,'gzip':packed}
            if key in pairs and (c.projection(pairs[key]['raw']),c.projection(pairs[key]['gzip']))!=(c.projection(row),c.projection(packed)):raise ValueError('Duplicate pair cross-wired to different raw/package bytes')
            pairs[key]=pair
    return pairs,references

def validate_decode_rows(rows,pairs):
    if type(rows) is not list or len(rows)!=len(pairs):raise ValueError('Every unique actual gzip pair required once')
    seen=set()
    for row in rows:
        k=(row.get('raw_path'),row.get('remote_gzip_path'))
        if k in seen or k not in pairs or row.get('actual_remote_fetch_this_run') is not True or row.get('decoded_byte_for_byte_equal_to_preserved_raw') is not True:raise ValueError('Missing/duplicate/cached gzip proof')
        seen.add(k);raw,packed=pairs[k]['raw'],pairs[k]['gzip']
        wanted={'raw_bytes':raw['size'],'raw_sha256':raw['sha256'],'remote_git_blob':packed['git_blob'],'remote_gzip_bytes':packed['size'],'remote_gzip_sha256':packed['sha256']}
        if any(type(row.get(key)) is not type(value) or row.get(key)!=value for key,value in wanted.items()):raise ValueError('Actual compressed/decoded tuple differs')
    if seen!=set(pairs):raise ValueError('Actual gzip pair coverage incomplete')
    return True

def validate_gzip_report(report,spec,commit,bindings):
    pairs,refs=representation_plan(spec)
    if report.get('status')!='PASS_ACTUAL_REMOTE_LOSSLESS' or report.get('commit_sha')!=commit or not c.typed_equal(report.get('exact_scope'),c.scope(spec)):raise ValueError('Actual gzip complete scope differs')
    for k in ('spec','stage','full_proof','full_ledger'):
        if report.get(k)!=bindings[k]:raise ValueError('Actual gzip proof binding differs')
    validate_decode_rows(report['actual_remote_decodes'],pairs)
    if (not c.typed_equal(report.get('manifest_references_inherited_from_actual_full_remote_proof'),refs)
            or type(report.get('actual_new_remote_gzip_fetches')) is not int or report['actual_new_remote_gzip_fetches']!=len(pairs)
            or type(report.get('actual_decoded_bytes')) is not int or report['actual_decoded_bytes']!=sum(v['raw']['size'] for v in pairs.values())):raise ValueError('Remote decode/ref accounting differs')
    if (type(report.get('extra_remote_manifest_fetches_claimed')) is not int or report['extra_remote_manifest_fetches_claimed']!=0
            or type(report.get('local_representation_iterations')) is not int or report['local_representation_iterations']!=c.scope(spec)['local_representation_iterations']):raise ValueError('Local repeats and remote fetch accounting confused')
    for ctx in ('source_before','source_after'):c.validate_metadata_report(report['metadata_reports'][ctx],ctx)
    return report

def execute(cfg):
    if set(cfg['fields'])!={'commit_sha'}:raise ValueError('Only exact actual commit field accepted')
    spec=c.bound_json(cfg['inputs']['spec']);stage,local,commit=validate_stage_dependencies(cfg,spec)
    full=c.bound_json(cfg['inputs']['full_proof']);validate_full_report(full,spec,commit,cfg['inputs']['spec'])
    if full.get('ledger')!=cfg['inputs']['full_ledger'] or full.get('stage')!=cfg['inputs']['stage']:raise ValueError('Actual full proof/ledger/stage binding differs')
    files,_=c.all_entries(spec);expected=c.groups(files)
    c.validate_actual_groups(c.ledger_rows(cfg['inputs']['full_ledger'],'actual_remote_blob_verified'),expected,'actual_remote_fetch_this_run')
    before=c.custody(commit);session=c.SecretSession();source_before=c.verify_sources(spec,session,'source_before');local=c.verify_local_lossless(spec)
    c.verify_git_scope(spec,commit);pairs,refs=representation_plan(spec)
    if before['api_core_counts_only']['remaining']<len(pairs)+16:raise ValueError('Insufficient current API budget for every fresh gzip')
    full_by_path={p:r for r in full['actual_remote_blob_groups'] for p in r['paths']}
    for ref in refs:
        proof=full_by_path.get(ref['manifest_path'])
        if proof is None or proof['git_blob']!=ref['manifest_git_blob'] or proof['sha256']!=ref['manifest_sha256']:raise ValueError('Manifest lacks actual full remote payload proof')
    ledgerpath=c.new_ledger(cfg['outputs']['ledger']);results=[];g=c.library('gzip')
    # Frozen G.consume_remote uses 8MiB compressed spool and 64KiB raw comparison.
    for pair in pairs.values():
        raw,packed=pair['raw'],pair['gzip'];rawpath=c.safe(c.FACTORY,raw['path'])
        result=c.raw_blob(packed['git_blob'],lambda handle,r=raw,p=packed,rp=rawpath:g.consume_remote(handle,rp,r,p,{'sha':p['git_blob']},c.HERE))
        results.append(result);c.ledger(ledgerpath,{'event':'actual_remote_gzip_decoded',**result})
    validate_decode_rows(results,pairs);source_after=c.verify_sources(spec,session,'source_after')
    c.verify_git_scope(spec,commit);after=c.custody(commit);c.recheck_inputs(cfg)
    receipt={'status':'PASS_ACTUAL_REMOTE_LOSSLESS','commit_sha':commit,'sole_parent_sha':c.PARENT,'tree_sha':full['tree_sha'],
             'spec':cfg['inputs']['spec'],'stage':cfg['inputs']['stage'],'full_proof':cfg['inputs']['full_proof'],'full_ledger':cfg['inputs']['full_ledger'],
             'exact_scope':c.scope(spec),'actual_new_remote_gzip_fetches':len(pairs),'actual_decoded_bytes':sum(v['raw']['size'] for v in pairs.values()),
             'local_representation_iterations':len(local),'local_decoded_bytes':sum(r['raw_bytes'] for r in local),
             'actual_remote_decodes':results,'manifest_references_inherited_from_actual_full_remote_proof':refs,
             'extra_remote_manifest_fetches_claimed':0,'metadata_reports':{'source_before':source_before,'source_after':source_after},
             'custody_before':before,'custody_after':after,'ledger':c.file_binding(ledgerpath),'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],
             'author_execution':'/root/backup_manager','actual_runner_execution':cfg['runner_execution'],'finished_at_utc':c.now(),
             'grants_data_approval':False,'self_inventory_boundary':'Separate gzip proof/ledger are postinventory local; no overall new inventory implied'}
    validate_gzip_report(receipt,spec,commit,cfg['inputs'])
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    return {'status':receipt['status'],'receipt':row,'actual_new_remote_gzip_fetches':len(pairs),'actual_decoded_bytes':receipt['actual_decoded_bytes']}
