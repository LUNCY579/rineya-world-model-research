from pathlib import Path
import hashlib,json
L=Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs')
r=(L/'recover_remote_checkpoint_exact_public_metadata_v1.py').read_text()
r=r.replace('Python sources may evolve only with an explicit option; frozen Git bytes still\nhave to match the original spec.', 'All current sources are held to the exact approved-release inventory; evolving\nsources are rejected. Four typed metadata exceptions remain fully byte bound.')
r=r.replace('AUTHOR = "/root/backup_manager"', 'sys.dont_write_bytecode = True\nAUTHOR = "/root/backup_manager"')
r=r.replace("if args.allow_evolving_source:raise ValueError", "if args.workers!=2:raise ValueError('Exactly two actual remote workers required')\n    if args.allow_evolving_source:raise ValueError")
r=r.replace("if args.source_freeze.is_symlink() or hash_file(args.source_freeze)['sha256']!=args.expected_freeze_sha256:", "if (source_path(FACTORY,args.source_freeze.absolute().relative_to(FACTORY).as_posix())!=args.source_freeze.absolute()\n            or not args.source_freeze.absolute().is_relative_to(FACTORY/'backup_logs')\n            or args.source_freeze.is_symlink() or hash_file(args.source_freeze)['sha256']!=args.expected_freeze_sha256):")
r=r.replace("if freeze['source']['sha256']!=hash_file(Path(__file__))['sha256'] or freeze['author_execution']!=AUTHOR:", "if (freeze['source']!={'path':Path(__file__).relative_to(FACTORY).as_posix(),**hash_file(Path(__file__))}\n            or freeze['author_execution']!=AUTHOR or freeze.get('frozen_before_first_import_test_execution') is not True):")
r=r.replace("    metadata_helper().validate_config()\n    return freeze", "    required={'backup_logs/belief-v4-approved-release-r1_STAGE.json','backup_logs/belief-v4-approved-release-r1_LOCAL_COMMIT.json',\n        'backup_logs/belief-v4-approved-release-r1_STAGE_LEDGER.jsonl'}\n    if not required<={p.absolute().relative_to(FACTORY).as_posix() for p in args.prior_evidence}:\n        raise ValueError('Actual stage/localcommit/ledger bindings required')\n    metadata_helper().validate_config()\n    return freeze")
start=r.index('def check_new_source_freeze')
validation='''def validate_metadata_report(report,stream=False):
    helper=metadata_helper();expected={row['path']:row for row in helper.EXPECTED_FILES}
    if (type(report) is not dict or report.get('status')!='PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS'
            or type(report.get('allowed_exact_match_detections')) is not int or report['allowed_exact_match_detections']!=4
            or type(report.get('unbound_secret_detections')) is not int or report['unbound_secret_detections']!=0
            or report.get('configuration_sha256')!=METADATA_CONFIG_SHA or report.get('existing_regexes_unchanged') is not True):
        raise ValueError('Metadata report truth/config/count differs')
    rows=report.get('four_exception_files')
    if type(rows) is not list or len(rows)!=4 or len({v.get('path') for v in rows})!=4 or {v.get('path') for v in rows}!=set(expected):
        raise ValueError('All four unique metadata reports required')
    flag='full_size_sha256_gitsha1_verified_before_exception_acceptance' if stream else 'full_size_and_sha_verified_before_acceptance'
    for value in rows:
        row=expected[value['path']]
        if (type(value.get('size')) is not int or value['size']!=row['size'] or value.get('sha256')!=row['sha256']
                or value.get('matches')!=row['matches'] or value.get(flag) is not True):
            raise ValueError('Metadata report full-byte/span proof differs')
        if stream and (not re.fullmatch(r'[0-9a-f]{40}',value.get('git_blob','')) or report.get('scanner_source_sha256')!=METADATA_SOURCE_SHA):
            raise ValueError('Metadata report actual Git/source proof missing')
    return report


def validate_exact_scope(spec):
    files=spec.get('files',[]);raws=spec.get('raw_size_exclusions',[])
    by={e['path']:e for e in files}
    if len(files)!=3158 or sum(e['size'] for e in files)!=1263090770 or len(raws)!=31 or sum(e['size'] for e in raws)!=1724728576:
        raise ValueError('Exact approved-release scope/count/bytes differs')
    if sum(len(e['lossless_representations']) for e in raws)!=32 or sum(e['size']*len(e['lossless_representations']) for e in raws)!=1766951239:
        raise ValueError('Exact local representation coverage differs')
    for row in metadata_helper().EXPECTED_FILES:
        if by.get(row['path'])!={k:row[k] for k in ('path','size','sha256')}:raise ValueError('Known metadata SPEC binding missing/changed')
    return True


'''
r=r[:start]+validation+r[start:]
r=r.replace('    inventory = spec["files"]', '    validate_exact_scope(spec)\n    inventory = spec["files"]')
r=r.replace('            groups, evolved = {}, []', '''            scoped={name[len(PREFIX)+1:] for name in remote_paths if name.startswith(PREFIX+'/')}
            if scoped!={e['path'] for e in inventory}:raise ValueError('Actual remote prefix coverage differs')
            groups, evolved = {}, []''')
r=r.replace('            raw_proofs = []', "            if len(groups)!=2474:raise ValueError('Exact unique blob scope differs')\n            raw_proofs = []")
r=r.replace('            receipt["capped_source_secret_scan_before"]=metadata_helper().report(require_complete=True)', '            receipt["capped_source_secret_scan_before"]=validate_metadata_report(metadata_helper().report(require_complete=True))')
r=r.replace('            receipt["capped_source_secret_scan_after"]=metadata_helper().report(require_complete=True)', '            receipt["capped_source_secret_scan_after"]=validate_metadata_report(metadata_helper().report(require_complete=True))')
r=r.replace('            receipt["actual_local_git_metadata_exception_scan"]=complete_metadata_stream_report("local_git")', '            receipt["actual_local_git_metadata_exception_scan"]=validate_metadata_report(complete_metadata_stream_report("local_git"),True)')
r=r.replace('                receipt["actual_remote_metadata_exception_scan"]=complete_metadata_stream_report("actual_remote")', '                receipt["actual_remote_metadata_exception_scan"]=validate_metadata_report(complete_metadata_stream_report("actual_remote"),True)')
r=r.replace('            check_new_source_freeze(args)\n            receipt["local_raw', '            if check_new_source_freeze(args)!=source_freeze:raise ValueError("Frozen preparation changed")\n            receipt["local_raw')
p=L/'recover_remote_checkpoint_exact_public_metadata_v2.py';p.open('x').write(r)
rsha=hashlib.sha256(p.read_bytes()).hexdigest()
g=(L/'verify_archival_remote_lossless_exact_public_metadata_v1.py').read_text().replace('69986a0dc5eebffda55746b3a22f01dcc98254f61b8a1d450a77475803bca46c',rsha).replace('recover_remote_checkpoint_exact_public_metadata_v1.py','recover_remote_checkpoint_exact_public_metadata_v2.py')
g=g.replace('    s.loader.exec_module(module)\n    return module', '    s.loader.exec_module(module)\n    if sha_file(path)!=DEPENDENCY_SHA:raise ValueError("Recovery dependency changed during import")\n    return module')
a=g.index('    for key in ("capped_source_secret_scan_before"');z=g.index('    remote = verified.get',a)
g=g[:a]+'''    for key in ("capped_source_secret_scan_before","capped_source_secret_scan_after","actual_local_git_metadata_exception_scan","actual_remote_metadata_exception_scan"):
        bounded.validate_metadata_report(verified.get(key,{}),key.startswith('actual_'))
'''+g[z:]
g=g.replace('def main():', '''def validate_preparation_freeze(freeze,bounded):
    if (freeze.get('author_execution')!=AUTHOR or freeze.get('frozen_before_first_import_test_execution') is not True
            or freeze.get('runner_source')!={'path':Path(__file__).relative_to(FACTORY).as_posix(),**bounded.hash_file(Path(__file__))}):
        raise ValueError('Gzip exact source/preexecution freeze differs')
    for row in freeze['frozen_inputs']:
        if bounded.hash_file(bounded.source_path(FACTORY,row['path']))!={k:row[k] for k in ('size','sha256')}:
            raise ValueError('Frozen gzip preparation input changed')
    bounded.metadata_helper().validate_config()
    return True


def main():''')
g=g.replace('    p.add_argument("--expected-raw-count", type=int, default=24)', '    p.add_argument("--expected-raw-count", type=int, required=True)')
g=g.replace('    bounded = load_bounded();output_paths(args, bounded)', '''    bounded = load_bounded();output_paths(args, bounded)
    if (args.runner!=AUTHOR or args.commit!=bounded.EXPECTED_COMMIT or args.expected_spec_sha256!=bounded.EXPECTED_SPEC_SHA
            or args.expected_raw_count!=31):raise ValueError('Exact runner/commit/spec/raw31 required')''')
g=g.replace('            if freeze.get("author_execution") != AUTHOR or freeze["runner_source"]["sha256"] != receipt["source_sha256"]:\n                raise ValueError("Actual runner lacks its exact source freeze")', '            validate_preparation_freeze(freeze,bounded)')
g=g.replace('            spec = read_bound(args.spec, args.expected_spec_sha256, bounded)', '            spec = read_bound(args.spec, args.expected_spec_sha256, bounded)\n            bounded.validate_exact_scope(spec)')
g=g.replace('            if (remote.get("sha") != args.commit', '            if ([p["sha"] for p in remote.get("parents",[])]!=[bounded.EXPECTED_PARENT] or remote.get("sha") != args.commit')
g=g.replace('            receipt.update(status="PASS_ACTUAL_REMOTE_GZIP_DECODE"', '''            validate_preparation_freeze(freeze,bounded)
            if (len(work)!=31 or len(manifest_refs)!=32 or len({r['manifest']['path'] for r in manifest_refs})!=31
                    or sum(e['raw_bytes'] for e in proofs)!=1724728576):raise ValueError('Frozen deduplicated gzip coverage/byte totals differ')
            receipt.update(status="PASS_ACTUAL_REMOTE_GZIP_DECODE"''')
p=L/'verify_archival_remote_lossless_exact_public_metadata_v2.py';p.open('x').write(g)
print(json.dumps({'recovery_sha256':rsha,'gzip_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
