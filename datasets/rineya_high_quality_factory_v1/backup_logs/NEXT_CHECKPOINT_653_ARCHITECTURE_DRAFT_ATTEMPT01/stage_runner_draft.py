"""UNEXECUTED stage architecture. No copy/add/commit/push implementation exists."""
def expected_whole_tree(contracts,parent_tree,files):
    # parent_tree/files are exact Git mode/object maps reverified at runtime.
    prefix=contracts.PREFIX+'/'
    expected={name:row for name,row in parent_tree.items() if not name.startswith(prefix)}
    for e in files:
        contracts.artifact(e,True);name=prefix+e['path'];old=parent_tree.get(name)
        mode=old['mode'] if old else '100644';contracts.require(mode in ('100644','100755'),'Symlink/nonregular entry forbidden')
        expected[name]={'mode':mode,'git_blob':e['git_blob']}
    return expected

def mutation_gate(contracts,frozen_scope,actual):
    contracts.require(actual['head']==contracts.PARENT and actual['remote_branch']==contracts.PARENT and actual['clean_before_copy'] is True,'Exact clean653 parent required')
    for key in ('full_original_path_secret_scans','complete_four_metadata_report','raw_lossless_byte_comparisons','all_parent_paths_preserved','release222_preserved','refs65_preserved','git6_preserved','private_ADMIN','disk_reserve','source_target_index_sha256_gitsha1','whole_index_outside_prefix_equal'):
        contracts.require(actual[key] is True,'Stage guard failed: '+key)
    contracts.strict_delta_diff(actual['unrestricted_diff_check'])
    contracts.require(actual['whole_index_tree']==actual['expected_whole_tree'],'Exact whole-index tree required')
    return {'authorized_archival_only':True,'grants_data_approval':False,'remote_status_after_nonforce_push':'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING'}

def engine_contract():
    return ('Freeze stage source/tests/typed custody adapter/config against actual newSPEC beforefirst import',
            'No deletion/reset/force/config/whitespace-policy change; literal-path add onlyaudited names; ignored names force-add onlyifexactinventory',
            '64KiB source-to-temp copy, fullSHA+GitSHA1 beforeatomicreplace; noexception stream for unknownaliases',
            'stream every actual local index blob, checkallpaths/hash/size and outsideprefix parentbytes beforecommit',
            'strict cached diff exit0 beforecommit and strict parent653->newcommit diff exit0 aftercommit; preserve failure alias on anynonzero',
            'soleparent653/localcommit retained/nonforcepush/privateADMIN/currentmain6guards/endPUSH_PENDING; neverBACKED_UP here')
