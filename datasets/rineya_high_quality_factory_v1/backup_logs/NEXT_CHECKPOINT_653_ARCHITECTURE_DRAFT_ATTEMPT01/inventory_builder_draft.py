"""UNEXECUTED inventory architecture. Filesystem/scan engine port follows approval."""
def select_scope(contracts,observed,previous,pending):
    # observed rows are literal names/stat-stable hashes at ONE cut, not future glob guesses.
    parent={e['path'] for e in previous['files']};deferred={}
    for row in pending:
        contracts.artifact(row['artifact']);name=row['artifact']['path']
        contracts.require(row['actual_actor_status_at_cut']=='RUNNING' and row['new_untracked_owned_path'] is True and name in observed and name not in parent,'Only observed NEW unfinished paths may defer')
        contracts.require(name not in deferred,'Duplicate pending path');deferred[name]=row
    selected=set(observed)-set(deferred)
    contracts.require(parent<=selected,'Never defer/delete a parent3158 path')
    return selected,deferred

def readiness_criteria(contracts,root_bindings):
    # These statuses authorize byte archival only, never candidate/data acceptance.
    for key in ('actual_root_integration_tests','actual_producer_code_source_freeze','actual_root_stable_cut_hold'):
        row=root_bindings[key];contracts.artifact(row['artifact']);contracts.require(row['completed'] is True,'Minimum root source/test readiness incomplete')
    contracts.require(root_bindings['data_approval_by_backup'] is False,'Backup cannot approve data')
    return True

def engine_contract():
    return ('64KiB full-size/SHA256/GitSHA1 original-path source scanner; complete4 exact a799/2e18 observations bothends',
            'all current>25MiB paths equal rawscope; preserve previous31 raw/historical279/current149 bindings; reject every other stale manifest',
            'decode every perraw manifest representation byte-for-byte; separately dedup remote raw/gzip pairs',
            '222 release capped/raw union +65readonly references +6 Git byte/existence guards +privateADMIN +clean653 dedicated +cleanmaina223',
            'before/after2*capped_inventory_bytes+512MiB reserve and bounded actualAPI core counts; exact source/config closure',
            'futureSPEC includes computed GitSHA1 per capped row; draft must never emit SPEC/readiness claim')
