"""UNEXECUTED actual gzip proof architecture; dynamic representation cardinalities."""
def plan_pairs(contracts,spec,full_payloads):
    pairs={};refs=[]
    for raw in spec['raw_size_exclusions']:
        for rep in raw['lossless_representations']:
            key=(raw['path'],rep['gzip']['path'])
            for role in ('gzip','manifest'):
                record=rep[role];contracts.artifact(record);actual=full_payloads[record['path']]
                contracts.require((actual['size'],actual['sha256'])==(record['size'],record['sha256']),'Lossless packed/manifest fullproof crosswire')
            contracts.require(key not in pairs or pairs[key]['gzip']==rep['gzip'],'Representation pairconflict')
            pairs[key]={'raw':raw,'gzip':rep['gzip'],'git_blob':full_payloads[rep['gzip']['path']]['git_blob']}
            refs.append({'raw_path':raw['path'],'gzip_path':rep['gzip']['path'],'manifest':rep['manifest'],'actual_fullproof_manifest_git_blob':full_payloads[rep['manifest']['path']]['git_blob']})
    return pairs,refs

def engine_contract():
    return ('NEWsourcefreeze bindsNEWrecovery actualSHA and actualcompletedfullreceipt/stage/spec/newcommit beforefirstpayloadfetch',
            'actualeach dedup(raw_path,gzip_path)fetchno-cache;64KiB compressed/decodedchunks +8MiB spool, CRC/end/size/SHA256/GitSHA1/statchecks',
            'byte-for-bytecompare everyactualdecoded chunk againstpreservedlocalraw; no rawwholememory/no cachedmanifest inference',
            'doNOThardcode prior31fetches/32refs/31globalpaths; derive futurepair/reference/globalmanifest sets separately',
            'ownactualnewfetches andremote decodedbytes separatefromfullproof localrepresentationiterations/bytes; manifests useexplicitinheritedactualfullpayloadproof boundary',
            'nooverallBACKED_UPclaim ordataapproval from gzip alone; exclusivefreshproof/ledger/failedaliases')
