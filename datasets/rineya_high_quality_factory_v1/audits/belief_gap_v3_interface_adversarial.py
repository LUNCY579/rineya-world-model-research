#!/usr/bin/env python3
"""Immutable post-first-judgment public-interface discrepancy interpretation.
No other reviewer verdict, producer diagnostics/manifests or private expectations read.
"""
import collections,json,pathlib
import belief_gap_v3_adversarial_audit as a

INHERITED={'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}
def immutable_json(path,report):
    text=json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():a.require(path.read_text()==text,'Refuse immutable supplemental report rewrite')
    else:path.write_text(text)

def main():
    frozen_paths=[a.INPUT,a.CAND,a.OUT/f'{a.BATCH}.input_predictions.jsonl',a.OUT/f'{a.BATCH}.first_judgments.jsonl',a.OUT/f'{a.BATCH}.jsonl',a.ROOT/'audits/execution/belief-gap-v3-adversarial.json',a.ROOT/'audits/execution/belief-gap-v3-supplement-adversarial.json',pathlib.Path(a.__file__)]
    before={str(p.relative_to(a.ROOT)):a.sha(p) for p in frozen_paths};a.require(a.sha(a.CAND)==a.CAND_SHA,'Frozen candidate')
    docs=[a.ROOT/'schemas/BELIEF_CONTROL_PROPERTIES.md',a.ROOT/'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md',a.ROOT/'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md'];rows=a.load(a.CAND);impacts=[];counts=collections.Counter();groups=collections.defaultdict(list)
    for r in rows:
        affected=[]
        for c in r['family_control_claims']:
            name=c.get('name',c.get('property'))
            if name in INHERITED:
                counts[(name,c['scope'],'pair_sample_ids' in c,'probe_id' in c)]+=1
                if c['scope']=='corpus_variant' and 'pair_sample_ids' not in c:
                    affected.append({'name':name,'actual_claim':c,'missing_required_binding':'pair_sample_ids','present_manifest_pointer_does_not_substitute_for_claim_binding':True})
        if affected:impacts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':a.EXECUTION,'source_family_key':r['source_family_key'],'auxiliary_scalar_variant':r['auxiliary_scalar_variant'],'supplemental_metadata_status':'REVISE_METADATA_UNDER_INHERITED_V2_REQUIREMENT','affected_claims':affected,'semantic_target_or_property_defect_asserted':False,'original_first_final_receipts_unchanged':True})
        groups[r['source_family_key']].append(r)
    a.require(len(rows)==len(impacts)==3456 and sum(len(x['affected_claims']) for x in impacts)==6912,'Actual all-row two-claim omission')
    pair_evidence=[];examples=[]
    for f,rs in sorted(groups.items()):
        keyed={(r['controlled_context'],r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer']):r for r in rs}
        for variant in ('identity','transposed'):
            for length,leftctx,rightctx in [(2,'E','G'),(3,'F','H')]:
                for k in range(27):
                    for renderer in ('forward','reverse'):
                        l,r=keyed[(leftctx,variant,k,renderer)],keyed[(rightctx,variant,k,renderer)];lt,rt=l['semantic_task'],r['semantic_task'];al,dl=a.solve(lt);ar,dr=a.solve(rt)
                        a.require(a.canon(lt['worlds'])==a.canon(rt['worlds']) and a.canon(lt['queries'])==a.canon(rt['queries']) and len(lt['observations'])==len(rt['observations'])==length,'Actual inherited same-table/query/length')
                        changes=[i for i,(x,y) in enumerate(zip(lt['observations'],rt['observations'])) if a.canon(x)!=a.canon(y)];a.require(changes==[1] and lt['observations'][1]['fact']==rt['observations'][1]['fact'],'Actual only index1 equals change')
                        a.require(lt['observations'][0]==rt['observations'][0] and dl['prefix_survivors'][1]==dr['prefix_survivors'][1] and not al['surviving_worlds'] and ar['surviving_worlds'],'Actual matched prefix and mixed outcomes')
                        evidence={'source_family_key':f,'auxiliary_scalar_variant':variant,'observation_length':length,'schedule_index':k,'renderer':renderer,'pair_sample_ids':[l['sample_id'],r['sample_id']],'record_hashes':[l['record_hash'],r['record_hash']],'same_first_observation':lt['observations'][0],'same_first_prefix_world_ids':dl['prefix_survivors'][1],'only_changed_observation_index':1,'empty_prefix_sizes':al['belief_sizes'],'nonempty_prefix_sizes':ar['belief_sizes'],'actual_both_inherited_relationships':sorted(INHERITED),'evidence_source':'Own recomputation from actual supplied candidate inputs, no producer manifest/expected result used.'}
                        pair_evidence.append(evidence)
                        if k==0 and renderer=='forward':examples.append(evidence)
    a.require(len(pair_evidence)==864,'Actual independently reconstructable inherited pairs864')
    index=a.ROOT/'audits/execution/belief-gap-v3-inherited-interface-affected-records.jsonl';pairs=a.ROOT/'audits/execution/belief-gap-v3-inherited-interface-actual-pairs.jsonl';a.freeze(index,impacts);a.freeze(pairs,pair_evidence)
    report={
        'reviewer_execution_id':a.EXECUTION,'reviewer_role':'Adversarial Red-Team; supplemental public-interface interpretation','batch':a.BATCH,'status':'completed_immutable_supplement','stage':'Post first/final judgments discrepancy follow-up; parent disclosed a formal-gate disagreement, no formal report/code or other reviewer verdict opened.',
        'candidate_sha256':a.sha(a.CAND),'candidate_records':len(rows),'public_interface_source_hashes':{str(p.relative_to(a.ROOT)):a.sha(p) for p in docs},'actual_review_program_sha256':a.sha(pathlib.Path(__file__)),'own_semantic_helper_sha256':a.sha(pathlib.Path(a.__file__)),
        'interpretation':{
            'supplemental_metadata_decision':'REVISE_METADATA_UNDER_INHERITED_V2_REQUIREMENT','severity':'medium','v3_clearly_supersedes_inherited_pair_binding':False,'conclusion':'V3 does not clearly revoke the specific inherited V2 pair-ID obligation. Under the retained specific requirement, all3456 rows omit pair_sample_ids on both inherited corpus_variant claims. This is a real metadata binding omission; independently true controls and correct targets remain established.',
            'specific_retained_rule':{'path':'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md','line':42,'exact_relevant_text':'Supplied E/G and F/H claims use `corpus_variant` with actual same-family pair IDs; their evidence manifest must identify the length-specific pair.'},
            'v3_precedence_limit':{'path':'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md','line':3,'exact_relevant_text':'the amendments below take precedence for v3 grouping only.'},
            'v3_ambiguous_general_sentence':{'path':'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md','line':48,'exact_relevant_text':'pair-specific evidence may include actual two-element `pair_sample_ids`.'},
            'v3_section_scope':{'path':'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md','lines':[38,42,44,45,46,52,59],'reason':'The optional wording follows the New family control table of three V3 identifiers. The interface says retain original/V2 canonical fields with precise additions and grouping; no item explicitly makes the inherited two identifiers aggregate-only or relaxes their pair requirement.'},
            'strongest_alternative_reading':'Because V3 is later and line48 says required supplied claims and optional pair-specific evidence without repeating a qualifier, a producer/reviewer can plausibly read it as permitting aggregates for every family claim. The full actual groups are independently reconstructable and a manifest/control pointer can name them. That plausible ambiguity explains the implementation; it does not clearly override the specific retained requirement or justify silently treating the old fields as optional.',
            'semantic_relationship_truth':'PASS:864 actual same-variant E/G and F/H pairs independently verified, covering both lengths, all families/shifts/renderers. No false task target, per-row property or actual control relationship is asserted by this supplement.',
            'manifest_read_scope':'Producer family-control manifest bytes were not opened. A claim contains a relative pointer; this review cannot independently certify whether that manifest identifies every length-specific pair. The directly observed missing claim pair-ID binding is distinct from that uninspected manifest-content condition.',
            'earlier_red_scope_correction':'My completed V3 program accepted aggregate encoding under the broad optional-field wording and proved the actual relationships, but did not enforce the inherited V2 pair-ID condition. Thus its zero metadata errors is incomplete under the retained specific reading. Original first/final evidence is preserved rather than rewritten; this supplemental disagreement must accompany it.',
            'recommended_resolution':'Resolve the public interface explicitly and prospectively. If retaining V2 binding, create a fresh metadata-corrected candidate with length-specific same-family pair claims/IDs and exact evidence; keep old bytes and histories. If an aggregate relaxation is intended, publish an explicit versioned amendment naming these inherited identifiers and document the discrepancy rather than silently weakening a frozen gate. Data targets/worlds need no semantic correction on this evidence.',
            'approval_or_adjudication_authority':False
        },
        'actual_claim_instance_counts':[{'name':k[0],'scope':k[1],'pair_sample_ids_present':k[2],'probe_id_present':k[3],'instances':v} for k,v in sorted(counts.items())],
        'affected_rows':len(impacts),'affected_claim_instances':sum(len(x['affected_claims']) for x in impacts),'affected_record_index_path':str(index.relative_to(a.ROOT)),'affected_record_index_sha256':a.sha(index),'actual_pair_evidence_path':str(pairs.relative_to(a.ROOT)),'actual_pair_evidence_sha256':a.sha(pairs),'actual_pairs_verified':len(pair_evidence),
        'concrete_minimal_claim_counterexample':{'sample_id':rows[0]['sample_id'],'record_hash':rows[0]['record_hash'],'scope':'First actual A/identity record still makes whole-family inherited supplied claims. The missing pair fields are visible without inspecting any target.','actual_inherited_claim_objects':[c for c in rows[0]['family_control_claims'] if c.get('name',c.get('property')) in INHERITED]},
        'concrete_actual_length_specific_witnesses':examples,'frozen_original_artifact_hashes':before,'frozen_original_artifacts_unchanged':all(a.sha(a.ROOT/p)==h for p,h in before.items()),'forbidden_producer_manifest_reference_or_other_verdict_reads':False,
        'scope_limit':'Guided same-model discrepancy interpretation after frozen first judgments, not new blinded approval, new semantic-family generation, adjudication or release action.'
    }
    a.require(report['frozen_original_artifacts_unchanged'],'Frozen original changed')
    output=a.ROOT/'audits/execution/belief-gap-v3-inherited-interface-adversarial.json';immutable_json(output,report)
    print(a.canon({'report_sha256':a.sha(output),'program_sha256':a.sha(pathlib.Path(__file__)),'affected_rows':len(impacts),'affected_claim_instances':report['affected_claim_instances'],'actual_pairs':len(pair_evidence),'conclusion':report['interpretation']['conclusion']}),flush=True)

if __name__=='__main__':main()
