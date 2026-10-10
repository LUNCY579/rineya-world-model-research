"""Public-only v3 fact-local auxiliary controls; no producer/gold imports.

Uses the author's immutable v1 typed-set and v2 A-H proof primitives. Internal
v2 claim projections bind actual row IDs solely to execute core semantics;
actual submitted claims are separately checked against full v3 witnesses.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
from . import belief_controls_independent as v1
from . import belief_controls_v2_independent as v2

DEPENDENCY_SHA256 = {'v1':'ec9f387485f2a2d58ab557db0a2a7d95c24e77cb3108afa503a80c575e04b03e','v2':'5b97b73df0b9248b1ad155307e75e2a73b995dff64e81232382a6c05a2fe4984'}
PUBLIC_V3_SHA256 = '3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11'
VARIANTS = ('identity','transposed')
NEW_FAMILY = frozenset(['auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'])
FAMILY = v2.FAMILY | NEW_FAMILY
BeliefControlError=v1.BeliefControlError
_need,_json,_meta=v1._need,v1._json,v1._meta


def _hash(value): return hashlib.sha256(_json(value).encode('utf8')).hexdigest()


def _claims(row):
    claims=_meta(row,'family_control_claims');_need(type(claims) is list,'family_control_claims list required')
    output=[]; identifiers={}; allowed={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'}
    for claim in claims:
        _need(type(claim) is dict,'claim object required');names=set(claim)&{'name','property'}
        _need(len(names)==1 and 'scope' in claim and not set(claim)-allowed,'claim encoding fields invalid')
        name,scope=claim[next(iter(names))],claim['scope']
        _need(type(name) is str and name in FAMILY and scope in ('corpus_variant','audit_probe'),'undefined family property or scope')
        if 'source_family_key' in claim:_need(type(claim['source_family_key']) is str and claim['source_family_key']==_meta(row,'source_family_key'),'claim source family mismatch')
        for key in ('evidence_manifest','control_id','probe_id'):
            if key in claim:_need(type(claim[key]) is str and bool(claim[key]),'nonempty descriptive '+key+' required')
        if 'evidence_manifest' in claim:
            path=claim['evidence_manifest'];_need(not path.startswith(('/','\\')) and not re.match(r'^[A-Za-z]:',path) and '..' not in path.replace('\\','/').split('/'),'relative evidence provenance required')
        if 'pair_sample_ids' in claim:
            pair=claim['pair_sample_ids'];_need(scope=='corpus_variant' and type(pair) is list and len(pair)==2 and all(type(s) is str and s for s in pair) and pair[0]!=pair[1],'two distinct actual corpus pair IDs required')
        if 'probe_id' in claim:_need(scope=='audit_probe','probe_id requires audit_probe scope')
        if name in v2.NEW_FAMILY:_need(('pair_sample_ids' in claim and 'evidence_manifest' in claim) if scope=='corpus_variant' else 'probe_id' in claim,'v2 control requires pair/evidence or executed probe ID')
        if name in NEW_FAMILY and scope=='audit_probe':_need('probe_id' in claim,'new audit control requires executed probe ID')
        identifier=claim.get('control_id',name);_need(identifier not in identifiers or identifiers[identifier]==scope,'duplicate descriptive key has inconsistent scope');identifiers[identifier]=scope
        output.append({**claim,'property':name,'scope':scope})
    return output


def _control(row):
    variant=_meta(row,'auxiliary_scalar_variant');_need(variant in VARIANTS,'required auxiliary_scalar_variant must be identity/transposed')
    _need(_meta(row,'scalar_variant') is None,'undocumented scalar_variant alias')
    control=_meta(row,'auxiliary_scalar_control');_need(type(control) is dict and set(control)=={'fact','baseline_value','supported_value'},'auxiliary_scalar_control exact fields required')
    fact,b,g=control['fact'],control['baseline_value'],control['supported_value']
    roles=_meta(row,'fact_role_assignments');_need(type(roles) is dict and fact==roles.get('observation_auxiliary') and type(fact) is str,'control fact must be actual auxiliary role')
    _need(type(b) is str and type(g) is str and b!=g,'two distinct strings required')
    _need(b=='linen' and g not in {'stone','unknown','linen'},'targeted pilot requires linen baseline and distinct nonmarker supported string')
    task=row['semantic_task'];_need(all(q['fact']!=fact for q in task['queries']),'auxiliary control fact must be unqueried')
    values=[w['facts'][fact] for w in task['worlds']];_need(b in values and g in values,'both controlled strings must be supported on actual auxiliary column')
    prefix,_=v1._visible(row)
    _need(not any(key in prefix for key in ('auxiliary_scalar_variant','auxiliary_scalar_control','controlled_context','family_control_claims','property_witnesses')),'control annotations leaked into prompt')
    return control


def check(row):
    """Per-row closed metadata and truthful property errors, no stored targets."""
    try:
        _need(type(row) is dict,'row object required');v1.derive(row.get('semantic_task'));_control(row);_claims(row)
        return v2.check(v2._projection(row))
    except (BeliefControlError,KeyError,TypeError,ValueError) as error:return ['belief_control_v3_row: '+str(error)]


def transpose(task,control):
    """Independent local transposition; every matching world/observation changed."""
    v1.derive(task);_need(type(control) is dict and set(control)=={'fact','baseline_value','supported_value'},'malformed transposition control')
    fact,b,g=control['fact'],control['baseline_value'],control['supported_value']
    _need(type(fact) is str and type(b) is str and type(g) is str and b!=g,'typed distinct string control required')
    _need(all(fact in w['facts'] for w in task['worlds']) and all(q['fact']!=fact for q in task['queries']),'known unqueried auxiliary fact required')
    result=deepcopy(task)
    def swap(value):
        return g if type(value) is str and value==b else b if type(value) is str and value==g else value
    # JSON has occurrence semantics. Construct from each original occurrence
    # so even a caller's reused Python dict cannot be involuted twice in-place.
    result['worlds']=[{'world_id':world['world_id'],'facts':{key:swap(value) if key==fact else value for key,value in world['facts'].items()}} for world in task['worlds']]
    result['observations']=[{'fact':observation['fact'],'equals':swap(observation['equals']) if observation['fact']==fact else observation['equals']} for observation in task['observations']]
    return result


def scalar_pair(left,right,control):
    a,b=v1.derive(left),v1.derive(right);transposed=transpose(left,control)
    _need(_json(transposed)==_json(right),'actual tasks differ from complete fact-local transposition')
    _need(_json(transpose(transposed,control))==_json(left),'transposition involution failed')
    fact=control['fact'];values=[w['facts'][fact] for w in left['worlds']]
    _need(control['baseline_value'] in values and control['supported_value'] in values,'both strings must occur')
    changes=[{'world_id':x['world_id'],'fact':fact,'before':x['facts'][fact],'after':y['facts'][fact]} for x,y in zip(left['worlds'],right['worlds']) if _json(x['facts'][fact])!=_json(y['facts'][fact])]
    _need(bool(changes),'nontrivial changed auxiliary column required')
    _need(a['support_ids']==b['support_ids'] and _json(a['target'])==_json(b['target']),'all prefix IDs and exact full target must be invariant')
    return {'satisfied':True,'control':control,'world_column_changes':changes,'changed_observation_indices':[i for i,(x,y) in enumerate(zip(left['observations'],right['observations'])) if _json(x)!=_json(y)],'left_prefix_supports':a['support_ids'],'right_prefix_supports':b['support_ids'],'left':a['target'],'right':b['target'],'input_sha256':[_hash(left),_hash(right)],'involution_verified':True}


def same_observations_pair(left,right,control):
    a,b=v1.derive(left),v1.derive(right);transposed=transpose(left,control)
    values=[w['facts'][control['fact']] for w in left['worlds']]
    _need(control['baseline_value'] in values and control['supported_value'] in values,'both controlled strings must occur in counterfactual table')
    _need(_json(left['observations'])==_json(right['observations']) and _json(left['queries'])==_json(right['queries']),'observations/ordered queries must be exactly identical')
    _need(_json(transposed['worlds'])==_json(right['worlds']) and _json(left['worlds'])!=_json(right['worlds']),'world differences must be nontrivial complete auxiliary-column transposition only')
    _need(len(left['observations']) in (2,3) and len(left['observations'])==len(right['observations']),'same-observation pair length 2 or3 required')
    _need(bool(a['support_ids'][1]) and a['support_ids'][1]==b['support_ids'][1],'first-prefix IDs must match exactly and be nonempty')
    _need(not a['supports'][-1] and bool(b['supports'][-1]),'same observations pair must be empty then nonempty')
    _need(left['observations'][0]['fact']!=control['fact'],'first constraint must be outside auxiliary column')
    if len(left['observations'])==3:_need(_json(left['observations'][2])==_json(left['observations'][0]),'three-observation pair must repeat first')
    return {'satisfied':True,'observation_length':len(left['observations']),'control':control,'left_prefix_supports':a['support_ids'],'right_prefix_supports':b['support_ids'],'left':a['target'],'right':b['target'],'input_sha256':[_hash(left),_hash(right)],'observations':left['observations'],'ordered_queries':left['queries'],'world_column_changes':[{'world_id':x['world_id'],'before':x['facts'][control['fact']],'after':y['facts'][control['fact']]} for x,y in zip(left['worlds'],right['worlds']) if _json(x['facts'][control['fact']])!=_json(y['facts'][control['fact']])]}


def _project_core(rows):
    """Bind internal core proof IDs; submitted claim truth is not inferred here."""
    by={( _meta(r,'source_family_key'),_meta(r,'controlled_context'),_meta(r,'schedule_index'),_meta(r,'renderer')):r for r in rows if type(r) is dict}
    projected=[]
    for row in rows:
        item=v2._projection(row);claims=[];family=_meta(row,'source_family_key');schedule=_meta(row,'schedule_index');renderer=_meta(row,'renderer')
        v2._setmeta(item,'controlled_context',_meta(row,'controlled_context'))
        for lc,rc in [('E','G'),('F','H')]:
            pair=[by.get((family,c,schedule,renderer)) for c in (lc,rc)]
            if all(pair):
                for name in v2.NEW_FAMILY:claims.append({'name':name,'scope':'corpus_variant','pair_sample_ids':[p['sample_id'] for p in pair],'evidence_manifest':'independent-internal-core/'+lc+rc+'.json'})
        v2._setmeta(item,'family_control_claims',claims);projected.append(item)
    return projected


def _extra_histograms(members,rows):
    hist=defaultdict(lambda:defaultdict(Counter));rule_counts=defaultdict(lambda:Counter());scalar_pairs=defaultdict(lambda:defaultdict(Counter))
    for m in members:
        task=rows[m['row_index']]['semantic_task'];status='empty' if not m['analysis']['target']['surviving_worlds'] else 'nonempty';variant=m['variant'];length=len(task['observations']);split=rows[m['row_index']]['split']
        marker=any(type(o['equals']) is str and o['equals'] in {'stone','unknown'} for o in task['observations']);combined=marker and any(type(o['equals']) is str and o['equals']=='linen' for o in task['observations'])
        labels=['overall','family:'+m['family'],'split:'+split,'variant:'+variant,'context:'+m['context'],'length:'+str(length),'family:'+m['family']+'/variant:'+variant+'/length:'+str(length),'split:'+split+'/variant:'+variant+'/length:'+str(length)]
        for label in labels:
            for rule,prediction in [('v1_original_scalar_marker_empty',marker),('prospective_marker_and_auxiliary_token_empty',combined)]:
                rule_counts[label+'/'+rule]['records']+=1;rule_counts[label+'/'+rule]['errors']+=prediction!=(status=='empty')
            atoms=[list(v1._atom(o['equals'])) for o in task['observations']]
            scalar_pairs[label][_json(atoms)][status]+=1
            fact=m['control']['fact'];freq=Counter(_json(list(v1._atom(w['facts'][fact]))) for w in task['worlds']);char=Counter(len(w['facts'][fact]) for w in task['worlds'] if type(w['facts'][fact]) is str)
            for key,value in [('auxiliary_column_typed_frequency',sorted(freq.items())),('auxiliary_column_string_lengths',sorted(char.items())),('complete_observation_typed_scalar_sequence',atoms)]:hist[label+'/'+key][_json(value)][status]+=1
    return {'frozen_rule_measurements':{key:{**dict(c),'status_accuracy':(c['records']-c['errors'])/c['records'],'complete_task_accuracy':None,'note':'Status-only rule; separately measured frozen complete-core coverage is in variant diagnostics.'} for key,c in rule_counts.items()},'scalar_sequence_outcome_histograms':{key:{bucket:dict(c) for bucket,c in buckets.items()} for key,buckets in scalar_pairs.items()},'column_feature_outcome_histograms':{key:{bucket:dict(c) for bucket,c in buckets.items()} for key,buckets in hist.items()}}


def deterministic_diagnostics(diagnostics):
    """Fit the same one-token baselines with a declared stable max-score tie.

    Counts come from actual typed histograms. Canonical feature strings ascend,
    then False before True; this affects only equally scoring representatives.
    Frozen v2 source/results remain unchanged.
    """
    result=deepcopy(diagnostics)
    for detail in result.values():
        totals=detail['outcomes'];n=detail['records']
        for field in ('typed_scalar_anywhere','typed_scalar_position'):
            best=None
            for feature,bucket in sorted(detail['histograms'][field].items()):
                for polarity in (False,True):
                    correct=(bucket.get('empty',0)+totals.get('nonempty',0)-bucket.get('nonempty',0)) if polarity else (bucket.get('nonempty',0)+totals.get('empty',0)-bucket.get('empty',0))
                    candidate={'feature':feature,'present_predicts_empty':polarity,'correct':correct,'errors':n-correct,'accuracy':correct/n,'fitting':'Strongest tested in-sample one-token binary predicate, both polarities.','tie_policy':'Canonical feature ascending, False before True; first maximum.'}
                    if best is None or correct>best['correct']:best=candidate
            detail['baselines']['strongest_'+field]=best
    return result


def check_batch(rows):
    """Full conditional A-H x scalar-variant custody and actual pair proofs."""
    if type(rows) is not list:return {'errors':['v3 batch requires a list'],'counts':{'records':0}}
    errors=[];members=[];families=defaultdict(list);pairings={};ids={};variant_rows={v:[] for v in VARIANTS}
    for i,row in enumerate(rows):
        label=row.get('sample_id',str(i)) if type(row) is dict else str(i);errors.extend(label+': '+error for error in check(row))
        try:
            _need(type(row) is dict,'row object required');control=_control(row);actual=v1.analyze(row['semantic_task']);encoding=v1._encoding(v2._projection(row),actual);encoding['context']=_meta(row,'controlled_context');variant=_meta(row,'auxiliary_scalar_variant')
            _need(label not in ids,'duplicate sample_id');m={'row_index':i,'sample_id':label,**encoding,'variant':variant,'control':control,'analysis':actual,'decoded':v1._decode(row,encoding)};members.append(m);ids[label]=m;families[m['family']].append(m)
            coordinate=(m['family'],m['context'],variant,m['schedule'],m['renderer']);_need(coordinate not in pairings,'duplicate full variant coordinate');pairings[coordinate]=m;variant_rows[variant].append(row)
        except (BeliefControlError,KeyError,TypeError,ValueError) as error:errors.append(str(label)+': v3 batch row: '+str(error))
    cores={};properties={name:[] for name in FAMILY};crossings=[];probes=[]
    for variant,subset in variant_rows.items():
        core=v2.check_batch(_project_core(subset));cores[variant]=core;errors.extend(variant+' core: '+error for error in core['errors'])
        for name,proofs in core['corpus_controls'].items():properties[name].extend({**p,'auxiliary_scalar_variant':variant} for p in proofs)
        crossings.extend({**p,'auxiliary_scalar_variant':variant} for p in core['crossings'])
        for p in core['audit_probes']:probes.append({**p,'auxiliary_scalar_variant':variant,'probe_id':variant+':'+p['probe_id']})
    for family,group in sorted(families.items()):
        if len(group)!=864 or {m['variant'] for m in group}!=set(VARIANTS) or {m['context'] for m in group}!=set(v2.CONTEXTS):errors.append(family+': complete 8contexts x2variants x27 x2 required')
        controls={_json([m['control']['baseline_value'],m['control']['supported_value']]) for m in group}
        if len(controls)!=1:errors.append(family+': identity-coordinate control values vary across lineage')
        # Return all transposed decoded tables to the actual identity coordinate.
        canonical=[]
        for m in group:
            task=m['decoded'];ctl={**m['control'],'fact':'observation_auxiliary'}
            if m['variant']=='transposed':task=transpose(task,ctl)
            canonical.append(_json(task['worlds']))
        if len(set(canonical))!=1:errors.append(family+': inverse-transposed decoded source tables differ')
        familycross=[p for p in crossings if p['family']==family]
        valid=len(familycross)==32 and all(p['satisfied'] and len(p['sample_ids'])==27 for p in familycross)
        if valid:properties['context_auxiliary_scalar_variant_crossing'].append({'family':family,'scope':'corpus_variant','sample_ids':[m['sample_id'] for m in group],'evidence_id':family+':full-conditional-variant-crossing','conditional_groups':32,'variant_context_pairs':[[c,v] for c in v2.CONTEXTS for v in VARIANTS],'satisfied':True})
        else:errors.append(family+': actual full conditional variant crossing fails')
        for schedule in range(27):
            for renderer in ('forward','reverse'):
                for context in v2.CONTEXTS:
                    left=pairings.get((family,context,'identity',schedule,renderer));right=pairings.get((family,context,'transposed',schedule,renderer))
                    if not left or not right:errors.append(family+': missing scalar pair '+context);continue
                    try:
                        proof=scalar_pair(rows[left['row_index']]['semantic_task'],rows[right['row_index']]['semantic_task'],left['control'])
                        lp,_=v1._visible(rows[left['row_index']]);rp,_=v1._visible(rows[right['row_index']]);_need(lp==rp,'variant-specific prompt instruction or annotation')
                        properties['auxiliary_scalar_transposition_invariance'].append({'family':family,'scope':'corpus_variant','sample_ids':[left['sample_id'],right['sample_id']],'context':context,'schedule':schedule,'renderer':renderer,'evidence_id':family+':scalar:'+context+':'+str(schedule)+':'+renderer,**proof})
                    except BeliefControlError as error:errors.append(family+': scalar pair '+context+': '+str(error))
                for lc,rc in [('E','G'),('F','H')]:
                    for lv,rv in [('identity','transposed'),('transposed','identity')]:
                        left=pairings.get((family,lc,lv,schedule,renderer));right=pairings.get((family,rc,rv,schedule,renderer))
                        if not left or not right:errors.append(family+': missing same-observation cross-variant pair '+lc+'/'+rc);continue
                        try:
                            proof=same_observations_pair(rows[left['row_index']]['semantic_task'],rows[right['row_index']]['semantic_task'],left['control'])
                            properties['same_observations_auxiliary_table_counterfactual'].append({'family':family,'scope':'corpus_variant','sample_ids':[left['sample_id'],right['sample_id']],'left_context':lc,'right_context':rc,'left_variant':lv,'right_variant':rv,'schedule':schedule,'renderer':renderer,'evidence_id':family+':sameobs:'+lc+rc+':'+lv+':'+str(schedule)+':'+renderer,**proof})
                        except BeliefControlError as error:errors.append(family+': same observations '+lc+'/'+rc+': '+str(error))
        for context in ('E','G'):
            base=pairings.get((family,context,'identity',0,'forward'))
            if base:
                obs=rows[base['row_index']]['semantic_task']['observations'];expected=base['control']['baseline_value' if context=='E' else 'supported_value']
                if len(obs)!=2 or obs[1]['fact']!=base['control']['fact'] or obs[1]['equals']!=expected:errors.append(family+': identity E/G second equality/control mismatch')
        # Literal placement conditions needed for particular frozen-rule counts.
        for m in group:
            obs=rows[m['row_index']]['semantic_task']['observations'];marker=any(type(o['equals']) is str and o['equals'] in {'stone','unknown'} for o in obs)
            if marker!=(m['context'] in 'EFGH'):errors.append(family+': actual old-marker placement condition fails')
    declarations=defaultdict(list);identifiers={}
    for m in members:
        try:
            for claim in _claims(rows[m['row_index']]):
                key=(m['family'],claim.get('control_id',claim['property']));scope=claim['scope']
                if key in identifiers and identifiers[key]!=scope:errors.append(m['family']+': descriptive evidence key inconsistent scope')
                identifiers[key]=scope;declarations[m['family'],_json(claim)].append(m['sample_id'])
        except BeliefControlError:pass
    ledger={(p['family'],p['probe_id']):p for p in probes};claim_checks=[]
    for (family,encoded),sources in sorted(declarations.items()):
        claim=json.loads(encoded);name,scope=claim['property'],claim['scope'];key=(family,claim.get('probe_id'))
        if scope=='audit_probe' and 'probe_id' in claim and key not in ledger:
            templates=[p for p in probes if p['family']==family and p['property']==name];bound=None
            if templates and name in v1.FAMILY:bound=v1._execute_declared_probe(templates[0],claim['probe_id'])
            else:
                template=next((p for p in properties[name] if p['family']==family),None)
                if template and name!='context_auxiliary_scalar_variant_crossing':
                    tasks=[deepcopy(rows[ids[s]['row_index']]['semantic_task']) for s in template['sample_ids']]
                    if name=='auxiliary_scalar_transposition_invariance':detail=scalar_pair(*tasks,template['control'])
                    elif name=='same_observations_auxiliary_table_counterfactual':detail=same_observations_pair(*tasks,template['control'])
                    elif name in v2.NEW_FAMILY:detail=v2.matched_pair(*tasks)
                    else:detail=next((d for n,d in v1._relations(*tasks) if n==name),None)
                    if detail:bound={'family':family,'property':name,'scope':'audit_probe','probe_id':claim['probe_id'],'tasks':tasks,'relation':detail,'executed_targets':[v1.derive(t)['target'] for t in tasks],'satisfied':True,'executed_under_declared_id':True,'added_corpus_records':0,'added_source_families':0}
                elif template and name=='context_auxiliary_scalar_variant_crossing':
                    # Execute all actual finite conditional groups under this ID;
                    # this ledger never replaces mandatory supplied coverage.
                    selected=[m for m in members if m['family']==family];by=defaultdict(list)
                    for m in selected:by[m['context'],m['variant'],m['renderer']].append(m)
                    valid=len(by)==32 and all(sorted(m['schedule'] for m in group)==list(range(27)) and v2._crossing(group,rows)['satisfied'] for group in by.values())
                    if valid:bound={'family':family,'property':name,'scope':'audit_probe','probe_id':claim['probe_id'],'executed_input_sha256':[_hash(rows[m['row_index']]['semantic_task']) for m in selected],'conditional_groups':32,'satisfied':True,'executed_under_declared_id':True,'added_corpus_records':0,'added_source_families':0}
            if bound:probes.append(bound);ledger[key]=bound
        if scope=='audit_probe' and 'probe_id' in claim and key in ledger and ledger[key]['property']!=name:errors.append(family+': declared probe ID conflicts with executed property')
        witnesses=[p for p in (properties[name] if scope=='corpus_variant' else probes) if p['family']==family and (scope=='corpus_variant' or p['property']==name)]
        if 'pair_sample_ids' in claim:
            pair=set(claim['pair_sample_ids'])
            if any(s not in ids or ids[s]['family']!=family for s in pair):errors.append(family+': pair IDs not actual same-family records')
            witnesses=[p for p in witnesses if pair<=set(p.get('sample_ids',[]))]
        if 'probe_id' in claim:witnesses=[p for p in witnesses if p['probe_id']==claim['probe_id']]
        valid=bool(witnesses);claim_checks.append({'family':family,'claim':claim,'declaring_sample_ids':sorted(set(sources)),'matched_witness_count':len(witnesses),'satisfied':valid,'evidence_ids':[p.get('evidence_id',p.get('probe_id')) for p in witnesses]})
        if not valid:errors.append(family+': missing actual scoped claim witness '+name+'/'+scope)
    for family in families:
        for name in NEW_FAMILY:
            if not any(c['family']==family and c['claim']['property']==name and c['claim']['scope']=='corpus_variant' and c['satisfied'] for c in claim_checks):errors.append(family+': missing required declared supplied '+name)
        for name in v2.NEW_FAMILY:
            for length in (2,3):
                if not any(c['family']==family and c['claim']['property']==name and c['claim']['scope']=='corpus_variant' and c['satisfied'] and any(p['family']==family and p['observation_length']==length and set(c['claim'].get('pair_sample_ids',[]))<=set(p['sample_ids']) for p in properties[name]) for c in claim_checks):errors.append(family+': missing required declared v2 supplied '+name+' length'+str(length))
    # Custody must agree across variant cores as well as within each core.
    for field in ('semantic_family_id','semantic_parent_id'):
        owners=defaultdict(set)
        for m in members:owners[rows[m['row_index']][field]].add(m['family'])
        if any(len(value)>1 for value in owners.values()):errors.append(field+': lineage crosses families/variants')
    if cores['identity']['source_index_mapping']!=cores['transposed']['source_index_mapping']:errors.append('source index-family mapping changes across scalar variants')
    controls=properties['same_observations_auxiliary_table_counterfactual'];paired_ids=[s for p in controls for s in p['sample_ids']]
    if len(controls)!=864 or len(set(paired_ids))!=len(paired_ids):errors.append('required864 disjoint actual same-observation opposite-status pairs absent')
    if len(properties['auxiliary_scalar_transposition_invariance'])!=1728:errors.append('required1728 actual scalar pairs absent')
    extras=_extra_histograms(members,rows)
    if len(members)==3456 and not errors:
        for name in ('v1_original_scalar_marker_empty','prospective_marker_and_auxiliary_token_empty'):
            if extras['frozen_rule_measurements']['overall/'+name]['errors']!=864:errors.append('measured frozen-rule errors differ from witnessed literal theorem '+name)
    decoded_tables={_json(sorted([w['facts'] for w in m['decoded']['worlds']],key=_json)) for m in members};visible_tables={_json(sorted(rows[m['row_index']]['semantic_task']['worlds'],key=_json)) for m in members}
    return {'errors':errors,'counts':{'records':len(rows),'controlled_records':len(members),'source_families':len(families),'semantic_base_contexts':len({(m['family'],m['context']) for m in members}),'scalar_variant_base_cases':len({(m['family'],m['context'],m['variant']) for m in members}),'base_context_query_observations':3*len({(m['family'],m['context'],m['variant']) for m in members}),'rotation_level_query_observations':3*sum(m['renderer']=='forward' for m in members),'rendered_query_observations':3*len(members),'conditional_crossing_histograms':len(crossings),'scalar_pairs':len(properties['auxiliary_scalar_transposition_invariance']),'field_order_pairs':len(properties['field_order_invariance']),'opposite_status_same_observation_pairs':len(controls),'opposite_status_paired_records':len(set(paired_ids)),'decoded_source_structures':len(cores['identity']['structural_signatures']),'distinct_decoded_scalar_tables':len(decoded_tables),'distinct_visible_tables':len(visible_tables),'executed_probes':len(probes),'added_probe_corpus_records':0,'added_probe_source_families':0},'rows':[{'sample_id':m['sample_id'],'family':m['family'],'context':m['context'],'variant':m['variant'],'supports':m['analysis']['support_ids'],'derived_target':m['analysis']['target']} for m in members],'crossings':crossings,'corpus_controls':properties,'audit_probes':probes,'claim_checks':claim_checks,'variant_shortcut_diagnostics':{v:deterministic_diagnostics(cores[v]['shortcut_diagnostics']) for v in VARIANTS},'extra_shortcut_diagnostics':extras,'same_observation_rule_lower_bound':{'disjoint_actual_pairs':len(controls),'minimum_errors_using_only_identical_matched_features':len(controls),'features':'ordered observations/queries, world/fact/query/observation counts, first-prefix IDs; excludes differing world payload table'},'source_index_mapping':cores['identity']['source_index_mapping'],'structural_signatures':cores['identity']['structural_signatures'],'audited_semantic_tasks_sha256':_hash([r['semantic_task'] for r in rows if type(r) is dict and 'semantic_task' in r]),'dependencies':DEPENDENCY_SHA256,'internal_core_claim_projection':'Submitted claims replaced only during primitive semantic execution by actual pair-ID bindings; every submitted claim independently validated above.','approval_verdict':None}
