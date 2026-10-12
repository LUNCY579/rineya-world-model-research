"""Executable acceptance gates; disagreement is preserved as quarantine evidence."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
from collections import Counter
from ..pipelines.common import canonical, digest

ALLOWED_INPUT={'environment','initial_state','rules','goal','constraints','candidates','instruction',
               'context','observations','information','possible_worlds','probability_assumption','search_request','utility_definition'}
ALLOWED_TOP={'schema_version','id','family','difficulty','split','provenance','input','targets','metadata','verification','quality_gate'}
FORBIDDEN_INPUT={'targets','oracle','verification','quality_gate','best_candidate_ids','ranking','success_probability','terminal_state','next_state','trajectory','realized_utility_delta'}


def rational(value):
    if isinstance(value,dict):
        if set(value)!={'numerator','denominator'} or type(value['numerator']) is not int or type(value['denominator']) is not int or value['denominator']<=0:
            raise ValueError('malformed exact rational')
        return Fraction(value['numerator'],value['denominator'])
    if type(value) is not int: raise ValueError('utility component must be exact integer or rational')
    return Fraction(value)


def recursive_input_check(value):
    if isinstance(value,dict):
        if FORBIDDEN_INPUT.intersection(value):raise ValueError('oracle-only key in model input')
        for v in value.values():recursive_input_check(v)
    elif isinstance(value,list):
        for v in value:recursive_input_check(v)
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite input')


def template_signature(task):
    s=task.get('initial_state', task.get('possible_worlds',[{}])[0].get('state'))
    if task['environment']=='ledger':size=[len(s['accounts']),len(s['entries'])]
    elif task['environment']=='blocksworld':size=[len(s['on'])]
    elif task['environment']=='register':size=[len(s['registers'])]
    else:size=[len(s)]
    return digest({'environment':task['environment'],'world_size':size,'rule_keys':sorted(task['rules']),'constraint_keys':sorted(task.get('constraints',{}))})


def semantic_fingerprint(record):
    # Bounded exhaustive entity canonicalization has a separate template-size
    # gate below; shared serialization is not shared transition/scoring logic.
    from ..generators.build import semantic_fingerprint as normalize
    from ..generators.profiles import resolve_record_profile
    profile=resolve_record_profile(record['provenance'])
    _validate_profile_world_size(record['input'],profile['config'],record['split'])
    if profile['id']=='v1':
        # Every stored V1 fingerprint and unknown-reference marker is an
        # immutable recovery authority. Do not alter the original normalizer.
        return normalize(record)
    from ..generators.version2 import restore_source_names
    restored=deepcopy(record)
    restored['input']=restore_source_names(record,config=profile['config'])
    _validate_profile_world_size(restored['input'],profile['config'],record['split'])
    return normalize(restored)


def _validate_profile_world_size(task, config, split):
    """Bound factorial namespace work before normalization of untrusted rows."""
    if split not in config['split_templates'] or type(task) is not dict:
        raise ValueError('unknown reserved-template split or task')
    spec=config['split_templates'][split]
    state=task.get('initial_state')
    if state is None:
        worlds=task.get('possible_worlds')
        state=worlds[0] if type(worlds) is list and worlds else None
    if type(state) is not dict:
        raise ValueError('reserved-template state must be an object')
    environment=task.get('environment')
    fields={'ledger':(('accounts',spec['accounts']),('entries',spec['entries'])),
            'blocksworld':(('on',spec['blocks']),),
            'register':(('registers',spec['accounts']),)}
    if environment not in fields:
        raise ValueError('unsupported reserved-template environment')
    if any(type(state.get(field)) is not dict or len(state[field])!=size
           for field,size in fields[environment]):
        raise ValueError(environment+' reserved-template world-size violation')


def validate_schema(record):
    if not isinstance(record,dict) or not set(record)<=ALLOWED_TOP:raise ValueError('record shape/unknown field')
    required=ALLOWED_TOP-{'verification','quality_gate'}
    if not required<=set(record):raise ValueError('missing canonical field')
    if record['schema_version']!='gcdl-1.0':raise ValueError('unsupported schema')
    if record['family'] not in list('ABCDEFGH'):raise ValueError('unknown family')
    if record['difficulty'] not in ['easy','medium','hard','adversarial']:raise ValueError('unknown difficulty')
    if record['split'] not in ['train','dev','sealed','ood']:raise ValueError('unknown split')
    p=record['provenance']
    if not {'seed','generator_version','environment_version','config_version','config_hash','template_id','template_group_id','parent_world_id','lineage_id','record_index'}<=set(p):raise ValueError('missing provenance')
    if type(p['seed']) is not int or p['seed']<0:raise ValueError('invalid seed')
    task=record['input']
    if not isinstance(task,dict) or not set(task)<=ALLOWED_INPUT:raise ValueError('unknown input field')
    if not {'environment','rules','goal','constraints','candidates'}<=set(task) or not ('initial_state' in task or 'possible_worlds' in task):raise ValueError('missing task field')
    if not 1<=len(task['candidates'])<=256:raise ValueError('candidate size budget')
    ids=[]
    for c in task['candidates']:
        if set(c)!={'id','plan'} or not isinstance(c['id'],str) or not isinstance(c['plan'],list) or len(c['plan'])>12:raise ValueError('malformed candidate')
        ids.append(c['id'])
    if len(set(ids))!=len(ids):raise ValueError('duplicate candidate IDs')
    recursive_input_check(task)
    declared_utility={'comparison':'lexicographic','components':['expected_feasibility','expected_goal_atoms_satisfied','negative_expected_successful_action_cost'],
        'feasibility':'legal execution and no constraint violations at initial or intermediate states','all_infeasible':'abstain; no best candidate'}
    if task.get('utility_definition')!=declared_utility:
        raise ValueError('unsupported or misleading declared utility definition')
    if not {'oracle','decision'}<=set(record['targets']) or not set(record['targets'])<={'oracle','decision','search','bounded_proof'}:
        raise ValueError('target schema mismatch')
    if record['family']=='F' and ('search' not in record['targets'] or 'search_request' not in task):
        raise ValueError('computation family missing matched-search evidence')
    if record['family']=='G' and not {'search','bounded_proof'}.intersection(record['targets']):
        raise ValueError('distillation family missing verified search evidence')
    from ..generators.build import record_id
    if record_id(record)!=record['id']:raise ValueError('stable ID mismatch')
    if template_signature(task)!=record['metadata']['protected_template_signature']:raise ValueError('template signature not derived from task')
    # Resolve the record's exact trusted archived profile, never the current
    # default. V1 identities and split reservations remain valid under V2 code.
    from ..generators.profiles import resolve_record_profile
    profile=resolve_record_profile(p)
    cfg=profile['config']
    spec=cfg['split_templates'][record['split']]
    if record['schema_version']!=profile['schema_version'] or p['template_id']!=spec['template']:
        raise ValueError('provenance version/config mismatch')
    _validate_profile_world_size(task,cfg,record['split'])
    if p['template_group_id']!=spec['template'] or p['lineage_id']!=spec['template']:raise ValueError('lineage split partition mismatch')
    if profile['id']=='v2':
        metadata=record['metadata']
        if record['family']=='C':
            _validate_v2_c_record(record)
        elif ({'goal_bundle_id','goal_index'} & set(metadata)
              or 'fixed_candidate_goal_triplet' in metadata.get('tags',[])):
            raise ValueError('v2 C bundle metadata must describe family C')
        from ..generators.version2 import validate_vocabulary
        validate_vocabulary(record,config=cfg)
    canonical(record)


def _validate_v2_c_record(record):
    task,metadata=record['input'],record['metadata']
    if task['environment']!='blocksworld':
        raise ValueError('v2 C requires Blocksworld')
    state=task.get('initial_state')
    goal=task['goal']
    if (type(state) is not dict or type(state.get('on')) is not dict
            or type(goal) is not dict or set(goal)!={'on'} or type(goal['on']) is not dict
            or set(goal['on'])!=set(state['on'])):
        raise ValueError('v2 C requires a full-forest goal')
    if goal['on']==state['on']:
        raise ValueError('v2 C goal must differ from initial state')
    if len(task['candidates'])!=6:
        raise ValueError('v2 C requires exactly six supplied candidates')
    index=metadata.get('goal_index')
    if type(index) is not int or not 0<=index<=2:
        raise ValueError('v2 C goal_index must be an integer from zero to two')
    # Reconstruct the identity independently using only the fixed inference
    # bundle. Goal, targets, record ID, indices and audit metadata are excluded.
    identity=digest({key:task[key] for key in ('environment','initial_state','rules',
                                             'constraints','candidates','utility_definition')})
    if metadata.get('goal_bundle_id')!=identity:
        raise ValueError('v2 C goal bundle identity does not match fixed model input')


def validate_v2_group(records, require_full_curriculum=False):
    """Check V2 atomic goal/vocabulary groups after authoritative row audits.

    Caller must first run ``audit_record`` on every row. This helper adds
    cross-record custody/contrast checks; it does not replace oracle auditing.
    Any failure requires the worker to quarantine the complete V2 batch.
    """
    from ..generators.profiles import resolve_record_profile
    from ..generators.version2 import restore_source_names
    if not records:
        raise ValueError('empty v2 group')
    if type(require_full_curriculum) is not bool:
        raise ValueError('require_full_curriculum must be boolean')
    profiles=[resolve_record_profile(row['provenance']) for row in records]
    if any(profile['id']!='v2' for profile in profiles):
        raise ValueError('v2 group cannot mix generation profiles')
    if len({row['id'] for row in records})!=len(records):
        raise ValueError('duplicate canonical record in v2 group')
    for row in records:
        validate_schema(row)
    if require_full_curriculum:
        counts=Counter(row['family'] for row in records)
        if any(counts[family]<minimum for family,minimum in profiles[0]['config']['family_minimums'].items()):
            raise ValueError('v2 group lacks required full-curriculum families')
    bundles={};pairs={}
    for row in records:
        metadata=row['metadata'];p=row['provenance']
        if row['family']=='C':
            bundles.setdefault(metadata['goal_bundle_id'],[]).append(row)
        vocabulary=metadata['vocabulary']
        if ('vocabulary_matched_control' in metadata.get('tags',[])
                or metadata.get('perturbation',{}).get('kind')=='vocabulary_matched_control'
                or vocabulary['stratum']=='familiar_control'):
            pairs.setdefault(p['parent_world_id'],[]).append(row)
    if not bundles and not pairs:
        raise ValueError('v2 group lacks a controlled goal bundle or vocabulary pair')
    for rows in bundles.values():
        if len(rows)!=3 or {row['metadata']['goal_index'] for row in rows}!={0,1,2}:
            raise ValueError('v2 C bundle must contain exactly indices zero, one and two')
        custody={(row['split'],row['provenance']['seed'],row['provenance']['parent_world_id'],
                  row['provenance']['template_group_id'],row['provenance']['lineage_id'],
                  row['metadata']['counterfactual_group_id']) for row in rows}
        if len(custody)!=1 or any(row['metadata']['counterfactual_group_id']!=row['provenance']['parent_world_id'] for row in rows):
            raise ValueError('v2 C bundle parent/split custody mismatch')
        non_goal=[]
        for row in rows:
            task=deepcopy(row['input']);task.pop('goal');non_goal.append(canonical(task))
        if len(set(non_goal))!=1 or len({canonical(row['input']['goal']) for row in rows})!=3:
            raise ValueError('v2 C bundle must change only its three distinct goals')
        best=[set(row['targets']['decision']['best_candidate_ids']) for row in rows]
        if any(not ids for ids in best) or any(best[a]&best[b] for a in range(3) for b in range(a)):
            raise ValueError('v2 C bundle requires disjoint nonempty exact best sets')
        for row,ids in zip(rows,best):
            if not any(candidate['id'] in ids and rational(candidate['feasible_probability'])==1
                       and rational(candidate['success_probability'])==1
                       for candidate in row['targets']['decision']['candidates']):
                raise ValueError('v2 C bundle requires a feasible successful best candidate per goal')
    for rows in pairs.values():
        if (len(rows)!=2 or {row['metadata']['vocabulary']['stratum'] for row in rows}
                !={'familiar_control','reserved_dev'}):
            raise ValueError('matched vocabulary group requires both DEV strata')
        if any(row['split']!='dev' or row['family']!='D' for row in rows):
            raise ValueError('matched vocabulary pair must be DEV family D')
        custody={(row['provenance']['seed'],row['provenance']['template_group_id'],
                  row['provenance']['lineage_id'],row['metadata']['counterfactual_group_id']) for row in rows}
        if len(custody)!=1 or any(row['metadata']['counterfactual_group_id']!=row['provenance']['parent_world_id'] for row in rows):
            raise ValueError('matched vocabulary pair parent custody mismatch')
        restored=[canonical(restore_source_names(row,config=profiles[0]['config'])) for row in rows]
        if len(set(restored))!=1 or rows[0]['targets']['decision']!=rows[1]['targets']['decision']:
            raise ValueError('matched vocabulary pair changed source task or exact decision labels')


def validate_decision(labels,oracle):
    outcomes=oracle['candidates']
    scores={c['id']:tuple(rational(v) for v in c['utility']) for c in outcomes}
    best=sorted(k for k,v in scores.items() if v==max(scores.values())) if max(v[0] for v in scores.values())>0 else []
    levels=sorted(set(scores.values()),reverse=True)
    ranking=[{'rank':i+1,'candidate_ids':sorted(k for k,v in scores.items() if v==level)} for i,level in enumerate(levels)]
    if labels['best_candidate_ids']!=best or labels['ranking']!=ranking:raise ValueError('ranking/tie mismatch')
    projected=[{'id':c['id'],'utility':c['utility'],'success_probability':c['success_probability'],'feasible_probability':c['feasible_probability']} for c in outcomes]
    if labels['candidates']!=projected:raise ValueError('pointwise label mismatch')
    for c in outcomes:
        for key in ['success_probability','legal_probability','feasible_probability']:
            prob=rational(c[key])
            if not 0<=prob<=1:raise ValueError('probability outside [0,1]')
    if labels['optimality']!={'scope':'supplied_candidates','proof':'exhaustive_candidate_execution'}:raise ValueError('unsupported candidate optimality')
    if not best:
        expected_label='no_feasible_candidate';expected_abstention='no_feasible_candidate'
    elif not any(rational(c['success_probability'])>0 for c in outcomes):
        expected_label='impossible_in_candidate_set';expected_abstention='no_candidate_can_succeed'
    else:
        expected_label='tied_best' if len(best)>1 else 'partially_observed' if oracle.get('world_count',1)>1 else 'determined'
        expected_abstention='not_required'
    uncertainty=labels['uncertainty']
    if uncertainty['label']!=expected_label or uncertainty['abstention']!=expected_abstention:
        raise ValueError('uncertainty/abstention label mismatch')


def search_result(task,budget):
    from ..oracles.independent import evaluate
    t=deepcopy(task);t['candidates']=t['candidates'][:budget]
    oracle=evaluate(t)
    scores={c['id']:tuple(rational(v) for v in c['utility']) for c in oracle['candidates']}
    ids=sorted(k for k,v in scores.items() if v==max(scores.values())) if max(v[0] for v in scores.values())>0 else []
    nodes=sum(len(o['steps']) for c in oracle['candidates'] for o in c['outcomes'])
    best=next(c for c in oracle['candidates'] if tuple(rational(v) for v in c['utility'])==max(scores.values()))
    return {'candidate_budget':len(t['candidates']),'candidate_evaluations':len(t['candidates']),
        'transition_node_evaluations':nodes,'best_candidate_ids':ids,'best_utility':best['utility']}


def validate_auxiliary(record):
    from ..oracles.independent import evaluate
    targets=record['targets'];task=record['input']
    search=targets.get('search')
    if search:
        actual_ids=[c['id'] for c in task['candidates']]
        if search['matched_candidate_ids']!=actual_ids or search['benefit_is_oracle_only'] is not True:
            raise ValueError('matched search candidate universe/target separation mismatch')
        cb=search['cheap']['candidate_budget'];eb=search['expanded']['candidate_budget']
        if type(cb) is not int or type(eb) is not int or not 1<=cb<=eb<=len(actual_ids):
            raise ValueError('invalid search budget')
        request=task.get('search_request')
        if request and (request.get('cheap_candidate_budget')!=cb or request.get('expanded_candidate_budget')!=eb or request.get('same_candidate_universe') is not True):
            raise ValueError('declared search budgets differ from measured policies')
        cheap=search_result(task,search['cheap']['candidate_budget'])
        expanded=search_result(task,search['expanded']['candidate_budget'])
        if search['cheap']!=cheap or search['expanded']!=expanded:raise ValueError('matched search supervision mismatch')
        delta=[rational(b)-rational(a) for a,b in zip(cheap['best_utility'],expanded['best_utility'])]
        if delta!=[rational(d) for d in search['realized_utility_delta']]:raise ValueError('value of compute delta mismatch')
        if search['additional_candidate_evaluations']!=expanded['candidate_evaluations']-cheap['candidate_evaluations']:raise ValueError('search candidate cost mismatch')
        if search['additional_transition_node_evaluations']!=expanded['transition_node_evaluations']-cheap['transition_node_evaluations']:raise ValueError('search node cost mismatch')
    proof=targets.get('bounded_proof')
    if proof:
        actions=proof['action_alphabet'];horizon=proof['horizon']
        if not 1<=len(actions)<=5 or type(horizon) is not int or not 0<=horizon<=2 or proof['global_unbounded_optimality_claim'] is not False or proof['scope']!='finite_declared_action_alphabet_and_horizon' or proof['proof']!='exhaustive_execution':raise ValueError('exhaustive proof scope invalid')
        plans=[{'id':f'proof_{i}','plan':list(plan)} for i,plan in enumerate(plan for h in range(horizon+1) for plan in product(actions,repeat=h))]
        t=deepcopy(task);t['candidates']=plans
        oracle=evaluate(t)
        scores=[tuple(rational(v) for v in c['utility']) for c in oracle['candidates']]
        if proof['enumerated_plan_count']!=len(plans) or tuple(rational(v) for v in proof['best_utility'])!=max(scores):raise ValueError('bounded optimality proof mismatch')
        exists=any(rational(c['success_probability'])==1 for c in oracle['candidates'])
        if proof['solution_exists_in_bound']!=exists:raise ValueError('bounded solution existence mismatch')
        if (proof['witness_plan'] is not None)!=exists:raise ValueError('bounded witness presence mismatch')
        if proof['witness_plan'] is not None:
            if len(proof['witness_plan'])>horizon or any(a not in actions for a in proof['witness_plan']):
                raise ValueError('witness outside declared proof bound')
            t['candidates']=[{'id':'witness','plan':proof['witness_plan']}]
            if rational(evaluate(t)['candidates'][0]['success_probability'])!=1:raise ValueError('proof witness does not succeed')


def audit_record(record):
    from ..oracles.primary import evaluate as primary
    from ..oracles.independent import evaluate as independent
    validate_schema(record)
    a=primary(record['input']);b=independent(record['input'])
    if a!=b:raise ValueError('primary/independent oracle disagreement')
    if record['targets']['oracle']!=b:raise ValueError('record/oracle outcome disagreement')
    validate_decision(record['targets']['decision'],b)
    validate_auxiliary(record)
    reordered=deepcopy(record['input']);reordered['candidates'].reverse()
    c=independent(reordered)
    if {v['id']:v for v in b['candidates']}!={v['id']:v for v in c['candidates']}:raise ValueError('option-order oracle inconsistency')
    from ..decision_labels.views import training_views
    views=training_views(record)
    for family in views.values():
        for row in family:recursive_input_check(row['input'])
    return {'schema':'PASS','authoritative_evidence':'PASS','differential_oracle':'PASS','ranking':'PASS',
        'option_order':'PASS','input_target_separation':'PASS','auxiliary_labels':'PASS','proof_scope':'candidate_set_or_explicit_finite_bound',
        'evidence_sha256':digest(b),'checker_context':'two separately implemented executable checkers; correlated engineering disclosed'}


def audit_group(records,db=None):
    accepted=[];rejected=[];causes={};batch_seen={};parents={};templates={};signatures={}
    for source in records:
        row=deepcopy(source)
        try:
            checks=audit_record(row)
            p=row['provenance'];split=row['split'];semantic=semantic_fingerprint(row);parent=p['parent_world_id'];template=p['template_group_id'];signature=template_signature(row['input'])
            for table,key in [(parents,parent),(templates,template),(signatures,signature)]:
                if key in table and table[key]!=split:raise ValueError('within-batch protected split overlap')
                table[key]=split
            if row['id'] in batch_seen:raise ValueError('duplicate ID within batch')
            if db is not None:
                matches=db.execute('SELECT id,parent,split FROM seen WHERE id=? OR semantic=? OR parent=? OR template=?',(row['id'],semantic,parent,template)).fetchall()
                for ident,existing_parent,existing_split in matches:
                    if existing_split!=split:raise ValueError('protected split leakage')
                    if ident==row['id']:raise ValueError('duplicate accepted ID')
                duplicate=db.execute('SELECT parent FROM seen WHERE semantic=?',(semantic,)).fetchall()
                if any(found[0]!=parent for found in duplicate):raise ValueError('semantic duplicate across parent worlds')
            if semantic in batch_seen and batch_seen[semantic]!=parent:raise ValueError('semantic duplicate across parent worlds')
            batch_seen[row['id']]=parent;batch_seen[semantic]=parent
            row['verification']=checks;row['quality_gate']={'status':'accepted','rejection_reasons':[]}
            accepted.append(row)
        except Exception as error:
            reason=type(error).__name__+': '+str(error)
            causes[reason]=causes.get(reason,0)+1
            row['quality_gate']={'status':'quarantined','rejection_reasons':[reason]}
            rejected.append(row)
    report={'verdict':'PASS' if not rejected else 'PARTIAL_QUARANTINE','accepted':len(accepted),'quarantined':len(rejected),
        'causes':causes,'oracle_agreement_on_accepted':1.0 if accepted else None,'schema_pass_rate_on_accepted':1.0 if accepted else None,
        'independent_human_or_agent_review':'separate release report required; executable checks are not an agent approval'}
    report['family_counts']=dict(Counter(r['family'] for r in accepted))
    report['difficulty_counts']=dict(Counter(r['difficulty'] for r in accepted))
    report['perturbation_tags']=dict(Counter(tag for r in accepted for tag in r['metadata']['tags']))
    report['unique_task_fingerprints']=len({semantic_fingerprint(r) for r in accepted})
    return accepted,rejected,report
