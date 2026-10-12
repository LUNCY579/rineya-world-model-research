"""Frozen bounded DEV diagnostics in memory; never publishes training records."""
import argparse
from copy import deepcopy
import json
import time

from .build import generate_group as legacy_group
from .profiles import resolve_profile
from .version2 import generate_goal_triplet, generate_vocabulary_pair
from ..auditors.integrity import audit_record, semantic_fingerprint, validate_v2_group
from ..decision_labels.rendering import parse_input_text, render_training_views
from ..pipelines.common import ROOT, canonical, digest, file_hash, immutable_bytes, source_hashes


def choices(task):
    candidates=task['candidates']
    qualifying=[c for c in candidates if c['plan'] and
                any(d['plan']==c['plan'][:-1] for d in candidates) and
                any(d['plan']==list(reversed(c['plan'])) for d in candidates)]
    return {'candidate_relationship':min(c['id'] for c in (qualifying or candidates)),
            'first_option':candidates[0]['id'],
            'shortest_plan_then_ID':min(candidates,key=lambda c:(len(c['plan']),c['id']))['id'],
            'longest_plan_then_ID':min(candidates,key=lambda c:(-len(c['plan']),c['id']))['id'],
            'lowest_opaque_ID':min(c['id'] for c in candidates)}


def freeze():
    spec=ROOT/'configs/v2_experiment.json'
    profile=resolve_profile('v2')
    code=source_hashes()
    receipt={'version':'v2-experiment-freeze-1','specification':json.loads(spec.read_text()),
             'specification_sha256':file_hash(spec),'code_hashes':code,'source_digest':digest(code),
             'v2_profile':{k:profile[k] for k in ('id','config_version','config_hash','generator_version','environment_version','config_file_sha256')},
             'lexicons_sha256':digest(profile['config']['lexicons']),
             'publication':False,'training_launched':False,
             'purpose':'Mechanical DEV comparison; hypotheses and source frozen before measurement'}
    path=ROOT/'manifests/V2_EXPERIMENT_FREEZE.json'
    immutable_bytes(path,(canonical(receipt)+'\n').encode())
    return receipt


def measure():
    path=ROOT/'manifests/V2_EXPERIMENT_FREEZE.json'
    frozen=json.loads(path.read_text())
    code=source_hashes()
    if (frozen['code_hashes']!=code or frozen['source_digest']!=digest(code)
            or frozen['specification_sha256']!=file_hash(ROOT/'configs/v2_experiment.json')):
        raise ValueError('frozen experiment source/specification changed; no adaptive rerun permitted')
    spec=frozen['specification']
    if (len(spec['goal_seeds'])!=32 or len(spec['vocabulary_seeds'])!=32
            or spec['max_tasks']!=160 or spec['max_wall_seconds']!=300
            or spec['publication'] is not False or spec['model_training'] is not False):
        raise ValueError('unsupported frozen experiment budget')
    profile=resolve_profile('v2');cfg=profile['config']
    if digest(cfg['lexicons'])!=frozen['lexicons_sha256']:
        raise ValueError('frozen lexicon changed')
    existing=ROOT/'reports/V2_CONTROLLED_EXPERIMENT.json'
    if existing.exists():
        result=json.loads(existing.read_text())
        if result.get('source_digest')!=digest(code) or result.get('freeze_sha256')!=file_hash(path):
            raise ValueError('existing result does not match frozen source/specification')
        if (result.get('version')!='v2-controlled-result-1' or result.get('verdict') not in ('PASS','FAIL')
                or result.get('goal_parents_attempted')!=32 or result.get('vocabulary_pairs')!=32
                or type(result.get('goal_parents_valid')) is not int
                or not 0<=result['goal_parents_valid']<=32
                or result.get('tasks_verified_in_memory')!=result['goal_parents_valid']*3+64
                or result.get('canonical_records_published')!=0 or result.get('training_launched') is not False
                or len(result.get('parent_receipts',[]))!=result['goal_parents_valid']+32
                or (result['verdict']=='PASS' and (result['goal_parents_valid']<spec['minimum_valid_goal_parents']
                    or result.get('oracle_or_label_mismatches')!=0 or result.get('reproduction_mismatches')!=0))):
            raise ValueError('completed result lacks bounded full experiment evidence')
        return result  # Observation/recovery never reruns completed DEV measurements.
    started=time.monotonic()
    deadline=started+spec['max_wall_seconds']
    receipts=[];exclusions=[];before=0;matched_before=0;after=0;task_count=0;valid=0
    secondary={name:0 for name in ('first_option','shortest_plan_then_ID','longest_plan_then_ID','lowest_opaque_ID')}
    exact_recovery={name:0 for name in ('candidate_relationship',*secondary)}
    before_exact=0;matched_before_exact=0;before_ties=0;after_ties=0
    rendered_checks=0

    def bounded():
        if time.monotonic()>deadline: raise TimeoutError('frozen DEV wall budget exhausted')
        if task_count>spec['max_tasks']: raise ValueError('frozen DEV task budget exceeded')

    def checked(record,seed):
        bounded()
        if record['split']!='dev' or record['provenance']['seed']!=seed:
            raise ValueError('controlled source seed/split mismatch')
        evidence=audit_record(record)
        return {'record_id':record['id'],'input_sha256':digest(record['input']),
                'target_sha256':digest(record['targets']),'evidence_sha256':evidence['evidence_sha256'],
                'semantic_fingerprint':semantic_fingerprint(record)}

    for seed in spec['goal_seeds']:
        bounded()
        old=next(r for r in legacy_group(seed,split='dev') if r['family']=='C')
        audit_record(old)
        old_choice=choices(old['input'])['candidate_relationship']
        old_best=set(old['targets']['decision']['best_candidate_ids'])
        old_correct=old_choice in old_best
        old_exact=old_best=={old_choice}
        before+=int(old_correct)
        before_exact+=int(old_exact);before_ties+=int(len(old_best)>1)
        records,diagnostic=generate_goal_triplet(seed,split='dev',config=cfg)
        again,again_diagnostic=generate_goal_triplet(seed,split='dev',config=cfg)
        if canonical(records)!=canonical(again) or diagnostic!=again_diagnostic:
            raise ValueError('C construction reproduction failure')
        if not records:
            exclusions.append(diagnostic)
            continue
        checks=[checked(r,seed) for r in records]
        validate_v2_group(records)
        chosen=[choices(r['input']) for r in records]
        if any(item!=chosen[0] for item in chosen):raise ValueError('goal-blind selector changed within triplet')
        best=[set(r['targets']['decision']['best_candidate_ids']) for r in records]
        correct=sum(chosen[i]['candidate_relationship'] in best[i] for i in range(3))
        if correct>1:raise ValueError('goal-blind one-third structural bound failed')
        after+=correct;matched_before+=int(old_correct);matched_before_exact+=int(old_exact)
        valid+=1;task_count+=len(records);after_ties+=sum(len(values)>1 for values in best)
        for name in secondary:
            secondary[name]+=sum(chosen[i][name] in best[i] for i in range(3))
        for name in exact_recovery:
            exact_recovery[name]+=sum(best[i]=={chosen[i][name]} for i in range(3))
        # This is an in-memory accepted validation view, not publication.
        rendered=[]
        for record in records:
            value=deepcopy(record)
            value['quality_gate']={'status':'accepted'}
            value['verification']=audit_record(value)
            rendered.append(list(render_training_views(value,'listwise',3)))
        for variant in range(3):
            payloads=[]
            for rows in rendered:
                task=parse_input_text(rows[variant]['input_text'])
                task.pop('goal')
                payloads.append(json.dumps(task,sort_keys=False,ensure_ascii=False))
                rendered_checks+=1
            if len(set(payloads))!=1:raise ValueError('rendered triplet changes more than goal, including key/candidate order')
        receipts.append({'seed':seed,'arm':'C','baseline_correct':old_correct,'after_correct':correct,
                         'baseline_best_candidate_ids':sorted(old_best),'baseline_best_set_size':len(old_best),
                         'baseline_singleton_exact_recovery':old_exact,
                         'goal_best_candidate_ids':[sorted(values) for values in best],
                         'goal_best_set_sizes':[len(values) for values in best],
                         'singleton_exact_recovery':{name:sum(best[i]=={chosen[i][name]} for i in range(3)) for name in exact_recovery},
                         'after_tasks':3,'diagnostic':diagnostic,'records':checks,'selectors':chosen[0]})
        bounded()
    lexical_strings=set();lexical_pairs=0
    train_strings={word for pool in cfg['lexicons']['train'].values() for word in pool}
    for seed in spec['vocabulary_seeds']:
        bounded()
        records,diagnostic=generate_vocabulary_pair(seed,split='dev',config=cfg)
        again,again_diagnostic=generate_vocabulary_pair(seed,split='dev',config=cfg)
        if canonical(records)!=canonical(again) or diagnostic!=again_diagnostic:
            raise ValueError('vocabulary reproduction failure')
        checks=[checked(r,seed) for r in records]
        validate_v2_group(records)
        if len(records)!=2 or checks[0]['semantic_fingerprint']!=checks[1]['semantic_fingerprint']:
            raise ValueError('vocabulary paired normalized-task mismatch')
        if records[0]['targets']['decision']!=records[1]['targets']['decision']:
            raise ValueError('vocabulary paired exact decision target mismatch')
        words={word for mapping in records[1]['metadata']['vocabulary']['maps'].values() for word in mapping.values()}
        if words & train_strings:raise ValueError('reserved DEV namespace string appears in TRAIN lexicon')
        lexical_strings.update(words);lexical_pairs+=1;task_count+=2
        receipts.append({'seed':seed,'arm':'V','diagnostic':diagnostic,'records':checks,
                         'reserved_namespace_strings':sorted(words)})
        bounded()
    if source_hashes()!=code:raise ValueError('source changed during frozen DEV measurement')
    passed=valid>=spec['minimum_valid_goal_parents'] and lexical_pairs==32
    result={'version':'v2-controlled-result-1','verdict':'PASS' if passed else 'FAIL',
            'source_digest':digest(code),'freeze_sha256':file_hash(path),
            'goal_parents_attempted':32,'goal_parents_valid':valid,'goal_parents_excluded':len(exclusions),
            'baseline_goal_blind_correct':before,'baseline_tasks':32,
            'matched_baseline_goal_blind_correct':matched_before,'matched_baseline_tasks':valid,
            'baseline_singleton_exact_recovery':before_exact,'matched_baseline_singleton_exact_recovery':matched_before_exact,
            'baseline_tied_tasks':before_ties,'after_tied_goal_tasks':after_ties,
            'after_goal_blind_correct':after,'after_goal_tasks':valid*3,
            'secondary_goal_blind_correct':secondary,'rendered_goal_only_checks':rendered_checks,
            'after_singleton_exact_recovery':exact_recovery,
            'vocabulary_pairs':lexical_pairs,'vocabulary_reserved_distinct_strings':len(lexical_strings),
            'vocabulary_training_overlap':0,'tasks_verified_in_memory':task_count,
            'oracle_or_label_mismatches':0,'reproduction_mismatches':0,
            'elapsed_seconds':time.monotonic()-started,'exclusions':exclusions,'parent_receipts':receipts,
            'canonical_records_published':0,'training_launched':False,
            'interpretation':'Mechanical contrast and complete-string lexical custody only; no trained-model result, global optimality or universal shortcut claim'}
    raw=(canonical(result)+'\n').encode()
    if len(raw)>1048576:raise ValueError('bounded research report exceeds one MiB')
    immutable_bytes(ROOT/'reports/V2_CONTROLLED_EXPERIMENT.json',raw)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    try:
        result=freeze() if args.freeze else measure()
    except Exception as error:
        evidence={'version':'v2-controlled-failure-1','error_class':type(error).__name__,
                  'error':str(error),'diagnostics':getattr(error,'diagnostics',None),
                  'source_digest':digest(source_hashes()),'publication':False,'training_launched':False}
        raw=(canonical(evidence)+'\n').encode()
        if len(raw)>1048576:raise ValueError('failure evidence exceeds bounded research report') from error
        immutable_bytes(ROOT/'reports'/('V2_EXPERIMENT_FAILURE-'+digest(evidence)[:16]+'.json'),raw)
        raise
    print(canonical({key:value for key,value in result.items() if key not in ('code_hashes','parent_receipts','specification')}))
    return 0 if result.get('verdict','PASS')=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
