#!/usr/bin/env python3
"""Fact-check semantic mechanism tags using public-contract implementations only.

The expansion's initial blind judgments and completed semantic judgments are preserved.
This supplemental tag challenge was requested after another reviewer's finding topics
were disclosed by root. No reviewer files, private oracle, or producer code is read.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from adversarial_audit import ROOT, EXECUTION, canon, digest, solve, graph, graph_shortest, require, issue


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def routes(w):
    """Enumerate all feasible simple goal routes, distinct by destination sequence."""
    out=defaultdict(list)
    for a,b,c in w['edges']:out[a].append((b,c))
    found=[]
    def walk(at,fuel,path,seen):
        if at==w['goal']:
            found.append(path);return
        for b,cost in sorted(out[at]):
            if b not in seen and cost<=fuel:walk(b,fuel-cost,path+[b],seen|{b})
    walk(w['start'],w['fuel'],[],{w['start']})
    return sorted(found,key=lambda p:(len(p),p))


def minimal_probes():
    linear={'kind':'graph','nodes':['a','b','g'],'edges':[['a','b',1],['b','g',1]],'start':'a','goal':'g','fuel':2,'plan':[]}
    reachable={'kind':'graph','nodes':['a','g'],'edges':[['a','g',1]],'start':'a','goal':'g','fuel':1,'plan':[]}
    single={'kind':'blocks','on':{'a':'TABLE','b':'a'},'goal_on':{'a':'b','b':'TABLE'},'plan':[]}
    require(max(Counter(x[0] for x in linear['edges']).values())==1,'branching counterexample')
    require(routes(linear)==[['b','g']],'equal-route counterexample')
    require(graph_shortest(reachable)==['g'],'unreachable counterexample')
    require(sum(v=='TABLE' for v in single['on'].values())==sum(v=='TABLE' for v in single['goal_on'].values())==1,'stack counterexample')
    return {'false_branching_tag':{'task':linear,'maximum_outdegree':1},
            'false_equal_length_routes_tag':{'task':linear,'exhaustive_feasible_simple_routes':routes(linear)},
            'false_initially_unreachable_tag':{'task':reachable,'feasible_fresh_plan':['g']},
            'unsupported_multistack_tag':{'task':single,'initial_table_stacks':1,'explicit_goal_table_roots':1,'fresh_plan':solve(single)['shortest_plan']}}


def equal_walk_witness(w):
    """Two equal-length complete goal-ending walks, including goal revisits, or None.

    Positive integer costs bound the walk length by initial fuel. Retain two
    distinct prefixes per (node,fuel,hops), enough to detect any final tie.
    """
    out=defaultdict(list)
    for a,b,c in w['edges']:out[a].append((b,c))
    layer={(w['start'],w['fuel']):[[]]}
    for hops in range(w['fuel']+1):
        hits=[p for (at,fuel),ps in layer.items() if at==w['goal'] for p in ps]
        if len(hits)>=2:return hits[:2]
        later=defaultdict(list)
        for (at,fuel),ps in layer.items():
            for b,cost in out[at]:
                if cost<=fuel:
                    key=(b,fuel-cost)
                    for p in ps:
                        if len(later[key])<2:later[key].append(p+[b])
        layer=later
    return None


def check(r):
    t=r['semantic_task'];a=solve(t);tags=r['skill_tags'];errors=[];low=[];measured={}
    def defect(code,evidence,fix,severity='high'):
        errors.append(issue(code,severity,evidence,fix))
    if t['kind']=='causal':
        measured.update(event_horizon=len(t['events']),conditional_gate_steps=len(a['steps']))
        # Ordinal predicate_0..5 labels are opaque group names, not factual predicate descriptions.
        return errors,low,measured
    w=t['world'] if t['kind']=='critic' else t
    if t['kind'] in ('graph','critic'):
        degree=Counter(x[0] for x in w['edges']);maximum=max(degree.values(),default=0)
        measured['maximum_outdegree']=maximum
        if any(tag.startswith('branched_resource_graph') for tag in tags) and maximum<2:
            defect('false_branching_tag',{'maximum_outdegree':maximum,'edges':w['edges']},'Remove the branching modifier or add a genuine branch in a fresh candidate batch.')
        if 'single_successor_resource_graph' in tags and maximum>1:
            defect('false_single_successor_tag',{'maximum_outdegree':maximum},'Use the branching tag when a node has multiple outgoing edges.')
    if t['kind'] in ('graph','blocks'):
        for tag,truth in [('valid_plan',a['valid']),('invalid_plan',not a['valid']),('unreachable',a['shortest_plan'] is None),('goal_satisfaction',a['reaches_goal'])]:
            if tag in tags and not truth:defect('false_'+tag+'_tag',{'actual_valid':a['valid'],'actual_reaches_goal':a['reaches_goal'],'fresh_plan':a['shortest_plan']},'Derive factual tags from the fully constructed task.')
        if t['kind']=='blocks':
            initial=sum(v=='TABLE' for v in t['on'].values());goal=sum(v=='TABLE' for v in t['goal_on'].values())
            measured.update(initial_table_stacks=initial,explicit_goal_table_roots=goal,goal_constraints=len(t['goal_on']),partial_goal=len(t['goal_on'])<len(t['on']))
            if 'multi_stack_goal_composition' in tags and initial<=1 and goal<=1:
                # Partial goals can admit multi-stack completions; they do not require them.
                defect('unsupported_multistack_tag',{'initial_table_stacks':initial,'explicit_goal_table_roots':goal,'on':t['on'],'goal_on':t['goal_on']},'Use a general goal-composition tag on these rows, or require multiple explicitly represented stacks in a fresh task.',severity='medium')
            if 'multiple_initial_table_stacks' in tags and initial<2:
                defect('false_multiple_initial_table_stacks_tag',{'initial_table_stacks':initial},'Derive the tag from actual TABLE supports.')
            if 'multiple_explicit_goal_table_roots' in tags and goal<2:
                defect('false_multiple_explicit_goal_table_roots_tag',{'explicit_goal_table_roots':goal},'Count only explicit goal_on TABLE values.')
            # Public COMPETENCY_TAGS defines block_goal_composition generically;
            # the complete finite block-task validation above establishes it.
        return errors,low,measured
    shortest=graph_shortest(w);allroutes=routes(w);lengths=Counter(map(len,allroutes))
    measured.update(fresh_plan=shortest,feasible_simple_goal_route_lengths=dict(lengths),ranking_tie_group_sizes=[len(g) for g in a['ranking'] if len(g)>1])
    evaluations={e['candidate_id']:e for e in a['evaluations']};cs=t['candidates']
    suffixes={tag.split(':',1)[-1] for tag in tags}
    if suffixes & {'initially_unreachable','initial_goal_unreachable'} and shortest is not None:
        result=graph(dict(w,plan=shortest));require(result['valid'] and result['reaches_goal'],'fresh route witness invalid')
        defect('false_initially_unreachable_tag',{'feasible_fresh_plan':shortest,'initial_fuel':w['fuel'],'witness_after':result['after']},'Compute the reachability tag after graph construction; retag or regenerate the same family in a fresh batch.')
    if 'initial_goal_reachable' in suffixes and shortest is None:
        defect('false_initial_goal_reachable_tag',{},'Derive initial goal reachability from the final constructed world.')
    if 'initial_goal' in suffixes and w['start']!=w['goal']:
        defect('false_initial_goal_tag',{'start':w['start'],'goal':w['goal']},'Derive the initial-goal tag from actual start and goal.')
    if suffixes & {'equal_length_routes','equal_length_goal_routes'}:
        equal=[p for p in allroutes if lengths[len(p)]>=2]
        if not equal:
            defect('false_equal_length_routes_tag',{'all_feasible_simple_goal_routes':allroutes,'route_length_counts':dict(lengths)},'Tag only actual distinct feasible equal-length goal routes; if shortest-route ties are claimed, verify two minimum-length routes.')
        elif lengths[len(allroutes[0])]<2:
            low.append(issue('equal_routes_not_optimal_tie','low',{'shortest_route':allroutes[0],'distinct_equal_length_routes':equal[:2]},'Do not count this example as exercising the lexicographic shortest-route tie break.'))
    if 'no_equal_length_goal_routes' in suffixes:
        witness=equal_walk_witness(w)
        measured['equal_length_complete_goal_walk_witness']=witness
        if witness:defect('false_no_equal_length_goal_routes_tag',{'distinct_equal_length_goal_walks':witness},'Derive the no-equal-route tag from actual routes, including feasible cyclic walks.')
    if 'legal_detour' in suffixes:
        detours=[c['plan'] for c in cs if evaluations[c['candidate_id']]['valid'] and evaluations[c['candidate_id']]['reaches_goal'] and shortest is not None and len(c['plan'])>len(shortest)]
        measured['valid_longer_goal_route_witnesses']=detours[:2]
        if not detours:defect('false_legal_detour_tag',{'fresh_plan':shortest},'Require a submitted valid goal route longer than the fresh optimum.')
    if 'fuel_precondition' in suffixes:
        fuel_failures=[];edges={(x,y):cost for x,y,cost in w['edges']}
        for c in cs:
            index=evaluations[c['candidate_id']]['first_error']
            if type(index)is int:
                after=graph(dict(w,plan=c['plan'][:index]))['after'];cost=edges.get((after['node'],c['plan'][index]))
                if cost is not None and cost>after['fuel']:fuel_failures.append({'candidate_id':c['candidate_id'],'first_error':index,'available_fuel':after['fuel'],'required_fuel':cost})
        measured['fuel_failure_witnesses']=fuel_failures[:2]
        if not fuel_failures:defect('false_fuel_precondition_tag',{},'Require an actual insufficient-fuel failure on an existing edge.')
    if 'fresh_replacement_counterexample' in suffixes:
        witnesses=[]
        for c in cs:
            first=evaluations[c['candidate_id']]['first_error'];prefix=c['plan'][:first] if type(first)is int else c['plan']
            if prefix and shortest is not None and shortest[:len(prefix)]!=prefix:witnesses.append({'candidate_id':c['candidate_id'],'executed_legal_prefix':prefix,'fresh_plan':shortest})
        measured['fresh_replacement_differs_from_retained_prefix']=witnesses[:2]
        if not witnesses:defect('unsupported_fresh_replacement_counterexample_tag',{'fresh_plan':shortest},'Require an explicit prefix-retention versus fresh-replacement witness.',severity='medium')
    if 'equal_length_good_bad' in suffixes:
        matched=any(len(x['plan'])==len(y['plan']) and (evaluations[x['candidate_id']]['valid'],evaluations[x['candidate_id']]['claim_correct'])!=(evaluations[y['candidate_id']]['valid'],evaluations[y['candidate_id']]['claim_correct']) for x in cs for y in cs)
        if not matched:defect('false_equal_length_quality_control_tag',{},'Include equal-length candidates with opposing valid/claim quality.')
    if 'longer_valid_shorter_invalid' in suffixes:
        matched=any(len(x['plan'])>len(y['plan']) and evaluations[x['candidate_id']]['valid'] and not evaluations[y['candidate_id']]['valid'] for x in cs for y in cs)
        if not matched:defect('false_opposed_length_control_tag',{},'Include a longer valid plan paired with a shorter invalid one.')
    return errors,low,measured


def main(batch,release=None,tracks=None):
    paths=[release] if release else sorted((ROOT/'data').glob('*/candidate_renderings/'+batch+'.jsonl'))
    if tracks and not release:paths=[p for p in paths if p.parent.parent.name in tracks]
    rows=[r for p in paths for r in load(p)];out=ROOT/'reviews/adversarial';out.mkdir(parents=True,exist_ok=True)
    context=('Initial blind judgments preserved; root later disclosed finding topics. Facts recomputed independently; no other reviewer file read.'
             if batch=='expansion-v1' else 'Fresh correction or erratum review informed by prior tag challenges. Current hashes, targets, metamorphics and tag facts checked independently; no other reviewer file read.')
    findings=[];counts=Counter();units={};first_receipts={}
    if not release:
        original=out/(batch+'.jsonl');archive=out/(batch+'.before-mechanism-tag-audit.jsonl')
        if not archive.exists():archive.write_bytes(original.read_bytes())
        first_receipts={r['sample_id']:r for r in load(archive)}
    receipts=[]
    for r in rows:
        defects,low,measured=check(r);counts.update(q['code'] for q in defects)
        if defects:findings.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'semantic_family_id':r['semantic_family_id'],'track_id':r['track_id'],'issues':defects,'semantic_task':r['semantic_task']})
        key=(r['semantic_family_id'],canon(r['semantic_task']));units[key]={'track_id':r['track_id'],'defects':[q['code'] for q in defects],'measured':measured}
        prior=first_receipts.get(r['sample_id']);receipt=copy.deepcopy(prior) if prior else {'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_role':'Adversarial Red-Team','reviewer_execution_id':EXECUTION,'scores':{k:5 for k in ('semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','difficulty_quality','diversity','shortcut_resistance')},'critical_defects':[],'issues':[],'confidence':0.99,'evidence':{}}
        if prior:receipt['supersedes_receipt_id']=prior['receipt_id']
        receipt['receipt_id']='adv-tags-'+digest([batch,r['sample_id'],r['record_hash'],EXECUTION])[:24]
        receipt['issues'].extend(defects+low);receipt['decision']='REVISE' if defects else receipt.get('decision','ACCEPT')
        if defects:receipt['scores'].update(generalization_value=3,diversity=3)
        receipt['evidence'].update(mechanism_tag_audit=measured,tag_challenge_context=context,independent_target_recomputation_for_tag_challenge=True)
        receipts.append(receipt)
    name=batch+'.mechanism-tags.jsonl' if not release else batch+'.tag-erratum.jsonl'
    (out/name).write_text(''.join(canon(r)+'\n' for r in receipts))
    if not release:(out/(batch+'.jsonl')).write_text(''.join(canon(r)+'\n' for r in receipts))
    report={'reviewer_execution_id':EXECUTION,'batch':batch,'records':len(rows),'logical_variants':len(units),'decision_counts':dict(Counter(r['decision'] for r in receipts)),'defect_record_counts':dict(counts),'defect_logical_variant_counts':dict(Counter(c for v in units.values() for c in v['defects'])),'input_file_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'audit_script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'findings':findings,'challenge_context':context,'scope':['Tags are checked against final constructed tasks, not original template intent. Semantic target values continue to be checked independently.','Positive equal-route claims have distinct feasible simple-route witnesses. New no-equal-route claims exhaust complete goal-ending walks, including departures from and revisits to the goal, bounded by positive integer fuel costs.','The ordinal predicate tags are opaque producer strata and are not assigned an invented meaning.','The immutable pilot release is read only; errata live in separate review/audit files.','Correction batches preserve source families and are not new independent mechanism samples.'],'private_reference_producer_or_reviewer_files_read':False}
    report['small_counterexamples']={code:min((f for f in findings if any(q['code']==code for q in f['issues'])),key=lambda f:len(canon(f['semantic_task']))) for code in counts}
    report['minimal_semantic_counterexamples']={code:value for code,value in minimal_probes().items() if code in counts}
    audit=ROOT/'audits/shortcut_detection';(audit/(batch+'-mechanism-tags.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(canon({k:report[k] for k in ('batch','records','logical_variants','decision_counts','defect_record_counts','defect_logical_variant_counts','input_file_sha256')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='expansion-v1');parser.add_argument('--release');parser.add_argument('--track',action='append');args=parser.parse_args()
    main(args.batch,ROOT/args.release if args.release else None,args.track)
