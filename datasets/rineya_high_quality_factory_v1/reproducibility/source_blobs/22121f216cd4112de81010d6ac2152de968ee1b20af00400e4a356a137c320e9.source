#!/usr/bin/env python3
"""Independent mechanism, lineage and difficulty evidence for expansion-v1."""
import copy
import hashlib
import json
import pathlib
import re
from collections import Counter,defaultdict
from adversarial_audit import ROOT,EXECUTION,canon,digest,causal_signature,graph_signature,block_signature,signature,solve,require,issue


def load(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def ast_shape(t):
    shape=causal_signature(t)
    return canon({k:shape[k] for k in ('rule','precondition')})


def depth(ast):return 1+max((depth(a) for a in ast.get('args',[])),default=0)


def world_motif(t):
    if t['kind']=='causal':return ast_shape(t)
    if t['kind']=='critic':return graph_signature(t['world'],full=False)
    if t['kind']=='graph':return graph_signature(t,full=False)
    t=copy.deepcopy(t);t['plan']=[]
    return block_signature(t)


def lineage(r):
    if 'source_family_key' in r:return r['source_family_key'],'declared'
    match=re.search(r'-p(\d+)$',r['semantic_family_id'])
    return (f"{r['track_id']}-seed{r['generation_seed']}-p{int(match.group(1)):04d}",'inferred_from_family_index') if match else (None,'unavailable')


def main():
    batch='expansion-v1';paths=sorted((ROOT/'data').glob('*/candidate_renderings/'+batch+'.jsonl'))
    rows=[r for p in paths for r in load(p)];units=list({(r['semantic_family_id'],canon(r['semantic_task'])):r for r in rows}.values())
    releases=sorted((ROOT/'data').glob('*/releases/*/canonical.jsonl'))
    history=[dict(r,_release_path=str(p.relative_to(ROOT))) for p in releases for r in load(p) if r.get('generation_batch')!=batch and batch not in r['sample_id']]
    current_keys={lineage(r)[0] for r in rows};old_keys={lineage(r)[0] for r in history}
    exact_history=defaultdict(list)
    motif_history=defaultdict(list)
    for r in history:
        exact_history[canon(r['semantic_task'])].append(r)
        if r['semantic_task']['kind'] in ('causal','graph','blocks','critic'):motif_history[world_motif(r['semantic_task'])].append(r)
    report={'reviewer_execution_id':EXECUTION,'batch':batch,'records':len(rows),'families':len({r['semantic_family_id'] for r in rows}),
            'logical_variants':len(units),'candidate_file_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'prior_release_file_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in releases},
            'source_family_key_overlap_with_prior_releases':sorted((current_keys&old_keys)-{None}),
            'prior_lineage_evidence_counts':dict(Counter(lineage(r)[1] for r in history)),'tracks':{},'issues':[],
            'limitations':['A numerical world instance is not a new mechanism. AST shapes strip scalar magnitudes/vocabulary; graph motifs strip fuel/costs and names; block motifs retain support/goal relations after canonical entity renaming.',
                           'Prior family keys are inferred from track/seed/index only when no declared key exists; this is explicitly weaker provenance evidence.',
                           'Prior releases and current development are exposed. Structure overlap is reported, not interpreted as sealed evaluation leakage.',
                           'No model training, benchmark retuning, publication or sealed evaluation access occurs.']}
    prior_exact=[];old_dev_current_train=[]
    for r in units:
        t=r['semantic_task'];exact=exact_history.get(canon(t),[])
        if exact:prior_exact.append({'sample_id':r['sample_id'],'prior_samples':[q['sample_id'] for q in exact],'prior_splits':sorted({q['split'] for q in exact})})
        if r['split']=='train':
            old=[q for q in motif_history.get(world_motif(t),[]) if q['split']=='dev_challenge']
            if old:old_dev_current_train.append({'sample_id':r['sample_id'],'prior_dev_samples':sorted({q['sample_id'] for q in old})})
    report['exact_semantic_task_overlap_with_prior_releases']=prior_exact
    report['current_train_with_prior_development_world_motif']=old_dev_current_train
    for track in sorted({r['track_id'] for r in rows}):
        rs=[r for r in units if r['track_id']==track];train=[r for r in rs if r['split']=='train'];dev=[r for r in rs if r['split']=='dev_challenge']
        motifs={world_motif(r['semantic_task']) for r in rs}
        train_motifs={world_motif(r['semantic_task']) for r in train};dev_motifs={world_motif(r['semantic_task']) for r in dev}
        prior_motifs={world_motif(r['semantic_task']) for r in history if r['track_id']==track and r['semantic_task']['kind'] in ('causal','graph','blocks','critic')}
        evidence={'families':len({r['semantic_family_id'] for r in rs}),'logical_variants':len(rs),'distinct_world_or_AST_motifs':len(motifs),
                  'train_motifs':len(train_motifs),'dev_motifs':len(dev_motifs),'train_dev_motif_overlap':len(train_motifs&dev_motifs),
                  'motifs_new_relative_to_prior_releases':len(motifs-prior_motifs),
                  'structural_difficulty_counts':dict(Counter(r['structural_difficulty'] for r in rs)),
                  'difficulty_matches_structural_field':all(r['difficulty']==r['structural_difficulty'] for r in rs)}
        if track=='RCM-001':
            evidence['AST_depth_counts']=dict(Counter(max(depth(r['semantic_task']['rule']),depth(r['semantic_task']['precondition'])) for r in rs))
            evidence['train_AST_shapes']=len({ast_shape(r['semantic_task']) for r in train});evidence['dev_AST_shapes']=len({ast_shape(r['semantic_task']) for r in dev})
            evidence['heldout_composition_shape_overlap']=[r['sample_id'] for r in dev if 'heldout_predicate_composition' in r.get('challenge_axes',[]) and ast_shape(r['semantic_task']) in {ast_shape(q['semantic_task']) for q in train}]
            train_status={r['semantic_task']['events'][0]['status'] for r in train}
            evidence['heldout_first_status_overlap']=[r['sample_id'] for r in dev if r['semantic_task']['events'][0]['status'] in train_status]
        else:
            for kind in sorted({r['semantic_task']['kind'] for r in rs}):
                sub=[r for r in rs if r['semantic_task']['kind']==kind]
                def world(r):return r['semantic_task']['world'] if kind=='critic' else r['semantic_task']
                def names(r):return set(world(r)['on']) if kind=='blocks' else set(world(r)['nodes'])
                tr=[r for r in sub if r['split']=='train'];dv=[r for r in sub if r['split']=='dev_challenge']
                tr_names=set().union(*(names(r) for r in tr));dv_names=set().union(*(names(r) for r in dv))
                tr_sizes={len(names(r)) for r in tr};dv_sizes={len(names(r)) for r in dv}
                part={'train_entity_counts':dict(Counter(len(names(r)) for r in tr)),'dev_entity_counts':dict(Counter(len(names(r)) for r in dv)),
                      'train_dev_entity_name_overlap':sorted(tr_names&dv_names),'train_dev_entity_count_overlap':sorted(tr_sizes&dv_sizes)}
                if kind in ('graph','critic'):
                    part['worlds_with_branching_outdegree_at_least_two']=sum(max(Counter(a for a,b,c in world(r)['edges']).values(),default=0)>=2 for r in sub)
                else:
                    part['partial_goal_worlds']=sum(len(world(r)['goal_on'])<len(world(r)['on']) for r in sub)
                    part['multiple_initial_table_stacks']=sum(sum(v=='TABLE' for v in world(r)['on'].values())>1 for r in sub)
                    if tr_sizes&dv_sizes:
                        report['issues'].append({'code':'block_size_not_disjoint','severity':'low','track':track,'evidence':sorted(tr_sizes&dv_sizes),
                                                 'fix':'Describe block development as held-out names and selected structures; block count alone is not a disjoint size axis.'})
                evidence[kind]=part
        report['tracks'][track]=evidence
    require(not report['source_family_key_overlap_with_prior_releases'],'source family key reused from prior release')
    require(not prior_exact,'exact semantic task reused from prior release')
    require(all(t['difficulty_matches_structural_field'] for t in report['tracks'].values()),'difficulty/structural mismatch')
    require(not report['tracks']['RCM-001']['heldout_composition_shape_overlap'],'claimed held-out AST composition seen in train')
    require(not report['tracks']['RCM-001']['heldout_first_status_overlap'],'claimed held-out first status seen in train')
    # Supplement existing independent target/metamorphic receipts; never replace frozen first judgments.
    receipt_path=ROOT/'reviews'/'adversarial'/(batch+'.jsonl');receipts=load(receipt_path);by_id={r['sample_id']:r for r in rows}
    for receipt in receipts:
        r=by_id[receipt['sample_id']];t=r['semantic_task']
        require(receipt['record_hash']==r['record_hash'],'receipt not bound to exact current record')
        measured={'world_or_AST_motif_sha256':digest(world_motif(t)),'structural_difficulty':r['structural_difficulty'],'source_family_key':r['source_family_key']}
        if t['kind']=='causal':measured.update(event_horizon=len(t['events']),AST_depth=max(depth(t['rule']),depth(t['precondition'])))
        else:
            w=t['world'] if t['kind']=='critic' else t
            measured['entity_count']=len(w['on']) if t['kind']=='blocks' else len(w['nodes'])
        receipt['evidence']['measured_task_structure']=measured
        receipt['issues']=[q for q in receipt['issues'] if q.get('code')!='block_size_not_disjoint']
        if r['split']=='dev_challenge' and t['kind']=='blocks' and report['tracks'][r['track_id']]['blocks']['train_dev_entity_count_overlap']:
            overlap=report['tracks'][r['track_id']]['blocks']['train_dev_entity_count_overlap']
            receipt['issues'].append(issue('block_size_not_disjoint','low',f'Block counts {overlap} occur in both expansion train and development; development block names remain disjoint.','Describe only actual held-out dimensions.'))
    receipt_path.write_text(''.join(canon(r)+'\n' for r in receipts))
    out=ROOT/'audits'/'diversity';out.mkdir(parents=True,exist_ok=True);(out/'expansion-v1-mechanisms-and-lineage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(canon({'records':len(rows),'families':report['families'],'logical_variants':len(units),'prior_exact_overlap':len(prior_exact),
                 'prior_lineage_overlap':len(report['source_family_key_overlap_with_prior_releases']),'old_dev_current_train_motifs':len(old_dev_current_train),'tracks':report['tracks'],'issues':report['issues']}))


if __name__=='__main__':main()
