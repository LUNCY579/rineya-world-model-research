#!/usr/bin/env python3
"""Small immutable input-only scope interpretation; /root/adversarial_auditor."""
import collections,copy,hashlib,importlib.util,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
def main():
    inp=ROOT/'reviews/input_only/planning-gap-v1.jsonl';pred=ROOT/'reviews/adversarial/planning-gap-v1.input_predictions.jsonl';prep=ROOT/'audits/rgm_planning_gap_adversarial_preparation.py'
    assert sha(inp)=='2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654'
    assert sha(pred)=='273fbdcc8005c034b1f443f5f4ff3309193c899edb461dfd33d69b2f8fda2ec3'
    assert sha(prep)=='2330e383f2a8e1ec608435ee7eacfd647e072e8715bb35d583c12798eaaa1ad3'
    spec=importlib.util.spec_from_file_location('own_preparation_scope',prep);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    source=pathlib.Path(__file__);freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_input_scope-{sha(source)}.py',source.read_bytes())
    rs=[json.loads(s) for s in inp.open()];ps={r['sample_id']:r for r in map(json.loads,pred.open())};units={canon(r['semantic_task']):r for r in rs}.values()
    bysplit=collections.defaultdict(collections.Counter);bad=[]
    for r in units:
        feasible=ps[r['sample_id']]['expected_target']['shortest_plan'] is not None;bysplit[r['split']][str(feasible)]+=1
        if not feasible:bad.append({'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'split':r['split'],'actual_task':r['semantic_task'],'fresh_plan':None})
    assert bysplit['dev_challenge']=={'True':32}
    actual=bad[0];t=actual['actual_task'];adj=collections.defaultdict(list)
    for a,b,c in t['edges']:adj[a].append((b,c))
    paths=[]
    def walk(n,seen,p,cost):
        if n==t['goal']:paths.append((cost,p));return
        for b,c in adj[n]:
            if b not in seen:walk(b,seen|{b},p+[(n,b,c)],cost+c)
    walk(t['start'],{t['start']},[],0);cost,path=min(paths)
    reduced=copy.deepcopy(t);reduced['nodes']=sorted({n for a,b,c in path for n in (a,b)});reduced['edges']=[list(e) for e in path];reduced['plan']=[]
    a,info=P.solve(reduced);assert a['shortest_plan'] is None and info['topology_minimum_moves'] is not None and cost>t['fuel']
    # Independent hand-derived minimal guard probes distinguish the two actual-corpus mutants with no failures.
    guard=[{'name':'noop_two_blocks','task':{'kind':'blocks','on':{'A':'TABLE','B':'TABLE'},'goal_on':{'A':'B'},'plan':[{'block':'A','to':'TABLE'}]},'manual_first_failure':'unchanged_support'},
           {'name':'occupied_destination_three_blocks','task':{'kind':'blocks','on':{'A':'TABLE','B':'TABLE','C':'B'},'goal_on':{},'plan':[{'block':'A','to':'B'}]},'manual_first_failure':'destination_not_clear'}]
    for g in guard:
        g['actual_target'],g['search_evidence']=P.solve(g['task']);assert g['actual_target']['valid'] is False and g['search_evidence']['first_failure']=={'index':0,'reasons':[g['manual_first_failure']]}
    out={'actual_author_execution':'/root/adversarial_auditor','stage':'INPUT_ONLY_SCOPE_FINDINGS','source_sha256':sha(source),'input_sha256':sha(inp),'prediction_sha256':sha(pred),'dataset_decision':None,
         'fresh_feasibility_counts_semantic_units':dict(bysplit),'development_constant_feasible_accuracy':{'correct':32,'total':32},
         'scope_issue':{'severity':'medium','kind':'development_infeasibility_coverage_absent','evidence':'All six infeasible semantic inputs are train; dev has no negative fresh-feasibility examples. This limits development checks of null shortest_plan, not exact-target correctness.',
                        'fix':'In a fresh prospective batch, place both fuel-blocked and fuel-lengthened families in development, retaining family-level custody and preserving this frozen batch.'},
         'all_actual_infeasible_samples':bad,'deletion_reduced_actual_counterexample':{'origin_sample_id':actual['sample_id'],'origin_record_hash':actual['record_hash'],'operation':'Retain one cheapest simple start-to-goal path; delete other edges/nodes and proposed actions. No new edge/cost/resource value introduced.',
             'task':reduced,'expected_target':a,'minimum_required_fuel':cost,'search_evidence':info,'minimality_scope':'Path-only deletion reduction; not claimed globally smallest equivalent full-target task.'},
         'two_new_hand_derived_guard_probes':guard,'guard_probe_count':2,'actual_corpus_guard_mutant_scope':'Allowing no-op or ignoring destination-clear produced no exact-target errors on 96 actual block semantic units. These separate minimal probes prove solver guard handling, not actual corpus coverage.',
         'no_canonical_targets_control_manifest_or_other_verdicts_read':True,'scope_limits':['Guided public preparation and same-model/common-design correlation remain disclosed.','No release novelty or efficacy claim.','Initial input judgments remain frozen; these are later scope findings, not rewrites.']}
    dest=ROOT/'audits/execution/planning-gap-v1-adversarial-input-scope.json';freeze(dest,(json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode());print(canon({'source_sha256':sha(source),'report_sha256':sha(dest),'counts':out['fresh_feasibility_counts_semantic_units'],'guard_probes':2}))
if __name__=='__main__':main()
