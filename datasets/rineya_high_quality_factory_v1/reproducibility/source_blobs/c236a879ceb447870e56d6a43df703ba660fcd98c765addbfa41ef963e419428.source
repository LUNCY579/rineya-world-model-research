#!/usr/bin/env python3
"""Separate corrective code audit, using own solvers and independently built exports.
No release/producer/reference/oracle/reviewer verdict files read/imported.
"""
import ast,collections,copy,functools,hashlib,json,pathlib,tempfile
from jsonschema import Draft202012Validator
import adversarial_audit as p0
import extended_adversarial_audit as ext
ROOT=p0.ROOT;EXECUTION=p0.EXECUTION;canon=p0.canon;digest=p0.digest;require=p0.require
V=ROOT/'factory/export_validation.py'
OUT=ROOT/'audits/execution/EXPORT_COMPLETENESS_CORRECTIVE_ADVERSARIAL.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def trajectory(t):
    # Selection concerns the submitted plan, not optimality. Avoid unnecessary shortest-plan search.
    kind=t['kind'];steps=[];valid=True
    if kind=='graph':
        edges=p0.graph_validate(t);state={'node':t['start'],'fuel':t['fuel']}
        for dest in t['plan']:
            cost=edges.get((state['node'],dest));legal=cost is not None and cost<=state['fuel']
            if legal:state={'node':dest,'fuel':state['fuel']-cost}
            steps.append({'legal':legal,'after':state.copy()})
            if not legal:valid=False;break
        goal=valid and state['node']==t['goal']
    else:
        p0.block_validate(t);state=t['on'].copy()
        for act in t['plan']:
            after=p0.block_move(state,act);legal=after is not None
            if legal:state=after
            steps.append({'legal':legal,'after':state.copy()})
            if not legal:valid=False;break
        goal=valid and all(state[a]==b for a,b in t['goal_on'].items())
    return {'steps':steps,'after':state,'valid':valid,'reaches_goal':goal}

@functools.lru_cache(maxsize=None)
def solved(key):
    t=json.loads(key)
    if t['kind'] in ('graph','blocks'):return trajectory(t)
    if t['kind']=='critic':return p0.critic(t)
    return ext.solve(t)

def ownsolve(t):return copy.deepcopy(solved(canon(t)))

def extract(source):
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_specialized']
    require(len(nodes)==1,'requested entry point')
    scope={'json':json,'Counter':collections.Counter,'ROOT':ROOT,'canonical':canon,'digest':digest,'load_jsonl':load,'Draft202012Validator':Draft202012Validator,'solve':ownsolve}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'corrective_validator_AST_only','exec'),scope)
    return scope['validate_specialized'],tree

def candidate(track,batch):
    p=ROOT/f'data/{track}/candidate_renderings/{batch}.jsonl';rs=load(p)
    for r in rs:
        require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'candidate hash')
        t=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);require(canon(t)==canon(r['semantic_task']),'typed prompt match')
    return p,rs

def outputs(rows,suffix):
    out=[]
    for r in rows:
        t=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);verdict=ownsolve(t)
        if suffix in ('memory_updates','skill_demonstrations','belief_updates'):out.append({'input':r['rendered_input'],'target':verdict})
        elif suffix=='process_reward':
            for c in t['candidates']:out.append({'input':r['rendered_input'],'candidate':c,'labels':ownsolve({'kind':'critic','world':t['world'],'candidates':[c]})['evaluations'][0]})
        elif suffix=='trajectories':
            if verdict['valid'] and verdict['reaches_goal']:out.append({'input':r['rendered_input'],'actions':t['plan'],'steps':verdict['steps'],'after':verdict['after']})
    return out

def preference_outputs(rows,alternate=False):
    # Independent representative choice; no exporter source imported or its hash-selection reused.
    grouped=collections.defaultdict(list)
    for r in rows:grouped[canon(r['semantic_task']['world'])].append(r)
    out=[];pair_summary=[]
    for worldkey,rs in sorted(grouped.items()):
        require(len(rs)==2,'actual fixture has two canonical field-order views per world')
        t=rs[0]['semantic_task'];rank={c:i for i,g in enumerate(ownsolve(t)['ranking']) for c in g};pairs=collections.defaultdict(list)
        for a in t['candidates']:
            for b in t['candidates']:
                if rank[a['candidate_id']]<rank[b['candidate_id']]:
                    delta=len(a['plan'])-len(b['plan']);bucket=(delta>0)-(delta<0);pairs[bucket].append((a,b))
        for bucket,ps in sorted(pairs.items()):
            # Lexical ID choice is a TEST fixture only; representative identity is deliberately arbitrary.
            ps.sort(key=lambda pair:canon(pair));chosen,rejected=ps[-1 if alternate else 0]
            for position in (0,1):
                options=[chosen,rejected] if position==0 else [rejected,chosen]
                prompt='Evaluate the two complete submitted candidates under the public graph critic semantics. World and response options JSON:\n'+canon({'world':t['world'],'options':options})
                out.append({'prompt':prompt,'chosen':canon(chosen),'rejected':canon(rejected)})
            pair_summary.append({'world_hash':digest(t['world']),'bucket':bucket,'strict_pair_options':len(ps),'chosen':chosen['candidate_id'],'rejected':rejected['candidate_id']})
    return out,pair_summary

def test_case(check,rows,suffix,items,name,expect_reject):
    with tempfile.TemporaryDirectory(prefix='red-corrective-export-') as tmp:
        p=pathlib.Path(tmp);file=p/name;file.write_text(''.join(canon(x)+'\n' for x in items))
        try:count=check(p,rows)
        except Exception as e:
            require(expect_reject,'Legitimate independent export rejected: '+name+' '+str(e));return {'test':suffix+'/'+name,'expectation':'REJECT','observed':'REJECT','error_type':type(e).__name__,'reason':str(e),'items':len(items)}
        require(not expect_reject,'Corrupt independent export accepted: '+name+' '+canon(count));return {'test':suffix+'/'+name,'expectation':'PASS','observed':'PASS','counts':count,'items':len(items)}

def mutations(items):
    require(len(items)>=2,'nonempty multirow fixture')
    return {'empty':[],'omitted':items[1:],'duplicated':items+[items[0]],'same_count_replaced':[items[0]]+items[:-1]}

def save(report):
    content=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
    if OUT.exists() and OUT.read_text()!=content:
        h=OUT.parent/'history'/f'{OUT.stem}-{sha(OUT)}.json';h.parent.mkdir(parents=True,exist_ok=True)
        if not h.exists():h.write_bytes(OUT.read_bytes())
    OUT.write_text(content)

def main():
    gate_source=V.read_text();gatehash=sha(V);check,tree=extract(gate_source)
    immutable=[ROOT/'reviews/adversarial'/f'memory-gap-v2.{s}' for s in ('input_predictions.jsonl','first_judgments.jsonl','jsonl')]+[ROOT/'audits/execution/memory-gap-v2-export-completeness-adversarial.first.json',ROOT/'audits/execution/memory-gap-v2-export-completeness-adversarial.json']
    bound={str(p.relative_to(ROOT)):sha(p) for p in immutable}
    tests=[];fixtures=[];inputs=[]
    configs=[('REM-001','memory-gap-v2','memory_updates'),('RMM-001','extended-pilot-v3','memory_updates'),('RSM-001','extended-pilot-v2','skill_demonstrations'),('RBM-001','extended-pilot-v2','belief_updates'),('ROM-001','extended-pilot-v2','belief_updates')]
    for track,batch,suffix in configs:
        path,rows=candidate(track,batch);inputs.append(path)
        for split in ('train','dev_challenge'):
            selected=[r for r in rows if r['split']==split];items=outputs(selected,suffix);require(len(items)==len(selected),'own complete row count')
            test=test_case(check,rows,suffix,items,split+'-'+suffix+'.jsonl',False);test['fixture']=track;tests.append(test)
            for mutation,bad in mutations(items).items():
                test=test_case(check,rows,suffix,bad,split+'-'+suffix+'.jsonl',True);test.update(fixture=track,mutation=mutation);tests.append(test)
            fixtures.append({'track':track,'batch':batch,'split':split,'suffix':suffix,'canonical_rows':len(selected),'legitimate_items':len(items)})
        print(canon({'phase':'one_per_row_complete','track':track,'rows':len(rows)}),flush=True)
    # All release-selected current critic candidates, from public candidate files (no releases read).
    rvrows=[]
    for batch in ('pilot-v4','expansion-v2','claim-gap-v2'):
        path,rs=candidate('RVM-001',batch);inputs.append(path);rvrows+=rs
    for split in ('train','dev_challenge'):
        selected=[r for r in rvrows if r['split']==split];items=outputs(selected,'process_reward')
        require(len(items)==sum(len(r['semantic_task']['candidates']) for r in selected),'all canonical candidate pairs')
        tests.append({**test_case(check,rvrows,'process_reward',items,split+'-process_reward.jsonl',False),'fixture':'RVM-001'})
        for mutation,bad in mutations(items).items():tests.append({**test_case(check,rvrows,'process_reward',bad,split+'-process_reward.jsonl',True),'fixture':'RVM-001','mutation':mutation})
        # Same-count substitution of a DIFFERENT valid canonical candidate, retaining correct labels.
        first=items[0];same=[x for x in items if x['input']==first['input'] and canon(x['candidate'])!=canon(first['candidate'])];require(same,'multiple candidates');bad=copy.deepcopy(items);bad[0]=same[0];tests.append({**test_case(check,rvrows,'process_reward',bad,split+'-process_reward.jsonl',True),'fixture':'RVM-001','mutation':'same_count_candidate_replaced_with_other_correctly_labeled_canonical_candidate'})
        fixtures.append({'track':'RVM-001','split':split,'suffix':'process_reward','canonical_rows':len(selected),'canonical_input_candidate_pairs':len(items)})
    print(canon({'phase':'process_complete','canonical_rows':len(rvrows)}),flush=True)
    path,rgrows=candidate('RGM-001','expansion-v3');inputs.append(path)
    for split in ('train','dev_challenge'):
        selected=[r for r in rgrows if r['split']==split];items=outputs(selected,'trajectories');require(0<len(items)<len(selected),'legal goal selection excludes illegal/nongoal submitted plans')
        tests.append({**test_case(check,rgrows,'trajectories',items,split+'-trajectories.jsonl',False),'fixture':'RGM-001'})
        for mutation,bad in mutations(items).items():tests.append({**test_case(check,rgrows,'trajectories',bad,split+'-trajectories.jsonl',True),'fixture':'RGM-001','mutation':mutation})
        nong=next(r for r in selected if not (ownsolve(r['semantic_task'])['valid'] and ownsolve(r['semantic_task'])['reaches_goal']));ver=ownsolve(nong['semantic_task']);invalid={'input':nong['rendered_input'],'actions':nong['semantic_task']['plan'],'steps':ver['steps'],'after':ver['after']};tests.append({**test_case(check,rgrows,'trajectories',items+[invalid],split+'-trajectories.jsonl',True),'fixture':'RGM-001','mutation':'add_truthfully_described_nonselected_submitted_trajectory'})
        fixtures.append({'track':'RGM-001','split':split,'suffix':'trajectories','canonical_rows':len(selected),'own_selected_legal_goal_items':len(items),'excluded_submitted_plans':len(selected)-len(items)})
    print(canon({'phase':'trajectory_complete','canonical_rows':len(rgrows)}),flush=True)
    preference_summary=[]
    for split in ('train','dev_challenge'):
        selected=[r for r in rvrows if r['split']==split];items,pairs=preference_outputs(selected);alternate,_=preference_outputs(selected,True)
        tests.append({**test_case(check,rvrows,'preference',items,split+'-preference.jsonl',False),'fixture':'RVM-001','representative':'independent arbitrary first strict pair per bucket'})
        tests.append({**test_case(check,rvrows,'preference',alternate,split+'-preference.jsonl',False),'fixture':'RVM-001','representative':'independent alternative last strict pair per bucket'})
        for mutation,bad in mutations(items).items():tests.append({**test_case(check,rvrows,'preference',bad,split+'-preference.jsonl',True),'fixture':'RVM-001','mutation':mutation})
        # Remove a whole world/bucket crossed pair; then replace it with another legitimate pair twice.
        require(len(items)>4,'multiple preference bucket groups');tests.append({**test_case(check,rvrows,'preference',items[2:],split+'-preference.jsonl',True),'fixture':'RVM-001','mutation':'omit_complete_crossed_pair'})
        tests.append({**test_case(check,rvrows,'preference',items[2:]+items[2:4],split+'-preference.jsonl',True),'fixture':'RVM-001','mutation':'same_count_missing_bucket_replaced_with_other_valid_crossed_pair'})
        # Same canonical input/world membership and schema cannot excuse reversing strict preference.
        bad=copy.deepcopy(items);bad[0]['chosen'],bad[0]['rejected']=bad[0]['rejected'],bad[0]['chosen'];tests.append({**test_case(check,rvrows,'preference',bad,split+'-preference.jsonl',True),'fixture':'RVM-001','mutation':'strict_pair_reversed'})
        preference_summary.append({'split':split,'canonical_rows':len(selected),'unique_worlds':len({canon(r['semantic_task']['world']) for r in selected}),'legitimate_preference_items':len(items),'world_bucket_groups':len(pairs),'bucket_counts':dict(collections.Counter(p['bucket'] for p in pairs)),'representative_identity_not_enforced':True,'all_strict_pair_options_by_world_bucket':pairs})
    print(canon({'phase':'preference_complete','canonical_rows':len(rvrows)}),flush=True)
    # Direct typed mutations pass Python container equality, yet must fail exact JSON joins.
    _,mr=candidate('REM-001','memory-gap-v2');mitems=outputs([r for r in mr if r['split']=='train'],'memory_updates')
    for kind in ('boolean_value_to_equal_integer','chosen_version_to_equal_float'):
        bad=copy.deepcopy(mitems)
        if kind=='boolean_value_to_equal_integer':
            i,j=next((i,j) for i,it in enumerate(bad) for j,x in enumerate(it['target']['results']) if type(x['value']) is bool);bad[i]['target']['results'][j]['value']=int(bad[i]['target']['results'][j]['value'])
        else:
            i,j=next((i,j) for i,it in enumerate(bad) for j,x in enumerate(it['target']['results']) if type(x['chosen_version']) is int);bad[i]['target']['results'][j]['chosen_version']=float(bad[i]['target']['results'][j]['chosen_version'])
        require(bad==mitems,'typed memory probe must demonstrate Python equality confound')
        tests.append({**test_case(check,mr,'memory_updates',bad,'train-memory_updates.jsonl',True),'fixture':'REM-001','mutation':kind,'typed_corruption':True})
    rtrain=[r for r in rvrows if r['split']=='train'];pi=outputs(rtrain,'process_reward')
    for kind in ('process_valid_boolean_to_equal_integer','candidate_claimed_fuel_to_equal_float'):
        bad=copy.deepcopy(pi)
        if kind=='process_valid_boolean_to_equal_integer':bad[0]['labels']['valid']=int(bad[0]['labels']['valid'])
        else:bad[0]['candidate']['claimed_after']['fuel']=float(bad[0]['candidate']['claimed_after']['fuel'])
        require(bad==pi,'typed process probe must demonstrate Python equality confound')
        tests.append({**test_case(check,rvrows,'process_reward',bad,'train-process_reward.jsonl',True),'fixture':'RVM-001','mutation':kind,'typed_corruption':True})
    ri=outputs([r for r in rgrows if r['split']=='train'],'trajectories');bad=copy.deepcopy(ri);i=next(i for i,it in enumerate(bad) if 'fuel' in it['after']);bad[i]['after']['fuel']=float(bad[i]['after']['fuel']);require(bad==ri,'typed trajectory probe must demonstrate Python equality confound');tests.append({**test_case(check,rgrows,'trajectories',bad,'train-trajectories.jsonl',True),'fixture':'RGM-001','mutation':'trajectory_after_fuel_to_equal_float','typed_corruption':True})
    prefs,_=preference_outputs(rtrain)
    for kind in ('preference_claimed_fuel_to_equal_float','preference_world_edge_cost_to_equal_float'):
        bad=copy.deepcopy(prefs);data=json.loads(bad[0]['prompt'].split('World and response options JSON:\n',1)[1])
        if kind=='preference_claimed_fuel_to_equal_float':
            chosen=json.loads(bad[0]['chosen']);chosen['claimed_after']['fuel']=float(chosen['claimed_after']['fuel']);bad[0]['chosen']=canon(chosen);ci=next(i for i,c in enumerate(data['options']) if c['candidate_id']==chosen['candidate_id']);data['options'][ci]=chosen
        else:data['world']['edges'][0][2]=float(data['world']['edges'][0][2])
        olddata=json.loads(prefs[0]['prompt'].split('World and response options JSON:\n',1)[1]);require(data==olddata,'typed preference probe must demonstrate Python equality confound');bad[0]['prompt']='Evaluate the two complete submitted candidates under the public graph critic semantics. World and response options JSON:\n'+canon(data)
        tests.append({**test_case(check,rvrows,'preference',bad,'train-preference.jsonl',True),'fixture':'RVM-001','mutation':kind,'typed_corruption':True})
    print(canon({'phase':'direct_typed_corruption_complete','rejected':sum(t.get('typed_corruption',False) for t in tests)}),flush=True)
    require(sha(V)==gatehash,'validator changed during independent review');require({str(p.relative_to(ROOT)):sha(p) for p in immutable}==bound,'frozen first/final data or original code finding changed')
    testsfile=ROOT/'tests/test_export_completeness.py'
    report={'reviewer_execution_id':EXECUTION,'reviewer_role':'Adversarial Red-Team','status':'completed','decision':'ACCEPT_CODE','validator_sha256':gatehash,'validator_path':str(V.relative_to(ROOT)),'root_regression_test_source_sha256':sha(testsfile),'root_regression_tests_read_not_run':'These tests import private oracle/reference and read releases. Own independent shadow tests below provide the execution evidence.','script_sha256':sha(pathlib.Path(__file__)),'own_solver_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'audits/adversarial_audit.py',ROOT/'audits/extended_adversarial_audit.py']},'candidate_input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'preserved_initial_code_and_data_artifact_sha256':bound,'fixtures':fixtures,'tests':tests,'legitimate_fixture_tests':sum(t['expectation']=='PASS' for t in tests),'corruption_rejections':sum(t['expectation']=='REJECT' for t in tests),'preferences':preference_summary,'material_blockers':[],'method':'Only requested validate_specialized function AST-executed with own typed solvers. Every fixture constructed independently from prompt-derived candidate inputs; own strict critic ranking, submitted legal-goal trajectory simulation, complete input/candidate coverage. No exporter representative-selection logic, producer/reference/oracle code, other reviewer verdicts or release files read.','scope_limits':['This approves exact-hash generic coverage/type/semantic join code, not new dataset releases or efficacy.','Preference completeness is per exact world/submitted-plan length-sign bucket, as declared; it permits any strict representative pair with crossed positions. All canonical worlds in exercised fixtures have exactly two field-order views.','Submitted trajectory selection tests legality and actual goal attainment; it does not require shortest plans.','Per-file missing-file presence is delegated to the existing release validator; this entry point checks completeness of supplied named files. No release validator or private oracle executed.','Separate execution still shares the underlying model; independence is an execution/custody claim, not independent model-error evidence.']}
    save(report);print(canon({'decision':report['decision'],'validator_sha256':gatehash,'report_sha256':sha(OUT),'legitimate_tests':report['legitimate_fixture_tests'],'corruption_rejections':report['corruption_rejections'],'fixtures':fixtures}),flush=True)

if __name__=='__main__':main()
