"""Prospective hand fixtures; no producer/candidate/gold reads.
Actual author: /root/formal_oracle; all first expected paths hand-derived.
"""
from collections import deque
from copy import deepcopy
from itertools import product
import json
import unittest
from oracles import rgm_planning_controls_independent as oracle
from oracles.independent import solve as original_solve

COUNTS = {'graph_histories': 0, 'block_transition_cases': 0, 'block_world_goal_pairs': 0,
          'hand_control_families': 0, 'malformed_cases': 0, 'mutations': 0,
          'partial_goal_extension_pairs': 0, 'held_goal_search_pairs': 0}

def action(b, d): return {'block': b, 'to': d}
def graph(edges=None, fuel=4, plan=None):
    return {'kind': 'graph', 'nodes': ['a', 'b', 'c', 'd', 'g'],
            'edges': edges if edges is not None else [['a','b',2],['b','g',2],['a','c',1],['c','d',1],['d','g',1]],
            'start':'a','goal':'g','fuel':fuel,'plan':['b','g'] if plan is None else plan}

def family_tasks(stratum):
    if stratum == 'A':
        a = graph(); b = deepcopy(a); b['fuel'] = 3
        c = deepcopy(a); c['plan'] = ['b']
        d = deepcopy(a); d['plan'] = ['b','a']
    elif stratum == 'B':
        a = graph([['a','b',1],['b','c',1],['c','g',1],['b','d',1],['d','g',1],['b','b',1]], 8, ['b','c','g'])
        b = deepcopy(a); b['edges'].remove(['b','c',1])
        c = deepcopy(a); c['plan'] = ['b','a']
        d = deepcopy(a); d['plan'] = ['b','b','c','g']
    elif stratum == 'C':
        a = {'kind':'blocks','on':{'a':'b','b':'TABLE','c':'TABLE','d':'c','e':'TABLE'},
             'goal_on':{'a':'b','b':'c','d':'e'},
             'plan':[action('a','TABLE'),action('d','e'),action('b','c'),action('a','b')]}
        b = deepcopy(a); b['plan'][1], b['plan'][2] = b['plan'][2], b['plan'][1]
        c = deepcopy(a); c['plan'] = [action('a','TABLE'),action('d','TABLE'),action('d','e'),action('b','c'),action('a','b')]
        d = deepcopy(a); d['plan'] = a['plan'][:1]
    else:
        a = {'kind':'blocks','on':{'a':'b','b':'c','c':'TABLE','d':'TABLE','e':'TABLE'},
             'goal_on':{'a':'d','b':'TABLE'},'plan':[action('a','d'),action('b','TABLE')]}
        b = deepcopy(a); b['goal_on']['c'] = 'e'
        c = deepcopy(b); c['plan'].append(action('c','e'))
        d = deepcopy(c); d['plan'] = [action('a','TABLE')] + d['plan']
    return [a,b,c,d]

def rows_manifest(stratum):
    tasks = family_tasks(stratum); rows = []; contexts = []
    index = {'A':180,'B':192,'C':204,'D':216}[stratum]
    key = 'hand-family-' + stratum
    for i, task in enumerate(tasks):
        context = stratum + str(i); ids = ['hand-'+context+'-forward','hand-'+context+'-reverse']
        # Factual tags derived by manual fixture assertions below; empty optional
        # lists do not claim unneeded properties on context cousins.
        props = (['later_action_requires_changed_block_clearance','required_temporary_goal_displacement']
                 if context == 'C0' else ['partial_block_goal'] if context == 'D0' else [])
        witnesses = [{'sample_id': identity, 'property': p, 'scope': 'row',
                      'evidence': {'minimum_moves': {'C0':4,'D0':2}[context]}} for identity in ids for p in props]
        contexts.append({'context_id':context,'sample_ids':ids,'row_properties':props,'row_witnesses':witnesses})
        for j, identity in enumerate(ids):
            rendered_task = task if not j else dict(reversed(list(task.items())))
            if j and task['kind'] == 'graph':
                rendered_task['nodes'] = list(reversed(task['nodes']))
                rendered_task['edges'] = list(reversed(task['edges']))
            rows.append({'sample_id':identity,'semantic_family_id':'lineage-'+stratum,
                         'semantic_parent_id':'parent-'+context,'track_id':'RGM-001','split':'train',
                         'generator_version':'rgm-planning-gap-1.0.0','source_provenance':{'source_family_key':key,'source_index':index},
                         'semantic_task':deepcopy(task),'skill_tags':props,
                         'rendered_input':'Full public contract applies. Task JSON:\n'+json.dumps(rendered_task,indent=2 if j else None),
                         'verified_target':{'do_not_read':True}})
    definitions = {'A':[('fuel_only_counterfactual',0,1,['fuel'])],
                   'B':[('directed_edge_deletion_counterfactual',0,1,['edges']),('same_world_valid_goal_detour',0,3,['plan'])],
                   'C':[('adjacent_action_order_counterfactual',0,1,['plan']),('same_world_valid_goal_detour',0,2,['plan'])],
                   'D':[('partial_goal_extension_counterfactual',0,1,['goal_on']),('same_world_valid_goal_detour',2,3,['plan'])]}
    controls = [{'control_id':'shared-physical-pair' if stratum == 'D' else 'hand-control-'+str(k),
                 'name':name,'scope':'corpus_variant','sample_ids':[contexts[a]['sample_ids'][0],contexts[b]['sample_ids'][0]],
                 'changed_task_fields':fields,'provenance':'supplied_tasks'} for k,(name,a,b,fields) in enumerate(definitions[stratum])]
    manifest = {'version':'hand-1','track_id':'RGM-001','batch_name':'planning-gap-v1','semantics_sha256':oracle.SEMANTICS_SHA256,
                'families':[{'source_family_key':key,'source_index':index,'split':'train','primary_stratum':stratum,
                             'base_sample_id':contexts[0]['sample_ids'][0],'contexts':contexts,'controls':controls}]}
    return rows, manifest

def stack_step(on, move):
    """Separately authored list-of-stacks transition for exhaustive comparison."""
    chains = []
    roots = sorted(b for b,s in on.items() if s == 'TABLE')
    for root in roots:
        chain = [root]
        while True:
            children = [b for b,s in on.items() if s == chain[-1]]
            if not children: break
            chain.append(children[0])
        chains.append(chain)
    block,dest = move['block'],move['to']
    source = next((x for x in chains if x[-1] == block),None)
    target = next((x for x in chains if x[-1] == dest),None)
    if source is None or dest == block or dest == on[block] or (dest != 'TABLE' and target is None): return None
    source.pop()
    if dest == 'TABLE': chains.append([block])
    else: target.append(block)
    result = {}
    for chain in chains:
        for i,b in enumerate(chain): result[b] = 'TABLE' if i == 0 else chain[i-1]
    return result

def valid_worlds(names):
    for supports in product(['TABLE']+names,repeat=len(names)):
        on = dict(zip(names,supports))
        if any(b == s for b,s in on.items()): continue
        non_table = [s for s in supports if s != 'TABLE']
        if len(set(non_table)) != len(non_table): continue
        valid = True
        for b in names:
            seen = set()
            while b != 'TABLE':
                if b in seen: valid = False; break
                seen.add(b); b = on[b]
        if valid: yield on

def exhaustive_stack_path(initial, goal, held=None):
    queue = deque([(initial,[])])
    seen = {json.dumps(initial,sort_keys=True)}
    while queue:
        state,path = queue.popleft()
        if all(state[b] == d for b,d in goal.items()): return path
        for b in sorted(state):
            for d in sorted(list(state)+['TABLE']):
                move = action(b,d); successor = stack_step(state,move)
                if successor is not None and all(successor[b] == d for b,d in (held or {}).items()):
                    identity = json.dumps(successor,sort_keys=True)
                    if identity not in seen:
                        seen.add(identity); queue.append((successor,path+[move]))
    return None

class RGMProspectiveTests(unittest.TestCase):
    def test_hand_derived_resource_objectives_and_boundary(self):
        a = graph(); result = oracle.analyze(a)
        self.assertEqual(result['target']['shortest_plan'],['b','g'])
        self.assertEqual(result['search']['minimum_moves'],2)
        self.assertEqual(result['search']['cheapest_fuel'],3)
        self.assertEqual(result['search']['topology_distance'],2)
        a['fuel']=3; result=oracle.analyze(a)
        self.assertEqual(result['target']['shortest_plan'],['c','d','g'])
        self.assertTrue(result['properties']['fuel_lengthens_fresh_minimum_moves'])
        a['fuel']=2; result=oracle.analyze(a)
        self.assertIsNone(result['target']['shortest_plan'])
        self.assertTrue(result['properties']['fuel_blocks_topologically_reachable_goal'])
        self.assertTrue(result['search']['complete'])
        self.assertLessEqual(result['search']['states'],len(a['nodes'])*(a['fuel']+1))
        # A different boundary fixture uses a two-edge cheapest route.
        a=graph([['a','b',1],['b','g',1],['a','c',1],['c','d',1],['d','g',1]],2)
        b=deepcopy(a); b['fuel']=1
        c=deepcopy(a); c['plan']=['b']
        d=deepcopy(a); d['plan']=['b','a']
        self.assertEqual(oracle._context_rules('A',[a,b,c,d],[oracle.analyze(x) for x in [a,b,c,d]]),'unreachable')

    def test_hand_derived_all_four_contexts(self):
        for stratum in 'ABCD':
            tasks=family_tasks(stratum)
            oracle._context_rules(stratum,tasks,[oracle.analyze(x) for x in tasks])
            for task in tasks:
                self.assertEqual(oracle._json(oracle.solve(task)),oracle._json(original_solve(task)))
            COUNTS['hand_control_families']+=1
        c=oracle.analyze(family_tasks('C')[0])
        self.assertEqual(c['target']['shortest_plan'],[action('a','TABLE'),action('d','e'),action('b','c'),action('a','b')])
        self.assertTrue(c['properties']['required_temporary_goal_displacement'])
        self.assertFalse(c['search']['held_goal_reachable'])
        self.assertTrue(c['held_search']['complete'])
        self.assertIn({'action_index':2,'precondition':'source_not_clear','relevant_block':'b','changed_by_action_index':0},c['clearance'])
        d=family_tasks('D')
        self.assertEqual([oracle.analyze(x)['search']['minimum_moves'] for x in d],[2,3,3,3])

    def test_graph_lex_full_sequence_and_multiplicity(self):
        task=family_tasks('B')[0]
        report=oracle.analyze(task)
        self.assertEqual(report['target']['shortest_plan'],['b','c','g'])
        self.assertEqual(report['search']['minimum_route_count'],2)
        rename={'a':'z','b':'y','c':'x','d':'a','g':'g'}
        changed={'kind':'graph','nodes':[rename[x] for x in reversed(task['nodes'])],
                 'edges':[[rename[a],rename[b],cost] for a,b,cost in reversed(task['edges'])],
                 'start':rename[task['start']],'goal':rename[task['goal']],'fuel':task['fuel'],
                 'plan':[rename[x] for x in task['plan']]}
        rr=oracle.analyze(changed)
        self.assertEqual(rr['target']['shortest_plan'],['y','a','g'])
        self.assertEqual(rr['search']['minimum_route_count'],2)
        self.assertEqual(rr['search']['minimum_moves'],report['search']['minimum_moves'])
        self.assertEqual(oracle._signature(oracle._json(task)),oracle._signature(oracle._json(changed)))

    def test_first_illegal_suffix_absorption_and_identity(self):
        for task in [family_tasks('B')[2],family_tasks('C')[1]]:
            report=oracle.analyze(task); i=report['first_error']
            self.assertEqual(report['target']['steps'][i]['after'],report['target']['steps'][i-1]['after'])
            self.assertEqual(len(report['target']['steps']),i+1)
            altered=deepcopy(task)
            altered['plan']=task['plan'][:i+1]+(['no-node'] if task['kind']=='graph' else [action('missing','TABLE')])
            self.assertEqual(oracle.solve(task),oracle.solve(altered))
            COUNTS['mutations']+=1
        task=graph(plan=[]); task['goal']=task['start']
        self.assertEqual(oracle.solve(task)['shortest_plan'],[])
        self.assertTrue(oracle.solve(task)['reaches_goal'])
        task=family_tasks('C')[0]; task['goal_on']={}; task['plan']=[]
        self.assertEqual(oracle.solve(task)['shortest_plan'],[])
        self.assertEqual(oracle.solve(task)['after'],task['on'])

    def test_type_and_world_malformed_fail_closed(self):
        bad=[]
        for field,value in [('fuel',True),('fuel',-1),('fuel',31),('plan',[1]),('nodes',['a','a'])]:
            x=graph(); x[field]=value; bad.append(x)
        for edge in [['a','b',True],['a','b',0],['a','b',1.0],['a','x',1],['a','b'],['a','b',None]]:
            x=graph(); x['edges']=[edge]; bad.append(x)
        x=graph(); x['edges']*=2; bad.append(x)
        for on,goal in [({'TABLE':'TABLE'},{}),({'a':'b','b':'a'},{}),({'a':'TABLE','b':'a','c':'a'},{}),
                        ({'a':'TABLE'},{'a':'a'}),({'a':'TABLE','b':'TABLE'},{'a':'b','b':'a'}),
                        ({'a':'TABLE','b':'TABLE','c':'TABLE'},{'a':'c','b':'c'})]:
            bad.append({'kind':'blocks','on':on,'goal_on':goal,'plan':[]})
        for task in bad:
            with self.assertRaises(oracle.OracleError): oracle.analyze(task)
            self.assertTrue(oracle.check({'semantic_task':task}))
            COUNTS['malformed_cases']+=1
        # Costs larger than the budget remain valid public input, just unusable.
        self.assertIsNone(oracle.solve(graph([['a','g',100]],30))['shortest_plan'])

    def test_exhaustive_small_graphs_vs_direct_path_enumeration(self):
        names=['a','b','g']; edge_pairs=[(a,b) for a in names for b in names if a != b]
        for mask in range(1 << len(edge_pairs)):
            edges=[[a,b,1+(i%2)] for i,(a,b) in enumerate(edge_pairs) if mask & (1 << i)]
            for fuel in range(5):
                task={'kind':'graph','nodes':names,'edges':edges,'start':'a','goal':'g','fuel':fuel,'plan':[]}
                paths=[('a',fuel,())]; wins=[]
                for length in range(fuel+1):
                    wins=[path for node,remaining,path in paths if node=='g']
                    if wins: break
                    paths=[(b,remaining-cost,path+(b,)) for node,remaining,path in paths
                           for a,b,cost in edges if a==node and cost<=remaining]
                rr=oracle.analyze(task)
                self.assertEqual(rr['target']['shortest_plan'],list(min(wins)) if wins else None)
                self.assertEqual(rr['search']['minimum_route_count'],len(wins))
                self.assertEqual(oracle._json(rr['target']),oracle._json(original_solve(task)))
                COUNTS['graph_histories']+=1

    def test_independent_stack_transition_and_small_goal_bfs(self):
        names=['a','b','c']; worlds=list(valid_worlds(names))
        self.assertEqual(len(worlds),13)
        for on in worlds:
            for b in names+['missing']:
                for dest in ['TABLE']+names+['missing']:
                    move=action(b,dest); expected=stack_step(on,move)
                    actual,_=oracle._block_step(frozenset(on.items()),move)
                    self.assertEqual(None if actual is None else dict(actual),expected)
                    COUNTS['block_transition_cases']+=1
            for goal in worlds:
                task={'kind':'blocks','on':on,'goal_on':goal,'plan':[]}
                expected=exhaustive_stack_path(on,goal)
                self.assertEqual(oracle.solve(task)['shortest_plan'],expected)
                self.assertEqual(oracle._json(oracle.solve(task)),oracle._json(original_solve(task)))
                COUNTS['block_world_goal_pairs']+=1

    def test_partial_goal_extension_and_held_relation_search_exhaustive(self):
        worlds=list(valid_worlds(['a','b','c']))
        for on in worlds:
            for full in worlds:
                for size in (0,1,2):
                    partial=dict(list(full.items())[:size])
                    task={'kind':'blocks','on':on,'goal_on':partial,'plan':[]}
                    rr=oracle.analyze(task)
                    stronger=deepcopy(task); stronger['goal_on']=full
                    self.assertLessEqual(rr['search']['minimum_moves'], oracle.analyze(stronger)['search']['minimum_moves'])
                    COUNTS['partial_goal_extension_pairs']+=1
                task={'kind':'blocks','on':on,'goal_on':full,'plan':[]}
                held={b:d for b,d in full.items() if on[b] == d}
                independently_constrained=exhaustive_stack_path(on,full,held)
                rr=oracle.analyze(task)
                self.assertEqual(rr['search']['held_goal_reachable'], independently_constrained is not None)
                self.assertEqual(rr['properties']['required_temporary_goal_displacement'], bool(held) and independently_constrained is None)
                COUNTS['held_goal_search_pairs']+=1

    def test_name_key_order_equivariance_and_displacement(self):
        task=family_tasks('C')[0]
        rename={b:'z'+str(i) for i,b in enumerate(reversed(sorted(task['on'])))}
        rename['TABLE']='TABLE'
        changed={'kind':'blocks','on':{rename[b]:rename[s] for b,s in reversed(list(task['on'].items()))},
                 'goal_on':{rename[b]:rename[s] for b,s in reversed(list(task['goal_on'].items()))},
                 'plan':[action(rename[x['block']],rename[x['to']]) for x in task['plan']]}
        aa,bb=oracle.analyze(task),oracle.analyze(changed)
        self.assertEqual(aa['search']['minimum_moves'],bb['search']['minimum_moves'])
        self.assertEqual(aa['properties'],bb['properties'])
        self.assertEqual(aa['target']['valid'],bb['target']['valid'])
        self.assertEqual(oracle._signature(oracle._json(task)),oracle._signature(oracle._json(changed)))
        self.assertEqual({rename[b]:rename[s] for b,s in aa['target']['after'].items()},bb['target']['after'])

    def test_manifest_positive_family_binding_and_ignored_stored_target(self):
        for stratum in 'ABCD':
            rows,manifest=rows_manifest(stratum)
            result=oracle.check_batch(rows,manifest,[])
            self.assertEqual(result['counts']['complete_families'],1,result['errors'])
            self.assertFalse(any(e.startswith('hand-family-') for e in result['errors']),result['errors'])
            for row in rows:
                self.assertEqual(oracle.check(row),[])
                row['verified_target']={'fabricated':'different'}
            self.assertEqual(oracle.check_batch(rows,manifest,[]),result)
            self.assertEqual(sum(result['conditional_histograms'].values()),8)

    def test_published_property_list_encoding_and_integer_version(self):
        rows,manifest=rows_manifest('C')
        baseline=oracle.check_batch(rows,manifest,[])
        manifest['version']=1
        self.assertEqual(oracle.check_batch(rows,manifest,[]),baseline)
        context=manifest['families'][0]['contexts'][0]
        context['row_properties']={sid:list(context['row_properties']) for sid in context['sample_ids']}
        result=oracle.check_batch(rows,manifest,[])
        self.assertTrue(any('public encoding supplement' in x for x in result['errors']))
        manifest['version']=True
        self.assertTrue(any('version' in x for x in oracle.check_batch(rows,manifest,[])['errors']))

    def test_manifest_mutations_actual_fields_and_scopes(self):
        rows,manifest=rows_manifest('C')
        cases=[]
        x=deepcopy(manifest); x['families'][0]['controls'][0]['changed_task_fields']=['on']; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['controls'][0]['scope']='audit_probe'; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['controls'][0]['name']='unpublished'; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['controls'][0]['sample_ids'][1]='foreign'; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['contexts'][0]['row_witnesses'][0]['evidence']['minimum_moves']=True; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['contexts'][0]['row_witnesses'].pop(); cases.append(x)
        x=deepcopy(manifest); x['families'][0]['contexts'][0]['sample_ids'][1]=x['families'][0]['contexts'][1]['sample_ids'][0]; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['controls']=[]; cases.append(x)
        x=deepcopy(manifest); x['families'][0]['contexts'].pop(); cases.append(x)
        for x in cases:
            rr=oracle.check_batch(rows,x,[])
            self.assertTrue(any(e.startswith('hand-family-') for e in rr['errors']),rr['errors'])
            COUNTS['mutations']+=1
        rr=deepcopy(rows); rr[0]['split']='dev_challenge'
        self.assertEqual(oracle.check_batch(rr,manifest,[])['counts']['complete_families'],0)
        rr=deepcopy(rows); rr[0]['source_provenance']['source_index']=True
        self.assertEqual(oracle.check_batch(rr,manifest,[])['counts']['complete_families'],0)
        rr=deepcopy(rows); rr[1]['rendered_input']=rr[0]['rendered_input']
        self.assertEqual(oracle.check_batch(rr,manifest,[])['counts']['complete_families'],0)
        rr=deepcopy(rows); rr[0]['rendered_input']+=' '+rr[0]['sample_id']
        self.assertEqual(oracle.check_batch(rr,manifest,[])['counts']['complete_families'],0)
        rr=deepcopy(rows); rr[0]['skill_tags'].append('fuel_blocks_topologically_reachable_goal')
        self.assertTrue(any('false row property' in x for x in oracle.check(rr[0])))

    def test_witness_scalar_and_input_locator_truth_fail_closed(self):
        task=family_tasks('C')[0]; result=oracle.analyze(task)
        prop='later_action_requires_changed_block_clearance'
        for witness in ({'fresh_minimum_moves':4},{'input_fact':'on.a'},
                        {'input_facts':{'on':{'a':'b'}}},
                        {'action_index':2,'precondition':'source_not_clear','relevant_block':'b','clearing_action_index':0}):
            oracle._witness(witness,prop,task,result)
        for witness in ({'fresh_minimum_moves':True},{'invented_measurement':900},
                        {'note':'a descriptive claim alone is not a witness'},
                        {'input_fact':'on.missing'}, {'input_facts':{'on':{'a':'TABLE'}}},
                        {'action_index':2,'changed_by_action_index':1}):
            with self.assertRaises(oracle.OracleError): oracle._witness(witness,prop,task,result)
            COUNTS['mutations']+=1
        failed=family_tasks('B')[2]; rr=oracle.analyze(failed)
        oracle._witness({'action_index':1},'legal_prefix_then_first_failure',failed,rr)
        with self.assertRaises(oracle.OracleError):
            oracle._witness({'action_index':0},'legal_prefix_then_first_failure',failed,rr)
        prior=graph(); prior['edges'][0][2]=True
        rows,manifest=rows_manifest('B')
        self.assertTrue(any('edge cost' in x for x in oracle.check_batch(rows,manifest,[prior])['errors']))

    def test_novelty_inventory_and_isomorphism_collisions(self):
        rows,manifest=rows_manifest('B')
        missing=oracle.check_batch(rows,manifest)
        self.assertTrue(any('UNKNOWN' in x for x in missing['errors']))
        rr=oracle.check_batch(rows,manifest,[deepcopy(rows[0])])
        self.assertTrue(any('lineage collision' in x or 'isomorphism collision' in x for x in rr['errors']))
        prior=deepcopy(rows[0]['semantic_task']); prior['nodes']=list(reversed(prior['nodes'])); prior['edges'].reverse()
        self.assertTrue(any('isomorphism collision' in x for x in oracle.check_batch(rows,manifest,[prior])['errors']))
        changed=deepcopy(prior); changed['edges'][0][2]+=1
        self.assertNotEqual(oracle._signature(oracle._json(prior)),oracle._signature(oracle._json(changed)))
        self.assertEqual(oracle._signature(oracle._json(prior),False,False),oracle._signature(oracle._json(changed),False,False))

    def test_rendered_payload_type_identity_and_real_order_crossing(self):
        task = graph(fuel=1)
        row = {'semantic_task': task, 'rendered_input': 'Contract example {"kind":"graph"}. Task JSON: ' + json.dumps(task)}
        self.assertEqual(oracle.check(row), [])
        altered = deepcopy(task); altered['fuel'] = True
        row['rendered_input'] = 'Task JSON: ' + json.dumps(altered)
        self.assertTrue(oracle.check(row))
        rows, manifest = rows_manifest('B')
        rows[1]['rendered_input'] = 'Task JSON: ' + json.dumps(rows[0]['semantic_task'], indent=4)
        result = oracle.check_batch(rows, manifest, [])
        self.assertTrue(any('ordering identical' in x for x in result['errors']))

    def test_non_goal_prefix_need_not_copy_the_base_submitted_route(self):
        tasks = family_tasks('A')
        tasks[2]['plan'] = ['c']
        self.assertEqual(oracle._context_rules('A', tasks, [oracle.analyze(x) for x in tasks]), 'lengthened')
        tasks = family_tasks('C')
        tasks[3]['plan'] = [action('d','e')]
        oracle._context_rules('C', tasks, [oracle.analyze(x) for x in tasks])

    def test_executed_probes_and_diagnostic_reproduction(self):
        for stratum in 'ABCD':
            for task in family_tasks(stratum):
                probes = oracle._executed_probes(task)
                self.assertTrue(all(x['invoked'] and x['completed'] and x['passed'] for x in probes))
                self.assertTrue(all(len(x['input_sha256']) == len(x['transformed_input_sha256']) == 64 for x in probes))
        fixtures = [{'size':5,'plan_length':1,'fuel':1,'edge_count':1,'tokens':['"a"'],
                     'valid':True,'reaches_goal':False,'fresh_feasible':True},
                    {'size':6,'plan_length':2,'fuel':2,'edge_count':2,'tokens':['"b"'],
                     'valid':False,'reaches_goal':True,'fresh_feasible':False}]
        self.assertEqual(oracle._shortcut_diagnostics(fixtures), oracle._shortcut_diagnostics(list(reversed(fixtures))))
        import os, subprocess, sys
        code = 'import json; from oracles.rgm_planning_controls_independent import _shortcut_diagnostics; print(json.dumps(_shortcut_diagnostics(' + repr(fixtures) + '),sort_keys=True))'
        outputs = [subprocess.check_output([sys.executable,'-c',code], env=dict(os.environ,PYTHONHASHSEED=seed,PYTHONDONTWRITEBYTECODE='1')) for seed in ('1','987')]
        self.assertEqual(outputs[0], outputs[1])

    def test_controls_reject_hidden_changes_and_invalid_goals(self):
        a,b,c,d=family_tasks('A')
        b['goal']='d'
        with self.assertRaises(oracle.OracleError): oracle._pair('fuel_only_counterfactual',a,b)
        a,b,c,d=family_tasks('B'); b['edges'].remove(['b','d',1])
        with self.assertRaises(oracle.OracleError): oracle._pair('directed_edge_deletion_counterfactual',a,b)
        a,b,c,d=family_tasks('C'); b['plan']=list(reversed(a['plan']))
        with self.assertRaises(oracle.OracleError): oracle._pair('adjacent_action_order_counterfactual',a,b)
        a,b,c,d=family_tasks('D'); b['goal_on']['d']='a'
        with self.assertRaises(oracle.OracleError): oracle._pair('partial_goal_extension_counterfactual',a,b)

if __name__ == '__main__': unittest.main()
