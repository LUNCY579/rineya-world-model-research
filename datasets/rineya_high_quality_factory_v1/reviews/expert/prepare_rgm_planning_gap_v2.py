#!/opt/anaconda3/bin/python3
"""Public-only corrective V2 expert preparation, actual /root/expert_reviewer.

Reuses only own frozen V1 prompt-derived transition/search implementation.
New V2 guard/trace/witness/renderer-control code and hand-derived calibration.
Never reads production candidates, manifests, gold, other reviewers or oracles.
"""
import copy, datetime, hashlib, importlib.util, json, pathlib, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'reviews/expert'
ROLE = '/root/expert_reviewer'
V1_PATH = OUT / 'prepare_rgm_planning_gap_v1.py'
V1_SHA = '82ae6d000d5289bf439599579eb6437de5a09468388ead4fb1cef95f9dd3a987'
FREEZE = 'research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json'
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(V1_PATH.read_bytes()) == V1_SHA
spec = importlib.util.spec_from_file_location('own_frozen_v1_solver', V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
eq = v1.eq

GUARDS = ('source_exists', 'source_clear', 'destination_exists_or_table',
          'source_destination_distinct', 'destination_clear_or_table', 'changed_support')
OMISSION = {'source_clear_ignored': 'source_clear',
            'destination_clear_ignored': 'destination_clear_or_table',
            'no_op_allowed': 'changed_support'}
PRIVILEGED_LABELS = ('sample_id', 'semantic_parent_id', 'semantic_family_id',
    'record_hash', 'verified_target', 'source_provenance', 'difficulty', 'split',
    'review_status', 'review_receipt_ids', 'controlled_context', 'primary_stratum')

def guard_truth(on, action):
    """All six predicates, independent of short-circuit reason selection."""
    block, dest = action['block'], action['to']
    occupied = {support for support in on.values() if support != 'TABLE'}
    return {
        'source_exists': block in on,
        'source_clear': block not in occupied,
        'destination_exists_or_table': dest == 'TABLE' or dest in on,
        'source_destination_distinct': block != dest,
        'destination_clear_or_table': dest == 'TABLE' or dest not in occupied,
        'changed_support': block in on and on[block] != dest,
    }

def diagnostic_replay(task, omission=None):
    """Separate all-guard transition diagnostic; never a canonical label source.

    Deliberately omitted guard is not restored by a post-state validator. This
    allows measured mutant states to be invalid under the original dynamics.
    """
    v1.validate(task)
    assert task['kind'] == 'blocks'
    if omission not in (None, *OMISSION):
        raise ValueError('unknown diagnostic omission')
    state = dict(task['on']); states = [dict(state)]; steps = []; vectors = []
    first = None
    for i, action in enumerate(task['plan']):
        guards = guard_truth(state, action)
        vectors.append(guards)
        legal = all(value for key, value in guards.items()
                    if key != OMISSION.get(omission))
        if legal:
            state = dict(state); state[action['block']] = action['to']
        steps.append({'legal': legal, 'after': dict(state)})
        states.append(dict(state))
        if not legal:
            first = i
            break
    valid = first is None
    return {'steps': steps, 'after': state, 'valid': valid,
            'reaches_goal': valid and v1.block_goal(state, task['goal_on'])}, {
                'prefix_states': states, 'guard_vectors': vectors, 'first_error_index': first}

def clearance_witnesses(task, replay_facts=None):
    """All prospective V2 canonical-source-first truthful witnesses."""
    if task['kind'] != 'blocks':
        return []
    strict, facts = diagnostic_replay(task) if replay_facts is None else replay_facts
    initial = task['on']; output = []
    for i, action in enumerate(task['plan'][:len(strict['steps'])]):
        if i < 1 or not strict['steps'][i]['legal']:
            continue
        initial_guards = guard_truth(initial, action)
        if not initial_guards['source_clear']:
            code, selected = 'source_not_clear', action['block']
        elif action['to'] != 'TABLE' and not initial_guards['destination_clear_or_table']:
            code, selected = 'destination_not_clear', action['to']
        else:
            continue
        obstruction = next((b for b, d in initial.items() if d == selected), None)
        if obstruction is None:
            continue
        for j in range(i):
            before, after = facts['prefix_states'][j:j + 2]
            if task['plan'][j]['block'] != obstruction:
                continue
            if selected not in before.values() or selected in after.values():
                continue
            if any(selected in facts['prefix_states'][k].values() for k in range(j + 1, i + 1)):
                continue
            output.append({'action_index': i, 'precondition': code,
                           'enabling_action_index': j, 'obstructing_block': obstruction})
    return output

def check_clearance_witness(task, evidence):
    required = ('action_index', 'precondition', 'enabling_action_index', 'obstructing_block')
    if not isinstance(evidence, dict) or not all(k in evidence for k in required):
        return False
    if type(evidence['action_index']) is not int or type(evidence['enabling_action_index']) is not int:
        return False
    if type(evidence['obstructing_block']) is not str or evidence['precondition'] not in (
            'source_not_clear', 'destination_not_clear') or type(evidence['precondition']) is not str:
        return False
    return any(all(eq(evidence[k], w[k]) for k in required) for w in clearance_witnesses(task))

def noop_fact(task):
    if task['kind'] != 'blocks':
        return None
    strict, facts = diagnostic_replay(task)
    i = facts['first_error_index']
    if i is None or i < 1:
        return None
    a = task['plan'][i]; state = facts['prefix_states'][i]; guards = facts['guard_vectors'][i]
    if all(guards[k] for k in GUARDS if k != 'changed_support') and not guards['changed_support']:
        if state.get(a['block']) == 'TABLE' and a['to'] == 'TABLE':
            return {'action_index': i, 'block': a['block'], 'current_support': 'TABLE'}
    return None

def check_noop_witness(task, evidence):
    if not isinstance(evidence, dict) or type(evidence.get('action_index')) is not int:
        return False
    fact = noop_fact(task)
    return fact is not None and all(k in evidence and eq(evidence[k], fact[k]) for k in fact)

def derive_visible(row):
    """Future stage-one API uses only semantic_task and student prompt."""
    task = row['semantic_task']
    rendered = row['rendered_input']
    if type(rendered) is not str or rendered.count('Task JSON:\n') != 1:
        raise ValueError('missing/ambiguous task delimiter')
    lead, payload = rendered.split('Task JSON:\n', 1)
    parsed = json.loads(payload)
    if not eq(parsed, task):
        raise ValueError('typed visible task/payload mismatch')
    for key in PRIVILEGED_LABELS:
        # Field labels in lead, not ordinary prose or legitimate task entity names.
        if re.search(r'(?<!\w)[\"\']?' + re.escape(key) + r'[\"\']?\s*[:=]', lead):
            raise ValueError('internal prompt lead label: ' + key)
    facts = v1.measurements(parsed)
    if task['kind'] == 'blocks':
        strict, extra = diagnostic_replay(parsed)
        base = {k: facts['target'][k] for k in ('steps', 'after', 'valid', 'reaches_goal')}
        if not eq(strict, base):
            raise AssertionError('own independent six-guard replay discrepancy')
        facts['v2_guard_facts'] = extra
        facts['v2_canonical_clearance_witnesses'] = clearance_witnesses(parsed, (strict, extra))
        facts['v2_isolated_noop_witness'] = noop_fact(parsed)
        facts['diagnostic_mutant_outputs'] = {
            mutant: diagnostic_replay(parsed, mutant)[0] for mutant in OMISSION}
    facts['input_task_sha256'] = sha(v1.canon(parsed).encode())
    return facts

def compare_renderer_control(control, records):
    """V2 generic pair binding; records contain independently solved fixture facts.

    Later original-candidate comparison can construct this interface after its
    predictions freeze. No hidden control/gold reads occur in preparation.
    """
    assert type(control.get('control_id')) is str and control['control_id']
    assert control.get('scope') == 'corpus_variant'
    assert control.get('provenance') == 'supplied_tasks'
    ids = control.get('sample_ids')
    assert type(ids) is list and len(ids) == 2 and all(type(i) is str and i for i in ids)
    assert len(set(ids)) == 2 and all(i in records for i in ids)
    left, right = (records[i] for i in ids)
    assert left['family'] == right['family'] and left['split'] == right['split']
    assert left['style'] == right['style']
    lt, rt = left['task'], right['task']
    fields = [key for key in sorted(set(lt) | set(rt)) if not eq(lt.get(key), rt.get(key))]
    assert fields == sorted(control['changed_task_fields'])
    name = control['name']; evidence = control.get('evidence', {})
    if name == 'fuel_only_counterfactual':
        assert fields == ['fuel'] and left['context'] == 'A0' and right['context'] == 'A1'
    elif name == 'directed_edge_deletion_counterfactual':
        assert fields == ['edges'] and left['context'] == 'B0' and right['context'] == 'B1'
        assert len(lt['edges']) == len(rt['edges']) + 1
        assert all(edge in lt['edges'] for edge in rt['edges'])
    elif name == 'adjacent_action_order_counterfactual':
        assert fields == ['plan'] and left['context'] == 'C0' and right['context'] == 'C1'
        index = evidence.get('swap_index')
        assert type(index) is int and 0 <= index < len(lt['plan']) - 1
        swapped = copy.deepcopy(lt['plan']); swapped[index:index + 2] = reversed(swapped[index:index + 2])
        assert eq(swapped, rt['plan'])
        _, facts = diagnostic_replay(rt); failure = facts['first_error_index']
        assert type(evidence.get('first_failure_index')) is int and failure == evidence['first_failure_index']
        guards = facts['guard_vectors'][failure]
        given = evidence.get('guard_truth')
        assert type(given) is dict and set(given) == set(GUARDS)
        assert all(type(given[k]) is bool for k in given) and eq(given, guards)
        reason = evidence.get('first_failure_reason')
        omitted = 'source_clear_ignored' if reason == 'source_not_clear' else 'destination_clear_ignored'
        assert reason in ('source_not_clear', 'destination_not_clear')
        selected = OMISSION[omitted]
        assert [k for k in GUARDS if not guards[k]] == [selected]
        mutant, _ = diagnostic_replay(rt, omitted)
        assert mutant['steps'][failure]['legal'] and not eq(mutant, diagnostic_replay(rt)[0])
    elif name == 'partial_goal_extension_counterfactual':
        assert fields == ['goal_on'] and left['context'] == 'D0' and right['context'] == 'D1'
        assert len(rt['goal_on']) > len(lt['goal_on'])
        assert all(rt['goal_on'].get(k) == val for k, val in lt['goal_on'].items())
        v1.validate(rt)
    elif name == 'isolated_noop_append_counterfactual':
        assert fields == ['plan'] and left['context'] == 'C3' and right['context'] == 'C4'
        k = evidence.get('prefix_length'); assert type(k) is int and k >= 1
        assert len(lt['plan']) == k and len(rt['plan']) == k + 1
        assert eq(rt['plan'][:k], lt['plan'])
        assert eq(rt['plan'][k], {'block': evidence.get('block'), 'to': 'TABLE'})
        assert evidence.get('current_support') == 'TABLE'
        assert v1.replay(lt)[0]['valid'] and not v1.replay(lt)[0]['reaches_goal']
        assert check_noop_witness(rt, {'action_index': k, 'block': evidence['block'], 'current_support': 'TABLE'})
    elif name == 'same_world_valid_goal_detour':
        assert fields == ['plan']
        a, b = v1.solve(lt)[0], v1.solve(rt)[0]
        assert a['valid'] and a['reaches_goal'] and b['valid'] and b['reaches_goal']
        assert len(rt['plan']) > len(b['shortest_plan'])
    else:
        raise AssertionError('undefined control name')
    return True

def require_both_renderer_declarations(controls, records):
    seen_ids = set(); groups = {}
    for c in controls:
        assert c['control_id'] not in seen_ids
        seen_ids.add(c['control_id']); compare_renderer_control(c, records)
        l, r = (records[i] for i in c['sample_ids'])
        key = (l['family'], c['name'], l['context'], r['context'])
        groups.setdefault(key, []).append(l['style'])
    assert all(sorted(styles) == ['forward', 'reverse'] for styles in groups.values())
    return groups

def preparation():
    started = time.monotonic()
    freeze_bytes = (ROOT / FREEZE).read_bytes(); freeze = json.loads(freeze_bytes)
    public = {}
    for p, h in freeze['public_specification_bindings'].items():
        data = (ROOT / p).read_bytes(); assert sha(data) == h
        public[p] = {'sha256': h, 'bytes': len(data)}
    assert len(public) == 7
    semantics = (ROOT / 'schemas/SEMANTICS.md').read_text()
    graph_contract = semantics.split('## Graph tasks\n')[1].split('\n## Block tasks')[0]
    block_contract = semantics.split('## Block tasks\n')[1].split('\n## Critic tasks')[0]
    checks = []
    def check(name, condition, evidence=None):
        if not condition: raise AssertionError(name)
        checks.append({'name': name, 'passed': True, 'evidence': evidence})
    def reject(name, call):
        error = None
        try: call()
        except (ValueError, AssertionError, TypeError, KeyError) as e: error = type(e).__name__ + ': ' + str(e)
        check(name, error is not None, {'actual_rejection': error})
    inherited = []; transitions = 0
    for f in v1.fixtures():
        result = v1.measurements(f['task']); expected = f['hand_expected']; target = result['target']
        check('own-V1-hand-fixture:' + f['name'], all(eq(target[k], expected[k]) for k in ('shortest_plan', 'valid', 'reaches_goal', 'after')))
        error = result['measurements']['first_error']
        check('own-V1-first-error:' + f['name'], (error['index'] if error else None) == expected['first_error_index'] and (error['reason'] if error else None) == expected['first_error_reason'])
        check('own-V1-required-properties:' + f['name'], set(expected['required_properties']) <= set(result['row_properties']))
        alt = v1.graph_minimum_alternative(f['task']) if f['task']['kind'] == 'graph' else v1.block_minimum_alternative(f['task'])
        check('own-second-search:' + f['name'], eq(alt, target['shortest_plan']))
        inherited.append({'name': f['name'], 'target': target, 'search': result['measurements']['search']})
        if f['task']['kind'] == 'blocks':
            for state in result['measurements']['prefix_states']:
                for b in list(state) + ['unknown']:
                    for d in list(state) + ['TABLE', 'unknown']:
                        a = {'block': b, 'to': d}; l, n, _ = v1.block_step(state, a)
                        ll, nn = v1.block_step_relation(state, a)
                        check('two-transition-agreement:' + str(transitions), l == ll and eq(n, nn))
                        transitions += 1
    for i, malformed in enumerate(v1.malformed()):
        reject('typed-malformed:' + str(i), lambda t=malformed: v1.solve(t))
    source = v1.b({'a': 'TABLE', 'b': 'a', 'c': 'b'}, {'a': 'c'}, [('c', 'TABLE'), ('a', 'c'), ('b', 'TABLE')])
    dest = v1.b({'a': 'TABLE', 'b': 'a', 'c': 'TABLE', 'd': 'c'}, {'a': 'c'}, [('b', 'TABLE'), ('a', 'c')])
    noop = v1.b({'a': 'TABLE', 'b': 'a', 'c': 'TABLE'}, {'a': 'c'}, [('b', 'TABLE'), ('b', 'TABLE')])
    new_hand = [
        ('selective_source', source, {'steps': [{'legal': True, 'after': {'a':'TABLE','b':'a','c':'TABLE'}}, {'legal': False, 'after': {'a':'TABLE','b':'a','c':'TABLE'}}], 'after': {'a':'TABLE','b':'a','c':'TABLE'}, 'valid': False, 'reaches_goal': False, 'shortest_plan': [{'block':'c','to':'TABLE'},{'block':'b','to':'TABLE'},{'block':'a','to':'c'}]}, 'source_clear_ignored', 'source_clear'),
        ('selective_destination', dest, {'steps': [{'legal': True, 'after': {'a':'TABLE','b':'TABLE','c':'TABLE','d':'c'}}, {'legal': False, 'after': {'a':'TABLE','b':'TABLE','c':'TABLE','d':'c'}}], 'after': {'a':'TABLE','b':'TABLE','c':'TABLE','d':'c'}, 'valid': False, 'reaches_goal': False, 'shortest_plan': [{'block':'b','to':'TABLE'},{'block':'d','to':'TABLE'},{'block':'a','to':'c'}]}, 'destination_clear_ignored', 'destination_clear_or_table'),
        ('isolated_TABLE_noop', noop, {'steps': [{'legal': True, 'after': {'a':'TABLE','b':'TABLE','c':'TABLE'}}, {'legal': False, 'after': {'a':'TABLE','b':'TABLE','c':'TABLE'}}], 'after': {'a':'TABLE','b':'TABLE','c':'TABLE'}, 'valid': False, 'reaches_goal': False, 'shortest_plan': [{'block':'b','to':'TABLE'},{'block':'a','to':'c'}]}, 'no_op_allowed', 'changed_support')]
    new_reports = []
    for name, task, hand, mutant, false_guard in new_hand:
        target = v1.solve(task)[0]; strict, facts = diagnostic_replay(task)
        check(name + ':complete_hand_target', eq(target, hand), {'hand_target': hand})
        check(name + ':separate_strict_replay', eq(strict, {k: target[k] for k in strict}))
        check(name + ':only_named_guard_false', [k for k in GUARDS if not facts['guard_vectors'][1][k]] == [false_guard])
        changed, mf = diagnostic_replay(task, mutant)
        check(name + ':actual_omitted_action_executed', changed['steps'][1]['legal'] and not eq(strict, changed), {'strict': strict, 'mutant': changed})
        if name == 'isolated_TABLE_noop':
            for m in ('source_clear_ignored', 'destination_clear_ignored'):
                check(name + ':other_guard_omission_unchanged:' + m, eq(strict, diagnostic_replay(task, m)[0]))
            check(name + ':typed_noop_witness', check_noop_witness(task, {'action_index': 1, 'block':'b','current_support':'TABLE'}))
            check(name + ':bool_noop_index_rejected', not check_noop_witness(task, {'action_index': True, 'block':'b','current_support':'TABLE'}))
        new_reports.append({'name': name, 'task': task, 'hand_target': hand, 'strict': strict, 'guard_facts': facts, 'mutant': changed, 'mutant_facts': mf})
    dual = v1.b({'a':'TABLE','b':'a','c':'TABLE','d':'c'}, {'a':'c'}, [('b','TABLE'),('d','TABLE'),('a','c')])
    good = {'action_index':2,'precondition':'source_not_clear','enabling_action_index':0,'obstructing_block':'b'}
    check('V2 dual-initial-blockage source-first witness', check_clearance_witness(dual, good))
    check('V2 truthful-destination-but-noncanonical witness rejected', not check_clearance_witness(dual, {'action_index':2,'precondition':'destination_not_clear','enabling_action_index':1,'obstructing_block':'d'}))
    for value in ('source','destination','blocked_source','blocked_destination',True,1,None,[],{}):
        e = dict(good); e['precondition'] = value
        check('V2 invalid witness enum:' + repr(value), not check_clearance_witness(dual, e))
    for key, value in [('action_index',True),('enabling_action_index',False),('action_index',99),('enabling_action_index',2),('obstructing_block','d'),('obstructing_block',None)]:
        e = dict(good); e[key] = value
        check('V2 invalid witness field:' + repr((key,value)), not check_clearance_witness(dual, e))
    dest_only = v1.b({'a':'TABLE','c':'TABLE','d':'c'}, {'a':'c'}, [('d','TABLE'),('a','c')])
    check('V2 destination witness when source initially clear', check_clearance_witness(dest_only, {'action_index':1,'precondition':'destination_not_clear','enabling_action_index':0,'obstructing_block':'d'}))
    undone = v1.b({'a':'TABLE','b':'a','c':'TABLE'}, {'a':'c'}, [('b','TABLE'),('b','a'),('b','TABLE'),('a','c')])
    early = {'action_index':3,'precondition':'source_not_clear','enabling_action_index':0,'obstructing_block':'b'}
    late = dict(early); late['enabling_action_index'] = 2
    check('V2 undone early clearance rejected', not check_clearance_witness(undone, early))
    check('V2 actual final continuous clearance accepted', check_clearance_witness(undone, late))
    compounded = v1.b({'a':'TABLE','b':'a','c':'TABLE','d':'TABLE'}, {'a':'c'}, [('d','c'),('b','a')])
    check('nonTABLE unchanged support compound-invalid not isolated', noop_fact(compounded) is None)
    _, cf = diagnostic_replay(compounded)
    check('compound guard vector exposes both failures', [k for k in GUARDS if not cf['guard_vectors'][1][k]] == ['destination_clear_or_table','changed_support'])
    both = v1.b(dual['on'], dual['goal_on'], [('a','c')])
    check('both-blocked action not killed by single clearance omission', all(not diagnostic_replay(both,m)[0]['valid'] for m in ('source_clear_ignored','destination_clear_ignored')))
    # Supplied toy adjacent-order controls, not a proposed production cohort.
    c0 = v1.b({'a':'TABLE','b':'a','c':'TABLE','d':'c'}, {'a':'c'}, [('b','TABLE'),('d','TABLE'),('a','c')])
    c1 = v1.b(c0['on'],c0['goal_on'], [('b','TABLE'),('a','c'),('d','TABLE')])
    _, first_facts = diagnostic_replay(c1)
    records = {}; controls = []
    for style in ('forward','reverse'):
        for ctx, task in [('C0',c0),('C1',c1),('C3',v1.b(noop['on'],noop['goal_on'],[('b','TABLE')])),('C4',noop)]:
            family = 'own-order-toy' if ctx in ('C0','C1') else 'own-noop-toy'
            records[style+'_'+ctx]={'task':task,'context':ctx,'family':family,'split':'train','style':style}
        controls.append({'control_id':'own-order-'+style,'name':'adjacent_action_order_counterfactual','scope':'corpus_variant','provenance':'supplied_tasks','sample_ids':[style+'_C0',style+'_C1'],'changed_task_fields':['plan'],'evidence':{'swap_index':1,'first_failure_index':1,'first_failure_reason':'destination_not_clear','guard_truth':first_facts['guard_vectors'][1]}})
        controls.append({'control_id':'own-noop-'+style,'name':'isolated_noop_append_counterfactual','scope':'corpus_variant','provenance':'supplied_tasks','sample_ids':[style+'_C3',style+'_C4'],'changed_task_fields':['plan'],'evidence':{'prefix_length':1,'block':'b','current_support':'TABLE'}})
    check('two-renderer separate declarations accepted', bool(require_both_renderer_declarations(controls,records)))
    reject('single-renderer control group rejected', lambda: require_both_renderer_declarations(controls[:2],records))
    duplicate=copy.deepcopy(controls);duplicate[-1]['control_id']=duplicate[0]['control_id']
    reject('duplicate global declaration IDs rejected',lambda:require_both_renderer_declarations(duplicate,records))
    mixed=copy.deepcopy(controls);mixed[0]['sample_ids'][1]='reverse_C1'
    reject('mixed renderer pair rejected',lambda:require_both_renderer_declarations(mixed,records))
    four=copy.deepcopy(controls);four[0]['sample_ids']+=['reverse_C0','reverse_C1']
    reject('four-ID combined pair rejected',lambda:require_both_renderer_declarations(four,records))
    typed=copy.deepcopy(controls);typed[0]['evidence']['guard_truth']['source_exists']=1
    reject('integer guard truth rejected',lambda:require_both_renderer_declarations(typed,records))
    index=copy.deepcopy(controls);index[0]['evidence']['swap_index']=True
    reject('boolean swap index rejected',lambda:require_both_renderer_declarations(index,records))
    wrong=copy.deepcopy(controls);wrong[0]['changed_task_fields']=['on']
    reject('false changed-field declaration rejected',lambda:require_both_renderer_declarations(wrong,records))
    rendered_checks=[]
    for name,task,hand,_,_ in new_hand:
        contract=graph_contract if task['kind']=='graph' else block_contract
        before=derive_visible({'semantic_task':task,'rendered_input':contract+'\nTask JSON:\n'+json.dumps(task)})
        reverse=copy.deepcopy(task);reverse['on']=dict(reversed(list(reverse['on'].items())));reverse['goal_on']=dict(reversed(list(reverse['goal_on'].items())))
        prompt='The difficulty may vary; split the reasoning into moves.\n'+contract+'\nTask JSON:\n'+json.dumps(reverse,indent=2)
        after=derive_visible({'semantic_task':task,'rendered_input':prompt})
        check('renderer typed-target/prefix invariance:'+name,eq(before,after))
        rendered_checks.append({'name':name,'task_sha256':before['input_task_sha256'],'target':before['target'],'v2_clearance':before['v2_canonical_clearance_witnesses'],'v2_noop':before['v2_isolated_noop_witness']})
        for key in ('difficulty','split','semantic_family_id','controlled_context'):
            reject('privileged field-label lead:'+name+':'+key,lambda k=key:derive_visible({'semantic_task':task,'rendered_input':k+'=hidden\n'+prompt}))
    graph=v1.g(['s','a','b','c','g'],[['s','a',1],['a','c',1],['c','g',1],['a','b',2],['b','g',1]],4,[])
    check('lex tie beyond shared first destination ignores spare fuel',eq(v1.solve(graph)[0]['shortest_plan'],['a','b','g']))
    truefuel=copy.deepcopy(graph);truefuel['fuel']=True
    reject('typed true versus integer visible task mismatch',lambda:derive_visible({'semantic_task':graph,'rendered_input':graph_contract+'\nTask JSON:\n'+json.dumps(truefuel)}))
    renamed=v1.rename_graph(graph,{'s':'root','a':'mid','b':'zeta','c':'alpha','g':'finish'})
    check('renaming preserves distance changes lexical tied route',len(v1.solve(renamed)[0]['shortest_plan'])==3 and eq(v1.solve(renamed)[0]['shortest_plan'],['mid','alpha','finish']))
    stopped=copy.deepcopy(source);stopped['plan']=source['plan']+source['plan']*2
    check('illegal suffix absorption V2 guards and canonical target',eq(v1.solve(stopped)[0],v1.solve(source)[0]) and eq(diagnostic_replay(stopped),diagnostic_replay(source)))
    monotone=v1.b({'a':'TABLE','b':'TABLE','c':'TABLE'},{'a':'b'},[])
    strengthened=copy.deepcopy(monotone);strengthened['goal_on']['c']='a'
    check('partial-goal extension strictly lengthens fresh minimum',len(v1.solve(monotone)[0]['shortest_plan'])==1 and len(v1.solve(strengthened)[0]['shortest_plan'])==2)
    timeout=False
    try:v1.block_minimum(strengthened,limit=1)
    except v1.UnknownSearch:timeout=True
    check('bounded incomplete search UNKNOWN distinct from null',timeout)
    return {'actual_author':ROLE,'stage':'PUBLIC_ONLY_V2_REVIEW_PREPARATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python_executable':sys.executable,'own_source_sha256':sha(pathlib.Path(__file__).read_bytes()),'own_frozen_solver_sha256':V1_SHA,
        'public_root_freeze':{'path':FREEZE,'sha256':sha(freeze_bytes),'bytes':len(freeze_bytes)},'public_doc_bindings':public,
        'actual_check_entries':len(checks),'all_pass':True,'transition_agreements':transitions,'inherited_hand_fixtures':inherited,
        'new_hand_fixtures':new_reports,'renderer_checks':rendered_checks,'checks':checks,'elapsed_seconds':time.monotonic()-started,
        'guidance':'Public-guided corrective revision of the same 48-family V1 cohort; own frozen V1 solver reused. Shared underlying-model and own prior review knowledge disclosed; no independent-model or fresh fully blind design claim.',
        'not_read':['V2 candidate/gold/control manifest/producer code','V2 formal implementation/tests/private handgold','root V2 gate expectations','other reviewer private verdicts'],
        'scope':'Prospective preparation only. No future data judgment, target approval, release or family-growth claim. New V2 witnesses are never retroactively applied to V1.',
        'planned_next_stage':'Only after root supplies freshly stripped public inputs: derive all 408 typed predictions and initial input judgments, freeze source/execution/expected bytes before canonical gold or private control comparison.'}

if __name__ == '__main__':
    report=preparation()
    path=OUT/'planning-gap-v2.expert_preparation.actual_execution.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'path':str(path),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'checks':report['actual_check_entries'],'all_pass':report['all_pass'],'elapsed_seconds':report['elapsed_seconds']}))
