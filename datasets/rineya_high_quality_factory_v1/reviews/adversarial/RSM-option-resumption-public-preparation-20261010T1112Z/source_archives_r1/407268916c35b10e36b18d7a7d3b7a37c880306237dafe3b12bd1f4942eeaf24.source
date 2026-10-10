"""Independent typed public option-resumption parser and explicit event machine."""
import copy
import json

ENVELOPE = {'kind','version','initial_state','options','entry','mode','history','checkpoint','event_budget'}
REGISTER = {'x','y'}


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(typed_equal(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a == b


def exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('Exact object shape')


def integer(value):
    if type(value) is not int:
        raise ValueError('Integer excludes boolean/coercion')


def name(value):
    if type(value) is not str or not value:
        raise ValueError('Nonempty literal name')


def validate(task):
    exact(task, ENVELOPE)
    if task['kind'] != 'option_resume' or type(task['version']) is not int or task['version'] != 1:
        raise ValueError('Kind/version')
    exact(task['initial_state'], REGISTER)
    for value in task['initial_state'].values(): integer(value)
    if type(task['mode']) is not str or task['mode'] not in {'fresh','resume','restart'}:
        raise ValueError('Mode')
    integer(task['event_budget'])
    if task['event_budget'] < 0:
        raise ValueError('Negative budget')
    options = task['options']
    if type(options) is not dict or not options:
        raise ValueError('Options map')
    for option, definition in options.items():
        name(option); exact(definition, {'params','initiation','success','body'})
        params = definition['params']
        if type(params) is not list:
            raise ValueError('Parameters list')
        for parameter in params: name(parameter)
        if len(params) != len(set(params)):
            raise ValueError('Duplicate parameter')
    calls = {option: set() for option in options}

    def expression(value, parameters):
        if type(value) is not dict or len(value) != 1:
            raise ValueError('Expression shape')
        if 'const' in value: integer(value['const'])
        elif 'reg' in value:
            if type(value['reg']) is not str or value['reg'] not in REGISTER: raise ValueError('Register')
        elif 'param' in value:
            if type(value['param']) is not str or value['param'] not in parameters: raise ValueError('Parameter reference')
        else: raise ValueError('Expression form')

    def predicate(value, parameters):
        exact(value, {'all'})
        if type(value['all']) is not list: raise ValueError('Predicate atoms list')
        for atom in value['all']:
            exact(atom, {'left','cmp','right'})
            if type(atom['cmp']) is not str or atom['cmp'] not in {'lt','eq','ge'}: raise ValueError('Comparison')
            expression(atom['left'], parameters); expression(atom['right'], parameters)

    def arguments(value, target, parameters):
        if type(target) is not str or target not in options: raise ValueError('Unknown option')
        exact(value, options[target]['params'])
        for item in value.values(): expression(item, parameters)

    def instructions(body, owner):
        if type(body) is not list: raise ValueError('Instruction list')
        for instruction in body:
            if type(instruction) is not dict or type(instruction.get('op')) is not str: raise ValueError('Instruction object')
            op = instruction['op']; parameters = options[owner]['params']
            if op in {'add','scale'}:
                exact(instruction, {'op','register','value'})
                if type(instruction['register']) is not str or instruction['register'] not in REGISTER: raise ValueError('Effect register')
                expression(instruction['value'], parameters)
            elif op == 'swap': exact(instruction, {'op'})
            elif op == 'call':
                exact(instruction, {'op','option','args'})
                arguments(instruction['args'], instruction['option'], parameters)
                calls[owner].add(instruction['option'])
            elif op == 'if':
                exact(instruction, {'op','test','then','else'})
                predicate(instruction['test'], parameters)
                instructions(instruction['then'], owner); instructions(instruction['else'], owner)
            else: raise ValueError('Unknown instruction')
    for option, definition in options.items():
        predicate(definition['initiation'], definition['params'])
        predicate(definition['success'], definition['params'])
        instructions(definition['body'], option)
    exact(task['entry'], {'option','args'})
    arguments(task['entry']['args'], task['entry']['option'], [])
    visiting = set(); finished = set()
    def acyclic(option):
        if option in visiting: raise ValueError('Recursive call graph')
        if option in finished: return
        visiting.add(option)
        for child in calls[option]: acyclic(child)
        visiting.remove(option); finished.add(option)
    for option in options: acyclic(option)
    return task


def strict_json(raw):
    def object_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out: raise ValueError('Duplicate JSON object key')
            out[key] = value
        return out
    def nonfinite(value): raise ValueError('Nonfinite JSON number')
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=nonfinite)


def expression(value, state, bindings):
    if 'const' in value: return value['const']
    if 'reg' in value: return state[value['reg']]
    return bindings[value['param']]


def predicate(value, state, bindings):
    for atom in value['all']:
        left = expression(atom['left'], state, bindings)
        right = expression(atom['right'], state, bindings)
        result = {'lt': lambda: left < right, 'eq': lambda: left == right, 'ge': lambda: left >= right}[atom['cmp']]()
        if not result: return False
    return True


def initial(state):
    return {'state': copy.deepcopy(state), 'frames': [], 'next_instance_id': 0,
            'root_pending': True, 'phase': 'running'}


def body_at(task, frame):
    body = task['options'][frame['option']]['body']
    path = frame['body_path']
    for index in range(0, len(path), 2):
        body = body[path[index]][path[index+1]]
    return body


def normalize(task, machine):
    while machine['frames']:
        top = machine['frames'][-1]
        if top['frame_kind'] != 'branch' or top['next_index'] < len(body_at(task, top)): break
        machine['frames'].pop()
    return machine


def event_step(task, machine):
    normalize(task, machine)
    if machine['phase'] != 'running': raise ValueError('Terminal event request')
    def enter(option, args, caller_bindings, site):
        bound = {key: expression(value, machine['state'], caller_bindings) for key,value in args.items()}
        instance = machine['next_instance_id']; machine['next_instance_id'] += 1
        passed = predicate(task['options'][option]['initiation'], machine['state'], bound)
        if passed:
            machine['frames'].append({'frame_kind':'option','instance_id':instance,'option':option,
                                      'bindings':bound,'body_path':[],'next_index':0})
        else: machine['phase'] = 'initiation_failed'
        return {'event':'enter','instance_id':instance,'option':option,'bindings':copy.deepcopy(bound),
                'call_site':copy.deepcopy(site),'initiation_ok':passed}
    if machine['root_pending']:
        machine['root_pending'] = False
        return enter(task['entry']['option'], task['entry']['args'], {}, None)
    frame = machine['frames'][-1]; body = body_at(task, frame); index = frame['next_index']
    if index == len(body):
        passed = predicate(task['options'][frame['option']]['success'], machine['state'], frame['bindings'])
        machine['frames'].pop(); normalize(task, machine)
        if not passed: machine['phase'] = 'success_failed'
        elif not machine['frames']: machine['phase'] = 'completed'
        return {'event':'return','instance_id':frame['instance_id'],'option':frame['option'],'success_ok':passed}
    instruction = body[index]; op = instruction['op']; frame['next_index'] += 1
    location = {'instance_id':frame['instance_id'],'option':frame['option'],
                'body_path':copy.deepcopy(frame['body_path']),'index':index}
    if op == 'call':
        return enter(instruction['option'], instruction['args'], frame['bindings'],
                     {'caller_instance':frame['instance_id'],'body_path':copy.deepcopy(frame['body_path']),'index':index})
    if op == 'if':
        passed = predicate(instruction['test'], machine['state'], frame['bindings'])
        chosen = 'then' if passed else 'else'
        machine['frames'].append({'frame_kind':'branch','instance_id':frame['instance_id'],'option':frame['option'],
                                  'bindings':copy.deepcopy(frame['bindings']),'body_path':frame['body_path']+[index,chosen],'next_index':0})
        normalize(task, machine)
        return {'event':'branch', **location, 'condition':passed}
    event = {'event':'effect', **location, 'op':op}
    if op == 'swap':
        machine['state']['x'],machine['state']['y'] = machine['state']['y'],machine['state']['x']
    else:
        value = expression(instruction['value'], machine['state'], frame['bindings'])
        register = instruction['register']
        machine['state'][register] = machine['state'][register]+value if op == 'add' else machine['state'][register]*value
        event.update(register=register, value=value)
    event['state'] = copy.deepcopy(machine['state']); normalize(task, machine)
    return event


def invalid(error):
    return {'status':'invalid','error':error,'checkpoint':None,'new_events':[],
            'previous_effects':None,'new_effects':0,'next_event':None}


def solve(task):
    try: validate(task)
    except (ValueError,TypeError,KeyError): return invalid('invalid_task')
    machine = initial(task['initial_state'])
    if type(task['history']) is not list: return invalid('invalid_history')
    for supplied in task['history']:
        if machine['phase'] != 'running': return invalid('invalid_history')
        expected = event_step(task, machine)
        if not typed_equal(supplied, expected): return invalid('invalid_history')
    normalize(task, machine)
    if task['mode'] == 'fresh':
        if task['history'] or task['checkpoint'] is not None: return invalid('invalid_checkpoint')
    elif not typed_equal(task['checkpoint'], machine): return invalid('invalid_checkpoint')
    previous = sum(event['event']=='effect' for event in task['history'])
    if task['mode'] == 'restart': machine = initial(machine['state'])
    events = []
    while machine['phase'] == 'running' and len(events) < task['event_budget']:
        events.append(event_step(task, machine))
    normalize(task, machine)
    status = 'paused' if machine['phase'] == 'running' else machine['phase']
    preview = event_step(task, copy.deepcopy(machine)) if status == 'paused' else None
    return {'status':status,'error':None,'checkpoint':copy.deepcopy(machine),'new_events':events,
            'previous_effects':previous,'new_effects':sum(e['event']=='effect' for e in events),'next_event':preview}
