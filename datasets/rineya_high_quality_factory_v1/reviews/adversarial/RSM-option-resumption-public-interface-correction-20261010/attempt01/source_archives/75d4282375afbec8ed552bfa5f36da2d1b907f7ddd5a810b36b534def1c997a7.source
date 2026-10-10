"""Fresh r2 scoped pair-ID typing correction; original r1 stays immutable.
Only control_id is nonempty; pair IDs are two distinct strings without a minLength.
These preparation helpers do not approve a corpus.
"""
import copy
import hashlib
import itertools
import json
from pathlib import PurePosixPath
import re
from frame_solver_r1 import typed_equal, expression, predicate, event_step, normalize

NAMES = ('uninterrupted_resume_suffix','resume_restart_replay','budget_pause_after_effect',
         'second_checkpoint_suffix','positive_bound_argument_counterfactual','initiation_argument_counterfactual',
         'success_contract_counterfactual','active_checkpoint_tamper_rejection','object_field_order_invariance')
PAIRS = dict(zip(NAMES[:-1], [('F','R'),('R','T'),('R','P'),('P','U'),('F','A'),('F','I'),('F','S'),('R','X')]))
CONTEXTS = {'F','R','T','P','U','A','I','S','X'}


def require(value, message):
    if not value: raise ValueError(message)


def differences(a, b, path=()):
    if type(a) is not type(b):return [path]
    if type(a) is dict:
        output=[]
        for key in sorted(set(a)|set(b)):
            output.extend([path+(key,)] if key not in a or key not in b else differences(a[key],b[key],path+(key,)))
        return output
    if type(a) is list:
        if len(a)!=len(b):return [path]
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,path+(i,))]
    return [] if a==b else [path]


def reverse_objects(value):
    if type(value) is dict:return {k:reverse_objects(value[k]) for k in reversed(value)}
    if type(value) is list:return [reverse_objects(v) for v in value]
    return value


def guard_values(pred, state, bindings):
    return [predicate({'all':[atom]},state,bindings) for atom in pred['all']]


def instruction_at(task, event):
    body=task['options'][event['option']]['body']
    for j in range(0,len(event['body_path']),2):body=body[event['body_path'][j]][event['body_path'][j+1]]
    return body[event['index']]


def check_controls(tasks, outputs):
    F,R,T,P,U,A,I,S,X=(tasks[k] for k in 'FRTPUAISX')
    f,r,t,p,u,a,i,s,x=(outputs[k] for k in 'FRTPUAISX')
    require(R['history'] and R['checkpoint']['phase']=='running','Actual nonterminal R prefix')
    require(typed_equal(R['history'],f['new_events'][:len(R['history'])]),'R exact F prefix')
    require(typed_equal(r['new_events'],f['new_events'][len(R['history']):]),'Uninterrupted suffix')
    require(f['status']==r['status']=='completed' and typed_equal(f['checkpoint']['state'],r['checkpoint']['state']),'Uninterrupted terminal')
    require(r['previous_effects']>0 and r['new_events'],'Committed effect and future work')
    require(all(typed_equal(F[k],R[k]) for k in ('initial_state','options','entry')),'Same original program/root')
    require(differences(R,T)==[('mode',)],'Restart mode only')
    require(t['new_events'] and t['new_events'][0]['event']=='enter' and t['new_events'][0]['initiation_ok'] is True,'Restart root admitted')
    require(not typed_equal(r['new_events'],t['new_events']) and (r['status']!=t['status'] or not typed_equal(r['checkpoint']['state'],t['checkpoint']['state'])),'Measured restart difference')
    bindings={};prior=copy.deepcopy(F['initial_state']);replayed=[]
    for event in R['history']:
        if event['event']=='enter':bindings[event['instance_id']]=event['bindings']
        if event['event']!='effect':continue
        instruction=instruction_at(F,event);local=bindings[event['instance_id']]
        def primitive(state):
            state=copy.deepcopy(state)
            if instruction['op']=='swap':state['x'],state['y']=state['y'],state['x']
            else:
                reg=instruction['register'];operand=expression(instruction['value'],state,local)
                state[reg]=state[reg]+operand if instruction['op']=='add' else state[reg]*operand
            return state
        once=primitive(prior);twice=primitive(once)
        same_locator=lambda candidate: candidate['event']=='effect' and all(candidate[k]==event[k] for k in ('option','body_path','index','op'))
        if not typed_equal(once,twice) and any(same_locator(candidate) for candidate in t['new_events']):
            replayed.append({'option':event['option'],'body_path':event['body_path'],'index':event['index'],
                             'original_pre_state':prior,'after_once':once,'after_twice':twice})
        prior=copy.deepcopy(event['state'])
    require(replayed,'Actual repeated non-idempotent effect')
    require(differences(R,P)==[('event_budget',)] and P['event_budget']>0,'Pause budget only')
    require(p['status']=='paused' and p['new_effects']>=1 and p['next_event'] is not None,'Second cut commits effect/pending work')
    require(typed_equal(p['new_events'],r['new_events'][:len(p['new_events'])]),'P actual continuation prefix')
    require(typed_equal(U['history'],R['history']+p['new_events']) and typed_equal(U['checkpoint'],p['checkpoint']),'Second history/checkpoint custody')
    require(U['event_budget']==48 and u['status']=='completed' and typed_equal(u['new_events'],r['new_events'][len(p['new_events']):]),'Second suffix')
    require(typed_equal(u['checkpoint'],f['checkpoint']),'Second terminal checkpoint')
    for label, task in [('A',A),('I',I)]:
        require(all(typed_equal(F[k],task[k]) for k in F if k!='entry'),'Root argument only '+label)
        require(F['entry']['option']==task['entry']['option'],'Entry option unchanged')
        require(sum(not typed_equal(F['entry']['args'][key],task['entry']['args'][key]) for key in F['entry']['args'])==1,'Exactly one changed root expression')
    require(a['status']=='completed','Positive argument both complete')
    later_difference=any(fe['event']==ae['event']=='effect' and fe.get('value')!=ae.get('value') for fe,ae in zip(f['new_events'],a['new_events']))
    later_difference |= any(fe['event']==ae['event']=='branch' and fe['condition']!=ae['condition'] for fe,ae in zip(f['new_events'],a['new_events']))
    later_difference |= any(fe['event']==ae['event']=='return' and fe['success_ok']!=ae['success_ok'] for fe,ae in zip(f['new_events'],a['new_events']))
    require(later_difference,'Actual later parameter-dependent use, beyond entry binding')
    require(i['status']=='initiation_failed' and i['new_effects']==0,'Initiation failure no effects')
    root=F['entry']['option'];fguard=guard_values(F['options'][root]['initiation'],F['initial_state'],f['new_events'][0]['bindings'])
    iguard=guard_values(I['options'][root]['initiation'],I['initial_state'],i['new_events'][0]['bindings'])
    require(all(fguard) and sum(not x for x in iguard)==1,'One decisive initiation atom, all others true')
    delta=differences(F,S)
    require(len(delta)==1 and delta[0][:4]==('options',root,'success','all') and delta[0][-1]=='const','One root-success literal')
    require(s['status']=='success_failed' and typed_equal([e for e in f['new_events'] if e['event']=='effect'],[e for e in s['new_events'] if e['event']=='effect']),'Success fails without rollback/effect change')
    sg=guard_values(S['options'][root]['success'],s['checkpoint']['state'],s['new_events'][0]['bindings'])
    require(sum(not x for x in sg)==1,'One decisive exit atom')
    delta=differences(R,X)
    require(len(delta)==1 and delta[0][0]=='checkpoint','Exactly one checkpoint scalar')
    leaf=delta[0]
    require(leaf[1]=='state' or (leaf[1]=='frames' and (leaf[-1]=='next_index' or 'bindings' in leaf)),'Active state/cursor/binding category')
    require(x['error']=='invalid_checkpoint' and x['new_events']==[],'Tamper rejected before any event')
    trusted=copy.deepcopy(X['checkpoint']);diagnostic=[]
    while trusted['phase']=='running' and len(diagnostic)<48:diagnostic.append(event_step(F,trusted))
    require(not typed_equal(diagnostic,r['new_events']),'Tampered actual retained copy changes future work')
    active=R['checkpoint'];options=[fr for fr in active['frames'] if fr['frame_kind']=='option'];branches=[fr for fr in active['frames'] if fr['frame_kind']=='branch']
    require(len(options)>=2 and branches,'Retained nested caller and branch')
    require(any(not predicate(F['options'][fr['option']]['initiation'],active['state'],fr['bindings']) for fr in options[1:]),'Previously entered nonroot initiation currently false')
    require(any(not predicate(instruction_at(F,{'option':fr['option'],'body_path':fr['body_path'][:-2],'index':fr['body_path'][-2]})['test'],active['state'],fr['bindings']) for fr in branches if fr['body_path'][-1]=='then'),'Chosen branch current guard reversal')
    return {'relations_verified':9,'repeated_effects':replayed,'X_actual_active_path':list(leaf),
            'retained_option_frames':len(options),'retained_branch_frames':len(branches)}


def identifier(item):
    present=[item[key] for key in ('name','property') if key in item]
    require(present and all(type(value) is str and value in NAMES for value in present),'Published literal identifier')
    require(all(value==present[0] for value in present),'Name/property agreement')
    return present[0]


def safe_manifest_name(value):
    require(type(value) is str and value and '\0' not in value,'Literal manifest path')
    path=PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix()==value and not any(p in {'.','..'} for p in path.parts),'Safe relative path')
    return value


def claim_key(claim):
    required={'scope','evidence_manifest','control_id','pair_sample_ids'}
    allowed=required|{'name','property','source_family_key','source_family_index','controlled_context','renderer','diagnostics'}
    require(type(claim) is dict and required<=set(claim) and set(claim)<=allowed,'R2 supported canonical claim fields; no pair hashes')
    name=identifier(claim);safe_manifest_name(claim['evidence_manifest'])
    require(claim['scope']=='corpus_variant','Claim scope')
    require(type(claim['control_id']) is str and claim['control_id'],'Nonempty control ID')
    pair=claim['pair_sample_ids']
    require(type(pair) is list and len(pair)==2 and all(type(v) is str for v in pair) and pair[0]!=pair[1],'Ordered distinct pair IDs')
    for key in ('source_family_key',):
        if key in claim:require(type(claim[key]) is str,'Optional string coordinate')
    if 'source_family_index' in claim:require(type(claim['source_family_index']) is int,'Optional integer excludes boolean')
    if 'controlled_context' in claim:require(claim['controlled_context'] in CONTEXTS,'Optional context')
    if 'renderer' in claim:require(claim['renderer'] in {'forward','reverse'},'Optional renderer')
    if 'diagnostics' in claim:require(type(claim['diagnostics']) is dict,'Optional diagnostics object')
    return (claim['control_id'],name,tuple(pair))


def resolve_claim(owner, claim, entries, members, candidate_sha):
    key=claim_key(claim)
    require(re.fullmatch('[0-9a-f]{64}',candidate_sha) is not None,'External actual whole candidate SHA')
    matches=[]
    for entry in entries:
        ident=identifier(entry)
        require(type(entry['control_id']) is str and entry['control_id'],'Entry nonempty control ID')
        require(type(entry['pair_sample_ids']) is list and len(entry['pair_sample_ids'])==2,'Entry ordered pair')
        if (entry['control_id'],ident,tuple(entry['pair_sample_ids']))==key:matches.append(entry)
    require(len(matches)==1,'Exactly one complete lookup key')
    entry=matches[0];first,second=(members[v] for v in key[2])
    require(entry['candidate_sha256']==candidate_sha and entry['scope']=='corpus_variant','Whole pending candidate/scope binding')
    for coordinate in ('source_family_key','source_family_index','split'):
        require(typed_equal(owner[coordinate],first[coordinate]) and typed_equal(owner[coordinate],second[coordinate]),'Owner/member coordinate '+coordinate)
    require(type(entry['source_family_index']) is int and entry['source_family_index']==owner['source_family_index'] and entry['source_family_key']==owner['source_family_key'],'Actual entry family binding')
    require(entry['controlled_contexts']==[first['controlled_context'],second['controlled_context']] and entry['renderers']==[first['renderer'],second['renderer']],'Actual entry coordinates/order')
    if key[1]=='object_field_order_invariance':
        require(first['controlled_context']==second['controlled_context']==owner['controlled_context'],'Same-context field pair')
        require([first['renderer'],second['renderer']]==['forward','reverse'] and owner['sample_id'] in key[2],'Owner own field pair')
    else:
        require([first['controlled_context'],second['controlled_context']]==list(PAIRS[key[1]]),'Required context order')
        require(first['renderer']==second['renderer']==owner['renderer'],'Owner/member style')
    for coordinate in ('source_family_key','source_family_index','controlled_context','renderer'):
        if coordinate in claim:require(typed_equal(claim[coordinate],owner[coordinate]),'Optional provenance agrees with owner')
    if 'pair_record_hashes' in entry:
        hashes=entry['pair_record_hashes']
        require(type(hashes) is list and len(hashes)==2 and all(type(v) is str and re.fullmatch('[0-9a-f]{64}',v) for v in hashes),'External ordered hashes')
        require(hashes==[first['record_hash'],second['record_hash']],'External hashes match independently bound members')
    return entry


def owner_claims(owner):
    require('family_claims' not in owner,'Superseded ignored claim alias prohibited')
    claims=owner['family_control_claims']
    require(type(claims) is list and len(claims)==9,'Nine supplied claims')
    keys=[claim_key(claim) for claim in claims]
    require({key[1] for key in keys}==set(NAMES),'One required claim per name')
    return keys


PUBLIC_PATHS=('schemas/RSM_OPTION_RESUMPTION_V1.md',
              'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md',
              'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json',
              'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md',
              'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md',
              'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md')


def manifest_header(manifest, source_bytes, candidate_sha, batch):
    require(type(manifest) is dict and {'metadata_version','batch','candidate_sha256','source_bindings','control_pairs'}<=set(manifest),'Minimum manifest fields')
    require(type(manifest['metadata_version']) is int and manifest['metadata_version']==1,'Typed metadata version')
    require(manifest['batch']==batch and manifest['candidate_sha256']==candidate_sha,'Batch/whole candidate binding')
    require(type(manifest['control_pairs']) is list and type(manifest['source_bindings']) is dict,'Collections')
    require(set(PUBLIC_PATHS)<=set(manifest['source_bindings']),'Six operative source bindings')
    for path,digest in manifest['source_bindings'].items():
        safe_manifest_name(path)
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),'Source lowercase SHA')
        require(path in source_bytes and hashlib.sha256(source_bytes[path]).hexdigest()==digest,'Actual source bytes independently supplied')
    return True


def program_signatures(task):
    """Full option and two coarse graphs; enumerate name/register alpha maps, literals erased.
    This signature measures fixture structure only; legacy/corpus novelty not yet inspected.
    """
    options=task['options'];root=task['entry']['option'];reachable=set()
    def called(body):
        result=[]
        for ins in body:
            if ins['op']=='call':result.append(ins['option'])
            elif ins['op']=='if':result.extend(called(ins['then']));result.extend(called(ins['else']))
        return result
    def visit(option):
        if option in reachable:return
        reachable.add(option)
        for child in called(options[option]['body']):visit(child)
    visit(root)
    def signature(view):
        selected=list(options) if view!='coarse_reachable' else list(reachable);representations=[]
        for ordering in itertools.permutations(selected):
            omap={name:i for i,name in enumerate(ordering)}
            for registers in [('x','y'),('y','x')]:
                rmap={registers[0]:'r0',registers[1]:'r1'}
                def expr(node,owner):
                    if 'reg' in node:return ['live',rmap[node['reg']]]
                    if view!='full':return ['scalar']
                    return ['literal'] if 'const' in node else ['parameter',options[owner]['params'].index(node['param'])]
                def guard(node,owner):return [[a['cmp'],expr(a['left'],owner),expr(a['right'],owner)] for a in node['all']]
                def body(instructions,owner):
                    result=[]
                    for ins in instructions:
                        op=ins['op']
                        if op in {'add','scale'}:result.append([op,rmap[ins['register']],expr(ins['value'],owner)])
                        elif op=='swap':result.append([op])
                        elif op=='if':result.append([op,guard(ins['test'],owner),body(ins['then'],owner),body(ins['else'],owner)])
                        else:
                            args=[expr(ins['args'][p],owner) for p in options[ins['option']]['params']]
                            result.append([op,omap[ins['option']],args] if view=='full' else [op,omap[ins['option']]])
                    return result
                programs=[]
                for owner in ordering:
                    definition=options[owner]
                    programs.append([len(definition['params']),guard(definition['initiation'],owner),guard(definition['success'],owner),body(definition['body'],owner)] if view=='full' else body(definition['body'],owner))
                representation=[omap[root],[expr(task['entry']['args'][p],root) for p in options[root]['params']],programs] if view=='full' else programs
                representations.append(json.dumps(representation,separators=(',',':')))
        return hashlib.sha256(min(representations).encode()).hexdigest()
    return {view:signature(view) for view in ('full','coarse_all','coarse_reachable')}
