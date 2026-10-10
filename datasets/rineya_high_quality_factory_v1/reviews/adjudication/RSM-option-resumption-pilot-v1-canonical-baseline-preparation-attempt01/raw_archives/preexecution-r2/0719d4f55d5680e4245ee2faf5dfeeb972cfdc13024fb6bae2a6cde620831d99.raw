"""Own bounded alpha-normalized public graph signatures; no private helpers."""
import itertools,json

def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def lift_legacy(t):
    if t['kind']!='skills':return t
    root='__own_lifted_main__'
    while root in t['procedures']:root+='_'
    def convert(items):
        out=[]
        for ins in items:
            op=ins['op']
            if op=='call':out.append({'op':'call','option':ins['name'],'args':{}})
            elif op=='if':
                p=ins['test'];out.append({'op':'if','test':{'all':[{'left':{'reg':p['register']},'cmp':p['cmp'],'right':{'const':p['value']}}]},'then':convert(ins['then']),'else':convert(ins['else'])})
            elif op=='swap':out.append({'op':'swap'})
            else:out.append({'op':op,'register':ins['register'],'value':{'const':ins['value']}})
        return out
    options={name:{'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':convert(body)} for name,body in t['procedures'].items()}
    options[root]={'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':convert(t['main'])}
    return {'kind':'option_resume','options':options,'entry':{'option':root,'args':{}}}

def reachable(t):
    def targets(items):
        found=set()
        for i in items:
            if i['op']=='call':found.add(i['option'])
            elif i['op']=='if':found.update(targets(i['then']));found.update(targets(i['else']))
        return found
    seen=set();pending=[t['entry']['option']]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);pending.extend(targets(t['options'][name]['body']))
    return seen

def signature(task,view):
    t=lift_legacy(task);full=view=='full';names=sorted(reachable(t) if view=='reachable_coarse' else t['options'])
    # Bounded supplied cohort has <=4 options; legacy includes synthetic main.
    if len(names)>7:raise ValueError('own bounded normalization >7 options requires fresh preparation')
    candidates=[]
    for permutation in itertools.permutations(names):
        labels={name:i for i,name in enumerate(permutation)}
        for registers in ({'x':'r0','y':'r1'},{'x':'r1','y':'r0'}):
            params={name:{p:i for i,p in enumerate(t['options'][name]['params'])} for name in names}
            def expr(e,owner):
                if 'reg' in e:return ['reg',registers[e['reg']]]
                if not full:return ['scalar']
                if 'param' in e:return ['param',params[owner][e['param']]]
                return ['const']
            def pred(p,owner):return [[expr(a['left'],owner),a['cmp'],expr(a['right'],owner)] for a in p['all']]
            def body(items,owner):
                out=[]
                for i in items:
                    op=i['op']
                    if op in ('add','scale'):out.append([op,registers[i['register']],expr(i['value'],owner)])
                    elif op=='swap':out.append(['swap'])
                    elif op=='if':out.append(['if',pred(i['test'],owner),body(i['then'],owner),body(i['else'],owner)])
                    else:
                        target=i['option'];args=[[params[target][p],expr(e,owner)] for p,e in i['args'].items()] if full else []
                        out.append(['call',labels[target],sorted(args,key=lambda v:v[0])] if full else ['call',labels[target]])
                return out
            options=[]
            for name in permutation:
                option=t['options'][name];items=body(option['body'],name)
                options.append([len(option['params']),pred(option['initiation'],name),pred(option['success'],name),items] if full else items)
            result={'procedures':options}
            if full:
                root=t['entry']['option'];result['entry']=[labels[root],sorted([[params[root][p],expr(e,root)] for p,e in t['entry']['args'].items()],key=lambda v:v[0])]
            candidates.append(encode(result))
    return min(candidates)
