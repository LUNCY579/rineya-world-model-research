"""Independent public full/coarse/reachable graph views; pure import.
No producer normalizer/reference/private source is read or reused.
"""
import hashlib
import itertools

def lift(task):
    if task['kind']=='option_resume':return task['options'],task['entry']['option']
    if task['kind']!='skills':raise ValueError('unsupported legacy input kind')
    root='@ROOT';names={p:'@PROC:'+p for p in task['procedures']}
    def block(xs):
        result=[]
        for i in xs:
            op=i['op']
            if op in ('add','scale'):result.append({'op':op,'register':i['register'],'value':{'const':i['value']}})
            elif op=='swap':result.append({'op':'swap'})
            elif op=='call':result.append({'op':'call','option':names[i['name']],'args':{}})
            elif op=='if':
                a=i['test'];result.append({'op':'if','test':{'all':[{'left':{'reg':a['register']},'cmp':a['cmp'],'right':{'const':a['value']}}]},'then':block(i['then']),'else':block(i['else'])})
            else:raise ValueError('unknown legacy primitive')
        return result
    ds={names[n]:{'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':block(b)} for n,b in task['procedures'].items()}
    ds[root]={'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':block(task['main'])}
    return ds,root

def signature(task,view):
    if view not in ('full','coarse_all','coarse_reachable'):raise ValueError('unknown view')
    defs,root=lift(task);names=list(defs)
    def calls(xs):
        out=set()
        for x in xs:
            if x['op']=='call':out.add(x['option'])
            if x['op']=='if':out|=calls(x['then'])|calls(x['else'])
        return out
    if view=='coarse_reachable':
        seen=set();pending=[root]
        while pending:
            n=pending.pop()
            if n in seen:continue
            seen.add(n);pending.extend(calls(defs[n]['body'])-seen)
        names=[n for n in names if n in seen]
    if len(names)>7:raise ValueError('bounded exhaustive graph normalization exceeds seven procedures')
    candidates=[];coarse=view!='full'
    for order in itertools.permutations(names):
        om={n:i for i,n in enumerate(order)};pm={n:{p:i for i,p in enumerate(defs[n]['params'])} for n in order}
        for swap in (False,True):
            reg=lambda n:('y' if n=='x' else 'x') if swap else n
            def expr(x,owner):
                if 'reg' in x:return ('reg',reg(x['reg']))
                if coarse:return ('scalar',)
                return ('const',) if 'const' in x else ('param',pm[owner][x['param']])
            def predicate(p,owner):return tuple((expr(a['left'],owner),a['cmp'],expr(a['right'],owner)) for a in p['all'])
            def body(xs,owner):
                out=[]
                for x in xs:
                    op=x['op']
                    if op in ('add','scale'):out.append((op,reg(x['register']),expr(x['value'],owner)))
                    elif op=='swap':out.append(('swap',))
                    elif op=='if':out.append(('if',predicate(x['test'],owner),body(x['then'],owner),body(x['else'],owner)))
                    else:
                        callee=x['option'];args=() if coarse else tuple(sorted((pm[callee][p],expr(v,owner)) for p,v in x['args'].items()))
                        out.append(('call',om[callee],args))
                return tuple(out)
            graph=[]
            for n in order:
                d=defs[n]
                graph.append((body(d['body'],n),) if coarse else (len(d['params']),predicate(d['initiation'],n),predicate(d['success'],n),body(d['body'],n)))
            candidates.append(repr((tuple(graph),None if coarse else om[root])))
    text=min(candidates)
    return {'sha256':hashlib.sha256(text.encode()).hexdigest(),'canonical_graph':text,'procedure_count':len(names),'view':view}
