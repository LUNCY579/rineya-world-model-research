"""Compile only executed results; never substitutes missing outcomes with zero."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dsv3wm.stats import wilson_interval,paired_mcnemar,holm
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('summary exists')
    summary={'scope':'native CPU seed0 operational pilot, exploratory, no confirmation labels',
             'arms':{},'paired_development':{},'confirmed_findings':[],
             'deferred':['A3/A4 native distillation training','multi-seed primary study','independent confirmation',
                         'specialist routing','compressed memory/MLA conversion','behavioral skill retention']}
    for arm in ('A0','A1','A2','A5'):
        def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        run=read(ROOT/'artifacts'/f'{arm}_s0/run.json')
        dev=read(ROOT/'results'/f'{arm}_cached_dev.json')
        plan=read(ROOT/'results'/f'{arm}_cached_planning.json')
        retention=read(ROOT/'results'/f'retention_{arm}_v1.json')
        entry={'training':run,'dynamics':dev,'planning':plan,'retention_loss_only':retention}
        if dev and dev['status']=='pilot-evaluated':
            entry['wilson_95_by_horizon']={h:{metric:wilson_interval(dev['summary'][h][metric]['count'],dev['summary'][h]['n'])
                 for metric in ('direct_correct','rollout_correct','legality_correct')} for h in dev['summary']}
        if plan and plan['status']=='pilot-evaluated' and plan['n']:
            entry['planning_wilson_95']=wilson_interval(plan['successes'],plan['n'])
        summary['arms'][arm]=entry
    pvalues={}
    for candidate in ('A1','A2'):
        for control in ('A0','A5'):
            c=summary['arms'][candidate]['dynamics'];b=summary['arms'][control]['dynamics']
            if not(c and b and c['status']==b['status']=='pilot-evaluated'):continue
            ids=[r['parent_world_id'] for r in b['per_parent']]
            if ids!=[r['parent_world_id'] for r in c['per_parent']] or len(set(ids))!=len(ids):
                raise ValueError('paired independent dev parents mismatch')
            key=f'{candidate}_vs_{control}_h4_rollout'
            outcome=paired_mcnemar([r['horizons']['4']['rollout_correct'] for r in b['per_parent']],
                                   [r['horizons']['4']['rollout_correct'] for r in c['per_parent']])
            summary['paired_development'][key]=outcome;pvalues[key]=outcome['p_two_sided']
    if pvalues:summary['exploratory_holm_family']=holm(pvalues)
    summary['uncertainty_warning']='Two dev parents and one seed cannot test efficacy; nonsignificance is not equivalence. Planning has one parent.'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2),encoding='utf-8',newline='\n')
    print(str(out))


if __name__=='__main__':main()
