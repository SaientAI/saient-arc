#!/usr/bin/env python3
"""Independent ARC-AGI-1 public training split representation-sensitivity probe.
No model retraining; variants are equivalent input encodings and inverse outputs.
The test targets are used ONLY for scoring after predictions are generated.
"""
from __future__ import annotations
import json, time, sys, random, os
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('SAIENT_ARC_DATA',str(ROOT/'data'/'evaluation')))
sys.path.insert(0,str(ROOT))
from arc_gremlin.saient_solver import solve_with_report,apply

def rot(g):
    return [list(row) for row in zip(*g[::-1])]
def unrot(g):
    return [list(row) for row in zip(*g)][::-1]
def recolor(g,m):
    return [[m.get(x,x) for x in row] for row in g]
def build_maps(t):
    grids=[]
    for ex in t['train']:
        grids.extend([ex['input'],ex['output']])
    grids.extend(ex['input'] for ex in t['test'])
    counts=Counter(c for g in grids for row in g for c in row if c!=0)
    palette=sorted(counts,key=lambda c:(-counts[c],c))
    m={0:0}; m.update({c:i+1 for i,c in enumerate(palette)})
    return m,{v:k for k,v in m.items()}
def transform(t,view):
    if view=='original':
        return {'train':t['train'],'test':[{'input':x['input']} for x in t['test']]}, lambda x:x
    if view=='rotate90':
        return {'train':[{'input':rot(e['input']),'output':rot(e['output'])} for e in t['train']],
                 'test':[{'input':rot(x['input'])} for x in t['test']]},unrot
    if view=='palette':
        m,inv=build_maps(t)
        return {'train':[{'input':recolor(e['input'],m),'output':recolor(e['output'],m)} for e in t['train']],
                 'test':[{'input':recolor(x['input'],m)} for x in t['test']]},lambda x:recolor(x,inv)
    raise ValueError(view)
def train_score(report,task):
    h=report.get('best_hypothesis')
    if h is None:return (0,0)
    exact=0; pixels=0.0
    for e in task['train']:
        g=apply(h,e['input']); out=e['output']
        if g==out:exact+=1
        if len(g)==len(out) and g and len(g[0])==len(out[0]):
            pixels+=sum(g[y][x]==out[y][x] for y in range(len(out)) for x in range(len(out[0])))/(len(out)*len(out[0]))
    return exact,pixels/len(task['train'])

def main():
    paths=sorted(DATA.glob('*.json'))
    assert len(paths)==400 or (os.environ.get('SAIENT_ARC_LIMIT') and paths), f'Expected 400 public evaluation tasks or a limited test fixture, found {len(paths)}'
    chosen=paths[:int(os.environ.get('SAIENT_ARC_LIMIT','0'))] if int(os.environ.get('SAIENT_ARC_LIMIT','0'))>0 else paths
    names=('original','rotate90','palette')
    results=[]
    print(f'start: {len(chosen)} ARC public evaluation tasks; frozen three-view policy; no model edits',flush=True)
    start=time.monotonic()
    for i,p in enumerate(chosen):
        t=json.loads(p.read_text())
        views={}
        for name in names:
            task,inv=transform(t,name)
            begin=time.monotonic()
            try:
                report=solve_with_report(task)
                pred=[inv(g) for g in report['predictions']]
                tr_exact,tr_pixel=train_score(report,task)
                actual=[x['output'] for x in t['test']]
                views[name]={'exact':len(pred)==len(actual) and all(a==b for a,b in zip(pred,actual)),
                             'outputs_exact':sum(a==b for a,b in zip(pred,actual)),
                             'train_exact':tr_exact,'train_total':len(t['train']),
                             'train_pixel':round(tr_pixel,6),'confidence':report.get('confidence',0),
                             'tier':report.get('selected_tier'),
                             'evals':report.get('evaluated_hypotheses',0),
                             'runtime_ms':round((time.monotonic()-begin)*1000,2),
                             'prediction':pred}
            except Exception as e:
                views[name]={'error':type(e).__name__+': '+str(e)[:140], 'exact':False, 'runtime_ms':round((time.monotonic()-begin)*1000,2)}
        # Score-based arbitration is selected without seeing test targets.
        # Prefer original on equal training agreement to avoid invented gains.
        ranked=sorted(names,key=lambda n:(views[n].get('train_exact',-1),views[n].get('train_pixel',-1),-names.index(n)),reverse=True)
        pick=ranked[0]
        results.append({'id':p.stem,'views':views,'selected':pick,'selected_exact':views[pick]['exact'],
                        'oracle_any':any(views[n]['exact'] for n in names),
                        'views_disagree':any(views[n].get('prediction')!=views['original'].get('prediction') for n in names[1:])})
        if (i+1)%40==0:
            print(f'progress={i+1}/{len(chosen)}, elapsed_s={time.monotonic()-start:.1f}',flush=True)
    summary={'benchmark':f'ARC-AGI-1 public evaluation, {len(chosen)} tasks; frozen representation views/policy','task_ids':[p.stem for p in chosen],
             'method':'original versus 90-degree globally rotated task versus globally palette-frequency-renamed task; inverse-transform predictions; each sees train inputs/outputs and test inputs, never test outputs before solving',
             'elapsed_s':round(time.monotonic()-start,3),'n':len(results),
             'solved_tasks':{name:sum(r['views'][name]['exact'] for r in results) for name in names},
             'selected_solved':sum(r['selected_exact'] for r in results),
             'oracle_any_solved':sum(r['oracle_any'] for r in results),
             'disagreements':sum(r['views_disagree'] for r in results),
             'unique_solves':{name:[r['id'] for r in results if r['views'][name]['exact'] and not any(r['views'][other]['exact'] for other in names if other!=name)] for name in names},
             'errors':{name:sum('error' in r['views'][name] for r in results) for name in names},
             'mean_runtime_ms':{name:round(sum(r['views'][name]['runtime_ms'] for r in results)/len(results),2) for name in names},
             'selection_counts':dict(Counter(r['selected'] for r in results))}
    folder=Path(os.environ.get('SAIENT_ARC_OUT',str(ROOT/'results')));folder.mkdir(parents=True,exist_ok=True)
    (folder/'eval_results.json').write_text(json.dumps({'summary':summary,'tasks':results},indent=2)+'\n')
    (folder/'eval_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY '+json.dumps({k:summary[k] for k in ('n','solved_tasks','selected_solved','oracle_any_solved','disagreements','unique_solves','errors','mean_runtime_ms','selection_counts','elapsed_s')},sort_keys=True),flush=True)
if __name__=='__main__':main()
