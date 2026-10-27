#!/usr/bin/env python3
"""ARC-AGI-1 public evaluation with a frozen three-view selection policy.
No model retraining; variants are equivalent input encodings and inverse outputs.
The test targets are used ONLY for scoring after predictions are generated.
"""
from __future__ import annotations
import json, time, sys, os
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

NAMES = ('original', 'rotate90', 'palette')


def predict_task(task):
    """Commit all view predictions and TRAIN-only selection without TEST labels.

    Metadata and TEST output fields are deliberately not copied into the solver
    input. Pixel agreement is rounded to six decimals before arbitration, as in
    the original public benchmark; exact ties prefer the earlier view.
    """
    clean = {
        'train': [{'input': [row[:] for row in e['input']],
                   'output': [row[:] for row in e['output']]} for e in task['train']],
        'test': [{'input': [row[:] for row in e['input']]} for e in task['test']],
    }
    views = {}
    for name in NAMES:
        transformed, inverse = transform(clean, name)
        begin = time.monotonic()
        try:
            report = solve_with_report(transformed)
            prediction = [inverse(g) for g in report['predictions']]
            exact, pixel = train_score(report, transformed)
            views[name] = {
                'train_exact': exact, 'train_total': len(clean['train']),
                'train_pixel': round(pixel, 6),
                'confidence': report.get('confidence', 0),
                'tier': report.get('selected_tier'),
                'evals': report.get('evaluated_hypotheses', 0),
                'runtime_ms': round((time.monotonic()-begin)*1000, 2),
                'prediction': prediction,
            }
        except Exception as exc:
            views[name] = {
                'error': type(exc).__name__+': '+str(exc)[:140],
                'runtime_ms': round((time.monotonic()-begin)*1000, 2),
            }
    selected = max(NAMES, key=lambda name: (
        views[name].get('train_exact', -1),
        views[name].get('train_pixel', -1), -NAMES.index(name)))
    return {
        'views': views, 'selected': selected,
        'views_disagree': any(views[name].get('prediction') !=
                             views['original'].get('prediction') for name in NAMES[1:]),
    }


def main():
    paths=sorted(DATA.glob('*.json'))
    assert len(paths)==400 or (os.environ.get('SAIENT_ARC_LIMIT') and paths), f'Expected 400 public evaluation tasks or a limited test fixture, found {len(paths)}'
    chosen=paths[:int(os.environ.get('SAIENT_ARC_LIMIT','0'))] if int(os.environ.get('SAIENT_ARC_LIMIT','0'))>0 else paths
    names=NAMES
    results=[]
    test_outputs=0
    print(f'start: {len(chosen)} ARC public evaluation tasks; frozen three-view policy; no model edits',flush=True)
    start=time.monotonic()
    for i,p in enumerate(chosen):
        task=json.loads(p.read_text())
        committed=predict_task(task)
        # No TEST output is read until all predictions and the selected view
        # have returned from the answer-independent prediction function.
        actual=[example['output'] for example in task['test']]
        test_outputs+=len(actual)
        views=committed['views']
        for name in names:
            pred=views[name].get('prediction', [])
            views[name]['exact']='error' not in views[name] and len(pred)==len(actual) and all(a==b for a,b in zip(pred,actual))
            views[name]['outputs_exact']=sum(a==b for a,b in zip(pred,actual))
        pick=committed['selected']
        results.append({'id':p.stem,**committed,'selected_exact':views[pick]['exact'],
                        'oracle_any':any(views[name]['exact'] for name in names)})
        if (i+1)%40==0:
            print(f'progress={i+1}/{len(chosen)}, elapsed_s={time.monotonic()-start:.1f}',flush=True)
    summary={'benchmark':f'ARC-AGI-1 public evaluation, {len(chosen)} tasks; frozen representation views/policy','task_ids':[p.stem for p in chosen],
             'method':'original versus 90-degree globally rotated task versus globally palette-frequency-renamed task; inverse-transform predictions; each sees train inputs/outputs and test inputs, never test outputs before solving',
             'elapsed_s':round(time.monotonic()-start,3),'n':len(results),
             'test_outputs':test_outputs,
             'selected_outputs_exact':sum(r['views'][r['selected']]['outputs_exact'] for r in results),
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
    print('SUMMARY '+json.dumps({k:summary[k] for k in ('n','test_outputs','selected_outputs_exact','solved_tasks','selected_solved','oracle_any_solved','disagreements','unique_solves','errors','mean_runtime_ms','selection_counts','elapsed_s')},sort_keys=True),flush=True)
if __name__=='__main__':main()
