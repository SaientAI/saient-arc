#!/usr/bin/env python3
"""Offline public training-task comparison; answers enter scoring only."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arc_gremlin.extended_solver import solve_extended, sanitize
from tools.benchmark import transform, train_score
from arc_gremlin.saient_solver import solve_with_report


def original_three_view(task):
    best, predictions = (-1,-1), []
    for name in ('original','rotate90','palette'):
        transformed,inverse = transform(task,name)
        report = solve_with_report(transformed)
        score = train_score(report,transformed)
        if score > best:
            best, predictions = score, [inverse(g) for g in report['predictions']]
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path,help='local PUBLIC ARC-AGI-1 training directory')
    p.add_argument('--ids',nargs='*')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--budget-ms',type=float,default=1000)
    p.add_argument('--max-candidates',type=int,default=512)
    a = p.parse_args()
    root = a.data.resolve()
    if not root.is_relative_to(ROOT): p.error('data must be inside the isolated workspace')
    paths = sorted(root.glob('*.json'))
    if a.ids: paths = [f for f in paths if f.stem in a.ids]
    paths = paths[:a.limit]
    if not paths: p.error('no local public tasks found')
    results = []
    for path in paths:
        if not path.resolve().is_relative_to(ROOT): p.error('task symlink escapes workspace')
        task = json.loads(path.read_text())
        clean = sanitize(task)
        start = time.monotonic()
        old = original_three_view(clean)
        old_ms = (time.monotonic()-start)*1000
        new = solve_extended(clean,budget_ms=a.budget_ms,max_candidates=a.max_candidates,multi_view=True,program_search=True)
        targets = [e['output'] for e in task['test']]
        def score(pred):
            return {'tasks_exact':int(pred == targets),'outputs_exact':sum(x==y for x,y in zip(pred,targets))}
        results.append({'id':path.stem,'original_three_view':score(old),'extended':score(new['predictions']),
                        'original_ms':round(old_ms,3),'extended_ms':new['elapsed_ms'],'selected':new['selected']})
    print(json.dumps({'tasks':len(results),'outputs':sum(len(json.loads(f.read_text())['test']) for f in paths),
                      'results':results,'note':'Original three-view is unbounded; extended uses task deadline.'},indent=2))


if __name__ == '__main__': main()
