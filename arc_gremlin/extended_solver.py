"""Opt-in task-budgeted baseline, representation views and safe program search."""
import argparse
import json
import math
import sys
import time
from collections import Counter
from contextlib import contextmanager
from .saient_solver import solve_with_report, apply
from .spatial_programs import freeze, agreement, synthesize, execute


class DeadlineExceeded(BaseException):
    pass


@contextmanager
def deadline_guard(deadline):
    """Interrupt trusted Python work; restore the caller's tracing hook.

    Cooperative at Python line boundaries in called functions: cannot preempt
    a long C operation. BaseException avoids baseline Exception recovery paths.
    No threads, processes, signal handlers or generated-code execution.
    """
    previous = sys.gettrace()
    def trace(frame, event, arg):
        if time.monotonic() >= deadline: raise DeadlineExceeded()
        return trace
    sys.settrace(trace)
    try: yield
    finally: sys.settrace(previous)


def sanitize(task):
    return {'train':[{'input':[list(r) for r in freeze(e['input'])], 'output':[list(r) for r in freeze(e['output'])]} for e in task['train']],
            'test':[{'input':[list(r) for r in freeze(e['input'])]} for e in task['test']]}


def view(task, name):
    if name == 'original': return task, lambda g:g
    if name == 'rotate90':
        forward = lambda g:[list(r) for r in zip(*g[::-1])]
        inverse = lambda g:[list(r) for r in zip(*g)][::-1]
    elif name == 'palette':
        counts = Counter(c for group in ('train','test') for e in task[group] for key in ('input','output') if key in e for row in e[key] for c in row if c)
        mapping = {0:0, **{c:i+1 for i,c in enumerate(sorted(counts,key=lambda c:(-counts[c],c)))}}
        reverse = {v:k for k,v in mapping.items()}
        forward = lambda g:[[mapping[c] for c in r] for r in g]
        inverse = lambda g:[[reverse.get(c,c) for c in r] for r in g]
    else: raise ValueError('unknown view')
    return {'train':[{'input':forward(e['input']),'output':forward(e['output'])} for e in task['train']],
            'test':[{'input':forward(e['input'])} for e in task['test']]}, inverse


def solve_extended(task, *, budget_ms=1000, max_candidates=512, multi_view=False,
                   program_search=False, baseline=True, connectivity=4, max_depth=2):
    if not math.isfinite(budget_ms) or budget_ms < 0 or type(max_candidates) is not int or max_candidates < 0:
        raise ValueError('invalid budget')
    if connectivity not in (4,8) or max_depth not in (1,2): raise ValueError('invalid search mode')
    start = time.monotonic()
    deadline = start + budget_ms/1000
    clean = sanitize(task)
    if not clean['train']: raise ValueError('training required')
    fallback = [e['input'] for e in clean['test']]
    best_score = agreement([e['input'] for e in clean['train']],clean['train'])
    predictions, selected, logs = fallback, 'identity', []
    attempts, baseline_evaluations, search_report = 0, 0, None
    names = ('original','rotate90','palette') if multi_view else ('original',)
    if baseline:
        for name in names:
            if time.monotonic() >= deadline: break
            attempts += 1
            try:
                with deadline_guard(deadline):
                    transformed, inverse = view(clean,name)
                    report = solve_with_report(transformed)
                    hypothesis = report.get('best_hypothesis')
                    train_predictions = [apply(hypothesis,e['input']) for e in transformed['train']]
                    score = agreement(train_predictions,transformed['train'])
                    candidate = [inverse(g) for g in report['predictions']]
                    for g in candidate: freeze(g)
                    if len(candidate) != len(clean['test']): raise ValueError('prediction count')
                baseline_evaluations += report.get('evaluated_hypotheses',0)
                logs.append({'source':name,'train_score':score,'hypothesis':getattr(hypothesis,'name',None)})
                if score > best_score or (score == best_score and selected == 'identity'):
                    predictions, best_score, selected = candidate, score, name
                if score[0] == len(clean['train']): break
            except DeadlineExceeded:
                logs.append({'source':name,'stopped':'deadline'})
                break
            except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError) as exc:
                logs.append({'source':name,'error':type(exc).__name__})
    progress = {'attempts':0,'duplicates':0,'invalid':0,'unique':0}
    if program_search and best_score[0] < len(clean['train']) and time.monotonic() < deadline:
        try:
            with deadline_guard(deadline):
                search_report = synthesize(clean['train'],max_candidates=max_candidates,budget_ms=budget_ms,
                                          deadline=deadline,connectivity=connectivity,max_depth=max_depth,progress=progress)
                program = search_report['program']
                score = search_report['train_score']
                # A held-out failure cannot replace a baseline, even on partial fit.
                if program is not None and search_report['holdout'] is not False and score > best_score:
                    candidate = [execute(program,e['input']) for e in clean['test']]
                    predictions, best_score, selected = candidate, score, 'program'
        except DeadlineExceeded:
            search_report = dict(progress, stop_reason='deadline', train_score=(0,0.0), holdout=None)
            logs.append({'source':'program','stopped':'deadline'})
        except ValueError as exc:
            logs.append({'source':'program','error':str(exc)})
    if search_report is not None:
        search_report = {k:v for k,v in search_report.items() if k != 'program'}
    elapsed = (time.monotonic()-start)*1000
    return {'predictions':predictions,'selected':selected,'train_score':best_score,'views':logs,
            'view_attempts':attempts,'baseline_evaluations':baseline_evaluations,'search':search_report,
            'elapsed_ms':round(elapsed,3),'budget_ms':budget_ms,'deadline_reached':time.monotonic() >= deadline,
            'options':{'multi_view':multi_view,'program_search':program_search,'baseline':baseline,
                       'connectivity':connectivity,'max_depth':max_depth,'max_candidates':max_candidates}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--budget-ms',type=float,default=1000)
    parser.add_argument('--max-candidates',type=int,default=512)
    parser.add_argument('--multi-view',action='store_true')
    parser.add_argument('--program-search',action='store_true')
    parser.add_argument('--no-baseline',action='store_true')
    parser.add_argument('--connectivity',type=int,choices=(4,8),default=4)
    parser.add_argument('--max-depth',type=int,choices=(1,2),default=2)
    args = vars(parser.parse_args())
    with open(args.pop('task')) as handle: task = json.load(handle)
    args['baseline'] = not args.pop('no_baseline')
    print(json.dumps(solve_extended(task,**args),indent=2))


if __name__ == '__main__': main()
