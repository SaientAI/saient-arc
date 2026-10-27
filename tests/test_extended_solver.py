import copy
import sys
import time
import unittest
from unittest.mock import patch
from arc_gremlin.extended_solver import solve_extended, deadline_guard, DeadlineExceeded
from arc_gremlin.saient_solver import Hypothesis
from test_spatial_programs import TASKS


def task(name='recolor'):
    examples = TASKS[name]
    return {'train':copy.deepcopy(examples[:2]),'test':copy.deepcopy(examples[2:])}


def weak_baseline(t):
    hyp = Hypothesis('identity',lambda g:g,0,1.0)
    return {'predictions':[e['input'] for e in t['test']], 'best_hypothesis':hyp,'evaluated_hypotheses':1}


class ExtendedTests(unittest.TestCase):
    def test_real_baseline_ignores_test_targets(self):
        a = {'train': [{'input': [[1, 0]], 'output': [[1, 0]]},
                       {'input': [[2, 0]], 'output': [[2, 0]]}],
             'test': [{'input': [[3, 0]], 'output': [[9]]}]}
        b = copy.deepcopy(a)
        b['test'][0]['output'] = [[8, 8], [8, 8]]
        before = copy.deepcopy(a)
        opts = dict(budget_ms=1000, multi_view=True, program_search=False)
        x, y = solve_extended(a, **opts), solve_extended(b, **opts)
        self.assertEqual(x['predictions'], [[[3, 0]]])
        self.assertEqual(x['predictions'], y['predictions'])
        self.assertEqual(x['selected'], y['selected'])
        self.assertEqual(x['train_score'], y['train_score'])
        self.assertEqual(x['views'], y['views'])
        self.assertEqual(a, before)

    def test_no_target_access_and_determinism(self):
        class Forbidden(dict):
            def __getitem__(self,key):
                if key == 'output': raise AssertionError('test target read')
                return super().__getitem__(key)
        a = task()
        b = task()
        b['test'] = [Forbidden(input=e['input']) for e in b['test']]
        before = copy.deepcopy(a)
        opts = dict(baseline=False,program_search=True,budget_ms=800,max_candidates=800)
        x, y = solve_extended(a,**opts), solve_extended(b,**opts)
        self.assertEqual(x['predictions'],y['predictions'])
        self.assertEqual(x['search'],y['search'])
        self.assertEqual(a,before)
        self.assertEqual(x['predictions'],[a['test'][0]['output']])

    def test_score_fallback(self):
        with patch('arc_gremlin.extended_solver.solve_with_report',side_effect=weak_baseline):
            result = solve_extended(task(),program_search=True,budget_ms=800,max_candidates=800)
        self.assertEqual(result['selected'],'program')
        self.assertEqual(result['train_score'][0],2)

    def test_baseline_tie_and_early_stop(self):
        t = task('crop')
        def perfect(t):
            hyp = Hypothesis('crop',lambda g:[[c for c in row if c == 2] for row in g if 2 in row],0,1.0)
            # Fixture exact hypothesis with known training lookup, never test targets.
            mapping = {str(e['input']):e['output'] for e in t['train']}
            hyp = Hypothesis('fixture',lambda g:mapping.get(str(g),g),0,1.0)
            return {'best_hypothesis':hyp,'predictions':[e['input'] for e in t['test']]}
        with patch('arc_gremlin.extended_solver.solve_with_report',side_effect=perfect) as mock:
            result = solve_extended(t,multi_view=True,program_search=True)
        self.assertEqual(mock.call_count,1)
        self.assertEqual(result['selected'],'original')
        self.assertIsNone(result['search'])

    def test_budget_and_ablation(self):
        result = solve_extended(task(),budget_ms=0,program_search=True,multi_view=True)
        self.assertEqual(result['view_attempts'],0)
        self.assertTrue(result['deadline_reached'])
        result = solve_extended(task(),baseline=False,program_search=True,max_candidates=1)
        self.assertLessEqual(result['search']['attempts'],1)
        result = solve_extended(task(),baseline=False,program_search=False)
        self.assertEqual(result['selected'],'identity')
        previous = sys.gettrace()
        start = time.monotonic()
        def busy():
            while True:
                pass
        with self.assertRaises(DeadlineExceeded):
            with deadline_guard(start+0.005):
                busy()
        self.assertIs(sys.gettrace(),previous)
        self.assertLess(time.monotonic()-start,0.5)

    def test_real_baseline_smoke(self):
        t = {'train':[{'input':[[1,0]],'output':[[1,0]]},{'input':[[2,0]],'output':[[2,0]]}],
             'test':[{'input':[[3,0]]}]}
        result = solve_extended(t,budget_ms=500,multi_view=True,program_search=True)
        self.assertEqual(result['predictions'],[[[3,0]]])
        self.assertLess(result['elapsed_ms'],1000)

    def test_multiview_target_isolation(self):
        seen = []
        def inspect(t):
            self.assertTrue(all(set(e) == {'input'} for e in t['test']))
            seen.append(t)
            return weak_baseline(t)
        with patch('arc_gremlin.extended_solver.solve_with_report',side_effect=inspect):
            a = solve_extended(task(),multi_view=True,program_search=True)
            changed = task()
            changed['test'][0]['output'] = [[9]]
            b = solve_extended(changed,multi_view=True,program_search=True)
        self.assertEqual(len(seen),6)
        self.assertEqual(a['predictions'],b['predictions'])
        self.assertEqual(a['search'],b['search'])

    def test_deadline_survives_baseline_exception_recovery(self):
        def busy(t):
            while True:
                try:
                    sum(range(20))
                except Exception:
                    continue
        with patch('arc_gremlin.extended_solver.solve_with_report',side_effect=busy):
            result = solve_extended(task(),budget_ms=5,multi_view=True,program_search=True)
        self.assertTrue(result['deadline_reached'])
        self.assertEqual(result['view_attempts'],1)
        self.assertLess(result['elapsed_ms'],200)
        for invalid in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): solve_extended(task(),budget_ms=invalid)



if __name__ == '__main__': unittest.main()
