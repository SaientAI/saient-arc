import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tools import benchmark


def fixture():
    return {'train': [{'input': [[1, 0]], 'output': [[1, 0]]},
                      {'input': [[2, 0]], 'output': [[2, 0]]}],
            'test': [{'input': [[3, 0]], 'output': [[9]]}]}


def stable_prediction(result):
    result = copy.deepcopy(result)
    for view in result['views'].values():
        view.pop('runtime_ms', None)
    return result


class BenchmarkTests(unittest.TestCase):
    def test_real_predictions_and_selection_ignore_test_labels(self):
        class Forbidden(dict):
            def __getitem__(self, key):
                if key == 'output':
                    raise AssertionError('TEST output read during prediction')
                return super().__getitem__(key)

        first = fixture()
        changed = fixture()
        changed['test'][0]['output'] = [[8, 8], [8, 8]]
        forbidden = fixture()
        forbidden['test'] = [Forbidden(input=[[3, 0]], output=[[7]])]
        before = copy.deepcopy(first)
        a = benchmark.predict_task(first)
        b = benchmark.predict_task(changed)
        c = benchmark.predict_task(forbidden)
        self.assertEqual(stable_prediction(a), stable_prediction(b))
        self.assertEqual(stable_prediction(a), stable_prediction(c))
        self.assertEqual(a['views'][a['selected']]['prediction'], [[[3, 0]]])
        self.assertEqual(first, before)

    def test_ranking_keeps_six_decimal_rounding_and_original_tie_break(self):
        def report(task):
            return {'predictions': [e['input'] for e in task['test']]}
        with patch.object(benchmark, 'solve_with_report', side_effect=report), \
             patch.object(benchmark, 'train_score', side_effect=[
                 (1, 0.5000001), (1, 0.5000004), (1, 0.4999999)]):
            result = benchmark.predict_task(fixture())
        self.assertEqual(result['selected'], 'original')
        self.assertTrue(all(v['train_pixel'] == 0.5 for v in result['views'].values()))

    def test_runner_scores_only_after_all_views_and_selection_are_committed(self):
        committed = []
        actual_predict = benchmark.predict_task

        class CheckedOutput(dict):
            def __getitem__(self, key):
                if key == 'output':
                    if not committed:
                        raise AssertionError('TEST scoring happened before prediction returned')
                    result = committed[0]
                    self_test.assertEqual(set(result['views']), set(benchmark.NAMES))
                    self_test.assertIn(result['selected'], result['views'])
                return super().__getitem__(key)

        def predict(task):
            result = actual_predict(task)
            committed.append(copy.deepcopy(result))
            return result

        self_test = self
        task = fixture()
        task['test'] = [CheckedOutput(input=[[3, 0]], output=[[3, 0]])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            (data/'fixture.json').write_text('{}')
            out = root/'results'
            with patch.object(benchmark, 'DATA', data), \
                 patch.object(benchmark.json, 'loads', return_value=task), \
                 patch.object(benchmark, 'predict_task', side_effect=predict), \
                 patch.dict(os.environ, {'SAIENT_ARC_LIMIT': '1', 'SAIENT_ARC_OUT': str(out)}), \
                 redirect_stdout(io.StringIO()):
                benchmark.main()
            saved = json.loads((out/'eval_results.json').read_text())
        self.assertEqual(len(committed), 1)
        self.assertEqual(saved['summary']['selected_solved'], 1)
        self.assertEqual(saved['summary']['test_outputs'], 1)
        self.assertEqual(saved['summary']['selected_outputs_exact'], 1)
        self.assertNotIn('exact', committed[0]['views']['original'])


if __name__ == '__main__':
    unittest.main()
