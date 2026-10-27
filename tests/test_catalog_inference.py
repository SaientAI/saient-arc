import copy
import unittest

from arc_gremlin import saient_solver as solver


class CatalogInferenceTests(unittest.TestCase):
    def features(self, examples):
        return solver.extract_features(examples)

    def test_uniform_row_marker_transfers_to_new_dimensions_and_palette(self):
        for marker in (6, 8):
            train = [
                {'input': [[2, 2, 2, 2], [3, 0, 3, 0]],
                 'output': [[marker] * 4, [0] * 4]},
                {'input': [[4, 4], [1, 0], [7, 7]],
                 'output': [[marker] * 2, [0] * 2, [marker] * 2]},
            ]
            hypotheses = solver.generate_hypotheses(self.features(train), False, False)
            hypothesis = next(h for h in hypotheses if h.name == 'uniform_rows_to_marker')
            # New dimensions and input colors, absent from both training examples.
            grid = [[9] * 5, [5, 0, 5, 0, 5], [3] * 5]
            self.assertEqual(hypothesis.fn(grid), [[marker] * 5, [0] * 5, [marker] * 5])

    def test_uniform_row_marker_rejects_conflicting_training_colors(self):
        train = [
            {'input': [[2, 2], [3, 0]], 'output': [[6, 6], [0, 0]]},
            {'input': [[4, 4], [1, 0]], 'output': [[8, 8], [0, 0]]},
        ]
        self.assertIsNone(solver._infer_uniform_rows_marker_rule(self.features(train)['pairs']))

    def crossing_train(self):
        return [
            {'input': [[1, 0, 0], [0, 0, 0], [0, 0, 3]],
             'output': [[1, 1, 7], [1, 0, 3], [7, 3, 3]]},
            {'input': [[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 6]],
             'output': [[4, 4, 4, 7], [0, 4, 0, 6], [0, 4, 0, 6], [6, 7, 6, 6]]},
        ]

    def test_crossing_color_transfers_without_fixed_color_or_size(self):
        hypothesis = solver._build_biaxial_marker_expand_hypothesis(self.features(self.crossing_train()))
        self.assertIsNotNone(hypothesis)
        grid = [[0] * 5, [2, 0, 0, 0, 0], [0] * 5, [0] * 5, [0, 0, 0, 9, 0]]
        expected = [[2, 0, 0, 9, 0], [2, 2, 2, 7, 2], [2, 0, 0, 9, 0],
                    [2, 0, 0, 9, 0], [7, 9, 9, 9, 9]]
        self.assertEqual(hypothesis.fn(grid), expected)

    def test_crossing_has_no_default_when_training_is_uninformative(self):
        train = [{'input': [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                  'output': [[1, 1, 1], [1, 0, 1], [1, 1, 1]]}]
        self.assertIsNone(solver._build_biaxial_marker_expand_hypothesis(self.features(train)))

    def test_crossing_rejects_conflicts_and_partial_match(self):
        train = self.crossing_train()
        conflicting = copy.deepcopy(train)
        conflicting[1]['output'][0][3] = 8
        partial = copy.deepcopy(train)
        partial[1]['output'][1][0] = 5
        for examples in (conflicting, partial):
            self.assertIsNone(solver._build_biaxial_marker_expand_hypothesis(self.features(examples)))


if __name__ == '__main__':
    unittest.main()
