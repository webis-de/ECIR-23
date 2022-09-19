from unittest import TestCase
from reconstruction_evaluation import ReconstructionEvaluation, calculate_error


class TestReconstructionEvaluation(TestCase):
    def test_extraction_of_results_01(self):
        model = ReconstructionEvaluation(0.000000001)
        systems = [
            {'system': 'a', 'ground_truth': 0.7, 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.8}},
            {'system': 'b', 'ground_truth': 0.8, 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
            {'system': 'c', 'ground_truth': 0.9, 'prediction': {'lower': 0.86, 'actual': 0.9, 'upper': 0.90}},
        ]

        expected = [
            {'post_hoc': {'system': 'a', 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.8}},
             'existing': [{'system': 'b', 'ground_truth': 0.8}, {'system': 'c', 'ground_truth': 0.9}],
             'expected_pairs': {('b', 'a'), ('c', 'a')},
             'predicted_pairs': {('c', 'a')},
             }, {
             'post_hoc': {'system': 'b', 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
             'existing': [{'system': 'a', 'ground_truth': 0.7}, {'system': 'c', 'ground_truth': 0.9}],
             'expected_pairs': {('b', 'a'), ('c', 'b')},
             'predicted_pairs': {('b', 'a'), ('c', 'b')},
             }, {
             'post_hoc': {'system': 'c', 'prediction': {'lower': 0.86, 'actual': 0.9, 'upper': 0.90}},
             'existing': [{'system': 'a', 'ground_truth': 0.7}, {'system': 'b', 'ground_truth': 0.8}],
             'expected_pairs': {('c', 'a'), ('c', 'b')},
             'predicted_pairs': {('c', 'a'), ('c', 'b')},
             }
        ]

        actual = model.construct_reconstruction_scenarious(systems)

        self.assertEquals(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertEquals(expected[i], actual[i])

        self.assertAlmostEquals(1.0, model.precision(systems), 4)
        self.assertAlmostEquals(0.833333, model.recall(systems), 4)

    def test_extraction_of_results_02(self):
        model = ReconstructionEvaluation(0.000000001)
        systems = [
            {'system': 'a', 'ground_truth': 0.9, 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.8}},
            {'system': 'b', 'ground_truth': 0.8, 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
            {'system': 'c', 'ground_truth': 0.7, 'prediction': {'lower': 0.86, 'actual': 0.9, 'upper': 0.90}},
        ]

        expected = [
            {'post_hoc': {'system': 'a', 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.8}},
             'existing': [{'system': 'b', 'ground_truth': 0.8}, {'system': 'c', 'ground_truth': 0.7}],
             'expected_pairs': {('a', 'b'), ('a', 'c')},
             'predicted_pairs': set(),
             }, {
             'post_hoc': {'system': 'b', 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
             'existing': [{'system': 'a', 'ground_truth': 0.9}, {'system': 'c', 'ground_truth': 0.7}],
             'expected_pairs': {('a', 'b'), ('b', 'c')},
             'predicted_pairs': {('a', 'b'), ('b', 'c')}
             }, {
             'post_hoc': {'system': 'c', 'prediction': {'lower': 0.86, 'actual': 0.9, 'upper': 0.90}},
             'existing': [{'system': 'a', 'ground_truth': 0.9}, {'system': 'b', 'ground_truth': 0.8}],
             'expected_pairs': {('a', 'c'), ('b', 'c')},
             'predicted_pairs': {('c', 'b')}
             }
        ]

        actual = model.construct_reconstruction_scenarious(systems)

        self.assertEquals(len(expected), len(actual))

        for i in range(len(expected)):
            self.assertEquals(expected[i], actual[i])

        self.assertAlmostEquals(0.66666, model.precision(systems), 4)
        self.assertAlmostEquals(0.33333, model.recall(systems), 4)

    def test_error_for_perfect_prediction(self):
        expected = 0

        self.assertEquals(expected, calculate_error(0, 1, 0.5, 0.5, False))
        self.assertEquals(expected, calculate_error(0, 1, 0.5, 0.5, True))

        self.assertEquals(expected, calculate_error(0.5, 0.5, 0.5, 0.5, False))
        self.assertEquals(expected, calculate_error(0.5, 0.5, 0.5, 0.5, True))

    def test_error_for_prediction_that_is_too_low_01(self):
        self.assertAlmostEquals(0.1, calculate_error(0, 1, 0.5, 0.4, False), 4)
        self.assertAlmostEquals(0.166666, calculate_error(0, 1, 0.5, 0.4, True), 4)

    def test_error_for_prediction_that_is_too_low_02(self):
        self.assertAlmostEquals(0.1, calculate_error(0, 0.6, 0.5, 0.4, False), 4)
        self.assertAlmostEquals(0.5, calculate_error(0, 0.6, 0.5, 0.4, True), 4)

    def test_error_for_prediction_that_is_too_low_03(self):
        self.assertAlmostEquals(0.1, calculate_error(0, 0.5, 0.5, 0.4, False), 4)
        self.assertAlmostEquals(1.0, calculate_error(0, 0.5, 0.5, 0.4, True), 4)

    def test_error_for_prediction_that_is_too_low_04(self):
        self.assertAlmostEquals(0.2, calculate_error(0, 0.7, 0.5, 0.3, False), 4)
        self.assertAlmostEquals(0.5, calculate_error(0, 0.7, 0.5, 0.3, True), 4)

    def test_error_for_prediction_that_is_too_high_01(self):
        self.assertAlmostEquals(-0.1, calculate_error(0, 1, 0.5, 0.6, False), 4)
        self.assertAlmostEquals(-0.166666, calculate_error(0, 1, 0.5, 0.6, True), 4)

    def test_error_for_prediction_that_is_too_high_02(self):
        self.assertAlmostEquals(-0.1, calculate_error(0.4, 1, 0.5, 0.6, False), 4)
        self.assertAlmostEquals(-0.5, calculate_error(0.4, 1, 0.5, 0.6, True), 4)

    def test_error_for_prediction_that_is_too_high_03(self):
        self.assertAlmostEquals(-0.1, calculate_error(0.5, 1, 0.5, 0.6, False), 4)
        self.assertAlmostEquals(-1, calculate_error(0.5, 1, 0.5, 0.6, True), 4)

    def test_error_for_prediction_that_is_too_high_04(self):
        self.assertAlmostEquals(-0.2, calculate_error(0.3, 1, 0.5, 0.7, False), 4)
        self.assertAlmostEquals(-0.5, calculate_error(0.3, 1, 0.5, 0.7, True), 4)
