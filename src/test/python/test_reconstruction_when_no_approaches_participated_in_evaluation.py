from unittest import TestCase
from reconstruction_evaluation import AllApproachesDidNotParticipateInPoolingReconstructionEvaluation, DataConstruction


class TestReconstructionEvaluation(TestCase):
    def test_small_example(self):
        topics = [
            {'system': 'a', 'ground_truth': 0.7, 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.8}},
            {'system': 'b', 'ground_truth': 0.8, 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
            {'system': 'c', 'ground_truth': 0.9, 'prediction': {'lower': 0.86, 'actual': 0.9, 'upper': 0.90}},
        ]

        expected_correct_pairs = {('b', 'a'), ('c', 'a'), ('c', 'b')}
        expected_retrieved_pairs = {('c', 'a'), ('c', 'b')}

        reconstruction_eval = AllApproachesDidNotParticipateInPoolingReconstructionEvaluation()

        self.assertEquals(expected_correct_pairs, reconstruction_eval.ground_truth_pairs(topics))
        self.assertEquals(expected_retrieved_pairs, reconstruction_eval.predicted_pairs(topics))
        self.assertAlmostEquals(1.0, reconstruction_eval.precision(topics), 4)
        self.assertAlmostEquals(0.6666667, reconstruction_eval.recall(topics), 4)

    def test_small_only_non_overlapping_predictions(self):
        topics = [
            {'system': 'a', 'ground_truth': 0.9, 'prediction': {'lower': 0.6, 'actual': 0.7, 'upper': 0.9}},
            {'system': 'b', 'ground_truth': 0.8, 'prediction': {'lower': 0.75, 'actual': 0.8, 'upper': 0.84}},
            {'system': 'c', 'ground_truth': 0.7, 'prediction': {'lower': 0.83, 'actual': 0.9, 'upper': 0.90}},
        ]

        expected_correct_pairs = {('a', 'b'), ('a', 'c'), ('b', 'c')}
        expected_retrieved_pairs = set()

        reconstruction_eval = AllApproachesDidNotParticipateInPoolingReconstructionEvaluation()

        self.assertEquals(expected_correct_pairs, reconstruction_eval.ground_truth_pairs(topics))
        self.assertEquals(expected_retrieved_pairs, reconstruction_eval.predicted_pairs(topics))
        self.assertAlmostEquals(0.0, reconstruction_eval.precision(topics), 4)
        self.assertAlmostEquals(0.0, reconstruction_eval.recall(topics), 4)

    def test_small_only_very_small_differences(self):
        topics = [
            {'system': 'a', 'ground_truth': 0.7, 'prediction': {'lower': 0.69, 'actual': 0.7, 'upper': 0.71}},
            {'system': 'b', 'ground_truth': 0.72, 'prediction': {'lower': 0.71, 'actual': 0.72, 'upper': 0.73}},
            {'system': 'c', 'ground_truth': 0.74, 'prediction': {'lower': 0.73, 'actual': 0.74, 'upper': 0.75}},
        ]

        expected_correct_pairs = {('b', 'a'), ('c', 'a'), ('c', 'b')}
        expected_retrieved_pairs = {('c', 'a')}

        reconstruction_eval = AllApproachesDidNotParticipateInPoolingReconstructionEvaluation()

        self.assertEquals(expected_correct_pairs, reconstruction_eval.ground_truth_pairs(topics))
        self.assertEquals(expected_retrieved_pairs, reconstruction_eval.predicted_pairs(topics))
        self.assertAlmostEquals(1.0, reconstruction_eval.precision(topics), 4)
        self.assertAlmostEquals(0.333333, reconstruction_eval.recall(topics), 4)

    def test_small_only_very_small_differences_with_tiny_threshold(self):
        topics = [
            {'system': 'a', 'ground_truth': 0.7, 'prediction': {'lower': 0.69, 'actual': 0.7, 'upper': 0.71}},
            {'system': 'b', 'ground_truth': 0.72, 'prediction': {'lower': 0.71, 'actual': 0.72, 'upper': 0.73}},
            {'system': 'c', 'ground_truth': 0.74, 'prediction': {'lower': 0.73, 'actual': 0.74, 'upper': 0.75}},
        ]

        expected_correct_pairs = {('b', 'a'), ('c', 'a'), ('c', 'b')}
        expected_retrieved_pairs = {('c', 'a')}

        reconstruction_eval = AllApproachesDidNotParticipateInPoolingReconstructionEvaluation(0.00001)

        self.assertEquals(expected_correct_pairs, reconstruction_eval.ground_truth_pairs(topics))
        self.assertEquals(expected_retrieved_pairs, reconstruction_eval.predicted_pairs(topics))
        self.assertAlmostEquals(1.0, reconstruction_eval.precision(topics), 4)
        self.assertAlmostEquals(0.333333, reconstruction_eval.recall(topics), 4)

    def test_small_only_very_small_differences_with_larger_threshold(self):
        topics = [
            {'system': 'a', 'ground_truth': 0.7, 'prediction': {'lower': 0.69, 'actual': 0.7, 'upper': 0.71}},
            {'system': 'b', 'ground_truth': 0.72, 'prediction': {'lower': 0.71, 'actual': 0.72, 'upper': 0.73}},
            {'system': 'c', 'ground_truth': 0.74, 'prediction': {'lower': 0.73, 'actual': 0.74, 'upper': 0.75}},
        ]

        expected_correct_pairs = {('c', 'a')}
        expected_retrieved_pairs = set()

        reconstruction_eval = AllApproachesDidNotParticipateInPoolingReconstructionEvaluation(0.02000001)

        self.assertEquals(expected_correct_pairs, reconstruction_eval.ground_truth_pairs(topics))
        self.assertEquals(expected_retrieved_pairs, reconstruction_eval.predicted_pairs(topics))
        self.assertAlmostEquals(0.0, reconstruction_eval.precision(topics), 4)
        self.assertAlmostEquals(0.0, reconstruction_eval.recall(topics), 4)

    def test_extraction_of_traditional_xy(self):
        input_data = {'320': [
            {'topic': '320',
             'system': 'a',
             'ground_truth': 0.111,
             'Min-Residual': 0.222,
             'Condensed': 0.333,
             'Max-Residual': 0.444,

             'PBS-RP-RMSE[0.8,1]': 0.555,
             'PBS-RP-RMSE': 0.666,
             'PBS-RP-RMSE[1,3]': 0.777,

             'PBS-R-RMSE[0.8,1]': 0.888,
             'PBS-R-RMSE': 0.999,
             'PBS-R-RMSE[1,3]': 1.111,

             'PBS-R-RMSE[0.8,1]': 1.222,
             'PBS-R-RMSE': 1.333,
             'PBS-R-RMSE[1,3]': 1.444,

             'PBS-RMSE[0.8,1]': 1.555,
             'PBS-RMSE': 1.666,
             'PBS-RMSE[1,3]': 1.777
             }, {'topic': '320',
             'system': 'b',
             'ground_truth': 0.111,
             'Min-Residual': 0.222,
             'Condensed': 0.333,
             'Max-Residual': 0.444,

             'PBS-RP-RMSE[0.8,1]': 0.555,
             'PBS-RP-RMSE': 0.666,
             'PBS-RP-RMSE[1,3]': 0.777,

             'PBS-R-RMSE[0.8,1]': 0.888,
             'PBS-R-RMSE': 0.999,
             'PBS-R-RMSE[1,3]': 1.111,

             'PBS-R-RMSE[0.8,1]': 1.222,
             'PBS-R-RMSE': 1.333,
             'PBS-R-RMSE[1,3]': 1.444,

             'PBS-RMSE[0.8,1]': 1.555,
             'PBS-RMSE': 1.666,
             'PBS-RMSE[1,3]': 1.777
             }
        ]}

        expected = {'320': [
            {'system': 'a', 'ground_truth': 0.111, 'prediction': {'lower': 0.222, 'actual': 0.333, 'upper': 0.444}},
            {'system': 'b', 'ground_truth': 0.111, 'prediction': {'lower': 0.222, 'actual': 0.333, 'upper': 0.444}},
        ]}

        actual = DataConstruction('Min-Residual', 'Condensed', 'Max-Residual').construct_data_for_reconstruction_evaluation(input_data)

        self.assertEquals(expected, actual)
