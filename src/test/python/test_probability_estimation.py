from unittest import TestCase
from trectools import TrecRun, TrecQrel
import pandas as pd
from probability_estimation import PoissonEstimator, CountProbabilityEstimator, RunIndependentCountProbabilityEstimator


class TestProbabilityEstimation(TestCase):
    def test_probability_estimation_for_only_irrelevant_docs(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'd1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd3', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'd1', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd2', 'rel': 0},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        estimator = PoissonEstimator()
        actual = estimator.estimate_probabilities(run, qrels)
        print(actual)
        self.assertAlmostEqual(1.0, actual[0], places=4)
        self.assertAlmostEqual(0.0000, actual[1], places=4)
        self.assertAlmostEqual(0.0000, actual[2], places=4)
        self.assertAlmostEqual(0.0000, actual[3], places=4)

    def test_probability_estimation_for_20_percent_relevant_docs_k_1(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'd1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd3', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd4', 'rank': 3, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'd1', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd4', 'rel': 1},
            {'query': '301', 'q0': 0, 'docid': 'd2', 'rel': 0},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        estimator = PoissonEstimator()
        actual = estimator.estimate_probabilities(run, qrels)
        print(actual)
        self.assertAlmostEqual(0.94692, actual[0], places=4)
        self.assertAlmostEqual(0.05307, actual[1], places=4)
        self.assertAlmostEqual(0.0000, actual[2], places=4)
        self.assertAlmostEqual(0.0000, actual[3], places=4)

    def test_probability_estimation_for_only_irrelevant_docs_with_counting(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'd1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd2', 'rank': 1, 'score': 2999, 'system': 'a'},
        ])
        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'd1', 'rel': 0},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        estimator = CountProbabilityEstimator()
        actual = estimator.estimate_probabilities(run, qrels)

        print(actual)
        self.assertAlmostEqual(0.9997, actual[0], places=4)
        self.assertAlmostEqual(0.0001, actual[1], places=4)
        self.assertAlmostEqual(0.0001, actual[2], places=4)
        self.assertAlmostEqual(0.0001, actual[3], places=4)

    def test_probability_estimation_for_33_percent_relevant_docs_with_counting(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'd1', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd2', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd3', 'rank': 2, 'score': 2998, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd4', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'd1', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd2', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd3', 'rel': 1},

            # Some noise
            {'query': '302', 'q0': 0, 'docid': 'doc-0', 'rel': 0},
            {'query': '302', 'q0': 0, 'docid': 'doc-1', 'rel': 1},
            {'query': '302', 'q0': 0, 'docid': 'doc-2', 'rel': 2},
        ])

        estimator = CountProbabilityEstimator()
        actual = estimator.estimate_probabilities(run, qrels)
        self.assertAlmostEqual(0.66653, actual[0], places=4)
        self.assertAlmostEqual(0.33326, actual[1], places=4)
        self.assertAlmostEqual(0.0001, actual[2], places=4)
        self.assertAlmostEqual(0.0001, actual[3], places=4)

    def test_probability_estimation_for_run_independent_evaluation_with_counting(self):
        run = TrecRun()
        run.run_data = pd.DataFrame([
            {'query': '301', 'q0': 'Q0', 'docid': 'd7', 'rank': 0, 'score': 3000, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd8', 'rank': 1, 'score': 2999, 'system': 'a'},
            {'query': '301', 'q0': 'Q0', 'docid': 'd9', 'rank': 2, 'score': 2998, 'system': 'a'},
        ])
        qrels = TrecQrel()
        qrels.qrels_data = pd.DataFrame([
            {'query': '301', 'q0': 0, 'docid': 'd1', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd2', 'rel': 0},
            {'query': '301', 'q0': 0, 'docid': 'd3', 'rel': 1},
        ])

        estimator = RunIndependentCountProbabilityEstimator()
        actual = estimator.estimate_probabilities(run, qrels)
        self.assertAlmostEqual(0.66653, actual[0], places=4)
        self.assertAlmostEqual(0.33326, actual[1], places=4)
        self.assertAlmostEqual(0.0001, actual[2], places=4)
        self.assertAlmostEqual(0.0001, actual[3], places=4)



# Add tests for an approach that uses simply the condensed thing as predictions
