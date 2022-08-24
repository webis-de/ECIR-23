from unittest import TestCase
from approvaltests import verify
from trectools import TrecEval
from error_analysis_approaches import *
from probability_estimation import *


def eval_measures(data):
    return {
        'condensed-ndcg@5': TrecEval(data['run'], data['qrels_incomplete']).get_ndcg(5, removeUnjudged=True),
        'ndcg@5': TrecEval(data['run'], data['qrels_complete']).get_ndcg(5),
    }


def __norm(i):
    return {k: round(v, 3) for k, v in i.items()}


def eval_probability_estimates(approach):
    appr = [
        ('good_run_unjudged_pos_5_easy_topic', good_run_unjudged_pos_5_easy_topic()),
        ('good_run_unjudged_pos_5_difficult_topic', good_run_unjudged_pos_5_difficult_topic()),
        ('bad_run_unjudged_pos_5_easy_topic', bad_run_unjudged_pos_5_easy_topic()),
        ('bad_run_unjudged_pos_5_difficult_topic', bad_run_unjudged_pos_5_difficult_topic()),
    ]

    return {k: __norm(approach.estimate_probabilities(v['run'], v['qrels'])) for k, v in appr}


class TestErrorAnalysisApproaches(TestCase):
    def test_unjudged_irrelevant_before_judged(self):
        input_data = unjudged_documents_before_judged_documents()
        expected = {"condensed-ndcg@5": 1.0, "ndcg@5": 0.0}

        actual = eval_measures(input_data)

        self.assertEquals(expected, actual)

    def test_probability_estimates_run_dependent(self):
        estimator = CountProbabilityEstimator()
        # Pool independent system distinguishes good run from bad run but not easy topic from difficult topic
        expected = {
            'good_run_unjudged_pos_5_easy_topic': {0: 0.25, 1: 0.75, 2: 0.0, 3: 0.0},
            'good_run_unjudged_pos_5_difficult_topic': {0: 0.25, 1: 0.75, 2: 0.0, 3: 0.0},
            'bad_run_unjudged_pos_5_easy_topic': {0: 0.75, 1: 0.25, 2: 0.0, 3: 0.0},
            'bad_run_unjudged_pos_5_difficult_topic': {0: 0.75, 1: 0.25, 2: 0.0, 3: 0.0}}

        actual = eval_probability_estimates(estimator)

        print(actual)
        self.assertEquals(expected, actual)

    def test_probability_estimates_cnt_pool_dependent(self):
        estimator = RunIndependentCountProbabilityEstimator()
        # Run independent system distinguishes easy topic from difficult topic but not good run from bad run
        expected = {
            'good_run_unjudged_pos_5_easy_topic': {0: 0.273, 1: 0.727, 2: 0.0, 3: 0.0},
            'bad_run_unjudged_pos_5_easy_topic': {0: 0.273, 1: 0.727, 2: 0.0, 3: 0.0},
            'good_run_unjudged_pos_5_difficult_topic': {0: 0.692, 1: 0.308, 2: 0.0, 3: 0.0},
            'bad_run_unjudged_pos_5_difficult_topic': {0: 0.692, 1: 0.308, 2: 0.0, 3: 0.0}
        }

        actual = eval_probability_estimates(estimator)

        print(actual)
        self.assertEquals(expected, actual)

    def test_probability_estimates_poison_rank_pool_dependent(self):
        estimator = PoissonEstimator()
        # Run/Pool Dependent system...
        expected = {
            'good_run_unjudged_pos_5_easy_topic': {0: 0.979, 1: 0.021, 2: 0, 3: 0},
            'good_run_unjudged_pos_5_difficult_topic': {0: 0.999, 1: 0.001, 2: 0, 3: 0},
            'bad_run_unjudged_pos_5_easy_topic': {0: 0.84, 1: 0.16, 2: 0, 3: 0},
            'bad_run_unjudged_pos_5_difficult_topic': {0: 0.957, 1: 0.043, 2: 0, 3: 0}}

        actual = eval_probability_estimates(estimator)

        print(actual)
        self.assertEquals(expected, actual)
