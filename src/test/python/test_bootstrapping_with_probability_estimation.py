from unittest import TestCase
from error_analysis_approaches import good_run_unjudged_pos_5_difficult_topic
from pool_bootstrap_util import ProbabilityEstimatedBootstrappingStrategey


class DummyEstimator:
    def __init__(self, ret):
        self.__ret = ret

    def estimate_probabilities(self, run, qrels):
        return self.__ret


class TestBootstrappingWithProbabilityEstimation(TestCase):
    def test_selection_of_always_irrelevent_results(self):
        estimator = DummyEstimator({0: 1, 1: 0, 2: 0, 3: 0})
        run, qrels = good_run_unjudged_pos_5_difficult_topic().values()

        bs = ProbabilityEstimatedBootstrappingStrategey(qrels, estimator)

        expected = {'{"u-1": "nr-2"}'}
        actual = set(bs.bootstraps_for_topic(run.run_data, repeat=20, seed=0))

        self.assertEquals(actual, expected)

    def test_selection_of_always_relevent_results(self):
        estimator = DummyEstimator({0: 0, 1: 1, 2: 0, 3: 0})
        run, qrels = good_run_unjudged_pos_5_difficult_topic().values()

        bs = ProbabilityEstimatedBootstrappingStrategey(qrels, estimator)

        expected = {'{"u-1": "r-4"}'}
        actual = set(bs.bootstraps_for_topic(run.run_data, repeat=20, seed=0))

        self.assertEquals(actual, expected)

    def test_selection_of_always_relevent_results_of_lower_score(self):
        estimator = DummyEstimator({0: 0, 1: 0, 2: 1, 3: 0})
        run, qrels = good_run_unjudged_pos_5_difficult_topic().values()

        bs = ProbabilityEstimatedBootstrappingStrategey(qrels, estimator)

        expected = {'{"u-1": "r-4"}'}
        actual = set(bs.bootstraps_for_topic(run.run_data, repeat=20, seed=0))

        self.assertEquals(actual, expected)

    def test_selection_of_sometimes_relevant_sometimes_irrelevant_results(self):
        estimator = DummyEstimator({0: .5, 1: .5, 2: 0, 3: 0})
        run, qrels = good_run_unjudged_pos_5_difficult_topic().values()

        bs = ProbabilityEstimatedBootstrappingStrategey(qrels, estimator)

        expected = ['{"u-1": "r-4"}', '{"u-1": "r-4"}', '{"u-1": "nr-2"}', '{"u-1": "nr-2"}', '{"u-1": "r-4"}',
                    '{"u-1": "nr-2"}', '{"u-1": "r-4"}', '{"u-1": "nr-2"}', '{"u-1": "nr-2"}', '{"u-1": "r-4"}',
                    '{"u-1": "r-4"}', '{"u-1": "r-4"}', '{"u-1": "nr-2"}', '{"u-1": "r-4"}', '{"u-1": "r-4"}',
                    '{"u-1": "nr-2"}', '{"u-1": "r-4"}', '{"u-1": "r-4"}', '{"u-1": "r-4"}', '{"u-1": "r-4"}'
                    ]
        actual = bs.bootstraps_for_topic(run.run_data, repeat=20, seed=0)
        print(actual)

        self.assertEquals(actual, expected)
