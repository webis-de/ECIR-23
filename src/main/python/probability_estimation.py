from scipy.stats import poisson
import pandas as pd
from trectools import TrecQrel, TrecRun
from run_file_processing import IncompletePools


def normalize_identifier(identifier):
    return identifier.replace('/', '-')


def load_pool_task(task):
    pooling = IncompletePools(
        pool_per_run_file=task['working_directory'] + '/processed/pool-documents-per-run-' + normalize_identifier(
            task['trec_identifier']) + '.json.gz')
    pools = {k: v for k, v in pooling.create_incomplete_pools_for_run(task['run'])}

    return pools


class ProbabilityEstimator:
    def estimate_probabilities(self, run, qrels):
        if type(run) == TrecRun:
            return self.estimate_probabilities(run.run_data, qrels)
        if type(qrels) == TrecQrel:
            return self.estimate_probabilities(run, qrels.qrels_data)

        tmp = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")
        tmp = tmp[tmp["rel"].isnull()]

        if len(tmp) == 0:
            # Default
            return {0: 0, 1: 0, 2: 0, 3: 0}

        if len(tmp) == len(run):
            # Default
            return {0: 1, 1: 0, 2: 0, 3: 0}

        ret = [0, 0, 0, 0]
        rel_levels = [0, 1, 2, 3]

        for rel_level in rel_levels:
            for rank in tmp['rank']:
                ret[rel_level] += self.estimate_single_probability(run, qrels, rel_level, k=rank)
        sum_ret = sum(ret)
        ret = [i/sum_ret for i in ret]

        return {rl: ret[rl] for rl in rel_levels}

    def smoothing(self):
        return 0.0001

    def estimate_single_probability(self, run, qrels, rel_level, k=None):
        pass


class CountProbabilityEstimator(ProbabilityEstimator):
    def estimate_single_probability(self, run, qrels, relevance_level, k=None):
        ret = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")
        ret = ret[~ret["rel"].isnull()]

        return max(len(ret[ret["rel"] == relevance_level])/len(ret), super().smoothing())


class RunIndependentCountProbabilityEstimator(ProbabilityEstimator):
    def estimate_single_probability(self, run, qrels, relevance_level, k=None):
        return max(len(qrels[qrels["rel"] == relevance_level])/len(qrels), super().smoothing())


class RunAndPoolDependentProbabilityEstimator(ProbabilityEstimator):
    def estimate_probabilities(self, run, qrels):
        if type(run) == TrecRun:
            return self.estimate_probabilities(run.run_data, qrels)
        if type(qrels) == TrecQrel:
            return self.estimate_probabilities(run, qrels.qrels_data)

        ret = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")

        num_judged = len(ret[~ret["rel"].isnull()])
        num_relevant = len(ret[ret["rel"] == 1])

        if num_judged <= 0.0000000000001:
            # Default
            return {0: 1, 1: 0, 2: 0, 3: 0}

        p_relevant = num_relevant/num_judged
        p_unjudged = (len(ret) - num_judged)/len(ret)

        if p_unjudged <= 0.0000000000001:
            # Default
            return {0: 0, 1: 0, 2: 0, 3: 0}

        p_relevant_from_pool = len(qrels[qrels['rel'] == 1])/len(qrels)

        p_relevant_given_unjudged = p_relevant_from_pool*p_relevant

        return {
            0: 1 - p_relevant_given_unjudged,
            1: p_relevant_given_unjudged,
            2: 0,
            3: 0,
        }

    def estimate_single_probability(self, run, qrels, relevance_level, k=None):
        raise ValueError('Not implemented')


class PoissonEstimator(CountProbabilityEstimator):
    def __init__(self, to_add=1, p_add_from_pool_given_unjudged=None, lower_p=None, upper_p=None):
        self.__to_add = to_add
        self.__p_add_from_pool_given_unjudged = p_add_from_pool_given_unjudged
        self.__lower_p = lower_p
        self.__upper_p = upper_p

    def estimate_probabilities(self, run, qrels):
        if type(run) == TrecRun:
            return self.estimate_probabilities(run.run_data, qrels)
        if type(qrels) == TrecQrel:
            return self.estimate_probabilities(run, qrels.qrels_data)

        ret = pd.merge(run, qrels[["query", "docid", "rel"]], how="left")

        num_judged = len(ret[~ret["rel"].isnull()])
        num_relevant = len(ret[ret["rel"] == 1])

        if num_judged <= 0.0000000000001:
            # Default
            return {0: 1, 1: 0, 2: 0, 3: 0}

        p_relevant = num_relevant/num_judged
        p_unjudged = (len(ret) - num_judged)/len(ret)

        if p_unjudged <= 0.0000000000001:
            # Default
            return {0: 0, 1: 0, 2: 0, 3: 0}

        p_relevant_from_pool = len(qrels[qrels['rel'] == 1])/len(qrels)

        p_add_from_pool_given_unjudged = poisson.pmf(k=num_relevant + self.__to_add, mu=p_relevant_from_pool)

        if self.__p_add_from_pool_given_unjudged:
            p_add_from_pool_given_unjudged = self.__p_add_from_pool_given_unjudged

        if self.__lower_p and self.__upper_p:
            p_add_from_pool_given_unjudged = ((self.__upper_p - self.__lower_p) * p_relevant_from_pool) + self.__lower_p

        p_relevant_given_unjudged = (p_add_from_pool_given_unjudged * p_relevant)/p_unjudged

        return {
            0: 1 - p_relevant_given_unjudged,
            1: p_relevant_given_unjudged,
            2: 0,
            3: 0,
        }

    def estimate_single_probability(self, run, qrels, relevance_level, k=None):
        raise ValueError('Not implemented')
