from scipy.stats import poisson
import pandas as pd
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
        tmp = pd.merge(run.run_data, qrels.qrels_data[["query", "docid", "rel"]], how="left")
        tmp = tmp[tmp["rel"].isnull()]

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
        ret = pd.merge(run.run_data, qrels.qrels_data[["query", "docid", "rel"]], how="left")
        ret = ret[~ret["rel"].isnull()]

        return max(len(ret[ret["rel"] == relevance_level])/len(ret), super().smoothing())


class RunIndependentCountProbabilityEstimator(ProbabilityEstimator):
    def estimate_single_probability(self, run, qrels, relevance_level,  k=None):
        qrels = qrels.qrels_data

        return max(len(qrels[qrels["rel"] == relevance_level])/len(qrels), super().smoothing())


class PoissonEstimator(CountProbabilityEstimator):
    def estimate_single_probability(self, run, qrels, relevance_level, k=None):
        return poisson.pmf(k=k, mu=super().estimate_single_probability(run, qrels, relevance_level))



# https://machinelearningmastery.com/softmax-activation-function-with-python/
