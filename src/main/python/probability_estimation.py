from scipy.stats import poisson
import pandas as pd
from run_file_processing import IncompletePools


def normalize_identifier(identifier):
    return identifier.replace('/', '-')


def load_pool_task(task):
    out_file = task['working_directory'] + '/' + task['out_file_name']

    qrel_file = 'src/main/resources/unprocessed/topics-and-qrels/qrels.robust04.txt'
    pooling = IncompletePools(
        pool_per_run_file=task['working_directory'] + '/processed/pool-documents-per-run-' + normalize_identifier(
            task['trec_identifier']) + '.json.gz')
    pools = {k: v for k, v in pooling.create_incomplete_pools_for_run(task['run'])}

    return pools


class CountProbabilityEstimator:
    def __init__(self, smoothing=0.0001):
        self.__smoothing = smoothing

    def estimate_probability(self, run, qrels, relevance_level):
        ret = pd.merge(run.run_data, qrels.qrels_data[["query", "docid", "rel"]], how="left")
        ret = ret[~ret["rel"].isnull()]

        return max(len(ret[ret["rel"] == relevance_level])/len(ret), self.__smoothing)


class RunIndependentCountProbabilityEstimator:
    def __init__(self, smoothing=0.0001):
        self.__smoothing = smoothing

    def estimate_probability(self, run, qrels, relevance_level):
        qrels = qrels.qrels_data

        return max(len(qrels[qrels["rel"] == relevance_level])/len(qrels), self.__smoothing)


class PoissonEstimator(CountProbabilityEstimator):
    def estimate_probability(self, run, qrels, relevance_level, k=None):
        return poisson.pmf(k=k, mu=super().estimate_probability(run, qrels, relevance_level))


# https://machinelearningmastery.com/softmax-activation-function-with-python/
