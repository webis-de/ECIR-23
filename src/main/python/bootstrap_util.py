from run_file_processing import normalize_run
import pandas as pd
import json
from copy import deepcopy
from tqdm import tqdm
from numpy import isnan
from trectools import TrecQrel, TrecEval, TrecRun

from fast_ndcg_implementation import MemorizedNdcg


class BootstrappingStrategey:
    def __init__(self, qrels=None):
        self.__qrels = qrels.qrels_data.copy()[["query", "docid", "rel"]]

    def single_bootstrap(self, qrels_and_run, unjudged_docs, rand):
        raise ValueError('Implement this method in implementations')

    def unjudged_documents(self, run):
        ret = pd.merge(run, self.__qrels, how="left")
        ret = ret[ret["rel"].isnull()]

        return sorted(list(set(list(ret["docid"].unique()))))

    def join_run_and_qrels(self, run):
        qid = run['query'].unique()
        if len(qid) != 1:
            raise ValueError('sdada')
        qid = qid[0]

        ret = pd.merge(self.__qrels, run[["query", "docid", "rank"]], how="left")

        return ret[ret['query'] == qid]

    def bootstraps_for_topic(self, run, repeat, seed=None):
        from random import Random

        rand = Random() if seed is None else Random(seed)
        unjudged_docs = self.unjudged_documents(run)
        qrels_and_run = self.join_run_and_qrels(run)

        ret = []
        for i in range(repeat):
            ret += [self.single_bootstrap(qrels_and_run, unjudged_docs, rand)]

        return [json.dumps(i, sort_keys=True) for i in ret]


class FullyIndependentBootstrappingStrategey(BootstrappingStrategey):
    def join_run_and_qrels(self, run):
        ret = super().join_run_and_qrels(run)
        ret = ret[~ret["rel"].isnull()]
        ret = ret[ret["rank"].isnull()]

        return sorted(list(set(ret['docid'].unique())))

    def single_bootstrap(self, qrels_and_run, unjudged_docs, rand):
        qrels_and_run = deepcopy(qrels_and_run)
        rand.shuffle(qrels_and_run)

        return {k: qrels_and_run.pop() for k in unjudged_docs}


class BootstrappEval:
    def __init__(self, measure, bootstrapping_strategy, verbose=True):
        self.__measure = measure
        self.__depth = int(measure.split('@')[-1])
        self.__verbose = verbose
        self.__bootstrapping_strategy = bootstrapping_strategy

        if not self.__measure.lower().startswith('ndcg@'):
            raise ValueError('Invalid Measure')
        self.__evaluator = MemorizedNdcg(self.__depth)

    def bootstrap(self, run, qrels, measure, repeat=5, seed=1):
        run = normalize_run(run, self.__depth)
        ret = {}
        topics = sorted(qrels.qrels_data['query'].unique())
        if self.__verbose:
            topics = tqdm(self.__verbose, desc="Bootstrapping")

        for topic in topics:
            assert topic not in ret
            ret[topic] = {measure: []}
            run_for_topic = run.run_data[run.run_data['query'] == topic]
            eval_scores = self.eval_scores(run_for_topic, qrels)

            for bootstrap in self.all_bootstrapps(run_for_topic, seed, repeat):
                ret[topic][measure] += [eval_scores[topic][bootstrap]]

        return ret

    def all_bootstrapps(self, run_for_topic, seed, repeat):
        return self.__bootstrapping_strategy.bootstraps_for_topic(run_for_topic, seed=seed, repeat=repeat)

    def eval_scores(self, run, qrels):
        ret = {}
        memorized_ndg_scores = self.__evaluator.get_ndcg(run, qrels, self.__depth)

        if type(qrels) == TrecQrel:
            qrels = qrels.qrels_data

        for topic in qrels['query'].unique():
            assert topic not in ret
            incomplete_ndcg = memorized_ndg_scores[topic]

            doc_to_qrel = {}
            for _, i in qrels[qrels['query'] == topic].iterrows():
                if i['docid'] in doc_to_qrel:
                    raise ValueError(f'I do not know how to handle duplicates in qrels: docid={i["docid"]} topic={topic}')

                doc_to_qrel[i['docid']] = i['rel']

            incomplete_ndcg.set_doc_to_qrel(doc_to_qrel)

            ret[topic] = incomplete_ndcg

        return ret

